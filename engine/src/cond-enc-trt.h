#pragma once
// cond-enc-trt.h — TensorRT native loader for the ACEStep condition encoder.
//
// The cond-encoder engine is a strongly-typed graph with weights EMBEDDED
// in the plan (NOT kREFIT_IDENTICAL like dit-trt.h). So this
// loader only deserializes — there is no ONNX sidecar, no refitter, no weight
// streaming. The lyric/timbre/text fusion (8L lyric encoder, 4L timbre encoder,
// text projector, CLS prepend, pack order) is baked entirely into the graph.
//
// Engine I/O contract (auto-detected at load time from the engine's binding
// dtypes — supports FP16, BF16, and FP32 engines):
//   input  "text_hidden"   {HALF|BF16|FLOAT} [B, S_text,  1024]
//   input  "lyric_embed"   {HALF|BF16|FLOAT} [B, S_lyric, 1024]
//   input  "timbre_feats"  {HALF|BF16|FLOAT} [B, S_ref,   64]
//   output "enc_hidden"    {HALF|BF16|FLOAT} [B, S_total, 2048]  (S_total = S_lyric+1+S_text)
//   profiles: B fixed at 1; text S in [1,512], lyric S in [1,1024],
//             timbre S in [1,512].
//
// FP16 vs BF16 vs FP32 trade-offs:
//   FP16 (HALF):  smallest engine, ±65504 range — can saturate to ±Inf/NaN
//                 on extreme activations (text_hidden can reach ±51, and
//                 attention scores can overflow). Runtime sanitizes NaN/Inf
//                 to finite range, corrupting ~5 elements per forward.
//   BF16 (BF16):  same engine size as FP16, ±3.4e38 range (matches FP32) —
//                 no overflow, no sanitization. PREFERRED on Ampere+ GPUs.
//   FP32 (FLOAT): largest engine, safest, slowest. No conversion overhead.
//
// The C++ host always passes FP32 to cond_enc_trt_forward and receives FP32
// back; the loader handles the FP32<->engine-dtype conversion internally.
//
// The output S dimension is DYNAMIC: the graph packs cat(lyric, timbre[0:1],
// text_proj), so S_total = S_lyric + 1 + S_text when timbre is present. The
// loader queries the runtime output shape after enqueue rather than assuming it.
//
// Output handoff. cond_enc_trt_forward writes FP32 hidden states into a caller
// host buffer in [B, S_total, H] row-major order. For B=1 this byte order equals
// the GGML cond_ggml_forward output layout (enc_hidden [2048, S_total] ggml =
// S-major, H-contiguous, packed lyric/timbre/text), so the TRT path can swap in
// where cond_ggml_forward is called today. The pack order is identical because
// it is baked into the TRT graph the ONNX export captured.
//
// null_condition_emb is NOT an engine input. The GGML CondGGML loads it for the
// classifier-free-guidance path, but that lives outside the encoder forward —
// the TRT engine consumes only the three real conditioning inputs. The caller
// uses the existing null_cond path; it is independent of this loader.
//
// EVICT_STRICT behavior: cond_enc_trt_free fully unloads the engine,
// runtime, context, stream, and all device buffers. On subsequent runs,
// the engine is reloaded from disk. With --keep-loaded, the full state
// is kept resident and reused across requests.

#ifdef HOT_STEP_TRT

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "NvInfer.h"

#if defined(HOT_STEP_TRT_VERSION_MAJOR) && HOT_STEP_TRT_VERSION_MAJOR != 11
#error "HOT_STEP_TRT requires TensorRT 11.x headers"
#endif

inline uint16_t cond_enc_trt_float_to_bf16(float f) {
    // BF16 is the upper 16 bits of FP32, with round-to-nearest-even.
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    // Round-to-nearest-even: add 0x7FFF + (LSB of result) to bias the truncation.
    uint32_t rounding_bias = 0x7FFF + ((x >> 16) & 1);
    return (uint16_t)((x + rounding_bias) >> 16);
}

inline float cond_enc_trt_bf16_to_float(uint16_t b) {
    // BF16 -> FP32: just zero-extend the upper 16 bits into the upper half of FP32.
    uint32_t x = (uint32_t) b << 16;
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

// ── Engine I/O dtype enum ────────────────────────────────────────────────────
//
// Auto-detected at load time from the output binding's dtype. Drives whether
// the upload path converts FP32->FP16 or FP32->BF16 (or passes FP32 through),
// and whether the download path runs the FP16 sanitization (FP16 only).

enum CondEncIoDtype {
    COND_ENC_IO_FP16 = 0,  // HALF  — ±65504 range, needs sanitization
    COND_ENC_IO_BF16 = 1,  // BF16  — ±3.4e38 range, no sanitization needed
    COND_ENC_IO_FP32 = 2,  // FLOAT — no conversion, no sanitization
};

// ── FP16 <-> FP32 helpers (full subnormal/Inf/NaN-correct) ──────────────────

inline float cond_enc_trt_half_to_float(uint16_t h) {
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;  // ±0
        } else {
            // Subnormal half → normalized float.
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; e++; } while ((m & 0x0400u) == 0);
            m &= 0x03FFu;
            u = sign | ((uint32_t) (127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);  // Inf / NaN
    } else {
        u = sign | ((uint32_t) (exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t cond_enc_trt_float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t        exp  = (int32_t) ((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x007FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) {
        // Inf / NaN: preserve mantissa non-zero-ness for NaN.
        return (uint16_t) (sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    if (exp >= 31) {
        return (uint16_t) (sign | 0x7C00u);  // overflow → Inf
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;  // underflow → ±0
        }
        // Subnormal half.
        uint32_t m = (mant | 0x00800000u) >> (uint32_t) (14 - exp);
        return (uint16_t) (sign | m);
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
}

// ── TRT Logger ──────────────────────────────────────────────────────────────

class CondEncTrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char * msg) noexcept override {
        if (severity > Severity::kWARNING) return;
        const char * prefix = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: prefix = "[CondEnc-TRT-INTERNAL] "; break;
            case Severity::kERROR:          prefix = "[CondEnc-TRT-ERROR] ";    break;
            case Severity::kWARNING:        prefix = "[CondEnc-TRT-WARN] ";     break;
            default: break;
        }
        fprintf(stderr, "%s%s\n", prefix, msg);
    }
};

// ── CondEncTrt context ──────────────────────────────────────────────────────

struct CondEncTrt {
    // TRT runtime objects.
    nvinfer1::IRuntime *    runtime = nullptr;
    nvinfer1::ICudaEngine * engine  = nullptr;

    // Evictable: freed by release_evictable, reallocated by forward.
    nvinfer1::IExecutionContext * context = nullptr;
    void *  d_text   = nullptr;  // engine-dtype [B*S_text*1024]
    void *  d_lyric  = nullptr;  // engine-dtype [B*S_lyric*1024]
    void *  d_timbre = nullptr;  // engine-dtype [B*S_ref*64]
    void *  d_out    = nullptr;  // engine-dtype [B*S_total*2048]
    size_t  buf_text_bytes   = 0;
    size_t  buf_lyric_bytes  = 0;
    size_t  buf_timbre_bytes = 0;
    size_t  buf_out_bytes    = 0;

    int  hidden_size = 2048;  // resolved from the output binding at load

    // Engine I/O dtype, auto-detected from the output binding at load time.
    // Drives upload/download conversion and whether sanitization runs.
    CondEncIoDtype io_dtype = COND_ENC_IO_FP16;

    CondEncTrtLogger logger;
    cudaStream_t     stream = nullptr;

    int64_t load_time_ms = 0;
};

// Free per-job device I/O buffers and execution context.
inline void cond_enc_trt_release_evictable(CondEncTrt * ctx) {
    if (!ctx) return;
    bool released = false;
    if (ctx->d_text)   { cudaFree(ctx->d_text);   ctx->d_text = nullptr;   released = true; }
    if (ctx->d_lyric)  { cudaFree(ctx->d_lyric);  ctx->d_lyric = nullptr;  released = true; }
    if (ctx->d_timbre) { cudaFree(ctx->d_timbre); ctx->d_timbre = nullptr; released = true; }
    if (ctx->d_out)    { cudaFree(ctx->d_out);    ctx->d_out = nullptr;    released = true; }
    if (ctx->context)  { delete ctx->context;     ctx->context = nullptr;  released = true; }
    ctx->buf_text_bytes   = 0;
    ctx->buf_lyric_bytes  = 0;
    ctx->buf_timbre_bytes = 0;
    ctx->buf_out_bytes    = 0;
    fprintf(stderr, "[CondEnc-TRT] %s\n",
            released ? "released device buffers + context" : "no evictable state to release");
}

// ── Load (deserialize only — embedded weights, no refit) ────────────────────

inline bool cond_enc_trt_load(CondEncTrt * ctx, const char * engine_path, int device_id = 0) {
    auto t0 = std::chrono::steady_clock::now();
    cudaSetDevice(device_id);

    FILE * f = fopen(engine_path, "rb");
    if (!f) {
        fprintf(stderr, "[CondEnc-TRT] Cannot open engine %s\n", engine_path);
        return false;
    }
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    int64_t engine_size = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
#else
    fseeko(f, 0, SEEK_END);
    int64_t engine_size = (int64_t)ftello(f);
    fseeko(f, 0, SEEK_SET);
#endif
    if (engine_size <= 0) {
        fprintf(stderr, "[CondEnc-TRT] Empty engine file %s\n", engine_path);
        fclose(f);
        return false;
    }
    std::vector<char> engine_data((size_t) engine_size);
    size_t rd = fread(engine_data.data(), 1, (size_t) engine_size, f);
    fclose(f);
    if (rd != (size_t) engine_size) {
        fprintf(stderr, "[CondEnc-TRT] Short read on engine %s (%zu/%lld)\n", engine_path, rd, (long long)engine_size);
        return false;
    }

    ctx->runtime = nvinfer1::createInferRuntime(ctx->logger);
    if (!ctx->runtime) {
        fprintf(stderr, "[CondEnc-TRT] Failed to create TRT runtime\n");
        return false;
    }
    ctx->engine = ctx->runtime->deserializeCudaEngine(engine_data.data(), (size_t) engine_size);
    if (!ctx->engine) {
        fprintf(stderr, "[CondEnc-TRT] Failed to deserialize engine %s\n", engine_path);
        return false;
    }

    // Resolve + validate the I/O contract against the embedded-weight graph.
    // The engine I/O dtype (FP16 / BF16 / FP32) is auto-detected from the
    // output binding and stored in ctx->io_dtype. The runtime adapts the
    // upload (FP32 -> engine dtype) and download (engine dtype -> FP32 +
    // sanitize if FP16) paths accordingly.
    bool saw_text = false, saw_lyric = false, saw_timbre = false, saw_output = false;
    int  num_io = ctx->engine->getNbIOTensors();
    nvinfer1::DataType detected_dtype = nvinfer1::DataType::kHALF;  // default
    for (int i = 0; i < num_io; i++) {
        const char * name  = ctx->engine->getIOTensorName(i);
        auto         dtype = ctx->engine->getTensorDataType(name);
        auto         mode  = ctx->engine->getTensorIOMode(name);
        const std::string n = name;
        if (n == "text_hidden") {
            saw_text = true;
            if (mode != nvinfer1::TensorIOMode::kINPUT ||
                (dtype != nvinfer1::DataType::kHALF &&
                 dtype != nvinfer1::DataType::kBF16 &&
                 dtype != nvinfer1::DataType::kFLOAT)) {
                fprintf(stderr, "[CondEnc-TRT] text_hidden has unexpected mode/dtype\n");
                return false;
            }
        } else if (n == "lyric_embed") {
            saw_lyric = true;
            if (mode != nvinfer1::TensorIOMode::kINPUT ||
                (dtype != nvinfer1::DataType::kHALF &&
                 dtype != nvinfer1::DataType::kBF16 &&
                 dtype != nvinfer1::DataType::kFLOAT)) {
                fprintf(stderr, "[CondEnc-TRT] lyric_embed has unexpected mode/dtype\n");
                return false;
            }
        } else if (n == "timbre_feats") {
            saw_timbre = true;
            if (mode != nvinfer1::TensorIOMode::kINPUT ||
                (dtype != nvinfer1::DataType::kHALF &&
                 dtype != nvinfer1::DataType::kBF16 &&
                 dtype != nvinfer1::DataType::kFLOAT)) {
                fprintf(stderr, "[CondEnc-TRT] timbre_feats has unexpected mode/dtype\n");
                return false;
            }
        } else if (n == "enc_hidden") {
            saw_output = true;
            if (mode != nvinfer1::TensorIOMode::kOUTPUT ||
                (dtype != nvinfer1::DataType::kHALF &&
                 dtype != nvinfer1::DataType::kBF16 &&
                 dtype != nvinfer1::DataType::kFLOAT)) {
                fprintf(stderr, "[CondEnc-TRT] enc_hidden has unexpected mode/dtype\n");
                return false;
            }
            detected_dtype = dtype;
            nvinfer1::Dims d = ctx->engine->getTensorShape(name);
            if (d.nbDims == 3 && d.d[2] > 0) {
                ctx->hidden_size = (int) d.d[2];
            }
        }
    }
    // Classify the detected dtype into our enum.
    if (detected_dtype == nvinfer1::DataType::kHALF) {
        ctx->io_dtype = COND_ENC_IO_FP16;
    } else if (detected_dtype == nvinfer1::DataType::kBF16) {
        ctx->io_dtype = COND_ENC_IO_BF16;
    } else {
        ctx->io_dtype = COND_ENC_IO_FP32;
    }
    {
        const char * dtype_names[] = {"FP16", "BF16", "FP32"};
        fprintf(stderr, "[CondEnc-TRT] I/O dtype: %s\n", dtype_names[ctx->io_dtype]);
    }
    if (!saw_text || !saw_lyric || !saw_timbre || !saw_output) {
        fprintf(stderr, "[CondEnc-TRT] Engine missing text_hidden/lyric_embed/timbre_feats/enc_hidden bindings\n");
        return false;
    }

    if (!ctx->stream) {
        cudaStreamCreate(&ctx->stream);
    }

    auto t1 = std::chrono::steady_clock::now();
    ctx->load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "[CondEnc-TRT] Deserialized %s (%lld bytes, H=%d, %lld ms)\n",
            engine_path, (long long)engine_size, ctx->hidden_size, (long long) ctx->load_time_ms);
    fflush(stderr);
    return true;
}

// ── Forward ─────────────────────────────────────────────────────────────────

// Upload one input from a host FP32 array, converting to the engine's I/O
// dtype (FP16 / BF16 / FP32). (Re)allocates the device buffer as needed.
// Returns false on a CUDA failure. d_buf / buf_bytes are the slot's tracked
// device pointer + capacity; label is the binding's diagnostic name.
inline bool cond_enc_trt_upload(CondEncTrt * ctx, const float * src, size_t n_elems,
                                void ** d_buf, size_t * buf_bytes, const char * label) {
    // Element size depends on the engine's I/O dtype.
    const size_t elem_bytes = (ctx->io_dtype == COND_ENC_IO_FP32) ? sizeof(float) : sizeof(uint16_t);
    const size_t bytes = n_elems * elem_bytes;
    if (*d_buf == nullptr || *buf_bytes < bytes) {
        if (*d_buf) cudaFree(*d_buf);
        if (cudaMalloc(d_buf, bytes) != cudaSuccess) {
            fprintf(stderr, "[CondEnc-TRT] cudaMalloc %s failed (%zu bytes)\n", label, bytes);
            *d_buf = nullptr;
            *buf_bytes = 0;
            return false;
        }
        *buf_bytes = bytes;
        fprintf(stderr, "[CondEnc-TRT] forward: %s buffer allocated (%zu bytes)\n", label, bytes);
    }

    if (ctx->io_dtype == COND_ENC_IO_FP32) {
        // No conversion — upload FP32 directly.
        if (cudaMemcpyAsync(*d_buf, src, bytes, cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess) {
            fprintf(stderr, "[CondEnc-TRT] H2D copy of %s failed\n", label);
            return false;
        }
        return true;
    }

    // FP16 or BF16: convert FP32 -> engine dtype on host, then upload.
    std::vector<uint16_t> half(n_elems);
    if (ctx->io_dtype == COND_ENC_IO_FP16) {
        for (size_t i = 0; i < n_elems; i++) {
            half[i] = cond_enc_trt_float_to_half(src[i]);
        }
    } else {  // COND_ENC_IO_BF16
        for (size_t i = 0; i < n_elems; i++) {
            half[i] = cond_enc_trt_float_to_bf16(src[i]);
        }
    }
    if (cudaMemcpyAsync(*d_buf, half.data(), bytes, cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess) {
        fprintf(stderr, "[CondEnc-TRT] H2D copy of %s failed\n", label);
        return false;
    }
    return true;
}

// Run the condition encoder on host FP32 conditioning inputs.
//   text_hidden:  host float32 [B*S_text*1024]   (row-major [B,S_text,1024])
//   lyric_embed:  host float32 [B*S_lyric*1024]   (row-major [B,S_lyric,1024])
//   timbre_feats: host float32 [B*S_ref*64]       (row-major [B,S_ref,64])
//   B, S_text, S_lyric, S_ref: batch + per-input sequence lengths (B==1)
//   out_f32:      host float32, resized to [B*S_total*H], row-major [B,S_total,H]
//   out_S_total:  receives the packed total sequence length (S_lyric+1+S_text)
// Returns false on any TRT/CUDA failure. Reallocates context + device buffers
// on demand so it works directly after cond_enc_trt_release_evictable.
inline bool cond_enc_trt_forward(CondEncTrt * ctx,
                                 const float * text_hidden,  int S_text,
                                 const float * lyric_embed,  int S_lyric,
                                 const float * timbre_feats, int S_ref,
                                 int           B,
                                 std::vector<float> & out_f32,
                                 int *         out_S_total) {
    if (!ctx || !ctx->engine) {
        fprintf(stderr, "[CondEnc-TRT] FATAL: forward called before load\n");
        return false;
    }
    if (B <= 0 || S_text <= 0 || S_lyric <= 0 || S_ref <= 0 ||
        !text_hidden || !lyric_embed || !timbre_feats) {
        fprintf(stderr, "[CondEnc-TRT] forward: invalid args (B=%d S_text=%d S_lyric=%d S_ref=%d)\n",
                B, S_text, S_lyric, S_ref);
        return false;
    }

    const int    H = ctx->hidden_size;
    const size_t n_text   = (size_t) B * S_text  * 1024;
    const size_t n_lyric  = (size_t) B * S_lyric * 1024;
    const size_t n_timbre = (size_t) B * S_ref   * 64;

    if (!ctx->context) {
        ctx->context = ctx->engine->createExecutionContext();
        if (!ctx->context) {
            fprintf(stderr, "[CondEnc-TRT] Failed to (re)create execution context\n");
            return false;
        }
        fprintf(stderr, "[CondEnc-TRT] forward: execution context allocated\n");
    }

    // Upload the three FP16 inputs.
    if (!cond_enc_trt_upload(ctx, text_hidden,  n_text,   &ctx->d_text,   &ctx->buf_text_bytes,   "text_hidden")  ||
        !cond_enc_trt_upload(ctx, lyric_embed,  n_lyric,  &ctx->d_lyric,  &ctx->buf_lyric_bytes,  "lyric_embed")  ||
        !cond_enc_trt_upload(ctx, timbre_feats, n_timbre, &ctx->d_timbre, &ctx->buf_timbre_bytes, "timbre_feats")) {
        return false;
    }

    if (!ctx->context->setInputShape("text_hidden",  nvinfer1::Dims3(B, S_text,  1024)) ||
        !ctx->context->setInputShape("lyric_embed",  nvinfer1::Dims3(B, S_lyric, 1024)) ||
        !ctx->context->setInputShape("timbre_feats", nvinfer1::Dims3(B, S_ref,   64))) {
        fprintf(stderr, "[CondEnc-TRT] setInputShape failed\n");
        return false;
    }

    // The output S dimension is dynamic — query it from the context now that the
    // input shapes are set, then size the output buffer + host result.
    nvinfer1::Dims od = ctx->context->getTensorShape("enc_hidden");
    if (od.nbDims != 3 || od.d[0] != B || od.d[2] != H || od.d[1] <= 0) {
        fprintf(stderr, "[CondEnc-TRT] unexpected enc_hidden runtime shape (nbDims=%d)\n", od.nbDims);
        return false;
    }
    const int    S_total  = (int) od.d[1];
    const size_t n_out    = (size_t) B * S_total * H;
    // Element size depends on engine I/O dtype (FP16/BF16 = 2 bytes, FP32 = 4 bytes).
    const size_t out_elem_bytes = (ctx->io_dtype == COND_ENC_IO_FP32) ? sizeof(float) : sizeof(uint16_t);
    const size_t out_bytes = n_out * out_elem_bytes;

    if (ctx->d_out == nullptr || ctx->buf_out_bytes < out_bytes) {
        if (ctx->d_out) cudaFree(ctx->d_out);
        if (cudaMalloc(&ctx->d_out, out_bytes) != cudaSuccess) {
            fprintf(stderr, "[CondEnc-TRT] cudaMalloc enc_hidden failed (%zu bytes)\n", out_bytes);
            ctx->d_out = nullptr;
            ctx->buf_out_bytes = 0;
            return false;
        }
        ctx->buf_out_bytes = out_bytes;
        fprintf(stderr, "[CondEnc-TRT] forward: enc_hidden buffer allocated (%zu bytes)\n", out_bytes);
    }

    if (!ctx->context->setTensorAddress("text_hidden",  ctx->d_text)   ||
        !ctx->context->setTensorAddress("lyric_embed",  ctx->d_lyric)  ||
        !ctx->context->setTensorAddress("timbre_feats", ctx->d_timbre) ||
        !ctx->context->setTensorAddress("enc_hidden",   ctx->d_out)) {
        fprintf(stderr, "[CondEnc-TRT] setTensorAddress failed\n");
        return false;
    }
    if (!ctx->context->enqueueV3(ctx->stream)) {
        fprintf(stderr, "[CondEnc-TRT] enqueueV3 failed\n");
        return false;
    }

    // Pull engine-dtype output to host, convert to FP32 in [B,S_total,H] order.
    out_f32.resize(n_out);
    if (ctx->io_dtype == COND_ENC_IO_FP32) {
        // No conversion — download FP32 directly into the output buffer.
        if (cudaMemcpyAsync(out_f32.data(), ctx->d_out, out_bytes,
                            cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess) {
            fprintf(stderr, "[CondEnc-TRT] D2H copy of enc_hidden failed\n");
            return false;
        }
        if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) {
            fprintf(stderr, "[CondEnc-TRT] stream sync failed\n");
            return false;
        }
        // FP32 engine can't overflow; no sanitization needed.
        if (out_S_total) *out_S_total = S_total;
        return true;
    }

    // FP16 or BF16: download as uint16, then convert to FP32.
    std::vector<uint16_t> half(n_out);
    if (cudaMemcpyAsync(half.data(), ctx->d_out, out_bytes,
                        cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess) {
        fprintf(stderr, "[CondEnc-TRT] D2H copy of enc_hidden failed\n");
        return false;
    }
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) {
        fprintf(stderr, "[CondEnc-TRT] stream sync failed\n");
        return false;
    }

    if (ctx->io_dtype == COND_ENC_IO_BF16) {
        // BF16 has FP32-equivalent range (±3.4e38) — no overflow possible,
        // no sanitization needed. Just widen to FP32 (zero-extend upper 16 bits).
        for (size_t i = 0; i < n_out; i++) {
            out_f32[i] = cond_enc_trt_bf16_to_float(half[i]);
        }
        if (out_S_total) *out_S_total = S_total;
        return true;
    }

    // FP16 path: extreme activations can saturate to ±Inf (and 0*Inf etc. to
    // NaN) at isolated channels. The DiT consumer requires a finite enc_hidden
    // — a single non-finite element poisons the whole denoise. Clamp on the
    // FP16→FP32 boundary: NaN→0, ±Inf→±65504 (the FP16 finite max), preserving
    // sign/magnitude of saturated values. The GGML FP32 cond path does not
    // saturate, so this only affects the FP16 engine. BF16/FP32 engines skip
    // this path entirely (handled above).
    size_t sanitized = 0;
    for (size_t i = 0; i < n_out; i++) {
        float v = cond_enc_trt_half_to_float(half[i]);
        if (v != v) {  // NaN
            v = 0.0f;
            sanitized++;
        } else if (v > 65504.0f) {
            v = 65504.0f;
            sanitized++;
        } else if (v < -65504.0f) {
            v = -65504.0f;
            sanitized++;
        }
        out_f32[i] = v;
    }
    if (sanitized) {
        fprintf(stderr, "[CondEnc-TRT] sanitized %zu non-finite FP16 output element(s) to finite range\n",
                sanitized);
    }
    if (out_S_total) *out_S_total = S_total;
    return true;
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

inline void cond_enc_trt_free(CondEncTrt * ctx) {
    if (!ctx) return;
    cond_enc_trt_release_evictable(ctx);
    if (ctx->stream)  { cudaStreamDestroy(ctx->stream); ctx->stream = nullptr; }
    if (ctx->engine)  { delete ctx->engine;  ctx->engine = nullptr; }
    if (ctx->runtime) { delete ctx->runtime; ctx->runtime = nullptr; }
    fprintf(stderr, "[CondEnc-TRT] Engine fully unloaded\n");
}

#endif  // HOT_STEP_TRT
