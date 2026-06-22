#pragma once
// text-enc-trt.h — TensorRT native loader for the Qwen3 text encoder.
//
// The text-encoder engine is a strongly-typed graph with weights EMBEDDED
// in the plan (NOT kREFIT_IDENTICAL like dit-trt.h). So this
// loader only deserializes — there is no ONNX sidecar, no refitter, no weight
// streaming. The embed lookup is baked into the graph: the engine consumes
// token IDs directly, not pre-looked-up embeddings.
//
// Engine I/O contract (auto-detected at load time from the engine's binding
// dtypes — supports FP16, BF16, and FP32 engines):
//   input  "input_ids"     INT64                [B, S]        (token IDs; embed baked in)
//   output "hidden_states" {HALF|BF16|FLOAT}    [B, S, 1024]  (last_hidden_state)
//   profile: min [1,1], opt [1,128], max [1,512]
//
// FP16 vs BF16 vs FP32 trade-offs:
//   FP16 (HALF):  smallest engine, ±65504 range — intermediate activations in
//                 the 28-layer Qwen3 encoder can saturate to ±Inf/NaN. Unlike
//                 cond-enc-trt.h, text-enc-trt.h has NO sanitization, so
//                 corruption passes straight through to text_hidden.
//   BF16 (BF16):  same engine size as FP16, ±3.4e38 range (matches FP32) —
//                 no overflow, no corruption. PREFERRED on Ampere+ GPUs.
//   FP32 (FLOAT): largest engine, safest, slowest. No conversion overhead.
//
// Output handoff. text_enc_trt_forward writes FP32 hidden states into a caller
// host buffer in [B, S, H] row-major order. For B=1 this byte order matches the
// GGML text encoder's qwen3_forward output ([H,S] ggml layout = S-major,
// H-contiguous), so the TRT path can swap in where qwen3_forward is called today.
// The loader handles the engine-dtype -> FP32 conversion internally.
//
// EVICT_STRICT behavior: text_enc_trt_free fully unloads the engine,
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

// ── BF16 <-> FP32 helpers ────────────────────────────────────────────────────
// BF16 is the upper 16 bits of FP32 (truncated, with round-to-nearest-even).
// Simpler than FP16 <-> FP32 because BF16 shares FP32's exponent format.

inline uint16_t text_enc_trt_float_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    // Round-to-nearest-even: add 0x7FFF + (LSB of result) to bias the truncation.
    uint32_t rounding_bias = 0x7FFF + ((x >> 16) & 1);
    return (uint16_t)((x + rounding_bias) >> 16);
}

inline float text_enc_trt_bf16_to_float(uint16_t b) {
    // BF16 -> FP32: just zero-extend the upper 16 bits into the upper half of FP32.
    uint32_t x = (uint32_t) b << 16;
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

// ── TRT Logger ──────────────────────────────────────────────────────────────

class TextEncTrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char * msg) noexcept override {
        if (severity > Severity::kWARNING) return;
        const char * prefix = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: prefix = "[TextEnc-TRT-INTERNAL] "; break;
            case Severity::kERROR:          prefix = "[TextEnc-TRT-ERROR] ";    break;
            case Severity::kWARNING:        prefix = "[TextEnc-TRT-WARN] ";     break;
            default: break;
        }
        fprintf(stderr, "%s%s\n", prefix, msg);
    }
};

// ── TextEncTrt context ──────────────────────────────────────────────────────

// Engine I/O dtype, auto-detected from the output binding at load time.
// Drives the download conversion (FP16/BF16 -> FP32) and element sizing.
enum TextEncIoDtype {
    TEXT_ENC_IO_FP16 = 0,  // HALF  — ±65504 range, no sanitization (legacy)
    TEXT_ENC_IO_BF16 = 1,  // BF16  — ±3.4e38 range, no overflow possible
    TEXT_ENC_IO_FP32 = 2,  // FLOAT — no conversion, no overflow
};

struct TextEncTrt {
    // TRT runtime objects.
    nvinfer1::IRuntime *    runtime = nullptr;
    nvinfer1::ICudaEngine * engine  = nullptr;

    // Evictable: freed by release_evictable, reallocated by forward.
    nvinfer1::IExecutionContext * context = nullptr;
    void *  d_input_ids    = nullptr;  // INT64 [B*S]
    void *  d_hidden       = nullptr;  // engine-dtype [B*S*H]
    size_t  buf_input_bytes  = 0;
    size_t  buf_hidden_bytes = 0;

    int  hidden_size = 1024;  // resolved from the output binding at load

    // Engine I/O dtype, auto-detected from the output binding at load time.
    TextEncIoDtype io_dtype = TEXT_ENC_IO_FP16;

    TextEncTrtLogger logger;
    cudaStream_t     stream = nullptr;

    int64_t load_time_ms = 0;
};

// Free per-job device I/O buffers and execution context.
inline void text_enc_trt_release_evictable(TextEncTrt * ctx) {
    if (!ctx) return;
    bool released = false;
    if (ctx->d_input_ids) { cudaFree(ctx->d_input_ids); ctx->d_input_ids = nullptr; released = true; }
    if (ctx->d_hidden)    { cudaFree(ctx->d_hidden);    ctx->d_hidden = nullptr;    released = true; }
    if (ctx->context)     { delete ctx->context;        ctx->context = nullptr;     released = true; }
    ctx->buf_input_bytes  = 0;
    ctx->buf_hidden_bytes = 0;
    fprintf(stderr, "[TextEnc-TRT] %s\n",
            released ? "released device buffers + context" : "no evictable state to release");
}

// ── Load (deserialize only — embedded weights, no refit) ────────────────────

inline bool text_enc_trt_load(TextEncTrt * ctx, const char * engine_path, int device_id = 0) {
    auto t0 = std::chrono::steady_clock::now();
    cudaSetDevice(device_id);

    FILE * f = fopen(engine_path, "rb");
    if (!f) {
        fprintf(stderr, "[TextEnc-TRT] Cannot open engine %s\n", engine_path);
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
        fprintf(stderr, "[TextEnc-TRT] Empty engine file %s\n", engine_path);
        fclose(f);
        return false;
    }
    std::vector<char> engine_data((size_t) engine_size);
    size_t rd = fread(engine_data.data(), 1, (size_t) engine_size, f);
    fclose(f);
    if (rd != (size_t) engine_size) {
        fprintf(stderr, "[TextEnc-TRT] Short read on engine %s (%zu/%lld)\n", engine_path, rd, (long long)engine_size);
        return false;
    }

    ctx->runtime = nvinfer1::createInferRuntime(ctx->logger);
    if (!ctx->runtime) {
        fprintf(stderr, "[TextEnc-TRT] Failed to create TRT runtime\n");
        return false;
    }
    ctx->engine = ctx->runtime->deserializeCudaEngine(engine_data.data(), (size_t) engine_size);
    if (!ctx->engine) {
        fprintf(stderr, "[TextEnc-TRT] Failed to deserialize engine %s\n", engine_path);
        return false;
    }

    // Resolve + validate the I/O contract. The output dtype (FP16/BF16/FP32)
    // is auto-detected and stored in ctx->io_dtype; the runtime adapts the
    // download + conversion path accordingly.
    bool saw_input = false, saw_output = false;
    int  num_io = ctx->engine->getNbIOTensors();
    nvinfer1::DataType detected_dtype = nvinfer1::DataType::kHALF;  // default
    for (int i = 0; i < num_io; i++) {
        const char * name  = ctx->engine->getIOTensorName(i);
        auto         dtype = ctx->engine->getTensorDataType(name);
        auto         mode  = ctx->engine->getTensorIOMode(name);
        if (std::string(name) == "input_ids") {
            saw_input = true;
            if (mode != nvinfer1::TensorIOMode::kINPUT || dtype != nvinfer1::DataType::kINT64) {
                fprintf(stderr, "[TextEnc-TRT] input_ids has unexpected mode/dtype\n");
                return false;
            }
        } else if (std::string(name) == "hidden_states") {
            saw_output = true;
            if (mode != nvinfer1::TensorIOMode::kOUTPUT ||
                (dtype != nvinfer1::DataType::kHALF &&
                 dtype != nvinfer1::DataType::kBF16 &&
                 dtype != nvinfer1::DataType::kFLOAT)) {
                fprintf(stderr, "[TextEnc-TRT] hidden_states has unexpected mode/dtype\n");
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
        ctx->io_dtype = TEXT_ENC_IO_FP16;
    } else if (detected_dtype == nvinfer1::DataType::kBF16) {
        ctx->io_dtype = TEXT_ENC_IO_BF16;
    } else {
        ctx->io_dtype = TEXT_ENC_IO_FP32;
    }
    {
        const char * dtype_names[] = {"FP16", "BF16", "FP32"};
        fprintf(stderr, "[TextEnc-TRT] I/O dtype: %s\n", dtype_names[ctx->io_dtype]);
    }
    if (!saw_input || !saw_output) {
        fprintf(stderr, "[TextEnc-TRT] Engine missing input_ids/hidden_states bindings\n");
        return false;
    }

    if (!ctx->stream) {
        cudaStreamCreate(&ctx->stream);
    }

    auto t1 = std::chrono::steady_clock::now();
    ctx->load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "[TextEnc-TRT] Deserialized %s (%lld bytes, H=%d, %lld ms)\n",
            engine_path, (long long)engine_size, ctx->hidden_size, (long long) ctx->load_time_ms);
    fflush(stderr);
    return true;
}

// ── Forward ─────────────────────────────────────────────────────────────────

// Run the text encoder on host int32 token IDs.
//   token_ids: host int32 [B*S] (matches the GGML qwen3_forward input contract)
//   B, S:      batch + sequence length (profile bounds: B==1, 1<=S<=512)
//   out_f32:   host float32 [B*S*H], written in [B, S, H] row-major
// Returns false on any TRT/CUDA failure. Reallocates context + device buffers
// on demand so it works directly after text_enc_trt_release_evictable.
inline bool text_enc_trt_forward(TextEncTrt * ctx,
                                 const int * token_ids,
                                 int         B,
                                 int         S,
                                 float *     out_f32) {
    if (!ctx || !ctx->engine) {
        fprintf(stderr, "[TextEnc-TRT] FATAL: forward called before load\n");
        return false;
    }
    if (B <= 0 || S <= 0 || !token_ids || !out_f32) {
        fprintf(stderr, "[TextEnc-TRT] forward: invalid args (B=%d S=%d)\n", B, S);
        return false;
    }

    const int    H        = ctx->hidden_size;
    const size_t n_tokens = (size_t) B * (size_t) S;
    const size_t n_hidden = n_tokens * (size_t) H;
    const size_t in_bytes  = n_tokens * sizeof(int64_t);
    // Element size depends on engine I/O dtype (FP16/BF16 = 2 bytes, FP32 = 4 bytes).
    const size_t out_elem_bytes = (ctx->io_dtype == TEXT_ENC_IO_FP32) ? sizeof(float) : sizeof(uint16_t);
    const size_t out_bytes = n_hidden * out_elem_bytes;

    if (!ctx->context) {
        ctx->context = ctx->engine->createExecutionContext();
        if (!ctx->context) {
            fprintf(stderr, "[TextEnc-TRT] Failed to (re)create execution context\n");
            return false;
        }
        fprintf(stderr, "[TextEnc-TRT] forward: execution context allocated\n");
    }

    // (Re)allocate device buffers when missing or too small.
    if (ctx->d_input_ids == nullptr || ctx->buf_input_bytes < in_bytes) {
        if (ctx->d_input_ids) cudaFree(ctx->d_input_ids);
        if (cudaMalloc(&ctx->d_input_ids, in_bytes) != cudaSuccess) {
            fprintf(stderr, "[TextEnc-TRT] cudaMalloc input failed (%zu bytes)\n", in_bytes);
            ctx->d_input_ids = nullptr;
            ctx->buf_input_bytes = 0;
            return false;
        }
        ctx->buf_input_bytes = in_bytes;
        fprintf(stderr, "[TextEnc-TRT] forward: input buffer allocated (%zu bytes)\n", in_bytes);
    }
    if (ctx->d_hidden == nullptr || ctx->buf_hidden_bytes < out_bytes) {
        if (ctx->d_hidden) cudaFree(ctx->d_hidden);
        if (cudaMalloc(&ctx->d_hidden, out_bytes) != cudaSuccess) {
            fprintf(stderr, "[TextEnc-TRT] cudaMalloc output failed (%zu bytes)\n", out_bytes);
            ctx->d_hidden = nullptr;
            ctx->buf_hidden_bytes = 0;
            return false;
        }
        ctx->buf_hidden_bytes = out_bytes;
        fprintf(stderr, "[TextEnc-TRT] forward: output buffer allocated (%zu bytes)\n", out_bytes);
    }

    // Upload token IDs as INT64 (the engine's input dtype).
    std::vector<int64_t> ids64(n_tokens);
    for (size_t i = 0; i < n_tokens; i++) {
        ids64[i] = (int64_t) token_ids[i];
    }
    if (cudaMemcpyAsync(ctx->d_input_ids, ids64.data(), in_bytes,
                        cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess) {
        fprintf(stderr, "[TextEnc-TRT] H2D copy of input_ids failed\n");
        return false;
    }

    if (!ctx->context->setInputShape("input_ids", nvinfer1::Dims2(B, S))) {
        fprintf(stderr, "[TextEnc-TRT] setInputShape failed (B=%d S=%d)\n", B, S);
        return false;
    }
    if (!ctx->context->setTensorAddress("input_ids", ctx->d_input_ids) ||
        !ctx->context->setTensorAddress("hidden_states", ctx->d_hidden)) {
        fprintf(stderr, "[TextEnc-TRT] setTensorAddress failed\n");
        return false;
    }
    if (!ctx->context->enqueueV3(ctx->stream)) {
        fprintf(stderr, "[TextEnc-TRT] enqueueV3 failed\n");
        return false;
    }

    // Pull engine-dtype output to host, convert to FP32 in [B,S,H] order.
    if (ctx->io_dtype == TEXT_ENC_IO_FP32) {
        // No conversion — download FP32 directly into the output buffer.
        if (cudaMemcpyAsync(out_f32, ctx->d_hidden, out_bytes,
                            cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess) {
            fprintf(stderr, "[TextEnc-TRT] D2H copy of hidden_states failed\n");
            return false;
        }
        if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) {
            fprintf(stderr, "[TextEnc-TRT] stream sync failed\n");
            return false;
        }
        return true;
    }

    // FP16 or BF16: download as uint16, then convert to FP32.
    std::vector<uint16_t> half(n_hidden);
    if (cudaMemcpyAsync(half.data(), ctx->d_hidden, out_bytes,
                        cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess) {
        fprintf(stderr, "[TextEnc-TRT] D2H copy of hidden_states failed\n");
        return false;
    }
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) {
        fprintf(stderr, "[TextEnc-TRT] stream sync failed\n");
        return false;
    }

    if (ctx->io_dtype == TEXT_ENC_IO_BF16) {
        // BF16 has FP32-equivalent range (±3.4e38) — no overflow possible.
        // Just widen to FP32 (zero-extend upper 16 bits).
        for (size_t i = 0; i < n_hidden; i++) {
            out_f32[i] = text_enc_trt_bf16_to_float(half[i]);
        }
        return true;
    }

    // FP16 path: convert to FP32. Note: text-enc-trt.h does NOT sanitize
    // NaN/Inf (unlike cond-enc-trt.h). This is the legacy behavior — FP16
    // engines may produce corrupted values if intermediate activations
    // overflow. BF16/FP32 engines (the preferred options) skip this path.
    for (size_t i = 0; i < n_hidden; i++) {
        const uint16_t h = half[i];
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
        memcpy(&out_f32[i], &u, sizeof(u));
    }
    return true;
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

inline void text_enc_trt_free(TextEncTrt * ctx) {
    if (!ctx) return;
    text_enc_trt_release_evictable(ctx);
    if (ctx->stream)  { cudaStreamDestroy(ctx->stream); ctx->stream = nullptr; }
    if (ctx->engine)  { delete ctx->engine;  ctx->engine = nullptr; }
    if (ctx->runtime) { delete ctx->runtime; ctx->runtime = nullptr; }
    fprintf(stderr, "[TextEnc-TRT] Engine fully unloaded\n");
}

#endif  // HOT_STEP_TRT
