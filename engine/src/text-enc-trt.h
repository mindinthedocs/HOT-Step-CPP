#pragma once
// text-enc-trt.h — TensorRT native loader for the Qwen3 text encoder.
//
// The text-encoder engine is a strongly-typed FP16 graph with weights EMBEDDED
// in the plan (NOT kREFIT_IDENTICAL like dit-trt.h). So this
// loader only deserializes — there is no ONNX sidecar, no refitter, no weight
// streaming. The embed lookup is baked into the graph: the engine consumes
// token IDs directly, not pre-looked-up embeddings.
//
// Engine I/O contract (from text_encoder.engine.metadata.json, verified by
// gen_text_enc_fixture.py):
//   input  "input_ids"     INT64 [B, S]        (token IDs; embed baked in)
//   output "hidden_states" HALF  [B, S, 1024]  (last_hidden_state)
//   profile: min [1,1], opt [1,128], max [1,512]
//
// Output handoff. text_enc_trt_forward writes FP32 hidden states into a caller
// host buffer in [B, S, H] row-major order. For B=1 this byte order matches the
// GGML text encoder's qwen3_forward output ([H,S] ggml layout = S-major,
// H-contiguous), so the TRT path can swap in where qwen3_forward is called today.
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

struct TextEncTrt {
    // TRT runtime objects.
    nvinfer1::IRuntime *    runtime = nullptr;
    nvinfer1::ICudaEngine * engine  = nullptr;

    // Evictable: freed by release_evictable, reallocated by forward.
    nvinfer1::IExecutionContext * context = nullptr;
    void *  d_input_ids    = nullptr;  // INT64 [B*S]
    void *  d_hidden       = nullptr;  // HALF  [B*S*H]
    size_t  buf_input_bytes  = 0;
    size_t  buf_hidden_bytes = 0;

    int  hidden_size = 1024;  // resolved from the output binding at load

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

    // Resolve + validate the I/O contract against the embedded-weight FP16 graph.
    bool saw_input = false, saw_output = false;
    int  num_io = ctx->engine->getNbIOTensors();
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
            if (mode != nvinfer1::TensorIOMode::kOUTPUT || dtype != nvinfer1::DataType::kHALF) {
                fprintf(stderr, "[TextEnc-TRT] hidden_states has unexpected mode/dtype\n");
                return false;
            }
            nvinfer1::Dims d = ctx->engine->getTensorShape(name);
            if (d.nbDims == 3 && d.d[2] > 0) {
                ctx->hidden_size = (int) d.d[2];
            }
        }
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
    const size_t out_bytes = n_hidden * sizeof(uint16_t);

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

    // Pull FP16 output to host, convert to FP32 in [B,S,H] order.
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
