#pragma once
// text-enc-ort.h: ONNX Runtime text encoder wrapper
//
// Wraps the Qwen3-Embedding-0.6B ONNX model for text encoding.
// Input:  token_ids [S] int32 (BPE tokens)
// Output: hidden_states [S * 1024] float (same layout as qwen3_forward)
//
// Also provides embed_lookup from a raw binary table (for lyric tokens).
//
// Part of HOT-Step CPP ONNX/TRT pipeline.

#ifndef TEXT_ENC_ORT_H
#define TEXT_ENC_ORT_H

#include "ort-trt-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef HOT_STEP_SUPERSEP

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>

// ── Text Encoder ORT context ──────────────────────────────────────────

struct TextEncOrt {
    Ort::Env            env;
    Ort::Session *      session;
    Ort::SessionOptions session_opts;
    std::string         model_path;
    bool                using_trt;
    int                 hidden_size;  // 1024 for Qwen3-0.6B

    // Embedding table for lyric lookup (loaded from embed_tokens.bin)
    std::vector<float>  embed_table;  // [vocab_size * hidden_size]
    int                 vocab_size;

    TextEncOrt() : env(ORT_LOGGING_LEVEL_WARNING, "text-enc-ort"),
                   session(nullptr), using_trt(false),
                   hidden_size(1024), vocab_size(0) {}
    ~TextEncOrt() { delete session; }
};

// ── Load ──────────────────────────────────────────────────────────────

static inline bool text_enc_ort_load(TextEncOrt * ctx, const char * onnx_path,
                                     const char * embed_bin_path = nullptr,
                                     int device_id = 0,
                                     const char * artifact_fingerprint = nullptr) {
    if (!ctx || !onnx_path) return false;

    ctx->model_path = onnx_path;
    ctx->session_opts.SetIntraOpNumThreads(1);
    ctx->session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ── TensorRT EP ────────────────────────────────────────────────
#if defined(GGML_USE_CUDA)
    std::string trt_cache_dir = hs_ort_trt_cache_dir(onnx_path, artifact_fingerprint, "text-enc");
    if (!hs_ort_trt_cache_ready(trt_cache_dir)) {
        fprintf(stderr, "[TextEnc-ORT] FATAL: prebuilt TensorRT cache missing or incomplete: %s\n",
                trt_cache_dir.c_str());
        fprintf(stderr, "[TextEnc-ORT] Build it offline with tools/onnx-export/build-ort-trt-engines.py\n");
        return false;
    }

    {
        try {
            Ort::TensorRTProviderOptions trt_opts;
            auto opts = hs_ort_trt_provider_options(
                onnx_path, artifact_fingerprint, "text-enc", device_id, false, (size_t)2 << 30);
            trt_opts.Update(opts);
            ctx->session_opts.AppendExecutionProvider_TensorRT_V2(*trt_opts);
            ctx->using_trt = true;
            fprintf(stderr, "[TextEnc-ORT] TRT EP appended (device %d, cache=%s, profile_max=%s)\n",
                    device_id, trt_cache_dir.c_str(), opts["trt_profile_max_shapes"].c_str());
        } catch (const std::exception & e) {
            fprintf(stderr, "[TextEnc-ORT] FATAL: TensorRT EP unavailable for prebuilt bundle: %s\n", e.what());
            ctx->using_trt = false;
            return false;
        }
    }

    try {
        OrtCUDAProviderOptions cuda_opts;
        memset(&cuda_opts, 0, sizeof(cuda_opts));
        cuda_opts.device_id             = device_id;
        cuda_opts.arena_extend_strategy = 1;
        ctx->session_opts.AppendExecutionProvider_CUDA(cuda_opts);
        fprintf(stderr, "[TextEnc-ORT] CUDA EP appended (device %d)\n", device_id);
    } catch (const std::exception & e) {
        fprintf(stderr, "[TextEnc-ORT] CUDA EP failed: %s — CPU fallback\n", e.what());
    }
#else
    fprintf(stderr, "[TextEnc-ORT] No GPU EP — CPU only\n");
#endif

    // ── Create session ─────────────────────────────────────────────
    try {
#ifdef _WIN32
        int wlen = MultiByteToWideChar(CP_UTF8, 0, onnx_path, -1, nullptr, 0);
        std::vector<wchar_t> wpath(wlen);
        MultiByteToWideChar(CP_UTF8, 0, onnx_path, -1, wpath.data(), wlen);
        ctx->session = new Ort::Session(ctx->env, wpath.data(), ctx->session_opts);
#else
        ctx->session = new Ort::Session(ctx->env, onnx_path, ctx->session_opts);
#endif
        fprintf(stderr, "[TextEnc-ORT] Session created: %s (TRT=%s)\n",
                onnx_path, ctx->using_trt ? "yes" : "no");
    } catch (const std::exception & e) {
        fprintf(stderr, "[TextEnc-ORT] FATAL: session creation failed: %s\n", e.what());
        ctx->session = nullptr;
        return false;
    }

    // ── Load embedding table for lyric lookup ──────────────────────
    if (embed_bin_path && embed_bin_path[0]) {
        FILE * f = fopen(embed_bin_path, "rb");
        if (f) {
            uint32_t V = 0, H = 0;
            if (fread(&V, 4, 1, f) == 1 && fread(&H, 4, 1, f) == 1) {
                ctx->vocab_size = (int)V;
                ctx->hidden_size = (int)H;
                ctx->embed_table.resize((size_t)V * H);
                size_t read = fread(ctx->embed_table.data(), sizeof(float), (size_t)V * H, f);
                if (read == (size_t)V * H) {
                    fprintf(stderr, "[TextEnc-ORT] Embedding table: [%d, %d] from %s\n",
                            V, H, embed_bin_path);
                } else {
                    fprintf(stderr, "[TextEnc-ORT] WARNING: embed table truncated\n");
                    ctx->embed_table.clear();
                }
            }
            fclose(f);
        } else {
            fprintf(stderr, "[TextEnc-ORT] WARNING: cannot open %s — lyric embed lookup disabled\n",
                    embed_bin_path);
        }
    }

    return true;
}

static inline void text_enc_ort_free(TextEncOrt * ctx) {
    if (!ctx) return;
    delete ctx->session;
    ctx->session = nullptr;
    ctx->embed_table.clear();
}

// ── Forward: token IDs → hidden states ────────────────────────────────
//
// token_ids: [S] int32
// output:    resized to [H * S] float, ggml layout (H contiguous, S rows)
// Returns 0 on success, -1 on error.

static inline int text_enc_ort_forward(TextEncOrt *           ctx,
                                       const int *            token_ids,
                                       int                    S,
                                       std::vector<float> &   output) {
    if (!ctx || !ctx->session || !token_ids || S <= 0) return -1;

    int H = ctx->hidden_size;

    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Input: [1, S] int64 (ONNX expects int64 for input_ids)
    std::vector<int64_t> ids64(S);
    for (int i = 0; i < S; i++) ids64[i] = (int64_t)token_ids[i];

    std::vector<int64_t> input_shape = {1, (int64_t)S};
    auto input_tensor = Ort::Value::CreateTensor<int64_t>(
        mem, ids64.data(), ids64.size(),
        input_shape.data(), input_shape.size());

    auto in_name  = ctx->session->GetInputNameAllocated(0, alloc);
    auto out_name = ctx->session->GetOutputNameAllocated(0, alloc);
    const char * in_names[]  = { in_name.get() };
    const char * out_names[] = { out_name.get() };

    std::vector<Ort::Value> outputs;
    try {
        outputs = ctx->session->Run(
            Ort::RunOptions{nullptr},
            in_names, &input_tensor, 1,
            out_names, 1);
    } catch (const std::exception & e) {
        fprintf(stderr, "[TextEnc-ORT] Inference failed: %s\n", e.what());
        return -1;
    }

    // Output: [1, S, H] — need to extract to [H * S] ggml layout
    auto & out_tensor = outputs[0];
    auto   out_shape  = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const float * out_data = out_tensor.GetTensorData<float>();

    // ONNX output is [1, S, H] row-major = H-contiguous per token
    // GGML layout is also H-contiguous per token (ne[0]=H, ne[1]=S)
    // So they match! Just copy directly.
    output.resize((size_t)H * S);
    memcpy(output.data(), out_data, (size_t)H * S * sizeof(float));

    return 0;
}

// ── Embedding lookup (for lyric tokens) ───────────────────────────────
//
// Pure CPU operation using the embed_tokens.bin table.
// token_ids: [S] int32
// output:    resized to [H * S] float, ggml layout (H contiguous, S rows)
// Returns 0 on success, -1 on error.

static inline int text_enc_ort_embed_lookup(TextEncOrt *         ctx,
                                            const int *          token_ids,
                                            int                  S,
                                            std::vector<float> & output) {
    if (!ctx || ctx->embed_table.empty() || !token_ids || S <= 0) return -1;

    int H = ctx->hidden_size;
    int V = ctx->vocab_size;

    output.resize((size_t)H * S);
    for (int i = 0; i < S; i++) {
        int id = token_ids[i];
        if (id < 0 || id >= V) {
            // Out of range: zero vector
            memset(output.data() + (size_t)i * H, 0, (size_t)H * sizeof(float));
        } else {
            memcpy(output.data() + (size_t)i * H,
                   ctx->embed_table.data() + (size_t)id * H,
                   (size_t)H * sizeof(float));
        }
    }

    return 0;
}

#else  // !HOT_STEP_SUPERSEP — stubs

struct TextEncOrt {
    int hidden_size = 1024;
};

static inline bool text_enc_ort_load(TextEncOrt *, const char *, const char * = nullptr, int = 0,
                                     const char * = nullptr) {
    fprintf(stderr, "[TextEnc-ORT] Not compiled (HOT_STEP_SUPERSEP not defined)\n");
    return false;
}

static inline void text_enc_ort_free(TextEncOrt *) {}

static inline int text_enc_ort_forward(TextEncOrt *, const int *, int, std::vector<float> &) {
    return -1;
}

static inline int text_enc_ort_embed_lookup(TextEncOrt *, const int *, int, std::vector<float> &) {
    return -1;
}

#endif  // HOT_STEP_SUPERSEP

#endif  // TEXT_ENC_ORT_H
