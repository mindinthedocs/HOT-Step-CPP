#pragma once
// cond-enc-ort.h: ONNX Runtime condition encoder wrapper
//
// Wraps the AceStep condition encoder ONNX model.
// Input:  text_hidden [S_text, 1024], lyric_embed [S_lyric, 1024],
//         timbre_feats [S_ref, 64]
// Output: enc_hidden [S_total, 2048] where S_total = S_lyric + 1 + S_text
//
// Also loads null_condition_emb from binary file.
//
// Part of HOT-Step CPP ONNX/TRT pipeline.

#ifndef COND_ENC_ORT_H
#define COND_ENC_ORT_H

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

// ── Condition Encoder ORT context ─────────────────────────────────────

struct CondEncOrt {
    Ort::Env            env;
    Ort::Session *      session;
    Ort::SessionOptions session_opts;
    std::string         model_path;
    bool                using_trt;
    int                 hidden_size;  // 2048 for AceStep encoder output

    // null_condition_emb [hidden_size] — for CFG padding
    std::vector<float>  null_cond_emb;

    CondEncOrt() : env(ORT_LOGGING_LEVEL_WARNING, "cond-enc-ort"),
                   session(nullptr), using_trt(false),
                   hidden_size(2048) {}
    ~CondEncOrt() { delete session; }
};

// ── Load ──────────────────────────────────────────────────────────────

static inline bool cond_enc_ort_load(CondEncOrt * ctx, const char * onnx_path,
                                     const char * null_cond_bin_path = nullptr,
                                     int device_id = 0,
                                     const char * artifact_fingerprint = nullptr) {
    if (!ctx || !onnx_path) return false;

    ctx->model_path = onnx_path;
    ctx->session_opts.SetIntraOpNumThreads(1);
    ctx->session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ── TensorRT EP ────────────────────────────────────────────────
#if defined(GGML_USE_CUDA)
    std::string trt_cache_dir = hs_ort_trt_cache_dir(onnx_path, artifact_fingerprint, "cond-enc");
    if (!hs_ort_trt_cache_ready(trt_cache_dir)) {
        fprintf(stderr, "[CondEnc-ORT] FATAL: prebuilt TensorRT cache missing or incomplete: %s\n",
                trt_cache_dir.c_str());
        fprintf(stderr, "[CondEnc-ORT] Build it offline with tools/onnx-export/build-ort-trt-engines.py\n");
        return false;
    }

    {
        try {
            Ort::TensorRTProviderOptions trt_opts;
            auto opts = hs_ort_trt_provider_options(
                onnx_path, artifact_fingerprint, "cond-enc", device_id, false, (size_t)2 << 30);
            trt_opts.Update(opts);
            ctx->session_opts.AppendExecutionProvider_TensorRT_V2(*trt_opts);
            ctx->using_trt = true;
            fprintf(stderr, "[CondEnc-ORT] TRT EP appended (device %d, cache=%s, profile_max=%s)\n",
                    device_id, trt_cache_dir.c_str(), opts["trt_profile_max_shapes"].c_str());
        } catch (const std::exception & e) {
            fprintf(stderr, "[CondEnc-ORT] FATAL: TensorRT EP unavailable for prebuilt bundle: %s\n", e.what());
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
        fprintf(stderr, "[CondEnc-ORT] CUDA EP appended (device %d)\n", device_id);
    } catch (const std::exception & e) {
        fprintf(stderr, "[CondEnc-ORT] CUDA EP failed: %s — CPU fallback\n", e.what());
    }
#else
    fprintf(stderr, "[CondEnc-ORT] No GPU EP — CPU only\n");
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
        fprintf(stderr, "[CondEnc-ORT] Session created: %s (TRT=%s)\n",
                onnx_path, ctx->using_trt ? "yes" : "no");
    } catch (const std::exception & e) {
        fprintf(stderr, "[CondEnc-ORT] FATAL: session creation failed: %s\n", e.what());
        ctx->session = nullptr;
        return false;
    }

    // ── Load null_condition_emb ────────────────────────────────────
    if (null_cond_bin_path && null_cond_bin_path[0]) {
        FILE * f = fopen(null_cond_bin_path, "rb");
        if (f) {
            uint32_t dim = 0;
            if (fread(&dim, 4, 1, f) == 1 && dim > 0 && dim <= 8192) {
                ctx->hidden_size = (int)dim;
                ctx->null_cond_emb.resize(dim);
                fread(ctx->null_cond_emb.data(), sizeof(float), dim, f);
                fprintf(stderr, "[CondEnc-ORT] null_condition_emb: [%d] from %s\n",
                        dim, null_cond_bin_path);
            }
            fclose(f);
        } else {
            fprintf(stderr, "[CondEnc-ORT] WARNING: cannot open %s\n", null_cond_bin_path);
        }
    }

    return true;
}

static inline void cond_enc_ort_free(CondEncOrt * ctx) {
    if (!ctx) return;
    delete ctx->session;
    ctx->session = nullptr;
    ctx->null_cond_emb.clear();
}

// ── Forward ───────────────────────────────────────────────────────────
//
// text_hidden:  [S_text * H_text] float (H_text=1024, ggml layout)
// lyric_embed:  [S_lyric * H_text] float
// timbre_feats: [S_ref * 64] float (or NULL for no reference)
// S_ref:        0 if no reference audio
//
// enc_hidden:   output resized to [S_total * H_enc] float (H_enc=2048)
// out_enc_S:    set to S_total
//
// Returns 0 on success, -1 on error.

static inline int cond_enc_ort_forward(CondEncOrt *          ctx,
                                       const float *         text_hidden,
                                       int                   S_text,
                                       const float *         lyric_embed,
                                       int                   S_lyric,
                                       const float *         timbre_feats,
                                       int                   S_ref,
                                       std::vector<float> &  enc_hidden,
                                       int *                 out_enc_S) {
    if (!ctx || !ctx->session || !text_hidden || !lyric_embed) return -1;

    int H_text = 1024;  // text encoder hidden size
    int H_enc  = ctx->hidden_size;  // 2048

    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // If no timbre, provide 1 frame of zeros (ONNX model always expects timbre input)
    std::vector<float> zero_timbre;
    if (!timbre_feats || S_ref <= 0) {
        S_ref = 1;
        zero_timbre.assign(64, 0.0f);
        timbre_feats = zero_timbre.data();
    }

    // Input tensors: [1, S, D]
    // text_hidden is [S_text * H_text] ggml layout → reshape to [1, S_text, H_text]
    std::vector<int64_t> text_shape   = {1, (int64_t)S_text, (int64_t)H_text};
    std::vector<int64_t> lyric_shape  = {1, (int64_t)S_lyric, (int64_t)H_text};
    std::vector<int64_t> timbre_shape = {1, (int64_t)S_ref, 64};

    auto text_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float *>(text_hidden), (size_t)S_text * H_text,
        text_shape.data(), text_shape.size());
    auto lyric_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float *>(lyric_embed), (size_t)S_lyric * H_text,
        lyric_shape.data(), lyric_shape.size());
    auto timbre_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float *>(timbre_feats), (size_t)S_ref * 64,
        timbre_shape.data(), timbre_shape.size());

    // I/O names — the exported ONNX has exactly 3 inputs and 1 output
    // with known names from the export script. Use fixed arrays.
    const char * in_names[]  = { "text_hidden", "lyric_embed", "timbre_feats" };
    const char * out_names[] = { "enc_hidden" };
    size_t n_inputs = 3;
    size_t n_outputs = 1;

    Ort::Value inputs[] = { std::move(text_tensor), std::move(lyric_tensor), std::move(timbre_tensor) };

    std::vector<Ort::Value> outputs;
    try {
        outputs = ctx->session->Run(
            Ort::RunOptions{nullptr},
            in_names, inputs, n_inputs,
            out_names, n_outputs);
    } catch (const std::exception & e) {
        fprintf(stderr, "[CondEnc-ORT] Inference failed: %s\n", e.what());
        return -1;
    }

    // Output: [1, S_total, H_enc]
    auto & out_tensor = outputs[0];
    auto   out_shape  = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const float * out_data = out_tensor.GetTensorData<float>();

    int S_total = (out_shape.size() >= 2) ? (int)out_shape[1] : 0;
    if (S_total <= 0) {
        fprintf(stderr, "[CondEnc-ORT] ERROR: unexpected output shape\n");
        return -1;
    }

    // Copy to output: [S_total * H_enc] ggml layout
    enc_hidden.resize((size_t)S_total * H_enc);
    memcpy(enc_hidden.data(), out_data, (size_t)S_total * H_enc * sizeof(float));

    if (out_enc_S) *out_enc_S = S_total;

    fprintf(stderr, "[CondEnc-ORT] Forward: text=%d + lyric=%d + ref=%d → enc_S=%d (TRT=%s)\n",
            S_text, S_lyric, S_ref, S_total, ctx->using_trt ? "yes" : "no");

    return 0;
}

#else  // !HOT_STEP_SUPERSEP — stubs

struct CondEncOrt {
    int hidden_size = 2048;
    std::vector<float> null_cond_emb;
};

static inline bool cond_enc_ort_load(CondEncOrt *, const char *, const char * = nullptr, int = 0,
                                     const char * = nullptr) {
    fprintf(stderr, "[CondEnc-ORT] Not compiled (HOT_STEP_SUPERSEP not defined)\n");
    return false;
}

static inline void cond_enc_ort_free(CondEncOrt *) {}

static inline int cond_enc_ort_forward(CondEncOrt *, const float *, int,
                                       const float *, int,
                                       const float *, int,
                                       std::vector<float> &, int *) {
    return -1;
}

#endif  // HOT_STEP_SUPERSEP

#endif  // COND_ENC_ORT_H
