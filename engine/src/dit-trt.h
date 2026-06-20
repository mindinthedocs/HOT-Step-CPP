#pragma once
// dit-trt.h — TensorRT native API wrapper for DiT inference + LoRA refitting
//
// Uses TRT directly (not ORT) because:
//  - IRefitter is required for runtime LoRA adapter switching
//  - ORT's TRT EP doesn't expose kREFIT_IDENTICAL or IRefitter API
//
// Architecture:
//  BUILD (once per GPU arch):
//    ONNX -> TRT engine with kREFIT_IDENTICAL (weights embedded)
//    → full engine file with all weights (~5GB)
//
//  RUNTIME:
//    Load engine → cache base weights via ONNX refitter → Run
//    On adapter switch: refit with merged weights (~0.5s)

#ifdef HOT_STEP_TRT

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <atomic>

// TRT headers
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "yyjson.h"

// dlopen / LoadLibrary for the HOT-Step plugin .so — included AFTER TRT headers
// to avoid macro conflicts (ERROR, NO_ERROR, DELETE, etc.) from windows.h.
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#if NV_TENSORRT_MAJOR != 11
#error "HOT_STEP_TRT requires TensorRT 11.x headers"
#endif

// ── TRT Logger ──────────────────────────────────────────────────────────────

class DitTrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Skip INFO/VERBOSE unless debugging
        if (severity > Severity::kWARNING) return;
        const char* prefix = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: prefix = "[TRT-INTERNAL] "; break;
            case Severity::kERROR:          prefix = "[TRT-ERROR] ";    break;
            case Severity::kWARNING:        prefix = "[TRT-WARN] ";     break;
            default: break;
        }
        fprintf(stderr, "%s%s\n", prefix, msg);
    }
};

inline void dit_trt_log_version(const char* where) {
    fprintf(stderr, "[DiT-TRT] TensorRT %d.%d.%d (%s)\n",
            NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
            where ? where : "runtime");
}

// ── DitTrt context ──────────────────────────────────────────────────────────

struct DitTrt {
    // TRT objects
    nvinfer1::IRuntime*            runtime  = nullptr;
    nvinfer1::ICudaEngine*         engine   = nullptr;
    nvinfer1::IExecutionContext*   context  = nullptr;

    // I/O tensor indices (resolved once at load time)
    // Inputs:  input_latents[B,T,192], enc_hidden[B,S,2048], t[B], t_r[B],
    //          attention_mask[B,T] (int64), encoder_attention_mask[B,S] (int64)
    // Outputs: velocity[B,T,64]
    int idx_input_latents = -1;
    int idx_enc_hidden    = -1;
    int idx_t             = -1;
    int idx_t_r           = -1;
    int idx_attention_mask         = -1;  // int64 [B,T] — self-attn padding
    int idx_encoder_attention_mask = -1;  // int64 [B,S] — cross-attn padding
    int idx_velocity      = -1;
    nvinfer1::DataType dtype_input_latents = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType dtype_enc_hidden    = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType dtype_velocity      = nvinfer1::DataType::kFLOAT;

    // Device buffers (allocated lazily, resized as needed)
    void*  d_input_latents = nullptr;
    void*  d_enc_hidden    = nullptr;
    void*  d_t             = nullptr;
    void*  d_t_r           = nullptr;
    void*  d_velocity      = nullptr;
    size_t buf_input_latents_bytes = 0;
    size_t buf_enc_hidden_bytes    = 0;
    size_t buf_velocity_bytes      = 0;

    // ONNX path (needed for weight refitting / adapter base cache)
    std::string onnx_path;

    // TensorRT weight streaming metadata (budget = full size so all weights
    // are resident in VRAM; streaming is enabled at build time to avoid OOM
    // during engine compilation, but at runtime nothing is streamed).
    int64_t streamable_weights_bytes = 0;
    int64_t weight_streaming_budget_bytes = 0;

    // Base weight cache (BF16 host memory, keyed by TRT weight name)
    // Populated on first load; used to revert adapter changes
    std::unordered_map<std::string, std::vector<uint16_t>> base_weights;

    // Current adapter state
    std::string current_adapter;  // empty = base model
    std::string current_adapter_key;  // path + scale + group scales + sidecar hash
    std::mutex  refit_mutex;

    // Weights that dynamo stored in transposed [in,out] orientation
    // (vs torch nn.Linear's [out,in]). LoRA deltas arrive in torch
    // orientation and must be transposed before adding to these base weights.
    // Loaded from refit_manifest.json sidecar emitted by export_dit.py.
    std::unordered_set<std::string> weights_transposed;

    // I/O tensor dtype for host ↔ GPU conversion.
    // bf16: dynamo bf16_mixed export (standard path, needs bf16↔fp32 staging)
    // fp16: hypothetical fp16 export (needs fp16↔fp32 staging)
    // fp32: FP8 QDQ model from modelopt (no staging needed, direct upload)
    enum IODtype { IO_BF16, IO_FP16, IO_FP32 };
    IODtype io_dtype = IO_BF16;

    // Logger
    DitTrtLogger logger;

    // Dedicated CUDA stream (avoids default-stream synchronisation penalties)
    cudaStream_t stream = nullptr;

    // Stats
    int64_t build_time_ms = 0;
    int64_t load_time_ms  = 0;
};

// Release per-job GPU buffers without destroying the TRT runtime, engine,
// execution context, or refitted weights.
inline void dit_trt_release_evictable(DitTrt* ctx) {
    if (!ctx) return;
    bool released = false;
    if (ctx->d_input_latents) { cudaFree(ctx->d_input_latents); ctx->d_input_latents = nullptr; released = true; }
    if (ctx->d_enc_hidden)    { cudaFree(ctx->d_enc_hidden);    ctx->d_enc_hidden = nullptr;    released = true; }
    if (ctx->d_t)             { cudaFree(ctx->d_t);             ctx->d_t = nullptr;             released = true; }
    if (ctx->d_t_r)           { cudaFree(ctx->d_t_r);           ctx->d_t_r = nullptr;           released = true; }
    if (ctx->d_velocity)      { cudaFree(ctx->d_velocity);      ctx->d_velocity = nullptr;      released = true; }
    ctx->buf_input_latents_bytes = 0;
    ctx->buf_enc_hidden_bytes    = 0;
    ctx->buf_velocity_bytes      = 0;
    fprintf(stderr, "[DiT-TRT] EVICT_STRICT: %s\n",
            released ? "released evictable buffers" : "no evictable buffers to release");
}

// ── Engine build (once per GPU architecture) ────────────────────────────────

// Build a TRT engine from ONNX and serialize to disk.
// Returns true on success.  engine_path will be created/overwritten.
//
// This is slow (5-30 minutes) but only needs to run once per GPU arch.
// The engine is built with:
//   - kREFIT_IDENTICAL: allows weight refitting with zero inference penalty
//   - kWEIGHT_STREAMING: avoids OOM during engine compilation
//   - strongly typed graph precision from ONNX
//   - weights embedded in engine file (no kSTRIP_PLAN)
inline bool dit_trt_build(
    const char* onnx_path,
    const char* engine_path,
    int         device_id = 0
) {
    DitTrtLogger logger;

    dit_trt_log_version("build");
    fprintf(stderr, "[DiT-TRT] Building engine from %s ...\n", onnx_path);
    fprintf(stderr, "[DiT-TRT] This will take 5-30 minutes (first run only).\n");
    auto t0 = std::chrono::steady_clock::now();

    // Set CUDA device
    cudaSetDevice(device_id);

    // Create builder
    auto builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        fprintf(stderr, "[DiT-TRT] Failed to create TRT builder\n");
        return false;
    }

    // Create network: strongly typed so TensorRT honors ONNX tensor dtypes,
    // typed initializers, and explicit Cast nodes from the precision policy.
    // STRONGLY_TYPED is mandatory in TRT 11; TRT honors the per-tensor dtypes
    // from the ONNX graph.
    uint32_t net_flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#if !defined(HOT_STEP_TRT_VERSION_MAJOR) || HOT_STEP_TRT_VERSION_MAJOR < 11
    net_flags |= 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    auto network = builder->createNetworkV2(net_flags);
    if (!network) {
        fprintf(stderr, "[DiT-TRT] Failed to create network\n");
        delete builder;
        return false;
    }
    fprintf(stderr, "[DiT-TRT] STRONGLY_TYPED network (ONNX precision policy)\n");

    // Parse ONNX
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnx_path,
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        fprintf(stderr, "[DiT-TRT] ONNX parse failed\n");
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Note: TRT 11 STRONGLY_TYPED forbids setPrecision/setOutputType/setType.
    // The "FP16 layernorm" warning is handled by TRT's internal optimizer which
    // auto-promotes Reduce/Pow to FP32 in layernorm patterns when it detects
    // overflow risk. If NaN persists, the ONNX must be re-exported with FP32 norms.

    // Builder config
    auto config = builder->createBuilderConfig();

    // STRONGLY_TYPED + TF32: graph types are authoritative (kFP16/kOBEY removed
    // in TRT 11). TF32 accelerates fp32 island ops on tensor cores.
    // No FP16/BF16 builder flags — STRONGLY_TYPED forbids them.
    config->setFlag(nvinfer1::BuilderFlag::kTF32);
    fprintf(stderr, "[DiT-TRT] STRONGLY_TYPED + TF32 (no global FP16/BF16 builder flags)\n");

    // Enable refittable engine (zero perf penalty with IDENTICAL)
    config->setFlag(nvinfer1::BuilderFlag::kREFIT_IDENTICAL);
    config->setFlag(nvinfer1::BuilderFlag::kWEIGHT_STREAMING);
    fprintf(stderr, "[DiT-TRT] kREFIT_IDENTICAL + kWEIGHT_STREAMING enabled (weights embedded in engine)\n");

    // Workspace (4 GB should be plenty for DiT)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               4ULL << 30);

    // Optimization profile with dynamic shapes
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        fprintf(stderr, "[DiT-TRT] Failed to create optimization profile\n");
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }
    auto set_profile_dims = [&](const char* name,
                                nvinfer1::OptProfileSelector selector,
                                nvinfer1::Dims dims,
                                const char* selector_label) -> bool {
        if (!profile->setDimensions(name, selector, dims)) {
            fprintf(stderr, "[DiT-TRT] Failed to set %s profile dims for %s\n",
                    selector_label, name);
            return false;
        }
        return true;
    };

    // input_latents: [B, T, 192]
    //   T ranges: min=64, opt=2048, max=8192
    if (!set_profile_dims("input_latents",
            nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 64, 192), "min") ||
        !set_profile_dims("input_latents",
            nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 2048, 192), "opt") ||
        !set_profile_dims("input_latents",
            nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2, 8192, 192), "max")) {
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // enc_hidden: [B, S, 2048]
    //   S ranges: min=64, opt=512, max=2048
    if (!set_profile_dims("enc_hidden",
            nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 64, 2048), "min") ||
        !set_profile_dims("enc_hidden",
            nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 2048), "opt") ||
        !set_profile_dims("enc_hidden",
            nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2, 2048, 2048), "max")) {
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // t: [B]
    if (!set_profile_dims("t",
            nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}}, "min") ||
        !set_profile_dims("t",
            nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {1}}, "opt") ||
        !set_profile_dims("t",
            nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {2}}, "max")) {
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // t_r: [B]
    if (!set_profile_dims("t_r",
            nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}}, "min") ||
        !set_profile_dims("t_r",
            nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {1}}, "opt") ||
        !set_profile_dims("t_r",
            nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {2}}, "max")) {
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // attention_mask: [B, T] int64 — self-attn padding mask.
    // Tied to the same T range as input_latents.
    if (!set_profile_dims("attention_mask",
            nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 64), "min") ||
        !set_profile_dims("attention_mask",
            nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 2048), "opt") ||
        !set_profile_dims("attention_mask",
            nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(2, 8192), "max")) {
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // encoder_attention_mask: [B, S] int64 — cross-attn padding mask.
    // Tied to the same S range as enc_hidden.
    if (!set_profile_dims("encoder_attention_mask",
            nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 64), "min") ||
        !set_profile_dims("encoder_attention_mask",
            nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 512), "opt") ||
        !set_profile_dims("encoder_attention_mask",
            nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(2, 2048), "max")) {
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    if (!profile->isValid()) {
        fprintf(stderr, "[DiT-TRT] Optimization profile is invalid\n");
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }
    if (config->addOptimizationProfile(profile) < 0) {
        fprintf(stderr, "[DiT-TRT] Failed to add optimization profile\n");
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Build serialized engine — this is the blocking call (5-30 min).
    // Launch a heartbeat thread to emit periodic log lines, preventing
    // the Node.js stall detector from killing the job.
    std::atomic<bool> build_done{false};
    std::thread heartbeat([&build_done, &t0]() {
        int tick = 0;
        while (!build_done.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            if (build_done.load(std::memory_order_relaxed)) break;
            tick++;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - t0).count();
            fprintf(stderr, "[DiT-TRT] Engine build in progress... (%llds elapsed)\n",
                    (long long)elapsed);
            fflush(stderr);
        }
    });

    auto serialized = builder->buildSerializedNetwork(*network, *config);

    build_done.store(true, std::memory_order_relaxed);
    heartbeat.join();

    if (!serialized) {
        fprintf(stderr, "[DiT-TRT] Engine build failed\n");
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Write to disk
    FILE* f = fopen(engine_path, "wb");
    if (!f) {
        fprintf(stderr, "[DiT-TRT] Cannot write to %s\n", engine_path);
        delete serialized;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }
    fwrite(serialized->data(), 1, serialized->size(), f);
    fclose(f);

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    fprintf(stderr, "[DiT-TRT] Engine saved to %s (%zu bytes, %.1f min)\n",
            engine_path, serialized->size(), ms / 60000.0);

    delete serialized;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true;
}

// ── Engine load + base weight refit ─────────────────────────────────────────

// Load the HOT-Step custom plugin library (libhotstep_plugins.so) so the
// ConvRotInt8Linear op is registered with TRT's plugin registry before any
// w8a8 engine is deserialized. The plugin .so is built by CMakeLists.txt
// (target: hotstep_plugins) and installed next to the engine binary.
//
// dlopen is idempotent: calling this multiple times is a no-op after the
// first successful load. The library stays resident for the process
// lifetime (we don't dlclose it) because TRT's plugin registry holds raw
// pointers into the .so.
//
// Returns true on success or if the plugin .so is already loaded. Returns
// false (and prints a diagnostic) if the .so cannot be found or the
// hotstep_register_plugins entry point fails.
inline bool dit_trt_load_hotstep_plugins() {
#if defined(_WIN32)
    static HMODULE g_plugin_handle = nullptr;
    if (g_plugin_handle) return true;

    // hotstep_plugins.dll depends on nvinfer_11.dll and cudart64_*.dll.
    // A bare LoadLibraryA("hotstep_plugins.dll") only searches the exe
    // directory, system dirs, and PATH — it won't find TRT DLLs in
    // %TENSORRT_ROOT%/bin/. Use AddDllDirectory to temporarily add the
    // TRT bin/ directory to the DLL search order, following the same
    // pattern as lm-trtllm.h.
    HMODULE h = nullptr;

    // Try loading with the TRT bin/ directory added to the search path.
    const char* trtRootEnv = std::getenv("TENSORRT_ROOT");
    if (trtRootEnv && trtRootEnv[0] != '\0') {
        std::string trtBinDir = std::string(trtRootEnv) + "\\bin";

        // Convert to wide string for AddDllDirectory
        int wlen = MultiByteToWideChar(CP_UTF8, 0, trtBinDir.c_str(), -1, nullptr, 0);
        std::wstring wTrtBinDir(wlen, 0);
        MultiByteToWideChar(CP_UTF8, 0, trtBinDir.c_str(), -1, &wTrtBinDir[0], wlen);

        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        DLL_DIRECTORY_COOKIE cookie = AddDllDirectory(wTrtBinDir.c_str());
        if (cookie) {
            fprintf(stderr, "[DiT-TRT] Added DLL dir for plugin: %s\n", trtBinDir.c_str());
        }

        h = LoadLibraryExA("hotstep_plugins.dll", NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);

        if (cookie) RemoveDllDirectory(cookie);
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }

    // Fallback: try bare LoadLibraryExA (searches exe dir + default dirs)
    if (!h) {
        h = LoadLibraryExA("hotstep_plugins.dll", NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }

    if (!h) {
        DWORD err = GetLastError();
        fprintf(stderr, "[DiT-TRT] WARNING: Cannot load hotstep_plugins.dll (error %lu). "
                        "w8a8 engines will not work.\n", err);
        return false;
    }
    g_plugin_handle = h;
    auto reg = reinterpret_cast<int(*)()>(GetProcAddress(h, "hotstep_register_plugins"));
    if (reg && reg() == 0) return true;
    fprintf(stderr, "[DiT-TRT] WARNING: hotstep_plugins.dll loaded but "
                    "hotstep_register_plugins() failed; w8a8 engines will not work.\n");
    return false;
#else
    static void* g_plugin_handle = nullptr;
    if (g_plugin_handle) return true;
    // dlopen searches $ORIGIN (set via RPATH) and the system LD_LIBRARY_PATH.
    void* h = dlopen("libhotstep_plugins.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        // Try the .dylib name on macOS.
        h = dlopen("libhotstep_plugins.dylib", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!h) {
        // Not fatal — q8map-fp16 and fp32 engines don't need the plugin.
        // w8a8 engines will fail at deserializeCudaEngine with a clear
        // "plugin not found" error.
        return false;
    }
    g_plugin_handle = h;
    auto reg = reinterpret_cast<int(*)()>(dlsym(h, "hotstep_register_plugins"));
    if (reg && reg() == 0) return true;
    fprintf(stderr, "[DiT-TRT] WARNING: libhotstep_plugins.so loaded but "
                    "hotstep_register_plugins() failed; w8a8 engines will not work.\n");
    return false;
#endif
}

// Log free/total VRAM on the current device for OOM diagnostics.
inline void dit_trt_log_vram(const char * label) {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        fprintf(stderr, "[DiT-TRT] VRAM(%s): free=%.1f MB / total=%.1f MB\n",
                label ? label : "",
                (double)free_bytes / (1ull << 20),
                (double)total_bytes / (1ull << 20));
    }
}

// Read the engine metadata sidecar and return whether the engine has embedded
// weights (strip_plan == false). When true, the runtime MUST skip the
// parser-refit + base_weights cache at load time because:
//   1. The weights are already in the engine — refitFromFile just overwrites
//      them with identical values, wasting CPU RAM (ONNX is parsed into host
//      memory) and creating a transient VRAM spike during refitCudaEngine().
//   2. The base_weights cache (a std::unordered_map copying every BF16/HALF
//      weight from the refitter into host memory) is only needed for adapter
//      revert. We build it lazily on the first adapter apply instead.
//
// On any read/parse error, returns false (conservative: fall back to the old
// refit-at-load behavior).
inline bool dit_trt_engine_has_embedded_weights(const char * engine_path) {
    if (!engine_path) return false;
    std::string metadata_path = std::string(engine_path) + ".metadata.json";
    FILE * mf = fopen(metadata_path.c_str(), "rb");
    if (!mf) return false;
    fseek(mf, 0, SEEK_END);
    long mlen = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    if (mlen <= 0) { fclose(mf); return false; }
    std::vector<char> mbuf((size_t)mlen + 1);
    size_t mrd = fread(mbuf.data(), 1, (size_t)mlen, mf);
    fclose(mf);
    if (mrd != (size_t)mlen) return false;
    mbuf[(size_t)mlen] = '\0';
    yyjson_doc * doc = yyjson_read(mbuf.data(), (size_t)mlen, 0);
    if (!doc) return false;
    yyjson_val * root = yyjson_doc_get_root(doc);
    bool strip_plan = true;  // conservative default
    bool ok = false;
    if (root && yyjson_is_obj(root)) {
        yyjson_val * sp = yyjson_obj_get(root, "strip_plan");
        if (sp && yyjson_is_bool(sp)) {
            strip_plan = yyjson_get_bool(sp);
            ok = true;
        }
    }
    yyjson_doc_free(doc);
    if (!ok) return false;
    return !strip_plan;  // embedded weights iff strip_plan == false
}

// Lazily build the base_weights cache by reading directly from the engine via
// a refitter (no ONNX file needed). Used by dit_trt_refit_adapter the first
// time an adapter is applied, so we can later revert via dit_trt_refit_base.
//
// Returns true on success (or if cache already populated). Returns false if
// the refitter cannot enumerate weights — in that case adapter revert will
// not be possible (but forward still works, since the engine is intact).
inline bool dit_trt_ensure_base_weights_cached(DitTrt * ctx) {
    if (!ctx || !ctx->engine) return false;
    if (!ctx->base_weights.empty()) return true;

    auto refitter = nvinfer1::createInferRefitter(*ctx->engine, ctx->logger);
    if (!refitter) {
        fprintf(stderr, "[DiT-TRT] ensure_base_weights: failed to create refitter\n");
        return false;
    }

    int32_t num_weights = refitter->getAllWeights(0, nullptr);
    if (num_weights <= 0) {
        delete refitter;
        fprintf(stderr, "[DiT-TRT] ensure_base_weights: refitter reports no weights\n");
        return false;
    }

    std::vector<const char*> names((size_t)num_weights);
    refitter->getAllWeights(num_weights, names.data());
    for (int32_t i = 0; i < num_weights; i++) {
        auto w = refitter->getNamedWeights(names[i]);
        if ((w.type == nvinfer1::DataType::kBF16 || w.type == nvinfer1::DataType::kHALF) && w.count > 0) {
            const uint16_t* data = static_cast<const uint16_t*>(w.values);
            ctx->base_weights[names[i]] = std::vector<uint16_t>(data, data + w.count);
        }
    }
    delete refitter;
    fprintf(stderr, "[DiT-TRT] Lazily cached %zu base weight tensors for adapter revert\n",
            ctx->base_weights.size());
    return true;
}

// Load a pre-built TRT engine.
//
// The engine was built with kREFIT_IDENTICAL + embedded weights
// (strip_plan == false). The runtime deserializes the engine directly into
// VRAM and skips the parser-refit / base_weights cache at load time:
//   - parser-refit (refitFromFile + refitCudaEngine) would re-read the ONNX,
//     stage all weights in host memory, and overwrite the engine's already-
//     embedded weights with identical values — a transient VRAM spike that
//     can OOM on small cards (e.g. 4B DiT-XL w8a8 on 6GB).
//   - base_weights cache is only needed for adapter revert; built lazily by
//     dit_trt_ensure_base_weights_cached() on the first adapter apply.
//
// The legacy path (parser-refit at load) is retained as a fallback for any
// engine that ships with strip_plan == true (no longer produced by the build
// script, but older bundles may exist on disk).
inline bool dit_trt_load(
    DitTrt*     ctx,
    const char* engine_path,
    const char* onnx_path,
    int         device_id = 0
) {
    auto t0 = std::chrono::steady_clock::now();
    cudaSetDevice(device_id);
    ctx->onnx_path = onnx_path;
    dit_trt_log_version("load");

    // Load the HOT-Step plugin library so w8a8 engines can find the
    // ConvRotInt8Linear op in TRT's plugin registry. Safe to call for
    // q8map-fp16 / fp32 engines too (the .so just won't be used).
    dit_trt_load_hotstep_plugins();
    dit_trt_log_vram("before deserialize");

    // Read engine file
    FILE* f = fopen(engine_path, "rb");
    if (!f) {
        fprintf(stderr, "[DiT-TRT] Cannot open engine %s\n", engine_path);
        return false;
    }
    // Use 64-bit file positioning: the DiT engine is 4GB+ and plain ftell()
    // returns long (32-bit on Windows MSVC), which overflows to -1 for files
    // >2GB. That -1 cast to size_t becomes SIZE_MAX (18446744073709551615),
    // causing a catastrophic allocation attempt. _ftelli64 (Windows) and
    // ftello (Unix) use 64-bit offsets.
#if defined(_WIN32)
    _fseeki64(f, 0, SEEK_END);
    int64_t engine_size_i64 = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
#else
    fseeko(f, 0, SEEK_END);
    int64_t engine_size_i64 = (int64_t)ftello(f);
    fseeko(f, 0, SEEK_SET);
#endif
    if (engine_size_i64 <= 0) {
        fclose(f);
        fprintf(stderr, "[DiT-TRT] FATAL: cannot determine engine file size (ftell returned %lld): %s\n",
                (long long)engine_size_i64, engine_path);
        return false;
    }
    size_t engine_size = (size_t)engine_size_i64;
    fprintf(stderr, "[DiT-TRT] Engine file size: %zu bytes (%.2f GB)\n",
            engine_size, (double)engine_size / (1ull << 30));
    fflush(stderr);

    // Allocate host buffer for the engine file. A 4GB+ contiguous allocation
    // can throw std::bad_alloc on Windows when the process address space is
    // fragmented (e.g. after loading+freeing the text-enc and cond-enc
    // engines, which together consumed ~2.4GB of contiguous host memory).
    // Without this try/catch the uncaught exception terminates the process
    // silently — no error message, no log line, just a crash. Catch it and
    // report a clear actionable error instead.
    std::vector<char> engine_data;
    try {
        engine_data.resize(engine_size);
    } catch (const std::bad_alloc & exc) {
        fclose(f);
        fprintf(stderr,
                "[DiT-TRT] FATAL: cannot allocate %zu bytes (%.2f GB) of host memory "
                "to read engine file. %s\n"
                "[DiT-TRT] This usually means process address space is fragmented after "
                "loading text-enc/cond-enc engines. Try one of:\n"
                "[DiT-TRT]   - rebuild the DiT engine with a smaller max_T profile\n"
                "[DiT-TRT]   - increase Windows pagefile / system RAM\n"
                "[DiT-TRT]   - close other memory-heavy processes before generation\n",
                engine_size, (double)engine_size / (1ull << 30), exc.what());
        return false;
    }
    size_t rd = fread(engine_data.data(), 1, engine_size, f);
    fclose(f);
    if (rd != engine_size) {
        fprintf(stderr, "[DiT-TRT] Short read on engine file (%zu/%zu bytes)\n", rd, engine_size);
        return false;
    }
    fprintf(stderr, "[DiT-TRT] Engine file read into host buffer (%zu bytes)\n", rd);
    fflush(stderr);

    // Deserialize
    ctx->runtime = nvinfer1::createInferRuntime(ctx->logger);
    if (!ctx->runtime) {
        fprintf(stderr, "[DiT-TRT] Failed to create TRT runtime\n");
        return false;
    }

    // Tell TRT it can spill temp data to a temp dir during deserialize if it
    // needs to. This helps on small-VRAM cards where the engine's compiled
    // plan + kernel cache don't all fit alongside the streamable-weight pool.
    // The dir is best-effort: if it doesn't exist or isn't writable, TRT
    // silently falls back to in-memory operation.
    if (const char * tmp = std::getenv("HOTSTEP_TRT_TEMP_DIR")) {
        std::string tmps(tmp);
        if (!tmps.empty()) {
            ctx->runtime->setTemporaryDirectory(tmps.c_str());
            fprintf(stderr, "[DiT-TRT] TRT temp dir: %s\n", tmps.c_str());
        }
    }

    fprintf(stderr, "[DiT-TRT] Deserializing engine (%zu bytes)...\n", engine_size);
    fflush(stderr);
    ctx->engine = ctx->runtime->deserializeCudaEngine(
        engine_data.data(), engine_size);
    if (!ctx->engine) {
        fprintf(stderr, "[DiT-TRT] Failed to deserialize engine\n");
        dit_trt_log_vram("after failed deserialize");
        return false;
    }

    fprintf(stderr, "[DiT-TRT] First deserialize: engine loaded (%zu bytes)\n", engine_size);
    dit_trt_log_vram("after deserialize");

    // Free the host-side engine file buffer now. TRT has copied what it needs
    // into its own internal structures (streamable weights stay in a TRT-managed
    // host pool; the plan and kernels are in VRAM). Holding engine_data on the
    // host side just wastes ~4GB of system RAM for the engine's lifetime.
    engine_data.clear();
    engine_data.shrink_to_fit();

    // ── Refit decision ────────────────────────────────────────────────────
    //
    // The current build (build-trt-engine.py) produces embedded-weight engines
    // (strip_plan == false) — the ONNX weights are baked into the engine file
    // at build time and are already in VRAM after deserializeCudaEngine. The
    // parser-refit (refitFromFile + refitCudaEngine) would re-read the ONNX,
    // stage all weights in host memory, and overwrite the engine's already-
    // embedded weights with identical values. This:
    //   - Wastes CPU RAM (the ONNX is parsed into host memory)
    //   - Creates a transient VRAM spike during refitCudaEngine() that can
    //     OOM on small cards (e.g. 4B DiT-XL w8a8 ~4GB on a 6GB card)
    //   - Buys us nothing: the engine weights are already correct.
    //
    // The base_weights cache (for adapter revert) is also skipped at load time
    // and built lazily by dit_trt_ensure_base_weights_cached() on the first
    // adapter apply — that reads directly from the engine via a refitter
    // (no ONNX file needed).
    //
    // The legacy path (parser-refit at load) is retained as a fallback for
    // any engine that ships with strip_plan == true (no longer produced by
    // the build script, but older bundles may exist on disk).
    bool embedded_weights = dit_trt_engine_has_embedded_weights(engine_path);
    fprintf(stderr, "[DiT-TRT] Engine %s embedded weights (strip_plan=%s)\n",
            embedded_weights ? "has" : "has NOT",
            embedded_weights ? "false" : "true");

    if (embedded_weights) {
        fprintf(stderr, "[DiT-TRT] Skipping parser-refit + base_weights cache "
                        "(embedded weights; cache built lazily on first adapter apply)\n");
    } else {
        // Legacy path: stripped-plan engine needs refit from ONNX to populate
        // weights. Also populates base_weights cache for adapter revert.
        fprintf(stderr, "[DiT-TRT] WARNING: engine has no embedded weights — "
                        "falling back to parser-refit at load (legacy path)\n");
        auto refitter = nvinfer1::createInferRefitter(*ctx->engine, ctx->logger);
        if (!refitter) {
            fprintf(stderr, "[DiT-TRT] Failed to create refitter\n");
            return false;
        }
        auto parser_refitter = nvonnxparser::createParserRefitter(*refitter, ctx->logger);
        if (!parser_refitter->refitFromFile(onnx_path)) {
            fprintf(stderr, "[DiT-TRT] Parser refit from ONNX failed\n");
            delete parser_refitter;
            delete refitter;
            return false;
        }
        if (!refitter->refitCudaEngine()) {
            fprintf(stderr, "[DiT-TRT] Engine refit failed\n");
            delete parser_refitter;
            delete refitter;
            return false;
        }
        // Cache refittable weights for adapter merges/reverts.
        int32_t num_weights = refitter->getAllWeights(0, nullptr);
        if (num_weights > 0) {
            std::vector<const char*> names(num_weights);
            refitter->getAllWeights(num_weights, names.data());
            for (int32_t i = 0; i < num_weights; i++) {
                auto w = refitter->getNamedWeights(names[i]);
                if ((w.type == nvinfer1::DataType::kBF16 || w.type == nvinfer1::DataType::kHALF) && w.count > 0) {
                    const uint16_t* data = static_cast<const uint16_t*>(w.values);
                    ctx->base_weights[names[i]] =
                        std::vector<uint16_t>(data, data + w.count);
                }
            }
            fprintf(stderr, "[DiT-TRT] Cached %zu base weight tensors for refit\n",
                    ctx->base_weights.size());
        }
        delete parser_refitter;
        delete refitter;
        dit_trt_log_vram("after legacy refit");
    }

    // Load refit manifest sidecar (weights_transposed list).
    // Generated by export_dit.py — records which weights dynamo stored
    // in transposed [in,out] orientation (vs torch [out,in]). LoRA deltas
    // arrive in torch orientation and must be transposed for these.
    // Loaded regardless of embedded_weights — adapter apply needs this map.
    {
        std::string manifest_path = std::string(onnx_path) + ".refit_manifest.json";
        FILE* fmap = fopen(manifest_path.c_str(), "rb");
        if (fmap) {
            fseek(fmap, 0, SEEK_END);
            long len = ftell(fmap);
            fseek(fmap, 0, SEEK_SET);
            std::vector<char> buf((size_t)len + 1);
            fread(buf.data(), 1, (size_t)len, fmap);
            fclose(fmap);
            buf[(size_t)len] = '\0';

            yyjson_doc* doc = yyjson_read(buf.data(), (size_t)len, 0);
            if (doc) {
                yyjson_val* root = yyjson_doc_get_root(doc);
                yyjson_val* arr = yyjson_obj_get(root, "weights_transposed");
                if (arr && yyjson_is_arr(arr)) {
                    size_t idx, max;
                    yyjson_val* item;
                    yyjson_arr_foreach(arr, idx, max, item) {
                        if (yyjson_is_str(item)) {
                            ctx->weights_transposed.insert(yyjson_get_str(item));
                        }
                    }
                }
                yyjson_doc_free(doc);
            }
            fprintf(stderr, "[DiT-TRT] Refit manifest: %zu transposed-layout weights\n",
                    ctx->weights_transposed.size());
        } else {
            fprintf(stderr, "[DiT-TRT] No refit manifest found (adapters will assume torch orientation)\n");
        }
    }

    // Weight streaming budget.
    //
    // The engine is built with kWEIGHT_STREAMING, so weights can either be
    // pinned in VRAM (budget = full) or streamed on demand from host memory
    // (budget < full). The previous default was budget = full, which OOMs on
    // small cards (e.g. 4B DiT-XL w8a8 ~4GB on a 6GB card has no room for
    // activations + CUDA context after pinning all weights).
    //
    // New default: budget = 0 (TRT "auto"). TRT 11 interprets budget=0 as
    // "pick a budget based on current free VRAM" — it will leave room for
    // activations and only pin what fits, streaming the rest. This trades a
    // small amount of inference speed (weights stream over PCIe on demand)
    // for actually fitting on the card.
    //
    // Override via env var:
    //   HOTSTEP_DIT_WEIGHT_STREAMING_BUDGET_MB=<n>  → pin <n> MB of weights
    //   HOTSTEP_DIT_WEIGHT_STREAMING_BUDGET_MB=full → pin all weights (old behavior)
    //   HOTSTEP_DIT_WEIGHT_STREAMING_BUDGET_MB=auto → TRT auto (new default)
    ctx->streamable_weights_bytes = ctx->engine->getStreamableWeightsSize();
    if (ctx->streamable_weights_bytes > 0) {
        int64_t budget_bytes = 0;  // 0 = TRT auto (leaves room for activations)
        const char * env_budget = std::getenv("HOTSTEP_DIT_WEIGHT_STREAMING_BUDGET_MB");
        if (env_budget && env_budget[0] != '\0') {
            std::string s(env_budget);
            if (s == "full") {
                budget_bytes = ctx->streamable_weights_bytes;
            } else if (s == "auto") {
                budget_bytes = 0;
            } else {
                try {
                    long long mb = std::stoll(s);
                    if (mb > 0) budget_bytes = mb * (1ll << 20);
                } catch (...) {
                    fprintf(stderr, "[DiT-TRT] WARNING: bad HOTSTEP_DIT_WEIGHT_STREAMING_BUDGET_MB=%s, using auto\n", env_budget);
                    budget_bytes = 0;
                }
            }
        }
        fprintf(stderr,
                "[DiT-TRT] Weight streaming: budget=%lld bytes (%.0f MB), streamable=%lld bytes (%.0f MB) [%s]\n",
                (long long)budget_bytes, (double)budget_bytes / (1ull << 20),
                (long long)ctx->streamable_weights_bytes, (double)ctx->streamable_weights_bytes / (1ull << 20),
                env_budget ? env_budget : "auto");
        if (!ctx->engine->setWeightStreamingBudgetV2(budget_bytes)) {
            fprintf(stderr, "[DiT-TRT] WARNING: failed to set weight-streaming budget\n");
        } else {
            ctx->weight_streaming_budget_bytes = ctx->engine->getWeightStreamingBudgetV2();
            fprintf(stderr, "[DiT-TRT] Weight streaming: actual budget=%lld bytes (%.0f MB)\n",
                    (long long)ctx->weight_streaming_budget_bytes,
                    (double)ctx->weight_streaming_budget_bytes / (1ull << 20));
        }
    }
    dit_trt_log_vram("after weight-streaming budget");

    // Create execution context
    ctx->context = ctx->engine->createExecutionContext();
    if (!ctx->context) {
        fprintf(stderr, "[DiT-TRT] Failed to create execution context\n");
        return false;
    }

    // Resolve I/O tensor indices and log dtypes
    int num_io = ctx->engine->getNbIOTensors();
    for (int i = 0; i < num_io; i++) {
        const char* name = ctx->engine->getIOTensorName(i);
        auto dtype = ctx->engine->getTensorDataType(name);
        auto mode = ctx->engine->getTensorIOMode(name);
        const char* dtype_str = "unknown";
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:  dtype_str = "fp32"; break;
            case nvinfer1::DataType::kHALF:   dtype_str = "fp16"; break;
            case nvinfer1::DataType::kBF16:   dtype_str = "bf16"; break;
            case nvinfer1::DataType::kINT32:  dtype_str = "int32"; break;
            case nvinfer1::DataType::kINT8:   dtype_str = "int8"; break;
            case nvinfer1::DataType::kBOOL:   dtype_str = "bool"; break;
            default: break;
        }
        const char* io_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
        fprintf(stderr, "[DiT-TRT] IO[%d] %-20s %s  %s\n", i, name, io_str, dtype_str);
        
        if (std::string(name) == "input_latents") {
            ctx->idx_input_latents = i;
            ctx->dtype_input_latents = dtype;
        }
        else if (std::string(name) == "enc_hidden") {
            ctx->idx_enc_hidden = i;
            ctx->dtype_enc_hidden = dtype;
        }
        else if (std::string(name) == "t")          ctx->idx_t = i;
        else if (std::string(name) == "t_r")        ctx->idx_t_r = i;
        else if (std::string(name) == "attention_mask") {
            ctx->idx_attention_mask = i;
        }
        else if (std::string(name) == "encoder_attention_mask") {
            ctx->idx_encoder_attention_mask = i;
        }
        else if (std::string(name) == "velocity") {
            ctx->idx_velocity = i;
            ctx->dtype_velocity = dtype;
        }
    }

    if (ctx->idx_input_latents < 0 || ctx->idx_enc_hidden < 0 ||
        ctx->idx_t < 0 || ctx->idx_t_r < 0 || ctx->idx_velocity < 0) {
        fprintf(stderr, "[DiT-TRT] Missing I/O tensors!\n");
        return false;
    }

    // The attention_mask and encoder_attention_mask inputs were added in a
    // later revision (timbre-reference fix). Engines built before the fix
    // don't have these bindings — refuse to load so the user re-builds the
    // engine from the updated ONNX. Silently continuing would re-introduce
    // the timbre-dilution bug.
    if (ctx->idx_attention_mask < 0 || ctx->idx_encoder_attention_mask < 0) {
        fprintf(stderr,
                "[DiT-TRT] FATAL: engine is missing attention_mask / encoder_attention_mask\n"
                "[DiT-TRT]        bindings. This engine was built from an older ONNX export\n"
                "[DiT-TRT]        that did not pass attention masks into the DiT graph, which\n"
                "[DiT-TRT]        causes the cross-attention to attend to null_cond_vec\n"
                "[DiT-TRT]        padding and silently ignore the timbre reference.\n"
                "[DiT-TRT]        Re-export the ONNX with the patched export_dit.py and\n"
                "[DiT-TRT]        rebuild the TRT engine with build-trt-engine.py.\n");
        return false;
    }

    // Detect I/O dtype: fp32 (FP8 QDQ from modelopt), fp16, or bf16 (dynamo)
    {
        const char* il_name = ctx->engine->getIOTensorName(ctx->idx_input_latents);
        auto il_dtype = ctx->engine->getTensorDataType(il_name);
        if (il_dtype == nvinfer1::DataType::kFLOAT) {
            ctx->io_dtype = DitTrt::IO_FP32;
        } else if (il_dtype == nvinfer1::DataType::kHALF) {
            ctx->io_dtype = DitTrt::IO_FP16;
        } else {
            ctx->io_dtype = DitTrt::IO_BF16;
        }
        const char* names[] = { "bf16", "fp16", "fp32" };
        fprintf(stderr, "[DiT-TRT] I/O dtype: %s\n", names[ctx->io_dtype]);
    }

    auto t1 = std::chrono::steady_clock::now();
    ctx->load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t0).count();
    fprintf(stderr, "[DiT-TRT] Load + refit complete (%lld ms)\n",
            (long long)ctx->load_time_ms);
    fflush(stderr);
    ctx->current_adapter.clear();

    // Create dedicated CUDA stream for inference
    if (!ctx->stream) {
        cudaStreamCreate(&ctx->stream);
    }

    return true;
}

// ── LoRA adapter refitting ──────────────────────────────────────────────────

// Refit engine with LoRA adapter deltas.
// `deltas` maps TRT weight names to merged weight data (W_base + delta).
// Each value should point to BF16 data with the same element count as base.
//
// Returns refit time in milliseconds.
inline int64_t dit_trt_refit_adapter(
    DitTrt*     ctx,
    const std::string& adapter_name,
    const std::unordered_map<std::string, const void*>& merged_weights
) {
    std::lock_guard<std::mutex> lock(ctx->refit_mutex);
    auto t0 = std::chrono::steady_clock::now();

    // Lazily build the base_weights cache on first adapter apply. When the
    // engine has embedded weights (the common case), dit_trt_load skipped
    // the parser-refit and did not populate this cache. We read base weights
    // directly from the engine via a refitter (no ONNX file needed) so the
    // adapter revert path (dit_trt_refit_base) can later restore them.
    if (ctx->base_weights.empty()) {
        if (!dit_trt_ensure_base_weights_cached(ctx)) {
            fprintf(stderr, "[DiT-TRT] Cannot apply adapter: failed to build base_weights cache\n");
            return -1;
        }
    }

    auto refitter = nvinfer1::createInferRefitter(*ctx->engine, ctx->logger);
    if (!refitter) {
        fprintf(stderr, "[DiT-TRT] Failed to create refitter for adapter\n");
        return -1;
    }

    int updated = 0;
    for (const auto& [name, data] : merged_weights) {
        auto it = ctx->base_weights.find(name);
        if (it == ctx->base_weights.end()) {
            fprintf(stderr, "[DiT-TRT] Warning: weight '%s' not in base cache\n",
                    name.c_str());
            continue;
        }

        nvinfer1::Weights w;
        w.type   = nvinfer1::DataType::kBF16;
        w.values = data;
        w.count  = static_cast<int64_t>(it->second.size());

        if (!refitter->setNamedWeights(name.c_str(), w)) {
            fprintf(stderr, "[DiT-TRT] Failed to set weight '%s'\n",
                    name.c_str());
        } else {
            updated++;
        }
    }

    if (!refitter->refitCudaEngine()) {
        fprintf(stderr, "[DiT-TRT] Adapter refit failed\n");
        delete refitter;
        return -1;
    }

    delete refitter;
    ctx->current_adapter = adapter_name;
    ctx->current_adapter_key = adapter_name;

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "[DiT-TRT] Adapter '%s' applied: %d weights refitted (%lld ms)\n",
            adapter_name.c_str(), updated, (long long)ms);
    return ms;
}

// Revert to base weights (remove adapter)
inline int64_t dit_trt_refit_base(DitTrt* ctx) {
    std::lock_guard<std::mutex> lock(ctx->refit_mutex);
    if (ctx->current_adapter.empty()) return 0;  // already base

    auto t0 = std::chrono::steady_clock::now();

    auto refitter = nvinfer1::createInferRefitter(*ctx->engine, ctx->logger);
    if (!refitter) return -1;

    for (const auto& [name, data] : ctx->base_weights) {
        nvinfer1::Weights w;
        w.type   = nvinfer1::DataType::kBF16;
        w.values = data.data();
        w.count  = static_cast<int64_t>(data.size());
        refitter->setNamedWeights(name.c_str(), w);
    }

    bool ok = refitter->refitCudaEngine();
    delete refitter;

    if (!ok) {
        fprintf(stderr, "[DiT-TRT] Base refit failed!\n");
        return -1;
    }

    ctx->current_adapter.clear();
    ctx->current_adapter_key.clear();

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "[DiT-TRT] Reverted to base weights (%lld ms)\n",
            (long long)ms);
    return ms;
}

// ── Single forward pass ─────────────────────────────────────────────────────

// Run one DiT forward pass (one diffusion timestep).
// All pointers are GPU device memory.
//
// input_latents:           [N, T, 192] fp16
// enc_hidden:              [N, S, 2048] fp16
// t:                       [N] fp32
// t_r:                     [N] fp32
// attention_mask:          [N, T] int64 — self-attn padding (1=attend, 0=pad)
// encoder_attention_mask:  [N, S] int64 — cross-attn padding (1=attend, 0=pad)
// velocity_out:            [N, T, 64] fp16 (output)
inline bool dit_trt_forward(
    DitTrt*     ctx,
    const void* input_latents,                // GPU, fp16 [N, T, 192]
    const void* enc_hidden,                   // GPU, fp16 [N, S, 2048]
    const float* t,                           // GPU, fp32 [N]
    const float* t_r,                         // GPU, fp32 [N]
    int N, int T, int S,
    const void* attention_mask,               // GPU, int64 [N, T]
    const void* encoder_attention_mask,       // GPU, int64 [N, S]
    void* velocity_out,                       // GPU, fp16 [N, T, 64]
    cudaStream_t stream = nullptr
) {
    if (!ctx || !ctx->engine || !ctx->context) {
        fprintf(stderr, "[DiT-TRT] FATAL: forward called before engine/context load\n");
        return false;
    }
    auto* context = ctx->context;

    // Set input shapes
    const char* input_latents_name = ctx->engine->getIOTensorName(ctx->idx_input_latents);
    const char* enc_hidden_name    = ctx->engine->getIOTensorName(ctx->idx_enc_hidden);
    const char* t_name             = ctx->engine->getIOTensorName(ctx->idx_t);
    const char* t_r_name           = ctx->engine->getIOTensorName(ctx->idx_t_r);
    const char* velocity_name      = ctx->engine->getIOTensorName(ctx->idx_velocity);
    const char* attention_mask_name         = ctx->engine->getIOTensorName(ctx->idx_attention_mask);
    const char* encoder_attention_mask_name = ctx->engine->getIOTensorName(ctx->idx_encoder_attention_mask);

    if (!context->setInputShape(input_latents_name, nvinfer1::Dims3(N, T, 192)) ||
        !context->setInputShape(enc_hidden_name,    nvinfer1::Dims3(N, S, 2048)) ||
        !context->setInputShape(t_name,             nvinfer1::Dims{1, {N}}) ||
        !context->setInputShape(t_r_name,           nvinfer1::Dims{1, {N}}) ||
        !context->setInputShape(attention_mask_name,         nvinfer1::Dims2(N, T)) ||
        !context->setInputShape(encoder_attention_mask_name, nvinfer1::Dims2(N, S))) {
        fprintf(stderr, "[DiT-TRT] FATAL: failed to set input shapes (N=%d T=%d S=%d)\n",
                N, T, S);
        return false;
    }

    // Set tensor addresses (setTensorAddress takes void*, cast away const)
    if (!context->setTensorAddress(input_latents_name, const_cast<void*>(input_latents)) ||
        !context->setTensorAddress(enc_hidden_name,    const_cast<void*>(enc_hidden)) ||
        !context->setTensorAddress(t_name,             const_cast<void*>(static_cast<const void*>(t))) ||
        !context->setTensorAddress(t_r_name,           const_cast<void*>(static_cast<const void*>(t_r))) ||
        !context->setTensorAddress(attention_mask_name,         const_cast<void*>(attention_mask)) ||
        !context->setTensorAddress(encoder_attention_mask_name, const_cast<void*>(encoder_attention_mask)) ||
        !context->setTensorAddress(velocity_name,      velocity_out)) {
        fprintf(stderr, "[DiT-TRT] FATAL: failed to bind tensor addresses\n");
        return false;
    }

    // Enqueue on stream
    bool ok = context->enqueueV3(stream ? stream : 0);
    if (!ok) {
        fprintf(stderr, "[DiT-TRT] enqueueV3 failed\n");
    }

    // First-call diagnostic: dump a few output values to catch NaN early
    static bool first_call = true;
    if (first_call && ok) {
        first_call = false;
        cudaStreamSynchronize(stream ? stream : 0);
        float probe[8] = {};
        size_t probe_bytes = sizeof(probe);
        auto io_dt = ctx->engine->getTensorDataType(velocity_name);
        if (io_dt == nvinfer1::DataType::kFLOAT) {
            cudaMemcpy(probe, velocity_out, probe_bytes, cudaMemcpyDeviceToHost);
        } else if (io_dt == nvinfer1::DataType::kHALF) {
            uint16_t h[8];
            cudaMemcpy(h, velocity_out, sizeof(h), cudaMemcpyDeviceToHost);
            for (int i = 0; i < 8; i++) {
                // FP16 → FP32
                uint32_t sign = ((uint32_t)h[i] & 0x8000) << 16;
                uint32_t exp = (h[i] >> 10) & 0x1F;
                uint32_t mant = h[i] & 0x03FF;
                uint32_t u = (exp == 0) ? sign :
                             (exp == 31) ? (sign | 0x7F800000 | (mant << 13)) :
                             (sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13));
                memcpy(&probe[i], &u, 4);
            }
        } else {
            // BF16
            uint16_t h[8];
            cudaMemcpy(h, velocity_out, sizeof(h), cudaMemcpyDeviceToHost);
            for (int i = 0; i < 8; i++) {
                uint32_t u = (uint32_t)h[i] << 16;
                memcpy(&probe[i], &u, 4);
            }
        }
        fprintf(stderr, "[DiT-TRT] DIAG first output: [%.4g, %.4g, %.4g, %.4g, %.4g, %.4g, %.4g, %.4g]\n",
                probe[0], probe[1], probe[2], probe[3],
                probe[4], probe[5], probe[6], probe[7]);
        fflush(stderr);
    }
    return ok;
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

inline void dit_trt_free(DitTrt* ctx) {
    dit_trt_release_evictable(ctx);
    if (ctx->stream)          { cudaStreamDestroy(ctx->stream); ctx->stream = nullptr; }
    // TensorRT objects are destroyed with delete, not destroy().
    if (ctx->context)         { delete ctx->context;            ctx->context = nullptr; }
    if (ctx->engine)          { delete ctx->engine;             ctx->engine = nullptr; }
    if (ctx->runtime)         { delete ctx->runtime;            ctx->runtime = nullptr; }
    ctx->base_weights.clear();
    ctx->weights_transposed.clear();
    ctx->current_adapter.clear();
    ctx->current_adapter_key.clear();
    ctx->onnx_path.clear();
    fprintf(stderr, "[DiT-TRT] Engine fully unloaded\n");
}

#endif // HOT_STEP_TRT
