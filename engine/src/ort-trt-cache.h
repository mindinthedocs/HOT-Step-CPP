#pragma once
// ort-trt-cache.h: fingerprinted TensorRT EP cache directories for ORT modules.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#    define HS_ORT_SEP "\\"
#else
#    define HS_ORT_SEP "/"
#endif

static inline uint64_t hs_ort_fnv1a_string(const std::string & text) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : text) {
        h ^= (uint64_t)c;
        h *= 1099511628211ULL;
    }
    return h;
}

static inline std::string hs_ort_hex64(uint64_t v) {
    static const char * hex = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[i] = hex[v & 0xf];
        v >>= 4;
    }
    return out;
}

static inline std::string hs_ort_dirname(const char * path) {
    std::string p = path ? path : "";
    while (!p.empty() && (p.back() == '/' || p.back() == '\\')) {
        p.pop_back();
    }
    auto slash = p.find_last_of("/\\");
    return (slash != std::string::npos) ? p.substr(0, slash) : ".";
}

static inline std::string hs_ort_join(const std::string & a, const std::string & b) {
    if (a.empty()) return b;
    if (a.back() == '/' || a.back() == '\\') return a + b;
    return a + HS_ORT_SEP + b;
}

static inline bool hs_ort_env_truthy(const char * name) {
    const char * v = std::getenv(name);
    if (!v || !v[0]) return false;
    std::string s = v;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

static inline int hs_ort_env_int(const char * name, int fallback, int min_v, int max_v) {
    const char * v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    char * end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed < min_v || parsed > max_v) {
        fprintf(stderr, "[ORT-TRT] WARNING: ignoring invalid %s=%s; using %d\n",
                name, v, fallback);
        return fallback;
    }
    return (int) parsed;
}

static inline void hs_ort_append_cache_profile_seed(std::string & seed, const char * module_tag) {
    const std::string tag = module_tag ? module_tag : "";
    seed += "\nbuilder=";
    seed += std::to_string(hs_ort_env_int("HOTSTEP_ORT_TRT_BUILDER_OPT_LEVEL", 0, 0, 5));
    if (tag == "text-enc") {
        int opt_text = hs_ort_env_int("HOTSTEP_ORT_TRT_TEXT_OPT_TOKENS", 128, 1, 4096);
        int max_text = hs_ort_env_int("HOTSTEP_ORT_TRT_TEXT_MAX_TOKENS", 512, opt_text, 8192);
        if (max_text < opt_text) max_text = opt_text;
        seed += "\nmin=input_ids:1x1";
        seed += "\nopt=input_ids:1x" + std::to_string(opt_text);
        seed += "\nmax=input_ids:1x" + std::to_string(max_text);
        return;
    }
    if (tag == "cond-enc") {
        int opt_text = hs_ort_env_int("HOTSTEP_ORT_TRT_TEXT_OPT_TOKENS", 128, 1, 4096);
        int max_text = hs_ort_env_int("HOTSTEP_ORT_TRT_TEXT_MAX_TOKENS", 512, opt_text, 8192);
        int opt_lyric = hs_ort_env_int("HOTSTEP_ORT_TRT_LYRIC_OPT_TOKENS", 256, 1, 4096);
        int max_lyric = hs_ort_env_int("HOTSTEP_ORT_TRT_LYRIC_MAX_TOKENS", 1024, opt_lyric, 8192);
        int opt_timbre = hs_ort_env_int("HOTSTEP_ORT_TRT_TIMBRE_OPT_FRAMES", 1, 1, 4096);
        int max_timbre = hs_ort_env_int("HOTSTEP_ORT_TRT_TIMBRE_MAX_FRAMES", 512, opt_timbre, 8192);
        if (max_text < opt_text) max_text = opt_text;
        if (max_lyric < opt_lyric) max_lyric = opt_lyric;
        if (max_timbre < opt_timbre) max_timbre = opt_timbre;
        seed += "\nmin=text_hidden:1x1x1024,lyric_embed:1x1x1024,timbre_feats:1x1x64";
        seed += "\nopt=text_hidden:1x" + std::to_string(opt_text) + "x1024,"
            "lyric_embed:1x" + std::to_string(opt_lyric) + "x1024,"
            "timbre_feats:1x" + std::to_string(opt_timbre) + "x64";
        seed += "\nmax=text_hidden:1x" + std::to_string(max_text) + "x1024,"
            "lyric_embed:1x" + std::to_string(max_lyric) + "x1024,"
            "timbre_feats:1x" + std::to_string(max_timbre) + "x64";
        return;
    }
    int opt_vae = hs_ort_env_int("HOTSTEP_ORT_TRT_VAE_OPT_FRAMES", 1024, 64, 8192);
    int max_vae = hs_ort_env_int("HOTSTEP_ORT_TRT_VAE_MAX_FRAMES", 2250, opt_vae, 16384);
    if (max_vae < opt_vae) max_vae = opt_vae;
    seed += "\nmin=latents:1x64x64";
    seed += "\nopt=latents:1x64x" + std::to_string(opt_vae);
    seed += "\nmax=latents:1x64x" + std::to_string(max_vae);
}

static inline std::string hs_ort_trt_cache_root(const char * onnx_path) {
    const char * env_root = std::getenv("HOTSTEP_ORT_TRT_ENGINE_ROOT");
    if (env_root && env_root[0]) {
        return env_root;
    }
    return hs_ort_join(hs_ort_dirname(onnx_path), "ort-trt-engines");
}

static inline std::string hs_ort_trt_cache_dir(const char * onnx_path,
                                               const char * artifact_fingerprint,
                                               const char * module_tag) {
    std::string root = hs_ort_trt_cache_root(onnx_path);
    std::string seed = module_tag ? module_tag : "ort";
    seed += "\n";
    seed += (artifact_fingerprint && artifact_fingerprint[0]) ? artifact_fingerprint : (onnx_path ? onnx_path : "");
    hs_ort_append_cache_profile_seed(seed, module_tag);
    std::string cache = hs_ort_join(root, std::string(module_tag ? module_tag : "ort") + "-"
                                          + hs_ort_hex64(hs_ort_fnv1a_string(seed)));

    std::error_code ec;
    std::filesystem::create_directories(cache, ec);
    if (ec) {
        fprintf(stderr, "[ORT-TRT] WARNING: cannot create TensorRT EP cache dir %s: %s\n",
                cache.c_str(), ec.message().c_str());
    }
    return cache;
}

static inline bool hs_ort_trt_cache_ready(const std::string & cache_dir) {
    std::error_code ec;
    if (!std::filesystem::is_directory(cache_dir, ec)) {
        return false;
    }
    bool has_engine = false;
    bool has_profile = false;
    for (const auto & entry : std::filesystem::directory_iterator(cache_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        has_engine = has_engine || ext == ".engine";
        has_profile = has_profile || ext == ".profile";
    }
    return has_engine && has_profile;
}

struct HsOrtTrtProfileShapes {
    std::string min_shapes;
    std::string opt_shapes;
    std::string max_shapes;
};

static inline HsOrtTrtProfileShapes hs_ort_trt_profile_shapes(const char * module_tag) {
    const std::string tag = module_tag ? module_tag : "";
    if (tag == "text-enc") {
        int opt_text = hs_ort_env_int("HOTSTEP_ORT_TRT_TEXT_OPT_TOKENS", 128, 1, 4096);
        int max_text = hs_ort_env_int("HOTSTEP_ORT_TRT_TEXT_MAX_TOKENS", 512, opt_text, 8192);
        if (max_text < opt_text) max_text = opt_text;
        return {
            "input_ids:1x1",
            "input_ids:1x" + std::to_string(opt_text),
            "input_ids:1x" + std::to_string(max_text),
        };
    }
    if (tag == "cond-enc") {
        int opt_text = hs_ort_env_int("HOTSTEP_ORT_TRT_TEXT_OPT_TOKENS", 128, 1, 4096);
        int max_text = hs_ort_env_int("HOTSTEP_ORT_TRT_TEXT_MAX_TOKENS", 512, opt_text, 8192);
        int opt_lyric = hs_ort_env_int("HOTSTEP_ORT_TRT_LYRIC_OPT_TOKENS", 256, 1, 4096);
        int max_lyric = hs_ort_env_int("HOTSTEP_ORT_TRT_LYRIC_MAX_TOKENS", 1024, opt_lyric, 8192);
        int opt_timbre = hs_ort_env_int("HOTSTEP_ORT_TRT_TIMBRE_OPT_FRAMES", 1, 1, 4096);
        int max_timbre = hs_ort_env_int("HOTSTEP_ORT_TRT_TIMBRE_MAX_FRAMES", 512, opt_timbre, 8192);
        if (max_text < opt_text) max_text = opt_text;
        if (max_lyric < opt_lyric) max_lyric = opt_lyric;
        if (max_timbre < opt_timbre) max_timbre = opt_timbre;
        return {
            "text_hidden:1x1x1024,lyric_embed:1x1x1024,timbre_feats:1x1x64",
            "text_hidden:1x" + std::to_string(opt_text) + "x1024,"
                "lyric_embed:1x" + std::to_string(opt_lyric) + "x1024,"
                "timbre_feats:1x" + std::to_string(opt_timbre) + "x64",
            "text_hidden:1x" + std::to_string(max_text) + "x1024,"
                "lyric_embed:1x" + std::to_string(max_lyric) + "x1024,"
                "timbre_feats:1x" + std::to_string(max_timbre) + "x64",
        };
    }

    int opt_vae = hs_ort_env_int("HOTSTEP_ORT_TRT_VAE_OPT_FRAMES", 1024, 64, 8192);
    int max_vae = hs_ort_env_int("HOTSTEP_ORT_TRT_VAE_MAX_FRAMES", 2250, opt_vae, 16384);
    if (max_vae < opt_vae) max_vae = opt_vae;
    return {
        "latents:1x64x64",
        "latents:1x64x" + std::to_string(opt_vae),
        "latents:1x64x" + std::to_string(max_vae),
    };
}

static inline std::unordered_map<std::string, std::string>
hs_ort_trt_provider_options(const char * onnx_path,
                            const char * artifact_fingerprint,
                            const char * module_tag,
                            int device_id,
                            bool fp16,
                            size_t workspace_bytes) {
    std::string cache_dir = hs_ort_trt_cache_dir(onnx_path, artifact_fingerprint, module_tag);
    std::string cache_root = hs_ort_trt_cache_root(onnx_path);
    HsOrtTrtProfileShapes shapes = hs_ort_trt_profile_shapes(module_tag);
    int builder_opt = hs_ort_env_int("HOTSTEP_ORT_TRT_BUILDER_OPT_LEVEL", 0, 0, 5);

    return {
        {"device_id", std::to_string(device_id)},
        {"trt_max_partition_iterations", "1000"},
        {"trt_min_subgraph_size", "1"},
        {"trt_max_workspace_size", std::to_string((unsigned long long) workspace_bytes)},
        {"trt_fp16_enable", fp16 ? "1" : "0"},
        {"trt_engine_cache_enable", "1"},
        {"trt_engine_cache_path", cache_dir},
        {"trt_engine_cache_prefix", module_tag ? module_tag : "ort"},
        {"trt_timing_cache_enable", "1"},
        {"trt_timing_cache_path", cache_root},
        {"trt_force_sequential_engine_build", "1"},
        {"trt_context_memory_sharing_enable", "1"},
        {"trt_builder_optimization_level", std::to_string(builder_opt)},
        {"trt_profile_min_shapes", shapes.min_shapes},
        {"trt_profile_opt_shapes", shapes.opt_shapes},
        {"trt_profile_max_shapes", shapes.max_shapes},
    };
}
