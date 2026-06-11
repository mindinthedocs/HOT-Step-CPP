// ace-ort-cache.cpp: prebuild ONNX Runtime TensorRT EP caches for side modules.

#include "ort-trt-cache.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#endif

struct Args {
    std::string text_onnx;
    std::string cond_onnx;
    std::string vae_onnx;
    std::string modules = "text,cond,vae";
    std::string out_dir;
    int device_id = 0;
    int builder_optimization_level = 0;
    size_t workspace_bytes = (size_t)2 << 30;
    int text_opt_tokens = 128;
    int text_max_tokens = 512;
    int lyric_opt_tokens = 256;
    int lyric_max_tokens = 1024;
    int timbre_opt_frames = 1;
    int timbre_max_frames = 512;
    int vae_opt_frames = 1024;
    int vae_max_frames = 2250;
};

static void usage() {
    std::fprintf(stderr,
        "usage: ace-ort-cache --modules text,cond,vae [--text-onnx path] [--cond-onnx path] [--vae-onnx path]\n"
        "                     [--out-dir dir] [--workspace-gb n] [--builder-optimization-level 0..5]\n");
}

static bool parse_int(const char * text, int * out) {
    if (!text || !out) return false;
    char * end = nullptr;
    long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    *out = (int)v;
    return true;
}

static bool parse_double(const char * text, double * out) {
    if (!text || !out) return false;
    char * end = nullptr;
    double v = std::strtod(text, &end);
    if (end == text || *end != '\0') return false;
    *out = v;
    return true;
}

static bool parse_args(int argc, char ** argv, Args * args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto value = [&]() -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", key.c_str());
                return nullptr;
            }
            return argv[++i];
        };

        if (key == "--help" || key == "-h") {
            usage();
            std::exit(0);
        } else if (key == "--text-onnx") {
            const char * v = value(); if (!v) return false; args->text_onnx = v;
        } else if (key == "--cond-onnx") {
            const char * v = value(); if (!v) return false; args->cond_onnx = v;
        } else if (key == "--vae-onnx") {
            const char * v = value(); if (!v) return false; args->vae_onnx = v;
        } else if (key == "--modules") {
            const char * v = value(); if (!v) return false; args->modules = v;
        } else if (key == "--out-dir") {
            const char * v = value(); if (!v) return false; args->out_dir = v;
        } else if (key == "--device-id") {
            const char * v = value(); if (!v || !parse_int(v, &args->device_id)) return false;
        } else if (key == "--builder-optimization-level") {
            const char * v = value(); if (!v || !parse_int(v, &args->builder_optimization_level)) return false;
        } else if (key == "--workspace-gb") {
            double gb = 0.0;
            const char * v = value(); if (!v || !parse_double(v, &gb) || gb <= 0.0) return false;
            args->workspace_bytes = (size_t)(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (key == "--text-opt-tokens") {
            const char * v = value(); if (!v || !parse_int(v, &args->text_opt_tokens)) return false;
        } else if (key == "--text-max-tokens") {
            const char * v = value(); if (!v || !parse_int(v, &args->text_max_tokens)) return false;
        } else if (key == "--lyric-opt-tokens") {
            const char * v = value(); if (!v || !parse_int(v, &args->lyric_opt_tokens)) return false;
        } else if (key == "--lyric-max-tokens") {
            const char * v = value(); if (!v || !parse_int(v, &args->lyric_max_tokens)) return false;
        } else if (key == "--timbre-opt-frames") {
            const char * v = value(); if (!v || !parse_int(v, &args->timbre_opt_frames)) return false;
        } else if (key == "--timbre-max-frames") {
            const char * v = value(); if (!v || !parse_int(v, &args->timbre_max_frames)) return false;
        } else if (key == "--vae-opt-frames") {
            const char * v = value(); if (!v || !parse_int(v, &args->vae_opt_frames)) return false;
        } else if (key == "--vae-max-frames") {
            const char * v = value(); if (!v || !parse_int(v, &args->vae_max_frames)) return false;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", key.c_str());
            return false;
        }
    }
    return true;
}

static void set_env_value(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static void set_env_value(const char * name, int value) {
    set_env_value(name, std::to_string(value));
}

static std::vector<std::string> split_modules(const std::string & text) {
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c); }), item.end());
        std::transform(item.begin(), item.end(), item.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (item == "text" || item == "text-enc" || item == "text_encoder") item = "text-enc";
        if (item == "cond" || item == "cond-enc" || item == "cond_encoder") item = "cond-enc";
        if (item == "vae" || item == "vae-dec" || item == "vae_decoder") item = "vae-dec";
        if (!item.empty() && std::find(out.begin(), out.end(), item) == out.end()) out.push_back(item);
    }
    return out;
}

static bool file_exists(const std::string & path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

static std::string normalize_path(const std::string & path) {
    std::error_code ec;
    std::filesystem::path abs = std::filesystem::absolute(path, ec);
    if (ec) return path;
    std::filesystem::path canon = std::filesystem::weakly_canonical(abs, ec);
    return ec ? abs.string() : canon.string();
}

static uint64_t fnv1a_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    uint64_t h = 1469598103934665603ULL;
    std::vector<unsigned char> buf(1024 * 1024);
    while (f) {
        f.read(reinterpret_cast<char *>(buf.data()), (std::streamsize)buf.size());
        std::streamsize n = f.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            h ^= (uint64_t)buf[(size_t)i];
            h *= 1099511628211ULL;
        }
    }
    return h;
}

static std::string artifact_fingerprint(const std::vector<std::string> & paths) {
    std::string fp;
    for (const auto & path : paths) {
        std::string normalized = normalize_path(path);
        fp += normalized;
        fp += "=";
        fp += hs_ort_hex64(fnv1a_file(normalized));
        fp += ";";
    }
    return fp;
}

static std::string dirname_of(const std::string & path) {
    auto slash = path.find_last_of("/\\");
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

static std::string join_path(const std::string & a, const std::string & b) {
    return hs_ort_join(a, b);
}

static Ort::Session make_session(Ort::Env & env, const std::string & path, Ort::SessionOptions & opts) {
#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t)wlen);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    return Ort::Session(env, wpath.data(), opts);
#else
    return Ort::Session(env, path.c_str(), opts);
#endif
}

struct CacheSummary {
    int engine_count = 0;
    int profile_count = 0;
    uint64_t engine_bytes = 0;
};

static CacheSummary validate_cache(const std::string & cache_dir) {
    CacheSummary summary;
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(cache_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (ext == ".engine") {
            summary.engine_count++;
            summary.engine_bytes += (uint64_t)entry.file_size(ec);
        } else if (ext == ".profile") {
            summary.profile_count++;
        }
    }
    return summary;
}

static void write_text_file(const std::filesystem::path & path, const std::string & text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f << text;
}

static void write_cache_metadata(const std::string & root_dir,
                                 const std::string & cache_dir,
                                 const std::string & module_tag,
                                 const std::string & artifact_fp,
                                 const CacheSummary & summary) {
    std::filesystem::path cache_path(cache_dir);
    std::ostringstream cache_json;
    cache_json
        << "{\n"
        << "  \"artifact\": \"hotstep-ort-trt-cache\",\n"
        << "  \"module\": \"" << module_tag << "\",\n"
        << "  \"cache_dir\": \"" << cache_path.filename().string() << "\",\n"
        << "  \"artifact_fingerprint\": \"" << artifact_fp << "\",\n"
        << "  \"engine_count\": " << summary.engine_count << ",\n"
        << "  \"profile_count\": " << summary.profile_count << ",\n"
        << "  \"engine_bytes\": " << summary.engine_bytes << "\n"
        << "}\n";
    write_text_file(cache_path / "hotstep-ort-trt-cache.json", cache_json.str());

    std::ostringstream root_json;
    root_json
        << "{\n"
        << "  \"artifact\": \"hotstep-ort-trt-engine-root\",\n"
        << "  \"modules\": [\"" << module_tag << "\"],\n"
        << "  \"builder_optimization_level\": " << hs_ort_env_int("HOTSTEP_ORT_TRT_BUILDER_OPT_LEVEL", 0, 0, 5) << "\n"
        << "}\n";
    write_text_file(std::filesystem::path(root_dir) / "ort-trt-engines.metadata.json", root_json.str());
}

static bool append_providers(Ort::SessionOptions & opts,
                             const std::string & onnx_path,
                             const std::string & artifact_fp,
                             const char * module_tag,
                             int device_id,
                             bool fp16,
                             size_t workspace_bytes) {
    try {
        Ort::TensorRTProviderOptions trt_opts;
        auto trt_map = hs_ort_trt_provider_options(
            onnx_path.c_str(), artifact_fp.c_str(), module_tag, device_id, fp16, workspace_bytes);
        trt_opts.Update(trt_map);
        opts.AppendExecutionProvider_TensorRT_V2(*trt_opts);
        std::fprintf(stderr, "[ace-ort-cache] TRT EP appended: module=%s cache=%s max=%s\n",
                     module_tag, trt_map["trt_engine_cache_path"].c_str(), trt_map["trt_profile_max_shapes"].c_str());
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[ace-ort-cache] FATAL: TensorRT EP unavailable for %s: %s\n", module_tag, e.what());
        return false;
    }
}

static bool append_cuda(Ort::SessionOptions & opts, int device_id) {
    try {
        OrtCUDAProviderOptions cuda_opts;
        std::memset(&cuda_opts, 0, sizeof(cuda_opts));
        cuda_opts.device_id = device_id;
        cuda_opts.arena_extend_strategy = 1;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[ace-ort-cache] WARNING: CUDA EP unavailable: %s\n", e.what());
        return false;
    }
}

static bool run_text(Ort::Session & session, int tokens) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input((size_t)tokens, 0);
    std::vector<int64_t> shape = {1, tokens};
    auto tensor = Ort::Value::CreateTensor<int64_t>(mem, input.data(), input.size(), shape.data(), shape.size());
    auto in_name = session.GetInputNameAllocated(0, alloc);
    auto out_name = session.GetOutputNameAllocated(0, alloc);
    const char * in_names[] = {in_name.get()};
    const char * out_names[] = {out_name.get()};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    return !outputs.empty();
}

static bool run_cond(Ort::Session & session, int text_tokens, int lyric_tokens, int timbre_frames) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> text((size_t)text_tokens * 1024, 0.0f);
    std::vector<float> lyric((size_t)lyric_tokens * 1024, 0.0f);
    std::vector<float> timbre((size_t)timbre_frames * 64, 0.0f);
    std::vector<int64_t> text_shape = {1, text_tokens, 1024};
    std::vector<int64_t> lyric_shape = {1, lyric_tokens, 1024};
    std::vector<int64_t> timbre_shape = {1, timbre_frames, 64};
    auto text_tensor = Ort::Value::CreateTensor<float>(mem, text.data(), text.size(), text_shape.data(), text_shape.size());
    auto lyric_tensor = Ort::Value::CreateTensor<float>(mem, lyric.data(), lyric.size(), lyric_shape.data(), lyric_shape.size());
    auto timbre_tensor = Ort::Value::CreateTensor<float>(mem, timbre.data(), timbre.size(), timbre_shape.data(), timbre_shape.size());
    Ort::Value inputs[] = {std::move(text_tensor), std::move(lyric_tensor), std::move(timbre_tensor)};
    const char * in_names[] = {"text_hidden", "lyric_embed", "timbre_feats"};
    const char * out_names[] = {"enc_hidden"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, inputs, 3, out_names, 1);
    return !outputs.empty();
}

static bool run_vae(Ort::Session & session, int frames) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> input((size_t)64 * frames, 0.0f);
    std::vector<int64_t> shape = {1, 64, frames};
    auto tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape.data(), shape.size());
    auto in_name = session.GetInputNameAllocated(0, alloc);
    auto out_name = session.GetOutputNameAllocated(0, alloc);
    const char * in_names[] = {in_name.get()};
    const char * out_names[] = {out_name.get()};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    return !outputs.empty();
}

static bool build_module(Ort::Env & env, const Args & args, const std::string & module_tag, const std::string & onnx_path) {
    if (!file_exists(onnx_path)) {
        std::fprintf(stderr, "[ace-ort-cache] FATAL: missing ONNX for %s: %s\n", module_tag.c_str(), onnx_path.c_str());
        return false;
    }

    std::string dir = dirname_of(onnx_path);
    std::vector<std::string> fp_paths = {onnx_path};
    bool fp16 = false;
    if (module_tag == "text-enc") {
        fp_paths.push_back(join_path(dir, "embed_tokens.bin"));
        fp_paths.push_back(join_path(dir, "vocab.json"));
        fp_paths.push_back(join_path(dir, "merges.txt"));
    } else if (module_tag == "cond-enc") {
        fp_paths.push_back(join_path(dir, "null_condition_emb.bin"));
    } else if (module_tag == "vae-dec") {
        fp16 = true;
    } else {
        std::fprintf(stderr, "[ace-ort-cache] FATAL: unknown module %s\n", module_tag.c_str());
        return false;
    }

    std::string artifact_fp = artifact_fingerprint(fp_paths);
    std::string cache_dir = hs_ort_trt_cache_dir(onnx_path.c_str(), artifact_fp.c_str(), module_tag.c_str());

    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(1);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (!append_providers(session_opts, onnx_path, artifact_fp, module_tag.c_str(), args.device_id, fp16, args.workspace_bytes)) {
        return false;
    }
    append_cuda(session_opts, args.device_id);

    try {
        std::fprintf(stderr, "[ace-ort-cache] creating session: %s\n", onnx_path.c_str());
        Ort::Session session = make_session(env, onnx_path, session_opts);
        bool ok = false;
        if (module_tag == "text-enc") {
            ok = run_text(session, args.text_opt_tokens);
        } else if (module_tag == "cond-enc") {
            ok = run_cond(session, args.text_opt_tokens, args.lyric_opt_tokens, args.timbre_opt_frames);
        } else {
            ok = run_vae(session, args.vae_opt_frames);
        }
        if (!ok) {
            std::fprintf(stderr, "[ace-ort-cache] FATAL: dry run failed for %s\n", module_tag.c_str());
            return false;
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[ace-ort-cache] FATAL: %s failed: %s\n", module_tag.c_str(), e.what());
        return false;
    }

    CacheSummary summary = validate_cache(cache_dir);
    if (summary.engine_count <= 0 || summary.profile_count <= 0) {
        std::fprintf(stderr, "[ace-ort-cache] FATAL: TensorRT cache incomplete for %s: %s\n",
                     module_tag.c_str(), cache_dir.c_str());
        return false;
    }

    write_cache_metadata(args.out_dir, cache_dir, module_tag, artifact_fp, summary);
    std::fprintf(stderr, "[ace-ort-cache] cache OK: module=%s engines=%d profiles=%d bytes=%llu\n",
                 module_tag.c_str(), summary.engine_count, summary.profile_count,
                 (unsigned long long)summary.engine_bytes);
    return true;
}

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        usage();
        return 2;
    }

    std::vector<std::string> modules = split_modules(args.modules);
    if (modules.empty()) {
        std::fprintf(stderr, "no modules selected\n");
        return 2;
    }
    if (args.out_dir.empty()) {
        const std::string * first = nullptr;
        if (!args.text_onnx.empty()) first = &args.text_onnx;
        else if (!args.cond_onnx.empty()) first = &args.cond_onnx;
        else if (!args.vae_onnx.empty()) first = &args.vae_onnx;
        if (!first) {
            std::fprintf(stderr, "--out-dir is required when no ONNX path is supplied\n");
            return 2;
        }
        args.out_dir = join_path(dirname_of(*first), "ort-trt-engines");
    }
    args.out_dir = normalize_path(args.out_dir);
    std::filesystem::create_directories(args.out_dir);

    set_env_value("HOTSTEP_ORT_TRT_ENGINE_ROOT", args.out_dir);
    set_env_value("HOTSTEP_ORT_TRT_BUILDER_OPT_LEVEL", args.builder_optimization_level);
    set_env_value("HOTSTEP_ORT_TRT_TEXT_OPT_TOKENS", args.text_opt_tokens);
    set_env_value("HOTSTEP_ORT_TRT_TEXT_MAX_TOKENS", args.text_max_tokens);
    set_env_value("HOTSTEP_ORT_TRT_LYRIC_OPT_TOKENS", args.lyric_opt_tokens);
    set_env_value("HOTSTEP_ORT_TRT_LYRIC_MAX_TOKENS", args.lyric_max_tokens);
    set_env_value("HOTSTEP_ORT_TRT_TIMBRE_OPT_FRAMES", args.timbre_opt_frames);
    set_env_value("HOTSTEP_ORT_TRT_TIMBRE_MAX_FRAMES", args.timbre_max_frames);
    set_env_value("HOTSTEP_ORT_TRT_VAE_OPT_FRAMES", args.vae_opt_frames);
    set_env_value("HOTSTEP_ORT_TRT_VAE_MAX_FRAMES", args.vae_max_frames);

    std::map<std::string, std::string> paths = {
        {"text-enc", args.text_onnx},
        {"cond-enc", args.cond_onnx},
        {"vae-dec", args.vae_onnx},
    };

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ace-ort-cache");
    for (const std::string & module : modules) {
        if (paths[module].empty()) {
            std::fprintf(stderr, "[ace-ort-cache] FATAL: missing path for module %s\n", module.c_str());
            return 2;
        }
        paths[module] = normalize_path(paths[module]);
        if (!build_module(env, args, module, paths[module])) {
            return 1;
        }
    }

    return 0;
}
