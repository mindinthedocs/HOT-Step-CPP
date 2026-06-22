#pragma once
// model-registry.h: scan directories for GGUF, SafeTensors, and ONNX models.
//
// Reads GGUF headers, SafeTensors config.json, and ONNX filenames to classify
// each model into lm/dit/text-enc/vae buckets.
// Adapter entries are .safetensors files or PEFT directories.
//
// Usage:
//   ModelRegistry reg;
//   registry_scan(&reg, "./models");
//   registry_scan_adapters(&reg, "./adapters");
//   const ModelEntry * dit = registry_find(reg.dit, "acestep-v15-turbo-Q8_0.gguf");

#include "config-json.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dirent.h>
#    include <sys/stat.h>
#endif

struct ModelEntry {
    std::string name;  // filename (e.g. "acestep-v15-turbo-Q8_0.gguf")
    std::string path;  // full path
};

struct AdapterEntry {
    std::string name;  // filename or directory name (e.g. "singer.safetensors" or "my-adapter")
    std::string path;  // full path (file or PEFT directory)
};

struct ModelRegistry {
    std::vector<ModelEntry>   lm;
    std::vector<ModelEntry>   dit;
    std::vector<ModelEntry>   text_enc;
    std::vector<ModelEntry>   vae;
    std::vector<ModelEntry>   pp_vae;  // PP-VAE (post-processing VAE, auto-detected)
    std::vector<AdapterEntry> adapters;
};

// find an entry by name in a bucket. returns NULL if not found.
static const ModelEntry * registry_find(const std::vector<ModelEntry> & bucket, const char * name) {
    for (const auto & e : bucket) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// Find the first non-ONNX entry in a bucket, or a specific named entry if it's non-ONNX.
// Use this when an ONNX model cannot serve the purpose (e.g., VAE encode — ONNX VAE files
// are decoder-only and don't contain encoder weights).
static const ModelEntry * registry_find_non_onnx(const std::vector<ModelEntry> & bucket, const char * name = nullptr) {
    auto is_onnx = [](const std::string & n) {
        return n.size() >= 5 && n.substr(n.size() - 5) == ".onnx";
    };
    if (name && name[0]) {
        // Specific name requested — return it only if it's not ONNX
        for (const auto & e : bucket) {
            if (e.name == name && !is_onnx(e.name)) {
                return &e;
            }
        }
        return nullptr;
    }
    // No specific name — return first non-ONNX entry
    for (const auto & e : bucket) {
        if (!is_onnx(e.name)) {
            return &e;
        }
    }
    return nullptr;
}

// find an adapter entry by name. returns NULL if not found.
static const AdapterEntry * registry_find_adapter(const ModelRegistry & reg, const char * name) {
    for (const auto & e : reg.adapters) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// classify a GGUF file by reading its header.
// returns: "lm", "dit", "text-enc", "vae", or "" if unrecognized.
static std::string registry_classify_gguf(const char * path) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context *   ctx    = gguf_init_from_file(path, params);
    if (!ctx) {
        return "";
    }

    std::string arch;
    int64_t     idx = gguf_find_key(ctx, "general.architecture");
    if (idx >= 0) {
        arch = gguf_get_val_str(ctx, idx);
    }
    gguf_free(ctx);

    // map GGUF architecture string to bucket name
    if (arch == "acestep-lm") {
        return "LM";
    }
    if (arch == "acestep-dit") {
        return "DiT";
    }
    if (arch == "acestep-text-enc") {
        return "Text-Enc";
    }
    if (arch == "acestep-vae") {
        return "VAE";
    }
    if (arch == "pp-vae") {
        return "PP-VAE";
    }
    return "";
}

// check if a string ends with a suffix
static bool str_ends_with(const std::string & s, const char * suffix) {
    size_t slen = strlen(suffix);
    return s.size() >= slen && s.compare(s.size() - slen, slen, suffix) == 0;
}

static std::string registry_classify_onnx_name(const std::string & name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "lm.onnx" || lower.find("lm_") == 0 || lower.find("lm-") == 0) {
        return "LM";
    }
    if (lower == "dit.onnx" || lower.find("dit_") == 0 || lower.find("dit-") == 0) {
        return "DiT";
    }
    if (lower == "vae.onnx" || lower.find("vae_") == 0 || lower.find("vae-") == 0) {
        return "VAE";
    }
    if (lower == "text_encoder.onnx" || lower.find("text_enc") == 0 ||
        lower.find("text-enc") == 0 || lower.find("text_encoder") == 0) {
        return "Text-Enc";
    }
    return "";
}

#ifdef _WIN32

// scan a directory for files matching a pattern (Windows)
static void registry_list_dir(const char * dir, std::vector<std::string> * names) {
    std::string      pattern = std::string(dir) + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

// list subdirectories (Windows)
static void registry_list_subdirs(const char * dir, std::vector<std::string> * names) {
    std::string      pattern = std::string(dir) + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && strcmp(fd.cFileName, ".") != 0 &&
            strcmp(fd.cFileName, "..") != 0) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

static bool registry_is_file(const char * path) {
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

#else

// scan a directory for files (POSIX)
static void registry_list_dir(const char * dir, std::vector<std::string> * names) {
    DIR * d = opendir(dir);
    if (!d) {
        return;
    }
    struct dirent * entry;
    while ((entry = readdir(d)) != nullptr) {
        // skip directories
        std::string full = std::string(dir) + "/" + entry->d_name;
        struct stat sb;
        if (stat(full.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
            names->push_back(entry->d_name);
        }
    }
    closedir(d);
}

// list subdirectories (POSIX)
static void registry_list_subdirs(const char * dir, std::vector<std::string> * names) {
    DIR * d = opendir(dir);
    if (!d) {
        return;
    }
    struct dirent * entry;
    while ((entry = readdir(d)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        std::string full = std::string(dir) + "/" + entry->d_name;
        struct stat sb;
        if (stat(full.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
            names->push_back(entry->d_name);
        }
    }
    closedir(d);
}

static bool registry_is_file(const char * path) {
    struct stat sb;
    return stat(path, &sb) == 0 && S_ISREG(sb.st_mode);
}

#endif

// path separator
#ifdef _WIN32
#    define REGISTRY_SEP "\\"
#else
#    define REGISTRY_SEP "/"
#endif

static void registry_add_onnx_entry(ModelRegistry * reg,
                                    const std::string & type,
                                    const std::string & name,
                                    const std::string & path) {
    ModelEntry entry = { name, path };
    if (type == "LM") {
        reg->lm.push_back(entry);
    } else if (type == "DiT") {
        reg->dit.push_back(entry);
    } else if (type == "Text-Enc") {
        reg->text_enc.push_back(entry);
    } else if (type == "VAE") {
        reg->vae.push_back(entry);
    }
}

static int registry_scan_onnx_bundle_dir(ModelRegistry * reg,
                                         const std::string & entry_prefix,
                                         const std::string & dir_path) {
    std::vector<std::string> files;
    registry_list_dir(dir_path.c_str(), &files);
    std::sort(files.begin(), files.end());

    int count = 0;
    for (const auto & f : files) {
        if (!str_ends_with(f, ".onnx")) {
            continue;
        }
        std::string type = registry_classify_onnx_name(f);
        if (type.empty()) {
            continue;
        }
        bool primary =
            (type == "DiT" && f == "dit.onnx") ||
            (type == "LM" && f == "lm_full.onnx") ||
            (type == "Text-Enc" && f == "text_encoder.onnx") ||
            (type == "VAE" && (f == "vae_decoder.onnx" || f == "vae.onnx"));
        std::string entry_name = primary ? entry_prefix : (entry_prefix + "/" + f);
        registry_add_onnx_entry(reg, type, entry_name, dir_path + REGISTRY_SEP + f);
        fprintf(stderr, "[Registry] %s -> %s (ONNX bundle: %s)\n",
                entry_name.c_str(), type.c_str(), f.c_str());
        count++;
    }
    return count;
}

// scan a directory for .gguf files, classify each by architecture.
// returns true if at least one model was found.
static bool registry_scan(ModelRegistry * reg, const char * models_dir) {
    std::vector<std::string> files;
    registry_list_dir(models_dir, &files);
    std::sort(files.begin(), files.end());

    int count = 0;
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".gguf")) {
            continue;
        }

        std::string full = std::string(models_dir) + REGISTRY_SEP + fname;
        std::string type = registry_classify_gguf(full.c_str());
        if (type.empty()) {
            fprintf(stderr, "[Registry] WARNING: skipping %s (unknown architecture)\n", fname.c_str());
            continue;
        }

        ModelEntry entry = { fname, full };
        if (type == "LM") {
            reg->lm.push_back(entry);
        } else if (type == "DiT") {
            reg->dit.push_back(entry);
        } else if (type == "Text-Enc") {
            reg->text_enc.push_back(entry);
        } else if (type == "VAE") {
            reg->vae.push_back(entry);
        } else if (type == "PP-VAE") {
            reg->pp_vae.push_back(entry);
        }

        fprintf(stderr, "[Registry] %s -> %s\n", fname.c_str(), type.c_str());
        count++;
    }

    // Scan for safetensors VAE files (classified by filename prefix)
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".safetensors")) {
            continue;
        }

        // Classify by lowercase filename prefix
        std::string lower = fname;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        std::string full = std::string(models_dir) + REGISTRY_SEP + fname;
        ModelEntry  entry = { fname, full };

        if (lower.find("pp-vae") == 0 || lower.find("pp_vae") == 0) {
            reg->pp_vae.push_back(entry);
            fprintf(stderr, "[Registry] %s -> PP-VAE (safetensors)\n", fname.c_str());
            count++;
        } else if (lower.find("vae") == 0 || lower.find("scragvae") == 0) {
            reg->vae.push_back(entry);
            fprintf(stderr, "[Registry] %s -> VAE (safetensors)\n", fname.c_str());
            count++;
        }
    }

    // Scan for ONNX files (pre-exported models for TRT acceleration)
    // Classified by filename prefix: dit_* -> DiT, vae_* -> VAE, etc.
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".onnx")) {
            continue;
        }

        std::string full = std::string(models_dir) + REGISTRY_SEP + fname;
        ModelEntry  entry = { fname, full };
        std::string type  = registry_classify_onnx_name(fname);

        if (type == "DiT") {
            reg->dit.push_back(entry);
            fprintf(stderr, "[Registry] %s -> DiT (ONNX)\n", fname.c_str());
            count++;
        } else if (type == "VAE") {
            reg->vae.push_back(entry);
            fprintf(stderr, "[Registry] %s -> VAE (ONNX)\n", fname.c_str());
            count++;
        } else if (type == "LM") {
            reg->lm.push_back(entry);
            fprintf(stderr, "[Registry] %s -> LM (ONNX)\n", fname.c_str());
            count++;
        } else if (type == "Text-Enc") {
            reg->text_enc.push_back(entry);
            fprintf(stderr, "[Registry] %s -> Text-Enc (ONNX)\n", fname.c_str());
            count++;
        } else {
            fprintf(stderr, "[Registry] WARNING: skipping %s (unrecognized ONNX prefix)\n", fname.c_str());
        }
    }

    // Scan subdirectories for safetensors model checkpoints (HuggingFace format).
    // Each subdirectory should contain model.safetensors (or sharded index) + config.json.
    // Classification uses config.json content, not directory name.
    std::vector<std::string> subdirs;
    registry_list_subdirs(models_dir, &subdirs);
    std::sort(subdirs.begin(), subdirs.end());

    for (const auto & dname : subdirs) {
        std::string dir_path = std::string(models_dir) + REGISTRY_SEP + dname;

        // Must have config.json
        std::string cfg_path = dir_path + REGISTRY_SEP + "config.json";
        if (!registry_is_file(cfg_path.c_str())) {
            continue;
        }

        // Must have model.safetensors or model.safetensors.index.json (sharded)
        std::string st_path = dir_path + REGISTRY_SEP + "model.safetensors";
        std::string st_index = dir_path + REGISTRY_SEP + "model.safetensors.index.json";
        if (!registry_is_file(st_path.c_str()) && !registry_is_file(st_index.c_str())) {
            continue;
        }

        // Classify by config.json content
        std::string type = config_json_classify(cfg_path.c_str());
        if (type.empty()) {
            fprintf(stderr, "[Registry] WARNING: skipping %s/ (unrecognized config.json)\n", dname.c_str());
            continue;
        }

        // ModelEntry.path is the directory path for safetensors models.
        // The loader will find model.safetensors + sidecar files inside.
        ModelEntry entry = { dname, dir_path };
        if (type == "LM") {
            reg->lm.push_back(entry);
        } else if (type == "DiT") {
            reg->dit.push_back(entry);
        } else if (type == "Text-Enc") {
            reg->text_enc.push_back(entry);
        } else if (type == "VAE") {
            reg->vae.push_back(entry);
        }

        bool is_sharded = registry_is_file(st_index.c_str());
        fprintf(stderr, "[Registry] %s/ -> %s (safetensors%s)\n", dname.c_str(), type.c_str(),
                is_sharded ? ", sharded" : "");
        count++;
    }

    // Scan subdirectories for ONNX model directories (TRT acceleration).
    // Each subdirectory should contain at least one .onnx file.
    // Classification by .onnx filename prefix (lm_* -> LM, dit_* -> DiT, etc.)
    // or by directory name prefix if ONNX filename is ambiguous.
    for (const auto & dname : subdirs) {
        std::string dir_path = std::string(models_dir) + REGISTRY_SEP + dname;

        // Skip directories already registered (safetensors path above)
        // by checking for model.safetensors — if it has that, it was already handled
        std::string st_check = dir_path + REGISTRY_SEP + "model.safetensors";
        std::string st_idx   = dir_path + REGISTRY_SEP + "model.safetensors.index.json";
        if (registry_is_file(st_check.c_str()) || registry_is_file(st_idx.c_str())) {
            continue;
        }

        // Register each recognized ONNX file in this subdirectory. Runtime
        // bundles contain dit.onnx plus sibling text/VAE files, so choosing an
        // arbitrary first .onnx would miss the DiT in some directory orders.
        std::vector<std::string> sub_files;
        registry_list_dir(dir_path.c_str(), &sub_files);
        std::sort(sub_files.begin(), sub_files.end());
        int onnx_count = 0;
        for (const auto & f : sub_files) {
            if (!str_ends_with(f, ".onnx")) {
                continue;
            }
            std::string type = registry_classify_onnx_name(f);
            if (type.empty()) {
                continue;
            }

            std::string entry_name = (type == "DiT") ? dname : (dname + "/" + f);
            ModelEntry entry = { entry_name, dir_path + REGISTRY_SEP + f };
            if (type == "LM") {
                reg->lm.push_back(entry);
            } else if (type == "DiT") {
                reg->dit.push_back(entry);
            } else if (type == "Text-Enc") {
                reg->text_enc.push_back(entry);
            } else if (type == "VAE") {
                reg->vae.push_back(entry);
            }
            fprintf(stderr, "[Registry] %s -> %s (ONNX: %s)\n", entry_name.c_str(), type.c_str(), f.c_str());
            onnx_count++;
            count++;
        }
        if (onnx_count == 0) {
            fprintf(stderr, "[Registry] WARNING: skipping %s/ (unrecognized ONNX directory)\n", dname.c_str());
        }
    }

    // Scan structured prebuilt bundle roots:
    //   models/dit/<bundle>/dit.onnx
    //   models/embedding/<bundle>/text_encoder.onnx
    //   models/lm/<bundle>/lm_full.onnx
    //   models/vae/<bundle>/vae_decoder.onnx
    const char * bundle_roots[] = { "dit", "embedding", "lm", "vae" };
    for (const char * root_name : bundle_roots) {
        std::string root_path = std::string(models_dir) + REGISTRY_SEP + root_name;
        std::vector<std::string> bundle_dirs;
        registry_list_subdirs(root_path.c_str(), &bundle_dirs);
        if (bundle_dirs.empty()) {
            continue;
        }
        std::sort(bundle_dirs.begin(), bundle_dirs.end());
        for (const auto & bundle_name : bundle_dirs) {
            std::string bundle_path = root_path + REGISTRY_SEP + bundle_name;
            std::string entry_prefix = std::string(root_name) + "/" + bundle_name;
            count += registry_scan_onnx_bundle_dir(reg, entry_prefix, bundle_path);
        }
    }

    // Scan native TRT bundles: models/trt-bundles/<name>/ each carrying a
    // manifest.json. Two bundle shapes are supported:
    //   - DiT bundle: manifest.json + dit.onnx (+ cond_encoder.onnx). Registered
    //     as a DiT entry so a synth request can select it via synth_model="<name>".
    //   - Qwen3-emb bundle: manifest.json + text_encoder.onnx (no dit.onnx).
    //     Registered as a Text-Enc entry so the user can select it as the text
    //     encoder. The synth path detects the sibling manifest.json and routes
    //     the text-encoder forward through the TRT engine.
    // Without this scan the generic subdir scan skips trt-bundles/ — its .onnx
    // files live one level deeper, in trt-bundles/<name>/ — so the bundles would
    // show in /props but not be selectable as DiT or Text-Enc.
    {
        std::string trt_root = std::string(models_dir) + REGISTRY_SEP + "trt-bundles";
        std::vector<std::string> bundle_dirs;
        registry_list_subdirs(trt_root.c_str(), &bundle_dirs);
        std::sort(bundle_dirs.begin(), bundle_dirs.end());
        for (const auto & bundle_name : bundle_dirs) {
            std::string bundle_path   = trt_root + REGISTRY_SEP + bundle_name;
            std::string manifest_path = bundle_path + REGISTRY_SEP + "manifest.json";
            if (!registry_is_file(manifest_path.c_str())) {
                continue;  // not a TRT bundle — no manifest.json
            }
            // registry_scan_onnx_bundle_dir registers whichever .onnx files are
            // present: dit.onnx (DiT), text_encoder.onnx (Text-Enc), etc. A DiT
            // bundle has dit.onnx; a Qwen3-emb bundle has text_encoder.onnx only.
            count += registry_scan_onnx_bundle_dir(reg, bundle_name, bundle_path);
        }
    }

    return count > 0;
}

// scan a directory for adapters.
// - .safetensors files: ComfyUI single-file format (alpha baked in)
// - subdirectories containing adapter_model.safetensors: PEFT format
// returns true if at least one adapter was found.
static bool registry_scan_adapters(ModelRegistry * reg, const char * adapters_dir) {
    int count = 0;

    // single .safetensors files
    std::vector<std::string> files;
    registry_list_dir(adapters_dir, &files);
    std::sort(files.begin(), files.end());
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".safetensors")) {
            continue;
        }
        std::string full = std::string(adapters_dir) + REGISTRY_SEP + fname;
        reg->adapters.push_back({ fname, full });
        fprintf(stderr, "[Registry] Adapter: %s (ComfyUI)\n", fname.c_str());
        count++;
    }

    // PEFT directories (contain adapter_model.safetensors)
    std::vector<std::string> subdirs;
    registry_list_subdirs(adapters_dir, &subdirs);
    std::sort(subdirs.begin(), subdirs.end());
    for (const auto & dname : subdirs) {
        std::string adapter =
            std::string(adapters_dir) + REGISTRY_SEP + dname + REGISTRY_SEP + "adapter_model.safetensors";
        if (registry_is_file(adapter.c_str())) {
            std::string full = std::string(adapters_dir) + REGISTRY_SEP + dname;
            reg->adapters.push_back({ dname, full });
            fprintf(stderr, "[Registry] Adapter: %s (PEFT)\n", dname.c_str());
            count++;
        }
    }

    return count > 0;
}
