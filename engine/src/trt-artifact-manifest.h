// trt-artifact-manifest.h: lightweight validation for DiT TRT sidecars.
#pragma once

#include "yyjson.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

struct TrtPrecisionManifestCheck {
    std::string path;
    std::string policy;
    size_t      matched_allowlist_count = 0;
    size_t      downcast_to_fp16_count  = 0;
    size_t      quantized_to_int8_count = 0;
    size_t      preserved_fp32_count    = 0;
    size_t      missing_initializer_count = 0;
    size_t      unmatched_pattern_count = 0;
    double      all_parameter_count = 0.0;
    double      matrix_parameter_count = 0.0;
    double      non_matrix_parameter_count = 0.0;
};

struct TrtEngineMetadataCheck {
    std::string path;
    std::string source_path;
    std::string policy;
    std::string profile;
    std::string tensorrt_version;
    int         tensorrt_major = -1;
    int         tensorrt_minor = -1;
    bool        weight_streaming = false;
    int         profile_max_batch = 0;
    int         profile_max_T = 0;
    int         profile_max_enc_S = 0;
};

static const int TRT_ARTIFACT_NATIVE_PROFILE_MAX_BATCH = 2;
// max_T = 3000 = 120s × 25Hz (2-minute test profile for 6GB cards).
// The DiT internally patchifies T→T/2 (12.5Hz), so the max internal sequence
// length is 1500 tokens. For 10-minute songs (max_T=15000) use the
// "full-10min" Python profile + rebuild — requires ≥8GB VRAM.
static const int TRT_ARTIFACT_NATIVE_PROFILE_MAX_T = 3000;
static const int TRT_ARTIFACT_NATIVE_PROFILE_MAX_ENC_S = 2048;

static inline void trt_artifact_set_error(std::string * err, const std::string & msg) {
    if (err) {
        *err = msg;
    }
}

static inline bool trt_artifact_read_file(const std::string & path, std::string & out, std::string * err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        trt_artifact_set_error(err, "cannot open " + path);
        return false;
    }
    out.clear();
    char buf[8192];
    for (;;) {
        size_t n = fread(buf, 1, sizeof(buf), f);
        if (n) {
            out.append(buf, n);
        }
        if (n < sizeof(buf)) {
            if (ferror(f)) {
                fclose(f);
                trt_artifact_set_error(err, "cannot read " + path);
                return false;
            }
            break;
        }
    }
    fclose(f);
    return true;
}

static inline std::string trt_artifact_stem_from_onnx(const std::string & onnx_path) {
    if (onnx_path.size() >= 5 && onnx_path.substr(onnx_path.size() - 5) == ".onnx") {
        return onnx_path.substr(0, onnx_path.size() - 5);
    }
    return onnx_path;
}

static inline std::string trt_artifact_precision_manifest_path(const std::string & onnx_path) {
    return trt_artifact_stem_from_onnx(onnx_path) + ".precision-manifest.json";
}

static inline std::string trt_artifact_dirname(const std::string & path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return std::string();
    }
    return path.substr(0, pos);
}

static inline bool trt_artifact_is_abs_path(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    if (path[0] == '/' || path[0] == '\\') {
        return true;
    }
    return path.size() >= 3 && path[1] == ':' && (path[2] == '\\' || path[2] == '/');
}

static inline std::string trt_artifact_join_relative(const std::string & base_file,
                                                     const std::string & maybe_relative) {
    if (maybe_relative.empty() || trt_artifact_is_abs_path(maybe_relative)) {
        return maybe_relative;
    }
    std::string dir = trt_artifact_dirname(base_file);
    if (dir.empty()) {
        return maybe_relative;
    }
    return dir + "/" + maybe_relative;
}

static inline yyjson_doc * trt_artifact_read_json(const std::string & path, std::string * err) {
    std::string text;
    if (!trt_artifact_read_file(path, text, err)) {
        return nullptr;
    }
    yyjson_doc * doc = yyjson_read(text.c_str(), text.size(), 0);
    if (!doc) {
        trt_artifact_set_error(err, "invalid JSON in " + path);
        return nullptr;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "JSON root is not an object in " + path);
        return nullptr;
    }
    return doc;
}

static inline bool trt_artifact_get_string(yyjson_val * root,
                                           const char * key,
                                           std::string & out,
                                           std::string * err) {
    yyjson_val * val = yyjson_obj_get(root, key);
    if (!val || !yyjson_is_str(val)) {
        trt_artifact_set_error(err, std::string("missing string field: ") + key);
        return false;
    }
    out.assign(yyjson_get_str(val), yyjson_get_len(val));
    return true;
}

static inline bool trt_artifact_get_bool(yyjson_val * root,
                                         const char * key,
                                         bool & out,
                                         std::string * err) {
    yyjson_val * val = yyjson_obj_get(root, key);
    if (!val || !yyjson_is_bool(val)) {
        trt_artifact_set_error(err, std::string("missing bool field: ") + key);
        return false;
    }
    out = yyjson_get_bool(val);
    return true;
}

static inline bool trt_artifact_get_num_optional(yyjson_val * root,
                                                 const char * key,
                                                 double & out,
                                                 std::string * err) {
    yyjson_val * val = yyjson_obj_get(root, key);
    if (!val) {
        out = 0.0;
        return true;
    }
    if (!yyjson_is_num(val)) {
        trt_artifact_set_error(err, std::string("field is not numeric: ") + key);
        return false;
    }
    out = yyjson_get_num(val);
    return true;
}

static inline bool trt_artifact_string_array(yyjson_val * root,
                                             const char * key,
                                             std::vector<std::string> * values,
                                             std::unordered_set<std::string> * value_set,
                                             size_t * count,
                                             std::string * err) {
    yyjson_val * arr = yyjson_obj_get(root, key);
    if (!arr || !yyjson_is_arr(arr)) {
        trt_artifact_set_error(err, std::string("missing string array field: ") + key);
        return false;
    }
    if (values) {
        values->clear();
    }
    if (value_set) {
        value_set->clear();
    }
    if (count) {
        *count = yyjson_arr_size(arr);
    }
    size_t idx, max;
    yyjson_val * item;
    yyjson_arr_foreach(arr, idx, max, item) {
        if (!yyjson_is_str(item)) {
            trt_artifact_set_error(err, std::string("array field contains non-string item: ") + key);
            return false;
        }
        std::string s(yyjson_get_str(item), yyjson_get_len(item));
        if (values) {
            values->push_back(s);
        }
        if (value_set) {
            value_set->insert(s);
        }
    }
    return true;
}

static inline bool trt_artifact_string_array_optional(yyjson_val * root,
                                                      const char * key,
                                                      std::vector<std::string> * values,
                                                      std::unordered_set<std::string> * value_set,
                                                      size_t * count,
                                                      std::string * err) {
    yyjson_val * arr = yyjson_obj_get(root, key);
    if (!arr) {
        if (values) values->clear();
        if (value_set) value_set->clear();
        if (count) *count = 0;
        return true;
    }
    return trt_artifact_string_array(root, key, values, value_set, count, err);
}

static inline bool trt_artifact_allowlist_counts_ok(yyjson_val * root, std::string * err) {
    yyjson_val * counts = yyjson_obj_get(root, "allowlist_pattern_match_counts");
    if (!counts || !yyjson_is_obj(counts)) {
        trt_artifact_set_error(err, "missing object field: allowlist_pattern_match_counts");
        return false;
    }
    if (yyjson_obj_size(counts) == 0) {
        trt_artifact_set_error(err, "allowlist_pattern_match_counts is empty");
        return false;
    }
    size_t idx, max;
    yyjson_val * key;
    yyjson_val * val;
    yyjson_obj_foreach(counts, idx, max, key, val) {
        if (!yyjson_is_num(val)) {
            trt_artifact_set_error(err, "allowlist_pattern_match_counts contains a non-numeric value");
            return false;
        }
        if (yyjson_get_num(val) <= 0.0) {
            std::string name(yyjson_get_str(key), yyjson_get_len(key));
            trt_artifact_set_error(err, "allowlist pattern matched zero tensors: " + name);
            return false;
        }
    }
    return true;
}

static inline bool trt_artifact_validate_precision_manifest(const std::string & onnx_path,
                                                            TrtPrecisionManifestCheck * out,
                                                            std::string * err) {
    TrtPrecisionManifestCheck check;
    check.path = trt_artifact_precision_manifest_path(onnx_path);

    yyjson_doc * doc = trt_artifact_read_json(check.path, err);
    if (!doc) {
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);

    if (!trt_artifact_get_string(root, "precision_policy", check.policy, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (check.policy != "q8map-fp16" && check.policy != "w8a8" && check.policy != "fp32") {
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "unsupported DiT precision policy: " + check.policy);
        return false;
    }

    std::unordered_set<std::string> matched_set;
    std::unordered_set<std::string> downcast_set;
    std::unordered_set<std::string> quantized_set;
    if (!trt_artifact_string_array(root, "matched_allowlist", nullptr, &matched_set,
                                   &check.matched_allowlist_count, err) ||
        !trt_artifact_string_array(root, "downcast_to_fp16", nullptr, &downcast_set,
                                   &check.downcast_to_fp16_count, err) ||
        !trt_artifact_string_array_optional(root, "quantized_to_int8", nullptr, &quantized_set,
                                            &check.quantized_to_int8_count, err) ||
        !trt_artifact_string_array(root, "preserved_fp32", nullptr, nullptr,
                                   &check.preserved_fp32_count, err) ||
        !trt_artifact_string_array(root, "missing_initializers", nullptr, nullptr,
                                   &check.missing_initializer_count, err) ||
        !trt_artifact_string_array(root, "unmatched_allowlist_patterns", nullptr, nullptr,
                                   &check.unmatched_pattern_count, err) ||
        !trt_artifact_get_num_optional(root, "all_parameter_count", check.all_parameter_count, err) ||
        !trt_artifact_get_num_optional(root, "matrix_parameter_count", check.matrix_parameter_count, err) ||
        !trt_artifact_get_num_optional(root, "non_matrix_parameter_count", check.non_matrix_parameter_count, err)) {
        yyjson_doc_free(doc);
        return false;
    }

    if (check.missing_initializer_count != 0) {
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "precision manifest reports missing ONNX initializers");
        return false;
    }

    if (check.policy == "q8map-fp16") {
        if (check.unmatched_pattern_count != 0) {
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "precision manifest reports unmatched allowlist patterns");
            return false;
        }
        if (check.matched_allowlist_count == 0 || check.downcast_to_fp16_count == 0) {
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "q8map-fp16 precision manifest matched or downcasted zero tensors");
            return false;
        }
        if (matched_set.size() != downcast_set.size() ||
            check.matched_allowlist_count != check.downcast_to_fp16_count) {
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "precision manifest downcast count does not match allowlist count");
            return false;
        }
        for (const std::string & name : matched_set) {
            if (downcast_set.find(name) == downcast_set.end()) {
                yyjson_doc_free(doc);
                trt_artifact_set_error(err, "precision manifest downcast set does not match allowlist set");
                return false;
            }
        }
        if (check.quantized_to_int8_count != 0) {
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "q8map-fp16 precision manifest unexpectedly contains INT8 quantized tensors");
            return false;
        }
        if (!trt_artifact_allowlist_counts_ok(root, err)) {
            yyjson_doc_free(doc);
            return false;
        }
    } else if (check.policy == "w8a8") {
        // w8a8 + ConvRot: INT8 weights AND INT8 activations, fused by the
        // ConvRotInt8Linear TRT plugin. Every matched matrix weight is
        // quantized to INT8 with per-output-channel symmetric scale, and
        // the rewrite emits a single ConvRotInt8Linear custom-op node per
        // MatMul/Gemm site. The manifest must record convrot metadata
        // (group_size, hadamard_initializer) so downstream consumers
        // can verify the rotation group size.
        if (check.unmatched_pattern_count != 0) {
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "precision manifest reports unmatched allowlist patterns");
            return false;
        }
        if (check.matched_allowlist_count == 0 || check.quantized_to_int8_count == 0) {
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "w8a8 precision manifest matched or quantized zero tensors");
            return false;
        }
        if (check.downcast_to_fp16_count != 0) {
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "w8a8 precision manifest unexpectedly contains downcast tensors");
            return false;
        }
        if (matched_set.size() != quantized_set.size() ||
            check.matched_allowlist_count != check.quantized_to_int8_count) {
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "precision manifest INT8 quantized count does not match allowlist count");
            return false;
        }
        for (const std::string & name : matched_set) {
            if (quantized_set.find(name) == quantized_set.end()) {
                yyjson_doc_free(doc);
                trt_artifact_set_error(err, "precision manifest INT8 quantized set does not match allowlist set");
                return false;
            }
        }
        // convrot metadata block (group_size, hadamard_initializer name)
        yyjson_val * convrot = yyjson_obj_get(root, "convrot");
        if (!convrot || !yyjson_is_obj(convrot)) {
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "w8a8 precision manifest is missing the convrot metadata block");
            return false;
        }
        yyjson_val * convrot_enabled = yyjson_obj_get(convrot, "enabled");
        if (!convrot_enabled || !yyjson_is_bool(convrot_enabled)) {
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "w8a8 precision manifest convrot.enabled must be a boolean");
            return false;
        }
        if (yyjson_get_bool(convrot_enabled)) {
            yyjson_val * gs = yyjson_obj_get(convrot, "group_size");
            if (!gs || !yyjson_is_num(gs)) {
                yyjson_doc_free(doc);
                trt_artifact_set_error(err, "w8a8 precision manifest convrot.group_size must be a number when ConvRot is enabled");
                return false;
            }
            yyjson_val * h_name = yyjson_obj_get(convrot, "hadamard_initializer");
            if (!h_name || !yyjson_is_str(h_name) || yyjson_get_len(h_name) == 0) {
                yyjson_doc_free(doc);
                trt_artifact_set_error(err, "w8a8 precision manifest convrot.hadamard_initializer must be a non-empty string when ConvRot is enabled");
                return false;
            }
        }
        if (!trt_artifact_allowlist_counts_ok(root, err)) {
            yyjson_doc_free(doc);
            return false;
        }
    } else if (check.downcast_to_fp16_count != 0) {
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "fp32 precision manifest unexpectedly contains downcast tensors");
        return false;
    } else if (check.quantized_to_int8_count != 0) {
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "fp32 precision manifest unexpectedly contains INT8 quantized tensors");
        return false;
    }

    yyjson_doc_free(doc);
    if (out) {
        *out = check;
    }
    return true;
}

static inline bool trt_artifact_parse_version(const std::string & version, int & major, int & minor) {
    major = -1;
    minor = -1;
    const char * p = version.c_str();
    char * end = nullptr;
    long maj = std::strtol(p, &end, 10);
    if (end == p || maj < 0) {
        return false;
    }
    major = (int) maj;
    if (*end == '.') {
        p = end + 1;
        long min = std::strtol(p, &end, 10);
        if (end == p || min < 0) {
            return false;
        }
        minor = (int) min;
    }
    return true;
}

static inline bool trt_artifact_shape_dim(yyjson_val * arr, size_t idx, int & out, std::string * err) {
    if (!arr || !yyjson_is_arr(arr)) {
        trt_artifact_set_error(err, "profile shape is not an array");
        return false;
    }
    yyjson_val * val = yyjson_arr_get(arr, idx);
    if (!val || !yyjson_is_int(val)) {
        trt_artifact_set_error(err, "profile shape contains a missing or non-integer dimension");
        return false;
    }
    out = yyjson_get_int(val);
    if (out <= 0) {
        trt_artifact_set_error(err, "profile shape contains a non-positive dimension");
        return false;
    }
    return true;
}

static inline bool trt_artifact_shape_rank(yyjson_val * arr, size_t expected, std::string * err) {
    if (!arr || !yyjson_is_arr(arr)) {
        trt_artifact_set_error(err, "profile shape is not an array");
        return false;
    }
    if (yyjson_arr_size(arr) != expected) {
        trt_artifact_set_error(err, "engine metadata profile_shapes has unexpected DiT rank");
        return false;
    }
    return true;
}

static inline yyjson_val * trt_artifact_profile_max_shape(yyjson_val * shapes, const char * tensor_name) {
    // Python builder layout:
    //   profile_shapes.max.input_latents = [N, T, C]
    yyjson_val * max_by_selector = yyjson_obj_get(shapes, "max");
    if (max_by_selector && yyjson_is_obj(max_by_selector)) {
        yyjson_val * arr = yyjson_obj_get(max_by_selector, tensor_name);
        if (arr) {
            return arr;
        }
    }

    // Native first-use layout:
    //   profile_shapes.input_latents.max = [N, T, C]
    yyjson_val * by_tensor = yyjson_obj_get(shapes, tensor_name);
    if (by_tensor && yyjson_is_obj(by_tensor)) {
        return yyjson_obj_get(by_tensor, "max");
    }
    return nullptr;
}

static inline bool trt_artifact_parse_profile_bounds(yyjson_val * shapes,
                                                     TrtEngineMetadataCheck & check,
                                                     std::string * err) {
    yyjson_val * input_max = trt_artifact_profile_max_shape(shapes, "input_latents");
    yyjson_val * enc_max   = trt_artifact_profile_max_shape(shapes, "enc_hidden");
    yyjson_val * t_max     = trt_artifact_profile_max_shape(shapes, "t");
    yyjson_val * t_r_max   = trt_artifact_profile_max_shape(shapes, "t_r");
    if (!input_max || !enc_max || !t_max || !t_r_max) {
        trt_artifact_set_error(err, "engine metadata profile_shapes is missing max bounds");
        return false;
    }
    int input_batch = 0;
    int input_T = 0;
    int input_channels = 0;
    int enc_batch = 0;
    int enc_S = 0;
    int enc_channels = 0;
    int t_batch = 0;
    int t_r_batch = 0;
    if (!trt_artifact_shape_rank(input_max, 3, err) ||
        !trt_artifact_shape_rank(enc_max, 3, err) ||
        !trt_artifact_shape_rank(t_max, 1, err) ||
        !trt_artifact_shape_rank(t_r_max, 1, err) ||
        !trt_artifact_shape_dim(input_max, 0, input_batch, err) ||
        !trt_artifact_shape_dim(input_max, 1, input_T, err) ||
        !trt_artifact_shape_dim(input_max, 2, input_channels, err) ||
        !trt_artifact_shape_dim(enc_max, 0, enc_batch, err) ||
        !trt_artifact_shape_dim(enc_max, 1, enc_S, err) ||
        !trt_artifact_shape_dim(enc_max, 2, enc_channels, err) ||
        !trt_artifact_shape_dim(t_max, 0, t_batch, err) ||
        !trt_artifact_shape_dim(t_r_max, 0, t_r_batch, err)) {
        return false;
    }
    if (input_channels != 192 || enc_channels != 2048) {
        trt_artifact_set_error(err, "engine metadata profile_shapes has unexpected DiT channel dimensions");
        return false;
    }
    if (input_batch != enc_batch || input_batch != t_batch || input_batch != t_r_batch) {
        trt_artifact_set_error(err, "engine metadata profile_shapes batch bounds are inconsistent");
        return false;
    }
    check.profile_max_batch = input_batch;
    check.profile_max_T = input_T;
    check.profile_max_enc_S = enc_S;
    return true;
}

static inline bool trt_artifact_engine_profile_covers(const TrtEngineMetadataCheck & check,
                                                      int required_batch,
                                                      int required_T,
                                                      int required_enc_S,
                                                      std::string * err) {
    if (check.profile_max_batch <= 0 || check.profile_max_T <= 0 || check.profile_max_enc_S <= 0) {
        trt_artifact_set_error(err, "engine metadata profile max bounds were not parsed");
        return false;
    }
    if (required_batch > check.profile_max_batch ||
        required_T > check.profile_max_T ||
        required_enc_S > check.profile_max_enc_S) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "engine profile too small: need batch=%d T=%d enc_S=%d, max batch=%d T=%d enc_S=%d",
                 required_batch, required_T, required_enc_S,
                 check.profile_max_batch, check.profile_max_T, check.profile_max_enc_S);
        trt_artifact_set_error(err, buf);
        return false;
    }
    return true;
}

static inline bool trt_artifact_native_engine_profile_covers(int required_batch,
                                                             int required_T,
                                                             int required_enc_S,
                                                             std::string * err) {
    TrtEngineMetadataCheck native_check;
    native_check.profile = "native-default";
    native_check.profile_max_batch = TRT_ARTIFACT_NATIVE_PROFILE_MAX_BATCH;
    native_check.profile_max_T = TRT_ARTIFACT_NATIVE_PROFILE_MAX_T;
    native_check.profile_max_enc_S = TRT_ARTIFACT_NATIVE_PROFILE_MAX_ENC_S;
    return trt_artifact_engine_profile_covers(native_check, required_batch, required_T, required_enc_S, err);
}

static inline bool trt_artifact_load_effective_engine_metadata(const std::string & metadata_path,
                                                               yyjson_doc ** doc_out,
                                                               yyjson_doc ** primary_doc_out,
                                                               yyjson_val ** root_out,
                                                               std::string * source_path_out,
                                                               std::string * err) {
    *doc_out = trt_artifact_read_json(metadata_path, err);
    if (!*doc_out) {
        return false;
    }
    *primary_doc_out = nullptr;
    yyjson_val * root = yyjson_doc_get_root(*doc_out);

    if (!yyjson_obj_get(root, "precision_policy")) {
        yyjson_val * primary = yyjson_obj_get(root, "primary_metadata");
        if (primary && yyjson_is_str(primary)) {
            std::string primary_path(yyjson_get_str(primary), yyjson_get_len(primary));
            std::string primary_relative = trt_artifact_join_relative(metadata_path, primary_path);
            std::string primary_err;
            *primary_doc_out = trt_artifact_read_json(primary_relative, &primary_err);
            if (!*primary_doc_out && primary_relative != primary_path) {
                *primary_doc_out = trt_artifact_read_json(primary_path, &primary_err);
                if (*primary_doc_out) {
                    primary_relative = primary_path;
                }
            }
            if (!*primary_doc_out) {
                yyjson_doc_free(*doc_out);
                *doc_out = nullptr;
                trt_artifact_set_error(err, "metadata alias points to unreadable primary metadata: " + primary_path);
                return false;
            }
            root = yyjson_doc_get_root(*primary_doc_out);
            if (source_path_out) {
                *source_path_out = primary_relative;
            }
        }
    }

    if (source_path_out && source_path_out->empty()) {
        *source_path_out = metadata_path;
    }
    *root_out = root;
    return true;
}

static inline bool trt_artifact_validate_engine_metadata(const std::string & metadata_path,
                                                         const std::string & expected_policy,
                                                         int expected_trt_major,
                                                         int expected_trt_minor,
                                                         TrtEngineMetadataCheck * out,
                                                         std::string * err) {
    yyjson_doc * doc = nullptr;
    yyjson_doc * primary_doc = nullptr;
    yyjson_val * root = nullptr;
    TrtEngineMetadataCheck check;
    check.path = metadata_path;
    if (!trt_artifact_load_effective_engine_metadata(metadata_path, &doc, &primary_doc,
                                                     &root, &check.source_path, err)) {
        return false;
    }

    bool strong = false;
    bool global_fp16 = false;
    bool global_bf16 = false;
    bool strip_plan = false;
    bool refit_identical = false;
    bool weight_streaming = false;
    if (!trt_artifact_get_string(root, "precision_policy", check.policy, err) ||
        !trt_artifact_get_string(root, "profile", check.profile, err) ||
        !trt_artifact_get_string(root, "tensorrt_version", check.tensorrt_version, err) ||
        !trt_artifact_get_bool(root, "strongly_typed_network", strong, err) ||
        !trt_artifact_get_bool(root, "global_fp16_builder_flag", global_fp16, err) ||
        !trt_artifact_get_bool(root, "global_bf16_builder_flag", global_bf16, err) ||
        !trt_artifact_get_bool(root, "strip_plan", strip_plan, err) ||
        !trt_artifact_get_bool(root, "refit_identical", refit_identical, err)) {
        if (primary_doc) yyjson_doc_free(primary_doc);
        yyjson_doc_free(doc);
        return false;
    }
    yyjson_val * weight_streaming_val = yyjson_obj_get(root, "weight_streaming");
    if (weight_streaming_val) {
        if (!yyjson_is_bool(weight_streaming_val)) {
            if (primary_doc) yyjson_doc_free(primary_doc);
            yyjson_doc_free(doc);
            trt_artifact_set_error(err, "engine metadata weight_streaming is not a bool");
            return false;
        }
        weight_streaming = yyjson_get_bool(weight_streaming_val);
    }
    check.weight_streaming = weight_streaming;

    if (check.policy != expected_policy) {
        if (primary_doc) yyjson_doc_free(primary_doc);
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "engine precision policy does not match DiT precision manifest");
        return false;
    }
    if (!strong || global_fp16 || global_bf16) {
        if (primary_doc) yyjson_doc_free(primary_doc);
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "engine metadata does not describe a strongly typed, graph-precision build");
        return false;
    }
    // HOT-Step DiT engines embed weights directly in the serialized plan
    // (no kSTRIP_PLAN) and are built with kREFIT_IDENTICAL so the runtime
    // can later refit weights with zero inference penalty. The Python
    // builder (build-trt-engine.py) and the Python validator
    // (acestep_trt_common.validate_engine_metadata) enforce the same
    // contract: strip_plan == false && refit_identical == true.
    if (strip_plan) {
        if (primary_doc) yyjson_doc_free(primary_doc);
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "engine metadata reports a stripped plan; HOT-Step DiT engines must embed weights (strip_plan=false)");
        return false;
    }
    if (!refit_identical) {
        if (primary_doc) yyjson_doc_free(primary_doc);
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "engine metadata is missing the refit-identical guarantee (refit_identical=true required)");
        return false;
    }
    yyjson_val * shapes = yyjson_obj_get(root, "profile_shapes");
    if (!shapes || !yyjson_is_obj(shapes)) {
        if (primary_doc) yyjson_doc_free(primary_doc);
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "engine metadata is missing profile_shapes");
        return false;
    }
    if (!trt_artifact_parse_profile_bounds(shapes, check, err)) {
        if (primary_doc) yyjson_doc_free(primary_doc);
        yyjson_doc_free(doc);
        return false;
    }
    if (!trt_artifact_parse_version(check.tensorrt_version, check.tensorrt_major, check.tensorrt_minor)) {
        if (primary_doc) yyjson_doc_free(primary_doc);
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "engine metadata has an unparseable TensorRT version");
        return false;
    }
    if (expected_trt_major >= 0 && check.tensorrt_major != expected_trt_major) {
        if (primary_doc) yyjson_doc_free(primary_doc);
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "engine TensorRT major version does not match runtime");
        return false;
    }
    if (expected_trt_minor >= 0 && check.tensorrt_minor != expected_trt_minor) {
        if (primary_doc) yyjson_doc_free(primary_doc);
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "engine TensorRT minor version does not match runtime");
        return false;
    }

    yyjson_val * pm = yyjson_obj_get(root, "precision_manifest");
    if (pm && yyjson_is_obj(pm)) {
        yyjson_val * pm_policy = yyjson_obj_get(pm, "precision_policy");
        if (pm_policy && yyjson_is_str(pm_policy)) {
            std::string nested_policy(yyjson_get_str(pm_policy), yyjson_get_len(pm_policy));
            if (nested_policy != expected_policy) {
                if (primary_doc) yyjson_doc_free(primary_doc);
                yyjson_doc_free(doc);
                trt_artifact_set_error(err, "engine precision manifest summary does not match DiT manifest");
                return false;
            }
        }
    }

    if (primary_doc) yyjson_doc_free(primary_doc);
    yyjson_doc_free(doc);
    if (out) {
        *out = check;
    }
    return true;
}

static inline std::string trt_artifact_json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static inline bool trt_artifact_write_native_engine_metadata(const std::string & engine_path,
                                                             const TrtPrecisionManifestCheck & manifest,
                                                             int trt_major,
                                                             int trt_minor,
                                                             int trt_patch,
                                                             bool weight_streaming,
                                                             std::string * err) {
    std::string metadata_path = engine_path + ".metadata.json";
    FILE * f = fopen(metadata_path.c_str(), "wb");
    if (!f) {
        trt_artifact_set_error(err, "cannot write " + metadata_path);
        return false;
    }
    std::string policy = trt_artifact_json_escape(manifest.policy);
    fprintf(f,
            "{\n"
            "  \"engine_builder\": \"hot-step-cpp-native\",\n"
            "  \"profile\": \"native-default\",\n"
            "  \"precision_policy\": \"%s\",\n"
            "  \"precision_manifest\": {\n"
            "    \"precision_policy\": \"%s\",\n"
            "    \"matched_allowlist_count\": %llu,\n"
            "    \"downcast_to_fp16_count\": %llu,\n"
            "    \"quantized_to_int8_count\": %llu,\n"
            "    \"preserved_fp32_count\": %llu,\n"
            "    \"all_parameter_count\": %.0f,\n"
            "    \"matrix_parameter_count\": %.0f,\n"
            "    \"non_matrix_parameter_count\": %.0f\n"
            "  },\n"
            "  \"tensorrt_version\": \"%d.%d.%d\",\n"
            "  \"strongly_typed_network\": true,\n"
            "  \"global_fp16_builder_flag\": false,\n"
            "  \"global_bf16_builder_flag\": false,\n"
            "  \"workspace_gb\": 4.0,\n"
            "  \"strip_plan\": false,\n"
            "  \"refit_identical\": true,\n"
            "  \"weight_streaming\": %s,\n"
            "  \"profile_shapes\": {\n"
            "    \"input_latents\": {\"min\": [1, 64, 192], \"opt\": [1, 2048, 192], \"max\": [%d, %d, 192]},\n"
            "    \"enc_hidden\": {\"min\": [1, 64, 2048], \"opt\": [1, 512, 2048], \"max\": [%d, %d, 2048]},\n"
            "    \"t\": {\"min\": [1], \"opt\": [1], \"max\": [%d]},\n"
            "    \"t_r\": {\"min\": [1], \"opt\": [1], \"max\": [%d]}\n"
            "  }\n"
            "}\n",
            policy.c_str(),
            policy.c_str(),
            (unsigned long long) manifest.matched_allowlist_count,
            (unsigned long long) manifest.downcast_to_fp16_count,
            (unsigned long long) manifest.quantized_to_int8_count,
            (unsigned long long) manifest.preserved_fp32_count,
            manifest.all_parameter_count,
            manifest.matrix_parameter_count,
            manifest.non_matrix_parameter_count,
            trt_major,
            trt_minor,
            trt_patch,
            weight_streaming ? "true" : "false",
            TRT_ARTIFACT_NATIVE_PROFILE_MAX_BATCH,
            TRT_ARTIFACT_NATIVE_PROFILE_MAX_T,
            TRT_ARTIFACT_NATIVE_PROFILE_MAX_BATCH,
            TRT_ARTIFACT_NATIVE_PROFILE_MAX_ENC_S,
            TRT_ARTIFACT_NATIVE_PROFILE_MAX_BATCH,
            TRT_ARTIFACT_NATIVE_PROFILE_MAX_BATCH);
    fclose(f);
    return true;
}
