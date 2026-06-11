// trt-bundle-manifest.h: native TRT bundle detection + manifest.json reader.
//
// A native TRT bundle is a directory under trt-bundles/<name>/ containing a
// manifest.json written during bundle export. The manifest is the SOLE
// TRT-bundle detector (partner §3.5): the presence of manifest.json in the
// selected bundle dir is what marks the directory as a TRT runtime bundle. This
// retires the legacy onnx_dit_has_synth_bundle() file-list probe.
//
// Schema (partner §4.4):
//   {
//     "version": <str>,
//     "source_model": <str>,
//     "variant": <str>,
//     "components": {
//       "dit":      {"engine": <rel>, "metadata": <rel>, "precision": <str>, "weight_streaming": <bool>},
//       "text_enc": {"engine": <rel>, "metadata": <rel>, "precision": <str>, "sidecars": [<rel>...]},
//       "cond_enc": {"engine": <rel>, "metadata": <rel>, "precision": <str>, "sidecars": [<rel>...]},
//       "fsq":      {"sidecar": <rel>, "metadata": <rel>}
//     }
//   }
//
// all-three-or-nothing (partner §5.7): a manifest that is present but missing
// dit | text_enc | cond_enc (or whose declared engine file is absent) is a HARD
// FAILURE. The runtime never silently falls back to GGUF encoders when a TRT
// manifest is present but incomplete.
#pragma once

#include "trt-artifact-manifest.h"  // yyjson read/get helpers, dirname/join

#include <string>
#include <vector>

struct TrtBundleComponent {
    std::string engine;            // resolved absolute/relative engine path (dit/text_enc/cond_enc)
    std::string metadata;          // resolved metadata path
    std::string precision;         // precision policy string (informational)
    bool        weight_streaming = false;  // dit only; false elsewhere
    std::vector<std::string> sidecars;     // text_enc/cond_enc sidecar paths
};

struct TrtBundleManifest {
    std::string manifest_path;     // <bundle>/manifest.json
    std::string bundle_dir;        // <...>/trt-bundles/<name>
    std::string version;
    std::string source_model;
    std::string variant;
    TrtBundleComponent dit;
    TrtBundleComponent text_enc;
    TrtBundleComponent cond_enc;
    std::string fsq_sidecar;       // <bundle>/fsq.safetensors (resolved)
    std::string fsq_metadata;      // <bundle>/fsq.metadata.json (resolved)
};

static inline std::string trt_bundle_manifest_path(const std::string & bundle_dir) {
    if (bundle_dir.empty()) {
        return std::string();
    }
    char back = bundle_dir.back();
    if (back == '/' || back == '\\') {
        return bundle_dir + "manifest.json";
    }
    return bundle_dir + "/manifest.json";
}

// Detection: a directory is a TRT bundle iff it carries a manifest.json. This is
// the sole TRT-bundle detector — do not reintroduce a file-list probe.
static inline bool trt_bundle_has_manifest(const std::string & bundle_dir) {
    std::string p = trt_bundle_manifest_path(bundle_dir);
    if (p.empty()) {
        return false;
    }
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

// Parse one component object (engine + metadata + precision [+ ws] [+ sidecars]).
// require_engine_present enforces that the declared engine file exists on disk —
// the all-three-or-nothing gate fails closed when it does not.
static inline bool trt_bundle_parse_component(yyjson_val *          components,
                                              const char *          key,
                                              const std::string &   manifest_path,
                                              bool                  want_weight_streaming,
                                              bool                  want_sidecars,
                                              TrtBundleComponent &  out,
                                              std::string *         err) {
    yyjson_val * comp = yyjson_obj_get(components, key);
    if (!comp || !yyjson_is_obj(comp)) {
        trt_artifact_set_error(err, std::string("manifest is missing component object: components.") + key);
        return false;
    }

    std::string engine_rel;
    std::string metadata_rel;
    if (!trt_artifact_get_string(comp, "engine", engine_rel, err)) {
        trt_artifact_set_error(err, std::string("manifest component ") + key + " is missing string field: engine");
        return false;
    }
    // metadata + precision are informational; tolerate absence so the gate keys
    // strictly on the engine artifact existing.
    yyjson_val * md = yyjson_obj_get(comp, "metadata");
    if (md && yyjson_is_str(md)) {
        metadata_rel.assign(yyjson_get_str(md), yyjson_get_len(md));
    }
    yyjson_val * prec = yyjson_obj_get(comp, "precision");
    if (prec && yyjson_is_str(prec)) {
        out.precision.assign(yyjson_get_str(prec), yyjson_get_len(prec));
    }

    out.engine   = trt_artifact_join_relative(manifest_path, engine_rel);
    out.metadata = metadata_rel.empty() ? std::string()
                                        : trt_artifact_join_relative(manifest_path, metadata_rel);

    if (want_weight_streaming) {
        yyjson_val * ws = yyjson_obj_get(comp, "weight_streaming");
        if (ws && yyjson_is_bool(ws)) {
            out.weight_streaming = yyjson_get_bool(ws);
        }
    }

    if (want_sidecars) {
        yyjson_val * arr = yyjson_obj_get(comp, "sidecars");
        if (arr && yyjson_is_arr(arr)) {
            size_t idx, max;
            yyjson_val * item;
            yyjson_arr_foreach(arr, idx, max, item) {
                if (yyjson_is_str(item)) {
                    std::string rel(yyjson_get_str(item), yyjson_get_len(item));
                    out.sidecars.push_back(trt_artifact_join_relative(manifest_path, rel));
                }
            }
        }
    }

    // all-three-or-nothing: the declared engine file must exist on disk.
    FILE * ef = fopen(out.engine.c_str(), "rb");
    if (!ef) {
        trt_artifact_set_error(err, std::string("manifest component ") + key +
                                        " declares an engine that is missing on disk: " + out.engine);
        return false;
    }
    fclose(ef);
    return true;
}

// Load + validate a bundle manifest. On success out carries every resolved
// component path. Fails closed (returns false, sets err) when manifest.json is
// absent, malformed, or missing/incomplete in any of dit | text_enc | cond_enc.
static inline bool trt_bundle_load_manifest(const std::string &   bundle_dir,
                                            TrtBundleManifest *   out,
                                            std::string *         err) {
    std::string manifest_path = trt_bundle_manifest_path(bundle_dir);
    if (manifest_path.empty()) {
        trt_artifact_set_error(err, "empty bundle dir");
        return false;
    }

    yyjson_doc * doc = trt_artifact_read_json(manifest_path, err);
    if (!doc) {
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);

    TrtBundleManifest m;
    m.manifest_path = manifest_path;
    m.bundle_dir    = bundle_dir;

    // version/source_model/variant are informational; tolerate absence.
    yyjson_val * v = yyjson_obj_get(root, "version");
    if (v && yyjson_is_str(v)) m.version.assign(yyjson_get_str(v), yyjson_get_len(v));
    yyjson_val * sm = yyjson_obj_get(root, "source_model");
    if (sm && yyjson_is_str(sm)) m.source_model.assign(yyjson_get_str(sm), yyjson_get_len(sm));
    yyjson_val * va = yyjson_obj_get(root, "variant");
    if (va && yyjson_is_str(va)) m.variant.assign(yyjson_get_str(va), yyjson_get_len(va));

    yyjson_val * components = yyjson_obj_get(root, "components");
    if (!components || !yyjson_is_obj(components)) {
        yyjson_doc_free(doc);
        trt_artifact_set_error(err, "manifest is missing the components object");
        return false;
    }

    if (!trt_bundle_parse_component(components, "dit", manifest_path,
                                    /*weight_streaming*/ true, /*sidecars*/ false, m.dit, err) ||
        !trt_bundle_parse_component(components, "text_enc", manifest_path,
                                    /*weight_streaming*/ false, /*sidecars*/ true, m.text_enc, err) ||
        !trt_bundle_parse_component(components, "cond_enc", manifest_path,
                                    /*weight_streaming*/ false, /*sidecars*/ true, m.cond_enc, err)) {
        yyjson_doc_free(doc);
        return false;
    }

    // FSQ is delivered through the existing fsq.safetensors sidecar mechanism;
    // it is not part of the all-three gate. Parse it when present.
    yyjson_val * fsq = yyjson_obj_get(components, "fsq");
    if (fsq && yyjson_is_obj(fsq)) {
        yyjson_val * sc = yyjson_obj_get(fsq, "sidecar");
        if (sc && yyjson_is_str(sc)) {
            std::string rel(yyjson_get_str(sc), yyjson_get_len(sc));
            m.fsq_sidecar = trt_artifact_join_relative(manifest_path, rel);
        }
        yyjson_val * md = yyjson_obj_get(fsq, "metadata");
        if (md && yyjson_is_str(md)) {
            std::string rel(yyjson_get_str(md), yyjson_get_len(md));
            m.fsq_metadata = trt_artifact_join_relative(manifest_path, rel);
        }
    }

    yyjson_doc_free(doc);
    if (out) {
        *out = m;
    }
    return true;
}
