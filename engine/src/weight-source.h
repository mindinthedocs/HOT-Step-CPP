#pragma once
// weight-source.h: format-agnostic tensor loading from GGUF or safetensors
//
// Unifies tensor data access so model loaders (DiT, Qwen3, etc.) work
// identically with both formats.  Generalises VaeWeightSource from vae.h.
//
// The GGUF path delegates to gguf-weights.h functions.
// The safetensors path uses safetensors.h (mmap + JSON header parse).
//
// Tensor name mapping:
//   GGUF names are the canonical names used by the loaders.
//   When loading from safetensors, a name_prefix (e.g. "model.") can be
//   prepended to match HuggingFace naming (only needed for Text Encoder).
//
// Dimension ordering:
//   GGUF stores shapes in ggml order (ne[0]=innermost, row-major).
//   Safetensors stores shapes in PyTorch order (dim[0]=outermost).
//   ws_shape() returns ggml order regardless of source format.

#include "gguf-weights.h"
#include "safetensors.h"
#include "weight-ctx.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#    define WS_SEP "\\"
#else
#    define WS_SEP "/"
#endif

// Multi-file safetensors container for sharded models.
// Each shard is an independently mmapped STFile. Tensor lookup searches all.
struct STMulti {
    std::vector<STFile> shards;

    // Find a tensor entry across all shards. Returns (shard_index, entry*).
    std::pair<int, const STEntry *> find(const char * name) const {
        for (int i = 0; i < (int) shards.size(); i++) {
            const STEntry * e = st_find(shards[i], name);
            if (e) return { i, e };
        }
        return { -1, nullptr };
    }

    // Get data pointer for a found entry
    const void * data(int shard_idx, const STEntry & e) const {
        return st_data(shards[shard_idx], e);
    }
};

static bool st_path_ends_with(const char * path, const char * suffix) {
    if (!path || !suffix) return false;
    size_t n = strlen(path);
    size_t m = strlen(suffix);
    return n >= m && strcmp(path + n - m, suffix) == 0;
}

static bool st_multi_open_file(STMulti * sm, const std::string & path) {
    STFile sf = {};
    if (!st_open(&sf, path.c_str())) {
        return false;
    }
    sm->shards.push_back(std::move(sf));
    return true;
}

// Open a single or sharded safetensors model from a directory.
// Looks for:
//   1. an exact .safetensors path, when the caller passed a file
//   2. model.safetensors (single file)
//   3. model.safetensors.index.json (sharded: reads weight_map, opens each shard)
//   4. diffusion_pytorch_model.safetensors (diffusers/VAE format)
// Returns true on success.
static bool st_multi_open(STMulti * sm, const char * dir) {
    sm->shards.clear();

    if (st_path_ends_with(dir, ".safetensors")) {
        if (st_multi_open_file(sm, dir)) {
            return true;
        }
        fprintf(stderr, "[WeightSource] No safetensors file found at %s\n", dir);
        return false;
    }

    // Try single file first
    std::string single = std::string(dir) + WS_SEP + "model.safetensors";
    {
        if (st_multi_open_file(sm, single)) {
            return true;
        }
    }

    // Try sharded: read index.json for shard filenames
    std::string index_path = std::string(dir) + WS_SEP + "model.safetensors.index.json";
    {
        FILE * f = fopen(index_path.c_str(), "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            std::string json(sz, '\0');
            fread(&json[0], 1, sz, f);
            fclose(f);

            // Extract unique shard filenames from weight_map values.
            // Minimal parsing: find "weight_map":{...}, collect unique quoted values.
            std::vector<std::string> shard_names;
            size_t wm = json.find("\"weight_map\"");
            if (wm != std::string::npos) {
                size_t brace = json.find('{', wm);
                size_t end = std::string::npos;
                if (brace != std::string::npos) {
                    // Find matching close brace
                    int depth = 1;
                    size_t p = brace + 1;
                    while (p < json.size() && depth > 0) {
                        if (json[p] == '{') depth++;
                        else if (json[p] == '}') depth--;
                        p++;
                    }
                    end = p;

                    // Scan for ": " followed by quoted string values
                    p = brace + 1;
                    while (p < end) {
                        size_t colon = json.find(':', p);
                        if (colon == std::string::npos || colon >= end) break;
                        // Find value string after colon
                        size_t q1 = json.find('"', colon + 1);
                        if (q1 == std::string::npos || q1 >= end) break;
                        size_t q2 = json.find('"', q1 + 1);
                        if (q2 == std::string::npos || q2 >= end) break;
                        std::string val = json.substr(q1 + 1, q2 - q1 - 1);
                        // Add if not already present
                        bool found = false;
                        for (const auto & s : shard_names) {
                            if (s == val) { found = true; break; }
                        }
                        if (!found) {
                            shard_names.push_back(val);
                        }
                        p = q2 + 1;
                    }
                }
            }

            if (!shard_names.empty()) {
                // Sort for deterministic order
                std::sort(shard_names.begin(), shard_names.end());
                for (const auto & name : shard_names) {
                    std::string shard_path = std::string(dir) + WS_SEP + name;
                    STFile sf = {};
                    if (!st_open(&sf, shard_path.c_str())) {
                        fprintf(stderr, "[WeightSource] FATAL: cannot open shard %s\n", shard_path.c_str());
                        // Close already-opened shards
                        for (auto & s : sm->shards) st_close(&s);
                        sm->shards.clear();
                        return false;
                    }
                    sm->shards.push_back(std::move(sf));
                }
                fprintf(stderr, "[WeightSource] Opened %d shards from %s\n",
                        (int) sm->shards.size(), index_path.c_str());
                return true;
            }
        }
    }

    // Try diffusers format (VAE)
    std::string diffusers = std::string(dir) + WS_SEP + "diffusion_pytorch_model.safetensors";
    {
        if (st_multi_open_file(sm, diffusers)) {
            return true;
        }
    }

    fprintf(stderr, "[WeightSource] No safetensors files found in %s\n", dir);
    return false;
}

static bool st_multi_open_preferred(STMulti * sm, const char * dir, const char * preferred_file) {
    sm->shards.clear();
    if (st_path_ends_with(dir, ".safetensors")) {
        return st_multi_open(sm, dir);
    }
    if (preferred_file && preferred_file[0]) {
        std::string preferred = std::string(dir) + WS_SEP + preferred_file;
        if (st_multi_open_file(sm, preferred)) {
            fprintf(stderr, "[WeightSource] Opened sidecar %s\n", preferred.c_str());
            return true;
        }
    }
    return st_multi_open(sm, dir);
}

static void st_multi_close(STMulti * sm) {
    for (auto & s : sm->shards) {
        st_close(&s);
    }
    sm->shards.clear();
}

// ─── WeightSource ────────────────────────────────────────────────────

struct WeightSource {
    bool        is_st = false;
    GGUFModel * gf    = nullptr;
    STMulti *   sm    = nullptr;   // multi-file safetensors (single or sharded)
    std::string name_prefix;       // prepended to lookup names for safetensors (e.g. "model.")

    // Resolve a canonical tensor name to the safetensors lookup name.
    // For most models this is identity; for Text Encoder we prepend "model.".
    std::string resolve_name(const char * name) const {
        if (is_st && !name_prefix.empty()) {
            return name_prefix + name;
        }
        return name;
    }

    // Check if a tensor exists
    bool exists(const char * name) const {
        std::string rn = resolve_name(name);
        if (is_st) {
            auto [idx, e] = sm->find(rn.c_str());
            return e != nullptr;
        } else {
            return ggml_get_tensor(gf->meta, name) != nullptr;
        }
    }

    // Get raw data pointer + type for a named tensor
    const void * data(const char * name, ggml_type & type) const {
        if (is_st) {
            std::string rn = resolve_name(name);
            auto [shard_idx, e] = sm->find(rn.c_str());
            if (!e) return nullptr;
            type = st_ggml_type(*e);
            return sm->data(shard_idx, *e);
        } else {
            struct ggml_tensor * mt = ggml_get_tensor(gf->meta, name);
            if (!mt) return nullptr;
            type = mt->type;
            return gf_get_data(*gf, name);
        }
    }

    // Get shape info (ggml order: ne[0]=innermost)
    bool shape(const char * name, int & n_dims, int64_t ne[4]) const {
        ne[0] = ne[1] = ne[2] = ne[3] = 1;
        if (is_st) {
            std::string rn = resolve_name(name);
            auto [shard_idx, e] = sm->find(rn.c_str());
            if (!e) return false;
            n_dims = e->n_dims;
            // Reverse: safetensors is PyTorch order [dim0..dimN], ggml is [neN..ne0]
            for (int i = 0; i < 4; i++) {
                int src = e->n_dims - 1 - i;
                ne[i] = (src >= 0 && src < e->n_dims) ? e->shape[src] : 1;
            }
            return true;
        } else {
            struct ggml_tensor * mt = ggml_get_tensor(gf->meta, name);
            if (!mt) return false;
            n_dims = ggml_n_dims(mt);
            for (int i = 0; i < 4; i++) ne[i] = mt->ne[i];
            return true;
        }
    }

    // Get nbytes of a tensor (for safetensors: data_end - data_start)
    size_t nbytes(const char * name) const {
        if (is_st) {
            std::string rn = resolve_name(name);
            auto [shard_idx, e] = sm->find(rn.c_str());
            if (!e) return 0;
            return e->data_end - e->data_start;
        } else {
            struct ggml_tensor * mt = ggml_get_tensor(gf->meta, name);
            if (!mt) return 0;
            return ggml_nbytes(mt);
        }
    }

    // Get ggml type of a tensor
    ggml_type type(const char * name) const {
        if (is_st) {
            std::string rn = resolve_name(name);
            auto [shard_idx, e] = sm->find(rn.c_str());
            if (!e) return GGML_TYPE_COUNT;
            return st_ggml_type(*e);
        } else {
            struct ggml_tensor * mt = ggml_get_tensor(gf->meta, name);
            if (!mt) return GGML_TYPE_COUNT;
            return mt->type;
        }
    }

    // Enumerate all tensor names (in canonical/loader form, prefix stripped).
    // For GGUF: direct from gguf_get_tensor_name.
    // For safetensors: iterate all STFile entries, strip name_prefix if set.
    std::vector<std::string> tensor_names() const {
        std::vector<std::string> names;
        if (is_st) {
            for (const auto & shard : sm->shards) {
                for (const auto & e : shard.entries) {
                    std::string n = e.name;
                    // Strip prefix if it matches
                    if (!name_prefix.empty() &&
                        n.compare(0, name_prefix.size(), name_prefix) == 0) {
                        n = n.substr(name_prefix.size());
                    }
                    names.push_back(n);
                }
            }
        } else {
            int64_t n = gguf_get_n_tensors(gf->gguf);
            names.reserve((size_t) n);
            for (int64_t i = 0; i < n; i++) {
                names.push_back(gguf_get_tensor_name(gf->gguf, i));
            }
        }
        return names;
    }
};

// ─── ws_load_* functions ─────────────────────────────────────────────
// Drop-in replacements for gf_load_tensor, gf_load_tensor_f32, etc.
// Work with both GGUF (via GGUFModel) and safetensors (via STMulti).

// Load tensor preserving original dtype. Returns ggml_tensor* (not yet allocated).
static struct ggml_tensor * ws_load_tensor(WeightCtx *         wctx,
                                           const WeightSource & ws,
                                           const std::string &  name) {
    // GGUF fast path: delegate to existing function
    if (!ws.is_st) {
        return gf_load_tensor(wctx, *ws.gf, name);
    }

    // Safetensors path
    std::string rn = ws.resolve_name(name.c_str());
    auto [shard_idx, e] = ws.sm->find(rn.c_str());
    if (!e) {
        fprintf(stderr, "[WeightSource] FATAL: tensor '%s' (as '%s') not found in safetensors\n",
                name.c_str(), rn.c_str());
        exit(1);
    }

    ggml_type type = st_ggml_type(*e);
    if (type == GGML_TYPE_COUNT) {
        fprintf(stderr, "[WeightSource] FATAL: unsupported dtype '%s' for tensor '%s'\n",
                e->dtype.c_str(), name.c_str());
        exit(1);
    }

    // Shape in ggml order (reversed from safetensors)
    int     n_dims = e->n_dims;
    int64_t ne[4] = { 1, 1, 1, 1 };
    for (int i = 0; i < 4; i++) {
        int src = e->n_dims - 1 - i;
        ne[i] = (src >= 0 && src < e->n_dims) ? e->shape[src] : 1;
    }

    struct ggml_tensor * tensor = ggml_new_tensor(wctx->ctx, type, n_dims, ne);
    ggml_set_name(tensor, name.c_str());

    const void * data = ws.sm->data(shard_idx, *e);
    size_t nbytes = e->data_end - e->data_start;

    wctx->pending.push_back({ tensor, data, nbytes, 0 });
    return tensor;
}

// Try to load, returns nullptr if not found (no exit)
static struct ggml_tensor * ws_try_load_tensor(WeightCtx *          wctx,
                                               const WeightSource & ws,
                                               const std::string &  name) {
    if (!ws.is_st) {
        return gf_try_load_tensor(wctx, *ws.gf, name);
    }
    std::string rn = ws.resolve_name(name.c_str());
    auto [shard_idx, e] = ws.sm->find(rn.c_str());
    if (!e) return nullptr;
    return ws_load_tensor(wctx, ws, name);
}

// Load tensor, converting to F32 at load time.
// Best for small tensors: norms, biases, scale_shift_tables.
static struct ggml_tensor * ws_load_tensor_f32(WeightCtx *          wctx,
                                               const WeightSource & ws,
                                               const std::string &  name) {
    // GGUF fast path
    if (!ws.is_st) {
        return gf_load_tensor_f32(wctx, *ws.gf, name);
    }

    // Safetensors path
    std::string rn = ws.resolve_name(name.c_str());
    auto [shard_idx, e] = ws.sm->find(rn.c_str());
    if (!e) {
        fprintf(stderr, "[WeightSource] FATAL: tensor '%s' (as '%s') not found in safetensors\n",
                name.c_str(), rn.c_str());
        exit(1);
    }

    ggml_type src_type = st_ggml_type(*e);

    // Shape in ggml order
    int     n_dims = e->n_dims;
    int64_t ne[4] = { 1, 1, 1, 1 };
    for (int i = 0; i < 4; i++) {
        int src = e->n_dims - 1 - i;
        ne[i] = (src >= 0 && src < e->n_dims) ? e->shape[src] : 1;
    }

    // If already F32, load directly
    if (src_type == GGML_TYPE_F32) {
        return ws_load_tensor(wctx, ws, name);
    }

    // Convert BF16/F16 → F32
    struct ggml_tensor * tensor = ggml_new_tensor(wctx->ctx, GGML_TYPE_F32, n_dims, ne);
    ggml_set_name(tensor, name.c_str());

    size_t n = 1;
    for (int i = 0; i < n_dims; i++) n *= (size_t) ne[i];

    auto    buf  = std::make_unique<float[]>(n);
    float * data = buf.get();

    const void * raw = ws.sm->data(shard_idx, *e);

    if (src_type == GGML_TYPE_BF16) {
        const uint16_t * p = (const uint16_t *) raw;
        for (size_t i = 0; i < n; i++) {
            data[i] = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &p[i]);
        }
    } else if (src_type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw, data, (int) n);
    } else {
        fprintf(stderr, "[WeightSource] WARNING: unsupported type for F32 convert '%s', loading as-is\n",
                name.c_str());
        return ws_load_tensor(wctx, ws, name);
    }

    wctx->pending.push_back({ tensor, data, n * sizeof(float), 0 });
    wctx->staging.push_back(std::move(buf));
    return tensor;
}

// Fuse Q, K, V projection weights into [ne0, q_ne1 + k_ne1 + v_ne1].
// For safetensors (all BF16): types always match, so fusion always succeeds.
// Returns NULL if types differ (caller falls back to separate loads).
static struct ggml_tensor * ws_load_qkv_fused(WeightCtx *          wctx,
                                              const WeightSource & ws,
                                              const std::string &  q_name,
                                              const std::string &  k_name,
                                              const std::string &  v_name) {
    // GGUF fast path
    if (!ws.is_st) {
        return gf_load_qkv_fused(wctx, *ws.gf, q_name, k_name, v_name);
    }

    // Safetensors path
    std::string q_rn = ws.resolve_name(q_name.c_str());
    std::string k_rn = ws.resolve_name(k_name.c_str());
    std::string v_rn = ws.resolve_name(v_name.c_str());

    auto [q_si, q_e] = ws.sm->find(q_rn.c_str());
    auto [k_si, k_e] = ws.sm->find(k_rn.c_str());
    auto [v_si, v_e] = ws.sm->find(v_rn.c_str());

    if (!q_e || !k_e || !v_e) {
        fprintf(stderr, "[WeightSource] FATAL: QKV tensor not found: %s / %s / %s\n",
                q_name.c_str(), k_name.c_str(), v_name.c_str());
        exit(1);
    }

    ggml_type q_type = st_ggml_type(*q_e);
    ggml_type k_type = st_ggml_type(*k_e);
    ggml_type v_type = st_ggml_type(*v_e);

    if (q_type != k_type || k_type != v_type) {
        return NULL;  // caller falls back to separate loads
    }

    // Shapes in ggml order: [ne0, ne1] where ne0 = inner dim (last in PyTorch)
    // PyTorch Q: [q_ne1, ne0] → ggml [ne0, q_ne1]
    int64_t ne0 = (q_e->n_dims >= 1) ? q_e->shape[q_e->n_dims - 1] : 1;
    int64_t q_ne1 = (q_e->n_dims >= 2) ? q_e->shape[0] : 1;
    int64_t k_ne1 = (k_e->n_dims >= 2) ? k_e->shape[0] : 1;
    int64_t v_ne1 = (v_e->n_dims >= 2) ? v_e->shape[0] : 1;

    int64_t fused_ne[2] = { ne0, q_ne1 + k_ne1 + v_ne1 };
    struct ggml_tensor * fused = ggml_new_tensor(wctx->ctx, q_type, 2, fused_ne);

    size_t row_size = ggml_row_size(q_type, ne0);
    size_t q_bytes = q_ne1 * row_size;
    size_t k_bytes = k_ne1 * row_size;
    size_t v_bytes = v_ne1 * row_size;

    wctx->pending.push_back({ fused, ws.sm->data(q_si, *q_e), q_bytes, 0 });
    wctx->pending.push_back({ fused, ws.sm->data(k_si, *k_e), k_bytes, q_bytes });
    wctx->pending.push_back({ fused, ws.sm->data(v_si, *v_e), v_bytes, q_bytes + k_bytes });
    return fused;
}

// Fuse two projection weights [ne0, a_ne1 + b_ne1] when types match.
// Returns NULL if types differ.
static struct ggml_tensor * ws_load_pair_fused(WeightCtx *          wctx,
                                               const WeightSource & ws,
                                               const std::string &  a_name,
                                               const std::string &  b_name) {
    // GGUF fast path
    if (!ws.is_st) {
        return gf_load_pair_fused(wctx, *ws.gf, a_name, b_name);
    }

    // Safetensors path
    std::string a_rn = ws.resolve_name(a_name.c_str());
    std::string b_rn = ws.resolve_name(b_name.c_str());

    auto [a_si, a_e] = ws.sm->find(a_rn.c_str());
    auto [b_si, b_e] = ws.sm->find(b_rn.c_str());

    if (!a_e || !b_e) return NULL;

    ggml_type a_type = st_ggml_type(*a_e);
    ggml_type b_type = st_ggml_type(*b_e);
    if (a_type != b_type) return NULL;

    // ne0 = inner dim (last in PyTorch order)
    int64_t ne0   = (a_e->n_dims >= 1) ? a_e->shape[a_e->n_dims - 1] : 1;
    int64_t a_ne1 = (a_e->n_dims >= 2) ? a_e->shape[0] : 1;
    int64_t b_ne1 = (b_e->n_dims >= 2) ? b_e->shape[0] : 1;

    // Verify ne0 matches
    int64_t b_ne0 = (b_e->n_dims >= 1) ? b_e->shape[b_e->n_dims - 1] : 1;
    if (ne0 != b_ne0) return NULL;

    int64_t fused_ne[2] = { ne0, a_ne1 + b_ne1 };
    struct ggml_tensor * fused = ggml_new_tensor(wctx->ctx, a_type, 2, fused_ne);

    size_t row_size = ggml_row_size(a_type, ne0);
    size_t a_bytes = a_ne1 * row_size;
    size_t b_bytes = b_ne1 * row_size;

    wctx->pending.push_back({ fused, ws.sm->data(a_si, *a_e), a_bytes, 0 });
    wctx->pending.push_back({ fused, ws.sm->data(b_si, *b_e), b_bytes, a_bytes });
    return fused;
}
