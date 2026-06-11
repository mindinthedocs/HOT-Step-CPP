// model-store.cpp: centralised ownership of GGML modules.
//
// A single hashmap keyed by ModelKey holds every GPU module the pipelines
// touch. Each entry carries the type-erased pointer, a refcount, and a
// deleter that knows how to free the underlying struct. On require, the
// store either hits the cache (refcount++) or evicts peers (STRICT) and
// loads the module. On release, the refcount drops; in STRICT with
// refcount == 0 the module is unloaded on the spot.
//
// CPU-resident modules (BPE, silence_latent, FSM, DiT metadata) live in a
// separate map with the same keying scheme minus the eviction logic.

#include "model-store.h"

#include "config-json.h"
#include "gguf-weights.h"
#include "silence-latent.h"
#include "timer.h"
#include "weight-source.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <vector>

namespace {

// Key hashing. Only the fields relevant for this ModelKind participate, so
// pipeline authors cannot accidentally drift a key by leaving a field that
// their kind does not care about at a different default than their peer.
// LM: kind + path + max_seq + n_kv_sets. DiT: kind + path + adapter_path
// + adapter_scale. Everything else: kind + path.
struct ModelKeyHash {
    size_t operator()(const ModelKey & k) const noexcept {
        size_t h = std::hash<int>{}(static_cast<int>(k.kind));
        h ^= std::hash<std::string>{}(k.path) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        if (k.kind == MODEL_LM) {
            h ^= std::hash<int>{}(k.max_seq) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(k.n_kv_sets) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        } else if (k.kind == MODEL_DIT) {
            h ^= std::hash<std::string>{}(k.adapter_path) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            // adapter_scale: hash the raw bit pattern so 1.0f and 1.00001f are distinct.
            uint32_t bits;
            memcpy(&bits, &k.adapter_scale, sizeof(bits));
            h ^= std::hash<uint32_t>{}(bits) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            // group scales: baked into merged weights, must invalidate on change.
            auto hash_f = [&](float v) {
                uint32_t b; memcpy(&b, &v, sizeof(b));
                h ^= std::hash<uint32_t>{}(b) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            };
            hash_f(k.adapter_group_scales.self_attn);
            hash_f(k.adapter_group_scales.cross_attn);
            hash_f(k.adapter_group_scales.mlp);
            hash_f(k.adapter_group_scales.cond_embed);
            hash_f(k.adapter_group_scales.time_embed);
            hash_f(k.adapter_group_scales.proj_in);
        } else if (k.kind == MODEL_DIT_TRT) {
            // Fold the engine/sidecar fingerprint into the key so an in-place
            // engine or sidecar change reloads. Adapter fields are deliberately
            // excluded: adapters refit in place on the resident engine.
            h ^= std::hash<std::string>{}(k.artifact_fingerprint) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct ModelKeyEq {
    bool operator()(const ModelKey & a, const ModelKey & b) const noexcept {
        if (a.kind != b.kind || a.path != b.path) {
            return false;
        }
        if (a.kind == MODEL_LM) {
            return a.max_seq == b.max_seq && a.n_kv_sets == b.n_kv_sets;
        }
        if (a.kind == MODEL_DIT) {
            return a.adapter_path == b.adapter_path
                && a.adapter_scale == b.adapter_scale
                && a.adapter_group_scales.self_attn  == b.adapter_group_scales.self_attn
                && a.adapter_group_scales.cross_attn == b.adapter_group_scales.cross_attn
                && a.adapter_group_scales.mlp        == b.adapter_group_scales.mlp
                && a.adapter_group_scales.cond_embed == b.adapter_group_scales.cond_embed
                && a.adapter_group_scales.time_embed == b.adapter_group_scales.time_embed
                && a.adapter_group_scales.proj_in    == b.adapter_group_scales.proj_in;
        }
        if (a.kind == MODEL_DIT_TRT) {
            return a.artifact_fingerprint == b.artifact_fingerprint;
        }
        return true;
    }
};

// A loaded GPU module. The store owns ptr and calls deleter when unloading.
// bytes is the resident weight buffer size at load time, used for logging
// and future budget-aware scheduling. label is a short human-readable name
// for the same purpose.
struct GpuEntry {
    void * ptr;
    size_t bytes;
    int    refcount;
    void (*deleter)(void *);
    const char * label;
};

// Reverse lookup: handle pointer -> key, so store_release can find the
// entry from just the pointer without the caller carrying the key around.
using HandleMap = std::unordered_map<void *, ModelKey>;

// CPU-resident entries. No eviction, no refcount. Keyed by path only for
// BPE/silence/FSM/DiTMeta, since those have no other variation.
struct CpuEntry {
    void * ptr;
    void (*deleter)(void *);
};

static bool read_f32_vector_sidecar(const std::string & path, std::vector<float> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    uint32_t dim = 0;
    if (fread(&dim, sizeof(dim), 1, f) != 1 || dim == 0 || dim > 65536) {
        fclose(f);
        return false;
    }
    out.resize(dim);
    size_t n = fread(out.data(), sizeof(float), dim, f);
    fclose(f);
    if (n != dim) {
        out.clear();
        return false;
    }
    return true;
}

static uint64_t store_fnv1a_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return 0;
    }
    uint64_t h = 1469598103934665603ULL;
    unsigned char buf[8192];
    for (;;) {
        size_t n = fread(buf, 1, sizeof(buf), f);
        for (size_t i = 0; i < n; i++) {
            h ^= (uint64_t) buf[i];
            h *= 1099511628211ULL;
        }
        if (n < sizeof(buf)) {
            break;
        }
    }
    fclose(f);
    return h;
}

static std::string store_hex64(uint64_t v) {
    static const char * hex = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[i] = hex[v & 0xf];
        v >>= 4;
    }
    return out;
}

static bool store_file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

static std::string store_dirname(const std::string & path) {
    std::string p = path;
    while (!p.empty() && (p.back() == '/' || p.back() == '\\')) {
        p.pop_back();
    }
    auto slash = p.find_last_of("/\\");
    return (slash != std::string::npos) ? p.substr(0, slash) : ".";
}

// Derive the TRT bundle directory that owns a component path. A bundle lives at
// <...>/trt-bundles/<name>/, and every component key (dit.engine,
// text_encoder.engine, cond_encoder.engine, manifest.json) shares that prefix.
// Returns <...>/trt-bundles/<name> with native separators, or "" when path is
// not under any trt-bundles/ directory (a GGML/GGUF module the prune ignores).
// Matching is separator-agnostic so mixed '/' and '\\' paths resolve the same.
static std::string store_bundle_dir(const std::string & path) {
    static const char * marker = "trt-bundles";
    const size_t        mlen   = 11;  // strlen("trt-bundles")
    for (size_t i = 0; (i = path.find(marker, i)) != std::string::npos; i += mlen) {
        // marker must be a path segment: preceded by a separator (or start) and
        // followed by a separator that begins the <name> segment.
        bool left_ok = (i == 0) || path[i - 1] == '/' || path[i - 1] == '\\';
        size_t after = i + mlen;
        if (!left_ok || after >= path.size()) {
            continue;
        }
        if (path[after] != '/' && path[after] != '\\') {
            continue;  // e.g. "trt-bundles-old", not the marker segment
        }
        // Skip the separator(s) after the marker, then take the <name> segment.
        size_t name_start = after;
        while (name_start < path.size() && (path[name_start] == '/' || path[name_start] == '\\')) {
            name_start++;
        }
        if (name_start >= path.size()) {
            return "";  // trailing "trt-bundles/" with no bundle name
        }
        size_t name_end = name_start;
        while (name_end < path.size() && path[name_end] != '/' && path[name_end] != '\\') {
            name_end++;
        }
        // Bundle dir = everything up to and including <name>.
        return path.substr(0, name_end);
    }
    return "";
}

// Two bundle dirs name the same bundle if their derived dirs match after
// trailing-separator strip and separator normalization. keep_bundle_dir may be
// passed as the bundle directory itself or as any component path under it, so
// normalize both sides through store_bundle_dir first (a component path) then
// fall back to a direct dir compare.
static std::string store_normalize_bundle_dir(const std::string & dir_or_path) {
    std::string b = store_bundle_dir(dir_or_path);
    if (b.empty()) {
        // Caller passed the bundle dir itself (…/trt-bundles/<name>) with no
        // component suffix; store_bundle_dir needs a separator after <name>, so
        // re-derive by appending a synthetic component.
        std::string probe = dir_or_path;
        while (!probe.empty() && (probe.back() == '/' || probe.back() == '\\')) {
            probe.pop_back();
        }
        b = store_bundle_dir(probe + "/_");
    }
    while (!b.empty() && (b.back() == '/' || b.back() == '\\')) {
        b.pop_back();
    }
    for (auto & c : b) {
        if (c == '\\') c = '/';
    }
    return b;
}

static std::string store_bpe_cache_key(const char * lm_path) {
    std::string key = lm_path ? lm_path : "";
    if (!lm_path) {
        return key;
    }
    size_t len = strlen(lm_path);
    bool is_gguf = (len >= 5 && strcmp(lm_path + len - 5, ".gguf") == 0);
    if (is_gguf) {
        return key;
    }
    std::string direct_vocab = key + WS_SEP + "vocab.json";
    std::string direct_merges = key + WS_SEP + "merges.txt";
    std::string sidecar_dir = key;
    if (!store_file_exists(direct_vocab) || !store_file_exists(direct_merges)) {
        sidecar_dir = store_dirname(key);
    }
    std::string vocab = sidecar_dir + WS_SEP + "vocab.json";
    std::string merges = sidecar_dir + WS_SEP + "merges.txt";
    return key + "|vocab=" + store_hex64(store_fnv1a_file(vocab))
        + "|merges=" + store_hex64(store_fnv1a_file(merges));
}

}  // namespace

struct ModelStore {
    EvictPolicy policy;

    std::unordered_map<ModelKey, GpuEntry, ModelKeyHash, ModelKeyEq> gpu;
    HandleMap                                                        handle_to_key;

    // CPU resident tables. Keyed by source path (LM GGUF for BPE/FSM, DiT
    // GGUF for silence/DiTMeta). Small total footprint, never evicted.
    std::unordered_map<std::string, CpuEntry> bpe_by_path;
    std::unordered_map<std::string, CpuEntry> silence_by_path;
    std::unordered_map<std::string, CpuEntry> fsm_by_path;
    std::unordered_map<std::string, CpuEntry> dit_meta_by_path;

    mutable std::mutex mtx;
};

// Caller holds s->mtx. Evicts every GPU entry whose key does not match the
// one we are about to load. Aborts if any conflicting module still has
// refcount > 0: that would mean two mutually exclusive modules are live at
// once, which violates the contract in STRICT mode.
static void evict_all_except(ModelStore * s, const ModelKey & keep) {
    for (auto it = s->gpu.begin(); it != s->gpu.end();) {
        ModelKeyEq eq;
        if (eq(it->first, keep)) {
            ++it;
            continue;
        }
        // Under EVICT_STRICT, evict everything that is not the module we want.
        GpuEntry & e = it->second;
        if (e.refcount > 0) {
            fprintf(stderr, "[Store] FATAL: evicting %s (refcount=%d) to make room in STRICT mode\n", e.label,
                    e.refcount);
            abort();
        }
        fprintf(stderr, "[Store] Evict %s (%.1f MB)\n", e.label, (float) e.bytes / (1024.0f * 1024.0f));
        s->handle_to_key.erase(e.ptr);
        e.deleter(e.ptr);
        it = s->gpu.erase(it);
    }
}

namespace {

template <typename T>
static T * install_entry(ModelStore *     s,
                         const ModelKey & k,
                         T *              obj,
                         size_t           bytes,
                         const char *     label,
                         void (*deleter)(void *)) {
    GpuEntry e;
    e.ptr      = obj;
    e.bytes    = bytes;
    e.refcount = 1;
    e.deleter  = deleter;
    e.label    = label;
    s->gpu.emplace(k, e);
    s->handle_to_key.emplace(obj, k);
    return obj;
}

template <typename T> static T * cache_hit(ModelStore * s, const ModelKey & k) {
    auto it = s->gpu.find(k);
    if (it == s->gpu.end()) {
        return nullptr;
    }
    it->second.refcount++;
    fprintf(stderr, "[Store] Hit %s (refcount=%d)\n", it->second.label, it->second.refcount);
    return static_cast<T *>(it->second.ptr);
}

}  // namespace

ModelStore * store_create(EvictPolicy policy) {
    auto * s  = new ModelStore();
    s->policy = policy;
    fprintf(stderr, "[Store] Created (policy=%s)\n", policy == EVICT_STRICT ? "STRICT" : "NEVER");
    return s;
}

void store_set_policy(ModelStore * s, EvictPolicy policy) {
    if (!s) return;
    std::lock_guard<std::mutex> lock(s->mtx);
    if (s->policy == policy) return;
    fprintf(stderr, "[Store] Policy changed: %s -> %s\n",
            s->policy == EVICT_STRICT ? "STRICT" : "NEVER",
            policy    == EVICT_STRICT ? "STRICT" : "NEVER");
    s->policy = policy;
}

EvictPolicy store_get_policy(ModelStore * s) {
    if (!s) return EVICT_STRICT;
    return s->policy;
}

void store_free(ModelStore * s) {
    if (!s) {
        return;
    }
    // GPU modules: release every entry regardless of refcount (shutdown).
    for (auto & kv : s->gpu) {
        GpuEntry & e = kv.second;
        e.deleter(e.ptr);
    }
    s->gpu.clear();
    s->handle_to_key.clear();

    // CPU modules.
    for (auto & kv : s->bpe_by_path) {
        kv.second.deleter(kv.second.ptr);
    }
    for (auto & kv : s->silence_by_path) {
        kv.second.deleter(kv.second.ptr);
    }
    for (auto & kv : s->fsm_by_path) {
        kv.second.deleter(kv.second.ptr);
    }
    for (auto & kv : s->dit_meta_by_path) {
        kv.second.deleter(kv.second.ptr);
    }
    delete s;
}

// Each require_* follows the same shape: lock, check cache, evict if needed,
// load, install entry. The deleter is a plain C function that matches the
// module's free signature, avoiding template plumbing.
static void del_lm(void * p) {
    qw3lm_free(static_cast<Qwen3LM *>(p));
    delete static_cast<Qwen3LM *>(p);
}

static void del_text_enc(void * p) {
    qwen3_free(static_cast<Qwen3GGML *>(p));
    delete static_cast<Qwen3GGML *>(p);
}

static void del_cond_enc(void * p) {
    cond_ggml_free(static_cast<CondGGML *>(p));
    delete static_cast<CondGGML *>(p);
}

static void del_dit(void * p) {
    dit_ggml_free(static_cast<DiTGGML *>(p));
    delete static_cast<DiTGGML *>(p);
}

static void del_vae_enc(void * p) {
    vae_enc_free(static_cast<VAEEncoder *>(p));
    delete static_cast<VAEEncoder *>(p);
}

static void del_vae_dec(void * p) {
    vae_ggml_free(static_cast<VAEGGML *>(p));
    delete static_cast<VAEGGML *>(p);
}

static void del_fsq_tok(void * p) {
    tok_ggml_free(static_cast<TokGGML *>(p));
    delete static_cast<TokGGML *>(p);
}

static void del_fsq_detok(void * p) {
    detok_ggml_free(static_cast<DetokGGML *>(p));
    delete static_cast<DetokGGML *>(p);
}

#ifdef HOT_STEP_TRT
static void del_text_enc_trt(void * p) {
    text_enc_trt_free(static_cast<TextEncTrt *>(p));
    delete static_cast<TextEncTrt *>(p);
}

static void del_cond_enc_trt(void * p) {
    cond_enc_trt_free(static_cast<CondEncTrt *>(p));
    delete static_cast<CondEncTrt *>(p);
}

static void del_dit_trt(void * p) {
    dit_trt_free(static_cast<DitTrt *>(p));
    delete static_cast<DitTrt *>(p);
}
#endif

// Weight buffer size helpers: different modules use different field names
// for their backend buffer. VAE and VAE-Enc expose m->buf directly, every
// other module uses a WeightCtx at m->wctx.buffer. We spell that out per
// module rather than templating, keeps grep-ability and avoids SFINAE.
static size_t bytes_of_lm(const Qwen3LM * m) {
    return m && m->wctx.buffer ? ggml_backend_buffer_get_size(m->wctx.buffer) : 0;
}

static size_t bytes_of_text_enc(const Qwen3GGML * m) {
    return m && m->wctx.buffer ? ggml_backend_buffer_get_size(m->wctx.buffer) : 0;
}

static size_t bytes_of_cond_enc(const CondGGML * m) {
    return m && m->wctx.buffer ? ggml_backend_buffer_get_size(m->wctx.buffer) : 0;
}

static size_t bytes_of_dit(const DiTGGML * m) {
    return m && m->wctx.buffer ? ggml_backend_buffer_get_size(m->wctx.buffer) : 0;
}

static size_t bytes_of_vae_enc(const VAEEncoder * m) {
    return m && m->buf ? ggml_backend_buffer_get_size(m->buf) : 0;
}

static size_t bytes_of_vae_dec(const VAEGGML * m) {
    return m && m->buf ? ggml_backend_buffer_get_size(m->buf) : 0;
}

static size_t bytes_of_fsq_tok(const TokGGML * m) {
    return m && m->wctx.buffer ? ggml_backend_buffer_get_size(m->wctx.buffer) : 0;
}

static size_t bytes_of_fsq_detok(const DetokGGML * m) {
    return m && m->wctx.buffer ? ggml_backend_buffer_get_size(m->wctx.buffer) : 0;
}

Qwen3LM * store_require_lm(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<Qwen3LM>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    }
    Timer     t;
    Qwen3LM * m = new Qwen3LM();
    if (!qw3lm_load(m, k.path.c_str(), k.max_seq, k.n_kv_sets)) {
        delete m;
        return nullptr;
    }
    install_entry(s, k, m, bytes_of_lm(m), "LM", del_lm);
    fprintf(stderr, "[Store] Load LM: %.0f ms\n", t.ms());
    return m;
}

Qwen3GGML * store_require_text_enc(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<Qwen3GGML>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    }
    Timer       t;
    Qwen3GGML * m = new Qwen3GGML();
    if (!qwen3_load_text_encoder(m, k.path.c_str())) {
        delete m;
        return nullptr;
    }
    install_entry(s, k, m, bytes_of_text_enc(m), "TextEnc", del_text_enc);
    fprintf(stderr, "[Store] Load TextEnc: %.0f ms\n", t.ms());
    return m;
}

CondGGML * store_require_cond_enc(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<CondGGML>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    }
    Timer      t;
    CondGGML * m = new CondGGML();
    if (!cond_ggml_load(m, k.path.c_str())) {
        delete m;
        return nullptr;
    }
    install_entry(s, k, m, bytes_of_cond_enc(m), "CondEnc", del_cond_enc);
    fprintf(stderr, "[Store] Load CondEnc: %.0f ms\n", t.ms());
    return m;
}

DiTGGML * store_require_dit(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<DiTGGML>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    } else {
        // EVICT_NEVER: still evict stale DiTs with the same base model but
        // different adapter params (path, scale, group_scales). Without this,
        // each adapter switch accumulates a full DiT copy (~3-4 GB) that is
        // never freed, eventually pushing into shared memory or OOM.
        // We already know the exact key is not cached (cache_hit returned null),
        // so any DiT with the same base path must have different adapter fields.
        for (auto it = s->gpu.begin(); it != s->gpu.end(); ) {
            if (it->first.kind == MODEL_DIT && it->first.path == k.path) {
                GpuEntry & e = it->second;
                if (e.refcount > 0) {
                    fprintf(stderr, "[Store] WARNING: stale DiT still in use (refcount=%d), "
                                    "cannot evict for adapter swap\n", e.refcount);
                    ++it;
                    continue;
                }
                fprintf(stderr, "[Store] Adapter swap: evicting DiT (adapter=%s → %s, %.1f MB)\n",
                        it->first.adapter_path.empty() ? "(none)" : it->first.adapter_path.c_str(),
                        k.adapter_path.empty()         ? "(none)" : k.adapter_path.c_str(),
                        (float) e.bytes / (1024.0f * 1024.0f));
                s->handle_to_key.erase(e.ptr);
                e.deleter(e.ptr);
                it = s->gpu.erase(it);
            } else {
                ++it;
            }
        }
    }
    Timer        t;
    DiTGGML *    m       = new DiTGGML();
    const char * adapter = k.adapter_path.empty() ? nullptr : k.adapter_path.c_str();
    if (!dit_ggml_load(m, k.path.c_str(), adapter, k.adapter_scale)) {
        // Retry once after a brief delay — CUDA context may need time to release
        // resources after a model eviction (common during XL↔base swaps).
        fprintf(stderr, "[Store] DiT load failed, retrying in 500ms (possible CUDA context issue)...\n");
        delete m;
        {
            // Temporarily unlock to allow other threads to proceed during sleep.
            // Note: lock_guard does not support unlock, so we use a scoped sleep
            // while still holding the lock. This is acceptable since the store
            // mutex is process-wide and the 500ms delay only matters here.
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        m = new DiTGGML();
        if (!dit_ggml_load(m, k.path.c_str(), adapter, k.adapter_scale)) {
            fprintf(stderr, "[Store] DiT load FAILED on retry. GPU context may be corrupted.\n");
            delete m;
            return nullptr;
        }
        fprintf(stderr, "[Store] DiT load succeeded on retry.\n");
    }
    install_entry(s, k, m, bytes_of_dit(m), "DiT", del_dit);
    fprintf(stderr, "[Store] Load DiT: %.0f ms\n", t.ms());
    return m;
}

VAEEncoder * store_require_vae_enc(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<VAEEncoder>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    }
    Timer        t;
    VAEEncoder * m = new VAEEncoder();
    vae_enc_load(m, k.path.c_str());  // exit(1) on failure, returns void
    install_entry(s, k, m, bytes_of_vae_enc(m), "VAE-Enc", del_vae_enc);
    fprintf(stderr, "[Store] Load VAE-Enc: %.0f ms\n", t.ms());
    return m;
}

VAEGGML * store_require_vae_dec(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<VAEGGML>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    }
    Timer     t;
    VAEGGML * m = new VAEGGML();
    // vae_ggml_load calls exit(1) on hard failure, but CUDA context issues
    // may manifest as a later crash. We wrap in try/catch for safety.
    try {
        vae_ggml_load(m, k.path.c_str());
    } catch (...) {
        fprintf(stderr, "[Store] VAE-Dec load failed, retrying in 500ms...\n");
        delete m;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        m = new VAEGGML();
        try {
            vae_ggml_load(m, k.path.c_str());
            fprintf(stderr, "[Store] VAE-Dec load succeeded on retry.\n");
        } catch (...) {
            fprintf(stderr, "[Store] VAE-Dec load FAILED on retry.\n");
            delete m;
            return nullptr;
        }
    }
    install_entry(s, k, m, bytes_of_vae_dec(m), "VAE-Dec", del_vae_dec);
    fprintf(stderr, "[Store] Load VAE-Dec: %.0f ms\n", t.ms());
    return m;
}

TokGGML * store_require_fsq_tok(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<TokGGML>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    }
    Timer     t;
    TokGGML * m = new TokGGML();
    if (!tok_ggml_load(m, k.path.c_str())) {
        delete m;
        return nullptr;
    }
    install_entry(s, k, m, bytes_of_fsq_tok(m), "FSQ-Tok", del_fsq_tok);
    fprintf(stderr, "[Store] Load FSQ-Tok: %.0f ms\n", t.ms());
    return m;
}

DetokGGML * store_require_fsq_detok(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<DetokGGML>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    }
    Timer       t;
    DetokGGML * m = new DetokGGML();
    if (!detok_ggml_load(m, k.path.c_str())) {
        delete m;
        return nullptr;
    }
    install_entry(s, k, m, bytes_of_fsq_detok(m), "FSQ-Detok", del_fsq_detok);
    fprintf(stderr, "[Store] Load FSQ-Detok: %.0f ms\n", t.ms());
    return m;
}

#ifdef HOT_STEP_TRT
TextEncTrt * store_require_text_enc_trt(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<TextEncTrt>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    }
    Timer        t;
    TextEncTrt * m = new TextEncTrt();
    if (!text_enc_trt_load(m, k.path.c_str())) {
        delete m;
        return nullptr;
    }
    // bytes == 0: TRT owns the engine VRAM; the store budget does not track it.
    install_entry(s, k, m, 0, "TextEnc-TRT", del_text_enc_trt);
    fprintf(stderr, "[Store] Load TextEnc-TRT: %.0f ms\n", t.ms());
    return m;
}

void store_release_text_enc_trt(ModelStore * s, TextEncTrt * handle) {
    if (!s || !handle) {
        return;
    }
    // Free the per-job device I/O buffers + execution context, then run
    // the generic refcount decrement.
    text_enc_trt_release_evictable(handle);
    store_release(s, handle);
}

CondEncTrt * store_require_cond_enc_trt(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<CondEncTrt>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    }
    Timer        t;
    CondEncTrt * m = new CondEncTrt();
    if (!cond_enc_trt_load(m, k.path.c_str())) {
        delete m;
        return nullptr;
    }
    // bytes == 0: TRT owns the engine VRAM; the store budget does not track it.
    install_entry(s, k, m, 0, "CondEnc-TRT", del_cond_enc_trt);
    fprintf(stderr, "[Store] Load CondEnc-TRT: %.0f ms\n", t.ms());
    return m;
}

void store_release_cond_enc_trt(ModelStore * s, CondEncTrt * handle) {
    if (!s || !handle) {
        return;
    }
    // Free the per-job device I/O buffers + execution context, then run
    // the generic refcount decrement.
    cond_enc_trt_release_evictable(handle);
    store_release(s, handle);
}

DitTrt * store_require_dit_trt(ModelStore * s, const ModelKey & k) {
    std::lock_guard<std::mutex> lock(s->mtx);
    if (auto * hit = cache_hit<DitTrt>(s, k)) {
        return hit;
    }
    if (s->policy == EVICT_STRICT) {
        evict_all_except(s, k);
    }
    Timer    t;
    DitTrt * m = new DitTrt();
    // Deserialize the engine (weights embedded) and cache base weights
    // from the ONNX refitter for adapter support.
    if (!dit_trt_load(m, k.path.c_str(), k.engine_onnx_path.c_str())) {
        delete m;
        return nullptr;
    }
    // bytes == 0: TRT owns the engine VRAM; the store budget does not track it.
    install_entry(s, k, m, 0, "DiT-TRT", del_dit_trt);
    fprintf(stderr, "[Store] Load DiT-TRT: %.0f ms\n", t.ms());
    return m;
}

void store_release_dit_trt(ModelStore * s, DitTrt * handle) {
    if (!s || !handle) {
        return;
    }
    // Free the per-job device I/O buffers + execution context,
    // then run the generic refcount decrement.
    dit_trt_release_evictable(handle);
    store_release(s, handle);
}
#endif

void store_release(ModelStore * s, void * handle) {
    if (!s || !handle) {
        return;
    }
    std::lock_guard<std::mutex> lock(s->mtx);
    auto                        hit = s->handle_to_key.find(handle);
    if (hit == s->handle_to_key.end()) {
        fprintf(stderr, "[Store] WARNING: release of unknown handle %p\n", handle);
        return;
    }
    auto gpu_it = s->gpu.find(hit->second);
    if (gpu_it == s->gpu.end()) {
        fprintf(stderr, "[Store] WARNING: release of handle %p whose entry is gone\n", handle);
        s->handle_to_key.erase(hit);
        return;
    }
    GpuEntry & e = gpu_it->second;
    assert(e.refcount > 0);
    e.refcount--;
    if (e.refcount == 0 && s->policy == EVICT_STRICT) {
        // Under EVICT_STRICT, every module is fully unloaded after use.
        // On the next request, the engine is reloaded from disk.
        fprintf(stderr, "[Store] Unload %s (%.1f MB)\n", e.label, (float) e.bytes / (1024.0f * 1024.0f));
        e.deleter(e.ptr);
        s->handle_to_key.erase(hit);
        s->gpu.erase(gpu_it);
    }
}

void store_prune_bundle_except(ModelStore * s, const std::string & keep_bundle_dir) {
    if (!s) {
        return;
    }
    std::lock_guard<std::mutex> lock(s->mtx);
    const std::string           keep = store_normalize_bundle_dir(keep_bundle_dir);
    for (auto it = s->gpu.begin(); it != s->gpu.end();) {
        std::string entry_bundle = store_normalize_bundle_dir(it->first.path);
        if (entry_bundle.empty() || entry_bundle == keep) {
            // Not a bundle module, or part of the bundle we are keeping.
            ++it;
            continue;
        }
        GpuEntry & e = it->second;
        if (e.refcount > 0) {
            fprintf(stderr,
                    "[Store] FATAL: prune cannot evict %s (refcount=%d) from bundle %s while "
                    "switching to %s; a different-bundle module is still in use\n",
                    e.label, e.refcount, entry_bundle.c_str(), keep.c_str());
            abort();
        }
        fprintf(stderr, "[Store] Bundle-prune evict %s (%.1f MB) from %s\n", e.label,
                (float) e.bytes / (1024.0f * 1024.0f), entry_bundle.c_str());
        s->handle_to_key.erase(e.ptr);
        e.deleter(e.ptr);
        it = s->gpu.erase(it);
    }
}

// Each accessor has the same shape: lookup, load on miss, return cached
// pointer. Deleter is a lightweight lambda since these types are simple.
BPETokenizer * store_bpe(ModelStore * s, const char * lm_path) {
    std::lock_guard<std::mutex> lock(s->mtx);
    std::string                 key = store_bpe_cache_key(lm_path);
    auto                        it  = s->bpe_by_path.find(key);
    if (it != s->bpe_by_path.end()) {
        return static_cast<BPETokenizer *>(it->second.ptr);
    }
    auto * bpe = new BPETokenizer();

    // Detect format: .gguf file or safetensors directory
    bool loaded = false;
    size_t len = strlen(lm_path);
    bool is_gguf = (len >= 5 && strcmp(lm_path + len - 5, ".gguf") == 0);

    if (is_gguf) {
        loaded = load_bpe_from_gguf(bpe, lm_path);
    } else {
        // Safetensors/ONNX directory: load from sidecar files
        std::string vocab_path  = std::string(lm_path) + WS_SEP + "vocab.json";
        std::string merges_path = std::string(lm_path) + WS_SEP + "merges.txt";
        loaded = load_bpe_from_files(bpe, vocab_path.c_str(), merges_path.c_str());

        // Fallback: ONNX LM models may live in a subdirectory (e.g. onnx/lm-4B/)
        // with tokenizer files in the parent directory (e.g. onnx/)
        if (!loaded) {
            std::string parent;
            {
                std::string p = lm_path;
                // Strip trailing separator
                while (!p.empty() && (p.back() == '/' || p.back() == '\\')) p.pop_back();
                auto slash = p.find_last_of("/\\");
                parent = (slash != std::string::npos) ? p.substr(0, slash) : ".";
            }
            vocab_path  = parent + WS_SEP + "vocab.json";
            merges_path = parent + WS_SEP + "merges.txt";
            loaded = load_bpe_from_files(bpe, vocab_path.c_str(), merges_path.c_str());
            if (loaded) {
                fprintf(stderr, "[BPE] Loaded from parent dir: %s\n", parent.c_str());
            }
        }
    }

    if (!loaded) {
        delete bpe;
        return nullptr;
    }
    CpuEntry e;
    e.ptr     = bpe;
    e.deleter = [](void * p) {
        delete static_cast<BPETokenizer *>(p);
    };
    s->bpe_by_path.emplace(key, e);
    return bpe;
}

const float * store_silence(ModelStore * s, const char * dit_path) {
    std::lock_guard<std::mutex> lock(s->mtx);
    std::string                 key = dit_path ? dit_path : "";
    auto                        it  = s->silence_by_path.find(key);
    if (it != s->silence_by_path.end()) {
        return static_cast<const std::vector<float> *>(it->second.ptr)->data();
    }

    auto * vec = new std::vector<float>();

    // Detect format: .gguf file or safetensors directory
    size_t len = strlen(dit_path);
    bool is_gguf = (len >= 5 && strcmp(dit_path + len - 5, ".gguf") == 0);

    if (is_gguf) {
        // GGUF: silence_latent is embedded as a tensor (already transposed by convert.py)
        GGUFModel gf = {};
        if (!gf_load(&gf, dit_path)) {
            fprintf(stderr, "[Store] FATAL: silence cannot open %s\n", dit_path);
            delete vec;
            return nullptr;
        }
        const void * sl = gf_get_data(gf, "silence_latent");
        if (!sl) {
            fprintf(stderr, "[Store] FATAL: silence_latent not found in %s\n", dit_path);
            gf_close(&gf);
            delete vec;
            return nullptr;
        }
        vec->resize(15000 * 64);
        memcpy(vec->data(), sl, 15000 * 64 * sizeof(float));
        gf_close(&gf);
    } else {
        // Safetensors directory: read from silence_latent.pt sidecar
        std::string pt_path = std::string(dit_path) + WS_SEP + "silence_latent.pt";
        if (!sl_read_silence_latent(pt_path.c_str(), *vec)) {
            fprintf(stderr, "[Store] FATAL: cannot read %s\n", pt_path.c_str());
            delete vec;
            return nullptr;
        }
    }

    CpuEntry e;
    e.ptr     = vec;
    e.deleter = [](void * p) {
        delete static_cast<std::vector<float> *>(p);
    };
    s->silence_by_path.emplace(key, e);
    return vec->data();
}

MetadataFSM * store_fsm(ModelStore * s, const char * lm_path, int vocab_size) {
    // BPE must exist first: FSM is built from the BPE + vocab_size.
    BPETokenizer * bpe = store_bpe(s, lm_path);
    if (!bpe) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(s->mtx);
    std::string                 key = lm_path ? lm_path : "";
    auto                        it  = s->fsm_by_path.find(key);
    if (it != s->fsm_by_path.end()) {
        return static_cast<MetadataFSM *>(it->second.ptr);
    }
    auto * fsm = new MetadataFSM();
    fsm->init(*bpe, vocab_size);
    CpuEntry e;
    e.ptr     = fsm;
    e.deleter = [](void * p) {
        delete static_cast<MetadataFSM *>(p);
    };
    s->fsm_by_path.emplace(key, e);
    return fsm;
}

const DiTMeta * store_dit_meta(ModelStore * s, const char * dit_path) {
    std::lock_guard<std::mutex> lock(s->mtx);
    std::string                 key = dit_path ? dit_path : "";
    auto                        it  = s->dit_meta_by_path.find(key);
    if (it != s->dit_meta_by_path.end()) {
        return static_cast<const DiTMeta *>(it->second.ptr);
    }
    auto * meta = new DiTMeta();
    if (!dit_ggml_load_config(&meta->cfg, dit_path)) {
        fprintf(stderr, "[Store] FATAL: DiT config cannot open %s\n", dit_path);
        delete meta;
        return nullptr;
    }

    // Detect format: .gguf file or safetensors directory
    size_t len = strlen(dit_path);
    bool is_gguf = (len >= 5 && strcmp(dit_path + len - 5, ".gguf") == 0);

    if (is_gguf) {
        // GGUF path: read metadata from GGUF KV + embedded tensors
        GGUFModel gf = {};
        if (!gf_load(&gf, dit_path)) {
            fprintf(stderr, "[Store] FATAL: DiT cannot reopen %s for metadata\n", dit_path);
            delete meta;
            return nullptr;
        }
        meta->is_turbo = gf_get_bool(gf, "acestep.is_turbo");

        // Detect blend/merge models from filename
        {
            std::string basename = dit_path;
            auto slash = basename.find_last_of("/\\");
            if (slash != std::string::npos) basename = basename.substr(slash + 1);
            for (auto & c : basename) c = (char)tolower((unsigned char)c);
            meta->is_merge = (basename.find("merge") != std::string::npos ||
                              basename.find("sftturbo") != std::string::npos);
        }

        // silence_latent
        const void * sl = gf_get_data(gf, "silence_latent");
        if (!sl) {
            fprintf(stderr, "[Store] FATAL: silence_latent not found in %s\n", dit_path);
            gf_close(&gf);
            delete meta;
            return nullptr;
        }
        meta->silence_full.resize(15000 * 64);
        memcpy(meta->silence_full.data(), sl, 15000 * 64 * sizeof(float));

        // null_condition_emb
        struct ggml_tensor * nce_meta = ggml_get_tensor(gf.meta, "null_condition_emb");
        if (nce_meta) {
            int          emb_n = (int) ggml_nelements(nce_meta);
            const void * raw   = gf_get_data(gf, "null_condition_emb");
            meta->null_cond_cpu.resize(emb_n);
            if (nce_meta->type == GGML_TYPE_BF16) {
                const uint16_t * src = (const uint16_t *) raw;
                for (int i = 0; i < emb_n; i++) {
                    uint32_t w = (uint32_t) src[i] << 16;
                    memcpy(&meta->null_cond_cpu[i], &w, 4);
                }
            } else if (nce_meta->type == GGML_TYPE_F32) {
                memcpy(meta->null_cond_cpu.data(), raw, emb_n * sizeof(float));
            } else {
                fprintf(stderr, "[Store] FATAL: null_condition_emb unexpected type %d\n", nce_meta->type);
                gf_close(&gf);
                delete meta;
                return nullptr;
            }
        }
        gf_close(&gf);
    } else {
        // Safetensors directory or ONNX file path: read from sidecar files
        // For ONNX: sidecars are in the parent directory
        // For safetensors: sidecars are in the model directory itself
        std::string sidecar_dir = dit_sidecar_dir(dit_path);
        bool is_onnx = dit_ends_with_onnx(dit_path);

        std::string cfg_path = sidecar_dir + WS_SEP + "config.json";
        meta->is_turbo = config_json_get_is_turbo(cfg_path.c_str());
        meta->is_merge = config_json_get_is_merge(cfg_path.c_str());

        // Also detect merge from directory/file name
        {
            std::string basename = dit_path;
            auto slash = basename.find_last_of("/\\");
            if (slash != std::string::npos) basename = basename.substr(slash + 1);
            for (auto & c : basename) c = (char)tolower((unsigned char)c);
            if (basename.find("merge") != std::string::npos ||
                basename.find("sftturbo") != std::string::npos ||
                basename.find("turbo") != std::string::npos) {
                meta->is_merge = true;
            }
        }

        // silence_latent from silence_latent.pt
        std::string pt_path = sidecar_dir + WS_SEP + "silence_latent.pt";
        if (!sl_read_silence_latent(pt_path.c_str(), meta->silence_full)) {
            fprintf(stderr, "[Store] FATAL: cannot read %s\n", pt_path.c_str());
            delete meta;
            return nullptr;
        }

        // null_condition_emb: read from safetensors model file (not available for ONNX)
        if (!is_onnx) {
            STMulti sm = {};
            if (st_multi_open(&sm, dit_path)) {
                auto [shard_idx, e] = sm.find("null_condition_emb");
                if (e) {
                    ggml_type nce_type = st_ggml_type(*e);
                    size_t    n = 1;
                    for (int d = 0; d < e->n_dims; d++) n *= (size_t) e->shape[d];
                    const void * raw = sm.data(shard_idx, *e);
                    meta->null_cond_cpu.resize(n);
                    if (nce_type == GGML_TYPE_BF16) {
                        const uint16_t * src = (const uint16_t *) raw;
                        for (size_t i = 0; i < n; i++) {
                            uint32_t w = (uint32_t) src[i] << 16;
                            memcpy(&meta->null_cond_cpu[i], &w, 4);
                        }
                    } else if (nce_type == GGML_TYPE_F32) {
                        memcpy(meta->null_cond_cpu.data(), raw, n * sizeof(float));
                    }
                }
                st_multi_close(&sm);
            }
        } else {
            std::string null_path = sidecar_dir + WS_SEP + "null_condition_emb.bin";
            if (!read_f32_vector_sidecar(null_path, meta->null_cond_cpu)) {
                fprintf(stderr, "[Store] FATAL: ONNX DiT requires %s for CFG padding\n", null_path.c_str());
                delete meta;
                return nullptr;
            }
            fprintf(stderr, "[Store] ONNX DiT: null_condition_emb [%zu] from %s\n",
                    meta->null_cond_cpu.size(), null_path.c_str());
        }
    }

    CpuEntry e;
    e.ptr     = meta;
    e.deleter = [](void * p) {
        delete static_cast<DiTMeta *>(p);
    };
    s->dit_meta_by_path.emplace(key, e);
    return meta;
}

size_t store_vram_bytes(const ModelStore * s) {
    if (!s) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(s->mtx);
    size_t                      total = 0;
    for (const auto & kv : s->gpu) {
        total += kv.second.bytes;
    }
    return total;
}

int store_gpu_module_count(const ModelStore * s) {
    if (!s) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(s->mtx);
    return (int) s->gpu.size();
}

#ifdef HOT_STEP_MODEL_STORE_TEST
namespace {
// Synthetic module backing a test cache entry: a heap object whose deleter
// bumps an externally-owned counter, so the test can prove eviction freed
// exactly the expected entries.
struct SyntheticModule {
    int * deleter_calls;
};
static void del_synthetic(void * p) {
    auto * m = static_cast<SyntheticModule *>(p);
    if (m->deleter_calls) {
        (*m->deleter_calls)++;
    }
    delete m;
}
}  // namespace

void store_test_install_synthetic(ModelStore *        s,
                                  ModelKind           kind,
                                  const std::string & path,
                                  size_t              bytes,
                                  int                 refcount,
                                  int *               deleter_calls) {
    std::lock_guard<std::mutex> lock(s->mtx);
    ModelKey k;
    k.kind = kind;
    k.path = path;
    auto * m       = new SyntheticModule();
    m->deleter_calls = deleter_calls;
    GpuEntry e;
    e.ptr      = m;
    e.bytes    = bytes;
    e.refcount = refcount;
    e.deleter  = del_synthetic;
    e.label    = "Synthetic";
    s->gpu.emplace(k, e);
    s->handle_to_key.emplace(m, k);
}

bool store_test_has_entry(const ModelStore * s, ModelKind kind, const std::string & path) {
    std::lock_guard<std::mutex> lock(s->mtx);
    ModelKey k;
    k.kind = kind;
    k.path = path;
    return s->gpu.find(k) != s->gpu.end();
}

void * store_test_install_synthetic_h(ModelStore *        s,
                                      ModelKind           kind,
                                      const std::string & path,
                                      const std::string & artifact_fingerprint,
                                      size_t              bytes,
                                      int                 refcount,
                                      int *               deleter_calls) {
    std::lock_guard<std::mutex> lock(s->mtx);
    ModelKey k;
    k.kind                 = kind;
    k.path                 = path;
    k.artifact_fingerprint = artifact_fingerprint;
    auto * m         = new SyntheticModule();
    m->deleter_calls = deleter_calls;
    GpuEntry e;
    e.ptr      = m;
    e.bytes    = bytes;
    e.refcount = refcount;
    e.deleter  = del_synthetic;
    e.label    = "Synthetic";
    s->gpu.emplace(k, e);
    s->handle_to_key.emplace(m, k);
    return m;
}

int store_test_refcount(const ModelStore * s,
                        ModelKind           kind,
                        const std::string & path,
                        const std::string & artifact_fingerprint) {
    std::lock_guard<std::mutex> lock(s->mtx);
    ModelKey k;
    k.kind                 = kind;
    k.path                 = path;
    k.artifact_fingerprint = artifact_fingerprint;
    auto it = s->gpu.find(k);
    return it == s->gpu.end() ? -1 : it->second.refcount;
}
#endif
