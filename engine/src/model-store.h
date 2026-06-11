#pragma once
// model-store.h: centralised ownership of GGML modules
//
// VRAM policy doctrine. READ THIS BEFORE CHANGING ANYTHING IN THIS FILE.
//
//   --keep-loaded (EVICT_NEVER)
//       Everything stays in VRAM. No reload, ever. The user is telling us
//       they have the budget for the full working set. Do not second-guess
//       them by adding smart eviction rules.
//
//   default (EVICT_STRICT)
//       Maximum VRAM optimisation. At most one GPU module resident at a
//       time. VAE tiles never coexist with DiT weights or LM weights by
//       construction, because only one module is ever loaded. No special
//       case needed. All modules are fully unloaded after use, including
//       TRT engines. On subsequent runs, engines are reloaded from disk.
//
//   invariant held under BOTH policies
//       Exactly ONE LM instance for the whole process. ace_lm (generate)
//       and ace_understand must share the same LM: duplicating it would
//       waste gigabytes for no gain. This is enforced by making the
//       ModelKey identical across both pipelines (same path, same
//       max_seq, same n_kv_sets).
//
//       Modules that carry an artifact fingerprint key on it so an
//       in-place update to a backing file or required sidecar reloads the
//       module without changing the load path.
//
// A ModelStore holds the GGML module instances that the pipelines need
// (Qwen3 LM, DiT, VAE encoder, VAE decoder, FSQ tokenizer, etc). Pipelines
// ask the store for a module by key and return it when done. The store
// decides what stays in VRAM and what gets evicted, following the policy
// above set at creation time.
//
// Keys
//   A module is uniquely identified by (kind, path, extras). Two requires
//   with the same key return the same instance. Two requires with different
//   extras (for instance two DiTs with different adapters) are two distinct
//   modules. The LM key deliberately fixes n_kv_sets at 2 * max_batch for
//   both ace_lm and ace_understand so they share one instance.
//
// Refcounting
//   Each module has a refcount. require increments it, release decrements.
//   In EVICT_STRICT, a module with refcount > 0 cannot be evicted: a
//   conflicting require is a programming error (asserts). This catches
//   accidental overlap between modules that must not coexist.
//
// Thread safety
//   All public entry points take a single mutex. Load / unload / hit
//   decisions are serialised. Compute itself runs outside the lock.

#include "bpe.h"
#include "cond-enc.h"
#include "cond-enc-trt.h"
#include "dit.h"
#include "dit-trt.h"
#include "fsq-detok.h"
#include "fsq-tok.h"
#include "hot-step-params.h"  // AdapterGroupScales
#include "metadata-fsm.h"
#include "qwen3-enc.h"
#include "qwen3-lm.h"
#include "text-enc-trt.h"
#include "vae-enc.h"
#include "vae.h"

#include <cstddef>
#include <string>

struct ModelStore;

enum ModelKind {
    MODEL_LM,           // Qwen3LM        from acestep-5Hz-lm-*.gguf
    MODEL_TEXT_ENC,     // Qwen3GGML      from Qwen3-Embedding-*.gguf
    MODEL_COND_ENC,     // CondGGML       from acestep-v15-*.gguf (cond_enc.*)
    MODEL_DIT,          // DiTGGML        from acestep-v15-*.gguf
    MODEL_VAE_ENC,      // VAEEncoder     from vae.gguf (encoder.*)
    MODEL_VAE_DEC,      // VAEGGML        from vae.gguf (decoder.*)
    MODEL_FSQ_TOK,      // TokGGML        from acestep-v15-*.gguf (tokenizer.*)
    MODEL_FSQ_DETOK,    // DetokGGML      from acestep-v15-*.gguf (detokenizer.*)
    // Native TensorRT bundle components. Each keys on its component path under
    // a single trt-bundles/<name>/ directory; a bundle's three modules share
    // that bundle-dir prefix. Under EVICT_STRICT, engines are fully unloaded
    // after use and reloaded from disk on next request. With --keep-loaded,
    // engines stay resident for reuse. store_prune_bundle_except evicts a whole
    // bundle group when switching bundles.
    MODEL_DIT_TRT,      // DitTrt         from trt-bundles/<name>/dit.engine
    MODEL_TEXT_ENC_TRT, // TextEncTrt     from trt-bundles/<name>/text_encoder.engine
    MODEL_COND_ENC_TRT, // CondEncTrt     from trt-bundles/<name>/cond_encoder.engine
};

struct ModelKey {
    ModelKind   kind = MODEL_LM;
    std::string path;  // GGUF path, engine path, or artifact directory
    std::string artifact_fingerprint;  // backing-file/sidecar content key (modules that opt in)
    // LM-only extras (ignored for other kinds):
    int         max_seq = 0;    // KV cache length
    int         n_kv_sets = 0;  // number of KV sets (1 or 2*max_batch with CFG)
    // DiT-only extras (ignored for other kinds):
    std::string adapter_path;   // "" when no adapter
    float       adapter_scale = 1.0f;  // significant when adapter_path is set
    AdapterGroupScales adapter_group_scales = {};  // per-group scale multipliers baked into merged weights
    // DIT_TRT-only load inputs (ignored for other kinds). The store key for
    // MODEL_DIT_TRT is (kind + path[engine] + artifact_fingerprint); adapter
    // fields are NOT part of the key because adapter state is refit in place on
    // the resident engine (one deserialized engine, refit per request), exactly
    // as the prior static singleton did. These two carry the extra load
    // arguments the native loader needs but that do not change identity:
    std::string engine_onnx_path;          // ONNX path for adapter base-weight caching
};

enum EvictPolicy {
    EVICT_STRICT,  // default: at most one GPU module resident at a time
    EVICT_NEVER,   // --keep-loaded: never evict, accumulate
};

// DiT metadata cached on the CPU: needed by text encoding and T resolution
// before the DiT itself is loaded on the GPU.
struct DiTMeta {
    DiTGGMLConfig      cfg;
    std::vector<float> silence_full;   // [15000, 64] f32, from silence_latent tensor
    std::vector<float> null_cond_cpu;  // [hidden_size] f32, empty when the model has none
    bool               is_turbo;
    bool               is_merge;  // base/turbo blend — skip turbo restrictions
};

ModelStore * store_create(EvictPolicy policy);
void         store_free(ModelStore * s);
void         store_set_policy(ModelStore * s, EvictPolicy policy);
EvictPolicy  store_get_policy(ModelStore * s);
// Typed GPU module accessors. Each returns a pointer owned by the store;
// never free it yourself. Returns NULL on load failure.
//
// After require, the module stays resident with a refcount > 0 until the
// matching release. In EVICT_STRICT, require evicts every other GPU module
// whose refcount is zero; if any conflicting module has refcount > 0 the
// store aborts (a programming error in the caller).
Qwen3LM *    store_require_lm(ModelStore * s, const ModelKey & k);
Qwen3GGML *  store_require_text_enc(ModelStore * s, const ModelKey & k);
CondGGML *   store_require_cond_enc(ModelStore * s, const ModelKey & k);
DiTGGML *    store_require_dit(ModelStore * s, const ModelKey & k);
VAEEncoder * store_require_vae_enc(ModelStore * s, const ModelKey & k);
VAEGGML *    store_require_vae_dec(ModelStore * s, const ModelKey & k);
TokGGML *    store_require_fsq_tok(ModelStore * s, const ModelKey & k);
DetokGGML *  store_require_fsq_detok(ModelStore * s, const ModelKey & k);

#ifdef HOT_STEP_TRT
// Native TRT text encoder (MODEL_TEXT_ENC_TRT). Under EVICT_STRICT the engine
// is fully unloaded after inference; under EVICT_NEVER it stays loaded.
// store_release_text_enc_trt frees the per-job device I/O buffers + execution
// context and decrements the refcount.
TextEncTrt * store_require_text_enc_trt(ModelStore * s, const ModelKey & k);
void         store_release_text_enc_trt(ModelStore * s, TextEncTrt * handle);

// Native TRT condition encoder (MODEL_COND_ENC_TRT). Same lifecycle as
// the text encoder above.
CondEncTrt * store_require_cond_enc_trt(ModelStore * s, const ModelKey & k);
void         store_release_cond_enc_trt(ModelStore * s, CondEncTrt * handle);

// Native TRT DiT (MODEL_DIT_TRT). The key is (kind + engine path +
// artifact_fingerprint); a changed engine/sidecar fingerprint reloads.
// Adapter (LoRA) state is NOT part of the key: the caller refits adapters in
// place on the returned handle (dit_trt_refit_* / adapter_trt_*), keeping a
// single engine across adapter switches. k.engine_onnx_path supplies the
// ONNX path needed for base-weight caching. Under EVICT_STRICT the engine
// is fully unloaded after inference; under EVICT_NEVER it stays loaded.
DitTrt *     store_require_dit_trt(ModelStore * s, const ModelKey & k);
void         store_release_dit_trt(ModelStore * s, DitTrt * handle);
#endif

// Release decrements the refcount for the module behind this handle.
// Pass exactly the pointer returned by require. After release, the pointer
// must not be used: in EVICT_STRICT it may be unloaded immediately.
void store_release(ModelStore * s, void * handle);

// Group-evict every GPU module that belongs to a TRT bundle directory OTHER
// than keep_bundle_dir. A module belongs to a bundle when its key.path resolves
// under some trt-bundles/<name>/ directory; that <name> directory is its bundle
// dir. Modules not under any trt-bundles/ directory (GGML/GGUF modules) are
// never touched. For each other-bundle entry: refcount 0 evicts it (deleter
// runs, bytes freed, removed from cache); refcount > 0 is a fatal error — a
// different-bundle module is still in use, which cannot happen under
// EVICT_STRICT and means a caller leaked a handle across a bundle switch.
// keep_bundle_dir is matched against the derived bundle dir, so passing either
// the bundle directory itself or any component path under it keeps that bundle.
void store_prune_bundle_except(ModelStore * s, const std::string & keep_bundle_dir);

// CPU-resident accessors. Loaded on first call, kept forever, never evicted.
// All small (a few MB total). Return NULL on load failure.
BPETokenizer *  store_bpe(ModelStore * s, const char * lm_path);
const float *   store_silence(ModelStore * s, const char * dit_path);
MetadataFSM *   store_fsm(ModelStore * s, const char * lm_path, int vocab_size);
const DiTMeta * store_dit_meta(ModelStore * s, const char * dit_path);

// Observability: sum of currently resident GPU module weight buffers, and
// the count of loaded GPU modules. Used by test-model-store to assert
// eviction policy invariants.
size_t store_vram_bytes(const ModelStore * s);
int    store_gpu_module_count(const ModelStore * s);

// RAII helper. Builds on top of store_release, nothing else.
struct ModelHandle {
    ModelStore * store;
    void *       ptr;

    ModelHandle(ModelStore * s, void * p) : store(s), ptr(p) {}

    ~ModelHandle() {
        if (store && ptr) {
            store_release(store, ptr);
        }
    }

    // non-copyable, movable
    ModelHandle(const ModelHandle &)             = delete;
    ModelHandle & operator=(const ModelHandle &) = delete;

    ModelHandle(ModelHandle && o) noexcept : store(o.store), ptr(o.ptr) {
        o.store = nullptr;
        o.ptr   = nullptr;
    }
};

#ifdef HOT_STEP_MODEL_STORE_TEST
// Test-only seam: install a synthetic GPU cache entry without loading a real
// module, so eviction policy (store_prune_bundle_except) can be exercised
// against a controlled cache. The deleter invokes *deleter_calls (the entry's
// "free") on eviction, letting the test prove the right entries were freed.
// Compiled only when HOT_STEP_MODEL_STORE_TEST is defined (the test target).
void store_test_install_synthetic(ModelStore *      s,
                                  ModelKind         kind,
                                  const std::string & path,
                                  size_t            bytes,
                                  int               refcount,
                                  int *             deleter_calls);
// Test-only: does the cache hold an entry for exactly (kind, path)?
bool store_test_has_entry(const ModelStore * s, ModelKind kind, const std::string & path);

// Test-only: install a synthetic entry and return its handle pointer so the
// test can drive store_release / store_require refcounting directly (the
// void-returning installer above is for prune tests that key only by path).
// artifact_fingerprint participates in the MODEL_DIT_TRT key, so it is settable
// here to exercise fingerprint-driven cache identity.
void * store_test_install_synthetic_h(ModelStore *        s,
                                      ModelKind           kind,
                                      const std::string & path,
                                      const std::string & artifact_fingerprint,
                                      size_t              bytes,
                                      int                 refcount,
                                      int *               deleter_calls);
// Test-only: current refcount of (kind, path, artifact_fingerprint), or -1 if
// the entry is absent. Lets the test assert require++/release-- arithmetic and
// resident-shell-on-release retention.
int store_test_refcount(const ModelStore * s,
                        ModelKind           kind,
                        const std::string & path,
                        const std::string & artifact_fingerprint);
#endif
