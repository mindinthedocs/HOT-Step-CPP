// pipeline-synth.cpp: ACE-Step synthesis pipeline implementation
//
// Thin orchestrator over a ModelStore. Holds no GPU module, no CPU-cached
// DiT state of its own: the store exposes DiTMeta (silence, null_cond, cfg,
// is_turbo) and each op acquires the GPU modules it needs on the fly.
//
// One function per task. Each task reads its inputs, poses its flags on
// SynthState, then calls the ops in a linear sequence. The dispatcher at the
// bottom picks the right task function from reqs[0].task_type.

#include "pipeline-synth.h"

#include "pipeline-synth-impl.h"
#include "pipeline-synth-ops.h"
#include "task-types.h"
#include "trt-artifact-manifest.h"

// LRC alignment (Phase 3)
#include "alignment-config.h"
#include "dit-alignment-graph.h"
#include "lrc-alignment.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static bool synth_file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

static std::string synth_fsq_metadata_path(const std::string & sidecar) {
    const char * suffix = ".safetensors";
    size_t suffix_len = strlen(suffix);
    if (sidecar.size() >= suffix_len &&
        sidecar.compare(sidecar.size() - suffix_len, suffix_len, suffix) == 0) {
        return sidecar.substr(0, sidecar.size() - suffix_len) + ".metadata.json";
    }
    return sidecar + ".metadata.json";
}

static bool synth_require_file(const char * label, const std::string & path, bool & ok) {
    if (synth_file_exists(path)) {
        return true;
    }
    fprintf(stderr, "[Synth-Load] FATAL: missing %s: %s\n", label, path.c_str());
    ok = false;
    return false;
}

static bool synth_validate_onnx_artifacts(const char * dit_path) {
    std::string onnx_path = dit_path ? dit_path : "";
    std::string onnx_dir  = dit_sidecar_dir(dit_path);
    std::string stem      = onnx_path;
    if (stem.size() >= 5 && stem.substr(stem.size() - 5) == ".onnx") {
        stem.resize(stem.size() - 5);
    }

    bool ok = true;
    synth_require_file("DiT ONNX", onnx_path, ok);
    synth_require_file("silence_latent.pt", onnx_dir + WS_SEP + "silence_latent.pt", ok);
    synth_require_file("null_condition_emb.bin", onnx_dir + WS_SEP + "null_condition_emb.bin", ok);
    synth_require_file("cond_encoder.onnx", onnx_dir + WS_SEP + "cond_encoder.onnx", ok);
    synth_require_file("DiT metadata", stem + ".metadata.json", ok);
    synth_require_file("precision manifest", stem + ".precision-manifest.json", ok);
    synth_require_file("refit manifest", onnx_path + ".refit_manifest.json", ok);
    synth_require_file("prebuilt DiT TensorRT engine", stem + ".engine", ok);
    synth_require_file("prebuilt DiT TensorRT engine metadata", stem + ".engine.metadata.json", ok);

    TrtPrecisionManifestCheck precision_manifest;
    if (synth_file_exists(stem + ".precision-manifest.json")) {
        std::string precision_err;
        if (!trt_artifact_validate_precision_manifest(onnx_path, &precision_manifest, &precision_err)) {
            fprintf(stderr, "[Synth-Load] FATAL: invalid DiT precision manifest: %s\n", precision_err.c_str());
            ok = false;
        } else {
            fprintf(stderr,
                    "[Synth-Load] Precision manifest OK: policy=%s matched=%llu downcast=%llu int8=%llu preserved=%llu\n",
                    precision_manifest.policy.c_str(),
                    (unsigned long long) precision_manifest.matched_allowlist_count,
                    (unsigned long long) precision_manifest.downcast_to_fp16_count,
                    (unsigned long long) precision_manifest.quantized_to_int8_count,
                    (unsigned long long) precision_manifest.preserved_fp32_count);
        }
    }

    std::string precision_report = stem + ".precision.json";
    if (synth_file_exists(precision_report)) {
        fprintf(stderr, "[Synth-Load] Precision report: %s\n", precision_report.c_str());
    }

    std::string engine_path = stem + ".engine";
    if (synth_file_exists(engine_path) && synth_file_exists(engine_path + ".metadata.json")) {
        fprintf(stderr, "[Synth-Load] Prebuilt TRT engine: %s\n", engine_path.c_str());
        std::string engine_metadata = engine_path + ".metadata.json";
        if (!precision_manifest.policy.empty()) {
            TrtEngineMetadataCheck engine_check;
            std::string engine_err;
            if (!trt_artifact_validate_engine_metadata(engine_metadata, precision_manifest.policy,
                                                       -1, -1, &engine_check, &engine_err)) {
                fprintf(stderr, "[Synth-Load] FATAL: prebuilt TRT engine metadata incompatible: %s\n",
                        engine_err.c_str());
                ok = false;
            } else {
                fprintf(stderr, "[Synth-Load] TRT engine metadata OK: %s (TRT %s, profile=%s)\n",
                        engine_check.source_path.c_str(),
                        engine_check.tensorrt_version.c_str(),
                        engine_check.profile.c_str());
            }
        }
    }

    std::string fsq_sidecar = onnx_dir + WS_SEP + "fsq.safetensors";
    if (synth_file_exists(fsq_sidecar)) {
        fprintf(stderr, "[Synth-Load] ONNX FSQ sidecar: %s\n", fsq_sidecar.c_str());
        synth_require_file("FSQ sidecar metadata", synth_fsq_metadata_path(fsq_sidecar), ok);
    } else {
        fprintf(stderr, "[Synth-Load] FATAL: missing ONNX FSQ sidecar: %s\n", fsq_sidecar.c_str());
        ok = false;
    }
    return ok;
}

void ace_synth_default_params(AceSynthParams * p) {
    p->text_encoder_path = NULL;
    p->dit_path          = NULL;
    p->aux_dit_path      = NULL;
    p->vae_path          = NULL;
    p->adapter_path      = NULL;
    p->adapter_scale     = 1.0f;
    p->use_fa            = true;
    p->clamp_fp16        = false;
    p->use_batch_cfg     = true;
    p->vae_chunk         = 1024;
    p->vae_overlap       = 64;
    p->dump_dir          = NULL;
    p->pp_vae_path       = NULL;
}

AceSynth * ace_synth_load(ModelStore * store, const AceSynthParams * params) {
    if (!store || !params) {
        fprintf(stderr, "[Synth-Load] ERROR: store and params are required\n");
        return NULL;
    }
    if (!params->dit_path) {
        fprintf(stderr, "[Synth-Load] ERROR: dit_path is NULL\n");
        return NULL;
    }
    bool is_onnx_dit = dit_ends_with_onnx(params->dit_path);
    if (is_onnx_dit && !synth_validate_onnx_artifacts(params->dit_path)) {
        return NULL;
    }

    bool have_text_path = params->text_encoder_path && params->text_encoder_path[0];
    bool text_encoder_is_onnx = have_text_path && dit_ends_with_onnx(params->text_encoder_path);
    bool have_text_gguf = have_text_path && !text_encoder_is_onnx;
    if (!have_text_gguf && !is_onnx_dit) {
        fprintf(stderr, "[Synth-Load] ERROR: text_encoder_path is NULL; a GGUF text encoder is required\n");
        return NULL;
    }

    bool have_vae_gguf = params->vae_path && params->vae_path[0];
    if (!have_vae_gguf) {
        fprintf(stderr, "[Synth-Load] ERROR: vae_path is NULL; a GGUF VAE decoder is required\n");
        return NULL;
    }

    AceSynth * ctx = new AceSynth();
    ctx->store     = store;
    ctx->params    = *params;

    bool use_gguf_aux_for_onnx = is_onnx_dit && params->aux_dit_path && params->aux_dit_path[0];
    if (use_gguf_aux_for_onnx) {
        fprintf(stderr, "[Synth-Load] ONNX DiT forward with GGUF DiT auxiliary submodels: %s\n",
                params->aux_dit_path);
    }

    // DiTMeta: config + silence_latent + null_condition_emb + is_turbo,
    // fetched once, valid for the store lifetime. For forward-only TRT DiT
    // mode, metadata follows the GGUF auxiliary source so silence/null
    // conditioning stays with the non-forward DiT components.
    const char * meta_path = use_gguf_aux_for_onnx ? params->aux_dit_path : params->dit_path;
    ctx->meta = store_dit_meta(store, meta_path);
    if (!ctx->meta) {
        fprintf(stderr, "[Synth-Load] FATAL: DiT metadata unavailable for %s\n", meta_path);
        delete ctx;
        return NULL;
    }
    ctx->Oc     = ctx->meta->cfg.out_channels;           // 64
    ctx->ctx_ch = ctx->meta->cfg.in_channels - ctx->Oc;  // 128

    // For ONNX DiT: sub-models (cond_enc, fsq_tok/detok) will be separate
    // ONNX files in the same directory as the DiT ONNX. Each model type is
    // self-contained — no cross-dependency on safetensors or GGUF.
    // For GGUF/SafeTensors: sub-models share the dit_path as before.
    std::string submodel_path = is_onnx_dit
        ? (use_gguf_aux_for_onnx ? std::string(params->aux_dit_path) : dit_sidecar_dir(params->dit_path))
        : std::string(params->dit_path);

    // ModelKeys. Each path identifies its GGUF; adapter info rides with the
    // DiT key because two DiTs with different adapters are distinct modules.
    ctx->text_enc_key.kind = MODEL_TEXT_ENC;
    ctx->text_enc_key.path = have_text_gguf ? params->text_encoder_path : "";
    if (is_onnx_dit && have_text_gguf) {
        fprintf(stderr, "[Synth-Load] ONNX DiT forward with GGUF TextEnc: %s\n", params->text_encoder_path);
    }

    ctx->cond_enc_key.kind = MODEL_COND_ENC;
    ctx->cond_enc_key.path = submodel_path;

    // FSQ tok/detok: ONNX DiT artifacts must be self-contained. Do not fall
    // back to a GGUF/safetensors DiT just to satisfy tokenization side data.
    // When cover mode or audio_codes need FSQ, the existing GGML FSQ runtime
    // reads a small fsq.safetensors sidecar from this artifact directory.
    std::string fsq_path = submodel_path;
    if (is_onnx_dit) {
        if (use_gguf_aux_for_onnx) {
            fprintf(stderr, "[Synth-Load] ONNX DiT: FSQ tokenizer/detokenizer sourced from GGUF auxiliary DiT %s\n",
                    fsq_path.c_str());
        } else {
            fprintf(stderr, "[Synth-Load] ONNX DiT: FSQ sidecar must live in %s; GGUF DiT fallback disabled\n",
                    fsq_path.c_str());
        }
    }

    ctx->fsq_tok_key.kind = MODEL_FSQ_TOK;
    ctx->fsq_tok_key.path = fsq_path;

    ctx->fsq_detok_key.kind = MODEL_FSQ_DETOK;
    ctx->fsq_detok_key.path = fsq_path;

    ctx->dit_key.kind                 = MODEL_DIT;
    ctx->dit_key.path                 = params->dit_path;
    ctx->dit_key.adapter_path         = params->adapter_path ? params->adapter_path : "";
    ctx->dit_key.adapter_scale        = params->adapter_scale;
    ctx->dit_key.adapter_group_scales = g_hotstep_params.adapter_group_scales;

    ctx->vae_enc_key.kind = MODEL_VAE_ENC;
    ctx->vae_enc_key.path = have_vae_gguf ? params->vae_path : "";

    ctx->vae_dec_key.kind = MODEL_VAE_DEC;
    ctx->vae_dec_key.path = have_vae_gguf ? params->vae_path : "";

    // PP-VAE: optional post-processing VAE (GGML)
    ctx->have_pp_vae = false;
    if (params->pp_vae_path && params->pp_vae_path[0]) {
        ctx->pp_vae_enc_key.kind = MODEL_VAE_ENC;
        ctx->pp_vae_enc_key.path = params->pp_vae_path;
        ctx->pp_vae_dec_key.kind = MODEL_VAE_DEC;
        ctx->pp_vae_dec_key.path = params->pp_vae_path;
        ctx->have_pp_vae         = true;
        fprintf(stderr, "[Synth-Load] PP-VAE: %s\n", params->pp_vae_path);
    }

    // is_onnx_pipeline: true when FSQ/runtime side data comes from the ONNX
    // DiT artifact directory (TRT DiT forward without a GGUF auxiliary DiT).
    // Drives FSQ sidecar resolution in the synthesis ops.
    ctx->is_onnx_pipeline = is_onnx_dit && !use_gguf_aux_for_onnx;

    // Native TRT bundle detection + all-three-or-nothing gate (partner §3.5/§5.7).
    // The selected ONNX DiT directory is the bundle dir; a manifest.json there is
    // the sole TRT-bundle detector. When a manifest is present, every component
    // (dit, text_enc, cond_enc) MUST be declared with an existing engine on disk:
    // trt_bundle_load_manifest enforces this and fails closed. We HARD FAIL the
    // load on an incomplete manifest rather than silently routing encoders to
    // GGUF. When no manifest is present, this is a GGML/ORT pipeline as before.
    if (ctx->is_onnx_pipeline) {
        std::string bundle_dir = dit_sidecar_dir(params->dit_path);
        if (trt_bundle_has_manifest(bundle_dir)) {
            std::string manifest_err;
            if (!trt_bundle_load_manifest(bundle_dir, &ctx->trt_manifest, &manifest_err)) {
                fprintf(stderr,
                        "[Synth-Load] FATAL: TRT bundle manifest at %s is incomplete; "
                        "all-three-or-nothing requires dit+text_enc+cond_enc engines: %s\n",
                        bundle_dir.c_str(), manifest_err.c_str());
                delete ctx;
                return NULL;
            }
            ctx->use_trt_bundle = true;

            ctx->text_enc_trt_key.kind = MODEL_TEXT_ENC_TRT;
            ctx->text_enc_trt_key.path = ctx->trt_manifest.text_enc.engine;
            ctx->cond_enc_trt_key.kind = MODEL_COND_ENC_TRT;
            ctx->cond_enc_trt_key.path = ctx->trt_manifest.cond_enc.engine;

            // Group eviction: evict any previously-resident bundle's engines
            // before this bundle loads, so two bundles never co-reside.
            store_prune_bundle_except(store, bundle_dir);

            fprintf(stderr,
                    "[Synth-Load] TRT bundle: version=%s source=%s variant=%s\n"
                    "[Synth-Load]   dit      = %s\n"
                    "[Synth-Load]   text_enc = %s\n"
                    "[Synth-Load]   cond_enc = %s\n",
                    ctx->trt_manifest.version.c_str(), ctx->trt_manifest.source_model.c_str(),
                    ctx->trt_manifest.variant.c_str(), ctx->trt_manifest.dit.engine.c_str(),
                    ctx->trt_manifest.text_enc.engine.c_str(), ctx->trt_manifest.cond_enc.engine.c_str());
        }
    }

    if (!ctx->use_trt_bundle) {
        // Switching to a GGML/GGUF pipeline: any TRT bundle that was resident
        // from a prior request must be evicted before the GGUF DiT loads, so the
        // two never co-reside on the GPU. evict_all_except keeps the MODEL_DIT_TRT
        // resident shell across a require for a different kind, and the TRT
        // encoders' release retains their zero-byte shells, so a plain GGUF
        // store_require_dit would NOT free the bundle's engine VRAM. Pruning with
        // an empty keep evicts every resident bundle engine (DiT/TextEnc/CondEnc)
        // while leaving GGML/GGUF modules untouched.
        store_prune_bundle_except(store, "");
    }

    fprintf(stderr, "[Synth-Load] Ready: turbo=%s, merge=%s, fa=%s, batch_cfg=%s\n",
            ctx->meta->is_turbo ? "yes" : "no",
            ctx->meta->is_merge ? "yes" : "no",
            params->use_fa ? "yes" : "no", params->use_batch_cfg ? "yes" : "no");
    if (params->clamp_fp16) {
        fprintf(stderr, "[Synth-Load] FP16 clamp enabled\n");
    }
    if (params->adapter_path) {
        fprintf(stderr, "[Synth-Load] Adapter: %s (scale=%.2f)\n", params->adapter_path, params->adapter_scale);
    }

    return ctx;
}

// Allocate job and init the SynthState fields every task poses the same way.
static AceSynthJob * alloc_job(AceSynth * ctx, const AceRequest * reqs, int batch_n) {
    AceSynthJob * job = new AceSynthJob();
    job->batch_n      = batch_n;
    SynthState & s    = job->state;
    s.Oc              = ctx->Oc;
    s.ctx_ch          = ctx->ctx_ch;
    s.left_pad_sec    = 0.0f;
    s.rr              = reqs[0];
    s.rs              = s.rr.repainting_start;
    s.re              = s.rr.repainting_end;
    s.use_sde         = (s.rr.infer_method == INFER_SDE);
    s.is_repaint      = false;
    s.is_lego_region  = false;
    s.have_cover      = false;
    s.T_cover         = 0;
    debug_init(&s.dbg, ctx->params.dump_dir);
    return job;
}

// Outpainting: pad src_audio with silence when the region extends beyond
// source bounds. Returns the (possibly padded) buffer and length via out
// Pad source for outpainting when the region extends beyond the source bounds.
// Audio path: zero-pads s.padded_src in samples, the VAE encoder will turn
// silence audio into ~silence latents.
// Latent path: pads s.cover_latents directly with the precomputed silence_full
// latent (the canonical VAE encoding of audio silence). No VAE encode needed.
// Sets s.left_pad_sec for adjust_region_coords.
static void apply_outpainting_padding(const AceSynth *   ctx,
                                      const AceRequest & r,
                                      const float *      src_audio,
                                      int                src_len,
                                      const float *      src_latents,
                                      int                src_T_latent,
                                      SynthState &       s,
                                      const float *&     enc_audio,
                                      int &              enc_len,
                                      const float *&     enc_latents,
                                      int &              enc_T_latent) {
    enc_audio    = src_audio;
    enc_len      = src_len;
    enc_latents  = src_latents;
    enc_T_latent = src_T_latent;

    float src_dur =
        (src_latents && src_T_latent > 0) ? (float) src_T_latent * 1920.0f / 48000.0f : (float) src_len / 48000.0f;
    float rs_raw   = r.repainting_start;
    float re_raw   = r.repainting_end;
    float end_time = (re_raw < 0.0f) ? src_dur : re_raw;
    float lpad     = (rs_raw < 0.0f) ? -rs_raw : 0.0f;
    float rpad     = (end_time > src_dur) ? end_time - src_dur : 0.0f;
    s.left_pad_sec = lpad;

    if (src_latents && src_T_latent > 0) {
        if (lpad <= 0.0f && rpad <= 0.0f) {
            return;
        }
        // Latent pad: round to 25Hz frames, splice silence_full at the edges.
        int           lpad_T  = (int) (lpad * 25.0f + 0.5f);
        int           rpad_T  = (int) (rpad * 25.0f + 0.5f);
        int           total_T = lpad_T + src_T_latent + rpad_T;
        const float * sil     = ctx->meta->silence_full.data();
        s.padded_latents.resize((size_t) total_T * 64);
        memcpy(s.padded_latents.data(), sil, (size_t) lpad_T * 64 * sizeof(float));
        memcpy(s.padded_latents.data() + (size_t) lpad_T * 64, src_latents, (size_t) src_T_latent * 64 * sizeof(float));
        memcpy(s.padded_latents.data() + (size_t) (lpad_T + src_T_latent) * 64, sil,
               (size_t) rpad_T * 64 * sizeof(float));
        enc_latents  = s.padded_latents.data();
        enc_T_latent = total_T;
        fprintf(stderr, "[Outpaint] latent pad left=%.1fs (%d) right=%.1fs (%d) total=%.1fs\n", lpad, lpad_T, rpad,
                rpad_T, (float) total_T * 1920.0f / 48000.0f);
        return;
    }

    // Audio path: always populate s.padded_src so the post-decode waveform
    // splice always has the source PCM to read from, with or without outpaint.
    int lpad_s       = (int) (lpad * 48000.0f);
    int rpad_s       = (int) (rpad * 48000.0f);
    int padded_total = src_len + lpad_s + rpad_s;
    s.padded_src.resize((size_t) padded_total * 2);
    if (lpad_s > 0 || rpad_s > 0) {
        memset(s.padded_src.data(), 0, s.padded_src.size() * sizeof(float));
    }
    memcpy(s.padded_src.data() + (size_t) lpad_s * 2, src_audio, (size_t) src_len * 2 * sizeof(float));
    enc_audio = s.padded_src.data();
    enc_len   = padded_total;
    if (lpad > 0.0f || rpad > 0.0f) {
        fprintf(stderr, "[Outpaint] audio pad left=%.1fs right=%.1fs total=%.1fs\n", lpad, rpad,
                (float) padded_total / 48000.0f);
    }
}

// Shift region coords into the padded reference frame, resolve sentinel end
// (-1) to either left pad boundary (outpaint) or source end (inpaint).
// src_dur is the unpadded source duration in seconds, agnostic of audio vs
// latent input. Returns false when the resolved range is empty or inverted.
static bool adjust_region_coords(SynthState & s, float src_dur) {
    s.rs += s.left_pad_sec;
    if (s.re < 0.0f) {
        s.re = (s.rr.repainting_start < 0.0f) ? s.left_pad_sec : src_dur + s.left_pad_sec;
    } else {
        s.re += s.left_pad_sec;
    }
    if (s.re <= s.rs) {
        fprintf(stderr, "[Region] ERROR: end (%.1f) <= start (%.1f)\n", s.re, s.rs);
        return false;
    }
    fprintf(stderr, "[Region] %.1fs..%.1fs (canvas=%.1fs)\n", s.rs, s.re, (float) s.T_cover * 1920.0f / 48000.0f);
    return true;
}

// Uppercase track name for instruction templates, warn on unknown names.
static std::string prepare_track(const std::string & track, const char * label) {
    std::string upper = track;
    for (char & ch : upper) {
        ch = (char) toupper((unsigned char) ch);
    }
    validate_track_names(track, label);
    return upper;
}

// Warn when a stem task runs on a turbo model: the training objective does
// not cover stem isolation, output degrades to incoherent noise.
static void warn_if_turbo_stem(const AceSynth * ctx, const char * task_name) {
    if (ctx->meta->is_turbo) {
        fprintf(stderr, "[Synth-Run] WARNING: %s requires base model, turbo output incoherent\n", task_name);
    }
}

// Pin VAE-Enc across two back-to-back encodes (source then timbre) so STRICT
// does not unload and reload the 160 MB weights between them. On return the
// pin is released, VAE-Enc can be evicted by the next require. The pin is
// only justified when at least one side actually goes through the encoder:
// when both src and ref arrive as pre-encoded latents, no pin is needed.
static bool pinned_encode_src_and_timbre(AceSynth *    ctx,
                                         const float * src_audio,
                                         int           src_len,
                                         const float * src_latents,
                                         int           src_T_latent,
                                         const float * ref_audio,
                                         int           ref_len,
                                         const float * ref_latents,
                                         int           ref_T_latent,
                                         SynthState &  s) {
    bool have_src_audio   = src_audio && src_len > 0;
    bool have_src_latents = src_latents && src_T_latent > 0;
    bool have_src         = have_src_audio || have_src_latents;
    bool have_ref_audio   = ref_audio && ref_len > 0;
    bool have_ref_latents = ref_latents && ref_T_latent > 0;
    bool have_ref         = have_ref_audio || have_ref_latents;
    if (!have_src && !have_ref) {
        // Neither encode touches the GPU: timbre takes the silence path.
        ops_encode_timbre(ctx, NULL, 0, NULL, 0, s);
        return true;
    }
    bool        need_vae_pin = (have_src_audio || have_ref_audio);
    ModelHandle vae_pin(ctx->store, need_vae_pin ? store_require_vae_enc(ctx->store, ctx->vae_enc_key) : NULL);
    if (need_vae_pin && !vae_pin.ptr) {
        fprintf(stderr, "[Pipeline-Synth] FATAL: store_require_vae_enc (pin) failed\n");
        return false;
    }
    if (have_src) {
        if (ops_encode_src(ctx, src_audio, src_len, src_latents, src_T_latent, s) != 0) {
            return false;
        }
    }
    ops_encode_timbre(ctx, ref_audio, ref_len, ref_latents, ref_T_latent, s);
    return true;
}

// ─── Determinism diagnostic ─────────────────────────────────────────────────
// Print per-buffer statistics for comparing identical generation runs.
// The exact sum is sensitive to single-bit differences, so if the sum
// matches across runs, the buffers are bit-identical.
static void diag_stats_f32(const char * label, const float * data, size_t n) {
    if (!data || n == 0) { return; }
    double sum = 0.0, sum_sq = 0.0;
    float  mn = data[0], mx = data[0];
    for (size_t i = 0; i < n; i++) {
        double v = (double) data[i];
        sum += v;
        sum_sq += v * v;
        if (data[i] < mn) { mn = data[i]; }
        if (data[i] > mx) { mx = data[i]; }
    }
    double mean = sum / (double) n;
    double rms  = sqrt(sum_sq / (double) n);
    fprintf(stderr, "[DIAG] %s: n=%zu mean=%.8f rms=%.8f min=%.6f max=%.6f sum=%.10f\n",
            label, n, mean, rms, mn, mx, sum);
}

static void synth_log_phase_memory(const AceSynth * ctx, const char * phase) {
    if (!ctx || !ctx->store) {
        return;
    }
    size_t bytes = store_vram_bytes(ctx->store);
    int    count = store_gpu_module_count(ctx->store);
    fprintf(stderr, "[Phase-Memory] %-22s policy=%s gpu_modules=%d tracked_weights=%.1f MB\n",
            phase,
            store_get_policy(ctx->store) == EVICT_STRICT ? "STRICT" : "NEVER",
            count,
            (double) bytes / (1024.0 * 1024.0));
}

// Common tail every task ends with once its inputs are encoded and flags are
// posed: resolve params, resolve T, build schedule, encode text, build
// context, init noise, run DiT. Returns 0 on success, -1 on error/cancel.
static int run_tail(AceSynth *         ctx,
                    const AceRequest * reqs,
                    int                batch_n,
                    SynthState &       s,
                    bool (*cancel)(void *),
                    void * cancel_data) {
    if (ops_resolve_params(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }
    if (ops_resolve_T(ctx, s) != 0) {
        return -1;
    }
    ops_build_schedule(s);
    synth_log_phase_memory(ctx, "tail-start");
    if (ops_encode_text(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }
    synth_log_phase_memory(ctx, "after-text-cond");
    diag_stats_f32("enc_hidden", s.enc_hidden.data(), s.enc_hidden.size());

    if (ops_build_context(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }
    synth_log_phase_memory(ctx, "after-context");
    diag_stats_f32("context", s.context.data(), s.context.size());

    ops_build_context_silence(ctx, batch_n, s);
    ops_init_noise(ctx, reqs, batch_n, s);
    diag_stats_f32("noise", s.noise.data(), s.noise.size());

    synth_log_phase_memory(ctx, "before-dit");
    if (ops_dit_generate(ctx, batch_n, s, cancel, cancel_data) != 0) {
        synth_log_phase_memory(ctx, "after-dit-error");
        return -1;
    }
    synth_log_phase_memory(ctx, "after-dit");
    diag_stats_f32("dit_output", s.output.data(), s.output.size());

    return 0;
}

// text2music: pure generation. No src audio. Optional timbre reference.
// Audio codes from LM (or absent) condition the DiT context.
static AceSynthJob * run_text2music(AceSynth *         ctx,
                                    const AceRequest * reqs,
                                    const float *      ref_audio,
                                    int                ref_len,
                                    const float *      ref_latents,
                                    int                ref_T_latent,
                                    int                batch_n,
                                    bool (*cancel)(void *),
                                    void * cancel_data) {
    AceSynthJob * job    = alloc_job(ctx, reqs, batch_n);
    SynthState &  s      = job->state;
    // audio_codes from the LM produce a latent context; empty codes fall back
    // to silence (DiT-only). The DiT was trained with the cover instruction on
    // latent context, so the two flags cascade from the same condition.
    s.use_source_context = !reqs[0].audio_codes.empty();
    s.instruction_str    = s.use_source_context ? DIT_INSTR_COVER : DIT_INSTR_TEXT2MUSIC;

    if (!pinned_encode_src_and_timbre(ctx, NULL, 0, NULL, 0, ref_audio, ref_len, ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// cover: src audio recomposed with FSQ roundtrip degrading the context, so
// the DiT diverges from the original while staying thematically aligned.
static AceSynthJob * run_cover(AceSynth *         ctx,
                               const AceRequest * reqs,
                               const float *      src_audio,
                               int                src_len,
                               const float *      src_latents,
                               int                src_T_latent,
                               const float *      ref_audio,
                               int                ref_len,
                               const float *      ref_latents,
                               int                ref_T_latent,
                               int                batch_n,
                               bool (*cancel)(void *),
                               void * cancel_data) {
    bool have_src = (src_audio && src_len > 0) || (src_latents && src_T_latent > 0);
    if (!have_src) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'cover' requires source audio or latents\n");
        return NULL;
    }
    AceSynthJob * job    = alloc_job(ctx, reqs, batch_n);
    SynthState &  s      = job->state;
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_COVER;

    if (!pinned_encode_src_and_timbre(ctx, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }

    // Snapshot clean VAE latents before the FSQ roundtrip degrades them.
    // cover_noise_strength blending needs the clean copy.
    s.noise_blend_latents = s.cover_latents;
    if (ops_fsq_roundtrip(ctx, s) != 0) {
        delete job;
        return NULL;
    }

    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// cover-nofsq: cover without the FSQ roundtrip. DiT works on clean 25Hz VAE
// latents, output stays close to source structure and timbre. Pass
// ref_audio = src_audio for best results.
static AceSynthJob * run_cover_nofsq(AceSynth *         ctx,
                                     const AceRequest * reqs,
                                     const float *      src_audio,
                                     int                src_len,
                                     const float *      src_latents,
                                     int                src_T_latent,
                                     const float *      ref_audio,
                                     int                ref_len,
                                     const float *      ref_latents,
                                     int                ref_T_latent,
                                     int                batch_n,
                                     bool (*cancel)(void *),
                                     void * cancel_data) {
    bool have_src = (src_audio && src_len > 0) || (src_latents && src_T_latent > 0);
    if (!have_src) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'cover-nofsq' requires source audio or latents\n");
        return NULL;
    }
    AceSynthJob * job    = alloc_job(ctx, reqs, batch_n);
    SynthState &  s      = job->state;
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_COVER;

    if (!pinned_encode_src_and_timbre(ctx, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// repaint: region-bounded inpaint/outpaint. Source padded with silence at the
// edges when the region extends beyond bounds, audio-side or latent-side
// depending on what the caller provided. DiT regenerates inside the region,
// phase 2 splices the result with the source latents and decodes once.
static AceSynthJob * run_repaint(AceSynth *         ctx,
                                 const AceRequest * reqs,
                                 const float *      src_audio,
                                 int                src_len,
                                 const float *      src_latents,
                                 int                src_T_latent,
                                 const float *      ref_audio,
                                 int                ref_len,
                                 const float *      ref_latents,
                                 int                ref_T_latent,
                                 int                batch_n,
                                 bool (*cancel)(void *),
                                 void * cancel_data) {
    bool have_audio   = src_audio && src_len > 0;
    bool have_latents = src_latents && src_T_latent > 0;
    if (!have_audio && !have_latents) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'repaint' requires source audio or src_latents\n");
        return NULL;
    }
    AceSynthJob * job    = alloc_job(ctx, reqs, batch_n);
    SynthState &  s      = job->state;
    s.is_repaint         = true;
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_REPAINT;

    const float * enc_audio    = NULL;
    int           enc_len      = 0;
    const float * enc_latents  = NULL;
    int           enc_T_latent = 0;
    apply_outpainting_padding(ctx, reqs[0], src_audio, src_len, src_latents, src_T_latent, s, enc_audio, enc_len,
                              enc_latents, enc_T_latent);

    if (!pinned_encode_src_and_timbre(ctx, enc_audio, enc_len, enc_latents, enc_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    float src_dur = have_latents ? (float) src_T_latent * 1920.0f / 48000.0f : (float) src_len / 48000.0f;
    if (!adjust_region_coords(s, src_dur)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// lego: stem generation. With valid rs/re: region-constrained, DiT generates
// only in the zone with full audio context. Without: whole-song generation.
// audio_cover_strength forced to 1.0 so all DiT steps hear the backing track.
// Both src_audio and src_latents are accepted; in region mode the source is
// padded with silence at the edges when the region extends beyond bounds.
static AceSynthJob * run_lego(AceSynth *         ctx,
                              const AceRequest * reqs,
                              const float *      src_audio,
                              int                src_len,
                              const float *      src_latents,
                              int                src_T_latent,
                              const float *      ref_audio,
                              int                ref_len,
                              const float *      ref_latents,
                              int                ref_T_latent,
                              int                batch_n,
                              bool (*cancel)(void *),
                              void * cancel_data) {
    bool have_audio   = src_audio && src_len > 0;
    bool have_latents = src_latents && src_T_latent > 0;
    if (!have_audio && !have_latents) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'lego' requires source audio or src_latents\n");
        return NULL;
    }
    AceSynthJob * job       = alloc_job(ctx, reqs, batch_n);
    SynthState &  s         = job->state;
    s.is_lego_region        = (s.rr.repainting_end > s.rr.repainting_start);
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Lego");
    s.instruction_str       = dit_instr_lego(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "lego");

    const float * enc_audio    = src_audio;
    int           enc_len      = src_len;
    const float * enc_latents  = src_latents;
    int           enc_T_latent = src_T_latent;
    if (s.is_lego_region) {
        apply_outpainting_padding(ctx, reqs[0], src_audio, src_len, src_latents, src_T_latent, s, enc_audio, enc_len,
                                  enc_latents, enc_T_latent);
    }

    if (!pinned_encode_src_and_timbre(ctx, enc_audio, enc_len, enc_latents, enc_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    float src_dur = have_latents ? (float) src_T_latent * 1920.0f / 48000.0f : (float) src_len / 48000.0f;
    if (s.is_lego_region && !adjust_region_coords(s, src_dur)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// extract: stem isolation from a full mix.
static AceSynthJob * run_extract(AceSynth *         ctx,
                                 const AceRequest * reqs,
                                 const float *      src_audio,
                                 int                src_len,
                                 const float *      src_latents,
                                 int                src_T_latent,
                                 const float *      ref_audio,
                                 int                ref_len,
                                 const float *      ref_latents,
                                 int                ref_T_latent,
                                 int                batch_n,
                                 bool (*cancel)(void *),
                                 void * cancel_data) {
    bool have_src = (src_audio && src_len > 0) || (src_latents && src_T_latent > 0);
    if (!have_src) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'extract' requires source audio or latents\n");
        return NULL;
    }
    AceSynthJob * job       = alloc_job(ctx, reqs, batch_n);
    SynthState &  s         = job->state;
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Extract");
    s.instruction_str       = dit_instr_extract(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "extract");

    if (!pinned_encode_src_and_timbre(ctx, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// complete: extend an isolated stem with more content.
static AceSynthJob * run_complete(AceSynth *         ctx,
                                  const AceRequest * reqs,
                                  const float *      src_audio,
                                  int                src_len,
                                  const float *      src_latents,
                                  int                src_T_latent,
                                  const float *      ref_audio,
                                  int                ref_len,
                                  const float *      ref_latents,
                                  int                ref_T_latent,
                                  int                batch_n,
                                  bool (*cancel)(void *),
                                  void * cancel_data) {
    bool have_src = (src_audio && src_len > 0) || (src_latents && src_T_latent > 0);
    if (!have_src) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'complete' requires source audio or latents\n");
        return NULL;
    }
    AceSynthJob * job       = alloc_job(ctx, reqs, batch_n);
    SynthState &  s         = job->state;
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Complete");
    s.instruction_str       = dit_instr_complete(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "complete");

    if (!pinned_encode_src_and_timbre(ctx, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// Phase 1 entry point. Dispatches on reqs[0].task_type to the right task
// function. task_type is always set: request_init defaults it to text2music,
// the JSON parser ignores empty strings.
AceSynthJob * ace_synth_job_run_dit(AceSynth *         ctx,
                                    const AceRequest * reqs,
                                    const float *      src_audio,
                                    int                src_len,
                                    const float *      src_latents,
                                    int                src_T_latent,
                                    const float *      ref_audio,
                                    int                ref_len,
                                    const float *      ref_latents,
                                    int                ref_T_latent,
                                    int                batch_n,
                                    bool (*cancel)(void *),
                                    void * cancel_data) {
    if (!ctx || !reqs || batch_n < 1 || batch_n > 9) {
        return NULL;
    }
    const std::string & task = reqs[0].task_type;
    if (task == TASK_TEXT2MUSIC) {
        return run_text2music(ctx, reqs, ref_audio, ref_len, ref_latents, ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_COVER) {
        return run_cover(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len, ref_latents,
                         ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_COVER_NOFSQ) {
        return run_cover_nofsq(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len,
                               ref_latents, ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_REPAINT) {
        return run_repaint(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len, ref_latents,
                           ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_LEGO) {
        return run_lego(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len, ref_latents,
                        ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_EXTRACT) {
        return run_extract(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len, ref_latents,
                           ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_COMPLETE) {
        return run_complete(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len, ref_latents,
                            ref_T_latent, batch_n, cancel, cancel_data);
    }
    fprintf(stderr, "[Synth-Run] ERROR: unknown task_type '%s'\n", task.c_str());
    return NULL;
}

// Pointer to the post-DiT latent for one track in s.output, time-major
// [T*64] f32. Lives until ace_synth_job_free.
const float * ace_synth_job_get_latent(const AceSynthJob * job, int track_idx, int * T_out) {
    const SynthState & s = job->state;
    *T_out               = s.T;
    return s.output.data() + (size_t) track_idx * s.T * s.Oc;
}

// Phase 2: latent splice (for repaint/lego) + VAE decode for every batch item.
int ace_synth_job_run_vae(AceSynth *    ctx,
                          AceSynthJob * job,
                          AceAudio *    out,
                          bool (*cancel)(void *),
                          void * cancel_data) {
    if (!ctx || !job || !out) {
        return -1;
    }

    // Route through postprocess plugin if one is selected
    synth_log_phase_memory(ctx, "before-vae");
    const std::string & pp = job->state.rr.postprocess_plugin;
    int rc;
    if (!pp.empty()) {
        rc = ops_vae_decode_postprocess(ctx, job->batch_n, out, job->state,
                                        pp.c_str(), cancel, cancel_data);
    } else {
        rc = ops_vae_decode(ctx, job->batch_n, out, job->state, cancel, cancel_data);
    }
    synth_log_phase_memory(ctx, rc == 0 ? "after-vae" : "after-vae-error");
    return rc;
}

// ── LRC core: run alignment with an already-held DiT ─────────────────────
// Called from ops_dit_generate while the DiT is still in scope, so no
// redundant acquire/merge. Results stored in s.lrc_results[].
int ops_lrc_extract(const AceSynth * ctx, DiTGGML * dit, int batch_n, SynthState & s) {
    s.lrc_done = true;
    s.lrc_results.assign(batch_n, std::string());

    if (!s.get_lrc) {
        fprintf(stderr, "[LRC] Skipped: get_lrc=false\n");
        return 0;
    }
    int pure_n = s.lyric_end_idx - s.lyric_start_idx;
    if (pure_n <= 0 || s.lyric_token_ids.empty()) {
        fprintf(stderr, "[LRC] Skipped: no lyric tokens (pure_n=%d)\n", pure_n);
        return 0;
    }

    Timer lrc_timer;

    // Resolve alignment config from model metadata
    AlignmentConfig align_cfg = alignment_config_resolve(
        "",  // TODO: read from DiTMeta when GGUF field is available
        ctx->meta->cfg.n_layers,
        ctx->meta->cfg.n_heads);
    if (!align_cfg.valid) {
        fprintf(stderr, "[LRC] WARNING: no valid alignment config, skipping\n");
        return 0;
    }

    // Force f32 attention for alignment (no flash_attn)
    bool saved_fa = dit->use_flash_attn;
    dit->use_flash_attn = false;

    // Extract attention scores (batch 0 only — LRC is per-song, not per-variation)
    int total_scores = align_cfg.total_heads * s.enc_S * s.S;
    std::vector<float> scores(total_scores);

    const int align_steps = 8;

    int rc = dit_alignment_extract(
        dit, align_cfg,
        s.output.data(),
        s.context.data(),
        s.enc_hidden.data(),
        s.T, s.S, s.enc_S,
        align_steps,
        scores.data());

    dit->use_flash_attn = saved_fa;

    if (rc != 0) {
        fprintf(stderr, "[LRC] WARNING: alignment extraction failed\n");
        return 0;
    }

    // Transpose scores from GGML column-major [enc_S, S] to C row-major [pure_n, S]
    std::vector<float> sliced_scores(align_cfg.total_heads * pure_n * s.S);
    for (int h = 0; h < align_cfg.total_heads; h++) {
        const float * src = scores.data() + h * s.enc_S * s.S;
        float *       dst = sliced_scores.data() + h * pure_n * s.S;
        for (int t = 0; t < pure_n; t++) {
            int enc_idx = s.lyric_start_idx + t;
            for (int f = 0; f < s.S; f++) {
                dst[t * s.S + f] = src[f * s.enc_S + enc_idx];
            }
        }
    }

    // Build pure lyric token IDs and texts
    std::vector<int> pure_ids(s.lyric_token_ids.begin() + s.lyric_start_idx,
                               s.lyric_token_ids.begin() + s.lyric_end_idx);

    // Run LRC alignment
    LrcResult result = lrc_align(
        sliced_scores.data(),
        align_cfg.total_heads, pure_n, s.S,
        pure_ids, s.lyric_token_texts,
        s.duration,
        2.0f,  // violence_level
        1);    // medfilt_width

    if (result.success && !result.lrc_text.empty()) {
        for (int b = 0; b < batch_n; b++) {
            s.lrc_results[b] = result.lrc_text;
        }
        fprintf(stderr, "[LRC] Generated %zu bytes in %.1f ms\n",
                result.lrc_text.size(), lrc_timer.ms());
    } else {
        fprintf(stderr, "[LRC] Alignment %s: %s (%.1f ms)\n",
                result.success ? "empty" : "failed",
                result.error.c_str(), lrc_timer.ms());
    }

    return 0;
}

// Phase 3: LRC timestamp generation (thin wrapper).
// If ops_lrc_extract already ran during Phase 1 (inside ops_dit_generate),
// just copy the pre-computed results — no DiT re-acquisition needed.
int ace_synth_job_run_lrc(AceSynth *    ctx,
                          AceSynthJob * job,
                          std::string * lrc_out,
                          int           batch_n) {
    if (!ctx || !job || !lrc_out) {
        return -1;
    }

    SynthState & s = job->state;

    // Initialize all outputs to empty
    for (int b = 0; b < batch_n; b++) {
        lrc_out[b].clear();
    }

    // If LRC was already computed inline during DiT phase, just copy results
    if (s.lrc_done) {
        int n = (int) s.lrc_results.size();
        if (n > batch_n) n = batch_n;
        for (int b = 0; b < n; b++) {
            lrc_out[b] = s.lrc_results[b];
        }
        return 0;
    }

    // Fallback: compute now (acquires DiT — will trigger adapter reload)
    fprintf(stderr, "[LRC] WARNING: running standalone LRC pass (DiT will be re-acquired)\n");

    if (!s.get_lrc) {
        return 0;
    }

    DiTGGML * dit = store_require_dit(ctx->store, ctx->dit_key);
    if (!dit) {
        fprintf(stderr, "[LRC] WARNING: store_require_dit failed, skipping\n");
        return 0;
    }
    ModelHandle dit_guard(ctx->store, dit);

    return ops_lrc_extract(ctx, dit, batch_n, s);
}

void ace_synth_job_free(AceSynthJob * job) {
    delete job;
}

void ace_audio_free(AceAudio * audio) {
    if (audio && audio->samples) {
        free(audio->samples);
        audio->samples   = NULL;
        audio->n_samples = 0;
    }
}

void ace_synth_free(AceSynth * ctx) {
    if (!ctx) {
        return;
    }
    delete ctx;
}
