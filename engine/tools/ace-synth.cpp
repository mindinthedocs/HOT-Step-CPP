// ace-synth.cpp: ACE-Step synthesis CLI
// Thin wrapper: parses args, scans the model registry, calls pipeline-synth,
// writes output files. Model selection (synth_model, adapter, output_format)
// comes from the request JSON. The registry resolves names to GGUF,
// SafeTensors, or ONNX runtime-bundle artifacts under --models <dir> and
// adapter artifacts under --adapters <dir>.

#include "audio-io.h"
#include "lua-plugin-registry.h"
#include "model-registry.h"
#include "model-store.h"
#include "pipeline-synth.h"
#include "request.h"
#include "synth-batch-runner.h"
#include "task-types.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    AceSynthParams d;
    ace_synth_default_params(&d);

    fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
    fprintf(stderr,
            "Usage: %s --models <dir> --request <json...> [options]\n\n"
            "Required:\n"
            "  --models <dir>          Directory of GGUF, SafeTensors, or ONNX runtime-bundle artifacts\n"
            "  --request <json...>     One or more request JSONs (from ace-lm --request)\n\n"
            "Optional:\n"
            "  --adapters <dir>        Directory of adapter files (enables JSON adapter field)\n"
            "  --src-audio <file>      Source audio (WAV or MP3)\n"
            "  --ref-audio <file>      Timbre reference audio (WAV or MP3)\n\n"
            "Model selection comes from the request JSON: synth_model picks the DiT,\n"
            "adapter picks an adapter from --adapters, output_format picks the output\n"
            "extension. When synth_model is empty the first DiT in the registry is used.\n"
            "ONNX DiT bundles can carry sibling text/condition/VAE artifacts;\n"
            "GGUF/SafeTensors DiT entries use the first GGUF/SafeTensors text-encoder and VAE entries.\n\n"
            "Audio encoding:\n"
            "  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)\n\n"
            "Memory control:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: %d)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: %d)\n\n"
            "Debug:\n"
            "  --gguf-text-encoder    Use registry GGUF Text-Enc even when synth_model is ONNX/TRT\n"
            "  --gguf-dit-aux         Use registry GGUF DiT for CondEnc/FSQ side data with ONNX/TRT DiT forward\n"
            "  --no-fa                 Disable flash attention\n"
            "  --no-batch-cfg          Split DiT CFG into two separate forwards\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --dump <dir>            Dump intermediate tensors\n",
            prog, d.vae_chunk, d.vae_overlap);
}

static const ModelEntry * first_non_onnx(const std::vector<ModelEntry> & bucket) {
    for (const auto & entry : bucket) {
        if (!str_ends_with(entry.path, ".onnx")) {
            return &entry;
        }
    }
    return nullptr;
}

static const ModelEntry * first_onnx(const std::vector<ModelEntry> & bucket) {
    for (const auto & entry : bucket) {
        if (str_ends_with(entry.path, ".onnx")) {
            return &entry;
        }
    }
    return nullptr;
}

static bool cli_file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

static std::string cli_dirname(const std::string & path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return path.substr(0, 1);
    }
    return path.substr(0, pos);
}

static bool onnx_dit_has_sibling_vae(const ModelEntry * entry) {
    if (!entry || !str_ends_with(entry->path, ".onnx")) {
        return false;
    }
    return cli_file_exists(cli_dirname(entry->path) + REGISTRY_SEP + "vae_decoder.onnx");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    // Defaults live in ace_synth_default_params. CLI locals read from params
    // so there is exactly one place in the codebase that picks the numbers.
    AceSynthParams params;
    ace_synth_default_params(&params);

    std::vector<const char *> request_paths;
    const char *              models_dir     = NULL;
    const char *              adapters_dir   = NULL;
    const char *              src_audio_path = NULL;
    const char *              ref_audio_path = NULL;
    const char *              dump_dir       = NULL;
    bool                      use_fa         = true;
    bool                      use_batch_cfg  = true;
    bool                      clamp_fp16     = false;
    bool                      gguf_text_enc  = false;
    bool                      gguf_dit_aux   = false;
    int                       vae_chunk      = params.vae_chunk;
    int                       vae_overlap    = params.vae_overlap;
    int                       mp3_kbps       = 128;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--request")) {
            // Collect all following non-option args
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                request_paths.push_back(argv[++i]);
            }
        } else if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!strcmp(argv[i], "--adapters") && i + 1 < argc) {
            adapters_dir = argv[++i];
        } else if (!strcmp(argv[i], "--src-audio") && i + 1 < argc) {
            src_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--ref-audio") && i + 1 < argc) {
            ref_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (!strcmp(argv[i], "--gguf-text-encoder")) {
            gguf_text_enc = true;
        } else if (!strcmp(argv[i], "--gguf-dit-aux")) {
            gguf_dit_aux = true;
        } else if (!strcmp(argv[i], "--no-fa")) {
            use_fa = false;
        } else if (!strcmp(argv[i], "--no-batch-cfg")) {
            use_batch_cfg = false;
        } else if (!strcmp(argv[i], "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!strcmp(argv[i], "--vae-chunk") && i + 1 < argc) {
            vae_chunk = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-overlap") && i + 1 < argc) {
            vae_overlap = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--mp3-bitrate") && i + 1 < argc) {
            mp3_kbps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!models_dir) {
        fprintf(stderr, "[CLI] ERROR: --models required\n");
        usage(argv[0]);
        return 1;
    }
    if (request_paths.empty()) {
        fprintf(stderr, "[CLI] ERROR: --request required\n");
        usage(argv[0]);
        return 1;
    }

    PluginRegistry::instance().init_from_executable(argv[0]);

    // Parse all requests first: the first request drives model selection.
    int                      batch_n = (int) request_paths.size();
    std::vector<AceRequest>  reqs(batch_n);
    std::vector<std::string> basenames(batch_n);
    for (int ri = 0; ri < batch_n; ri++) {
        const char * rpath = request_paths[ri];
        if (!request_parse(&reqs[ri], rpath)) {
            fprintf(stderr, "[Ace-Synth] FATAL: failed to parse %s\n", rpath);
            return 1;
        }
        request_dump(&reqs[ri], stderr);
        if (reqs[ri].caption.empty() && reqs[ri].task_type != TASK_LEGO && reqs[ri].task_type != TASK_EXTRACT &&
            reqs[ri].task_type != TASK_COMPLETE) {
            fprintf(stderr, "[Ace-Synth] FATAL: caption is empty in %s\n", rpath);
            return 1;
        }
        // output basename: strip .json suffix
        basenames[ri] = rpath;
        size_t dot    = basenames[ri].rfind(".json");
        if (dot != std::string::npos) {
            basenames[ri] = basenames[ri].substr(0, dot);
        }
    }
    fprintf(stderr, "[Ace-Synth] Batch: %d request(s)\n", batch_n);

    // Scan the registry and resolve model paths from the first request.
    ModelRegistry registry;
    if (!registry_scan(&registry, models_dir)) {
        fprintf(stderr, "[Ace-Synth] FATAL: cannot scan --models %s\n", models_dir);
        return 1;
    }
    if (adapters_dir) {
        registry_scan_adapters(&registry, adapters_dir);
    }
    if (registry.dit.empty()) {
        fprintf(stderr, "[Ace-Synth] FATAL: registry needs a DiT model\n");
        return 1;
    }
    const ModelEntry * dit_entry =
        reqs[0].synth_model.empty() ? &registry.dit[0] : registry_find(registry.dit, reqs[0].synth_model.c_str());
    if (!dit_entry) {
        fprintf(stderr, "[Ace-Synth] FATAL: synth_model '%s' not found in registry\n", reqs[0].synth_model.c_str());
        return 1;
    }
    const bool dit_is_onnx = str_ends_with(dit_entry->path, ".onnx");
    const ModelEntry * text_entry = (dit_is_onnx && !gguf_text_enc) ? first_onnx(registry.text_enc) : first_non_onnx(registry.text_enc);
    const ModelEntry * aux_dit_entry = (dit_is_onnx && gguf_dit_aux) ? first_non_onnx(registry.dit) : nullptr;
    if (dit_is_onnx && gguf_dit_aux && !aux_dit_entry) {
        fprintf(stderr, "[Ace-Synth] FATAL: --gguf-dit-aux requested but no GGUF/SafeTensors DiT was found\n");
        return 1;
    }
    if (dit_is_onnx && !gguf_text_enc && !text_entry) {
        fprintf(stderr, "[Ace-Synth] FATAL: selected ONNX DiT requires an ONNX text-encoder bundle\n");
        return 1;
    }
    const ModelEntry * vae_entry  = nullptr;
    if (!reqs[0].vae.empty()) {
        vae_entry = registry_find(registry.vae, reqs[0].vae.c_str());
        if (!vae_entry) {
            fprintf(stderr, "[Ace-Synth] FATAL: vae '%s' not found in registry\n", reqs[0].vae.c_str());
            return 1;
        }
    } else if (!dit_is_onnx) {
        vae_entry = first_non_onnx(registry.vae);
    } else if (!onnx_dit_has_sibling_vae(dit_entry) && !registry.vae.empty()) {
        vae_entry = &registry.vae[0];
    }
    if (!dit_is_onnx || gguf_text_enc || gguf_dit_aux) {
        if (!text_entry || !vae_entry) {
            fprintf(stderr, "[Ace-Synth] FATAL: selected route requires GGUF/SafeTensors text-encoder and VAE models\n");
            return 1;
        }
        if (str_ends_with(text_entry->path, ".onnx")) {
            fprintf(stderr, "[Ace-Synth] FATAL: selected route requires a GGUF/SafeTensors text encoder\n");
            return 1;
        }
    }
    const AdapterEntry * adapter_entry = NULL;
    if (!reqs[0].adapter.empty()) {
        adapter_entry = registry_find_adapter(registry, reqs[0].adapter.c_str());
        if (!adapter_entry) {
            fprintf(stderr, "[Ace-Synth] FATAL: adapter '%s' not found (use --adapters <dir>)\n",
                    reqs[0].adapter.c_str());
            return 1;
        }
    }

    // Resolve output_format to (is_mp3, wav_fmt).
    bool      is_mp3  = true;
    WavFormat wav_fmt = WAV_S16;
    if (!audio_parse_format(reqs[0].output_format.c_str(), is_mp3, wav_fmt)) {
        fprintf(stderr, "[Ace-Synth] FATAL: invalid output_format '%s' (use: mp3, wav16, wav24, wav32)\n",
                reqs[0].output_format.c_str());
        return 1;
    }

    // Fill params from registry lookups and CLI flags.
    params.text_encoder_path = text_entry ? text_entry->path.c_str() : NULL;
    params.dit_path          = dit_entry->path.c_str();
    params.aux_dit_path      = aux_dit_entry ? aux_dit_entry->path.c_str() : NULL;
    params.vae_path          = NULL;
    if (vae_entry && !str_ends_with(vae_entry->path, ".onnx")) {
        params.vae_path = vae_entry->path.c_str();
    }
    params.adapter_path      = adapter_entry ? adapter_entry->path.c_str() : NULL;
    params.adapter_scale     = reqs[0].adapter_scale;
    params.use_fa            = use_fa;
    params.use_batch_cfg     = use_batch_cfg;
    params.clamp_fp16        = clamp_fp16;
    params.vae_chunk         = vae_chunk;
    params.vae_overlap       = vae_overlap;
    params.dump_dir          = dump_dir;

    // Local store with the default STRICT policy: at most one GPU module
    // resident at a time for this one-shot CLI. No module sharing across runs,
    // so EVICT_STRICT frees the DiT before the VAE loads, and so on.
    ModelStore * store = store_create(EVICT_STRICT);
    AceSynth *   ctx   = ace_synth_load(store, &params);
    if (!ctx) {
        store_free(store);
        return 1;
    }

    // Read source audio (cover/lego mode)
    float * src_interleaved = NULL;
    int     src_len         = 0;
    if (src_audio_path) {
        int     T_audio = 0;
        float * planar  = audio_read_48k(src_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Ace-Synth] FATAL: cannot read --src-audio %s\n", src_audio_path);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        fprintf(stderr, "[Ace-Synth] Source audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

        src_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        src_len = T_audio;
    }

    // Read reference audio (timbre conditioning)
    float * ref_interleaved = NULL;
    int     ref_len         = 0;
    if (ref_audio_path) {
        int     T_audio = 0;
        float * planar  = audio_read_48k(ref_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Ace-Synth] FATAL: cannot read --ref-audio %s\n", ref_audio_path);
            free(src_interleaved);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        fprintf(stderr, "[Ace-Synth] Reference audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);
        ref_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        ref_len = T_audio;
    }

    // Generate every request in one DiT batch. synth_batch_size expands each
    // request into per-seed variants in groups[0]. Total clamped to DiT max 9.
    int total_alloc = 0;
    for (int ri = 0; ri < batch_n; ri++) {
        int sbs = reqs[ri].synth_batch_size;
        total_alloc += sbs < 1 ? 1 : (sbs > 9 ? 9 : sbs);
    }
    if (total_alloc > 9) {
        fprintf(stderr, "[Ace-Synth] Batch %d exceeds DiT max 9, clamping\n", total_alloc);
        total_alloc = 9;
    }
    std::vector<AceAudio>                all_audio(total_alloc);
    std::vector<std::string>             all_basenames(total_alloc);
    std::vector<int>                     all_synth_indices(total_alloc);
    std::vector<std::vector<AceRequest>> groups(1);
    groups[0].reserve(total_alloc);

    int off = 0;
    for (int ri = 0; ri < batch_n && off < total_alloc; ri++) {
        int sbs = reqs[ri].synth_batch_size;
        if (sbs < 1) {
            sbs = 1;
        }
        if (sbs > 9) {
            sbs = 9;
        }
        if (off + sbs > total_alloc) {
            sbs = total_alloc - off;
        }

        // resolve seed once per original request
        request_resolve_seed(&reqs[ri]);
        const long long base_seed = reqs[ri].seed;

        for (int i = 0; i < sbs; i++) {
            AceRequest r = reqs[ri];
            r.seed       = base_seed + i;
            groups[0].push_back(r);
            all_basenames[off + i]     = basenames[ri];
            all_synth_indices[off + i] = i;
        }
        off += sbs;
    }

    if (total_alloc > 1) {
        fprintf(stderr, "[Ace-Synth] Batch: %d track(s) from %d request(s)\n", total_alloc, batch_n);
    }

    // Two-phase run: DiT resident for all groups, then VAE for all jobs.
    const int rc = synth_batch_run(ctx, groups, src_interleaved, src_len,
                                   nullptr, 0,  // src_latents
                                   ref_interleaved, ref_len,
                                   nullptr, 0,  // ref_latents
                                   all_audio.data());
    if (rc != 0) {
        fprintf(stderr, "[Ace-Synth] ERROR: batch run failed\n");
        for (auto & a : all_audio) {
            ace_audio_free(&a);
        }
        free(src_interleaved);
        free(ref_interleaved);
        ace_synth_free(ctx);
        store_free(store);
        return 1;
    }

    // Write output files
    for (int b = 0; b < (int) all_audio.size(); b++) {
        if (!all_audio[b].samples) {
            continue;
        }
        const char * ext = is_mp3 ? ".mp3" : ".wav";
        char         out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s%d%s", all_basenames[b].c_str(), all_synth_indices[b], ext);
        if (!audio_write(out_path, all_audio[b].samples, all_audio[b].n_samples, 48000, mp3_kbps, wav_fmt)) {
            fprintf(stderr, "[Ace-Synth Batch%d] FATAL: failed to write %s\n", b, out_path);
        }
        ace_audio_free(&all_audio[b]);
    }

    free(src_interleaved);
    free(ref_interleaved);
    ace_synth_free(ctx);
    store_free(store);
    fprintf(stderr, "[Ace-Synth] All done\n");
    return 0;
}
