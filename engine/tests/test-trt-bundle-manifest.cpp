// test-trt-bundle-manifest.cpp: manifest reader for DiT + Qwen3-emb bundles.
//
// Behavior claims:
//   - DiT bundle manifest: dit + cond_enc mandatory; text_enc ignored if present.
//     A manifest missing dit or cond_enc HARD-FAILS.
//   - Qwen3-emb bundle manifest: text_enc mandatory (no dit, no cond_enc).
//     A manifest with only text_enc loads OK.
//   - A manifest declaring neither dit nor text_enc HARD-FAILS.
//   - A manifest declaring an engine file that is absent on disk HARD-FAILS.
//   - Detection (trt_bundle_has_manifest) keys solely on manifest.json presence.

#include "trt-bundle-manifest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

static int g_fail = 0;
#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (cond) {                                                         \
            fprintf(stderr, "ok: %s\n", msg);                              \
        } else {                                                            \
            fprintf(stderr, "FAIL: %s\n", msg);                            \
            g_fail++;                                                       \
        }                                                                   \
    } while (0)

static void write_file(const fs::path & p, const std::string & content) {
    std::ofstream f(p, std::ios::binary);
    f << content;
}

int main() {
    fs::path root = fs::temp_directory_path() / "acestep-manifest-test" / "trt-bundles";
    std::error_code ec;
    fs::remove_all(root.parent_path(), ec);

    // ── DiT bundle: dit + cond_enc + fsq (no text_enc) — the project default ──
    {
        fs::path dit = root / "dit-bundle";
        fs::create_directories(dit, ec);
        write_file(dit / "dit.engine", "ENGINE-DIT");
        write_file(dit / "cond_encoder.engine", "ENGINE-COND");
        const char * manifest =
            "{\n"
            "  \"version\": \"0.1.0\",\n"
            "  \"source_model\": \"acestep-v15-sft\",\n"
            "  \"variant\": \"q8map-fp16\",\n"
            "  \"components\": {\n"
            "    \"dit\":      {\"engine\": \"dit.engine\", \"metadata\": \"dit.metadata.json\", \"precision\": \"q8map-fp16\", \"weight_streaming\": true},\n"
            "    \"cond_enc\": {\"engine\": \"cond_encoder.engine\", \"metadata\": \"cond_encoder.metadata.json\", \"precision\": \"fp16\", \"sidecars\": []},\n"
            "    \"fsq\":      {\"sidecar\": \"fsq.safetensors\", \"metadata\": \"fsq.metadata.json\"}\n"
            "  }\n"
            "}\n";
        write_file(dit / "manifest.json", manifest);

        CHECK(trt_bundle_has_manifest(dit.string()),
              "DiT bundle detected (manifest.json present)");

        TrtBundleManifest m;
        std::string err;
        bool ok = trt_bundle_load_manifest(dit.string(), &m, &err);
        CHECK(ok, "DiT bundle (dit + cond_enc, no text_enc) loads");
        CHECK(m.dit.engine.find("dit.engine") != std::string::npos,
              "DiT engine resolved");
        CHECK(m.cond_enc.engine.find("cond_encoder.engine") != std::string::npos,
              "cond_enc engine resolved");
        CHECK(m.text_enc.engine.empty(),
              "text_enc.engine empty (not in manifest)");
        CHECK(m.dit.weight_streaming, "dit weight_streaming parsed");
        if (!ok) fprintf(stderr, "[test] dit bundle error: %s\n", err.c_str());
    }

    // ── Qwen3-emb bundle: text_enc only (no dit, no cond_enc) ──
    {
        fs::path emb = root / "qwen3-emb";
        fs::create_directories(emb, ec);
        write_file(emb / "text_encoder.engine", "ENGINE-TEXT");
        write_file(emb / "embed_tokens.bin", "EMBED-TABLE");
        const char * manifest =
            "{\n"
            "  \"version\": \"0.1.0\",\n"
            "  \"source_model\": \"Qwen3-Embedding-0.6B\",\n"
            "  \"variant\": \"fp16\",\n"
            "  \"components\": {\n"
            "    \"text_enc\": {\"engine\": \"text_encoder.engine\", \"metadata\": \"text_encoder.metadata.json\", \"precision\": \"fp16\", \"sidecars\": [\"embed_tokens.bin\", \"vocab.json\", \"merges.txt\"]}\n"
            "  }\n"
            "}\n";
        write_file(emb / "manifest.json", manifest);

        TrtBundleManifest m;
        std::string err;
        bool ok = trt_bundle_load_manifest(emb.string(), &m, &err);
        CHECK(ok, "Qwen3-emb bundle (text_enc only) loads");
        CHECK(m.text_enc.engine.find("text_encoder.engine") != std::string::npos,
              "text_enc engine resolved");
        CHECK(m.dit.engine.empty(), "dit.engine empty (not in manifest)");
        CHECK(m.cond_enc.engine.empty(), "cond_enc.engine empty (not in manifest)");
        CHECK(m.text_enc.sidecars.size() == 3,
              "text_enc sidecars parsed (3 entries)");
        if (!ok) fprintf(stderr, "[test] qwen3-emb bundle error: %s\n", err.c_str());
    }

    // ── DiT bundle missing cond_enc → HARD FAIL ──
    {
        fs::path bad = root / "dit-missing-cond";
        fs::create_directories(bad, ec);
        write_file(bad / "dit.engine", "ENGINE-DIT");
        const char * manifest =
            "{\n"
            "  \"components\": {\n"
            "    \"dit\": {\"engine\": \"dit.engine\"}\n"
            "  }\n"
            "}\n";
        write_file(bad / "manifest.json", manifest);

        TrtBundleManifest m;
        std::string err;
        bool ok = trt_bundle_load_manifest(bad.string(), &m, &err);
        CHECK(!ok, "DiT bundle missing cond_enc HARD-FAILS");
        CHECK(err.find("cond_enc") != std::string::npos,
              "error names cond_enc");
        fprintf(stderr, "[test] dit-missing-cond error: %s\n", err.c_str());
    }

    // ── Manifest with neither dit nor text_enc → HARD FAIL ──
    {
        fs::path bad = root / "empty-bundle";
        fs::create_directories(bad, ec);
        const char * manifest =
            "{\n"
            "  \"components\": {\n"
            "    \"fsq\": {\"sidecar\": \"fsq.safetensors\"}\n"
            "  }\n"
            "}\n";
        write_file(bad / "manifest.json", manifest);

        TrtBundleManifest m;
        std::string err;
        bool ok = trt_bundle_load_manifest(bad.string(), &m, &err);
        CHECK(!ok, "manifest with neither dit nor text_enc HARD-FAILS");
        CHECK(err.find("neither") != std::string::npos,
              "error mentions neither dit nor text_enc");
        fprintf(stderr, "[test] empty-bundle error: %s\n", err.c_str());
    }

    // ── Manifest declaring an absent engine → HARD FAIL ──
    {
        fs::path bad = root / "missing-engine";
        fs::create_directories(bad, ec);
        write_file(bad / "dit.engine", "ENGINE-DIT");
        // cond_encoder.engine intentionally NOT written.
        const char * manifest =
            "{\n"
            "  \"components\": {\n"
            "    \"dit\":      {\"engine\": \"dit.engine\"},\n"
            "    \"cond_enc\": {\"engine\": \"cond_encoder.engine\"}\n"
            "  }\n"
            "}\n";
        write_file(bad / "manifest.json", manifest);

        TrtBundleManifest m;
        std::string err;
        bool ok = trt_bundle_load_manifest(bad.string(), &m, &err);
        CHECK(!ok, "manifest declaring absent engine HARD-FAILS");
        CHECK(err.find("cond_encoder.engine") != std::string::npos &&
              err.find("missing on disk") != std::string::npos,
              "error names the missing engine file");
        fprintf(stderr, "[test] missing-engine error: %s\n", err.c_str());
    }

    fs::remove_all(root.parent_path(), ec);

    if (g_fail == 0) {
        fprintf(stderr, "PASS: manifest reader (DiT + Qwen3-emb bundles) verified\n");
        return 0;
    }
    fprintf(stderr, "FAILED: %d assertion(s)\n", g_fail);
    return 1;
}
