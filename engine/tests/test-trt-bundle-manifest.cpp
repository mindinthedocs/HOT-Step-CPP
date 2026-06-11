// test-trt-bundle-manifest.cpp: all-three-or-nothing fail-closed gate.
//
// Behavior claim: when a TRT bundle manifest is present but missing any of
// dit | text_enc | cond_enc (or a declared engine file is absent on disk),
// trt_bundle_load_manifest HARD-FAILS with a clear error and never reports the
// bundle as loadable. A complete manifest (all three engines present) loads and
// resolves every component path. Detection (trt_bundle_has_manifest) keys solely
// on manifest.json presence.
//
// Break-it: removing the all-three gate (accepting an incomplete manifest, or
// not checking engine-file existence) makes the missing-cond_enc and
// missing-engine cases load successfully -> these assertions fail.

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
    fs::path root = fs::temp_directory_path() / "acestep-w17-manifest-test" / "trt-bundles";
    std::error_code ec;
    fs::remove_all(root.parent_path(), ec);

    // Engine files all three components can point at (present on disk).
    fs::path good = root / "goodbundle";
    fs::create_directories(good, ec);
    write_file(good / "dit.engine", "ENGINE-DIT");
    write_file(good / "text_encoder.engine", "ENGINE-TEXT");
    write_file(good / "cond_encoder.engine", "ENGINE-COND");

    const char * complete_manifest =
        "{\n"
        "  \"version\": \"0.1.0\",\n"
        "  \"source_model\": \"acestep-v15-sft\",\n"
        "  \"variant\": \"q8map-fp16\",\n"
        "  \"components\": {\n"
        "    \"dit\":      {\"engine\": \"dit.engine\", \"metadata\": \"dit.engine.metadata.json\", \"precision\": \"q8map-fp16\", \"weight_streaming\": true},\n"
        "    \"text_enc\": {\"engine\": \"text_encoder.engine\", \"metadata\": \"text_encoder.engine.metadata.json\", \"precision\": \"fp16\", \"sidecars\": [\"embed_tokens.bin\"]},\n"
        "    \"cond_enc\": {\"engine\": \"cond_encoder.engine\", \"metadata\": \"cond_encoder.engine.metadata.json\", \"precision\": \"fp16\", \"sidecars\": []},\n"
        "    \"fsq\":      {\"sidecar\": \"fsq.safetensors\", \"metadata\": \"fsq.metadata.json\"}\n"
        "  }\n"
        "}\n";

    // ── Detection keys solely on manifest.json presence ──
    CHECK(!trt_bundle_has_manifest(good.string()),
          "no manifest.json yet -> not detected as a TRT bundle");
    write_file(good / "manifest.json", complete_manifest);
    CHECK(trt_bundle_has_manifest(good.string()),
          "manifest.json present -> detected as a TRT bundle");

    // ── Positive: complete manifest loads + resolves every component ──
    {
        TrtBundleManifest m;
        std::string err;
        bool ok = trt_bundle_load_manifest(good.string(), &m, &err);
        CHECK(ok, "complete manifest (all three engines present) loads");
        CHECK(m.dit.engine.find("dit.engine") != std::string::npos &&
                  m.text_enc.engine.find("text_encoder.engine") != std::string::npos &&
                  m.cond_enc.engine.find("cond_encoder.engine") != std::string::npos,
              "all three component engine paths resolved");
        CHECK(m.dit.weight_streaming, "dit weight_streaming parsed");
        CHECK(m.text_enc.sidecars.size() == 1 &&
                  m.text_enc.sidecars[0].find("embed_tokens.bin") != std::string::npos,
              "text_enc sidecar resolved");
        CHECK(m.version == "0.1.0" && m.source_model == "acestep-v15-sft",
              "manifest version/source_model parsed");
    }

    // ── Fail-closed: manifest missing the cond_enc component ──
    {
        fs::path bad = root / "badbundle-missing-cond";
        fs::create_directories(bad, ec);
        write_file(bad / "dit.engine", "ENGINE-DIT");
        write_file(bad / "text_encoder.engine", "ENGINE-TEXT");
        const char * missing_cond =
            "{\n"
            "  \"version\": \"0.1.0\",\n"
            "  \"source_model\": \"acestep-v15-sft\",\n"
            "  \"variant\": \"q8map-fp16\",\n"
            "  \"components\": {\n"
            "    \"dit\":      {\"engine\": \"dit.engine\"},\n"
            "    \"text_enc\": {\"engine\": \"text_encoder.engine\"}\n"
            "  }\n"
            "}\n";
        write_file(bad / "manifest.json", missing_cond);

        // Detection still sees a manifest...
        CHECK(trt_bundle_has_manifest(bad.string()),
              "missing-cond_enc bundle still has a manifest.json (detected)");

        // ...but loading HARD-FAILS (all-three-or-nothing).
        TrtBundleManifest m;
        std::string err;
        bool ok = trt_bundle_load_manifest(bad.string(), &m, &err);
        CHECK(!ok, "manifest missing cond_enc HARD-FAILS (all-three-or-nothing)");
        CHECK(err.find("cond_enc") != std::string::npos,
              "error names the missing component (cond_enc)");
        fprintf(stderr, "[test] missing-cond_enc error: %s\n", err.c_str());
    }

    // ── Fail-closed: component declares an engine that is absent on disk ──
    {
        fs::path bad = root / "badbundle-missing-engine";
        fs::create_directories(bad, ec);
        write_file(bad / "dit.engine", "ENGINE-DIT");
        write_file(bad / "text_encoder.engine", "ENGINE-TEXT");
        // cond_encoder.engine intentionally NOT written.
        const char * missing_engine =
            "{\n"
            "  \"version\": \"0.1.0\",\n"
            "  \"source_model\": \"acestep-v15-sft\",\n"
            "  \"variant\": \"q8map-fp16\",\n"
            "  \"components\": {\n"
            "    \"dit\":      {\"engine\": \"dit.engine\"},\n"
            "    \"text_enc\": {\"engine\": \"text_encoder.engine\"},\n"
            "    \"cond_enc\": {\"engine\": \"cond_encoder.engine\"}\n"
            "  }\n"
            "}\n";
        write_file(bad / "manifest.json", missing_engine);

        TrtBundleManifest m;
        std::string err;
        bool ok = trt_bundle_load_manifest(bad.string(), &m, &err);
        CHECK(!ok, "manifest declaring an absent engine HARD-FAILS");
        CHECK(err.find("cond_encoder.engine") != std::string::npos &&
                  err.find("missing on disk") != std::string::npos,
              "error names the missing engine file on disk");
        fprintf(stderr, "[test] missing-engine error: %s\n", err.c_str());
    }

    fs::remove_all(root.parent_path(), ec);

    if (g_fail == 0) {
        fprintf(stderr, "PASS: trt_bundle manifest all-three-or-nothing fail-closed verified\n");
        return 0;
    }
    fprintf(stderr, "FAILED: %d assertion(s)\n", g_fail);
    return 1;
}
