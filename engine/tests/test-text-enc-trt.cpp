// test-text-enc-trt.cpp
// Parity + lifecycle test for the native TRT text encoder (text-enc-trt.h).
//
// Proves the C++ deserialize+forward reproduces the PyTorch FP32 reference, that
// the evict→reactivate path returns identical output, and (mutation-kill) that
// the cosine assertion actually fails when the output is perturbed.
//
// Fixture (gitignored, produced by .enc-build/gen_text_enc_fixture.py):
//   .enc-build/text_enc_parity_fixture.bin  — "TEXTENC1" magic, B,S,H ints,
//                                              int32[B*S] token_ids,
//                                              float32[B*S*H] PyTorch reference
//   .enc-build/text_encoder.engine          — strongly-typed FP16 engine
//
// Usage:
//   test-text-enc-trt <engine_path> <fixture_path>
//   test-text-enc-trt                      (defaults to ../.enc-build/* and
//                                           .enc-build/* relative to cwd)
//
// Build: CMake target test-text-enc-trt, only when TRT_ENABLED. See
// engine/CMakeLists.txt.

#include "text-enc-trt.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef HOT_STEP_TRT
int main() {
    fprintf(stderr, "test-text-enc-trt: built without HOT_STEP_TRT; nothing to test\n");
    return 0;
}
#else

static int g_failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); \
            g_failures++;                                             \
        } else {                                                     \
            fprintf(stderr, "ok: %s\n", (msg));                      \
        }                                                            \
    } while (0)

struct Fixture {
    int                  B = 0, S = 0, H = 0;
    std::vector<int>     token_ids;  // [B*S]
    std::vector<float>   reference;  // [B*S*H], row-major [B,S,H]
};

static bool load_fixture(const char * path, Fixture & fx) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FAIL: cannot open fixture %s\n", path);
        return false;
    }
    char magic[8] = {};
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "TEXTENC1", 8) != 0) {
        fprintf(stderr, "FAIL: bad fixture magic in %s\n", path);
        fclose(f);
        return false;
    }
    int32_t hdr[3] = {};
    if (fread(hdr, sizeof(int32_t), 3, f) != 3) {
        fprintf(stderr, "FAIL: bad fixture header\n");
        fclose(f);
        return false;
    }
    fx.B = hdr[0]; fx.S = hdr[1]; fx.H = hdr[2];
    const size_t n_tok = (size_t) fx.B * fx.S;
    const size_t n_hid = n_tok * fx.H;
    fx.token_ids.resize(n_tok);
    fx.reference.resize(n_hid);
    if (fread(fx.token_ids.data(), sizeof(int), n_tok, f) != n_tok ||
        fread(fx.reference.data(), sizeof(float), n_hid, f) != n_hid) {
        fprintf(stderr, "FAIL: fixture body short read\n");
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

static double cosine(const float * a, const float * b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

int main(int argc, char ** argv) {
    const char * engine_path  = (argc > 1) ? argv[1] : "../.enc-build/text_encoder.engine";
    const char * fixture_path = (argc > 2) ? argv[2] : "../.enc-build/text_enc_parity_fixture.bin";

    Fixture fx;
    if (!load_fixture(fixture_path, fx)) {
        fprintf(stderr, "\nFAILED: fixture load (path=%s)\n", fixture_path);
        return 1;
    }
    fprintf(stderr, "[test] fixture B=%d S=%d H=%d\n", fx.B, fx.S, fx.H);

    TextEncTrt ctx = {};
    if (!text_enc_trt_load(&ctx, engine_path)) {
        fprintf(stderr, "\nFAILED: engine load (path=%s)\n", engine_path);
        return 1;
    }
    CHECK(ctx.hidden_size == fx.H, "engine hidden_size matches fixture H");

    const size_t n_hid = (size_t) fx.B * fx.S * fx.H;
    std::vector<float> out1(n_hid, 0.0f);

    // ── Parity: C++ forward vs PyTorch reference ──
    if (!text_enc_trt_forward(&ctx, fx.token_ids.data(), fx.B, fx.S, out1.data())) {
        fprintf(stderr, "\nFAILED: forward #1\n");
        text_enc_trt_free(&ctx);
        return 1;
    }
    double cos1 = cosine(fx.reference.data(), out1.data(), n_hid);
    fprintf(stderr, "[test] COSINE(ref, cpp_forward_1) = %.8f\n", cos1);
    CHECK(cos1 > 0.99, "parity cosine > 0.99");

    // ── Mutation-kill: a perturbed output must drop below the threshold ──
    // Confirms the assertion is real, not vacuous. Flip the sign of the first
    // half of the elements; this is a direct, plausible "loader wrong" shape.
    {
        std::vector<float> mutated = out1;
        for (size_t i = 0; i < n_hid / 2; i++) {
            mutated[i] = -mutated[i];
        }
        double cos_mut = cosine(fx.reference.data(), mutated.data(), n_hid);
        fprintf(stderr, "[test] COSINE(ref, mutated) = %.8f\n", cos_mut);
        CHECK(cos_mut < 0.99, "mutation-kill: perturbed output fails the 0.99 bar");
    }

    // ── Evict → reactivate: free I/O buffers + context, forward again ──
    text_enc_trt_release_evictable(&ctx);
    CHECK(ctx.context == nullptr, "release_evictable freed execution context");
    CHECK(ctx.d_input_ids == nullptr && ctx.d_hidden == nullptr,
          "release_evictable freed device I/O buffers");
    CHECK(ctx.engine != nullptr && ctx.runtime != nullptr,
          "release_evictable kept the resident engine shell");

    std::vector<float> out2(n_hid, 0.0f);
    if (!text_enc_trt_forward(&ctx, fx.token_ids.data(), fx.B, fx.S, out2.data())) {
        fprintf(stderr, "\nFAILED: forward #2 after reactivation\n");
        text_enc_trt_free(&ctx);
        return 1;
    }
    CHECK(ctx.context != nullptr, "forward reallocated execution context after evict");
    CHECK(ctx.d_input_ids != nullptr && ctx.d_hidden != nullptr,
          "forward reallocated device I/O buffers after evict");

    // Same input + same resident engine → bit-identical output.
    bool identical = (memcmp(out1.data(), out2.data(), n_hid * sizeof(float)) == 0);
    double cos_reuse = cosine(out1.data(), out2.data(), n_hid);
    fprintf(stderr, "[test] reactivated output identical=%d cosine(out1,out2)=%.8f\n",
            (int) identical, cos_reuse);
    CHECK(identical, "evict→reactivate yields bit-identical output");

    text_enc_trt_free(&ctx);
    CHECK(ctx.engine == nullptr && ctx.runtime == nullptr, "free unloaded the engine shell");

    if (g_failures == 0) {
        fprintf(stderr, "\nPASS: text_enc_trt parity + evict/reactivate verified\n");
        return 0;
    }
    fprintf(stderr, "\nFAILED with %d failure(s)\n", g_failures);
    return 1;
}

#endif  // HOT_STEP_TRT
