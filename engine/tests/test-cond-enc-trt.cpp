// test-cond-enc-trt.cpp
// Parity + lifecycle test for the native TRT condition encoder (cond-enc-trt.h).
//
// Proves the C++ deserialize+forward reproduces the PyTorch FP32 reference, that
// the evict→reactivate path returns identical output, and (mutation-kill) that
// the cosine assertion actually fails when the output is perturbed.
//
// Fixture (gitignored, produced by .enc-build/gen_cond_enc_fixture.py):
//   .enc-build/cond_enc_parity_fixture.bin  — "CONDENC1" magic, header ints,
//                                              float32 text_hidden/lyric_embed/
//                                              timbre_feats, float32 enc_hidden
//                                              PyTorch reference
//   .enc-build/cond_encoder.engine          — strongly-typed FP16 engine
//
// Fixture binary format (little-endian):
//   char[8]  magic = "CONDENC1"
//   int32    B, S_text, S_lyric, S_ref, S_total, H
//   float32[B*S_text*1024]   text_hidden
//   float32[B*S_lyric*1024]  lyric_embed
//   float32[B*S_ref*64]      timbre_feats
//   float32[B*S_total*H]     reference (PyTorch FP32 enc_hidden, [B,S_total,H])
//
// Usage:
//   test-cond-enc-trt <engine_path> <fixture_path>
//   test-cond-enc-trt   (defaults to ../.enc-build/* relative to cwd)
//
// Build: CMake target test-cond-enc-trt, only when TRT_ENABLED. See
// engine/CMakeLists.txt.

#include "cond-enc-trt.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef HOT_STEP_TRT
int main() {
    fprintf(stderr, "test-cond-enc-trt: built without HOT_STEP_TRT; nothing to test\n");
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
    int                B = 0, S_text = 0, S_lyric = 0, S_ref = 0, S_total = 0, H = 0;
    std::vector<float> text_hidden;   // [B*S_text*1024]
    std::vector<float> lyric_embed;   // [B*S_lyric*1024]
    std::vector<float> timbre_feats;  // [B*S_ref*64]
    std::vector<float> reference;     // [B*S_total*H], row-major [B,S_total,H]
};

static bool load_fixture(const char * path, Fixture & fx) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FAIL: cannot open fixture %s\n", path);
        return false;
    }
    char magic[8] = {};
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "CONDENC1", 8) != 0) {
        fprintf(stderr, "FAIL: bad fixture magic in %s\n", path);
        fclose(f);
        return false;
    }
    int32_t hdr[6] = {};
    if (fread(hdr, sizeof(int32_t), 6, f) != 6) {
        fprintf(stderr, "FAIL: bad fixture header\n");
        fclose(f);
        return false;
    }
    fx.B = hdr[0]; fx.S_text = hdr[1]; fx.S_lyric = hdr[2];
    fx.S_ref = hdr[3]; fx.S_total = hdr[4]; fx.H = hdr[5];
    const size_t n_text  = (size_t) fx.B * fx.S_text  * 1024;
    const size_t n_lyric = (size_t) fx.B * fx.S_lyric * 1024;
    const size_t n_timb  = (size_t) fx.B * fx.S_ref   * 64;
    const size_t n_ref   = (size_t) fx.B * fx.S_total * fx.H;
    fx.text_hidden.resize(n_text);
    fx.lyric_embed.resize(n_lyric);
    fx.timbre_feats.resize(n_timb);
    fx.reference.resize(n_ref);
    if (fread(fx.text_hidden.data(),  sizeof(float), n_text,  f) != n_text  ||
        fread(fx.lyric_embed.data(),  sizeof(float), n_lyric, f) != n_lyric ||
        fread(fx.timbre_feats.data(), sizeof(float), n_timb,  f) != n_timb  ||
        fread(fx.reference.data(),    sizeof(float), n_ref,   f) != n_ref) {
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
    const char * engine_path  = (argc > 1) ? argv[1] : "../.enc-build/cond_encoder.engine";
    const char * fixture_path = (argc > 2) ? argv[2] : "../.enc-build/cond_enc_parity_fixture.bin";

    Fixture fx;
    if (!load_fixture(fixture_path, fx)) {
        fprintf(stderr, "\nFAILED: fixture load (path=%s)\n", fixture_path);
        return 1;
    }
    fprintf(stderr, "[test] fixture B=%d S_text=%d S_lyric=%d S_ref=%d S_total=%d H=%d\n",
            fx.B, fx.S_text, fx.S_lyric, fx.S_ref, fx.S_total, fx.H);

    CondEncTrt ctx = {};
    if (!cond_enc_trt_load(&ctx, engine_path)) {
        fprintf(stderr, "\nFAILED: engine load (path=%s)\n", engine_path);
        return 1;
    }
    CHECK(ctx.hidden_size == fx.H, "engine hidden_size matches fixture H");

    const size_t n_out = (size_t) fx.B * fx.S_total * fx.H;
    std::vector<float> out1;
    int S_total_1 = 0;

    // ── Parity: C++ forward vs PyTorch reference ──
    if (!cond_enc_trt_forward(&ctx,
                              fx.text_hidden.data(),  fx.S_text,
                              fx.lyric_embed.data(),  fx.S_lyric,
                              fx.timbre_feats.data(), fx.S_ref,
                              fx.B, out1, &S_total_1)) {
        fprintf(stderr, "\nFAILED: forward #1\n");
        cond_enc_trt_free(&ctx);
        return 1;
    }
    CHECK(S_total_1 == fx.S_total, "forward packed S_total matches fixture (lyric+1+text)");
    CHECK(out1.size() == n_out, "forward output element count matches [B,S_total,H]");
    double cos1 = cosine(fx.reference.data(), out1.data(), n_out);
    fprintf(stderr, "[test] COSINE(ref, cpp_forward_1) = %.8f\n", cos1);
    CHECK(cos1 > 0.99, "parity cosine > 0.99");

    // ── Mutation-kill: a perturbed output must drop below the threshold ──
    // Confirms the >0.99 assertion is real, not vacuous. A sign error in the
    // FP16->FP32 decode (a direct, plausible "loader wrong" shape) negates every
    // element; that must collapse the cosine. We negate the whole output to
    // model that decode bug. (Flipping only a fraction does not move the cosine
    // far on this large 193x2048 output, which is exactly why a fractional
    // mutation would be a weak, near-vacuous check.)
    {
        std::vector<float> mutated = out1;
        for (size_t i = 0; i < n_out; i++) {
            mutated[i] = -mutated[i];
        }
        double cos_mut = cosine(fx.reference.data(), mutated.data(), n_out);
        fprintf(stderr, "[test] COSINE(ref, mutated_sign_flip) = %.8f\n", cos_mut);
        CHECK(cos_mut < 0.99, "mutation-kill: sign-flipped output fails the 0.99 bar");
    }

    // ── Evict → reactivate: free I/O buffers + context, forward again ──
    cond_enc_trt_release_evictable(&ctx);
    CHECK(ctx.context == nullptr, "release_evictable freed execution context");
    CHECK(ctx.d_text == nullptr && ctx.d_lyric == nullptr &&
          ctx.d_timbre == nullptr && ctx.d_out == nullptr,
          "release_evictable freed all device I/O buffers");
    CHECK(ctx.engine != nullptr && ctx.runtime != nullptr,
          "release_evictable kept the resident engine shell");

    std::vector<float> out2;
    int S_total_2 = 0;
    if (!cond_enc_trt_forward(&ctx,
                              fx.text_hidden.data(),  fx.S_text,
                              fx.lyric_embed.data(),  fx.S_lyric,
                              fx.timbre_feats.data(), fx.S_ref,
                              fx.B, out2, &S_total_2)) {
        fprintf(stderr, "\nFAILED: forward #2 after reactivation\n");
        cond_enc_trt_free(&ctx);
        return 1;
    }
    CHECK(ctx.context != nullptr, "forward reallocated execution context after evict");
    CHECK(ctx.d_text != nullptr && ctx.d_lyric != nullptr &&
          ctx.d_timbre != nullptr && ctx.d_out != nullptr,
          "forward reallocated device I/O buffers after evict");

    // Same input + same resident engine → bit-identical output.
    bool identical = (out1.size() == out2.size()) &&
                     (memcmp(out1.data(), out2.data(), n_out * sizeof(float)) == 0);
    double cos_reuse = cosine(out1.data(), out2.data(), n_out);
    fprintf(stderr, "[test] reactivated output identical=%d cosine(out1,out2)=%.8f\n",
            (int) identical, cos_reuse);
    CHECK(identical, "evict→reactivate yields bit-identical output");

    cond_enc_trt_free(&ctx);
    CHECK(ctx.engine == nullptr && ctx.runtime == nullptr, "free unloaded the engine shell");

    if (g_failures == 0) {
        fprintf(stderr, "\nPASS: cond_enc_trt parity + evict/reactivate verified\n");
        return 0;
    }
    fprintf(stderr, "\nFAILED with %d failure(s)\n", g_failures);
    return 1;
}

#endif  // HOT_STEP_TRT
