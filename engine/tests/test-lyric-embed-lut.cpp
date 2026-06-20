// test-lyric-embed-lut.cpp
// Parity oracle for the TRT-path lyric-embedding lookup (lyric-embed-lut.h).
//
// The TRT cond path produces lyric_embed by row-gathering the bundle's raw
// embed_tokens.bin. The GGML path produces it via qwen3_embed_lookup
// (ggml_get_rows against the same model.embed_tokens.weight tensor, loaded from
// the GGUF text encoder). The .bin is that table extracted to f32 by
// export_text_enc.py, so for a fixed set of token IDs the two row-gathers must
// match. The GGUF table is Q8_0-quantized, so parity is high-cosine /
// near-equal, not bit-equal — a per-row cosine ~0.999 (well above 0.99) proves
// the lookup is faithful and kills the old 0.0f stub and any
// row-stride/dtype/table error.
//
// A mutation-kill confirms the cosine assertion is real, not vacuous.
//
// Usage:
//   test-lyric-embed-lut <embed_tokens.bin> <text_encoder.gguf>
//   test-lyric-embed-lut   (defaults to the on-box bundle + GGUF text encoder)
//
// Build: CMake target test-lyric-embed-lut, only when TRT_ENABLED. Run on demand
// against the on-disk bundle; not in the default build.

#include "lyric-embed-lut.h"

#ifndef HOT_STEP_TRT
#include <cstdio>
int main() {
    fprintf(stderr, "test-lyric-embed-lut: built without HOT_STEP_TRT; nothing to test\n");
    return 0;
}
#else

#include "qwen3-enc.h"

#include <cmath>
#include <cstdio>
#include <vector>

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
    const char * bin_path  = (argc > 1) ? argv[1]
        : "../models/trt-bundles/trt-acestep-v15-sft-q8map-fp16/embed_tokens.bin";
    const char * gguf_path = (argc > 2) ? argv[2]
        : "../models/Qwen3-Embedding-0.6B-Q8_0.gguf";

    // ── Load the raw .bin table (TRT-path source) ──
    LyricEmbedLut lut;
    if (!lyric_embed_lut_load(bin_path, lut)) {
        fprintf(stderr, "\nFAILED: embed_tokens.bin load (path=%s)\n", bin_path);
        return 1;
    }
    fprintf(stderr, "[test] .bin table [%d, %d]\n", lut.rows, lut.cols);
    CHECK(lut.cols == 1024, "embed table width is 1024");
    CHECK(lut.rows > 100000, "embed table has the Qwen3-Embedding vocab row count");

    // Fixed token IDs spanning low/mid/high vocab, including the lyric end marker
    // 151643 (<|endoftext|>) the live path actually emits, and the table's last
    // valid row (rows-1) to exercise the top edge.
    std::vector<int> ids = {0, 1, 100, 1234, 9999, 64002, 100000, 151643, lut.rows - 1};
    const int S = (int) ids.size();

    // ── TRT-path lookup: .bin row-gather ──
    std::vector<float> bin_out;  // [S*cols] row-major, H contiguous per token
    if (!lyric_embed_lut_lookup(lut, ids.data(), S, bin_out)) {
        fprintf(stderr, "\nFAILED: .bin lookup (bounds/format)\n");
        return 1;
    }

    // ── GGML-path producer: qwen3_embed_lookup against the GGUF text encoder ──
    Qwen3GGML m = {};
    if (!qwen3_load_text_encoder(&m, gguf_path)) {
        fprintf(stderr, "\nFAILED: GGML text encoder load (path=%s)\n", gguf_path);
        return 1;
    }
    const int H = m.cfg.hidden_size;
    CHECK(H == lut.cols, "GGML hidden_size matches .bin width");

    std::vector<float> ggml_out((size_t) H * S, 0.0f);  // [H*S] H-contiguous per token
    qwen3_embed_lookup(&m, ids.data(), S, ggml_out.data());

    // Both layouts are H-contiguous per token, so compare element-wise / per-row.
    double whole_cos = cosine(bin_out.data(), ggml_out.data(), (size_t) H * S);
    fprintf(stderr, "[test] whole-tensor COSINE(.bin, qwen3_embed_lookup) = %.8f\n", whole_cos);
    CHECK(whole_cos > 0.99, "lyric-embed parity cosine > 0.99 (Q8_0 quant tolerance)");

    double min_row_cos = 1.0;
    for (int t = 0; t < S; t++) {
        double rc = cosine(bin_out.data() + (size_t) t * H,
                           ggml_out.data() + (size_t) t * H, H);
        if (rc < min_row_cos) min_row_cos = rc;
        fprintf(stderr, "[test]   row id=%d cos=%.8f\n", ids[t], rc);
    }
    fprintf(stderr, "[test] min per-row cosine = %.8f\n", min_row_cos);
    CHECK(min_row_cos > 0.99, "every row's parity cosine > 0.99 (no wrong-stride row)");

    // ── Mutation-kill: a wrong-stride lookup (shift the table by one row) must
    //    drop below the bar, proving the assertion is non-vacuous. ──
    {
        std::vector<int> shifted = ids;
        for (int & v : shifted) v = (v + 1) % lut.rows;
        std::vector<float> bad;
        lyric_embed_lut_lookup(lut, shifted.data(), S, bad);
        double bad_cos = cosine(bad.data(), ggml_out.data(), (size_t) H * S);
        fprintf(stderr, "[test] mutation (off-by-one row) COSINE = %.8f\n", bad_cos);
        CHECK(bad_cos < 0.99, "mutation-kill: wrong-row lookup fails the 0.99 bar");
    }

    // ── Bounds-check: an out-of-range id must be rejected, not silently read. ──
    {
        std::vector<int> oob = {lut.rows};  // one past the last valid row
        std::vector<float> dummy;
        bool ok = lyric_embed_lut_lookup(lut, oob.data(), 1, dummy);
        CHECK(!ok, "bounds-check: out-of-range token id is rejected");
    }

    qwen3_free(&m);

    if (g_failures == 0) {
        fprintf(stderr, "\nPASS: lyric-embed lookup parity vs qwen3_embed_lookup verified\n");
        return 0;
    }
    fprintf(stderr, "\nFAILED with %d failure(s)\n", g_failures);
    return 1;
}

#endif  // HOT_STEP_TRT
