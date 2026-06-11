#pragma once
// lyric-embed-lut.h — CPU lyric-embedding lookup for the native TRT path.
//
// On the GGML path the lyric embedding is produced by qwen3_embed_lookup
// (qwen3-enc.h), a ggml_get_rows against the text encoder's embed_tokens tensor.
// On the TRT path the GGML Qwen3 encoder is NOT loaded (all-three-or-nothing
// forbids the GGUF text encoder when a TRT bundle is selected), so the lookup
// reads the bundle's raw embed_tokens.bin sidecar directly.
//
// embed_tokens.bin format (written by tools/onnx-export/export_text_enc.py
// export_embed_table): an 8-byte header of two little-endian uint32 — V (rows,
// vocab) and H (cols, hidden) — followed by V*H raw float32 values in row-major
// [V, H] order. It is the same model.embed_tokens.weight tensor the GGML text
// encoder loads as "embed_tokens.weight"; the .bin is that table extracted to
// raw f32, so a row-gather here equals qwen3_embed_lookup's output up to the
// GGML quantization of the loaded tensor.
//
// The lookup produces lyric_embed [S_lyric, H] f32 row-major (H contiguous per
// token), the exact layout cond_enc_trt_forward consumes for its lyric_embed
// input ([B,S_lyric,1024] row-major, B==1).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct LyricEmbedLut {
    int                rows = 0;  // V (vocab / table row count)
    int                cols = 0;  // H (hidden width, expected 1024)
    std::vector<float> table;     // [rows*cols] f32 row-major; row r at [r*cols]
};

// Parse embed_tokens.bin: 8-byte [uint32 V, uint32 H] header + V*H f32.
// Derives the shape from the file (does NOT assume 151669x1024) and validates
// the declared dimensions against the actual file size. Returns false on any
// I/O / format / size mismatch.
inline bool lyric_embed_lut_load(const char * path, LyricEmbedLut & lut) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[LyricEmbed] FATAL: cannot open embed table %s\n", path);
        return false;
    }
    uint32_t hdr[2] = {0, 0};
    if (fread(hdr, sizeof(uint32_t), 2, f) != 2) {
        fprintf(stderr, "[LyricEmbed] FATAL: short read on header %s\n", path);
        fclose(f);
        return false;
    }
    const uint32_t V = hdr[0];
    const uint32_t H = hdr[1];
    if (V == 0 || H == 0 || H > (1u << 20) || V > (1u << 24)) {
        fprintf(stderr, "[LyricEmbed] FATAL: implausible embed shape V=%u H=%u in %s\n", V, H, path);
        fclose(f);
        return false;
    }

    // Validate against the actual file size: 8-byte header + V*H f32.
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "[LyricEmbed] FATAL: fseek failed on %s\n", path);
        fclose(f);
        return false;
    }
    const long file_size = ftell(f);
    const long expect     = 8L + (long) ((size_t) V * H * sizeof(float));
    if (file_size != expect) {
        fprintf(stderr,
                "[LyricEmbed] FATAL: %s size %ld != header-implied %ld (V=%u H=%u f32)\n",
                path, file_size, expect, V, H);
        fclose(f);
        return false;
    }
    if (fseek(f, 8, SEEK_SET) != 0) {
        fprintf(stderr, "[LyricEmbed] FATAL: fseek to body failed on %s\n", path);
        fclose(f);
        return false;
    }

    const size_t n = (size_t) V * H;
    lut.table.resize(n);
    const size_t rd = fread(lut.table.data(), sizeof(float), n, f);
    fclose(f);
    if (rd != n) {
        fprintf(stderr, "[LyricEmbed] FATAL: body short read %zu/%zu on %s\n", rd, n, path);
        lut.table.clear();
        return false;
    }
    lut.rows = (int) V;
    lut.cols = (int) H;
    fprintf(stderr, "[LyricEmbed] loaded embed table [%d, %d] f32 from %s (%ld bytes)\n",
            lut.rows, lut.cols, path, file_size);
    return true;
}

// Row-gather: token_ids[0..S) -> out[S*cols] f32 row-major. Bounds-checks every
// token id against the real row count; an out-of-range id is a fatal error
// (returns false) rather than a silent garbage row. out is resized to S*cols.
inline bool lyric_embed_lut_lookup(const LyricEmbedLut & lut, const int * token_ids, int S,
                                   std::vector<float> & out) {
    if (lut.table.empty() || lut.rows <= 0 || lut.cols <= 0) {
        fprintf(stderr, "[LyricEmbed] FATAL: lookup on unloaded table\n");
        return false;
    }
    out.resize((size_t) S * lut.cols);
    for (int t = 0; t < S; t++) {
        const int id = token_ids[t];
        if (id < 0 || id >= lut.rows) {
            fprintf(stderr, "[LyricEmbed] FATAL: token id %d out of range [0,%d) at pos %d\n",
                    id, lut.rows, t);
            return false;
        }
        memcpy(out.data() + (size_t) t * lut.cols,
               lut.table.data() + (size_t) id * lut.cols,
               (size_t) lut.cols * sizeof(float));
    }
    return true;
}

// Process-wide cache keyed by absolute path so the 593 MB table loads once per
// server session and is reused across gens. Returns nullptr on load failure.
inline const LyricEmbedLut * lyric_embed_lut_get_cached(const std::string & path) {
    static std::mutex                                       mu;
    static std::unordered_map<std::string, LyricEmbedLut>   cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(path);
    if (it != cache.end()) {
        return &it->second;
    }
    LyricEmbedLut lut;
    if (!lyric_embed_lut_load(path.c_str(), lut)) {
        return nullptr;
    }
    auto res = cache.emplace(path, std::move(lut));
    return &res.first->second;
}
