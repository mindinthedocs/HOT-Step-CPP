# ONNX Export Tools

Scripts for exporting HOT-Step model components to ONNX format for use with
TensorRT or ONNX Runtime inference.

## Prerequisites

- Python 3.10+ with the hot-step-9000 venv
- PyTorch with CUDA support
- `diffusers`, `onnx`, `onnxruntime` (or `onnxruntime-gpu` for GPU/TRT)

## Scripts

### `build-trt-engine.py` - DiT TensorRT Engine Build

Builds the standalone DiT TensorRT engine from `dit.onnx`. This is the direct
runtime engine path and writes `dit.engine` plus metadata when `--runtime-alias`
is enabled.

### `build-ort-trt-engines.py` - ORT TensorRT Side-Module Cache Build

Prebuilds the ONNX Runtime TensorRT EP engine caches for side modules that are
loaded through ORT instead of the direct DiT engine path:

- `text_encoder.onnx`
- `cond_encoder.onnx`
- `vae_decoder.onnx`

Default output is an `ort-trt-engines` directory next to the ONNX bundle. The
C++ runtime looks there by default, so a deployable bundle can contain:

```text
dit.onnx
dit.onnx.data
dit.engine
text_encoder.onnx
cond_encoder.onnx
vae_decoder.onnx
ort-trt-engines/
```

**Usage:**
```powershell
& .venv\Scripts\python.exe tools\onnx-export\build-ort-trt-engines.py `
    --bundle-dir models\acestep-v15-xl-sftturbo50-dit-trt11-w8a16 `
    --modules text,cond,vae
```

For a custom cache location, pass `--out-dir` and set the generated
`use-prebuilt-ort-trt.ps1` environment values before launching the runtime.

To prevent runtime-side engine builds, set:

```powershell
$env:HOTSTEP_ORT_TRT_PREBUILT_ONLY = "1"
```

With that flag, missing or incomplete ORT TensorRT caches are fatal instead of
falling through to a first-run build.

Profile knobs must match between prebuild and runtime. If you override any of
these during prebuild, set the same environment variables before launching:

- `HOTSTEP_ORT_TRT_TEXT_OPT_TOKENS`, `HOTSTEP_ORT_TRT_TEXT_MAX_TOKENS`
- `HOTSTEP_ORT_TRT_LYRIC_OPT_TOKENS`, `HOTSTEP_ORT_TRT_LYRIC_MAX_TOKENS`
- `HOTSTEP_ORT_TRT_TIMBRE_OPT_FRAMES`, `HOTSTEP_ORT_TRT_TIMBRE_MAX_FRAMES`
- `HOTSTEP_ORT_TRT_VAE_OPT_FRAMES`, `HOTSTEP_ORT_TRT_VAE_MAX_FRAMES`
- `HOTSTEP_ORT_TRT_BUILDER_OPT_LEVEL`

`export_runtime_bundle.py --build-ort-engines` runs this script after export.

### `export_vae.py` — VAE Decoder Export

Exports the `AutoencoderOobleck` VAE decoder to ONNX format. Only the decoder
half is exported (encoder is not needed for inference — we decode latents to
audio).

**Tensor spec:**
| Name | Shape | Description |
|------|-------|-------------|
| `latents` (input) | `[B, 64, T]` | Latent channels, latent frames @ 25Hz |
| `audio` (output) | `[B, 2, S]` | Stereo audio, S = T × 1920 @ 48kHz |

Dynamic axes: batch (dim 0) and temporal dims (latent_frames, samples).

**Usage:**
```powershell
# Basic export
& .venv\Scripts\python.exe tools\onnx-export\export_vae.py `
    --vae-path checkpoints\vae `
    --output models\onnx\vae_decoder.onnx

# Export with validation (compares ONNX vs PyTorch output)
& .venv\Scripts\python.exe tools\onnx-export\export_vae.py `
    --vae-path checkpoints\vae `
    --output models\onnx\vae_decoder.onnx `
    --validate
```

**Options:**
- `--vae-path` — Path to VAE checkpoint directory (config.json + safetensors)
- `--output` — Output path for the ONNX file
- `--opset` — ONNX opset version (default: 18)
- `--validate` — Compare ONNX output against PyTorch using onnxruntime

### `test_trt_vae.py` — TensorRT Validation & Benchmark

Benchmarks the exported ONNX model using CUDA EP vs TensorRT EP. Reports
latency, speedup ratio, and numerical accuracy.

**Usage:**
```powershell
& .venv\Scripts\python.exe tools\onnx-export\test_trt_vae.py `
    --onnx models\onnx\vae_decoder.onnx
```

**Options:**
- `--onnx` — Path to the exported ONNX file
- `--trt-cache` — Directory for TRT engine cache (default: `models/onnx/trt_cache/`)
- `--iterations` — Number of benchmark iterations (default: 20)

**Requirements for TRT EP:**
- `onnxruntime-gpu` (not `onnxruntime`)
- TensorRT libraries on PATH
- The script gracefully falls back if TRT EP is unavailable

## Output Files

| File | Size | Git-tracked? |
|------|------|-------------|
| `models/onnx/vae_decoder.onnx` | ~330 MB | ❌ No (gitignored) |
| `models/onnx/trt_cache/*.engine` | ~200 MB | ❌ No (gitignored) |

## Notes

- VAE export is FP32. ONNX Runtime may use CUDA or TensorRT EP for execution.
- DiT export uses `q8map-fp16` by default: the exporter starts from an FP32
  graph, downcasts only the hardcoded `Q8_0`-equivalent DiT matrix-weight
  allowlist to FP16, and inserts Cast nodes so a strongly typed TensorRT build
  honors the graph policy. The DiT builder does not set a blanket FP16/BF16
  builder flag.
- DiT `w8a16` uses the same hardcoded matrix-weight allowlist, quantizes those
  ONNX initializers to symmetric INT8 per output channel, and inserts
  DequantizeLinear/Cast nodes so TensorRT sees INT8 weight storage feeding FP16
  MatMul/Gemm islands with FP32 graph I/O. This is native ONNX Q/DQ weight-only
  quantization, not a TensorRT-LLM engine path.
- DiT `fp32` export leaves all DiT weights FP32 for validation.
- The VAE is a simple 1D convolutional network (no attention layers), so ONNX
  export is straightforward — no trace-safe patches needed.
- ScragVAE (674MB) can also be exported using the same script with a different
  `--vae-path`.
