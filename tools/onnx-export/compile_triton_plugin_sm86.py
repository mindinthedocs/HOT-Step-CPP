#!/usr/bin/env python3
"""Ahead-of-Time (AOT) compiler script for the fused Triton ConvRot INT8 kernel.

sm_86-ONLY build (for first-pass validation), but structurally designed to
seamlessly support future architectures (sm_75, sm_89, sm_90) via declarative
hardware restriction sets.

Compiles the fused Triton kernel for the 128x128 tile on sm_86 (RTX 30-series)
across all group_size / bias combinations, exporting them as embedded static
C++ hex arrays in: engine/src/plugins/assets/convrot_int8_kernel_cubin.h

Tested with Triton 3.7.1 (latest). The sm_86 (Ampere) INT8 `tl.dot` path
works on every modern Triton release; sm_75 (Turing) is broken upstream
between 3.3 and 3.7.1 and requires a separate patch — not included here.

Phase-1 contract (butterfly rotation)
-------------------------------------
- GROUP_SIZE in {0, 64, 256} (subset of {0, 4, 16, 64, 256, 1024}).
- BLOCK_K = GROUP_SIZE for ROT configs (statically asserted in the kernel).
- 128x128 tiles compiled for sm_86 only.
- num_warps=4 (128 threads). num_stages is AUTO-SELECTED per cubin to fit
  within the target architecture's per-block shared-memory opt-in limit:
  the compiler tries num_stages=max_stages first, then decrements down to 1,
  and picks the largest value whose compiled cubin's `metadata.shared` does
  not exceed the architecture's shared memory ceiling. This is critical because
  Triton 3.7.1 does NOT auto-reduce num_stages for FP32-IO G256 kernels —
  without this guard, the G256 FP32IO 128x128 cubin bakes 320 KB of dynamic
  shared memory into the kernel, the launch silently returns CUDA_SUCCESS
  (cuLaunchKernel only validates the opt-in ceiling, not the kernel's actual
  requirement), and the kernel then faults mid-execution with an out-of-bounds
  shared memory access. The asynchronous fault is then surfaced by the NEXT
  CUDA Driver API call (cuModuleLoadData) as CUDA_ERROR_ILLEGAL_INSTRUCTION
  (700), producing the misleading "cuModuleLoadData failed: 700" log line
  that has nothing to do with the cubin data being loaded.

Shared-memory size embedding
----------------------------
For every cubin, the generated header also emits a
`kCONVROT_INT8_KERNEL_{CONFIG}_SHARED` constant holding the kernel's
actual dynamic shared-memory requirement (in bytes), as reported by
`ccinfo.metadata.shared`. The C++ plugin reads this constant and passes
it to cuLaunchKernel as `sharedMemBytes`, instead of blindly passing the
device's opt-in ceiling. This (a) avoids over-allocating shared memory
for small kernels and (b) makes the runtime check
  `m_shared_bytes <= getDeviceMaxDynamicSharedMem()`
a real guard instead of a tautology.

Cubin naming
------------
  kCONVROT_INT8_KERNEL_{TILE}_G{GROUP}_{BIAS|NOBIAS}_{DTYPE}
  e.g. kCONVROT_INT8_KERNEL_128X128_G256_BIAS_FP32IO

If Triton is not importable in the build environment, this script REFUSES
to write a stub header (silent stub cubins cause cuLaunchKernel failures
at runtime).
"""

import os
import sys
from pathlib import Path

# Add current directory to path to allow importing trt_plugins
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Architecture Restriction Sets ──────────────────────────────────────────
# Future-proof hardware definitions. Each target architecture enforces its own
# specific dynamic shared memory limits, default warps, and pipeline stages.
# This makes adapting to sm_75 (Turing) or sm_90 (Hopper) a simple lookup.

class ArchTarget:
    def __init__(self, name: str, sm: int, shared_limit_bytes: int, max_stages: int, default_warps: int):
        self.name = name
        self.sm = sm
        self.shared_limit_bytes = shared_limit_bytes
        self.max_stages = max_stages
        self.default_warps = default_warps

# Define the supported target architectures and their hardware restrictions.
# - sm_75 (Turing): 64 KB dynamic shared memory ceiling.
# - sm_86 (Ampere): 99 KB dynamic shared memory ceiling (opt-in via cuFuncSetAttribute).
# - sm_89 (Ada): 100 KB dynamic shared memory ceiling.
# - sm_90 (Hopper): 228 KB dynamic shared memory ceiling.
ARCH_TARGETS = {
    86: ArchTarget("sm_86", sm=86, shared_limit_bytes=99 * 1024, max_stages=3, default_warps=4),
    75: ArchTarget("sm_75", sm=75, shared_limit_bytes=64 * 1024, max_stages=2, default_warps=4),
    89: ArchTarget("sm_89", sm=89, shared_limit_bytes=100 * 1024, max_stages=3, default_warps=4),
    90: ArchTarget("sm_90", sm=90, shared_limit_bytes=228 * 1024, max_stages=4, default_warps=4),
}

# Currently active build target (sm_86 by default for first-pass validation)
ACTIVE_ARCH = ARCH_TARGETS[86]
SM86_SHARED_LIMIT_BYTES = ACTIVE_ARCH.shared_limit_bytes

# Header version sentinel. The C++ plugin #errors if this is missing or
# lower than expected, which catches the "I forgot to regenerate the header"
# failure mode at C++ compile time rather than at plugin load time.
CONVROT_INT8_CUBIN_HEADER_VERSION = 3

try:
    import triton
    import triton.compiler
    from trt_plugins.convrot_int8_kernel import fused_convrot_gemm_rowwise_kernel, fused_convrot_gemm_rowwise_m1_kernel
    TRITON_AVAILABLE = True
except ImportError as exc:
    TRITON_AVAILABLE = False
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# --- Configuration ----------------------------------------------------------
# Edit this list to control which group sizes get compiled cubins.
GROUP_SIZES = [0, 64, 256]

# Tile preference order: for each (group_size, bias, dtype) config, the
# compiler tries these tiles in order and picks the first one that fits
# within the architecture's shared-memory ceiling with the highest possible
# num_stages. Larger BLOCK_M is preferred for better tensor-core utilization
# on large M; smaller BLOCK_M is used as a fallback to preserve pipelining
# (ns >= 2) when the 128x128 tile would require ns=1.
#
# Rationale: G256 FP32IO 128x128 needs 320 KB at ns=3 (exceeds 99 KB) and
# 192 KB at ns=2 (still exceeds). Only ns=1 fits (64 KB), but ns=1 kills
# software pipelining, making the GEMM memory-bound and ~3-5x slower.
# By trying 64x128 (96 KB at ns=2) we preserve 2-stage pipelining with only
# a 2x increase in grid blocks — a much better tradeoff than ns=1.
TILE_PREFERENCE = [
    (128, 128),  # Best for large M, but may not fit with ns >= 2 for FP32IO G256
    (64,  128),  # Good fallback: 2x more blocks, but ns=2 fits for G256 FP32IO
    (32,  128),  # Last resort: 4x more blocks, but ns=3 fits for G256 FP32IO
]

# Legacy tile config name for the header symbol. We always emit the cubin
# under the "128X128" name for backward compatibility with the C++ lookup
# macros, but the actual BLOCK_M may differ. The true BLOCK_M/BLOCK_N are
# emitted as _BLOCK_M / _BLOCK_N constants that the C++ plugin reads at
# runtime.
TILE_NAME = "128X128"

# Bias options: (suffix, has_bias)
BIAS_CONFIGS = [
    ("NOBIAS", False),
    ("BIAS",   True),
]

# Dtype variants: (suffix, input_fp16, output_fp16).
# The DiT exporter (tools/onnx-export/export_dit.py) emits BOTH FP16 and FP32
# plugin boundary dtypes:
#   * FP16 for non-attention layers (default of _w8a8_plugin_boundary_dtype)
#     when HOTSTEP_W8A8_PLUGIN_IO_DTYPE=FP16 is set
#   * FP32 for q/k/v/o/gate/up/down_proj layers (hardcoded in export_dit.py)
#     and for ALL layers when HOTSTEP_W8A8_PLUGIN_IO_DTYPE=FP32 (the default)
#
# Without FP32 variants in the cubin header, the plugin's configurePlugin()
# rejects all FP32-boundary layers at build time, causing the DiT engine build
# to fail with the PluginV3Runner::execute assertion.
DTYPE_CONFIGS = [
    ("FP16IO", True,  True),   # FP16 input, FP16 output
    ("FP32IO", False, False),  # FP32 input, FP32 output (DiT default)
]


def compile_tactic(block_m, block_n, block_k, group_size, has_bias,
                   arch=86, input_fp16=True, output_fp16=True,
                   num_warps=4, num_stages=3):
    """Compile a single kernel tactic using Triton's backend.

    Returns the cubin bytes for the given configuration.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Compilation cannot run.")

    target = triton.backends.compiler.GPUTarget("cuda", arch, 32)

    # NOTE: no H_ptr in the signature — butterfly rotation is H-matrix-free.
    signature = {
        "X_ptr": "*fp16" if input_fp16 else "*fp32",
        "W_ptr": "*i8",
        "Y_ptr": "*fp16" if output_fp16 else "*fp32",
        "W_scale_ptr": "*fp32",
        "Bias_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_xm": "i32",
        "stride_xk": "i32",
        "stride_wn": "i32",
        "stride_wk": "i32",
        "stride_ym": "i32",
        "stride_yn": "i32"
    }

    constants = {
        "BLOCK_M": int(block_m),
        "BLOCK_N": int(block_n),
        "BLOCK_K": int(block_k),
        "GROUP_SIZE": int(group_size),
        "HAS_BIAS": bool(has_bias),
        "INPUT_FP16": bool(input_fp16),
        "OUTPUT_FP16": bool(output_fp16)
    }

    # ASTSource kwarg was renamed in Triton 3.3:
    #   Triton <= 3.2.x : constants=
    #   Triton >= 3.3.x : constexprs=
    # Accept either so the script keeps working across versions.
    try:
        src = triton.compiler.ASTSource(
            fn=fused_convrot_gemm_rowwise_kernel,
            constexprs=constants,
            signature=signature
        )
    except TypeError:
        src = triton.compiler.ASTSource(
            fn=fused_convrot_gemm_rowwise_kernel,
            constants=constants,
            signature=signature
        )

    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    ccinfo = triton.compile(src, target=target, options=options.__dict__)

    return ccinfo.asm["cubin"], ccinfo.metadata.shared, ccinfo.metadata.num_stages


def compile_tactic_with_auto_stages(block_m, block_n, block_k, group_size, has_bias,
                                    arch=86, input_fp16=True, output_fp16=True,
                                    num_warps=None, max_stages=None,
                                    shared_limit_bytes=None):
    """Compile a single kernel tactic, auto-reducing num_stages to fit the target
    architecture's per-block shared-memory opt-in limit.

    Triton 3.7.1 does NOT auto-reduce num_stages when the compiled kernel's
    dynamic shared memory would exceed the device limit. For the FP32-IO G256
    128x128 configuration, num_stages=3 bakes 320 KB of dynamic shared memory
    into the cubin — 3x the sm_86 opt-in ceiling of 99 KB. The kernel then
    faults at runtime when it accesses shared memory beyond the opt-in
    ceiling, surfacing as a misleading CUDA_ERROR_ILLEGAL_INSTRUCTION (700)
    on the next CUDA Driver API call.

    This helper dynamically looks up the target architecture's restrictions,
    tries num_stages = max_stages, max_stages-1, ..., 1 and picks the largest
    value whose compiled cubin's `metadata.shared` does not exceed the limit.
    For configurations that fit at max_stages (e.g. FP16-IO, G0, G64, all M1
    variants), this is a no-op and the cubin is identical to what compile_tactic()
    would have produced. Only the overflowing configurations are downgraded.

    Returns a tuple (cubin_bytes, shared_bytes, chosen_num_stages).
    """
    arch_target = ARCH_TARGETS.get(arch, ACTIVE_ARCH)
    if num_warps is None:
        num_warps = arch_target.default_warps
    if max_stages is None:
        max_stages = arch_target.max_stages
    if shared_limit_bytes is None:
        shared_limit_bytes = arch_target.shared_limit_bytes

    last_error = None
    for ns in range(max_stages, 0, -1):
        try:
            cubin, shared, actual_ns = compile_tactic(
                block_m, block_n, block_k, group_size, has_bias,
                arch=arch, input_fp16=input_fp16, output_fp16=output_fp16,
                num_warps=num_warps, num_stages=ns,
            )
        except Exception as exc:
            last_error = exc
            continue
        # Triton may auto-reduce num_stages further than what we requested
        # (notably for FP16-IO G256, where it always returns ns=1 regardless
        # of the request). Use the actual value reported by the metadata.
        if shared <= shared_limit_bytes:
            return cubin, shared, actual_ns
        # Otherwise try a smaller num_stages.
    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"compile_tactic_with_auto_stages: no num_stages in [1, {max_stages}] "
        f"fits within {shared_limit_bytes} bytes for "
        f"(block_m={block_m}, block_n={block_n}, block_k={block_k}, "
        f"group_size={group_size}, has_bias={has_bias}, "
        f"input_fp16={input_fp16}, output_fp16={output_fp16}, arch=sm_{arch}). "
        f"Last attempt needed {shared} bytes."
    )


def compile_m1_tactic(block_n, block_k, group_size, has_bias,
                      arch=86, input_fp16=True, output_fp16=True,
                      num_warps=4, num_stages=3):
    """Compile the Triton-only M==1 specialized kernel."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Compilation cannot run.")

    target = triton.backends.compiler.GPUTarget("cuda", arch, 32)
    signature = {
        "X_ptr": "*fp16" if input_fp16 else "*fp32",
        "W_ptr": "*i8",
        "Y_ptr": "*fp16" if output_fp16 else "*fp32",
        "W_scale_ptr": "*fp32",
        "Bias_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_xm": "i32",
        "stride_xk": "i32",
        "stride_wn": "i32",
        "stride_wk": "i32",
        "stride_ym": "i32",
        "stride_yn": "i32"
    }
    constants = {
        "BLOCK_N": int(block_n),
        "BLOCK_K": int(block_k),
        "GROUP_SIZE": int(group_size),
        "HAS_BIAS": bool(has_bias),
        "INPUT_FP16": bool(input_fp16),
        "OUTPUT_FP16": bool(output_fp16)
    }
    try:
        src = triton.compiler.ASTSource(
            fn=fused_convrot_gemm_rowwise_m1_kernel,
            constexprs=constants,
            signature=signature
        )
    except TypeError:
        src = triton.compiler.ASTSource(
            fn=fused_convrot_gemm_rowwise_m1_kernel,
            constants=constants,
            signature=signature
        )
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    return ccinfo.asm["cubin"], ccinfo.metadata.shared, ccinfo.metadata.num_stages


def compile_m1_tactic_with_auto_stages(block_n, block_k, group_size, has_bias,
                                       arch=86, input_fp16=True, output_fp16=True,
                                       num_warps=None, max_stages=None,
                                       shared_limit_bytes=None):
    """M1 variant of compile_tactic_with_auto_stages.

    The M1 kernel uses elementwise multiply + reduce instead of tl.dot, so
    its shared-memory footprint is tiny (1–2 KB) and num_stages=3 always
    fits. This wrapper exists for API symmetry with the 128x128 path so
    the C++ header layout is identical across both tile families.
    """
    arch_target = ARCH_TARGETS.get(arch, ACTIVE_ARCH)
    if num_warps is None:
        num_warps = arch_target.default_warps
    if max_stages is None:
        max_stages = arch_target.max_stages
    if shared_limit_bytes is None:
        shared_limit_bytes = arch_target.shared_limit_bytes

    last_error = None
    for ns in range(max_stages, 0, -1):
        try:
            cubin, shared, actual_ns = compile_m1_tactic(
                block_n, block_k, group_size, has_bias,
                arch=arch, input_fp16=input_fp16, output_fp16=output_fp16,
                num_warps=num_warps, num_stages=ns,
            )
        except Exception as exc:
            last_error = exc
            continue
        if shared <= shared_limit_bytes:
            return cubin, shared, actual_ns
    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"compile_m1_tactic_with_auto_stages: no num_stages in [1, {max_stages}] "
        f"fits within {shared_limit_bytes} bytes for (block_n={block_n}, "
        f"block_k={block_k}, group_size={group_size}, has_bias={has_bias}, "
        f"input_fp16={input_fp16}, output_fp16={output_fp16}, arch=sm_{arch})."
    )


def compile_best_tactic(block_k, group_size, has_bias,
                        arch=86, input_fp16=True, output_fp16=True,
                        tile_preference=None,
                        num_warps=None, max_stages=None,
                        shared_limit_bytes=None):
    """Try multiple tile sizes and pick the one with the best pipelining.

    For each tile in tile_preference (largest BLOCK_M first), try num_stages
    from max_stages down to 1. Return the first (tile, ns) combo where the
    compiled cubin's shared memory fits within shared_limit_bytes, preferring
    larger BLOCK_M and larger num_stages.

    This is the key performance optimization: for G256 FP32IO, the 128x128
    tile only fits with ns=1 (64 KB, no pipelining). By trying 64x128, we
    get ns=2 (96 KB, 2-stage pipeline) — a 2-3x kernel speedup with only
    a 2x increase in grid blocks. For G256 FP16IO, 128x128 ns=3 (64 KB)
    fits, so the larger tile is preferred.

    Returns a tuple (cubin_bytes, shared_bytes, chosen_num_stages,
                     block_m, block_n).
    """
    arch_target = ARCH_TARGETS.get(arch, ACTIVE_ARCH)
    if num_warps is None:
        num_warps = arch_target.default_warps
    if max_stages is None:
        max_stages = arch_target.max_stages
    if shared_limit_bytes is None:
        shared_limit_bytes = arch_target.shared_limit_bytes
    if tile_preference is None:
        tile_preference = TILE_PREFERENCE

    best_result = None  # (cubin, shared, ns, block_m, block_n)
    best_score = -1     # higher = better; score = num_stages * 100 + block_m

    for block_m, block_n in tile_preference:
        for ns in range(max_stages, 0, -1):
            try:
                cubin, shared, actual_ns = compile_tactic(
                    block_m, block_n, block_k, group_size, has_bias,
                    arch=arch, input_fp16=input_fp16, output_fp16=output_fp16,
                    num_warps=num_warps, num_stages=ns,
                )
            except Exception:
                continue
            if shared <= shared_limit_bytes:
                score = actual_ns * 100 + block_m
                if score > best_score:
                    best_score = score
                    best_result = (cubin, shared, actual_ns, block_m, block_n)
                break  # found the best ns for this tile; try next tile

    if best_result is not None:
        return best_result

    # Last resort: ns=1 with the smallest tile
    block_m, block_n = tile_preference[-1]
    cubin, shared, actual_ns = compile_tactic(
        block_m, block_n, block_k, group_size, has_bias,
        arch=arch, input_fp16=input_fp16, output_fp16=output_fp16,
        num_warps=num_warps, num_stages=1,
    )
    if shared > shared_limit_bytes:
        raise RuntimeError(
            f"compile_best_tactic: even ns=1 with BLOCK_M={block_m} needs "
            f"{shared} bytes > {shared_limit_bytes} limit."
        )
    return cubin, shared, actual_ns, block_m, block_n


def _format_cubin_as_c_array(cubin_bytes: bytes) -> list:
    """Render raw bytes as C++ initializer list lines, wrapped at ~120 chars."""
    tokens = [f"0x{b:02x}" for b in cubin_bytes]
    lines = []
    line = "    "
    for token in tokens:
        if len(line) + len(token) + 2 > 120:
            lines.append(line.rstrip())
            line = "    "
        line += token + ", "
    lines.append(line.rstrip(", "))
    return lines


def main():
    # Output dir is two levels up from this script + engine/src/plugins/assets.
    dest_dir = Path(__file__).resolve().parent.parent.parent / "engine" / "src" / "plugins" / "assets"
    dest_dir.mkdir(parents=True, exist_ok=True)
    header_path = dest_dir / "convrot_int8_kernel_cubin.h"

    if not TRITON_AVAILABLE:
        print(
            "[compile_triton_plugin] FATAL: Triton or the kernel module is not "
            "importable. Refusing to write a stub header.\n"
            f"  ImportError: {_IMPORT_ERROR!r}\n"
            "  Fix: pip install triton  and re-run this script.",
            file=sys.stderr,
        )
        return 1

    print(f"[compile_triton_plugin] Compiling Triton Custom Kernel Combinations ({ACTIVE_ARCH.name} active)")
    print(f"  Triton version: {triton.__version__}")
    print(f"  Group sizes: {GROUP_SIZES}")
    print(f"  Tile preference: {TILE_PREFERENCE}")
    print(f"  Dtype variants: {[(s, i, o) for s, i, o in DTYPE_CONFIGS]}")
    print(f"  {ACTIVE_ARCH.name} shared-mem opt-in ceiling: {ACTIVE_ARCH.shared_limit_bytes} bytes ({ACTIVE_ARCH.shared_limit_bytes/1024:.0f} KB)")
    print(f"  Auto tile+num_stages selection: tries tiles in order, picks best ns that fits")

    header_lines = [
        "#pragma once",
        "#include <cstddef>",
        "",
        "// Fused Triton ConvRot INT8 compiled binary sizes and payloads.",
        "// AUTO-GENERATED by tools/onnx-export/compile_triton_plugin_sm86.py - do not edit.",
        "//",
        "// Phase-1 butterfly rotation: H_{GROUP_SIZE} applied via in-register",
        "// H_4 Kronecker butterfly stages. No H matrix in shared memory.",
        f"// Group sizes compiled: {GROUP_SIZES}",
        f"// Tile preference: {TILE_PREFERENCE}",
        f"// Dtype variants compiled: {[(s, 'in_fp16=' + str(i), 'out_fp16=' + str(o)) for s, i, o in DTYPE_CONFIGS]}",
        f"// {ACTIVE_ARCH.name} shared-mem opt-in ceiling: {ACTIVE_ARCH.shared_limit_bytes} bytes",
        "//",
        "// Per-cubin (BLOCK_M, num_stages) is auto-selected by trying tiles in",
        "// TILE_PREFERENCE order (largest BLOCK_M first) and picking the first",
        "// that fits within the shared-mem ceiling with the highest num_stages.",
        "// The actual BLOCK_M/BLOCK_N are emitted as _BLOCK_M / _BLOCK_N constants",
        "// and read by the C++ plugin for grid calculation. This avoids the ns=1",
        "// performance cliff for G256 FP32IO by falling back to BLOCK_M=64 with ns=2.",
        "",
        "// Header version sentinel. The C++ plugin #errors if this is missing or",
        "// lower than expected, catching the \"I forgot to regenerate the header\"",
        "// failure mode at C++ compile time rather than at plugin load time.",
        f"#define CONVROT_INT8_CUBIN_HEADER_VERSION {CONVROT_INT8_CUBIN_HEADER_VERSION}",
        "",
        "// Per-dtype availability markers consumed by convrot_int8_linear_plugin.cpp",
        "// to skip symbol references for dtypes that were not compiled (avoids C2065",
        "// undeclared-identifier errors at C++ compile time).",
    ]

    # Emit one #define per dtype variant actually compiled.
    for dtype_suffix, _, _ in DTYPE_CONFIGS:
        header_lines.append(f"#define CONVROT_INT8_HAS_DTYPE_{dtype_suffix} 1")
    header_lines.append("#define CONVROT_INT8_HAS_M1_SPECIALIZED 1")
    header_lines.append("")

    total_cubins = 0
    sm = ACTIVE_ARCH.sm
    for group_size in GROUP_SIZES:
        # BLOCK_K = GROUP_SIZE for ROT configs. For NOROT (group_size=0),
        # use 64 as a default that keeps shared memory small.
        block_k = group_size if group_size > 0 else 64

        for bias_suffix, has_bias in BIAS_CONFIGS:
            for dtype_suffix, input_fp16, output_fp16 in DTYPE_CONFIGS:
                # Symbol naming convention:
                #   kCONVROT_INT8_KERNEL_{TILE_NAME}_G{GROUP}_{BIAS|NOBIAS}_{DTYPE_SUFFIX}
                # The tile name is always "128X128" for backward compatibility,
                # but the actual BLOCK_M may differ (see _BLOCK_M constant).
                config_name = f"{TILE_NAME}_G{group_size}_{bias_suffix}_{dtype_suffix}"
                print(f"  Compiling {config_name} (K={block_k}, group_size={group_size}, "
                      f"bias={has_bias}, in_fp16={input_fp16}, out_fp16={output_fp16}, sm={sm})...")

                try:
                    cubin_bytes, shared_bytes, chosen_ns, chosen_bm, chosen_bn = compile_best_tactic(
                        block_k, group_size, has_bias,
                        arch=sm, input_fp16=input_fp16, output_fp16=output_fp16,
                    )
                except Exception as exc:
                    print(f"    FAILED: {exc}", file=sys.stderr)
                    return 1

                arch_target = ARCH_TARGETS.get(sm, ACTIVE_ARCH)
                fits = shared_bytes <= arch_target.shared_limit_bytes
                print(f"    -> BLOCK_M={chosen_bm}, BLOCK_N={chosen_bn}, ns={chosen_ns}, "
                      f"shared={shared_bytes} B ({shared_bytes/1024:.1f} KB, "
                      f"{'FITS' if fits else 'EXCEEDS!!'}), cubin={len(cubin_bytes)} B")
                if not fits:
                    print(f"    INTERNAL ERROR: compile_best_tactic returned a cubin that still exceeds "
                          f"the limit. Refusing to write a broken cubin.", file=sys.stderr)
                    return 1

                header_lines.append(
                    f"constexpr size_t kCONVROT_INT8_KERNEL_{config_name}_SIZE = {len(cubin_bytes)};"
                )
                header_lines.append(
                    f"constexpr size_t kCONVROT_INT8_KERNEL_{config_name}_SHARED = {shared_bytes};"
                )
                header_lines.append(
                    f"constexpr int kCONVROT_INT8_KERNEL_{config_name}_BLOCK_M = {chosen_bm};"
                )
                header_lines.append(
                    f"constexpr int kCONVROT_INT8_KERNEL_{config_name}_BLOCK_N = {chosen_bn};"
                )
                header_lines.append(
                    f"alignas(16) constexpr unsigned char kCONVROT_INT8_KERNEL_{config_name}[] = {{"
                )
                header_lines.extend(_format_cubin_as_c_array(cubin_bytes))
                header_lines.append("};")
                header_lines.append("")
                total_cubins += 1

    # M==1 specialized Triton kernels. Same ABI/signature as generic,
    # but one program covers one N tile and reduces over K explicitly.
    # The M1 kernel uses elementwise multiply+reduce (no tl.dot), so its
    # shared-memory footprint is tiny (1-2 KB) and num_stages=3 always fits.
    block_n_m1 = 128
    for group_size in GROUP_SIZES:
        block_k = group_size if group_size > 0 else 64
        for bias_suffix, has_bias in BIAS_CONFIGS:
            for dtype_suffix, input_fp16, output_fp16 in DTYPE_CONFIGS:
                config_name = f"M1_G{group_size}_{bias_suffix}_{dtype_suffix}"
                print(f"  Compiling {config_name} (Ntile={block_n_m1}, Ktile={block_k}, "
                      f"group_size={group_size}, bias={has_bias}, in_fp16={input_fp16}, "
                      f"out_fp16={output_fp16}, sm={sm})...")
                try:
                    cubin_bytes, shared_bytes, chosen_ns = compile_m1_tactic_with_auto_stages(
                        block_n_m1, block_k, group_size, has_bias,
                        arch=sm, input_fp16=input_fp16, output_fp16=output_fp16,
                    )
                except Exception as exc:
                    print(f"    FAILED: {exc}", file=sys.stderr)
                    return 1
                print(f"    -> ns={chosen_ns}, shared={shared_bytes} B "
                      f"({shared_bytes/1024:.1f} KB), cubin={len(cubin_bytes)} B")
                header_lines.append(
                    f"constexpr size_t kCONVROT_INT8_KERNEL_{config_name}_SIZE = {len(cubin_bytes)};"
                )
                header_lines.append(
                    f"constexpr size_t kCONVROT_INT8_KERNEL_{config_name}_SHARED = {shared_bytes};"
                )
                header_lines.append(
                    f"constexpr int kCONVROT_INT8_KERNEL_{config_name}_BLOCK_M = 1;"
                )
                header_lines.append(
                    f"constexpr int kCONVROT_INT8_KERNEL_{config_name}_BLOCK_N = {block_n_m1};"
                )
                header_lines.append(
                    f"alignas(16) constexpr unsigned char kCONVROT_INT8_KERNEL_{config_name}[] = {{"
                )
                header_lines.extend(_format_cubin_as_c_array(cubin_bytes))
                header_lines.append("};")
                header_lines.append("")
                total_cubins += 1

    header_lines.append("// End of generated Triton cubins.")

    header_path.write_text("\n".join(header_lines) + "\n")
    print(f"\n[compile_triton_plugin] SUCCESS! Wrote {total_cubins} cubins to {header_path}")
    print(f"  Total header size: {header_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
