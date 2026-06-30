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
CONVROT_INT8_CUBIN_HEADER_VERSION = 5

try:
    import triton
    import triton.compiler
    from trt_plugins.convrot_int8_kernel import (
        kernel1_convrot_quant,
        kernel2_gemm_dequant,
    )
    TRITON_AVAILABLE = True
except ImportError as exc:
    TRITON_AVAILABLE = False
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# Only the production group size is compiled. Do not re-add 0 or 256 for
# compatibility without also re-evaluating exporter behavior and engine JSONs.
GROUP_SIZES = [64]

# Profiling picked this exact pair as the default winner for both M==1 and M>1:
#   two_kernel_bk64_bm128_bn128
# Keep the fallback entries for compiler/shared-memory portability, but the
# first entry should be selected on sm86 with Triton 3.7.1.
TILE_PREFERENCE_K1 = [
    (128, 64),
    (64, 64),
]
TILE_PREFERENCE_K2 = [
    (128, 128),
    (64, 128),
    (32, 128),
]

TILE_NAME = "128X128"
BIAS_CONFIGS = [("NOBIAS", False), ("BIAS", True)]
DTYPE_CONFIGS = [
    ("FP16IO", True, True),
    ("FP32IO", False, False),
]


def block_k_for_group_size(group_size: int) -> int:
    """Return the activation quantization tile width for a given group size."""
    if group_size != 64:
        raise ValueError(f"Only group_size=64 is compiled, got {group_size}")
    return 64


def _make_ast_source(fn, signature, constants):
    """Triton renamed ``constants=`` to ``constexprs=`` in newer releases."""
    try:
        return triton.compiler.ASTSource(fn=fn, constexprs=constants, signature=signature)
    except TypeError:
        return triton.compiler.ASTSource(fn=fn, constants=constants, signature=signature)


def _compile(fn, signature, constants, *, arch: int, num_warps: int, num_stages: int, maxnreg=None):
    """AOT-compile a Triton kernel.

    In Triton 3.7.1, AOT and JIT use the SAME compilation pipeline
    (ast_to_ttir → ttgir → llir → ptx → cubin).  The only difference is
    who picks num_warps/num_stages: JIT defaults to (4, 3) from CUDAOptions;
    AOT requires explicit values.  There is NO automatic heuristic in JIT
    either — the benchmark's @triton.autotune is what sweeps configs.

    maxnreg: if set, passed to PTX as .maxnreg to cap register usage and
             force the compiler to spill to local memory instead of maxing
             out at 255 regs (which tanks occupancy to 8.3%).
    """
    target = triton.backends.compiler.GPUTarget("cuda", arch, 32)
    src = _make_ast_source(fn, signature, constants)
    backend = triton.compiler.make_backend(target)
    opts_dict = {"num_warps": num_warps, "num_stages": num_stages}
    if maxnreg is not None:
        opts_dict["maxnreg"] = maxnreg
    options = backend.parse_options(opts_dict)
    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    # Parse register count from PTX (.reg directive)
    num_regs = _parse_ptx_num_regs(ccinfo.asm.get("ptx", ""))
    return ccinfo.asm["cubin"], ccinfo.metadata.shared, ccinfo.metadata.num_stages, ccinfo.metadata.num_warps, num_regs


def _parse_ptx_num_regs(ptx_src: str) -> int:
    """Extract the register count from PTX .reg directive.

    PTX contains lines like:
        .reg .b32 %r<225>;
    or
        .reg .pred %p<32>;
    The total register count is the sum of all .reg declarations,
    but for occupancy estimation the .b32 count is what matters
    (predicates use a separate register file on sm_86).
    """
    import re
    max_reg = 0
    for m in re.finditer(r'\.reg\s+\.b\d+\s+%r<(\d+)>', ptx_src):
        max_reg = max(max_reg, int(m.group(1)))
    if max_reg == 0:
        # Fallback: try .reg .b32 %rh<n> or any .reg with %r
        for m in re.finditer(r'%r<(\d+)>', ptx_src):
            max_reg = max(max_reg, int(m.group(1)))
    return max_reg


def compile_kernel1(block_m, block_k, group_size, arch=86, input_fp16=True, num_warps=4, num_stages=3, maxnreg=None):
    signature = {
        "X_ptr": "*fp16" if input_fp16 else "*fp32",
        "X_q_ptr": "*i8",
        "X_scale_ptr": "*fp32",  # was *fp16 — FP32 for quality
        "M": "i32",
        "K": "i32",
        "stride_xm": "i32",
        "stride_xk": "i32",
        "stride_xqm": "i32",
        "stride_xqk": "i32",
        "stride_xsm": "i32",
        "stride_xsg": "i32",
    }
    constants = {
        "BLOCK_M": int(block_m),
        "BLOCK_K": int(block_k),
        "GROUP_SIZE": int(group_size),
        "INPUT_FP16": bool(input_fp16),
    }
    return _compile(
        kernel1_convrot_quant,
        signature,
        constants,
        arch=arch,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def compile_kernel2(block_m, block_n, block_k, has_bias, arch=86, output_fp16=True, num_warps=4, num_stages=3, group_m=4, maxnreg=None):
    signature = {
        "X_q_ptr": "*i8",
        "X_scale_ptr": "*fp32",  # was *fp16 — FP32 for quality
        "W_q_ptr": "*i8",
        "W_scale_ptr": "*fp32",
        "Bias_ptr": "*fp32",
        "Y_ptr": "*fp16" if output_fp16 else "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_xqm": "i32",
        "stride_xqk": "i32",
        "stride_xsm": "i32",
        "stride_xsg": "i32",
        "stride_wn": "i32",
        "stride_wk": "i32",
        "stride_ym": "i32",
        "stride_yn": "i32",
    }
    constants = {
        "BLOCK_M": int(block_m),
        "BLOCK_N": int(block_n),
        "BLOCK_K": int(block_k),
        "GROUP_M": int(group_m),
        "HAS_BIAS": bool(has_bias),
        "OUTPUT_FP16": bool(output_fp16),
    }
    return _compile(
        kernel2_gemm_dequant,
        signature,
        constants,
        arch=arch,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def _select_kernel1_variant(group_size: int, input_fp16: bool):
    block_k = block_k_for_group_size(group_size)
    last_exc = None
    for block_m, candidate_block_k in TILE_PREFERENCE_K1:
        if candidate_block_k != block_k:
            continue
        try:
            cubin, shared, num_stages, num_warps, num_regs = compile_kernel1(
                block_m,
                candidate_block_k,
                group_size,
                arch=ACTIVE_ARCH.sm,
                input_fp16=input_fp16,
            )
        except Exception as exc:  # pragma: no cover - best-effort diagnostics path
            last_exc = exc
            print(
                f"    !! FAILED BLOCK_M={block_m}, BLOCK_K={candidate_block_k}: "
                f"{exc.__class__.__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if shared <= ACTIVE_ARCH.shared_limit_bytes:
            print(
                f"    -> SELECTED BLOCK_M={block_m}, nw={num_warps}, ns={num_stages}, "
                f"shared={shared} B ({shared/1024:.1f} KB), regs={num_regs}/thread, cubin={len(cubin)} B",
                flush=True,
            )
            return cubin, shared, num_stages, num_warps, num_regs, block_m, candidate_block_k
        print(
            f"    !! REJECTED BLOCK_M={block_m}, BLOCK_K={candidate_block_k}: "
            f"shared={shared} exceeds limit {ACTIVE_ARCH.shared_limit_bytes}",
            file=sys.stderr,
            flush=True,
        )

    print(
        f"    -> Falling back to BLOCK_M=64, BLOCK_K={block_k}, num_stages=1",
        flush=True,
    )
    try:
        cubin, shared, num_stages, num_warps, num_regs = compile_kernel1(
            64,
            block_k,
            group_size,
            arch=ACTIVE_ARCH.sm,
            input_fp16=input_fp16,
            num_stages=1,
        )
    except Exception as exc:  # pragma: no cover - best-effort diagnostics path
        print(
            f"    !! FALLBACK FAILED BLOCK_M=64, BLOCK_K={block_k}, num_stages=1: "
            f"{exc.__class__.__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(f"kernel1 compile failed for group_size={group_size}") from (last_exc or exc)
    if shared > ACTIVE_ARCH.shared_limit_bytes:
        raise RuntimeError(
            f"kernel1 shared memory {shared} exceeds {ACTIVE_ARCH.shared_limit_bytes} bytes "
            f"for group_size={group_size}"
        )
    return cubin, shared, num_stages, num_warps, num_regs, 64, block_k


def _select_kernel2_variant(group_size: int, has_bias: bool, output_fp16: bool):
    """Compile K2 with the best (num_warps, num_stages, GROUP_M, maxnreg) config.

    Post-fix profiling (report9.jsonl) showed that even with num_warps=8,
    num_stages=2, the compiler STILL produces 255 regs/thread.  This caps
    occupancy at 16.7% (1 block/SM) because 255×256=65280 ≈ 65536 reg limit.

    The fix: force maxnreg=128 via the PTX .maxnreg directive.  This makes
    the compiler spill to local memory instead of maxing registers, trading
    some ILP for 2x occupancy (16.7% → 33.3% when shared ≤ 49 KB).

    Candidate ordering: try maxnreg-capped configs first (best occupancy),
    then uncapped fallbacks.
    """
    block_k = block_k_for_group_size(group_size)
    last_exc = None

    # (block_m, block_n, num_warps, num_stages, group_m, maxnreg)
    # maxnreg=None means let the compiler decide (produces 255 regs — bad).
    # maxnreg=128 forces 2 blocks/SM when shared ≤ 49 KB.
    CANDIDATES = [
        # maxnreg-capped: target 33% occupancy (2 blocks/SM)
        (128, 128, 8, 1, 4, 128),   # 1 stage → ~32 KB shared, 128 regs → 2 blocks/SM
        (128, 128, 8, 2, 4, 128),   # 2 stages → ~64 KB shared, but 128 regs → still 1 block (shared-limited)
        (128, 128, 4, 2, 4, 128),   # 4 warps, 2 stages, capped regs
        (128, 128, 4, 1, 4, 128),   # 4 warps, 1 stage, capped regs
        # Uncapped fallbacks (may produce 255 regs — low occupancy)
        (128, 128, 8, 2, 4, None),  # current production config (report9)
        (128, 128, 4, 3, 4, None),  # original pre-fix config
        (64, 128, 4, 3, 8, None),   # smaller M-tile
        (64, 128, 2, 5, 8, None),   # flash-attention INT8 pattern
        (32, 128, 4, 3, 8, None),   # last resort
    ]

    for block_m, block_n, num_warps, num_stages, group_m, maxnreg in CANDIDATES:
        try:
            cubin, shared, actual_ns, actual_nw, num_regs = compile_kernel2(
                block_m,
                block_n,
                block_k,
                has_bias,
                arch=ACTIVE_ARCH.sm,
                output_fp16=output_fp16,
                num_warps=num_warps,
                num_stages=num_stages,
                group_m=group_m,
                maxnreg=maxnreg,
            )
        except Exception as exc:
            last_exc = exc
            mr = maxnreg if maxnreg else "auto"
            print(
                f"    !! FAILED BM={block_m} BN={block_n} W={num_warps} S={num_stages} GM={group_m} MR={mr}: "
                f"{exc.__class__.__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if shared <= ACTIVE_ARCH.shared_limit_bytes:
            mr = maxnreg if maxnreg else "auto"
            print(
                f"    -> SELECTED BM={block_m} BN={block_n} W={actual_nw} S={actual_ns} GM={group_m} "
                f"MR={mr} shared={shared} B ({shared/1024:.1f} KB), regs={num_regs}/thread, cubin={len(cubin)} B",
                flush=True,
            )
            # Warn if register usage will tank occupancy.
            # sm_86: 65536 regs/SM, 1536 threads/SM max.
            # occupancy = min(regs_limit, shared_limit, threads_limit) where
            #   regs_limit = 65536 / (num_regs * warps * 32)
            #   shared_limit = 99 * 1024 / shared_bytes
            #   threads_limit = 1536 / (warps * 32)
            threads_per_block = actual_nw * 32
            regs_limit = 65536 // (num_regs * threads_per_block) if num_regs > 0 else 0
            shared_limit = (99 * 1024) // shared if shared > 0 else 0
            thread_limit = 1536 // threads_per_block
            blocks_per_sm = min(regs_limit, shared_limit, thread_limit)
            occupancy = blocks_per_sm * threads_per_block / 1536 * 100
            if occupancy < 25:
                print(
                    f"    !! WARNING: Low occupancy ~{occupancy:.0f}% (regs={num_regs}, "
                    f"shared={shared/1024:.0f}KB → {blocks_per_sm} blocks/SM). "
                    f"Consider maxnreg=128 or fewer stages.",
                    file=sys.stderr,
                    flush=True,
                )
            return cubin, shared, actual_ns, actual_nw, num_regs, block_m, block_n
        mr = maxnreg if maxnreg else "auto"
        print(
            f"    !! REJECTED BM={block_m} BN={block_n} W={num_warps} S={num_stages} GM={group_m} MR={mr}: "
            f"shared={shared} exceeds limit {ACTIVE_ARCH.shared_limit_bytes}",
            file=sys.stderr,
            flush=True,
        )

    raise RuntimeError(
        f"kernel2 compile failed for group_size={group_size}, has_bias={has_bias}: "
        f"all candidates exceeded shared memory limit"
    ) from last_exc


def _format_cubin_as_c_array(cubin_bytes: bytes) -> list[str]:
    """Render raw bytes as C++ initializer list lines, wrapped at ~120 chars."""
    tokens = [f"0x{b:02x}" for b in cubin_bytes]
    lines: list[str] = []
    line = "    "
    for token in tokens:
        if len(line) + len(token) + 2 > 120:
            lines.append(line.rstrip())
            line = "    "
        line += token + ", "
    lines.append(line.rstrip(", "))
    return lines


def main():
    print("[compile_triton_plugin] script entered", flush=True)
    dest_dir = Path(__file__).resolve().parent.parent.parent / "engine" / "src" / "plugins" / "assets"
    dest_dir.mkdir(parents=True, exist_ok=True)
    header_path = dest_dir / "convrot_int8_kernel_cubin.h"

    if not TRITON_AVAILABLE:
        print(
            "[compile_triton_plugin] Triton is not available; refusing to write a stub header.\n"
            f"  ImportError: {_IMPORT_ERROR!r}",
            file=sys.stderr,
        )
        return 1

    print(f"[compile_triton_plugin] Compiling Triton Custom Kernel Combinations ({ACTIVE_ARCH.name} active)", flush=True)
    print(f"  Triton version: {triton.__version__}", flush=True)
    print(f"  Group sizes: {GROUP_SIZES}", flush=True)
    print(f"  K1 tile preference: {TILE_PREFERENCE_K1}", flush=True)
    print(f"  K2 tile preference: {TILE_PREFERENCE_K2}", flush=True)
    print(f"  Dtype variants: {[(s, i, o) for s, i, o in DTYPE_CONFIGS]}", flush=True)
    print(f"  {ACTIVE_ARCH.name} shared-mem opt-in ceiling: {ACTIVE_ARCH.shared_limit_bytes} bytes ({ACTIVE_ARCH.shared_limit_bytes/1024:.0f} KB)", flush=True)

    header_lines = [
        "#pragma once",
        "#include <cstddef>",
        "",
        "// AUTO-GENERATED by tools/onnx-export/compile_triton_plugin_sm86.py.",
        "// Contains only the two-kernel ConvRot path: G64 x {BIAS,NOBIAS} x {FP16IO,FP32IO}.",
        "// The C++ launcher reads the per-cubin SHARED / BLOCK_* constants at runtime.",
        f"#define CONVROT_INT8_CUBIN_HEADER_VERSION {CONVROT_INT8_CUBIN_HEADER_VERSION}",
        "",
    ]
    for dtype_suffix, _, _ in DTYPE_CONFIGS:
        header_lines.append(f"#define CONVROT_INT8_HAS_DTYPE_{dtype_suffix} 1")
    header_lines.append("#define CONVROT_INT8_HAS_M1_SPECIALIZED 0")
    header_lines.append("")

    total_cubins = 0

    # Kernel 1 inventory.
    for group_size in GROUP_SIZES:
        for dtype_suffix, input_fp16, _ in DTYPE_CONFIGS:
            config_name = f"K1_{TILE_NAME}_G{group_size}_{dtype_suffix}"
            print(f"  Compiling {config_name} (group_size={group_size}, in_fp16={input_fp16}, sm={ACTIVE_ARCH.sm})...", flush=True)
            cubin, shared, ns, nw, num_regs, block_m, block_k = _select_kernel1_variant(group_size, input_fp16)
            print(f"    -> BLOCK_M={block_m}, BLOCK_K={block_k}, nw={nw}, ns={ns}, shared={shared} B ({shared/1024:.1f} KB), regs={num_regs}/thread, cubin={len(cubin)} B", flush=True)
            header_lines.append(f"constexpr size_t kCONVROT_INT8_KERNEL1_{config_name}_SIZE = {len(cubin)};")
            header_lines.append(f"constexpr size_t kCONVROT_INT8_KERNEL1_{config_name}_SHARED = {shared};")
            header_lines.append(f"constexpr int kCONVROT_INT8_KERNEL1_{config_name}_BLOCK_M = {block_m};")
            header_lines.append(f"constexpr int kCONVROT_INT8_KERNEL1_{config_name}_BLOCK_K = {block_k};")
            header_lines.append(f"constexpr int kCONVROT_INT8_KERNEL1_{config_name}_NUM_WARPS = {nw};")
            header_lines.append(f"alignas(16) constexpr unsigned char kCONVROT_INT8_KERNEL1_{config_name}[] = {{")
            header_lines.extend(_format_cubin_as_c_array(cubin))
            header_lines.append("};")
            header_lines.append("")
            total_cubins += 1

    # Kernel 2 inventory.
    for group_size in GROUP_SIZES:
        for bias_suffix, has_bias in BIAS_CONFIGS:
            for dtype_suffix, _, output_fp16 in DTYPE_CONFIGS:
                config_name = f"K2_{TILE_NAME}_G{group_size}_{bias_suffix}_{dtype_suffix}"
                print(f"  Compiling {config_name} (group_size={group_size}, bias={has_bias}, out_fp16={output_fp16}, sm={ACTIVE_ARCH.sm})...", flush=True)
                cubin, shared, ns, nw, num_regs, block_m, block_n = _select_kernel2_variant(group_size, has_bias, output_fp16)
                print(f"    -> BLOCK_M={block_m}, BLOCK_N={block_n}, nw={nw}, ns={ns}, shared={shared} B ({shared/1024:.1f} KB), regs={num_regs}/thread, cubin={len(cubin)} B", flush=True)
                header_lines.append(f"constexpr size_t kCONVROT_INT8_KERNEL2_{config_name}_SIZE = {len(cubin)};")
                header_lines.append(f"constexpr size_t kCONVROT_INT8_KERNEL2_{config_name}_SHARED = {shared};")
                header_lines.append(f"constexpr int kCONVROT_INT8_KERNEL2_{config_name}_BLOCK_M = {block_m};")
                header_lines.append(f"constexpr int kCONVROT_INT8_KERNEL2_{config_name}_BLOCK_N = {block_n};")
                header_lines.append(f"constexpr int kCONVROT_INT8_KERNEL2_{config_name}_NUM_WARPS = {nw};")
                header_lines.append(f"alignas(16) constexpr unsigned char kCONVROT_INT8_KERNEL2_{config_name}[] = {{")
                header_lines.extend(_format_cubin_as_c_array(cubin))
                header_lines.append("};")
                header_lines.append("")
                total_cubins += 1

    # M==1 intentionally has no dedicated inventory. The C++ plugin launches
    # the same K1/K2 cubins for all M, including M==1, because profiling showed
    # the two-kernel bk64/bm128/bn128 path is the faster default.

    header_lines.append("// End of generated Triton cubins.")
    header_path.write_text("\n".join(header_lines) + "\n")
    print(f"\n[compile_triton_plugin] SUCCESS! Wrote {total_cubins} cubins to {header_path}", flush=True)
    print(f"  Total header size: {header_path.stat().st_size} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
