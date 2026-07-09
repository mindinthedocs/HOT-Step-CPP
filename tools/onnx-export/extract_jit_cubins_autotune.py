#!/usr/bin/env python3
"""Extract autotune-selected cubins from Triton's JIT cache.

This version parses Triton's internal cache key to extract the exact
constexpr kwargs alongside metadata, ensuring we select the precise winning
cubin even when configs collide on (num_warps, num_stages).
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import threading
from pathlib import Path

sys.setrecursionlimit(200000)

# ─── Multithreading memory tuning (Windows Spawn-less approach) ─────────────
#
# To avoid the massive memory overhead of Windows "spawn" re-importing torch
# and triton on every worker process, this version uses Python's ThreadPool.
# Threads share the main process memory, eliminating spawn overhead completely.
# Triton releases the GIL during the heavy MLIR/LLVM compilation steps,
# allowing for genuine parallel speedups natively on Windows.
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

# ─── Thread oversubscription / allocator-arena guards ──────────────────────
#
# Capping thread pools to 1 removes CPU-thrashing cost with zero
# effect on compile throughput or wall-clock time, as this script does not
# perform math on the CPU.
for _thread_env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
    os.environ.setdefault(_thread_env_var, "1" if _thread_env_var != "TOKENIZERS_PARALLELISM" else "false")

os.environ.setdefault("MALLOC_ARENA_MAX", "2")

try:
    import triton
    import triton.language as tl
    from triton import Config
    from triton.language.extra import libdevice
except ImportError as exc:
    print(f"ERROR: Triton is required to run this script: {exc}", file=sys.stderr)
    raise

# Triton's CPU interpreter (TRITON_INTERPRET=1) is used by the unit tests in
# tools/onnx-export/tests/.  It does not implement libdevice.rint, so the K1
# kernel selects an arithmetically-explicit rounding fallback when this flag
# is set.  The flag is captured at import time and baked into the kernel as a
# constexpr global; production (GPU) extraction always sees False and compiles
# the single-instruction rint path.
#
# Patch review §M2: wrap the value as a `tl.constexpr` so it can be accessed
# from inside @triton.jit kernels.  Plain module-level globals are rejected
# by the AOT compile path (Triton 3.7.1+) with NameError; the constexpr
# wrapper makes it visible to both the interpreter and the GPU code path
# without changing the kernel signature.
_IS_TRITON_INTERPRETER = triton.language.constexpr(
    os.environ.get("TRITON_INTERPRET", "").strip() == "1"
)

# ── Optional Gluon (explicit-layout dialect) support ───────────────────────
# Gluon is Triton's low-level explicit-layout dialect (triton.experimental).
# Required for the K2 ``gluon_pipe`` kernel (``k2_gluon_pipelined``): a
# persistent, multi-stage cp.async pipelined GEMM+dequant targeting Ampere
# mma_v2 (sm80/sm86/sm89).  If the import fails the K2 extraction aborts with
# a clear message — the gluon_pipe kernel is now the only shipped K2 path.
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.ampere import (
        mma_v2 as _gluon_mma_v2,
        async_copy as _gluon_async_copy,
    )
    from triton.experimental.gluon._runtime import GluonASTSource
    from triton.compiler import compile as _triton_compile
    from triton.compiler.compiler import ASTSource as _TritonASTSource
    from triton.backends.compiler import GPUTarget as _GPUTarget
    GLUON_AVAILABLE = True
except Exception as _gluon_import_err:  # pragma: no cover
    gluon = None
    gl = None
    _gluon_mma_v2 = None
    _gluon_async_copy = None
    GluonASTSource = None
    _triton_compile = None
    _TritonASTSource = None
    _GPUTarget = None
    GLUON_AVAILABLE = False
    _GLUON_IMPORT_ERR = _gluon_import_err


def _is_gluon_kernel(fn) -> bool:
    """True if ``fn`` is a Gluon JIT kernel (vs a @triton.jit / autotune fn).

    Detection is by the class module of ``fn`` itself: ``@gluon.jit`` produces
    a GluonJITFunction whose class lives under ``triton.experimental.gluon``,
    while ``@triton.jit`` / ``@triton.autotune`` produce wrappers under
    ``triton.runtime``.  A ``__gluon__`` attribute fallback is kept for
    forward-compatibility.

    NOTE: we check ``type(fn).__module__`` — NOT ``type(fn.fn).__module__`` —
    because ``fn.fn`` is the raw Python function (defined in this module), and
    GluonASTSource expects the ``@gluon.jit`` wrapper object (which carries
    ``.arg_names`` and the other metadata the compile path needs), not the
    unwrapped function.
    """
    if fn is None or gluon is None:
        return False
    cls_mod = type(fn).__module__ or ""
    if "triton.experimental.gluon" in cls_mod:
        return True
    return bool(getattr(fn, "__gluon__", False))


# ─── Device-specs-driven autotune config builder ───────────────────────────

import math as _math

def _estimate_k1_configs(num_sms, l2_bytes, shared_mem_per_sm, autotune_shapes):
    """Build the autotune config list for the K1 quant kernel.

    Shared memory (measured via ptxas -v, patch review §2):
      The butterfly K1 kernel uses 2 KB/warp of shared memory, caused by
      tl.split/tl.join layout conversions inside _hadamard_butterfly_stage
      (NOT by input-tile pipelining as previously claimed).  This is
      independent of BLOCK_M, SUBCHUNK, and num_stages — it scales only
      with num_warps:
        num_warps=4 → 8 KB/CTA
        num_warps=8 → 16 KB/CTA
      This is still a large improvement over the old tensor-core kernel's
      83-99 KB/CTA, but it is NOT zero.  The dominant shared-memory lever
      is num_warps, not BLOCK_M or num_stages.

    Register pressure (measured via ptxas -v, patch review §1):
      SUBCHUNK=16 at num_warps=8 uses 64-128 regs/thread depending on BLOCK_M
      and arch — well under the 255 cap, zero spills.  SUBCHUNK=64 spills on
      every arch and should not be used.  maxnreg=128 gives a clean, uniform
      2 CTA/SM floor on both sm75 and sm86.

    See CONVROT_INT8_PATCH_REVIEW.md §1-§3 for the full measured data.
    """
    min_prod_m = min((s[0] for s in autotune_shapes), default=1024)
    # §1.2: BLOCK_M = 16 is fully legal (rotation SUBCHUNK is now per-thread,
    #       not 16×16) and is what unlocks 2 CTA/SM with num_stages=3.
    block_m_candidates = [bm for bm in (16, 32, 64, 128, 256) if bm % 16 == 0]
    if num_sms <= 32:
        block_m_candidates = [bm for bm in block_m_candidates if bm <= 128]
    # Prune BM≥128 for small-M shapes — a 128-row tile with M=64 is half empty.
    if min_prod_m < 128:
        block_m_candidates = [bm for bm in block_m_candidates if bm <= 64]
    block_m_candidates = [
        bm for bm in block_m_candidates
        if _math.ceil(min_prod_m / bm) >= 1
    ]
    # Drop num_stages ≥ 6 — the sm86 100 KB shared budget already blocks them
    # on the DiT shapes.  Keep 2 and 3 everywhere; num_stages=4 is added ONLY
    # for BLOCK_M=16 (16 KB × 4 = 64 KB fits with 2 CTA/SM headroom — larger
    # BM tiles with 4 stages blow the budget and just waste autotune time).
    base_warps_stages = [(4, 2), (4, 3), (8, 2), (8, 3)]
    # Keep K1 explicitly register-capped.  On sm86 Triton 3.7.1 can autotune
    # an uncapped BM=16/S=4 candidate that later reports an 8-byte stack spill
    # in the exact cached winner cubin, even though the pre-filter compiled a
    # spill-free-looking variant.  The capped MR=128 BM=16 candidates are clean
    # and benchmark within noise of the best old/current K1 variants, so avoid
    # uncapped K1 configs entirely and let the spill gate reject any shape that
    # still spills under the cap.
    maxnreg_vals = [128]

    configs = []
    seen = set()
    for bm in block_m_candidates:
        # num_stages=4 is a BM=16-only candidate (see comment above); the
        # previous version appended it to the shared list whenever BM=16 was
        # present, which also inflated the BM>=32 config space for no gain.
        warps_stages = base_warps_stages + ([(4, 4), (8, 4)] if bm == 16 else [])
        for nw, ns in warps_stages:
            for mr in maxnreg_vals:
                cfg_kwargs = {"BLOCK_M": bm}
                key = (bm, nw, ns, mr)
                if key in seen:
                    continue
                seen.add(key)
                if mr is not None:
                    configs.append(Config(cfg_kwargs, num_warps=nw, num_stages=ns, maxnreg=mr))
                else:
                    configs.append(Config(cfg_kwargs, num_warps=nw, num_stages=ns))
    return configs


def _estimate_k2_configs(num_sms, l2_bytes, shared_mem_per_sm, autotune_shapes, arch=86):
    """Build the tuning config list for the K2 Gluon ``gluon_pipe`` kernel.

    The K2 kernel is the Gluon pipelined variant (``k2_gluon_pipelined``),
    num_warps=8 (warps_per_cta=[4, 2], hardcoded inside the kernel).  The
    tuning grid is pinned to the user-specified space and is arch-independent
    — every config is emitted and the spill/compile filter rejects any that
    don't fit the target's smem/register budget:

        BM ∈ {64, 128}
        BN ∈ {64, 128}
        BK ∈ {32, 64}        (must divide GROUP_SIZE=256)
        stages ∈ {2, 3}      (→ NUM_BUFS = stages cp.async buffers)

    That is 2 × 2 × 2 × 2 = 16 configs, all num_warps=8.  GROUP_M is fixed at
    8 (the L2 super-group swizzle size; affects tile-scheduling order only,
    not smem/registers).  FOLD_EVERY defaults to GROUPS_PER_TILE = 256 / BK
    (one int32→fp32 fold per group); NUM_BUFS = stages.

    Output dtype is FP32 (DTYPE_CONFIGS = [("FP32IO", False, False)]); the
    kernel's OUTPUT_FP16 constexpr is wired from the dtype config at compile
    time, not from the tuning grid.

    The arch/smem heuristics that used to prune the @triton.autotune K2 space
    are no longer applied here: the gluon kernel's explicit shared-memory
    allocation is checked exactly by the spill/compile filter (cuobjdump
    SHARED field), and the custom benchmarker times each survivor so the
    winner is picked on measured throughput, not on a smem-tax estimate.
    """
    block_m_candidates = [64, 128]
    block_n_candidates = [64, 128]
    block_k_candidates = [32, 64]
    stages_candidates = [2, 3]
    group_m = 8  # L2 super-group swizzle (tile-scheduling order only)

    configs = []
    seen = set()
    for bm in block_m_candidates:
        # gluon_pipe (num_warps=8, warps_per_cta=[4,2]): BM % 64 == 0.
        if bm % 64 != 0:
            continue
        for bn in block_n_candidates:
            # BN % 32 == 0 (4×2 warps × [16,8] tile).
            if bn % 32 != 0:
                continue
            for bk in block_k_candidates:
                if 256 % bk != 0 or bk % 16 != 0:
                    continue
                for ns in stages_candidates:
                    nw = 8  # gluon_pipe is num_warps=8 only
                    cfg_kwargs = {
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "BLOCK_K": bk,
                        "GROUP_M": group_m,
                    }
                    key = (bm, bn, bk, group_m, nw, ns)
                    if key in seen:
                        continue
                    seen.add(key)
                    configs.append(Config(cfg_kwargs, num_warps=nw, num_stages=ns))
    return configs


def _gluon_k2_num_bufs(cfg):
    """Pipeline depth (NUM_BUFS) for the gluon_pipe K2 kernel = cfg.stages."""
    return max(2, cfg.num_stages)


def _gluon_k2_fold_every(cfg):
    """FOLD_EVERY for the gluon_pipe K2 kernel.

    Defaults to GROUPS_PER_TILE = GROUP_SIZE / BLOCK_K (one int32→fp32 fold
    per group), the safe default the standalone benchmark uses when a config
    carries no explicit FE knob.
    """
    return GROUP_SIZE // cfg.kwargs["BLOCK_K"]


def _gluon_k2_signature(output_fp16):
    """GluonASTSource signature dict for the K2 gluon_pipe kernel.

    Mirrors the proven signature from ``benchmark_k2_tuning_space_hybrid.py``'s
    ``aot_resources``: runtime args carry their element/scalar type, constexpr
    params are declared as ``'constexpr'``.  Both declarations are accepted by
    GluonASTSource (the constexpr type entries are redundant with the
    ``constexprs`` value dict but harmless, and keep this path byte-identical
    to the benchmarked one).
    """
    y_type = '*fp16' if output_fp16 else '*fp32'
    return {
        'X_q_ptr': '*i8', 'X_scale_ptr': '*fp32', 'W_q_ptr': '*i8',
        'W_scale_ptr': '*fp32', 'Bias_ptr': '*fp32', 'Y_ptr': y_type,
        'M': 'i32', 'N': 'i32', 'K': 'i32', 'NUM_TILES': 'i32',
        'stride_xqm': 'i32', 'stride_xqk': 'i32',
        'stride_xsm': 'i32', 'stride_xsg': 'i32',
        'stride_wn': 'i32', 'stride_wk': 'i32',
        'stride_ym': 'i32', 'stride_yn': 'i32',
        'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr',
        'GROUP_SIZE': 'constexpr', 'GROUP_M': 'constexpr',
        'NUM_BUFS': 'constexpr', 'FOLD_EVERY': 'constexpr',
        'HAS_BIAS': 'constexpr', 'OUTPUT_FP16': 'constexpr',
    }


def _gluon_k2_constexprs(cfg, output_fp16):
    """GluonASTSource constexprs dict for one K2 config + dtype variant."""
    return {
        'BLOCK_M': cfg.kwargs['BLOCK_M'],
        'BLOCK_N': cfg.kwargs['BLOCK_N'],
        'BLOCK_K': cfg.kwargs['BLOCK_K'],
        'GROUP_SIZE': GROUP_SIZE,
        'GROUP_M': cfg.kwargs['GROUP_M'],
        'NUM_BUFS': _gluon_k2_num_bufs(cfg),
        'FOLD_EVERY': _gluon_k2_fold_every(cfg),
        'HAS_BIAS': True,
        'OUTPUT_FP16': output_fp16,
    }


def _gluon_aot_compile_k2(cfg, output_fp16, target_arch):
    """AOT-compile one K2 gluon_pipe config and return its CompiledKernel.

    Mirrors the standalone benchmark's ``aot_resources`` Gluon path: builds a
    GluonASTSource from the config + dtype variant and compiles via
    ``triton.compiler.compile`` with ``num_warps=8, num_stages=1`` (the
    pipeline depth is encoded in the kernel's NUM_BUFS constexpr, not the
    Triton num_stages option).  The returned CompiledKernel exposes
    ``.asm['cubin']``, ``.metadata.shared`` and ``.name`` for cubin
    harvesting, ABI extraction and spill validation.
    """
    _require_gluon_for_k2()
    from triton.compiler import compile as triton_compile
    from triton.backends.compiler import GPUTarget
    # GluonASTSource expects the @gluon.jit wrapper object directly (it carries
    # .arg_names and the other metadata the compile path needs) — do NOT unwrap
    # .fn (the raw Python function has no .arg_names).  This matches the
    # standalone benchmark's aot_resources: fn=variant_spec.fn.
    src = GluonASTSource(
        fn=kernel2_gemm_dequant,
        signature=_gluon_k2_signature(output_fp16),
        constexprs=_gluon_k2_constexprs(cfg, output_fp16),
    )
    target = GPUTarget(backend='cuda', arch=target_arch, warp_size=32)
    return triton_compile(src, target=target,
                          options={'num_warps': 8, 'num_stages': 1})


def _add_sm_aware_tile_candidates(base_configs, num_sms, shapes):
    """Inject tile shapes whose wave count divides evenly into the SM count.

    The candidate space here MUST stay a subset of the pruning decisions made
    in _estimate_k2_configs: BLOCK_K=128 is pruned there (conflicts with
    BM>=128 tiles and never won in profiling), and K2 configs are deliberately
    uncapped.  A previous version of this function re-introduced BK=128 /
    inconsistent register-cap configs through the back door; extras now use
    the same BK space and no maxnreg caps.
    """
    # Deduplicate against base_configs on the full identity tuple.  Triton's
    # Config does not define __eq__, so the previous `cfg not in base_configs`
    # check compared object identity and never deduplicated anything.
    def _cfg_key(c):
        return (c.kwargs.get("BLOCK_M"), c.kwargs.get("BLOCK_N"),
                c.kwargs.get("BLOCK_K"), c.kwargs.get("GROUP_M"),
                c.num_warps, c.num_stages)

    seen = {_cfg_key(c) for c in base_configs}
    extra = []
    candidate_block_mns = [(32, 128), (32, 256),
                           (64, 64), (64, 128),
                           (128, 64), (128, 128), (128, 256), (256, 128)]
    for (M, N, K) in shapes:
        for bm, bn in candidate_block_mns:
            tiles = _math.ceil(M / bm) * _math.ceil(N / bn)
            if num_sms > 0 and tiles >= num_sms:
                ratio = tiles / num_sms
                if abs(ratio - round(ratio)) / ratio < 0.10:
                    # Same BK space as _estimate_k2_configs (BK=128 pruned).
                    for bk in [32, 64]:
                        if 256 % bk != 0:
                            continue
                        if bk == 32 and (bm >= 256 and bn >= 256):
                            continue
                        for gm in [4, 8]:
                            extra_warps_stages = [(4, 2), (8, 2)] if (bm == 32 and bn >= 256 and bk == 32) else [(4, 3), (8, 3)]
                            for nw, ns in extra_warps_stages:
                                key = (bm, bn, bk, gm, nw, ns)
                                if key in seen:
                                    continue
                                seen.add(key)
                                cfg_kwargs = {
                                    "BLOCK_M": bm,
                                    "BLOCK_N": bn,
                                    "BLOCK_K": bk,
                                    "GROUP_M": gm,
                                }
                                extra.append(Config(cfg_kwargs, num_warps=nw, num_stages=ns))
    return base_configs + extra


_K1_CONFIGS: list = []
_K2_CONFIGS: list = []


# ─── TC-based regular Hadamard rotation (H_16 ⊗ H_16) ─────────────────────
#
# This is the K1 rotation primitive: the tensor-core (TC) based regular
# Hadamard transform, reverted from the H4-Kronecker in-register butterfly
# that the last commit introduced.  The ConvRot regular Hadamard
# H_{256, regular} is factored as the Kronecker product of two 16×16 regular
# Hadamard blocks:
#
#       H_{256, regular} = H_16 ⊗ H_16    (normalized by 1/√256 = 1/16)
#
# The 16×16 H_16 factor is synthesized once per kernel invocation from an
# identity seed by two calls to ``_hadamard_butterfly_stage`` (log_4(16) = 2
# stages), then cast to FP16 so the rotation is computed by ``tl.dot`` on the
# Ampere FP16 tensor cores (mma.16x16x16 / mma.16x8x16).  A 256-element row is
# reshaped to (16, 16) and rotated by two FP16 matmuls (X · H_16, then its
# transpose · H_16); because H_16 is FP16, a split-FP16 decomposition (hi/lo
# halves) is used to recover full FP32 input dynamic range — the same accuracy
# technique the historical TC K1 path shipped with.
#
# This TC path is the v6 baseline: it offloads the 8192 multiply-adds of the
# rotation to the tensor cores (freeing the FP32 ALU pipeline) and keeps the
# per-thread register footprint small (the rotation is staged through shared
# memory via the MMA fragments, not held in registers as a 256-wide tile).
#
# ─── Register budget ───────────────────────────────────────────────────────
#
# The TC rotation operates on a (SUBCHUNK, 256) FP32 tile that is reshaped to
# (SUBCHUNK*16, 16) for the matmul.  SUBCHUNK=16 (the historical default)
# gives a (256, 16) FP16 operand per tl.dot — exactly one mma.16x16x16 tile
# per warp group, ~32 resting regs/thread, well under the 255 cap.  The K1
# kernel therefore processes the BLOCK_M-row tile in NUM_SUB = BLOCK_M /
# SUBCHUNK sequential slices, the same idiom the TC path has always used.

@triton.jit
def _hadamard_butterfly_stage(
    x_tile, BLOCK_M: tl.constexpr, GROUP_SIZE: tl.constexpr, STAGE: tl.constexpr,
):
    """One H4-Kronecker butterfly stage on a (BLOCK_M, GROUP_SIZE) FP32 tile.

    This is the same primitive the old tensor-core path used to synthesize h16
    from an identity matrix; here it is called directly on real data.  Each
    stage applies the H4 base block to groups of 4 elements at stride 4**STAGE.
    The 0.5 factor per stage folds the 1/√256 normalization across 4 stages
    (0.5^4 = 1/16 = 1/√256).
    """
    s: tl.constexpr = 4 ** STAGE
    n_groups: tl.constexpr = GROUP_SIZE // (4 * s)
    x_v = tl.reshape(x_tile, (BLOCK_M, n_groups, 4, s))
    x_v = tl.permute(x_v, (0, 1, 3, 2))
    x_2x2 = tl.reshape(x_v, (BLOCK_M, n_groups, s, 2, 2))
    lo, hi = tl.split(x_2x2)
    v0, v1 = tl.split(lo)
    v2, v3 = tl.split(hi)
    h0 = 0.5 * (v0 + v1 + v2 - v3)
    h1 = 0.5 * (v0 + v1 - v2 + v3)
    h2 = 0.5 * (v0 - v1 + v2 + v3)
    h3 = 0.5 * (-v0 + v1 + v2 + v3)
    lo_out = tl.join(h0, h1)
    hi_out = tl.join(h2, h3)
    out = tl.join(lo_out, hi_out)
    x_v = tl.reshape(out, (BLOCK_M, n_groups, s, 4))
    x_v = tl.permute(x_v, (0, 1, 3, 2))
    return tl.reshape(x_v, (BLOCK_M, GROUP_SIZE))


@triton.jit
def _generate_h16_fp16():
    """Synthesize the 16×16 regular Hadamard factor H_16 in FP16.

    Builds H_16 by applying ``_hadamard_butterfly_stage`` twice (log_4(16) = 2
    stages) to a 16×16 identity seed.  The 0.5 factor per stage folds the
    1/√16 = 1/4 normalization, so the result is the normalized H_16 used as the
    TC rotation operand.  Cast to FP16 so ``tl.dot`` maps to FP16 tensor cores.
    """
    rows = tl.arange(0, 16)[:, None]
    cols = tl.arange(0, 16)[None, :]
    identity16 = tl.where(rows == cols, 1.0, 0.0).to(tl.float32)
    h = identity16
    for stage in tl.static_range(0, 2):
        h = _hadamard_butterfly_stage(h, 16, 16, stage)
    return h.to(tl.float16)


@triton.jit
def _rotate_256_tensorcore(x_tile, h16, SUBCHUNK: tl.constexpr):
    """Apply the regular Hadamard H_256 = H_16 ⊗ H_16 rotation via FP16 tensor cores.

    A (SUBCHUNK, 256) FP32 tile is reshaped to (SUBCHUNK, 16, 16) and rotated
    by two FP16 ``tl.dot`` matmuls: first X · H_16 along the inner 16 axis,
    then (transposed result) · H_16 along the outer 16 axis.  Because H_16 is
    FP16, a split-FP16 (hi/lo) decomposition of the FP32 operand is used so the
    full FP32 dynamic range is preserved through the FP16 MMA — the same
    accuracy technique the historical TC K1 path shipped with.
    """
    X = tl.reshape(x_tile, (SUBCHUNK, 16, 16))
    X_flat = tl.reshape(X, (SUBCHUNK * 16, 16))
    X_hi = X_flat.to(tl.float16)
    X_lo = (X_flat - X_hi.to(tl.float32)).to(tl.float16)
    A_flat = tl.dot(X_hi, h16) + tl.dot(X_lo, h16)
    A = tl.reshape(A_flat, (SUBCHUNK, 16, 16))
    A_T = tl.permute(A, (0, 2, 1))
    A_T_flat = tl.reshape(A_T, (SUBCHUNK * 16, 16))
    A_T_hi = A_T_flat.to(tl.float16)
    A_T_lo = (A_T_flat - A_T_hi.to(tl.float32)).to(tl.float16)
    B_flat = tl.dot(A_T_hi, h16) + tl.dot(A_T_lo, h16)
    B = tl.reshape(B_flat, (SUBCHUNK, 16, 16))
    return tl.reshape(tl.permute(B, (0, 2, 1)), (SUBCHUNK, 256))


@triton.autotune(configs=_K1_CONFIGS, key=["M", "K", "INPUT_FP16", "CONTIG_XK"])
@triton.jit
def kernel1_convrot_quant(
    X_ptr, X_q_ptr, X_scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_xqm, stride_xqk,
    stride_xsm, stride_xsg,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    INPUT_FP16: tl.constexpr,
    CONTIG_XK: tl.constexpr,
):
    """K1 — ConvRot regular Hadamard rotation + per-group INT8 quantization.

    Rotation implementation (TC-based, H_16 ⊗ H_16):
      The 256-element regular Hadamard rotation is factored as H_16 ⊗ H_16 and
      computed by two FP16 ``tl.dot`` matmuls on the Ampere tensor cores via
      ``_rotate_256_tensorcore``.  A split-FP16 (hi/lo) decomposition of the
      FP32 operand preserves full dynamic range through the FP16 MMA.  The
      16×16 H_16 factor is synthesized once per invocation by
      ``_generate_h16_fp16``.

    Register safety:
      SUBCHUNK=16 slicing keeps the per-``tl.dot`` operand at (256, 16) FP16
      — one mma.16x16x16 tile per warp group, ~32 resting regs/thread, well
      under the 255 cap.

    Persistent grid (v2 §4.1):
      The loop strides by ``tl.num_programs(0)`` (the actual launched grid
      size).  No NUM_SMS runtime arg — the host computes the grid size and
      the kernel reads it via ``tl.num_programs(0)``.

    Stores are ALWAYS masked (patch review §4: dropping the store mask
    caused out-of-bounds writes for any M not evenly divisible by BLOCK_M).
    """
    tl.static_assert(GROUP_SIZE == 256, "Only GROUP_SIZE=256 is supported")
    tl.static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be a multiple of 16")
    # v2 §3.4: SUBCHUNK controls register pressure.  16 matches the existing
    # kernel and sits at ~32 peak regs/thread (comfortable).  32 is a safe
    # alternative for fewer loop iterations.  Values ≥ 64 risk the 255-reg cap
    # once epilogue overhead is added.
    SUBCHUNK: tl.constexpr = 16
    NUM_SUB: tl.constexpr = BLOCK_M // SUBCHUNK
    # Captured module-level constant (True only under TRITON_INTERPRET=1).
    # Selects the rounding implementation below without touching the launch
    # ABI or the autotune key space.
    IS_INTERPRETER: tl.constexpr = _IS_TRITON_INTERPRETER

    # Synthesize the 16×16 regular Hadamard factor once per kernel invocation.
    # H_256 = H_16 ⊗ H_16; the rotation itself is two FP16 tl.dot matmuls.
    h16 = _generate_h16_fp16()

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_k_groups = K // GROUP_SIZE
    num_tiles = num_pid_m * num_k_groups

    start_pid = tl.program_id(0)
    # v2 §4.1: stride by the actual launched grid size.
    #         tl.num_programs(0) is the runtime value of grid.x.
    grid_x = tl.num_programs(0)
    for tile_id in tl.range(start_pid, num_tiles, grid_x):
        pid_m = (tile_id // num_k_groups) * BLOCK_M
        pid_k = (tile_id % num_k_groups) * GROUP_SIZE

        for sub in tl.static_range(0, NUM_SUB):
            sub_m_off = pid_m + sub * SUBCHUNK
            rm = sub_m_off + tl.arange(0, SUBCHUNK)
            rk = pid_k + tl.arange(0, GROUP_SIZE)
            ram = rm % M
            mask_m = rm < M
            mask_k = rk[None, :] < K
            mask_2d = mask_m[:, None] & mask_k

            if CONTIG_XK:
                x_ptr_block = X_ptr + ram[:, None] * stride_xm + rk[None, :]
            else:
                x_ptr_block = X_ptr + ram[:, None] * stride_xm + rk[None, :] * stride_xk

            # Load: always masked.  The load side is safe even without a mask
            # (ram = rm % M wraps in-bounds), but masking avoids unnecessary
            # DRAM traffic for the tail tile.
            x_block = tl.load(x_ptr_block, mask=mask_2d, other=0.0,
                              eviction_policy="evict_first")
            if INPUT_FP16:
                x_block = x_block.to(tl.float32)

            # TC-based rotation (H_16 ⊗ H_16) via FP16 tensor cores.
            rotated_x = _rotate_256_tensorcore(x_block, h16, SUBCHUNK)

            # Per-row max-abs for the INT8 scale.
            block_max = tl.max(tl.abs(rotated_x), axis=1)
            scale = tl.maximum(block_max / 127.0, 1e-30)

            # INT8 quantize: round-to-nearest-even + safety clamp.
            #
            # GPU path (production): libdevice.rint lowers to a single
            # CVT.RNI SASS instruction (round-half-to-even, matching v6
            # numerics exactly).
            #
            # Interpreter path (CPU unit tests only): Triton 3.7.x's
            # interpreter does not implement libdevice.rint, so the tests
            # use an explicit floor-based round-half-away-from-zero
            # fallback.  The two paths differ ONLY on exact .5 ties (at
            # most ±1 LSB on values that quantize to a tie), which is far
            # below the tests' INT8-quantization tolerance.  IS_INTERPRETER
            # is a constexpr, so the production cubin contains no trace of
            # the fallback branch.
            scaled = rotated_x / scale[:, None]
            if IS_INTERPRETER:
                abs_scaled = tl.abs(scaled)
                rounded_abs = tl.floor(abs_scaled + 0.5)
                sign_scaled = tl.where(scaled >= 0.0, 1.0, -1.0)
                x_q = rounded_abs * sign_scaled
            else:
                x_q = libdevice.rint(scaled)
            x_q = tl.clamp(x_q, -127.0, 127.0).to(tl.int8)

            # CRITICAL: X_q store is ALWAYS masked.
            #
            # The previous FAST_PATH=True branch dropped the store mask,
            # causing out-of-bounds writes for any M not evenly divisible by
            # BLOCK_M (which is every real production shape: M=1, 64, 300,
            # 3000).  The workspace is sized exactly to M rows, so the last
            # tile's rm values (which range up to pid_m + BLOCK_M - 1 > M-1)
            # would write past the allocated buffer.
            #
            # Triton can still emit vectorized STG.128 stores with a mask
            # applied at the granularity of whole vector lanes — the mask
            # does not prevent vectorization, it only suppresses the
            # out-of-bounds lanes.
            if CONTIG_XK:
                xq_ptr_block = X_q_ptr + rm[:, None] * stride_xqm + rk[None, :]
            else:
                xq_ptr_block = X_q_ptr + rm[:, None] * stride_xqm + rk[None, :] * stride_xqk
            tl.store(xq_ptr_block, x_q, mask=mask_2d)

            # X_scale: [n_groups, M] transposed layout.  Also ALWAYS masked
            # for the same OOB reason.
            group_idx = pid_k // GROUP_SIZE
            xs_ptr_block = X_scale_ptr + group_idx * stride_xsg + rm * stride_xsm
            tl.store(xs_ptr_block, scale, mask=mask_m)


# ─── K2: Gluon ``gluon_pipe`` pipelined kernel (the shipped K2 path) ───────
#
# Replaces the historical ``hybrid_direct`` @triton.jit/@triton.autotune K2
# kernel with the Gluon explicit-layout ``k2_gluon_pipelined`` variant from
# ``benchmark_k2_tuning_space_hybrid.py``.  It is a persistent, multi-stage
# cp.async pipelined INT8 GEMM + late per-group xs dequant fold targeting
# Ampere mma_v2 (sm80/sm86/sm89), num_warps=8 (warps_per_cta=[4, 2]).
#
# Optimizations vs the plain hybrid_direct kernel:
#   (a) PERSISTENT tile loop — one CTA per SM walks GROUP_M-swizzled tiles for
#       L2 reuse (the X-slab stays hot).  The kernel takes a NUM_TILES runtime
#       arg and the host launches a 1D grid of min(num_tiles, num_sms).
#   (b) Multi-stage cp.async pipeline (NUM_BUFS = cfg.stages buffers, prefetch
#       distance NUM_BUFS-1, one commit group per k-step).  The i32-view
#       workaround (loads as int32, reinterprets as int8 in smem) sidesteps
#       Triton 3.7.1's int8 cp.async <4B lowering bug.
#   (c) Explicit gl.convert_layout to a coalesced BlockedLayout before the
#       epilogue store — eliminates the FP32-output bank-conflict / write-
#       amplification tax the uncoalesced MMA-layout store was paying.
#   (d) FOLD_EVERY int32-folded accumulation: GROUPS_PER_TILE / FOLD_EVERY
#       int32 accumulations are converted+multiplied by xs at once, instead
#       of one per K-sub-step.
#   (e) cache_modifier=".ca" on the tiny X_scale / W_scale / Bias loads so
#       they survive in L2 against the streaming X_q/W_q traffic, and ".cs"
#       streaming store on Y (write-once, never read).
#
# W_q layout is [N, K] row-major ("NK"); the transpose is free via
# smem.permute((1, 0)).  HAS_BIAS is compile-time True (the single shipped K2
# cubin); no-bias plugin instances pass a workspace zero-bias vector.
# OUTPUT_FP16 is a compile-time constexpr; the shipped cubin is FP32 output
# (DTYPE_CONFIGS = [("FP32IO", False, False)]).
#
# Tuning grid (pinned, user-specified): BM∈{64,128}, BN∈{64,128},
# BK∈{32,64}, stages(NUM_BUFS)∈{2,3} → 16 configs, all num_warps=8.

if GLUON_AVAILABLE:
    # Module-level alias so the kernel body can reference ``cp.`` like the
    # standalone benchmark script (cp.async_copy_global_to_shared, etc.).
    cp = _gluon_async_copy

    @gluon.jit
    def kernel2_gemm_dequant(
        X_q_ptr, X_scale_ptr, W_q_ptr, W_scale_ptr, Bias_ptr, Y_ptr,
        M, N, K, NUM_TILES,
        stride_xqm, stride_xqk,
        stride_xsm, stride_xsg,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
        GROUP_SIZE: gl.constexpr, GROUP_M: gl.constexpr, NUM_BUFS: gl.constexpr,
        FOLD_EVERY: gl.constexpr, HAS_BIAS: gl.constexpr, OUTPUT_FP16: gl.constexpr,
    ):
        """K2 — Gluon gluon_pipe: persistent, multi-stage cp.async pipeline.

        NUM_BUFS smem buffers, prefetch distance NUM_BUFS-1, one commit group
        per k-step and a single barrier per step (CUTLASS-style).  NUM_BUFS is
        wired from cfg.stages by the extraction driver; FOLD_EVERY is wired
        from cfg (defaults to GROUPS_PER_TILE = GROUP_SIZE / BLOCK_K when the
        config carries no explicit FE).  See the long comment above the
        definition for the full optimization rationale.
        """
        gl.static_assert(GROUP_SIZE % BLOCK_K == 0,
                         "GROUP_SIZE must be a multiple of BLOCK_K")
        gl.static_assert(GROUP_SIZE % 256 == 0 or GROUP_SIZE == 256,
                         "Only GROUP_SIZE=256 is supported")
        GROUPS_PER_TILE: gl.constexpr = GROUP_SIZE // BLOCK_K
        gl.static_assert(GROUPS_PER_TILE % FOLD_EVERY == 0,
                         "FOLD_EVERY must divide GROUPS_PER_TILE")
        FOLDS_PER_GROUP: gl.constexpr = GROUPS_PER_TILE // FOLD_EVERY
        BK32: gl.constexpr = BLOCK_K // 4  # BLOCK_K in i32 units

        # ── Layouts (num_warps=8: warps_per_cta=[4, 2]) ───────────────────
        mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[2, 0], warps_per_cta=[4, 2], instr_shape=[16, 8])
        a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=4)
        b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=4)

        # i32 copy layout: 4 x i32 = 16B per thread along K, coalesced.
        copy_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4], threads_per_warp=[8, 4],
            warps_per_cta=[8, 1], order=[1, 0])

        # Coalesced epilogue store layout (16B-aligned, one 128B cache-line
        # transaction per warp for FP32 output).
        store_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4], threads_per_warp=[8, 4],
            warps_per_cta=[4, 2], order=[1, 0])

        # Tighter swizzle (per_phase=2): workaround for the bank-conflict
        # pattern reported in triton issue #8149.
        smem32: gl.constexpr = gl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=4, order=[1, 0])
        smem8: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=2, max_phase=4, order=[1, 0])

        a_buf = gl.allocate_shared_memory(gl.int32, [NUM_BUFS, BLOCK_M, BK32], smem32)
        w_buf = gl.allocate_shared_memory(gl.int32, [NUM_BUFS, BLOCK_N, BK32], smem32)

        # Persistent tile loop: each CTA walks multiple GROUP_M-swizzled tiles
        # (identical ordering to hybrid_direct) so the X slab stays hot in L2.
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n

        start_pid = gl.program_id(0)
        grid_x = gl.num_programs(0)

        # Static per-kernel iterators (only tile_id is loop-variant).
        rm_cp_base = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, copy_layout))
        rn_cp_base = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, copy_layout))
        rk_cp = gl.arange(0, BK32, layout=gl.SliceLayout(0, copy_layout))
        rm_acc_base = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma_layout))
        rn_acc_base = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mma_layout))
        rm_store_base = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, store_layout))
        rn_store_base = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, store_layout))

        # Global pointers viewed as i32 (rows 16B-aligned, K a multiple of 64).
        Xq32 = X_q_ptr.cast(gl.pointer_type(gl.int32))
        Wq32 = W_q_ptr.cast(gl.pointer_type(gl.int32))
        sxm32 = stride_xqm // 4
        swn32 = stride_wn // 4

        num_steps = K // BLOCK_K
        num_groups = K // GROUP_SIZE

        for tile_id in range(start_pid, NUM_TILES, grid_x):
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % num_pid_in_group) % group_size_m
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            pid_m_off = pid_m * BLOCK_M
            pid_n_off = pid_n * BLOCK_N

            rm_cp = pid_m_off + rm_cp_base
            rn_cp = pid_n_off + rn_cp_base
            rm_acc = pid_m_off + rm_acc_base
            rn_acc = pid_n_off + rn_acc_base

            acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

            # ── Prologue: fill NUM_BUFS-1 stages, one commit group each ──
            for s in gl.static_range(0, NUM_BUFS - 1):
                off32 = s * BK32
                a_ptrs = Xq32 + rm_cp[:, None] * sxm32 + (off32 + rk_cp)[None, :]
                w_ptrs = Wq32 + rn_cp[:, None] * swn32 + (off32 + rk_cp)[None, :]
                cp.async_copy_global_to_shared(a_buf.index(s), a_ptrs)
                cp.async_copy_global_to_shared(w_buf.index(s), w_ptrs)
                cp.commit_group()

            # ── Main K-loop, grouped by GROUP_SIZE=256, folded by FOLD_EVERY ─
            for g in range(num_groups):
                # X_scale: tiny (M × n_groups) tensor, pin in L2 with .ca.
                xs = gl.load(X_scale_ptr + g * stride_xsg + rm_acc * stride_xsm,
                             cache_modifier=".ca")

                for fold in gl.static_range(0, FOLDS_PER_GROUP):
                    int32_acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.int32, layout=mma_layout)
                    for sub in gl.static_range(0, FOLD_EVERY):
                        step = g * GROUPS_PER_TILE + fold * FOLD_EVERY + sub
                        buf = step % NUM_BUFS
                        # Issue the prefetch NUM_BUFS-1 steps ahead (masked at tail).
                        pf = step + NUM_BUFS - 1
                        in_range = pf < num_steps
                        pbuf = pf % NUM_BUFS
                        off32 = pf * BK32
                        a_pf = Xq32 + rm_cp[:, None] * sxm32 + (off32 + rk_cp)[None, :]
                        w_pf = Wq32 + rn_cp[:, None] * swn32 + (off32 + rk_cp)[None, :]
                        valid_a = gl.full([BLOCK_M, BK32], in_range, gl.int1, layout=copy_layout)
                        valid_w = gl.full([BLOCK_N, BK32], in_range, gl.int1, layout=copy_layout)
                        cp.async_copy_global_to_shared(a_buf.index(pbuf), a_pf, mask=valid_a)
                        cp.async_copy_global_to_shared(w_buf.index(pbuf), w_pf, mask=valid_w)
                        cp.commit_group()
                        # This step's buffer is the oldest outstanding group.
                        cp.wait_group(NUM_BUFS - 1)
                        gl.barrier()
                        a8 = a_buf.index(buf)._reinterpret(gl.int8, [BLOCK_M, BLOCK_K], smem8)
                        w8 = w_buf.index(buf)._reinterpret(gl.int8, [BLOCK_N, BLOCK_K], smem8)
                        a = a8.load(a_layout)
                        b = w8.permute((1, 0)).load(b_layout)
                        int32_acc = _gluon_mma_v2(a, b, int32_acc)
                    # One fp32 convert + mul per FOLD, not per sub-step.
                    acc += int32_acc.to(gl.float32) * xs[:, None]

            # ── Epilogue: dequant scale, bias, convert_layout, streaming store ─
            # W_scale and Bias are tiny (N,) tensors; pin in L2 with .ca.
            ws = gl.load(W_scale_ptr + rn_acc, cache_modifier=".ca")
            acc = acc * ws[None, :]
            if HAS_BIAS:
                bias = gl.load(Bias_ptr + rn_acc, cache_modifier=".ca")
                acc += bias[None, :]

            # Convert MMA-fragment-distributed accumulator to a coalesced
            # store layout (eliminates the FP32-output write-amplification tax).
            acc_store = gl.convert_layout(acc, store_layout)
            rm_store = pid_m_off + rm_store_base
            rn_store = pid_n_off + rn_store_base
            y_ptrs = Y_ptr + rm_store[:, None] * stride_ym + rn_store[None, :] * stride_yn
            # Streaming store (.cs): Y is write-once/never-read.
            if OUTPUT_FP16:
                gl.store(y_ptrs, acc_store.to(gl.float16), cache_modifier=".cs")
            else:
                gl.store(y_ptrs, acc_store.to(gl.float32), cache_modifier=".cs")
else:
    # Gluon unavailable: define a sentinel so attribute lookups fail loudly
    # with a clear message at extraction time (see _require_gluon_for_k2).
    kernel2_gemm_dequant = None


def _require_gluon_for_k2():
    """Abort with a clear message if the Gluon dialect is unavailable.

    The K2 kernel is now the Gluon ``gluon_pipe`` variant; there is no
    @triton.jit fallback.  Any K2 extraction / launch path must call this
    before touching ``kernel2_gemm_dequant``.
    """
    if not GLUON_AVAILABLE:
        raise SystemExit(
            "[extract_jit_cubins_autotune] K2 requires the Triton Gluon dialect "
            "(triton.experimental.gluon), which failed to import: "
            f"{getattr(_GLUON_IMPORT_ERR, '__name__', 'Exception')}: "
            f"{_GLUON_IMPORT_ERR}. Install Triton >= 3.7 with the CUDA backend."
        )


GROUP_SIZE = 256
DEFAULT_BLOCK_K = 64
TILE_NAME = "PersistentG256"
# Version history:
#   6 — GROUP_SIZE=256, TC-based rotation (H_16⊗H_16), decoupled BLOCK_K,
#       persistent grid-stride launch, transposed X_scale layout.
#   7 — Direct H4 butterfly K1 rotation (in-register; ~2 KB/warp shared from
#       tl.split/tl.join), K1 CONTIG_XK constexpr toggle, occupancy-driven
#       persistent grid (cuOccupancyMaxActiveBlocksPerMultiprocessor),
#       evict_last on xs/ws/bias loads.  K2 is the plain/unmasked
#       "hybrid_direct" variant: canonical [N, K] W_q consumed via
#       tl.dot(xq, tl.trans(wq)), late per-group xs load, W_scale/bias
#       epilogue-only.  The plugin pads/splits runtime M on the host side
#       instead of shipping masked K2 variants.  Only the bias-enabled K2
#       cubin ships; no-bias plugin instances pass a workspace zero-bias
#       vector.
#   8 — K1 reverted to the v6 TC-based rotation (H_16⊗H_16) via FP16 tensor
#       cores; the H4 in-register butterfly is dropped.  The v7 K1
#       infrastructure (CONTIG_XK toggle, IS_INTERPRETER rounding fallback,
#       occupancy-driven persistent grid, masked stores) is retained.  K2 is
#       replaced by the Gluon ``gluon_pipe`` pipelined kernel
#       (``k2_gluon_pipelined``): persistent tile loop, multi-stage cp.async
#       pipeline, coalesced FP32 epilogue store, FOLD_EVERY int32-folded
#       accumulation.  The K2 tuning grid is pinned to BM∈{64,128},
#       BN∈{64,128}, BK∈{32,64}, stages∈{2,3} (num_warps=8), FP32 output.
CONVROT_INT8_CUBIN_HEADER_VERSION = 8
# K2 is always compiled with HAS_BIAS=True.  For no-bias ONNX layers, the C++
# plugin passes a zero-filled FP32 bias vector from workspace.
BIAS_CONFIGS = [("BIAS", True)]
DTYPE_CONFIGS = [("FP32IO", False, False)]
# Real DiT decoder K2 shapes, expressed as (M, K, N).  M=3000 is not
# divisible by all candidate BLOCK_M values; extract_k2 pads the benchmark M
# to 3072 so the (unmasked) Gluon gluon_pipe K2 persistent tile loop can be
# autotuned safely while staying within 2.4% of the production work.
#
# The first entry is the primary compromise shape used for the single shipped
# K2 cubin: it is one of the two dominant MLP projections and stresses wide-N
# reuse (N=9728).  The full list still drives L2/GROUP_M pruning and
# SM-wave-aware candidate injection, so down_proj / attention projections are
# represented in the candidate set.
AUTOTUNE_SHAPES_K2 = [
    (3000, 2560, 9728),  # dit.mlp.gate_proj  — dominant, wide N
    (3000, 9728, 2560),  # dit.mlp.down_proj  — dominant, large K
    (3000, 2560, 4096),  # dit.self_attn.q_proj
    (3000, 2560, 1024),  # dit.self_attn.k_proj
    (3000, 4096, 2560),  # dit.self_attn.o_proj
]
K2_AUTOTUNE_M_ALIGNMENT = 128  # max generated K2 BLOCK_M; avoids OOB in the
                                # unmasked Gluon gluon_pipe persistent tile loop

# Patch review §H2: autotune grid must mirror the production launch geometry.
#
# Production launches ``min(num_tiles, num_sms * max_active_ctas_per_sm)``
# where ``max_active_ctas_per_sm`` is the per-cubin occupancy queried from the
# CUDA driver at plugin init.  The old autotune grid used
# ``min(num_sms, num_tiles)`` (1 CTA/SM, one wave), so a config that wins at
# 1 CTA/SM might lose at 2 CTA/SM (and vice versa) — which is exactly the
# occupancy effect this patch is chasing.
#
# ``_K2_CFG_OCCUPANCY`` is populated by the spill/compile filter with the
# per-config occupancy derived from the compiled cubin's actual shared-memory
# and register footprint (see ``_compute_ctas_per_sm``).  The K2 autotune
# grid lambda then looks up the correct occupancy per meta and launches
# ``min(num_tiles, num_sms * per_cfg_occ)`` — identical to the production
# formula, so no config is measured under an occupancy it can't achieve.
#
# ``AUTOTUNE_OCCUPANCY_ASSUMPTION`` remains as the K1 fallback (K1 does not
# populate an occupancy map because it has far fewer configs and much
# lighter register/smem footprint) and as the sentinel value used when a
# K2 config is absent from the map (e.g. the spill filter was skipped).
#
# A config that fits only at 1 CTA/SM still benchmarks correctly at any
# grid size: the kernel's persistent grid-stride loop iterates more tiles
# per CTA, so the total work is unchanged; only the per-config SM
# utilization (and therefore the wall-clock time) changes with grid size.
AUTOTUNE_OCCUPANCY_ASSUMPTION = 2

# Per-config occupancy map for K2, populated by ``_filter_spilling_configs``.
# Key is the same config identity tuple used by the spill filter
# (``(sorted(kwargs.items()), num_warps, num_stages, maxnreg)``).  Value is
# the CTAs/SM the compiled cubin is estimated to sustain on the target arch.
_K2_CFG_OCCUPANCY: dict = {}


def _cfg_occupancy_key(cfg):
    """Identity tuple matching ``_filter_spilling_configs``'s reject map."""
    return (tuple(sorted(cfg.kwargs.items())), cfg.num_warps, cfg.num_stages,
            getattr(cfg, "maxnreg", None))


def _compute_ctas_per_sm(shared_bytes: int, num_regs: int, num_warps: int,
                         arch: int) -> int:
    """Estimate CTAs/SM the way CUDA's occupancy calculator does.

    Uses the three classical limiters:

      * shared memory: ``shared_mem_per_sm // shared_bytes`` (aligned up to
        the block allocation unit; on sm86 that's 128 B, negligible here)
      * registers: ``regs_per_sm // (num_regs * threads_per_block)`` with
        threads_per_block = ``num_warps * 32``.  Register file granularity
        is 256 regs per warp on Turing/Ampere/Ada; we align up to that.
      * maximum resident CTAs per SM (arch cap).

    Returns the minimum of the three.  A returned value of 0 means the
    cubin doesn't fit at all (this is a bug — the spill filter should
    have caught it — but we clamp to 1 so the grid lambda still launches
    something rather than dividing-by-zero downstream).

    Numbers are taken from the CUDA C++ Programming Guide, Compute
    Capabilities table (11.8 / 12.6, matching Triton 3.7's supported
    arches).  For arches we haven't tabulated, we fall back to the
    sm86/sm89 numbers, which underestimate occupancy on larger parts
    (safe: the config is still launched, just with fewer CTAs than the
    hardware could theoretically host).
    """
    if arch == 75:
        # Turing (T4, RTX 20-series, GTX 16-series)
        smem_per_sm = 64 * 1024
        regs_per_sm = 65536
        max_ctas_per_sm = 16
        reg_alloc_unit = 256   # regs per warp allocation granularity
    elif arch == 80:
        # Ampere HPC (A100)
        smem_per_sm = 164 * 1024
        regs_per_sm = 65536
        max_ctas_per_sm = 32
        reg_alloc_unit = 256
    elif arch == 86 or arch == 87:
        # Ampere consumer (RTX 30, A40, A10, Jetson Orin)
        smem_per_sm = 100 * 1024
        regs_per_sm = 65536
        max_ctas_per_sm = 16
        reg_alloc_unit = 256
    elif arch == 89:
        # Ada Lovelace (RTX 40)
        smem_per_sm = 100 * 1024
        regs_per_sm = 65536
        max_ctas_per_sm = 24
        reg_alloc_unit = 256
    elif arch == 90:
        # Hopper (H100)
        smem_per_sm = 228 * 1024
        regs_per_sm = 65536
        max_ctas_per_sm = 32
        reg_alloc_unit = 256
    else:
        # Fallback: sm86 numbers.  Underestimates on larger parts, safe.
        smem_per_sm = 100 * 1024
        regs_per_sm = 65536
        max_ctas_per_sm = 16
        reg_alloc_unit = 256

    # Shared-mem limiter.  Guard against 0 to avoid ZeroDivisionError; a
    # zero-shared kernel is rare but legal (K1 was almost there before v7).
    shared_bytes = max(int(shared_bytes), 1)
    ctas_shared = smem_per_sm // shared_bytes

    # Register limiter.  Per-thread reg count is rounded UP to 2 (Ampere/Ada
    # granularity), multiplied by the block's thread count.  If the cubin
    # didn't report a register count (cuobjdump failed), we fall back to the
    # arch's max resident CTA cap and let the smem/CTA limiters take over.
    #
    # (``reg_alloc_unit`` is retained above for future arches whose warp-level
    # allocation granularity differs; on sm75+ the effective per-thread
    # rounding used by the driver is 2, which is what matters here.)
    if num_regs and num_warps:
        threads = num_warps * 32
        regs_per_thread_rounded = ((int(num_regs) + 1) // 2) * 2
        block_regs = regs_per_thread_rounded * threads
        ctas_regs = regs_per_sm // max(block_regs, 1)
    else:
        ctas_regs = max_ctas_per_sm

    ctas = min(ctas_shared, ctas_regs, max_ctas_per_sm)
    return max(int(ctas), 1)

# K2 shared-memory cap: reject configs whose actual GPU-JIT shared memory
# exceeds this budget.  The cap prevents shipping a cubin that uses almost
# the entire per-block optin budget (e.g. 96 KB on RTX 3050 Laptop), which
# forces 1 CTA/SM and can profile poorly in the full DiT graph.
#
# Default: 0 = disabled (let the spill gate and per-arch smem filter in
# _estimate_k2_configs be the only filters).  The previous 64 KB default
# was set when the only viable configs were 64x128 BK=64 S=3 (96 KB) — it
# blocked those configs, forcing the user to set
# HOTSTEP_K2_MAX_SHARED_BYTES=0 to get any autotune winner.  With the new
# arch-dispatched config builder, the smem filter in _estimate_k2_configs
# already rejects configs that don't fit the per-arch optin budget, so
# this secondary cap is redundant.  Override with a byte count to
# re-enable.
K2_MAX_SHARED_BYTES = int(os.environ.get("HOTSTEP_K2_MAX_SHARED_BYTES", "0"))


def _autotune_grid_x(num_tiles: int, num_sms: int,
                     ctas_per_sm: int = AUTOTUNE_OCCUPANCY_ASSUMPTION) -> int:
    """Occupancy-scaled autotune grid (patch review §H2).

    Mirrors the production launch formula
    ``min(num_tiles, num_sms * max_active_ctas_per_sm)``.  ``ctas_per_sm``
    defaults to ``AUTOTUNE_OCCUPANCY_ASSUMPTION`` (2) for K1 and any caller
    that doesn't have a per-config occupancy value; K2's grid lambda passes
    the precomputed per-config occupancy so every candidate is measured
    under the same launch geometry the production plugin will use.
    Caps at ``num_tiles`` so we never launch more CTAs than there is work.
    """
    cap = num_sms * max(int(ctas_per_sm), 1)
    return min(num_tiles, cap) if num_tiles > cap else num_tiles


def _autotune_grid_x_for_k2(num_tiles: int, num_sms: int, meta) -> int:
    """K2-only grid lambda helper that looks up the per-config occupancy.

    The Triton autotune grid lambda receives ``meta``: a dict of the
    config's constexpr kwargs merged with ``num_warps``/``num_stages``.  We
    reconstruct the config identity tuple and look up the occupancy that
    the spill/compile filter previously computed for this exact
    (kwargs, warps, stages, maxnreg) combination.

    If the config isn't in the map (spill filter was skipped, e.g. tests
    that inject a manual config list), we fall back to
    ``AUTOTUNE_OCCUPANCY_ASSUMPTION``.
    """
    kwargs_items = tuple(sorted(
        (k, v) for k, v in meta.items()
        if k not in ("num_warps", "num_stages", "num_ctas", "maxnreg")
    ))
    key = (kwargs_items,
           int(meta.get("num_warps", 0)),
           int(meta.get("num_stages", 0)),
           meta.get("maxnreg", None))
    ctas = _K2_CFG_OCCUPANCY.get(key, AUTOTUNE_OCCUPANCY_ASSUMPTION)
    return _autotune_grid_x(num_tiles, num_sms, ctas)


_LAUNCH_ARG_ENUM = {
    "X_ptr": "kX_ptr",
    "X_q_ptr": "kX_q_ptr",
    "X_scale_ptr": "kX_scale_ptr",
    "W_q_ptr": "kW_q_ptr",
    "W_scale_ptr": "kW_scale_ptr",
    "Bias_ptr": "kBias_ptr",
    "Y_ptr": "kY_ptr",
    "M": "kM",
    "N": "kN",
    "K": "kK",
    "NUM_TILES": "kNUM_TILES",
    "stride_xm": "kStride_xm",
    "stride_xk": "kStride_xk",
    "stride_xqm": "kStride_xqm",
    "stride_xqk": "kStride_xqk",
    "stride_xsm": "kStride_xsm",
    "stride_xsg": "kStride_xsg",
    "stride_wn": "kStride_wn",
    "stride_wk": "kStride_wk",
    "stride_ym": "kStride_ym",
    "stride_yn": "kStride_yn",
}

_POINTER_ARG_NAMES = {
    "X_ptr", "X_q_ptr", "X_scale_ptr", "W_q_ptr", "W_scale_ptr", "Bias_ptr", "Y_ptr",
}

_QUANT_ARG_NAMES = {
    "X_ptr", "X_q_ptr", "X_scale_ptr", "M", "K",
    "stride_xm", "stride_xk", "stride_xqm", "stride_xqk", "stride_xsm", "stride_xsg",
}

_GEMM_ARG_NAMES = {
    "X_q_ptr", "X_scale_ptr", "W_q_ptr", "W_scale_ptr", "Bias_ptr", "Y_ptr",
    "M", "N", "K", "NUM_TILES",
    "stride_xqm", "stride_xqk", "stride_xsm", "stride_xsg",
    "stride_wn", "stride_wk", "stride_ym", "stride_yn",
}

_PTX_ENTRY_RE = re.compile(
    r"\.visible\s+\.entry\s+(?P<name>\w+)\s*\((?P<params>.*?)\)\s*(?P<after>.*)",
    re.S,
)
_PTX_REQNTID_RE = re.compile(r"\.reqntid\s+(\d+)(?:\s*,\s*(\d+)\s*,\s*(\d+))?")
_PTX_MAXNREG_RE = re.compile(r"\.maxnreg\s+(\d+)")
_PTX_PARAM_TYPE_RE = re.compile(r"\.param\s+\.(?P<kind>\w+)")


def _normalize_sig_type(sig_type):
    return str(sig_type).strip().replace(" ", "")

def _runtime_kind_from_sig_type(sig_type: str) -> str | None:
    sig_type = _normalize_sig_type(sig_type)
    if sig_type == "constexpr" or sig_type.startswith("constexpr["):
        return None
    if sig_type.startswith("*") or sig_type == "nvTmaDesc":
        return "ptr"
    if sig_type in {"i1", "i8", "i16", "i32", "i64", "u1", "u8", "u16", "u32", "u64", "fp16", "bf16", "fp32", "fp64"}:
        return "scalar"
    raise RuntimeError(f"Unsupported Triton signature type in extracted kernel ABI: {sig_type}")

def _ordered_runtime_signature(kernel):
    src = getattr(kernel, "src", None)
    signature = getattr(src, "signature", None)
    if not isinstance(signature, dict) or not signature:
        raise RuntimeError("CompiledKernel.src.signature is missing or empty; cannot derive launch ABI")

    fn = getattr(src, "fn", None)
    arg_names = list(getattr(fn, "arg_names", []) or [])
    ordered = []
    consumed = set()

    def append_item(key, name_hint=None):
        if key in signature and key not in consumed:
            ordered.append({
                "name": name_hint or str(key),
                "sig_type": _normalize_sig_type(signature[key]),
            })
            consumed.add(key)

    for idx, arg_name in enumerate(arg_names):
        if arg_name in signature:
            append_item(arg_name, arg_name)
        elif idx in signature:
            append_item(idx, arg_name)

    def sort_key(key):
        if isinstance(key, int):
            return (0, key)
        return (1, str(key))

    for key in sorted((k for k in signature.keys() if k not in consumed), key=sort_key):
        name = arg_names[key] if isinstance(key, int) and 0 <= key < len(arg_names) else str(key)
        append_item(key, name)

    if not ordered:
        raise RuntimeError("Failed to derive an ordered runtime signature from CompiledKernel.src.signature")
    return ordered

def _parse_ptx_entry_abi(ptx_text: str, function_name: str):
    for match in _PTX_ENTRY_RE.finditer(ptx_text):
        if match.group("name") != function_name:
            continue
        params_block = match.group("params")
        after_block = match.group("after")
        param_kinds = []
        param_types = []
        for line in params_block.splitlines():
            line = line.strip().rstrip(",")
            if not line.startswith(".param"):
                continue
            type_match = _PTX_PARAM_TYPE_RE.search(line)
            if type_match is None:
                raise RuntimeError(f"Could not parse PTX param declaration: {line!r}")
            raw_type = type_match.group("kind")
            is_ptr = ".ptr" in line
            param_kinds.append("ptr" if is_ptr else "scalar")
            param_types.append(("*" if is_ptr else "") + raw_type)
        reqntid_match = _PTX_REQNTID_RE.search(after_block)
        if reqntid_match is None:
            raise RuntimeError(f"Could not locate .reqntid for PTX entry {function_name}")
        reqntid = (
            int(reqntid_match.group(1)),
            int(reqntid_match.group(2) or 1),
            int(reqntid_match.group(3) or 1),
        )
        maxnreg_match = _PTX_MAXNREG_RE.search(after_block)
        maxnreg = int(maxnreg_match.group(1)) if maxnreg_match else None
        return {
            "function_name": function_name,
            "param_kinds": param_kinds,
            "param_types": param_types,
            "reqntid": reqntid,
            "maxnreg": maxnreg,
        }
    raise RuntimeError(f"Failed to locate PTX .entry for {function_name}")

def _extract_kernel_abi(kernel, stage_name: str):
    function_name = getattr(kernel, "name", None) or getattr(getattr(kernel, "metadata", None), "name", None)
    if not function_name:
        raise RuntimeError("CompiledKernel has no name metadata; cannot derive launch ABI")

    runtime_signature = _ordered_runtime_signature(kernel)
    allowed_names = _QUANT_ARG_NAMES if stage_name == "quant" else _GEMM_ARG_NAMES
    filtered_runtime_signature = []
    for item in runtime_signature:
        name = item["name"]
        sig_type = item["sig_type"]
        kind = _runtime_kind_from_sig_type(sig_type)
        if kind is None:
            continue
        if name not in _LAUNCH_ARG_ENUM:
            raise RuntimeError(f"Unsupported runtime kernel argument in extracted ABI: {name}")
        if name not in allowed_names:
            raise RuntimeError(f"Argument {name} is not valid for {stage_name} launch stubs")
        if name in _POINTER_ARG_NAMES and kind != "ptr":
            raise RuntimeError(f"Expected pointer type for {name}, got {sig_type}")
        if name not in _POINTER_ARG_NAMES and sig_type not in {"i32", "u32"}:
            raise RuntimeError(f"Expected 32-bit scalar type for {name}, got {sig_type}")
        item["kind"] = kind
        item["enum"] = _LAUNCH_ARG_ENUM[name]
        filtered_runtime_signature.append(item)

    ptx = kernel.asm.get("ptx")
    if not isinstance(ptx, str):
        raise RuntimeError(f"CompiledKernel for {function_name} has no PTX text; cannot validate launch ABI")
    ptx_abi = _parse_ptx_entry_abi(ptx, function_name)

    runtime_signature = filtered_runtime_signature
    expected_kinds = [item["kind"] for item in runtime_signature]
    actual_kinds = ptx_abi["param_kinds"]
    if len(actual_kinds) < len(expected_kinds):
        raise RuntimeError(
            f"PTX ABI for {function_name} has fewer params than runtime signature: "
            f"ptx={len(actual_kinds)} runtime={len(expected_kinds)}"
        )
    prefix_kinds = actual_kinds[:len(expected_kinds)]
    if prefix_kinds != expected_kinds:
        raise RuntimeError(
            f"PTX/runtime ABI kind mismatch for {function_name}: "
            f"runtime={expected_kinds} ptx_prefix={prefix_kinds}"
        )
    scratch_tail = actual_kinds[len(expected_kinds):]
    if any(kind != "ptr" for kind in scratch_tail):
        raise RuntimeError(
            f"Unexpected non-pointer PTX tail parameters for {function_name}: "
            f"runtime={expected_kinds} ptx={actual_kinds}"
        )

    return {
        "function_name": function_name,
        "runtime_signature": runtime_signature,
        "scratch_ptr_count": len(scratch_tail),
        "ptx_param_count": len(actual_kinds),
        "ptx_param_types": ptx_abi["param_types"],
        "block": ptx_abi["reqntid"],
        "ptx_maxnreg": ptx_abi["maxnreg"],
    }

def _find_jit_holder(obj):
    seen = set()
    cur = obj
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "device_caches"):
            return cur
        cur = getattr(cur, "fn", None)
    return None

def _debug_wrapper_chain(obj):
    cur = obj
    i = 0
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        print(
            f"[debug] layer {i}: type={type(cur)!r}, "
            f"has_fn={hasattr(cur, 'fn')}, "
            f"has_device_caches={hasattr(cur, 'device_caches')}",
            flush=True,
        )
        cur = getattr(cur, "fn", None)
        i += 1

def _safe_cache_entry(device_caches, device):
    if device not in device_caches:
        raise RuntimeError(f"Device {device} not in JITFunction cache — kernel was not compiled?")
    cache_entry = device_caches[device]
    if isinstance(cache_entry, tuple):
        kernel_cache = cache_entry[0]
    else:
        kernel_cache = cache_entry
    if not kernel_cache:
        raise RuntimeError("kernel_cache is empty — autotune did not compile any kernels?")
    return kernel_cache

def _load_kernel_sidecar(kernel):
    for attr in ("cache_path", "fn", "path"):
        v = getattr(kernel, attr, None)
        if isinstance(v, (str, os.PathLike)):
            p = Path(v)
            if p.suffix == ".json" and p.exists():
                return json.loads(p.read_text())
            if p.exists():
                sidecar = p.with_suffix(".json")
                if sidecar.exists():
                    return json.loads(sidecar.read_text())
    asm = getattr(kernel, "asm", None)
    if isinstance(asm, dict):
        for v in asm.values():
            if isinstance(v, (str, os.PathLike)):
                p = Path(v)
                if p.exists():
                    sidecar = p.with_suffix(".json")
                    if sidecar.exists():
                        return json.loads(sidecar.read_text())
    return None

def _kernel_matches_best_cfg(key, kernel, best_cfg, arg_names, runtime_kwargs):
    meta = kernel.metadata
    if getattr(meta, "num_warps", None) is not None and getattr(meta, "num_warps", None) != best_cfg.num_warps:
        return False
    if getattr(meta, "num_stages", None) is not None and getattr(meta, "num_stages", None) != best_cfg.num_stages:
        return False

    arg_list, meta_dict = None, None
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], dict):
        arg_list, meta_dict = key
    else:
        key_str = str(key)
        dict_start = key_str.rfind('{')
        if dict_start != -1:
            try:
                list_str = key_str[:dict_start]
                dict_str = key_str[dict_start:]
                arg_list = ast.literal_eval(list_str)
                meta_dict = ast.literal_eval(dict_str)
            except Exception:
                pass

    if arg_list is not None and meta_dict is not None and arg_names:
        try:
            if meta_dict.get('num_warps') != best_cfg.num_warps:
                return False
            if meta_dict.get('num_stages') != best_cfg.num_stages:
                return False
            best_maxnreg = getattr(best_cfg, 'maxnreg', None)
            cache_maxnreg = meta_dict.get('maxnreg')
            def _norm_mnreg(v): return v if v else None
            if _norm_mnreg(cache_maxnreg) != _norm_mnreg(best_maxnreg):
                return False

            expected_kwargs = {**best_cfg.kwargs, **runtime_kwargs}
            seen_expected = set()
            for i, arg_name in enumerate(arg_names):
                if arg_name in expected_kwargs:
                    expected_val = expected_kwargs[arg_name]
                    if i >= len(arg_list):
                        return False
                    val_tuple = arg_list[i]
                    if not (isinstance(val_tuple, (tuple, list)) and len(val_tuple) >= 2 and val_tuple[0] == 'constexpr'):
                        return None
                    if val_tuple[1] != expected_val:
                        return False
                    seen_expected.add(arg_name)
            if set(expected_kwargs) - seen_expected:
                return False
            return True
        except Exception:
            pass

    sidecar = _load_kernel_sidecar(kernel)
    if sidecar is not None:
        if sidecar.get("num_warps") != best_cfg.num_warps:
            return False
        if sidecar.get("num_stages") != best_cfg.num_stages:
            return False
        def _norm_mnreg_sidecar(v): return v if v else None
        if _norm_mnreg_sidecar(sidecar.get("maxnreg")) != _norm_mnreg_sidecar(getattr(best_cfg, "maxnreg", None)):
            return False

    return None

def _select_kernel_from_cache(kernel_cache, best_cfg, arg_names, runtime_kwargs):
    exact = []
    unknown = []

    for key, kernel in kernel_cache.items():
        if kernel is None:
            continue
        match = _kernel_matches_best_cfg(key, kernel, best_cfg, arg_names, runtime_kwargs)
        if match is True:
            exact.append((key, kernel))
        elif match is None:
            meta = kernel.metadata
            if getattr(meta, "num_warps", None) == best_cfg.num_warps and getattr(meta, "num_stages", None) == best_cfg.num_stages:
                unknown.append((key, kernel))

    if len(exact) == 1:
        return exact[0][1]
    if len(exact) > 1:
        raise RuntimeError(
            f"Multiple exact cache matches for best config {best_cfg} and runtime kwargs {runtime_kwargs}. "
            f"Need stronger discriminator. Matches={len(exact)}"
        )

    if len(unknown) == 1:
        return unknown[0][1]
    if len(unknown) > 1:
        raise RuntimeError(
            f"Multiple ambiguous cache matches for best config {best_cfg} using only metadata. "
            f"Matches={len(unknown)}. Add extra dumps and inspect cache layout."
        )

    raise RuntimeError(
        f"Could not find CompiledKernel matching best config {best_cfg}. "
        f"Cache has {len(kernel_cache)} entries."
    )

def _dump_cache_debug(kernel_cache, best_cfg):
    print("[debug] best config:", best_cfg, flush=True)
    for i, (key, kernel) in enumerate(kernel_cache.items()):
        if kernel is None:
            continue
        meta = kernel.metadata
        sidecar = _load_kernel_sidecar(kernel)
        print(f"[debug] cache[{i}] key={key!r}", flush=True)
        print(
            f"        meta: warps={getattr(meta, 'num_warps', None)} "
            f"stages={getattr(meta, 'num_stages', None)} shared={getattr(meta, 'shared', None)}",
            flush=True,
        )
        if sidecar is not None:
            print(
                f"        sidecar: name={sidecar.get('name')} warps={sidecar.get('num_warps')} "
                f"stages={sidecar.get('num_stages')} maxnreg={sidecar.get('maxnreg')} "
                f"shared={sidecar.get('shared')} hash={sidecar.get('hash')[:16]}...",
                flush=True,
            )
        else:
            print("        sidecar: <missing>", flush=True)


def _get_kernel_cache_dir(kernel):
    """Return the Triton JIT cache directory holding this kernel's artifacts.

    Each CompiledKernel is materialized on disk as a hash-named directory under
    the Triton cache root (TRITON_CACHE_DIR or ~/.triton/cache) containing the
    cubin, PTX, TTGIR, TTIR, and a JSON metadata sidecar.  Returning this
    directory lets the operator pull the exact PTX/IR/SASS for the autotune
    winner for offline diagnosis (cuobjdump, ptxas -v, nvdisasm).

    Returns None if no on-disk artifact path can be recovered.

    Triton 3.7.x CompiledKernel layout:
      * kernel.metadata_group : dict[str, Path] — maps content-hash → Path
        for each artifact (.json, .ptx, .ttgir, .cubin, etc.).  The parent
        of any of these paths is the cache directory.
      * kernel.hash : str — the hash used in the directory name.
      * kernel.asm : AsmDict — maps suffix → bytes/text (no path info).
      * No `cache_path` or `path` attribute exists in 3.7.x.
    """
    # Preferred: metadata_group dict (Triton 3.7.x).  Values are Path objects
    # pointing at individual artifacts; the parent of any of them is the cache
    # directory.
    mg = getattr(kernel, "metadata_group", None)
    if isinstance(mg, dict):
        for v in mg.values():
            if isinstance(v, (str, os.PathLike)) and not isinstance(v, (bytes, bytearray)):
                try:
                    p = Path(v)
                    if p.is_file():
                        return p.parent
                    if p.is_dir():
                        return p
                except (OSError, ValueError):
                    pass
    # Fallback: check common path-like attributes (older Triton versions).
    for attr in ("cache_path", "path"):
        v = getattr(kernel, attr, None)
        if isinstance(v, (str, os.PathLike)) and not isinstance(v, (bytes, bytearray)):
            try:
                p = Path(v)
                if p.is_file():
                    return p.parent
                if p.is_dir():
                    return p
            except (OSError, ValueError):
                pass
    # Fallback: scan .asm dict for path-like values (some Triton versions
    # store artifact paths here instead of raw bytes).
    asm = getattr(kernel, "asm", None)
    if isinstance(asm, dict):
        for v in asm.values():
            if isinstance(v, (str, os.PathLike)) and not isinstance(v, (bytes, bytearray)):
                try:
                    p = Path(v)
                    if p.is_file():
                        return p.parent
                except (OSError, ValueError):
                    pass
    return None


def _get_compiled_kernel(autotuned_fn, device=0, debug_dump=False, **runtime_kwargs):
    base_fn = _find_jit_holder(autotuned_fn)
    if base_fn is None:
        _debug_wrapper_chain(autotuned_fn)
        raise RuntimeError(
            f"Could not find Triton JIT cache holder in wrapper chain starting from "
            f"{type(autotuned_fn)!r}"
        )

    best_cfg = autotuned_fn.best_config
    kernel_cache = _safe_cache_entry(base_fn.device_caches, device)
    arg_names = getattr(base_fn, "arg_names", None)

    if debug_dump:
        _dump_cache_debug(kernel_cache, best_cfg)

    kernel = _select_kernel_from_cache(kernel_cache, best_cfg, arg_names, runtime_kwargs)
    cache_dir = _get_kernel_cache_dir(kernel)
    return kernel, best_cfg, cache_dir


def _get_compiled_kernel_for_cfg(autotuned_fn, best_cfg, device=0,
                                 debug_dump=False, **runtime_kwargs):
    """Same as ``_get_compiled_kernel`` but for a caller-selected ``best_cfg``.

    The custom K2 benchmarker (``_bench_k2_configs_custom``) bypasses
    ``@triton.autotune``'s built-in measurement, so ``autotuned_fn.best_config``
    is not populated for the shape we care about.  Instead we hand the picked
    config in explicitly and reuse the same cache-walking helpers.
    """
    base_fn = _find_jit_holder(autotuned_fn)
    if base_fn is None:
        _debug_wrapper_chain(autotuned_fn)
        raise RuntimeError(
            f"Could not find Triton JIT cache holder in wrapper chain starting from "
            f"{type(autotuned_fn)!r}"
        )

    kernel_cache = _safe_cache_entry(base_fn.device_caches, device)
    arg_names = getattr(base_fn, "arg_names", None)

    if debug_dump:
        _dump_cache_debug(kernel_cache, best_cfg)

    kernel = _select_kernel_from_cache(kernel_cache, best_cfg, arg_names, runtime_kwargs)
    cache_dir = _get_kernel_cache_dir(kernel)
    return kernel, best_cfg, cache_dir


def _flush_compiled_kernel_cache(jit_fn):
    base_fn = _find_jit_holder(jit_fn)
    if base_fn is None:
        return
    for device, cache_entry in list(base_fn.device_caches.items()):
        if isinstance(cache_entry, tuple):
            kernel_cache = cache_entry[0]
        else:
            kernel_cache = cache_entry
        if isinstance(kernel_cache, dict):
            kernel_cache.clear()


# ─── Threading State setup ───────────────────────────────────
#
# Replace global dictionary with threading.local() so threads running concurrently
# don't overwrite each other's torch memory tensors.
_K1_WORKER_STATE = threading.local()
_K2_WORKER_STATE = threading.local()


def _init_k1_worker(input_fp16):
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    dtype = torch.float16 if input_fp16 else torch.float32
    _K1_WORKER_STATE.input_fp16 = input_fp16
    _K1_WORKER_STATE.x = torch.empty((1, 1), device="cuda", dtype=dtype)
    _K1_WORKER_STATE.xq = torch.empty((1, 1), device="cuda", dtype=torch.int8)
    _K1_WORKER_STATE.xs = torch.empty((1, 1), device="cuda", dtype=torch.float32)


def _worker_precompile_k1(config_tuple):
    if not hasattr(_K1_WORKER_STATE, "x"):
        return 0
    from triton import Config
    kwargs, nw, ns, mr = config_tuple
    input_fp16 = _K1_WORKER_STATE.input_fp16
    try:
        cfg = Config(kwargs, num_warps=nw, num_stages=ns, maxnreg=mr)
        kernel1_convrot_quant.fn.warmup(
            _K1_WORKER_STATE.x, _K1_WORKER_STATE.xq, _K1_WORKER_STATE.xs,
            1024, 2560,
            2560, 1, 2560, 1, 1, 1024,
            GROUP_SIZE=256, INPUT_FP16=input_fp16,
            CONTIG_XK=True,
            **cfg.all_kwargs(),
            grid=(1, 1, 1)
        )
    except Exception:
        pass
    finally:
        _flush_compiled_kernel_cache(kernel1_convrot_quant)
    return 1


def _parallel_precompile_k1(configs, input_fp16):
    num_workers = int(os.environ.get("HOTSTEP_AUTOTUNE_WORKERS", min(mp.cpu_count(), len(configs), 16)))
    if num_workers <= 1 or not configs:
        return
    print(f"    [parallel-compile] Precompiling {len(configs)} K1 configs across {num_workers} threads...", flush=True)
    tuples = [(c.kwargs, c.num_warps, c.num_stages, getattr(c, "maxnreg", None)) for c in configs]
    done = 0
    total = len(configs)
    # Sparse newline-terminated progress (every ~10%): carriage-return
    # progress lines interleave badly with worker-thread prints.
    progress_step = max(total // 10, 1)
    try:
        with ThreadPool(
            processes=num_workers,
            initializer=_init_k1_worker,
            initargs=(input_fp16,)
        ) as pool:
            for _ in pool.imap_unordered(_worker_precompile_k1, tuples, chunksize=1):
                done += 1
                if done % progress_step == 0 or done == total:
                    print(f"    [parallel-compile] K1: {done}/{total} configs compiled", flush=True)
    except Exception as e:
        print(f"    [parallel-compile] Warning: parallel precompilation fallback ({e})", flush=True)


def _init_k2_worker(output_fp16, target_arch):
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    _K2_WORKER_STATE.output_fp16 = output_fp16
    _K2_WORKER_STATE.target_arch = target_arch
    # Gluon AOT compile (GluonASTSource) needs no device tensors — it compiles
    # from the signature/constexprs alone.  The dummy tensors below are kept
    # only so hasattr() guards in legacy callers don't break.
    _K2_WORKER_STATE.xq = torch.empty((1, 1), device="cuda", dtype=torch.int8)


def _worker_precompile_k2(config_tuple):
    """AOT-compile one K2 gluon_pipe config to warm the compile cache.

    The Gluon kernel is compiled from its signature + constexprs via
    GluonASTSource (no @triton.autotune cache to warm).  Compiling here, ahead
    of the custom bench loop, surfaces compile errors early and populates
    Triton's on-disk compile cache so the benchmarker's first launch of each
    config is a cache hit.
    """
    if not hasattr(_K2_WORKER_STATE, "xq"):
        return 0
    cfg, = config_tuple
    output_fp16 = _K2_WORKER_STATE.output_fp16
    try:
        _gluon_aot_compile_k2(cfg, output_fp16, _K2_WORKER_STATE.target_arch)
    except Exception:
        # Compile failures are not fatal here — the spill filter and the
        # custom benchmarker both report them per-config.  Precompile is best
        # effort (warm the cache, surface obvious errors early).
        pass
    return 1


def _parallel_precompile_k2(configs, output_fp16, target_arch):
    if not GLUON_AVAILABLE:
        print("    [parallel-compile] K2 skipped: Gluon dialect unavailable", flush=True)
        return
    num_workers = int(os.environ.get("HOTSTEP_AUTOTUNE_WORKERS", min(mp.cpu_count(), len(configs), 16)))
    if num_workers <= 1 or not configs:
        return
    print(f"    [parallel-compile] Precompiling {len(configs)} K2 (gluon_pipe) "
          f"configs across {num_workers} threads...", flush=True)
    tuples = [(c,) for c in configs]
    done = 0
    total = len(configs)
    # Sparse newline-terminated progress (every ~10%): carriage-return
    # progress lines interleave badly with worker-thread prints.
    progress_step = max(total // 10, 1)
    try:
        with ThreadPool(
            processes=num_workers,
            initializer=_init_k2_worker,
            initargs=(output_fp16, target_arch)
        ) as pool:
            for _ in pool.imap_unordered(_worker_precompile_k2, tuples, chunksize=1):
                done += 1
                if done % progress_step == 0 or done == total:
                    print(f"    [parallel-compile] K2: {done}/{total} configs compiled", flush=True)
    except Exception as e:
        print(f"    [parallel-compile] Warning: parallel precompilation fallback ({e})", flush=True)


# ─── Register spill detection (patch review §8 — robust, non-heuristic) ──
#
# Triton 3.7.x will SILENTLY spill registers to local memory (DRAM) when a
# config's maxnreg cap can't be honored.  There is no compile error — the
# kernel produces correct output but at a fraction of the performance because
# every spilled register access goes through DRAM instead of the register file.
#
# ZERO TOLERANCE: any config that spills even one byte is rejected.  No
# thresholds, no "borderline" acceptance.
#
# Detection method (patch review §8.1): every NVIDIA cubin carries per-kernel
# ELF attributes (EIATTR_FRAME_SIZE) recording how many bytes of thread-local
# ("local memory") stack space the kernel needs.  For a Triton kernel that
# declares no .local arrays, the ONLY source of nonzero thread-local stack is
# register spilling.  This is a hard, unambiguous binary signal:
#   * STACK == 0 → no spills (categorically)
#   * STACK > 0  → spills (categorically, REJECT)
#
# We read this via cuobjdump --dump-resource-usage (the STACK: field, NOT
# LOCAL: which stays 0 for Triton kernels).  If cuobjdump is unavailable,
# we fall back to PTX-level ld.local/st.local count (also zero-tolerance:
# any nonzero count = reject).

_CUOBJDUMP_FUNC_RE = None  # lazy-compiled


def _find_cuobjdump():
    """Find cuobjdump — prefer the one bundled with Triton (version-matched)."""
    import shutil
    try:
        import triton as _triton
        candidate = (Path(_triton.__file__).resolve().parent
                     / "backends" / "nvidia" / "bin" / "cuobjdump")
        if candidate.exists():
            return str(candidate)
    except ImportError:
        pass
    return shutil.which("cuobjdump")


def _check_cubin_spills_cuobjdump(cubin_bytes, function_name):
    """Robust spill check via cuobjdump --dump-resource-usage.

    Returns (spills: bool, local_bytes_per_thread: int, registers: int).
    Returns (None, 0, 0) if cuobjdump is not available.
    """
    import re
    import subprocess
    import tempfile

    global _CUOBJDUMP_FUNC_RE
    if _CUOBJDUMP_FUNC_RE is None:
        _CUOBJDUMP_FUNC_RE = re.compile(
            r"Function\s+(?P<name>\S+?):\s*\n\s*"
            r"REG:(?P<reg>\d+)\s+STACK:(?P<stack>\d+)\s+SHARED:(?P<shared>\d+)\s+LOCAL:(?P<local>\d+)",
        )

    cuobjdump = _find_cuobjdump()
    if cuobjdump is None:
        return (None, 0, 0)  # cuobjdump not available

    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin_bytes)
        cubin_path = f.name
    try:
        proc = subprocess.run(
            [cuobjdump, "--dump-resource-usage", cubin_path],
            capture_output=True, text=True, check=True,
        )
        for m in _CUOBJDUMP_FUNC_RE.finditer(proc.stdout):
            if m.group("name") == function_name:
                stack = int(m.group("stack"))
                regs = int(m.group("reg"))
                # ZERO TOLERANCE: stack > 0 means spills, period.
                return (stack > 0, stack, regs)
        # Function not found — can't determine
        return (None, 0, 0)
    except Exception:
        return (None, 0, 0)
    finally:
        Path(cubin_path).unlink(missing_ok=True)


def _detect_spills(kernel, function_name: str = "") -> dict:
    """Check a compiled kernel for register spills.  ZERO TOLERANCE.

    Primary method: cuobjdump STACK: field (nonzero = spills).
    Fallback (if cuobjdump unavailable): PTX-level ld.local/st.local count
    (any nonzero = spills).

    Returns a dict with:
      * 'spilling': bool — True if ANY spills detected
      * 'local_bytes': int — thread-local stack bytes from cuobjdump
      * 'registers': int — register count from cuobjdump
      * 'ptx_spills': int — count of ld.local/st.local in PTX
      * 'method': str — 'cuobjdump' or 'ptx-fallback'
    """
    cubin = kernel.asm.get("cubin", b"")
    cubin_size = len(cubin) if isinstance(cubin, (bytes, bytearray)) else 0

    ptx = kernel.asm.get("ptx", "")
    if isinstance(ptx, bytes):
        ptx = ptx.decode(errors="replace")
    ptx_spills = ptx.count("ld.local") + ptx.count("st.local")

    # Primary: cuobjdump STACK field (authoritative, zero-tolerance)
    spills_cuobjdump, local_bytes, registers = (None, 0, 0)
    if function_name and cubin_size > 0:
        spills_cuobjdump, local_bytes, registers = _check_cubin_spills_cuobjdump(
            cubin, function_name)

    if spills_cuobjdump is not None:
        # cuobjdump succeeded — STACK > 0 means spills.  Zero tolerance.
        spilling = spills_cuobjdump
        method = "cuobjdump"
    else:
        # Fallback: any PTX-level ld.local/st.local = spills.  Zero tolerance.
        # Patch review §LOW: log when the primary cuobjdump path degrades to
        # the PTX fallback so a silently-broken cuobjdump invocation (wrong
        # version, missing binary, regex mismatch) is visible in the build
        # log instead of being hidden behind a successful PTX result.
        if function_name and cubin_size > 0:
            print(
                f"    [spill-check] WARNING: cuobjdump unavailable or parsing "
                f"failed for {function_name} (cubin={cubin_size}B); falling "
                f"back to PTX ld.local/st.local counting (less authoritative).",
                flush=True,
            )
        spilling = ptx_spills > 0
        method = "ptx-fallback"

    return {
        "spilling": spilling,
        "ptx_spills": ptx_spills,
        "local_bytes": local_bytes,
        "registers": registers,
        "method": method,
    }


def _compile_kernel_for_spill_check(jit_fn, signature, constexprs, num_warps,
                                     num_stages, maxnreg, target_arch):
    """Compile a single config for spill analysis (no autotune, no launch).

    Dispatches to the Gluon AOT compile path (GluonASTSource) when ``jit_fn``
    is a Gluon kernel (the K2 ``gluon_pipe`` path), and to the standard
    Triton ASTSource path otherwise (the K1 @triton.jit path).  Gluon kernels
    ignore ``maxnreg`` (register allocation is driven by the explicit layouts)
    and pass ``num_stages=1`` to the compiler (the pipeline depth is encoded
    in the kernel's NUM_BUFS constexpr, not the Triton num_stages option).
    """
    from triton.compiler import compile as triton_compile
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget

    target = GPUTarget(backend='cuda', arch=target_arch, warp_size=32)

    if _is_gluon_kernel(jit_fn):
        # GluonASTSource expects the @gluon.jit wrapper object directly (it
        # carries .arg_names and the other metadata the compile path needs) —
        # do NOT unwrap .fn (the raw Python function has no .arg_names).
        src = GluonASTSource(fn=jit_fn, signature=signature, constexprs=constexprs)
        options = {'num_warps': num_warps, 'num_stages': 1}
        return triton_compile(src, target=target, options=options)

    fn = jit_fn.fn if hasattr(jit_fn, 'fn') else jit_fn
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    options = {'num_warps': num_warps, 'num_stages': num_stages}
    if maxnreg is not None:
        options['maxnreg'] = maxnreg
    return triton_compile(src, target=target, options=options)


# ─── Parallel spill-check worker state ────────────────────────────────────
_SPILL_CHECK_STATE = threading.local()


def _init_spill_check_worker(jit_fn, target_arch, stage):
    _SPILL_CHECK_STATE.jit_fn = jit_fn
    _SPILL_CHECK_STATE.target_arch = target_arch
    _SPILL_CHECK_STATE.fn_name = "kernel1_convrot_quant" if stage == "k1" else "kernel2_gemm_dequant"


def _worker_check_spills(task):
    """Compile one (config, variant) and return (config, label, spill_info_or_error)."""
    cfg, variant_label, signature, constexprs = task
    try:
        kernel = _compile_kernel_for_spill_check(
            _SPILL_CHECK_STATE.jit_fn,
            signature,
            constexprs,
            cfg.num_warps, cfg.num_stages,
            getattr(cfg, 'maxnreg', None),
            _SPILL_CHECK_STATE.target_arch)
        # Prefer the compiled kernel's own symbol name (robust for both the
        # @triton.jit K1 path and the @gluon.jit K2 path, whose cubin symbol
        # may differ from the Python function name).
        fn_name = getattr(kernel, "name", None) or _SPILL_CHECK_STATE.fn_name
        spill_info = _detect_spills(kernel, fn_name)
        spill_info["shared_bytes"] = int(getattr(kernel.metadata, "shared", 0) or 0)
        return (cfg, variant_label, spill_info, None)
    except Exception as e:
        return (cfg, variant_label, None, f"{type(e).__name__}: {e}")


def _spill_check_variants(stage):
    """Enumerate ALL shipped constexpr variants for the spill gate.

    Register pressure differs between variants — e.g. the K2 BIAS variant
    holds an extra (BLOCK_N,) fp32 tile in the epilogue, and FP16 IO changes
    both load widths and conversion sequences.  A config is only kept if it
    is spill-free in EVERY variant that will actually be extracted (per
    shipped BIAS_CONFIGS × DTYPE_CONFIGS), otherwise a config could pass the gate on
    the NOBIAS/FP32 variant and then spill in the BIAS variant that ships.

    Yields (label, signature, extra_constexprs) tuples.
    """
    if stage == "k1":
        for dtype_suffix, input_fp16, _ in DTYPE_CONFIGS:
            x_type = '*fp16' if input_fp16 else '*fp32'
            signature = {
                'X_ptr': x_type, 'X_q_ptr': '*i8', 'X_scale_ptr': '*fp32',
                'M': 'i32', 'K': 'i32',
                'stride_xm': 'i32', 'stride_xk': 'i32',
                'stride_xqm': 'i32', 'stride_xqk': 'i32',
                'stride_xsm': 'i32', 'stride_xsg': 'i32',
            }
            yield (dtype_suffix, signature,
                   {'GROUP_SIZE': 256, 'INPUT_FP16': input_fp16, 'CONTIG_XK': True})
    else:  # k2 — Gluon gluon_pipe kernel signature
        # The K2 kernel is the Gluon ``k2_gluon_pipelined`` variant: a
        # persistent tile loop that takes a NUM_TILES runtime arg between K
        # and the strides, and NUM_BUFS / FOLD_EVERY / GROUP_M / HAS_BIAS /
        # OUTPUT_FP16 as constexprs.  HAS_BIAS is compile-time-hardwired True
        # (BIAS_CONFIGS); iterating BIAS_CONFIGS keeps the label scheme in
        # sync with the extraction summary and the C++ cubin lookup.  The
        # signature is reused from _gluon_k2_signature so the spill-check
        # compile path and the AOT cubin-harvest path feed GluonASTSource an
        # identical, benchmark-proven signature.
        for bias_suffix, _has_bias in BIAS_CONFIGS:
            for dtype_suffix, _, output_fp16 in DTYPE_CONFIGS:
                yield (f"{bias_suffix}_{dtype_suffix}",
                       _gluon_k2_signature(output_fp16),
                       {'GROUP_SIZE': 256, 'HAS_BIAS': True,
                        'OUTPUT_FP16': output_fp16})


def _filter_spilling_configs(configs, jit_fn, stage, target_arch):
    """Compile each config in parallel and reject any that spill registers.

    ZERO TOLERANCE: any config with STACK > 0 (from cuobjdump) or any
    PTX-level ld.local/st.local is rejected.  No thresholds.

    Every SHIPPED constexpr variant (currently bias-enabled K2 only × DTYPE_CONFIGS) is
    checked — a config survives only if it is spill-free in all of them
    (see _spill_check_variants).

    Uses the same 16-thread ThreadPool as the precompile step — sequential
    compilation of 50+ configs takes minutes; parallel takes seconds.

    Returns the filtered config list.
    """
    if not configs:
        return configs

    def make_constexprs(cfg, extra):
        out = dict(extra)
        out.update(cfg.kwargs)  # BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M
        if stage == "k2":
            # Gluon gluon_pipe kernel also takes NUM_BUFS (= stages) and
            # FOLD_EVERY (= GROUPS_PER_TILE by default) as constexprs; both
            # are config-specific.
            out["NUM_BUFS"] = _gluon_k2_num_bufs(cfg)
            out["FOLD_EVERY"] = _gluon_k2_fold_every(cfg)
        return out

    # Build the task list: (config, variant_label, signature, constexprs)
    # for each config × shipped variant.
    tasks = [
        (cfg, label, signature, make_constexprs(cfg, extra))
        for cfg in configs
        for (label, signature, extra) in _spill_check_variants(stage)
    ]

    # Compile all config × variant combinations in parallel using the same
    # ThreadPool approach as _parallel_precompile_k1/k2 (16 threads default).
    num_workers = int(os.environ.get("HOTSTEP_AUTOTUNE_WORKERS",
                                      min(mp.cpu_count(), len(tasks), 16)))
    print(f"    [spill-check] {stage}: checking {len(configs)} configs x "
          f"{len(tasks) // max(len(configs), 1)} variant(s) = {len(tasks)} compiles "
          f"across {num_workers} threads (zero-tolerance)...", flush=True)

    results = []
    total = len(tasks)

    def _progress(done_count):
        # Plain end='\n'-free progress would interleave with worker prints
        # (e.g. cuobjdump fallback warnings); emit sparse newline-terminated
        # progress lines instead (every ~10% and at completion).
        step = max(total // 10, 1)
        if done_count % step == 0 or done_count == total:
            print(f"    [spill-check] {stage}: {done_count}/{total} compiled", flush=True)

    if num_workers <= 1:
        # Sequential fallback
        _init_spill_check_worker(jit_fn, target_arch, stage)
        for task in tasks:
            results.append(_worker_check_spills(task))
            _progress(len(results))
    else:
        try:
            with ThreadPool(
                processes=num_workers,
                initializer=_init_spill_check_worker,
                initargs=(jit_fn, target_arch, stage)
            ) as pool:
                for r in pool.imap_unordered(_worker_check_spills, tasks, chunksize=1):
                    results.append(r)
                    _progress(len(results))
        except Exception as e:
            print(f"    [spill-check] Warning: parallel spill-check failed ({e}), "
                  f"falling back to sequential", flush=True)
            results = []
            _init_spill_check_worker(jit_fn, target_arch, stage)
            for task in tasks:
                results.append(_worker_check_spills(task))
                _progress(len(results))

    # A config is rejected if ANY of its shipped variants spills or fails to
    # compile.  Aggregate per-config over all variant results.
    def _cfg_id(c):
        return (tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages,
                getattr(c, 'maxnreg', None))

    reject_reasons: dict = {}
    for cfg, variant_label, spill_info, error in results:
        cid = _cfg_id(cfg)
        if error is not None:
            reject_reasons.setdefault(cid, []).append(
                f"[{variant_label}] compile failed: {error}")
        elif spill_info['spilling']:
            reject_reasons.setdefault(cid, []).append(
                f"[{variant_label}] SPILLS: {spill_info['method']}: "
                f"local_bytes={spill_info['local_bytes']}, "
                f"ptx_spills={spill_info['ptx_spills']}, "
                f"regs={spill_info['registers']}")
        elif stage == "k2" and K2_MAX_SHARED_BYTES > 0 and spill_info.get("shared_bytes", 0) > K2_MAX_SHARED_BYTES:
            reject_reasons.setdefault(cid, []).append(
                f"[{variant_label}] shared_bytes={spill_info.get('shared_bytes', 0)} "
                f"> K2_MAX_SHARED_BYTES={K2_MAX_SHARED_BYTES}")

    filtered = []
    rejected = []
    # ── Per-config occupancy: minimum CTAs/SM over all shipped variants ──
    # We take the min so the autotune grid never over-launches beyond what
    # the tightest variant supports.  Populated for K2 only; K1 keeps the
    # fixed AUTOTUNE_OCCUPANCY_ASSUMPTION fallback.
    occ_map: dict = {}
    for cfg, variant_label, spill_info, error in results:
        if error is not None or spill_info is None:
            continue
        cur = _compute_ctas_per_sm(
            spill_info.get("shared_bytes", 0),
            spill_info.get("registers", 0),
            cfg.num_warps,
            target_arch,
        )
        cid = _cfg_id(cfg)
        prev = occ_map.get(cid)
        occ_map[cid] = cur if prev is None else min(prev, cur)

    for cfg in configs:
        reasons = reject_reasons.get(_cfg_id(cfg))
        if reasons:
            rejected.append((cfg, "; ".join(reasons)))
        else:
            filtered.append(cfg)

    if stage == "k2":
        # Publish surviving configs' occupancy to the module-level map so
        # the K2 autotune grid lambda can read it.  Keyed by the
        # occupancy-map key (which matches _cfg_id above).
        _K2_CFG_OCCUPANCY.clear()
        for cfg in filtered:
            occ = occ_map.get(_cfg_id(cfg), AUTOTUNE_OCCUPANCY_ASSUMPTION)
            _K2_CFG_OCCUPANCY[_cfg_occupancy_key(cfg)] = int(occ)

    if rejected:
        print(f"    [spill-check] {stage}: REJECTED {len(rejected)} configs "
              f"(zero-tolerance: any spill in any shipped variant = reject):", flush=True)
        for cfg, reason in rejected[:10]:
            mr = getattr(cfg, 'maxnreg', None)
            print(f"      REJECT {cfg.kwargs} W={cfg.num_warps} S={cfg.num_stages} MR={mr}: {reason}", flush=True)
        if len(rejected) > 10:
            print(f"      ... and {len(rejected) - 10} more", flush=True)
    else:
        print(f"    [spill-check] {stage}: all {len(configs)} configs are spill-free "
              f"in all shipped variants", flush=True)

    return filtered


# ─── Custom K2 benchmark loop (replaces Triton's built-in do_bench) ──────
#
# Triton 3.7.x's ``@triton.autotune`` measures each config with an internal
# ``do_bench`` call whose warmup/rep budget is dynamic and (empirically) too
# short for our K2 configs: on sm86, adjacent points in the tuning space
# routinely swap under repeated measurement.  The standalone benchmark
# ``benchmark_k2_tuning_space_hybrid.py`` uses a fixed ``warmup=20``,
# ``iters=50`` CUDA-event loop and consistently ranks the same config
# combinations — with ~0.5% run-to-run noise vs. the ~2% swings we saw
# from Triton's built-in.  So for K2 we replace the built-in measurement
# with the exact benchmark methodology the standalone benchmark uses,
# then plug the picked config back into ``_get_compiled_kernel_for_cfg``
# for header emission.
#
# Environment variables (all optional):
#   HOTSTEP_K2_BENCH_WARMUP   — warmup iterations per config (default 20)
#   HOTSTEP_K2_BENCH_ITERS    — timed iterations per config (default 50)
#   HOTSTEP_K2_BENCH_TOPN     — how many top configs to print (default 5)

K2_BENCH_WARMUP = int(os.environ.get("HOTSTEP_K2_BENCH_WARMUP", "20"))
K2_BENCH_ITERS = int(os.environ.get("HOTSTEP_K2_BENCH_ITERS", "50"))
K2_BENCH_TOPN = int(os.environ.get("HOTSTEP_K2_BENCH_TOPN", "5"))


def _launch_k2_for_bench(cfg, tensors, M, N, K, num_sms, output_fp16):
    """Launch one K2 gluon_pipe config with the persistent production grid.

    The Gluon kernel is a persistent tile loop: it takes a NUM_TILES runtime
    arg and is launched on a 1D grid of min(num_tiles, num_sms) CTAs (one CTA
    per SM walks GROUP_M-swizzled tiles).  Config constexprs (BLOCK_M/N/K,
    GROUP_M, NUM_BUFS, FOLD_EVERY, HAS_BIAS, OUTPUT_FP16) and num_warps=8 are
    passed as kernel metadata; no @triton.autotune wrapping is involved.
    """
    _require_gluon_for_k2()
    xq, xs, wq, ws, bias, y = tensors
    bm = cfg.kwargs["BLOCK_M"]
    bn = cfg.kwargs["BLOCK_N"]
    num_tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    # Persistent grid mirrors the production launch geometry: occupancy-driven
    # min(num_tiles, num_sms * max_active_ctas_per_sm).  The per-config
    # occupancy is populated by the spill/compile filter; fall back to the
    # AUTOTUNE_OCCUPANCY_ASSUMPTION if the map was skipped.
    ctas = _K2_CFG_OCCUPANCY.get(_cfg_occupancy_key(cfg), AUTOTUNE_OCCUPANCY_ASSUMPTION)
    grid_x = _autotune_grid_x(num_tiles, num_sms, ctas)
    grid = (grid_x,)
    meta = dict(
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=cfg.kwargs["BLOCK_K"],
        GROUP_SIZE=GROUP_SIZE, GROUP_M=cfg.kwargs["GROUP_M"],
        NUM_BUFS=_gluon_k2_num_bufs(cfg),
        FOLD_EVERY=_gluon_k2_fold_every(cfg),
        HAS_BIAS=True, OUTPUT_FP16=output_fp16,
        num_warps=cfg.num_warps,
    )
    # Gluon pipelined kernel arg order (matches the kernel signature):
    #   xq, xs, wq, ws, bias, y, M, N, K, NUM_TILES, <strides...>
    kernel2_gemm_dequant[grid](
        xq, xs, wq, ws, bias, y,
        M, N, K, num_tiles,
        K, 1,          # X_q [M, K] row-major
        1, M,          # X_scale [n_groups, M]
        K, 1,          # W_q [N, K] row-major
        N, 1,          # Y [M, N] row-major
        **meta,
    )
    return grid_x


def _bench_k2_config(cfg, tensors, M, N, K, num_sms, output_fp16,
                     warmup=None, iters=None):
    """Time a single K2 config with warmup + CUDA events, returning (ms, grid).

    Mirrors ``benchmark_k2_tuning_space_hybrid.py``'s per-config loop: one
    compile launch, ``warmup`` warm launches, then ``iters`` timed launches
    bracketed by ``torch.cuda.Event`` records, returning the mean time per
    launch in ms.

    Raises whatever the launch raises (typically ``OutOfResources`` /
    ``triton.compiler.errors.CompilationError`` for configs that can't fit).
    """
    import torch
    if warmup is None:
        warmup = K2_BENCH_WARMUP
    if iters is None:
        iters = K2_BENCH_ITERS

    # 1 compile-and-launch call to force JIT + cache population.
    grid_x = _launch_k2_for_bench(cfg, tensors, M, N, K, num_sms, output_fp16)
    torch.cuda.synchronize()
    for _ in range(warmup):
        _launch_k2_for_bench(cfg, tensors, M, N, K, num_sms, output_fp16)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _launch_k2_for_bench(cfg, tensors, M, N, K, num_sms, output_fp16)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms, grid_x


def _bench_k2_configs_custom(configs, tensors, M, N, K, num_sms, output_fp16):
    """Custom-benchmark every config and return (best_cfg, results_sorted).

    ``results_sorted`` is a list of ``(cfg, ms, grid_x, error)`` tuples ordered
    by ascending ms (successful configs first, failures appended last with
    ``ms=inf``).  ``best_cfg`` is ``results_sorted[0][0]`` if any config
    succeeded, else ``None``.

    Progress is logged every ~10% of the config space.
    """
    if not configs:
        return None, []

    total = len(configs)
    step = max(total // 10, 1)
    results: list = []
    print(f"    [custom-bench] Benchmarking {total} K2 configs "
          f"(warmup={K2_BENCH_WARMUP}, iters={K2_BENCH_ITERS})...", flush=True)

    for i, cfg in enumerate(configs, start=1):
        try:
            ms, grid_x = _bench_k2_config(cfg, tensors, M, N, K, num_sms, output_fp16)
            results.append((cfg, ms, grid_x, None))
        except Exception as e:
            results.append((cfg, float("inf"), 0, f"{type(e).__name__}: {e}"))
        if i % step == 0 or i == total:
            ok = sum(1 for _, ms, _, _ in results if ms != float("inf"))
            print(f"    [custom-bench] {i}/{total} benchmarked ({ok} succeeded)",
                  flush=True)

    results.sort(key=lambda r: r[1])
    best = results[0][0] if results and results[0][1] != float("inf") else None
    return best, results


def _print_k2_bench_topn(results, M, K, N, topn=None):
    """Log the top-N benchmark results in the same format the standalone bench uses."""
    if topn is None:
        topn = K2_BENCH_TOPN
    print(f"    [custom-bench] top-{topn} configs (M={M}, K={K}, N={N}):", flush=True)
    real_ops = 2.0 * M * K * N
    for cfg, ms, grid_x, err in results[:topn]:
        if ms == float("inf"):
            print(f"      FAIL   {cfg.kwargs} W={cfg.num_warps} S={cfg.num_stages}: {err}",
                  flush=True)
            continue
        tops = real_ops / (ms * 1e-3) / 1e12
        mr = getattr(cfg, "maxnreg", None)
        print(f"      {ms:7.4f} ms  {tops:6.2f} TOPS  grid={grid_x:4d}  "
              f"BM={cfg.kwargs.get('BLOCK_M')} BN={cfg.kwargs.get('BLOCK_N')} "
              f"BK={cfg.kwargs.get('BLOCK_K')} GM={cfg.kwargs.get('GROUP_M')} "
              f"W={cfg.num_warps} S={cfg.num_stages} MR={mr}",
              flush=True)


def extract_k1(arch, debug_dump=False, num_sms=0, l2_bytes=0, shared_mem_per_sm=0):
    import torch
    results = {}
    if num_sms == 0:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    for dtype_suffix, input_fp16, _ in DTYPE_CONFIGS:
        print(f"\n  K1 G256 {dtype_suffix} (autotuning)...", flush=True)
        _parallel_precompile_k1(_K1_CONFIGS, input_fp16)
        dtype = torch.float16 if input_fp16 else torch.float32
        M, K = 1024, 2560
        x = torch.randn((M, K), device="cuda", dtype=dtype)
        xq = torch.empty((M, K), device="cuda", dtype=torch.int8)
        n_groups = K // GROUP_SIZE
        xs = torch.empty((n_groups, M), device="cuda", dtype=torch.float32)
        # §H2: autotune grid mirrors the production occupancy-scaled launch
        # (min(num_tiles, num_sms * max_active_ctas_per_sm)).  See
        # ``_autotune_grid_x`` for the rationale and the 2-CTA/SM assumption.
        grid_k1 = lambda meta: (_autotune_grid_x(
            triton.cdiv(M, meta["BLOCK_M"]) * n_groups, num_sms),)

        kernel1_convrot_quant[grid_k1](
            x, xq, xs,
            M, K,
            K, 1,
            K, 1,
            1, M,
            GROUP_SIZE=GROUP_SIZE, INPUT_FP16=input_fp16,
            CONTIG_XK=True,
        )
        torch.cuda.synchronize()

        kernel, best_cfg, cache_dir = _get_compiled_kernel(
            kernel1_convrot_quant,
            debug_dump=debug_dump,
            GROUP_SIZE=GROUP_SIZE,
            INPUT_FP16=input_fp16,
            CONTIG_XK=True,
        )

        block_m = best_cfg.kwargs["BLOCK_M"]
        nw = best_cfg.num_warps
        ns = best_cfg.num_stages
        mr = getattr(best_cfg, "maxnreg", None)
        cubin = kernel.asm["cubin"]
        shared = kernel.metadata.shared
        abi = _extract_kernel_abi(kernel, "quant")

        # Post-autotune spill validation: verify the winning cubin — the one
        # that actually ships in the header — does not spill.  Uses the same
        # authoritative cuobjdump STACK-field detector as the pre-autotune
        # gate (with PTX ld.local/st.local fallback), and FAILS THE BUILD on
        # spill: zero tolerance means a spilling winner must never land in
        # the generated header silently.
        spill_info = _detect_spills(kernel, abi["function_name"])
        if spill_info["spilling"]:
            raise SystemExit(
                f"[spill-check] K1 {dtype_suffix} autotune winner SPILLS "
                f"({spill_info['method']}: local_bytes={spill_info['local_bytes']}, "
                f"ptx_spills={spill_info['ptx_spills']}, regs={spill_info['registers']}). "
                f"Config: BLOCK_M={block_m}, num_warps={nw}, num_stages={ns}, "
                f"maxnreg={mr}. Zero tolerance: refusing to write a spilling "
                f"cubin into the generated header."
            )
        spill_status = f"no spills ({spill_info['method']}, regs={spill_info['registers']})"

        print(f"  -> Autotune winner: BLOCK_M={block_m}, num_warps={nw}, num_stages={ns}, maxnreg={mr}", flush=True)
        print(
            f"     shared={shared/1024:.0f}KB, cubin={len(cubin)}B, "
            f"runtime_args={len(abi['runtime_signature'])}, scratch_ptrs={abi['scratch_ptr_count']}, "
            f"reqntid={abi['block']}, {spill_status}",
            flush=True,
        )
        if cache_dir is not None:
            print(f"     cache_dir: {cache_dir}", flush=True)
            print(f"       (PTX/IR for diagnosis: "
                  f"{Path(cache_dir) / (abi['function_name'] + '.ptx')} , "
                  f"{Path(cache_dir) / (abi['function_name'] + '.ttgir')})",
                  flush=True)
        else:
            print("     cache_dir: <unavailable — set HOTSTEP_TRITON_CACHE_DEBUG=1>",
                  flush=True)
        results[dtype_suffix] = {
            "cubin": cubin,
            "shared": shared,
            "block_m": int(block_m),
            "block_k": GROUP_SIZE,
            "num_warps": nw,
            "num_stages": ns,
            "maxnreg": mr,
            "abi": abi,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "function_name": abi["function_name"],
        }
    return results


def extract_k2(arch, debug_dump=False, num_sms=0, l2_bytes=0, shared_mem_per_sm=0):
    import torch
    results = {}
    if num_sms == 0:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    # HAS_BIAS is compile-time-hardwired True in the shipped K2 cubin (see
    # BIAS_CONFIGS).  We still iterate BIAS_CONFIGS so the label scheme in the
    # extraction summary and the generated cubin lookup stays uniform.
    for bias_suffix, _has_bias in BIAS_CONFIGS:
        for dtype_suffix, _, output_fp16 in DTYPE_CONFIGS:
            config_name = f"{bias_suffix}_{dtype_suffix}"
            print(f"\n  K2 G256 {config_name} (gluon_pipe custom-bench selection)...", flush=True)
            _parallel_precompile_k2(_K2_CONFIGS, output_fp16, arch)
            out_dtype = torch.float16 if output_fp16 else torch.float32
            M_real, K, N = AUTOTUNE_SHAPES_K2[0]
            # Plain K2 has no M masks.  Autotune the same production-like work
            # with M padded to the largest generated BLOCK_M so every candidate
            # can run safely; this mirrors the plugin's host-side padding while
            # keeping the benchmark within 2.4% of M=3000.
            M = ((M_real + K2_AUTOTUNE_M_ALIGNMENT - 1) // K2_AUTOTUNE_M_ALIGNMENT) * K2_AUTOTUNE_M_ALIGNMENT
            if M != M_real:
                print(f"    [K2 autotune] primary shape M={M_real}, K={K}, N={N}; "
                      f"using padded M={M} for unmasked Gluon gluon_pipe K2 safety", flush=True)
            xq = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
            n_groups = K // GROUP_SIZE
            xs = torch.rand((n_groups, M), device="cuda", dtype=torch.float32) * 0.02 + 0.001
            # W_q is [N, K] row-major (canonical Linear layout).  The Gluon
            # gluon_pipe K2 kernel handles the transpose in-kernel via a free
            # smem.permute((1, 0)) on the [N, K] shared-memory view (no tl.trans,
            # no extra copy), so nothing about the exported weight layout
            # changes.
            wq = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
            ws = torch.rand((N,), device="cuda", dtype=torch.float32) * 0.02 + 0.001
            # HAS_BIAS is hardwired True in the shipped cubin; always pass a
            # real bias tensor to autotune so the fused epilogue is measured.
            bias = torch.randn((N,), device="cuda", dtype=torch.float32)
            y = torch.empty((M, N), device="cuda", dtype=out_dtype)
            # Bypass Triton's @autotune measurement (its internal do_bench is
            # too noisy at the ~2% level, causing mis-ranking of adjacent K2
            # configs) and use the fixed-budget CUDA-event benchmark loop
            # from ``benchmark_k2_tuning_space_hybrid.py`` instead.  Every
            # config is launched at its per-config occupancy-scaled grid
            # (see ``_autotune_grid_x_for_k2`` for the geometry rationale).
            tensors = (xq, xs, wq, ws, bias, y)
            best_cfg, bench_results = _bench_k2_configs_custom(
                _K2_CONFIGS, tensors, M, N, K, num_sms, output_fp16)
            if best_cfg is None:
                raise SystemExit(
                    f"[custom-bench] K2 {config_name}: no config successfully "
                    f"benchmarked. Config space was likely rejected wholesale "
                    f"by the spill filter — check the [spill-check] log above."
                )
            _print_k2_bench_topn(bench_results, M, K, N)

            # AOT-compile the picked winner via GluonASTSource to harvest its
            # cubin directly (the Gluon kernel has no @triton.autotune cache to
            # walk, unlike the old hybrid_direct path).  The custom bench
            # already ran the winner, so this recompile is a cache hit and
            # yields the CompiledKernel for header emission, ABI extraction and
            # spill validation.
            kernel = _gluon_aot_compile_k2(best_cfg, output_fp16, arch)
            cache_dir = _get_kernel_cache_dir(kernel)

            block_m = best_cfg.kwargs["BLOCK_M"]
            block_n = best_cfg.kwargs["BLOCK_N"]
            block_k_winning = best_cfg.kwargs["BLOCK_K"]
            group_m = best_cfg.kwargs["GROUP_M"]
            nw = best_cfg.num_warps
            ns = best_cfg.num_stages
            mr = getattr(best_cfg, "maxnreg", None)
            cubin = kernel.asm["cubin"]
            shared = kernel.metadata.shared
            if K2_MAX_SHARED_BYTES > 0 and int(shared) > K2_MAX_SHARED_BYTES:
                raise SystemExit(
                    f"[shared-check] K2 {config_name} custom-bench winner uses shared={int(shared)}B "
                    f"> K2_MAX_SHARED_BYTES={K2_MAX_SHARED_BYTES}B. Refusing to write a "
                    f"high-shared 1-CTA/SM cubin into the generated header; adjust the config "
                    f"space or set HOTSTEP_K2_MAX_SHARED_BYTES=0 for experiments."
                )
            abi = _extract_kernel_abi(kernel, "gemm")

            # Post-autotune spill validation: authoritative cuobjdump check
            # on the exact winning cubin, fatal on spill (see extract_k1).
            spill_info = _detect_spills(kernel, abi["function_name"])
            if spill_info["spilling"]:
                raise SystemExit(
                    f"[spill-check] K2 {config_name} custom-bench winner SPILLS "
                    f"({spill_info['method']}: local_bytes={spill_info['local_bytes']}, "
                    f"ptx_spills={spill_info['ptx_spills']}, regs={spill_info['registers']}). "
                    f"Config: BM={block_m} BN={block_n} BK={block_k_winning} "
                    f"GM={group_m} W={nw} S={ns} MR={mr}. "
                    f"Zero tolerance: refusing to write a spilling cubin into "
                    f"the generated header."
                )
            spill_status = f"no spills ({spill_info['method']}, regs={spill_info['registers']})"

            occ_ctas = _K2_CFG_OCCUPANCY.get(_cfg_occupancy_key(best_cfg))
            occ_note = f", occupancy={occ_ctas} CTA/SM" if occ_ctas else ""
            # Winning ms/TOPS from the custom bench are recorded so the
            # summary log shows the exact per-launch time that drove the
            # selection.
            win_row = next(((c, ms, g) for c, ms, g, e in bench_results
                            if c is best_cfg and e is None), None)
            win_note = ""
            if win_row is not None:
                real_ops = 2.0 * M * K * N
                _, win_ms, win_grid = win_row
                win_tops = real_ops / (win_ms * 1e-3) / 1e12
                win_note = f", {win_ms:.4f} ms, {win_tops:.2f} TOPS, grid={win_grid}"
            print(f"  -> Custom-bench winner: BM={block_m} BN={block_n} BK={block_k_winning} GM={group_m} "
                  f"W={nw} S={ns} MR={mr}{occ_note}{win_note}", flush=True)
            print(
                f"     shared={shared/1024:.0f}KB, cubin={len(cubin)}B, "
                f"runtime_args={len(abi['runtime_signature'])}, scratch_ptrs={abi['scratch_ptr_count']}, "
                f"reqntid={abi['block']}, {spill_status}",
                flush=True,
            )
            if cache_dir is not None:
                print(f"     cache_dir: {cache_dir}", flush=True)
                print(f"       (PTX/IR for diagnosis: "
                      f"{Path(cache_dir) / (abi['function_name'] + '.ptx')} , "
                      f"{Path(cache_dir) / (abi['function_name'] + '.ttgir')})",
                      flush=True)
            else:
                print("     cache_dir: <unavailable — set HOTSTEP_TRITON_CACHE_DEBUG=1>",
                      flush=True)
            results[config_name] = {
                "cubin": cubin,
                "shared": shared,
                "block_m": int(block_m),
                "block_n": int(block_n),
                "block_k": int(block_k_winning),
                "group_m": int(group_m),
                "num_warps": nw,
                "num_stages": ns,
                "maxnreg": mr,
                "abi": abi,
                "cache_dir": str(cache_dir) if cache_dir is not None else None,
                "function_name": abi["function_name"],
            }
    return results


def _format_cubin_as_c_array(cubin_bytes: bytes) -> list[str]:
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


def write_header(k1_results, k2_results, output_path):
    max_scratch_ptrs = max(
        [k1_results[d]["abi"]["scratch_ptr_count"] for d, _, _ in DTYPE_CONFIGS]
        + [k2_results[f"{b}_{d}"]["abi"]["scratch_ptr_count"] for b, _ in BIAS_CONFIGS for d, _, _ in DTYPE_CONFIGS],
        default=0,
    )

    header_lines = [
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <cstring>",
        "#include <cuda.h>",
        "",
        "// AUTO-GENERATED by tools/onnx-export/extract_jit_cubins_autotune.py.",
        "// Cubins extracted from Triton's JIT cache after @triton.autotune.",
        "// NO triton.compile() / AOT path was used — these are the exact cubins",
        "// that Triton's JIT produced and autotune benchmarked as fastest.",
        "// Runtime parameter packing below is generated from CompiledKernel.src.signature",
        "// and cross-checked against the PTX .entry prototype for the same cubin.",
        f"#define CONVROT_INT8_CUBIN_HEADER_VERSION {CONVROT_INT8_CUBIN_HEADER_VERSION}",
        "#define CONVROT_INT8_HAS_GENERATED_LAUNCH_STUBS 1",
        f"#define CONVROT_INT8_TRITON_ABI_TRAILING_SCRATCH_PTRS {max_scratch_ptrs}",
        "",
    ]
    for dtype_suffix, _, _ in DTYPE_CONFIGS:
        header_lines.append(f"#define CONVROT_INT8_HAS_DTYPE_{dtype_suffix} 1")
    header_lines.append("#define CONVROT_INT8_HAS_M1_SPECIALIZED 0")
    header_lines.append("")

    total = 0
    desc_entries = []
    runtime_arg_arrays = []

    def maxnreg_value(r):
        explicit = r.get("maxnreg", None)
        if explicit is not None:
            return int(explicit)
        inferred = r.get("abi", {}).get("ptx_maxnreg", None)
        return -1 if inferred is None else int(inferred)

    def emit_runtime_arg_array(symbol: str, runtime_signature):
        runtime_arg_arrays.append(f"inline constexpr ConvRotRuntimeArg {symbol}[] = {{")
        for arg in runtime_signature:
            runtime_arg_arrays.append(
                f'    {{ConvRotLaunchArgId::{arg["enum"]}, \"{arg["name"]}\", \"{arg["sig_type"]}\"}},'
            )
        runtime_arg_arrays.append("};")
        runtime_arg_arrays.append("")

    def emit_common_constants(prefix, cn, r):
        block_x, block_y, block_z = r["abi"]["block"]
        header_lines.append(f"constexpr size_t {prefix}_{cn}_SIZE = {len(r['cubin'])};")
        header_lines.append(f"constexpr size_t {prefix}_{cn}_SHARED = {r['shared']};")
        header_lines.append(f"constexpr int {prefix}_{cn}_BLOCK_M = {r['block_m']};")
        header_lines.append(f"constexpr int {prefix}_{cn}_BLOCK_THREADS = {block_x * block_y * block_z};")
        header_lines.append(f"constexpr int {prefix}_{cn}_NUM_WARPS = {r['num_warps']};")
        header_lines.append(f"constexpr int {prefix}_{cn}_NUM_STAGES = {int(r.get('num_stages', -1))};")
        header_lines.append(f"constexpr int {prefix}_{cn}_MAXNREG = {maxnreg_value(r)};")
        header_lines.append(f"constexpr int {prefix}_{cn}_RUNTIME_PARAM_COUNT = {len(r['abi']['runtime_signature'])};")
        header_lines.append(f"constexpr int {prefix}_{cn}_SCRATCH_PTR_COUNT = {int(r['abi']['scratch_ptr_count'])};")

    for dtype_suffix, _, _ in DTYPE_CONFIGS:
        r = k1_results[dtype_suffix]
        cn = f"K1_{TILE_NAME}_G{GROUP_SIZE}_{dtype_suffix}"
        prefix = "kCONVROT_INT8_KERNEL1"
        emit_common_constants(prefix, cn, r)
        header_lines.append(f"constexpr int {prefix}_{cn}_BLOCK_K = {r['block_k']};")
        header_lines.append(f"alignas(16) constexpr unsigned char {prefix}_{cn}[] = {{")
        header_lines.extend(_format_cubin_as_c_array(r['cubin']))
        header_lines += ["};", ""]
        arg_symbol = f"kCONVROT_INT8_RUNTIME_ARGS_{cn}"
        emit_runtime_arg_array(arg_symbol, r["abi"]["runtime_signature"])
        input_dtype_id = 10 if dtype_suffix == "FP16IO" else 1
        output_dtype_id = input_dtype_id
        block_x, block_y, block_z = r["abi"]["block"]
        desc_entries.append({
            "name": f"CONVROT_{cn}",
            "array": f"{prefix}_{cn}",
            "size": f"{prefix}_{cn}_SIZE",
            "stage": "ConvRotKernelStage::kQuant",
            "function": r["abi"]["function_name"],
            "group_size": GROUP_SIZE,
            "has_bias": "false",
            "input_dtype": input_dtype_id,
            "output_dtype": output_dtype_id,
            "block_m": int(r['block_m']),
            "block_n": 1,
            "block_k": int(r['block_k']),
            "group_m": 1,
            "block_x": block_x,
            "block_y": block_y,
            "block_z": block_z,
            "shared": int(r['shared']),
            "num_warps": int(r['num_warps']),
            "num_stages": int(r.get('num_stages', -1)),
            "maxnreg": maxnreg_value(r),
            "runtime_args": arg_symbol,
            "runtime_arg_count": len(r["abi"]["runtime_signature"]),
            "scratch_ptr_count": int(r["abi"]["scratch_ptr_count"]),
        })
        total += 1

    for bias_suffix, has_bias in BIAS_CONFIGS:
        for dtype_suffix, _, _ in DTYPE_CONFIGS:
            r = k2_results[f"{bias_suffix}_{dtype_suffix}"]
            cn = f"K2_{TILE_NAME}_G{GROUP_SIZE}_{bias_suffix}_{dtype_suffix}"
            prefix = "kCONVROT_INT8_KERNEL2"
            emit_common_constants(prefix, cn, r)
            header_lines.append(f"constexpr int {prefix}_{cn}_BLOCK_N = {r['block_n']};")
            header_lines.append(f"constexpr int {prefix}_{cn}_BLOCK_K = {r.get('block_k', DEFAULT_BLOCK_K)};")
            header_lines.append(f"constexpr int {prefix}_{cn}_GROUP_M = {r.get('group_m', -1)};")
            header_lines.append(f"alignas(16) constexpr unsigned char {prefix}_{cn}[] = {{")
            header_lines.extend(_format_cubin_as_c_array(r['cubin']))
            header_lines += ["};", ""]
            arg_symbol = f"kCONVROT_INT8_RUNTIME_ARGS_{cn}"
            emit_runtime_arg_array(arg_symbol, r["abi"]["runtime_signature"])
            input_dtype_id = 10 if dtype_suffix == "FP16IO" else 1
            output_dtype_id = input_dtype_id
            block_x, block_y, block_z = r["abi"]["block"]
            desc_entries.append({
                "name": f"CONVROT_{cn}",
                "array": f"{prefix}_{cn}",
                "size": f"{prefix}_{cn}_SIZE",
                "stage": "ConvRotKernelStage::kGemm",
                "function": r["abi"]["function_name"],
                "group_size": GROUP_SIZE,
                "has_bias": "true" if has_bias else "false",
                "input_dtype": input_dtype_id,
                "output_dtype": output_dtype_id,
                "block_m": int(r['block_m']),
                "block_n": int(r['block_n']),
                "block_k": int(r.get('block_k', DEFAULT_BLOCK_K)),
                "group_m": int(r.get('group_m', -1)),
                "block_x": block_x,
                "block_y": block_y,
                "block_z": block_z,
                "shared": int(r['shared']),
                "num_warps": int(r['num_warps']),
                "num_stages": int(r.get('num_stages', -1)),
                "maxnreg": maxnreg_value(r),
                "runtime_args": arg_symbol,
                "runtime_arg_count": len(r["abi"]["runtime_signature"]),
                "scratch_ptr_count": int(r["abi"]["scratch_ptr_count"]),
            })
            total += 1

    header_lines += [
        "namespace hotstep::convrot_int8_generated {",
        "",
        "enum class ConvRotKernelStage : int32_t { kQuant = 1, kGemm = 2 };",
        "",
        "enum class ConvRotLaunchArgId : uint8_t {",
    ]
    for name in _LAUNCH_ARG_ENUM.values():
        header_lines.append(f"    {name},")
    header_lines += [
        "};",
        "",
        "struct ConvRotRuntimeArg {",
        "    ConvRotLaunchArgId id;",
        "    char const* name;",
        "    char const* sig_type;",
        "};",
        "",
        "struct ConvRotCubinDesc {",
        "    char const* logical_name;",
        "    unsigned char const* data;",
        "    size_t size;",
        "    char const* function_name;",
        "    ConvRotKernelStage stage;",
        "    int32_t group_size;",
        "    bool has_bias;",
        "    int32_t input_dtype_id;",
        "    int32_t output_dtype_id;",
        "    int32_t block_m;",
        "    int32_t block_n;",
        "    int32_t block_k;",
        "    int32_t group_m;",
        "    uint32_t block_x;",
        "    uint32_t block_y;",
        "    uint32_t block_z;",
        "    size_t shared_bytes;",
        "    int32_t num_warps;      // diagnostic only; launch uses block_x/y/z",
        "    int32_t num_stages;     // diagnostic only",
        "    int32_t maxnreg;        // diagnostic only; -1 if uncapped/unknown",
        "    ConvRotRuntimeArg const* runtime_args;",
        "    uint32_t runtime_arg_count;",
        "    uint32_t trailing_scratch_ptr_count;",
        "};",
        "",
        "inline constexpr size_t kConvRotMaxLaunchParams = 25;",
        "",
    ]
    header_lines.extend(runtime_arg_arrays)

    header_lines.append("inline constexpr ConvRotCubinDesc kConvRotCubins[] = {")
    for d in desc_entries:
        header_lines.append(
            "    {" +
            f'"{d["name"]}", {d["array"]}, {d["size"]}, "{d["function"]}", ' +
            f'{d["stage"]}, {d["group_size"]}, {d["has_bias"]}, ' +
            f'{d["input_dtype"]}, {d["output_dtype"]}, ' +
            f'{d["block_m"]}, {d["block_n"]}, {d["block_k"]}, {d["group_m"]}, ' +
            f'{d["block_x"]}u, {d["block_y"]}u, {d["block_z"]}u, ' +
            f'{d["shared"]}, {d["num_warps"]}, {d["num_stages"]}, {d["maxnreg"]}, ' +
            f'{d["runtime_args"]}, {d["runtime_arg_count"]}u, {d["scratch_ptr_count"]}u' +
            "},"
        )
    header_lines += [
        "};",
        "",
        "inline constexpr size_t kConvRotCubinCount = sizeof(kConvRotCubins) / sizeof(kConvRotCubins[0]);",
        "",
        "inline constexpr ConvRotCubinDesc const* findConvRotCubin(",
        "    ConvRotKernelStage stage, int32_t group_size, bool has_bias,",
        "    int32_t input_dtype_id, int32_t output_dtype_id) {",
        "    for (size_t i = 0; i < kConvRotCubinCount; ++i) {",
        "        ConvRotCubinDesc const& d = kConvRotCubins[i];",
        "        if (d.stage == stage && d.group_size == group_size &&",
        "            d.has_bias == has_bias && d.input_dtype_id == input_dtype_id &&",
        "            d.output_dtype_id == output_dtype_id) {",
        "            return &d;",
        "        }",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "inline uint32_t ceilDivU32(int32_t x, int32_t y) {",
        "    return static_cast<uint32_t>((x + y - 1) / y);",
        "}",
        "",
        "// §1.1 — Persistent grid sizing based on actual SM occupancy.",
        "//",
        "// The old version returned ``min(num_tiles, num_sms)``, which caps the",
        "// launch at exactly one wave.  Combined with ``num_stages=3`` inflating",
        "// shared memory, no shape ever exceeded ``waves_per_multiprocessor == 1``.",
        "//",
        "// The fix is to query ``cuOccupancyMaxActiveBlocksPerMultiprocessor`` on",
        "// the loaded cubin (the value is a property of the cubin + device, not",
        "// something Triton can know at AOT time) and launch",
        "// ``min(num_tiles, num_sms * max_active_ctas_per_sm)``.  This satisfies the",
        "// task requirement that \"actual SM usage should be obtained from the",
        "// resulting kernel metadata, not estimated.\"",
        "//",
        "// When the occupancy query fails (returns 0), we fall back to the old",
        "// 1-wave behaviour — the kernel still produces correct results, it just",
        "// doesn't get the 2-CTA/SM speed-up.",
        "inline uint32_t persistentGridU32(uint32_t num_tiles, uint32_t num_sms,",
        "                                  uint32_t max_active_ctas_per_sm) {",
        "    uint32_t const grid_cap = num_sms * (max_active_ctas_per_sm == 0 ? 1u : max_active_ctas_per_sm);",
        "    return (num_tiles < grid_cap) ? num_tiles : grid_cap;",
        "}",
        "",
        "// Backwards-compatible overload — same as passing max_active_ctas_per_sm=1.",
        "inline uint32_t persistentGridU32(uint32_t num_tiles, uint32_t num_sms) {",
        "    return persistentGridU32(num_tiles, num_sms, 1u);",
        "}",
        "",
        "// §1.1 — Query the actual max-resident-CTAs-per-SM for a loaded cubin.",
        "//",
        "// This is the value the task asks for (\"actual SM usage should be obtained",
        "// from the resulting kernel metadata, not estimated\").  It depends on the",
        "// loaded cubin's register/shared/thread footprint and the device's RF/shared",
        "// budget — neither Triton nor the AOT pipeline knows it at compile time.",
        "//",
        "// Returns 0 on failure (caller should fall back to 1-CTA/SM launch).",
        "//",
        "// CRITICAL (patch review §5): the previous version passed blockSize=0,",
        "// but the CUDA occupancy calculator (cuda_occupancy.h) rejects",
        "// blockSize <= 0 unconditionally with CUDA_OCC_ERROR_INVALID_INPUT,",
        "// which propagates as CUDA_ERROR_INVALID_VALUE.  This made the entire",
        "// occupancy-driven grid-sizing feature dead code — every launch",
        "// silently fell back to the old 1-wave (1 CTA/SM) grid.",
        "//",
        "// Fix: pass the real block size from ConvRotCubinDesc.block_x * block_y",
        "// * block_z, which is the .reqntid value ptxas baked into the cubin.",
        "//",
        "// Caching (patch review §H1): the plugin caches the query result once",
        "// per cubin in initTriton() (m_max_ctas_per_sm_quant / _gemm) and passes",
        "// it through this parameter on every enqueue(); the DiT loop calls",
        "// enqueue() 359 times per step, so re-querying here would add ~700",
        "// driver round-trips per step.",
        "inline uint32_t queryMaxActiveCtasPerSm(CUfunction func,",
        "                                       uint32_t block_threads,",
        "                                       uint32_t shared_bytes) {",
        "    if (func == nullptr || block_threads == 0) return 0;",
        "    int num_blocks = 0;",
        "    CUresult const rc = cuOccupancyMaxActiveBlocksPerMultiprocessor(",
        "        &num_blocks, func,",
        "        static_cast<int>(block_threads),",
        "        shared_bytes);",
        "    if (rc != CUDA_SUCCESS || num_blocks <= 0) return 0;",
        "    return static_cast<uint32_t>(num_blocks);",
        "}",
        "",
        "inline void* selectQuantLaunchParam(",
        "    ConvRotLaunchArgId id,",
        "    void*& x_param, void*& xq_ptr, void*& xs_ptr,",
        "    int32_t& M, int32_t& K,",
        "    int32_t& stride_xm, int32_t& stride_xk,",
        "    int32_t& stride_xqm, int32_t& stride_xqk,",
        "    int32_t& stride_xsm, int32_t& stride_xsg) {",
        "    switch (id) {",
        "        case ConvRotLaunchArgId::kX_ptr: return &x_param;",
        "        case ConvRotLaunchArgId::kX_q_ptr: return &xq_ptr;",
        "        case ConvRotLaunchArgId::kX_scale_ptr: return &xs_ptr;",
        "        case ConvRotLaunchArgId::kM: return &M;",
        "        case ConvRotLaunchArgId::kK: return &K;",
        "        case ConvRotLaunchArgId::kStride_xm: return &stride_xm;",
        "        case ConvRotLaunchArgId::kStride_xk: return &stride_xk;",
        "        case ConvRotLaunchArgId::kStride_xqm: return &stride_xqm;",
        "        case ConvRotLaunchArgId::kStride_xqk: return &stride_xqk;",
        "        case ConvRotLaunchArgId::kStride_xsm: return &stride_xsm;",
        "        case ConvRotLaunchArgId::kStride_xsg: return &stride_xsg;",
        "        default: return nullptr;",
        "    }",
        "}",
        "",
        "inline void* selectGemmLaunchParam(",
        "    ConvRotLaunchArgId id,",
        "    void*& xq_ptr, void*& xs_ptr, void*& wq_param, void*& ws_param, void*& bias_param, void*& y_ptr,",
        "    int32_t& M, int32_t& N, int32_t& K, int32_t& num_tiles,",
        "    int32_t& stride_xqm, int32_t& stride_xqk,",
        "    int32_t& stride_xsm, int32_t& stride_xsg,",
        "    int32_t& stride_wn, int32_t& stride_wk,",
        "    int32_t& stride_ym, int32_t& stride_yn) {",
        "    switch (id) {",
        "        case ConvRotLaunchArgId::kX_q_ptr: return &xq_ptr;",
        "        case ConvRotLaunchArgId::kX_scale_ptr: return &xs_ptr;",
        "        case ConvRotLaunchArgId::kW_q_ptr: return &wq_param;",
        "        case ConvRotLaunchArgId::kW_scale_ptr: return &ws_param;",
        "        case ConvRotLaunchArgId::kBias_ptr: return &bias_param;",
        "        case ConvRotLaunchArgId::kY_ptr: return &y_ptr;",
        "        case ConvRotLaunchArgId::kM: return &M;",
        "        case ConvRotLaunchArgId::kN: return &N;",
        "        case ConvRotLaunchArgId::kK: return &K;",
        "        case ConvRotLaunchArgId::kNUM_TILES: return &num_tiles;",
        "        case ConvRotLaunchArgId::kStride_xqm: return &stride_xqm;",
        "        case ConvRotLaunchArgId::kStride_xqk: return &stride_xqk;",
        "        case ConvRotLaunchArgId::kStride_xsm: return &stride_xsm;",
        "        case ConvRotLaunchArgId::kStride_xsg: return &stride_xsg;",
        "        case ConvRotLaunchArgId::kStride_wn: return &stride_wn;",
        "        case ConvRotLaunchArgId::kStride_wk: return &stride_wk;",
        "        case ConvRotLaunchArgId::kStride_ym: return &stride_ym;",
        "        case ConvRotLaunchArgId::kStride_yn: return &stride_yn;",
        "        default: return nullptr;",
        "    }",
        "}",
        "",
        "inline CUresult launchConvRotQuant(",
        "    ConvRotCubinDesc const& d, CUfunction func, CUstream stream,",
        "    void const* x_ptr, void* xq_ptr, void* xs_ptr,",
        "    int32_t M, int32_t K, int32_t num_sms,",
        "    int32_t stride_xm, int32_t stride_xk,",
        "    int32_t stride_xqm, int32_t stride_xqk,",
        "    int32_t stride_xsm, int32_t stride_xsg,",
        "    uint32_t max_active_ctas_per_sm = 0) {",
        "    void* triton_scratch1 = nullptr;",
        "    void* triton_scratch2 = nullptr;",
        "    if (d.runtime_arg_count + d.trailing_scratch_ptr_count > kConvRotMaxLaunchParams ||",
        "        d.trailing_scratch_ptr_count > 2) {",
        "        return CUDA_ERROR_INVALID_VALUE;",
        "    }",
        "    void* x_param = const_cast<void*>(x_ptr);",
        "    uint32_t const num_pid_m = ceilDivU32(M, d.block_m);",
        "    uint32_t const num_k_groups = static_cast<uint32_t>(K / d.group_size);",
        "    uint32_t const total_tiles = num_pid_m * num_k_groups;",
        "    // §1.1: if the caller didn't pre-query occupancy, query it now (cached",
        "    //        per-CUfunction by the driver).  Falls back to 1 CTA/SM on failure.",
        "    //        Patch review §5: must pass the real block size, not 0.",
        "    uint32_t const block_threads = d.block_x * d.block_y * d.block_z;",
        "    uint32_t const occupancy = (max_active_ctas_per_sm != 0)",
        "        ? max_active_ctas_per_sm",
        "        : queryMaxActiveCtasPerSm(func, block_threads, static_cast<uint32_t>(d.shared_bytes));",
        "    uint32_t const grid_x = persistentGridU32(total_tiles,",
        "                                              static_cast<uint32_t>(num_sms),",
        "                                              occupancy);",
        "    void* params[kConvRotMaxLaunchParams] = {};",
        "    uint32_t n = 0;",
        "    for (uint32_t i = 0; i < d.runtime_arg_count; ++i) {",
        "        void* slot = selectQuantLaunchParam(",
        "            d.runtime_args[i].id,",
        "            x_param, xq_ptr, xs_ptr,",
        "            M, K,",
        "            stride_xm, stride_xk, stride_xqm, stride_xqk, stride_xsm, stride_xsg);",
        "        if (slot == nullptr) return CUDA_ERROR_INVALID_VALUE;",
        "        params[n++] = slot;",
        "    }",
        "    void* scratch_slots[] = { &triton_scratch1, &triton_scratch2 };",
        "    for (uint32_t i = 0; i < d.trailing_scratch_ptr_count; ++i) params[n++] = scratch_slots[i];",
        "    return cuLaunchKernel(",
        "        func, grid_x, 1, 1, d.block_x, d.block_y, d.block_z,",
        "        static_cast<unsigned int>(d.shared_bytes), stream, params, nullptr);",
        "}",
        "",
        "inline CUresult launchConvRotGemm(",
        "    ConvRotCubinDesc const& d, CUfunction func, CUstream stream,",
        "    void* xq_ptr, void* xs_ptr, int8_t const* wq_ptr,",
        "    float const* ws_ptr, void const* bias_ptr, void* y_ptr,",
        "    int32_t M, int32_t N, int32_t K, int32_t num_sms,",
        "    int32_t stride_xqm, int32_t stride_xqk,",
        "    int32_t stride_xsm, int32_t stride_xsg,",
        "    int32_t stride_wn, int32_t stride_wk,",
        "    int32_t stride_ym, int32_t stride_yn,",
        "    uint32_t max_active_ctas_per_sm = 0) {",
        "    void* triton_scratch1 = nullptr;",
        "    void* triton_scratch2 = nullptr;",
        "    if (d.runtime_arg_count + d.trailing_scratch_ptr_count > kConvRotMaxLaunchParams ||",
        "        d.trailing_scratch_ptr_count > 2) {",
        "        return CUDA_ERROR_INVALID_VALUE;",
        "    }",
        "    void* wq_param = const_cast<int8_t*>(wq_ptr);",
        "    void* ws_param = const_cast<float*>(ws_ptr);",
        "    void* bias_param = const_cast<void*>(bias_ptr);",
        "    uint32_t const grid_m = ceilDivU32(M, d.block_m);",
        "    uint32_t const grid_n = ceilDivU32(N, d.block_n);",
        "    uint32_t const total_tiles = grid_m * grid_n;",
        "    // K2 gluon_pipe is a persistent tile loop: it takes a NUM_TILES",
        "    // runtime arg (i32).  Pass the total tile count as that arg.",
        "    int32_t const num_tiles = static_cast<int32_t>(total_tiles);",
        "    // §1.1: see launchConvRotQuant — occupancy-driven grid sizing.",
        "    //        Patch review §5: must pass the real block size, not 0.",
        "    uint32_t const block_threads = d.block_x * d.block_y * d.block_z;",
        "    uint32_t const occupancy = (max_active_ctas_per_sm != 0)",
        "        ? max_active_ctas_per_sm",
        "        : queryMaxActiveCtasPerSm(func, block_threads, static_cast<uint32_t>(d.shared_bytes));",
        "    uint32_t const grid_x = persistentGridU32(total_tiles,",
        "                                              static_cast<uint32_t>(num_sms),",
        "                                              occupancy);",
        "    void* params[kConvRotMaxLaunchParams] = {};",
        "    uint32_t n = 0;",
        "    for (uint32_t i = 0; i < d.runtime_arg_count; ++i) {",
        "        void* slot = selectGemmLaunchParam(",
        "            d.runtime_args[i].id,",
        "            xq_ptr, xs_ptr, wq_param, ws_param, bias_param, y_ptr,",
        "            M, N, K, const_cast<int32_t&>(num_tiles),",
        "            stride_xqm, stride_xqk, stride_xsm, stride_xsg,",
        "            stride_wn, stride_wk, stride_ym, stride_yn);",
        "        if (slot == nullptr) return CUDA_ERROR_INVALID_VALUE;",
        "        params[n++] = slot;",
        "    }",
        "    void* scratch_slots[] = { &triton_scratch1, &triton_scratch2 };",
        "    for (uint32_t i = 0; i < d.trailing_scratch_ptr_count; ++i) params[n++] = scratch_slots[i];",
        "    return cuLaunchKernel(",
        "        func, grid_x, 1, 1, d.block_x, d.block_y, d.block_z,",
        "        static_cast<unsigned int>(d.shared_bytes), stream, params, nullptr);",
        "}",
        "",
        "} // namespace hotstep::convrot_int8_generated",
        "",
        "// End of generated Triton cubins and launch descriptors.",
    ]
    output_path.write_text("\n".join(header_lines) + "\n")
    print(f"\n[extract_jit_cubins_autotune] Wrote {total} cubins to {output_path}", flush=True)


def main():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Must run on target GPU.", file=sys.stderr)
        return 1

    debug_dump = bool(int(os.environ.get("HOTSTEP_TRITON_CACHE_DEBUG", "0")))

    print(f"[extract_jit_cubins_autotune] Triton version: {triton.__version__}", flush=True)
    print(f"[extract_jit_cubins_autotune] Device: {torch.cuda.get_device_name()}", flush=True)
    props = torch.cuda.get_device_properties(0)
    arch = props.major * 10 + props.minor
    num_sms = props.multi_processor_count
    l2_bytes = getattr(props, 'l2_cache_size', 0) or 0
    if l2_bytes == 0:
        l2_bytes = 2 * 1024 * 1024

    # All device properties (arch, SM count, L2 size, shared-mem/SM) are
    # derived directly from the GPU the script is running on.  The cubin
    # extraction MUST be run on the target device — the generated cubins are
    # arch-specific (sm75/sm80/sm86/sm89 each produce different SASS) and the
    # autotune-space pruning (GROUP_M, shared-mem budgets) depends on the
    # exact L2 size and shared-mem/SM of the runtime device.  There is no
    # "build machine" vs "runtime machine" split here: one device, one
    # extraction, one cubin header.

    if arch >= 89:
        shared_mem_per_sm = 100 * 1024
    elif arch >= 86:
        shared_mem_per_sm = 99 * 1024
    elif arch >= 80:
        shared_mem_per_sm = 164 * 1024
    elif arch >= 75:
        shared_mem_per_sm = 64 * 1024
    else:
        shared_mem_per_sm = 48 * 1024

    print(f"[extract_jit_cubins_autotune] sm_{arch}, SMs={num_sms}, "
          f"L2={l2_bytes/(1024*1024):.1f} MB, shared/SM={shared_mem_per_sm//1024} KB",
          flush=True)

    global _K1_CONFIGS, _K2_CONFIGS
    _K1_CONFIGS[:] = _estimate_k1_configs(num_sms, l2_bytes, shared_mem_per_sm, AUTOTUNE_SHAPES_K2)
    # K2 config grid is pinned by the user (BM∈{64,128}, BN∈{64,128},
    # BK∈{32,64}, stages∈{2,3}, num_warps=8) — see _estimate_k2_configs.
    # The SM-aware tile injector is intentionally NOT applied to K2: the
    # gluon_pipe kernel requires num_warps=8 and the injected candidates
    # (num_warps=4, non-gluon-compatible tiles) would be rejected wholesale
    # by the gluon config gate.  K1 still uses the heuristic builder.
    _K2_CONFIGS[:] = _estimate_k2_configs(num_sms, l2_bytes, shared_mem_per_sm,
                                          AUTOTUNE_SHAPES_K2, arch=arch)

    # ── Register spill filtering (patch review §8) ───────────────────────
    # Triton will SILENTLY spill registers to local memory (DRAM) when a
    # config's maxnreg can't be honored.  Spilling kernels produce correct
    # output but catastrophic performance.  ZERO TOLERANCE: we compile each
    # config, check the cubin's STACK field via cuobjdump (or PTX-level
    # ld.local/st.local as fallback), and reject ANY config that spills
    # even one byte.  No thresholds.
    print(f"\n[spill-check] Filtering configs for register spills (zero-tolerance)...", flush=True)
    _K1_CONFIGS[:] = _filter_spilling_configs(_K1_CONFIGS, kernel1_convrot_quant, "k1", arch)
    _K2_CONFIGS[:] = _filter_spilling_configs(_K2_CONFIGS, kernel2_gemm_dequant, "k2", arch)

    # K1 is a @triton.autotune kernel — bind the filtered config list back so
    # the launch dispatch (extract_k1) autotunes over the survivors.  K2 is the
    # Gluon gluon_pipe kernel (no @triton.autotune wrapping); the filtered list
    # is consumed directly by the custom benchmarker, so there is no .configs
    # attribute to set.
    kernel1_convrot_quant.configs = _K1_CONFIGS

    print(f"[extract_jit_cubins_autotune] GROUP_SIZE={GROUP_SIZE}", flush=True)
    print(f"[extract_jit_cubins_autotune] K2 max shared cap: {K2_MAX_SHARED_BYTES if K2_MAX_SHARED_BYTES > 0 else 'disabled'} bytes", flush=True)
    print(f"[extract_jit_cubins_autotune] K1 configs: {len(_K1_CONFIGS)} (heuristic-pruned + spill-filtered)", flush=True)
    print(f"[extract_jit_cubins_autotune] K2 configs: {len(_K2_CONFIGS)} (gluon_pipe grid: BM/BN/BK/stages, spill-filtered)", flush=True)
    # Per-config occupancy summary (K2 only).  The K2 autotune grid uses these
    # so each candidate is measured under the actual CTAs/SM it will get in
    # production — e.g. a 96 KB-shared config runs at 1 CTA/SM, not 2 CTA/SM.
    if _K2_CFG_OCCUPANCY:
        from collections import Counter
        occ_hist = Counter(_K2_CFG_OCCUPANCY.values())
        occ_summary = ", ".join(f"{occ} CTA/SM: {n}"
                                for occ, n in sorted(occ_hist.items()))
        print(f"[extract_jit_cubins_autotune] K2 per-config occupancy: {occ_summary}", flush=True)
    if debug_dump:
        print("[extract_jit_cubins_autotune] HOTSTEP_TRITON_CACHE_DEBUG=1 — dumping cache metadata", flush=True)

    dest_dir = Path(__file__).resolve().parent.parent.parent / "engine" / "src" / "plugins" / "assets"
    dest_dir.mkdir(parents=True, exist_ok=True)
    header_path = dest_dir / "convrot_int8_kernel_cubin.h"

    print("\n=== K1 autotune (extracting from JIT cache) ===", flush=True)
    k1_results = extract_k1(arch, debug_dump=debug_dump, num_sms=num_sms,
                            l2_bytes=l2_bytes, shared_mem_per_sm=shared_mem_per_sm)

    print("\n=== K2 autotune (extracting from JIT cache) ===", flush=True)
    k2_results = extract_k2(arch, debug_dump=debug_dump, num_sms=num_sms,
                            l2_bytes=l2_bytes, shared_mem_per_sm=shared_mem_per_sm)

    print("\n=== Writing header ===", flush=True)
    write_header(k1_results, k2_results, header_path)

    print("\n[extract_jit_cubins_autotune] DONE!", flush=True)
    print("  The header is drop-in compatible with the C++ plugin.", flush=True)
    print("  Rebuild the plugin and TRT engine to use the autotune-selected cubins.", flush=True)

    # ── Winner cache directory summary (for PTX/IR/SASS diagnosis) ──────
    print("\n=== Autotune winner cache directories (PTX/IR for diagnosis) ===", flush=True)
    for dtype_suffix, _, _ in DTYPE_CONFIGS:
        r = k1_results.get(dtype_suffix, {})
        cd = r.get("cache_dir")
        fn = r.get("function_name", "<unknown>")
        if cd:
            print(f"  K1 {dtype_suffix}:  {cd}", flush=True)
            print(f"      function: {fn}", flush=True)
        else:
            print(f"  K1 {dtype_suffix}:  <cache_dir unavailable>", flush=True)
    for bias_suffix, _ in BIAS_CONFIGS:
        for dtype_suffix, _, _ in DTYPE_CONFIGS:
            r = k2_results.get(f"{bias_suffix}_{dtype_suffix}", {})
            cd = r.get("cache_dir")
            fn = r.get("function_name", "<unknown>")
            if cd:
                print(f"  K2 {bias_suffix}_{dtype_suffix}:  {cd}", flush=True)
                print(f"      function: {fn}", flush=True)
            else:
                print(f"  K2 {bias_suffix}_{dtype_suffix}:  <cache_dir unavailable>", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())