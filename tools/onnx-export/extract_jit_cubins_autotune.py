#!/usr/bin/env python3
"""Build and extract autotune-selected ConvRot INT8 cubins.

K1 is selected from Triton's JIT cache.  K2 uses Gluon's explicit-layout
mma_v2/cp.async path and a fixed-budget, weighted production-workload benchmark.
Only the production FP16IO specializations are emitted.
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

# K2 uses Gluon rather than Triton's automatic layout/pipeline selection.  Keep
# this import optional so the CPU interpreter tests can still exercise the
# Triton reference kernel on installations without the experimental dialect;
# production extraction calls _require_gluon_for_k2() and has no fallback.
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon._runtime import GluonASTSource
    from triton.experimental.gluon.language.nvidia.ampere import (
        async_copy as _gluon_async_copy,
        mma_v2 as _gluon_mma_v2,
    )
    GLUON_AVAILABLE = True
    _GLUON_IMPORT_ERROR = None
except Exception as _gluon_import_error:  # pragma: no cover - wheel dependent
    gluon = None
    gl = None
    GluonASTSource = None
    _gluon_async_copy = None
    _gluon_mma_v2 = None
    GLUON_AVAILABLE = False
    _GLUON_IMPORT_ERROR = _gluon_import_error


def _is_gluon_kernel(fn) -> bool:
    """Return whether *fn* is a Gluon JIT function."""
    marker = getattr(fn, "is_gluon", None)
    if callable(marker):
        try:
            return bool(marker())
        except Exception:
            pass
    return "triton.experimental.gluon" in (type(fn).__module__ or "")


def _require_gluon_for_k2():
    if GLUON_AVAILABLE:
        return
    raise SystemExit(
        "[extract_jit_cubins_autotune] K2 requires Triton's Gluon dialect "
        "(triton.experimental.gluon) and an sm80-sm89 CUDA target; import "
        f"failed with {type(_GLUON_IMPORT_ERROR).__name__}: {_GLUON_IMPORT_ERROR}"
    )


# ─── Device-specs-driven autotune config builder ───────────────────────────

import math as _math

def _estimate_k1_configs_sm80_89(num_sms, l2_bytes, shared_mem_per_sm,
                                  autotune_shapes):
    """K1 (v13) candidate family for Ampere/Ada (sm80-sm89).

    Explores BLOCK_M ∈ {4, 8, 16, 32, 64} and num_warps ∈ {4, 8}.
    Higher BLOCK_M processes 16-64 rows per CTA in parallel, yielding full
    vectorized memory throughput and sub-millisecond K1 execution time.
    """
    del num_sms, l2_bytes, shared_mem_per_sm, autotune_shapes

    configs = []
    for bm in (4, 8, 16, 32, 64):
        for nw in (4, 8):
            for mr in (128, None):
                kwargs = {"BLOCK_M": bm}
                if mr is None:
                    configs.append(Config(kwargs, num_warps=nw, num_stages=1))
                else:
                    configs.append(Config(kwargs, num_warps=nw, num_stages=1,
                                          maxnreg=mr))
    return configs


def _estimate_k1_configs_sm75_placeholder(*_args, **_kwargs):
    """Placeholder for the future Turing MMA-fixed wheel/kernel family."""
    raise SystemExit(
        "[K1 dispatch] sm75 kernel family is a deliberate placeholder in this "
        "patch. Add the Turing implementation and exactness vectors before "
        "enabling extraction; no fallback kernel is permitted."
    )


def _estimate_k1_configs_rdna3_placeholder(*_args, **_kwargs):
    """Placeholder for the future gfx11 wave32/HIP kernel family."""
    raise SystemExit(
        "[K1 dispatch] RDNA3/gfx11 kernel family is a deliberate placeholder "
        "in this patch. Add the HIP implementation and exactness vectors before "
        "enabling extraction; no fallback kernel is permitted."
    )


def _estimate_k1_configs(num_sms, l2_bytes, shared_mem_per_sm, autotune_shapes,
                         target_backend="cuda", target_arch=86):
    """Strict target dispatch for K1; unsupported devices never fall back."""
    backend = str(target_backend).lower()
    if backend == "cuda":
        arch = int(target_arch)
        if arch == 75:
            return _estimate_k1_configs_sm75_placeholder(
                num_sms, l2_bytes, shared_mem_per_sm, autotune_shapes)
        if 80 <= arch <= 89:
            return _estimate_k1_configs_sm80_89(
                num_sms, l2_bytes, shared_mem_per_sm, autotune_shapes)
        raise SystemExit(
            f"[K1 dispatch] unsupported CUDA architecture sm{arch}; strict "
            "mode has no fallback kernel"
        )

    if backend == "hip":
        arch = str(target_arch).split(":", 1)[0]
        if arch.startswith("gfx11"):
            return _estimate_k1_configs_rdna3_placeholder(
                num_sms, l2_bytes, shared_mem_per_sm, autotune_shapes)
        raise SystemExit(
            f"[K1 dispatch] unsupported HIP architecture {arch}; strict mode "
            "has no fallback kernel"
        )

    raise SystemExit(
        f"[K1 dispatch] unsupported Triton backend {target_backend!r}; strict "
        "mode has no fallback kernel"
    )


def _estimate_k2_configs(num_sms, l2_bytes, shared_mem_per_sm, autotune_shapes, arch=86):
    """Build the explicit-layout Gluon K2 search space.

    The attached sm86 winner is BM64/BN128/BK64/W4 with an MMA layout of
    ``warps_per_cta=[1, 4]`` and two reusable operand buffers.  Automatic
    layout selection cannot express the copy/MMA/store choices independently,
    so Gluon tunes the useful neighbouring layouts explicitly:

      * asymmetric 64x128 and 128x64 W4 tiles (including the attached layout),
      * 64x64 W4 and 128x128 W8 traffic alternatives,
      * a W2/BM16 family for the real M=1 bias workload,
      * BK32/BK64 plus profiler-driven BK128 double-buffer candidates,
      * double/triple buffering and the measured GROUP_M choices.

    ``WARPS_M`` controls the MMA warp topology; ``NUM_BUFFERS`` controls the
    hand-written cp.async ring.  ``Config.num_stages`` mirrors NUM_BUFFERS only
    for diagnostics/config identity.  Gluon's compiler pipeline itself is
    always invoked with num_stages=1 because the kernel already owns staging.
    """
    del num_sms, l2_bytes, autotune_shapes
    arch = int(arch)
    if not (80 <= arch <= 89):
        raise SystemExit(
            f"[K2 dispatch] Gluon mma_v2/cp.async K2 supports sm80-sm89, got sm{arch}"
        )

    # (BM, BN, warps, WARPS_M candidates, GROUP_M candidates)
    families = (
        (16, 64, 2, (1,), (4,)),
        (16, 128, 2, (1,), (4,)),
        (64, 64, 4, (1, 2, 4), (4, 8)),
        (64, 128, 4, (1, 2), (4, 8)),
        (64, 128, 8, (1, 2, 4), (4, 8)),
        (64, 256, 8, (1, 2, 4), (2, 4, 8)),
        (128, 64, 4, (2, 4), (2, 4, 8)),
        (128, 128, 4, (2, 4), (2, 4)),
        (128, 128, 8, (2, 4), (2, 4, 8)),
        (128, 256, 8, (2, 4), (2, 4, 8)),
    )

    configs = []
    seen = set()
    for bm, bn, num_warps, warp_ms, group_ms in families:
        block_ks = (64,) if bm == 16 else (32, 64, 128)
        for warp_m in warp_ms:
            if num_warps % warp_m:
                continue
            warp_n = num_warps // warp_m
            if bm % (16 * warp_m) or bn % (8 * warp_n):
                continue
            for bk in block_ks:
                if GROUP_SIZE % bk:
                    continue
                # BK128 halves the number of wait/barrier/loop-control steps;
                # its double buffer is already as deep in bytes and compute as
                # the current BK64 triple-buffer winner.
                buffer_candidates = (2,) if bk == 128 else (2, 3)
                for num_buffers in buffer_candidates:
                    # Unlike the old automatic pipeline, the operand rings are
                    # exact.  Layout-conversion scratch depends on WARPS_M and
                    # is checked from compiled metadata by the strict gate.
                    shared_bytes = num_buffers * bk * (bm + bn)
                    if shared_mem_per_sm and shared_bytes > shared_mem_per_sm:
                        continue
                    for group_m in group_ms:
                        for schedule_2d in (False, True):
                            key = (bm, bn, bk, group_m, num_warps,
                                   warp_m, num_buffers, schedule_2d)
                            if key in seen:
                                continue
                            seen.add(key)
                            configs.append(Config({
                                "BLOCK_M": bm,
                                "BLOCK_N": bn,
                                "BLOCK_K": bk,
                                "GROUP_M": group_m,
                                "WARPS_M": warp_m,
                                "NUM_BUFFERS": num_buffers,
                                "SCHEDULE_2D": schedule_2d,
                            }, num_warps=num_warps,
                               num_stages=num_buffers))
    return configs


_K1_CONFIGS: list = []
_K2_CONFIGS: list = []


# ─── H_4 Kronecker butterfly (direct regular Hadamard transform) ──────────
#
# This is the K1 rotation primitive, revised per CONVROT_INT8_DEEP_DIVE_V2.md §3.
#
# The ConvRot regular Hadamard H_{256, regular} is the 4-fold Kronecker power of
# the H4 base block:
#
#       | 1  1  1 -1 |
#  H4 = | 1  1 -1  1 |        (every row/column sums to 2 — "regular")
#       | 1 -1  1  1 |
#       |-1  1  1  1 |
#
#  H_{256, regular} = H4 ⊗ H4 ⊗ H4 ⊗ H4    (normalized by 1/√256 = 1/16)
#
# The previous FWHT+sign-flip decomposition (CONVROT_INT8_PERFORMANCE_REPORT.md
# §2) factored this as P256 · H_sylvester · D256 and required three constant
# tables (D256, perm256, sign256) plus a tl.gather.  The v2 deep-dive showed
# this is unnecessarily indirect: ``_hadamard_butterfly_stage`` — already in
# the codebase, originally written only to synthesize the dense h16 matrix for
# the old tensor-core path — implements one stage of the H4-Kronecker butterfly
# *directly on the data*.  Calling it 4 times (log_4(256) = 4 stages) on a
# (SUBCHUNK, 256) FP32 tile reproduces x @ H_{256, regular}.T exactly, with:
#
#   * **zero** extra constant tables (no D256 / perm256 / sign256)
#   * **zero** tl.gather / tl.permute-as-scatter
#   * identical arithmetic cost (1024 add/sub — same total FLOPs, just a
#     different factorization of the same linear operator)
#   * pure FP32 throughout → the split-FP16 accuracy hack is structurally
#     unnecessary, not just "less needed"
#
# Verified numerically to float64 machine precision (max abs diff < 4e-15 vs
# direct H256r @ x).  See CONVROT_INT8_DEEP_DIVE_V2.md §3.1–§3.3.
#
# ─── Register budget (v2 §3.4, rigorously re-derived) ────────────────────
#
# Triton's BlockedEncoding assigns elements to threads as a bijection: for a
# tile of E elements distributed over T threads/CTA, each thread holds exactly
# E/T elements in registers.  You do NOT get to pick "lanes per row"
# independently of the tile shape — the two are locked together.
#
# For the butterfly primitive, the worst-instant live-set is 2× the resting
# per-thread element count (v0..v3 and h0..h3 are all live simultaneously
# between computing h0 and the final join).  So the safe tile size for one
# butterfly call is:
#
#     SUBCHUNK × 256 / T  ≤  ~120 regs/thread  (leaving headroom for epilogue)
#
# At T = 256 threads/CTA (the standard num_warps=8 config):
#     SUBCHUNK = 16  →  16 resting,  32 peak   ← recommended default (matches
#                                                 the existing kernel's own
#                                                 SUBCHUNK, known-good)
#     SUBCHUNK = 32  →  32 resting,  64 peak   ← safe, fewer loop iterations
#     SUBCHUNK = 64  →  64 resting, 128 peak   ← tight, only if maxnreg allows
#     SUBCHUNK ≥ 128 →  exceeds 255-reg cap    ← IMPOSSIBLE
#
# The K1 kernel therefore processes the BLOCK_M-row tile in NUM_SUB = BLOCK_M /
# SUBCHUNK sequential slices, exactly the idiom the existing tensor-core path
# already used.  This is purely a register-pressure control knob — slicing is
# bit-identical to whole-tile because rows are fully independent under this
# transform (verified, max abs diff = 0.0 in float64).

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
def _convrot_rotate_256_subchunk(x_slice, SUBCHUNK: tl.constexpr, GROUP_SIZE: tl.constexpr):
    """Apply the full 4-stage regular Hadamard rotation to a (SUBCHUNK, GROUP_SIZE) slice.

    log_4(256) = 4 stages.  The 0.5 factor per stage folds the 1/√256 = 1/16
    normalization, so no separate scale multiply is needed.
    """
    rotated = x_slice
    for stage in tl.static_range(0, 4):
        rotated = _hadamard_butterfly_stage(rotated, SUBCHUNK, GROUP_SIZE, stage)
    return rotated


@triton.autotune(configs=_K1_CONFIGS, key=["K", "INPUT_FP16", "CONTIG_XK"])
@triton.jit
def kernel1_convrot_quant(
    X_ptr, X_q_ptr, X_scale_ptr,
    M, K, G,
    stride_xm, stride_xk,
    stride_xqm, stride_xqk,
    stride_xsm,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    INPUT_FP16: tl.constexpr,
    CONTIG_XK: tl.constexpr,
):
    """K1 (v13) — ConvRot H256 in-register rotation + per-row INT8 quant.

    Pipeline (100% in-register math, zero global memory workspace scratch):
      1. For each group g in [0, G):
            Load [BLOCK_M, GROUP_SIZE] FP16 input, convert to FP32, apply 4 H4
            Kronecker butterfly stages (= H256 regular Hadamard), reduce max-abs per row.
      2. Compute per-row FP32 scale (scale = max(row_max / 127.0, 1e-30)).
      3. For each group g in [0, G):
            Re-rotate tile in registers, divide by per-row scale,
            round/clamp to INT8, and store to X_q_ptr.
      4. Write per-row FP32 scale to X_scale_ptr.
    """
    tl.static_assert(GROUP_SIZE == 256, "Only GROUP_SIZE=256 is supported")
    tl.static_assert(BLOCK_M >= 1, "BLOCK_M must be >= 1")
    IS_INTERPRETER: tl.constexpr = _IS_TRITON_INTERPRETER

    num_pid_m = tl.cdiv(M, BLOCK_M)
    start_pid = tl.program_id(0)
    grid_x = tl.num_programs(0)
    rk_group = tl.arange(0, GROUP_SIZE)

    for pid_m_idx in tl.range(start_pid, num_pid_m, grid_x):
        pid_m = pid_m_idx * BLOCK_M
        rm = pid_m + tl.arange(0, BLOCK_M)
        mask_m = rm < M
        mask_2d = mask_m[:, None]

        # Pass 1: Reduce max-abs per row across all K elements (H256 rotated)
        block_max = tl.zeros([BLOCK_M], dtype=tl.float32)
        for g in tl.range(G):
            rk = g * GROUP_SIZE + rk_group
            if CONTIG_XK:
                x_ptr_block = X_ptr + rm[:, None] * stride_xm + rk[None, :]
            else:
                x_ptr_block = X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
            x_block = tl.load(x_ptr_block, mask=mask_2d, other=0.0,
                              eviction_policy="evict_first")
            if INPUT_FP16:
                x_block = x_block.to(tl.float32)

            rotated_x = x_block
            for stage in tl.static_range(0, 4):
                rotated_x = _hadamard_butterfly_stage(
                    rotated_x, BLOCK_M, GROUP_SIZE, stage)

            block_max = tl.maximum(block_max, tl.max(tl.abs(rotated_x), axis=1))

        scale = tl.maximum(block_max / 127.0, 1e-30)

        # Pass 2: Rotate, quantize to INT8, and store to X_q_ptr
        for g in tl.range(G):
            rk = g * GROUP_SIZE + rk_group
            if CONTIG_XK:
                x_ptr_block = X_ptr + rm[:, None] * stride_xm + rk[None, :]
            else:
                x_ptr_block = X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
            x_block = tl.load(x_ptr_block, mask=mask_2d, other=0.0,
                              eviction_policy="evict_first")
            if INPUT_FP16:
                x_block = x_block.to(tl.float32)

            rotated_x = x_block
            for stage in tl.static_range(0, 4):
                rotated_x = _hadamard_butterfly_stage(
                    rotated_x, BLOCK_M, GROUP_SIZE, stage)

            scaled = rotated_x / scale[:, None]
            if IS_INTERPRETER:
                abs_scaled = tl.abs(scaled)
                rounded_abs = tl.floor(abs_scaled + 0.5)
                sign_scaled = tl.where(scaled >= 0.0, 1.0, -1.0)
                x_q = rounded_abs * sign_scaled
            else:
                x_q = libdevice.rint(scaled)
            x_q = tl.clamp(x_q, -127.0, 127.0).to(tl.int8)

            if CONTIG_XK:
                xq_ptr = X_q_ptr + rm[:, None] * stride_xqm + rk[None, :]
            else:
                xq_ptr = X_q_ptr + rm[:, None] * stride_xqm + rk[None, :] * stride_xqk
            tl.store(xq_ptr, x_q, mask=mask_2d)

        # Per-row scale: one FP32 per M row
        tl.store(X_scale_ptr + rm * stride_xsm, scale, mask=mask_m)


@triton.jit
def _kernel2_compute_tile(
    X_q_ptr, X_scale_ptr, W_q_ptr, W_scale_ptr, Bias_ptr, Y_ptr,
    M, N, K,
    stride_xqm, stride_xqk,
    stride_xsm,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    pid_m, pid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,
    DYNAMIC_INNER: tl.constexpr,
):
    """Numerically locked K2 (v13) tile body — plain INT8 GEMM + per-row dequant.

    v13 simplifies the K2 contract: instead of one INT32 fold per 256-wide
    activation-scale group followed by a per-group FP32 accumulation, the
    per-row activation scale (one FP32 per M row, written by K1 v13) lets us
    accumulate a SINGLE full-K INT32 sum and apply the per-row X_scale *
    per-row W_scale outer product in the epilogue.

    Arithmetic contract:
      int32_acc = sum_k X_q[m, k] * W_q[n, k]                 # full-K INT8×INT8
      y[m, n]   = X_scale[m] * W_scale[n] * int32_acc[m, n]   # per-row FP32 dequant
      y[m, n]  += bias[n]                                     # if HAS_BIAS

    This matches ``gemm_convrot_hqq_sym_rowwise_act_rowwise`` in
    ``quantization_sim_real.py`` (the ConvRot_HQQ_g256_per_row GEMM model).
    """
    GROUPS_PER_TILE: tl.constexpr = GROUP_SIZE // BLOCK_K
    pid_m_off = pid_m * BLOCK_M
    pid_n_off = pid_n * BLOCK_N
    rm = pid_m_off + tl.arange(0, BLOCK_M)
    rn = pid_n_off + tl.arange(0, BLOCK_N)
    mask_m = rm < M

    # Single full-K INT32 accumulator.  The per-group FP32 fold of v7-v12 is
    # gone — X_scale is now per-row, so the entire K dimension shares one
    # activation scale and one weight scale per output row.
    int32_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_group_start in tl.range(0, K, GROUP_SIZE):
        if DYNAMIC_INNER:
            for k_offset in tl.range(
                    k_group_start, k_group_start + GROUP_SIZE, BLOCK_K):
                cols_k = k_offset + tl.arange(0, BLOCK_K)
                xq = tl.load(
                    X_q_ptr + rm[:, None] * stride_xqm + cols_k[None, :] * stride_xqk)
                wq = tl.load(
                    W_q_ptr + rn[:, None] * stride_wn + cols_k[None, :] * stride_wk)
                int32_acc += tl.dot(xq, tl.trans(wq), out_dtype=tl.int32)
        else:
            for sub in tl.static_range(0, GROUPS_PER_TILE):
                k_offset = k_group_start + sub * BLOCK_K
                cols_k = k_offset + tl.arange(0, BLOCK_K)
                xq = tl.load(
                    X_q_ptr + rm[:, None] * stride_xqm + cols_k[None, :] * stride_xqk)
                # Canonical W_q layout remains [N,K].  Keeping this exact load
                # and transpose path preserves the v10 numerical reference.
                wq = tl.load(
                    W_q_ptr + rn[:, None] * stride_wn + cols_k[None, :] * stride_wk)
                int32_acc += tl.dot(xq, tl.trans(wq), out_dtype=tl.int32)

    # Per-row X_scale (one FP32 per M row) — replaces the per-group xs load.
    xs = tl.load(X_scale_ptr + rm * stride_xsm, mask=mask_m, other=0.0,
                 eviction_policy="evict_last")
    # Per-row W_scale (one FP32 per N row) — unchanged from v7-v12.
    ws = tl.load(W_scale_ptr + rn, eviction_policy="evict_last")

    acc = int32_acc.to(tl.float32) * xs[:, None] * ws[None, :]
    if HAS_BIAS:
        bias = tl.load(Bias_ptr + rn, eviction_policy="evict_last")
        acc += bias[None, :]

    y_ptr_block = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    if OUTPUT_FP16:
        tl.store(y_ptr_block, acc.to(tl.float16), mask=mask_m[:, None])
    else:
        tl.store(y_ptr_block, acc.to(tl.float32), mask=mask_m[:, None])


@triton.jit
def kernel2_gemm_dequant_reference(
    X_q_ptr, X_scale_ptr, W_q_ptr, W_scale_ptr, Bias_ptr, Y_ptr,
    M, N, K,
    stride_xqm, stride_xqk,
    stride_xsm,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,
    SCHEDULE_2D: tl.constexpr = False,
    DYNAMIC_INNER: tl.constexpr = False,
):
    """K2 (v13) — plain INT8×INT8 → INT32 GEMM + per-row FP32 outer-product dequant.

    v13 simplification: the per-group INT32 fold + per-group FP32 accumulation
    of v7-v12 is replaced by a single full-K INT32 sum followed by an
    outer-product FP32 dequant with per-row X_scale (one FP32 per M row,
    produced by K1 v13) and per-row W_scale (one FP32 per N row, unchanged).

    This matches the ``ConvRot_HQQ_g256_per_row`` simulation GEMM
    (``gemm_convrot_hqq_sym_rowwise_act_rowwise``)::

        Y[m, n] = X_s_row[m] * W_scale_row[n] * dot_i32(X_q[m, :], W_q[n, :]) + bias[n]

    Bias handling remains a compile-time specialization in both this oracle and
    shipping Gluon, so NOBIAS skips the bias load/add.

    sm86 optimization notes (Nsight Compute profiled, carried over from v10-v12):

      1. ``X_scale`` and ``W_scale`` are epilogue-only, dead throughout the K
         loop.  Both are tagged ``eviction_policy="evict_last"`` because they
         are reused across tiles via the L2 super-group swizzle.

      2. ``eviction_policy`` is DROPPED on xq/wq loads (Nsight showed L1 hit
         rate ~3% — the hints add LSU overhead for zero benefit on streaming
         INT8 operands that never fit L1).

      3. ``DYNAMIC_INNER=True`` preserves the ordered four-BK INT32 sum but
         reuses one software-pipelined shared-memory buffer.

      4. The original 1-D GROUP_M persistent scheduler remains the numerical
         reference.  ``SCHEDULE_2D=True`` uses a bounded 2-D persistent grid
         and removes repeated dynamic tile division/modulo while calling the
         identical tile body.
    """
    tl.static_assert(GROUP_SIZE == 256, "Only GROUP_SIZE=256 is supported")
    tl.static_assert(GROUP_SIZE % BLOCK_K == 0,
                     "GROUP_SIZE must be a multiple of BLOCK_K")

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    if SCHEDULE_2D:
        # Division-free persistent scheduler.  The host launches a bounded 2-D
        # grid.  Each CTA owns a deterministic strided set of M/N tiles, so the
        # per-output arithmetic is exactly the same as the baseline scheduler.
        start_pid_m = tl.program_id(0)
        start_pid_n = tl.program_id(1)
        grid_m = tl.num_programs(0)
        grid_n = tl.num_programs(1)
        for pid_m in tl.range(start_pid_m, num_pid_m, grid_m):
            for pid_n in tl.range(start_pid_n, num_pid_n, grid_n):
                _kernel2_compute_tile(
                    X_q_ptr, X_scale_ptr, W_q_ptr, W_scale_ptr, Bias_ptr, Y_ptr,
                    M, N, K,
                    stride_xqm, stride_xqk, stride_xsm,
                    stride_wn, stride_wk, stride_ym, stride_yn,
                    pid_m, pid_n,
                    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE,
                    HAS_BIAS, OUTPUT_FP16, DYNAMIC_INNER,
                )
    else:
        # Existing 1-D GROUP_M super-group scheduler retained as the reference
        # and as an autotuned candidate.
        num_tiles = num_pid_m * num_pid_n
        num_pid_in_group = GROUP_M * num_pid_n
        start_pid = tl.program_id(0)
        grid_x = tl.num_programs(0)
        for tile_id in tl.range(start_pid, num_tiles, grid_x):
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
            _kernel2_compute_tile(
                X_q_ptr, X_scale_ptr, W_q_ptr, W_scale_ptr, Bias_ptr, Y_ptr,
                M, N, K,
                stride_xqm, stride_xqk, stride_xsm,
                stride_wn, stride_wk, stride_ym, stride_yn,
                pid_m, pid_n,
                BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE,
                HAS_BIAS, OUTPUT_FP16, DYNAMIC_INNER,
            )

# ─── K2 shipping kernel: explicit-layout Gluon IMMA + cp.async ─────────────
#
# The Triton kernel above is intentionally retained only as the immutable v10
# numerical oracle.  It is never benchmarked as a shipping candidate and its
# cubin is never emitted.  Gluon makes the three layouts and the cp.async ring
# explicit, avoiding the four separately-unrolled shared allocations visible in
# the attached TTGIR while retaining the exact G256 INT32/FP32 fold boundaries.
if GLUON_AVAILABLE:
    _k2_cp = _gluon_async_copy

    @gluon.jit
    def _kernel2_gluon_mma_step(
        step, int32_acc, a_bufs, w_bufs,
        Xq32, Wq32, rm_cp, rn_cp, rk_cp, a_row_mask, sxm32, swn32,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr,
        copy_layout: gl.constexpr, smem8: gl.constexpr,
        a_layout: gl.constexpr, b_layout: gl.constexpr,
        ISSUE_PREFETCH: gl.constexpr, WAIT_GROUP: gl.constexpr,
    ):
        """Consume one cp.async stage and optionally refill the freed stage.

        Issuing global-to-shared async prefetch immediately after wait_group/barrier
        dispatches memory requests to the L2/DRAM interface first, overlapping memory
        latency with shared-memory loads and Tensor Core MMA calculations.
        """
        BK32: gl.constexpr = BLOCK_K // 4
        _k2_cp.wait_group(WAIT_GROUP)
        gl.barrier()

        if ISSUE_PREFETCH:
            prefetch_step = step + NUM_BUFFERS - 1
            write_buf = prefetch_step % NUM_BUFFERS
            off32 = prefetch_step * BK32
            a_ptrs = Xq32 + rm_cp[:, None] * sxm32 + (off32 + rk_cp)[None, :]
            w_ptrs = Wq32 + rn_cp[:, None] * swn32 + (off32 + rk_cp)[None, :]
            _k2_cp.async_copy_global_to_shared(
                a_bufs.index(write_buf), a_ptrs, mask=a_row_mask[:, None],
                cache_modifier=".cg")
            _k2_cp.async_copy_global_to_shared(
                w_bufs.index(write_buf), w_ptrs, cache_modifier=".cg")
            _k2_cp.commit_group()

        read_buf = step % NUM_BUFFERS
        a8 = a_bufs.index(read_buf)._reinterpret(
            gl.int8, [BLOCK_M, BLOCK_K], smem8)
        w8 = w_bufs.index(read_buf)._reinterpret(
            gl.int8, [BLOCK_N, BLOCK_K], smem8)
        a = a8.load(a_layout)
        b = w8.permute((1, 0)).load(b_layout)

        return _gluon_mma_v2(a, b, int32_acc)


    @gluon.jit(do_not_specialize_on_alignment=("M",))
    def kernel2_gemm_dequant(
        X_q_ptr, X_scale_ptr, W_q_ptr, W_scale_ptr, Bias_ptr, Y_ptr,
        M, N, K,
        stride_xqm, stride_xqk,
        stride_xsm,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr, GROUP_SIZE: gl.constexpr,
        GROUP_M: gl.constexpr, WARPS_M: gl.constexpr,
        NUM_BUFFERS: gl.constexpr, HAS_BIAS: gl.constexpr,
        OUTPUT_FP16: gl.constexpr,
        SCHEDULE_2D: gl.constexpr = False,
    ):
        """Persistent INT8 GEMM (v13) with per-row X_scale + per-row W_scale.

        v13 simplification: the per-group INT32 fold + per-group FP32
        accumulation of v7-v12 is replaced by a single full-K INT32 sum.  The
        per-row X_scale (one FP32 per M row, written by K1 v13) and per-row
        W_scale (one FP32 per N row, unchanged) are applied as an outer-product
        FP32 dequant in the epilogue::

            y[m, n] = X_scale[m] * W_scale[n] * int32_acc[m, n] + bias[n]

        The cp.async ring and MMA warp topology are unchanged from v11-v12;
        only the per-group ``xs`` load + FP32 fold is removed, and a single
        per-row ``xs`` load is added to the epilogue alongside ``ws``.
        """
        gl.static_assert(GROUP_SIZE == 256, "only GROUP_SIZE=256 is supported")
        gl.static_assert(GROUP_SIZE % BLOCK_K == 0,
                         "GROUP_SIZE must be divisible by BLOCK_K")
        gl.static_assert(NUM_BUFFERS >= 2 and NUM_BUFFERS <= 3,
                         "K2 tunes double and triple buffering")
        gl.static_assert(OUTPUT_FP16, "FP32IO is legacy and is not shipped")
        gl.static_assert(gl.num_warps() % WARPS_M == 0,
                         "WARPS_M must divide num_warps")

        BK32: gl.constexpr = BLOCK_K // 4
        WARPS_N: gl.constexpr = gl.num_warps() // WARPS_M

        mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[2, 0], warps_per_cta=[WARPS_M, WARPS_N],
            instr_shape=[16, 8])
        a_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=mma_layout, k_width=4)
        b_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=mma_layout, k_width=4)

        COPY_THREADS_K: gl.constexpr = BLOCK_K // 16
        COPY_THREADS_M: gl.constexpr = 32 // COPY_THREADS_K
        copy_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[COPY_THREADS_M, COPY_THREADS_K],
            warps_per_cta=[gl.num_warps(), 1], order=[1, 0])

        # Match warps_per_cta in store_layout to mma_layout's [WARPS_M, WARPS_N]
        # to eliminate cross-warp shared-memory layout-conversion scratch (~32KB)
        # and allow 2 CTAs/SM occupancy on sm86/sm89.
        warp_tile_m: gl.constexpr = BLOCK_M // WARPS_M
        warp_tile_n: gl.constexpr = BLOCK_N // WARPS_N

        if warp_tile_n >= 64:
            store_threads_m: gl.constexpr = 4
            store_threads_n: gl.constexpr = 8
        elif warp_tile_n >= 32:
            store_threads_m: gl.constexpr = 8
            store_threads_n: gl.constexpr = 4
        else:
            store_threads_m: gl.constexpr = 16
            store_threads_n: gl.constexpr = 2

        vec_m: gl.constexpr = warp_tile_m // store_threads_m
        vec_n: gl.constexpr = warp_tile_n // store_threads_n

        store_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[vec_m, vec_n],
            threads_per_warp=[store_threads_m, store_threads_n],
            warps_per_cta=[WARPS_M, WARPS_N],
            order=[1, 0],
        )

        smem32: gl.constexpr = gl.SwizzledSharedLayout(
            vec=4, per_phase=2, max_phase=4, order=[1, 0])
        smem8: gl.constexpr = gl.SwizzledSharedLayout(
            vec=16, per_phase=2, max_phase=4, order=[1, 0])
        a_bufs = gl.allocate_shared_memory(
            gl.int32, [NUM_BUFFERS, BLOCK_M, BK32], smem32)
        w_bufs = gl.allocate_shared_memory(
            gl.int32, [NUM_BUFFERS, BLOCK_N, BK32], smem32)

        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_tiles = num_pid_m * num_pid_n
        num_pid_in_group = GROUP_M * num_pid_n
        start_pid = gl.program_id(0)
        grid_x = gl.num_programs(0)

        rm_cp_base = gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, copy_layout))
        rn_cp_base = gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(1, copy_layout))
        rk_cp = gl.arange(0, BK32, layout=gl.SliceLayout(0, copy_layout))
        rm_acc_base = gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, mma_layout))
        rn_acc_base = gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, mma_layout))
        rm_store_base = gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, store_layout))
        rn_store_base = gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, store_layout))

        Xq32 = X_q_ptr.cast(gl.pointer_type(gl.int32))
        Wq32 = W_q_ptr.cast(gl.pointer_type(gl.int32))
        sxm32 = stride_xqm // 4
        swn32 = stride_wn // 4
        total_steps = K // BLOCK_K
        steady_steps = total_steps - (NUM_BUFFERS - 1)

        if SCHEDULE_2D:
            start_pid_m = gl.program_id(0)
            start_pid_n = gl.program_id(1)
            grid_m = gl.num_programs(0)
            grid_n = gl.num_programs(1)
            for pid_m in range(start_pid_m, num_pid_m, grid_m):
                for pid_n in range(start_pid_n, num_pid_n, grid_n):
                    pid_m_off = pid_m * BLOCK_M
                    pid_n_off = pid_n * BLOCK_N

                    rm_cp = pid_m_off + rm_cp_base
                    rn_cp = pid_n_off + rn_cp_base
                    rm_acc = pid_m_off + rm_acc_base
                    rn_acc = pid_n_off + rn_acc_base
                    a_row_mask = rm_cp < M
                    acc_row_mask = rm_acc < M

                    for stage in gl.static_range(0, NUM_BUFFERS - 1):
                        off32 = stage * BK32
                        a_ptrs = Xq32 + rm_cp[:, None] * sxm32 + (off32 + rk_cp)[None, :]
                        w_ptrs = Wq32 + rn_cp[:, None] * swn32 + (off32 + rk_cp)[None, :]
                        _k2_cp.async_copy_global_to_shared(
                            a_bufs.index(stage), a_ptrs, mask=a_row_mask[:, None],
                            cache_modifier=".cg")
                        _k2_cp.async_copy_global_to_shared(w_bufs.index(stage), w_ptrs,
                                          cache_modifier=".cg")
                        _k2_cp.commit_group()

                    int32_acc = gl.zeros(
                        (BLOCK_M, BLOCK_N), dtype=gl.int32, layout=mma_layout)

                    for step in range(0, steady_steps):
                        int32_acc = _kernel2_gluon_mma_step(
                            step, int32_acc, a_bufs, w_bufs,
                            Xq32, Wq32, rm_cp, rn_cp, rk_cp, a_row_mask, sxm32, swn32,
                            BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS,
                            copy_layout, smem8, a_layout, b_layout,
                            ISSUE_PREFETCH=True, WAIT_GROUP=NUM_BUFFERS - 2)

                    for drain in gl.static_range(0, NUM_BUFFERS - 1):
                        step = steady_steps + drain
                        int32_acc = _kernel2_gluon_mma_step(
                            step, int32_acc, a_bufs, w_bufs,
                            Xq32, Wq32, rm_cp, rn_cp, rk_cp, a_row_mask, sxm32, swn32,
                            BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS,
                            copy_layout, smem8, a_layout, b_layout,
                            ISSUE_PREFETCH=False,
                            WAIT_GROUP=NUM_BUFFERS - 2 - drain)

                    xs = gl.load(X_scale_ptr + rm_acc * stride_xsm,
                                 mask=acc_row_mask, other=0.0,
                                 eviction_policy="evict_last")
                    ws = gl.load(W_scale_ptr + rn_acc, eviction_policy="evict_last")
                    acc = int32_acc.to(gl.float32) * xs[:, None] * ws[None, :]
                    if HAS_BIAS:
                        bias = gl.load(Bias_ptr + rn_acc, eviction_policy="evict_last")
                        acc = acc + bias[None, :]

                    acc_store = gl.convert_layout(acc, store_layout)
                    rm_store = pid_m_off + rm_store_base
                    rn_store = pid_n_off + rn_store_base
                    y_ptrs = (Y_ptr + rm_store[:, None] * stride_ym +
                              rn_store[None, :] * stride_yn)
                    gl.store(y_ptrs, acc_store.to(gl.float16),
                             mask=(rm_store < M)[:, None])
        else:
            for tile_id in range(start_pid, num_tiles, grid_x):
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
                tile_in_group = tile_id % num_pid_in_group
                pid_m = first_pid_m + (tile_in_group % group_size_m)
                pid_n = tile_in_group // group_size_m
                pid_m_off = pid_m * BLOCK_M
                pid_n_off = pid_n * BLOCK_N

                rm_cp = pid_m_off + rm_cp_base
                rn_cp = pid_n_off + rn_cp_base
                rm_acc = pid_m_off + rm_acc_base
                rn_acc = pid_n_off + rn_acc_base
                a_row_mask = rm_cp < M
                acc_row_mask = rm_acc < M

                for stage in gl.static_range(0, NUM_BUFFERS - 1):
                    off32 = stage * BK32
                    a_ptrs = Xq32 + rm_cp[:, None] * sxm32 + (off32 + rk_cp)[None, :]
                    w_ptrs = Wq32 + rn_cp[:, None] * swn32 + (off32 + rk_cp)[None, :]
                    _k2_cp.async_copy_global_to_shared(
                        a_bufs.index(stage), a_ptrs, mask=a_row_mask[:, None],
                        cache_modifier=".cg")
                    _k2_cp.async_copy_global_to_shared(w_bufs.index(stage), w_ptrs,
                                      cache_modifier=".cg")
                    _k2_cp.commit_group()

                int32_acc = gl.zeros(
                    (BLOCK_M, BLOCK_N), dtype=gl.int32, layout=mma_layout)

                for step in range(0, steady_steps):
                    int32_acc = _kernel2_gluon_mma_step(
                        step, int32_acc, a_bufs, w_bufs,
                        Xq32, Wq32, rm_cp, rn_cp, rk_cp, a_row_mask, sxm32, swn32,
                        BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS,
                        copy_layout, smem8, a_layout, b_layout,
                        ISSUE_PREFETCH=True, WAIT_GROUP=NUM_BUFFERS - 2)

                for drain in gl.static_range(0, NUM_BUFFERS - 1):
                    step = steady_steps + drain
                    int32_acc = _kernel2_gluon_mma_step(
                        step, int32_acc, a_bufs, w_bufs,
                        Xq32, Wq32, rm_cp, rn_cp, rk_cp, a_row_mask, sxm32, swn32,
                        BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS,
                        copy_layout, smem8, a_layout, b_layout,
                        ISSUE_PREFETCH=False,
                        WAIT_GROUP=NUM_BUFFERS - 2 - drain)

                xs = gl.load(X_scale_ptr + rm_acc * stride_xsm,
                             mask=acc_row_mask, other=0.0,
                             eviction_policy="evict_last")
                ws = gl.load(W_scale_ptr + rn_acc, eviction_policy="evict_last")
                acc = int32_acc.to(gl.float32) * xs[:, None] * ws[None, :]
                if HAS_BIAS:
                    bias = gl.load(Bias_ptr + rn_acc, eviction_policy="evict_last")
                    acc = acc + bias[None, :]

                acc_store = gl.convert_layout(acc, store_layout)
                rm_store = pid_m_off + rm_store_base
                rn_store = pid_n_off + rn_store_base
                y_ptrs = (Y_ptr + rm_store[:, None] * stride_ym +
                          rn_store[None, :] * stride_yn)
                gl.store(y_ptrs, acc_store.to(gl.float16),
                         mask=(rm_store < M)[:, None])
else:
    kernel2_gemm_dequant = None


GROUP_SIZE = 256
DEFAULT_BLOCK_K = 64
TILE_NAME = "PersistentG256PerRow"
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
#       instead of shipping masked K2 variants.  Version 8 restores separate
#       compile-time BIAS and NOBIAS K2 cubins.
#   9 — numerically locked sm80-sm89 K1 family: BM16/32/64 W4 candidates,
#       exact-specialization zero-spill gate (STACK + PTX + final SASS +
#       scratch), per-config K1 occupancy, generic runtime-M extraction,
#       bitwise BM16/W4 equivalence/canary gate, and redundant modulo removal.
#  10 — numerically locked K2 tile helper plus an optional division-free 2-D
#       persistent scheduler.  Selection uses a weighted multi-shape benchmark
#       of the exact plugin prefix/tail/copy sequence and rejects any candidate
#       that differs bitwise from the v9 BM128/BN64/BK64 reference.
#  11 — K2 is an explicit-layout Gluon mma_v2 kernel with a correctly drained
#       cp.async ring, coalesced FP16 stores, and tunable MMA warp topology.
#       FP32IO is retired: extraction emits only FP16IO K1/K2 cubins.
#  12 — K2 predicates its final M tile so the plugin uses one persistent launch
#       and no longer allocates, launches, or copies a padded output tail.
#  13 — Per-row activation scaling + in-register H256 Hadamard rotation (ConvRot_HQQ_
#       g256_per_row simulation strategy). K1 fuses in-register H256 rotation +
#       per-row INT8 quant in one kernel without cross-group butterfly or workspace
#       memory scratch; X_scale is [M] FP32 (one scale per activation row, NOT
#       per 256-block). K2 is simplified to a single full-K INT8×INT8 → INT32 matmul
#       plus an outer-product FP32 dequant (xs[m] * ws[n] * acc), removing the per-
#       K-group INT32 fold. stride_xsg and workspace pointers are removed.
#       Backward compatibility with v12 cubins is NOT preserved.
CONVROT_INT8_CUBIN_HEADER_VERSION = 13
# K2 ships separate compile-time bias and no-bias specializations.
BIAS_CONFIGS = [("NOBIAS", False), ("BIAS", True)]
# FP32IO is a legacy graph boundary.  Do not compile, benchmark, emit, or make
# the plugin require cubins that production never selects.
DTYPE_CONFIGS = [("FP16IO", True, True)]
# Real DiT decoder K2 shapes, expressed as (M, K, N).  M=3000 intentionally
# exercises the masked final tile used by the version-12 single-launch path.
#
# The first entry is the dominant wide-N projection.  The weighted custom
# benchmark below evaluates every listed production family; config construction
# is intentionally independent of one synthetic primary shape.
AUTOTUNE_SHAPES_K2 = [
    (3000, 2560, 9728),  # dit.mlp.gate/up projections
    (3000, 9728, 2560),  # dit.mlp.down projection
    (3000, 2560, 4096),  # dit.self_attn.q projection
    (3000, 2560, 1024),  # dit.self_attn.k/v projections
    (3000, 4096, 2560),  # dit.self_attn.o projection
]
# Actual ConvRot inventory from the TensorRT inspector JSON has 32 DiT layers:
#   NOBIAS: gate/up=64, down=32, self+cross Q=64, self+cross O=64,
#           self K/V=64, cross K/V=64.
#   BIAS:   six M=1 time-embedding projections plus one condition embedder.
# The benchmark models total runtime as sum(call_count * measured_latency).
# Measured latency already contains M*K*N, so multiplying call counts by FLOPs
# again would incorrectly square the compute weighting.
K2_DEFAULT_NOBIAS_WORKLOAD = (
    (3000, 2560, 9728, 64.0, "mlp.gate/up"),
    (3000, 9728, 2560, 32.0, "mlp.down"),
    (3000, 2560, 4096, 64.0, "self/cross q"),
    (3000, 2560, 1024, 64.0, "self k/v"),
    (2048, 2560, 1024, 64.0, "cross k/v"),
    (3000, 4096, 2560, 64.0, "self/cross o"),
)
K2_DEFAULT_BIAS_WORKLOAD = (
    (1, 256, 2560, 2.0, "time_embed.linear_1"),
    (1, 2560, 2560, 2.0, "time_embed.linear_2"),
    (1, 2560, 15360, 2.0, "time_embed.time_proj"),
    (2048, 2048, 2560, 1.0, "condition_embedder"),
)
K2_AUTOTUNE_M_ALIGNMENT = 128  # max generated K2 BLOCK_M; avoids OOB in plain K2
# K1 uses the same workspace pitch contract as K2 but the real, unpadded M so
# the shipped cubin makes no false divisibility assumption about runtime M.
K1_AUTOTUNE_M = 3000
K1_AUTOTUNE_M_ALIGNMENT = K2_AUTOTUNE_M_ALIGNMENT

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
# ``AUTOTUNE_OCCUPANCY_ASSUMPTION`` is only a defensive fallback when a
# config is absent from the exact compile/resource map (for example, a unit
# test that injects a manual config).  Production K1 and K2 tuning both use
# per-config occupancy populated by the strict spill gate.
#
# A config that fits only at 1 CTA/SM still benchmarks correctly at any
# grid size: the kernel's persistent grid-stride loop iterates more tiles
# per CTA, so the total work is unchanged; only the per-config SM
# utilization (and therefore the wall-clock time) changes with grid size.
AUTOTUNE_OCCUPANCY_ASSUMPTION = 2

# Per-config occupancy maps, populated by the exact-specialization compile /
# spill filter.  Key is ``(sorted(kwargs.items()), warps, stages, maxnreg)``.
# K1 used to assume 2 CTA/SM even though the sm86 winner runs at 4 CTA/SM;
# both stages now benchmark with the same occupancy-scaled grid as production.
_K1_CFG_OCCUPANCY: dict = {}
_K2_CFG_OCCUPANCY: dict = {}
# K2 additionally keeps bias-specific occupancy so NOBIAS benchmarking is not
# under-launched merely because the BIAS epilogue needs more registers.
_K2_CFG_OCCUPANCY_BY_BIAS: dict = {}
# A layout may be legal for NOBIAS and spill only after adding the BIAS
# epilogue (or vice versa).  Keep those candidate sets independent so the hot
# NOBIAS workload does not lose a valid fast cubin merely because BIAS selects a
# different winner.
_K2_CFG_VALID_BY_BIAS: dict = {}


def _cfg_occupancy_key(cfg):
    """Identity tuple matching ``_filter_spilling_configs``'s reject map."""
    return (tuple(sorted(cfg.kwargs.items())), cfg.num_warps, cfg.num_stages,
            getattr(cfg, "maxnreg", None))


def _max_shared_bytes_per_block(arch: int) -> int:
    if arch == 75:
        return 48 * 1024
    if arch == 80 or arch == 87:
        return 163 * 1024
    if 86 <= arch <= 89:
        return 99 * 1024
    return 48 * 1024


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
        smem_per_sm, max_ctas_per_sm, max_warps_per_sm = 64 * 1024, 16, 32
    elif arch == 80:
        # Ampere HPC (A100)
        smem_per_sm, max_ctas_per_sm, max_warps_per_sm = 164 * 1024, 32, 64
    elif arch == 86:
        # Ampere consumer (RTX 30, A40, A10)
        smem_per_sm, max_ctas_per_sm, max_warps_per_sm = 100 * 1024, 16, 48
    elif arch == 87:
        # Ampere embedded (Jetson Orin): cc8.7 has the 164-KB SMEM class.
        smem_per_sm, max_ctas_per_sm, max_warps_per_sm = 164 * 1024, 16, 48
    elif arch == 89:
        # Ada Lovelace (RTX 40)
        smem_per_sm, max_ctas_per_sm, max_warps_per_sm = 100 * 1024, 24, 48
    elif arch == 90:
        # Hopper (H100)
        smem_per_sm, max_ctas_per_sm, max_warps_per_sm = 228 * 1024, 32, 64
    else:
        # Strict K1 dispatch should make this unreachable; conservative sm86.
        smem_per_sm, max_ctas_per_sm, max_warps_per_sm = 100 * 1024, 16, 48
    regs_per_sm = 65536
    reg_alloc_unit = 256   # 32-bit registers per warp allocation granularity

    # Shared-mem limiter.  Guard against 0 to avoid ZeroDivisionError; a
    # zero-shared kernel is rare but legal (K1 was almost there before v7).
    shared_bytes = max(int(shared_bytes), 1)
    ctas_shared = smem_per_sm // shared_bytes

    # Register allocation is rounded per warp to a 256-register unit on these
    # architectures.  Include the resident-warp limiter as well; the previous
    # approximation rounded per thread and omitted it.
    if num_regs and num_warps:
        regs_per_warp_raw = int(num_regs) * 32
        regs_per_warp = ((regs_per_warp_raw + reg_alloc_unit - 1) //
                         reg_alloc_unit) * reg_alloc_unit
        block_regs = regs_per_warp * int(num_warps)
        ctas_regs = regs_per_sm // max(block_regs, 1)
        ctas_warps = max_warps_per_sm // int(num_warps)
    else:
        ctas_regs = max_ctas_per_sm
        ctas_warps = max_ctas_per_sm

    ctas = min(ctas_shared, ctas_regs, ctas_warps, max_ctas_per_sm)
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
    defaults to a defensive assumption only when a manually injected config
    has no resource entry.  Normal K1 and K2 tuning pass exact-config
    occupancy from the strict compile/spill gate.
    Caps at ``num_tiles`` so we never launch more CTAs than there is work.
    """
    cap = num_sms * max(int(ctas_per_sm), 1)
    return min(num_tiles, cap) if num_tiles > cap else num_tiles


def _autotune_grid_x_for_stage(num_tiles: int, num_sms: int, meta,
                                occupancy_map: dict) -> int:
    """Look up exact-config occupancy for a Triton autotune grid lambda."""
    kwargs_items = tuple(sorted(
        (k, v) for k, v in meta.items()
        if k not in ("num_warps", "num_stages", "num_ctas", "maxnreg")
    ))
    key = (kwargs_items,
           int(meta.get("num_warps", 0)),
           int(meta.get("num_stages", 0)),
           meta.get("maxnreg", None))
    ctas = occupancy_map.get(key, AUTOTUNE_OCCUPANCY_ASSUMPTION)
    return _autotune_grid_x(num_tiles, num_sms, ctas)


def _autotune_grid_x_for_k1(num_tiles: int, num_sms: int, meta) -> int:
    return _autotune_grid_x_for_stage(num_tiles, num_sms, meta,
                                      _K1_CFG_OCCUPANCY)


def _autotune_grid_x_for_k2(num_tiles: int, num_sms: int, meta) -> int:
    return _autotune_grid_x_for_stage(num_tiles, num_sms, meta,
                                      _K2_CFG_OCCUPANCY)


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
    "G": "kG",
    "stride_xm": "kStride_xm",
    "stride_xk": "kStride_xk",
    "stride_xqm": "kStride_xqm",
    "stride_xqk": "kStride_xqk",
    "stride_xsm": "kStride_xsm",
    "stride_wn": "kStride_wn",
    "stride_wk": "kStride_wk",
    "stride_ym": "kStride_ym",
    "stride_yn": "kStride_yn",
}

_POINTER_ARG_NAMES = {
    "X_ptr", "X_q_ptr", "X_scale_ptr", "W_q_ptr", "W_scale_ptr", "Bias_ptr", "Y_ptr",
}

# K1 (v13) launches with: activations, outputs, runtime dims (M, K, G), and strides.
_QUANT_ARG_NAMES = {
    "X_ptr", "X_q_ptr", "X_scale_ptr",
    "M", "K", "G",
    "stride_xm", "stride_xk",
    "stride_xqm", "stride_xqk",
    "stride_xsm",
}

# K2 (v13) launches with the per-row X_scale ([M] FP32, stride_xsm only — no
# stride_xsg), per-row W_scale, optional bias, and the canonical [N, K] W_q
# layout.  The GEMM is a single full-K INT8×INT8 → INT32 matmul; the per-row
# outer-product FP32 dequant lives in the epilogue.
_GEMM_ARG_NAMES = {
    "X_q_ptr", "X_scale_ptr", "W_q_ptr", "W_scale_ptr", "Bias_ptr", "Y_ptr",
    "M", "N", "K",
    "stride_xqm", "stride_xqk", "stride_xsm",
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
    # ASTSource keeps specialized integer arguments in ``signature`` but records
    # their true compile-time status separately in ``constants``.  They are not
    # PTX parameters and must not appear in the generated C++ launch ABI.
    constant_indices = {
        int(key[0]) for key in (getattr(src, "constants", {}) or {})
        if isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], int)
    }

    def append_item(key, name_hint=None):
        if key in signature and key not in consumed:
            name = name_hint or str(key)
            try:
                index = arg_names.index(name)
            except ValueError:
                index = key if isinstance(key, int) else None
            consumed.add(key)
            if index in constant_indices:
                return
            ordered.append({
                "name": name,
                "sig_type": _normalize_sig_type(signature[key]),
            })

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

    The weighted K2 workload benchmark bypasses ``@triton.autotune``'s built-in
    measurement, so ``autotuned_fn.best_config`` is not populated.  Instead we
    hand the picked config in explicitly and reuse the cache-walking helpers.
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
_K2_ACTIVE_HAS_BIAS = True


def _init_k1_worker(input_fp16):
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    dtype = torch.float16 if input_fp16 else torch.float32
    _K1_WORKER_STATE.input_fp16 = input_fp16
    _K1_WORKER_STATE.x = torch.empty((1, 256), device="cuda", dtype=dtype)
    _K1_WORKER_STATE.xq = torch.empty((1, 256), device="cuda", dtype=torch.int8)
    _K1_WORKER_STATE.xs = torch.empty((1,), device="cuda", dtype=torch.float32)


def _worker_precompile_k1(config_tuple):
    if not hasattr(_K1_WORKER_STATE, "x"):
        return 0
    from triton import Config
    kwargs, nw, ns, mr = config_tuple
    input_fp16 = _K1_WORKER_STATE.input_fp16
    try:
        cfg = Config(kwargs, num_warps=nw, num_stages=ns, maxnreg=mr)
        K_warmup = 256
        G_warmup = K_warmup // GROUP_SIZE  # = 1
        kernel1_convrot_quant.fn.warmup(
            _K1_WORKER_STATE.x, _K1_WORKER_STATE.xq, _K1_WORKER_STATE.xs,
            1, K_warmup, G_warmup,
            K_warmup, 1, K_warmup, 1, 1,
            GROUP_SIZE=256,
            INPUT_FP16=input_fp16,
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
    _K2_WORKER_STATE.output_fp16 = output_fp16
    _K2_WORKER_STATE.target_arch = target_arch


def _worker_precompile_k2(config_tuple):
    from triton import Config
    kwargs, nw, ns, mr = config_tuple
    try:
        if mr is None:
            cfg = Config(kwargs, num_warps=nw, num_stages=ns)
        else:
            cfg = Config(kwargs, num_warps=nw, num_stages=ns, maxnreg=mr)
        _compile_gluon_k2(
            cfg, _K2_ACTIVE_HAS_BIAS, _K2_WORKER_STATE.output_fp16,
            _K2_WORKER_STATE.target_arch)
    except Exception:
        # The authoritative spill/compile gate reports exact failures.  This
        # pass is only a best-effort disk-cache warmer.
        pass
    return 1


def _parallel_precompile_k2(configs, output_fp16, target_arch):
    _require_gluon_for_k2()
    num_workers = int(os.environ.get(
        "HOTSTEP_AUTOTUNE_WORKERS", min(mp.cpu_count(), len(configs), 16)))
    if num_workers <= 1 or not configs:
        return
    print(f"    [parallel-compile] Precompiling {len(configs)} Gluon K2 configs "
          f"across {num_workers} threads...", flush=True)
    tuples = [(c.kwargs, c.num_warps, c.num_stages,
               getattr(c, "maxnreg", None)) for c in configs]
    done = 0
    total = len(configs)
    progress_step = max(total // 10, 1)
    try:
        with ThreadPool(
            processes=num_workers,
            initializer=_init_k2_worker,
            initargs=(output_fp16, target_arch),
        ) as pool:
            for _ in pool.imap_unordered(
                    _worker_precompile_k2, tuples, chunksize=1):
                done += 1
                if done % progress_step == 0 or done == total:
                    print(f"    [parallel-compile] K2: {done}/{total} configs compiled",
                          flush=True)
    except Exception as exc:
        print(f"    [parallel-compile] Warning: Gluon precompilation fallback "
              f"({exc})", flush=True)


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


def _count_cubin_local_sass_ops(cubin_bytes, function_name: str) -> int | None:
    """Count final-SASS local-memory loads/stores (LDL/STL/LLD/LST).

    PTX can be spill-free and ptxas can still introduce spills during final
    register allocation.  Conversely, a stack-size parser failure must not
    silently turn into an acceptance.  Inspecting the final machine code is an
    independent third signal.  Returns None when cuobjdump cannot disassemble.
    """
    import subprocess
    import tempfile

    cuobjdump = _find_cuobjdump()
    if cuobjdump is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin_bytes)
        cubin_path = f.name
    try:
        proc = subprocess.run(
            [cuobjdump, "--dump-sass", cubin_path],
            capture_output=True, text=True, check=True,
        )
        text = proc.stdout
        # Restrict to the requested function when cuobjdump emitted several.
        marker = f"Function : {function_name}"
        if marker in text:
            text = text.split(marker, 1)[1]
            next_fn = text.find("Function : ")
            if next_fn >= 0:
                text = text[:next_fn]
        # Modern SASS uses LDL/STL; older generations may print LLD/LST.
        return len(re.findall(r"\b(?:LDL|STL|LLD|LST)(?:\.[A-Z0-9_.]+)?\b", text))
    except Exception:
        return None
    finally:
        Path(cubin_path).unlink(missing_ok=True)


def _detect_spills(kernel, function_name: str = "") -> dict:
    """Strict, redundant spill check on the final compiled kernel.

    A candidate is rejected if *any* available signal reports thread-local or
    scratch traffic:

      1. cubin EIATTR_FRAME_SIZE / cuobjdump STACK is nonzero;
      2. PTX contains ld.local or st.local;
      3. final SASS contains LDL/STL (or legacy LLD/LST);
      4. Triton metadata requests nonzero global scratch.

    The checks intentionally overlap.  The generic PTX check catches explicit
    local arrays, while STACK and SASS catch spills introduced later by ptxas.
    This kernel declares no local arrays, so zero is the only accepted value.
    Failure to run either final-binary check is itself a rejection; the wheel
    must ship a working, version-matched cuobjdump.
    """
    cubin = kernel.asm.get("cubin", b"")
    cubin_size = len(cubin) if isinstance(cubin, (bytes, bytearray)) else 0

    ptx = kernel.asm.get("ptx", "")
    if isinstance(ptx, bytes):
        ptx = ptx.decode(errors="replace")
    ptx_spills = ptx.count("ld.local") + ptx.count("st.local")

    spills_cuobjdump, local_bytes, registers = (None, 0, 0)
    sass_local_ops = None
    if function_name and cubin_size > 0:
        spills_cuobjdump, local_bytes, registers = _check_cubin_spills_cuobjdump(
            cubin, function_name)
        sass_local_ops = _count_cubin_local_sass_ops(cubin, function_name)

    global_scratch = int(
        getattr(getattr(kernel, "metadata", None), "global_scratch_size", 0) or 0
    )

    available = []
    if spills_cuobjdump is not None:
        available.append("cuobjdump-stack")
    if sass_local_ops is not None:
        available.append("sass-local")
    available.append("ptx-local")
    available.append("triton-scratch")

    incomplete_final_check = (
        bool(function_name) and cubin_size > 0 and
        (spills_cuobjdump is None or sass_local_ops is None)
    )
    spilling = (
        incomplete_final_check
        or bool(spills_cuobjdump)
        or ptx_spills > 0
        or (sass_local_ops is not None and sass_local_ops > 0)
        or global_scratch > 0
    )

    if incomplete_final_check:
        print(
            f"    [spill-check] REJECT: incomplete final-binary inspection for "
            f"{function_name}: stack={spills_cuobjdump}, "
            f"sass_local_ops={sass_local_ops}. Strict mode requires both "
            f"cuobjdump resource and SASS checks.",
            flush=True,
        )

    return {
        "spilling": spilling,
        "ptx_spills": ptx_spills,
        "sass_local_ops": -1 if sass_local_ops is None else sass_local_ops,
        "global_scratch": global_scratch,
        "local_bytes": local_bytes,
        "registers": registers,
        "method": "+".join(available),
    }


def _compile_kernel_for_spill_check(jit_fn, signature, constexprs, attrs,
                                     num_warps, num_stages, maxnreg,
                                     target_arch):
    """Compile one exact shipping specialization for spill analysis.

    Gluon already contains an explicit cp.async pipeline, so compiler
    ``num_stages`` must remain one; the Config's stage field is only the
    explicit NUM_BUFFERS identity/diagnostic value.
    """
    from triton.compiler import compile as triton_compile
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget

    if _is_gluon_kernel(jit_fn):
        _require_gluon_for_k2()
        src = GluonASTSource(fn=jit_fn, signature=signature,
                             constexprs=constexprs, attrs=attrs or {})
        compiler_stages = 1
    else:
        # ASTSource requires the JITFunction wrapper (arg_names/cache_key), not
        # its raw Python ``.fn``.  Autotuner wrappers are unwrapped only until
        # that JIT holder is reached.
        fn = _find_jit_holder(jit_fn)
        if fn is None:
            raise TypeError(f"could not locate Triton JITFunction for {jit_fn!r}")
        src = ASTSource(fn=fn, signature=signature, constexprs=constexprs,
                        attrs=attrs or {})
        compiler_stages = num_stages
    target = GPUTarget(backend='cuda', arch=target_arch, warp_size=32)
    options = {'num_warps': num_warps, 'num_stages': compiler_stages}
    if maxnreg is not None:
        options['maxnreg'] = maxnreg
    return triton_compile(src, target=target, options=options)


def _compile_gluon_k2(cfg, has_bias, output_fp16, target_arch):
    """Compile the exact AOT Gluon specialization used for launch/emission."""
    _require_gluon_for_k2()
    wanted = f"{'BIAS' if has_bias else 'NOBIAS'}_FP16IO"
    for label, signature, extra, attrs in _spill_check_variants("k2"):
        if label != wanted:
            continue
        if not output_fp16:
            raise RuntimeError("FP32IO K2 compilation is retired")
        constexprs = dict(extra)
        constexprs.update(cfg.kwargs)
        return _compile_kernel_for_spill_check(
            kernel2_gemm_dequant, signature, constexprs, attrs,
            cfg.num_warps, cfg.num_stages,
            getattr(cfg, "maxnreg", None), target_arch)
    raise RuntimeError(f"missing Gluon K2 spill/ABI variant {wanted}")


# ─── Parallel spill-check worker state ────────────────────────────────────
_SPILL_CHECK_STATE = threading.local()


def _init_spill_check_worker(jit_fn, target_arch, stage):
    _SPILL_CHECK_STATE.jit_fn = jit_fn
    _SPILL_CHECK_STATE.target_arch = target_arch
    _SPILL_CHECK_STATE.fn_name = "kernel1_convrot_quant" if stage == "k1" else "kernel2_gemm_dequant"


def _worker_check_spills(task):
    """Compile one exact (config, variant) shipping specialization."""
    cfg, variant_label, signature, constexprs, attrs = task
    try:
        kernel = _compile_kernel_for_spill_check(
            _SPILL_CHECK_STATE.jit_fn,
            signature,
            constexprs,
            attrs,
            cfg.num_warps, cfg.num_stages,
            getattr(cfg, 'maxnreg', None),
            _SPILL_CHECK_STATE.target_arch)
        spill_info = _detect_spills(kernel, _SPILL_CHECK_STATE.fn_name)
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

    Yields (label, signature, extra_constexprs, attrs) tuples.
    """
    if stage == "k1":
        for dtype_suffix, input_fp16, _ in DTYPE_CONFIGS:
            x_type = '*fp16' if input_fp16 else '*fp32'
            signature = {
                'X_ptr': x_type, 'X_q_ptr': '*i8', 'X_scale_ptr': '*fp32',
                'M': 'i32', 'K': 'i32', 'G': 'i32',
                'stride_xm': 'i32', 'stride_xk': 'i32',
                'stride_xqm': 'i32', 'stride_xqk': 'i32',
                'stride_xsm': 'i32',
            }
            constexprs = {
                'GROUP_SIZE': 256,
                'INPUT_FP16': input_fp16,
                'CONTIG_XK': True,
                'stride_xk': 1,
                'stride_xqk': 1,
                'stride_xsm': 1,
            }
            # Argument indices:
            # 0..2 : pointers (16-div)
            # 3    : M (generic)
            # 4    : K (16-div)
            # 5    : G (generic)
            # 6    : stride_xm (16-div)
            # 7, 9, 10: unit strides (constexpr 1)
            # 8    : stride_xqm (16-div)
            attrs = {
                (i,): [["tt.divisibility", 16]]
                for i in (0, 1, 2, 4, 6, 8)
            }
            yield (dtype_suffix, signature, constexprs, attrs)
    else:  # k2
        # Spill-check both shipped compile-time bias variants.  Register
        # pressure can differ because BIAS retains an extra epilogue vector.
        for bias_suffix, _has_bias in BIAS_CONFIGS:
            for dtype_suffix, _, output_fp16 in DTYPE_CONFIGS:
                y_type = '*fp16' if output_fp16 else '*fp32'
                # v13 K2 signature: per-row X_scale (no stride_xsg — X_scale
                # is [M] FP32, indexed only by stride_xsm).
                signature = {
                    'X_q_ptr': '*i8', 'X_scale_ptr': '*fp32', 'W_q_ptr': '*i8',
                    'W_scale_ptr': '*fp32', 'Bias_ptr': '*fp32', 'Y_ptr': y_type,
                    'M': 'i32', 'N': 'i32', 'K': 'i32',
                    'stride_xqm': 'i32', 'stride_xqk': 'i32',
                    'stride_xsm': 'i32',
                    'stride_wn': 'i32', 'stride_wk': 'i32',
                    'stride_ym': 'i32', 'stride_yn': 'i32',
                }
                constexprs = {
                    'GROUP_SIZE': 256,
                    'HAS_BIAS': _has_bias,
                    'OUTPUT_FP16': output_fp16,
                    'stride_xqk': 1,
                    'stride_xsm': 1,
                    'stride_wk': 1,
                    'stride_yn': 1,
                }
                # M is the real, tightly packed row count (e.g. 3000) and
                # must remain generic.  Pointers, N/K and the K/N row pitches
                # retain their 16-byte divisibility contracts.
                # Argument indices (0-based):
                #   0..5  : pointers (16-div)
                #   6     : M (generic)
                #   7, 8  : N, K (16-div)
                #   9     : stride_xqm = K (16-div)
                #   10, 11: unit strides (constexpr 1)
                #   12    : stride_wn = K (16-div)
                #   13    : stride_wk (constexpr 1)
                #   14    : stride_ym = N (16-div)
                #   15    : stride_yn (constexpr 1)
                attrs = {
                    (i,): [["tt.divisibility", 16]]
                    for i in (0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 14)
                }
                yield (f"{bias_suffix}_{dtype_suffix}", signature,
                       constexprs, attrs)


def _filter_spilling_configs(configs, jit_fn, stage, target_arch):
    """Compile each config in parallel and reject any that spill registers.

    ZERO TOLERANCE: any config with STACK > 0 (from cuobjdump) or any
    PTX-level ld.local/st.local is rejected.  No thresholds.

    Every shipped constexpr variant is checked.  K1 legality is global; K2
    legality is tracked independently for BIAS and NOBIAS so one epilogue's
    register pressure cannot suppress a valid candidate for the other.

    Uses the same 16-thread ThreadPool as the precompile step — sequential
    compilation of 50+ configs takes minutes; parallel takes seconds.

    Returns the filtered config list.
    """
    if not configs:
        return configs

    def make_constexprs(cfg, extra):
        out = dict(extra)
        out.update(cfg.kwargs)  # BLOCK_M[/BLOCK_N/BLOCK_K/GROUP_M]
        return out

    # Build (config, label, signature, constexprs, attrs) for every variant.
    tasks = [
        (cfg, label, signature, make_constexprs(cfg, extra), attrs)
        for cfg in configs
        for (label, signature, extra, attrs) in _spill_check_variants(stage)
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

    # K1 has one shipping dtype and therefore rejects a config globally.  K2
    # records legality per BIAS/NOBIAS specialization: an epilogue-only spill in
    # BIAS must not remove an otherwise valid NOBIAS candidate from the hot path.
    def _cfg_id(c):
        return (tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages,
                getattr(c, 'maxnreg', None))

    reject_reasons: dict = {}
    variant_reject_reasons: dict = {}

    def _record_reject(cid, variant_label, reason):
        tagged = f"[{variant_label}] {reason}"
        reject_reasons.setdefault(cid, []).append(tagged)
        if stage == "k2":
            has_bias_variant = variant_label.startswith("BIAS_")
            variant_reject_reasons.setdefault(
                (cid, has_bias_variant), []).append(tagged)

    for cfg, variant_label, spill_info, error in results:
        cid = _cfg_id(cfg)
        if error is not None:
            _record_reject(cid, variant_label, f"compile failed: {error}")
        elif spill_info['spilling']:
            _record_reject(
                cid, variant_label,
                f"SPILLS: {spill_info['method']}: "
                f"local_bytes={spill_info['local_bytes']}, "
                f"ptx_spills={spill_info['ptx_spills']}, "
                f"sass_local_ops={spill_info.get('sass_local_ops', -1)}, "
                f"global_scratch={spill_info.get('global_scratch', -1)}, "
                f"regs={spill_info['registers']}")
        elif (stage == "k2" and spill_info.get("shared_bytes", 0) >
              _max_shared_bytes_per_block(target_arch)):
            _record_reject(
                cid, variant_label,
                f"shared_bytes={spill_info.get('shared_bytes', 0)} > "
                f"architectural block limit={_max_shared_bytes_per_block(target_arch)}")
        elif (stage == "k2" and K2_MAX_SHARED_BYTES > 0 and
              spill_info.get("shared_bytes", 0) > K2_MAX_SHARED_BYTES):
            _record_reject(
                cid, variant_label,
                f"shared_bytes={spill_info.get('shared_bytes', 0)} > "
                f"K2_MAX_SHARED_BYTES={K2_MAX_SHARED_BYTES}")

    # ── Per-config occupancy from each exact shipping specialization ──────
    occ_map: dict = {}
    variant_occ_map: dict = {}
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
        if stage == "k2":
            has_bias_variant = variant_label.startswith("BIAS_")
            vkey = (cid, has_bias_variant)
            vprev = variant_occ_map.get(vkey)
            variant_occ_map[vkey] = cur if vprev is None else min(vprev, cur)

    filtered = []
    rejected = []
    variant_disabled = []
    if stage == "k2":
        _K2_CFG_VALID_BY_BIAS.clear()
        for cfg in configs:
            cid = _cfg_id(cfg)
            legal_any = False
            for has_bias in (False, True):
                reasons = variant_reject_reasons.get((cid, has_bias))
                valid = not reasons
                _K2_CFG_VALID_BY_BIAS[(_cfg_occupancy_key(cfg), has_bias)] = valid
                legal_any = legal_any or valid
                if reasons:
                    variant_disabled.append(
                        (cfg, has_bias, "; ".join(reasons)))
            if legal_any:
                filtered.append(cfg)
            else:
                rejected.append((cfg, "; ".join(reject_reasons.get(
                    cid, ["no legal shipping specialization"]))))
    else:
        for cfg in configs:
            reasons = reject_reasons.get(_cfg_id(cfg))
            if reasons:
                rejected.append((cfg, "; ".join(reasons)))
            else:
                filtered.append(cfg)

    occupancy_dst = _K1_CFG_OCCUPANCY if stage == "k1" else _K2_CFG_OCCUPANCY
    occupancy_dst.clear()
    if stage == "k2":
        _K2_CFG_OCCUPANCY_BY_BIAS.clear()
        for cfg in filtered:
            cid = _cfg_id(cfg)
            valid_occs = []
            for has_bias in (False, True):
                key = (_cfg_occupancy_key(cfg), has_bias)
                if not _K2_CFG_VALID_BY_BIAS.get(key, False):
                    continue
                occ = variant_occ_map.get(
                    (cid, has_bias), AUTOTUNE_OCCUPANCY_ASSUMPTION)
                _K2_CFG_OCCUPANCY_BY_BIAS[key] = int(occ)
                valid_occs.append(int(occ))
            occupancy_dst[_cfg_occupancy_key(cfg)] = (
                min(valid_occs) if valid_occs else AUTOTUNE_OCCUPANCY_ASSUMPTION)
    else:
        for cfg in filtered:
            occ = occ_map.get(_cfg_id(cfg), AUTOTUNE_OCCUPANCY_ASSUMPTION)
            occupancy_dst[_cfg_occupancy_key(cfg)] = int(occ)

    if rejected:
        print(f"    [spill-check] {stage}: globally rejected {len(rejected)} "
              f"config(s):", flush=True)
        for cfg, reason in rejected[:10]:
            mr = getattr(cfg, 'maxnreg', None)
            print(f"      REJECT {cfg.kwargs} W={cfg.num_warps} "
                  f"S={cfg.num_stages} MR={mr}: {reason}", flush=True)
        if len(rejected) > 10:
            print(f"      ... and {len(rejected) - 10} more", flush=True)
    if stage == "k2" and variant_disabled:
        print(f"    [spill-check] k2: disabled {len(variant_disabled)} "
              f"individual BIAS/NOBIAS specialization(s); other variants remain",
              flush=True)
        for cfg, has_bias, reason in variant_disabled[:10]:
            print(f"      DISABLE {'BIAS' if has_bias else 'NOBIAS'} "
                  f"{cfg.kwargs}: {reason}", flush=True)
    if not rejected and not variant_disabled:
        print(f"    [spill-check] {stage}: all {len(configs)} configs are "
              f"spill-free in all shipped variants", flush=True)

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
# from Triton's built-in.  K2 v12 retains that fixed event methodology and
# benchmarks the exact one-launch masked-M plugin path over every production
# shape, rather than one artificial M=3072 gate-projection launch.
#
# Environment variables (all optional):
#   HOTSTEP_K2_BENCH_WARMUP   — warmup iterations per config (default 20)
#   HOTSTEP_K2_BENCH_ITERS    — timed iterations per config (default 50)
#   HOTSTEP_K2_BENCH_TOPN     — how many top configs to print (default 5)
#   HOTSTEP_K2_LAYER_JSON     — optional TensorRT inspector JSON for exact counts
#   HOTSTEP_K2_PROFILE        — min/opt/max profile rows from that JSON (default max)

K2_BENCH_WARMUP = int(os.environ.get("HOTSTEP_K2_BENCH_WARMUP", "20"))
K2_BENCH_ITERS = int(os.environ.get("HOTSTEP_K2_BENCH_ITERS", "50"))
K2_BENCH_TOPN = int(os.environ.get("HOTSTEP_K2_BENCH_TOPN", "5"))


def _inspector_profile_rows(doc, tensor_name, profile_kind):
    shape_key = {"min": "MinShape", "opt": "OptShape", "max": "MaxShape"}[profile_kind]
    layers = doc.get("layers", {})
    for tensor in layers.get("I/O Tensors", []):
        if tensor.get("Name") != tensor_name:
            continue
        infos = tensor.get("ProfileInfo", [])
        if not infos:
            break
        shape = infos[0].get(shape_key, [])
        if len(shape) < 2:
            break
        rows = 1
        for dim in shape[:-1]:
            rows *= int(dim)
        return rows
    raise ValueError(f"could not read {shape_key} rows for {tensor_name!r}")


def _k2_workload_from_inspector(path, has_bias):
    """Derive exact K2 call counts and M classes from a TRT inspector JSON."""
    doc = json.loads(Path(path).read_text())
    profile_kind = os.environ.get("HOTSTEP_K2_PROFILE", "max").strip().lower()
    if profile_kind not in ("min", "opt", "max"):
        raise ValueError("HOTSTEP_K2_PROFILE must be min, opt, or max")
    latent_m = _inspector_profile_rows(doc, "input_latents", profile_kind)
    encoder_m = _inspector_profile_rows(doc, "enc_hidden", profile_kind)

    layer_list = doc.get("layers", {}).get("Layers", [])
    producer = {
        out.get("Name"): layer
        for layer in layer_list for out in layer.get("Outputs", [])
        if out.get("Name")
    }
    metadata_re = re.compile(r"gs=\d+,K=(\d+),N=(\d+),bias=(\d+)")
    counts = {}
    labels = {}
    for layer in layer_list:
        if layer.get("PluginType") != "ConvRotInt8Linear":
            continue
        match = metadata_re.search(layer.get("PluginMetadata", ""))
        if match is None:
            continue
        K, N, bias_flag = map(int, match.groups())
        if bool(bias_flag) != bool(has_bias):
            continue
        inputs = layer.get("Inputs", [])
        weight_name = ""
        if len(inputs) >= 2:
            weight_name = producer.get(inputs[1].get("Name"), {}).get("Name", "")
        if "time_embed" in weight_name:
            M = 1
        elif ("cross_attn.k_proj" in weight_name or
              "cross_attn.v_proj" in weight_name or
              "condition_embedder" in weight_name):
            M = encoder_m
        else:
            M = latent_m
        key = (M, K, N)
        counts[key] = counts.get(key, 0.0) + 1.0
        labels.setdefault(key, weight_name or layer.get("Name", "unnamed"))
    if not counts:
        raise ValueError(f"no ConvRot bias={int(bool(has_bias))} layers found in {path}")
    return tuple((M, K, N, count, labels[(M, K, N)])
                 for (M, K, N), count in sorted(counts.items()))


def _k2_workload(has_bias):
    inspector_path = os.environ.get("HOTSTEP_K2_LAYER_JSON", "").strip()
    if inspector_path:
        workload = _k2_workload_from_inspector(inspector_path, has_bias)
        print(f"    [K2 workload] loaded {len(workload)} shape classes from "
              f"{inspector_path} (profile={os.environ.get('HOTSTEP_K2_PROFILE', 'max')})",
              flush=True)
        return workload
    return K2_DEFAULT_BIAS_WORKLOAD if has_bias else K2_DEFAULT_NOBIAS_WORKLOAD


def _launch_k2_for_bench(cfg, tensors, M, N, K, num_sms, output_fp16,
                         reference=False):
    """Launch one K2 config with the occupancy-scaled production grid.

    v13: X_scale is per-row ([M] FP32), so there is no stride_xsg parameter.
    Shipping candidates call the raw Gluon JIT function with compiler staging
    disabled (the kernel has its own cp.async ring).  ``reference=True`` calls
    the retained Triton oracle and is used only for the bitwise gate.
    """
    xq, xs, wq, ws, bias, y = tensors
    num_pid_m = triton.cdiv(M, cfg.kwargs["BLOCK_M"])
    num_pid_n = triton.cdiv(N, cfg.kwargs["BLOCK_N"])
    num_tiles = num_pid_m * num_pid_n
    # Per-config occupancy mirrors the production launch geometry.
    cfg_occ_key = _cfg_occupancy_key(cfg)
    ctas = _K2_CFG_OCCUPANCY_BY_BIAS.get(
        (cfg_occ_key, bool(_K2_ACTIVE_HAS_BIAS)),
        _K2_CFG_OCCUPANCY.get(cfg_occ_key, AUTOTUNE_OCCUPANCY_ASSUMPTION))
    grid_cap = max(int(num_sms) * max(int(ctas), 1), 1)
    if cfg.kwargs.get("SCHEDULE_2D", False):
        grid_m = min(num_pid_m, grid_cap)
        grid_n = min(num_pid_n, max(grid_cap // max(grid_m, 1), 1))
        grid = (grid_m, grid_n)
        launched_ctas = grid_m * grid_n
    else:
        grid_x = _autotune_grid_x(num_tiles, num_sms, ctas)
        grid = (grid_x,)
        launched_ctas = grid_x
    kwargs = dict(cfg.kwargs)
    kwargs["GROUP_SIZE"] = GROUP_SIZE
    kwargs["HAS_BIAS"] = _K2_ACTIVE_HAS_BIAS
    kwargs["OUTPUT_FP16"] = output_fp16
    kwargs["num_warps"] = cfg.num_warps
    # Explicit Gluon cp.async staging must not be pipelined a second time by
    # compiler options.  The reference retains its historical Config stages.
    kwargs["num_stages"] = cfg.num_stages if reference else 1
    mr = getattr(cfg, "maxnreg", None)
    if mr is not None:
        kwargs["maxnreg"] = mr
    launch_fn = (kernel2_gemm_dequant_reference if reference
                 else kernel2_gemm_dequant)
    if launch_fn is None:
        _require_gluon_for_k2()
    # v13: no stride_xsg (X_scale is [M] FP32, indexed by stride_xsm=1 only).
    launch_fn[grid](
        xq, xs, wq, ws, bias, y,
        M, N, K,
        K, 1,
        1,
        K, 1,        # W_q [N, K] row-major
        N, 1,
        **kwargs,
    )
    return launched_ctas


def _make_k2_production_bench_pack(configs, M, N, K, has_bias, output_fp16,
                                   seed):
    """Allocate deterministic tensors for exact plugin-sequence benchmarking.

    v13: X_scale is per-row ([M] FP32) instead of per-group ([G, M] FP32).
    The pitch-keyed dict is retained because the v10 oracle still needs a
    padded M for bitwise comparison, but each entry is now 1-D ([pitch]).
    """
    import torch

    max_bm = max(c.kwargs["BLOCK_M"] for c in configs)
    if max_bm > K2_AUTOTUNE_M_ALIGNMENT:
        raise RuntimeError(
            f"K2 candidate BLOCK_M={max_bm} exceeds plugin workspace contract "
            f"{K2_AUTOTUNE_M_ALIGNMENT}")
    max_m_padded = _math.ceil(M / max_bm) * max_bm
    gen = torch.Generator(device="cuda")
    gen.manual_seed(int(seed))

    xq = torch.randint(-127, 128, (max_m_padded, K), generator=gen,
                       device="cuda", dtype=torch.int8)
    if max_m_padded > M:
        xq[M:].zero_()
    # v13: per-row X_scale (one FP32 per M row).  Production K1 writes [M]
    # FP32; the oracle's padded-M comparison path pads with zeros.
    xs_valid = torch.rand((M,), generator=gen, device="cuda",
                          dtype=torch.float32) * 0.02 + 0.001
    pitches = sorted(
        {M} | {_math.ceil(M / c.kwargs["BLOCK_M"]) * c.kwargs["BLOCK_M"]
               for c in configs})
    xs_by_pitch = {}
    for pitch in pitches:
        xs = torch.zeros((pitch,), device="cuda", dtype=torch.float32)
        xs[:M].copy_(xs_valid)
        xs_by_pitch[pitch] = xs

    wq = torch.randint(-127, 128, (N, K), generator=gen,
                       device="cuda", dtype=torch.int8)
    ws = torch.rand((N,), generator=gen, device="cuda",
                    dtype=torch.float32) * 0.02 + 0.001
    bias = (torch.randn((N,), generator=gen, device="cuda", dtype=torch.float32)
            if has_bias else torch.empty((1,), device="cuda", dtype=torch.float32))
    out_dtype = torch.float16 if output_fp16 else torch.float32
    y = torch.empty((M, N), device="cuda", dtype=out_dtype)
    y_tail = torch.empty((max_bm, N), device="cuda", dtype=out_dtype)
    return {
        "xq": xq, "xs_by_pitch": xs_by_pitch, "wq": wq, "ws": ws,
        "bias": bias, "y": y, "y_tail": y_tail,
    }


def _launch_k2_production_sequence(cfg, pack, M, N, K, num_sms,
                                   output_fp16, reference=False):
    """Mirror the shipping launch; retain split-tail only for the v10 oracle.

    v13: X_scale is per-row ([M] FP32); the pitch lookup still works because
    _make_k2_production_bench_pack stores 1-D tensors keyed by pitch.
    """
    if not reference:
        # Version 12+ masks the final M tile in-kernel.  Public prequantized
        # tensors and private K1 tensors are both tightly packed, and K2 writes
        # the real output directly in one persistent launch.
        xs = pack["xs_by_pitch"][M]
        return _launch_k2_for_bench(
            cfg,
            (pack["xq"], xs, pack["wq"], pack["ws"], pack["bias"], pack["y"]),
            M, N, K, num_sms, output_fp16,
            reference=False)

    # The immutable unmasked Triton oracle still needs the old split/copy path;
    # only its resulting bits are compared, never its timing.
    bm = cfg.kwargs["BLOCK_M"]
    m_aligned = M - (M % bm)
    m_padded = _math.ceil(M / bm) * bm
    xs = pack["xs_by_pitch"][m_padded]
    total_ctas = 0

    if m_aligned > 0:
        total_ctas += _launch_k2_for_bench(
            cfg,
            (pack["xq"], xs, pack["wq"], pack["ws"], pack["bias"], pack["y"]),
            m_aligned, N, K, num_sms, output_fp16,
            reference=reference)
    if m_aligned < M:
        tail_rows = M - m_aligned
        xq_tail = pack["xq"][m_aligned:]
        xs_tail = xs[m_aligned:]
        y_tail = pack["y_tail"][:bm]
        total_ctas += _launch_k2_for_bench(
            cfg,
            (xq_tail, xs_tail, pack["wq"], pack["ws"], pack["bias"], y_tail),
            bm, N, K, num_sms, output_fp16,
            reference=reference)
        # Contiguous copy_ lowers to the same D2D copy class used by enqueue.
        pack["y"][m_aligned:M].copy_(y_tail[:tail_rows])
    return total_ctas


def _bench_k2_config_production(cfg, pack, M, N, K, num_sms, output_fp16,
                                warmup=None, iters=None):
    """Time the exact single-launch masked-M production path."""
    import torch
    warmup = K2_BENCH_WARMUP if warmup is None else warmup
    iters = K2_BENCH_ITERS if iters is None else iters

    grid_ctas = _launch_k2_production_sequence(
        cfg, pack, M, N, K, num_sms, output_fp16)
    torch.cuda.synchronize()
    for _ in range(warmup):
        _launch_k2_production_sequence(cfg, pack, M, N, K, num_sms, output_fp16)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _launch_k2_production_sequence(cfg, pack, M, N, K, num_sms, output_fp16)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, grid_ctas


def _k2_reference_config():
    """The exact K2 baseline that shipped before this optimization patch."""
    return Config({
        "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64,
        "GROUP_M": 2, "SCHEDULE_2D": False, "DYNAMIC_INNER": False,
    }, num_warps=8, num_stages=3)


def _tensor_bitwise_equal(a, b):
    import torch
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    if a.dtype == torch.float16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a, b)


def _bench_k2_workload(configs, num_sms, has_bias, output_fp16):
    """Select one K2 cubin on the weighted, exact plugin workload.

    Each candidate must first match the v10 oracle bitwise on every production
    shape.  Timing covers the version-12 single persistent launch at real M.
    """
    import torch

    if not configs:
        return None, []
    reference = _k2_reference_config()
    workload = _k2_workload(has_bias)
    state = {
        id(cfg): {"cfg": cfg, "weighted_ms": 0.0, "weight": 0.0,
                  "shape_ms": [], "error": None}
        for cfg in configs
    }

    for shape_idx, (M, K, N, weight, shape_label) in enumerate(workload):
        legal = [c for c in configs
                 if N % c.kwargs["BLOCK_N"] == 0 and K % GROUP_SIZE == 0]
        if not legal:
            continue
        pack = _make_k2_production_bench_pack(
            legal + [reference], M, N, K, has_bias, output_fp16,
            seed=0x4B320000 + shape_idx)

        # Current shipped arithmetic/scheduler is the immutable numerical
        # reference, but it is timed only if present as a normal candidate.
        _launch_k2_production_sequence(
            reference, pack, M, N, K, num_sms, output_fp16,
            reference=True)
        torch.cuda.synchronize()
        if not torch.isfinite(pack["y"]).all().item():
            raise SystemExit(
                f"[K2 exactness] v9 reference produced NaN/Inf for {M}x{K}x{N}")
        y_reference = pack["y"].clone()

        shape_rows = []
        for cfg in legal:
            entry = state[id(cfg)]
            if entry["error"] is not None:
                continue
            try:
                _launch_k2_production_sequence(
                    cfg, pack, M, N, K, num_sms, output_fp16)
                torch.cuda.synchronize()
                if not torch.isfinite(pack["y"]).all().item():
                    raise RuntimeError("candidate produced NaN/Inf")
                if not _tensor_bitwise_equal(pack["y"], y_reference):
                    if pack["y"].dtype == torch.float32:
                        mismatch = int((pack["y"].view(torch.int32) !=
                                        y_reference.view(torch.int32)).sum().item())
                    else:
                        mismatch = int((pack["y"].view(torch.int16) !=
                                        y_reference.view(torch.int16)).sum().item())
                    raise RuntimeError(f"bitwise output mismatch ({mismatch} values)")

                ms, grid_ctas = _bench_k2_config_production(
                    cfg, pack, M, N, K, num_sms, output_fp16)
                entry["weighted_ms"] += float(weight) * ms
                entry["weight"] += float(weight)
                entry["shape_ms"].append((shape_idx, ms, grid_ctas))
                shape_rows.append((cfg, ms, grid_ctas, None))
            except Exception as exc:
                entry["error"] = f"shape {M}x{K}x{N}: {type(exc).__name__}: {exc}"
                shape_rows.append((cfg, float("inf"), 0, entry["error"]))

        shape_rows.sort(key=lambda row: row[1])
        print(f"    [K2 workload] shape {shape_idx + 1}/{len(workload)} "
              f"calls={weight:g} label={shape_label}; exact plugin sequence",
              flush=True)
        _print_k2_bench_topn(shape_rows, M, K, N)
        del y_reference, pack

    results = []
    expected_weight = float(sum(row[3] for row in workload))
    for entry in state.values():
        cfg = entry["cfg"]
        err = entry["error"]
        if err is not None or entry["weight"] != expected_weight:
            results.append((cfg, float("inf"), 0,
                            err or "not legal/benchmarked for every shape"))
        else:
            # Keep the modeled total, not an average.  This is directly
            # interpretable as aggregate K2 milliseconds per engine invocation.
            results.append((cfg, entry["weighted_ms"], -1, None))
    results.sort(key=lambda row: row[1])
    best = results[0][0] if results and results[0][1] != float("inf") else None
    return best, results


def _print_k2_workload_topn(results, topn=None):
    if topn is None:
        topn = K2_BENCH_TOPN
    print(f"    [K2 workload] top-{topn} by modeled total K2 ms per engine invocation:",
          flush=True)
    for cfg, ms, _, err in results[:topn]:
        if ms == float("inf"):
            print(f"      FAIL  {cfg}: {err}", flush=True)
            continue
        s2d_str = " S2D=1" if cfg.kwargs.get("SCHEDULE_2D", False) else ""
        print(f"      total={ms:9.4f} ms  BM={cfg.kwargs['BLOCK_M']} "
              f"BN={cfg.kwargs['BLOCK_N']} BK={cfg.kwargs['BLOCK_K']} "
              f"GM={cfg.kwargs['GROUP_M']} WM={cfg.kwargs.get('WARPS_M', '-')} "
              f"BUF={cfg.kwargs.get('NUM_BUFFERS', '-')} W={cfg.num_warps}{s2d_str} "
              f"MR={getattr(cfg, 'maxnreg', None)}", flush=True)


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
        s2d_str = " S2D=1" if cfg.kwargs.get("SCHEDULE_2D", False) else ""
        print(f"      {ms:7.4f} ms  {tops:6.2f} TOPS  grid={grid_x:4d}  "
              f"BM={cfg.kwargs.get('BLOCK_M')} BN={cfg.kwargs.get('BLOCK_N')} "
              f"BK={cfg.kwargs.get('BLOCK_K')} GM={cfg.kwargs.get('GROUP_M')} "
              f"WM={cfg.kwargs.get('WARPS_M', '-')} "
              f"BUF={cfg.kwargs.get('NUM_BUFFERS', '-')} "
              f"W={cfg.num_warps}{s2d_str} MR={mr}", flush=True)


def _launch_k1_raw_config(cfg, x, xq, xs, M, K, G, num_sms):
    """Launch one explicit K1 config without the autotune wrapper."""
    import torch
    num_tiles = triton.cdiv(M, cfg.kwargs["BLOCK_M"])
    occupancy = _K1_CFG_OCCUPANCY.get(
        _cfg_occupancy_key(cfg), AUTOTUNE_OCCUPANCY_ASSUMPTION)
    grid_x = _autotune_grid_x(num_tiles, num_sms, occupancy)
    kernel1_convrot_quant.fn[(grid_x,)](
        x, xq, xs,
        M, K, G,
        K, 1,            # stride_xm, stride_xk
        K, 1,            # stride_xqm, stride_xqk
        1,               # stride_xsm
        GROUP_SIZE=GROUP_SIZE,
        INPUT_FP16=(x.dtype == torch.float16),
        CONTIG_XK=True,
        **cfg.all_kwargs(),
    )


def _validate_k1_winner_bitwise(best_cfg, x, M, K, G, num_sms):
    """Require bitwise equality to the BM1 reference arithmetic."""
    import torch

    canary_rows = 16
    ref_cfg = Config({"BLOCK_M": 1}, num_warps=4, num_stages=1, maxnreg=128)

    def allocate_outputs():
        q = torch.full((M + canary_rows, K), -34, device="cuda",
                       dtype=torch.int8)
        s = torch.full((M + canary_rows,), float("nan"), device="cuda",
                       dtype=torch.float32)
        return q, s

    patterns = []
    patterns.append(("autotune-random", x))
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0x4B315F4558414354)  # "K1_EXACT"
    wide = torch.randn((M, K), generator=gen, device="cuda", dtype=x.dtype)
    row_exp = ((torch.arange(M, device="cuda") % 17) - 8).to(torch.float32)
    wide *= torch.pow(torch.tensor(2.0, device="cuda"), row_exp)[:, None]
    patterns.append(("wide-exponent-random", wide))

    for label, inp in patterns:
        q_ref, s_ref = allocate_outputs()
        q_got, s_got = allocate_outputs()
        _launch_k1_raw_config(ref_cfg, inp, q_ref, s_ref,
                              M, K, G, num_sms)
        _launch_k1_raw_config(best_cfg, inp, q_got, s_got,
                              M, K, G, num_sms)
        torch.cuda.synchronize()

        if not torch.isfinite(s_ref[:M]).all().item():
            raise SystemExit(f"[K1 exactness] reference produced NaN/Inf ({label})")
        if not torch.isfinite(s_got[:M]).all().item():
            raise SystemExit(f"[K1 exactness] candidate produced NaN/Inf ({label})")
        if not torch.equal(q_ref[:M], q_got[:M]):
            mismatches = int((q_ref[:M] != q_got[:M]).sum().item())
            raise SystemExit(
                f"[K1 exactness] INT8 mismatch for {label}: {mismatches} bytes; "
                f"candidate={best_cfg}")
        if not torch.equal(s_ref[:M].view(torch.int32),
                           s_got[:M].view(torch.int32)):
            mismatches = int((s_ref[:M].view(torch.int32) !=
                              s_got[:M].view(torch.int32)).sum().item())
            raise SystemExit(
                f"[K1 exactness] FP32 scale bit mismatch for {label}: "
                f"{mismatches} values; candidate={best_cfg}")

        if not torch.all(q_ref[M:] == -34).item() or not torch.all(q_got[M:] == -34).item():
            raise SystemExit(f"[K1 exactness] X_q tail canary corrupted ({label})")
        if not torch.isnan(s_ref[M:]).all().item() or not torch.isnan(s_got[M:]).all().item():
            raise SystemExit(f"[K1 exactness] X_scale tail canary corrupted ({label})")

    print(f"    [K1 exactness] PASS: candidate is bitwise equal to BM1 "
          f"for {len(patterns)} distributions; canaries intact", flush=True)


def extract_k1(arch, debug_dump=False, num_sms=0, l2_bytes=0, shared_mem_per_sm=0):
    import torch
    results = {}
    if num_sms == 0:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    for dtype_suffix, input_fp16, _ in DTYPE_CONFIGS:
        print(f"\n  K1 G256 {dtype_suffix} (autotuning)...", flush=True)
        _parallel_precompile_k1(_K1_CONFIGS, input_fp16)
        dtype = torch.float16 if input_fp16 else torch.float32
        M, K = K1_AUTOTUNE_M, 2560
        G = K // GROUP_SIZE
        torch.manual_seed(0x4B31)
        x = torch.randn((M, K), device="cuda", dtype=dtype)
        xq = torch.empty((M, K), device="cuda", dtype=torch.int8)
        xs = torch.empty((M,), device="cuda", dtype=torch.float32)
        grid_k1 = lambda meta: (_autotune_grid_x_for_k1(
            triton.cdiv(M, meta["BLOCK_M"]), num_sms, meta),)

        kernel1_convrot_quant[grid_k1](
            x, xq, xs,
            M, K, G,
            K, 1,
            K, 1,
            1,
            GROUP_SIZE=GROUP_SIZE,
            INPUT_FP16=input_fp16,
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
        spill_status = (
            f"no spills ({spill_info['method']}, regs={spill_info['registers']}, "
            f"sass_local={spill_info.get('sass_local_ops', -1)}, "
            f"scratch={spill_info.get('global_scratch', -1)})"
        )

        _validate_k1_winner_bitwise(best_cfg, x, M, K, G, num_sms)

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
    # Build and benchmark separate compile-time bias/no-bias K2 variants.
    for bias_suffix, _has_bias in BIAS_CONFIGS:
        global _K2_ACTIVE_HAS_BIAS
        _K2_ACTIVE_HAS_BIAS = _has_bias
        for dtype_suffix, _, output_fp16 in DTYPE_CONFIGS:
            config_name = f"{bias_suffix}_{dtype_suffix}"
            variant_configs = [
                cfg for cfg in _K2_CONFIGS
                if _K2_CFG_VALID_BY_BIAS.get(
                    (_cfg_occupancy_key(cfg), bool(_has_bias)), True)
            ]
            print(f"\n  K2 G256 {config_name} (Gluon custom-bench selection, "
                  f"{len(variant_configs)} legal configs)...", flush=True)
            _parallel_precompile_k2(variant_configs, output_fp16, arch)
            # Select on all production shape families using the exact masked-M
            # single-launch enqueue path.  Every candidate must first match the
            # previously shipped K2 output
            # bit-for-bit on every shape.
            best_cfg, bench_results = _bench_k2_workload(
                variant_configs, num_sms, _has_bias, output_fp16)
            if best_cfg is None:
                failures = [err for _, _, _, err in bench_results if err]
                raise SystemExit(
                    f"[K2 workload] {config_name}: no spill-free, bitwise-exact "
                    f"config passed all shapes. First failures: {failures[:3]}")
            _print_k2_workload_topn(bench_results)

            # Gluon has no @triton.autotune wrapper/cache walk.  Compile the
            # exact signature/attrs specialization used by the spill gate and
            # harvest that CompiledKernel directly (normally a disk-cache hit).
            kernel = _compile_gluon_k2(
                best_cfg, _has_bias, output_fp16, arch)
            cache_dir = _get_kernel_cache_dir(kernel)
            if debug_dump:
                print(f"    [gluon-cache] winner cache_dir={cache_dir}", flush=True)

            block_m = best_cfg.kwargs["BLOCK_M"]
            block_n = best_cfg.kwargs["BLOCK_N"]
            block_k_winning = best_cfg.kwargs["BLOCK_K"]
            group_m = best_cfg.kwargs["GROUP_M"]
            warp_m = best_cfg.kwargs["WARPS_M"]
            num_buffers = best_cfg.kwargs["NUM_BUFFERS"]
            schedule_2d = bool(best_cfg.kwargs.get("SCHEDULE_2D", False))
            dynamic_inner = True
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
                    f"GM={group_m} WM={warp_m} W={nw} BUF={num_buffers} MR={mr}. "
                    f"Zero tolerance: refusing to write a spilling cubin into "
                    f"the generated header."
                )
            spill_status = f"no spills ({spill_info['method']}, regs={spill_info['registers']})"

            occ_ctas = _K2_CFG_OCCUPANCY_BY_BIAS.get(
                (_cfg_occupancy_key(best_cfg), bool(_has_bias)),
                _K2_CFG_OCCUPANCY.get(_cfg_occupancy_key(best_cfg)))
            occ_note = f", occupancy={occ_ctas} CTA/SM" if occ_ctas else ""
            win_row = next(((c, ms) for c, ms, _, e in bench_results
                            if c is best_cfg and e is None), None)
            win_note = ""
            if win_row is not None:
                _, win_ms = win_row
                win_note = f", modeled total K2={win_ms:.4f} ms/engine-call"
            print(f"  -> Gluon custom-bench winner: BM={block_m} BN={block_n} "
                  f"BK={block_k_winning} GM={group_m} WM={warp_m} "
                  f"W={nw} BUF={num_buffers} MR={mr}"
                  f"{occ_note}{win_note}", flush=True)
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
                "warp_m": int(warp_m),
                "num_buffers": int(num_buffers),
                "schedule_2d": schedule_2d,
                "dynamic_inner": dynamic_inner,
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
        "// K1 comes from Triton's JIT cache; K2 is the exact GluonASTSource",
        "// specialization compiled, bitwise-gated, and custom-benchmarked here.",
        "// Only production FP16IO specializations are embedded.",
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
            "schedule_2d": "false",
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
                "schedule_2d": "true" if r.get('schedule_2d', False) else "false",
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
        "    bool schedule_2d;",
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
            f'{d["schedule_2d"]}, {d["block_x"]}u, {d["block_y"]}u, {d["block_z"]}u, ' +
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
        "    int32_t& M, int32_t& K, int32_t& G,",
        "    int32_t& stride_xm, int32_t& stride_xk,",
        "    int32_t& stride_xqm, int32_t& stride_xqk,",
        "    int32_t& stride_xsm) {",
        "    switch (id) {",
        "        case ConvRotLaunchArgId::kX_ptr: return &x_param;",
        "        case ConvRotLaunchArgId::kX_q_ptr: return &xq_ptr;",
        "        case ConvRotLaunchArgId::kX_scale_ptr: return &xs_ptr;",
        "        case ConvRotLaunchArgId::kM: return &M;",
        "        case ConvRotLaunchArgId::kK: return &K;",
        "        case ConvRotLaunchArgId::kG: return &G;",
        "        case ConvRotLaunchArgId::kStride_xm: return &stride_xm;",
        "        case ConvRotLaunchArgId::kStride_xk: return &stride_xk;",
        "        case ConvRotLaunchArgId::kStride_xqm: return &stride_xqm;",
        "        case ConvRotLaunchArgId::kStride_xqk: return &stride_xqk;",
        "        case ConvRotLaunchArgId::kStride_xsm: return &stride_xsm;",
        "        default: return nullptr;",
        "    }",
        "}",
        "",
        "inline void* selectGemmLaunchParam(",
        "    ConvRotLaunchArgId id,",
        "    void*& xq_ptr, void*& xs_ptr, void*& wq_param, void*& ws_param, void*& bias_param, void*& y_ptr,",
        "    int32_t& M, int32_t& N, int32_t& K,",
        "    int32_t& stride_xqm, int32_t& stride_xqk,",
        "    int32_t& stride_xsm,",
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
        "        case ConvRotLaunchArgId::kStride_xqm: return &stride_xqm;",
        "        case ConvRotLaunchArgId::kStride_xqk: return &stride_xqk;",
        "        case ConvRotLaunchArgId::kStride_xsm: return &stride_xsm;",
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
        "    int32_t M, int32_t K, int32_t G, int32_t num_sms,",
        "    int32_t stride_xm, int32_t stride_xk,",
        "    int32_t stride_xqm, int32_t stride_xqk,",
        "    int32_t stride_xsm,",
        "    uint32_t max_active_ctas_per_sm = 0) {",
        "    void* triton_scratch1 = nullptr;",
        "    void* triton_scratch2 = nullptr;",
        "    if (d.runtime_arg_count + d.trailing_scratch_ptr_count > kConvRotMaxLaunchParams ||",
        "        d.trailing_scratch_ptr_count > 2) {",
        "        return CUDA_ERROR_INVALID_VALUE;",
        "    }",
        "    void* x_param = const_cast<void*>(x_ptr);",
        "    // v13: K1 grid is (cdiv(M, BLOCK_M),) — one CTA per BLOCK_M rows.",
        "    // Each CTA processes the FULL K dimension of its row(s).",
        "    uint32_t const num_pid_m = ceilDivU32(M, d.block_m);",
        "    uint32_t const total_tiles = num_pid_m;",
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
        "            M, K, G,",
        "            stride_xm, stride_xk, stride_xqm, stride_xqk,",
        "            stride_xsm);",
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
        "    int32_t stride_xsm,",
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
        "    // §1.1: see launchConvRotQuant — occupancy-driven grid sizing.",
        "    //        Patch review §5: must pass the real block size, not 0.",
        "    uint32_t const block_threads = d.block_x * d.block_y * d.block_z;",
        "    uint32_t const occupancy = (max_active_ctas_per_sm != 0)",
        "        ? max_active_ctas_per_sm",
        "        : queryMaxActiveCtasPerSm(func, block_threads, static_cast<uint32_t>(d.shared_bytes));",
        "    uint32_t const raw_grid_cap = static_cast<uint32_t>(num_sms) * (occupancy == 0 ? 1u : occupancy);",
        "    uint32_t const grid_cap = raw_grid_cap == 0 ? 1u : raw_grid_cap;",
        "    uint32_t launch_grid_x = 1u;",
        "    uint32_t launch_grid_y = 1u;",
        "    if (d.schedule_2d) {",
        "        launch_grid_x = (grid_m < grid_cap) ? grid_m : grid_cap;",
        "        uint32_t const denom = launch_grid_x == 0 ? 1u : launch_grid_x;",
        "        uint32_t const remaining = grid_cap / denom;",
        "        uint32_t const y_cap = remaining == 0 ? 1u : remaining;",
        "        launch_grid_y = (grid_n < y_cap) ? grid_n : y_cap;",
        "    } else {",
        "        launch_grid_x = persistentGridU32(total_tiles,",
        "                                          static_cast<uint32_t>(num_sms),",
        "                                          occupancy);",
        "    }",
        "    void* params[kConvRotMaxLaunchParams] = {};",
        "    uint32_t n = 0;",
        "    for (uint32_t i = 0; i < d.runtime_arg_count; ++i) {",
        "        void* slot = selectGemmLaunchParam(",
        "            d.runtime_args[i].id,",
        "            xq_ptr, xs_ptr, wq_param, ws_param, bias_param, y_ptr,",
        "            M, N, K,",
        "            stride_xqm, stride_xqk, stride_xsm,",
        "            stride_wn, stride_wk, stride_ym, stride_yn);",
        "        if (slot == nullptr) return CUDA_ERROR_INVALID_VALUE;",
        "        params[n++] = slot;",
        "    }",
        "    void* scratch_slots[] = { &triton_scratch1, &triton_scratch2 };",
        "    for (uint32_t i = 0; i < d.trailing_scratch_ptr_count; ++i) params[n++] = scratch_slots[i];",
        "    return cuLaunchKernel(",
        "        func, launch_grid_x, launch_grid_y, 1, d.block_x, d.block_y, d.block_z,",
        "        static_cast<unsigned int>(d.shared_bytes), stream, params, nullptr);",
        "}",
        "",
        "} // namespace hotstep::convrot_int8_generated",
        "",
        "// End of generated Triton/Gluon cubins and launch descriptors.",
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
    triton_target = triton.runtime.driver.active.get_current_target()
    target_backend = str(triton_target.backend).lower()
    target_arch = triton_target.arch
    if target_backend == "cuda":
        arch = int(target_arch)
    else:
        # The only non-CUDA route currently reaches a strict K1 placeholder;
        # keep the native gfx string rather than inventing a CUDA-like number.
        arch = str(target_arch)
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

    if target_backend == "cuda":
        if arch >= 89:
            shared_mem_per_sm = 100 * 1024
        elif arch == 87:
            shared_mem_per_sm = 164 * 1024
        elif arch >= 86:
            shared_mem_per_sm = 100 * 1024
        elif arch >= 80:
            shared_mem_per_sm = 164 * 1024
        elif arch >= 75:
            shared_mem_per_sm = 64 * 1024
        else:
            shared_mem_per_sm = 48 * 1024
        arch_label = f"sm_{arch}"
    else:
        shared_mem_per_sm = int(
            getattr(props, "shared_memory_per_multiprocessor", 0) or
            getattr(props, "shared_memory_per_block", 0) or
            64 * 1024
        )
        arch_label = str(arch)

    print(f"[extract_jit_cubins_autotune] backend={target_backend}, "
          f"arch={arch_label}, SM/CUs={num_sms}, "
          f"L2={l2_bytes/(1024*1024):.1f} MB, shared/SM={shared_mem_per_sm//1024} KB",
          flush=True)

    global _K1_CONFIGS, _K2_CONFIGS
    # Strict K1 family dispatch happens before any K2 construction.  sm75 and
    # gfx11 intentionally stop at their placeholders in this patch.
    _K1_CONFIGS[:] = _estimate_k1_configs(
        num_sms, l2_bytes, shared_mem_per_sm, AUTOTUNE_SHAPES_K2,
        target_backend=target_backend, target_arch=target_arch)
    if target_backend != "cuda":
        raise SystemExit("[K2 dispatch] Gluon K2 requires the CUDA backend")
    _require_gluon_for_k2()
    # The Gluon builder is already the complete, deliberately bounded search
    # space; do not feed it the old Triton auto-layout candidate injectors.
    _K2_CONFIGS[:] = _estimate_k2_configs(
        num_sms, l2_bytes, shared_mem_per_sm, AUTOTUNE_SHAPES_K2, arch=arch)
    # The plugin's workspace/tail contract is strict; never benchmark a tile
    # that could win extraction but be rejected by enqueue.
    _K2_CONFIGS[:] = [c for c in _K2_CONFIGS
                      if c.kwargs["BLOCK_M"] <= K2_AUTOTUNE_M_ALIGNMENT]

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

    kernel1_convrot_quant.configs = _K1_CONFIGS

    print(f"[extract_jit_cubins_autotune] GROUP_SIZE={GROUP_SIZE}", flush=True)
    print(f"[extract_jit_cubins_autotune] K2 max shared cap: {K2_MAX_SHARED_BYTES if K2_MAX_SHARED_BYTES > 0 else 'disabled'} bytes", flush=True)
    print(f"[extract_jit_cubins_autotune] K1 configs: {len(_K1_CONFIGS)} (heuristic-pruned + spill-filtered)", flush=True)
    print(f"[extract_jit_cubins_autotune] K2 configs: {len(_K2_CONFIGS)} "
          f"(explicit Gluon layouts + spill-filtered)", flush=True)
    # Every timed candidate uses the occupancy derived from its exact compiled
    # specialization, matching the production persistent-grid formula.
    from collections import Counter
    for stage_name, occ_map in (("K1", _K1_CFG_OCCUPANCY),
                                ("K2", _K2_CFG_OCCUPANCY)):
        if occ_map:
            occ_hist = Counter(occ_map.values())
            occ_summary = ", ".join(f"{occ} CTA/SM: {n}"
                                    for occ, n in sorted(occ_hist.items()))
            print(f"[extract_jit_cubins_autotune] {stage_name} per-config "
                  f"occupancy: {occ_summary}", flush=True)
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