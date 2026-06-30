r"""Triton kernels for the ConvRot INT8 TensorRT plugin.

The runtime plugin uses one execution family for every M, including M == 1:

``kernel1_convrot_quant`` + ``kernel2_gemm_dequant``
    Kernel 1 rotates and quantizes activations once, writing the reusable INT8
    activation tile and its per-group scale to workspace. Kernel 2 reuses that
    workspace across all N tiles.

Earlier revisions also compiled a dedicated single-launch M==1 kernel. Profiling
on the real M==1 DiT workload showed the two-kernel BK64/BM128/BN128 path wins
there too, so the specialized M1 entry point was intentionally removed. Keeping a
single launch family also avoids another set of cubins and runtime dispatch
state.

The rotation itself is still the in-register H_4 Kronecker butterfly used by
Phase 1 of the project. No dense Hadamard matrix is staged in shared memory.
Only GROUP_SIZE == 64 is compiled for the TensorRT plugin now; legacy no-rotation
(group_size == 0) and group_size == 256 cubins are deliberately not generated.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


# ─── H_4 butterfly primitives ──────────────────────────────────────────────
#
# The ConvRot transform applies a regular Hadamard matrix whose size is a
# power of 4. The TensorRT plugin currently compiles only GROUP_SIZE == 64, but
# the primitive is left expressed in stages to keep the math clear and to make
# future re-tuning straightforward if another group size is deliberately added.
# We implement the transform as successive in-register H_4 Kronecker stages;
# no dense [GROUP_SIZE, GROUP_SIZE] matrix is materialized in shared memory.

@triton.jit
def _hadamard_butterfly_stage(
    x_tile,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    STAGE: tl.constexpr,
):
    """Apply one H_4 Kronecker stage to a [BLOCK_M, GROUP_SIZE] tile."""
    s: tl.constexpr = 4 ** STAGE
    n_groups: tl.constexpr = GROUP_SIZE // (4 * s)

    # [BM, GS] -> [BM, n_groups, 4, s] -> [BM, n_groups, s, 4]
    x_v = tl.reshape(x_tile, (BLOCK_M, n_groups, 4, s))
    x_v = tl.permute(x_v, (0, 1, 3, 2))

    # Split the final 4-lane axis into a 2x2 structure so Triton's split/join
    # helpers can be used cleanly.
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

    # Undo the temporary reshapes/permutations.
    x_v = tl.reshape(out, (BLOCK_M, n_groups, s, 4))
    x_v = tl.permute(x_v, (0, 1, 3, 2))
    return tl.reshape(x_v, (BLOCK_M, GROUP_SIZE))


@triton.jit
def _hadamard_butterfly(x_tile, BLOCK_M: tl.constexpr, GROUP_SIZE: tl.constexpr):
    """Apply H_{GROUP_SIZE} to the last axis of a [BLOCK_M, GROUP_SIZE] tile."""
    if GROUP_SIZE == 64:
        NUM_STAGES: tl.constexpr = 3
    else:
        tl.static_assert(False, "GROUP_SIZE must be 64 for the compiled TRT plugin")

    for stage in tl.static_range(0, NUM_STAGES):
        x_tile = _hadamard_butterfly_stage(x_tile, BLOCK_M, GROUP_SIZE, stage)
    return x_tile


# ─── Kernel 1: rotate + quantize activations ───────────────────────────────

@triton.jit
def kernel1_convrot_quant(
    X_ptr,
    X_q_ptr,
    X_scale_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_xqm,
    stride_xqk,
    stride_xsm,
    stride_xsg,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    INPUT_FP16: tl.constexpr,
):
    """Rotate one activation block, quantize it, and write reusable workspace.

    The production TensorRT plugin compiles GROUP_SIZE == BLOCK_K == 64 only.
    There is no group_size == 0 no-rotation sentinel in the compiled cubin set;
    exporters should only create ConvRotInt8Linear nodes for weights that were
    rotated offline with the same 64-wide grouping.
    """
    tl.static_assert(GROUP_SIZE == 64, "Only group_size=64 is compiled")
    tl.static_assert(BLOCK_K == GROUP_SIZE, "BLOCK_K must equal GROUP_SIZE")

    pid_m = tl.program_id(0) * BLOCK_M
    pid_k = tl.program_id(1) * BLOCK_K

    rm = pid_m + tl.arange(0, BLOCK_M)
    rk = pid_k + tl.arange(0, BLOCK_K)
    ram = rm % M

    mask = (rm[:, None] < M) & (rk[None, :] < K)

    x_ptr_block = X_ptr + ram[:, None] * stride_xm + rk[None, :] * stride_xk
    x_block = tl.load(x_ptr_block, mask=mask, other=0.0, eviction_policy="evict_first")
    if INPUT_FP16:
        x_block = x_block.to(tl.float32)

    rotated_x = _hadamard_butterfly(x_block, BLOCK_M, GROUP_SIZE)

    # One scale per row per K-group.
    block_max = tl.max(tl.abs(rotated_x), axis=1)
    scale = tl.maximum(block_max / 127.0, 1e-30)

    x_q = libdevice.rint(rotated_x / scale[:, None])
    x_q = tl.clamp(x_q, -127.0, 127.0).to(tl.int8)

    xq_ptr_block = X_q_ptr + rm[:, None] * stride_xqm + rk[None, :] * stride_xqk
    tl.store(xq_ptr_block, x_q, mask=mask)

    group_idx = pid_k // BLOCK_K
    xs_ptr_block = X_scale_ptr + rm * stride_xsm + group_idx * stride_xsg
    # Store scale as FP32 (not FP16) — FP16 storage caused measurable quality
    # reduction vs the ggml/non-Triton baselines.  The per-group scale is
    # applied to every element of the INT32 partial accumulator, so even
    # small FP16 rounding errors compound across K_groups.
    tl.store(xs_ptr_block, scale, mask=rm < M)


# ─── Kernel 2: INT8 GEMM + per-group dequant ───────────────────────────────

@triton.jit
def kernel2_gemm_dequant(
    X_q_ptr,
    X_scale_ptr,
    W_q_ptr,
    W_scale_ptr,
    Bias_ptr,
    Y_ptr,
    M,
    N,
    K,
    stride_xqm,
    stride_xqk,
    stride_xsm,
    stride_xsg,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,
):
    """Consume quantized activations and accumulate dequantized FP32 output.

    Includes L2-cache-friendly program swizzle (super-grouping) and eviction
    policy hints.  GROUP_M controls how many M-tiles are grouped together in
    the launch order — programs within a group share X_q L2 residency.

    On 2 MB L2 (GA107M / RTX 3050 Laptop): GROUP_M=4 with BLOCK_M=128, K=2560
    → 4 × 128 × 2560 = 1.28 MB, fits in L2 with room for W streaming.
    GROUP_M=8 would be 2.56 MB — thrashes the cache.
    """
    tl.static_assert(BLOCK_K == 64, "Kernel 2 expects 64-wide activation quant groups")

    pid = tl.program_id(0)
    # NOTE: num_pid_m / num_pid_n must NOT be `tl.constexpr` — M and N are
    # runtime int32 parameters, so tl.cdiv returns a runtime value.  The
    # swizzle math below works fine at runtime; it does not need compile-time
    # constants.  Forcing `: tl.constexpr` here raises
    # "_semantic argument must be provided outside of JIT functions".
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2 swizzle: super-group M-tiles so adjacent pids share X_q L2 residency.
    # This is the canonical Triton matmul tutorial swizzle (03-matrix-multiplication).
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_m = pid_m * BLOCK_M
    pid_n = pid_n * BLOCK_N

    rm = pid_m + tl.arange(0, BLOCK_M)
    rn = pid_n + tl.arange(0, BLOCK_N)
    ram = rm % M
    rbn = rn % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # W_scale is per-channel [N] — load once, reuse across all K-groups.
    ws = tl.load(W_scale_ptr + rn, mask=rn < N, other=0.0)

    for k_offset in tl.range(0, K, BLOCK_K):
        cols_k = k_offset + tl.arange(0, BLOCK_K)

        # X_q is reused across K-groups within one program → evict_last (keep in L2).
        xq_ptr_block = X_q_ptr + ram[:, None] * stride_xqm + cols_k[None, :] * stride_xqk
        xq = tl.load(xq_ptr_block,
                     mask=(rm[:, None] < M) & (cols_k[None, :] < K),
                     other=0,
                     eviction_policy="evict_last")

        # W_q is streamed once per K-group, not reused → evict_first (don't pollute L2).
        wq_ptr_block = W_q_ptr + rbn[:, None] * stride_wn + cols_k[None, :] * stride_wk
        wq = tl.load(wq_ptr_block,
                     mask=(rn[:, None] < N) & (cols_k[None, :] < K),
                     other=0,
                     eviction_policy="evict_first")

        partial = tl.dot(xq, tl.trans(wq), allow_tf32=False)

        group_idx = k_offset // BLOCK_K
        xs_ptr_block = X_scale_ptr + ram * stride_xsm + group_idx * stride_xsg
        # X_scale is now stored as FP32 (was FP16) — no upcast needed.
        xs = tl.load(xs_ptr_block, mask=rm < M, other=0.0)

        acc += partial.to(tl.float32) * xs[:, None] * ws[None, :]

    if HAS_BIAS:
        bias = tl.load(Bias_ptr + rn, mask=rn < N, other=0.0)
        acc += bias[None, :]

    y_ptr_block = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    if OUTPUT_FP16:
        tl.store(y_ptr_block, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))
    else:
        tl.store(y_ptr_block, acc.to(tl.float32), mask=(rm[:, None] < M) & (rn[None, :] < N))
