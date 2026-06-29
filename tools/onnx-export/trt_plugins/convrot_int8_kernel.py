r"""Fused ConvRot INT8 Linear kernel (Phase 1, butterfly rotation).

The entire w8a8 + ConvRot path is fused into a single Triton kernel:

  1. Load activation tile x [BLOCK_M, BLOCK_K] from HBM
  2. ConvRot rotation via in-register H_4 Kronecker butterfly (no H matrix)
  3. Per-row dynamic INT8 quantization (two-pass: max-abs then quantize)
  4. INT8 x INT8 -> INT32 GEMM via Tensor Cores (tl.dot)
  5. Dequant + bias + FP16 cast epilogue

The rotation uses the recursive structure H_{4^k} = H_4 \otimes H_{4^{k-1}},
implemented as k successive in-register H_4 butterfly stages with strides
1, 4, 16, ..., 4^{k-1}. This avoids materializing the full [GROUP_SIZE x
GROUP_SIZE] Hadamard matrix in shared memory (which would be 128 KB for
GROUP_SIZE=256 in FP16 — exceeding the sm_86 hardware limit of 99 KB).

All butterfly operations (reshape, permute, split, join, elementwise
arithmetic) are pure register operations. No HBM traffic, no shared memory.
The kernel remains a single fused launch — fusion is NOT broken.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


# ─── H_4 butterfly primitives ───────────────────────────────────────────────
#
# H_4 = (1/2) * [[ 1  1  1 -1],
#                [ 1  1 -1  1],
#                [ 1 -1  1  1],
#                [-1  1  1  1]]
#
# Applied to (v0, v1, v2, v3):
#   out0 = ( v0 + v1 + v2 - v3) / 2
#   out1 = ( v0 + v1 - v2 + v3) / 2
#   out2 = ( v0 - v1 + v2 + v3) / 2
#   out3 = (-v0 + v1 + v2 + v3) / 2
#
# This matches the sign pattern in the original CUDA kernel
# (convrot_activation_quant_kernel's regular_hadamard_transform).

@triton.jit
def _hadamard_butterfly_stage(x_tile, BLOCK_M: tl.constexpr, GROUP_SIZE: tl.constexpr,
                              STAGE: tl.constexpr):
    """One H_4 Kronecker butterfly stage.

    Pulled into its own JIT helper so each (unrolled) iteration of the
    enclosing `tl.static_range` gets a fresh constexpr scope. Inside the
    loop body of `_hadamard_butterfly`, annotating `s` / `n_groups` with
    `: tl.constexpr` triggers "constexpr cannot be reassigned" on the
    second iteration, and *not* annotating them causes Triton to fold
    them into `constexpr[tensor]` instead of `constexpr[int]` (which
    `tl.reshape` rejects). A separate JIT function is the canonical fix:
    every call site binds its own `s` / `n_groups` constexprs from the
    STAGE argument.
    """
    s: tl.constexpr = 4 ** STAGE
    n_groups: tl.constexpr = GROUP_SIZE // (4 * s)

    # Reshape [BLOCK_M, GROUP_SIZE] -> [BLOCK_M, n_groups, 4, s]
    x_v = tl.reshape(x_tile, (BLOCK_M, n_groups, 4, s))

    # Permute to [BLOCK_M, n_groups, s, 4]
    x_v = tl.permute(x_v, (0, 1, 3, 2))

    # Apply H_4 to the last axis. Triton's `tl.split` requires a split
    # dimension of exactly 2. We reshape [..., 4] -> [..., 2, 2] safely
    # using explicit integer parameters to avoid AST parsing bugs.
    x_2x2 = tl.reshape(x_v, (BLOCK_M, n_groups, s, 2, 2))

    # Split [..., 2, 2] -> two tensors of shape [..., 2]
    lo, hi = tl.split(x_2x2)

    # Split [..., 2] -> two tensors of shape [...] (the dim is cleanly dropped)
    v0, v1 = tl.split(lo)
    v2, v3 = tl.split(hi)

    # In-register FMA
    h0 = 0.5 * ( v0 + v1 + v2 - v3)
    h1 = 0.5 * ( v0 + v1 - v2 + v3)
    h2 = 0.5 * ( v0 - v1 + v2 + v3)
    h3 = 0.5 * (-v0 + v1 + v2 + v3)

    # Recombine: join creates a new minor dimension of size 2.
    lo_out = tl.join(h0, h1)  # -> [BLOCK_M, n_groups, s, 2]
    hi_out = tl.join(h2, h3)  # -> [BLOCK_M, n_groups, s, 2]
    out = tl.join(lo_out, hi_out)  # -> [BLOCK_M, n_groups, s, 2, 2]

    # Flatten back to [BLOCK_M, n_groups, s, 4]
    x_v = tl.reshape(out, (BLOCK_M, n_groups, s, 4))

    # Permute back to [BLOCK_M, n_groups, 4, s]
    x_v = tl.permute(x_v, (0, 1, 3, 2))

    # Reshape back to flat [BLOCK_M, GROUP_SIZE]
    return tl.reshape(x_v, (BLOCK_M, GROUP_SIZE))


@triton.jit
def _hadamard_butterfly(x_tile, BLOCK_M: tl.constexpr, GROUP_SIZE: tl.constexpr):
    """Apply H_{GROUP_SIZE} to the last axis of x_tile via k H_4 butterfly stages.

    x_tile: [BLOCK_M, GROUP_SIZE] tile in registers.
    BLOCK_M: Size of the M dimension (rows).
    GROUP_SIZE: must be a power of 4 (4, 16, 64, 256, 1024).
    Returns: rotated tile of same shape.
    """
    # Compile-time stage count (log_4(GROUP_SIZE))
    if GROUP_SIZE == 4:
        NUM_STAGES: tl.constexpr = 1
    elif GROUP_SIZE == 16:
        NUM_STAGES: tl.constexpr = 2
    elif GROUP_SIZE == 64:
        NUM_STAGES: tl.constexpr = 3
    elif GROUP_SIZE == 256:
        NUM_STAGES: tl.constexpr = 4
    elif GROUP_SIZE == 1024:
        NUM_STAGES: tl.constexpr = 5
    else:
        tl.static_assert(False, "GROUP_SIZE must be in {4, 16, 64, 256, 1024}")

    for stage in tl.static_range(0, NUM_STAGES):
        # Delegate to a per-stage helper. Each call binds its own fresh
        # constexprs (`s`, `n_groups`) computed from `stage`, sidestepping
        # the two `tl.constexpr` reassignment / type-folding problems
        # documented inside `_hadamard_butterfly_stage`.
        x_tile = _hadamard_butterfly_stage(x_tile, BLOCK_M, GROUP_SIZE, stage)

    return x_tile


@triton.jit
def _hadamard_butterfly_stage_vec(x_vec, GROUP_SIZE: tl.constexpr, STAGE: tl.constexpr):
    """One H_4 Kronecker butterfly stage for a single row vector."""
    s: tl.constexpr = 4 ** STAGE
    n_groups: tl.constexpr = GROUP_SIZE // (4 * s)
    x_v = tl.reshape(x_vec, (n_groups, 4, s))
    x_v = tl.permute(x_v, (0, 2, 1))  # [n_groups, s, 4]
    x_2x2 = tl.reshape(x_v, (n_groups, s, 2, 2))
    lo, hi = tl.split(x_2x2)
    v0, v1 = tl.split(lo)
    v2, v3 = tl.split(hi)
    h0 = 0.5 * ( v0 + v1 + v2 - v3)
    h1 = 0.5 * ( v0 + v1 - v2 + v3)
    h2 = 0.5 * ( v0 - v1 + v2 + v3)
    h3 = 0.5 * (-v0 + v1 + v2 + v3)
    lo_out = tl.join(h0, h1)
    hi_out = tl.join(h2, h3)
    out = tl.join(lo_out, hi_out)
    x_v = tl.reshape(out, (n_groups, s, 4))
    x_v = tl.permute(x_v, (0, 2, 1))
    return tl.reshape(x_v, (GROUP_SIZE,))


@triton.jit
def _hadamard_butterfly_vec(x_vec, GROUP_SIZE: tl.constexpr):
    """Apply H_{GROUP_SIZE} to one row vector via H_4 butterfly stages."""
    if GROUP_SIZE == 4:
        NUM_STAGES: tl.constexpr = 1
    elif GROUP_SIZE == 16:
        NUM_STAGES: tl.constexpr = 2
    elif GROUP_SIZE == 64:
        NUM_STAGES: tl.constexpr = 3
    elif GROUP_SIZE == 256:
        NUM_STAGES: tl.constexpr = 4
    elif GROUP_SIZE == 1024:
        NUM_STAGES: tl.constexpr = 5
    else:
        tl.static_assert(False, "GROUP_SIZE must be in {4, 16, 64, 256, 1024}")
    for stage in tl.static_range(0, NUM_STAGES):
        x_vec = _hadamard_butterfly_stage_vec(x_vec, GROUP_SIZE, stage)
    return x_vec


# ─── Fused kernel ───────────────────────────────────────────────────────────

@triton.jit
def fused_convrot_gemm_rowwise_kernel(
    X_ptr, W_ptr, Y_ptr, W_scale_ptr, Bias_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INPUT_FP16: tl.constexpr,
    OUTPUT_FP16: tl.constexpr
):
    """Fused ConvRot + row-wise INT8 quant + GEMM + epilogue.

    Parameters (14 total, no H_ptr — butterfly is H-matrix-free):
      X_ptr        — [M, K] FP16 or FP32 activations
      W_ptr        — [N, K] INT8 quantized weights
      Y_ptr        — [M, N] FP16 or FP32 output
      W_scale_ptr  — [N] FP32 per-output-channel weight scales
      Bias_ptr     — [N] FP32 bias (unused if HAS_BIAS=False)
      M, N, K      — matmul dims
      stride_* — row-major strides (tensors are kLINEAR)
      BLOCK_M/N/K  — tile sizes (constexpr, baked into cubin at AOT compile)
      GROUP_SIZE   — ConvRot group size (0 = no rotation, else power of 4)
      HAS_BIAS     — whether Bias_ptr is valid
      INPUT_FP16   — X_ptr dtype is FP16 (True) or FP32 (False)
      OUTPUT_FP16  — Y_ptr dtype is FP16 (True) or FP32 (False)

    Phase-1 contract:
      * GROUP_SIZE ∈ {0, 4, 16, 64, 256, 1024}.
      * When GROUP_SIZE > 0, BLOCK_K MUST equal GROUP_SIZE. The K-loop iterates
        over the full K dimension in GROUP_SIZE-sized chunks, and each chunk
        gets one full butterfly rotation. This invariant is statically asserted.
      * The rotation uses in-register H_4 Kronecker butterflies — no H matrix
        is loaded from HBM, no shared memory is used for rotation.
    """
    # Compile-time safety: BLOCK_K must equal GROUP_SIZE for ROT configs so
    # each K-tile gets exactly one full butterfly rotation.
    if GROUP_SIZE > 0:
        tl.static_assert(
            BLOCK_K == GROUP_SIZE,
            "BLOCK_K must equal GROUP_SIZE when rotation is enabled "
            "(each K-tile gets one full butterfly rotation)."
        )

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = (pid % num_pid_m) * BLOCK_M
    pid_n = (pid // num_pid_m) * BLOCK_N

    # Compute un-wrapped row and col indices for masking
    rm = pid_m + tl.arange(0, BLOCK_M)
    rn = pid_n + tl.arange(0, BLOCK_N)

    # Wrap out-of-bounds indices so cp.async/TMA pointer calculations stay valid.
    # Note: Do NOT use tl.max_contiguous or tl.multiple_of here, as those compiler hints
    # force Triton 3.x to assume linear contiguity, optimizing away the modulo wrapping
    # and causing CUDA_ERROR_ILLEGAL_ADDRESS on out-of-bounds row evaluation!
    ram = rm % M
    rbn = rn % N

    # FP32 accumulator for the GEMM
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── PASS 1: compute per-row max-abs of rotated x across the full K dim ──
    row_max = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_offset in tl.range(0, K, BLOCK_K):
        cols_x = k_offset + tl.arange(0, BLOCK_K)
        x_ptr_block = X_ptr + ram[:, None] * stride_xm + cols_x[None, :] * stride_xk
        x_block = tl.load(x_ptr_block,
                          mask=(rm[:, None] < M) & (cols_x[None, :] < K),
                          other=0.0)

        # Cast to FP32 for high-precision rotation and reduction
        if INPUT_FP16:
            x_block = x_block.to(tl.float32)

        # ConvRot rotation via in-register H_4 butterfly (no H matrix, no SMEM)
        if GROUP_SIZE > 0:
            rotated_x = _hadamard_butterfly(x_block, BLOCK_M, GROUP_SIZE)
        else:
            rotated_x = x_block

        # Row-wise max-abs, accumulate across K tiles
        block_max = tl.max(tl.abs(rotated_x), axis=1)  # [BLOCK_M]
        row_max = tl.maximum(row_max, block_max)

    # Per-row scale: scale = max(|x_rot|) / 127, clamped to avoid div-by-zero
    row_scale = tl.maximum(row_max / 127.0, 1e-30)  # [BLOCK_M]

    # ── PASS 2: rotate, quantize, and GEMM ──
    for k_offset in tl.range(0, K, BLOCK_K):
        cols_x = k_offset + tl.arange(0, BLOCK_K)
        x_ptr_block = X_ptr + ram[:, None] * stride_xm + cols_x[None, :] * stride_xk
        x_block = tl.load(x_ptr_block,
                          mask=(rm[:, None] < M) & (cols_x[None, :] < K),
                          other=0.0)

        if INPUT_FP16:
            x_block = x_block.to(tl.float32)

        # ConvRot rotation via butterfly (recompute — same as pass 1)
        if GROUP_SIZE > 0:
            rotated_x = _hadamard_butterfly(x_block, BLOCK_M, GROUP_SIZE)
        else:
            rotated_x = x_block

        # Symmetric INT8 quantization: rint(x_rot / scale) clamped to [-127, 127]
        x_q = libdevice.rint(rotated_x / row_scale[:, None])
        x_q = tl.clamp(x_q, -127.0, 127.0).to(tl.int8)

        # Load weight tile [BLOCK_N, BLOCK_K]
        cols_w = k_offset + tl.arange(0, BLOCK_K)
        w_ptr_block = W_ptr + rbn[:, None] * stride_wn + cols_w[None, :] * stride_wk
        w_q = tl.load(w_ptr_block,
                      mask=(rn[:, None] < N) & (cols_w[None, :] < K),
                      other=0)

        # INT8 x INT8 -> INT32 GEMM via Tensor Cores
        partial_acc = tl.dot(x_q, tl.trans(w_q), allow_tf32=False)
        accumulator += partial_acc.to(tl.float32)

    # ── EPILOGUE: dequant + bias + cast ──
    w_scale = tl.load(W_scale_ptr + rn, mask=rn < N)  # [BLOCK_N]

    # Output = acc * row_scale * w_scale + bias
    out_scale = row_scale[:, None] * w_scale[None, :]  # [BLOCK_M, BLOCK_N]
    y_block = accumulator * out_scale

    if HAS_BIAS:
        bias = tl.load(Bias_ptr + rn, mask=rn < N)  # [BLOCK_N]
        y_block += bias[None, :]

    # Cast and store
    y_ptr_block = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    if OUTPUT_FP16:
        tl.store(y_ptr_block, y_block.to(tl.float16),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))
    else:
        tl.store(y_ptr_block, y_block.to(tl.float32),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.jit
def fused_convrot_gemm_rowwise_m1_kernel(
    X_ptr, W_ptr, Y_ptr, W_scale_ptr, Bias_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INPUT_FP16: tl.constexpr,
    OUTPUT_FP16: tl.constexpr
):
    """Triton-only specialized fused path for M == 1.

    The generic 128x128 tl.dot cubin is correct in principle, but for DiT's
    one-row time-embedding projections it produces very large G256 kernels.
    This variant keeps the same single Triton launch and same math, but maps
    one program to one output-channel tile and reduces over K explicitly.  It
    avoids the huge 128-row tensor-core tile for the M=1 case that produced
    sticky CUDA illegal-address errors in runtime testing.
    """
    if GROUP_SIZE > 0:
        tl.static_assert(BLOCK_K == GROUP_SIZE,
                         "BLOCK_K must equal GROUP_SIZE for M1 rotated configs")

    pid_n = tl.program_id(0) * BLOCK_N
    cols_n = pid_n + tl.arange(0, BLOCK_N)
    rbn = cols_n % N

    # PASS 1: row max across all rotated K chunks for the single row.
    row_max = tl.full((), 0.0, dtype=tl.float32)
    for k_offset in tl.range(0, K, BLOCK_K):
        cols_k = k_offset + tl.arange(0, BLOCK_K)
        x_vec = tl.load(X_ptr + cols_k * stride_xk,
                        mask=cols_k < K,
                        other=0.0)
        if INPUT_FP16:
            x_vec = x_vec.to(tl.float32)
        if GROUP_SIZE > 0:
            x_rot = _hadamard_butterfly_vec(x_vec, GROUP_SIZE)
        else:
            x_rot = x_vec
        row_max = tl.maximum(row_max, tl.max(tl.abs(x_rot), axis=0))

    row_scale = tl.maximum(row_max / 127.0, 1e-30)

    # PASS 2: rotate/quantize and explicitly reduce W_q dot x_q for one row.
    acc = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for k_offset in tl.range(0, K, BLOCK_K):
        cols_k = k_offset + tl.arange(0, BLOCK_K)
        x_vec = tl.load(X_ptr + cols_k * stride_xk,
                        mask=cols_k < K,
                        other=0.0)
        if INPUT_FP16:
            x_vec = x_vec.to(tl.float32)
        if GROUP_SIZE > 0:
            x_rot = _hadamard_butterfly_vec(x_vec, GROUP_SIZE)
        else:
            x_rot = x_vec
        x_q = libdevice.rint(x_rot / row_scale)
        x_q = tl.clamp(x_q, -127.0, 127.0).to(tl.int32)

        w = tl.load(W_ptr + rbn[:, None] * stride_wn + cols_k[None, :] * stride_wk,
                    mask=(cols_n[:, None] < N) & (cols_k[None, :] < K),
                    other=0).to(tl.int32)
        acc += tl.sum(w * x_q[None, :], axis=1)

    w_scale = tl.load(W_scale_ptr + cols_n, mask=cols_n < N, other=0.0)
    y_vec = acc.to(tl.float32) * row_scale * w_scale
    if HAS_BIAS:
        bias = tl.load(Bias_ptr + cols_n, mask=cols_n < N, other=0.0)
        y_vec += bias

    y_ptrs = Y_ptr + cols_n * stride_yn
    if OUTPUT_FP16:
        tl.store(y_ptrs, y_vec.to(tl.float16), mask=cols_n < N)
    else:
        tl.store(y_ptrs, y_vec.to(tl.float32), mask=cols_n < N)
