r"""Triton kernels for the ConvRot INT8 TensorRT plugin (GROUP_SIZE=256, two-kernel design).

Architecture (see ``CONVROT_OPTIMAL_TWO_KERNEL_SPEC.md`` for the full design):

``kernel1_convrot_quant`` + ``kernel2_gemm_dequant``
    Kernel 1 rotates and quantizes activations once into a reusable INT8 workspace
    (``X_q``) plus a per-group FP32 scale workspace (``X_scale``).  Kernel 2
    consumes that workspace across all N output-channel tiles.

Key design decisions vs. the legacy GROUP_SIZE=64 implementation:

* **Tensor-Core rotation.**  The 256-point regular Hadamard rotation is implemented
  as a separable ``H_16 \otimes H_16`` Kronecker factorization computed via
  ``tl.dot`` (four FP16 MMAs with FP32 accumulate), NOT as an elementwise
  ``tl.split``/``tl.join`` H_4 butterfly.  This moves the rotation off the CUDA
  cores (which would otherwise sit idle during kernel 2's INT8 GEMM) and removes
  the register-pressure ceiling that a naive 64→256 group-size change hit.

* **In-kernel H_16 generation.**  The 16×16 regular Hadamard factor is
  reconstructed in-register by applying the existing (verified) H_4 Kronecker
  butterfly to a 16×16 identity matrix — once per CTA, not per tile.  This
  avoids any host-uploaded device buffer or C++-side plumbing: the exact same
  math that already produced correct GROUP_SIZE==64 rotations is reused, just
  applied to an identity input.  Every entry is exactly ±0.25 (FP16-exact).

* **Fixed 16-row sub-chunk rotation.**  Kernel 1 processes its BLOCK_M-row tile
  in fixed 16-row sub-chunks (``SUBCHUNK = 16``).  The rotation's ``tl.dot``
  shape is always ``(256, 16) × (16, 16)`` regardless of BLOCK_M, so the
  rotation's register/shared-memory footprint is independent of BLOCK_M.  This
  lets BLOCK_M be tuned purely for occupancy/tiling without rotation-driven
  register pressure scaling.

* **Hi/lo FP16 split.**  The activation operand of each rotation MMA is split
  into a high-half (exactly representable in FP16) and a low-half (residual),
  and two separate ``tl.dot`` calls are summed.  This recovers near-FP32
  accuracy from FP16 tensor-core MMAs.  The ``H_16`` weight operand does NOT
  need splitting — its entries are exactly ±0.25 (exactly representable in
  FP16).  This split is **not optional** — a naive single-pass FP16 rotation
  loses ~1% of INT8 output codes on outlier-heavy inputs.

* **Decoupled BLOCK_K from GROUP_SIZE in kernel 2.**  Kernel 2's INT8 MMA tile
  width (``BLOCK_K``, autotuned) is independent of the quantization group width
  (``GROUP_SIZE`` = 256).  The kernel accumulates INT32 partial dot products
  across ``GROUP_SIZE // BLOCK_K`` sub-tiles before applying the FP32 dequant
  scale once per 256-wide group.

* **Transposed ``X_scale`` workspace layout.**  ``X_scale`` is stored as
  ``[n_groups, M]`` row-major (``stride_xsm = 1, stride_xsg = M``) so that
  both kernels' per-row scale access is coalesced.

* **Persistent, grid-stride launch.**  Both kernels are launched with a 1D grid
  of ``min(NUM_SMS, num_tiles)`` CTAs and loop over tiles via
  ``tl.range(start_pid, num_tiles, NUM_SMS)``.  This eliminates wave-
  quantization loss for small grids.

Constants (non-negotiable per spec):
* ``GROUP_SIZE = 256`` (fixed).
* INT8 × INT8 → INT32 GEMM with FP32 dequant.
* Regular (non-Sylvester) H_4-Kronecker Hadamard rotation family.
* No calibration; per-row-per-group dynamic activation scales, per-channel
  static weight scales.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


# ─── H_4 butterfly primitive (used only to synthesize the 16x16 H factor) ──
#
# The ConvRot transform applies a regular Hadamard matrix whose size is a
# power of 4. This primitive is retained (unchanged) purely as the mechanism
# for reconstructing the exact, normalized 16x16 regular-Hadamard factor
# in-register (`_generate_h16_fp16` below) by applying it to an identity
# matrix. It is deliberately NOT used to rotate the [BLOCK_M, 256]
# activation tile itself anymore — see the module docstring for why.

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
def _generate_h16_fp16():
    """Reconstruct the normalized 16x16 regular-Hadamard factor in registers.

    Applies the (verified, unchanged) H_4 Kronecker butterfly to a 16x16
    identity matrix — since the butterfly implements a linear operator,
    applying it to the identity yields exactly that operator's matrix. This
    is two butterfly stages (16 == 4**2), executed once per kernel launch
    (not once per row-tile), and avoids needing any new host-uploaded
    device buffer or C++-side plumbing: the exact same math that already
    produced correct GROUP_SIZE==64 rotations is reused, just applied to an
    identity input instead of the activation tile.

    Every entry of the result is exactly ±0.25 (verified numerically), which
    is exactly representable in FP16 with zero rounding error — this is
    what allows `_rotate_256_tensorcore` below to skip the hi/lo split for
    the Hadamard-factor (weight) operand of its `tl.dot` calls and only
    split the activation operand.
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
    """Rotate a [SUBCHUNK, 256] fp32 tile by the normalized H_256 factor.

    Uses the separable Kronecker factorization H_256 = H_16 (x) H_16
    (verified bit-identical to the direct H_4^{(x)4} Kronecker construction
    in fp64), computed as two Tensor-Core `tl.dot` passes over 16x16 tiles:

      1. Reshape the flat 256-wide row into a [16, 16] tile (row-major, so
         element ``16*a + b`` sits at position ``[a, b]``).
      2. Apply H_16 along the inner axis ``b`` via right-multiply
         (``tl.dot``).
      3. Transpose the [16, 16] tile and apply H_16 along the (now inner)
         axis ``a`` via a second right-multiply `tl.dot` call.
      4. Transpose back and flatten to [SUBCHUNK, 256].

    Only the activation operand of each `tl.dot` is split into an FP16
    "hi" part (the value rounded to FP16) and an FP16 "lo" part (the
    residual, also rounded to FP16), with both `tl.dot` results accumulated
    in FP32 before being combined. This recovers accuracy statistically
    indistinguishable from an fp64-exact reference; a naive, unsplit
    single-pass FP16 rotation was measured to flip roughly 1% of INT8
    output codes relative to fp64-exact, which is not acceptable given
    GROUP_SIZE==256 exists specifically to improve INT8 quality. The
    Hadamard-factor operand (`h16`) never needs splitting because every one
    of its entries is exactly ±0.25 (exactly representable in FP16).

    x_tile: [SUBCHUNK, 256] fp32.
    h16: [16, 16] fp16, from `_generate_h16_fp16()`.
    Returns: [SUBCHUNK, 256] fp32, rotated.
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


# ─── Kernel 1: rotate + quantize activations (persistent, grid-stride) ────

@triton.jit
def kernel1_convrot_quant(
    X_ptr,
    X_q_ptr,
    X_scale_ptr,
    M,
    K,
    NUM_SMS,
    stride_xm,
    stride_xk,
    stride_xqm,
    stride_xqk,
    stride_xsm,
    stride_xsg,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    INPUT_FP16: tl.constexpr,
):
    """Rotate activation K-groups, quantize them, and write reusable workspace.

    Persistent, grid-stride kernel: launched with a grid of size
    ``min(NUM_SMS, num_pid_m * num_k_groups)``; each program loops over a
    strided share of ``(M-tile, K-group)`` work items so that wave
    quantization cannot leave SMs idle on small grids.

    ``BLOCK_M`` must be a multiple of 16 (the Tensor-Core rotation processes
    the tile in fixed 16-row sub-chunks — see ``SUBCHUNK`` below — so this is
    an implementation requirement of the rotation, not a GEMM-tile
    constraint). ``GROUP_SIZE`` must be 256; the rotation is only implemented
    for the 16x16 (x) 16x16 separable factorization.

    ``X_scale`` is written in TRANSPOSED ``[n_groups, M]`` layout
    (``stride_xsm`` is the per-row stride, ``stride_xsg`` is the per-group
    stride — pass ``stride_xsm=1, stride_xsg=M`` for a contiguous
    ``[n_groups, M]`` buffer) so that the store below, which holds
    ``group_idx`` fixed and varies ``rm`` across the tile, is coalesced.
    """
    tl.static_assert(GROUP_SIZE == 256, "Only GROUP_SIZE=256 is supported")
    tl.static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be a multiple of 16")
    SUBCHUNK: tl.constexpr = 16
    NUM_SUB: tl.constexpr = BLOCK_M // SUBCHUNK

    # Generate H_16 once per CTA by applying the H_4 butterfly to a 16x16
    # identity.  256 elements, 2 stages — negligible cost, no host buffer.
    h16 = _generate_h16_fp16()

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_k_groups = K // GROUP_SIZE
    num_tiles = num_pid_m * num_k_groups

    start_pid = tl.program_id(0)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        pid_m = (tile_id // num_k_groups) * BLOCK_M
        pid_k = (tile_id % num_k_groups) * GROUP_SIZE

        # Process the BLOCK_M-row tile in fixed 16-row sub-chunks: the
        # Tensor-Core rotation's `tl.dot` calls flatten each sub-chunk's
        # [SUBCHUNK, 16, 16] tile to [SUBCHUNK*16, 16], and keeping SUBCHUNK
        # fixed at 16 (rather than scaling it with BLOCK_M) is what keeps
        # this kernel's register/shared-memory footprint independent of
        # BLOCK_M, letting BLOCK_M be tuned purely for occupancy/tiling.
        for sub in tl.static_range(0, NUM_SUB):
            rm = pid_m + sub * SUBCHUNK + tl.arange(0, SUBCHUNK)
            rk = pid_k + tl.arange(0, GROUP_SIZE)
            ram = rm % M
            mask_m = rm < M
            mask = mask_m[:, None] & (rk[None, :] < K)

            x_ptr_block = X_ptr + ram[:, None] * stride_xm + rk[None, :] * stride_xk
            x_block = tl.load(x_ptr_block, mask=mask, other=0.0, eviction_policy="evict_first")
            if INPUT_FP16:
                x_block = x_block.to(tl.float32)

            rotated_x = _rotate_256_tensorcore(x_block, h16, SUBCHUNK)

            # One scale per row per K-group.
            block_max = tl.max(tl.abs(rotated_x), axis=1)
            scale = tl.maximum(block_max / 127.0, 1e-30)

            x_q = libdevice.rint(rotated_x / scale[:, None])
            x_q = tl.clamp(x_q, -127.0, 127.0).to(tl.int8)

            xq_ptr_block = X_q_ptr + rm[:, None] * stride_xqm + rk[None, :] * stride_xqk
            tl.store(xq_ptr_block, x_q, mask=mask)

            group_idx = pid_k // GROUP_SIZE
            # TRANSPOSED [n_groups, M] addressing: group_idx * stride_xsg is
            # the fixed per-group base offset, rm * stride_xsm walks
            # contiguously within that group's row when stride_xsm == 1.
            xs_ptr_block = X_scale_ptr + group_idx * stride_xsg + rm * stride_xsm
            # Store scale as FP32 (not FP16) — FP16 storage caused measurable
            # quality reduction vs the ggml/non-Triton baselines. The
            # per-group scale is applied to every element of the INT32
            # partial accumulator, so even small FP16 rounding errors
            # compound across K_groups.
            tl.store(xs_ptr_block, scale, mask=mask_m)


# ─── Kernel 2: INT8 GEMM + per-group dequant (persistent, grid-stride) ────

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
    NUM_SMS,
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
    GROUP_SIZE: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,
):
    """Consume quantized activations and accumulate dequantized FP32 output.

    Persistent, grid-stride kernel (see kernel 1's docstring for the
    rationale) sized to ``min(NUM_SMS, num_pid_m * num_pid_n)``.

    Includes L2-cache-friendly program swizzle (super-grouping) and eviction
    policy hints. GROUP_M controls how many M-tiles are grouped together in
    the launch order — programs within a group share X_q L2 residency.

    ``BLOCK_K`` (the INT8 `tl.dot` MMA tile width) is decoupled from
    ``GROUP_SIZE`` (the quantization group width, fixed at 256): the K loop
    walks one 256-wide quantization group at a time, and for each group
    accumulates INT32 partial products across ``GROUP_SIZE // BLOCK_K``
    narrower MMA sub-tiles before applying that group's FP32 dequant scale
    once. This amortizes the per-group scale load and FP32 rescale FMA over
    a wider swath of INT8 GEMM work than a design that rescaled once per
    (narrower) MMA tile, while leaving the proven-good MMA tile shapes
    (e.g. BLOCK_K=64) exactly as tunable as before.

    ``X_scale`` is read in the same TRANSPOSED ``[n_groups, M]`` layout
    kernel 1 writes (see kernel 1's docstring) for coalesced access.
    """
    tl.static_assert(GROUP_SIZE == 256, "Only GROUP_SIZE=256 is supported")
    tl.static_assert(GROUP_SIZE % BLOCK_K == 0, "GROUP_SIZE must be a multiple of BLOCK_K")
    GROUPS_PER_TILE: tl.constexpr = GROUP_SIZE // BLOCK_K

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n

    start_pid = tl.program_id(0)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        # L2 swizzle: super-group M-tiles so adjacent tile_ids share X_q L2
        # residency. This is the canonical Triton matmul tutorial swizzle
        # (03-matrix-multiplication), applied to the persistent tile index
        # instead of directly to `program_id`.
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        pid_m_off = pid_m * BLOCK_M
        pid_n_off = pid_n * BLOCK_N

        rm = pid_m_off + tl.arange(0, BLOCK_M)
        rn = pid_n_off + tl.arange(0, BLOCK_N)
        ram = rm % M
        rbn = rn % N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # W_scale is per-channel [N] — load once, reuse across all K-groups.
        ws = tl.load(W_scale_ptr + rn, mask=rn < N, other=0.0)

        for k_group_start in tl.range(0, K, GROUP_SIZE):
            group_idx = k_group_start // GROUP_SIZE
            xs_ptr_block = X_scale_ptr + group_idx * stride_xsg + ram * stride_xsm
            # X_scale is FP32 — no upcast needed.
            xs = tl.load(xs_ptr_block, mask=rm < M, other=0.0)

            int32_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for sub in tl.static_range(0, GROUPS_PER_TILE):
                k_offset = k_group_start + sub * BLOCK_K
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

                # INT8 × INT8 → INT32 MMA, fused-accumulate into int32_acc.
                int32_acc += tl.dot(xq, tl.trans(wq), out_dtype=tl.int32)

            # Dequantize once per 256-wide group (not once per BLOCK_K
            # sub-tile): GROUPS_PER_TILE fewer X_scale loads and FP32
            # rescale FMAs per output tile than rescaling at MMA-tile
            # granularity would need.
            acc += int32_acc.to(tl.float32) * xs[:, None] * ws[None, :]

        if HAS_BIAS:
            bias = tl.load(Bias_ptr + rn, mask=rn < N, other=0.0)
            acc += bias[None, :]

        y_ptr_block = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
        mask_y = (rm[:, None] < M) & (rn[None, :] < N)
        if OUTPUT_FP16:
            tl.store(y_ptr_block, acc.to(tl.float16), mask=mask_y)
        else:
            tl.store(y_ptr_block, acc.to(tl.float32), mask=mask_y)
