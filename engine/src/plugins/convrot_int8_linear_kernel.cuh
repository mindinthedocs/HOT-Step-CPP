/*
 * convrot_int8_linear_kernel.cuh — CUDA kernels for ConvRotInt8Linear (v13).
 *
 * Only the ConvRot activation rotation + per-row INT8 quantization
 * launch helper lives here.  The actual kernels are AOT-compiled Triton/Gluon
 * cubins (see extract_jit_cubins_autotune.py); this header is a compatibility
 * shim for the legacy launcher API.
 *
 * v13 contract:
 *   * K1 rotates activations with H256 in registers and computes per-row INT8
 *     quantization.  X_scale is [M] FP32 (one scale per activation row, NOT
 *     per 256-block).
 *   * K2 is a plain INT8×INT8 → INT32 GEMM with a per-row X_scale × per-row
 *     W_scale outer-product FP32 dequant in the epilogue.  The per-group
 *     INT32 fold of v7-v12 is gone.
 *
 * ConvRot rotation kernel (v13)
 * -----------------------------
 * One CTA per BLOCK_M rows of x [K]. Each block:
 *   1. For each group g in [0, G): loads [BLOCK_M, GROUP_SIZE] FP16 input,
 *      converts to FP32, applies 4 H4 Kronecker stages (H256),
 *      accumulates max-abs per row.
 *   2. Computes per-row max-abs scale (one FP32 per M row).
 *   3. Quantizes: x_q = clip(round(x_rot / scale), -127, 127).
 *
 * Outputs: x_q [M, K] INT8, x_scale [M] FP32  (per-row, NOT per-group)
 *
 * INT8 GEMM + per-row dequant kernel (v13)
 * ----------------------------------------
 * One block per output tile [BLOCK_M × BLOCK_N]. Each block:
 *   1. Accumulates a single full-K INT32 sum: int32_acc = sum_k X_q[m,k] * W_q[n,k]
 *   2. Loads x_scale[m] (per-row FP32) and w_scale[n] (per-row FP32).
 *   3. Computes y = int32_acc * x_scale * w_scale + bias  (FP32 epilogue).
 *   4. Writes FP16 output by default (or FP32 for explicit fallback).
 */

#pragma once

#ifdef HOT_STEP_TRT

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace hotstep {

/*
 * convrot_activation_quant_kernel — fused ConvRot rotation + per-row INT8 quant.
 *
 * Template parameters:
 *   GROUP_SIZE   - ConvRot block size (must divide K)
 *   BLOCK_K      - threads per block (must be >= GROUP_SIZE, power of 2)
 *
 * Grid: (M,)
 * Shared memory: 2 * GROUP_SIZE * sizeof(float)
 */
template <typename X_T, int GROUP_SIZE, int BLOCK_K>
__global__ void convrot_activation_quant_kernel(
    X_T const* __restrict__ x_ptr,         // [M, K] FP16/FP32
    int8_t* __restrict__ x_q_ptr,          // [M, K] INT8 (output)
    float* __restrict__ x_scale_ptr,       // [M] FP32 (output)
    int32_t M, int32_t K);

/*
 * dequant_bias_epilogue_kernel — fused dequant + bias from INT32 GEMM output.
 *
 * Template parameters:
 *   BLOCK_M, BLOCK_N - tile size for the output
 *   HAS_BIAS          - 1 if bias_ptr is valid
 *
 * Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
 */
template <typename Y_T, typename BIAS_T, int BLOCK_M, int BLOCK_N, bool HAS_BIAS>
__global__ void dequant_bias_epilogue_kernel(
    int32_t const* __restrict__ acc_ptr,   // [M, N] INT32 (from cuBLASLt)
    Y_T* __restrict__ y_ptr,               // [M, N] FP16/FP32 (output)
    float const* __restrict__ x_scale,     // [M] FP32
    float const* __restrict__ w_scale,     // [N] FP32
    BIAS_T const* __restrict__ bias_ptr,   // [N] FP16/FP32 (or nullptr)
    int32_t M, int32_t N);

// ── Launch helpers ──────────────────────────────────────────────────

bool launch_convrot_activation_quant(
    void const* x, int8_t* x_q, float* x_scale,
    int32_t M, int32_t K, int32_t group_size, int32_t input_dtype,
    cudaStream_t stream);

bool launch_dequant_bias_epilogue(
    int32_t const* acc, void* y,
    float const* x_scale, float const* w_scale, void const* bias,
    int32_t M, int32_t N, bool has_bias, int32_t bias_dtype,
    int32_t output_dtype, cudaStream_t stream);

}  // namespace hotstep

#endif  // HOT_STEP_TRT
