/*
 * convrot_int8_linear_kernel.cuh — CUDA kernels for ConvRotInt8Linear.
 *
 * Only the ConvRot activation rotation + per-row INT8 quantization kernel
 * lives here. The INT8 GEMM is handled by cuBLASLt (called from the plugin's
 * enqueue method), and the dequant+bias epilogue is a separate pointwise
 * kernel.
 *
 * ConvRot rotation kernel
 * -----------------------
 * One block per row of x [K]. Each block:
 *   1. Applies the regular Hadamard as H4 Kronecker butterflies per group
 *      (the H input is kept for graph/API compatibility but is not staged)
 *   2. Computes per-row max-abs from the rotated values
 *   3. Computes per-row max-abs → scale = max(|x_rot|) / 127
 *   4. Recomputes the rotation and quantizes:
 *      x_q = clip(round(x_rot / scale), -127, 127)
 *
 * Outputs: x_q [M, K] INT8, x_scale [M] FP32
 *
 * Dequant+epilogue kernel
 * -----------------------
 * One block per output tile [BLOCK_M × BLOCK_N]. Each block:
 *   1. Reads INT32 accumulator from cuBLASLt output
 *   2. Loads x_scale[m] and w_scale[n]
 *   3. Computes y = acc * x_scale * w_scale + bias
 *   4. Writes FP32 output
 */

#pragma once

#ifdef HOT_STEP_TRT

#include <cuda_runtime.h>
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
template <int GROUP_SIZE, int BLOCK_K>
__global__ void convrot_activation_quant_kernel(
    float const* __restrict__ x_ptr,       // [M, K] FP32
    int8_t* __restrict__ x_q_ptr,          // [M, K] INT8 (output)
    float* __restrict__ x_scale_ptr,       // [M] FP32 (output)
    float const* __restrict__ H_ptr,       // [GROUP_SIZE, GROUP_SIZE] FP32
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
template <int BLOCK_M, int BLOCK_N, bool HAS_BIAS>
__global__ void dequant_bias_epilogue_kernel(
    int32_t const* __restrict__ acc_ptr,   // [M, N] INT32 (from cuBLASLt)
    float* __restrict__ y_ptr,             // [M, N] FP32 (output)
    float const* __restrict__ x_scale,     // [M] FP32
    float const* __restrict__ w_scale,     // [N] FP32
    float const* __restrict__ bias_ptr,    // [N] FP32 (or nullptr)
    int32_t M, int32_t N);

// ── Launch helpers ──────────────────────────────────────────────────

bool launch_convrot_activation_quant(
    float const* x, int8_t* x_q, float* x_scale, float const* H,
    int32_t M, int32_t K, int32_t group_size, cudaStream_t stream);

bool launch_dequant_bias_epilogue(
    int32_t const* acc, float* y,
    float const* x_scale, float const* w_scale, float const* bias,
    int32_t M, int32_t N, bool has_bias, cudaStream_t stream);

}  // namespace hotstep

#endif  // HOT_STEP_TRT
