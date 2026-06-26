/*
 * convrot_int8_linear_kernel.cu — CUDA implementation of ConvRot + quant + epilogue.
 *
 * The INT8 GEMM is NOT in this file — it's handled by cuBLASLt in the plugin's
 * enqueue() method. This file only contains:
 *
 *   1. convrot_activation_quant_kernel — ConvRot rotation + per-row INT8 quant
 *   2. dequant_bias_epilogue_kernel — INT32 → FP32 dequant + bias add
 *
 * Ported from the ComfyUI-INT8-Fast Triton kernel (convrot.rotate_activation +
 * int8_fused_kernel._quantize_rowwise_kernel). The regular Hadamard matrix is
 * applied as H4 Kronecker butterflies, so the kernel does not need to stage a
 * dense group_size x group_size matrix in shared memory.
 */

#ifdef HOT_STEP_TRT

#include "convrot_int8_linear_kernel.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

namespace hotstep {

namespace {

template <typename T>
__device__ __forceinline__ float load_to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float load_to_float<__half>(__half v) {
    return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T store_from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ __half store_from_float<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ float load_to_float<uint16_t>(uint16_t v) {
    return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

template <>
__device__ __forceinline__ uint16_t store_from_float<uint16_t>(float v) {
    uint32_t bits = __float_as_uint(v);
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────
// ConvRot activation rotation + per-row INT8 quantization
// ──────────────────────────────────────────────────────────────────────────

template <int GROUP_SIZE>
__device__ __forceinline__ void regular_hadamard_transform(float* vals, float* tmp)
{
    int32_t tid = threadIdx.x;
    for (int32_t stride = 1; stride < GROUP_SIZE; stride *= 4) {
        int32_t span = stride * 4;
        int32_t combos = GROUP_SIZE / 4;
        for (int32_t c = tid; c < combos; c += blockDim.x) {
            int32_t base = (c / stride) * span + (c % stride);
            float x0 = vals[base];
            float x1 = vals[base + stride];
            float x2 = vals[base + 2 * stride];
            float x3 = vals[base + 3 * stride];
            tmp[base]              = 0.5f * ( x0 + x1 + x2 - x3);
            tmp[base + stride]     = 0.5f * ( x0 + x1 - x2 + x3);
            tmp[base + 2 * stride] = 0.5f * ( x0 - x1 + x2 + x3);
            tmp[base + 3 * stride] = 0.5f * (-x0 + x1 + x2 + x3);
        }
        __syncthreads();
        for (int32_t i = tid; i < GROUP_SIZE; i += blockDim.x) {
            vals[i] = tmp[i];
        }
        __syncthreads();
    }
}

template <typename X_T, int GROUP_SIZE, int BLOCK_K>
__global__ void convrot_activation_quant_kernel(
    X_T const* __restrict__ x_ptr,
    int8_t* __restrict__ x_q_ptr,
    float* __restrict__ x_scale_ptr,
    int32_t M, int32_t K)
{
    int32_t row = blockIdx.x;
    if (row >= M) return;
    int32_t tid = threadIdx.x;

    X_T const* x_row = x_ptr + (size_t)row * K;
    int8_t* xq_row = x_q_ptr + (size_t)row * K;

    extern __shared__ float smem[];
    float* s_x = smem;                       // [GROUP_SIZE]
    float* s_tmp = smem + GROUP_SIZE;        // [GROUP_SIZE]

    int32_t n_groups = K / GROUP_SIZE;
    float thread_max = 0.0f;

    // Phase 1: rotate each group and reduce max abs.
    for (int32_t g = 0; g < n_groups; ++g) {
        int32_t base = g * GROUP_SIZE;
        for (int32_t i = tid; i < GROUP_SIZE; i += blockDim.x) {
            s_x[i] = load_to_float(x_row[base + i]);
        }
        __syncthreads();

        regular_hadamard_transform<GROUP_SIZE>(s_x, s_tmp);
        for (int32_t i = tid; i < GROUP_SIZE; i += blockDim.x) {
            thread_max = fmaxf(thread_max, fabsf(s_x[i]));
        }
        __syncthreads();
    }

    // Tree reduce within the block
    __shared__ float s_reduce[32];
    int32_t lane = tid & 31;
    int32_t warp = tid >> 5;
    for (int32_t off = 16; off > 0; off >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, off));
    }
    if (lane == 0) s_reduce[warp] = thread_max;
    __syncthreads();
    int32_t num_warps = (BLOCK_K + 31) / 32;
    if (warp == 0) {
        thread_max = (lane < num_warps) ? s_reduce[lane] : 0.0f;
        for (int32_t off = 16; off > 0; off >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, off));
        }
        if (lane == 0) {
            float scale = fmaxf(thread_max / 127.0f, 1e-30f);
            x_scale_ptr[row] = scale;
            s_reduce[0] = scale;
        }
    }
    __syncthreads();
    float scale = s_reduce[0];
    float inv_scale = 1.0f / scale;

    // Phase 2: recompute the same rotation, quantize into shared memory,
    // and perform vectorized, coalesced 32-bit writes to global memory.
    for (int32_t g = 0; g < n_groups; ++g) {
        int32_t base = g * GROUP_SIZE;
        for (int32_t i = tid; i < GROUP_SIZE; i += blockDim.x) {
            s_x[i] = load_to_float(x_row[base + i]);
        }
        __syncthreads();

        regular_hadamard_transform<GROUP_SIZE>(s_x, s_tmp);
        
        int8_t* s_q = reinterpret_cast<int8_t*>(s_tmp);
        for (int32_t i = tid; i < GROUP_SIZE; i += blockDim.x) {
            float v = s_x[i];
            float q = rintf(v * inv_scale);
            q = fmaxf(-127.0f, fminf(127.0f, q));
            s_q[i] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // Perform vectorized coalesced 32-bit (4-byte) writes to global memory.
        int32_t const* s_q_int = reinterpret_cast<int32_t const*>(s_q);
        int32_t* xq_row_int = reinterpret_cast<int32_t*>(xq_row + base);
        int32_t num_ints = GROUP_SIZE / 4;
        for (int32_t i = tid; i < num_ints; i += blockDim.x) {
            xq_row_int[i] = s_q_int[i];
        }
        __syncthreads();
    }
}

// ──────────────────────────────────────────────────────────────────────────
// No-rotation activation quant (group_size == 0)
// ──────────────────────────────────────────────────────────────────────────
// Used when in_features % group_size != 0 — the ConvRot rotation is skipped
// and the activation is directly per-row INT8 quantized. Same quantization
// logic as the rotation kernel, just without the H-matrix rotation phase.

template <typename X_T, int BLOCK_K>
__global__ void activation_quant_norot_kernel(
    X_T const* __restrict__ x_ptr,
    int8_t* __restrict__ x_q_ptr,
    float* __restrict__ x_scale_ptr,
    int32_t M, int32_t K)
{
    int32_t row = blockIdx.x;
    if (row >= M) return;
    int32_t tid = threadIdx.x;

    X_T const* x_row = x_ptr + (size_t)row * K;
    int8_t* xq_row = x_q_ptr + (size_t)row * K;

    // Phase 1: per-row max-abs reduction (tiled over K).
    float thread_max = 0.0f;
    for (int32_t tile = 0; tile < K; tile += BLOCK_K) {
        for (int32_t i = tid; i < BLOCK_K && tile + i < K; i += BLOCK_K) {
            float v = fabsf(load_to_float(x_row[tile + i]));
            thread_max = fmaxf(thread_max, v);
        }
    }

    // Tree reduce within the block
    __shared__ float s_reduce[32];
    int32_t lane = tid & 31;
    int32_t warp = tid >> 5;
    for (int32_t off = 16; off > 0; off >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, off));
    }
    if (lane == 0) s_reduce[warp] = thread_max;
    __syncthreads();
    int32_t num_warps = (BLOCK_K + 31) / 32;
    if (warp == 0) {
        thread_max = (lane < num_warps) ? s_reduce[lane] : 0.0f;
        for (int32_t off = 16; off > 0; off >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, off));
        }
        if (lane == 0) {
            float scale = fmaxf(thread_max / 127.0f, 1e-30f);
            x_scale_ptr[row] = scale;
            s_reduce[0] = scale;
        }
    }
    __syncthreads();
    float scale = s_reduce[0];

    // Phase 2: quantize and store.
    for (int32_t tile = 0; tile < K; tile += BLOCK_K) {
        for (int32_t i = tid; i < BLOCK_K && tile + i < K; i += BLOCK_K) {
            float v = load_to_float(x_row[tile + i]);
            float q = rintf(v / scale);
            q = fmaxf(-127.0f, fminf(127.0f, q));
            xq_row[tile + i] = static_cast<int8_t>(q);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Dequant + bias epilogue (applied to cuBLASLt INT32 output)
// ──────────────────────────────────────────────────────────────────────────

template <typename Y_T, typename BIAS_T, int BLOCK_M, int BLOCK_N, bool HAS_BIAS>
__global__ void dequant_bias_epilogue_kernel(
    int32_t const* __restrict__ acc_ptr,
    Y_T* __restrict__ y_ptr,
    float const* __restrict__ x_scale,
    float const* __restrict__ w_scale,
    BIAS_T const* __restrict__ bias_ptr,
    int32_t M, int32_t N)
{
    int32_t bm = blockIdx.x * BLOCK_M;
    int32_t bn = blockIdx.y * BLOCK_N;
    int32_t tm = threadIdx.y;
    int32_t tn = threadIdx.x;

    int32_t m = bm + tm;
    int32_t n = bn + tn;
    if (m >= M || n >= N) return;

    int32_t acc = acc_ptr[m * N + n];
    float xs = x_scale[m];
    float ws = w_scale[n];
    float v = static_cast<float>(acc) * xs * ws;
    if (HAS_BIAS) {
        v += load_to_float(bias_ptr[n]);
    }
    y_ptr[m * N + n] = store_from_float<Y_T>(v);
}

// ──────────────────────────────────────────────────────────────────────────
// Launch helpers
// ──────────────────────────────────────────────────────────────────────────

template <typename X_T>
bool launch_convrot_activation_quant_typed(
    X_T const* x, int8_t* x_q, float* x_scale,
    int32_t M, int32_t K, int32_t group_size, cudaStream_t stream)
{
    if (group_size == 0) {
        dim3 grid(M);
        dim3 block(256);
        activation_quant_norot_kernel<X_T, 256>
            <<<grid, block, 0, stream>>>(x, x_q, x_scale, M, K);
        return cudaGetLastError() == cudaSuccess;
    }
    if (K % group_size != 0) return false;

    int32_t smem_bytes = 2 * group_size * sizeof(float);
    dim3 grid(M);
    dim3 block(256);
    switch (group_size) {
        case 4:
            convrot_activation_quant_kernel<X_T, 4, 256>
                <<<grid, block, smem_bytes, stream>>>(x, x_q, x_scale, M, K);
            break;
        case 16:
            convrot_activation_quant_kernel<X_T, 16, 256>
                <<<grid, block, smem_bytes, stream>>>(x, x_q, x_scale, M, K);
            break;
        case 64:
            convrot_activation_quant_kernel<X_T, 64, 256>
                <<<grid, block, smem_bytes, stream>>>(x, x_q, x_scale, M, K);
            break;
        case 256:
            convrot_activation_quant_kernel<X_T, 256, 256>
                <<<grid, block, smem_bytes, stream>>>(x, x_q, x_scale, M, K);
            break;
        case 1024:
            convrot_activation_quant_kernel<X_T, 1024, 256>
                <<<grid, block, smem_bytes, stream>>>(x, x_q, x_scale, M, K);
            break;
        default:
            return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

bool launch_convrot_activation_quant(
    void const* x, int8_t* x_q, float* x_scale,
    int32_t M, int32_t K, int32_t group_size, int32_t input_dtype,
    cudaStream_t stream)
{
    if (input_dtype == 1) {
        return launch_convrot_activation_quant_typed(
            static_cast<__half const*>(x), x_q, x_scale, M, K, group_size, stream);
    }
    if (input_dtype == 2) {
        return launch_convrot_activation_quant_typed(
            static_cast<uint16_t const*>(x), x_q, x_scale, M, K, group_size, stream);
    }
    return launch_convrot_activation_quant_typed(
        static_cast<float const*>(x), x_q, x_scale, M, K, group_size, stream);
}

template <typename Y_T, typename BIAS_T>
bool launch_dequant_bias_epilogue_typed(
    int32_t const* acc, void* y,
    float const* x_scale, float const* w_scale, void const* bias,
    int32_t M, int32_t N, bool has_bias, cudaStream_t stream)
{
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(BLOCK_M, BLOCK_N);
    auto* y_typed = static_cast<Y_T*>(y);
    auto const* bias_typed = static_cast<BIAS_T const*>(bias);

    if (has_bias) {
        dequant_bias_epilogue_kernel<Y_T, BIAS_T, BLOCK_M, BLOCK_N, true>
            <<<grid, block, 0, stream>>>(acc, y_typed, x_scale, w_scale, bias_typed, M, N);
    } else {
        dequant_bias_epilogue_kernel<Y_T, BIAS_T, BLOCK_M, BLOCK_N, false>
            <<<grid, block, 0, stream>>>(acc, y_typed, x_scale, w_scale, bias_typed, M, N);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool launch_dequant_bias_epilogue(
    int32_t const* acc, void* y,
    float const* x_scale, float const* w_scale, void const* bias,
    int32_t M, int32_t N, bool has_bias, int32_t bias_dtype,
    int32_t output_dtype, cudaStream_t stream)
{
    if (output_dtype == 1 && bias_dtype == 1) {
        return launch_dequant_bias_epilogue_typed<__half, __half>(acc, y, x_scale, w_scale, bias, M, N, has_bias, stream);
    }
    if (output_dtype == 1 && bias_dtype == 2) {
        return launch_dequant_bias_epilogue_typed<__half, uint16_t>(acc, y, x_scale, w_scale, bias, M, N, has_bias, stream);
    }
    if (output_dtype == 1) {
        return launch_dequant_bias_epilogue_typed<__half, float>(acc, y, x_scale, w_scale, bias, M, N, has_bias, stream);
    }
    if (output_dtype == 2 && bias_dtype == 1) {
        return launch_dequant_bias_epilogue_typed<uint16_t, __half>(acc, y, x_scale, w_scale, bias, M, N, has_bias, stream);
    }
    if (output_dtype == 2 && bias_dtype == 2) {
        return launch_dequant_bias_epilogue_typed<uint16_t, uint16_t>(acc, y, x_scale, w_scale, bias, M, N, has_bias, stream);
    }
    if (output_dtype == 2) {
        return launch_dequant_bias_epilogue_typed<uint16_t, float>(acc, y, x_scale, w_scale, bias, M, N, has_bias, stream);
    }
    if (bias_dtype == 1) {
        return launch_dequant_bias_epilogue_typed<float, __half>(acc, y, x_scale, w_scale, bias, M, N, has_bias, stream);
    }
    if (bias_dtype == 2) {
        return launch_dequant_bias_epilogue_typed<float, uint16_t>(acc, y, x_scale, w_scale, bias, M, N, has_bias, stream);
    }
    return launch_dequant_bias_epilogue_typed<float, float>(acc, y, x_scale, w_scale, bias, M, N, has_bias, stream);
}
}  // namespace hotstep

#endif  // HOT_STEP_TRT
