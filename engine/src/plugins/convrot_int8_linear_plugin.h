/*
 * convrot_int8_linear_plugin.h — HOT-Step ConvRotInt8Linear TensorRT 11 plugin.
 *
 * Fuses the entire w8a8 + ConvRot path into a single TensorRT custom op,
 * backed by an AOT-compiled Triton kernel launched via the CUDA Driver API:
 *
 *   1. Online activation rotation via in-register H_4 Kronecker butterfly
 *      (H_{GROUP_SIZE} = H_4 ⊗ H_4 ⊗ ... — no H matrix in shared memory)
 *   2. Per-row dynamic INT8 quantization of rotated x
 *   3. INT8 × INT8 → INT32 matmul on Tensor Cores (Triton tl.dot)
 *   4. Dequantize via per-row activation scale and per-output-channel
 *      weight scale
 *   5. Add bias (if present) and write FP16 by default
 *
 * Phase 1 architecture (cuBLASLt-free, butterfly rotation):
 *   - Single Triton kernel `fused_convrot_gemm_rowwise_kernel` AOT-compiled
 *     by tools/onnx-export/compile_triton_plugin.py into cubin variants keyed
 *     by (tile, group_size, bias) and embedded as C++ byte arrays in
 *     engine/src/plugins/assets/convrot_int8_kernel_cubin.h.
 *   - Loaded at runtime via cuModuleLoadData / cuModuleGetFunction.
 *   - Per-block shared-memory carveout opted in via cuFuncSetAttribute so
 *     Triton's multi-stage pipelined GEMM tiles can use >48 KB of static
 *     shared memory without cuLaunchKernel returning CUDA_ERROR_INVALID_VALUE.
 *   - Rotation uses NO H matrix pointer — the butterfly is entirely
 *     in-register. This avoids the 128 KB shared-memory cost of staging
 *     H_{256} as a dense matrix (which exceeds the sm_86 99 KB limit).
 *
 * Op signature (ONNX custom op, domain "hotstep", type "ConvRotInt8Linear"):
 *
 *   inputs[0]: x              FP16 [..., in_features] by default
 *   inputs[1]: weight_q       INT8 [out_features, in_features]
 *   inputs[2]: weight_scale   FP32 [out_features]
 *   inputs[3]: bias           FP32 [out_features]  (optional; 1D tensors stay FP32)
 *
 *   output[0]: y              FP16 [..., out_features] by default
 *
 * Attributes: group_size, in_features, out_features, has_bias,
 *             input_dtype={FP16,FP32}, output_dtype={FP16,FP32},
 *             preferred_format={LINEAR,HWC8,CHW32,HWC16}
 *
 * group_size contract: 0 (no rotation) or any power of 4 (4, 16, 64, 256, 1024).
 * The butterfly kernel supports all of these uniformly. The default export
 * uses group_size=256 (per tools/onnx-export/convrot.py).
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef HOT_STEP_TRT

#include <cuda.h>          // CUmodule / CUfunction (opaque as void* below)
#include "NvInfer.h"
#include "NvInferPluginBase.h"
#include "NvInferRuntimePlugin.h"

namespace hotstep {

constexpr char const* const kCONVROT_INT8_LINEAR_PLUGIN_NAME    = "ConvRotInt8Linear";
constexpr char const* const kCONVROT_INT8_LINEAR_PLUGIN_VERSION = "2";
constexpr char const* const kCONVROT_INT8_LINEAR_PLUGIN_NAMESPACE = "hotstep";

constexpr char const* const kFIELD_GROUP_SIZE   = "group_size";
constexpr char const* const kFIELD_IN_FEATURES  = "in_features";
constexpr char const* const kFIELD_OUT_FEATURES = "out_features";
constexpr char const* const kFIELD_HAS_BIAS     = "has_bias";
constexpr char const* const kFIELD_INPUT_DTYPE  = "input_dtype";
constexpr char const* const kFIELD_OUTPUT_DTYPE = "output_dtype";
constexpr char const* const kFIELD_INPUT_DTYPE_ID  = "input_dtype_id";
constexpr char const* const kFIELD_OUTPUT_DTYPE_ID = "output_dtype_id";
constexpr char const* const kFIELD_PREFERRED_FORMAT = "preferred_format";

/*
 * ConvRotInt8LinearPlugin — IPluginV3 implementation around an AOT-compiled
 * Triton fused kernel launched via the CUDA Driver API.
 *
 * Build phase flow:
 *   1. configurePlugin() → store M/K/N from dynamic shape descriptors
 *   2. TRT allocates plugin workspace reported by getWorkspaceSize()
 *      (returns 0 in Phase 1 — the Triton kernel uses no plugin workspace).
 *
 * Runtime phase flow:
 *   1. onShapeChange() → cuModuleLoadData the right cubin variant for
 *      (group_size, has_bias) in the current sm86-first inventory,
 *      cuModuleGetFunction for `fused_convrot_gemm_rowwise_kernel`, and
 *      cuFuncSetAttribute to opt in to large per-block shared memory.
 *   2. enqueue() → build the 14-arg kernel param block and cuLaunchKernel.
 */
class ConvRotInt8LinearPlugin
    : public nvinfer1::IPluginV3,
      public nvinfer1::IPluginV3OneCore,
      public nvinfer1::IPluginV3OneBuild,
      public nvinfer1::IPluginV3OneRuntime {
public:
    ConvRotInt8LinearPlugin(int32_t group_size, int32_t in_features,
                            int32_t out_features, int32_t has_bias,
                            int32_t input_dtype_id = 10,
                            int32_t output_dtype_id = 10,
                            std::string preferred_format = "HWC8");
    ConvRotInt8LinearPlugin(void const* data, size_t length);
    ~ConvRotInt8LinearPlugin() override;

    // ── IPluginV3 ────────────────────────────────────────────────────
    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;

    // ── IPluginV3OneCore ─────────────────────────────────────────────
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    // Note: setPluginNamespace() was removed in TRT 11. The namespace is
    // set via the creator registration and getPluginNamespace() returns
    // the stored value.

    // ── IPluginV3OneBuild ────────────────────────────────────────────
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
                               nvinfer1::DataType const* inputTypes,
                               int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
                            nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                            nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
                             nvinfer1::DynamicPluginTensorDesc const* out,
                             int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs,
                            int32_t nbInputs,
                            nvinfer1::DynamicPluginTensorDesc const* outputs,
                            int32_t nbOutputs) const noexcept override;
    // Note: destroy() was removed in TRT 11. The destructor handles cleanup.

    // ── Custom Tactics (sm86-first: only one custom tactic, plus implicit default 0) ────
    int32_t getNbTactics() noexcept override;
    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override;
    char const* getTimingCacheID() noexcept override;
    int32_t getFormatCombinationLimit() noexcept override;
    char const* getMetadataString() noexcept override;

    // ── IPluginV3OneRuntime ──────────────────────────────────────────
    int32_t setTactic(int32_t tactic) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
                          nvinfer1::PluginTensorDesc const* out,
                          int32_t nbOutputs) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                    nvinfer1::PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;  // TRT 11: no const
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    // TRT 11 new pure virtual: attachToContext. Returns a clone attached to
    // the given resource context. For our simple plugin, just return clone().
    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* ctx) noexcept override;

    // ── Serialization ────────────────────────────────────────────────
    size_t getSerializationSize() const noexcept;
    void serialize(void* buffer) const noexcept;

private:
    // Build-time params (serialized)
    int32_t m_group_size{0};
    int32_t m_in_features{0};
    int32_t m_out_features{0};
    int32_t m_has_bias{0};
    // ONNX TensorProto dtype ids: FLOAT=1, FLOAT16=10, BFLOAT16=16.
    int32_t m_input_dtype_id{10};
    int32_t m_output_dtype_id{10};
    std::string m_preferred_format{"HWC8"};
    std::string m_namespace{kCONVROT_INT8_LINEAR_PLUGIN_NAMESPACE};

    // Runtime state
    int32_t m_tactic{0};           // tactic requested by TRT (0 implicit default, 2 custom sm86 path)
    int32_t m_M{0};                 // rows (from onShapeChange)
    int32_t m_K{0};                 // in_features (from onShapeChange)
    int32_t m_N{0};                 // out_features (= m_out_features)

    // Triton / CUDA module state (lazily initialized in onShapeChange)
    void* m_module{nullptr};      // Cast to CUmodule in .cpp
    void* m_kernelFunc{nullptr};  // Cast to CUfunction in .cpp
    // No m_d_H — butterfly rotation uses no GPU-allocated Hadamard matrix.
    // Dynamic shared-memory requirement (bytes) of the loaded cubin, as
    // reported by Triton's compile metadata and embedded in the cubin header
    // as kCONVROT_INT8_KERNEL_*_SHARED. Passed to cuLaunchKernel as
    // sharedMemBytes. Critically, this is NOT the device's opt-in ceiling —
    // it is the kernel's actual requirement, so cuLaunchKernel allocates
    // exactly the right amount and the runtime check
    //   m_shared_bytes <= getDeviceMaxDynamicSharedMem()
    // is a real guard instead of a tautology.
    size_t m_shared_bytes{0};
    // Tile dimensions baked into the loaded cubin. Set by initTriton() from
    // the kCONVROT_INT8_KERNEL_*_BLOCK_M / _BLOCK_N constants in the cubin
    // header. The compile script may pick different BLOCK_M values per config
    // to maximize num_stages within the device's shared-memory limit (e.g.
    // G256 FP32IO uses BLOCK_M=64 with ns=2 instead of BLOCK_M=128 with ns=1
    // to preserve software pipelining). Used by enqueue() for grid math.
    int32_t m_block_m{128};
    int32_t m_block_n{128};

    // Mutable field collection for serialization
    mutable std::vector<nvinfer1::PluginField> m_fields;
    mutable nvinfer1::PluginFieldCollection m_fc{};

    // ── Internal helpers ─────────────────────────────────────────────
    bool initTriton();
    void destroyTriton();
};

/*
 * ConvRotInt8LinearPluginCreator — IPluginCreatorV3One factory.
 */
class ConvRotInt8LinearPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    ConvRotInt8LinearPluginCreator();
    ~ConvRotInt8LinearPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    // Note: setPluginNamespace() was removed in TRT 11.

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV3* createPlugin(char const* name,
                                      nvinfer1::PluginFieldCollection const* fc,
                                      nvinfer1::TensorRTPhase phase) noexcept override;

private:
    std::string m_namespace{kCONVROT_INT8_LINEAR_PLUGIN_NAMESPACE};
    std::vector<nvinfer1::PluginField> m_fields;
    nvinfer1::PluginFieldCollection m_fc{};
};

}  // namespace hotstep

#endif  // HOT_STEP_TRT
