/*
 * convrot_int8_linear_plugin.h — HOT-Step ConvRotInt8Linear TensorRT 11 plugin.
 *
 * This plugin exposes the ConvRot + W8A8 linear path as one TensorRT custom op.
 * The runtime uses the same two-kernel Triton pipeline for every M:
 *   1. Rotate and quantize activations once into reusable workspace.
 *   2. Reuse that INT8 workspace across all output-channel tiles.
 *
 * A previous specialized M==1 fused kernel was removed after profiling showed
 * the two-kernel BK64/BM128/BN128 path is faster for the M==1 workload too.
 *
 * The Hadamard/ConvRot transform itself is still implemented as in-register
 * H_4 Kronecker butterflies. No dense H matrix is staged in shared memory and
 * no H tensor is passed through the plugin boundary.
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

constexpr char const* const kCONVROT_INT8_LINEAR_PLUGIN_NAME = "ConvRotInt8Linear";
constexpr char const* const kCONVROT_INT8_LINEAR_PLUGIN_VERSION = "2";
constexpr char const* const kCONVROT_INT8_LINEAR_PLUGIN_NAMESPACE = "hotstep";

constexpr char const* const kFIELD_GROUP_SIZE = "group_size";
constexpr char const* const kFIELD_IN_FEATURES = "in_features";
constexpr char const* const kFIELD_OUT_FEATURES = "out_features";
constexpr char const* const kFIELD_HAS_BIAS = "has_bias";
constexpr char const* const kFIELD_INPUT_DTYPE = "input_dtype";
constexpr char const* const kFIELD_OUTPUT_DTYPE = "output_dtype";
constexpr char const* const kFIELD_INPUT_DTYPE_ID = "input_dtype_id";
constexpr char const* const kFIELD_OUTPUT_DTYPE_ID = "output_dtype_id";
constexpr char const* const kFIELD_PREFERRED_FORMAT = "preferred_format";

/*
 * ConvRotInt8LinearPlugin — IPluginV3 wrapper around AOT-compiled Triton
 * kernels launched through the CUDA Driver API.
 *
 * Build phase:
 *   * configurePlugin() validates the profile bounds and eagerly checks that
 *     the requested Triton cubins exist.
 *   * getWorkspaceSize() sizes the reusable X_q / X_scale workspace from the
 *     profile's MAX dimensions.
 *
 * Runtime phase:
 *   * onShapeChange() caches the concrete M/K/N and ensures the two Triton
 *     kernels are loaded for the selected dtype/bias configuration.
 *   * enqueue() launches the two-kernel pipeline on TensorRT's CUDA stream.
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

    // ── Custom Tactics ───────────────────────────────────────────────
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
    nvinfer1::IPluginV3* clone() noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;
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

    // Runtime shape and tactic state
    int32_t m_tactic{0};
    int32_t m_M{0};
    int32_t m_K{0};
    int32_t m_N{0};


    // M>1 kernel 1: activation rotation + quantization
    void* m_module_quant{nullptr};
    void* m_kernelFunc_quant{nullptr};
    size_t m_shared_bytes_quant{0};
    int32_t m_block_m_quant{128};
    int32_t m_block_k_quant{64};

    // M>1 kernel 2: INT8 GEMM + per-group dequant
    void* m_module_gemm{nullptr};
    void* m_kernelFunc_gemm{nullptr};
    size_t m_shared_bytes_gemm{0};
    int32_t m_block_m_gemm{128};
    int32_t m_block_n_gemm{128};

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
