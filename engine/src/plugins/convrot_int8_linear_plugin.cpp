/*
 * convrot_int8_linear_plugin.cpp — IPluginV3 implementation for ConvRotInt8Linear.
 *
 * Architecture (same as TRT-LLM's smooth_quant_gemm_plugin):
 *
 *   enqueue() pipeline:
 *     1. ConvRot activation rotation + per-row INT8 quant (custom CUDA kernel)
 *     2. INT8 × INT8 → INT32 matmul (cuBLASLt cublasLtMatmul)
 *     3. Dequant + bias epilogue (custom CUDA kernel; FP16 output by default)
 *
 * cuBLASLt setup:
 *   - Matmul descriptor: CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = CUBLAS_COMPUTE_32I
 *     (INT8 inputs, INT32 accumulation)
 *   - cuBLASLt layouts are column-major views over row-major buffers:
 *       A: w_q row-major [N, K] viewed as column-major [K, N], opA=T
 *       B: x_q row-major [M, K] viewed as column-major [K, M], opB=N
 *       C: acc row-major [M, N] viewed as column-major [N, M]
 *   - The GEMM computes C_cm = W_cm^T × X_cm, which is physically
 *     acc_rm = x_q × w_q^T.
 *
 * Build vs Runtime:
 *   - Build phase: the plugin has both build and runtime capabilities. TRT
 *     calls configurePlugin() with min/opt/max shapes and getWorkspaceSize().
 *   - Runtime phase: only runtime capability is needed. onShapeChange()
 *     re-creates the cuBLASLt descriptors for the actual runtime shapes,
 *     enqueue() runs the pipeline.
 */

#ifdef HOT_STEP_TRT

#include "convrot_int8_linear_plugin.h"
#include "convrot_int8_linear_kernel.cuh"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace hotstep {

namespace {

constexpr size_t kCUBLASLT_WORKSPACE_BYTES = 32ull * 1024ull * 1024ull;

size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

char* alignPtr(char* ptr, size_t alignment) {
    uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
    raw = (raw + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
    return reinterpret_cast<char*>(raw);
}

int64_t flattenedRows(nvinfer1::Dims const& dims) {
    if (dims.nbDims < 1) return 0;
    int64_t rows = 1;
    for (int32_t i = 0; i < dims.nbDims - 1; ++i) {
        if (dims.d[i] <= 0) return 0;
        rows *= dims.d[i];
    }
    return rows;
}

int32_t lastDim(nvinfer1::Dims const& dims) {
    if (dims.nbDims < 1) return 0;
    return dims.d[dims.nbDims - 1];
}

int32_t normalizeDtypeId(int32_t value, int32_t defaultValue = 10) {
    // ONNX TensorProto enum: FLOAT=1, FLOAT16=10, BFLOAT16=16.
    if (value == 1 || value == 10 || value == 16) return value;
    return defaultValue;
}

int32_t parseDtypeId(char const* value, int32_t defaultValue = 10) {
    if (value == nullptr || value[0] == '\0') return defaultValue;
    std::string s(value);
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (s == "FP16" || s == "FLOAT16" || s == "HALF") return 10;
    if (s == "BF16" || s == "BFLOAT16") return 16;
    if (s == "FP32" || s == "FLOAT" || s == "FLOAT32") return 1;
    return defaultValue;
}

nvinfer1::DataType trtTypeFromDtypeId(int32_t dtypeId) {
    if (dtypeId == 10) return nvinfer1::DataType::kHALF;
    if (dtypeId == 16) return nvinfer1::DataType::kBF16;
    return nvinfer1::DataType::kFLOAT;
}

int32_t kernelDtypeFromTrt(nvinfer1::DataType dtype) {
    if (dtype == nvinfer1::DataType::kHALF) return 1;
    if (dtype == nvinfer1::DataType::kBF16) return 2;
    return 0;
}

char const* dtypeNameFromId(int32_t dtypeId) {
    if (dtypeId == 10) return "FP16";
    if (dtypeId == 16) return "BF16";
    return "FP32";
}

bool readPluginString(nvinfer1::PluginField const& f, char* dst, size_t dstSize) {
    if (dst == nullptr || dstSize == 0) return false;
    dst[0] = '\0';
    if (f.data == nullptr || f.length <= 0) return false;
    auto const* src = static_cast<char const*>(f.data);
    size_t maxLen = static_cast<size_t>(f.length);
    size_t n = 0;
    while (n < maxLen && src[n] != '\0') ++n;
    n = std::min(n, dstSize - 1);
    std::memcpy(dst, src, n);
    dst[n] = '\0';
    return true;
}

bool validGroupSize(int32_t groupSize, int32_t K) {
    if (groupSize == 0) return true;
    switch (groupSize) {
        case 4:
        case 16:
        case 64:
        case 256:
        case 1024:
            return K > 0 && K % groupSize == 0;
        default:
            return false;
    }
}

size_t pluginWorkspaceBytes(int64_t M, int64_t K, int64_t N) {
    if (M <= 0 || K <= 0 || N <= 0) return 0;
    // Runtime aligns the workspace pointer itself; reserve worst-case headroom
    // so those alignment bumps never exceed TensorRT's requested size.
    size_t cursor = 256;
    cursor = alignUp(cursor, 16);
    cursor += static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(int8_t);
    cursor = alignUp(cursor, 16);
    cursor += static_cast<size_t>(M) * sizeof(float);
    cursor = alignUp(cursor, 256);
    cursor += static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t);
    cursor = alignUp(cursor, 256);
    cursor += kCUBLASLT_WORKSPACE_BYTES;
    return cursor;
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────
// ConvRotInt8LinearPlugin
// ──────────────────────────────────────────────────────────────────────────

ConvRotInt8LinearPlugin::ConvRotInt8LinearPlugin(
    int32_t group_size, int32_t in_features, int32_t out_features, int32_t has_bias,
    int32_t input_dtype_id, int32_t output_dtype_id, std::string preferred_format)
    : m_group_size(group_size),
      m_in_features(in_features),
      m_out_features(out_features),
      m_has_bias(has_bias),
      m_input_dtype_id(normalizeDtypeId(input_dtype_id)),
      m_output_dtype_id(normalizeDtypeId(output_dtype_id)),
      m_preferred_format(std::move(preferred_format)) {}

ConvRotInt8LinearPlugin::ConvRotInt8LinearPlugin(void const* data, size_t length) {
    if (length != getSerializationSize()) return;
    uint8_t const* d = static_cast<uint8_t const*>(data);
    int32_t input_dtype_id = 10, output_dtype_id = 10;
    std::memcpy(&m_group_size,   d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&m_in_features,  d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&m_out_features, d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&m_has_bias,     d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&input_dtype_id,  d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&output_dtype_id, d, sizeof(int32_t)); d += sizeof(int32_t);
    m_input_dtype_id = normalizeDtypeId(input_dtype_id);
    m_output_dtype_id = normalizeDtypeId(output_dtype_id);
}

ConvRotInt8LinearPlugin::~ConvRotInt8LinearPlugin() {
    destroyCublasLt();
}

// ── IPluginV3 ──────────────────────────────────────────────────────────────

nvinfer1::IPluginCapability* ConvRotInt8LinearPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    if (type == nvinfer1::PluginCapabilityType::kBUILD)
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    return static_cast<nvinfer1::IPluginV3OneCore*>(this);
}

// ── IPluginV3OneCore ──────────────────────────────────────────────────────

char const* ConvRotInt8LinearPlugin::getPluginName() const noexcept {
    return kCONVROT_INT8_LINEAR_PLUGIN_NAME;
}
char const* ConvRotInt8LinearPlugin::getPluginVersion() const noexcept {
    return kCONVROT_INT8_LINEAR_PLUGIN_VERSION;
}
char const* ConvRotInt8LinearPlugin::getPluginNamespace() const noexcept {
    return m_namespace.c_str();
}
// Note: setPluginNamespace() removed in TRT 11; namespace is set via creator.

// ── IPluginV3OneBuild ──────────────────────────────────────────────────────

int32_t ConvRotInt8LinearPlugin::getNbOutputs() const noexcept { return 1; }

int32_t ConvRotInt8LinearPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const*, int32_t) const noexcept {
    if (nbOutputs < 1) return -1;
    outputTypes[0] = trtTypeFromDtypeId(m_output_dtype_id);
    return 0;
}

int32_t ConvRotInt8LinearPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const*, int32_t,
    nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    if (nbOutputs < 1 || nbInputs < 1) return -1;
    auto const& x_dims = inputs[0];
    int32_t ndim = x_dims.nbDims;
    if (ndim < 1) return -1;
    outputs[0].nbDims = ndim;
    for (int32_t i = 0; i < ndim - 1; ++i)
        outputs[0].d[i] = x_dims.d[i];
    outputs[0].d[ndim - 1] = exprBuilder.constant(m_out_features);
    return 0;
}

bool ConvRotInt8LinearPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept {
    if (pos < 0 || pos >= nbInputs + nbOutputs) return false;

    // The current kernel indexes tensors as contiguous row-major memory. Keep
    // kLINEAR as the only accepted physical layout until the packed-layout
    // kernels land; the v2 dtype/tactic contract is independent of that future
    // optimization. The ONNX-side preferred_format attribute is still parsed
    // and serialized so future kernels can enable packed formats without graph
    // changes.
    if (inOut[pos].desc.format != nvinfer1::PluginFormat::kLINEAR) return false;

    auto type = inOut[pos].desc.type;
    auto inputType = trtTypeFromDtypeId(m_input_dtype_id);
    auto outputType = trtTypeFromDtypeId(m_output_dtype_id);
    if (pos < nbInputs) {
        switch (pos) {
            case 0: return type == inputType;                        // x
            case 1: return type == nvinfer1::DataType::kINT8;        // weight_q
            case 2: return type == nvinfer1::DataType::kFLOAT;       // weight_scale
            case 3: return type == inputType;                        // H (ignored by butterfly impl)
            case 4: return m_has_bias && type == nvinfer1::DataType::kFLOAT; // bias stays FP32 (1D sane rule)
            default: return false;
        }
    }
    return type == outputType;  // output y
}

int32_t ConvRotInt8LinearPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    (void)out;
    if (nbInputs < (m_has_bias ? 5 : 4) || nbOutputs < 1) return -1;
    int64_t optM = flattenedRows(in[0].opt);
    int32_t optK = lastDim(in[0].opt);
    if (optM <= 0 || optK <= 0 || optK != m_in_features) return -1;
    if (!validGroupSize(m_group_size, optK)) return -1;
    if (optM > std::numeric_limits<int32_t>::max()) return -1;
    m_M = static_cast<int32_t>(optM);
    m_K = optK;
    m_N = m_out_features;
    return 0;
}

// Note: destroy() removed in TRT 11; the destructor handles cleanup.

size_t ConvRotInt8LinearPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept {
    if (nbInputs < 1) return 0;
    int64_t maxM = flattenedRows(inputs[0].max);
    int32_t maxK = lastDim(inputs[0].max);
    if (maxM <= 0 || maxK <= 0) {
        maxM = flattenedRows(inputs[0].desc.dims);
        maxK = lastDim(inputs[0].desc.dims);
    }
    return pluginWorkspaceBytes(maxM, maxK, m_out_features);
}

// ── Custom Tactics (IPluginV3OneBuild) ──────────────────────────────────────

int32_t ConvRotInt8LinearPlugin::getNbTactics() noexcept {
    return 4;
}

int32_t ConvRotInt8LinearPlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept {
    if (tactics == nullptr || nbTactics < 4) return 1;
    // TensorRT reserves tactic 0 for the implicit default path; advertised
    // custom tactic IDs must be unique and non-zero (NvInferRuntime.h).
    // getValidTactics returns an error code, not the number of tactics.
    tactics[0] = 1;
    tactics[1] = 2;
    tactics[2] = 3;
    tactics[3] = 4;
    return 0;
}

char const* ConvRotInt8LinearPlugin::getTimingCacheID() noexcept {
    return "ConvRotInt8Linear.v2";
}

int32_t ConvRotInt8LinearPlugin::getFormatCombinationLimit() noexcept {
    // One physical layout combination today (kLINEAR), with dtype selected by
    // input_dtype/output_dtype. Packed-layout tactics can raise this later
    // without changing the ONNX custom-op schema.
    return 1;
}

char const* ConvRotInt8LinearPlugin::getMetadataString() noexcept {
    // Return a string describing the plugin config (for engine inspector).
    // This is called once; the buffer must live as long as the plugin.
    static thread_local std::string meta;
    std::ostringstream oss;
    oss << "ConvRotInt8Linear.v2(gs=" << m_group_size
        << ",K=" << m_in_features
        << ",N=" << m_out_features
        << ",bias=" << m_has_bias
        << ",in=" << dtypeNameFromId(m_input_dtype_id)
        << ",out=" << dtypeNameFromId(m_output_dtype_id)
        << ",fmt=" << m_preferred_format
        << ",tactic=" << m_tactic << ")";
    meta = oss.str();
    return meta.c_str();
}

// ── IPluginV3OneRuntime ────────────────────────────────────────────────────

int32_t ConvRotInt8LinearPlugin::setTactic(int32_t tactic) noexcept {
    if (tactic < 0 || tactic > 4) return -1;
    // Tactic 0 is TensorRT's implicit default. Tactics 1-4 are the advertised
    // forward-compatible build contract and currently fall through to the same
    // cuBLASLt implementation, so ONNX export and TRT timing-cache keys are
    // decoupled from future kernel optimization work.
    m_tactic = tactic;
    return 0;
}

int32_t ConvRotInt8LinearPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    if (nbInputs < (m_has_bias ? 5 : 4) || nbOutputs < 1) return -1;
    // Extract M, K, N from the descriptors.
    (void)out;
    auto const& x_desc = in[0].dims;
    int64_t runtimeM = flattenedRows(x_desc);
    if (runtimeM <= 0 || runtimeM > std::numeric_limits<int32_t>::max()) return -1;
    m_M = static_cast<int32_t>(runtimeM);
    m_K = lastDim(x_desc);
    m_N = m_out_features;

    if (m_K != m_in_features) return -1;
    if (!validGroupSize(m_group_size, m_K)) return -1;
    if (m_M <= 0 || m_K <= 0 || m_N <= 0) return -1;

    // (Re)create cuBLASLt descriptors for the new shapes.
    destroyCublasLt();
    if (!initCublasLt()) return -1;
    return 0;
}

int32_t ConvRotInt8LinearPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {
    try {
        (void)inputDesc;
        (void)outputDesc;
        int32_t M = m_M, K = m_K, N = m_N;
        if (M <= 0 || K <= 0 || N <= 0) return -1;

        void const* x_ptr      = inputs[0];
        int8_t const* wq_ptr   = static_cast<int8_t const*>(inputs[1]);
        float const* ws_ptr    = static_cast<float const*>(inputs[2]);
        void const* H_ptr      = inputs[3];
        void const* bias_ptr   = m_has_bias ? inputs[4] : nullptr;
        void* y_ptr            = outputs[0];

        if (workspace == nullptr) return -1;

        // Workspace layout:
        //   x_q:       [M, K] INT8    (quantized activations)
        //   x_s:       [M] FP32       (per-row activation scales)
        //   acc:       [M, N] INT32   (cuBLASLt GEMM output)
        //   cublas_ws: cuBLASLt scratch, separate from plugin scratch
        size_t xq_bytes = (size_t)M * K * sizeof(int8_t);
        size_t xs_bytes = (size_t)M * sizeof(float);
        size_t acc_bytes = (size_t)M * N * sizeof(int32_t);

        char* cursor = static_cast<char*>(workspace);
        cursor = alignPtr(cursor, 16);
        int8_t* xq_w = reinterpret_cast<int8_t*>(cursor);
        cursor += xq_bytes;
        cursor = alignPtr(cursor, 16);
        float* xs_w = reinterpret_cast<float*>(cursor);
        cursor += xs_bytes;
        cursor = alignPtr(cursor, 256);
        int32_t* acc_w = reinterpret_cast<int32_t*>(cursor);
        cursor += acc_bytes;
        cursor = alignPtr(cursor, 256);
        void* cublas_workspace = cursor;

        int32_t const input_dtype = kernelDtypeFromTrt(inputDesc[0].type);
        int32_t const bias_dtype = m_has_bias ? kernelDtypeFromTrt(inputDesc[4].type) : input_dtype;
        int32_t const output_dtype = kernelDtypeFromTrt(outputDesc[0].type);

        // Phase 1: ConvRot rotation + per-row INT8 quantization
        if (!launch_convrot_activation_quant(x_ptr, xq_w, xs_w, H_ptr,
                                             M, K, m_group_size, input_dtype,
                                             stream)) {
            std::fprintf(stderr, "[ConvRotInt8Linear] convrot_activation_quant failed\n");
            return -1;
        }

        // Phase 2: INT8 × INT8 → INT32 GEMM via cuBLASLt.
        // cuBLASLt sees column-major views:
        //   opA(w_q_cm[K,N]) = [N,K], opB(x_q_cm[K,M]) = [K,M],
        //   C_cm[N,M] maps physically to acc_rm[M,N].
        int32_t alpha = 1, beta = 0;
        cublasStatus_t cublas_status = cublasLtMatmul(m_cublasLt, m_matmulDesc,
            &alpha,
            wq_ptr, m_layoutA,
            xq_w, m_layoutB,
            &beta,
            acc_w, m_layoutC,
            acc_w, m_layoutC,
            nullptr,
            cublas_workspace, kCUBLASLT_WORKSPACE_BYTES,
            stream);

        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "[ConvRotInt8Linear] cublasLtMatmul failed: %d (tactic=%d)\n",
                         cublas_status, m_tactic);
            return -1;
        }

        // Phase 3: Dequant + bias epilogue
        if (!launch_dequant_bias_epilogue(acc_w, y_ptr, xs_w, ws_ptr, bias_ptr,
                                          M, N, m_has_bias != 0, bias_dtype,
                                          output_dtype, stream)) {
            std::fprintf(stderr, "[ConvRotInt8Linear] dequant_bias_epilogue failed\n");
            return -1;
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

nvinfer1::IPluginV3* ConvRotInt8LinearPlugin::clone() noexcept {
    auto* p = new ConvRotInt8LinearPlugin(m_group_size, m_in_features,
                                          m_out_features, m_has_bias,
                                          m_input_dtype_id, m_output_dtype_id,
                                          m_preferred_format);
    p->m_namespace = m_namespace;
    p->m_tactic = m_tactic;
    return p;
}

nvinfer1::IPluginV3* ConvRotInt8LinearPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* ctx) noexcept {
    // TRT 11 requires this. For our simple plugin that doesn't use
    // plugin resources, just return a clone.
    (void)ctx;
    return clone();
}

nvinfer1::PluginFieldCollection const* ConvRotInt8LinearPlugin::getFieldsToSerialize() noexcept {
    m_fields.clear();
    m_fields.push_back({kFIELD_GROUP_SIZE,   &m_group_size,   nvinfer1::PluginFieldType::kINT32, 1});
    m_fields.push_back({kFIELD_IN_FEATURES,  &m_in_features,  nvinfer1::PluginFieldType::kINT32, 1});
    m_fields.push_back({kFIELD_OUT_FEATURES, &m_out_features, nvinfer1::PluginFieldType::kINT32, 1});
    m_fields.push_back({kFIELD_HAS_BIAS,     &m_has_bias,     nvinfer1::PluginFieldType::kINT32, 1});
    static thread_local std::string input_dtype;
    static thread_local std::string output_dtype;
    input_dtype = dtypeNameFromId(m_input_dtype_id);
    output_dtype = dtypeNameFromId(m_output_dtype_id);
    static thread_local int32_t input_dtype_id;
    static thread_local int32_t output_dtype_id;
    input_dtype_id = m_input_dtype_id;
    output_dtype_id = m_output_dtype_id;
    m_fields.push_back({kFIELD_INPUT_DTYPE,  input_dtype.c_str(),  nvinfer1::PluginFieldType::kCHAR, static_cast<int32_t>(input_dtype.size() + 1)});
    m_fields.push_back({kFIELD_OUTPUT_DTYPE, output_dtype.c_str(), nvinfer1::PluginFieldType::kCHAR, static_cast<int32_t>(output_dtype.size() + 1)});
    m_fields.push_back({kFIELD_INPUT_DTYPE_ID,  &input_dtype_id,  nvinfer1::PluginFieldType::kINT32, 1});
    m_fields.push_back({kFIELD_OUTPUT_DTYPE_ID, &output_dtype_id, nvinfer1::PluginFieldType::kINT32, 1});
    m_fields.push_back({kFIELD_PREFERRED_FORMAT, m_preferred_format.c_str(), nvinfer1::PluginFieldType::kCHAR, static_cast<int32_t>(m_preferred_format.size() + 1)});
    m_fc.nbFields = static_cast<int32_t>(m_fields.size());
    m_fc.fields = m_fields.data();
    return &m_fc;
}

size_t ConvRotInt8LinearPlugin::getSerializationSize() const noexcept {
    return 6 * sizeof(int32_t);  // group_size, in_features, out_features, has_bias, input_dtype_id, output_dtype_id
}

void ConvRotInt8LinearPlugin::serialize(void* buffer) const noexcept {
    uint8_t* d = static_cast<uint8_t*>(buffer);
    std::memcpy(d, &m_group_size,   sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_in_features,  sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_out_features, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_has_bias,     sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_input_dtype_id,  sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_output_dtype_id, sizeof(int32_t)); d += sizeof(int32_t);
}

// ── cuBLASLt initialization ────────────────────────────────────────────────

bool ConvRotInt8LinearPlugin::initCublasLt() {
    if (m_cublasLt) return true;  // already initialized

    if (cublasLtCreate(&m_cublasLt) != CUBLAS_STATUS_SUCCESS) return false;

    // Create matmul descriptor: INT8 × INT8 → INT32. For 32I compute,
    // cuBLASLt requires CUDA_R_32I scale type and integer alpha/beta.
    if (cublasLtMatmulDescCreate(&m_matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I)
        != CUBLAS_STATUS_SUCCESS) {
        destroyCublasLt();
        return false;
    }

    // Column-major reinterpretation of row-major buffers:
    // A = w_q_cm[K, N], opA=T; B = x_q_cm[K, M], opB=N.
    cublasOperation_t opT = CUBLAS_OP_T;
    cublasOperation_t opN = CUBLAS_OP_N;
    if (cublasLtMatmulDescSetAttribute(m_matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT))
            != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(m_matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN))
            != CUBLAS_STATUS_SUCCESS) {
        destroyCublasLt();
        return false;
    }

    if (cublasLtMatrixLayoutCreate(&m_layoutA, CUDA_R_8I, m_K, m_N, m_K)
            != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&m_layoutB, CUDA_R_8I, m_K, m_M, m_K)
            != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&m_layoutC, CUDA_R_32I, m_N, m_M, m_N)
            != CUBLAS_STATUS_SUCCESS) {
        destroyCublasLt();
        return false;
    }
    return true;
}

void ConvRotInt8LinearPlugin::destroyCublasLt() {
    if (m_layoutC) { cublasLtMatrixLayoutDestroy(m_layoutC); m_layoutC = nullptr; }
    if (m_layoutB) { cublasLtMatrixLayoutDestroy(m_layoutB); m_layoutB = nullptr; }
    if (m_layoutA) { cublasLtMatrixLayoutDestroy(m_layoutA); m_layoutA = nullptr; }
    if (m_matmulDesc) { cublasLtMatmulDescDestroy(m_matmulDesc); m_matmulDesc = nullptr; }
    if (m_cublasLt) { cublasLtDestroy(m_cublasLt); m_cublasLt = nullptr; }
}

// ──────────────────────────────────────────────────────────────────────────
// ConvRotInt8LinearPluginCreator
// ──────────────────────────────────────────────────────────────────────────

ConvRotInt8LinearPluginCreator::ConvRotInt8LinearPluginCreator() {
    m_fields.emplace_back(kFIELD_GROUP_SIZE,   nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_IN_FEATURES,  nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_OUT_FEATURES, nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_HAS_BIAS,     nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_INPUT_DTYPE,  nullptr, nvinfer1::PluginFieldType::kCHAR, 0);
    m_fields.emplace_back(kFIELD_OUTPUT_DTYPE, nullptr, nvinfer1::PluginFieldType::kCHAR, 0);
    m_fields.emplace_back(kFIELD_INPUT_DTYPE_ID,  nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_OUTPUT_DTYPE_ID, nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_PREFERRED_FORMAT, nullptr, nvinfer1::PluginFieldType::kCHAR, 0);
    m_fc.nbFields = static_cast<int32_t>(m_fields.size());
    m_fc.fields = m_fields.data();
}

char const* ConvRotInt8LinearPluginCreator::getPluginName() const noexcept {
    return kCONVROT_INT8_LINEAR_PLUGIN_NAME;
}
char const* ConvRotInt8LinearPluginCreator::getPluginVersion() const noexcept {
    return kCONVROT_INT8_LINEAR_PLUGIN_VERSION;
}
char const* ConvRotInt8LinearPluginCreator::getPluginNamespace() const noexcept {
    return m_namespace.c_str();
}
// Note: setPluginNamespace() removed in TRT 11; namespace set via constructor.
nvinfer1::PluginFieldCollection const* ConvRotInt8LinearPluginCreator::getFieldNames() noexcept {
    return &m_fc;
}

nvinfer1::IPluginV3* ConvRotInt8LinearPluginCreator::createPlugin(
    char const*, nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase) noexcept {
    int32_t group_size = 0, in_features = 0, out_features = 0, has_bias = 0;
    int32_t input_dtype_id = 10;
    int32_t output_dtype_id = 10;
    std::string preferred_format = "HWC8";
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        auto const& f = fc->fields[i];
        if (f.name == nullptr || f.data == nullptr) continue;
        if (std::strcmp(f.name, kFIELD_GROUP_SIZE) == 0)
            group_size = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_IN_FEATURES) == 0)
            in_features = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_OUT_FEATURES) == 0)
            out_features = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_HAS_BIAS) == 0)
            has_bias = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_INPUT_DTYPE) == 0) {
            if (f.type == nvinfer1::PluginFieldType::kINT32) {
                input_dtype_id = (*static_cast<int32_t const*>(f.data) != 0) ? 10 : 1;
            } else {
                char tmp[16];
                if (readPluginString(f, tmp, sizeof(tmp))) input_dtype_id = parseDtypeId(tmp, input_dtype_id);
            }
        } else if (std::strcmp(f.name, kFIELD_OUTPUT_DTYPE) == 0) {
            if (f.type == nvinfer1::PluginFieldType::kINT32) {
                output_dtype_id = (*static_cast<int32_t const*>(f.data) != 0) ? 10 : 1;
            } else {
                char tmp[16];
                if (readPluginString(f, tmp, sizeof(tmp))) output_dtype_id = parseDtypeId(tmp, output_dtype_id);
            }
        } else if (std::strcmp(f.name, kFIELD_INPUT_DTYPE_ID) == 0) {
            input_dtype_id = normalizeDtypeId(*static_cast<int32_t const*>(f.data), input_dtype_id);
        } else if (std::strcmp(f.name, kFIELD_OUTPUT_DTYPE_ID) == 0) {
            output_dtype_id = normalizeDtypeId(*static_cast<int32_t const*>(f.data), output_dtype_id);
        } else if (std::strcmp(f.name, kFIELD_PREFERRED_FORMAT) == 0) {
            char tmp[32];
            if (readPluginString(f, tmp, sizeof(tmp))) preferred_format = tmp;
        }
    }
    return new ConvRotInt8LinearPlugin(group_size, in_features, out_features, has_bias,
                                       input_dtype_id, output_dtype_id, preferred_format);
}

}  // namespace hotstep

// ──────────────────────────────────────────────────────────────────────────
// Static registration
// ──────────────────────────────────────────────────────────────────────────

// TRT 11: REGISTER_TENSORRT_PLUGIN macro changed and doesn't work
// the same way. Use manual registration via hotstep_register_plugins() below.
// REGISTER_TENSORRT_PLUGIN(hotstep::ConvRotInt8LinearPluginCreator);

// Portable DLL export macro for the plugin entry point.
// On Windows, __declspec(dllexport) is required for GetProcAddress() to find
// the symbol. On Linux/macOS, default visibility is sufficient.
#if defined(_WIN32)
#  define HOTSTEP_PLUGIN_EXPORT __declspec(dllexport)
#else
#  define HOTSTEP_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
HOTSTEP_PLUGIN_EXPORT int hotstep_register_plugins() {
    try {
        // TRT 11: getPluginRegistry() is an extern "C" free function declared
        // in NvInferRuntime.h (included transitively via NvInfer.h).
        // It is in the global namespace, NOT in nvinfer1.
        auto* registry = getPluginRegistry();
        if (!registry) return 1;
        static hotstep::ConvRotInt8LinearPluginCreator creator;
        registry->registerCreator(creator, hotstep::kCONVROT_INT8_LINEAR_PLUGIN_NAMESPACE);
        return 0;
    } catch (...) {
        return 2;
    }
}
}  // extern "C"

#endif  // HOT_STEP_TRT
