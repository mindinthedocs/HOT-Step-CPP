/*
 * convrot_int8_linear_plugin.cpp — IPluginV3 implementation for ConvRotInt8Linear.
 *
 * Architecture (same as TRT-LLM's smooth_quant_gemm_plugin):
 *
 *   enqueue() pipeline:
 *     1. ConvRot activation rotation + per-row INT8 quant (custom CUDA kernel)
 *     2. INT8 × INT8 → INT32 matmul (cuBLASLt cublasLtMatmul)
 *     3. Dequant + bias epilogue (custom CUDA kernel)
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
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <sstream>

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
    int32_t group_size, int32_t in_features, int32_t out_features, int32_t has_bias)
    : m_group_size(group_size),
      m_in_features(in_features),
      m_out_features(out_features),
      m_has_bias(has_bias) {}

ConvRotInt8LinearPlugin::ConvRotInt8LinearPlugin(void const* data, size_t length) {
    if (length != getSerializationSize()) return;
    uint8_t const* d = static_cast<uint8_t const*>(data);
    std::memcpy(&m_group_size,   d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&m_in_features,  d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&m_out_features, d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&m_has_bias,     d, sizeof(int32_t)); d += sizeof(int32_t);
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
    outputTypes[0] = nvinfer1::DataType::kFLOAT;  // FP32 output
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
    if (inOut[pos].desc.format != nvinfer1::PluginFormat::kLINEAR) return false;
    auto type = inOut[pos].desc.type;
    if (pos < nbInputs) {
        switch (pos) {
            case 0: return type == nvinfer1::DataType::kFLOAT;       // x
            case 1: return type == nvinfer1::DataType::kINT8;        // weight_q
            case 2: return type == nvinfer1::DataType::kFLOAT;       // weight_scale
            case 3: return type == nvinfer1::DataType::kFLOAT;       // H
            case 4: return m_has_bias && type == nvinfer1::DataType::kFLOAT;  // bias
            default: return false;
        }
    }
    return type == nvinfer1::DataType::kFLOAT;  // output y
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
    return 0;
}

int32_t ConvRotInt8LinearPlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept {
    (void)tactics;
    (void)nbTactics;
    return 0;
}

char const* ConvRotInt8LinearPlugin::getTimingCacheID() noexcept {
    return "ConvRotInt8Linear";
}

int32_t ConvRotInt8LinearPlugin::getFormatCombinationLimit() noexcept {
    // Only 1 format combination (kLINEAR FP32/INT8) — no need for TRT to
    // try multiple format combos.
    return 1;
}

char const* ConvRotInt8LinearPlugin::getMetadataString() noexcept {
    // Return a string describing the plugin config (for engine inspector).
    // This is called once; the buffer must live as long as the plugin.
    static thread_local std::string meta;
    std::ostringstream oss;
    oss << "ConvRotInt8Linear(gs=" << m_group_size
        << ",K=" << m_in_features
        << ",N=" << m_out_features
        << ",bias=" << m_has_bias << ")";
    meta = oss.str();
    return meta.c_str();
}

// ── IPluginV3OneRuntime ────────────────────────────────────────────────────

int32_t ConvRotInt8LinearPlugin::setTactic(int32_t tactic) noexcept {
    m_tactic = 0;
    return tactic == 0 ? 0 : -1;
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

        float const* x_ptr     = static_cast<float const*>(inputs[0]);
        int8_t const* wq_ptr   = static_cast<int8_t const*>(inputs[1]);
        float const* ws_ptr    = static_cast<float const*>(inputs[2]);
        float const* H_ptr     = static_cast<float const*>(inputs[3]);
        float const* bias_ptr  = m_has_bias ? static_cast<float const*>(inputs[4]) : nullptr;
        float* y_ptr           = static_cast<float*>(outputs[0]);

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

        // Phase 1: ConvRot rotation + per-row INT8 quantization
        if (!launch_convrot_activation_quant(x_ptr, xq_w, xs_w, H_ptr,
                                             M, K, m_group_size, stream)) {
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
                                          M, N, m_has_bias != 0, stream)) {
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
                                          m_out_features, m_has_bias);
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
    m_fc.nbFields = static_cast<int32_t>(m_fields.size());
    m_fc.fields = m_fields.data();
    return &m_fc;
}

size_t ConvRotInt8LinearPlugin::getSerializationSize() const noexcept {
    return 4 * sizeof(int32_t);  // group_size, in_features, out_features, has_bias
}

void ConvRotInt8LinearPlugin::serialize(void* buffer) const noexcept {
    uint8_t* d = static_cast<uint8_t*>(buffer);
    std::memcpy(d, &m_group_size,   sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_in_features,  sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_out_features, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_has_bias,     sizeof(int32_t)); d += sizeof(int32_t);
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
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        auto const& f = fc->fields[i];
        if (std::strcmp(f.name, kFIELD_GROUP_SIZE) == 0)
            group_size = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_IN_FEATURES) == 0)
            in_features = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_OUT_FEATURES) == 0)
            out_features = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_HAS_BIAS) == 0)
            has_bias = *static_cast<int32_t const*>(f.data);
    }
    return new ConvRotInt8LinearPlugin(group_size, in_features, out_features, has_bias);
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
