"""HOT-Step custom TensorRT plugin for the w8a8 + ConvRot recipe.

The actual plugin kernel is implemented in C++ as a proper IPluginV3 +
IPluginCreatorV3 (the only stable TRT 11 plugin API, per
https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html).
This Python package is the ONNX-op emitter + the NumPy reference
implementation used by unit tests. It does NOT register the kernel with
TRT — that happens in the C++ ``libhotstep_plugins.so`` which the engine
runtime dlopens before deserializing any w8a8 engine.

Build-time flow
---------------
1. ``export_dit.py --precision w8a8`` emits ONNX with
   ``hotstep::ConvRotInt8Linear`` custom-op nodes (one per MatMul/Gemm
   site). See ``_rewrite_w8a8_weight_ops_with_plugin`` in export_dit.py.
2. ``build-trt-engine.py --precision-policy w8a8`` parses the ONNX with
   TRT's OnnxParser. The parser looks up the custom op in TRT's plugin
   registry — the registry entry was created by the static
   ``REGISTER_TENSORRT_PLUGIN`` macro in
   ``engine/src/plugins/convrot_int8_linear_plugin.cpp``.
3. TRT compiles the graph (including the plugin) into a serialized
   engine. The plugin kernel is JIT-compiled to PTX/CUBIN by TRT's
   builder.
4. At runtime, ``dit-trt.h:dit_trt_load`` dlopens
   ``libhotstep_plugins.so`` (which re-registers the creator under the
   "hotstep" namespace) and then deserializes the engine.

ONNX custom-op identity
-----------------------
  * domain: ``hotstep``
  * op_type: ``ConvRotInt8Linear``
  * inputs: [x, weight_q, weight_scale, H, (bias)]
  * outputs: [y]
  * attributes: group_size, in_features, out_features, has_bias

When ``tensorrt`` is not installed (unit-test environments), the
``register_plugins()`` function returns False and the NumPy reference
implementation (``conv_rot_int8_linear_reference``) is the only path
exercised. The reference mirrors the C++ kernel math exactly.
"""

from .convrot_int8_plugin import (
    CONVROT_INT8_LINEAR_OP_NAME,
    CONVROT_INT8_LINEAR_OP_NAMESPACE,
    conv_rot_int8_linear_reference,
    make_convrot_int8_linear_onnx_node,
    register_plugins,
    is_registered,
)

__all__ = [
    "CONVROT_INT8_LINEAR_OP_NAME",
    "CONVROT_INT8_LINEAR_OP_NAMESPACE",
    "conv_rot_int8_linear_reference",
    "make_convrot_int8_linear_onnx_node",
    "register_plugins",
    "is_registered",
]
