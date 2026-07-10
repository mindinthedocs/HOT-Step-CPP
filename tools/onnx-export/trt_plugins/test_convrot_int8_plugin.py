#!/usr/bin/env python3
"""Unit tests for the Python-side ConvRotInt8Linear v3 contract.

These tests deliberately avoid importing TensorRT or ONNX. They validate the
stable ONNX-emission surface and the NumPy reference boundary dtype, which are
the pieces available in a CPU-only CI environment.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convrot_int8_plugin import (
    CONVROT_INT8_LINEAR_PLUGIN_VERSION,
    _find_plugin_library,
    conv_rot_int8_linear_reference,
    make_convrot_int8_linear_onnx_node,
    make_convrot_quantize_onnx_node,
)


class _FakeTensorProto:
    FLOAT = 1
    FLOAT16 = 10
    BFLOAT16 = 16


class _FakeHelper:
    @staticmethod
    def make_node(op_type, inputs, outputs, name=None, domain=None, **attrs):
        return {
            "op_type": op_type,
            "inputs": list(inputs),
            "outputs": list(outputs),
            "name": name,
            "domain": domain,
            "attrs": dict(attrs),
        }


class ConvRotInt8PluginV3Tests(unittest.TestCase):
    def test_reference_defaults_to_fp16_boundary_output(self) -> None:
        x = np.array([[0.25, -0.5, 1.0, -2.0]], dtype=np.float32)
        weight_q = np.array([[1, -2, 3, -4], [-4, 3, -2, 1]], dtype=np.int8)
        weight_scale = np.array([0.1, 0.2], dtype=np.float32)
        h_unused = np.empty((0, 0), dtype=np.float32)

        y = conv_rot_int8_linear_reference(
            x,
            weight_q,
            weight_scale,
            h_unused,
            bias=None,
            group_size=0,
            in_features=4,
            out_features=2,
            has_bias=False,
        )

        self.assertEqual(y.dtype, np.float16)
        self.assertEqual(y.shape, (1, 2))

    def test_reference_keeps_fp32_when_requested(self) -> None:
        x = np.array([[0.25, -0.5, 1.0, -2.0]], dtype=np.float32)
        weight_q = np.array([[1, -2, 3, -4]], dtype=np.int8)
        weight_scale = np.array([0.1], dtype=np.float32)
        y = conv_rot_int8_linear_reference(
            x,
            weight_q,
            weight_scale,
            np.empty((0, 0), dtype=np.float32),
            bias=np.array([0.125], dtype=np.float32),
            group_size=0,
            in_features=4,
            out_features=1,
            has_bias=True,
            output_dtype="FP32",
        )
        self.assertEqual(y.dtype, np.float32)

    def test_make_node_emits_v2_dtype_and_format_attributes(self) -> None:
        node = make_convrot_int8_linear_onnx_node(
            helper_module=_FakeHelper,
            tensor_proto_module=_FakeTensorProto,
            x_name="x_fp16",
            weight_q_name="w_q",
            weight_scale_name="w_scale",
            bias_name="bias_fp16",
            output_name="y",
            node_name="linear/ConvRotInt8Linear",
            group_size=256,
            in_features=2048,
            out_features=6144,
            has_bias=True,
        )

        self.assertEqual(node["domain"], "hotstep")
        self.assertEqual(node["op_type"], "ConvRotInt8Linear")
        self.assertEqual(node["inputs"], ["x_fp16", "w_q", "w_scale", "bias_fp16"])
        self.assertEqual(node["outputs"], ["y"])
        self.assertEqual(node["attrs"]["input_dtype"], "FP16")
        self.assertEqual(node["attrs"]["output_dtype"], "FP16")
        self.assertEqual(node["attrs"]["input_dtype_id"], _FakeTensorProto.FLOAT16)
        self.assertEqual(node["attrs"]["output_dtype_id"], _FakeTensorProto.FLOAT16)
        self.assertEqual(node["attrs"]["plugin_version"], CONVROT_INT8_LINEAR_PLUGIN_VERSION)
        self.assertEqual(node["attrs"]["plugin_namespace"], "hotstep")
        self.assertEqual(node["attrs"]["preferred_format"], "HWC8")

    def test_make_node_allows_fp32_output_fallback(self) -> None:
        node = make_convrot_int8_linear_onnx_node(
            helper_module=_FakeHelper,
            tensor_proto_module=_FakeTensorProto,
            x_name="x_fp16",
            weight_q_name="proj_out.weight",
            weight_scale_name="proj_out.weight.w8a8_scale",
            bias_name="",
            output_name="velocity",
            node_name="proj_out/ConvRotInt8Linear",
            group_size=256,
            in_features=2048,
            out_features=64,
            has_bias=False,
            output_dtype="FP32",
        )
        self.assertEqual(node["inputs"], ["x_fp16", "proj_out.weight", "proj_out.weight.w8a8_scale"])
        self.assertEqual(node["attrs"]["output_dtype"], "FP32")
        self.assertEqual(node["attrs"]["output_dtype_id"], _FakeTensorProto.FLOAT)

    def test_fused_qk_rope_epilogue_contract(self) -> None:
        node = make_convrot_int8_linear_onnx_node(
            helper_module=_FakeHelper,
            tensor_proto_module=_FakeTensorProto,
            x_name="x",
            weight_q_name="q.weight",
            weight_scale_name="q.scale",
            bias_name="",
            output_name="q_fp16",
            node_name="q/ConvRotInt8Linear",
            group_size=256,
            in_features=2560,
            out_features=4096,
            has_bias=False,
            input_dtype="FP32",
            output_dtype="FP16",
            epilogue_kind=2,
            aux_input_names=["q_gamma", "rope_cos", "rope_sin"],
        )
        self.assertEqual(node["inputs"][-3:], ["q_gamma", "rope_cos", "rope_sin"])
        self.assertEqual(node["attrs"]["epilogue_kind"], 2)
        self.assertEqual(node["attrs"]["input_dtype_id"], _FakeTensorProto.FLOAT)
        self.assertEqual(node["attrs"]["output_dtype_id"], _FakeTensorProto.FLOAT16)

    def test_prequantized_k2_contract(self) -> None:
        node = make_convrot_int8_linear_onnx_node(
            helper_module=_FakeHelper,
            tensor_proto_module=_FakeTensorProto,
            x_name="xq",
            activation_scale_name="xs",
            weight_q_name="wq",
            weight_scale_name="ws",
            bias_name="",
            output_name="y",
            node_name="linear/Prequantized",
            group_size=256,
            in_features=2560,
            out_features=1024,
            has_bias=False,
            prequantized=True,
        )
        self.assertEqual(node["inputs"], ["xq", "xs", "wq", "ws"])
        self.assertEqual(node["attrs"]["prequantized"], 1)

    def test_quantize_only_contract(self) -> None:
        node = make_convrot_quantize_onnx_node(
            _FakeHelper, _FakeTensorProto, "x", "xq", "xs", "quant",
            256, 2560, input_dtype="FP32")
        self.assertEqual(node["inputs"], ["x"])
        self.assertEqual(node["outputs"], ["xq", "xs"])
        self.assertEqual(node["attrs"]["quantize_only"], 1)

    def test_epilogue_aux_arity_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            make_convrot_int8_linear_onnx_node(
                _FakeHelper, _FakeTensorProto, "x", "wq", "ws", "", "y", "bad",
                256, 2560, 4096, False, epilogue_kind=2,
                aux_input_names=["gamma"])

    def test_find_plugin_library_honors_explicit_env_file(self) -> None:
        if sys.platform == "win32":
            lib_name = "hotstep_plugins.dll"
        elif sys.platform == "darwin":
            lib_name = "libhotstep_plugins.dylib"
        else:
            lib_name = "libhotstep_plugins.so"

        old_file = os.environ.get("HOTSTEP_PLUGIN_LIBRARY")
        old_path = os.environ.get("HOTSTEP_PLUGINS_PATH")
        try:
            with tempfile.TemporaryDirectory() as td:
                fake = Path(td) / lib_name
                fake.write_bytes(b"not a real shared library; path lookup only")
                os.environ["HOTSTEP_PLUGIN_LIBRARY"] = str(fake)
                os.environ.pop("HOTSTEP_PLUGINS_PATH", None)
                self.assertEqual(Path(_find_plugin_library()), fake)
        finally:
            if old_file is None:
                os.environ.pop("HOTSTEP_PLUGIN_LIBRARY", None)
            else:
                os.environ["HOTSTEP_PLUGIN_LIBRARY"] = old_file
            if old_path is None:
                os.environ.pop("HOTSTEP_PLUGINS_PATH", None)
            else:
                os.environ["HOTSTEP_PLUGINS_PATH"] = old_path

    def test_bf16_boundary_is_rejected_by_selective_policy(self) -> None:
        with self.assertRaises(ValueError):
            make_convrot_int8_linear_onnx_node(
                helper_module=_FakeHelper,
                tensor_proto_module=_FakeTensorProto,
                x_name="x_bf16",
                weight_q_name="w_q",
                weight_scale_name="w_scale",
                bias_name="",
                output_name="y_bf16",
                node_name="linear/ConvRotInt8Linear",
                group_size=256,
                in_features=2048,
                out_features=2048,
                has_bias=False,
                input_dtype="BF16",
                output_dtype="BF16",
            )


if __name__ == "__main__":
    unittest.main()
