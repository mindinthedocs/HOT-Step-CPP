#!/usr/bin/env python3
"""
export_dit.py — Export AceStep DiT forward pass to ONNX for TensorRT acceleration.

Exports the SINGLE FORWARD PASS (one diffusion timestep) of the DiT model,
wrapping the full 32-layer transformer + attention mask computation + RoPE
into a single ONNX graph with 4 simplified inputs.

Precision recipes (--precision):
  q8map-fp16 — Default. Export FP32, then downcast only the hardcoded
               Q8_0-equivalent DiT matrix-weight allowlist to FP16 and add
               explicit Cast nodes so TensorRT 11 strongly typed builds honor
               the graph-level policy.
  w8a8      — INT8 weights + INT8 activations, fused by the
               ConvRotInt8Linear TRT plugin. Export FP32, quantize the
               matrix-weight allowlist to INT8 with per-output-channel
               symmetric scales (after ConvRot rotation), AND emit a
               single ConvRotInt8Linear custom-op node per MatMul/Gemm
               site. The plugin fuses online activation rotation, per-row
               dynamic INT8 quantization, INT8×INT8 matmul, dequant, and
               bias add into one GPU kernel launch. ConvRot (regular
               Hadamard rotation) is applied offline to weights and online
               to activations so per-row quantization survives
               diffusion-model outliers. See tools/onnx-export/trt_plugins/
               and engine/src/plugins/ for the plugin implementation.
  fp32       — Full FP32. Correct but slow. Baseline for validation.

Usage:
    python export_dit.py --model-dir <path-to-safetensors-model> --output <output.onnx>
    python export_dit.py --model-dir <path> --output <path> --precision w8a8

The diffusion loop, guidance (APG/CFG), and solvers stay in C++.
TRT compiles the ONNX graph once; LoRA adapters use IRefitter weight swapping.
"""

import argparse
import json
import re
import sys
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from acestep_trt_common import (
    ExportMetadata,
    TRT_FP16_WEIGHT_ALLOWLIST_PATTERNS,
    classify_tensor_names,
    write_export_metadata,
    write_json,
)

# We need the model's own code
# The model dir contains modeling_acestep_v15_xl_base.py


class PatchEmbedLinear(nn.Module):
    """Replace Conv1d(C_in, C_out, K, stride=K) with reshape + Linear.

    TensorRT has historically lacked reliable kernels for these 1D convolution
    patch shapes in some precision modes. This is mathematically equivalent:
      Conv1d: input[B, C_in, T] → output[B, C_out, T//K]
      Linear: input[B, C_in, T] → unfold[B, T//K, C_in*K] → Linear → [B, C_out, T//K]
    """
    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        C_out, C_in, K = conv.weight.shape
        self.kernel_size = K
        self.linear = nn.Linear(
            C_in * K,
            C_out,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        # Conv weight [C_out, C_in, K] → Linear weight [C_out, C_in*K]
        if not conv.weight.is_meta:
            self.linear.weight.data = conv.weight.data.reshape(C_out, -1).clone()
        if conv.bias is not None and not conv.bias.is_meta:
            self.linear.bias.data = conv.bias.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] (surrounding Lambda transposes replaced with Identity)
        B, T, C = x.shape
        K = self.kernel_size
        # Unfold patches: [B, T, C] → [B, T//K, K, C] → [B, T//K, C, K] → [B, T//K, C*K]
        x = x.reshape(B, T // K, K, C).transpose(2, 3).reshape(B, T // K, C * K)
        out = self.linear(x)                   # [B, T//K, C_out]
        return out


class UnPatchLinear(nn.Module):
    """Replace ConvTranspose1d(C_in, C_out, K, stride=K) with Linear + reshape.

    TensorRT has historically lacked reliable kernels for these 1D transposed
    convolution patch shapes. This is mathematically equivalent:
      ConvTranspose1d: input[B, C_in, T//K] → output[B, C_out, T]
      Linear: input[B, T//K, C_in] → Linear → [B, T//K, C_out*K] → fold → [B, C_out, T]
    """
    def __init__(self, deconv: nn.ConvTranspose1d):
        super().__init__()
        C_in, C_out, K = deconv.weight.shape
        self.kernel_size = K
        self.C_out = C_out
        self.linear = nn.Linear(
            C_in,
            C_out * K,
            bias=deconv.bias is not None,
            device=deconv.weight.device,
            dtype=deconv.weight.dtype,
        )
        # ConvTranspose1d weight [C_in, C_out, K] → Linear weight [C_out*K, C_in]
        if not deconv.weight.is_meta:
            self.linear.weight.data = deconv.weight.data.permute(1, 2, 0).reshape(C_out * K, C_in).clone()
        if deconv.bias is not None and not deconv.bias.is_meta:
            # ConvTranspose1d bias [C_out] → Linear bias [C_out*K] (repeat per patch)
            self.linear.bias.data = deconv.bias.data.repeat_interleave(K).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T//K, C_in] (surrounding Lambda transposes replaced with Identity)
        B, T_small, C_in = x.shape
        K = self.kernel_size
        x = self.linear(x)                                  # [B, T//K, C_out*K]
        # Fold patches: [B, T//K, C_out*K] → [B, T//K, C_out, K] → [B, T//K, K, C_out] → [B, T_small*K, C_out]
        x = x.reshape(B, T_small, self.C_out, K).transpose(2, 3).reshape(B, T_small * K, self.C_out)
        return x


class DiTForwardWrapper(nn.Module):
    """
    Wrapper around AceStepDiTModel.forward() that simplifies the interface
    for ONNX export.

    ONNX inputs (6 total):
        input_latents:           [B, T, 192]  — pre-concatenated [context_latents, xt]
        enc_hidden:              [B, S, 2048] — encoder hidden states
        t:                       [B]          fp32 — current timestep
        t_r:                     [B]          fp32 — reference timestep
        attention_mask:          [B, T]       int64 — self-attn padding mask (1=attend, 0=pad)
        encoder_attention_mask:  [B, S]       int64 — cross-attn padding mask (1=attend, 0=pad)

    ONNX output:
        velocity:                [B, T, 64]   — predicted flow velocity

    The two int64 masks are passed straight through to the underlying DiT model
    as ``attention_mask`` and ``encoder_attention_mask``. The model converts
    them to 4D additive biases internally and applies its trained sliding-window
    pattern on even self-attention layers (layer_type=0). Position IDs and
    RoPE are still computed internally from T and S.

    Without these masks the DiT cross-attends to the ``null_cond_vec`` padding
    that the C++ pipeline packs into ``enc_hidden`` for batched CFG / multi-
    request generation, diluting the timbre token (and lyric/text tokens) and
    effectively ignoring the reference audio. Passing the masks lets the model
    mask out padding positions the same way the GGML path does via its
    ``ca_mask`` graph input.
    """

    def __init__(self, dit_model, precision="q8map-fp16"):
        super().__init__()
        self.dit = dit_model
        self.config = dit_model.config
        self.precision = precision

    def forward(self, input_latents, enc_hidden, t, t_r,
                attention_mask, encoder_attention_mask):
        """
        Args:
            input_latents:           [B, T, 192] — concatenated context + noise latents
            enc_hidden:              [B, S, 2048] — encoder hidden states
            t:                       [B] — timestep
            t_r:                     [B] — reference timestep
            attention_mask:          [B, T] int64 — self-attn padding mask (1=real, 0=pad)
            encoder_attention_mask:  [B, S] int64 — cross-attn padding mask (1=real, 0=pad)
        Returns:
            velocity:                [B, T, 64] — predicted velocity
        """
        # Split input_latents into context (128 dim) and noise (64 dim)
        context_latents = input_latents[:, :, :128]
        hidden_states = input_latents[:, :, 128:]

        outputs = self.dit(
            hidden_states=hidden_states,
            timestep=t,
            timestep_r=t_r,
            attention_mask=attention_mask,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=False,
            past_key_values=None,
            output_attentions=False,
        )

        # outputs[0] is the velocity prediction [B, T, 64]
        velocity = outputs[0]
        return velocity


def replace_conv_with_linear(dit_model):
    """Replace Conv1d/ConvTranspose1d with equivalent Linear ops.

    PatchEmbedLinear/UnPatchLinear reformulate patch convolutions as
    reshape+matmul, which TensorRT handles more consistently.

    Must be called for all supported precision recipes.
    """
    if hasattr(dit_model, 'proj_in') and isinstance(dit_model.proj_in, nn.Sequential):
        for i, mod in enumerate(dit_model.proj_in):
            if isinstance(mod, nn.Conv1d):
                dit_model.proj_in[i] = PatchEmbedLinear(mod)
                print(f"[export_dit] Conv→Linear: proj_in[{i}] Conv1d → PatchEmbedLinear")
        if len(dit_model.proj_in) == 3:
            dit_model.proj_in[0] = nn.Identity()
            dit_model.proj_in[2] = nn.Identity()

    if hasattr(dit_model, 'proj_out') and isinstance(dit_model.proj_out, nn.Sequential):
        for i, mod in enumerate(dit_model.proj_out):
            if isinstance(mod, nn.ConvTranspose1d):
                dit_model.proj_out[i] = UnPatchLinear(mod)
                print(f"[export_dit] Conv→Linear: proj_out[{i}] ConvTranspose1d → UnPatchLinear")
        if len(dit_model.proj_out) == 3:
            dit_model.proj_out[0] = nn.Identity()
            dit_model.proj_out[2] = nn.Identity()

    return dit_model


def _safetensor_paths(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        return [model_dir / name for name in sorted(set(index["weight_map"].values()))]
    if single_path.exists():
        return [single_path]
    raise SystemExit(f"[export_dit] ERROR: No model.safetensors found in {model_dir}")


def _load_safetensor_header(path: Path) -> tuple[dict, int]:
    import struct

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def _build_safetensor_index(model_dir: Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for path in _safetensor_paths(model_dir):
        header, data_start = _load_safetensor_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            entries[name] = {
                "path": path,
                "dtype": info["dtype"],
                "shape": list(info["shape"]),
                "offset": data_start + int(begin),
                "length": int(end) - int(begin),
            }
    return entries


def _copy_file_range(src_path: Path, src_offset: int, length: int, out, chunk_bytes: int = 16 * 1024 * 1024) -> None:
    remaining = int(length)
    with open(src_path, "rb") as src:
        src.seek(int(src_offset))
        while remaining:
            block = src.read(min(chunk_bytes, remaining))
            if not block:
                raise IOError(f"unexpected EOF while copying {src_path}")
            out.write(block)
            remaining -= len(block)


def _shape_numel(shape: list[int] | tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _reserve_external_range(out, length: int) -> tuple[int, int]:
    offset = out.tell()
    if length:
        out.seek(length - 1, os.SEEK_CUR)
        out.write(b"\0")
        out.flush()
    return offset, int(length)


def _write_numpy_external(out, arr) -> tuple[int, int]:
    import numpy as np

    contiguous = np.ascontiguousarray(arr)
    offset = out.tell()
    out.write(memoryview(contiguous).cast("B"))
    return offset, int(contiguous.nbytes)


def _constant_tensor_from_node(node):
    import numpy as np
    from onnx import numpy_helper

    if node.op_type != "Constant" or len(node.output) != 1:
        return None
    name = node.output[0]
    for attr in node.attribute:
        if attr.name == "value":
            tensor = type(attr.t)()
            tensor.CopyFrom(attr.t)
            tensor.name = name
            return tensor
        if attr.name == "value_int":
            return numpy_helper.from_array(np.asarray(attr.i, dtype=np.int64), name=name)
        if attr.name == "value_ints":
            return numpy_helper.from_array(np.asarray(list(attr.ints), dtype=np.int64), name=name)
        if attr.name == "value_float":
            return numpy_helper.from_array(np.asarray(attr.f, dtype=np.float32), name=name)
        if attr.name == "value_floats":
            return numpy_helper.from_array(np.asarray(list(attr.floats), dtype=np.float32), name=name)
    return None


def _static_tensor_from_node(node, constant_values: dict):
    import numpy as np
    from onnx import numpy_helper

    tensor = _constant_tensor_from_node(node)
    if tensor is not None:
        return tensor
    if len(node.output) != 1:
        return None
    name = node.output[0]
    try:
        if node.op_type == "Identity" and len(node.input) == 1 and node.input[0] in constant_values:
            return numpy_helper.from_array(np.asarray(constant_values[node.input[0]]), name=name)
        if node.op_type == "Reshape" and len(node.input) >= 2:
            if node.input[0] not in constant_values or node.input[1] not in constant_values:
                return None
            shape = [int(x) for x in np.asarray(constant_values[node.input[1]]).reshape(-1)]
            arr = np.reshape(constant_values[node.input[0]], shape)
            return numpy_helper.from_array(np.asarray(arr), name=name)
    except Exception:
        return None
    return None


def _write_tensor_proto_external(tensor, data_path: str, data_location: str):
    import numpy as np
    from onnx import TensorProto, numpy_helper

    if tensor.data_type == TensorProto.STRING:
        return None
    arr = numpy_helper.to_array(tensor)
    if arr.dtype == np.dtype("O"):
        return None
    with open(data_path, "ab") as data_out:
        offset, length = _write_numpy_external(data_out, arr)
    return _external_initializer(
        tensor.name,
        int(tensor.data_type),
        [int(dim) for dim in tensor.dims],
        data_location,
        offset,
        length,
    )


def _fold_constant_nodes_to_initializers_for_trt(
    model,
    data_path: str | None = None,
    data_location: str | None = None,
) -> dict:
    """Convert static Constant nodes into initializers for TensorRT parser rules."""
    from onnx import TensorProto

    consumed_names = {input_name for node in model.graph.node for input_name in node.input if input_name}
    graph_output_names = {output.name for output in model.graph.output}
    initializer_names = {init.name for init in model.graph.initializer}
    constant_values = _constant_arrays(model)
    folded = []
    skipped = []
    removed_dead_constants = []
    new_initializers = []
    new_nodes = []

    for node in model.graph.node:
        tensor = _static_tensor_from_node(node, constant_values)
        if tensor is not None and node.output and node.output[0] not in consumed_names and node.output[0] not in graph_output_names:
            if node.op_type == "Constant":
                removed_dead_constants.append(node.output[0])
                continue
            new_nodes.append(node)
            continue
        if tensor is None:
            new_nodes.append(node)
            continue

        name = node.output[0]
        if name in initializer_names:
            new_nodes.append(node)
            skipped.append(name)
            continue
        if data_path and data_location:
            initializer = _write_tensor_proto_external(tensor, data_path, data_location)
            if initializer is None:
                new_nodes.append(node)
                skipped.append(name)
                continue
        else:
            initializer = type(tensor)()
            initializer.CopyFrom(tensor)
            initializer.data_location = TensorProto.DEFAULT
        new_initializers.append(initializer)
        initializer_names.add(name)
        try:
            from onnx import numpy_helper
            constant_values[name] = numpy_helper.to_array(tensor)
        except Exception:
            pass
        folded.append(name)

    if folded or removed_dead_constants:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        model.graph.initializer.extend(new_initializers)

    return {
        "constant_nodes_folded_to_initializers": len(folded),
        "folded_initializer_names": folded,
        "dead_constant_nodes_removed": len(removed_dead_constants),
        "skipped_constant_names": skipped,
    }


def _source_key_for_exported_param(param_name: str) -> str:
    if param_name.startswith("dit."):
        return "decoder." + param_name[len("dit."):]
    if param_name.startswith("decoder."):
        return param_name
    return "decoder." + param_name


def _graph_signature_initializer_targets(onnx_program) -> dict[str, str]:
    targets: dict[str, str] = {}
    graph_signature = getattr(getattr(onnx_program, "exported_program", None), "graph_signature", None)
    if graph_signature is None:
        return targets
    for spec in getattr(graph_signature, "input_specs", []):
        kind = str(getattr(spec, "kind", ""))
        if not (kind.endswith("PARAMETER") or kind.endswith("BUFFER")):
            continue
        arg = getattr(spec, "arg", None)
        arg_name = getattr(arg, "name", None)
        target = getattr(spec, "target", None)
        if arg_name and target:
            targets[str(arg_name)] = str(target)
    return targets


def _metadata_props(obj) -> dict[str, str]:
    raw_props = getattr(obj, "metadata_props", {})
    if hasattr(raw_props, "items"):
        return {str(key): str(value) for key, value in raw_props.items()}
    props = {}
    for prop in raw_props:
        key = getattr(prop, "key", None)
        value = getattr(prop, "value", None)
        if key is not None and value is not None:
            props[str(key)] = str(value)
    return props


def _onnx_ir_initializer_targets(onnx_program) -> dict[str, str]:
    targets_by_arg = _graph_signature_initializer_targets(onnx_program)
    result: dict[str, str] = {}
    graph = getattr(getattr(onnx_program, "model", None), "graph", None)
    if graph is None:
        return result
    initializers = getattr(graph, "initializers", {})
    items = initializers.items() if hasattr(initializers, "items") else []
    for name, init in items:
        props = _metadata_props(init)
        arg_name = props.get("pkg.torch.onnx.original_node_name") or str(name)
        target = targets_by_arg.get(arg_name)
        if target:
            result[str(name)] = target
    return result


def _onnx_ir_dtype_to_tensor_proto(dtype) -> int:
    from onnx import TensorProto

    name = str(dtype).split(".")[-1].upper()
    mapping = {
        "FLOAT": TensorProto.FLOAT,
        "FLOAT16": TensorProto.FLOAT16,
        "BFLOAT16": TensorProto.BFLOAT16,
        "INT64": TensorProto.INT64,
        "INT32": TensorProto.INT32,
    }
    if name not in mapping:
        raise SystemExit(f"unsupported ONNX IR initializer dtype: {dtype}")
    return mapping[name]


def _onnx_ir_initializer_specs(onnx_program) -> dict[str, dict]:
    graph = getattr(getattr(onnx_program, "model", None), "graph", None)
    if graph is None:
        return {}
    initializers = getattr(graph, "initializers", {})
    items = initializers.items() if hasattr(initializers, "items") else []
    specs: dict[str, dict] = {}
    for name, init in items:
        shape = [int(dim) for dim in getattr(init, "shape", [])]
        specs[str(name)] = {
            "dims": shape,
            "data_type": _onnx_ir_dtype_to_tensor_proto(getattr(init, "dtype", None)),
        }
    return specs


def _rename_graph_uses(model_proto, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return
    for node in model_proto.graph.node:
        for i, ref in enumerate(node.input):
            if ref == old_name:
                node.input[i] = new_name
    for value_info in list(model_proto.graph.input) + list(model_proto.graph.value_info) + list(model_proto.graph.output):
        if value_info.name == old_name:
            value_info.name = new_name


def _remove_graph_inputs(model_proto, names: set[str]) -> None:
    kept = [value_info for value_info in model_proto.graph.input if value_info.name not in names]
    del model_proto.graph.input[:]
    model_proto.graph.input.extend(kept)


def _remove_initializer_graph_inputs(model_proto) -> int:
    initializer_names = {init.name for init in model_proto.graph.initializer}
    before = len(model_proto.graph.input)
    _remove_graph_inputs(model_proto, initializer_names)
    return before - len(model_proto.graph.input)


def _external_initializer(name: str, data_type: int, dims: list[int], location: str, offset: int, length: int):
    from onnx import TensorProto, helper

    init = helper.make_tensor(name=name, data_type=data_type, dims=dims, vals=[])
    init.data_location = TensorProto.EXTERNAL
    del init.external_data[:]
    for key, value in (
        ("location", location),
        ("offset", str(int(offset))),
        ("length", str(int(length))),
    ):
        entry = init.external_data.add()
        entry.key = key
        entry.value = value
    return init


def _validate_external_data_artifacts(output_path: str) -> dict:
    import onnx
    from onnx import TensorProto

    model = onnx.load(output_path, load_external_data=False)
    output_dir = Path(output_path).parent
    missing_external = []
    embedded = []
    by_location: dict[str, dict] = {}

    for init in model.graph.initializer:
        if init.data_location != TensorProto.EXTERNAL:
            embedded.append(init.name)
            continue
        entries = {item.key: item.value for item in init.external_data}
        location = entries.get("location")
        offset = int(entries.get("offset", "0"))
        length_value = entries.get("length")
        if not location or length_value is None:
            missing_external.append(init.name)
            continue
        length = int(length_value)
        info = by_location.setdefault(location, {"max_end": 0, "bytes": 0, "count": 0})
        info["max_end"] = max(info["max_end"], offset + length)
        info["bytes"] += length
        info["count"] += 1

    if embedded:
        raise SystemExit(
            "low-memory export left embedded ONNX initializer(s); expected all weights external: "
            + ", ".join(embedded[:12])
            + (" ..." if len(embedded) > 12 else "")
        )
    if missing_external:
        raise SystemExit(
            "low-memory export wrote initializer(s) without complete external-data metadata: "
            + ", ".join(missing_external[:12])
            + (" ..." if len(missing_external) > 12 else "")
        )
    if not by_location:
        raise SystemExit("low-memory export produced no external initializers")

    files = []
    for location, info in sorted(by_location.items()):
        path = output_dir / location
        if not path.is_file():
            raise SystemExit(f"low-memory export is missing external data file: {path}")
        size = path.stat().st_size
        if size < info["max_end"]:
            raise SystemExit(
                f"external data file is truncated: {path} has {size} bytes, "
                f"but initializers reference up to {info['max_end']} bytes"
            )
        files.append(
            {
                "path": location,
                "size_bytes": size,
                "referenced_bytes": info["bytes"],
                "initializer_count": info["count"],
            }
        )

    return {
        "initializer_count": len(model.graph.initializer),
        "external_data_files": files,
        "external_data_total_size_bytes": sum(item["size_bytes"] for item in files),
        "external_data_referenced_bytes": sum(item["referenced_bytes"] for item in files),
    }


def _tensor_value_info_shapes(model) -> dict[str, list[int | None]]:
    shapes: dict[str, list[int | None]] = {}
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int | None] = []
        for dim in tensor_type.shape.dim:
            dims.append(int(dim.dim_value) if dim.HasField("dim_value") else None)
        shapes[value_info.name] = dims
    return shapes


def _constant_arrays(model):
    import onnx
    import numpy as np
    from onnx import numpy_helper

    values = {}
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            values[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                values[node.output[0]] = numpy_helper.to_array(attr.t)
            elif attr.name == "value_int":
                values[node.output[0]] = np.asarray(attr.i, dtype=np.int64)
            elif attr.name == "value_ints":
                values[node.output[0]] = np.asarray(list(attr.ints), dtype=np.int64)
    return values


def _static_int_array(constant_arrays: dict, name: str):
    import numpy as np

    if not name:
        return None
    arr = constant_arrays.get(name)
    if arr is None or not np.issubdtype(arr.dtype, np.integer):
        return None
    return arr.astype(np.int64, copy=False)


def _node_attr_int(node, name: str, default: int) -> int:
    import onnx

    for attr in node.attribute:
        if attr.name == name and attr.type == onnx.AttributeProto.INT:
            return int(attr.i)
    return int(default)


def _normalize_axis(axis: int, rank: int | None) -> int:
    if rank is not None and axis < 0:
        return int(axis + rank)
    return int(axis)


def _rewrite_split_to_sequence_for_trt(model) -> dict:
    """Lower SplitToSequence+SequenceAt pairs to TensorRT-supported Split nodes."""
    from collections import defaultdict
    from onnx import TensorProto, helper

    constant_arrays = _constant_arrays(model)
    shapes = _tensor_value_info_shapes(model)
    consumers: dict[str, list] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)

    remove_ids = set()
    inserted_split_count = 0
    removed_sequence_at_count = 0
    new_nodes = []

    for node in model.graph.node:
        if id(node) in remove_ids:
            continue
        if node.op_type != "SplitToSequence":
            new_nodes.append(node)
            continue
        if len(node.input) < 1 or len(node.output) != 1:
            raise SystemExit(f"cannot rewrite malformed SplitToSequence node: {node.name or node.output}")

        sequence_name = node.output[0]
        sequence_consumers = consumers.get(sequence_name, [])
        sequence_at_nodes = [consumer for consumer in sequence_consumers if consumer.op_type == "SequenceAt"]
        unsupported = [consumer for consumer in sequence_consumers if consumer.op_type != "SequenceAt"]
        if unsupported:
            names = ", ".join((consumer.name or consumer.op_type) for consumer in unsupported[:6])
            raise SystemExit(
                f"cannot rewrite SplitToSequence {node.name or sequence_name}: "
                f"unsupported sequence consumer(s): {names}"
            )
        if not sequence_at_nodes:
            inserted_split_count += 1
            continue

        raw_axis = _node_attr_int(node, "axis", 0)
        keepdims = _node_attr_int(node, "keepdims", 1)
        input_shape = shapes.get(node.input[0])
        output_shape = shapes.get(sequence_at_nodes[0].output[0]) if sequence_at_nodes[0].output else None
        rank = len(input_shape) if input_shape is not None else (len(output_shape) if output_shape is not None else None)
        axis = _normalize_axis(raw_axis, rank)

        indexed_outputs: dict[int, list[str]] = defaultdict(list)
        max_index = -1
        for sequence_at in sequence_at_nodes:
            if len(sequence_at.output) != 1:
                raise SystemExit(f"cannot rewrite malformed SequenceAt node: {sequence_at.name or sequence_at.output}")
            if len(sequence_at.input) >= 2 and sequence_at.input[1]:
                index_arr = _static_int_array(constant_arrays, sequence_at.input[1])
                if index_arr is None or index_arr.size != 1:
                    raise SystemExit(f"cannot rewrite dynamic SequenceAt index: {sequence_at.name or sequence_at.output}")
                index = int(index_arr.reshape(-1)[0])
            else:
                index = 0
            indexed_outputs[index].append(sequence_at.output[0])
            max_index = max(max_index, index)
            remove_ids.add(id(sequence_at))

        split_input_name = node.input[1] if len(node.input) >= 2 and node.input[1] else ""
        split_arr = _static_int_array(constant_arrays, split_input_name)
        split_inputs = [node.input[0]]
        split_attrs = {"axis": axis}
        if split_arr is not None and split_arr.ndim > 0 and split_arr.size > 1:
            output_count = int(split_arr.size)
            split_inputs.append(split_input_name)
        else:
            if split_arr is not None and split_arr.size == 1:
                split_size = int(split_arr.reshape(-1)[0])
            else:
                split_size = 1
            dim = input_shape[axis] if input_shape is not None and 0 <= axis < len(input_shape) else None
            if dim is not None and split_size > 0:
                output_count = int((dim + split_size - 1) // split_size)
            else:
                output_count = max_index + 1
            split_attrs["num_outputs"] = output_count

        if output_count <= 0:
            raise SystemExit(f"cannot rewrite SplitToSequence with zero outputs: {node.name or sequence_name}")
        normalized_outputs = {}
        for index, outputs in list(indexed_outputs.items()):
            normalized_index = index + output_count if index < 0 else index
            if normalized_index < 0 or normalized_index >= output_count:
                raise SystemExit(
                    f"SequenceAt index {index} is out of range for {node.name or sequence_name} "
                    f"with {output_count} split outputs"
                )
            normalized_outputs[normalized_index] = outputs

        split_outputs = [f"{sequence_name}_trt_split_{i}" for i in range(output_count)]
        new_nodes.append(
            helper.make_node(
                "Split",
                split_inputs,
                split_outputs,
                name=node.name or f"{sequence_name}/SplitForTensorRT",
                **split_attrs,
            )
        )
        inserted_split_count += 1

        for index in range(output_count):
            for output_name in normalized_outputs.get(index, []):
                if keepdims:
                    new_nodes.append(
                        helper.make_node(
                            "Identity",
                            [split_outputs[index]],
                            [output_name],
                            name=f"{output_name}/SequenceAtIdentity",
                        )
                    )
                else:
                    axes_name = f"{output_name}_sequence_squeeze_axes"
                    new_nodes.append(
                        helper.make_node(
                            "Constant",
                            [],
                            [axes_name],
                            name=f"{output_name}/SequenceSqueezeAxes",
                            value=helper.make_tensor(axes_name, TensorProto.INT64, [1], [axis]),
                        )
                    )
                    new_nodes.append(
                        helper.make_node(
                            "Squeeze",
                            [split_outputs[index], axes_name],
                            [output_name],
                            name=f"{output_name}/SequenceAtSqueeze",
                        )
                    )
                removed_sequence_at_count += 1

    if inserted_split_count:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    return {
        "split_to_sequence_rewritten": inserted_split_count,
        "sequence_at_removed": removed_sequence_at_count,
    }


def _write_param_tensor_external(out, tensor: torch.Tensor, limit_mb: int = 256) -> tuple[int, int]:
    import numpy as np

    param = tensor.detach().cpu().contiguous()
    nbytes = int(param.numel() * param.element_size())
    if nbytes > limit_mb * 1024 * 1024:
        raise SystemExit(
            f"cannot materialize fallback parameter of {nbytes / 1e6:.1f} MB; "
            "expected to stream it from safetensors instead"
        )
    offset = out.tell()
    arr = np.asarray(param.numpy())
    out.write(memoryview(arr).cast("B"))
    return offset, nbytes


def _read_f32_source_array(source: dict, limit_mb: int = 256):
    import numpy as np

    if source["dtype"] != "F32":
        raise SystemExit(f"derived DiT tensor streaming expects FP32 safetensors, got {source['dtype']}")
    if int(source["length"]) > limit_mb * 1024 * 1024:
        raise SystemExit(
            f"refusing to materialize derived tensor source of {int(source['length']) / 1e6:.1f} MB"
        )
    with open(source["path"], "rb") as f:
        f.seek(int(source["offset"]))
        block = f.read(int(source["length"]))
    if len(block) != int(source["length"]):
        raise IOError(f"unexpected EOF while reading {source['path']}")
    return np.frombuffer(block, dtype="<f4").reshape(source["shape"]).copy()


def _match_derived_array_to_dims(name: str, arr, dims: list[int]):
    if list(arr.shape) == dims:
        return arr
    if arr.ndim == 2 and list(arr.T.shape) == dims:
        return arr.T.copy()
    raise SystemExit(f"derived tensor shape mismatch for {name}: derived {list(arr.shape)} vs ONNX {dims}")


def _derived_dit_array_for_exported_param(name: str, source_index: dict[str, dict], dims: list[int]):
    import numpy as np

    match = re.match(r"^dit\.proj_in\.([0-9]+)\.linear\.(weight|bias)$", name)
    if match:
        idx, kind = match.groups()
        source_name = f"decoder.proj_in.{idx}.{'weight' if kind == 'weight' else 'bias'}"
        source = source_index.get(source_name)
        if source is None:
            return None
        arr = _read_f32_source_array(source)
        if kind == "weight":
            arr = arr.reshape(arr.shape[0], -1)
        return _match_derived_array_to_dims(name, arr.astype(np.float32, copy=False), dims)

    match = re.match(r"^dit\.proj_out\.([0-9]+)\.linear\.(weight|bias)$", name)
    if match:
        idx, kind = match.groups()
        source_name = f"decoder.proj_out.{idx}.{'weight' if kind == 'weight' else 'bias'}"
        source = source_index.get(source_name)
        if source is None:
            return None
        arr = _read_f32_source_array(source)
        if kind == "weight":
            arr = arr.transpose(1, 2, 0).reshape(arr.shape[1] * arr.shape[2], arr.shape[0])
        else:
            weight_source = source_index.get(f"decoder.proj_out.{idx}.weight")
            if weight_source is None or len(weight_source["shape"]) != 3:
                raise SystemExit(f"cannot derive repeat factor for {name}")
            arr = np.repeat(arr, int(weight_source["shape"][2]))
        return _match_derived_array_to_dims(name, arr.astype(np.float32, copy=False), dims)

    return None


def _derived_runtime_buffer_array(name: str, wrapper: nn.Module, dims: list[int]):
    import numpy as np

    if not (
        name.endswith(".rotary_emb.inv_freq")
        or name.endswith(".rotary_emb.original_inv_freq")
    ):
        return None
    dit = getattr(wrapper, "dit", None)
    rotary = getattr(dit, "rotary_emb", None)
    config = getattr(dit, "config", None)
    if rotary is None or config is None:
        return None
    try:
        rotary_cpu = type(rotary)(config=config, device="cpu")
        buffer_name = name.rsplit(".", 1)[-1]
        tensor = getattr(rotary_cpu, buffer_name)
    except Exception as exc:
        raise SystemExit(f"cannot derive low-memory rotary buffer {name}: {exc}") from None
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return _match_derived_array_to_dims(name, arr, dims)

def _float32_to_bfloat16_uint16(arr):
    import numpy as np

    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    bits = f32.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return np.ascontiguousarray((rounded >> np.uint32(16)).astype(np.uint16))


def _w8a8_plugin_boundary_dtype() -> str:
    # GGML Q8_0 linears produce FP32 tensors, and RMSNorm/AdaLN/residual math
    # stays FP32. Keep that as the default for audio quality. FP16/BF16 are
    # explicit experimental modes for profiling or overflow investigations.
    value = os.environ.get("HOTSTEP_W8A8_PLUGIN_IO_DTYPE", "FP32").strip().upper()
    if value in {"FP16", "FLOAT16", "HALF"}:
        return "FP16"
    if value in {"BF16", "BFLOAT16"}:
        return "BF16"
    if value in {"FP32", "FLOAT", "FLOAT32"}:
        return "FP32"
    raise SystemExit(
        "HOTSTEP_W8A8_PLUGIN_IO_DTYPE must be one of FP16, BF16, or FP32; "
        f"got {value!r}"
    )


def _onnx_dtype_for_plugin_boundary(TensorProto, dtype_name: str) -> int:
    if dtype_name == "BF16":
        return TensorProto.BFLOAT16
    if dtype_name == "FP16":
        return TensorProto.FLOAT16
    return TensorProto.FLOAT


def _numpy_boundary_array(arr, dtype_name: str):
    import numpy as np

    if dtype_name == "BF16":
        return _float32_to_bfloat16_uint16(arr)
    if dtype_name == "FP16":
        return np.ascontiguousarray(arr, dtype=np.float16)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _stream_f32_as_bf16(source: dict, out, chunk_bytes: int) -> tuple[int, int]:
    import numpy as np

    if source["dtype"] != "F32":
        raise SystemExit(f"low-memory BF16 streaming expects FP32 safetensors, got {source['dtype']}")
    offset = out.tell()
    remaining = int(source["length"])
    if remaining % 4:
        raise SystemExit(f"FP32 safetensor byte length is not divisible by 4 for {source['path']}")
    max_bytes = max(4, int(chunk_bytes) // 4 * 4)
    with open(source["path"], "rb") as f:
        f.seek(int(source["offset"]))
        while remaining:
            block = f.read(min(max_bytes, remaining))
            if not block:
                raise IOError(f"unexpected EOF while reading {source['path']}")
            arr = np.frombuffer(block, dtype="<f4")
            out_arr = _float32_to_bfloat16_uint16(arr)
            out.write(memoryview(out_arr).cast("B"))
            remaining -= len(block)
    return offset, int(source["length"] // 2)


def _source_layout(source: dict, onnx_dims: list[int]) -> tuple[list[int], bool]:
    source_dims = [int(x) for x in source["shape"]]
    if source_dims == onnx_dims:
        return source_dims, False
    if len(source_dims) == 2 and source_dims[::-1] == onnx_dims:
        return source_dims, True
    raise SystemExit(f"shape mismatch: ONNX {onnx_dims} vs safetensors {source_dims}")


def _read_f32_rows(source: dict, source_dims: list[int], row_start: int, rows: int):
    import numpy as np

    if source["dtype"] != "F32":
        raise SystemExit(f"low-memory streaming expects FP32 safetensors, got {source['dtype']}")
    row_width = _shape_numel(source_dims[1:])
    nbytes = int(rows) * row_width * 4
    with open(source["path"], "rb") as f:
        f.seek(int(source["offset"]) + int(row_start) * row_width * 4)
        block = f.read(nbytes)
    if len(block) != nbytes:
        raise IOError(f"unexpected EOF while reading {source['path']}")
    return np.frombuffer(block, dtype="<f4").reshape((int(rows), *source_dims[1:]))


def _rows_per_chunk(shape: list[int], chunk_bytes: int) -> int:
    row_bytes = max(1, _shape_numel(shape[1:]) * 4)
    return max(1, int(chunk_bytes) // row_bytes)


def _stream_f32_as_f16(source: dict, out, chunk_bytes: int) -> tuple[int, int]:
    import numpy as np

    if source["dtype"] != "F32":
        raise SystemExit(f"low-memory FP16 streaming expects FP32 safetensors, got {source['dtype']}")
    offset = out.tell()
    remaining = int(source["length"])
    if remaining % 4:
        raise SystemExit(f"FP32 safetensor byte length is not divisible by 4 for {source['path']}")
    max_bytes = max(4, int(chunk_bytes) // 4 * 4)
    with open(source["path"], "rb") as f:
        f.seek(int(source["offset"]))
        while remaining:
            block = f.read(min(max_bytes, remaining))
            if not block:
                raise IOError(f"unexpected EOF while reading {source['path']}")
            arr = np.frombuffer(block, dtype="<f4")
            out_arr = arr.astype("<f2")
            out.write(memoryview(out_arr).cast("B"))
            remaining -= len(block)
    return offset, int(source["length"] // 2)


def _stream_f32_transposed(
    source: dict,
    source_dims: list[int],
    data_path: str,
    out,
    dst_dtype,
    chunk_bytes: int,
    transform=None,
) -> tuple[int, int]:
    import numpy as np

    if len(source_dims) != 2:
        raise SystemExit(f"streamed transpose only supports 2D tensors, got {source_dims}")
    dst_shape = (int(source_dims[1]), int(source_dims[0]))
    dtype = np.dtype(dst_dtype)
    offset, length = _reserve_external_range(out, _shape_numel(dst_shape) * dtype.itemsize)
    mapped = np.memmap(data_path, dtype=dtype, mode="r+", offset=offset, shape=dst_shape, order="C")
    rows_per_chunk = _rows_per_chunk(source_dims, chunk_bytes)
    for row_start in range(0, source_dims[0], rows_per_chunk):
        rows = min(rows_per_chunk, source_dims[0] - row_start)
        chunk = _read_f32_rows(source, source_dims, row_start, rows)
        converted = transform(chunk, row_start) if transform is not None else chunk.astype(dtype, copy=False)
        mapped[:, row_start:row_start + rows] = converted.T
    mapped.flush()
    del mapped
    out.seek(offset + length)
    return offset, int(length)


def _write_low_memory_precision_manifest(
    output_path: str,
    wrapper: nn.Module,
    precision: str,
    downcast_to_lowp: list[str],
    quantized_to_int8: list[str],
    rewrite_report: dict | None = None,
    w8_axes: dict[str, int] | None = None,
    w8a8_rotated_by_name: dict[str, bool] | None = None,
    w8a8_group_size: int | None = None,
    w8a8_h_name: str | None = None,
) -> dict:
    all_param_names = [name for name, _ in wrapper.named_parameters()]
    matrix_param_names = [name for name, p in wrapper.named_parameters() if p.dim() >= 2]
    non_matrix_param_names = sorted(set(all_param_names) - set(matrix_param_names))
    fp16_names, fp32_matrix_names = classify_tensor_names(matrix_param_names)
    pattern_match_counts = {
        pattern: sum(1 for name in matrix_param_names if re.match(pattern, name))
        for pattern in TRT_FP16_WEIGHT_ALLOWLIST_PATTERNS
    }
    unmatched_patterns = [pattern for pattern, count in pattern_match_counts.items() if count == 0]
    preserved_names = sorted(fp32_matrix_names + non_matrix_param_names)
    report = {
        "precision_policy": precision,
        "matched_allowlist": fp16_names if precision in {"q8map-fp16", "w8a8"} else [],
        "downcast_to_lowp": sorted(downcast_to_lowp),
        "quantized_to_int8": sorted(quantized_to_int8),
        "preserved_fp32": sorted(all_param_names) if precision == "fp32" else preserved_names,
        "missing_initializers": [],
        "allowlist_patterns": TRT_FP16_WEIGHT_ALLOWLIST_PATTERNS,
        "allowlist_pattern_match_counts": pattern_match_counts,
        "unmatched_allowlist_patterns": unmatched_patterns,
        "all_parameter_count": len(all_param_names),
        "matrix_parameter_count": len(matrix_param_names),
        "non_matrix_parameter_count": len(non_matrix_param_names),
        "preserved_fp32_matrix": fp32_matrix_names,
        "preserved_fp32_non_matrix": non_matrix_param_names,
        "low_memory_export": True,
    }
    if rewrite_report:
        report["rewrite"] = rewrite_report
    if precision == "w8a8":
        report["w8a8_axis_by_name"] = w8_axes or {}
        report["weight_activation_quantization"] = {
            "weight_dtype": "int8",
            "activation_dtype": "int8",
            "compute_dtype": "fp32",
            "weight_scale_dtype": "fp32",
            "weight_scale_granularity": "per-output-channel",
            "activation_scale_dtype": "fp32",
            "activation_scale_granularity": "per-row",
            "scheme": "symmetric",
            "q_range": [-127, 127],
            "matmul": "ConvRotInt8Linear",
            "plugin_contract_version": 2,
            "plugin_boundary_dtype": _w8a8_plugin_boundary_dtype().lower(),
            "plugin_input_dtype_default": _w8a8_plugin_boundary_dtype().lower(),
            "plugin_output_dtype_default": _w8a8_plugin_boundary_dtype().lower(),
            "plugin_output_dtype_fallback": "fp32",
        }
        rotated_by_name = w8a8_rotated_by_name or {}
        report["convrot"] = {
            "enabled": any(rotated_by_name.values()) if rotated_by_name else False,
            "group_size": int(w8a8_group_size) if w8a8_group_size is not None else 0,
            "hadamard_initializer": w8a8_h_name or "",
            "rotated_weights": sorted(name for name, r in rotated_by_name.items() if r),
            "skipped_weights": sorted(name for name, r in rotated_by_name.items() if not r),
        }
    if precision in {"q8map-fp16", "w8a8"}:
        if not fp16_names:
            raise SystemExit(f"hardcoded DiT {precision} allowlist matched zero exported parameters")
        if unmatched_patterns:
            raise SystemExit(
                f"hardcoded DiT {precision} allowlist pattern(s) matched zero exported parameters: "
                + ", ".join(unmatched_patterns)
            )
    write_json(Path(output_path).with_suffix(".precision-manifest.json"), report)
    write_json(Path(output_path).with_suffix(".precision.json"), report)
    return report


def _externalize_initializers_from_safetensors(
    onnx_program,
    output_path: str,
    model_dir: Path,
    wrapper: nn.Module,
    precision: str = "fp32",
    chunk_mb: int = 16,
    graph_shell_path: str | None = None,
) -> dict:
    """Attach ONNX weights by streaming raw safetensors bytes into external data."""
    import onnx
    import numpy as np
    from onnx import TensorProto

    model_proto = onnx.load(graph_shell_path or output_path, load_external_data=False)
    target_by_init = _onnx_ir_initializer_targets(onnx_program)
    if not target_by_init:
        raise SystemExit("low-memory export could not recover PyTorch parameter names from ONNX metadata")
    initializer_specs = _onnx_ir_initializer_specs(onnx_program)
    if not initializer_specs:
        raise SystemExit("low-memory export could not recover initializer specs from ONNX IR metadata")

    if precision not in {"fp32", "q8map-fp16", "w8a8"}:
        raise SystemExit(f"unknown low-memory precision policy: {precision}")

    source_index = _build_safetensor_index(model_dir)
    wrapper_params = dict(wrapper.named_parameters())
    wrapper_tensors = {
        **wrapper_params,
        **dict(wrapper.named_buffers()),
    }
    matrix_param_names = [name for name, p in wrapper.named_parameters() if p.dim() >= 2]
    fp16_names, _ = classify_tensor_names(matrix_param_names)
    selected_names = set(fp16_names) if precision in {"q8map-fp16", "w8a8"} else set()
    chunk_bytes = max(1, int(chunk_mb)) * 1024 * 1024

    data_path = output_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
    data_location = os.path.basename(data_path)

    records = []
    removed_inputs = set()

    for init_name, target_name in sorted(target_by_init.items()):
        init_spec = initializer_specs.get(init_name)
        if init_spec is None:
            continue
        if init_spec["data_type"] != TensorProto.FLOAT:
            raise SystemExit(f"low-memory export currently supports FP32 initializers only, got {init_name}")

        dims = [int(dim) for dim in init_spec["dims"]]
        source_key = _source_key_for_exported_param(target_name)
        source = source_index.get(source_key)
        source_dims = None
        transposed = False
        derived_array = None
        if source is not None:
            if source["dtype"] != "F32":
                raise SystemExit(
                    f"low-memory export expected FP32 safetensors after GGUF conversion, "
                    f"but {source_key} is {source['dtype']}"
                )
            source_dims, transposed = _source_layout(source, dims)
        else:
            derived_array = _derived_dit_array_for_exported_param(target_name, source_index, dims)
            if derived_array is None:
                derived_array = _derived_runtime_buffer_array(target_name, wrapper, dims)
            if derived_array is not None:
                source_dims = [int(x) for x in derived_array.shape]
            else:
                param = wrapper_tensors.get(target_name)
                if param is None or getattr(param, "is_meta", False):
                    raise SystemExit(
                        f"no safetensors source or low-memory derivation found for {target_name}"
                    )
                param_shape = [int(x) for x in param.shape]
                if param_shape == dims:
                    source_dims = param_shape
                elif len(param_shape) == 2 and param_shape[::-1] == dims:
                    source_dims = param_shape
                    transposed = True
                else:
                    raise SystemExit(f"fallback parameter shape mismatch for {target_name}: {param_shape} vs {dims}")
            if derived_array is None:
                param = wrapper_tensors.get(target_name)
            else:
                param = None

        _rename_graph_uses(model_proto, init_name, target_name)
        removed_inputs.add(init_name)
        removed_inputs.add(target_name)
        records.append(
            {
                "init_name": init_name,
                "target_name": target_name,
                "dims": dims,
                "source_key": source_key,
                "source": source,
                "source_dims": source_dims,
                "transposed": transposed,
                "derived_array": derived_array,
            }
        )

    record_names = {record["target_name"] for record in records}
    selected_initializer_names = set()
    if precision in {"q8map-fp16", "w8a8"}:
        selected_initializer_names = selected_names & record_names
        missing = sorted(selected_names - selected_initializer_names)
        if missing:
            raise SystemExit(
                f"hardcoded DiT {precision} allowlist matched parameters missing from ONNX initializers: "
                + ", ".join(missing[:12])
                + (" ..." if len(missing) > 12 else "")
            )

    _remove_graph_inputs(model_proto, removed_inputs)
    w8_axes: dict[str, int] = {}
    if precision == "w8a8":
        w8_axes = _collect_w8a8_weight_axes(model_proto, selected_initializer_names)
        missing_axes = sorted(selected_initializer_names - set(w8_axes))
        if missing_axes:
            raise SystemExit(
                "hardcoded DiT W8A8 allowlist matched parameters that were not MatMul/Gemm weights: "
                + ", ".join(missing_axes[:12])
                + (" ..." if len(missing_axes) > 12 else "")
            )

    copied_fp32 = 0
    streamed_fp16 = 0
    streamed_int8 = 0
    fallback = 0
    transposed = []
    added_initializers = []
    downcast_to_lowp = []
    quantized_to_int8 = []
    scale_names: dict[str, str] = {}
    zero_point_names: dict[str, str] = {}
    w8a8_boundary_dtype = _w8a8_plugin_boundary_dtype() if precision == "w8a8" else "FP16"
    w8a8_boundary_tensorproto = _onnx_dtype_for_plugin_boundary(TensorProto, w8a8_boundary_dtype)
    # Sane quantization rule from the GGUF path: 1D tensors (norms/biases) stay
    # FP32. Even when a bias is folded into ConvRotInt8Linear, keep the ONNX
    # initializer and plugin input FP32; only activations/H use the lowp boundary
    # dtype selected by HOTSTEP_W8A8_PLUGIN_IO_DTYPE.
    # w8a8-specific bookkeeping
    w8a8_rotated_by_name: dict[str, bool] = {}
    w8a8_group_size: int | None = None
    w8a8_h_name: str | None = None

    # The w8a8 path needs a shared H initializer (the regular Hadamard
    # matrix used by both the offline weight rotation and the online
    # activation rotation subgraph). We build the NumPy array up-front
    # so the per-weight quantizer can pass it to rotate_weight(); the
    # initializer itself is written to the external data file inside the
    # loop below, alongside the per-weight INT8/scale data, so the
    # _validate_external_data_artifacts pass (which rejects embedded
    # initializers) stays happy.
    _w8a8_H_np = None
    if precision == "w8a8":
        from convrot import build_hadamard, is_valid_group_size as _convrot_valid

        w8a8_group_size = _convrot_default_group_size()
        if not _convrot_valid(w8a8_group_size):
            raise SystemExit(
                f"W8A8 streaming ConvRot group_size must be a power of 4, got {w8a8_group_size}"
            )
        w8a8_h_name = "convrot.hadamard"
        _w8a8_H_np = build_hadamard(w8a8_group_size).astype(np.float32)

    with open(data_path, "wb") as data_out:
        for record in records:
            name = record["target_name"]
            dims = record["dims"]
            source = record["source"]
            source_dims = record["source_dims"]
            is_transposed = bool(record["transposed"])
            derived_array = record["derived_array"]
            if is_transposed:
                transposed.append(name)

            if precision == "w8a8" and name in selected_initializer_names:
                # W8A8 + ConvRot streaming path. ConvRot rotates each
                # weight row along in_features, so we must materialize
                # ConvRot rotates each weight row along in_features, so we
                # must materialize the full weight in memory before
                # quantizing. Peak memory: one weight at a time
                # (~50 MB for the largest DiT matmul: 6144 × 2048 × 4).
                axis = w8_axes[name]
                # 1. Materialize the weight in ONNX storage layout (dims).
                if source is not None:
                    src_arr = _read_f32_source_array(source)  # shape == source_dims
                    if is_transposed:
                        arr = np.ascontiguousarray(src_arr.T)  # → ONNX layout
                    else:
                        arr = np.ascontiguousarray(src_arr)
                elif derived_array is not None:
                    arr = np.ascontiguousarray(derived_array, dtype=np.float32)
                else:
                    param = wrapper_tensors[name].detach().cpu().contiguous()
                    arr = param.numpy().T if is_transposed else param.numpy()
                    arr = np.ascontiguousarray(arr, dtype=np.float32)
                    fallback += 1
                # 2. Apply ConvRot rotation + per-output-channel INT8 quant.
                q_int8, scale_fp32, rotated = _quantize_w8a8_weight_array(
                    arr, axis=axis, group_size=w8a8_group_size, H=_w8a8_H_np
                )
                w8a8_rotated_by_name[name] = rotated
                # 3. Write the INT8 weight to external data.
                offset, length = _write_numpy_external(data_out, q_int8)
                added_initializers.append(
                    _external_initializer(name, TensorProto.INT8, list(q_int8.shape), data_location, offset, length)
                )
                # 4. Write the FP32 per-output-channel scale.
                scale_name = f"{name}.w8a8_scale"
                scale_offset, scale_length = _write_numpy_external(
                    data_out, np.ascontiguousarray(scale_fp32, dtype=np.float32)
                )
                added_initializers.append(
                    _external_initializer(
                        scale_name,
                        TensorProto.FLOAT,
                        [int(q_int8.shape[0])],
                        data_location,
                        scale_offset,
                        scale_length,
                    )
                )
                scale_names[name] = scale_name
                quantized_to_int8.append(name)
                streamed_int8 += 1
                continue

            if precision == "q8map-fp16" and name in selected_initializer_names:
                boundary_dtype = "FP16"
                boundary_tensorproto = TensorProto.FLOAT16
                if source is not None:
                    if is_transposed:
                        dst_dtype = "<f2" if boundary_dtype == "FP16" else ("<f4" if boundary_dtype == "FP32" else "<u2")
                        transform = None if boundary_dtype != "BF16" else (lambda chunk, _row_start: _float32_to_bfloat16_uint16(chunk))
                        offset, length = _stream_f32_transposed(source, source_dims, data_path, data_out, dst_dtype, chunk_bytes, transform=transform)
                    elif boundary_dtype == "BF16":
                        offset, length = _stream_f32_as_bf16(source, data_out, chunk_bytes)
                    elif boundary_dtype == "FP16":
                        offset, length = _stream_f32_as_f16(source, data_out, chunk_bytes)
                    else:
                        offset = data_out.tell()
                        _copy_file_range(source["path"], source["offset"], source["length"], data_out, chunk_bytes)
                        length = int(source["length"])
                elif derived_array is not None:
                    offset, length = _write_numpy_external(data_out, _numpy_boundary_array(derived_array, boundary_dtype))
                else:
                    param = wrapper_tensors[name].detach().cpu().contiguous()
                    arr = param.numpy().T if is_transposed else param.numpy()
                    offset, length = _write_numpy_external(data_out, _numpy_boundary_array(arr, boundary_dtype))
                    fallback += 1
                added_initializers.append(_external_initializer(name, boundary_tensorproto, dims, data_location, offset, length))
                if precision == "q8map-fp16":
                    downcast_to_lowp.append(name)
                streamed_fp16 += 1
                continue

            if source is not None:
                if is_transposed:
                    offset, length = _stream_f32_transposed(source, source_dims, data_path, data_out, "<f4", chunk_bytes)
                else:
                    offset = data_out.tell()
                    _copy_file_range(source["path"], source["offset"], source["length"], data_out, chunk_bytes)
                    length = int(source["length"])
                copied_fp32 += 1
            elif derived_array is not None:
                offset, length = _write_numpy_external(data_out, derived_array.astype(np.float32, copy=False))
                fallback += 1
            else:
                param = wrapper_tensors[name].detach().cpu().contiguous()
                arr = param.numpy().T if is_transposed else param.numpy()
                offset, length = _write_numpy_external(data_out, arr.astype(np.float32, copy=False))
                fallback += 1
            added_initializers.append(_external_initializer(name, TensorProto.FLOAT, dims, data_location, offset, length))

    # Write the shared ConvRot H matrix to the external data file with the
    # plugin boundary dtype. We keep the FP32 copy in memory for offline weight
    # rotation; only the ONNX boundary initializer is narrowed.
    if precision == "w8a8" and _w8a8_H_np is not None and w8a8_h_name:
        with open(data_path, "ab") as h_data_out:
            h_data_out.seek(0, os.SEEK_END)
            h_offset = h_data_out.tell()
            h_arr = _numpy_boundary_array(_w8a8_H_np, w8a8_boundary_dtype)
            h_data_out.write(memoryview(h_arr).cast("B"))
            h_length = int(h_arr.nbytes)
        added_initializers.append(
            _external_initializer(
                w8a8_h_name,
                w8a8_boundary_tensorproto,
                list(_w8a8_H_np.shape),
                data_location,
                h_offset,
                h_length,
            )
        )

    del model_proto.graph.initializer[:]
    model_proto.graph.initializer.extend(added_initializers)

    rewrite_report = {}
    if precision == "q8map-fp16":
        rewrite_report = _rewrite_fp16_weight_ops(model_proto, set(downcast_to_lowp))
    elif precision == "w8a8":
        quant_report = {
            "quantized": sorted(quantized_to_int8),
            "missing": [],
            "skipped": [],
            "scale_names": scale_names,
            "axis_by_name": {name: 0 for name in quantized_to_int8},
            "rotated_by_name": dict(w8a8_rotated_by_name),
            "convrot_group_size": int(w8a8_group_size) if w8a8_group_size is not None else 0,
            "convrot_h_name": w8a8_h_name or "",
        }
        rewrite_report = _w8a8_rewrite_dispatch(model_proto, quant_report)
        if not rewrite_report.get("rewritten_nodes"):
            raise SystemExit("W8A8 quantized weights but rewrote zero MatMul/Gemm nodes")

    sequence_rewrite_report = _rewrite_split_to_sequence_for_trt(model_proto)
    if sequence_rewrite_report["split_to_sequence_rewritten"]:
        rewrite_report["sequence_split_rewrite"] = sequence_rewrite_report
    if precision == "w8a8" and w8a8_boundary_dtype != "FP32":
        # The SplitToSequence compatibility pass may introduce new Split/
        # Identity/Squeeze nodes after the initial plugin rewrite. In lowp test
        # modes, run the island type-fix pass once more so TensorRT strongly
        # typed parsing does not see Add/Mul/etc. with mixed dtypes.
        rewrite_report["fp16_island_rewrite_after_sequence"] = _rewrite_w8a8_fp16_islands(model_proto, set(), w8a8_boundary_dtype)
    # Attention rewrite: replace decomposed SDPA (MatMul+Softmax+MatMul)
    # with a single ONNX Attention-23 op running in FP16 internally.
    # This is mandatory for the attention block — matches GGML Q8_0's
    # flash-attention precision and lets TRT dispatch to its FMHA kernel.
    # Must run AFTER the w8a8 plugin rewrite (so Q/K/V come from
    # ConvRotInt8Linear outputs) and AFTER SplitToSequence lowering, but
    # BEFORE constant folding (so the dead Q/K scaling nodes get cleaned up).
    attention_rewrite_report = rewrite_attention_to_onnx_attention_fp16(model_proto)
    if attention_rewrite_report.get("rewritten", 0):
        rewrite_report["attention_rewrite"] = attention_rewrite_report
        if attention_rewrite_report.get("skipped"):
            print(f"[export_dit] Attention rewrite: {attention_rewrite_report['rewritten']} blocks fused, "
                  f"skipped={attention_rewrite_report['skipped']}")
        else:
            print(f"[export_dit] Attention rewrite: {attention_rewrite_report['rewritten']} blocks fused")
    elif attention_rewrite_report.get("skipped"):
        print(f"[export_dit] Attention rewrite: 0 blocks fused, skipped={attention_rewrite_report['skipped']}")

    constant_fold_report = _fold_constant_nodes_to_initializers_for_trt(
        model_proto,
        data_path=data_path,
        data_location=data_location,
    )
    if constant_fold_report["constant_nodes_folded_to_initializers"]:
        rewrite_report["constant_initializer_fold"] = constant_fold_report
    removed_initializer_inputs = _remove_initializer_graph_inputs(model_proto)
    if removed_initializer_inputs:
        rewrite_report["initializer_graph_inputs_removed"] = removed_initializer_inputs

    onnx.save(model_proto, output_path)
    onnx.checker.check_model(output_path)
    artifact_report = _validate_external_data_artifacts(output_path)
    w8a8_axes_param = (
        {name: int(w8_axes[name]) for name in quantized_to_int8}
        if precision == "w8a8"
        else None
    )
    precision_report = _write_low_memory_precision_manifest(
        output_path,
        wrapper,
        precision,
        downcast_to_lowp,
        quantized_to_int8,
        rewrite_report,
        w8a8_axes_param,
        w8a8_rotated_by_name=w8a8_rotated_by_name if precision == "w8a8" else None,
        w8a8_group_size=w8a8_group_size,
        w8a8_h_name=w8a8_h_name,
    )

    manifest = {
        "version": 2,
        "onnx_path": os.path.basename(output_path),
        "weights_transposed": sorted(transposed),
        "weights_renamed": len(records),
        "external_initializers": len(model_proto.graph.initializer),
        "low_memory_export": True,
        "streamed_fp32_from_safetensors": copied_fp32,
        "streamed_fp16_from_safetensors": streamed_fp16,
        "streamed_int8_from_safetensors": streamed_int8,
        "streamed_from_safetensors": copied_fp32 + streamed_fp16 + streamed_int8,
        "materialized_fallback_parameters": fallback,
        "constant_nodes_folded_to_initializers": constant_fold_report["constant_nodes_folded_to_initializers"],
        "external_data": os.path.basename(data_path),
        "stream_chunk_mb": int(chunk_mb),
        "external_data_files": artifact_report["external_data_files"],
        "external_data_total_size_bytes": artifact_report["external_data_total_size_bytes"],
        "external_data_referenced_bytes": artifact_report["external_data_referenced_bytes"],
    }
    manifest_path = output_path + ".refit_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return {"manifest": manifest, "precision_report": precision_report}


def _node_attr(onnx, node, name: str, default):
    for attr in node.attribute:
        if attr.name != name:
            continue
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
    return default


def _producer_by_output(model) -> dict[str, object]:
    return {out: node for node in model.graph.node for out in node.output if out}


def _transpose_perm(onnx, node, rank: int | None = None) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == "perm":
            return [int(x) for x in attr.ints]
    if rank is not None:
        return list(reversed(range(rank)))
    return None


def _trace_selected_source_axis(model, input_name: str, selected_names: set[str], output_axis: int | None = None):
    import onnx

    producers = _producer_by_output(model)
    name = input_name
    axis = output_axis
    seen = set()
    while name and name not in seen:
        seen.add(name)
        if name in selected_names:
            return name, axis
        node = producers.get(name)
        if node is None:
            return None
        if node.op_type in {"Identity", "Cast"}:
            name = node.input[0] if node.input else ""
            continue
        if node.op_type == "Transpose":
            if axis is not None:
                perm = _transpose_perm(onnx, node, axis + 1)
                if perm is None or axis >= len(perm):
                    return None
                axis = int(perm[axis])
            name = node.input[0] if node.input else ""
            continue
        return None
    return None


def _trace_selected_source(model, input_name: str, selected_names: set[str]) -> str | None:
    result = _trace_selected_source_axis(model, input_name, selected_names, None)
    return result[0] if result else None


def _rewrite_fp16_weight_ops(model, fp16_initializer_names: set[str]) -> dict:
    """Route only selected MatMul/Gemm weights through FP16 compute islands."""
    import onnx
    from onnx import TensorProto, helper

    initializer_names = {init.name for init in model.graph.initializer}
    rewritten = []
    new_nodes = []

    for node in model.graph.node:
        if node.op_type not in {"Gemm", "MatMul"}:
            new_nodes.append(node)
            continue

        if node.op_type == "MatMul":
            if len(node.input) != 2 or len(node.output) != 1:
                raise SystemExit(f"cannot rewrite MatMul node with unexpected arity: {node.name or node.output}")
            weight_source = _trace_selected_source(model, node.input[1], fp16_initializer_names)
            if weight_source is None:
                new_nodes.append(node)
                continue
            out = node.output[0]
            matmul_inputs = list(node.input)
            if matmul_inputs[0] not in initializer_names and _trace_selected_source(model, matmul_inputs[0], fp16_initializer_names) is None:
                cast_name = f"{out}_input0_fp16"
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        [matmul_inputs[0]],
                        [cast_name],
                        name=f"{node.name or out}/CastInput0ToFP16",
                        to=TensorProto.FLOAT16,
                    )
                )
                matmul_inputs[0] = cast_name
            fp16_out = f"{out}_fp16"
            new_nodes.append(helper.make_node("MatMul", matmul_inputs, [fp16_out], name=node.name))
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    [fp16_out],
                    [out],
                    name=f"{node.name or out}/CastOutputToFP32",
                    to=TensorProto.FLOAT,
                )
            )
            rewritten.append({"op_type": "MatMul", "node": node.name, "weights": [weight_source]})
            continue

        if node.op_type == "Gemm":
            if len(node.input) < 2 or len(node.output) != 1:
                raise SystemExit(f"cannot rewrite Gemm node with unexpected arity: {node.name or node.output}")
            weight_source = _trace_selected_source(model, node.input[1], fp16_initializer_names)
            if weight_source is None:
                new_nodes.append(node)
                continue
            alpha = float(_node_attr(onnx, node, "alpha", 1.0))
            beta = float(_node_attr(onnx, node, "beta", 1.0))
            trans_a = int(_node_attr(onnx, node, "transA", 0))
            trans_b = int(_node_attr(onnx, node, "transB", 0))
            if alpha != 1.0 or beta != 1.0:
                raise SystemExit(f"cannot rewrite Gemm with alpha/beta != 1: {node.name or node.output}")

            out = node.output[0]
            a_name = node.input[0]
            b_name = node.input[1]
            c_name = node.input[2] if len(node.input) >= 3 and node.input[2] else ""

            a_fp16 = f"{out}_a_fp16"
            new_nodes.append(
                helper.make_node("Cast", [a_name], [a_fp16], name=f"{node.name or out}/CastAToFP16", to=TensorProto.FLOAT16)
            )
            matmul_a = a_fp16
            if trans_a:
                matmul_a = f"{out}_a_fp16_t"
                new_nodes.append(helper.make_node("Transpose", [a_fp16], [matmul_a], name=f"{node.name or out}/TransposeA"))

            matmul_b = b_name
            if trans_b:
                matmul_b = f"{out}_b_fp16_t"
                new_nodes.append(helper.make_node("Transpose", [b_name], [matmul_b], name=f"{node.name or out}/TransposeB"))

            fp16_out = f"{out}_fp16"
            fp32_out = out if not c_name else f"{out}_fp32"
            new_nodes.append(helper.make_node("MatMul", [matmul_a, matmul_b], [fp16_out], name=node.name))
            new_nodes.append(
                helper.make_node("Cast", [fp16_out], [fp32_out], name=f"{node.name or out}/CastOutputToFP32", to=TensorProto.FLOAT)
            )
            if c_name:
                new_nodes.append(helper.make_node("Add", [fp32_out, c_name], [out], name=f"{node.name or out}/AddBias"))
            rewritten.append({"op_type": "Gemm", "node": node.name, "weights": [weight_source], "bias": c_name or None})
            continue

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return {"rewritten_nodes": rewritten}


# =============================================================================
# W8A8 + ConvRot implementation
# =============================================================================
#
# Implements the w8a8 + ConvRot path as a TensorRT custom plugin:
#
#   * Offline (weight):  W_rot = W @ H_block^T, then per-output-channel
#                        symmetric INT8 quantization of W_rot. The rotation
#                        spreads diffusion-model outliers across channels
#                        inside each group, so per-channel scales no longer
#                        get hoisted by a single huge magnitude.
#
#   * Online (activation): x_rot = x @ H_block inside the TensorRT
#                          ConvRotInt8Linear plugin, followed by a per-row
#                          dynamic INT8 quantizer.
#
#   * Compute:           ConvRotInt8Linear runs INT8 GEMM, dequantizes with
#                        x_scale and W_scale, and adds bias when present.
#
# H_block is shared across every rotated site — it is built once via
# convrot.build_hadamard(group_size) and stored as a single FP32 initializer.
# This mirrors the ComfyUI-INT8-Fast reference, where the same H is reused
# across all rotated Linear layers.
#
# ConvRot is skipped per-weight when in_features % group_size != 0; the w8a8
# path still applies, just without the rotation benefit. This matches the
# reference behavior and keeps the implementation robust to unusual Linear
# shapes (e.g. head_dim-projection layers).


def _detect_max_convrot_group_size() -> int:
    """Return the default group_size for the current Triton plugin path.

    The two-kernel Triton implementation (per CONVROT_OPTIMAL_TWO_KERNEL_SPEC.md)
    uses GROUP_SIZE=256 with a Tensor-Core-based H_16⊗H_16 separable rotation.
    The rotation group, the activation quantization group, and the GEMM dequant
    group are all aligned to the same 256-wide chunk.
    """
    return 256


def _convrot_default_group_size() -> int:
    """Resolve the ConvRot group size: env override → auto-detect → safe default."""
    from convrot import CONVROT_GROUP_SIZE, is_valid_group_size

    override = os.environ.get("HOTSTEP_CONVROT_GROUP_SIZE")
    if not override:
        # The fixed kernel no longer picks group size by shared-memory capacity.
        detected = _detect_max_convrot_group_size()
        if detected != CONVROT_GROUP_SIZE:
            print(f"[export_dit] ConvRot group_size auto-detected: {detected} "
                  f"(override with HOTSTEP_CONVROT_GROUP_SIZE)")
        return detected
    try:
        value = int(override)
    except ValueError as exc:
        raise SystemExit(
            f"HOTSTEP_CONVROT_GROUP_SIZE must be an integer power of 4, got {override!r}"
        ) from exc

    if not is_valid_group_size(value):
        raise SystemExit(
            f"HOTSTEP_CONVROT_GROUP_SIZE must be a power of 4, got {value}"
        )
    if value != 256:
        raise SystemExit(
            f"HOTSTEP_CONVROT_GROUP_SIZE {value} is not compiled into the TRT plugin; supported value is 256"
        )
    return value


def _emit_convrot_h_initializer(model, group_size: int, name: str) -> str:
    """Add (or reuse) a single FP16 H initializer shared by all rotation sites.

    Returns the initializer name so callers can reference it from their MatMul
    nodes. Storing H once keeps the graph small (32 layers × 7 linears would
    otherwise duplicate the same 64×64 matrix 224 times).
    """
    import numpy as np
    from onnx import numpy_helper

    for init in model.graph.initializer:
        if init.name == name:
            return name
    from convrot import build_hadamard

    boundary_dtype = _w8a8_plugin_boundary_dtype()
    H = _numpy_boundary_array(build_hadamard(group_size), boundary_dtype)
    model.graph.initializer.extend([numpy_helper.from_array(H, name=name)])
    return name


def _quantize_w8a8_weight_array(
    arr: "np.ndarray",
    axis: int,
    group_size: int | None,
    H: "np.ndarray | None",
) -> tuple["np.ndarray", "np.ndarray", bool]:
    """Apply ConvRot (if applicable) + per-axis symmetric INT8 quantization.

    Args:
        arr: 2D weight array. The ``axis`` parameter identifies which axis
            holds the OUTPUT channel (the quantization axis). The OTHER axis
            holds in_features and is the ConvRot rotation axis. Both
            ``[out, in]`` (axis=0, PyTorch Linear convention) and
            ``[in, out]`` (axis=1, ONNX MatMul / Gemm transB=1 convention)
            layouts are supported.
        axis: Quantization axis (the OUTPUT channel axis in ``arr``).
        group_size: ConvRot group size, or None to skip rotation.
        H: Precomputed Hadamard matrix of shape ``[group_size, group_size]``,
            or None if no rotation.

    Returns:
        (q_int8, scale_fp32, rotated) where ``q_int8`` is canonical
        ``[out_features, in_features]`` for the TensorRT plugin,
        ``scale_fp32`` has one value per output channel, and ``rotated``
        indicates whether ConvRot was applied.
    """
    import numpy as np

    if arr.ndim != 2:
        raise SystemExit(
            f"W8A8 quantization currently supports 2D weights only, got {arr.shape}"
        )
    if axis not in (0, 1):
        raise SystemExit(
            f"W8A8 quantization axis must be 0 or 1, got {axis} for shape {arr.shape}"
        )

    # Work in canonical [out, in] layout internally so the rotation and
    # per-output-channel scale logic is the same regardless of storage.
    if axis == 0:
        w_canonical = np.ascontiguousarray(arr, dtype=np.float32)
    else:  # axis == 1, arr is [in, out]
        w_canonical = np.ascontiguousarray(arr.T, dtype=np.float32)

    rotated = False
    if group_size is not None and H is not None and w_canonical.shape[1] % group_size == 0:
        from convrot import rotate_weight

        w_canonical = rotate_weight(w_canonical, H, group_size)
        rotated = True

    # Per-output-channel (axis 0 of canonical) symmetric INT8 quantization.
    max_abs = np.max(np.abs(w_canonical), axis=1)
    scale = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
    q_canonical = np.clip(
        np.rint(w_canonical / scale.reshape(-1, 1)), -127, 127
    ).astype(np.int8)

    # The TensorRT plugin has a single layout contract: [out, in].
    return np.ascontiguousarray(q_canonical), scale, rotated


def _collect_w8a8_weight_axes(model, selected_names: set[str]) -> dict[str, int]:
    """Collect the per-weight quantization axis (output channel) for each
    selected initializer. Traces through Cast/Identity/Transpose producers
    to recover the source axis even when the weight is stored transposed
    (MatMul) or accessed via Gemm transB=1.

    The axis is the OUTPUT channel axis of the original Linear weight
    (axis 0 for canonical [out, in] layout, possibly transposed through
    a Transpose node before the MatMul). We trace through Cast/Identity/
    Transpose producers to recover the source axis.
    """
    import onnx

    axes: dict[str, int] = {}

    def add_axis(name: str, axis: int, node_name: str) -> None:
        prev = axes.get(name)
        if prev is not None and prev != axis:
            raise SystemExit(
                f"cannot assign two W8A8 quantization axes for {name}: "
                f"{prev} vs {axis} at {node_name}"
            )
        axes[name] = axis

    for node in model.graph.node:
        if node.op_type not in {"Gemm", "MatMul"}:
            continue
        node_name = node.name or (node.output[0] if node.output else node.op_type)
        if node.op_type == "MatMul":
            if len(node.input) != 2:
                raise SystemExit(
                    f"cannot W8A8-rewrite MatMul with unexpected arity: {node_name}"
                )
            traced = _trace_selected_source_axis(model, node.input[1], selected_names, 1)
            if traced is None:
                continue
            weight_name, source_axis = traced
            add_axis(weight_name, source_axis, node_name)
            continue
        if node.op_type == "Gemm":
            if len(node.input) < 2:
                raise SystemExit(
                    f"cannot W8A8-rewrite Gemm with unexpected arity: {node_name}"
                )
            trans_b = int(_node_attr(onnx, node, "transB", 0))
            traced = _trace_selected_source_axis(
                model, node.input[1], selected_names, 0 if trans_b else 1
            )
            if traced is None:
                continue
            weight_name, source_axis = traced
            add_axis(weight_name, source_axis, node_name)
    return axes


def _quantize_w8a8_initializers_with_convrot(
    model,
    selected_names: set[str],
    axes: dict[str, int],
    group_size: int,
) -> dict:
    """Replace selected FP32 initializers with INT8 + FP32 scale, applying ConvRot.

    The H initializer is added once (shared across all rotated sites) so the
    activation rotation subgraph can reference the same constant. Per-weight
    zero-points are skipped — symmetric quantization uses zero_point=0 by
    construction, and the ConvRotInt8Linear plugin does not require an
    explicit zero-point input.
    """
    import numpy as np
    from onnx import TensorProto, numpy_helper

    from convrot import build_hadamard, is_valid_group_size

    if not is_valid_group_size(group_size):
        raise SystemExit(
            f"W8A8 ConvRot group_size must be a power of 4, got {group_size}"
        )
    if group_size != 256:
        raise SystemExit(
            f"W8A8 ConvRot group_size {group_size} is not compiled into the TRT plugin; supported value is 256"
        )
    H_np = build_hadamard(group_size).astype(np.float32)

    # Single shared H initializer for activation rotation. The v2 plugin
    # contract uses FP16 activation/H/bias tensors by default; the CUDA
    # butterfly implementation does not read H at runtime, but typing it as
    # FP16 keeps the ONNX contract cast-free at the plugin boundary.
    H_name = "convrot.hadamard"
    H_already_present = any(init.name == H_name for init in model.graph.initializer)
    if not H_already_present:
        boundary_dtype = _w8a8_plugin_boundary_dtype()
        model.graph.initializer.extend([numpy_helper.from_array(_numpy_boundary_array(H_np, boundary_dtype), name=H_name)])

    initializers = {init.name: init for init in model.graph.initializer}
    quantized: list[str] = []
    missing: list[str] = []
    skipped: list[str] = []
    scale_names: dict[str, str] = {}
    axis_by_name: dict[str, int] = {}
    rotated_by_name: dict[str, bool] = {}

    for name in sorted(selected_names):
        init = initializers.get(name)
        if init is None:
            missing.append(name)
            continue
        if name not in axes:
            skipped.append(name)
            continue
        arr = numpy_helper.to_array(init)
        if arr.ndim < 2:
            skipped.append(name)
            continue
        axis = axes[name]
        if axis < 0:
            axis += arr.ndim
        if axis < 0 or axis >= arr.ndim:
            raise SystemExit(
                f"W8A8 axis {axis} is out of range for {name} shape {arr.shape}"
            )
        # ConvRot rotates along in_features. For the canonical Linear layout
        # [out, in], in_features is axis 1. If the ONNX initializer is stored
        # transposed ([in, out]), we rotate along axis 0 instead. The
        # _collect_w8a8_weight_axes trace already returns the axis that
        # corresponds to the OUTPUT channel of the Linear (i.e. the
        # quantization axis); the rotation axis is the OTHER one.
        rotation_axis = 1 - axis if arr.ndim == 2 else None
        local_group = group_size if (rotation_axis is not None and arr.shape[rotation_axis] % group_size == 0) else None
        local_H = H_np if local_group is not None else None

        q, scale, rotated = _quantize_w8a8_weight_array(arr, axis, local_group, local_H)

        init.CopyFrom(numpy_helper.from_array(q, name=name))
        init.data_type = TensorProto.INT8

        scale_name = f"{name}.w8a8_scale"
        model.graph.initializer.extend(
            [numpy_helper.from_array(scale.astype(np.float32), name=scale_name)]
        )
        scale_names[name] = scale_name
        # Initializer has been canonicalized for the plugin: [out, in].
        axis_by_name[name] = 0
        rotated_by_name[name] = rotated
        quantized.append(name)

    return {
        "quantized": quantized,
        "missing": missing,
        "skipped": skipped,
        "scale_names": scale_names,
        "axis_by_name": axis_by_name,
        "rotated_by_name": rotated_by_name,
        "convrot_group_size": group_size,
        "convrot_h_name": H_name,
        "convrot_applied": any(rotated_by_name.values()),
    }


# ──────────────────────────────────────────────────────────────────────────
# Attention rewrite: decomposed SDPA → ONNX Attention-23 op (FP16 lowp island)
# ──────────────────────────────────────────────────────────────────────────
# Ported from tools/onnx-export/optimize_dit_onnx.py. Replaces the manually
# decomposed ``MatMul(Q,Kᵀ) → Mul(scale) → Add(mask) → Softmax → MatMul(.,V)``
# pattern (produced by torch dynamo's sdpa decomposition) with a single
# ONNX ``Attention`` op (opset 23). This lets TensorRT dispatch to its
# built-in FMHA kernel instead of the current FP32 MatMul+Softmax+MatMul
# chain, which is ~30% of total inference time.
#
# The attention island runs in FP16 for sm_75/T4 compatibility (T4 has no BF16 support).
# flash-attention precision). TRT's Attention op handles softmax
# precision internally (always FP32 accumulation) — the
# ``softmax_precision`` attribute is NOT supported by TRT's parser.
# The surrounding graph stays in its current dtype (FP32 for the baseline);
# Casts are inserted at the attention I/O boundary.


def _attn_node_by_output(model) -> dict:
    """Map tensor name → producing node."""
    return {out: node for node in model.graph.node for out in node.output if out}


def _attn_consumers_by_input(model) -> dict:
    """Map tensor name → list of consuming nodes."""
    import collections
    consumers: dict[str, list] = collections.defaultdict(list)
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers[inp].append(node)
    return consumers


def _attn_ensure_main_opset(model, min_version: int) -> None:
    """Bump the main (ai.onnx) opset to at least ``min_version``."""
    from onnx import helper
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            if opset.version < min_version:
                opset.version = min_version
            return
    model.opset_import.extend([helper.make_opsetid("", min_version)])


def _is_sqrt_scalar_node(node, prod: dict) -> bool:
    """True if ``node`` is a ``Sqrt(CastLike(scalar, q_tensor))`` pattern."""
    if node is None or node.op_type != "Sqrt" or len(node.input) != 1:
        return False
    src = prod.get(node.input[0])
    return src is None or src.op_type in {"CastLike", "Cast", "Constant", "Reshape", "Squeeze"}


def _split_scaled_mul(mul_node, prod: dict):
    """Return ``(data_input, sqrt_scale_input)`` for ``Mul(data, Sqrt(scale))``."""
    if mul_node is None or mul_node.op_type != "Mul" or len(mul_node.input) != 2:
        return None
    a, b = mul_node.input
    pa, pb = prod.get(a), prod.get(b)
    if _is_sqrt_scalar_node(pa, prod):
        return b, a
    if _is_sqrt_scalar_node(pb, prod):
        return a, b
    return None


def _trace_untranspose_k(name: str, prod: dict):
    """Undo the exporter's ``K^T`` reshape/transpose/reshape wrapper."""
    r2 = prod.get(name)
    if r2 is None or r2.op_type != "Reshape" or len(r2.input) < 1:
        return None
    tr = prod.get(r2.input[0])
    if tr is None or tr.op_type != "Transpose" or len(tr.input) < 1:
        return None
    r1 = prod.get(tr.input[0])
    if r1 is None or r1.op_type != "Reshape" or len(r1.input) < 1:
        return None
    return r1.input[0]


def rewrite_attention_to_onnx_attention_fp16(model) -> dict:
    """Rewrite decomposed SDPA islands to ONNX ``Attention`` op (opset 23, FP16).

    Pattern matched::

        Mul(Q, sqrt(scale))            Mul(K^T, sqrt(scale))
             |                              |
              MatMul(Q, K^T) → Add(mask) → Softmax → MatMul(., V) → output

    Replacement::

        Cast Q/K/V/mask → fp16
        Attention(Q_fp16, K_fp16, V_fp16, mask_fp16, scale=1/sqrt(D),
                  is_causal=0)
        Cast output → FP32 (reuses the original AV MatMul output name)

    The attention island runs in FP16 in this export variant. This is required
    for sm_75/T4 compatibility because T4 has no BF16 support. TensorRT handles
    softmax precision internally (FP32 accumulation by default); the
    ``softmax_precision`` attribute is rejected by TRT's ONNX parser
    (``!hasSoftmaxPrecision`` assertion). The surrounding residual graph stays
    in its current dtype (FP32 for the baseline); Casts are inserted at the
    attention I/O boundary.

    Returns a report dict with ``rewritten`` count and ``skipped`` reasons.
    """
    from onnx import TensorProto, helper

    prod = _attn_node_by_output(model)
    consumers = _attn_consumers_by_input(model)
    remove_node_names: set[str] = set()
    replacements: dict[str, list] = {}
    rewritten = []
    skipped: dict[str, int] = {}

    used_names = {name for node in model.graph.node
                  for name in list(node.input) + list(node.output) if name}
    used_names.update(init.name for init in model.graph.initializer)

    def unique(base: str) -> str:
        name = base
        i = 0
        while name in used_names:
            i += 1
            name = f"{base}_{i}"
        used_names.add(name)
        return name

    for softmax in list(model.graph.node):
        if softmax.op_type != "Softmax" or len(softmax.input) != 1 or len(softmax.output) != 1:
            continue
        sm_in, sm_out = softmax.input[0], softmax.output[0]
        add = prod.get(sm_in)
        if add is None or add.op_type != "Add" or len(add.input) != 2:
            skipped["softmax_input_not_add"] = skipped.get("softmax_input_not_add", 0) + 1
            continue

        # Find QK matmul and additive mask input.
        qk = None
        mask = None
        for inp in add.input:
            p = prod.get(inp)
            if p is not None and p.op_type == "MatMul":
                qk = p
            else:
                mask = inp
        if qk is None or mask is None or len(qk.input) != 2 or len(qk.output) != 1:
            skipped["no_qk_or_mask"] = skipped.get("no_qk_or_mask", 0) + 1
            continue

        q_mul = prod.get(qk.input[0])
        k_mul = prod.get(qk.input[1])
        q_parts = _split_scaled_mul(q_mul, prod)
        k_parts = _split_scaled_mul(k_mul, prod)
        if q_parts is None or k_parts is None:
            skipped["scaled_mul_not_matched"] = skipped.get("scaled_mul_not_matched", 0) + 1
            continue
        q_name, _q_scale = q_parts
        k_transposed_name, _k_scale = k_parts
        k_name = _trace_untranspose_k(k_transposed_name, prod)
        if k_name is None:
            skipped["k_untranspose_not_matched"] = skipped.get("k_untranspose_not_matched", 0) + 1
            continue

        av_consumers = [node for node in consumers.get(sm_out, [])
                        if node.op_type == "MatMul" and len(node.input) == 2]
        if len(av_consumers) != 1:
            skipped["av_consumer_count"] = skipped.get("av_consumer_count", 0) + 1
            continue
        av = av_consumers[0]
        if av.input[0] != sm_out:
            skipped["av_softmax_not_first_input"] = skipped.get("av_softmax_not_first_input", 0) + 1
            continue
        v_name = av.input[1]
        if len(av.output) != 1:
            skipped["av_output_count"] = skipped.get("av_output_count", 0) + 1
            continue
        out_name = av.output[0]

        base = av.name or softmax.name or out_name
        q_fp16 = unique(out_name + "_attention_q_fp16")
        k_fp16 = unique(out_name + "_attention_k_fp16")
        v_fp16 = unique(out_name + "_attention_v_fp16")
        mask_fp16 = unique(out_name + "_attention_mask_fp16")
        y_fp16 = unique(out_name + "_attention_y_fp16")

        new_nodes = [
            helper.make_node("Cast", [q_name], [q_fp16],
                             name=base + "/CastQTofp16", to=TensorProto.FLOAT16),
            helper.make_node("Cast", [k_name], [k_fp16],
                             name=base + "/CastKTofp16", to=TensorProto.FLOAT16),
            helper.make_node("Cast", [v_name], [v_fp16],
                             name=base + "/CastVTofp16", to=TensorProto.FLOAT16),
            helper.make_node("Cast", [mask], [mask_fp16],
                             name=base + "/CastMaskTofp16", to=TensorProto.FLOAT16),
            helper.make_node(
                "Attention",
                [q_fp16, k_fp16, v_fp16, mask_fp16],
                [y_fp16],
                name=base + "/Attentionfp16",
                scale=0.08838834764831845,  # 1/sqrt(128) for head_dim=128
                is_causal=0,
            ),
            helper.make_node("Cast", [y_fp16], [out_name],
                             name=base + "/CastAttentionOutputToFP32",
                             to=TensorProto.FLOAT),
        ]
        replacements[av.name] = new_nodes
        remove_node_names.update({softmax.name, add.name, qk.name, av.name})
        # Q/K scaling nodes become dead after QK removal; remove when uniquely named.
        if q_mul is not None:
            remove_node_names.add(q_mul.name)
        if k_mul is not None:
            remove_node_names.add(k_mul.name)
        rewritten.append({"softmax": softmax.name, "qk": qk.name,
                          "av": av.name, "output": out_name})

    if not rewritten:
        return {"rewritten": 0, "skipped": dict(skipped)}

    new_graph_nodes = []
    for node in model.graph.node:
        if node.name in replacements:
            new_graph_nodes.extend(replacements[node.name])
            continue
        if node.name in remove_node_names:
            continue
        new_graph_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_graph_nodes)
    _attn_ensure_main_opset(model, 23)
    return {"rewritten": len(rewritten), "examples": rewritten[:8], "skipped": dict(skipped)}


def _w8a8_rewrite_dispatch(model, quant_report: dict) -> dict:
    """Emit the fused ``ConvRotInt8Linear`` custom op for every w8a8 site.

    HOT-Step's w8a8 path is plugin-only: ONNX is the intermediate model
    format that TensorRT parses and compiles into a serialized engine,
    and the ConvRotInt8Linear op is registered as a TRT custom plugin
    (see ``trt_plugins/`` and ``engine/src/plugins/``). The plugin
    fuses the online activation rotation, per-row INT8 quantization,
    INT8 × INT8 matmul, dequant, and bias add into a single GPU kernel
    launch — a direct port of the ComfyUI-INT8-Fast Triton kernel.
    """
    return _rewrite_w8a8_weight_ops_with_plugin(model, quant_report)


def _cast_attr_to_dtype(onnx, node) -> int | None:
    for attr in node.attribute:
        if attr.name == "to" and attr.type == onnx.AttributeProto.INT:
            return int(attr.i)
    return None


def _string_attr(node, name: str, default: str = "") -> str:
    import onnx

    for attr in node.attribute:
        if attr.name == name and attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode("utf-8", errors="replace")
    return default


def _rewrite_w8a8_fp16_islands(model, seed_lowp_tensors: set[str], target_dtype_name: str = "FP16") -> dict:
    """Make TensorRT strongly-typed elementwise islands agree on the plugin lowp dtype.

    ConvRotInt8Linear v2 emits FP16 in this T4-compatible variant. In a
    strongly typed TRT network, elementwise nodes such as
    Add/Mul are not allowed to mix FP32 and lowp tensors. This pass propagates
    the target lowp dtype through dtype-preserving ONNX ops and inserts casts
    only on the non-lowp side of mixed arithmetic islands.
    """
    import onnx
    from onnx import TensorProto, helper

    target_dtype_name = target_dtype_name.upper()
    target_tensorproto = _onnx_dtype_for_plugin_boundary(TensorProto, target_dtype_name)
    lowp = set(seed_lowp_tensors)
    for init in model.graph.initializer:
        if init.data_type == target_tensorproto:
            lowp.add(init.name)

    same_type_ops = {
        "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Mean",
    }
    unary_preserve_ops = {
        "Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh",
        "Ceil", "Cos", "Cosh", "Elu", "Erf", "Exp", "Floor", "Gelu",
        "HardSigmoid", "LeakyRelu", "Log", "Neg", "Reciprocal", "Relu",
        "Round", "Selu", "Sigmoid", "Sin", "Sinh", "Softmax", "Sqrt",
        "Tan", "Tanh", "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin",
        "ReduceProd", "ReduceL2",
    }
    data_movement_ops = {
        "Identity", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Slice",
        "Gather", "GatherElements", "GatherND", "Flatten", "Expand", "Tile",
        "Pad", "DepthToSpace", "SpaceToDepth",
        # PyTorch export commonly lowers chunk()/getitem() through sequence ops;
        # the TensorRT compatibility pass later rewrites these to Split, so the
        # FP16 propagation pass must understand both forms.
        "Split", "SplitToSequence", "SequenceAt",
    }
    matmul_like_ops = {"MatMul", "Gemm"}

    inserted = []
    new_nodes = []
    used_names = {name for node in model.graph.node for name in list(node.input) + list(node.output) if name}
    used_names.update(init.name for init in model.graph.initializer)

    def unique_name(base: str) -> str:
        base = base.replace(" ", "_")
        candidate = base
        idx = 0
        while candidate in used_names:
            idx += 1
            candidate = f"{base}_{idx}"
        used_names.add(candidate)
        return candidate

    def cast_to_lowp(tensor_name: str, node_name: str, input_index: int) -> str:
        if tensor_name in lowp:
            return tensor_name
        suffix = target_dtype_name.lower()
        cast_out = unique_name(f"{tensor_name}_to_{suffix}_for_{node_name}_{input_index}")
        cast_node = helper.make_node(
            "Cast",
            [tensor_name],
            [cast_out],
            name=unique_name(f"{node_name}/CastInput{input_index}To{target_dtype_name}"),
            to=target_tensorproto,
        )
        new_nodes.append(cast_node)
        lowp.add(cast_out)
        inserted.append({"node": cast_node.name, "input": tensor_name, "output": cast_out})
        return cast_out

    def clone_with_inputs(node, inputs):
        cloned = onnx.NodeProto()
        cloned.CopyFrom(node)
        del cloned.input[:]
        cloned.input.extend(inputs)
        return cloned

    for node in model.graph.node:
        node_name = node.name or (node.output[0] if node.output else node.op_type)

        if node.op_type == "Cast":
            new_nodes.append(node)
            if _cast_attr_to_dtype(onnx, node) == target_tensorproto:
                lowp.update(o for o in node.output if o)
            continue

        if node.op_type == "ConvRotInt8Linear":
            new_nodes.append(node)
            if _string_attr(node, "output_dtype", "FP16").upper() == target_dtype_name:
                lowp.update(o for o in node.output if o)
            continue

        if node.op_type in same_type_ops and any(i in lowp for i in node.input if i):
            patched_inputs = [cast_to_lowp(i, node_name, idx) if i and i not in lowp else i
                              for idx, i in enumerate(node.input)]
            new_nodes.append(clone_with_inputs(node, patched_inputs))
            lowp.update(o for o in node.output if o)
            continue

        if node.op_type == "Concat" and any(i in lowp for i in node.input if i):
            patched_inputs = [cast_to_lowp(i, node_name, idx) if i and i not in lowp else i
                              for idx, i in enumerate(node.input)]
            new_nodes.append(clone_with_inputs(node, patched_inputs))
            lowp.update(o for o in node.output if o)
            continue

        if node.op_type == "Where" and len(node.input) >= 3 and any(i in lowp for i in node.input[1:3] if i):
            patched_inputs = list(node.input)
            for idx in (1, 2):
                if patched_inputs[idx] and patched_inputs[idx] not in lowp:
                    patched_inputs[idx] = cast_to_lowp(patched_inputs[idx], node_name, idx)
            new_nodes.append(clone_with_inputs(node, patched_inputs))
            lowp.update(o for o in node.output if o)
            continue

        if node.op_type == "Clip" and node.input and node.input[0] in lowp:
            patched_inputs = list(node.input)
            for idx in range(1, len(patched_inputs)):
                if patched_inputs[idx] and patched_inputs[idx] not in lowp:
                    patched_inputs[idx] = cast_to_lowp(patched_inputs[idx], node_name, idx)
            new_nodes.append(clone_with_inputs(node, patched_inputs))
            lowp.update(o for o in node.output if o)
            continue

        if node.op_type in matmul_like_ops and any(i in lowp for i in node.input if i):
            patched_inputs = [cast_to_lowp(i, node_name, idx) if i and i not in lowp else i
                              for idx, i in enumerate(node.input)]
            new_nodes.append(clone_with_inputs(node, patched_inputs))
            lowp.update(o for o in node.output if o)
            continue

        if node.op_type in unary_preserve_ops and node.input and node.input[0] in lowp:
            new_nodes.append(node)
            lowp.update(o for o in node.output if o)
            continue

        if node.op_type in data_movement_ops and node.input and node.input[0] in lowp:
            new_nodes.append(node)
            lowp.update(o for o in node.output if o)
            continue

        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return {
        "lowp_boundary_casts_inserted": inserted,
        "lowp_tensor_count": len(lowp),
    }


def _rewrite_w8a8_weight_ops_with_plugin(model, quant_report: dict) -> dict:
    """Replace MatMul/Gemm nodes with a single ConvRotInt8Linear custom op.

    This is the only w8a8 path in production: ONNX is the intermediate
    model format that TensorRT parses and compiles into a serialized
    engine. The ``ConvRotInt8Linear`` op is registered as a TRT custom
    plugin (see ``trt_plugins/`` and ``engine/src/plugins/``) that
    fuses the online activation rotation, per-row INT8 quantization,
    INT8 × INT8 matmul, dequant, and bias add into a single GPU kernel
    launch — a direct port of the ComfyUI-INT8-Fast Triton kernel.

    The emitted ONNX node has:
      * domain: ``hotstep``
      * op_type: ``ConvRotInt8Linear``
      * inputs: [x, weight_q, weight_scale, H, (bias)]
      * outputs: [y]
      * attributes: group_size, in_features, out_features, has_bias

    See ``trt_plugins/convrot_int8_plugin.py`` for the kernel
    implementation and the NumPy reference used in unit tests.
    """
    import onnx
    from onnx import TensorProto, helper

    # Lazy import — the plugin module depends on numpy only at import time.
    try:
        from trt_plugins import (
            CONVROT_INT8_LINEAR_OP_NAME as _OP_NAME,
            CONVROT_INT8_LINEAR_OP_NAMESPACE as _OP_NS,
            make_convrot_int8_linear_onnx_node as _make_node,
        )
    except ImportError as exc:
        raise SystemExit(
            "w8a8 plugin path requires the trt_plugins package "
            f"(tools/onnx-export/trt_plugins/); import failed: {exc}"
        ) from exc

    quantized_names = set(quant_report["quantized"])
    axis_by_name = quant_report["axis_by_name"]
    rotated_by_name = quant_report["rotated_by_name"]
    group_size = quant_report["convrot_group_size"]
    H_name = quant_report["convrot_h_name"]

    # Ensure the custom op domain is registered on the model so onnx.checker
    # doesn't reject the graph. The opset version matches the rest of the
    # graph (18).
    existing_domains = {d.domain for d in model.opset_import}
    if _OP_NS not in existing_domains:
        model.opset_import.extend([helper.make_opsetid(_OP_NS, 18)])

    # Build a map: weight_name → in_features (K dim of the matmul).
    initializer_index = {init.name: init for init in model.graph.initializer}
    weight_in_features: dict[str, int] = {}
    for name in quantized_names:
        init = initializer_index.get(name)
        if init is None or len(init.dims) != 2:
            continue
        axis = axis_by_name[name]
        in_axis = 1 - axis
        weight_in_features[name] = int(init.dims[in_axis])

    # Build a producer index so we can detect the ``MatMul → Add(bias)``
    # pattern (PyTorch's Linear export idiom). When the Add's bias input is
    # a 1-D initializer matching out_features, we fold it into the custom op
    # instead of leaving a dangling Add after the MatMul rewrite.
    producer_by_output: dict[str, object] = {}
    for n in model.graph.node:
        for o in n.output:
            if o:
                producer_by_output[o] = n

    # Consumers of each MatMul output — used to detect the bias-Add pattern.
    consumers_of_output: dict[str, list] = {}
    for n in model.graph.node:
        for i in n.input:
            if i:
                consumers_of_output.setdefault(i, []).append(n)

    rewritten = []
    new_nodes: list = []
    consumed_matmul_outputs: set[str] = set()
    lowp_plugin_outputs: set[str] = set()
    boundary_dtype = _w8a8_plugin_boundary_dtype()
    boundary_tensorproto = _onnx_dtype_for_plugin_boundary(TensorProto, boundary_dtype)
    cast_nodes_inserted = []

    for node in model.graph.node:
        # Skip nodes whose output has already been consumed by a previous
        # custom-op emission (e.g. the Add we folded into the MatMul's
        # custom op).
        if node.output and node.output[0] in consumed_matmul_outputs:
            continue

        if node.op_type not in {"Gemm", "MatMul"}:
            new_nodes.append(node)
            continue

        node_name = node.name or (node.output[0] if node.output else node.op_type)
        out = node.output[0] if node.output else f"{node_name}/out"

        if node.op_type == "MatMul":
            if len(node.input) != 2 or len(node.output) != 1:
                raise SystemExit(
                    f"cannot W8A8-plugin-rewrite MatMul with unexpected arity: {node_name}"
                )
            weight_source = _trace_selected_source(model, node.input[1], quantized_names)
            if weight_source is None:
                new_nodes.append(node)
                continue
            x_name = node.input[0]
            bias_name = ""
            has_bias = False

            # Detect a trailing ``Add(matmul_out, bias)`` whose bias is a 1-D
            # initializer with matching out_features. PyTorch dynamo emits
            # Linear as MatMul + Add(bias) rather than a Gemm, so this is the
            # common path for the DiT graph. We fold the Add into the custom
            # op's bias input.
            matmul_out = node.output[0]
            matmul_consumers = consumers_of_output.get(matmul_out, [])
            for consumer in matmul_consumers if len(matmul_consumers) == 1 else []:
                if consumer.op_type != "Add" or len(consumer.input) != 2:
                    continue
                # One of the Add inputs is our MatMul output; the other must
                # be a 1-D initializer (the bias).
                other_input = (consumer.input[1] if consumer.input[0] == matmul_out
                               else consumer.input[0])
                bias_init = initializer_index.get(other_input)
                if bias_init is None or len(bias_init.dims) != 1:
                    continue
                # out_features is the OTHER dim of the weight; check the bias
                # length matches.
                weight_axis = axis_by_name[weight_source]
                expected_out_features = int(initializer_index[weight_source].dims[weight_axis])
                if bias_init.dims[0] != expected_out_features:
                    continue
                # Fold: bias_name = other_input, has_bias = True, out = Add output
                bias_name = other_input
                has_bias = True
                out = consumer.output[0]
                consumed_matmul_outputs.add(consumer.output[0])
                break
        else:  # Gemm
            if len(node.input) < 2 or len(node.output) != 1:
                raise SystemExit(
                    f"cannot W8A8-plugin-rewrite Gemm with unexpected arity: {node_name}"
                )
            weight_source = _trace_selected_source(model, node.input[1], quantized_names)
            if weight_source is None:
                new_nodes.append(node)
                continue
            alpha = float(_node_attr(onnx, node, "alpha", 1.0))
            beta = float(_node_attr(onnx, node, "beta", 1.0))
            trans_a = int(_node_attr(onnx, node, "transA", 0))
            if alpha != 1.0 or beta != 1.0:
                raise SystemExit(
                    f"cannot W8A8-plugin-rewrite Gemm with alpha/beta != 1: {node_name}"
                )
            if trans_a:
                raise SystemExit(
                    f"cannot W8A8-plugin-rewrite Gemm with transA != 0: {node_name}"
                )
            x_name = node.input[0]
            bias_name = node.input[2] if len(node.input) >= 3 and node.input[2] else ""
            has_bias = bool(bias_name)

        in_features = weight_in_features.get(weight_source)
        if in_features is None:
            raise SystemExit(
                f"W8A8 weight {weight_source} has no recorded in_features for {node_name}"
            )
        # out_features is the OTHER dim of the weight.
        weight_init = initializer_index[weight_source]
        weight_axis = axis_by_name[weight_source]
        out_features = int(weight_init.dims[weight_axis])

        # The TensorRT plugin no longer compiles the group_size=0 no-rotation
        # sentinel. Only rewrite layers whose weights were actually rotated
        # offline with the compiled group size; leave incompatible MatMul/Gemm
        # nodes in the graph instead of creating a plugin instance that cannot
        # load a cubin at build time.
        do_convrot = bool(rotated_by_name.get(weight_source, False)) and in_features % group_size == 0
        if not do_convrot:
            # The weight has already been replaced by its INT8 initializer in
            # the W8A8 quantization pass, so preserving the original MatMul/Gemm
            # would leave an invalid graph. Older plugin builds used
            # group_size=0 to cover this no-rotation W8A8 case, but that cubin
            # is no longer generated. Fail loudly if a future model introduces
            # a non-divisible K rather than emitting a plugin node that cannot
            # be built by TensorRT.
            raise SystemExit(
                f"W8A8 plugin rewrite for {node_name} would require removed group_size=0 "
                f"fallback (weight={weight_source}, in_features={in_features}, group_size={group_size})"
            )
        plugin_group_size = int(group_size)

        # Using FP32 for accuracy. It's worse with BF16 and catastrophic with FP16 here
        if any(t in weight_source for t in ["q_proj", "k_proj", "v_proj"]):
            input_dtype = "FP32"
            output_dtype = "FP32"
        elif "o_proj" in weight_source:
            input_dtype = "FP32"
            output_dtype = "FP32"
        elif any(t in weight_source for t in ["gate_proj", "up_proj"]):
            input_dtype = "FP32"
            output_dtype = "FP32"
        elif "down_proj" in weight_source:
            input_dtype = "FP32"
            output_dtype = "FP32"
        elif "proj_out" in weight_source:
            input_dtype = boundary_dtype
            output_dtype = "FP32"
        else:
            input_dtype = boundary_dtype
            output_dtype = boundary_dtype

        plugin_x_name = x_name
        if input_dtype != "FP32" and x_name not in lowp_plugin_outputs:
            plugin_x_name = f"{out}_convrot_x_{input_dtype.lower()}"
            cast_node = helper.make_node(
                "Cast",
                [x_name],
                [plugin_x_name],
                name=f"{node_name}/CastInputTo{input_dtype}",
                to=_onnx_dtype_for_plugin_boundary(TensorProto, input_dtype),
            )
            new_nodes.append(cast_node)
            cast_nodes_inserted.append({"node": cast_node.name, "to": input_dtype, "reason": "plugin_input"})

        # Sane quantization rule: folded 1D bias tensors stay FP32. The plugin
        # epilogue reads FP32 bias regardless of activation/output boundary dtype.
        plugin_bias_name = bias_name

        custom_node = _make_node(
            helper_module=helper,
            tensor_proto_module=TensorProto,
            x_name=plugin_x_name,
            weight_q_name=weight_source,
            weight_scale_name=quant_report["scale_names"][weight_source],
            bias_name=plugin_bias_name,
            output_name=out,
            node_name=f"{node_name}/ConvRotInt8Linear",
            group_size=plugin_group_size,
            in_features=in_features,
            out_features=out_features,
            has_bias=has_bias,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            preferred_format="HWC8",
        )
        new_nodes.append(custom_node)
        if output_dtype != "FP32":
            lowp_plugin_outputs.add(out)
        rewritten.append({
            "op_type": node.op_type,
            "node": node.name,
            "weights": [weight_source],
            "plugin": _OP_NAME,
            "convrot_applied": do_convrot,
            "input_dtype": input_dtype,
            "output_dtype": output_dtype,
        })

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    fp16_island_report = (
        _rewrite_w8a8_fp16_islands(model, lowp_plugin_outputs, boundary_dtype if boundary_dtype != "FP32" else "FP16")
        if lowp_plugin_outputs
        else {"lowp_boundary_casts_inserted": [], "lowp_tensor_count": 0}
    )
    return {
        "rewritten_nodes": rewritten,
        "plugin_op": _OP_NAME,
        "plugin_namespace": _OP_NS,
        "convrot_rotated_weights": sum(1 for v in rotated_by_name.values() if v),
        "convrot_group_size": group_size,
        "convrot_h_name": H_name,
        "casts_inserted": cast_nodes_inserted,
        "fp16_plugin_outputs": sorted(lowp_plugin_outputs),
        "plugin_boundary_dtype": boundary_dtype,
        "fp16_island_rewrite": fp16_island_report,
    }


def load_dit_model(
    model_dir: str,
    device: str = "cpu",
    precision: str = "q8map-fp16",
):
    """Load the AceStepDiTModel from a safetensors checkpoint."""
    model_dir = Path(model_dir)

    # Fix Windows encoding issues with transformers emoji output
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # Monkey-patch transformers auto_docstring to avoid lookup failure
    # for custom model types not registered in HF model registry
    try:
        import transformers.utils.auto_docstring as _ad
        _orig = _ad.auto_docstring
        _ad.auto_docstring = lambda *a, **kw: (lambda cls: cls)  # no-op decorator
    except Exception:
        pass

    # Add model dir to sys.path so we can import the model code.
    # Also add the Demon app root — model config files are re-export stubs
    # that import from the acestep package (from Demon).
    sys.path.insert(0, str(model_dir))
    demon_root = Path(model_dir).parent.parent.parent / "Demon"
    if demon_root.exists():
        sys.path.insert(0, str(demon_root))
        print(f"[export_dit] Added {demon_root} to sys.path for acestep package")
        # The model config stubs reference acestep.models.common but the
        # actual module is acestep.models. Create a shim alias.
        try:
            import acestep.models as _am
            sys.modules["acestep.models.common"] = _am
            # Also create the subpackage entry so Python's import system is happy
            import types
            if not hasattr(_am, "common"):
                _am.common = _am
        except ImportError:
            print("[export_dit] WARNING: Could not import acestep.models")

    from modeling_acestep_v15_xl_base import AceStepDiTModel # only xl for now, it IS the same for all xl
    from configuration_acestep_v15 import AceStepConfig

    # Load config
    import json
    with open(model_dir / "config.json") as f:
        config_dict = json.load(f)

    config = AceStepConfig(**config_dict)
    # Force SDPA for ONNX export (no flash attention)
    config._attn_implementation = "sdpa"

    if device != "cpu":
        raise SystemExit("low-memory export keeps weights CPU-mapped; use --device cpu")

    print(f"[export_dit] Loading model from {model_dir}...")
    print(f"[export_dit] Precision recipe: {precision}")
    print("[export_dit] Low-memory export: meta init only; weights stream after graph export")
    t0 = time.time()

    with torch.device("meta"):
        dit_model = AceStepDiTModel(config)
    dit_model = dit_model.to(dtype=torch.float32)

    # Supported handoff policies keep the PyTorch export in FP32. q8map-fp16
    # and w8a8 transform selected ONNX initializers after export according to
    # the hardcoded Q8_0-equivalent map.
    if precision not in {"q8map-fp16", "w8a8", "fp32"}:
        raise ValueError(
            f"Unknown precision: {precision}. Use 'q8map-fp16', 'w8a8', or 'fp32'."
        )

    # Replace Conv1d/ConvTranspose1d with Linear equivalents for ALL precision modes.
    # TensorRT handles the patch path more reliably as reshape+matmul.
    dit_model = replace_conv_with_linear(dit_model)
    dit_model.eval()

    t1 = time.time()
    print(f"[export_dit] Model loaded in {t1-t0:.1f}s")
    print(f"[export_dit] DiT: {sum(p.numel() for p in dit_model.parameters())/1e9:.2f}B params")

    # Log dtype distribution
    dtypes = {}
    for p in dit_model.parameters():
        dt = str(p.dtype)
        dtypes[dt] = dtypes.get(dt, 0) + p.numel()
    for dt, count in sorted(dtypes.items()):
        print(f"[export_dit]   {dt}: {count/1e6:.1f}M params")

    return dit_model, config


def export_onnx(dit_model, config, output_path: str, opset: int = 18,
                precision: str = "q8map-fp16", source_model: str = "",
                model_dir: str = "", stream_chunk_mb: int = 16):
    """Export the DiT forward pass to ONNX."""
    device = next(dit_model.parameters()).device

    if precision not in {"q8map-fp16", "w8a8", "fp32"}:
        raise SystemExit(f"unknown precision policy: {precision}")

    # The hardcoded policy is applied to ONNX initializers after FP32 export.
    tensor_dtype = torch.float32

    wrapper = DiTForwardWrapper(dit_model, precision=precision)
    wrapper.eval()

    # Create dummy inputs for tracing.
    #
    # WARNING — symbolic-dimension aliasing trap:
    # torch.export's ShapeEnv unifies two dynamic dimensions when they share the
    # same *concrete* value during tracing. The DiT patchifies the latent
    # sequence (T -> T // patch_size), so if the dummy encoder length S happens to
    # equal T // patch_size, the exported graph binds the patchified latent length
    # to the *encoder* sequence symbol. The result is a Reshape whose target
    # volume tracks enc_hidden instead of input_latents, which TensorRT later
    # rejects ("reshaping failed ... would change volume") for any profile where
    # T // patch_size != S. Keep T, T // patch_size and S pairwise distinct.
    patch_size = int(getattr(config, "patch_size", 2))
    B = 1
    T = 512   # latent sequence length (must be divisible by patch_size)
    S = 320   # encoder sequence length: distinct from T and from T // patch_size
    assert T % patch_size == 0, f"dummy T={T} must be divisible by patch_size={patch_size}"
    assert S != T and S != T // patch_size, (
        f"dummy trace shapes alias symbolic dims (T={T}, T//patch_size={T // patch_size}, "
        f"S={S}); choose S so that T, T//patch_size and S are pairwise distinct"
    )

    dummy_input_latents = torch.randn(B, T, 192, device=device, dtype=tensor_dtype)
    dummy_enc_hidden = torch.randn(B, S, 2048, device=device, dtype=tensor_dtype)
    dummy_t = torch.tensor([0.5], device=device, dtype=torch.float32)  # always fp32
    dummy_t_r = torch.tensor([0.5], device=device, dtype=torch.float32)  # always fp32
    # Padding masks for self-/cross-attention. All ones during tracing — the
    # model expands these to 4D additive biases and applies its trained
    # sliding-window pattern on even layers. At runtime the C++ host fills
    # 0 for padded positions so cross-attention ignores null_cond_vec tail
    # padding in enc_hidden (the timbre-dilution bug) and self-attention
    # ignores any latent padding beyond real_S[b].
    dummy_attention_mask = torch.ones(B, T, device=device, dtype=torch.long)
    dummy_encoder_attention_mask = torch.ones(B, S, device=device, dtype=torch.long)

    print(f"[export_dit] Tracing with shapes: input_latents={list(dummy_input_latents.shape)}, "
          f"enc_hidden={list(dummy_enc_hidden.shape)}, t={list(dummy_t.shape)}, "
          f"attention_mask={list(dummy_attention_mask.shape)}, "
          f"encoder_attention_mask={list(dummy_encoder_attention_mask.shape)}")
    print(f"[export_dit] Input dtype: {tensor_dtype}, t/t_r dtype: fp32, masks dtype: int64")

    print("[export_dit] Skipping PyTorch forward preflight for low-memory export")

    # Export to ONNX
    print(f"[export_dit] Exporting to ONNX (opset {opset})...")
    t0 = time.time()

    # Dynamo requires dynamic_shapes (not dynamic_axes)
    # Each input gets a dict mapping dim index → Dim object.
    # attention_mask shares the same seq_len Dim as input_latents (axis 1 == T),
    # and encoder_attention_mask shares enc_seq_len with enc_hidden (axis 1 == S).
    # Tying the Dims guarantees the runtime shapes agree, which TRT requires.
    #
    # patch_size constraint: the C++ host (pipeline-synth-ops.cpp:518
    # ops_resolve_T) rounds T up to a multiple of patch_size before calling
    # DiT, so at runtime T % patch_size == 0 always holds. We express this
    # to dynamo via a DERIVED DIM: `_seq_len_post = Dim(min=32, max=4096)`
    # is the post-patch sequence length, and `seq_len = patch_size *
    # _seq_len_post` is the pre-patch length. This tells dynamo that
    # seq_len is always a multiple of patch_size, which lets the modeling
    # file's `tensor.unfold(patch_size, patch_size).max(dim=-1)` downsample
    # the self-attn padding mask without triggering shape specialization.
    #
    # (torch 2.12's Dim API does not support a `modulo` parameter, so we
    # use the derived-dim approach instead.)
    batch = torch.export.Dim("batch", min=1, max=4)
    # _seq_len_post bounds: 32..7500 covers both the "default" profile
    # (max_T=3000 -> max post-patch=1500) and "full-10min" (max_T=15000 ->
    # max post-patch=7500). seq_len = patch_size * _seq_len_post.
    _seq_len_post = torch.export.Dim("_seq_len_post", min=32, max=7500)  # T // patch_size
    seq_len = patch_size * _seq_len_post                                  # T, always a multiple of patch_size
    enc_seq_len = torch.export.Dim("enc_seq_len", min=64, max=2048)

    dynamic_shapes = {
        "input_latents": {0: batch, 1: seq_len},
        "enc_hidden":    {0: batch, 1: enc_seq_len},
        "t":             {0: batch},
        "t_r":           {0: batch},
        "attention_mask":         {0: batch, 1: seq_len},
        "encoder_attention_mask": {0: batch, 1: enc_seq_len},
    }

    onnx_program = torch.onnx.export(
        wrapper,
        (dummy_input_latents, dummy_enc_hidden, dummy_t, dummy_t_r,
         dummy_attention_mask, dummy_encoder_attention_mask),
        None,
        opset_version=opset,
        input_names=["input_latents", "enc_hidden", "t", "t_r",
                     "attention_mask", "encoder_attention_mask"],
        output_names=["velocity"],
        dynamic_shapes=dynamic_shapes,
        export_params=False,
        keep_initializers_as_inputs=True,
        external_data=True,
        dynamo=True,
        optimize=False,
    )

    if not model_dir:
        raise SystemExit("low-memory export requires model_dir for safetensors streaming")
    graph_shell_path = output_path + ".graph-shell.onnx"
    for stale_path in (output_path, output_path + ".data", graph_shell_path, graph_shell_path + ".data"):
        if os.path.exists(stale_path):
            os.remove(stale_path)
    print("[export_dit] Saving graph without embedded initializers...")
    onnx_program.save(
        graph_shell_path,
        include_initializers=False,
        keep_initializers_as_inputs=True,
        external_data=True,
    )
    print(f"[export_dit] Streaming {precision} safetensors into ONNX external data...")
    try:
        stream_result = _externalize_initializers_from_safetensors(
            onnx_program,
            output_path,
            Path(model_dir),
            wrapper,
            precision=precision,
            chunk_mb=stream_chunk_mb,
            graph_shell_path=graph_shell_path,
        )
    except BaseException:
        for stale_path in (output_path, output_path + ".data"):
            if os.path.exists(stale_path):
                os.remove(stale_path)
        raise
    finally:
        if os.path.exists(graph_shell_path):
            os.remove(graph_shell_path)
    manifest = stream_result["manifest"]
    precision_report = stream_result["precision_report"]
    write_export_metadata(
        Path(output_path).with_suffix(".metadata.json"),
        ExportMetadata(
            family="acestep-v15",
            source_model=source_model,
            profile="dynamic-4input",
            precision_policy=precision,
            tensor_names_fp16=precision_report.get("matched_allowlist", []),
            tensor_names_fp32=precision_report.get("preserved_fp32", []),
        ),
    )

    t1 = time.time()
    data_path = output_path + ".data"
    onnx_size = os.path.getsize(output_path)
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    print(f"[export_dit] Low-memory stream manifest: {manifest['streamed_from_safetensors']} streamed, "
          f"{manifest['materialized_fallback_parameters']} small fallback")
    print(f"[export_dit] ONNX trace completed in {t1-t0:.1f}s")
    print(f"[export_dit] Exported to {output_path}")
    print(f"[export_dit] ONNX graph file: {onnx_size/1e6:.1f} MB")
    print(f"[export_dit] ONNX external weight data: {data_size/1e9:.2f} GB ({os.path.basename(data_path)})")
    print("[export_dit] Keep the .onnx and .onnx.data files together; pass the .onnx path to TensorRT.")
    for item in manifest.get("external_data_files", []):
        print(
            f"[export_dit] External data validated: {item['path']} "
            f"{item['size_bytes']/1e9:.2f} GB, {item['initializer_count']} initializers"
        )
    print(f"[export_dit] Total export time: {time.time()-t0:.1f}s")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export AceStep DiT to ONNX")
    parser.add_argument("--model-dir", required=True,
                        help="Path to the model directory (containing model.safetensors + config.json)")
    parser.add_argument("--output", default=None,
                        help="Output ONNX file path (default: models/onnx/dit_<model_name>.onnx)")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (default: 18)")
    parser.add_argument("--precision", default="q8map-fp16",
                        choices=["q8map-fp16", "w8a8", "fp32"],
                        help="Precision recipe (default: q8map-fp16). 'w8a8' = INT8 weights + INT8 activations with ConvRot rotation.")
    parser.add_argument("--device", default="cpu",
                        help="Device for model loading (default: cpu)")
    parser.add_argument("--stream-chunk-mb", type=int, default=16,
                        help="Chunk size in MB for low-memory tensor streaming (default: 16)")
    parser.add_argument("--convrot-group-size", type=int, default=None,
                        help="ConvRot Hadamard group size for w8a8 (default: 64). "
                             "Only 64 is compiled into the TensorRT plugin. "
                             "Overrides the HOTSTEP_CONVROT_GROUP_SIZE env var.")
    parser.add_argument("--force", action="store_true",
                        help="Re-export even if the output ONNX already exists")
    args = parser.parse_args()
    if args.convrot_group_size is not None:
        os.environ["HOTSTEP_CONVROT_GROUP_SIZE"] = str(args.convrot_group_size)

    # Default output path
    if args.output is None:
        model_name = Path(args.model_dir).name
        onnx_dir = Path(args.model_dir).parent.parent / "models" / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(onnx_dir / f"dit_{model_name}.onnx")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Skip if output already exists (resumable builds)
    if not args.force and os.path.isfile(args.output):
        print(f"[export_dit] Output already exists: {args.output} (use --force to re-export)")
        return

    # Load model
    dit_model, config = load_dit_model(
        args.model_dir,
        device=args.device,
        precision=args.precision,
    )

    # Export
    export_onnx(dit_model, config, args.output, opset=args.opset,
                precision=args.precision, source_model=args.model_dir,
                model_dir=args.model_dir, stream_chunk_mb=args.stream_chunk_mb)

    print("[export_dit] Done!")


if __name__ == "__main__":
    main()