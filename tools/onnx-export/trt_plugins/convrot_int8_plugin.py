"""ConvRotInt8Linear TensorRT plugin — Python helpers.

The production implementation lives in C++/Triton:
  * ``engine/src/plugins/convrot_int8_linear_plugin.{h,cpp}``
  * ``tools/onnx-export/extract_jit_cubins_autotune.py``

This module keeps the Python-side glue in one place:

* ``conv_rot_int8_linear_reference`` mirrors the plugin math in NumPy so unit
  tests can validate correctness without requiring TensorRT.
* ``make_convrot_int8_linear_onnx_node`` emits the ONNX custom-op node used by
  the exporter.
* ``register_plugins`` / ``is_registered`` load the compiled TensorRT plugin
  library when one is available.

Plugin contract
---------------
The regular exported ONNX custom op has three mandatory inputs and one optional input:

::

    ConvRotInt8Linear(
        x            : FP16/FP32[..., in_features]
        weight_q     : INT8[out_features, in_features]
        weight_scale : FP32[out_features]
        bias         : FP32[out_features] (optional)
    ) -> FP16/FP32[..., out_features]

Version 3 also supports a K1-only node returning ``(X_q, X_scale)`` and a
prequantized K2-only form consuming those tensors. Self-attention Q/K/V use
one K1-only node and three K2-only nodes.

The ConvRot Hadamard matrix is *not* an ONNX input. It is only used in Python
reference code and in the offline weight-rotation/export flow. Runtime rotation
inside the Triton kernels is implemented directly as an in-register butterfly.

For every M, including M==1, the runtime plugin uses the same two-kernel Triton
path:

1. Rotate each activation K-group and quantize it once into workspace.
2. Reuse that INT8 workspace across all output-channel tiles.

The former dedicated M==1 fused kernel was removed after profiling showed the
two-kernel BK64/BM128/BN128 path wins on the real M==1 workload too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

# Public op identity — used by export_dit.py when emitting the ONNX custom
# op and by build-trt-engine.py when verifying the plugin is registered.
CONVROT_INT8_LINEAR_OP_NAMESPACE = "hotstep"
CONVROT_INT8_LINEAR_OP_NAME = "ConvRotInt8Linear"
CONVROT_INT8_LINEAR_PLUGIN_VERSION = "3"

# Module-level flag set by register_plugins(). Used by is_registered() to
# avoid re-registering on every call (TRT logs a warning otherwise).
# In production, the actual registration happens in the C++ .so; this flag
# only tracks whether the Python side has confirmed the .so is loaded.
_PLUGIN_REGISTERED = False
# Handle to the loaded C++ plugin library (kept alive to prevent unloading)
_plugin_lib_handle = None


def conv_rot_int8_linear_reference(
    x: np.ndarray,
    weight_q: np.ndarray,
    weight_scale: np.ndarray,
    H: np.ndarray,
    bias: Optional[np.ndarray],
    group_size: int,
    in_features: int,
    out_features: int,
    has_bias: bool,
    output_dtype: str = "FP16",
) -> np.ndarray:
    """NumPy reference implementation of the ConvRotInt8Linear op.

    The runtime plugin uses the two-kernel path for all M: rotate/quantize
    activations into reusable workspace, then run an INT8 GEMM that dequantizes
    one K-group at a time. This reference follows that math:

      1. Apply the activation-side ConvRot transform group-wise. The compiled
         TensorRT plugin supports ``group_size == 256`` (per
         CONVROT_OPTIMAL_TWO_KERNEL_SPEC.md).  ``group_size == 0`` is also
         accepted as a no-rotation sentinel for testing with small inputs.
      2. Split K into ``group_size``-wide quantization groups.
      3. For each group, compute one activation scale per row, quantize the
         group to INT8, form the INT32 dot product against the matching weight
         slice, and immediately dequantize that partial sum.
      4. Add bias once after all K-groups have been accumulated.

    ``H`` is only used here for reference math. The runtime Triton kernels
    generate a 16x16 regular-Hadamard factor in registers and compute the
    full ``H_256`` via a separable ``tl.dot``-based Kronecker transform (not
    a dense matrix multiply).
    """
    if x.shape[-1] != in_features:
        raise ValueError(f"x last dim {x.shape[-1]} != in_features {in_features}")
    if weight_q.shape != (out_features, in_features):
        raise ValueError(f"weight_q shape {weight_q.shape} != ({out_features}, {in_features})")
    if weight_scale.shape != (out_features,):
        raise ValueError(f"weight_scale shape {weight_scale.shape} != ({out_features},)")
    if group_size not in (0, 256):
        raise ValueError(f"ConvRotInt8Linear plugin supports group_size=256 (or 0 for no-rotation test), got {group_size}")
    if group_size != 0 and H.shape != (group_size, group_size):
        raise ValueError(f"H shape {H.shape} != ({group_size}, {group_size})")
    if group_size != 0 and in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
    if has_bias:
        if bias is None:
            raise ValueError("has_bias=True but bias is None")
        if bias.shape != (out_features,):
            raise ValueError(f"bias shape {bias.shape} != ({out_features},)")
    elif bias is not None:
        raise ValueError("has_bias=False but bias is not None")

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    orig_shape = x_f32.shape

    if group_size == 0:
        # No-rotation sentinel (testing only): skip the Hadamard, treat the
        # entire row as one quantization group.
        x_rot = x_f32
        block_k = in_features if in_features > 0 else 1
    else:
        n_groups = in_features // group_size
        x_grouped = x_f32.reshape(*orig_shape[:-1], n_groups, group_size)
        H_f32 = np.ascontiguousarray(H, dtype=np.float32)
        x_rot = np.matmul(x_grouped, H_f32).reshape(*orig_shape[:-1], in_features)
        block_k = group_size

    x_rot_2d = x_rot.reshape(-1, in_features)
    weight_scale_2d = weight_scale.reshape(1, out_features)
    out_2d = np.zeros((x_rot_2d.shape[0], out_features), dtype=np.float32)

    for start in range(0, in_features, block_k):
        end = min(start + block_k, in_features)
        x_rot_g = x_rot_2d[:, start:end]
        group_max = np.max(np.abs(x_rot_g), axis=1, keepdims=True)
        group_scale = np.maximum(group_max, np.float32(1e-30)) / np.float32(127.0)
        x_q_g = np.clip(np.rint(x_rot_g / group_scale), -127, 127).astype(np.int8)

        w_q_g = weight_q[:, start:end].astype(np.int32)
        partial = np.matmul(x_q_g.astype(np.int32), w_q_g.T)
        out_2d += partial.astype(np.float32) * group_scale * weight_scale_2d

    if has_bias:
        bias_f32 = np.ascontiguousarray(bias, dtype=np.float32)
        out_2d += bias_f32.reshape(1, out_features)

    out = out_2d.reshape(*orig_shape[:-1], out_features)
    if output_dtype.upper() in {"FP16", "FLOAT16", "HALF"}:
        return out.astype(np.float16)
    if output_dtype.upper() in {"FP32", "FLOAT", "FLOAT32"}:
        return out.astype(np.float32)
    raise ValueError(f"unsupported output_dtype {output_dtype!r}; expected FP16 or FP32")


def _find_plugin_library() -> str | None:
    """Locate the hotstep_plugins shared library on disk.

    This function is used by the Python TensorRT engine builder before parsing
    a w8a8 ONNX graph. Be deliberately generous about build layouts: Windows
    users may build from ``engine/buildcuda.cmd`` (``engine/build``), from an IDE
    multi-config directory (``Release/`` or ``Debug/``), or place the DLL on
    ``PATH``. ``ctypes.util.find_library`` is not reliable for arbitrary DLLs on
    Windows, so PATH is scanned explicitly.

    Environment overrides:
      * ``HOTSTEP_PLUGIN_LIBRARY`` — full path to the DLL/.so/.dylib
      * ``HOTSTEP_PLUGINS_PATH`` — one or more directories/files (os.pathsep)
    """
    import ctypes.util

    this_dir = Path(__file__).resolve().parent
    # Walk up to find the repo root (directory containing engine/).
    repo_root = this_dir
    for _ in range(8):
        if (repo_root / "engine" / "CMakeLists.txt").is_file():
            break
        repo_root = repo_root.parent

    if sys.platform == "win32":
        lib_name = "hotstep_plugins.dll"
    elif sys.platform == "darwin":
        lib_name = "libhotstep_plugins.dylib"
    else:
        lib_name = "libhotstep_plugins.so"

    # Exact-file override.
    explicit = os.environ.get("HOTSTEP_PLUGIN_LIBRARY", "").strip()
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p)

    candidates: list[Path] = []

    def add_candidate(path_like) -> None:
        if not path_like:
            return
        try:
            p = Path(path_like)
        except TypeError:
            return
        if p not in candidates:
            candidates.append(p)

    # Directory/file override list.
    for entry in os.environ.get("HOTSTEP_PLUGINS_PATH", "").split(os.pathsep):
        if entry:
            add_candidate(entry)

    # Common in-tree build/output directories.
    for d in (
        this_dir,
        repo_root / "engine" / "build",
        repo_root / "engine" / "buildcuda",
        repo_root / "engine" / "build-cuda",
        repo_root / "engine" / "build-trt",
        repo_root / "build",
        repo_root / "buildcuda",
        repo_root / "build-trt",
        Path.cwd(),
    ):
        add_candidate(d)

    # TRT/CUDA and process library search paths.
    trt_root = os.environ.get("TENSORRT_ROOT", os.environ.get("TRT_ROOT", ""))
    if trt_root:
        for sub in ("bin", "lib", "lib64", ""):
            add_candidate(Path(trt_root) / sub if sub else Path(trt_root))

    path_env = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
    for entry in os.environ.get(path_env, "").split(os.pathsep):
        if entry:
            add_candidate(entry)

    # Direct checks plus common config subdirectories. If a candidate itself is
    # a file, accept it only if it has the expected library basename.
    subdirs = ("", "Release", "Debug", "RelWithDebInfo", "MinSizeRel", "bin", "lib", "lib64")
    for d in candidates:
        if d.is_file() and d.name == lib_name:
            return str(d)
        if not d.is_dir():
            continue
        for sub in subdirs:
            hit = (d / sub / lib_name) if sub else (d / lib_name)
            if hit.is_file():
                return str(hit)

    # Bounded recursive fallback under likely local build roots. This catches
    # multi-config generators that append extra target/config directories while
    # avoiding a full repository scan unless the directory exists.
    recursive_roots = [
        repo_root / "engine" / "build",
        repo_root / "engine" / "buildcuda",
        repo_root / "engine" / "build-cuda",
        repo_root / "build",
    ]
    for root in recursive_roots:
        if root.is_dir():
            try:
                for hit in root.rglob(lib_name):
                    if hit.is_file():
                        return str(hit)
            except OSError:
                pass

    # Last resort: let the OS/Python try to resolve it. On Windows this often
    # returns None for project-local DLLs, hence the explicit PATH scan above.
    sys_name = ctypes.util.find_library("hotstep_plugins")
    if sys_name:
        return sys_name

    return None


def register_plugins() -> bool:
    """Load the HOT-Step C++ plugin library and register it with TRT.

    Uses ctypes to dlopen/LoadLibrary the ``hotstep_plugins`` shared library
    and call ``hotstep_register_plugins()``, which registers the
    ConvRotInt8Linear IPluginCreatorV3One with TRT's plugin registry.
    This must be called BEFORE TRT's ONNX parser encounters the
    ``hotstep::ConvRotInt8Linear`` custom-op node.

    Returns True on success. Returns False (without raising) if:
      - The C++ plugin library cannot be found
      - TRT is not installed
      - The registration function returns non-zero
    """
    global _PLUGIN_REGISTERED, _plugin_lib_handle
    if _PLUGIN_REGISTERED:
        return True

    # Verify TRT is importable (the plugin DLL links against nvinfer)
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        return False

    lib_path = _find_plugin_library()
    if lib_path is None:
        print("[trt_plugins] WARNING: hotstep_plugins shared library not found; "
              "w8a8 engine builds will fail.", file=sys.stderr)
        return False

    import ctypes
    try:
        if sys.platform == "win32":
            # On Windows, the plugin DLL depends on TRT DLLs (nvinfer_11.dll etc.)
            # and CUDA DLLs. Add all likely search directories before LoadLibrary.
            # os.add_dll_directory() was added in Python 3.8 and is the supported
            # way to extend LoadLibraryEx search paths.
            _dll_dirs = []
            # 1. The directory containing the plugin DLL itself
            _dll_dirs.append(str(Path(lib_path).parent))
            # 2. TRT bin/ and lib/ from environment variables
            for env_var in ("TENSORRT_ROOT", "TRT_ROOT"):
                trt_root = os.environ.get(env_var, "")
                if trt_root:
                    for sub in ("bin", "lib", ""):
                        d = str(Path(trt_root) / sub) if sub else trt_root
                        if Path(d).is_dir():
                            _dll_dirs.append(d)
            # 3. Try to discover TRT location from the tensorrt Python package
            if not any("TensorRT" in d or "tensorrt" in d for d in _dll_dirs):
                try:
                    import tensorrt
                    trt_package_dir = str(Path(tensorrt.__file__).parent)
                    if Path(trt_package_dir).is_dir():
                        _dll_dirs.append(trt_package_dir)
                    # Also check parent (the SDK root)
                    trt_parent = str(Path(trt_package_dir).parent)
                    for sub in ("bin", "lib", ""):
                        d = str(Path(trt_parent) / sub) if sub else trt_parent
                        if Path(d).is_dir():
                            _dll_dirs.append(d)
                except Exception:
                    pass
            # 4. Common Windows TRT installation paths
            for common in (Path("D:/ai/TensorRT"), Path("C:/TensorRT"),
                           Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "NVIDIA" / "TensorRT"):
                if common.is_dir():
                    for sub in ("bin", "lib", ""):
                        d = str(common / sub) if sub else str(common)
                        if Path(d).is_dir():
                            _dll_dirs.append(d)
            # 5. CUDA toolkit directories from PATH
            for p in os.environ.get("PATH", "").split(os.pathsep):
                if "cuda" in p.lower() and Path(p).is_dir():
                    _dll_dirs.append(p)
            for d in dict.fromkeys(_dll_dirs):  # deduplicate while preserving order
                try:
                    os.add_dll_directory(d)
                except (OSError, FileNotFoundError):
                    pass
            _plugin_lib_handle = ctypes.cdll.LoadLibrary(lib_path)
        else:
            _plugin_lib_handle = ctypes.cdll.LoadLibrary(lib_path)
    except OSError as exc:
        print(f"[trt_plugins] WARNING: Cannot load {lib_path}: {exc}", file=sys.stderr)
        if sys.platform == "win32":
            print("[trt_plugins] HINT: Set TENSORRT_ROOT environment variable "
                  "to your TensorRT SDK directory (e.g. set TENSORRT_ROOT=D:\\ai\\TensorRT). "
                  "The plugin DLL depends on nvinfer_11.dll and cudart64_*.dll from the TRT bin/ directory.",
                  file=sys.stderr)
        return False

    # Call the registration entry point
    try:
        _plugin_lib_handle.hotstep_register_plugins.restype = ctypes.c_int
        rc = _plugin_lib_handle.hotstep_register_plugins()
        if rc != 0:
            print(f"[trt_plugins] WARNING: hotstep_register_plugins() returned {rc}", file=sys.stderr)
            return False
    except AttributeError as exc:
        print(f"[trt_plugins] WARNING: {lib_path} has no hotstep_register_plugins symbol: {exc}",
              file=sys.stderr)
        return False

    _PLUGIN_REGISTERED = True
    print(f"[trt_plugins] Registered ConvRotInt8Linear plugin from {lib_path}")
    return True


def is_registered() -> bool:
    """Return True if the ConvRotInt8Linear plugin is available.

    In production this means the C++ .so has been loaded by the engine
    runtime. In unit-test environments (no TRT installed) this returns
    False, and callers should fall back to the NumPy reference
    implementation.
    """
    return _PLUGIN_REGISTERED


# =============================================================================
# ONNX custom-op emission helpers
# =============================================================================

def _dtype_attr_id(tensor_proto_module, dtype: str) -> int:
    value = str(dtype).upper()
    if value in {"BF16", "BFLOAT16"}:
        return tensor_proto_module.BFLOAT16
    if value in {"FP32", "FLOAT", "FLOAT32"}:
        return tensor_proto_module.FLOAT
    return tensor_proto_module.FLOAT16


def make_convrot_int8_linear_onnx_node(
    helper_module,
    tensor_proto_module,
    x_name: str,
    weight_q_name: str,
    weight_scale_name: str,
    bias_name: str,
    output_name: str,
    node_name: str,
    group_size: int,
    in_features: int,
    out_features: int,
    has_bias: bool,
    input_dtype: str = "FP16",
    output_dtype: str = "FP16",
    preferred_format: str = "HWC8",
    prequantized: bool = False,
    activation_scale_name: str = "",
):
    """Build an ONNX custom-op node that maps to the ConvRotInt8Linear plugin.

    The node uses domain ``hotstep`` and op type ``ConvRotInt8Linear``. The
    build-time constants (group_size, in_features, out_features, has_bias)
    are carried as ONNX attributes so the TRT plugin creator can read them
    at parse time via the PluginFieldCollection. TensorRT's fallback plugin
    importer does not use the ONNX domain opset as the plugin version, so the
    v3 creator version/namespace are also emitted explicitly as
    ``plugin_version`` and ``plugin_namespace`` attributes.

    Args:
        helper_module: ``onnx.helper`` module (passed in to avoid a hard
            import dependency at module load time).
        tensor_proto_module: ``onnx.TensorProto`` module.
        x_name, weight_q_name, weight_scale_name, bias_name,
            output_name, node_name: ONNX tensor / node names.
        group_size, in_features, out_features: build-time constants.
        has_bias: whether the bias input is present.
        input_dtype: activation/H dtype requested from the plugin. Bias remains FP32.
        output_dtype: output dtype requested from the plugin (FP16 by default).
        preferred_format: layout hint reserved for future packed-format kernels.

    Returns:
        An ``onnx.NodeProto`` for the custom op.
    """
    if prequantized:
        if not activation_scale_name:
            raise ValueError("prequantized ConvRotInt8Linear requires activation_scale_name")
        inputs = [x_name, activation_scale_name, weight_q_name, weight_scale_name]
    else:
        inputs = [x_name, weight_q_name, weight_scale_name]
    if has_bias:
        inputs.append(bias_name)

    attrs = {
        "group_size": int(group_size),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "has_bias": int(1 if has_bias else 0),
        "input_dtype": str(input_dtype).upper(),
        "output_dtype": str(output_dtype).upper(),
        # Redundant integer dtype attrs make TensorRT PluginField parsing
        # robust across parser builds that report string attrs differently.
        # Values are ONNX TensorProto enum IDs: FLOAT=1, FLOAT16=10.
        "input_dtype_id": _dtype_attr_id(tensor_proto_module, input_dtype),
        "output_dtype_id": _dtype_attr_id(tensor_proto_module, output_dtype),
        # TensorRT ONNX parser fallback-plugin lookup keys. Without these it
        # defaults to plugin version "1" and empty namespace, which cannot find
        # the v3 creator registered as (ConvRotInt8Linear, "3", "hotstep").
        "plugin_version": CONVROT_INT8_LINEAR_PLUGIN_VERSION,
        "plugin_namespace": CONVROT_INT8_LINEAR_OP_NAMESPACE,
        "preferred_format": str(preferred_format).upper(),
        "prequantized": int(bool(prequantized)),
        "quantize_only": 0,
    }

    return helper_module.make_node(
        CONVROT_INT8_LINEAR_OP_NAME,
        inputs,
        [output_name],
        name=node_name,
        domain=CONVROT_INT8_LINEAR_OP_NAMESPACE,
        **attrs,
    )


def make_convrot_quantize_onnx_node(
    helper_module,
    tensor_proto_module,
    x_name: str,
    xq_name: str,
    xscale_name: str,
    node_name: str,
    group_size: int,
    in_features: int,
    input_dtype: str = "FP16",
):
    """Emit the K1-only form used by shared self-attention Q/K/V.

    ``X_q`` is INT8 with the same shape as X. ``X_scale`` is FP32 with
    shape ``[K/group_size, *X.shape[:-1]]`` and physical layout
    ``[groups, flattened_rows]``. Existing K1 cubins are used unchanged.
    """
    normalized_dtype = str(input_dtype).upper()
    if normalized_dtype not in {
        "FP16", "FLOAT16", "HALF", "FP32", "FLOAT", "FLOAT32"
    }:
        raise ValueError("ConvRot quantizer supports FP32/FP16 input only")

    # Keep the serialized boundary pair homogeneous.  Production extraction
    # now embeds only FP16IO; actual outputs remain explicitly INT8 and FP32.
    boundary_dtype_id = _dtype_attr_id(tensor_proto_module, input_dtype)
    attrs = {
        "group_size": int(group_size),
        "in_features": int(in_features),
        "out_features": int(in_features),
        "has_bias": 0,
        "input_dtype": normalized_dtype,
        "output_dtype": normalized_dtype,
        "input_dtype_id": boundary_dtype_id,
        "output_dtype_id": boundary_dtype_id,
        "plugin_version": CONVROT_INT8_LINEAR_PLUGIN_VERSION,
        "plugin_namespace": CONVROT_INT8_LINEAR_OP_NAMESPACE,
        "preferred_format": "HWC8",
        "prequantized": 0,
        "quantize_only": 1,
    }
    return helper_module.make_node(
        CONVROT_INT8_LINEAR_OP_NAME,
        [x_name],
        [xq_name, xscale_name],
        name=node_name,
        domain=CONVROT_INT8_LINEAR_OP_NAMESPACE,
        **attrs,
    )
