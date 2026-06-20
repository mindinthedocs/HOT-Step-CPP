"""ConvRotInt8Linear TensorRT plugin — Python helpers.

The actual IPluginV3 + IPluginCreatorV3 implementation lives in C++:
  engine/src/plugins/convrot_int8_linear_plugin.{h,cpp}
  engine/src/plugins/convrot_int8_linear_kernel.{cuh,cu}

This module provides:
  * ``conv_rot_int8_linear_reference`` — NumPy reference implementation
    used by unit tests to validate the C++ kernel's math without
    requiring a TRT install.
  * ``make_convrot_int8_linear_onnx_node`` — emits the ONNX custom-op
    node with the correct domain, op_type, inputs, outputs, and
    attributes. Called by ``export_dit._rewrite_w8a8_weight_ops_with_plugin``.
  * ``register_plugins`` / ``is_registered`` — thin shims that return
    False when the C++ .so isn't loaded (unit-test environments) and
    True when it is (production builds, after the engine runtime
    dlopens ``libhotstep_plugins.so``).

The plugin fuses the entire w8a8 + ConvRot path into a single TensorRT
custom op:

  1. Online activation rotation: x_rot = x @ H_block (group-wise Hadamard)
  2. Per-row dynamic INT8 quantization of x_rot
  3. INT8 × INT8 → INT32 matmul against the pre-rotated, per-output-channel
     INT8 weight
  4. Dequantize to FP32 via the per-row activation scale and
     per-output-channel weight scale
  5. Add bias (if present)

The kernel is a direct port of the ComfyUI-INT8-Fast Triton kernel
(``int8_fused_kernel.triton_int8_linear_per_row`` + ``convrot.rotate_activation``)
to the TensorRT 11 IPluginV3 API.

Op signature
------------

::

    ConvRotInt8Linear(
        x            : FP32[..., in_features]            # activation
        weight_q     : INT8[out_features, in_features]   # ConvRot-rotated, per-channel INT8
        weight_scale : FP32[out_features]                # per-output-channel symmetric scale
        H            : FP32[group_size, group_size]      # shared regular Hadamard matrix
        bias         : FP32[out_features] (optional)     # additive bias
    ) -> FP32[..., out_features]

Attributes (carried as ONNX op attributes, surfaced to the plugin at build
time):

  * ``group_size``   : int   — ConvRot block size (power of 4); 0 = skip rotation
  * ``in_features``  : int   — K dimension of the matmul
  * ``out_features`` : int   — N dimension of the matmul
  * ``has_bias``     : bool  — whether the bias input is present

Fusion notes
------------
The plugin is a single TRT layer, so TRT's standard fusion passes treat
it as an atomic op. Surrounding pointwise ops (Cast, Add, Mul, etc.)
that TRT would normally fuse into a MatMul epilogue are NOT fused into
this plugin — that's intentional, because the plugin already includes
the dequant + bias epilogue. Any residual pointwise ops the export
pipeline emits (e.g. a final Cast to FP32 to satisfy the graph I/O
policy) will fuse with the plugin's output naturally via TRT's
pointwise fusion.

The plugin supports only ``kLINEAR`` format (no vectorized layouts)
because the ConvRot rotation requires element-wise access patterns that
don't align with HWNC/CHWN32 etc. TensorRT will still tile the GEMM
internally via cuBLASLt INT8 paths.
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
) -> np.ndarray:
    """NumPy reference implementation of the ConvRotInt8Linear op.

    Mirrors the math of the C++ kernel in
    ``engine/src/plugins/convrot_int8_linear_kernel.cu`` line-for-line:

      1. Reshape x from ``[..., in_features]`` to ``[..., n_groups, group_size]``
         and apply the shared regular Hadamard matrix H (ConvRot rotation).
         When ``group_size == 0`` the rotation is skipped (the plugin
         treats 0 as a "no rotation" sentinel for layers where
         in_features % group_size != 0).
      2. Per-row symmetric INT8 quantization of the rotated activation:
         scale = max(|x_rot|, -1) / 127; q = round(x_rot / scale);
         clip to [-127, 127].
      3. INT8 × INT8 → INT32 matmul against weight_q^T (which is already
         ConvRot-rotated offline and per-output-channel quantized).
      4. Dequantize: y = int32_acc * x_scale[..., None] * weight_scale[None, :]
      5. Add bias if present.

    Args:
        x: FP32 activation, shape ``[..., in_features]``.
        weight_q: INT8 weight in canonical ``[out_features, in_features]``
            layout, already ConvRot-rotated offline and per-output-channel
            quantized.
        weight_scale: FP32 per-output-channel weight scale, shape ``[out_features]``.
        H: FP32 normalized regular Hadamard matrix, shape
            ``[group_size, group_size]``. May be empty when ``group_size == 0``.
        bias: FP32 bias, shape ``[out_features]``, or None if ``has_bias`` is False.
        group_size: ConvRot block size (power of 4). 0 = skip rotation.
        in_features: K dimension of the matmul.
        out_features: N dimension of the matmul.
        has_bias: Whether the bias input is present (must match the bias arg).

    Returns:
        FP32 output, shape ``[..., out_features]``.
    """
    if x.shape[-1] != in_features:
        raise ValueError(
            f"x last dim {x.shape[-1]} != in_features {in_features}"
        )
    if weight_q.shape != (out_features, in_features):
        raise ValueError(
            f"weight_q shape {weight_q.shape} != ({out_features}, {in_features})"
        )
    if weight_scale.shape != (out_features,):
        raise ValueError(
            f"weight_scale shape {weight_scale.shape} != ({out_features},)"
        )
    if group_size != 0:
        if H.shape != (group_size, group_size):
            raise ValueError(
                f"H shape {H.shape} != ({group_size}, {group_size})"
            )
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features {in_features} not divisible by group_size {group_size}"
            )
    if has_bias:
        if bias is None:
            raise ValueError("has_bias=True but bias is None")
        if bias.shape != (out_features,):
            raise ValueError(
                f"bias shape {bias.shape} != ({out_features},)"
            )
    elif bias is not None:
        raise ValueError("has_bias=False but bias is not None")

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    orig_shape = x_f32.shape

    # 1. ConvRot activation rotation: x_rot = x @ H_block.
    #    Skipped when group_size == 0 (the "no rotation" sentinel).
    if group_size != 0:
        n_groups = in_features // group_size
        x_grouped = x_f32.reshape(*orig_shape[:-1], n_groups, group_size)
        H_f32 = np.ascontiguousarray(H, dtype=np.float32)
        x_rot_grouped = np.matmul(x_grouped, H_f32)
        x_rot = x_rot_grouped.reshape(*orig_shape[:-1], in_features)
    else:
        x_rot = x_f32

    # 2. Per-row symmetric INT8 quantization of x_rot.
    abs_max = np.max(np.abs(x_rot), axis=-1, keepdims=True)
    abs_max_clamped = np.maximum(abs_max, np.float32(1e-30))
    x_scale = (abs_max_clamped / np.float32(127.0)).astype(np.float32)
    x_scaled = x_rot / x_scale
    x_q = np.clip(np.rint(x_scaled), -127, 127).astype(np.int8)

    # 3. INT8 × INT8 → INT32 matmul.
    x_q_2d = x_q.reshape(-1, in_features)
    acc = np.matmul(
        x_q_2d.astype(np.int32),
        weight_q.astype(np.int32).T,
    )

    # 4. Dequantize: acc * x_scale[M, 1] * weight_scale[None, N]
    x_scale_2d = x_scale.reshape(-1, 1)
    out_2d = (acc.astype(np.float32)
              * x_scale_2d
              * weight_scale.reshape(1, out_features))

    # 5. Add bias if present.
    if has_bias:
        bias_f32 = np.ascontiguousarray(bias, dtype=np.float32)
        out_2d = out_2d + bias_f32.reshape(1, out_features)

    # Reshape back to [..., out_features].
    return out_2d.reshape(*orig_shape[:-1], out_features)


def _find_plugin_library() -> str | None:
    """Locate the hotstep_plugins shared library on disk.

    Search order:
      1. Same directory as this Python file (editable install / dev)
      2. ``engine/build/`` relative to the repo root (CMake build dir)
      3. ``engine/buildcuda/`` (alternative CMake build dir)
      4. ``PATH`` / ``LD_LIBRARY_PATH`` (system paths)
    Returns the path as a string, or None if not found.
    """
    import ctypes.util
    import glob

    this_dir = Path(__file__).resolve().parent
    # Walk up to find the repo root (directory containing engine/)
    repo_root = this_dir
    for _ in range(6):
        if (repo_root / "engine" / "CMakeLists.txt").is_file():
            break
        repo_root = repo_root.parent

    if sys.platform == "win32":
        lib_name = "hotstep_plugins.dll"
    elif sys.platform == "darwin":
        lib_name = "libhotstep_plugins.dylib"
    else:
        lib_name = "libhotstep_plugins.so"

    # Candidate directories to search
    candidates = [
        this_dir,                          # same dir as this .py
        repo_root / "engine" / "build",    # standard CMake build dir
        repo_root / "engine" / "buildcuda",# alternative build dir
    ]
    # Also add any directory from the TRT bin/ path if TENSORRT_ROOT is set
    trt_root = os.environ.get("TENSORRT_ROOT", os.environ.get("TRT_ROOT", ""))
    if trt_root:
        candidates.append(Path(trt_root) / "bin")

    for d in candidates:
        p = d / lib_name
        if p.is_file():
            return str(p)
        # On Windows, CMake may put DLLs in Release/ or Debug/ subdirs
        for sub in ("Release", "Debug", ""):
            hit = d / sub / lib_name
            if hit.is_file():
                return str(hit)

    # Last resort: let the OS find it via PATH / LD_LIBRARY_PATH
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

def make_convrot_int8_linear_onnx_node(
    helper_module,
    tensor_proto_module,
    x_name: str,
    weight_q_name: str,
    weight_scale_name: str,
    H_name: str,
    bias_name: str,
    output_name: str,
    node_name: str,
    group_size: int,
    in_features: int,
    out_features: int,
    has_bias: bool,
):
    """Build an ONNX custom-op node that maps to the ConvRotInt8Linear plugin.

    The node uses domain ``hotstep`` and op type ``ConvRotInt8Linear``. The
    build-time constants (group_size, in_features, out_features, has_bias)
    are carried as ONNX attributes so the TRT plugin creator can read them
    at parse time via the PluginFieldCollection.

    Args:
        helper_module: ``onnx.helper`` module (passed in to avoid a hard
            import dependency at module load time).
        tensor_proto_module: ``onnx.TensorProto`` module.
        x_name, weight_q_name, weight_scale_name, H_name, bias_name,
            output_name, node_name: ONNX tensor / node names.
        group_size, in_features, out_features: build-time constants.
        has_bias: whether the bias input is present.

    Returns:
        An ``onnx.NodeProto`` for the custom op.
    """
    inputs = [x_name, weight_q_name, weight_scale_name, H_name]
    if has_bias:
        inputs.append(bias_name)

    attrs = {
        "group_size": int(group_size),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "has_bias": int(1 if has_bias else 0),
    }

    return helper_module.make_node(
        CONVROT_INT8_LINEAR_OP_NAME,
        inputs,
        [output_name],
        name=node_name,
        domain=CONVROT_INT8_LINEAR_OP_NAMESPACE,
        **attrs,
    )
