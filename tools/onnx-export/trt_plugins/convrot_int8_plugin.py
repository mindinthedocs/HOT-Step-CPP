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
  4. Dequantize via the per-row activation scale and per-output-channel
     weight scale
  5. Add bias (if present) and return FP16 by default

The kernel is a direct port of the ComfyUI-INT8-Fast Triton kernel
(``int8_fused_kernel.triton_int8_linear_per_row`` + ``convrot.rotate_activation``)
to the TensorRT 11 IPluginV3 API.

Op signature
------------

::

    ConvRotInt8Linear(
        x            : FP16[..., in_features]            # activation (default)
        weight_q     : INT8[out_features, in_features]   # ConvRot-rotated, per-channel INT8
        weight_scale : FP32[out_features]                # per-output-channel symmetric scale
        H            : FP16[group_size, group_size]      # shared regular Hadamard matrix
        bias         : FP32[out_features] (optional)     # additive bias; 1D tensors stay FP32
    ) -> FP16[..., out_features]

Attributes (carried as ONNX op attributes, surfaced to the plugin at build
time):

  * ``group_size``   : int   — ConvRot block size (power of 4); 0 = skip rotation
  * ``in_features``  : int   — K dimension of the matmul
  * ``out_features`` : int   — N dimension of the matmul
  * ``has_bias``     : bool  — whether the bias input is present
  * ``input_dtype``  : str   — ``FP16`` by default, ``FP32`` fallback
  * ``output_dtype`` : str   — ``FP16`` by default, ``FP32`` fallback
  * ``preferred_format`` : str — layout hint for future packed-format kernels

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

The v2 ONNX contract is independent of kernel optimization choices: the graph
emits dtype and layout-preference attributes once, while TensorRT chooses plugin
tactics during engine build. The current C++ kernel accepts contiguous
``kLINEAR`` tensors and exposes a stable tactic surface; future packed-format
kernels can consume the existing ``preferred_format`` attribute without another
ONNX rewrite.
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
CONVROT_INT8_LINEAR_PLUGIN_VERSION = "2"

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
        Output with dtype selected by ``output_dtype`` (FP16 by default), shape
        ``[..., out_features]``.
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

    # Reshape back to [..., out_features] and apply the v2 plugin output
    # contract. Accumulation and scaling stay FP32; only the boundary tensor
    # is narrowed, matching the CUDA epilogue.
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
):
    """Build an ONNX custom-op node that maps to the ConvRotInt8Linear plugin.

    The node uses domain ``hotstep`` and op type ``ConvRotInt8Linear``. The
    build-time constants (group_size, in_features, out_features, has_bias)
    are carried as ONNX attributes so the TRT plugin creator can read them
    at parse time via the PluginFieldCollection. TensorRT's fallback plugin
    importer does not use the ONNX domain opset as the plugin version, so the
    v2 creator version/namespace are also emitted explicitly as
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
        # the v2 creator registered as (ConvRotInt8Linear, "2", "hotstep").
        "plugin_version": CONVROT_INT8_LINEAR_PLUGIN_VERSION,
        "plugin_namespace": CONVROT_INT8_LINEAR_OP_NAMESPACE,
        "preferred_format": str(preferred_format).upper(),
    }

    return helper_module.make_node(
        CONVROT_INT8_LINEAR_OP_NAME,
        inputs,
        [output_name],
        name=node_name,
        domain=CONVROT_INT8_LINEAR_OP_NAMESPACE,
        **attrs,
    )
