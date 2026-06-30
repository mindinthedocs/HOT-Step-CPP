#!/usr/bin/env python3
"""Extract autotune-selected cubins from Triton's JIT cache.

This version parses Triton's internal cache key to extract the exact
constexpr kwargs alongside metadata, ensuring we select the precise winning
cubin even when configs collide on (num_warps, num_stages).
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

sys.setrecursionlimit(200000)

try:
    import triton
    import triton.language as tl
    from triton import Config
    from triton.language.extra import libdevice
except ImportError as exc:
    print(f"ERROR: Triton is required to run this script: {exc}", file=sys.stderr)
    raise


_K1_CONFIGS = [
    Config({"BLOCK_M": 64},  num_warps=2, num_stages=4),
    Config({"BLOCK_M": 64},  num_warps=4, num_stages=3),
    Config({"BLOCK_M": 64},  num_warps=4, num_stages=4),
    Config({"BLOCK_M": 128}, num_warps=4, num_stages=3),
    Config({"BLOCK_M": 128}, num_warps=4, num_stages=4),
    Config({"BLOCK_M": 128}, num_warps=8, num_stages=2),
    Config({"BLOCK_M": 128}, num_warps=8, num_stages=3),
]

_K2_CONFIGS = [
    Config({"BLOCK_M": 64,  "BLOCK_N": 128, "GROUP_M": 4}, num_warps=2, num_stages=5),
    Config({"BLOCK_M": 64,  "BLOCK_N": 128, "GROUP_M": 4}, num_warps=4, num_stages=3),
    Config({"BLOCK_M": 64,  "BLOCK_N": 128, "GROUP_M": 4}, num_warps=4, num_stages=4),
    Config({"BLOCK_M": 64,  "BLOCK_N": 256, "GROUP_M": 4}, num_warps=4, num_stages=4),
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 2}, num_warps=4, num_stages=3),
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 4}, num_warps=4, num_stages=3),
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 4}, num_warps=4, num_stages=4),
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 4}, num_warps=8, num_stages=2),
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 4}, num_warps=8, num_stages=3),
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 4}, num_warps=8, num_stages=1),
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 6}, num_warps=8, num_stages=2),
    Config({"BLOCK_M": 128, "BLOCK_N": 256, "GROUP_M": 2}, num_warps=8, num_stages=3),
    Config({"BLOCK_M": 128, "BLOCK_N": 256, "GROUP_M": 4}, num_warps=8, num_stages=3),
    Config({"BLOCK_M": 256, "BLOCK_N": 128, "GROUP_M": 2}, num_warps=8, num_stages=3),
    Config({"BLOCK_M": 256, "BLOCK_N": 256, "GROUP_M": 2}, num_warps=8, num_stages=2),
]


@triton.jit
def _hadamard_butterfly_stage(x_tile, BLOCK_M: tl.constexpr, GROUP_SIZE: tl.constexpr, STAGE: tl.constexpr):
    s: tl.constexpr = 4 ** STAGE
    n_groups: tl.constexpr = GROUP_SIZE // (4 * s)
    x_v = tl.reshape(x_tile, (BLOCK_M, n_groups, 4, s))
    x_v = tl.permute(x_v, (0, 1, 3, 2))
    x_2x2 = tl.reshape(x_v, (BLOCK_M, n_groups, s, 2, 2))
    lo, hi = tl.split(x_2x2)
    v0, v1 = tl.split(lo)
    v2, v3 = tl.split(hi)
    h0 = 0.5 * (v0 + v1 + v2 - v3)
    h1 = 0.5 * (v0 + v1 - v2 + v3)
    h2 = 0.5 * (v0 - v1 + v2 + v3)
    h3 = 0.5 * (-v0 + v1 + v2 + v3)
    lo_out = tl.join(h0, h1)
    hi_out = tl.join(h2, h3)
    out = tl.join(lo_out, hi_out)
    x_v = tl.reshape(out, (BLOCK_M, n_groups, s, 4))
    x_v = tl.permute(x_v, (0, 1, 3, 2))
    return tl.reshape(x_v, (BLOCK_M, GROUP_SIZE))


@triton.autotune(configs=_K1_CONFIGS, key=["M", "K", "INPUT_FP16"])
@triton.jit
def kernel1_convrot_quant(
    X_ptr, X_q_ptr, X_scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_xqm, stride_xqk,
    stride_xsm, stride_xsg,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    INPUT_FP16: tl.constexpr,
):
    pid_m = tl.program_id(0) * BLOCK_M
    pid_k = tl.program_id(1) * BLOCK_K
    rm = pid_m + tl.arange(0, BLOCK_M)
    rk = pid_k + tl.arange(0, BLOCK_K)
    ram = rm % M
    mask = (rm[:, None] < M) & (rk[None, :] < K)
    x_ptr_block = X_ptr + ram[:, None] * stride_xm + rk[None, :] * stride_xk
    x_block = tl.load(x_ptr_block, mask=mask, other=0.0, eviction_policy="evict_first")
    if INPUT_FP16:
        x_block = x_block.to(tl.float32)
    for stage in tl.static_range(0, 3):
        x_block = _hadamard_butterfly_stage(x_block, BLOCK_M, GROUP_SIZE, stage)
    block_max = tl.max(tl.abs(x_block), axis=1)
    scale = tl.maximum(block_max / 127.0, 1e-30)
    x_q = libdevice.rint(x_block / scale[:, None])
    x_q = tl.clamp(x_q, -127.0, 127.0).to(tl.int8)
    xq_ptr_block = X_q_ptr + rm[:, None] * stride_xqm + rk[None, :] * stride_xqk
    tl.store(xq_ptr_block, x_q, mask=mask)
    group_idx = pid_k // BLOCK_K
    xs_ptr_block = X_scale_ptr + rm * stride_xsm + group_idx * stride_xsg
    tl.store(xs_ptr_block, scale, mask=rm < M)


@triton.autotune(configs=_K2_CONFIGS, key=["M", "N", "K", "HAS_BIAS", "OUTPUT_FP16"])
@triton.jit
def kernel2_gemm_dequant(
    X_q_ptr, X_scale_ptr, W_q_ptr, W_scale_ptr, Bias_ptr, Y_ptr,
    M, N, K,
    stride_xqm, stride_xqk,
    stride_xsm, stride_xsg,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    pid_m = pid_m * BLOCK_M
    pid_n = pid_n * BLOCK_N
    rm = pid_m + tl.arange(0, BLOCK_M)
    rn = pid_n + tl.arange(0, BLOCK_N)
    ram = rm % M
    rbn = rn % N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ws = tl.load(W_scale_ptr + rn, mask=rn < N, other=0.0)
    for k_offset in tl.range(0, K, BLOCK_K):
        cols_k = k_offset + tl.arange(0, BLOCK_K)
        xq_ptr_block = X_q_ptr + ram[:, None] * stride_xqm + cols_k[None, :] * stride_xqk
        xq = tl.load(xq_ptr_block, mask=(rm[:, None] < M) & (cols_k[None, :] < K), other=0, eviction_policy="evict_last")
        wq_ptr_block = W_q_ptr + rbn[:, None] * stride_wn + cols_k[None, :] * stride_wk
        wq = tl.load(wq_ptr_block, mask=(rn[:, None] < N) & (cols_k[None, :] < K), other=0, eviction_policy="evict_first")
        partial = tl.dot(xq, tl.trans(wq), allow_tf32=False)
        group_idx = k_offset // BLOCK_K
        xs_ptr_block = X_scale_ptr + ram * stride_xsm + group_idx * stride_xsg
        xs = tl.load(xs_ptr_block, mask=rm < M, other=0.0)
        acc += partial.to(tl.float32) * xs[:, None] * ws[None, :]
    if HAS_BIAS:
        bias = tl.load(Bias_ptr + rn, mask=rn < N, other=0.0)
        acc += bias[None, :]
    y_ptr_block = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    if OUTPUT_FP16:
        tl.store(y_ptr_block, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))
    else:
        tl.store(y_ptr_block, acc.to(tl.float32), mask=(rm[:, None] < M) & (rn[None, :] < N))


GROUP_SIZE = 64
BLOCK_K = 64
TILE_NAME = "128X128"
CONVROT_INT8_CUBIN_HEADER_VERSION = 5
BIAS_CONFIGS = [("NOBIAS", False), ("BIAS", True)]
DTYPE_CONFIGS = [("FP16IO", True, True), ("FP32IO", False, False)]
AUTOTUNE_SHAPES_K2 = [
    (1024, 2560, 9728),
    (1024, 2560, 2560),
    (1024, 2560, 4096),
    (1024, 2560, 1024),
    (384, 2560, 1024),
]


_LAUNCH_ARG_ENUM = {
    "X_ptr": "kX_ptr",
    "X_q_ptr": "kX_q_ptr",
    "X_scale_ptr": "kX_scale_ptr",
    "W_q_ptr": "kW_q_ptr",
    "W_scale_ptr": "kW_scale_ptr",
    "Bias_ptr": "kBias_ptr",
    "Y_ptr": "kY_ptr",
    "M": "kM",
    "N": "kN",
    "K": "kK",
    "stride_xm": "kStride_xm",
    "stride_xk": "kStride_xk",
    "stride_xqm": "kStride_xqm",
    "stride_xqk": "kStride_xqk",
    "stride_xsm": "kStride_xsm",
    "stride_xsg": "kStride_xsg",
    "stride_wn": "kStride_wn",
    "stride_wk": "kStride_wk",
    "stride_ym": "kStride_ym",
    "stride_yn": "kStride_yn",
}

_POINTER_ARG_NAMES = {
    "X_ptr",
    "X_q_ptr",
    "X_scale_ptr",
    "W_q_ptr",
    "W_scale_ptr",
    "Bias_ptr",
    "Y_ptr",
}

_QUANT_ARG_NAMES = {
    "X_ptr",
    "X_q_ptr",
    "X_scale_ptr",
    "M",
    "K",
    "stride_xm",
    "stride_xk",
    "stride_xqm",
    "stride_xqk",
    "stride_xsm",
    "stride_xsg",
}

_GEMM_ARG_NAMES = {
    "X_q_ptr",
    "X_scale_ptr",
    "W_q_ptr",
    "W_scale_ptr",
    "Bias_ptr",
    "Y_ptr",
    "M",
    "N",
    "K",
    "stride_xqm",
    "stride_xqk",
    "stride_xsm",
    "stride_xsg",
    "stride_wn",
    "stride_wk",
    "stride_ym",
    "stride_yn",
}

_PTX_ENTRY_RE = re.compile(
    r"\.visible\s+\.entry\s+(?P<name>\w+)\s*\((?P<params>.*?)\)\s*(?P<after>.*)",
    re.S,
)
_PTX_REQNTID_RE = re.compile(r"\.reqntid\s+(\d+)(?:\s*,\s*(\d+)\s*,\s*(\d+))?")
_PTX_MAXNREG_RE = re.compile(r"\.maxnreg\s+(\d+)")
_PTX_PARAM_TYPE_RE = re.compile(r"\.param\s+\.(?P<kind>\w+)")


def _normalize_sig_type(sig_type):
    return str(sig_type).strip().replace(" ", "")


def _runtime_kind_from_sig_type(sig_type: str) -> str | None:
    sig_type = _normalize_sig_type(sig_type)
    # Triton keeps compile-time meta-parameters in some signature views as the
    # sentinel string "constexpr". They are not part of the runtime launch ABI
    # and must be filtered out before we build C++ launch stubs.
    if sig_type == "constexpr" or sig_type.startswith("constexpr["):
        return None
    if sig_type.startswith("*") or sig_type == "nvTmaDesc":
        return "ptr"
    if sig_type in {"i1", "i8", "i16", "i32", "i64", "u1", "u8", "u16", "u32", "u64", "fp16", "bf16", "fp32", "fp64"}:
        return "scalar"
    raise RuntimeError(f"Unsupported Triton signature type in extracted kernel ABI: {sig_type}")


def _ordered_runtime_signature(kernel):
    src = getattr(kernel, "src", None)
    signature = getattr(src, "signature", None)
    if not isinstance(signature, dict) or not signature:
        raise RuntimeError("CompiledKernel.src.signature is missing or empty; cannot derive launch ABI")

    fn = getattr(src, "fn", None)
    arg_names = list(getattr(fn, "arg_names", []) or [])
    ordered = []
    consumed = set()

    def append_item(key, name_hint=None):
        if key in signature and key not in consumed:
            ordered.append({
                "name": name_hint or str(key),
                "sig_type": _normalize_sig_type(signature[key]),
            })
            consumed.add(key)

    for idx, arg_name in enumerate(arg_names):
        if arg_name in signature:
            append_item(arg_name, arg_name)
        elif idx in signature:
            append_item(idx, arg_name)

    def sort_key(key):
        if isinstance(key, int):
            return (0, key)
        return (1, str(key))

    for key in sorted((k for k in signature.keys() if k not in consumed), key=sort_key):
        name = arg_names[key] if isinstance(key, int) and 0 <= key < len(arg_names) else str(key)
        append_item(key, name)

    if not ordered:
        raise RuntimeError("Failed to derive an ordered runtime signature from CompiledKernel.src.signature")
    return ordered


def _parse_ptx_entry_abi(ptx_text: str, function_name: str):
    for match in _PTX_ENTRY_RE.finditer(ptx_text):
        if match.group("name") != function_name:
            continue
        params_block = match.group("params")
        after_block = match.group("after")
        param_kinds = []
        param_types = []
        for line in params_block.splitlines():
            line = line.strip().rstrip(",")
            if not line.startswith(".param"):
                continue
            type_match = _PTX_PARAM_TYPE_RE.search(line)
            if type_match is None:
                raise RuntimeError(f"Could not parse PTX param declaration: {line!r}")
            raw_type = type_match.group("kind")
            is_ptr = ".ptr" in line
            param_kinds.append("ptr" if is_ptr else "scalar")
            param_types.append(("*" if is_ptr else "") + raw_type)
        reqntid_match = _PTX_REQNTID_RE.search(after_block)
        if reqntid_match is None:
            raise RuntimeError(f"Could not locate .reqntid for PTX entry {function_name}")
        reqntid = (
            int(reqntid_match.group(1)),
            int(reqntid_match.group(2) or 1),
            int(reqntid_match.group(3) or 1),
        )
        maxnreg_match = _PTX_MAXNREG_RE.search(after_block)
        maxnreg = int(maxnreg_match.group(1)) if maxnreg_match else None
        return {
            "function_name": function_name,
            "param_kinds": param_kinds,
            "param_types": param_types,
            "reqntid": reqntid,
            "maxnreg": maxnreg,
        }
    raise RuntimeError(f"Failed to locate PTX .entry for {function_name}")


def _extract_kernel_abi(kernel, stage_name: str):
    function_name = getattr(kernel, "name", None) or getattr(getattr(kernel, "metadata", None), "name", None)
    if not function_name:
        raise RuntimeError("CompiledKernel has no name metadata; cannot derive launch ABI")

    runtime_signature = _ordered_runtime_signature(kernel)
    allowed_names = _QUANT_ARG_NAMES if stage_name == "quant" else _GEMM_ARG_NAMES
    filtered_runtime_signature = []
    for item in runtime_signature:
        name = item["name"]
        sig_type = item["sig_type"]
        kind = _runtime_kind_from_sig_type(sig_type)
        if kind is None:
            # Compile-time-only Triton meta-parameter, not part of the launch ABI.
            continue
        if name not in _LAUNCH_ARG_ENUM:
            raise RuntimeError(f"Unsupported runtime kernel argument in extracted ABI: {name}")
        if name not in allowed_names:
            raise RuntimeError(f"Argument {name} is not valid for {stage_name} launch stubs")
        if name in _POINTER_ARG_NAMES and kind != "ptr":
            raise RuntimeError(f"Expected pointer type for {name}, got {sig_type}")
        if name not in _POINTER_ARG_NAMES and sig_type not in {"i32", "u32"}:
            raise RuntimeError(f"Expected 32-bit scalar type for {name}, got {sig_type}")
        item["kind"] = kind
        item["enum"] = _LAUNCH_ARG_ENUM[name]
        filtered_runtime_signature.append(item)

    ptx = kernel.asm.get("ptx")
    if not isinstance(ptx, str):
        raise RuntimeError(f"CompiledKernel for {function_name} has no PTX text; cannot validate launch ABI")
    ptx_abi = _parse_ptx_entry_abi(ptx, function_name)

    runtime_signature = filtered_runtime_signature
    expected_kinds = [item["kind"] for item in runtime_signature]
    actual_kinds = ptx_abi["param_kinds"]
    if len(actual_kinds) < len(expected_kinds):
        raise RuntimeError(
            f"PTX ABI for {function_name} has fewer params than runtime signature: "
            f"ptx={len(actual_kinds)} runtime={len(expected_kinds)}"
        )
    prefix_kinds = actual_kinds[:len(expected_kinds)]
    if prefix_kinds != expected_kinds:
        raise RuntimeError(
            f"PTX/runtime ABI kind mismatch for {function_name}: "
            f"runtime={expected_kinds} ptx_prefix={prefix_kinds}"
        )
    scratch_tail = actual_kinds[len(expected_kinds):]
    if any(kind != "ptr" for kind in scratch_tail):
        raise RuntimeError(
            f"Unexpected non-pointer PTX tail parameters for {function_name}: "
            f"runtime={expected_kinds} ptx={actual_kinds}"
        )

    return {
        "function_name": function_name,
        "runtime_signature": runtime_signature,
        "scratch_ptr_count": len(scratch_tail),
        "ptx_param_count": len(actual_kinds),
        "ptx_param_types": ptx_abi["param_types"],
        "block": ptx_abi["reqntid"],
        "ptx_maxnreg": ptx_abi["maxnreg"],
    }


def _find_jit_holder(obj):
    seen = set()
    cur = obj
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "device_caches"):
            return cur
        cur = getattr(cur, "fn", None)
    return None


def _debug_wrapper_chain(obj):
    cur = obj
    i = 0
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        print(
            f"[debug] layer {i}: type={type(cur)!r}, "
            f"has_fn={hasattr(cur, 'fn')}, "
            f"has_device_caches={hasattr(cur, 'device_caches')}",
            flush=True,
        )
        cur = getattr(cur, "fn", None)
        i += 1


def _safe_cache_entry(device_caches, device):
    if device not in device_caches:
        raise RuntimeError(f"Device {device} not in JITFunction cache — kernel was not compiled?")
    cache_entry = device_caches[device]
    if isinstance(cache_entry, tuple):
        kernel_cache = cache_entry[0]
    else:
        kernel_cache = cache_entry
    if not kernel_cache:
        raise RuntimeError("kernel_cache is empty — autotune did not compile any kernels?")
    return kernel_cache


def _load_kernel_sidecar(kernel):
    """Load Triton sidecar JSON metadata for a compiled kernel if available."""
    for attr in ("cache_path", "fn", "path"):
        v = getattr(kernel, attr, None)
        if isinstance(v, (str, os.PathLike)):
            p = Path(v)
            if p.suffix == ".json" and p.exists():
                return json.loads(p.read_text())
            if p.exists():
                sidecar = p.with_suffix(".json")
                if sidecar.exists():
                    return json.loads(sidecar.read_text())
    asm = getattr(kernel, "asm", None)
    if isinstance(asm, dict):
        for v in asm.values():
            if isinstance(v, (str, os.PathLike)):
                p = Path(v)
                if p.exists():
                    sidecar = p.with_suffix(".json")
                    if sidecar.exists():
                        return json.loads(sidecar.read_text())
    return None


def _kernel_matches_best_cfg(key, kernel, best_cfg, arg_names, runtime_kwargs):
    """Robustly verify if a cache entry matches the best config AND runtime arguments by parsing the cache key string."""
    meta = kernel.metadata

    # 1. Quick validation on metadata (if it exists)
    if getattr(meta, "num_warps", None) is not None and getattr(meta, "num_warps", None) != best_cfg.num_warps:
        return False
    if getattr(meta, "num_stages", None) is not None and getattr(meta, "num_stages", None) != best_cfg.num_stages:
        return False

    # 2. Parse the cache key dict string to isolate exact kwargs (Block M, N, plus runtime constexprs)
    arg_list, meta_dict = None, None
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], dict):
        arg_list, meta_dict = key
    else:
        key_str = str(key)
        dict_start = key_str.rfind('{')
        if dict_start != -1:
            try:
                list_str = key_str[:dict_start]
                dict_str = key_str[dict_start:]
                arg_list = ast.literal_eval(list_str)
                meta_dict = ast.literal_eval(dict_str)
            except Exception:
                pass

    if arg_list is not None and meta_dict is not None and arg_names:
        try:
            # Check warps/stages from meta dict.
            if meta_dict.get('num_warps') != best_cfg.num_warps:
                return False
            if meta_dict.get('num_stages') != best_cfg.num_stages:
                return False

            # Merge autotune kwargs and explicit runtime constexpr kwargs to
            # verify the exact specialization. The generated C++ descriptors
            # below are only safe if the cubin bytes and launch metadata come
            # from this same exact CompiledKernel.
            expected_kwargs = {**best_cfg.kwargs, **runtime_kwargs}
            seen_expected = set()
            for i, arg_name in enumerate(arg_names):
                if arg_name in expected_kwargs:
                    expected_val = expected_kwargs[arg_name]
                    if i >= len(arg_list):
                        return False
                    val_tuple = arg_list[i]
                    # Expecting format ('constexpr', value). If the key shape
                    # is different, do not claim an exact match; fall through to
                    # sidecar/unknown handling below.
                    if not (isinstance(val_tuple, (tuple, list)) and len(val_tuple) >= 2 and val_tuple[0] == 'constexpr'):
                        return None
                    if val_tuple[1] != expected_val:
                        return False
                    seen_expected.add(arg_name)
            if set(expected_kwargs) - seen_expected:
                return False
            return True
        except Exception:
            pass

    # 3. Fallback: check sidecar JSON if key parsing failed
    sidecar = _load_kernel_sidecar(kernel)
    if sidecar is not None:
        if sidecar.get("num_warps") != best_cfg.num_warps:
            return False
        if sidecar.get("num_stages") != best_cfg.num_stages:
            return False
        if sidecar.get("maxnreg") != best_cfg.maxnreg:
            return False

    return None


def _select_kernel_from_cache(kernel_cache, best_cfg, arg_names, runtime_kwargs):
    exact = []
    unknown = []

    for key, kernel in kernel_cache.items():
        if kernel is None:
            continue

        match = _kernel_matches_best_cfg(key, kernel, best_cfg, arg_names, runtime_kwargs)
        if match is True:
            exact.append((key, kernel))
        elif match is None:
            meta = kernel.metadata
            if getattr(meta, "num_warps", None) == best_cfg.num_warps and getattr(meta, "num_stages", None) == best_cfg.num_stages:
                unknown.append((key, kernel))

    if len(exact) == 1:
        return exact[0][1]
    if len(exact) > 1:
        raise RuntimeError(
            f"Multiple exact cache matches for best config {best_cfg} and runtime kwargs {runtime_kwargs}. "
            f"Need stronger discriminator. Matches={len(exact)}"
        )

    if len(unknown) == 1:
        return unknown[0][1]
    if len(unknown) > 1:
        raise RuntimeError(
            f"Multiple ambiguous cache matches for best config {best_cfg} using only metadata. "
            f"Matches={len(unknown)}. Add extra dumps and inspect cache layout."
        )

    raise RuntimeError(
        f"Could not find CompiledKernel matching best config {best_cfg}. "
        f"Cache has {len(kernel_cache)} entries."
    )


def _dump_cache_debug(kernel_cache, best_cfg):
    print("[debug] best config:", best_cfg, flush=True)
    for i, (key, kernel) in enumerate(kernel_cache.items()):
        if kernel is None:
            continue
        meta = kernel.metadata
        sidecar = _load_kernel_sidecar(kernel)
        print(f"[debug] cache[{i}] key={key!r}", flush=True)
        print(
            f"        meta: warps={getattr(meta, 'num_warps', None)} "
            f"stages={getattr(meta, 'num_stages', None)} shared={getattr(meta, 'shared', None)}",
            flush=True,
        )
        if sidecar is not None:
            print(
                f"        sidecar: name={sidecar.get('name')} warps={sidecar.get('num_warps')} "
                f"stages={sidecar.get('num_stages')} maxnreg={sidecar.get('maxnreg')} "
                f"shared={sidecar.get('shared')} hash={sidecar.get('hash')[:16]}...",
                flush=True,
            )
        else:
            print("        sidecar: <missing>", flush=True)


def _get_compiled_kernel(autotuned_fn, device=0, debug_dump=False, **runtime_kwargs):
    """Retrieve kernel, passing runtime constexpr kwargs to disambiguate autotuned parameters."""
    base_fn = _find_jit_holder(autotuned_fn)
    if base_fn is None:
        _debug_wrapper_chain(autotuned_fn)
        raise RuntimeError(
            f"Could not find Triton JIT cache holder in wrapper chain starting from "
            f"{type(autotuned_fn)!r}"
        )

    best_cfg = autotuned_fn.best_config
    kernel_cache = _safe_cache_entry(base_fn.device_caches, device)
    arg_names = getattr(base_fn, "arg_names", None)

    if debug_dump:
        _dump_cache_debug(kernel_cache, best_cfg)

    kernel = _select_kernel_from_cache(kernel_cache, best_cfg, arg_names, runtime_kwargs)
    return kernel, best_cfg


def extract_k1(arch, debug_dump=False):
    import torch
    results = {}
    for dtype_suffix, input_fp16, _ in DTYPE_CONFIGS:
        print(f"\n  K1 G64 {dtype_suffix} (autotuning)...", flush=True)
        dtype = torch.float16 if input_fp16 else torch.float32
        M, K = 1024, 2560
        x = torch.randn((M, K), device="cuda", dtype=dtype)
        xq = torch.empty((M, K), device="cuda", dtype=torch.int8)
        xs = torch.empty((M, K // 64), device="cuda", dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K // BLOCK_K)

        kernel1_convrot_quant[grid](
            x, xq, xs, M, K, K, 1, K, 1, K // 64, 1,
            BLOCK_K=BLOCK_K, GROUP_SIZE=GROUP_SIZE, INPUT_FP16=input_fp16,
        )
        torch.cuda.synchronize()

        # Pass runtime constexpr values for accurate cache discrimination
        kernel, best_cfg = _get_compiled_kernel(
            kernel1_convrot_quant,
            debug_dump=debug_dump,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP_SIZE,
            INPUT_FP16=input_fp16
        )

        block_m = best_cfg.kwargs["BLOCK_M"]
        nw = best_cfg.num_warps
        ns = best_cfg.num_stages
        cubin = kernel.asm["cubin"]
        shared = kernel.metadata.shared
        abi = _extract_kernel_abi(kernel, "quant")
        print(f"  -> Autotune winner: BLOCK_M={block_m}, num_warps={nw}, num_stages={ns}", flush=True)
        print(
            f"     shared={shared/1024:.0f}KB, cubin={len(cubin)}B, "
            f"runtime_args={len(abi['runtime_signature'])}, scratch_ptrs={abi['scratch_ptr_count']}, "
            f"reqntid={abi['block']}",
            flush=True,
        )
        results[dtype_suffix] = {
            "cubin": cubin,
            "shared": shared,
            "block_m": int(block_m),
            "block_k": BLOCK_K,
            "num_warps": nw,
            "num_stages": ns,
            "maxnreg": getattr(best_cfg, "maxnreg", None),
            "abi": abi,
        }
    return results


def extract_k2(arch, debug_dump=False):
    import torch
    results = {}
    for bias_suffix, has_bias in BIAS_CONFIGS:
        for dtype_suffix, _, output_fp16 in DTYPE_CONFIGS:
            config_name = f"{bias_suffix}_{dtype_suffix}"
            print(f"\n  K2 G64 {config_name} (autotuning)...", flush=True)
            out_dtype = torch.float16 if output_fp16 else torch.float32
            M, K, N = AUTOTUNE_SHAPES_K2[0]
            xq = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
            xs = torch.rand((M, K // 64), device="cuda", dtype=torch.float32) * 0.02 + 0.001
            wq = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
            ws = torch.rand((N,), device="cuda", dtype=torch.float32) * 0.02 + 0.001
            bias = torch.randn((N,), device="cuda", dtype=torch.float32) if has_bias else torch.empty((1,), device="cuda", dtype=torch.float32)
            y = torch.empty((M, N), device="cuda", dtype=out_dtype)
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

            kernel2_gemm_dequant[grid](
                xq, xs, wq, ws, bias, y,
                M, N, K, K, 1, K // 64, 1, K, 1, N, 1,
                BLOCK_K=BLOCK_K, HAS_BIAS=has_bias, OUTPUT_FP16=output_fp16,
            )
            torch.cuda.synchronize()

            # Pass runtime constexpr values for accurate cache discrimination
            kernel, best_cfg = _get_compiled_kernel(
                kernel2_gemm_dequant,
                debug_dump=debug_dump,
                BLOCK_K=BLOCK_K,
                HAS_BIAS=has_bias,
                OUTPUT_FP16=output_fp16
            )

            block_m = best_cfg.kwargs["BLOCK_M"]
            block_n = best_cfg.kwargs["BLOCK_N"]
            group_m = best_cfg.kwargs["GROUP_M"]
            nw = best_cfg.num_warps
            ns = best_cfg.num_stages
            mr = getattr(best_cfg, "maxnreg", None)
            cubin = kernel.asm["cubin"]
            shared = kernel.metadata.shared
            abi = _extract_kernel_abi(kernel, "gemm")
            print(f"  -> Autotune winner: BM={block_m} BN={block_n} GM={group_m} W={nw} S={ns} MR={mr}", flush=True)
            print(
                f"     shared={shared/1024:.0f}KB, cubin={len(cubin)}B, "
                f"runtime_args={len(abi['runtime_signature'])}, scratch_ptrs={abi['scratch_ptr_count']}, "
                f"reqntid={abi['block']}",
                flush=True,
            )
            results[config_name] = {
                "cubin": cubin,
                "shared": shared,
                "block_m": int(block_m),
                "block_n": int(block_n),
                "block_k": BLOCK_K,
                "group_m": int(group_m),
                "num_warps": nw,
                "num_stages": ns,
                "maxnreg": mr,
                "abi": abi,
            }
    return results


def _format_cubin_as_c_array(cubin_bytes: bytes) -> list[str]:
    tokens = [f"0x{b:02x}" for b in cubin_bytes]
    lines: list[str] = []
    line = "    "
    for token in tokens:
        if len(line) + len(token) + 2 > 120:
            lines.append(line.rstrip())
            line = "    "
        line += token + ", "
    lines.append(line.rstrip(", "))
    return lines


def write_header(k1_results, k2_results, output_path):
    """Emit cubins plus an authoritative C++ launch descriptor/stub layer.

    The generated header records the exact runtime launch ABI for each cubin by
    reading CompiledKernel.src.signature and validating it against the PTX
    `.entry` prototype. This keeps the embedded cubin bytes, launch geometry,
    dynamic shared-memory size, and host-side parameter packing in lock-step.
    """
    max_scratch_ptrs = max(
        [k1_results[d]["abi"]["scratch_ptr_count"] for d, _, _ in DTYPE_CONFIGS]
        + [k2_results[f"{b}_{d}"]["abi"]["scratch_ptr_count"] for b, _ in BIAS_CONFIGS for d, _, _ in DTYPE_CONFIGS],
        default=0,
    )

    header_lines = [
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <cstring>",
        "#include <cuda.h>",
        "",
        "// AUTO-GENERATED by tools/onnx-export/extract_jit_cubins_autotune.py.",
        "// Cubins extracted from Triton's JIT cache after @triton.autotune.",
        "// NO triton.compile() / AOT path was used — these are the exact cubins",
        "// that Triton's JIT produced and autotune benchmarked as fastest.",
        "// Runtime parameter packing below is generated from CompiledKernel.src.signature",
        "// and cross-checked against the PTX .entry prototype for the same cubin.",
        f"#define CONVROT_INT8_CUBIN_HEADER_VERSION {CONVROT_INT8_CUBIN_HEADER_VERSION}",
        "#define CONVROT_INT8_HAS_GENERATED_LAUNCH_STUBS 1",
        f"#define CONVROT_INT8_TRITON_ABI_TRAILING_SCRATCH_PTRS {max_scratch_ptrs}",
        "",
    ]
    for dtype_suffix, _, _ in DTYPE_CONFIGS:
        header_lines.append(f"#define CONVROT_INT8_HAS_DTYPE_{dtype_suffix} 1")
    header_lines.append("#define CONVROT_INT8_HAS_M1_SPECIALIZED 0")
    header_lines.append("")

    total = 0
    desc_entries = []
    runtime_arg_arrays = []

    def maxnreg_value(r):
        explicit = r.get("maxnreg", None)
        if explicit is not None:
            return int(explicit)
        inferred = r.get("abi", {}).get("ptx_maxnreg", None)
        return -1 if inferred is None else int(inferred)

    def emit_runtime_arg_array(symbol: str, runtime_signature):
        runtime_arg_arrays.append(f"inline constexpr ConvRotRuntimeArg {symbol}[] = {{")
        for arg in runtime_signature:
            runtime_arg_arrays.append(
                f'    {{ConvRotLaunchArgId::{arg["enum"]}, \"{arg["name"]}\", \"{arg["sig_type"]}\"}},'
            )
        runtime_arg_arrays.append("};")
        runtime_arg_arrays.append("")

    def emit_common_constants(prefix, cn, r):
        block_x, block_y, block_z = r["abi"]["block"]
        header_lines.append(f"constexpr size_t {prefix}_{cn}_SIZE = {len(r['cubin'])};")
        header_lines.append(f"constexpr size_t {prefix}_{cn}_SHARED = {r['shared']};")
        header_lines.append(f"constexpr int {prefix}_{cn}_BLOCK_M = {r['block_m']};")
        header_lines.append(f"constexpr int {prefix}_{cn}_BLOCK_THREADS = {block_x * block_y * block_z};")
        header_lines.append(f"constexpr int {prefix}_{cn}_NUM_WARPS = {r['num_warps']};")
        header_lines.append(f"constexpr int {prefix}_{cn}_NUM_STAGES = {int(r.get('num_stages', -1))};")
        header_lines.append(f"constexpr int {prefix}_{cn}_MAXNREG = {maxnreg_value(r)};")
        header_lines.append(f"constexpr int {prefix}_{cn}_RUNTIME_PARAM_COUNT = {len(r['abi']['runtime_signature'])};")
        header_lines.append(f"constexpr int {prefix}_{cn}_SCRATCH_PTR_COUNT = {int(r['abi']['scratch_ptr_count'])};")

    for dtype_suffix, _, _ in DTYPE_CONFIGS:
        r = k1_results[dtype_suffix]
        cn = f"K1_{TILE_NAME}_G{GROUP_SIZE}_{dtype_suffix}"
        prefix = "kCONVROT_INT8_KERNEL1"
        emit_common_constants(prefix, cn, r)
        header_lines.append(f"constexpr int {prefix}_{cn}_BLOCK_K = {r['block_k']};")
        header_lines.append(f"alignas(16) constexpr unsigned char {prefix}_{cn}[] = {{")
        header_lines.extend(_format_cubin_as_c_array(r['cubin']))
        header_lines += ["};", ""]
        arg_symbol = f"kCONVROT_INT8_RUNTIME_ARGS_{cn}"
        emit_runtime_arg_array(arg_symbol, r["abi"]["runtime_signature"])
        input_dtype_id = 10 if dtype_suffix == "FP16IO" else 1
        output_dtype_id = input_dtype_id
        block_x, block_y, block_z = r["abi"]["block"]
        desc_entries.append({
            "name": f"CONVROT_{cn}",
            "array": f"{prefix}_{cn}",
            "size": f"{prefix}_{cn}_SIZE",
            "stage": "ConvRotKernelStage::kQuant",
            "function": r["abi"]["function_name"],
            "group_size": GROUP_SIZE,
            "has_bias": "false",
            "input_dtype": input_dtype_id,
            "output_dtype": output_dtype_id,
            "block_m": int(r['block_m']),
            "block_n": 1,
            "block_k": int(r['block_k']),
            "group_m": 1,
            "block_x": block_x,
            "block_y": block_y,
            "block_z": block_z,
            "shared": int(r['shared']),
            "num_warps": int(r['num_warps']),
            "num_stages": int(r.get('num_stages', -1)),
            "maxnreg": maxnreg_value(r),
            "runtime_args": arg_symbol,
            "runtime_arg_count": len(r["abi"]["runtime_signature"]),
            "scratch_ptr_count": int(r["abi"]["scratch_ptr_count"]),
        })
        total += 1

    for bias_suffix, has_bias in BIAS_CONFIGS:
        for dtype_suffix, _, _ in DTYPE_CONFIGS:
            r = k2_results[f"{bias_suffix}_{dtype_suffix}"]
            cn = f"K2_{TILE_NAME}_G{GROUP_SIZE}_{bias_suffix}_{dtype_suffix}"
            prefix = "kCONVROT_INT8_KERNEL2"
            emit_common_constants(prefix, cn, r)
            header_lines.append(f"constexpr int {prefix}_{cn}_BLOCK_N = {r['block_n']};")
            header_lines.append(f"constexpr int {prefix}_{cn}_BLOCK_K = {r.get('block_k', BLOCK_K)};")
            header_lines.append(f"constexpr int {prefix}_{cn}_GROUP_M = {r.get('group_m', -1)};")
            header_lines.append(f"alignas(16) constexpr unsigned char {prefix}_{cn}[] = {{")
            header_lines.extend(_format_cubin_as_c_array(r['cubin']))
            header_lines += ["};", ""]
            arg_symbol = f"kCONVROT_INT8_RUNTIME_ARGS_{cn}"
            emit_runtime_arg_array(arg_symbol, r["abi"]["runtime_signature"])
            input_dtype_id = 10 if dtype_suffix == "FP16IO" else 1
            output_dtype_id = input_dtype_id
            block_x, block_y, block_z = r["abi"]["block"]
            desc_entries.append({
                "name": f"CONVROT_{cn}",
                "array": f"{prefix}_{cn}",
                "size": f"{prefix}_{cn}_SIZE",
                "stage": "ConvRotKernelStage::kGemm",
                "function": r["abi"]["function_name"],
                "group_size": GROUP_SIZE,
                "has_bias": "true" if has_bias else "false",
                "input_dtype": input_dtype_id,
                "output_dtype": output_dtype_id,
                "block_m": int(r['block_m']),
                "block_n": int(r['block_n']),
                "block_k": int(r.get('block_k', BLOCK_K)),
                "group_m": int(r.get('group_m', -1)),
                "block_x": block_x,
                "block_y": block_y,
                "block_z": block_z,
                "shared": int(r['shared']),
                "num_warps": int(r['num_warps']),
                "num_stages": int(r.get('num_stages', -1)),
                "maxnreg": maxnreg_value(r),
                "runtime_args": arg_symbol,
                "runtime_arg_count": len(r["abi"]["runtime_signature"]),
                "scratch_ptr_count": int(r["abi"]["scratch_ptr_count"]),
            })
            total += 1

    header_lines += [
        "namespace hotstep::convrot_int8_generated {",
        "",
        "enum class ConvRotKernelStage : int32_t { kQuant = 1, kGemm = 2 };",
        "",
        "enum class ConvRotLaunchArgId : uint8_t {",
    ]
    for name in _LAUNCH_ARG_ENUM.values():
        header_lines.append(f"    {name},")
    header_lines += [
        "};",
        "",
        "struct ConvRotRuntimeArg {",
        "    ConvRotLaunchArgId id;",
        "    char const* name;",
        "    char const* sig_type;",
        "};",
        "",
        "struct ConvRotCubinDesc {",
        "    char const* logical_name;",
        "    unsigned char const* data;",
        "    size_t size;",
        "    char const* function_name;",
        "    ConvRotKernelStage stage;",
        "    int32_t group_size;",
        "    bool has_bias;",
        "    int32_t input_dtype_id;",
        "    int32_t output_dtype_id;",
        "    int32_t block_m;",
        "    int32_t block_n;",
        "    int32_t block_k;",
        "    int32_t group_m;",
        "    uint32_t block_x;",
        "    uint32_t block_y;",
        "    uint32_t block_z;",
        "    size_t shared_bytes;",
        "    int32_t num_warps;      // diagnostic only; launch uses block_x/y/z",
        "    int32_t num_stages;     // diagnostic only",
        "    int32_t maxnreg;        // diagnostic only; -1 if uncapped/unknown",
        "    ConvRotRuntimeArg const* runtime_args;",
        "    uint32_t runtime_arg_count;",
        "    uint32_t trailing_scratch_ptr_count;",
        "};",
        "",
        "inline constexpr size_t kConvRotMaxLaunchParams = 24;",
        "",
    ]
    header_lines.extend(runtime_arg_arrays)

    header_lines.append("inline constexpr ConvRotCubinDesc kConvRotCubins[] = {")
    for d in desc_entries:
        header_lines.append(
            "    {" +
            f'"{d["name"]}", {d["array"]}, {d["size"]}, "{d["function"]}", ' +
            f'{d["stage"]}, {d["group_size"]}, {d["has_bias"]}, ' +
            f'{d["input_dtype"]}, {d["output_dtype"]}, ' +
            f'{d["block_m"]}, {d["block_n"]}, {d["block_k"]}, {d["group_m"]}, ' +
            f'{d["block_x"]}u, {d["block_y"]}u, {d["block_z"]}u, ' +
            f'{d["shared"]}, {d["num_warps"]}, {d["num_stages"]}, {d["maxnreg"]}, ' +
            f'{d["runtime_args"]}, {d["runtime_arg_count"]}u, {d["scratch_ptr_count"]}u' +
            "},"
        )
    header_lines += [
        "};",
        "",
        "inline constexpr size_t kConvRotCubinCount = sizeof(kConvRotCubins) / sizeof(kConvRotCubins[0]);",
        "",
        "inline constexpr ConvRotCubinDesc const* findConvRotCubin(",
        "    ConvRotKernelStage stage, int32_t group_size, bool has_bias,",
        "    int32_t input_dtype_id, int32_t output_dtype_id) {",
        "    for (size_t i = 0; i < kConvRotCubinCount; ++i) {",
        "        ConvRotCubinDesc const& d = kConvRotCubins[i];",
        "        if (d.stage == stage && d.group_size == group_size &&",
        "            d.has_bias == has_bias && d.input_dtype_id == input_dtype_id &&",
        "            d.output_dtype_id == output_dtype_id) {",
        "            return &d;",
        "        }",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "inline uint32_t ceilDivU32(int32_t x, int32_t y) {",
        "    return static_cast<uint32_t>((x + y - 1) / y);",
        "}",
        "",
        "inline void* selectQuantLaunchParam(",
        "    ConvRotLaunchArgId id,",
        "    void*& x_param, void*& xq_ptr, void*& xs_ptr,",
        "    int32_t& M, int32_t& K,",
        "    int32_t& stride_xm, int32_t& stride_xk,",
        "    int32_t& stride_xqm, int32_t& stride_xqk,",
        "    int32_t& stride_xsm, int32_t& stride_xsg) {",
        "    switch (id) {",
        "        case ConvRotLaunchArgId::kX_ptr: return &x_param;",
        "        case ConvRotLaunchArgId::kX_q_ptr: return &xq_ptr;",
        "        case ConvRotLaunchArgId::kX_scale_ptr: return &xs_ptr;",
        "        case ConvRotLaunchArgId::kM: return &M;",
        "        case ConvRotLaunchArgId::kK: return &K;",
        "        case ConvRotLaunchArgId::kStride_xm: return &stride_xm;",
        "        case ConvRotLaunchArgId::kStride_xk: return &stride_xk;",
        "        case ConvRotLaunchArgId::kStride_xqm: return &stride_xqm;",
        "        case ConvRotLaunchArgId::kStride_xqk: return &stride_xqk;",
        "        case ConvRotLaunchArgId::kStride_xsm: return &stride_xsm;",
        "        case ConvRotLaunchArgId::kStride_xsg: return &stride_xsg;",
        "        default: return nullptr;",
        "    }",
        "}",
        "",
        "inline void* selectGemmLaunchParam(",
        "    ConvRotLaunchArgId id,",
        "    void*& xq_ptr, void*& xs_ptr, void*& wq_param, void*& ws_param, void*& bias_param, void*& y_ptr,",
        "    int32_t& M, int32_t& N, int32_t& K,",
        "    int32_t& stride_xqm, int32_t& stride_xqk,",
        "    int32_t& stride_xsm, int32_t& stride_xsg,",
        "    int32_t& stride_wn, int32_t& stride_wk,",
        "    int32_t& stride_ym, int32_t& stride_yn) {",
        "    switch (id) {",
        "        case ConvRotLaunchArgId::kX_q_ptr: return &xq_ptr;",
        "        case ConvRotLaunchArgId::kX_scale_ptr: return &xs_ptr;",
        "        case ConvRotLaunchArgId::kW_q_ptr: return &wq_param;",
        "        case ConvRotLaunchArgId::kW_scale_ptr: return &ws_param;",
        "        case ConvRotLaunchArgId::kBias_ptr: return &bias_param;",
        "        case ConvRotLaunchArgId::kY_ptr: return &y_ptr;",
        "        case ConvRotLaunchArgId::kM: return &M;",
        "        case ConvRotLaunchArgId::kN: return &N;",
        "        case ConvRotLaunchArgId::kK: return &K;",
        "        case ConvRotLaunchArgId::kStride_xqm: return &stride_xqm;",
        "        case ConvRotLaunchArgId::kStride_xqk: return &stride_xqk;",
        "        case ConvRotLaunchArgId::kStride_xsm: return &stride_xsm;",
        "        case ConvRotLaunchArgId::kStride_xsg: return &stride_xsg;",
        "        case ConvRotLaunchArgId::kStride_wn: return &stride_wn;",
        "        case ConvRotLaunchArgId::kStride_wk: return &stride_wk;",
        "        case ConvRotLaunchArgId::kStride_ym: return &stride_ym;",
        "        case ConvRotLaunchArgId::kStride_yn: return &stride_yn;",
        "        default: return nullptr;",
        "    }",
        "}",
        "",
        "inline CUresult launchConvRotQuant(",
        "    ConvRotCubinDesc const& d, CUfunction func, CUstream stream,",
        "    void const* x_ptr, void* xq_ptr, void* xs_ptr,",
        "    int32_t M, int32_t K,",
        "    int32_t stride_xm, int32_t stride_xk,",
        "    int32_t stride_xqm, int32_t stride_xqk,",
        "    int32_t stride_xsm, int32_t stride_xsg) {",
        "    void* triton_scratch1 = nullptr;",
        "    void* triton_scratch2 = nullptr;",
        "    if (d.runtime_arg_count + d.trailing_scratch_ptr_count > kConvRotMaxLaunchParams ||",
        "        d.trailing_scratch_ptr_count > 2) {",
        "        return CUDA_ERROR_INVALID_VALUE;",
        "    }",
        "    void* x_param = const_cast<void*>(x_ptr);",
        "    uint32_t const grid_x = ceilDivU32(M, d.block_m);",
        "    uint32_t const grid_y = ceilDivU32(K, d.block_k);",
        "    void* params[kConvRotMaxLaunchParams] = {};",
        "    uint32_t n = 0;",
        "    for (uint32_t i = 0; i < d.runtime_arg_count; ++i) {",
        "        void* slot = selectQuantLaunchParam(",
        "            d.runtime_args[i].id,",
        "            x_param, xq_ptr, xs_ptr,",
        "            M, K,",
        "            stride_xm, stride_xk, stride_xqm, stride_xqk, stride_xsm, stride_xsg);",
        "        if (slot == nullptr) return CUDA_ERROR_INVALID_VALUE;",
        "        params[n++] = slot;",
        "    }",
        "    void* scratch_slots[] = { &triton_scratch1, &triton_scratch2 };",
        "    for (uint32_t i = 0; i < d.trailing_scratch_ptr_count; ++i) params[n++] = scratch_slots[i];",
        "    return cuLaunchKernel(",
        "        func, grid_x, grid_y, 1, d.block_x, d.block_y, d.block_z,",
        "        static_cast<unsigned int>(d.shared_bytes), stream, params, nullptr);",
        "}",
        "",
        "inline CUresult launchConvRotGemm(",
        "    ConvRotCubinDesc const& d, CUfunction func, CUstream stream,",
        "    void* xq_ptr, void* xs_ptr, int8_t const* wq_ptr,",
        "    float const* ws_ptr, void const* bias_ptr, void* y_ptr,",
        "    int32_t M, int32_t N, int32_t K,",
        "    int32_t stride_xqm, int32_t stride_xqk,",
        "    int32_t stride_xsm, int32_t stride_xsg,",
        "    int32_t stride_wn, int32_t stride_wk,",
        "    int32_t stride_ym, int32_t stride_yn) {",
        "    void* triton_scratch1 = nullptr;",
        "    void* triton_scratch2 = nullptr;",
        "    if (d.runtime_arg_count + d.trailing_scratch_ptr_count > kConvRotMaxLaunchParams ||",
        "        d.trailing_scratch_ptr_count > 2) {",
        "        return CUDA_ERROR_INVALID_VALUE;",
        "    }",
        "    void* wq_param = const_cast<int8_t*>(wq_ptr);",
        "    void* ws_param = const_cast<float*>(ws_ptr);",
        "    void* bias_param = const_cast<void*>(bias_ptr);",
        "    uint32_t const grid_m = ceilDivU32(M, d.block_m);",
        "    uint32_t const grid_n = ceilDivU32(N, d.block_n);",
        "    void* params[kConvRotMaxLaunchParams] = {};",
        "    uint32_t n = 0;",
        "    for (uint32_t i = 0; i < d.runtime_arg_count; ++i) {",
        "        void* slot = selectGemmLaunchParam(",
        "            d.runtime_args[i].id,",
        "            xq_ptr, xs_ptr, wq_param, ws_param, bias_param, y_ptr,",
        "            M, N, K,",
        "            stride_xqm, stride_xqk, stride_xsm, stride_xsg,",
        "            stride_wn, stride_wk, stride_ym, stride_yn);",
        "        if (slot == nullptr) return CUDA_ERROR_INVALID_VALUE;",
        "        params[n++] = slot;",
        "    }",
        "    void* scratch_slots[] = { &triton_scratch1, &triton_scratch2 };",
        "    for (uint32_t i = 0; i < d.trailing_scratch_ptr_count; ++i) params[n++] = scratch_slots[i];",
        "    return cuLaunchKernel(",
        "        func, grid_m * grid_n, 1, 1, d.block_x, d.block_y, d.block_z,",
        "        static_cast<unsigned int>(d.shared_bytes), stream, params, nullptr);",
        "}",
        "",
        "} // namespace hotstep::convrot_int8_generated",
        "",
        "// End of generated Triton cubins and launch descriptors.",
    ]
    output_path.write_text("\n".join(header_lines) + "\n")
    print(f"\n[extract_jit_cubins_autotune] Wrote {total} cubins to {output_path}", flush=True)

def main():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Must run on target GPU.", file=sys.stderr)
        return 1

    debug_dump = bool(int(os.environ.get("HOTSTEP_TRITON_CACHE_DEBUG", "0")))

    print(f"[extract_jit_cubins_autotune] Triton version: {triton.__version__}", flush=True)
    print(f"[extract_jit_cubins_autotune] Device: {torch.cuda.get_device_name()}", flush=True)
    props = torch.cuda.get_device_properties(0)
    arch = props.major * 10 + props.minor
    print(f"[extract_jit_cubins_autotune] sm_{arch}, SMs={props.multi_processor_count}, L2={getattr(props, 'l2_cache_size', 0)/(1024*1024):.1f} MB", flush=True)
    print(f"[extract_jit_cubins_autotune] K1 configs: {len(_K1_CONFIGS)}", flush=True)
    print(f"[extract_jit_cubins_autotune] K2 configs: {len(_K2_CONFIGS)}", flush=True)
    if debug_dump:
        print("[extract_jit_cubins_autotune] HOTSTEP_TRITON_CACHE_DEBUG=1 — dumping cache metadata", flush=True)

    dest_dir = Path(__file__).resolve().parent.parent.parent / "engine" / "src" / "plugins" / "assets"
    dest_dir.mkdir(parents=True, exist_ok=True)
    header_path = dest_dir / "convrot_int8_kernel_cubin.h"

    print("\n=== K1 autotune (extracting from JIT cache) ===", flush=True)
    k1_results = extract_k1(arch, debug_dump=debug_dump)

    print("\n=== K2 autotune (extracting from JIT cache) ===", flush=True)
    k2_results = extract_k2(arch, debug_dump=debug_dump)

    print("\n=== Writing header ===", flush=True)
    write_header(k1_results, k2_results, header_path)

    print("\n[extract_jit_cubins_autotune] DONE!", flush=True)
    print("  The header is drop-in compatible with the C++ plugin.", flush=True)
    print("  Rebuild the plugin and TRT engine to use the autotune-selected cubins.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
