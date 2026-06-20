#!/usr/bin/env python3
"""Shared TensorRT 11 helpers for HOT-Step ONNX tooling."""

from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Pattern


TRT_VERSION_REQUIRED_MAJOR = int(os.environ.get("HOT_STEP_TRT_MAJOR", "11"))
FSQ_SIDECAR_NAME = "fsq.safetensors"
FSQ_TENSOR_PREFIXES = ("tokenizer.", "detokenizer.")

DIT_DEFAULTS = {
    "family": "acestep-v15",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "n_layers": 32,
    "n_heads": 16,
    "n_kv_heads": 8,
    "head_dim": 128,
    "in_channels": 192,
    "out_channels": 64,
    "patch_size": 2,
    "encoder_hidden_size": 2048,
}

TRT_PROFILES = {
    # "default" — 2-minute test profile for 6GB cards.
    # max_T=3000 = 120s × 25Hz. The DiT internally patchifies T→T/2 (12.5Hz),
    # so max internal sequence length = 1500 tokens. Combined with the 4GB
    # w8a8 engine this leaves ~1GB for activations on a 6GB card.
    # For longer songs use the "full-10min" profile (requires ≥8GB VRAM).
    "default": {
        "min_T": 64,
        "opt_T": 2048,
        "max_T": 3000,
        "min_enc_S": 64,
        "opt_enc_S": 512,
        "max_enc_S": 2048,
        "min_N": 1,
        "opt_N": 1,
        "max_N": 1,
    },
    # "full-10min" — 10-minute profile for ≥8GB cards.
    # max_T=15000 = 600s × 25Hz (the silence_latent max). Internal sequence
    # length after patchify = 7500 tokens at 12.5Hz. This is the official
    # ACE-Step v1.5 max song length.
    "full-10min": {
        "min_T": 64,
        "opt_T": 2048,
        "max_T": 15000,
        "min_enc_S": 64,
        "opt_enc_S": 512,
        "max_enc_S": 2048,
        "min_N": 1,
        "opt_N": 1,
        "max_N": 1,
    },
    "pj-ode-320": {
        "min_T": 64,
        "opt_T": 2250,
        "max_T": 2250,
        "min_enc_S": 32,
        "opt_enc_S": 384,
        "max_enc_S": 1024,
        "min_N": 1,
        "opt_N": 2,
        "max_N": 2,
    },
}

# Q8_0-equivalent DiT decoder matrix allowlist.
#
# Accepted prefixes reflect the different names produced by safetensors,
# wrapper.named_parameters(), and TensorRT refit manifests:
#   decoder.layers...
#   dit.layers...
#   layers...
# Biases, norms, modulation/AdaLN, silence/null sidecars, FSQ/tokenizer,
# condition encoder, text/LM, VAE, and convolution weights are excluded.
TRT_FP16_WEIGHT_ALLOWLIST_PATTERNS = [
    r"^(?:decoder\.|dit\.)?time_embed(?:_r)?\.linear_[12]\.weight$",
    r"^(?:decoder\.|dit\.)?time_embed(?:_r)?\.time_proj\.weight$",
    r"^(?:decoder\.|dit\.)?condition_embedder\.weight$",
    r"^(?:decoder\.|dit\.)?layers\.[0-9]+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$",
    r"^(?:decoder\.|dit\.)?layers\.[0-9]+\.cross_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$",
    r"^(?:decoder\.|dit\.)?layers\.[0-9]+\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$",
]

TRT_FP16_WEIGHT_ALLOWLIST = [re.compile(p) for p in TRT_FP16_WEIGHT_ALLOWLIST_PATTERNS]


def fsq_required_tensor_names() -> list[str]:
    names = [
        "tokenizer.audio_acoustic_proj.weight",
        "tokenizer.audio_acoustic_proj.bias",
        "tokenizer.attention_pooler.embed_tokens.weight",
        "tokenizer.attention_pooler.embed_tokens.bias",
        "tokenizer.attention_pooler.special_token",
        "tokenizer.attention_pooler.norm.weight",
        "tokenizer.quantizer.project_in.weight",
        "tokenizer.quantizer.project_in.bias",
        "tokenizer.quantizer.project_out.weight",
        "tokenizer.quantizer.project_out.bias",
        "detokenizer.embed_tokens.weight",
        "detokenizer.embed_tokens.bias",
        "detokenizer.special_tokens",
        "detokenizer.norm.weight",
        "detokenizer.proj_out.weight",
        "detokenizer.proj_out.bias",
    ]
    for prefix in ("tokenizer.attention_pooler", "detokenizer"):
        for layer in range(2):
            p = f"{prefix}.layers.{layer}"
            names.extend(
                [
                    f"{p}.input_layernorm.weight",
                    f"{p}.post_attention_layernorm.weight",
                    f"{p}.self_attn.q_proj.weight",
                    f"{p}.self_attn.k_proj.weight",
                    f"{p}.self_attn.v_proj.weight",
                    f"{p}.self_attn.o_proj.weight",
                    f"{p}.self_attn.q_norm.weight",
                    f"{p}.self_attn.k_norm.weight",
                    f"{p}.mlp.gate_proj.weight",
                    f"{p}.mlp.up_proj.weight",
                    f"{p}.mlp.down_proj.weight",
                ]
            )
    return names


@dataclass(frozen=True)
class ExportMetadata:
    family: str
    source_model: str
    profile: str
    precision_policy: str
    tensor_names_fp16: list[str]
    tensor_names_fp32: list[str]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def portable_path(path: Path | str, base_dir: Path | str) -> str:
    import os

    path_obj = Path(path)
    base_obj = Path(base_dir)
    try:
        return str(path_obj.resolve().relative_to(base_obj.resolve()))
    except ValueError:
        try:
            return os.path.relpath(path_obj.resolve(), base_obj.resolve())
        except ValueError:
            try:
                return str(path_obj.resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                return path_obj.name if path_obj.is_absolute() else str(path_obj)


def hex64(value: int) -> str:
    return f"{value & 0xffffffffffffffff:016x}"


def fnv1a_bytes(data: bytes, seed: int = 1469598103934665603) -> int:
    h = seed
    for byte in data:
        h ^= byte
        h = (h * 1099511628211) & 0xffffffffffffffff
    return h


def fnv1a_text(text: str) -> int:
    return fnv1a_bytes(text.encode("utf-8"))


def fnv1a_file(path: Path) -> int:
    h = 1469598103934665603
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h = fnv1a_bytes(chunk, h)
    return h


def artifact_fingerprint(paths: Iterable[Path | str]) -> str:
    parts: list[str] = []
    for item in paths:
        path = Path(item).resolve()
        file_hash = fnv1a_file(path) if path.is_file() else 0
        parts.append(f"{path}={hex64(file_hash)};")
    return "".join(parts)


def write_export_metadata(path: Path, metadata: ExportMetadata) -> None:
    payload = asdict(metadata)
    if payload.get("source_model"):
        payload["source_model"] = portable_path(payload["source_model"], path.parent)
    write_json(path, payload)


def precision_manifest_path(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(".precision-manifest.json")


def validate_dit_precision_manifest(onnx_path: Path, precision_policy: str | None = None) -> dict:
    manifest_path = precision_manifest_path(onnx_path)
    if not manifest_path.is_file():
        raise SystemExit(f"DiT precision manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Cannot read DiT precision manifest {manifest_path}: {exc}") from None

    got_policy = manifest.get("precision_policy")
    if got_policy not in {"q8map-fp16", "w8a8", "fp32"}:
        raise SystemExit(f"Unsupported DiT precision policy in manifest: {got_policy!r}")
    if precision_policy is not None and got_policy != precision_policy:
        raise SystemExit(
            f"Precision policy mismatch: requested {precision_policy!r}, "
            f"manifest has {got_policy!r}."
        )

    missing = manifest.get("missing_initializers") or []
    if missing:
        raise SystemExit(
            "DiT precision manifest reports allowlisted tensors missing from ONNX initializers: "
            + ", ".join(str(x) for x in missing[:12])
            + (" ..." if len(missing) > 12 else "")
        )

    downcast = manifest.get("downcast_to_fp16") or []
    quantized = manifest.get("quantized_to_int8") or []
    matched = manifest.get("matched_allowlist") or []
    unmatched_patterns = manifest.get("unmatched_allowlist_patterns") or []
    if got_policy == "q8map-fp16":
        if unmatched_patterns:
            raise SystemExit(
                "DiT precision manifest reports allowlist pattern(s) with zero matches: "
                + ", ".join(str(x) for x in unmatched_patterns)
            )
        if not matched:
            raise SystemExit("DiT precision manifest matched zero allowlisted tensors.")
        if not downcast:
            raise SystemExit("DiT precision manifest downcasted zero tensors.")
        if set(downcast) != set(matched) or len(downcast) != len(matched):
            raise SystemExit("DiT precision manifest downcast set does not match allowlist match set.")
        if quantized:
            raise SystemExit("q8map-fp16 precision manifest unexpectedly contains INT8 quantized tensors.")
    elif got_policy == "w8a8":
        # w8a8 + ConvRot: INT8 weights AND INT8 activations, fused by the
        # ConvRotInt8Linear TRT plugin. Same allowlist contract as q8map-fp16
        # (every matched matrix weight is quantized to INT8 with
        # per-output-channel symmetric scale), but the rewrite also emits
        # the ConvRotInt8Linear custom-op node and records convrot metadata
        # (group_size, hadamard_initializer name) so downstream consumers
        # (engine builder, refit tooling) can verify the rotation group size.
        if unmatched_patterns:
            raise SystemExit(
                "DiT precision manifest reports allowlist pattern(s) with zero matches: "
                + ", ".join(str(x) for x in unmatched_patterns)
            )
        if not matched:
            raise SystemExit("DiT precision manifest matched zero allowlisted tensors.")
        if downcast:
            raise SystemExit("W8A8 precision manifest unexpectedly contains FP16-downcast tensors.")
        if not quantized:
            raise SystemExit("W8A8 precision manifest quantized zero tensors.")
        if set(quantized) != set(matched) or len(quantized) != len(matched):
            raise SystemExit("W8A8 precision manifest quantized set does not match allowlist match set.")
        convrot = manifest.get("convrot") or {}
        if not isinstance(convrot, dict):
            raise SystemExit("W8A8 precision manifest is missing the convrot metadata block.")
        if convrot.get("enabled") and not convrot.get("hadamard_initializer"):
            raise SystemExit(
                "W8A8 precision manifest reports ConvRot enabled but is missing the hadamard_initializer name."
            )
        gs = convrot.get("group_size")
        if convrot.get("enabled") and not isinstance(gs, int):
            raise SystemExit("W8A8 precision manifest convrot.group_size must be an integer when enabled.")
    else:
        if downcast:
            raise SystemExit("FP32 precision manifest unexpectedly contains downcast tensors.")
        if quantized:
            raise SystemExit("FP32 precision manifest unexpectedly contains INT8 quantized tensors.")

    return {
        "path": str(manifest_path),
        "precision_policy": got_policy,
        "matched_allowlist_count": len(matched),
        "downcast_to_fp16_count": len(downcast),
        "quantized_to_int8_count": len(quantized),
        "preserved_fp32_count": len(manifest.get("preserved_fp32") or []),
        "all_parameter_count": manifest.get("all_parameter_count"),
        "matrix_parameter_count": manifest.get("matrix_parameter_count"),
        "non_matrix_parameter_count": manifest.get("non_matrix_parameter_count"),
        "allowlist_pattern_match_counts": manifest.get("allowlist_pattern_match_counts") or {},
    }


def _load_json_object(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"JSON file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Cannot read JSON file {path}: {exc}") from None
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root is not an object: {path}")
    return payload


def _metadata_primary_path(alias_path: Path, primary: str) -> Path:
    primary_path = Path(primary)
    if primary_path.is_absolute():
        return primary_path
    relative_path = alias_path.parent / primary_path
    if relative_path.is_file():
        return relative_path
    return primary_path


def _parse_version(version: str) -> tuple[int, int | None]:
    try:
        parts = str(version).split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else None
    except Exception as exc:
        raise SystemExit(f"Cannot parse TensorRT version {version!r}.") from exc
    return major, minor


def _profile_max_shape(profile_shapes: dict, tensor_name: str) -> list[int]:
    if isinstance(profile_shapes.get("max"), dict):
        value = profile_shapes["max"].get(tensor_name)
    else:
        tensor_shapes = profile_shapes.get(tensor_name)
        value = tensor_shapes.get("max") if isinstance(tensor_shapes, dict) else None
    if not isinstance(value, list) or not value or not all(isinstance(x, int) and x > 0 for x in value):
        raise SystemExit(f"Engine metadata profile_shapes is missing valid max bounds for {tensor_name}.")
    return value


def _profile_shape_bounds(profile_shapes: dict) -> dict:
    input_max = _profile_max_shape(profile_shapes, "input_latents")
    enc_max = _profile_max_shape(profile_shapes, "enc_hidden")
    t_max = _profile_max_shape(profile_shapes, "t")
    t_r_max = _profile_max_shape(profile_shapes, "t_r")
    if len(input_max) != 3 or len(enc_max) != 3 or len(t_max) != 1 or len(t_r_max) != 1:
        raise SystemExit("Engine metadata profile_shapes has unexpected DiT rank.")
    if input_max[2] != DIT_DEFAULTS["in_channels"] or enc_max[2] != DIT_DEFAULTS["encoder_hidden_size"]:
        raise SystemExit("Engine metadata profile_shapes has unexpected DiT channel dimensions.")
    if input_max[0] != enc_max[0] or input_max[0] != t_max[0] or input_max[0] != t_r_max[0]:
        raise SystemExit("Engine metadata profile_shapes batch bounds are inconsistent.")
    return {
        "profile_max_batch": input_max[0],
        "profile_max_T": input_max[1],
        "profile_max_enc_S": enc_max[1],
    }


def validate_engine_metadata(
    metadata_path: Path,
    precision_policy: str,
    expected_trt_major: int | None = TRT_VERSION_REQUIRED_MAJOR,
    expected_trt_minor: int | None = None,
) -> dict:
    alias_payload = _load_json_object(metadata_path)
    payload = alias_payload
    source_path = metadata_path
    if "precision_policy" not in payload and payload.get("primary_metadata"):
        source_path = _metadata_primary_path(metadata_path, str(payload["primary_metadata"]))
        payload = _load_json_object(source_path)

    got_policy = payload.get("precision_policy")
    if got_policy != precision_policy:
        raise SystemExit(
            f"Engine precision policy mismatch: expected {precision_policy!r}, "
            f"metadata has {got_policy!r}."
        )
    if payload.get("strongly_typed_network") is not True:
        raise SystemExit("Engine metadata does not describe a strongly typed network.")
    if payload.get("global_fp16_builder_flag") is not False or payload.get("global_bf16_builder_flag") is not False:
        raise SystemExit("Engine metadata indicates global FP16/BF16 builder flags were used.")
    if payload.get("refit_identical") is not True:
        raise SystemExit("Engine metadata is missing the refit-identical guarantee.")
    if payload.get("strip_plan") is not False:
        raise SystemExit("Engine metadata reports a stripped plan; HOT-Step engines must embed weights.")
    weight_streaming = payload.get("weight_streaming", False)
    if not isinstance(weight_streaming, bool):
        raise SystemExit("Engine metadata weight_streaming must be a bool when present.")
    profile_shapes_payload = payload.get("profile_shapes")
    if not isinstance(profile_shapes_payload, dict) or not profile_shapes_payload:
        raise SystemExit("Engine metadata is missing profile_shapes.")
    profile_bounds = _profile_shape_bounds(profile_shapes_payload)

    version = payload.get("tensorrt_version")
    if not isinstance(version, str):
        raise SystemExit("Engine metadata is missing tensorrt_version.")
    major, minor = _parse_version(version)
    if expected_trt_major is not None and major != expected_trt_major:
        raise SystemExit(f"Engine TensorRT major mismatch: expected {expected_trt_major}, metadata has {version}.")
    if expected_trt_minor is not None and minor != expected_trt_minor:
        raise SystemExit(f"Engine TensorRT minor mismatch: expected {expected_trt_minor}, metadata has {version}.")

    precision_summary = payload.get("precision_manifest")
    if isinstance(precision_summary, dict):
        nested_policy = precision_summary.get("precision_policy")
        if nested_policy is not None and nested_policy != precision_policy:
            raise SystemExit("Engine metadata precision_manifest summary does not match precision_policy.")

    return {
        "path": str(metadata_path),
        "source_path": str(source_path),
        "precision_policy": got_policy,
        "profile": payload.get("profile", ""),
        "tensorrt_version": version,
        "tensorrt_major": major,
        "tensorrt_minor": minor,
        "weight_streaming": weight_streaming,
        **profile_bounds,
    }


def is_fp16_weight_name(name: str, allowlist: Iterable[Pattern[str]] = TRT_FP16_WEIGHT_ALLOWLIST) -> bool:
    return any(pattern.match(name) for pattern in allowlist)


def classify_tensor_names(names: Iterable[str]) -> tuple[list[str], list[str]]:
    fp16_names: list[str] = []
    fp32_names: list[str] = []
    for name in sorted(names):
        if is_fp16_weight_name(name):
            fp16_names.append(name)
        else:
            fp32_names.append(name)
    return fp16_names, fp32_names


def check_trt_major(trt) -> None:
    version = getattr(trt, "__version__", "0")
    major_text = str(version).split(".", 1)[0]
    try:
        major = int(major_text)
    except ValueError as exc:
        raise SystemExit(f"Cannot parse TensorRT version {version!r}.") from exc
    if major != TRT_VERSION_REQUIRED_MAJOR:
        raise SystemExit(f"TensorRT {TRT_VERSION_REQUIRED_MAJOR}.x is required, found {version}.")


def profile_shapes(profile_name: str) -> dict:
    if profile_name not in TRT_PROFILES:
        raise ValueError(f"unknown profile {profile_name!r}; choices: {sorted(TRT_PROFILES)}")
    p = TRT_PROFILES[profile_name]

    def shapes(n: int, t: int, enc_s: int) -> dict:
        return {
            "input_latents": (n, t, DIT_DEFAULTS["in_channels"]),
            "enc_hidden": (n, enc_s, DIT_DEFAULTS["encoder_hidden_size"]),
            "t": (n,),
            "t_r": (n,),
            # Self-/cross-attention padding masks (int64). Tied to the same
            # T / S dims as input_latents / enc_hidden. Without these the DiT
            # cross-attends to null_cond_vec padding and dilutes the timbre
            # token — see export_dit.py:DiTForwardWrapper docstring.
            "attention_mask": (n, t),
            "encoder_attention_mask": (n, enc_s),
            "velocity": (n, t, DIT_DEFAULTS["out_channels"]),
        }

    return {
        "min": shapes(p["min_N"], p["min_T"], p["min_enc_S"]),
        "opt": shapes(p["opt_N"], p["opt_T"], p["opt_enc_S"]),
        "max": shapes(p["max_N"], p["max_T"], p["max_enc_S"]),
    }


def default_engine_stem(profile_name: str, precision_policy: str, trt_version: str) -> str:
    p = TRT_PROFILES[profile_name]
    system = platform.system().lower()
    if system == "windows":
        os_tag = "win64"
    elif system == "linux":
        os_tag = "linux-x86_64"
    else:
        os_tag = system or "unknown-os"
    return f"{DIT_DEFAULTS['family']}.{os_tag}.trt{trt_version}.{precision_policy}.T{p['max_T']}"
