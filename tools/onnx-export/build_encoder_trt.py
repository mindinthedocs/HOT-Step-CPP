#!/usr/bin/env python3
"""Build strongly-typed FP16 TensorRT 11 engines for the HOT-Step encoders.

This is the encoder counterpart to ``build-trt-engine.py`` (which is
DiT-specific: it requires a DiT precision manifest and the
REFIT_IDENTICAL refit lifecycle). The encoders ship embedded FP16 weights, so
this builder reuses the shared TRT-11 infrastructure from
``acestep_trt_common`` (version guard, fingerprinting, JSON/metadata helpers,
portable paths) and adds ONLY the encoder-specific optimization profiles:

    text-enc:  input_ids [1, seq]
    cond-enc:  text_hidden [1, text_seq, 1024]
               lyric_embed [1, lyric_seq, 1024]
               timbre_feats[1, ref,      64]

TRT 11 strongly-typed networks derive precision entirely from the
ONNX graph's tensor dtypes, so a FP16-exported ONNX yields a FP16 engine with
no per-precision builder flags. Embedded-weight engines (no strip+refit) are
correct for the encoders; only the DiT runtime needs the refit lifecycle.

Usage:
    python build_encoder_trt.py --module text-enc --onnx text_encoder.onnx \
        --out-dir .enc-build
    python build_encoder_trt.py --module cond-enc --onnx cond_encoder.onnx \
        --out-dir .enc-build
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from acestep_trt_common import (
    ExportMetadata,
    TRT_VERSION_REQUIRED_MAJOR,
    artifact_fingerprint,
    check_trt_major,
    portable_path,
    write_export_metadata,
    write_json,
)


# Encoder-specific dynamic-shape profiles. These are the ONLY profile knobs the
# encoders need on top of the shared infra; the DiT profiles in
# acestep_trt_common.TRT_PROFILES do not apply here.
ENCODER_PROFILES: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {
    "text-enc": {
        "min": {"input_ids": (1, 1)},
        "opt": {"input_ids": (1, 128)},
        "max": {"input_ids": (1, 512)},
    },
    "cond-enc": {
        "min": {
            "text_hidden": (1, 1, 1024),
            "lyric_embed": (1, 1, 1024),
            "timbre_feats": (1, 1, 64),
        },
        "opt": {
            "text_hidden": (1, 128, 1024),
            "lyric_embed": (1, 256, 1024),
            "timbre_feats": (1, 512, 64),
        },
        "max": {
            # timbre_feats max matches DiT max_T (8192 frames @ 25fps ≈ 327s)
            # so the reference audio can be as long as the song itself —
            # the common case in song covers.
            "text_hidden": (1, 512, 1024),
            "lyric_embed": (1, 1024, 1024),
            "timbre_feats": (1, 8192, 64),
        },
    },
}

ENCODER_FAMILY = "acestep-v15-encoder"


def import_or_die(module_name: str, package_hint: str | None = None):
    try:
        return __import__(module_name)
    except ImportError:
        raise SystemExit(
            f"Missing Python dependency '{module_name}'. Install {package_hint or module_name}."
        ) from None
    except Exception as exc:
        detail = str(exc).splitlines()[0].strip() or exc.__class__.__name__
        raise SystemExit(
            f"Cannot import '{module_name}' ({detail}). "
            f"Ensure {package_hint or module_name} runtime libraries are on PATH."
        ) from None


def trt_version_string(trt) -> str:
    return getattr(trt, "__version__", "unknown")


def network_flags(trt) -> int:
    strongly_typed = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    if strongly_typed is None:
        raise SystemExit("TensorRT Python API does not expose STRONGLY_TYPED network creation.")
    flags = 1 << int(strongly_typed)
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit_batch is not None:
        flags |= 1 << int(explicit_batch)
    return flags


def parse_onnx(parser, onnx_path: Path) -> None:
    if parser.parse_from_file(str(onnx_path)):
        return
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    raise SystemExit("TensorRT ONNX parse failed:\n" + "\n".join(errors))


def add_profile(builder, config, network, module: str) -> None:
    shapes = ENCODER_PROFILES[module]
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        if name not in shapes["opt"]:
            raise SystemExit(f"Input {name!r} is not covered by the {module!r} profile.")
        profile.set_shape(name, shapes["min"][name], shapes["opt"][name], shapes["max"][name])
    if not profile:
        raise SystemExit("TensorRT optimization profile is invalid.")
    config.add_optimization_profile(profile)


def network_bindings(network) -> list[dict]:
    bindings: list[dict] = []
    for i in range(network.num_inputs):
        t = network.get_input(i)
        bindings.append({"kind": "input", "name": t.name, "shape": list(t.shape), "dtype": str(t.dtype)})
    for i in range(network.num_outputs):
        t = network.get_output(i)
        bindings.append({"kind": "output", "name": t.name, "shape": list(t.shape), "dtype": str(t.dtype)})
    return bindings


def collect_weight_names(network) -> list[str]:
    names: list[str] = []
    for i in range(network.num_layers):
        names.append(network.get_layer(i).name)
    return names


def _existing_encoder_engine_complete(out_dir: Path, stem: str) -> tuple[Path, Path, Path] | None:
    """Return (engine_path, metadata_path, layers_path) if a complete encoder
    build is already on disk; else None.

    The encoder builder writes a fixed-name engine + metadata + layers next to
    each other in --out-dir. All three must be present and non-empty to skip:
    a zero-byte engine means a previous build was interrupted and would
    silently produce a corrupt-model skip on the next run.
    """
    engine_path = out_dir / f"{stem}.engine"
    metadata_path = out_dir / f"{stem}.engine.metadata.json"
    layers_path = out_dir / f"{stem}.layers.json"
    if not (engine_path.is_file() and metadata_path.is_file() and layers_path.is_file()):
        return None
    if engine_path.stat().st_size == 0 or metadata_path.stat().st_size == 0:
        return None
    return engine_path, metadata_path, layers_path


def build_engine(args) -> tuple[Path, Path, Path]:
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX file not found: {onnx_path}")
    module = args.module
    if module not in ENCODER_PROFILES:
        raise SystemExit(f"Unknown encoder module {module!r}; choices: {sorted(ENCODER_PROFILES)}")

    out_dir = Path(args.out_dir)
    stem = args.engine_stem or ("text_encoder" if module == "text-enc" else "cond_encoder")

    # Resumable build: skip the (multi-minute) TRT compilation when a complete
    # engine + metadata + layers triple is already on disk. The orchestrator
    # (trt_bundle_manager) checks Step.outputs at its level; this in-script
    # gate makes the script safe to invoke directly without recompiling.
    # --force overrides so a user can rebuild after a corrupt engine.
    if not args.force:
        existing = _existing_encoder_engine_complete(out_dir, stem)
        if existing is not None:
            engine_path, metadata_path, layers_path = existing
            print(f"[build_encoder_trt] Skipping: existing {module} engine found:")
            print(f"[build_encoder_trt]   engine:   {engine_path} ({engine_path.stat().st_size:,} bytes)")
            print(f"[build_encoder_trt]   metadata: {metadata_path}")
            print(f"[build_encoder_trt]   layers:   {layers_path}")
            print(f"[build_encoder_trt] Use --force to rebuild.")
            return engine_path, metadata_path, layers_path

    trt = import_or_die("tensorrt", "TensorRT 11")
    check_trt_major(trt)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags(trt))
    parser = trt.OnnxParser(network, logger)
    parse_onnx(parser, onnx_path)

    config = builder.create_builder_config()
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = args.builder_optimization_level
    if args.workspace_gb > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1024 ** 3)))

    add_profile(builder, config, network, module)

    # Strongly-typed network: precision is carried by the FP16 ONNX graph; there
    # are no per-precision builder flags. Encoders embed their weights (no
    # REFIT_IDENTICAL refit lifecycle — that is DiT-only).
    print(f"[build_encoder_trt] Building strongly-typed FP16 engine for {module}...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT failed to build a serialized engine.")

    out_dir.mkdir(parents=True, exist_ok=True)
    engine_path = out_dir / f"{stem}.engine"
    metadata_path = out_dir / f"{stem}.engine.metadata.json"
    layers_path = out_dir / f"{stem}.layers.json"

    engine_path.write_bytes(bytes(serialized))

    bindings = network_bindings(network)
    fp16_names, fp32_names = _classify_binding_dtypes(bindings)

    profile = ENCODER_PROFILES[module]
    metadata = ExportMetadata(
        family=ENCODER_FAMILY,
        source_model=str(onnx_path.resolve()),
        profile=module,
        precision_policy="strongly-typed-fp16",
        tensor_names_fp16=fp16_names,
        tensor_names_fp32=fp32_names,
    )
    write_export_metadata(metadata_path, metadata)

    # Append the build-shape and provenance facts that the encoder runtime and
    # C++ parity fixtures consume, next to the portable ExportMetadata.
    extra = {
        "module": module,
        "onnx": portable_path(onnx_path, metadata_path.parent),
        "engine": portable_path(engine_path, metadata_path.parent),
        "tensorrt_version": trt_version_string(trt),
        "strongly_typed_network": True,
        "global_fp16_builder_flag": False,
        "global_bf16_builder_flag": False,
        "embedded_weights": True,
        "strip_plan": False,
        "refit_identical": False,
        "workspace_gb": args.workspace_gb,
        "builder_optimization_level": args.builder_optimization_level,
        "engine_bytes": engine_path.stat().st_size,
        "artifact_fingerprint": artifact_fingerprint([onnx_path]),
        "bindings": bindings,
        "profile_shapes": {
            level: {name: list(shape) for name, shape in profile[level].items()}
            for level in ("min", "opt", "max")
        },
    }
    merged = json.loads(metadata_path.read_text(encoding="utf-8"))
    merged.update(extra)
    write_json(metadata_path, merged)

    write_json(layers_path, {"bindings": bindings, "layers": collect_weight_names(network)})

    return engine_path, metadata_path, layers_path


def _classify_binding_dtypes(bindings: list[dict]) -> tuple[list[str], list[str]]:
    fp16: list[str] = []
    fp32: list[str] = []
    for b in bindings:
        target = fp16 if "HALF" in b["dtype"].upper() else fp32
        target.append(b["name"])
    return sorted(fp16), sorted(fp32)


def main() -> int:
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build a strongly-typed FP16 TRT 11 encoder engine")
    parser.add_argument("--module", required=True, choices=sorted(ENCODER_PROFILES.keys()))
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--out-dir", default=".enc-build")
    parser.add_argument("--engine-stem", default="")
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--builder-optimization-level", type=int, default=3,
                        choices=range(0, 6), metavar="{0..5}")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Rebuild even if a complete engine + metadata + "
                             "layers triple already exists on disk (default: skip).")
    args = parser.parse_args()

    engine_path, metadata_path, layers_path = build_engine(args)
    print(f"engine: {engine_path}")
    print(f"metadata: {metadata_path}")
    print(f"layers: {layers_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
