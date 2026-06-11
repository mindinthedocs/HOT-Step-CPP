#!/usr/bin/env python3
"""Build a TensorRT 11 engine for a HOT-Step DiT ONNX export."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

from acestep_trt_common import (
    DIT_DEFAULTS,
    TRT_VERSION_REQUIRED_MAJOR,
    TRT_PROFILES,
    check_trt_major,
    default_engine_stem,
    profile_shapes,
    validate_dit_precision_manifest,
    validate_engine_metadata,
    write_json,
)


def import_or_die(module_name: str, package_hint: str | None = None):
    try:
        return __import__(module_name)
    except ImportError as exc:
        hint = package_hint or module_name
        raise SystemExit(f"Missing Python dependency '{module_name}'. Install {hint}.") from None
    except Exception as exc:
        hint = package_hint or module_name
        detail = str(exc).splitlines()[0].strip() or exc.__class__.__name__
        raise SystemExit(f"Cannot import '{module_name}' ({detail}). Ensure {hint} runtime libraries are on PATH.") from None


def trt_version_string(trt) -> str:
    return getattr(trt, "__version__", "unknown")


def network_flags(trt) -> int:
    flags = 0
    strongly_typed = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    if strongly_typed is None:
        raise SystemExit("TensorRT Python API does not expose STRONGLY_TYPED network creation.")
    flags |= 1 << int(strongly_typed)
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit_batch is not None:
        flags |= 1 << int(explicit_batch)
    return flags


def parse_onnx(parser, onnx_path: Path) -> None:
    if parser.parse_from_file(str(onnx_path)):
        return
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    raise SystemExit("TensorRT ONNX parse failed:\n" + "\n".join(errors))


def add_profile(builder, config, network, trt, profile_name: str) -> None:
    shapes = profile_shapes(profile_name)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        if tensor.name not in shapes["opt"]:
            raise SystemExit(f"Input {tensor.name!r} is not covered by profile {profile_name!r}.")
        profile.set_shape(tensor.name, shapes["min"][tensor.name], shapes["opt"][tensor.name], shapes["max"][tensor.name])
    if not profile:
        raise SystemExit("TensorRT optimization profile is invalid.")
    config.add_optimization_profile(profile)


def maybe_set_builder_flag(trt, config, flag_name: str) -> bool:
    flag = getattr(trt.BuilderFlag, flag_name, None)
    if flag is None:
        return False
    config.set_flag(flag)
    return True


def require_builder_flag(trt, config, flag_name: str) -> bool:
    if not maybe_set_builder_flag(trt, config, flag_name):
        raise SystemExit(f"TensorRT Python API does not expose BuilderFlag.{flag_name}.")
    return True


def network_bindings(network) -> list[dict]:
    bindings: list[dict] = []
    for i in range(network.num_inputs):
        t = network.get_input(i)
        bindings.append({"kind": "input", "name": t.name, "shape": tuple(t.shape), "dtype": str(t.dtype)})
    for i in range(network.num_outputs):
        t = network.get_output(i)
        bindings.append({"kind": "output", "name": t.name, "shape": tuple(t.shape), "dtype": str(t.dtype)})
    return bindings


def relative_path(path: Path, base_dir: Path) -> str:
    import os
    try:
        return str(Path(path).resolve().relative_to(base_dir.resolve()))
    except ValueError:
        try:
            return os.path.relpath(Path(path).resolve(), base_dir.resolve())
        except ValueError:
            try:
                return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                return str(path)


def precision_manifest_summary_for_base(summary: dict, onnx_path: Path, base_dir: Path) -> dict:
    portable = dict(summary)
    portable["path"] = relative_path(onnx_path.with_suffix(".precision-manifest.json"), base_dir)
    return portable


class _Spinner:
    """Threaded console spinner so the user knows Python hasn't frozen."""
    def __init__(self, msg: str):
        self._msg = msg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        chars = "|/-\\"
        i = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{self._msg} {chars[i % len(chars)]}")
            sys.stdout.flush()
            time.sleep(0.15)
            i += 1
        sys.stdout.write(f"\r{self._msg} ... done\n")
        sys.stdout.flush()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


def attach_progress_monitor(trt, config) -> None:
    """Attach a TensorRT IProgressMonitor if the Python API exposes it."""
    progress_cls = getattr(trt, "IProgressMonitor", None) or getattr(trt, "IBuilderProgressMonitor", None)
    if progress_cls is None:
        print("[TRT Build] IProgressMonitor not available in this TensorRT Python API; using spinner only.")
        return

    class _Monitor(progress_cls):
        def __init__(self):
            super().__init__()
            self._abort = False
            self._phase_steps = {}

        def phase_start(self, phase_name, parent_phase, num_steps):
            self._phase_steps[phase_name] = num_steps
            print(f"[TRT Build] >>> {phase_name}  ({num_steps} steps)")
            return True

        def phase_finish(self, phase_name):
            print(f"[TRT Build] <<< {phase_name}")

        def step_complete(self, phase_name, step):
            total_steps = self._phase_steps.get(phase_name, 0)
            # Throttle prints to ~5 % increments
            mod = max(1, total_steps // 20) if total_steps > 0 else 1
            if step % mod == 0 or step == total_steps:
                pct = 100.0 * step / total_steps if total_steps else 0
                print(f"[TRT Build] {phase_name}: {step}/{total_steps} ({pct:.0f}%)")
            return True

        def abort_requested(self):
            return self._abort

    try:
        config.progress_monitor = _Monitor()
        print("[TRT Build] Progress monitor attached.")
    except Exception as exc:
        print(f"[TRT Build] Could not attach progress monitor: {exc}")


def write_runtime_aliases(
    args,
    onnx_path: Path,
    engine_path: Path,
    layers_path: Path,
    metadata_path: Path,
    metadata_payload: dict,
) -> tuple[Path, Path] | None:
    if not args.runtime_alias:
        return None

    alias_engine = onnx_path.with_suffix(".engine")
    alias_metadata = onnx_path.with_suffix(".engine.metadata.json")
    alias_base = alias_metadata.parent
    if alias_engine.resolve() != engine_path.resolve():
        shutil.copyfile(engine_path, alias_engine)

    write_json(
        alias_metadata,
        {
            "runtime_engine_alias": relative_path(alias_engine, alias_base),
            "primary_engine": relative_path(engine_path, alias_base),
            "primary_layers": relative_path(layers_path, alias_base),
            "primary_metadata": relative_path(metadata_path, alias_base),
            "onnx": relative_path(onnx_path, alias_base),
            "note": "HOT-Step runtime looks for this alias next to the DiT ONNX.",
            "profile": metadata_payload["profile"],
            "precision_policy": metadata_payload["precision_policy"],
            "precision_manifest": precision_manifest_summary_for_base(
                metadata_payload["precision_manifest"], onnx_path, alias_base),
            "tensorrt_version": metadata_payload["tensorrt_version"],
            "strongly_typed_network": metadata_payload["strongly_typed_network"],
            "global_fp16_builder_flag": metadata_payload["global_fp16_builder_flag"],
            "global_bf16_builder_flag": metadata_payload["global_bf16_builder_flag"],
            "workspace_gb": metadata_payload["workspace_gb"],
            "builder_optimization_level": metadata_payload["builder_optimization_level"],
            "strip_plan": metadata_payload["strip_plan"],
            "refit_identical": metadata_payload["refit_identical"],
            "weight_streaming": metadata_payload["weight_streaming"],
            "profile_shapes": metadata_payload["profile_shapes"],
        },
    )
    return alias_engine, alias_metadata


def build_engine(args) -> tuple[Path, Path, Path, tuple[Path, Path] | None]:
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX file not found: {onnx_path}")
    if args.precision_policy not in {"q8map-fp16", "w8a16", "fp32"}:
        raise SystemExit(f"Unsupported precision policy: {args.precision_policy}")
    precision_manifest_summary = validate_dit_precision_manifest(onnx_path, args.precision_policy)
    if args.strip_plan:
        print("[TRT Build] WARNING: --strip-plan is deprecated and ignored. Engines now embed weights.")
    if not args.refit_identical:
        raise SystemExit("HOT-Step DiT runtime engines require REFIT_IDENTICAL; do not pass --no-refit-identical.")

    trt = import_or_die("tensorrt", "TensorRT 11")
    check_trt_major(trt)

    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags(trt))
    parser = trt.OnnxParser(network, logger)
    parse_onnx(parser, onnx_path)

    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = args.builder_optimization_level
    elif args.builder_optimization_level != 3:
        raise SystemExit("TensorRT Python API does not expose builder_optimization_level.")
    if args.workspace_gb > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1024**3)))
    refit_identical_enabled = require_builder_flag(trt, config, "REFIT_IDENTICAL")
    weight_streaming_enabled = False
    if args.weight_streaming:
        weight_streaming_enabled = require_builder_flag(trt, config, "WEIGHT_STREAMING")

    attach_progress_monitor(trt, config)

    # ── Persistent timing cache ───────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    cache_path = Path(args.timing_cache) if args.timing_cache else out_dir / "trt_timing.cache"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        cache_data = cache_path.read_bytes()
        print(f"[TRT Build] Loading timing cache from {cache_path} ({len(cache_data):,} bytes)")
    else:
        cache_data = b""
        print(f"[TRT Build] No timing cache at {cache_path} — starting fresh.")
    timing_cache = config.create_timing_cache(cache_data)
    # ignore_mismatch=False: silently falls back to a fresh cache if the
    # cached measurements were taken on a different GPU/driver combination.
    config.set_timing_cache(timing_cache, ignore_mismatch=False)
    # ─────────────────────────────────────────────────────────────────────────

    add_profile(builder, config, network, trt, args.profile)

    # The ONNX graph carries the precision policy through typed initializers
    # and Cast nodes. TensorRT 11 removed blanket per-precision builder flags.

    print("[TRT Build] Starting engine build...")
    spinner = _Spinner("[TRT Build] Building engine")
    spinner.start()

    serialized = None
    try:
        serialized = builder.build_serialized_network(network, config)
    finally:
        spinner.stop()

        # ── Persist timing cache (Moved to ensure it saves even on failure/OOM) ──
        try:
            updated_cache = config.get_timing_cache()
            if updated_cache is not None:
                serialized_cache = updated_cache.serialize()
                cache_path.write_bytes(bytes(serialized_cache))
                print(f"[TRT Build] Timing cache saved to {cache_path} ({len(bytes(serialized_cache)):,} bytes)")
        except Exception as exc:
            print(f"[TRT Build] Warning: could not save timing cache: {exc}")
        # ─────────────────────────────────────────────────────────────────────────

    if serialized is None:
        raise SystemExit("TensorRT failed to build a serialized engine.")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.engine_stem or default_engine_stem(args.profile, args.precision_policy, trt_version_string(trt))
    engine_path = out_dir / f"{stem}.engine"
    layers_path = out_dir / f"{stem}.layers.json"
    metadata_path = out_dir / f"{stem}.metadata.json"
    metadata_base = metadata_path.parent

    engine_path.write_bytes(bytes(serialized))

    layer_payload: dict = {
        "bindings": network_bindings(network),
        "precision_manifest": precision_manifest_summary_for_base(precision_manifest_summary, onnx_path, layers_path.parent),
        "layers": None,
    }
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(serialized))
    if engine is not None:
        inspector = engine.create_engine_inspector()
        fmt = getattr(trt.LayerInformationFormat, "JSON", trt.LayerInformationFormat.ONELINE)
        info = inspector.get_engine_information(fmt)
        try:
            layer_payload["layers"] = json.loads(info)
        except Exception:
            layer_payload["layers"] = info
    write_json(layers_path, layer_payload)

    metadata_payload = {
        "onnx": relative_path(onnx_path, metadata_base),
        "engine": relative_path(engine_path, metadata_base),
        "runtime_engine_alias": relative_path(onnx_path.with_suffix(".engine"), metadata_base) if args.runtime_alias else "",
        "profile": args.profile,
        "precision_policy": args.precision_policy,
        "precision_manifest": precision_manifest_summary_for_base(precision_manifest_summary, onnx_path, metadata_base),
        "tensorrt_version": trt_version_string(trt),
        "strongly_typed_network": True,
        "global_fp16_builder_flag": False,
        "global_bf16_builder_flag": False,
        "workspace_gb": args.workspace_gb,
        "builder_optimization_level": args.builder_optimization_level,
        "strip_plan": False,
        "refit_identical": refit_identical_enabled,
        "weight_streaming": weight_streaming_enabled,
        "model_defaults": DIT_DEFAULTS,
        "profile_shapes": profile_shapes(args.profile),
    }
    write_json(metadata_path, metadata_payload)
    validate_engine_metadata(metadata_path, args.precision_policy, TRT_VERSION_REQUIRED_MAJOR)

    runtime_alias = write_runtime_aliases(args, onnx_path, engine_path, layers_path, metadata_path, metadata_payload)
    if runtime_alias:
        validate_engine_metadata(runtime_alias[1], args.precision_policy, TRT_VERSION_REQUIRED_MAJOR)
    return engine_path, layers_path, metadata_path, runtime_alias


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--profile", default="default", choices=sorted(TRT_PROFILES.keys()))
    parser.add_argument("--out-dir", default="engines")
    parser.add_argument("--precision-policy", choices=["q8map-fp16", "w8a16", "fp32"], default="q8map-fp16")
    parser.add_argument("--workspace-gb", type=float, default=1.0)
    parser.add_argument("--builder-optimization-level", type=int, default=0, choices=range(0, 6), metavar="{0..5}")
    parser.add_argument("--strip-plan", action="store_true", default=False,
                        help="(Deprecated, ignored) Engines now always embed weights.")
    parser.add_argument("--no-strip-plan", action="store_false", dest="strip_plan")
    parser.add_argument("--refit-identical", action="store_true", default=True)
    parser.add_argument("--no-refit-identical", action="store_false", dest="refit_identical")
    parser.set_defaults(weight_streaming=True)
    parser.add_argument("--weight-streaming", action="store_true", dest="weight_streaming",
                        help="Build with kWEIGHT_STREAMING to avoid OOM during engine compilation (default). "
                             "At runtime, budget is set to full engine size so all weights are in VRAM.")
    parser.add_argument("--no-weight-streaming", action="store_false", dest="weight_streaming",
                        help="Build without TensorRT weight streaming (may OOM on large models).")
    parser.add_argument("--timing-cache", default="",
                        help="Path to persistent TRT timing cache file. "
                             "Default: <out-dir>/trt_timing.cache. "
                             "Shared across builds on the same GPU/driver.")
    parser.add_argument("--engine-stem", default="")
    parser.add_argument("--runtime-alias", action="store_true", default=True,
                        help="Also write <onnx-stem>.engine for the C++ runtime transparent lookup.")
    parser.add_argument("--no-runtime-alias", action="store_false", dest="runtime_alias")
    args = parser.parse_args()

    engine_path, layers_path, metadata_path, runtime_alias = build_engine(args)
    print(f"engine: {engine_path}")
    print(f"layers: {layers_path}")
    print(f"metadata: {metadata_path}")
    if runtime_alias:
        print(f"runtime_engine_alias: {runtime_alias[0]}")
        print(f"runtime_alias_metadata: {runtime_alias[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
