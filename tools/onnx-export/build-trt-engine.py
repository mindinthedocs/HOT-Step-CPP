#!/usr/bin/env python3
"""Build a TensorRT 11 engine for a HOT-Step DiT ONNX export."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from acestep_trt_common import (
    DIT_DEFAULTS,
    TRT_VERSION_REQUIRED_MAJOR,
    TRT_PROFILES,
    check_trt_major,
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


def _existing_dit_engine_complete(onnx_path: Path, precision_policy: str) -> bool:
    """Return True when a previously-completed DiT build is still on disk.

    The build writes the engine directly next to the ONNX as
    ``<onnx-stem>.engine`` + ``<onnx-stem>.engine.metadata.json``.
    Both must be present and non-empty to skip; a zero-byte engine means
    a previous build was interrupted and would silently produce a corrupt-
    model skip on the next run.
    """
    engine_path = onnx_path.with_suffix(".engine")
    metadata_path = onnx_path.with_suffix(".engine.metadata.json")
    if not (engine_path.is_file() and metadata_path.is_file()):
        return False
    if engine_path.stat().st_size == 0 or metadata_path.stat().st_size == 0:
        return False
    # Sanity-check the metadata's precision_policy matches what we'd build.
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return payload.get("precision_policy") == precision_policy
    except Exception:
        return False


def build_engine(args) -> tuple[Path, Path, Path]:
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX file not found: {onnx_path}")
    if args.precision_policy not in {"q8map-fp16", "w8a8", "fp32"}:
        raise SystemExit(f"Unsupported precision policy: {args.precision_policy}")

    # The engine, metadata, and layers are written directly next to the ONNX
    # file as <onnx-stem>.engine, <onnx-stem>.engine.metadata.json, and
    # <onnx-stem>.layers.json. This is the single source of truth — no
    # engines/ subdirectory, no version-stemmed filename, no runtime alias,
    # no copy/link. The C++ runtime derives the engine path the same way
    # (onnx_path with .onnx → .engine) and reads it directly.
    engine_path   = onnx_path.with_suffix(".engine")
    metadata_path = onnx_path.with_suffix(".engine.metadata.json")
    layers_path   = onnx_path.with_suffix(".layers.json")

    # Resumable build: if a complete engine + metadata already exist on disk,
    # skip the (potentially multi-hour) TRT compilation. --force overrides.
    if not args.force and _existing_dit_engine_complete(onnx_path, args.precision_policy):
        print(f"[TRT Build] Skipping: existing DiT engine found: {engine_path}")
        print(f"[TRT Build]   metadata: {metadata_path}")
        print(f"[TRT Build] Use --force to rebuild.")
        return engine_path, layers_path, metadata_path

    precision_manifest_summary = validate_dit_precision_manifest(onnx_path, args.precision_policy)
    if args.strip_plan:
        print("[TRT Build] WARNING: --strip-plan is deprecated and ignored. Engines now embed weights.")
    if not args.refit_identical:
        raise SystemExit("HOT-Step DiT runtime engines require REFIT_IDENTICAL; do not pass --no-refit-identical.")

    trt = import_or_die("tensorrt", "TensorRT 11")
    check_trt_major(trt)

    # Register HOT-Step custom ops (ConvRotInt8Linear) with TRT's plugin
    # registry before parsing the ONNX graph. w8a8 engines are built from
    # ONNX graphs that use the ``hotstep::ConvRotInt8Linear`` custom op;
    # without this registration the ONNX parser fails at the first
    # custom-op node. q8map-fp16 and fp32 engines don't use the plugin.
    if args.precision_policy == "w8a8":
        try:
            from trt_plugins import register_plugins, is_registered, CONVROT_INT8_LINEAR_OP_NAME
        except ImportError as exc:
            raise SystemExit(
                f"w8a8 engine build requires the trt_plugins package "
                f"(tools/onnx-export/trt_plugins/); import failed: {exc}"
            ) from exc
        if not is_registered():
            ok = register_plugins()
            if not ok:
                raise SystemExit(
                    "trt_plugins.register_plugins() failed — the hotstep_plugins "
                    "C++ shared library (.dll/.so) could not be loaded. "
                    "Build it first (engine/buildcuda.cmd) and ensure it's on "
                    "PATH or in engine/build/ or engine/buildcuda/."
                )
        print(f"[TRT Build] Registered HOT-Step plugin: {CONVROT_INT8_LINEAR_OP_NAME}")

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
    # Default cache location: next to the ONNX file as <onnx-stem>.timing.cache.
    # This keeps all build artifacts co-located with the source ONNX.
    if args.timing_cache:
        cache_path = Path(args.timing_cache)
    else:
        cache_path = onnx_path.with_suffix(".timing.cache")
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

    # Write engine, layers, and metadata directly next to the ONNX file.
    # engine_path / metadata_path / layers_path were already derived at the
    # top of build_engine() from onnx_path.with_suffix(".engine") etc.
    engine_path.parent.mkdir(parents=True, exist_ok=True)
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

    return engine_path, layers_path, metadata_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--profile", default="default", choices=sorted(TRT_PROFILES.keys()))
    parser.add_argument("--precision-policy", choices=["q8map-fp16", "w8a8", "fp32"], default="q8map-fp16")
    parser.add_argument("--workspace-gb", type=float, default=2,
                        help="Workspace size in GB (default: 4.6). w8a8 DiT builds need ~5GB.")
    parser.add_argument("--builder-optimization-level", type=int, default=5, choices=range(0, 6), metavar="{0..5}")
    parser.add_argument("--strip-plan", action="store_true", default=False,
                        help="(Deprecated, ignored) Engines now always embed weights.")
    parser.add_argument("--no-strip-plan", action="store_false", dest="strip_plan")
    parser.add_argument("--refit-identical", action="store_true", default=True)
    parser.add_argument("--no-refit-identical", action="store_false", dest="refit_identical")
    parser.set_defaults(weight_streaming=True)
    parser.add_argument("--weight-streaming", action="store_true", dest="weight_streaming",
                        help="Build with kWEIGHT_STREAMING to avoid OOM during engine compilation (default).")
    parser.add_argument("--no-weight-streaming", action="store_false", dest="weight_streaming",
                        help="Build without TensorRT weight streaming (may OOM on large models).")
    parser.add_argument("--timing-cache", default="",
                        help="Path to persistent TRT timing cache file. "
                             "Default: <onnx-stem>.timing.cache next to the ONNX. "
                             "Shared across builds on the same GPU/driver.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Rebuild even if the engine + metadata already exist on disk (default: skip).")
    args = parser.parse_args()

    engine_path, layers_path, metadata_path = build_engine(args)
    print(f"engine: {engine_path}")
    print(f"layers: {layers_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
