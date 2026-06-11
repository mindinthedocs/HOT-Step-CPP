#!/usr/bin/env python3
"""Prebuild ONNX Runtime TensorRT EP engine caches for HOT-Step side modules.

This builds the TensorRT engines that ONNX Runtime would otherwise create on
the first synthesis run for text_encoder.onnx, cond_encoder.onnx, and
vae_decoder.onnx. DiT engines are still built with build-trt-engine.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
import threading
import time
from pathlib import Path

import numpy as np

from acestep_trt_common import (
    ORT_TRT_DEFAULTS,
    ORT_TRT_ENGINE_ROOT_NAME,
    artifact_fingerprint,
    ort_trt_provider_options,
    portable_path,
    validate_ort_trt_cache,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
_DLL_DIR_HANDLES = []


def add_windows_nvidia_dll_dirs() -> None:
    """Make CUDA/cuDNN DLLs from NVIDIA Python wheels visible to ORT."""
    if os.name != "nt":
        return

    roots: list[Path] = []
    for value in (sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")):
        if value:
            roots.append(Path(value))
    roots.extend(Path(p) for p in sys.path if p)

    seen: set[Path] = set()
    dll_dirs: list[Path] = []
    for root in roots:
        nvidia_root = root / "nvidia"
        for rel in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin"):
            candidate = nvidia_root / rel
            if candidate.is_dir():
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    dll_dirs.append(resolved)

    if not dll_dirs:
        return

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for dll_dir in reversed(dll_dirs):
        text = str(dll_dir)
        if text not in path_entries:
            path_entries.insert(0, text)
    os.environ["PATH"] = os.pathsep.join(path_entries)

    for dll_dir in dll_dirs:
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))
        except (FileNotFoundError, OSError):
            pass


def candidate_native_builders() -> list[Path]:
    names = ["ace-ort-cache.exe"] if os.name == "nt" else ["ace-ort-cache"]
    repo_root = SCRIPT_DIR.parents[1]
    candidates: list[Path] = []
    env_builder = os.environ.get("HOTSTEP_ORT_CACHE_BUILDER")
    if env_builder:
        candidates.append(Path(env_builder))
    for build_root in (repo_root / "engine").glob("build*"):
        for name in names:
            candidates.append(build_root / name)
            candidates.append(build_root / "Release" / name)
            candidates.append(build_root / "RelWithDebInfo" / name)
    return candidates


def find_native_builder() -> Path | None:
    for candidate in candidate_native_builders():
        if candidate.is_file():
            return candidate.resolve()
    return None


def native_runtime_path_env(builder: Path) -> dict[str, str]:
    env = os.environ.copy()
    extra: list[Path] = [builder.parent]
    repo_root = SCRIPT_DIR.parents[1]
    trt10_roots: list[Path] = []
    trt10_env = env.get("HOTSTEP_ORT_TRT10_RUNTIME")
    if trt10_env:
        trt10_roots.append(Path(trt10_env))
    for candidate in (
        repo_root / "artifacts" / "deps" / "ort-trt10-python-runtime",
        repo_root.parent / "hot-step-trt11-work" / "artifacts" / "deps" / "ort-trt10-python-runtime",
        Path.cwd() / "hot-step-trt11-work" / "artifacts" / "deps" / "ort-trt10-python-runtime",
    ):
        trt10_roots.append(candidate)
    for root in trt10_roots:
        if not root.is_dir():
            continue
        for rel in ("tensorrt_libs", "nvidia/cuda_runtime/bin", "nvidia/cufft/bin", "nvidia/nvjitlink/bin"):
            candidate = root / rel
            if candidate.is_dir():
                extra.append(candidate.resolve())
    for candidate in (
        repo_root / "engine" / "deps" / "tensorrt" / "bin",
        repo_root / "engine" / "deps" / "onnxruntime" / "lib",
    ):
        if candidate.is_dir():
            extra.append(candidate.resolve())
    for name in ("TRT_ROOT", "TENSORRT_ROOT"):
        value = env.get(name)
        if value and (Path(value) / "bin").is_dir():
            extra.append((Path(value) / "bin").resolve())
    for name in ("CUDA_PATH", "CUDA_HOME"):
        value = env.get(name)
        if value and (Path(value) / "bin").is_dir():
            extra.append((Path(value) / "bin").resolve())
    if os.name == "nt":
        program_files = env.get("ProgramFiles")
        if program_files:
            cuda_root = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
            for candidate in cuda_root.glob("v*/bin"):
                if candidate.is_dir():
                    extra.append(candidate.resolve())
    add_windows_nvidia_dll_dirs()
    for path_text in os.environ.get("PATH", "").split(os.pathsep):
        if path_text:
            extra.append(Path(path_text))

    seen: set[str] = set()
    path_entries: list[str] = []
    for path in extra:
        text = str(path)
        key = text.lower() if os.name == "nt" else text
        if key not in seen:
            seen.add(key)
            path_entries.append(text)
    env["PATH"] = os.pathsep.join(path_entries)
    return env


def run_native_builder(builder: Path, paths: dict[str, Path], modules: list[str], args) -> None:
    module_arg = ",".join({
        "text-enc": "text",
        "cond-enc": "cond",
        "vae-dec": "vae",
    }[module] for module in modules)
    cmd = [
        str(builder),
        "--modules", module_arg,
        "--out-dir", str(args.out_dir),
        "--device-id", str(args.device_id),
        "--workspace-gb", str(args.workspace_gb),
        "--builder-optimization-level", str(args.builder_optimization_level),
        "--text-opt-tokens", str(args.text_opt_tokens),
        "--text-max-tokens", str(args.text_max_tokens),
        "--lyric-opt-tokens", str(args.lyric_opt_tokens),
        "--lyric-max-tokens", str(args.lyric_max_tokens),
        "--timbre-opt-frames", str(args.timbre_opt_frames),
        "--timbre-max-frames", str(args.timbre_max_frames),
        "--vae-opt-frames", str(args.vae_opt_frames),
        "--vae-max-frames", str(args.vae_max_frames),
    ]
    if "text-enc" in paths:
        cmd.extend(["--text-onnx", str(paths["text-enc"])])
    if "cond-enc" in paths:
        cmd.extend(["--cond-onnx", str(paths["cond-enc"])])
    if "vae-dec" in paths:
        cmd.extend(["--vae-onnx", str(paths["vae-dec"])])
    print(f"[ORT-TRT Build] using native TRT11 cache builder: {builder}")
    subprocess.run(cmd, check=True, env=native_runtime_path_env(builder))


class Spinner:
    def __init__(self, message: str):
        self.message = message
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        def run() -> None:
            frames = "|/-\\"
            i = 0
            while not self.stop_event.wait(0.2):
                sys.stdout.write(f"\r{self.message} {frames[i % len(frames)]}")
                sys.stdout.flush()
                i += 1
            sys.stdout.write(f"\r{self.message} done\n")
            sys.stdout.flush()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join()


def import_or_die():
    add_windows_nvidia_dll_dirs()
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise SystemExit(
            "onnxruntime-gpu is required to prebuild ORT TensorRT engines."
        ) from exc
    return ort


def parse_modules(text: str) -> list[str]:
    aliases = {
        "text": "text-enc",
        "text-enc": "text-enc",
        "text_encoder": "text-enc",
        "cond": "cond-enc",
        "cond-enc": "cond-enc",
        "cond_encoder": "cond-enc",
        "vae": "vae-dec",
        "vae-dec": "vae-dec",
        "vae_decoder": "vae-dec",
    }
    out: list[str] = []
    for item in text.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise SystemExit(f"unknown module {item!r}; use text,cond,vae")
        tag = aliases[key]
        if tag not in out:
            out.append(tag)
    if not out:
        raise SystemExit("no modules selected")
    return out


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    return path


def bundle_file(bundle_dir: Path | None, explicit: str | None, name: str, label: str) -> Path:
    if explicit:
        return require_file(Path(explicit), label)
    if not bundle_dir:
        raise SystemExit(f"{label} requires --bundle-dir or an explicit path")
    return require_file(bundle_dir / name, label)


def sidecar_path(onnx_path: Path, name: str) -> Path:
    return onnx_path.resolve().parent / name


def module_artifact_fingerprint(module_tag: str, onnx_path: Path) -> str:
    if module_tag == "text-enc":
        return artifact_fingerprint([
            onnx_path,
            sidecar_path(onnx_path, "embed_tokens.bin"),
            sidecar_path(onnx_path, "vocab.json"),
            sidecar_path(onnx_path, "merges.txt"),
        ])
    if module_tag == "cond-enc":
        return artifact_fingerprint([
            onnx_path,
            sidecar_path(onnx_path, "null_condition_emb.bin"),
        ])
    if module_tag == "vae-dec":
        return artifact_fingerprint([onnx_path])
    raise ValueError(module_tag)


def profile_kwargs(args) -> dict:
    if args.text_opt_tokens > args.text_max_tokens:
        raise SystemExit("--text-opt-tokens must be <= --text-max-tokens")
    if args.lyric_opt_tokens > args.lyric_max_tokens:
        raise SystemExit("--lyric-opt-tokens must be <= --lyric-max-tokens")
    if args.timbre_opt_frames > args.timbre_max_frames:
        raise SystemExit("--timbre-opt-frames must be <= --timbre-max-frames")
    if args.vae_opt_frames > args.vae_max_frames:
        raise SystemExit("--vae-opt-frames must be <= --vae-max-frames")
    return {
        "text_opt_tokens": args.text_opt_tokens,
        "text_max_tokens": args.text_max_tokens,
        "lyric_opt_tokens": args.lyric_opt_tokens,
        "lyric_max_tokens": args.lyric_max_tokens,
        "timbre_opt_frames": args.timbre_opt_frames,
        "timbre_max_frames": args.timbre_max_frames,
        "vae_opt_frames": args.vae_opt_frames,
        "vae_max_frames": args.vae_max_frames,
    }


def dummy_inputs(module_tag: str, args) -> dict[str, np.ndarray]:
    if module_tag == "text-enc":
        return {
            "input_ids": np.zeros((1, args.text_opt_tokens), dtype=np.int64),
        }
    if module_tag == "cond-enc":
        return {
            "text_hidden": np.zeros((1, args.text_opt_tokens, 1024), dtype=np.float32),
            "lyric_embed": np.zeros((1, args.lyric_opt_tokens, 1024), dtype=np.float32),
            "timbre_feats": np.zeros((1, args.timbre_opt_frames, 64), dtype=np.float32),
        }
    if module_tag == "vae-dec":
        return {
            "latents": np.zeros((1, 64, args.vae_opt_frames), dtype=np.float32),
        }
    raise ValueError(module_tag)


def build_module(ort, module_tag: str, onnx_path: Path, args) -> dict:
    artifact_fp = module_artifact_fingerprint(module_tag, onnx_path)
    options, cache_dir = ort_trt_provider_options(
        onnx_path=onnx_path,
        module_tag=module_tag,
        artifact_fp=artifact_fp,
        out_dir=args.out_dir,
        device_id=args.device_id,
        fp16=(module_tag == "vae-dec"),
        workspace_gb=args.workspace_gb,
        builder_optimization_level=args.builder_optimization_level,
        profile_kwargs=profile_kwargs(args),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ORT-TRT Build] module={module_tag}")
    print(f"[ORT-TRT Build] onnx={onnx_path}")
    print(f"[ORT-TRT Build] cache={cache_dir}")
    print(f"[ORT-TRT Build] profile_max={options['trt_profile_max_shapes']}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [
        ("TensorrtExecutionProvider", options),
        ("CUDAExecutionProvider", {"device_id": args.device_id}),
        "CPUExecutionProvider",
    ]

    t0 = time.time()
    spinner = Spinner(f"[ORT-TRT Build] Creating {module_tag} session")
    spinner.start()
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        inputs = dummy_inputs(module_tag, args)
        sess.run(None, inputs)
    finally:
        spinner.stop()
    elapsed = time.time() - t0

    active = sess.get_providers()
    if "TensorrtExecutionProvider" not in active:
        raise SystemExit(f"{module_tag}: TensorRTExecutionProvider was not active; providers={active}")

    cache_summary = validate_ort_trt_cache(cache_dir)
    payload = {
        "artifact": "hotstep-ort-trt-cache",
        "module": module_tag,
        "onnx": portable_path(onnx_path, cache_dir),
        "cache_dir": str(cache_dir.name),
        "artifact_fingerprint": artifact_fp,
        "provider_options": {
            k: (portable_path(v, cache_dir) if k.endswith("_path") else v)
            for k, v in options.items()
        },
        "provider_order": active,
        "build_seconds": elapsed,
        **cache_summary,
    }
    write_json(cache_dir / "hotstep-ort-trt-cache.json", payload)
    print(
        "[ORT-TRT Build] cache OK: "
        f"engines={cache_summary['engine_count']} "
        f"profiles={cache_summary['profile_count']} "
        f"bytes={cache_summary['engine_bytes']:,}"
    )
    return payload


def write_env_file(out_dir: Path, args) -> None:
    lines = [
        "# Use this before launching the HOT-Step runtime when --out-dir is not the default bundle folder.",
        f"$env:HOTSTEP_ORT_TRT_ENGINE_ROOT = \"{out_dir}\"",
        "$env:HOTSTEP_ORT_TRT_PREBUILT_ONLY = \"1\"",
        f"$env:HOTSTEP_ORT_TRT_TEXT_OPT_TOKENS = \"{args.text_opt_tokens}\"",
        f"$env:HOTSTEP_ORT_TRT_TEXT_MAX_TOKENS = \"{args.text_max_tokens}\"",
        f"$env:HOTSTEP_ORT_TRT_LYRIC_OPT_TOKENS = \"{args.lyric_opt_tokens}\"",
        f"$env:HOTSTEP_ORT_TRT_LYRIC_MAX_TOKENS = \"{args.lyric_max_tokens}\"",
        f"$env:HOTSTEP_ORT_TRT_TIMBRE_OPT_FRAMES = \"{args.timbre_opt_frames}\"",
        f"$env:HOTSTEP_ORT_TRT_TIMBRE_MAX_FRAMES = \"{args.timbre_max_frames}\"",
        f"$env:HOTSTEP_ORT_TRT_VAE_OPT_FRAMES = \"{args.vae_opt_frames}\"",
        f"$env:HOTSTEP_ORT_TRT_VAE_MAX_FRAMES = \"{args.vae_max_frames}\"",
        f"$env:HOTSTEP_ORT_TRT_BUILDER_OPT_LEVEL = \"{args.builder_optimization_level}\"",
        "",
    ]
    (out_dir / "use-prebuilt-ort-trt.ps1").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild HOT-Step ORT TensorRT engine caches")
    parser.add_argument("--bundle-dir", default=None,
                        help="Runtime artifact directory containing text_encoder.onnx/cond_encoder.onnx/vae_decoder.onnx")
    parser.add_argument("--text-onnx", default=None, help="Override text_encoder.onnx path")
    parser.add_argument("--cond-onnx", default=None, help="Override cond_encoder.onnx path")
    parser.add_argument("--vae-onnx", default=None, help="Override vae_decoder.onnx path")
    parser.add_argument("--modules", default="text,cond,vae",
                        help="Comma-separated modules to build: text,cond,vae")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"Engine cache root. Default: <bundle-dir>/{ORT_TRT_ENGINE_ROOT_NAME}")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--builder-optimization-level", type=int,
                        default=ORT_TRT_DEFAULTS["builder_optimization_level"],
                        choices=range(0, 6), metavar="{0..5}")
    parser.add_argument("--text-opt-tokens", type=int, default=ORT_TRT_DEFAULTS["text_opt_tokens"])
    parser.add_argument("--text-max-tokens", type=int, default=ORT_TRT_DEFAULTS["text_max_tokens"])
    parser.add_argument("--lyric-opt-tokens", type=int, default=ORT_TRT_DEFAULTS["lyric_opt_tokens"])
    parser.add_argument("--lyric-max-tokens", type=int, default=ORT_TRT_DEFAULTS["lyric_max_tokens"])
    parser.add_argument("--timbre-opt-frames", type=int, default=ORT_TRT_DEFAULTS["timbre_opt_frames"])
    parser.add_argument("--timbre-max-frames", type=int, default=ORT_TRT_DEFAULTS["timbre_max_frames"])
    parser.add_argument("--vae-opt-frames", type=int, default=ORT_TRT_DEFAULTS["vae_opt_frames"])
    parser.add_argument("--vae-max-frames", type=int, default=ORT_TRT_DEFAULTS["vae_max_frames"])
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else None
    if args.out_dir is None:
        if not bundle_dir:
            raise SystemExit("--out-dir is required when --bundle-dir is not supplied")
        args.out_dir = bundle_dir / ORT_TRT_ENGINE_ROOT_NAME
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    modules = parse_modules(args.modules)
    paths: dict[str, Path] = {}
    if "text-enc" in modules:
        paths["text-enc"] = bundle_file(bundle_dir, args.text_onnx, "text_encoder.onnx", "text encoder ONNX")
    if "cond-enc" in modules:
        paths["cond-enc"] = bundle_file(bundle_dir, args.cond_onnx, "cond_encoder.onnx", "condition encoder ONNX")
    if "vae-dec" in modules:
        paths["vae-dec"] = bundle_file(bundle_dir, args.vae_onnx, "vae_decoder.onnx", "VAE decoder ONNX")

    native_builder = find_native_builder()
    if native_builder is not None:
        run_native_builder(native_builder, paths, modules, args)
        return

    ort = import_or_die()
    providers = ort.get_available_providers()
    if "TensorrtExecutionProvider" not in providers:
        raise SystemExit(f"onnxruntime-gpu does not expose TensorrtExecutionProvider; available={providers}")

    summaries = []
    for module_tag in modules:
        summaries.append(build_module(ort, module_tag, paths[module_tag], args))

    write_json(args.out_dir / "ort-trt-engines.metadata.json", {
        "artifact": "hotstep-ort-trt-engine-root",
        "modules": summaries,
        "profile_defaults": profile_kwargs(args),
        "builder_optimization_level": args.builder_optimization_level,
    })
    write_env_file(args.out_dir, args)
    print(f"[ORT-TRT Build] wrote metadata: {args.out_dir / 'ort-trt-engines.metadata.json'}")
    print(f"[ORT-TRT Build] wrote env helper: {args.out_dir / 'use-prebuilt-ort-trt.ps1'}")


if __name__ == "__main__":
    main()
