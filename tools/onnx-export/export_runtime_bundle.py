#!/usr/bin/env python3
"""Build a HOT-Step ONNX/TensorRT runtime artifact directory.

This script orchestrates the existing single-purpose exporters and copies the
CPU sidecars that the C++ ONNX/TRT runtime requires next to dit.onnx. It keeps
conversion separate from generation, so the source model can be read during an
offline setup step instead of during low-VRAM synthesis.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from acestep_trt_common import (
    FSQ_SIDECAR_NAME,
    FSQ_TENSOR_PREFIXES,
    ORT_TRT_ENGINE_ROOT_NAME,
    fsq_required_tensor_names,
    validate_dit_precision_manifest,
    validate_engine_metadata,
    validate_ort_trt_cache,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"missing {label}: {path}")


def copy_sidecar(src: Path, dst: Path, label: str) -> None:
    require_file(src, label)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"[bundle] {label}: {dst}")


def run_tool(args: list[str]) -> None:
    print("[bundle] run:", " ".join(args))
    subprocess.run(args, check=True)


def validate_fsq_sidecar_metadata(output_dir: Path) -> dict:
    sidecar = output_dir / FSQ_SIDECAR_NAME
    metadata_path = sidecar.with_suffix(".metadata.json")
    require_file(sidecar, "FSQ sidecar")
    require_file(metadata_path, "FSQ sidecar metadata")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Cannot read FSQ sidecar metadata {metadata_path}: {exc}") from None
    if not isinstance(payload, dict):
        raise SystemExit(f"FSQ sidecar metadata root is not an object: {metadata_path}")
    if payload.get("artifact") != "fsq-sidecar":
        raise SystemExit("FSQ sidecar metadata does not describe an fsq-sidecar artifact.")

    required = set(fsq_required_tensor_names())
    tensors = payload.get("tensors")
    if not isinstance(tensors, list) or not all(isinstance(x, str) for x in tensors):
        raise SystemExit("FSQ sidecar metadata is missing the tensor list.")
    tensor_set = set(tensors)
    missing = sorted(required - tensor_set)
    if missing:
        preview = "\n  ".join(missing[:24])
        more = "" if len(missing) <= 24 else f"\n  ... {len(missing) - 24} more"
        raise SystemExit(f"FSQ sidecar metadata is missing required tensors:\n  {preview}{more}")

    required_count = payload.get("required_tensor_count")
    tensor_count = payload.get("tensor_count")
    if required_count != len(required):
        raise SystemExit("FSQ sidecar metadata has the wrong required tensor count.")
    if not isinstance(tensor_count, int) or tensor_count < len(required):
        raise SystemExit("FSQ sidecar metadata has an invalid tensor count.")

    prefixes = payload.get("tensor_prefixes")
    if prefixes != list(FSQ_TENSOR_PREFIXES):
        raise SystemExit("FSQ sidecar metadata has unexpected tensor prefixes.")
    return {
        "path": str(sidecar),
        "metadata": str(metadata_path),
        "tensor_count": tensor_count,
        "required_tensor_count": len(required),
    }


def validate_ort_trt_engine_root(engine_root: Path) -> dict:
    metadata_path = engine_root / "ort-trt-engines.metadata.json"
    require_file(metadata_path, "ORT TensorRT engine root metadata")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Cannot read ORT TensorRT metadata {metadata_path}: {exc}") from None
    if payload.get("artifact") != "hotstep-ort-trt-engine-root":
        raise SystemExit("ORT TensorRT metadata does not describe a HOT-Step engine root.")
    modules = payload.get("modules")
    if not isinstance(modules, list) or not modules:
        raise SystemExit("ORT TensorRT metadata has no module cache entries.")
    summaries = []
    for item in modules:
        if not isinstance(item, dict) or not isinstance(item.get("cache_dir"), str):
            raise SystemExit("ORT TensorRT metadata has an invalid module cache entry.")
        summaries.append(validate_ort_trt_cache(engine_root / item["cache_dir"]))
    return {
        "metadata": str(metadata_path),
        "module_count": len(summaries),
        "engine_count": sum(x["engine_count"] for x in summaries),
        "profile_count": sum(x["profile_count"] for x in summaries),
    }


def validate_bundle(output_dir: Path,
                    include_fsq: bool,
                    include_vae: bool,
                    build_engine: bool,
                    build_ort_engines: bool,
                    ort_engine_dir: Path | None,
                    precision_policy: str) -> None:
    required = [
        ("DiT ONNX", output_dir / "dit.onnx"),
        ("DiT metadata", output_dir / "dit.metadata.json"),
        ("DiT precision manifest", output_dir / "dit.precision-manifest.json"),
        ("DiT refit manifest", output_dir / "dit.onnx.refit_manifest.json"),
        ("config.json", output_dir / "config.json"),
        ("silence_latent.pt", output_dir / "silence_latent.pt"),
        ("null_condition_emb.bin", output_dir / "null_condition_emb.bin"),
        ("text_encoder.onnx", output_dir / "text_encoder.onnx"),
        ("embed_tokens.bin", output_dir / "embed_tokens.bin"),
        ("cond_encoder.onnx", output_dir / "cond_encoder.onnx"),
        ("vocab.json", output_dir / "vocab.json"),
        ("merges.txt", output_dir / "merges.txt"),
    ]
    if include_fsq:
        required.append(("FSQ sidecar", output_dir / FSQ_SIDECAR_NAME))
        required.append(("FSQ sidecar metadata", (output_dir / FSQ_SIDECAR_NAME).with_suffix(".metadata.json")))
    if include_vae:
        required.append(("VAE decoder ONNX", output_dir / "vae_decoder.onnx"))
    if build_engine:
        required.append(("DiT TensorRT engine", output_dir / "dit.engine"))
        required.append(("DiT TensorRT engine metadata", output_dir / "dit.engine.metadata.json"))
    if build_ort_engines:
        root = ort_engine_dir if ort_engine_dir else output_dir / ORT_TRT_ENGINE_ROOT_NAME
        required.append(("ORT TensorRT engine root metadata", root / "ort-trt-engines.metadata.json"))

    missing: list[str] = []
    for label, path in required:
        if not path.is_file():
            missing.append(f"{label}: {path}")
    if missing:
        raise SystemExit("bundle validation failed; missing:\n  " + "\n  ".join(missing))

    precision_summary = validate_dit_precision_manifest(output_dir / "dit.onnx", precision_policy)
    print(
        "[bundle] precision manifest OK: "
        f"policy={precision_summary['precision_policy']} "
        f"matched={precision_summary['matched_allowlist_count']} "
        f"downcast={precision_summary['downcast_to_fp16_count']} "
        f"int8={precision_summary.get('quantized_to_int8_count', 0)} "
        f"preserved={precision_summary['preserved_fp32_count']}"
    )
    if include_fsq:
        fsq_summary = validate_fsq_sidecar_metadata(output_dir)
        print(
            "[bundle] FSQ sidecar OK: "
            f"tensors={fsq_summary['tensor_count']} "
            f"required={fsq_summary['required_tensor_count']} "
            f"metadata={fsq_summary['metadata']}"
        )
    engine_metadata = output_dir / "dit.engine.metadata.json"
    if build_engine:
        engine_summary = validate_engine_metadata(engine_metadata, precision_summary["precision_policy"])
        print(
            "[bundle] engine metadata OK: "
            f"TRT={engine_summary['tensorrt_version']} "
            f"profile={engine_summary['profile']} "
            f"weight_streaming={engine_summary['weight_streaming']} "
            f"max=batch:{engine_summary['profile_max_batch']} "
            f"T:{engine_summary['profile_max_T']} "
            f"enc_S:{engine_summary['profile_max_enc_S']} "
                f"source={engine_summary['source_path']}"
        )
    if build_ort_engines:
        ort_root = ort_engine_dir if ort_engine_dir else output_dir / ORT_TRT_ENGINE_ROOT_NAME
        ort_summary = validate_ort_trt_engine_root(ort_root)
        print(
            "[bundle] ORT TensorRT caches OK: "
            f"modules={ort_summary['module_count']} "
            f"engines={ort_summary['engine_count']} "
            f"profiles={ort_summary['profile_count']} "
            f"metadata={ort_summary['metadata']}"
        )

    print(f"[bundle] validated runtime artifact directory: {output_dir}")


def _exports_complete(output_dir: Path, include_fsq: bool, include_vae: bool) -> bool:
    """Return True when all ONNX exports and sidecars are present on disk.

    Checked independently of engine builds so that a failed TRT build does not
    force re-exporting the (already-correct) ONNX files.
    """
    required = [
        output_dir / "dit.onnx",
        output_dir / "dit.metadata.json",
        output_dir / "dit.precision-manifest.json",
        output_dir / "dit.onnx.refit_manifest.json",
        output_dir / "config.json",
        output_dir / "silence_latent.pt",
        output_dir / "null_condition_emb.bin",
        output_dir / "text_encoder.onnx",
        output_dir / "embed_tokens.bin",
        output_dir / "cond_encoder.onnx",
        output_dir / "vocab.json",
        output_dir / "merges.txt",
    ]
    if include_fsq:
        required.append(output_dir / FSQ_SIDECAR_NAME)
    if include_vae:
        required.append(output_dir / "vae_decoder.onnx")
    return all(p.is_file() for p in required)


def _dit_engine_complete(output_dir: Path) -> bool:
    """Return True when the DiT TRT engine is present and non-empty.

    build-trt-engine.py writes dit.engine directly next to dit.onnx (single
    source of truth — no engines/ subdir, no alias). A zero-byte engine means
    a previous build was interrupted.
    """
    engine = output_dir / "dit.engine"
    return engine.is_file() and engine.stat().st_size > 0


def _ort_engines_complete(output_dir: Path, ort_engine_dir: Path | None) -> bool:
    """Return True when the ORT-TRT engine cache root metadata exists."""
    root = ort_engine_dir if ort_engine_dir else output_dir / ORT_TRT_ENGINE_ROOT_NAME
    return (root / "ort-trt-engines.metadata.json").is_file()


def try_reuse_existing_bundle(output_dir: Path,
                              include_fsq: bool,
                              include_vae: bool,
                              build_engine: bool,
                              build_ort_engines: bool,
                              ort_engine_dir: Path | None,
                              precision_policy: str) -> bool:
    if not output_dir.is_dir():
        return False
    try:
        validate_bundle(output_dir, include_fsq, include_vae, build_engine,
                        build_ort_engines, ort_engine_dir, precision_policy)
    except SystemExit as exc:
        print(f"[bundle] existing artifact directory is incomplete or incompatible; exporting ({exc})")
        return False
    print("[bundle] existing runtime artifact directory is complete; skipping export/build")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a complete HOT-Step ONNX/TRT runtime bundle")
    parser.add_argument("--dit-dir", required=True, help="DiT source model directory")
    parser.add_argument("--text-encoder-dir", required=True, help="Qwen3 text encoder source directory")
    parser.add_argument("--vae-path", default=None, help="Optional VAE source checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="Runtime artifact output directory")
    parser.add_argument("--precision", choices=["q8map-fp16", "w8a8", "fp32"], default="q8map-fp16")
    parser.add_argument("--profile", default="pj-ode-320", help="TensorRT build profile")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="cpu", help="Device for DiT/text/condition export")
    parser.add_argument("--build-engine", action="store_true",
                        help="Build dit.engine with TensorRT 11 after ONNX export")
    parser.add_argument("--build-ort-engines", action="store_true",
                        help="Prebuild ONNX Runtime TensorRT caches for text/cond/VAE side modules")
    parser.add_argument("--ort-engine-dir", default=None,
                        help=f"ORT TensorRT cache root. Default: <output-dir>/{ORT_TRT_ENGINE_ROOT_NAME}")
    parser.add_argument("--ort-modules", default=None,
                        help="Comma-separated ORT modules to prebuild. Default: text,cond plus vae when exported")
    parser.add_argument("--skip-fsq", action="store_true",
                        help="Skip fsq.safetensors extraction for text-only validation bundles")
    parser.add_argument("--skip-vae", action="store_true",
                        help="Skip VAE decoder export even when --vae-path is supplied")
    parser.add_argument("--verify", action="store_true", help="Run exporter ONNX verification steps")
    args = parser.parse_args()

    dit_dir = Path(args.dit_dir)
    text_dir = Path(args.text_encoder_dir)
    vae_path = Path(args.vae_path) if args.vae_path else None
    output_dir = Path(args.output_dir)
    ort_engine_dir = Path(args.ort_engine_dir) if args.ort_engine_dir else None
    include_fsq = not args.skip_fsq
    include_vae = vae_path is not None and not args.skip_vae

    # Fast path: everything already done (idempotent re-run of a complete build).
    if try_reuse_existing_bundle(output_dir, include_fsq, include_vae, args.build_engine,
                                 args.build_ort_engines, ort_engine_dir, args.precision):
        return

    require_dir(dit_dir, "DiT source directory")
    require_dir(text_dir, "text encoder source directory")
    if vae_path is not None:
        require_dir(vae_path, "VAE source directory")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── ONNX export stage ────────────────────────────────────────────────────
    # Each export script has its own --force / skip-if-exists guard, but we
    # also check here at the orchestration level so we can skip the entire
    # export block (including sidecars) without spawning any subprocesses.
    if not _exports_complete(output_dir, include_fsq, include_vae):
        copy_sidecar(dit_dir / "config.json", output_dir / "config.json", "config.json")
        copy_sidecar(dit_dir / "silence_latent.pt", output_dir / "silence_latent.pt", "silence_latent.pt")
        copy_sidecar(text_dir / "vocab.json", output_dir / "vocab.json", "vocab.json")
        copy_sidecar(text_dir / "merges.txt", output_dir / "merges.txt", "merges.txt")

        dit_onnx = output_dir / "dit.onnx"
        text_onnx = output_dir / "text_encoder.onnx"
        cond_onnx = output_dir / "cond_encoder.onnx"

        run_tool([
            sys.executable,
            str(SCRIPT_DIR / "export_dit.py"),
            "--model-dir", str(dit_dir),
            "--output", str(dit_onnx),
            "--opset", str(args.opset),
            "--precision", args.precision,
            "--device", args.device,
            *(["--verify"] if args.verify else []),
        ])

        run_tool([
            sys.executable,
            str(SCRIPT_DIR / "export_text_enc.py"),
            "--model-dir", str(text_dir),
            "--output", str(text_onnx),
            "--dit-dir", str(dit_dir),
            "--opset", str(args.opset),
            "--device", args.device,
            *(["--verify"] if args.verify else []),
        ])

        run_tool([
            sys.executable,
            str(SCRIPT_DIR / "export_cond_enc.py"),
            "--model-dir", str(dit_dir),
            "--output", str(cond_onnx),
            "--opset", str(args.opset),
            "--device", args.device,
            *(["--verify"] if args.verify else []),
        ])

        if include_fsq:
            run_tool([
                sys.executable,
                str(SCRIPT_DIR / "export_fsq_sidecar.py"),
                "--model-dir", str(dit_dir),
                "--output-dir", str(output_dir),
            ])

        if include_vae:
            run_tool([
                sys.executable,
                str(SCRIPT_DIR / "export_vae.py"),
                "--vae-path", str(vae_path),
                "--output", str(output_dir / "vae_decoder.onnx"),
                "--opset", str(args.opset),
            ])
    else:
        print("[bundle] ONNX exports already present — skipping export stage")

    dit_onnx = output_dir / "dit.onnx"
    text_onnx = output_dir / "text_encoder.onnx"
    cond_onnx = output_dir / "cond_encoder.onnx"

    # ── DiT TRT engine build ─────────────────────────────────────────────────
    # build-trt-engine.py writes dit.engine directly next to dit.onnx.
    if args.build_engine:
        if not _dit_engine_complete(output_dir):
            run_tool([
                sys.executable,
                str(SCRIPT_DIR / "build-trt-engine.py"),
                "--onnx", str(dit_onnx),
                "--profile", args.profile,
                "--precision-policy", args.precision,
            ])
        else:
            print("[bundle] DiT TRT engine already present — skipping build-trt-engine")

    # ── ORT-TRT encoder/VAE engine builds ───────────────────────────────────
    # Skip when the engine-root metadata file exists (written as the last action
    # of build-ort-trt-engines.py, so its presence means the whole build
    # completed successfully).
    if args.build_ort_engines:
        if not _ort_engines_complete(output_dir, ort_engine_dir):
            ort_modules = args.ort_modules if args.ort_modules else ("text,cond,vae" if include_vae else "text,cond")
            run_tool([
                sys.executable,
                str(SCRIPT_DIR / "build-ort-trt-engines.py"),
                "--bundle-dir", str(output_dir),
                "--modules", ort_modules,
                *(["--out-dir", str(ort_engine_dir)] if ort_engine_dir else []),
            ])
        else:
            print("[bundle] ORT-TRT engine caches already present — skipping build-ort-trt-engines")

    validate_bundle(output_dir, include_fsq, include_vae, args.build_engine,
                    args.build_ort_engines, ort_engine_dir, args.precision)


if __name__ == "__main__":
    main()
