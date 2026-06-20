#!/usr/bin/env python3
"""Export and prebuild HOT-Step runtime bundles in the split folder layout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_models(text: str) -> list[str]:
    aliases = {
        "dit": "dit",
        "embedding": "embedding",
        "embed": "embedding",
        "text": "embedding",
        "text-enc": "embedding",
        "lm": "lm",
        "vae": "vae",
        "all": "all",
    }
    out: list[str] = []
    for raw in text.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise SystemExit(f"unknown model selection {raw!r}; use dit,embedding,lm,vae,all")
        value = aliases[key]
        if value == "all":
            return ["dit", "embedding", "lm", "vae"]
        if value not in out:
            out.append(value)
    return out or ["dit", "embedding", "lm", "vae"]


def run_tool(args: list[str]) -> None:
    print("[bundles] run:", " ".join(str(x) for x in args))
    subprocess.run([str(x) for x in args], check=True)


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise SystemExit(f"missing {label}: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    return path


def find_sidecar(input_root: Path, model_dir: Path, names: tuple[str, ...], label: str) -> Path:
    candidates: list[Path] = []
    for name in names:
        candidates.append(model_dir / name)
        candidates.append(input_root / "safetensors" / name)
        candidates.append(input_root / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"missing {label}; checked: " + ", ".join(str(p) for p in candidates))


def copy_file(src: Path, dst: Path, label: str) -> None:
    require_file(src, label)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"[bundles] {label}: {dst}")


def prepare_bundle(path: Path, force_rewrite: bool) -> bool:
    if path.exists() and not force_rewrite:
        print(f"[bundles] exists, skipping: {path}")
        return False
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return True


def bundle_name(prefix: str, module: str, precision: str) -> str:
    if module == "dit":
        return f"{prefix}-dit-trt11-{precision}"
    return f"{prefix}-{module}-trt11"


def build_dit(args, input_root: Path, src: Path, dst: Path) -> None:
    silence = find_sidecar(input_root, src, ("silence_latent.pt", "silent_latent.pt"), "silence_latent.pt")
    copy_file(silence, dst / "silence_latent.pt", "silence_latent.pt")

    dit_onnx = dst / "dit.onnx"
    run_tool([
        sys.executable,
        SCRIPT_DIR / "export_dit.py",
        "--model-dir", src,
        "--output", dit_onnx,
        "--opset", args.opset,
        "--precision", args.precision,
        "--device", args.device,
    ])
    run_tool([
        sys.executable,
        SCRIPT_DIR / "export_cond_enc.py",
        "--model-dir", src,
        "--output", dst / "cond_encoder.onnx",
        "--opset", args.opset,
        "--device", args.device,
    ])
    run_tool([
        sys.executable,
        SCRIPT_DIR / "export_fsq_sidecar.py",
        "--model-dir", src,
        "--output-dir", dst,
    ])
    run_tool([
        sys.executable,
        SCRIPT_DIR / "build-trt-engine.py",
        "--onnx", dit_onnx,
        "--profile", args.profile,
        "--precision-policy", args.precision,
        "--workspace-gb", args.dit_workspace_gb,
        "--builder-optimization-level", args.builder_optimization_level,
    ])
    run_tool([
        sys.executable,
        SCRIPT_DIR / "build-ort-trt-engines.py",
        "--cond-onnx", dst / "cond_encoder.onnx",
        "--modules", "cond",
        "--out-dir", dst / "ort-trt-engines",
        "--workspace-gb", args.ort_workspace_gb,
        "--builder-optimization-level", args.ort_builder_optimization_level,
    ])


def build_embedding(args, input_root: Path, src: Path, dst: Path) -> None:
    copy_file(find_sidecar(input_root, src, ("vocab.json",), "vocab.json"), dst / "vocab.json", "vocab.json")
    copy_file(find_sidecar(input_root, src, ("merges.txt",), "merges.txt"), dst / "merges.txt", "merges.txt")
    run_tool([
        sys.executable,
        SCRIPT_DIR / "export_text_enc.py",
        "--model-dir", src,
        "--output", dst / "text_encoder.onnx",
        "--opset", args.opset,
        "--device", args.device,
    ])
    run_tool([
        sys.executable,
        SCRIPT_DIR / "build-ort-trt-engines.py",
        "--text-onnx", dst / "text_encoder.onnx",
        "--modules", "text",
        "--out-dir", dst / "ort-trt-engines",
        "--workspace-gb", args.ort_workspace_gb,
        "--builder-optimization-level", args.ort_builder_optimization_level,
    ])


def build_lm(args, input_root: Path, src: Path, dst: Path) -> None:
    copy_file(find_sidecar(input_root, src, ("vocab.json",), "vocab.json"), dst / "vocab.json", "vocab.json")
    copy_file(find_sidecar(input_root, src, ("merges.txt",), "merges.txt"), dst / "merges.txt", "merges.txt")
    run_tool([
        sys.executable,
        SCRIPT_DIR / "export_lm.py",
        "--model-dir", src,
        "--output", dst,
        "--device", args.device,
        "--opset", args.opset,
        "--full-only",
    ])
    run_tool([
        sys.executable,
        SCRIPT_DIR / "build-lm-trt-engine.py",
        "--onnx", dst / "lm_full.onnx",
        "--engine", dst / "lm_full.engine",
        "--max-seq-len", args.lm_max_seq_len,
        "--max-input-len", args.lm_max_input_len,
        "--opt-past-len", args.lm_opt_past_len,
        "--workspace-gb", args.lm_workspace_gb,
        "--builder-optimization-level", args.builder_optimization_level,
    ])


def build_vae(args, _input_root: Path, src: Path, dst: Path) -> None:
    run_tool([
        sys.executable,
        SCRIPT_DIR / "export_vae.py",
        "--vae-path", src,
        "--output", dst / "vae_decoder.onnx",
        "--opset", args.opset,
    ])
    run_tool([
        sys.executable,
        SCRIPT_DIR / "build-ort-trt-engines.py",
        "--vae-onnx", dst / "vae_decoder.onnx",
        "--modules", "vae",
        "--out-dir", dst / "ort-trt-engines",
        "--workspace-gb", args.ort_workspace_gb,
        "--builder-optimization-level", args.ort_builder_optimization_level,
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export/build split HOT-Step runtime bundles")
    parser.add_argument("--input-root", required=True, type=Path,
                        help="Root containing safetensors/dit, safetensors/embedding, safetensors/lm, safetensors/vae")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Destination root that will contain dit/, embedding/, lm/, vae/")
    parser.add_argument("--models", default="all",
                        help="Comma-separated selection: dit,embedding,lm,vae,all")
    parser.add_argument("--name-prefix", default=None,
                        help="Bundle name prefix (default: input root folder name)")
    parser.add_argument("--precision", choices=["q8map-fp16", "w8a8", "fp32"], default="w8a8")
    parser.add_argument("--profile", default="pj-ode-320")
    parser.add_argument("--opset", default="18")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-rewrite", action="store_true",
                        help="Rewrite existing destination bundles")
    parser.add_argument("--builder-optimization-level", default="0")
    parser.add_argument("--ort-builder-optimization-level", default="0")
    parser.add_argument("--dit-workspace-gb", default="2")
    parser.add_argument("--lm-workspace-gb", default="2")
    parser.add_argument("--ort-workspace-gb", default="2")
    parser.add_argument("--lm-max-seq-len", default="8192")
    parser.add_argument("--lm-max-input-len", default="2048")
    parser.add_argument("--lm-opt-past-len", default="512")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    require_dir(input_root / "safetensors", "safetensors root")

    selected = parse_models(args.models)
    prefix = args.name_prefix or input_root.name
    builders = {
        "dit": build_dit,
        "embedding": build_embedding,
        "lm": build_lm,
        "vae": build_vae,
    }
    source_labels = {
        "dit": "DiT safetensors folder",
        "embedding": "embedding safetensors folder",
        "lm": "LM safetensors folder",
        "vae": "VAE safetensors folder",
    }

    for module in selected:
        src = require_dir(input_root / "safetensors" / module, source_labels[module])
        dst = output_root / module / bundle_name(prefix, module, args.precision)
        if not prepare_bundle(dst, args.force_rewrite):
            continue
        builders[module](args, input_root, src, dst)
        print(f"[bundles] complete: {dst}")


if __name__ == "__main__":
    main()
