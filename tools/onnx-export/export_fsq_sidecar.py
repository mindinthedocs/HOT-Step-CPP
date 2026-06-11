#!/usr/bin/env python3
"""Extract HOT-Step FSQ tokenizer/detokenizer weights into fsq.safetensors.

The ONNX DiT runtime must not reopen the full DiT GGUF or source model just to
run cover-mode FSQ or decode LM audio_codes. This script copies only the
`tokenizer.*` and `detokenizer.*` tensors needed by the existing GGML FSQ
runtime into a small safetensors sidecar for the ONNX/TRT artifact directory.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from acestep_trt_common import (
    FSQ_SIDECAR_NAME,
    FSQ_TENSOR_PREFIXES,
    fsq_required_tensor_names,
    portable_path,
    write_json,
)


def import_or_die(module_name: str, package_hint: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = package_hint or module_name
        raise SystemExit(f"Missing Python dependency '{module_name}'. Install {hint}.") from None


def safetensor_files(model_dir: Path) -> list[Path]:
    if model_dir.is_file() and model_dir.name.endswith(".safetensors"):
        return [model_dir]

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(payload.get("weight_map", {}).values()))
        return [model_dir / name for name in shard_names]

    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single]

    shards = sorted(model_dir.glob("model-*.safetensors"))
    if shards:
        return shards

    raise SystemExit(f"No safetensors model files found in {model_dir}")


def extract_fsq_tensors(model_dir: Path) -> dict:
    safetensors = import_or_die("safetensors", "safetensors")
    selected = {}
    for path in safetensor_files(model_dir):
        with safetensors.safe_open(str(path), framework="pt", device="cpu") as sf:
            for name in sf.keys():
                if name.startswith(FSQ_TENSOR_PREFIXES):
                    selected[name] = sf.get_tensor(name).detach().cpu()

    required = fsq_required_tensor_names()
    missing = [name for name in required if name not in selected]
    if missing:
        preview = "\n".join(f"  - {name}" for name in missing[:32])
        more = "" if len(missing) <= 32 else f"\n  ... {len(missing) - 32} more"
        raise SystemExit(f"FSQ sidecar is missing required tensors:\n{preview}{more}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Source DiT safetensors directory or file")
    parser.add_argument("--output-dir", default="models/onnx", help="ONNX/TRT artifact directory")
    parser.add_argument("--output-name", default=FSQ_SIDECAR_NAME)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = out_dir / args.output_name

    tensors = extract_fsq_tensors(model_dir)
    safetensors_torch = import_or_die("safetensors.torch", "safetensors")
    safetensors_torch.save_file(
        tensors,
        str(sidecar_path),
        metadata={
            "hot_step_artifact": "fsq-sidecar",
            "tensor_count": str(len(tensors)),
        },
    )

    metadata_path = sidecar_path.with_suffix(".metadata.json")
    write_json(
        metadata_path,
        {
            "artifact": "fsq-sidecar",
            "source_model": portable_path(model_dir, metadata_path.parent),
            "sidecar": portable_path(sidecar_path, metadata_path.parent),
            "tensor_count": len(tensors),
            "required_tensor_count": len(fsq_required_tensor_names()),
            "tensor_prefixes": list(FSQ_TENSOR_PREFIXES),
            "tensors": sorted(tensors),
        },
    )
    print(f"fsq sidecar: {sidecar_path}")
    print(f"metadata: {metadata_path}")
    print(f"tensors: {len(tensors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
