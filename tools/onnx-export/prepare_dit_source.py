#!/usr/bin/env python3
"""Source-asset preparation for the DiT bundle export.

The low-memory DiT export (``export_dit.py``) requires F32 source safetensors,
but the published ``ACE-Step/acestep-v15-sft`` snapshot ships BF16. And the
``silence_latent.pt`` sidecar the FSQ tokenizer needs (read by the C++
``silence-latent.h`` as ``[64,15000]`` f32) is not always present in the source
directory; it can be reconstructed from the BF16 GGUF, which embeds it as a
``[15000,64]`` f32 tensor.

Both steps are idempotent and detection-driven: ``ensure_f32_dit`` converts only
when the source is BF16, ``ensure_silence_latent`` reconstructs only when the
sidecar is absent. The orchestrator runs them before ``export-dit`` so a fresh
``bundle`` against the published BF16 snapshot succeeds without manual prep.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# The canonical base modeling file name; also used by trt_bundle_manager.py as a
# prepare-step output marker so that a staging dir missing this file forces
# prepare-dit to re-run and copy it from the (freshly downloaded) source.
DIT_MODELING_PY = "modeling_acestep_v15_base.py"

# Modeling/config files copied alongside the converted weights so the F32
# staging dir is a drop-in replacement for the source snapshot.
_DIT_AUX_FILES = (
    "config.json",
    "silence_latent.pt",
    "apg_guidance.py",
    "configuration_acestep_v15.py",
    DIT_MODELING_PY,
)

SILENCE_LATENT = "silence_latent.pt"
_SILENCE_DIM0 = 64
_SILENCE_DIM1 = 15000


def safetensors_is_bf16(model_path: Path) -> bool:
    with safe_open(str(model_path), framework="pt") as f:
        for key in f.keys():
            return f.get_slice(key).get_dtype() == "BF16"
    return False


def _shard_paths(src_dir: Path) -> list[Path] | None:
    """Return sorted list of sharded safetensors paths, or None if unsharded."""
    index_path = src_dir / "model.safetensors.index.json"
    if index_path.is_file():
        import json
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        return [src_dir / name for name in shard_names]
    return None


def _convert_shard_bf16_to_f32(src_shard: Path, dst_shard: Path) -> int:
    """Convert a single BF16 safetensors shard to F32. Returns tensor count."""
    with safe_open(str(src_shard), framework="pt") as f:
        metadata = f.metadata() or {}
        tensors = {k: f.get_tensor(k).to(torch.float32).contiguous() for k in f.keys()}
    save_file(tensors, str(dst_shard), metadata=metadata)
    return len(tensors)


def ensure_f32_dit(src_dir: Path, staging_dir: Path) -> Path:
    """Return a DiT source dir whose safetensors are F32.

    Handles both single-file (``model.safetensors``) and sharded
    (``model-NNNNN-of-MMMMM.safetensors`` + ``model.safetensors.index.json``)
    layouts. If ``src_dir`` already holds F32 weights, it is returned unchanged.
    If the weights are BF16, they are converted into ``staging_dir`` (with the
    config and modeling files copied alongside) and ``staging_dir`` is returned.
    The conversion is skipped when the staging weights already exist.
    """

    # Sharded model: model-00001-of-00004.safetensors + index JSON
    shard_paths = _shard_paths(src_dir)
    if shard_paths is not None:
        # Check dtype from the first shard
        if not safetensors_is_bf16(shard_paths[0]):
            return src_dir

        staging_dir.mkdir(parents=True, exist_ok=True)
        all_exist = all((staging_dir / p.name).is_file() for p in shard_paths)
        if not all_exist:
            total = 0
            for i, src_shard in enumerate(shard_paths):
                dst_shard = staging_dir / src_shard.name
                if dst_shard.is_file():
                    continue
                n = _convert_shard_bf16_to_f32(src_shard, dst_shard)
                total += n
                print(f"[prepare-dit] BF16->F32 shard {i+1}/{len(shard_paths)}: "
                      f"{src_shard.name} ({n} tensors)")
            print(f"[prepare-dit] BF16->F32 total {total} tensors across {len(shard_paths)} shards")

        # Copy the index JSON so downstream code can find the shards
        src_index = src_dir / "model.safetensors.index.json"
        dst_index = staging_dir / "model.safetensors.index.json"
        if src_index.is_file() and not dst_index.is_file():
            shutil.copy2(src_index, dst_index)
            print(f"[prepare-dit] copied model.safetensors.index.json")

        for name in _DIT_AUX_FILES:
            src = src_dir / name
            dst = staging_dir / name
            if src.is_file() and not dst.is_file():
                shutil.copy2(src, dst)
                print(f"[prepare-dit] copied {name}")
        return staging_dir

    # Single-file model: model.safetensors
    model = src_dir / "model.safetensors"
    if not model.is_file():
        raise FileNotFoundError(f"DiT source missing model.safetensors: {model}")
    if not safetensors_is_bf16(model):
        return src_dir

    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_model = staging_dir / "model.safetensors"
    if not staged_model.is_file():
        with safe_open(str(model), framework="pt") as f:
            metadata = f.metadata() or {}
            tensors = {k: f.get_tensor(k).to(torch.float32).contiguous() for k in f.keys()}
        save_file(tensors, str(staged_model), metadata=metadata)
        print(f"[prepare-dit] BF16->F32 wrote {len(tensors)} tensors to {staged_model}")

    for name in _DIT_AUX_FILES:
        src = src_dir / name
        dst = staging_dir / name
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            print(f"[prepare-dit] copied {name}")
    return staging_dir


def _read(fmt: str, f) -> tuple:
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def _read_str(f) -> str:
    (n,) = _read("<Q", f)
    return f.read(n).decode("utf-8")


# GGUF metadata value-type ids -> struct formats ("str"/"arr" handled inline).
_GGUF_VTYPE = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 8: "str", 9: "arr", 10: "<Q", 11: "<q", 12: "<d",
}
_GGUF_MAGIC = 0x46554747
_GGML_TYPE_F32 = 0


def _read_value(f, vtype: int):
    if vtype == 8:
        return _read_str(f)
    if vtype == 9:
        (elem_t,) = _read("<I", f)
        (count,) = _read("<Q", f)
        return [_read_value(f, elem_t) for _ in range(count)]
    return _read(_GGUF_VTYPE[vtype], f)[0]


def _silence_from_gguf(gguf_path: Path) -> torch.Tensor:
    with open(gguf_path, "rb") as f:
        magic, _version = _read("<II", f)
        if magic != _GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: magic={magic:#x} ({gguf_path})")
        n_tensors, n_kv = _read("<QQ", f)
        for _ in range(n_kv):
            _read_str(f)
            (vtype,) = _read("<I", f)
            _read_value(f, vtype)
        tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (n_dims,) = _read("<I", f)
            dims = [_read("<Q", f)[0] for _ in range(n_dims)]
            (ttype,) = _read("<I", f)
            (offset,) = _read("<Q", f)
            tensors.append((name, dims, ttype, offset))
        align = 32
        data_start = (f.tell() + align - 1) // align * align

    target = next((t for t in tensors if "silence" in t[0].lower()), None)
    if target is None:
        raise ValueError(f"no silence tensor in GGUF: {gguf_path}")
    name, dims, ttype, offset = target
    if ttype != _GGML_TYPE_F32:
        raise ValueError(f"silence tensor type {ttype} != f32 ({gguf_path})")

    # GGUF stores ne fastest-first; convert.py wrote the [15000,64] numpy array
    # so ggml ne = (64, 15000). Read raw, reshape [15000,64], transpose to the
    # PyTorch [64,15000] layout silence-latent.h reads back.
    ne0, ne1 = dims[0], dims[1]
    with open(gguf_path, "rb") as f:
        f.seek(data_start + offset)
        raw = f.read(ne0 * ne1 * 4)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(ne1, ne0)
    pt = torch.from_numpy(arr.T.copy())
    if tuple(pt.shape) != (_SILENCE_DIM0, _SILENCE_DIM1):
        raise ValueError(f"reconstructed silence_latent shape {tuple(pt.shape)} != (64, 15000)")
    print(f"[prepare-dit] silence_latent {name} ggml ne=({ne0},{ne1}) -> torch {tuple(pt.shape)}")
    return pt


def ensure_silence_latent(target_dir: Path, gguf_path: Path) -> Path:
    """Reconstruct ``silence_latent.pt`` into ``target_dir`` if it is absent.

    Returns the path to the sidecar. When present it is left untouched; when
    absent it is reconstructed from ``gguf_path`` (which embeds the tensor) and
    saved in the ``[64,15000]`` f32 PyTorch layout the C++ reader expects.
    """

    out = target_dir / SILENCE_LATENT
    if out.is_file():
        return out
    if not gguf_path.is_file():
        raise FileNotFoundError(
            f"silence_latent.pt absent from {target_dir} and GGUF not found at "
            f"{gguf_path} to reconstruct it from"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_silence_from_gguf(gguf_path), out)
    print(f"[prepare-dit] saved {out}")
    return out


def prepare_dit_dir(src_dir: Path, staging_dir: Path, gguf_path: Path) -> Path:
    """Materialize a ready-to-export DiT source dir and return its path.

    Converts BF16->F32 into ``staging_dir`` when needed (else returns ``src_dir``),
    then guarantees ``silence_latent.pt`` exists in the chosen dir, reconstructing
    it from ``gguf_path`` if absent. Idempotent: a second call is a fast no-op.
    """

    prepared = ensure_f32_dit(src_dir, staging_dir)
    ensure_silence_latent(prepared, gguf_path)
    return prepared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare_dit_source",
        description="Prepare F32 DiT weights + silence_latent.pt for the bundle export",
    )
    parser.add_argument("--src-dir", required=True, help="Source DiT safetensors directory (may be BF16)")
    parser.add_argument("--staging-dir", required=True, help="F32 staging directory to write/reuse")
    parser.add_argument("--gguf", required=True, help="BF16 GGUF to reconstruct silence_latent.pt from if absent")
    args = parser.parse_args(argv)

    staged = prepare_dit_dir(Path(args.src_dir), Path(args.staging_dir), Path(args.gguf))
    print(f"[prepare-dit] DiT source ready: {staged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
