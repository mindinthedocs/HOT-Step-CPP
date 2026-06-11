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

# Modeling/config files copied alongside the converted weights so the F32
# staging dir is a drop-in replacement for the source snapshot.
_DIT_AUX_FILES = (
    "config.json",
    "silence_latent.pt",
    "apg_guidance.py",
    "configuration_acestep_v15.py",
    "modeling_acestep_v15_base.py",
)

SILENCE_LATENT = "silence_latent.pt"
_SILENCE_DIM0 = 64
_SILENCE_DIM1 = 15000


def safetensors_is_bf16(model_path: Path) -> bool:
    with safe_open(str(model_path), framework="pt") as f:
        for key in f.keys():
            return f.get_slice(key).get_dtype() == "BF16"
    return False


def ensure_f32_dit(src_dir: Path, staging_dir: Path) -> Path:
    """Return a DiT source dir whose ``model.safetensors`` is F32.

    If ``src_dir`` already holds F32 weights, it is returned unchanged. If the
    weights are BF16, they are converted into ``staging_dir`` (with the config
    and modeling files copied alongside) and ``staging_dir`` is returned. The
    conversion is skipped when the staging weights already exist.
    """

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
