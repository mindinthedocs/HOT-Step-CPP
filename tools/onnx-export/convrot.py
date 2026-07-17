#!/usr/bin/env python3
"""ConvRot group-wise Hadamard rotation for INT8 weight + activation quantization.

Adapted from https://github.com/BobJohnson24/ComfyUI-INT8-Fast (MIT license)
for the HOT-Step ONNX/TensorRT tooling. The reference implementation runs the
rotation in PyTorch at module-load time; this port emits a pure NumPy offline
weight rotation plus an ONNX graph fragment that performs the matching online
activation rotation. The two halves must use the SAME normalized regular
Hadamard matrix (``build_hadamard`` below) and the SAME ``group_size`` or the
linear-layer math breaks.

Background (see QuaRot 2024 / ConvRot 2025):
  * Standard Sylvester Hadamard matrices have an all-1s first column. When used
    for block rotation, that column sums every input row into one output
    channel, amplifying diffusion-model outliers and hurting INT8 quality.
  * Regular Hadamard matrices built from the H4 base block (Theorem 3.3) have
    every row and column summing to 2, so no single channel hoards the
    outlier energy.
  * The matrix is normalized to be orthogonal: H @ H.T = I. As a result the
    offline ``W_rot = W @ H^T`` followed by online ``x_rot = x @ H`` keeps
    ``x_rot @ W_rot^T == x @ W^T`` exactly (in floating point), so the
    rotation itself is mathematically transparent; only the quantization
    granularity changes.

Group size contract:
  * ``group_size`` must be a power of 4 (4, 16, 64, 256, 1024, ...).
  * ``in_features`` of every rotated weight must be divisible by group_size.
  * Default 256 matches the ComfyUI-INT8-Fast reference and divides the DiT
    hidden_size (2048) and intermediate_size (6144) cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Default group size. Must be a power of 4.
#
# 256 matches the PyTorch/ComfyUI reference and divides the DiT allowlist
# dimensions (hidden_size=2048, intermediate_size=6144) cleanly. The TRT CUDA
# kernel applies the regular Hadamard via H4 Kronecker butterflies, so it no
# longer stages a dense group_size×group_size H matrix in shared memory.
#
# Override via --convrot-group-size or HOTSTEP_CONVROT_GROUP_SIZE env var
# for experiments with smaller/faster groups.
CONVROT_GROUP_SIZE = 256

# Per-process cache of normalized regular Hadamard matrices, keyed by size.
_HADAMARD_CACHE: Dict[int, np.ndarray] = {}


def is_valid_group_size(group_size: int) -> bool:
    """Return True iff group_size is a power of 4 >= 4 (ConvRot constraint)."""
    if group_size < 4:
        return False
    if (group_size & (group_size - 1)) != 0:
        return False  # not a power of 2
    try:
        return math.log(group_size, 4).is_integer()
    except ValueError:
        return False


def build_hadamard(size: int) -> np.ndarray:
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot).

    Size must be a power of 4 (4, 16, 64, 256, 1024, ...). Uses the Kronecker
    construction H_{4^{k+1}} = H_{4^k} \\otimes H_4 starting from the H4 base
    block defined in Theorem 3.3 of the ConvRot paper. The matrix is then
    divided by sqrt(size) so that H @ H.T = I exactly.
    """
    if size in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[size]
    if not is_valid_group_size(size):
        raise ValueError(
            f"Regular Hadamard size must be a power of 4 (4, 16, 64, 256, ...), got {size}"
        )

    # Base H4 from Theorem 3.3 / Eq 9 in the ConvRot paper. Every row and
    # every column sums to exactly 2 — this is the property that distinguishes
    # a "regular" Hadamard matrix from the Sylvester construction and prevents
    # the all-ones column that amplifies diffusion-model outliers.
    H4 = np.array(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    H = H4
    current_size = 4
    while current_size < size:
        H = np.kron(H, H4)
        current_size *= 4

    # Orthogonalize so H @ H.T == I (within float32 precision). The ComfyUI
    # reference uses the same normalization.
    H_normalized = (H / np.sqrt(np.float32(size))).astype(np.float32)
    _HADAMARD_CACHE[size] = H_normalized
    return H_normalized


def rotate_weight(weight: np.ndarray, H: np.ndarray, group_size: int) -> np.ndarray:
    """Offline weight rotation: W_rot = W @ H_block^T.

    Operates on the canonical PyTorch Linear weight layout ``[out_features,
    in_features]``. Each row of W is split into ``in_features // group_size``
    contiguous groups of ``group_size`` columns, and each group is rotated by
    ``H^T`` (a no-op transpose-wise because H is symmetric, but kept explicit
    for clarity). The operation is mathematically equivalent to multiplying
    the full weight by a block-diagonal matrix whose blocks are H, but avoids
    materializing the (mostly zero) block-diagonal matrix.

    Args:
        weight: Shape ``[out_features, in_features]``, FP32 or any float.
        H: Normalized regular Hadamard matrix, shape ``[group_size, group_size]``.
        group_size: Block size; must divide ``in_features``.

    Returns:
        Rotated weight, same shape as input, dtype float32.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"rotate_weight expects a 2D [out, in] tensor, got shape {weight.shape}"
        )
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(
            f"in_features {in_f} not divisible by convrot group_size {group_size}"
        )
    n_groups = in_f // group_size

    w_f32 = np.ascontiguousarray(weight, dtype=np.float32)
    # (out, in) -> (out, n_groups, group_size)
    W_grouped = w_f32.reshape(out_f, n_groups, group_size)
    # Apply H^T to each group. H is symmetric so H.T == H mathematically,
    # but we keep .T to mirror the formula in the docstring.
    H_t = np.ascontiguousarray(H.T, dtype=np.float32)
    W_rot = np.matmul(W_grouped, H_t)
    return W_rot.reshape(out_f, in_f)


def rotate_activation(x: np.ndarray, H: np.ndarray, group_size: int) -> np.ndarray:
    """Online activation rotation: x_rot = x @ H_block (NumPy reference).

    Used here for numerical verification; runtime activation rotation is
    performed by the TensorRT ConvRotInt8Linear plugin.

    Args:
        x: Shape ``[..., features]``; last dim must be divisible by group_size.
        H: Normalized regular Hadamard matrix, shape ``[group_size, group_size]``.
        group_size: Block size; must divide ``features``.

    Returns:
        Rotated activation, same shape as input.
    """
    if x.ndim < 1:
        raise ValueError(f"rotate_activation expects ndim >= 1, got {x.shape}")
    features = x.shape[-1]
    if features % group_size != 0:
        raise ValueError(
            f"activation features {features} not divisible by convrot group_size {group_size}"
        )
    n_groups = features // group_size

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    orig_shape = x_f32.shape
    # (..., features) -> (..., n_groups, group_size)
    x_grouped = x_f32.reshape(*orig_shape[:-1], n_groups, group_size)
    H_dev = np.ascontiguousarray(H, dtype=np.float32)
    x_rot = np.matmul(x_grouped, H_dev)
    return x_rot.reshape(orig_shape)


def verify_rotation_pair(weight: np.ndarray, x: np.ndarray, group_size: int) -> dict:
    """Sanity check: rotating weight and activation must preserve ``x @ W^T``.

    Returns a dict with the original / rotated matmul max-abs difference.
    A passing check (max_diff < 1e-3 for FP32) confirms the rotation pair is
    a transparent similarity transform.
    """
    H = build_hadamard(group_size)
    W_rot = rotate_weight(weight, H, group_size)
    x_rot = rotate_activation(x, H, group_size)

    ref = x @ weight.T
    rot = x_rot @ W_rot.T
    diff = float(np.max(np.abs(ref - rot)))
    return {
        "group_size": group_size,
        "max_abs_diff": diff,
        "mean_abs_diff": float(np.mean(np.abs(ref - rot))),
        "pass": diff < 1e-3,
    }


def rotate_weight_with_butterfly(
    weight: np.ndarray, H: np.ndarray, group_size: int,
    **kwargs,
) -> np.ndarray:
    """Offline weight rotation: applies ConvRot H_{group_size} rotation."""
    return rotate_weight(weight, H, group_size)
