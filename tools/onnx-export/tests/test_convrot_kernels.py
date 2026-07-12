#!/usr/bin/env python3
"""Unit tests for the K1 rotation+quant kernel and the hybrid_direct K2 GEMM.

These tests use Triton's interpreter mode (``TRITON_INTERPRET=1``) so they
run on a CPU-only CI machine without a CUDA device.  They validate:

  * The K1 direct H4 Kronecker butterfly rotation matches a NumPy reference
    built from the dense H256r matrix.
  * The K1 store mask is honoured for M not divisible by BLOCK_M (canary
    bytes placed immediately after the workspace buffers remain intact).
  * The plain/unmasked K2 hybrid_direct GEMM+dequant kernel matches a
    NumPy reference for the canonical [N, K] W_q layout (the kernel folds
    the transpose in via ``tl.trans(wq)``).
  * Host-side M padding for plain K2 is emulated by zero-padding X_q /
    X_scale, writing a padded output tile, and slicing back the real rows
    (this mirrors the C++ plugin's enqueue-time tail handling).
  * The end-to-end K1 -> K2 pipeline matches the FP32 reference
    ``y = x @ W^T + bias`` within INT8 quant tolerance.

These tests are the canonical correctness gate for the optimisations
described in the version 7 patch.  They do NOT exercise the autotune search
(which requires a real GPU) -- only the math.

Run with:
    TRITON_INTERPRET=1 python -m pytest tools/onnx-export/tests/test_convrot_kernels.py
or:
    TRITON_INTERPRET=1 python tools/onnx-export/tests/test_convrot_kernels.py
"""
from __future__ import annotations

import os
# Must be set before any triton import.
os.environ.setdefault("TRITON_INTERPRET", "1")

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Make the onnx-export package importable.
TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from convrot import build_hadamard, rotate_weight
import extract_jit_cubins_autotune as m


# ─── Helpers ──────────────────────────────────────────────────────────────

def _kron_pow(M, k):
    out = M
    for _ in range(k - 1):
        out = np.kron(out, M)
    return out


def _build_h256r():
    """Build the dense normalized H256r matrix the way convrot.build_hadamard does."""
    H4r = np.array([[1, 1, 1, -1],
                    [1, 1, -1, 1],
                    [1, -1, 1, 1],
                    [-1, 1, 1, 1]], dtype=np.float32)
    H256r = _kron_pow(H4r, 4)
    return (H256r / np.sqrt(np.float32(256))).astype(np.float32)


def _set_k1_config(BLOCK_M):
    m.kernel1_convrot_quant.configs = [
        m.triton.Config({"BLOCK_M": BLOCK_M}, num_warps=4, num_stages=1)
    ]


def _set_k2_config(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    m.kernel2_gemm_dequant.configs = [
        m.triton.Config(
            {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
             "GROUP_M": GROUP_M},
            num_warps=4, num_stages=1,
        )
    ]


# ─── Tests ────────────────────────────────────────────────────────────────


class TestK1DirectRotation(unittest.TestCase):
    """The K1 direct H4 Kronecker butterfly must match the dense H256r @ x reference."""

    def setUp(self):
        np.random.seed(42)
        self.H256r = _build_h256r()

    def _reference(self, x_np):
        return x_np @ self.H256r.T

    def _triton_k1(self, x_np, BLOCK_M):
        x_torch = torch.from_numpy(x_np.copy())
        M, K = x_torch.shape
        xq = torch.empty((M, K), dtype=torch.int8, device=x_torch.device)
        xs = torch.empty((K // 256, M), dtype=torch.float32, device=x_torch.device)
        _set_k1_config(BLOCK_M)
        m.kernel1_convrot_quant[(1,)](
            x_torch, xq, xs,
            M, K,
            K, 1, K, 1, 1, M,        # strides
            GROUP_SIZE=256, INPUT_FP16=False,
            CONTIG_XK=True,
        )
        return xq, xs

    def _run_for_block_m(self, BLOCK_M):
        x_np = (np.random.randn(BLOCK_M, 256) * 0.1).astype(np.float32)
        xq, xs = self._triton_k1(x_np, BLOCK_M)
        # Reverse the INT8 quant to recover the rotated FP32 values.
        rotated_recovered = xq.to(torch.float32).numpy() * xs[0, :, None].numpy()
        rotated_ref = self._reference(x_np)
        max_diff = float(np.max(np.abs(rotated_recovered - rotated_ref)))
        max_ref = float(np.max(np.abs(rotated_ref)))
        rel_diff = max_diff / max_ref
        self.assertLess(rel_diff, 0.05,
                        f"BLOCK_M={BLOCK_M}: rel_diff={rel_diff:.4e}")

    def test_block_m_16(self):
        self._run_for_block_m(16)

    def test_block_m_32(self):
        self._run_for_block_m(32)

    def test_block_m_64(self):
        self._run_for_block_m(64)


class TestK1UnalignedM(unittest.TestCase):
    """K1 must NOT write out of bounds when M is not a multiple of BLOCK_M.

    The previous FAST_PATH=True code path dropped the store mask, causing OOB
    writes for every real production shape (M=1, 64, 300, 3000 — none are
    multiples of typical BLOCK_M values).  These tests use canary bytes
    (0xDE pattern) placed immediately after the workspace buffers and verify
    they remain untouched after the kernel runs.
    """

    def setUp(self):
        np.random.seed(999)
        self.H256r = _build_h256r()

    def _reference(self, x_np):
        return x_np @ self.H256r.T

    def _run_with_canary(self, M, BLOCK_M):
        """Run K1 with workspace padded by canary bytes, verify no OOB write."""
        K = 256
        x_np = (np.random.randn(M, K) * 0.1).astype(np.float32)

        # Allocate workspace with canary padding after the valid region.
        CANARY_INT8 = -34  # 0xDE in two's complement
        canary_rows = BLOCK_M  # enough to catch any tile-overflow
        xq_padded = np.full((M + canary_rows, K), CANARY_INT8, dtype=np.int8)
        xq_full = torch.from_numpy(xq_padded.copy())
        xq_view = xq_full[:M, :]  # the kernel writes here; rows [M, M+canary) are canary

        # X_scale: [n_groups, M + canary_cols] fp32 + canary.  Use NaN as the
        # canary.  Pass the true group stride so a future K > 256 test would
        # not silently read the canary region as the second group's row data.
        n_groups = K // 256
        canary_cols = canary_rows
        true_stride_xsg = M + canary_cols
        xs_padded = np.full((n_groups, true_stride_xsg), float('nan'), dtype=np.float32)
        xs_full = torch.from_numpy(xs_padded.copy())
        xs_view = xs_full[:, :M]

        x_torch = torch.from_numpy(x_np.copy())
        _set_k1_config(BLOCK_M)
        m.kernel1_convrot_quant[(1,)](
            x_torch, xq_view, xs_view,
            M, K,
            K, 1, K, 1, 1, true_stride_xsg,
            GROUP_SIZE=256, INPUT_FP16=False,
            CONTIG_XK=True,
        )

        # Verify canary bytes are untouched (no OOB write).
        xq_canary = xq_full[M:, :].numpy()
        self.assertTrue(np.all(xq_canary == CANARY_INT8),
                        f"BLOCK_M={BLOCK_M}, M={M}: X_q canary corrupted! "
                        f"Non-canary values: {xq_canary[xq_canary != CANARY_INT8][:10]}")

        xs_canary = xs_full[:, M:].numpy()
        self.assertTrue(np.all(np.isnan(xs_canary)),
                        f"BLOCK_M={BLOCK_M}, M={M}: X_scale canary corrupted! "
                        f"Non-NaN values: {xs_canary[~np.isnan(xs_canary)][:10]}")

        # Also verify the in-bounds values are correct.
        rotated_recovered = xq_view.to(torch.float32).numpy() * xs_view[0, :M, None].numpy()
        rotated_ref = self._reference(x_np)
        max_diff = float(np.max(np.abs(rotated_recovered - rotated_ref)))
        max_ref = float(np.max(np.abs(rotated_ref)))
        rel_diff = max_diff / max_ref if max_ref > 0 else 0
        self.assertLess(rel_diff, 0.05,
                        f"BLOCK_M={BLOCK_M}, M={M}: rel_diff={rel_diff:.4e}")

    def test_m1_block_m16(self):
        """M=1 (time-embedding shape) — the extreme unaligned case."""
        self._run_with_canary(M=1, BLOCK_M=16)

    def test_m64_block_m128(self):
        """M=64, BLOCK_M=128 — M < BLOCK_M, last tile is entirely OOB."""
        self._run_with_canary(M=64, BLOCK_M=128)

    def test_m300_block_m32(self):
        """M=300 (lyric encoder), BLOCK_M=32 — 300 % 32 = 12 ≠ 0."""
        self._run_with_canary(M=300, BLOCK_M=32)

    def test_m3000_block_m128(self):
        """M=3000 (DiT decoder — the dominant workload), BLOCK_M=128.
        3000 % 128 = 88 ≠ 0 → last tile writes 40 rows past the buffer."""
        self._run_with_canary(M=3000, BLOCK_M=128)

    def test_m3000_block_m64(self):
        """M=3000, BLOCK_M=64 — 3000 % 64 = 48 ≠ 0."""
        self._run_with_canary(M=3000, BLOCK_M=64)


class TestK2GemmDequant(unittest.TestCase):
    """The hybrid_direct K2 kernel matches a NumPy reference for the canonical
    [N, K] W_q layout.  BIAS and NOBIAS are separate compile-time variants;
    their equivalent zero-bias math is validated below.
    """

    def setUp(self):
        np.random.seed(123)
        self.M, self.K, self.N = 32, 256, 64
        self.group_size = 256

    def _reference(self, xq_np, xs_np, wq_N_K, ws_np, bias_np, has_bias):
        M, K = xq_np.shape
        N = wq_N_K.shape[0]
        y = np.zeros((M, N), dtype=np.float32)
        n_groups = K // self.group_size
        for g in range(n_groups):
            ks = g * self.group_size
            ke = ks + self.group_size
            # W_q is [N, K] row-major — same reference math whether we consume
            # it as W.T ([K,N]) or via a real transpose.
            partial = xq_np[:, ks:ke].astype(np.int32) @ wq_N_K[:, ks:ke].T.astype(np.int32)
            y += partial.astype(np.float32) * xs_np[g, :, None] * ws_np[None, :]
        if has_bias:
            y += bias_np[None, :]
        return y

    def _triton_k2(self, xq, xs, wq_N_K, ws, bias, has_bias):
        M, K = xq.shape
        N = wq_N_K.shape[0]
        y = torch.empty((M, N), dtype=torch.float32, device=xq.device)
        _set_k2_config(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=4)

        # HAS_BIAS is compile-time True in the shipped cubin.  For no-bias
        # layers the plugin passes a zero-filled bias vector.
        bias_arg = bias if has_bias else torch.zeros((N,), dtype=torch.float32, device=xq.device)
        # W_q is [N, K] row-major: stride_wn=K, stride_wk=1.
        m.kernel2_gemm_dequant[(1,)](
            xq, xs, wq_N_K, ws, bias_arg, y,
            M, N, K,
            K, 1, 1, M,
            K, 1,
            N, 1,
            GROUP_SIZE=256, HAS_BIAS=has_bias, OUTPUT_FP16=False,
        )
        return y

    def _build_inputs(self):
        xq = torch.from_numpy(
            np.random.randint(-127, 128, size=(self.M, self.K), dtype=np.int8).copy())
        xs = torch.from_numpy(
            (np.random.rand(self.K // self.group_size, self.M) * 0.02 + 0.001).astype(np.float32).copy())
        wq_N_K = torch.from_numpy(
            np.random.randint(-127, 128, size=(self.N, self.K), dtype=np.int8).copy())
        ws = torch.from_numpy(
            (np.random.rand(self.N) * 0.02 + 0.001).astype(np.float32).copy())
        bias = torch.from_numpy(
            (np.random.randn(self.N) * 0.01).astype(np.float32).copy())
        return xq, xs, wq_N_K, ws, bias

    def test_zero_bias_matches_nobias_math(self):
        xq, xs, wq, ws, bias = self._build_inputs()
        y = self._triton_k2(xq, xs, wq, ws, bias, has_bias=False)
        y_ref = self._reference(xq.numpy(), xs.numpy(), wq.numpy(), ws.numpy(), bias.numpy(), False)
        max_diff = float(np.max(np.abs(y.numpy() - y_ref)))
        self.assertLess(max_diff, 1e-3, f"zero-bias/nobias: max_diff={max_diff:.4e}")

    def test_with_bias(self):
        xq, xs, wq, ws, bias = self._build_inputs()
        y = self._triton_k2(xq, xs, wq, ws, bias, has_bias=True)
        y_ref = self._reference(xq.numpy(), xs.numpy(), wq.numpy(), ws.numpy(), bias.numpy(), True)
        max_diff = float(np.max(np.abs(y.numpy() - y_ref)))
        self.assertLess(max_diff, 1e-3, f"with-bias: max_diff={max_diff:.4e}")


class TestK2HostPadding(unittest.TestCase):
    """The production plugin pads/splits M so the plain K2 kernel never needs
    masks.  This test emulates the tail path used by enqueue(): actual rows are
    followed by zero X_q/X_scale rows, K2 writes a padded tile, and only the
    real rows are consumed.
    """

    def test_unaligned_m_tail_tile(self):
        np.random.seed(456)
        M, M_PAD, K, N = 5, 32, 256, 64
        group_size = 256

        xq_np = np.random.randint(-127, 128, size=(M, K), dtype=np.int8)
        xs_np = (np.random.rand(K // group_size, M) * 0.02 + 0.001).astype(np.float32)
        wq_np = np.random.randint(-127, 128, size=(N, K), dtype=np.int8)  # [N, K]
        ws_np = (np.random.rand(N) * 0.02 + 0.001).astype(np.float32)
        bias_np = (np.random.randn(N) * 0.01).astype(np.float32)

        xq_pad_np = np.zeros((M_PAD, K), dtype=np.int8)
        xq_pad_np[:M, :] = xq_np
        xs_pad_np = np.zeros((K // group_size, M_PAD), dtype=np.float32)
        xs_pad_np[:, :M] = xs_np

        xq = torch.from_numpy(xq_pad_np.copy())
        xs = torch.from_numpy(xs_pad_np.copy())
        wq = torch.from_numpy(wq_np.copy())
        ws = torch.from_numpy(ws_np.copy())
        bias = torch.from_numpy(bias_np.copy())
        y_pad = torch.empty((M_PAD, N), dtype=torch.float32, device=xq.device)

        _set_k2_config(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=4)
        # W_q is [N, K] row-major.
        m.kernel2_gemm_dequant[(1,)](
            xq, xs, wq, ws, bias, y_pad,
            M_PAD, N, K,
            K, 1, 1, M_PAD,
            K, 1,
            N, 1,
            GROUP_SIZE=group_size, HAS_BIAS=True, OUTPUT_FP16=False,
        )

        # Reference only for the real rows.  Padded rows should be finite but
        # are intentionally ignored by the plugin copy-back step.
        y_ref = np.zeros((M, N), dtype=np.float32)
        partial = xq_np.astype(np.int32) @ wq_np.T.astype(np.int32)
        y_ref += partial.astype(np.float32) * xs_np[0, :, None] * ws_np[None, :]
        y_ref += bias_np[None, :]
        max_diff = float(np.max(np.abs(y_pad[:M, :].numpy() - y_ref)))
        self.assertLess(max_diff, 1e-3, f"padded-tail: max_diff={max_diff:.4e}")
        self.assertTrue(np.all(np.isfinite(y_pad.numpy())))


class TestOccupancyCompute(unittest.TestCase):
    """Occupancy math must match the CUDA occupancy calculator for the
    canonical (shared, regs, warps, arch) points documented in the K2
    autotune-grid comment (§H2)."""

    def test_sm86_96kb_shared_is_1_cta(self):
        # 96 KB shared on sm86 (100 KB budget) → 1 CTA/SM.  This is the
        # exact scenario that caused the wrong K2 autotune winner before
        # the per-config occupancy fix landed.
        self.assertEqual(
            m._compute_ctas_per_sm(shared_bytes=96 * 1024,
                                   num_regs=160, num_warps=8, arch=86),
            1,
        )

    def test_sm86_48kb_shared_is_2_cta(self):
        # 48 KB shared, 128 regs, W=8 → 2 CTA/SM.
        self.assertEqual(
            m._compute_ctas_per_sm(shared_bytes=48 * 1024,
                                   num_regs=128, num_warps=8, arch=86),
            2,
        )

    def test_sm86_register_bound(self):
        # 255 regs × 256 threads/CTA = 65280 regs/CTA ≈ full 65536 RF → 1 CTA.
        self.assertEqual(
            m._compute_ctas_per_sm(shared_bytes=32 * 1024,
                                   num_regs=255, num_warps=8, arch=86),
            1,
        )

    def test_sm86_capped_by_max_ctas(self):
        # Tiny shared and tiny reg use → still capped at max_ctas_per_sm.
        occ = m._compute_ctas_per_sm(shared_bytes=1024, num_regs=16,
                                     num_warps=1, arch=86)
        self.assertLessEqual(occ, 16)   # sm86 hard cap
        self.assertGreaterEqual(occ, 1)

    def test_missing_regs_falls_back_safely(self):
        # cuobjdump-missing case: num_regs=0 → we still return a positive CTA
        # count (bounded by smem and the arch's max-CTA limit).
        occ = m._compute_ctas_per_sm(shared_bytes=32 * 1024, num_regs=0,
                                     num_warps=8, arch=86)
        self.assertGreaterEqual(occ, 1)


class TestK1TargetDispatchAndGrid(unittest.TestCase):
    """K1 has one enabled sm80-sm89 family and strict placeholders."""

    def setUp(self):
        m._K1_CFG_OCCUPANCY.clear()

    def tearDown(self):
        m._K1_CFG_OCCUPANCY.clear()

    def test_sm86_candidate_family_is_numerically_locked(self):
        cfgs = m._estimate_k1_configs(
            20, 2 * 1024 * 1024, 100 * 1024, m.AUTOTUNE_SHAPES_K2,
            target_backend="cuda", target_arch=86)
        self.assertEqual(len(cfgs), 5)
        self.assertEqual({c.kwargs["BLOCK_M"] for c in cfgs}, {16, 32, 64})
        self.assertTrue(all(c.num_warps == 4 for c in cfgs))
        self.assertTrue(all(c.num_stages == 1 for c in cfgs))
        self.assertEqual(
            sum(c.kwargs["BLOCK_M"] == 16 for c in cfgs), 1,
            "BM16 capped/uncapped forms are resource duplicates",
        )

    def test_strict_placeholders_have_no_fallback(self):
        with self.assertRaises(SystemExit):
            m._estimate_k1_configs(20, 0, 0, [], "cuda", 75)
        with self.assertRaises(SystemExit):
            m._estimate_k1_configs(96, 0, 0, [], "hip", "gfx1100")
        with self.assertRaises(SystemExit):
            m._estimate_k1_configs(1, 0, 0, [], "cuda", 90)

    def test_spill_signature_matches_shipping_specialization(self):
        label, signature, constexprs, attrs = next(m._spill_check_variants("k1"))
        self.assertEqual(label, "FP32IO")
        self.assertEqual(constexprs["stride_xk"], 1)
        self.assertEqual(constexprs["stride_xqk"], 1)
        self.assertEqual(constexprs["stride_xsm"], 1)
        # Runtime M (argument index 3) must remain generic for M=300/3000.
        self.assertNotIn((3,), attrs)
        for index in (0, 1, 2, 4, 5, 7, 10):
            self.assertIn((index,), attrs)
        self.assertEqual(signature["M"], "i32")

    def test_k1_grid_uses_per_config_occupancy(self):
        cfg = m.triton.Config({"BLOCK_M": 32}, num_warps=4,
                              num_stages=1, maxnreg=128)
        m._K1_CFG_OCCUPANCY[m._cfg_occupancy_key(cfg)] = 4
        meta = {"BLOCK_M": 32, "num_warps": 4, "num_stages": 1,
                "maxnreg": 128}
        self.assertEqual(m._autotune_grid_x_for_k1(1000, 20, meta), 80)
        self.assertEqual(m._autotune_grid_x_for_k1(10, 20, meta), 10)


class TestAutotuneGridForK2(unittest.TestCase):
    """The K2 grid lambda must launch ``num_sms * per_cfg_occupancy`` CTAs
    (capped at num_tiles), reading occupancy from ``_K2_CFG_OCCUPANCY``."""

    def setUp(self):
        m._K2_CFG_OCCUPANCY.clear()

    def tearDown(self):
        m._K2_CFG_OCCUPANCY.clear()

    def test_known_cfg_1_cta_per_sm(self):
        cfg = m.triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 2},
            num_warps=8, num_stages=2)
        m._K2_CFG_OCCUPANCY[m._cfg_occupancy_key(cfg)] = 1
        meta = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 2,
                "num_warps": 8, "num_stages": 2}
        # 1800 tiles, 32 SMs, cfg = 1 CTA/SM → launch 32 CTAs.
        self.assertEqual(m._autotune_grid_x_for_k2(1800, 32, meta), 32)

    def test_known_cfg_2_cta_per_sm(self):
        cfg = m.triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
            num_warps=8, num_stages=3)
        m._K2_CFG_OCCUPANCY[m._cfg_occupancy_key(cfg)] = 2
        meta = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4,
                "num_warps": 8, "num_stages": 3}
        # 1800 tiles, 32 SMs, cfg = 2 CTA/SM → launch 64 CTAs.
        self.assertEqual(m._autotune_grid_x_for_k2(1800, 32, meta), 64)

    def test_unseen_cfg_falls_back(self):
        meta = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4,
                "num_warps": 4, "num_stages": 2}
        # Nothing in the map → fallback to AUTOTUNE_OCCUPANCY_ASSUMPTION.
        self.assertEqual(
            m._autotune_grid_x_for_k2(1800, 32, meta),
            32 * m.AUTOTUNE_OCCUPANCY_ASSUMPTION,
        )

    def test_num_tiles_clamp(self):
        cfg = m.triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 2},
            num_warps=8, num_stages=2)
        m._K2_CFG_OCCUPANCY[m._cfg_occupancy_key(cfg)] = 1
        meta = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 2,
                "num_warps": 8, "num_stages": 2}
        # Fewer tiles than SMs → launch num_tiles.
        self.assertEqual(m._autotune_grid_x_for_k2(10, 32, meta), 10)


class TestEndToEndPipeline(unittest.TestCase):
    """K1 (rotate+quant) → K2 (GEMM+dequant) ≈ x @ W^T + bias."""

    def test_pipeline_matches_fp32_reference(self):
        np.random.seed(2025)
        M, K, N = 32, 256, 64
        group_size = 256

        # Build the offline-rotated + quantized weight (canonical [N, K]).
        w_fp32 = (np.random.randn(N, K) * 0.05).astype(np.float32)
        H = build_hadamard(group_size)
        w_rot = rotate_weight(w_fp32, H, group_size)
        w_scale = np.maximum(np.max(np.abs(w_rot), axis=1) / 127.0, 1e-10)
        w_q_int8 = np.clip(np.round(w_rot / w_scale[:, None]),
                           -127, 127).astype(np.int8)
        bias = (np.random.randn(N) * 0.01).astype(np.float32)
        x_fp32 = (np.random.randn(M, K) * 0.5).astype(np.float32)

        # FP32 reference (no rotation, no quant).
        y_ref = x_fp32 @ w_fp32.T + bias[None, :]

        # Triton pipeline.
        x_t = torch.from_numpy(x_fp32.copy())
        wq_t = torch.from_numpy(w_q_int8.copy())    # [N, K]
        ws_t = torch.from_numpy(w_scale.copy())
        bias_t = torch.from_numpy(bias.copy())

        # K1
        xq = torch.empty((M, K), dtype=torch.int8, device=x_t.device)
        xs = torch.empty((K // group_size, M), dtype=torch.float32, device=x_t.device)
        _set_k1_config(BLOCK_M=32)
        m.kernel1_convrot_quant[(1,)](
            x_t, xq, xs,
            M, K,
            K, 1, K, 1, 1, M,
            GROUP_SIZE=group_size, INPUT_FP16=False,
            CONTIG_XK=True,
        )

        # K2 — hybrid_direct consumes W_q [N, K] via tl.trans(wq).
        y = torch.empty((M, N), dtype=torch.float32, device=x_t.device)
        _set_k2_config(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=4)
        m.kernel2_gemm_dequant[(1,)](
            xq, xs, wq_t, ws_t, bias_t, y,
            M, N, K,
            K, 1, 1, M,
            K, 1,   # W_q [N, K] row-major: wn=K, wk=1
            N, 1,
            GROUP_SIZE=group_size, HAS_BIAS=True, OUTPUT_FP16=False,
        )

        y_np = y.numpy()
        max_abs_err = float(np.max(np.abs(y_np - y_ref)))
        max_ref = float(np.max(np.abs(y_ref)))
        rel_err = max_abs_err / max_ref
        self.assertLess(rel_err, 0.05,
                        f"end-to-end rel_err={rel_err:.4e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
