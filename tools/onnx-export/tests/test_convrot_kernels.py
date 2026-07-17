#!/usr/bin/env python3
"""Unit tests for K1 (v13) and the retained Triton K2 numerical oracle.

The shipping K2 is Gluon and is compiled/bitwise-gated on the target GPU. These
CPU tests use Triton's interpreter mode (``TRITON_INTERPRET=1``) to validate the
oracle, host padding, and FP16IO arithmetic without a CUDA device.  v13 changes:

  * K1 now fuses in-register H256 rotation + per-row INT8 quant.
    X_scale is [M] FP32 (one scale per activation row, NOT per 256-block).
  * K2 is simplified to a single full-K INT8×INT8 → INT32 matmul plus an
    outer-product FP32 dequant (xs[m] * ws[n] * acc).  The per-group INT32
    fold of v7-v12 is gone; stride_xsg is removed from the launch ABI.

These tests complement the version-13 extraction-time GPU gate.  They do not
exercise Gluon execution or the autotune search, which require a real GPU.

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
from unittest import mock
from pathlib import Path

import numpy as np
import torch

# Make the onnx-export package importable.
TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from convrot import build_hadamard, rotate_weight, rotate_activation
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


def _k2_reference_meta(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
                       SCHEDULE_2D=False):
    """Compile-time knobs for the interpreter-only v13 numerical oracle."""
    return {
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "GROUP_M": GROUP_M, "SCHEDULE_2D": SCHEDULE_2D,
        "DYNAMIC_INNER": False,
    }


def _full_k1_reference(x_np, H256r, group_size):
    """NumPy reference for v13 K1: H256 rotation + per-row quant.

    Returns (x_q, x_scale) where x_q is [M, K] int8 and x_scale is [M] fp32.
    """
    M, K = x_np.shape
    # H256 rotation (per group of 256).
    G = K // group_size
    x_grouped = x_np.reshape(M, G, group_size)
    x_rot = np.matmul(x_grouped, H256r).reshape(M, K).astype(np.float32)
    # Per-row scale + INT8 quant.
    row_max = np.max(np.abs(x_rot), axis=1)
    row_scale = np.maximum(row_max, np.float32(1e-30)) / np.float32(127.0)
    x_q = np.clip(np.rint(x_rot / row_scale[:, None]),
                  -127, 127).astype(np.int8)
    return x_q, row_scale


# ─── Tests ────────────────────────────────────────────────────────────────


class TestK1DirectRotation(unittest.TestCase):
    """The K1 (v13) H256 rotation + per-row quant must match the NumPy reference.

    With G=1 (K=256), only H256 + per-row quant applies.
    We test BLOCK_M=1 (the v13 default for small K).
    """

    def setUp(self):
        np.random.seed(42)
        self.H256r = _build_h256r()

    def _triton_k1(self, x_np, BLOCK_M):
        x_torch = torch.from_numpy(x_np.copy())
        M, K = x_torch.shape
        G = K // 256
        xq = torch.empty((M, K), dtype=torch.int8, device=x_torch.device)
        xs = torch.empty((M,), dtype=torch.float32, device=x_torch.device)
        _set_k1_config(BLOCK_M)
        m.kernel1_convrot_quant[(1,)](
            x_torch, xq, xs,
            M, K, G,
            K, 1,            # stride_xm, stride_xk
            K, 1,            # stride_xqm, stride_xqk
            1,               # stride_xsm
            GROUP_SIZE=256,
            INPUT_FP16=False,
            CONTIG_XK=True,
        )
        return xq, xs

    def test_block_m_1(self):
        # K=256, G=1 — H256 + per-row quant.
        BLOCK_M = 1
        x_np = (np.random.randn(BLOCK_M, 256) * 0.1).astype(np.float32)
        xq, xs = self._triton_k1(x_np, BLOCK_M)
        xq_ref, xs_ref = _full_k1_reference(x_np, self.H256r, 256)
        np.testing.assert_array_equal(xq.numpy(), xq_ref)
        np.testing.assert_allclose(xs.numpy(), xs_ref, rtol=1e-6, atol=1e-7)

    def test_block_m_2(self):
        BLOCK_M = 2
        x_np = (np.random.randn(BLOCK_M, 256) * 0.1).astype(np.float32)
        xq, xs = self._triton_k1(x_np, BLOCK_M)
        xq_ref, xs_ref = _full_k1_reference(x_np, self.H256r, 256)
        np.testing.assert_array_equal(xq.numpy(), xq_ref)
        np.testing.assert_allclose(xs.numpy(), xs_ref, rtol=1e-6, atol=1e-7)


class TestK1MultiGroup(unittest.TestCase):
    """K1 with G>=4 performs per-group H256 rotations."""

    def setUp(self):
        np.random.seed(7)
        self.H256r = _build_h256r()

    def test_k1024_g4(self):
        # K=1024, G=4 — per-group H256 rotations + per-row quant.
        BLOCK_M = 1
        K = 1024
        x_np = (np.random.randn(BLOCK_M, K) * 0.1).astype(np.float32)
        xq, xs = self._triton_k1(x_np, BLOCK_M, K)
        xq_ref, xs_ref = _full_k1_reference(x_np, self.H256r, 256)
        np.testing.assert_array_equal(xq.numpy(), xq_ref)
        np.testing.assert_allclose(xs.numpy(), xs_ref, rtol=1e-6, atol=1e-7)

    def _triton_k1(self, x_np, BLOCK_M, K):
        x_torch = torch.from_numpy(x_np.copy())
        M = x_np.shape[0]
        G = K // 256
        xq = torch.empty((M, K), dtype=torch.int8, device=x_torch.device)
        xs = torch.empty((M,), dtype=torch.float32, device=x_torch.device)
        _set_k1_config(BLOCK_M)
        m.kernel1_convrot_quant[(1,)](
            x_torch, xq, xs,
            M, K, G,
            K, 1, K, 1, 1,
            GROUP_SIZE=256,
            INPUT_FP16=False,
            CONTIG_XK=True,
        )
        return xq, xs


class TestK1UnalignedM(unittest.TestCase):
    """K1 must NOT write out of bounds when M is not a multiple of BLOCK_M.

    These tests use canary bytes (0xDE pattern) placed immediately after the
    workspace buffers and verify they remain untouched after the kernel runs.
    """

    def setUp(self):
        np.random.seed(999)
        self.H256r = _build_h256r()

    def _run_with_canary(self, M, BLOCK_M):
        K = 256
        x_np = (np.random.randn(M, K) * 0.1).astype(np.float32)

        # Allocate workspace outputs with canary padding after the valid region.
        CANARY_INT8 = -34  # 0xDE in two's complement
        canary_rows = max(BLOCK_M, 4)
        xq_padded = np.full((M + canary_rows, K), CANARY_INT8, dtype=np.int8)
        xq_full = torch.from_numpy(xq_padded.copy())
        xq_view = xq_full[:M, :]

        # X_scale: [M + canary_cols] fp32 + canary (NaN).
        canary_cols = canary_rows
        xs_padded = np.full((M + canary_cols,), float('nan'), dtype=np.float32)
        xs_full = torch.from_numpy(xs_padded.copy())
        xs_view = xs_full[:M]

        x_torch = torch.from_numpy(x_np.copy())
        G = K // 256
        _set_k1_config(BLOCK_M)
        m.kernel1_convrot_quant[(1,)](
            x_torch, xq_view, xs_view,
            M, K, G,
            K, 1, K, 1, 1,
            GROUP_SIZE=256,
            INPUT_FP16=False,
            CONTIG_XK=True,
        )

        # Verify canary bytes are untouched (no OOB write).
        xq_canary = xq_full[M:, :].numpy()
        self.assertTrue(np.all(xq_canary == CANARY_INT8),
                        f"BLOCK_M={BLOCK_M}, M={M}: X_q canary corrupted!")

        xs_canary = xs_full[M:].numpy()
        self.assertTrue(np.all(np.isnan(xs_canary)),
                        f"BLOCK_M={BLOCK_M}, M={M}: X_scale canary corrupted!")

        # Also verify the in-bounds values are correct.
        xq_ref, xs_ref = _full_k1_reference(x_np, self.H256r, 256)
        np.testing.assert_array_equal(xq_view.numpy(), xq_ref)
        np.testing.assert_allclose(xs_view.numpy(), xs_ref, rtol=1e-6, atol=1e-7)

    def test_m1_block_m1(self):
        """M=1, BLOCK_M=1 — exact fit."""
        self._run_with_canary(M=1, BLOCK_M=1)

    def test_m3_block_m2(self):
        """M=3, BLOCK_M=2 — last tile writes 1 row past the buffer."""
        self._run_with_canary(M=3, BLOCK_M=2)

    def test_m5_block_m2(self):
        """M=5, BLOCK_M=2 — last tile writes 1 row past."""
        self._run_with_canary(M=5, BLOCK_M=2)


class TestK2GemmDequant(unittest.TestCase):
    """The K2 (v13) oracle matches NumPy for the per-row X_scale contract.

    BIAS and NOBIAS are separate compile-time variants, as in shipping Gluon.
    """

    def setUp(self):
        np.random.seed(123)
        self.M, self.K, self.N = 32, 256, 64
        self.group_size = 256

    def _reference(self, xq_np, xs_row_np, wq_N_K, ws_np, bias_np, has_bias):
        """v13 reference: single full-K INT32 matmul + per-row outer-product dequant."""
        M, K = xq_np.shape
        N = wq_N_K.shape[0]
        dot = xq_np.astype(np.int32) @ wq_N_K.T.astype(np.int32)  # [M, N] int32
        y = (xs_row_np.astype(np.float32)[:, None]
             * ws_np.astype(np.float32)[None, :]
             * dot.astype(np.float32))
        if has_bias:
            y += bias_np[None, :]
        return y

    def _triton_k2(self, xq, xs, wq_N_K, ws, bias, has_bias,
                   schedule_2d=False):
        M, K = xq.shape
        N = wq_N_K.shape[0]
        y = torch.empty((M, N), dtype=torch.float16, device=xq.device)
        meta = _k2_reference_meta(
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=4,
            SCHEDULE_2D=schedule_2d)

        bias_arg = bias if has_bias else torch.empty(
            (1,), dtype=torch.float32, device=xq.device)
        # v13: no stride_xsg (X_scale is [M] FP32, stride_xsm=1 only).
        m.kernel2_gemm_dequant_reference[(1,)](
            xq, xs, wq_N_K, ws, bias_arg, y,
            M, N, K,
            K, 1,
            1,
            K, 1,
            N, 1,
            GROUP_SIZE=256, HAS_BIAS=has_bias, OUTPUT_FP16=True,
            **meta,
        )
        return y

    def _build_inputs(self):
        xq = torch.from_numpy(
            np.random.randint(-127, 128, size=(self.M, self.K), dtype=np.int8).copy())
        # v13: X_scale is [M] FP32 (per-row), NOT [G, M].
        xs = torch.from_numpy(
            (np.random.rand(self.M) * 0.02 + 0.001).astype(np.float32).copy())
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
        y_ref = self._reference(
            xq.numpy(), xs.numpy(), wq.numpy(), ws.numpy(), bias.numpy(), False
        ).astype(np.float16)
        max_diff = float(np.max(np.abs(y.numpy() - y_ref)))
        self.assertLess(max_diff, 1e-3, f"zero-bias/nobias: max_diff={max_diff:.4e}")

    def test_with_bias(self):
        xq, xs, wq, ws, bias = self._build_inputs()
        y = self._triton_k2(xq, xs, wq, ws, bias, has_bias=True)
        y_ref = self._reference(
            xq.numpy(), xs.numpy(), wq.numpy(), ws.numpy(), bias.numpy(), True
        ).astype(np.float16)
        max_diff = float(np.max(np.abs(y.numpy() - y_ref)))
        self.assertLess(max_diff, 1e-3, f"with-bias: max_diff={max_diff:.4e}")

    def test_2d_scheduler_is_bitwise_identical(self):
        xq, xs, wq, ws, bias = self._build_inputs()
        y_1d = self._triton_k2(xq, xs, wq, ws, bias, has_bias=True,
                               schedule_2d=False)
        y_2d = self._triton_k2(xq, xs, wq, ws, bias, has_bias=True,
                               schedule_2d=True)
        self.assertTrue(torch.equal(y_1d.view(torch.int32),
                                    y_2d.view(torch.int32)))


class TestK2OraclePadding(unittest.TestCase):
    """The v13 oracle still uses padding for bitwise comparison.

    Shipping Gluon K2 masks the final tile and is compile-checked separately.
    """

    def test_unaligned_m_tail_tile(self):
        np.random.seed(456)
        M, M_PAD, K, N = 5, 32, 256, 64

        xq_np = np.random.randint(-127, 128, size=(M, K), dtype=np.int8)
        # v13: X_scale is [M] fp32 (per-row).
        xs_np = (np.random.rand(M) * 0.02 + 0.001).astype(np.float32)
        wq_np = np.random.randint(-127, 128, size=(N, K), dtype=np.int8)  # [N, K]
        ws_np = (np.random.rand(N) * 0.02 + 0.001).astype(np.float32)
        bias_np = (np.random.randn(N) * 0.01).astype(np.float32)

        xq_pad_np = np.zeros((M_PAD, K), dtype=np.int8)
        xq_pad_np[:M, :] = xq_np
        xs_pad_np = np.zeros((M_PAD,), dtype=np.float32)
        xs_pad_np[:M] = xs_np

        xq = torch.from_numpy(xq_pad_np.copy())
        xs = torch.from_numpy(xs_pad_np.copy())
        wq = torch.from_numpy(wq_np.copy())
        ws = torch.from_numpy(ws_np.copy())
        bias = torch.from_numpy(bias_np.copy())
        y_pad = torch.empty((M_PAD, N), dtype=torch.float16, device=xq.device)

        meta = _k2_reference_meta(
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=4)
        # v13: no stride_xsg.
        m.kernel2_gemm_dequant_reference[(1,)](
            xq, xs, wq, ws, bias, y_pad,
            M_PAD, N, K,
            K, 1,
            1,
            K, 1,
            N, 1,
            GROUP_SIZE=256, HAS_BIAS=True, OUTPUT_FP16=True,
            **meta,
        )

        # Reference only for the real rows.
        dot = xq_np.astype(np.int32) @ wq_np.T.astype(np.int32)
        y_ref = (xs_np[:, None].astype(np.float32)
                 * ws_np[None, :].astype(np.float32)
                 * dot.astype(np.float32))
        y_ref += bias_np[None, :]
        y_ref = y_ref.astype(np.float16)
        max_diff = float(np.max(np.abs(y_pad[:M, :].numpy() - y_ref)))
        self.assertLess(max_diff, 1e-3, f"padded-tail: max_diff={max_diff:.4e}")
        self.assertTrue(np.all(np.isfinite(y_pad.numpy())))


class TestOccupancyCompute(unittest.TestCase):
    """Occupancy math must match the CUDA occupancy calculator for the
    canonical (shared, regs, warps, arch) points documented in the K2
    autotune-grid comment (§H2)."""

    def test_sm86_96kb_shared_is_1_cta(self):
        self.assertEqual(
            m._compute_ctas_per_sm(shared_bytes=96 * 1024,
                                   num_regs=160, num_warps=8, arch=86),
            1,
        )

    def test_sm86_48kb_shared_is_2_cta(self):
        self.assertEqual(
            m._compute_ctas_per_sm(shared_bytes=48 * 1024,
                                   num_regs=128, num_warps=8, arch=86),
            2,
        )

    def test_sm86_register_bound(self):
        self.assertEqual(
            m._compute_ctas_per_sm(shared_bytes=32 * 1024,
                                   num_regs=255, num_warps=8, arch=86),
            1,
        )

    def test_sm86_capped_by_max_ctas(self):
        occ = m._compute_ctas_per_sm(shared_bytes=1024, num_regs=16,
                                     num_warps=1, arch=86)
        self.assertLessEqual(occ, 16)
        self.assertGreaterEqual(occ, 1)

    def test_missing_regs_falls_back_safely(self):
        occ = m._compute_ctas_per_sm(shared_bytes=32 * 1024, num_regs=0,
                                     num_warps=8, arch=86)
        self.assertGreaterEqual(occ, 1)


class TestK1TargetDispatchAndGrid(unittest.TestCase):
    """K1 (v13) has one enabled sm80-sm89 family (BLOCK_M ∈ {1, 2})."""

    def setUp(self):
        m._K1_CFG_OCCUPANCY.clear()

    def tearDown(self):
        m._K1_CFG_OCCUPANCY.clear()

    def test_sm86_candidate_family_is_v13_block_m_1_2(self):
        cfgs = m._estimate_k1_configs(
            20, 2 * 1024 * 1024, 100 * 1024, m.AUTOTUNE_SHAPES_K2,
            target_backend="cuda", target_arch=86)
        # v13: BLOCK_M ∈ {1, 2}, each with (maxnreg=128, maxnreg=None).
        self.assertEqual(len(cfgs), 4)
        self.assertEqual({c.kwargs["BLOCK_M"] for c in cfgs}, {1, 2})
        self.assertTrue(all(c.num_warps == 4 for c in cfgs))
        self.assertTrue(all(c.num_stages == 1 for c in cfgs))

    def test_strict_placeholders_have_no_fallback(self):
        with self.assertRaises(SystemExit):
            m._estimate_k1_configs(20, 0, 0, [], "cuda", 75)
        with self.assertRaises(SystemExit):
            m._estimate_k1_configs(96, 0, 0, [], "hip", "gfx1100")
        with self.assertRaises(SystemExit):
            m._estimate_k1_configs(1, 0, 0, [], "cuda", 90)

    def test_spill_signature_matches_v13_shipping_specialization(self):
        label, signature, constexprs, attrs = next(m._spill_check_variants("k1"))
        self.assertEqual(label, "FP16IO")
        self.assertTrue(constexprs["INPUT_FP16"])
        self.assertEqual(m.DTYPE_CONFIGS, [("FP16IO", True, True)])
        self.assertEqual(constexprs["stride_xk"], 1)
        self.assertEqual(constexprs["stride_xqk"], 1)
        self.assertEqual(constexprs["stride_xsm"], 1)
        self.assertNotIn("stride_xsg", signature)
        self.assertNotIn("stride_xsg", constexprs)
        self.assertNotIn("WORKSPACE_ptr", signature)
        self.assertNotIn("BF_GROUPS_ptr", signature)
        self.assertIn("G", signature)
        self.assertEqual(signature["M"], "i32")

    def test_k1_grid_uses_per_config_occupancy(self):
        cfg = m.triton.Config({"BLOCK_M": 1}, num_warps=4,
                              num_stages=1, maxnreg=128)
        m._K1_CFG_OCCUPANCY[m._cfg_occupancy_key(cfg)] = 4
        meta = {"BLOCK_M": 1, "num_warps": 4, "num_stages": 1,
                "maxnreg": 128}
        # v13: K1 grid is cdiv(M, BLOCK_M) — no per-group multiplier.
        self.assertEqual(m._autotune_grid_x_for_k1(1000, 20, meta), 80)
        self.assertEqual(m._autotune_grid_x_for_k1(10, 20, meta), 10)


class TestGluonK2ConfigSpace(unittest.TestCase):
    def test_contains_fp16_only_inventory(self):
        cfgs = m._estimate_k2_configs(
            20, 2 * 1024 * 1024, 100 * 1024,
            m.AUTOTUNE_SHAPES_K2, arch=86)
        self.assertEqual(m.DTYPE_CONFIGS, [("FP16IO", True, True)])
        self.assertTrue(all(c.kwargs["NUM_BUFFERS"] == c.num_stages
                            for c in cfgs))
        self.assertTrue(all(c.kwargs["BLOCK_M"] <= m.K2_AUTOTUNE_M_ALIGNMENT
                            for c in cfgs))
        self.assertTrue(any(c.kwargs["BLOCK_K"] == 128 for c in cfgs))

    def test_v13_k2_signature_drops_stride_xsg(self):
        _, signature, constexprs, attrs = next(m._spill_check_variants("k2"))
        self.assertNotIn("stride_xsg", signature)
        self.assertNotIn("stride_xsg", constexprs)
        # M (argument index 6) is not 16-divisible.
        self.assertNotIn((6,), attrs)

    def test_shipping_benchmark_is_one_real_m_launch(self):
        cfg = next(c for c in m._estimate_k2_configs(
            20, 2 * 1024 * 1024, 100 * 1024,
            m.AUTOTUNE_SHAPES_K2, arch=86)
            if c.kwargs["BLOCK_M"] == 128)
        pack = {
            "xq": object(), "xs_by_pitch": {3000: object()},
            "wq": object(), "ws": object(), "bias": object(),
            "y": object(), "y_tail": object(),
        }
        with mock.patch.object(m, "_launch_k2_for_bench", return_value=20) as launch:
            self.assertEqual(m._launch_k2_production_sequence(
                cfg, pack, 3000, 4096, 2560, 20, True), 20)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.args[2:5], (3000, 4096, 2560))
        # v13: no stride_xsg kwarg.
        self.assertNotIn("stride_xsg", launch.call_args.kwargs)


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
        self.assertEqual(m._autotune_grid_x_for_k2(1800, 32, meta), 32)

    def test_known_cfg_2_cta_per_sm(self):
        cfg = m.triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
            num_warps=8, num_stages=3)
        m._K2_CFG_OCCUPANCY[m._cfg_occupancy_key(cfg)] = 2
        meta = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4,
                "num_warps": 8, "num_stages": 3}
        self.assertEqual(m._autotune_grid_x_for_k2(1800, 32, meta), 64)

    def test_unseen_cfg_falls_back(self):
        meta = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4,
                "num_warps": 4, "num_stages": 2}
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
        self.assertEqual(m._autotune_grid_x_for_k2(10, 32, meta), 10)


class TestEndToEndPipeline(unittest.TestCase):
    """K1 (v13: rotate + per-row quant) → K2 (per-row dequant)
    ≈ x @ W^T + bias."""

    def test_pipeline_matches_fp32_reference(self):
        np.random.seed(2025)
        M, K, N = 32, 256, 64
        group_size = 256

        # Build the offline-rotated + quantized weight.
        w_fp32 = (np.random.randn(N, K) * 0.05).astype(np.float32)
        H = build_hadamard(group_size)
        # v13 offline weight prep: H256 + per-row HQQ INT8.
        from convrot import rotate_weight
        w_rot = rotate_weight(w_fp32, H, group_size)
        w_scale = np.maximum(np.max(np.abs(w_rot), axis=1) / 127.0, 1e-10)
        w_q_int8 = np.clip(np.round(w_rot / w_scale[:, None]),
                           -127, 127).astype(np.int8)
        bias = (np.random.randn(N) * 0.01).astype(np.float32)
        x_fp32 = (np.random.randn(M, K) * 0.5).astype(np.float32)

        # Shipping boundary is FP16; compare against the same rounded input.
        x_fp16 = x_fp32.astype(np.float16)
        y_ref = x_fp16.astype(np.float32) @ w_fp32.T + bias[None, :]

        # Triton reference pipeline (the GPU gate compares Gluon bitwise).
        x_t = torch.from_numpy(x_fp16.copy())
        wq_t = torch.from_numpy(w_q_int8.copy())    # [N, K]
        ws_t = torch.from_numpy(w_scale.copy())
        bias_t = torch.from_numpy(bias.copy())

        # K1 (v13): rotate + per-row quant.
        xq = torch.empty((M, K), dtype=torch.int8, device=x_t.device)
        xs = torch.empty((M,), dtype=torch.float32, device=x_t.device)
        G = K // group_size
        _set_k1_config(BLOCK_M=1)
        m.kernel1_convrot_quant[(1,)](
            x_t, xq, xs,
            M, K, G,
            K, 1, K, 1, 1,
            GROUP_SIZE=group_size,
            INPUT_FP16=True,
            CONTIG_XK=True,
        )

        # K2 (v13) oracle: single full-K INT8 GEMM + per-row dequant.
        y = torch.empty((M, N), dtype=torch.float16, device=x_t.device)
        meta = _k2_reference_meta(
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=4)
        m.kernel2_gemm_dequant_reference[(1,)](
            xq, xs, wq_t, ws_t, bias_t, y,
            M, N, K,
            K, 1,
            1,
            K, 1,   # W_q [N, K] row-major: wn=K, wk=1
            N, 1,
            GROUP_SIZE=group_size, HAS_BIAS=True, OUTPUT_FP16=True,
            **meta,
        )

        y_np = y.to(torch.float32).numpy()
        max_abs_err = float(np.max(np.abs(y_np - y_ref)))
        max_ref = float(np.max(np.abs(y_ref)))
        rel_err = max_abs_err / max_ref
        self.assertLess(rel_err, 0.05,
                        f"end-to-end rel_err={rel_err:.4e}")


class TestRotationInvariance(unittest.TestCase):
    """H256 rotation on both W and X preserves x @ W^T."""

    def test_rotation_pair_preserves_matmul(self):
        np.random.seed(11)
        K, N, M = 256 * 4, 8, 3
        W = (np.random.randn(N, K) * 0.1).astype(np.float32)
        X = (np.random.randn(M, K) * 0.5).astype(np.float32)
        H = build_hadamard(256)

        W_rot = rotate_weight(W, H, 256)

        # Apply H256 rotation to X online.
        G = K // 256
        X_grouped = X.reshape(M, G, 256)
        X_rot = np.matmul(X_grouped, H).reshape(M, K).astype(np.float32)

        ref = X @ W.T
        rot = X_rot @ W_rot.T
        max_diff = float(np.max(np.abs(ref - rot)))
        self.assertLess(max_diff, 1e-3,
                        f"rotation pair max_abs_diff={max_diff:.4e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
