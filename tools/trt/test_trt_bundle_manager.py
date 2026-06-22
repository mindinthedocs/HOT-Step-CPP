#!/usr/bin/env python3
"""Behavior tests for the TRT-bundle orchestrator.

Run: ``python -m unittest tools.trt.test_trt_bundle_manager`` (or, from this
dir, ``python -m unittest test_trt_bundle_manager``).

Proven behaviors:
  - the DiT component registry is well-formed (dit + cond_enc are MVP; text_enc
    is registered but NOT in the DiT MVP — the GGUF Qwen3 is the text encoder at
    runtime, or a standalone Qwen3-emb bundle can be built separately);
  - cond_enc no longer declares a text_enc dependency (it loads weights from the
    DiT safetensors);
  - a DiT bundle ``bundle`` dry-run produces the correct ordered step list (no
    text_enc steps);
  - the DiT bundle manifest writer omits text_enc — the C++ reader treats it as
    a DiT-only bundle;
  - ``build_embedding_plan`` produces a standalone Qwen3-emb plan with only
    text_enc steps;
  - the embedding manifest writer emits only text_enc;
  - resume skips a step whose outputs already exist, and a corrupted resume
    check (a missing output) makes a previously-done step run again.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trt_bundle_manager as tbm


def _dit_plan(tmp: Path) -> list[tbm.Step]:
    return tbm.build_plan(
        "q8map-fp16",
        tmp / "bundle",
        tmp / "dit-src",
    )


def _embedding_plan(tmp: Path) -> list[tbm.Step]:
    return tbm.build_embedding_plan(
        tmp / "emb-bundle",
        tmp / "text-src",
    )


class RegistryTests(unittest.TestCase):
    def test_dit_mvp_components(self) -> None:
        # DiT MVP is dit + cond_enc only. text_enc is registered for the
        # standalone embedding bundle but is NOT in the DiT MVP.
        self.assertEqual(set(tbm.MVP_COMPONENTS), {"dit", "cond_enc"})
        for name in tbm.MVP_COMPONENTS:
            self.assertIn(name, tbm.REGISTRY)

    def test_embedding_components(self) -> None:
        self.assertEqual(set(tbm.EMBEDDING_COMPONENTS), {"text_enc"})

    def test_text_enc_registered(self) -> None:
        self.assertIn("text_enc", tbm.REGISTRY)

    def test_dit_precision_recipes(self) -> None:
        dit = tbm.REGISTRY["dit"]
        self.assertEqual(dit.precision_options, ("q8map-fp16", "w8a8", "fp32"))
        self.assertEqual(dit.default_precision, "q8map-fp16")

    def test_encoders_are_bf16(self) -> None:
        for name in ("text_enc", "cond_enc"):
            self.assertEqual(tbm.REGISTRY[name].default_precision, "bf16")

    def test_cond_enc_does_not_depend_on_text_enc(self) -> None:
        self.assertEqual(tbm.REGISTRY["cond_enc"].depends_on, ())

    def test_cond_enc_owns_null_condition_emb(self) -> None:
        self.assertIn("null_condition_emb.bin", tbm.REGISTRY["cond_enc"].sidecar_files)
        self.assertNotIn("null_condition_emb.bin", tbm.REGISTRY["text_enc"].sidecar_files)

    def test_text_enc_owns_embed_tokens(self) -> None:
        self.assertIn("embed_tokens.bin", tbm.REGISTRY["text_enc"].sidecar_files)

    def test_bad_default_precision_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tbm.ComponentSpec(
                name="x", display_name="x", export_script="x.py",
                build_flag_weight_stream=False,
                precision_options=("fp16",), default_precision="fp32",
                sidecar_files=(), onnx_dynamic_shapes=False,
            )


class DitPlanOrderTests(unittest.TestCase):
    def test_ordered_step_names(self) -> None:
        # No text_enc steps in a DiT bundle.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            names = [s.name for s in _dit_plan(Path(d))]
        self.assertEqual(
            names,
            [
                "download",
                "prepare-dit",
                "export-cond_enc",
                "export-dit",
                "build-cond_enc",
                "build-dit",
                "fsq-sidecar",
                "collect-sidecars",
                "manifest",
            ],
        )

    def test_no_text_enc_steps_in_dit_plan(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = _dit_plan(Path(d))
        for s in steps:
            self.assertNotIn("text_enc", s.name)

    def test_export_precedes_build_precedes_manifest(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            names = [s.name for s in _dit_plan(Path(d))]
        self.assertLess(names.index("export-dit"), names.index("build-dit"))
        self.assertLess(names.index("export-cond_enc"), names.index("build-cond_enc"))
        self.assertLess(names.index("build-dit"), names.index("manifest"))

    def test_prepare_precedes_dit_consumers(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            names = [s.name for s in _dit_plan(Path(d))]
        self.assertLess(names.index("prepare-dit"), names.index("export-cond_enc"))
        self.assertLess(names.index("prepare-dit"), names.index("export-dit"))

    def test_cond_enc_export_passes_bf16(self) -> None:
        # cond_enc must be exported as BF16 (preferred) or FP16 (sm_75 fallback):
        # build_encoder_trt.py builds a strongly-typed engine, so the ONNX graph
        # dtypes propagate to the engine IO tensors. The C++ cond-enc runtime
        # (engine/src/cond-enc-trt.h) auto-detects the engine I/O dtype.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _dit_plan(Path(d))}
        cmd = steps["export-cond_enc"].command
        assert cmd is not None
        self.assertIn("--bf16", cmd)

    def test_dit_engine_in_outputs(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _dit_plan(Path(d))}
        outs = [p.name for p in steps["build-dit"].outputs]
        self.assertIn("dit.engine", outs)

    def test_dit_manifest_matches_cpp_reader_schema(self) -> None:
        # The DiT bundle manifest must NOT include text_enc.
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _dit_plan(Path(d))}
            steps["manifest"].run()
            payload = json.loads((Path(d) / "bundle" / "manifest.json").read_text())
        self.assertIn("version", payload)
        self.assertEqual(payload["variant"], "q8map-fp16")
        comps = payload["components"]
        self.assertEqual(comps["dit"]["engine"], "dit.engine")
        self.assertEqual(comps["cond_enc"]["engine"], "cond_encoder.engine")
        self.assertEqual(comps["fsq"]["sidecar"], "fsq.safetensors")
        self.assertTrue(comps["dit"]["weight_streaming"])
        self.assertNotIn("text_enc", comps)

    def test_manifest_records_encoder_precision(self) -> None:
        # DiT precision is the build variant; encoders default to BF16 on
        # Ampere+ GPUs (FP16-equivalent size, FP32-equivalent dynamic range —
        # no overflow). FP16 remains available as a precision_option for sm_75.
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _dit_plan(Path(d))}
            steps["manifest"].run()
            payload = json.loads((Path(d) / "bundle" / "manifest.json").read_text())
        comps = payload["components"]
        self.assertEqual(comps["dit"]["precision"], "q8map-fp16")
        self.assertEqual(comps["cond_enc"]["precision"], "bf16")


class EmbeddingPlanTests(unittest.TestCase):
    def test_ordered_step_names(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            names = [s.name for s in _embedding_plan(Path(d))]
        self.assertEqual(
            names,
            [
                "download",
                "export-text_enc",
                "build-text_enc",
                "collect-sidecars",
                "manifest",
            ],
        )

    def test_no_dit_or_cond_steps(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = _embedding_plan(Path(d))
        for s in steps:
            self.assertNotIn("dit", s.name)
            self.assertNotIn("cond_enc", s.name)

    def test_text_enc_export_passes_fp16(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _embedding_plan(Path(d))}
        cmd = steps["export-text_enc"].command
        assert cmd is not None
        self.assertIn("--fp16", cmd)

    def test_embedding_manifest_matches_cpp_reader_schema(self) -> None:
        # The embedding bundle manifest must include ONLY text_enc (no dit,
        # no cond_enc, no fsq).
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _embedding_plan(Path(d))}
            steps["manifest"].run()
            payload = json.loads((Path(d) / "emb-bundle" / "manifest.json").read_text())
        self.assertIn("version", payload)
        comps = payload["components"]
        self.assertEqual(comps["text_enc"]["engine"], "text_encoder.engine")
        self.assertEqual(comps["text_enc"]["precision"], "fp16")
        self.assertIn("embed_tokens.bin", comps["text_enc"]["sidecars"])
        self.assertNotIn("dit", comps)
        self.assertNotIn("cond_enc", comps)
        self.assertNotIn("fsq", comps)


class ResumeTests(unittest.TestCase):
    def _make_complete_step(self, out: Path) -> tbm.Step:
        return tbm.Step("probe", (), (out,), 1.0, command=("noop",))

    def test_step_complete_when_output_exists(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "made.bin"
            step = self._make_complete_step(out)
            self.assertFalse(step.is_complete())  # output missing -> not done
            out.write_text("x")
            self.assertTrue(step.is_complete())  # output present -> done

    def test_run_plan_skips_completed_step_runs_missing(self) -> None:
        # Build a real DiT plan, pre-create the outputs of the cond_enc export
        # step so resume skips it, leave others missing, and prove via the
        # emitted JSON status which steps were skipped vs. would run.
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            steps = _dit_plan(root)
            export_cond = next(s for s in steps if s.name == "export-cond_enc")
            for out in export_cond.outputs:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("done")

            buf = io.StringIO()
            with redirect_stdout(buf):
                tbm.run_plan(steps, dry_run=True)
            lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]
            by_name = {ln["name"]: ln["status"] for ln in lines if "name" in ln}

            self.assertEqual(by_name["export-cond_enc"], "skipped")
            self.assertEqual(by_name["export-dit"], "dry-run")

            # Mutation: corrupt the resume check by deleting one declared output.
            next(iter(export_cond.outputs)).unlink()
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                tbm.run_plan(steps, dry_run=True)
            lines2 = [json.loads(x) for x in buf2.getvalue().splitlines() if x.strip()]
            by_name2 = {ln["name"]: ln["status"] for ln in lines2 if "name" in ln}
            self.assertEqual(by_name2["export-cond_enc"], "dry-run")

    def test_failed_step_emits_json_error_and_nonzero(self) -> None:
        import io
        import json
        from contextlib import redirect_stderr

        def boom() -> None:
            raise RuntimeError("kaboom")

        with __import__("tempfile").TemporaryDirectory() as d:
            out = Path(d) / "never.bin"
            step = tbm.Step("explode", (), (out,), 1.0, action=boom)
            err = io.StringIO()
            with redirect_stderr(err):
                rc = tbm.run_plan([step])
            self.assertEqual(rc, 1)
            payload = json.loads(err.getvalue().splitlines()[-1])
            self.assertTrue(payload["error"])
            self.assertEqual(payload["step"], "explode")
            self.assertIn("kaboom", payload["message"])


class SourcePrepTests(unittest.TestCase):
    def _write_safetensors(self, path: Path, dtype) -> None:
        import torch
        from safetensors.torch import save_file
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"w": torch.zeros(2, 2, dtype=dtype)}, str(path))
        (path.parent / "config.json").write_text("{}")

    def test_bf16_source_repoints_dit_steps_to_staging(self) -> None:
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dit_src = root / "dit-src"
            self._write_safetensors(dit_src / "model.safetensors", torch.bfloat16)
            out = root / "bundle"
            steps = {s.name: s for s in tbm.build_plan("q8map-fp16", out, dit_src)}
            staging = out / "dit-fp32-source"
            cmd = steps["export-dit"].command
            assert cmd is not None
            self.assertIn(str(staging), cmd)
            self.assertNotIn(str(dit_src), cmd)
            outs = {p.name for p in steps["prepare-dit"].outputs}
            self.assertEqual(outs, {"model.safetensors", "silence_latent.pt"})

    def test_f32_source_used_in_place(self) -> None:
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dit_src = root / "dit-src"
            self._write_safetensors(dit_src / "model.safetensors", torch.float32)
            out = root / "bundle"
            steps = {s.name: s for s in tbm.build_plan("q8map-fp16", out, dit_src)}
            cmd = steps["export-dit"].command
            assert cmd is not None
            self.assertIn(str(dit_src), cmd)
            self.assertNotIn(str(out / "dit-fp32-source"), cmd)


if __name__ == "__main__":
    unittest.main()
