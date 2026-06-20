#!/usr/bin/env python3
"""Behavior tests for the TRT-bundle orchestrator.

Run: ``python -m unittest tools.trt.test_trt_bundle_manager`` (or, from this
dir, ``python -m unittest test_trt_bundle_manager``).

Proven behaviors:
  - the component registry is well-formed (the dit/text_enc/cond_enc set, DiT
    precision recipes, cond_enc depends on text_enc);
  - a ``bundle`` dry-run produces the correct ordered step list;
  - resume skips a step whose outputs already exist, and a corrupted resume
    check (a missing output) makes a previously-done step run again.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trt_bundle_manager as tbm


def _plan(tmp: Path) -> list[tbm.Step]:
    return tbm.build_plan(
        "q8map-fp16",
        tmp / "bundle",
        tmp / "dit-src",
        tmp / "text-src",
    )


class RegistryTests(unittest.TestCase):
    def test_mvp_components_present(self) -> None:
        self.assertEqual(set(tbm.MVP_COMPONENTS), {"dit", "text_enc", "cond_enc"})
        for name in tbm.MVP_COMPONENTS:
            self.assertIn(name, tbm.REGISTRY)

    def test_dit_precision_recipes(self) -> None:
        dit = tbm.REGISTRY["dit"]
        self.assertEqual(dit.precision_options, ("q8map-fp16", "w8a8", "fp32"))
        self.assertEqual(dit.default_precision, "q8map-fp16")

    def test_encoders_are_fp16(self) -> None:
        for name in ("text_enc", "cond_enc"):
            self.assertEqual(tbm.REGISTRY[name].default_precision, "fp16")

    def test_cond_enc_depends_on_text_enc(self) -> None:
        self.assertEqual(tbm.REGISTRY["cond_enc"].depends_on, ("text_enc",))

    def test_text_enc_owns_embed_tokens(self) -> None:
        self.assertIn("embed_tokens.bin", tbm.REGISTRY["text_enc"].sidecar_files)

    def test_cond_enc_owns_null_condition_emb(self) -> None:
        # cond_enc is the single owner of null_condition_emb.
        self.assertIn("null_condition_emb.bin", tbm.REGISTRY["cond_enc"].sidecar_files)
        self.assertNotIn("null_condition_emb.bin", tbm.REGISTRY["text_enc"].sidecar_files)

    def test_bad_default_precision_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tbm.ComponentSpec(
                name="x", display_name="x", export_script="x.py",
                build_flag_weight_stream=False,
                precision_options=("fp16",), default_precision="fp32",
                sidecar_files=(), onnx_dynamic_shapes=False,
            )


class PlanOrderTests(unittest.TestCase):
    def test_ordered_step_names(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            names = [s.name for s in _plan(Path(d))]
        self.assertEqual(
            names,
            [
                "download",
                "prepare-dit",
                "export-text_enc",
                "export-cond_enc",
                "export-dit",
                "build-text_enc",
                "build-cond_enc",
                "build-dit",
                "fsq-sidecar",
                "collect-sidecars",
                "manifest",
            ],
        )

    def test_export_precedes_build_precedes_manifest(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            names = [s.name for s in _plan(Path(d))]
        self.assertLess(names.index("export-dit"), names.index("build-dit"))
        self.assertLess(names.index("export-text_enc"), names.index("export-cond_enc"))
        self.assertLess(names.index("build-dit"), names.index("manifest"))

    def test_prepare_precedes_dit_consumers(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            names = [s.name for s in _plan(Path(d))]
        self.assertLess(names.index("prepare-dit"), names.index("export-cond_enc"))
        self.assertLess(names.index("prepare-dit"), names.index("export-dit"))

    def test_text_enc_export_has_no_dit_dir_flag(self) -> None:
        # text_enc must NOT be a second null_condition_emb producer.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _plan(Path(d))}
        cmd = steps["export-text_enc"].command
        assert cmd is not None
        self.assertNotIn("--dit-dir", cmd)
        self.assertIn("--fp16", cmd)

    def test_cond_enc_export_passes_fp16(self) -> None:
        # cond_enc must be exported as FP16: build_encoder_trt.py builds a
        # strongly-typed engine, so the ONNX graph dtypes propagate to the
        # engine IO tensors. The C++ cond-enc runtime (engine/src/cond-enc-trt.h)
        # validates text_hidden/lyric_embed/timbre_feats as kHALF inputs and
        # enc_hidden as a kHALF output — an FP32 ONNX would fail that contract
        # at runtime with "text_hidden has unexpected mode/dtype".
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _plan(Path(d))}
        cmd = steps["export-cond_enc"].command
        assert cmd is not None
        self.assertIn("--fp16", cmd)

    def test_dit_engine_in_outputs(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _plan(Path(d))}
        outs = [p.name for p in steps["build-dit"].outputs]
        self.assertIn("dit.engine", outs)

    def test_manifest_matches_cpp_reader_schema(self) -> None:
        # The writer must emit the field structure trt_bundle_load_manifest
        # parses: root version/source_model/variant + a components object keyed
        # dit/text_enc/cond_enc/fsq, each with a relative engine (or sidecar for
        # fsq). A wrong field name or nesting breaks the C++ all-three gate.
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _plan(Path(d))}
            steps["manifest"].run()
            payload = json.loads((Path(d) / "bundle" / "manifest.json").read_text())
        self.assertIn("version", payload)
        self.assertIn("source_model", payload)
        self.assertEqual(payload["variant"], "q8map-fp16")
        comps = payload["components"]
        self.assertEqual(comps["dit"]["engine"], "dit.engine")
        self.assertEqual(comps["text_enc"]["engine"], "text_encoder.engine")
        self.assertEqual(comps["cond_enc"]["engine"], "cond_encoder.engine")
        self.assertEqual(comps["fsq"]["sidecar"], "fsq.safetensors")
        self.assertTrue(comps["dit"]["weight_streaming"])
        self.assertEqual(comps["text_enc"]["sidecars"], ["embed_tokens.bin", "vocab.json", "merges.txt"])

    def test_manifest_records_onbox_precision_not_bf16(self) -> None:
        # DiT precision is the build variant, encoders fp16, never bf16
        # (sm_75 has no BF16 tensor cores).
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            steps = {s.name: s for s in _plan(Path(d))}
            steps["manifest"].run()
            payload = json.loads((Path(d) / "bundle" / "manifest.json").read_text())
        comps = payload["components"]
        self.assertEqual(comps["dit"]["precision"], "q8map-fp16")
        self.assertEqual(comps["text_enc"]["precision"], "fp16")
        self.assertEqual(comps["cond_enc"]["precision"], "fp16")
        self.assertNotIn("bf16", json.dumps(payload))


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
        # Build a real plan, pre-create the outputs of the first export step so
        # resume skips it, leave others missing, and prove via the emitted JSON
        # status which steps were skipped vs. would run.
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            steps = _plan(root)
            export_text = next(s for s in steps if s.name == "export-text_enc")
            for out in export_text.outputs:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("done")

            buf = io.StringIO()
            with redirect_stdout(buf):
                tbm.run_plan(steps, dry_run=True)
            lines = [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]
            by_name = {ln["name"]: ln["status"] for ln in lines if "name" in ln}

            self.assertEqual(by_name["export-text_enc"], "skipped")
            self.assertEqual(by_name["export-cond_enc"], "dry-run")

            # Mutation: corrupt the resume check by deleting one declared output.
            # The previously-skipped step must now run instead of being skipped.
            next(iter(export_text.outputs)).unlink()
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                tbm.run_plan(steps, dry_run=True)
            lines2 = [json.loads(x) for x in buf2.getvalue().splitlines() if x.strip()]
            by_name2 = {ln["name"]: ln["status"] for ln in lines2 if "name" in ln}
            self.assertEqual(by_name2["export-text_enc"], "dry-run")

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
            steps = {s.name: s for s in tbm.build_plan("q8map-fp16", out, dit_src, root / "text-src")}
            staging = out / "dit-fp32-source"
            # DiT export reads --model-dir from the F32 staging dir, not the BF16 src.
            cmd = steps["export-dit"].command
            assert cmd is not None
            self.assertIn(str(staging), cmd)
            self.assertNotIn(str(dit_src), cmd)
            # prepare-dit declares the staged weights + silence_latent as its outputs.
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
            steps = {s.name: s for s in tbm.build_plan("q8map-fp16", out, dit_src, root / "text-src")}
            cmd = steps["export-dit"].command
            assert cmd is not None
            self.assertIn(str(dit_src), cmd)
            self.assertNotIn(str(out / "dit-fp32-source"), cmd)


if __name__ == "__main__":
    unittest.main()
