#!/usr/bin/env python3
"""Component-registry-driven, resumable orchestrator for TRT bundles.

Produces a self-contained ``trt-bundles/<name>/`` directory (DiT + encoder
``.engine`` files + FSQ + CPU sidecars + ``manifest.json``) by INVOKING the
existing single-purpose export/build scripts. It does not re-implement the
subprocess-invoke or sidecar-copy logic: it reuses ``run_tool`` and
``copy_sidecar`` from ``tools/onnx-export/export_runtime_bundle.py`` (the
existing exporter-orchestration core) and adds the component registry, the
resumable step model, the CLI, and the JSON progress protocol.

Resume is stateless and disk-to-disk: a step is skipped when every output it
declares already exists. There is no cleanup on failure, so a re-run resumes
from the first step whose outputs are missing.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_EXPORT_DIR = SCRIPT_DIR.parent / "onnx-export"
TOOLS_DIR = SCRIPT_DIR.parent
if str(ONNX_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(ONNX_EXPORT_DIR))

# REUSE the existing exporter-orchestration core (do not re-implement).
from export_runtime_bundle import copy_sidecar, run_tool  # noqa: E402
from prepare_dit_source import (  # noqa: E402
    DIT_MODELING_PY,
    SILENCE_LATENT,
    prepare_dit_dir,
    safetensors_is_bf16,
)


# --------------------------------------------------------------------------
# Component registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSpec:
    """Declarative description of one bundle component.

    Forward-looking fields (``onnx_dynamic_shapes``, ``depends_on``,
    ``build_flag_weight_stream``) carry the shape future components (vae, lm)
    will need; the MVP registry only populates dit/text_enc/cond_enc.
    """

    name: str
    display_name: str
    export_script: str
    build_flag_weight_stream: bool
    precision_options: tuple[str, ...]
    default_precision: str
    sidecar_files: tuple[str, ...]
    onnx_dynamic_shapes: bool
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.default_precision not in self.precision_options:
            raise ValueError(
                f"{self.name}: default_precision {self.default_precision!r} "
                f"not in precision_options {self.precision_options!r}"
            )


# DiT precision recipes: q8map-fp16 (default) / w8a16 / fp32, enforced on-disk by
# acestep_trt_common.validate_dit_precision_manifest. Encoders run FP16
# strongly-typed: sm_75 has no BF16 tensor cores.
REGISTRY: dict[str, ComponentSpec] = {
    "dit": ComponentSpec(
        name="dit",
        display_name="DiT transformer",
        export_script="export_dit.py",
        build_flag_weight_stream=True,
        precision_options=("q8map-fp16", "w8a16", "fp32"),
        default_precision="q8map-fp16",
        sidecar_files=("config.json", "silence_latent.pt"),
        onnx_dynamic_shapes=True,
    ),
    "text_enc": ComponentSpec(
        name="text_enc",
        display_name="Qwen3 text encoder",
        export_script="export_text_enc.py",
        build_flag_weight_stream=False,
        precision_options=("fp16",),
        default_precision="fp16",
        sidecar_files=("embed_tokens.bin", "vocab.json", "merges.txt"),
        onnx_dynamic_shapes=True,
    ),
    "cond_enc": ComponentSpec(
        name="cond_enc",
        display_name="Condition encoder",
        export_script="export_cond_enc.py",
        build_flag_weight_stream=False,
        precision_options=("fp16",),
        default_precision="fp16",
        sidecar_files=("null_condition_emb.bin",),
        onnx_dynamic_shapes=True,
        depends_on=("text_enc",),
    ),
}

MVP_COMPONENTS: tuple[str, ...] = ("dit", "text_enc", "cond_enc")


# --------------------------------------------------------------------------
# Resumable step model
# --------------------------------------------------------------------------


@dataclass
class Step:
    """One unit of bundle work.

    Exactly one of ``command`` (a subprocess argv run via the reused
    ``run_tool``) or ``action`` (a Python callable, e.g. sidecar copy / manifest
    write) drives the step. ``outputs`` are the files whose collective existence
    means the step is already done — the basis for stateless resume.
    """

    name: str
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    progress_weight: float
    command: tuple[str, ...] | None = None
    action: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if (self.command is None) == (self.action is None):
            raise ValueError(f"step {self.name!r}: exactly one of command/action required")
        if not self.outputs:
            raise ValueError(f"step {self.name!r}: must declare at least one output")

    def is_complete(self) -> bool:
        return all(out.exists() for out in self.outputs)

    def run(self) -> None:
        if self.command is not None:
            run_tool(list(self.command))
        else:
            assert self.action is not None
            self.action()


# --------------------------------------------------------------------------
# JSON progress protocol
# --------------------------------------------------------------------------


def _emit_progress(index: int, name: str, progress: float, status: str) -> None:
    line = {"step": index, "name": name, "progress": round(progress, 4), "status": status}
    print(json.dumps(line), flush=True)


def _emit_error(name: str, message: str) -> None:
    err = {"error": True, "step": name, "message": message}
    print(json.dumps(err), file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Plan construction
# --------------------------------------------------------------------------


def _exporter(script: str, args: Sequence[str]) -> tuple[str, ...]:
    return (sys.executable, str(ONNX_EXPORT_DIR / script), *args)


def _tools_script(script: str, args: Sequence[str]) -> tuple[str, ...]:
    return (sys.executable, str(TOOLS_DIR / script), *args)


def _build_encoder(module: str, onnx: Path, stem: str, output_dir: Path) -> tuple[str, ...]:
    return _exporter(
        "build_encoder_trt.py",
        ["--module", module, "--onnx", str(onnx), "--out-dir", str(output_dir), "--engine-stem", stem],
    )


DEFAULT_GGUF = "models/acestep-v15-sft-BF16.gguf"


def _resolve_dit_source(dit_dir: Path, output_dir: Path) -> tuple[Path, Path | None]:
    """Pick the DiT source dir downstream steps read, plus a staging dir.

    The low-memory DiT export needs F32 weights. When ``dit_dir`` ships BF16,
    downstream steps must read a converted F32 staging dir instead; the ``prepare``
    step materializes it. F32 sources are used in place (no staging). Detection is
    a cheap safetensors-header read; the conversion itself happens in ``prepare``.
    """

    model = dit_dir / "model.safetensors"
    if model.is_file() and safetensors_is_bf16(model):
        staging = output_dir / "dit-fp32-source"
        return staging, staging
    return dit_dir, None


def build_plan(
    variant: str,
    output_dir: Path,
    dit_dir: Path,
    text_encoder_dir: Path,
    *,
    components: Sequence[str] = MVP_COMPONENTS,
    source_model: str | None = None,
    gguf_path: Path | None = None,
) -> list[Step]:
    """Build the ordered, resumable step list for a variant.

    Order: download -> prepare -> export(text_enc, cond_enc, dit) -> build(text_enc,
    cond_enc, dit) -> fsq -> sidecars -> manifest. ``prepare`` converts BF16 DiT
    weights to the F32 the exporter needs and reconstructs ``silence_latent.pt``
    from the GGUF when absent, so a fresh bundle runs from the published snapshot
    without manual prep. Encoders export/build before DiT because cond_enc depends
    on the text-encoder embed table, and the heavy DiT build is last so a resume
    re-runs only it after an OOM.
    """

    selected = [REGISTRY[name] for name in components]
    if "dit" not in components:
        raise ValueError("plan requires the dit component")

    dit_src, staging = _resolve_dit_source(dit_dir, output_dir)
    gguf = gguf_path or (SCRIPT_DIR.parent.parent / DEFAULT_GGUF)

    dit_onnx = output_dir / "dit.onnx"
    text_onnx = output_dir / "text_encoder.onnx"
    cond_onnx = output_dir / "cond_encoder.onnx"
    dit_engine = output_dir / "dit.engine"
    text_engine = output_dir / "text_encoder.engine"
    cond_engine = output_dir / "cond_encoder.engine"
    fsq_sidecar = output_dir / "fsq.safetensors"
    manifest = output_dir / "manifest.json"
    prepare_outputs = (dit_src / "model.safetensors", dit_src / SILENCE_LATENT, dit_src / DIT_MODELING_PY)

    # text_enc is invoked WITHOUT --dit-dir so it is NOT a second producer of
    # null_condition_emb — cond_enc is the sole owner. build-trt-engine writes
    # the runtime alias dit.engine next to dit.onnx via --runtime-alias.
    steps: list[Step] = [
        Step("download", (), (dit_dir / "config.json", text_encoder_dir / "config.json"),
             1.0, action=lambda: _require_sources(dit_dir, text_encoder_dir)),
        Step("prepare-dit", (dit_dir / "config.json",), prepare_outputs, 1.0,
             action=lambda: prepare_dit_dir(dit_dir, staging or dit_src, gguf)),
        Step("export-text_enc", (text_encoder_dir / "config.json",),
             (text_onnx, output_dir / "embed_tokens.bin"), 2.0,
             command=_exporter("export_text_enc.py",
                               ["--model-dir", str(text_encoder_dir), "--output", str(text_onnx), "--fp16"])),
        Step("export-cond_enc", (dit_src / "config.json",),
             (cond_onnx, output_dir / "null_condition_emb.bin"), 2.0,
             command=_exporter("export_cond_enc.py",
                               ["--model-dir", str(dit_src), "--output", str(cond_onnx)])),
        Step("export-dit", (dit_src / "config.json",), (dit_onnx,), 4.0,
             command=_exporter("export_dit.py",
                               ["--model-dir", str(dit_src), "--output", str(dit_onnx),
                                "--precision", variant])),
        Step("build-text_enc", (text_onnx,), (text_engine,), 2.0,
             command=_build_encoder("text-enc", text_onnx, "text_encoder", output_dir)),
        Step("build-cond_enc", (cond_onnx,), (cond_engine,), 2.0,
             command=_build_encoder("cond-enc", cond_onnx, "cond_encoder", output_dir)),
        Step("build-dit", (dit_onnx,), (dit_engine,), 6.0,
             command=_tools_script("build-trt-engine.py",
                                   ["--onnx", str(dit_onnx), "--out-dir", str(output_dir / "engines"),
                                    "--precision-policy", variant])),
        Step("fsq-sidecar", (dit_src / "config.json",), (fsq_sidecar,), 1.0,
             command=_exporter("export_fsq_sidecar.py",
                               ["--model-dir", str(dit_src), "--output-dir", str(output_dir)])),
        _sidecar_copy_step(output_dir, dit_src, text_encoder_dir, selected),
        Step("manifest", (dit_engine, text_engine, cond_engine, fsq_sidecar), (manifest,), 1.0,
             action=lambda: write_manifest(manifest, variant, dit_dir, source_model=source_model)),
    ]
    return steps


def _require_sources(dit_dir: Path, text_encoder_dir: Path) -> None:
    for label, path in (("DiT source", dit_dir), ("text-encoder source", text_encoder_dir)):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"{label} not cached: expected {path / 'config.json'} "
                "(download the source safetensors before running the bundle)"
            )


def _sidecar_copy_step(
    output_dir: Path,
    dit_dir: Path,
    text_encoder_dir: Path,
    selected: Sequence[ComponentSpec],
) -> Step:
    # Sidecars copied verbatim from source dirs (vs sidecars produced by an
    # exporter step such as embed_tokens / null_condition_emb / fsq).
    copy_map: dict[str, Path] = {
        "config.json": dit_dir / "config.json",
        "silence_latent.pt": dit_dir / "silence_latent.pt",
        "vocab.json": text_encoder_dir / "vocab.json",
        "merges.txt": text_encoder_dir / "merges.txt",
    }
    wanted: dict[str, Path] = {}
    for spec in selected:
        for fname in spec.sidecar_files:
            if fname in copy_map:
                wanted[fname] = copy_map[fname]

    def copy_all() -> None:
        for fname, src in wanted.items():
            copy_sidecar(src, output_dir / fname, fname)

    return Step(
        "collect-sidecars",
        tuple(wanted.values()),
        tuple(output_dir / fname for fname in wanted),
        1.0,
        action=copy_all,
    )


MANIFEST_VERSION = "1"
DIT_ENGINE = "dit.engine"
TEXT_ENC_ENGINE = "text_encoder.engine"
COND_ENC_ENGINE = "cond_encoder.engine"
FSQ_SIDECAR = "fsq.safetensors"


def _build_manifest(variant: str, source_model: str) -> dict[str, object]:
    """The manifest payload matching the C++ reader (``trt-bundle-manifest.h``).

    Field names track ``trt_bundle_load_manifest`` exactly: a ``components``
    object keyed ``dit``/``text_enc``/``cond_enc``/``fsq``, each with a relative
    ``engine`` (or ``sidecar`` for fsq), optional ``metadata``/``precision``/
    ``sidecars``/``weight_streaming``. Paths are relative to the manifest (bundle
    root). The DiT precision is the build variant (``q8map-fp16``/``w8a16``/
    ``fp32``); the encoders are FP16 — never ``bf16``, since sm_75 has no BF16
    tensor cores.
    """

    dit_spec = REGISTRY["dit"]
    text_spec = REGISTRY["text_enc"]
    cond_spec = REGISTRY["cond_enc"]
    return {
        "version": MANIFEST_VERSION,
        "source_model": source_model,
        "variant": variant,
        "components": {
            "dit": {
                "engine": DIT_ENGINE,
                "metadata": "dit.metadata.json",
                "precision": variant,
                "weight_streaming": dit_spec.build_flag_weight_stream,
            },
            "text_enc": {
                "engine": TEXT_ENC_ENGINE,
                "metadata": "text_encoder.metadata.json",
                "precision": text_spec.default_precision,
                "sidecars": list(text_spec.sidecar_files),
            },
            "cond_enc": {
                "engine": COND_ENC_ENGINE,
                "metadata": "cond_encoder.metadata.json",
                "precision": cond_spec.default_precision,
                "sidecars": list(cond_spec.sidecar_files),
            },
            "fsq": {
                "sidecar": FSQ_SIDECAR,
                "metadata": "fsq.metadata.json",
            },
        },
    }


def write_manifest(manifest: Path, variant: str, dit_dir: Path,
                   *, source_model: str | None = None) -> None:
    """Emit ``manifest.json`` per the schema the C++ reader parses.

    ``source_model`` is the explicit model label when supplied (so the badge
    reads e.g. ``acestep-v15-sft`` not an HF snapshot hash); else the dir name.
    """

    payload = _build_manifest(variant, source_model=source_model or dit_dir.name)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[trt-bundle] manifest: {manifest}")


# --------------------------------------------------------------------------
# Plan execution
# --------------------------------------------------------------------------


def run_plan(steps: Sequence[Step], *, dry_run: bool = False) -> int:
    total = len(steps)
    completed = 0.0
    total_weight = sum(s.progress_weight for s in steps) or 1.0
    for index, step in enumerate(steps):
        if step.is_complete():
            completed += step.progress_weight
            _emit_progress(index, step.name, completed / total_weight, "skipped")
            continue
        _emit_progress(index, step.name, completed / total_weight, "running")
        if dry_run:
            completed += step.progress_weight
            _emit_progress(index, step.name, completed / total_weight, "dry-run")
            continue
        try:
            step.run()
        except Exception as exc:  # surface the failing step + propagate non-zero
            _emit_error(step.name, str(exc))
            return 1
        completed += step.progress_weight
        _emit_progress(index, step.name, completed / total_weight, "done")
    _emit_progress(total, "bundle", 1.0, "complete")
    return 0


