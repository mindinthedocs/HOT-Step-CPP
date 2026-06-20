#!/usr/bin/env python3
"""Command-line front end for the TRT-bundle orchestrator.

Thin argparse layer over ``trt_bundle_manager``: it resolves a variant, builds
the resumable plan, slices it per subcommand (download / export / build / full
bundle), and prints the plan for ``info`` / the registry for ``list``. The plan
construction, step execution, source preparation, and manifest writing all live
in ``trt_bundle_manager``; this module only maps CLI args onto them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trt_bundle_manager import (  # noqa: E402
    MVP_COMPONENTS,
    REGISTRY,
    Step,
    build_plan,
    run_plan,
)


def _resolve_variant(variant: str) -> str:
    options = REGISTRY["dit"].precision_options
    if variant not in options:
        raise SystemExit(f"unknown variant {variant!r}; choices: {', '.join(options)}")
    return variant


def _plan_from_args(args: argparse.Namespace) -> list[Step]:
    return build_plan(
        _resolve_variant(args.variant), Path(args.output_dir).resolve(),
        Path(args.dit_dir).resolve(), Path(args.text_encoder_dir).resolve(),
        source_model=args.source_model,
        gguf_path=Path(args.gguf).resolve() if args.gguf else None,
    )


def _filter_component(steps: list[Step], component: str | None) -> list[Step]:
    if not component:
        return steps
    return [s for s in steps if s.name.endswith(component) or s.name == "fsq-sidecar"]


def _stage(steps: list[Step], prefixes: tuple[str, ...]) -> list[Step]:
    return [s for s in steps if s.name.split("-")[0] in prefixes]


def _cmd_bundle(args: argparse.Namespace) -> int:
    return run_plan(_plan_from_args(args), dry_run=args.dry_run)


def _cmd_download(args: argparse.Namespace) -> int:
    return run_plan(_stage(_plan_from_args(args), ("download",)), dry_run=args.dry_run)


def _cmd_export(args: argparse.Namespace) -> int:
    steps = _stage(_plan_from_args(args), ("prepare", "export", "fsq"))
    return run_plan(_filter_component(steps, args.component), dry_run=args.dry_run)


def _cmd_build(args: argparse.Namespace) -> int:
    steps = _stage(_plan_from_args(args), ("build",))
    return run_plan(_filter_component(steps, args.component), dry_run=args.dry_run)


def _cmd_list(args: argparse.Namespace) -> int:
    for name in MVP_COMPONENTS:
        spec = REGISTRY[name]
        print(f"{spec.name}\t{spec.display_name}\t{spec.default_precision}\t[{', '.join(spec.precision_options)}]")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    steps = _plan_from_args(args)
    print(json.dumps({
        "variant": _resolve_variant(args.variant),
        "output_dir": str(Path(args.output_dir)),
        "components": list(MVP_COMPONENTS),
        "steps": [{"name": s.name, "outputs": [str(o) for o in s.outputs]} for s in steps],
    }, indent=2))
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--variant", default=REGISTRY["dit"].default_precision,
                   help="DiT precision recipe (q8map-fp16 | w8a8 | fp32)")
    p.add_argument("--output-dir", required=True, help="Bundle output directory")
    p.add_argument("--dit-dir", required=True, help="DiT source safetensors directory")
    p.add_argument("--text-encoder-dir", required=True, help="Qwen3 text encoder source directory")
    p.add_argument("--source-model", default=None, help="Manifest source_model badge label (else DiT dir name)")
    p.add_argument("--gguf", default=None,
                   help="BF16 GGUF used to reconstruct silence_latent.pt when absent from --dit-dir")
    p.add_argument("--dry-run", action="store_true", help="Emit the step plan without executing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trt_bundle_manager",
                                     description="Resumable TRT-bundle orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bundle = sub.add_parser("bundle", help="Run every step (download -> prepare -> export -> build -> manifest)")
    _add_common(p_bundle)
    p_bundle.set_defaults(func=_cmd_bundle)

    p_dl = sub.add_parser("download", help="Verify source safetensors are cached")
    _add_common(p_dl)
    p_dl.set_defaults(func=_cmd_download)

    p_export = sub.add_parser("export", help="Run source prep + ONNX export steps")
    _add_common(p_export)
    p_export.add_argument("--component", choices=MVP_COMPONENTS, help="Limit to one component")
    p_export.add_argument("--precision", help="(forward) per-component precision override")
    p_export.set_defaults(func=_cmd_export)

    p_build = sub.add_parser("build", help="Run TRT engine build steps")
    _add_common(p_build)
    p_build.add_argument("--component", choices=MVP_COMPONENTS, help="Limit to one component")
    p_build.add_argument("--precision", help="(forward) per-component precision override")
    p_build.set_defaults(func=_cmd_build)

    p_info = sub.add_parser("info", help="Print the resolved step plan as JSON")
    _add_common(p_info)
    p_info.set_defaults(func=_cmd_info)

    p_list = sub.add_parser("list", help="List registered components")
    p_list.set_defaults(func=_cmd_list)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
