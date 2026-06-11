#!/usr/bin/env python3
"""Build a prebuilt TensorRT engine for the HOT-Step LM ONNX export."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lm_full.engine from lm_full.onnx")
    parser.add_argument("--onnx", required=True, type=Path, help="Path to lm_full.onnx")
    parser.add_argument("--engine", default=None, type=Path,
                        help="Output engine path (default: <onnx stem>.engine)")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--opt-past-len", type=int, default=512)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--builder-optimization-level", type=int, default=5,
                        choices=range(0, 6), metavar="{0..5}")
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


class Logger:
    def __init__(self, trt):
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.INFO)


def enum_value(enum_cls, name: str):
    return getattr(enum_cls, name)


def set_config_flag(config, trt, name: str) -> bool:
    if not hasattr(trt.BuilderFlag, name):
        return False
    config.set_flag(enum_value(trt.BuilderFlag, name))
    return True


def input_profile_shapes(name: str, rank: int, args: argparse.Namespace):
    if name in ("input_ids", "position_ids"):
        return (1, 1), (1, 1), (1, args.max_input_len)
    if name == "attention_mask":
        return (1, 1), (1, min(args.opt_past_len, args.max_seq_len)), (1, args.max_seq_len)
    if name.startswith("past_key_") or name.startswith("past_value_"):
        return (
            (1, args.n_kv_heads, 1, args.head_dim),
            (1, args.n_kv_heads, min(args.opt_past_len, args.max_seq_len), args.head_dim),
            (1, args.n_kv_heads, args.max_seq_len, args.head_dim),
        )
    if rank == 2:
        return (1, 1), (1, 1), (1, args.max_input_len)
    if rank == 4:
        return (
            (1, args.n_kv_heads, 1, args.head_dim),
            (1, args.n_kv_heads, min(args.opt_past_len, args.max_seq_len), args.head_dim),
            (1, args.n_kv_heads, args.max_seq_len, args.head_dim),
        )
    raise SystemExit(f"cannot infer LM profile for input {name!r} with rank {rank}")


def main() -> None:
    args = parse_args()
    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        raise SystemExit(f"missing ONNX model: {onnx_path}")
    engine_path = args.engine.resolve() if args.engine else onnx_path.with_suffix(".engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tensorrt as trt
    except Exception as exc:
        raise SystemExit("TensorRT Python bindings are required to build LM engines") from exc

    try:
        import cuda.bindings.driver as cuda_driver
        cuda_driver.cuInit(0)
    except Exception:
        pass

    logger = Logger(trt).logger
    builder = trt.Builder(logger)
    if builder is None:
        raise SystemExit("failed to create TensorRT builder")

    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    elif hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    print(f"[LM-TRT] Parsing {onnx_path}")
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("LM ONNX parse failed")

    config = builder.create_builder_config()
    config.builder_optimization_level = args.builder_optimization_level
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(args.workspace_gb * (1024 ** 3)),
        )
    set_config_flag(config, trt, "TF32")
    set_config_flag(config, trt, "REFIT_IDENTICAL")
    set_config_flag(config, trt, "WEIGHT_STREAMING")

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        min_shape, opt_shape, max_shape = input_profile_shapes(inp.name, len(inp.shape), args)
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
        print(f"[LM-TRT] profile {inp.name}: min={min_shape} opt={opt_shape} max={max_shape}")
    config.add_optimization_profile(profile)

    print("[LM-TRT] Building serialized engine")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("LM TensorRT build failed")
    engine_path.write_bytes(bytes(serialized))
    elapsed = time.time() - t0
    metadata = {
        "artifact": "hotstep-lm-trt-engine",
        "onnx": onnx_path.name,
        "engine": engine_path.name,
        "max_seq_len": args.max_seq_len,
        "max_input_len": args.max_input_len,
        "opt_past_len": args.opt_past_len,
        "n_kv_heads": args.n_kv_heads,
        "head_dim": args.head_dim,
        "workspace_gb": args.workspace_gb,
        "builder_optimization_level": args.builder_optimization_level,
        "build_seconds": elapsed,
        "engine_bytes": engine_path.stat().st_size,
    }
    engine_path.with_suffix(engine_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[LM-TRT] Engine saved: {engine_path} ({metadata['engine_bytes']:,} bytes, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
