#!/usr/bin/env python3
"""
export_text_enc.py — Export Qwen3-Embedding text encoder to ONNX.

The text encoder is a standard Qwen3Model (28 layers, H=1024, causal attention)
that takes BPE token IDs and produces hidden states for the condition encoder.

Usage:
    python export_text_enc.py --model-dir <path-to-Qwen3-Embedding-0.6B> --output <output.onnx>

Exports:
    text_encoder.onnx — Full 28-layer transformer
        Input:  input_ids [B, S] int64
        Output: hidden_states [B, S, 1024] fp16

    embed_lookup.bin — Raw embedding table (vocab_size * hidden_size * 2 bytes, BF16)
        Used for lyric token embedding lookup on CPU (no ONNX needed).
"""

import argparse
import os
import sys
import time
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TextEncoderWrapper(nn.Module):
    """Wrapper around Qwen3Model that returns hidden_states as a flat tensor.

    ONNX inputs:
        input_ids:     [B, S] int64 — BPE token IDs

    ONNX output:
        hidden_states: [B, S, 1024] fp16 — last hidden state

    CRITICAL: The GGML text encoder (qwen3-enc.h:351) hardcodes is_causal=true,
    applying a lower-triangular causal mask (token i attends only to tokens 0..i).
    The HuggingFace Qwen3Model with attention_mask=None generates a causal mask
    ONLY for decoder models (is_decoder=True). Qwen3-Embedding is an encoder
    model, so attention_mask=None produces BIDIRECTIONAL attention — all tokens
    attend to all tokens. This mismatch causes completely different hidden
    states (the "text[0]" section of enc_hidden diverges entirely between TRT
    and GGML).

    Fix: explicitly build and pass a causal attention mask [B, S] (1=valid),
    which HF converts to a lower-triangular additive bias. This matches the
    GGML path's causal masking bit-for-bit.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        B, S = input_ids.shape
        # Build causal attention mask: [B, S] all-ones.
        # HF's _prepare_4d_causal_attention_mask converts this to a
        # lower-triangular additive bias (0.0 for j<=i, -inf for j>i),
        # matching GGML's qwen3_forward mask construction.
        attention_mask = torch.ones(B, S, device=input_ids.device, dtype=torch.long)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,  # causal mask: all valid, lower-triangular
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state


def load_model(model_dir: str, device: str = "cpu", dtype=torch.float32,
               low_memory_export: bool = True):
    """Load Qwen3-Embedding model from safetensors."""
    model_dir = Path(model_dir)
    
    # Fix Windows encoding issues
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    print(f"[export_text_enc] Loading model from {model_dir}...")
    t0 = time.time()
    
    from transformers import AutoModel, AutoConfig
    
    config = AutoConfig.from_pretrained(str(model_dir))
    # Force SDPA for ONNX export (no flash attention)
    config._attn_implementation = "sdpa"
    
    model = AutoModel.from_pretrained(
        str(model_dir),
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=low_memory_export,
    )
    model = model.to(device)
    model.eval()
    
    t1 = time.time()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[export_text_enc] Model loaded in {t1-t0:.1f}s ({n_params:.0f}M params)")
    print(f"[export_text_enc] Config: {config.num_hidden_layers}L, H={config.hidden_size}, "
          f"heads={config.num_attention_heads}/{config.num_key_value_heads}")
    
    return model, config


def export_onnx(model, config, output_path: str, opset: int = 18,
                low_memory_export: bool = True):
    """Export the text encoder to ONNX."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    wrapper = TextEncoderWrapper(model)
    wrapper.eval()
    
    # Dummy inputs for tracing
    B = 1
    S = 128  # typical sequence length
    
    dummy_input_ids = torch.randint(0, config.vocab_size, (B, S), device=device, dtype=torch.long)
    
    print(f"[export_text_enc] Tracing with shapes: input_ids={list(dummy_input_ids.shape)}")
    
    # Test forward pass
    print("[export_text_enc] Testing forward pass...")
    with torch.no_grad():
        test_out = wrapper(dummy_input_ids)
    print(f"[export_text_enc] Output shape: {list(test_out.shape)} "
          f"(expected [{B}, {S}, {config.hidden_size}])")
    
    # Export to ONNX
    print(f"[export_text_enc] Exporting to ONNX (opset {opset})...")
    t0 = time.time()
    
    torch.onnx.export(
        wrapper,
        (dummy_input_ids,),
        output_path,
        opset_version=opset,
        input_names=["input_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids":     {0: "batch", 1: "seq_len"},
            "hidden_states": {0: "batch", 1: "seq_len"},
        },
        do_constant_folding=not low_memory_export,
        export_params=True,
    )
    
    t1 = time.time()
    file_size = os.path.getsize(output_path)
    print(f"[export_text_enc] Exported to {output_path}")
    print(f"[export_text_enc] File size: {file_size/1e6:.1f} MB")
    print(f"[export_text_enc] Export time: {t1-t0:.1f}s")
    
    return output_path


def export_embed_table(model, config, output_path: str):
    """Export the embedding table as a raw binary file for lyric lookup.
    
    The lyric path uses embed_tokens lookup only (no transformer layers).
    We export the table as float32 for direct CPU indexing.
    
    Format: raw float32 array [vocab_size, hidden_size]
    """
    embed_weight = model.embed_tokens.weight.detach().cpu().float().numpy()
    V, H = embed_weight.shape
    
    with open(output_path, "wb") as f:
        # Header: vocab_size (int32), hidden_size (int32)
        f.write(struct.pack("<II", V, H))
        # Raw float32 weights
        f.write(embed_weight.tobytes())
    
    file_size = os.path.getsize(output_path)
    print(f"[export_text_enc] Embedding table: [{V}, {H}] -> {output_path} ({file_size/1e6:.1f} MB)")


def export_null_cond(model_dir: str, output_path: str):
    """Export null_condition_emb from the DiT model as raw float32.
    
    This is a [2048] float32 vector used for classifier-free guidance padding.
    Read from the DiT safetensors since it lives there.
    """
    from safetensors.torch import load_file
    
    model_dir = Path(model_dir)
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        # Try multi-shard
        for p in sorted(model_dir.glob("model-*.safetensors")):
            st = load_file(str(p))
            if "null_condition_emb" in st:
                vec = st["null_condition_emb"].detach().cpu().float().numpy()
                with open(output_path, "wb") as f:
                    f.write(struct.pack("<I", vec.shape[0]))
                    f.write(vec.tobytes())
                print(f"[export_text_enc] null_condition_emb: [{vec.shape[0]}] -> {output_path}")
                return
        print("[export_text_enc] WARNING: null_condition_emb not found")
        return
    
    st = load_file(str(st_path))
    if "null_condition_emb" not in st:
        print("[export_text_enc] WARNING: null_condition_emb not found in model.safetensors")
        return
    
    vec = st["null_condition_emb"].detach().cpu().float().numpy()
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", vec.shape[0]))
        f.write(vec.tobytes())
    print(f"[export_text_enc] null_condition_emb: [{vec.shape[0]}] -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-Embedding text encoder to ONNX")
    parser.add_argument("--model-dir", required=True,
                        help="Path to Qwen3-Embedding-0.6B directory")
    parser.add_argument("--output", default=None,
                        help="Output ONNX file (default: models/onnx/text_encoder.onnx)")
    parser.add_argument("--dit-dir", default=None,
                        help="Path to DiT model dir (for null_condition_emb export)")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (default: 18)")
    parser.add_argument("--device", default="cpu",
                        help="Device for model loading (default: cpu)")
    parser.add_argument("--fp16", action="store_true",
                        help="Export the ONNX graph in FP16 (native half weights, "
                             "the strongly-typed TRT 11 encoder precision for sm_75)")
    parser.add_argument("--bf16", action="store_true",
                        help="Export the ONNX graph in BF16 (bfloat16 weights). "
                             "Preferred over --fp16 for the text-enc engine: BF16 has "
                             "the same dynamic range as FP32 (8 exponent bits), so "
                             "intermediate activations in the 28-layer Qwen3 encoder "
                             "can't saturate to ±Inf/NaN. The C++ text-enc runtime "
                             "auto-detects the engine I/O dtype. Requires Ampere+ GPU "
                             "(sm80+) for native BF16 throughput.")
    parser.add_argument("--force", action="store_true",
                        help="Re-export even if the output ONNX already exists")
    parser.add_argument("--low-memory-export", dest="low_memory_export", action="store_true", default=True,
                        help="Use low CPU memory loading and skip ONNX constant folding (default)")
    parser.add_argument("--no-low-memory-export", dest="low_memory_export", action="store_false",
                        help="Use the legacy eager loader")
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        onnx_dir = Path(args.model_dir).parent / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(onnx_dir / "text_encoder.onnx")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dir = os.path.dirname(args.output)
    
    # Skip if output already exists (resumable builds).
    # Check both the ONNX shell and the external data file: a crash mid-export
    # can leave the shell written but the data file absent or truncated, which
    # would cause a silent corrupt-model skip on the next run.
    if not args.force:
        onnx_path = Path(args.output)
        data_path = Path(args.output + ".data")
        shell_ok = onnx_path.is_file()
        data_ok = (not data_path.exists()) or (data_path.stat().st_size > 0)
        if shell_ok and data_ok:
            print(f"[export_text_enc] Output already exists: {args.output} (use --force to re-export)")
            return
    
    # Load model
    # The text-enc TRT runtime (engine/src/text-enc-trt.h) auto-detects the
    # engine I/O dtype at load time and supports FP16 / BF16 / FP32 engines.
    #   --fp16: smallest engine, FP16 dynamic range (±65504) — intermediate
    #           activations in the 28-layer Qwen3 encoder can saturate to
    #           ±Inf/NaN, subtly degrading text_hidden which propagates to
    #           cond_enc and DiT. No sanitization in text-enc-trt.h (unlike
    #           cond-enc-trt.h), so corruption passes straight through.
    #   --bf16: same engine size as FP16, BF16 dynamic range (±3.4e38) matches
    #           FP32 — no overflow, no corruption. PREFERRED for text-enc.
    #           Requires Ampere+ GPU (sm80+) for native throughput.
    #   (default) FP32: largest engine, safest, slowest.
    if args.bf16 and args.fp16:
        raise SystemExit("--bf16 and --fp16 are mutually exclusive")
    if args.bf16:
        export_dtype = torch.bfloat16
    elif args.fp16:
        export_dtype = torch.float16
    else:
        export_dtype = torch.float32
    model, config = load_model(args.model_dir, device=args.device, dtype=export_dtype,
                               low_memory_export=args.low_memory_export)
    
    # Export ONNX
    export_onnx(model, config, args.output, opset=args.opset,
                low_memory_export=args.low_memory_export)
    
    # Export embedding table for lyric lookup
    embed_path = os.path.join(output_dir, "embed_tokens.bin")
    export_embed_table(model, config, embed_path)
    
    # Export null_condition_emb if DiT dir provided
    if args.dit_dir:
        null_cond_path = os.path.join(output_dir, "null_condition_emb.bin")
        export_null_cond(args.dit_dir, null_cond_path)
    
    print("[export_text_enc] Done!")


if __name__ == "__main__":
    main()
