#!/usr/bin/env python3
"""Build a deterministic synthetic MoE GGUF test vehicle.

Carried over from a prior run's uncommitted WIP in the sibling work/ clone.
SUPERSEDED: this fixture does NOT load in llama-cli (exit 134, "tensor buffer not
set") because all its layers are SWA, so `tau` is unused, `inp_pos` is pruned and
never allocated. Use `make_synthetic_gguf.py` in this directory for a loadable
fixture; see `tools/audit/PARITY_REPORT.md`. Kept only to preserve the WIP and
record the failure mode.

This is a *test vehicle*, not a model. Every tensor value comes from a seeded
numpy PRNG, and every metadata value is a fixed constant. No Inkling weights are
read, downloaded or embedded. No performance claim is made by this fixture.

Usage:
    python make_moe_gguf.py --out /tmp/fixture.gguf [--seed 0] [--manifest PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "gguf-py"))

import gguf  # noqa: E402

ARCH = "inkling2_loop"
MODEL_NAME = "synthetic-inkling2-loop"

N_LAYER = 4
N_EMBD = 16
N_HEAD = 2
N_HEAD_KV = 1
HEAD_DIM = 8
KV_DIM = 8
N_FF_EXP = 8
N_EXPERT = 128
N_EXPERT_USED = 8
LOOP_START = 2
LOOP_END = 4
N_PASS = 3
GATE_RANK = 64
CORE_LAYERS = (2, 3)
CTRL_NAMES = ("q", "k", "v", "ffn")

N_CTX = 64
VOCAB_SIZE = 32
N_REL = 4
N_EXTENT = 8
N_CONV = 2
DIM_SCALE = 0.05

GGUF_MAX_BYTES = 4 * 1024 * 1024


def constants() -> dict:
    return {
        "architecture": ARCH,
        "n_layers": N_LAYER,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_head_kv": N_HEAD_KV,
        "head_dim": HEAD_DIM,
        "n_ff_exp": N_FF_EXP,
        "expert_count": N_EXPERT,
        "expert_used_count": N_EXPERT_USED,
        "loop_start": LOOP_START,
        "loop_end": LOOP_END,
        "loop_count": N_PASS,
        "gate_rank": GATE_RANK,
        "core_layers": list(CORE_LAYERS),
        "context_length": N_CTX,
        "vocab_size": VOCAB_SIZE,
        "relative_length": N_REL,
        "relative_extent": N_EXTENT,
        "conv_l_cache": N_CONV,
        "weight_scale": DIM_SCALE,
    }


def tensor_specs() -> dict:
    """Return name -> numpy shape, in the order written to the GGUF."""
    specs = {
        "token_embd.weight": (VOCAB_SIZE, N_EMBD),
        "token_embd_norm.weight": (N_EMBD,),
        "output.weight": (VOCAB_SIZE, N_EMBD),
        "output_norm.weight": (N_EMBD,),
        "output_res_query.weight": (N_EMBD,),
    }
    for i in range(N_LAYER):
        p = f"blk.{i}."
        specs.update({
            p + "attn_q.weight": (HEAD_DIM * N_HEAD, N_EMBD),
            p + "attn_k.weight": (KV_DIM, N_EMBD),
            p + "attn_v.weight": (KV_DIM, N_EMBD),
            p + "attn_output.weight": (N_EMBD, N_EMBD),
            p + "attn_gate.weight": (N_EMBD, N_EMBD),
            p + "attn_q_clip.weight": (N_HEAD, HEAD_DIM),
            p + "attn_sink_proj.weight": (N_HEAD, N_EMBD),
            p + "attn_rel.weight": (N_REL * N_HEAD, N_EMBD),
            p + "attn_rel_bias.weight": (N_EXTENT, N_REL),
            p + "attn_rel_clip.weight": (N_HEAD, N_REL),
            p + "attn_k_conv.weight": (KV_DIM, N_CONV),
            p + "attn_v_conv.weight": (KV_DIM, N_CONV),
            p + "attn_post_conv.weight": (N_EMBD, N_CONV),
            p + "ffn_post_conv.weight": (N_EMBD, N_CONV),
            p + "attn_norm.weight": (N_EMBD,),
            p + "ffn_res_query.weight": (N_EMBD,),
            p + "ffn_norm.weight": (N_EMBD,),
            p + "ffn_gate_inp.weight": (N_EXPERT, N_EMBD),
            p + "ffn_gate_up_exps.weight": (N_EXPERT, 2 * N_FF_EXP, N_EMBD),
            p + "ffn_down_exps.weight": (N_EXPERT, N_EMBD, N_FF_EXP),
        })
        if i > 0:
            specs[p + "attn_res_query.weight"] = (1, N_EMBD)

    for il in CORE_LAYERS:
        p = f"blk.{il}."
        specs[p + "loop_msg_query.weight"] = (len(CTRL_NAMES), N_EMBD)
        for ctrl in CTRL_NAMES:
            specs[p + f"loop_{ctrl}_ctrl_down.weight"] = (GATE_RANK, 2 * N_EMBD)
            specs[p + f"loop_{ctrl}_ctrl_gain.weight"] = (N_EMBD, GATE_RANK)
            specs[p + f"loop_{ctrl}_ctrl_forget.weight"] = (N_EMBD, GATE_RANK)
    return specs


def build_tensors(seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def w(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * DIM_SCALE).astype(np.float32)

    return {name: w(*shape) for name, shape in tensor_specs().items()}


def add_metadata(writer: gguf.GGUFWriter) -> None:
    k = lambda key: f"{ARCH}.{key}"  # noqa: E731

    writer.add_name(MODEL_NAME)
    writer.add_uint32(k("context_length"), N_CTX)
    writer.add_uint32(k("embedding_length"), N_EMBD)
    writer.add_uint32(k("block_count"), N_LAYER)
    writer.add_uint32(k("expert_count"), N_EXPERT)
    writer.add_uint32(k("expert_used_count"), N_EXPERT_USED)
    writer.add_uint32(k("expert_feed_forward_length"), N_FF_EXP)
    writer.add_uint32(k("attention.head_count"), N_HEAD)
    writer.add_uint32(k("attention.head_count_kv"), N_HEAD_KV)

    writer.add_float32(k("attention.layer_norm_rms_epsilon"), 1e-5)
    writer.add_float32(k("attention.qk_clip"), 1.0)
    writer.add_uint32(k("attn_res.block_size"), N_CONV)
    writer.add_uint32(k("shortconv.l_cache"), N_CONV)
    writer.add_uint32(k("attention.relative_length"), N_REL)
    writer.add_uint32(k("attention.relative_extent"), N_EXTENT)
    writer.add_float32(k("attention.relative_clip"), 1.0)
    writer.add_uint32(k("attention.log_scale_floor"), 1)
    writer.add_float32(k("attention.log_scale_alpha"), 1.0)

    writer.add_uint32(k("vocab_size"), VOCAB_SIZE)
    writer.add_uint32(k("vocab_size_unpadded"), VOCAB_SIZE)
    writer.add_float32(k("activation.situ_beta"), 1.0)
    writer.add_float32(k("activation.situ_linear_beta"), 1.0)
    writer.add_float32(k("expert_weights_scale"), 1.0)

    writer.add_uint32(k("attention.sliding_window"), N_EXTENT)
    writer.add_array(k("attention.sliding_window_pattern"), [True] * N_LAYER)
    writer.add_uint32(k("rope.dimension_count"), 0)

    writer.add_uint32(k("loop.start"), LOOP_START)
    writer.add_uint32(k("loop.end"), LOOP_END)
    writer.add_uint32(k("loop.count"), N_PASS)
    writer.add_uint32(k("loop.gate_rank"), GATE_RANK)
    writer.add_bool(k("loop.varying"), False)

    writer.add_string("tokenizer.ggml.model", "test")
    tokens = ["<unk>", "<s>", "</s>"] + [f"t{i}" for i in range(VOCAB_SIZE - 3)]
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.token_type", [1] * VOCAB_SIZE)


def write_gguf(path: str, seed: int) -> dict:
    tensors = build_tensors(seed)
    writer = gguf.GGUFWriter(path, arch=ARCH)
    add_metadata(writer)
    for name, array in tensors.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: str, gguf_path: str, seed: int, tensors: dict) -> dict:
    manifest = {
        "generator": "make_moe_gguf.py",
        "seed": seed,
        "architecture": ARCH,
        "gguf_sha256": file_hash(gguf_path),
        "gguf_size_bytes": os.path.getsize(gguf_path),
        "constants": constants(),
        "tensor_count": len(tensors),
        "tensors": {
            name: {"shape": list(a.shape), "dtype": str(a.dtype)}
            for name, a in sorted(tensors.items())
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed (default: 0)")
    ap.add_argument("--manifest", default=None, help="manifest path (default: OUT.manifest.json)")
    args = ap.parse_args()

    tensors = write_gguf(args.out, args.seed)
    manifest_path = args.manifest or (args.out + ".manifest.json")
    manifest = write_manifest(manifest_path, args.out, args.seed, tensors)
    print("gguf_sha256 " + manifest["gguf_sha256"])
    print("gguf_size_bytes " + str(manifest["gguf_size_bytes"]))
    print("wrote " + os.path.abspath(args.out))
    print("wrote " + os.path.abspath(manifest_path))


if __name__ == "__main__":
    main()
