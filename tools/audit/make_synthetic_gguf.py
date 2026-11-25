#!/usr/bin/env python3
"""Build a tiny, fully synthetic Inkling2-loop GGUF (no checkpoint, no download).

This generator is for the issue #4 parity audit only. It uses the upstream
gguf-py writer API to emit every metadata key and tensor that
src/models/inkling2.cpp + src/models/inkling2-loop.cpp require, with
deterministic small random weights from numpy's PCG64. The tokenizer is the
built-in ``test`` tokenizer (hashed token ids), which avoids needing a real
GPT-2 vocabulary while still producing a loadable, runnable model.

Nothing here is a real Inkling weight. The model is intentionally tiny so a
CPU-only llama.cpp build can load and run all three loop passes in seconds.

Usage:
    PYTHONPATH=/tmp/build/inkengine-llama/gguf-py \
    python3 tools/audit/make_synthetic_gguf.py --out /tmp/ink-parity/synthetic.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gguf import GGUFWriter

ARCH = "inkling2_loop"

# Structural constants. 128 experts are kept (top-8 routing); all other
# dimensions are minimal but valid per load_arch_hparams() checks.
N_LAYER = 2          # physical layers: 0 = prelude, 1 = loop core
CORE_START = 1
CORE_END = 2
N_PASSES = 3
GATE_RANK = 8
N_EMBD = 32
N_HEAD = 2
N_HEAD_KV = 1
HEAD_DIM = 16
N_REL = 4
N_REL_EXTENT = 8
N_SWA = 8
N_CONV = 2
N_EXPERT = 128
N_EXPERT_USED = 8
N_FF_EXP = 8
ATTN_RES_BLOCK = 1
VOCAB_PADDED = 16
VOCAB_UNPADDED = 16
CTX = 64
LOOP_CTRL_COUNT = 4


def key(name: str) -> str:
    return f"{ARCH}.{name}"


def tensor_key(name: str) -> str:
    return f"tokenizer.ggml.{name}"


def build(out: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def rnd(ggml_shape: tuple[int, ...], scale: float = 0.02) -> np.ndarray:
        # GGUF stores dimensions reversed relative to numpy; GGUFWriter reverses
        # the numpy shape when writing, so allocate the reversed GGML shape.
        arr = rng.standard_normal(ggml_shape[::-1], dtype=np.float32) * scale
        return np.ascontiguousarray(arr)

    gw = GGUFWriter(out, ARCH)
    gw.add_name("synthetic-inkling2-loop-parity-audit")
    gw.add_file_type(0)  # LLAMA_FTYPE_ALL_F32
    gw.add_quantization_version(1)

    # General hparams from llama_model::load_hparams.
    gw.add_uint32(key("context_length"), CTX)
    gw.add_uint32(key("embedding_length"), N_EMBD)
    gw.add_uint32(key("block_count"), N_LAYER)
    gw.add_uint32(key("vocab_size"), VOCAB_PADDED)
    gw.add_uint32(key("vocab_size_unpadded"), VOCAB_UNPADDED)
    gw.add_uint32(key("expert_count"), N_EXPERT)
    gw.add_uint32(key("expert_used_count"), N_EXPERT_USED)
    gw.add_uint32(key("attention.head_count"), N_HEAD)
    gw.add_uint32(key("attention.head_count_kv"), N_HEAD_KV)
    gw.add_uint32(key("feed_forward_length"), N_FF_EXP)

    # llama_model_inkling2::load_arch_hparams.
    gw.add_float32(key("attention.layer_norm_rms_epsilon"), 1e-5)
    gw.add_float32(key("attention.qk_clip"), 10.0)
    gw.add_uint32(key("attn_res.block_size"), ATTN_RES_BLOCK)
    gw.add_uint32(key("shortconv.l_cache"), N_CONV)
    gw.add_uint32(key("attention.relative_length"), N_REL)
    gw.add_uint32(key("attention.relative_extent"), N_REL_EXTENT)
    gw.add_float32(key("attention.relative_clip"), 10.0)
    gw.add_uint32(key("attention.log_scale_floor"), 1)
    gw.add_float32(key("attention.log_scale_alpha"), 0.1)
    gw.add_float32(key("activation.situ_beta"), 1.0)
    gw.add_float32(key("activation.situ_linear_beta"), 1.0)
    gw.add_float32(key("expert_weights_scale"), 1.0)
    gw.add_float32(key("final_logit_softcapping"), 0.0)
    gw.add_uint32(key("expert_feed_forward_length"), N_FF_EXP)
    gw.add_uint32(key("attention.key_length_swa"), HEAD_DIM)
    gw.add_uint32(key("attention.value_length_swa"), HEAD_DIM)
    gw.add_uint32(key("rope.dimension_count"), 0)
    gw.add_uint32(key("attention.sliding_window"), N_SWA)
    # Layer 0 is full attention so `tau` (and therefore inp_pos) is consumed by
    # the graph; layer 1 is SWA and is the loop core.
    IS_SWA = [False, True]
    gw.add_array(key("attention.sliding_window_pattern"), IS_SWA)

    # llama_model_inkling2_loop::load_arch_hparams.
    gw.add_uint32(key("loop.start"), CORE_START)
    gw.add_uint32(key("loop.end"), CORE_END)
    gw.add_uint32(key("loop.count"), N_PASSES)
    gw.add_uint32(key("loop.gate_rank"), GATE_RANK)
    gw.add_bool(key("loop.varying"), False)

    # Minimal test tokenizer: no merges, no byte table needed.
    gw.add_tokenizer_model("test")
    gw.add_token_list([f"tok{i:03d}" for i in range(VOCAB_UNPADDED)])
    gw.add_token_types([1] * VOCAB_UNPADDED)
    gw.add_uint32(tensor_key("bos_token_id"), 0)
    gw.add_uint32(tensor_key("eos_token_id"), 1)
    gw.add_bool(tensor_key("add_bos_token"), False)
    gw.add_bool(tensor_key("add_eos_token"), False)

    # Embeddings and output head (llama_model_inkling2::load_arch_tensors).
    def add(name: str, shape: tuple[int, ...], scale: float = 0.02) -> None:
        gw.add_tensor(name, rnd(shape, scale))

    add("token_embd.weight", (N_EMBD, VOCAB_PADDED))
    add("token_embd_norm.weight", (N_EMBD,), 1.0)
    add("output.weight", (N_EMBD, VOCAB_PADDED))
    add("output_norm.weight", (N_EMBD,), 1.0)
    add("output_res_query.weight", (N_EMBD,))

    for i in range(N_LAYER):
        p = f"blk.{i}."
        add(p + "attn_q.weight", (N_EMBD, HEAD_DIM * N_HEAD))
        add(p + "attn_k.weight", (N_EMBD, HEAD_DIM * N_HEAD_KV))
        add(p + "attn_v.weight", (N_EMBD, HEAD_DIM * N_HEAD_KV))
        add(p + "attn_output.weight", (HEAD_DIM * N_HEAD, N_EMBD))
        add(p + "attn_gate.weight", (N_EMBD, HEAD_DIM * N_HEAD))
        add(p + "attn_q_clip.weight", (HEAD_DIM, N_HEAD), 1.0)
        add(p + "attn_sink_proj.weight", (N_EMBD, N_HEAD))
        add(p + "attn_rel.weight", (N_EMBD, N_REL * N_HEAD))
        add(p + "attn_rel_bias.weight", (N_REL, N_REL_EXTENT if not IS_SWA[i] else N_SWA))
        add(p + "attn_rel_clip.weight", (N_REL, N_HEAD), 1.0)
        add(p + "attn_k_conv.weight", (N_CONV, HEAD_DIM * N_HEAD_KV))
        add(p + "attn_v_conv.weight", (N_CONV, HEAD_DIM * N_HEAD_KV))
        add(p + "attn_post_conv.weight", (N_CONV, N_EMBD))
        add(p + "ffn_post_conv.weight", (N_CONV, N_EMBD))
        if IS_SWA[i]:
            add(p + "attn_norm.weight", (N_EMBD,), 1.0)
        else:
            add(p + "attn_q_inp_norm.weight", (N_EMBD,), 1.0)
            add(p + "attn_k_inp_norm.weight", (N_EMBD,), 1.0)
            add(p + "attn_v_inp_norm.weight", (N_EMBD,), 1.0)
            add(p + "attn_k_conv_proj.weight", (N_EMBD, N_CONV * N_HEAD_KV))
            add(p + "attn_v_conv_proj.weight", (N_EMBD, N_CONV * N_HEAD_KV))
        if i > 0:
            add(p + "attn_res_query.weight", (N_EMBD, 1 if IS_SWA[i] else 3))
        add(p + "ffn_res_query.weight", (N_EMBD,))
        add(p + "ffn_norm.weight", (N_EMBD,), 1.0)
        add(p + "ffn_gate_inp.weight", (N_EMBD, N_EXPERT))
        add(p + "exp_probs_b.bias", (N_EXPERT,), 0.0)
        add(p + "ffn_gate_up_exps.weight", (N_EMBD, 2 * N_FF_EXP, N_EXPERT))
        add(p + "ffn_down_exps.weight", (N_FF_EXP, N_EMBD, N_EXPERT))

    # Loop controller tensors (llama_model_inkling2_loop::load_arch_tensors).
    for i in range(CORE_START, CORE_END):
        p = f"blk.{i}."
        add(p + "loop_msg_query.weight", (N_EMBD, LOOP_CTRL_COUNT))
        for ctrl in ("q", "k", "v", "ffn"):
            add(p + f"loop_{ctrl}_ctrl_down.weight", (2 * N_EMBD, GATE_RANK))
            add(p + f"loop_{ctrl}_ctrl_gain.weight", (GATE_RANK, N_EMBD))
            add(p + f"loop_{ctrl}_ctrl_forget.weight", (GATE_RANK, N_EMBD))

    out.parent.mkdir(parents=True, exist_ok=True)
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "generator": "tools/audit/make_synthetic_gguf.py",
        "seed": seed,
        "sha256": digest,
        "size_bytes": out.stat().st_size,
        "architecture": ARCH,
        "n_layer": N_LAYER,
        "core_start": CORE_START,
        "core_end": CORE_END,
        "n_passes": N_PASSES,
        "gate_rank": GATE_RANK,
        "n_expert": N_EXPERT,
        "n_expert_used": N_EXPERT_USED,
        "n_ff_exp": N_FF_EXP,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_head_kv": N_HEAD_KV,
        "head_dim": HEAD_DIM,
        "vocab_padded": VOCAB_PADDED,
        "vocab_unpadded": VOCAB_UNPADDED,
        "n_tensors": sum(len(t) for t in gw.tensors),
        "tokenizer": "test",
        "real_weights": False,
    }
    (out.with_suffix(out.suffix + ".manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = build(args.out, args.seed)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
