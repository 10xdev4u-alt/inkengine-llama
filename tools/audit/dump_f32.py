#!/usr/bin/env python3
"""Dump F32 intermediates from a synthetic MoE fixture for bit-exact audits.

The fixture is produced by make_moe_fixture.py. For every (layer, pass, token)
this writes router logits, top-k ids, dequantized selected expert weights,
expert outputs, the cross-loop message mix and the final layer output, plus a
per-(layer, pass, expert) route-traffic count.

Dequantization uses loopkit's NVFP4 emulator when it is importable (e.g.
PYTHONPATH=/tmp/build/loopkit/src), otherwise the byte-identical local fallback
in make_moe_fixture.py. Both paths produce the same dump SHA-256.

Usage:
    python dump_f32.py --fixture FIXTURE.npz --out dump.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_moe_fixture as mk  # noqa: E402  (script dir inserted above)

# Prefer loopkit's emulator when the sibling repo is on PYTHONPATH; the local
# fallback is identical and keeps this tool usable stand-alone.
try:
    from loopkit import nvfp4 as _nvfp4  # type: ignore
    NVFP4_SOURCE = "loopkit"
except Exception:  # pragma: no cover - exercised only without PYTHONPATH
    _nvfp4 = mk
    NVFP4_SOURCE = "local-fallback"


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * scale


def matvec(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Broadcasting matvec with a fixed summation order (no BLAS reduction)."""
    return np.sum(np.asarray(w, dtype=np.float64) * np.asarray(x, dtype=np.float64)[..., None, :], axis=-1)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def bounded_act(x: np.ndarray) -> np.ndarray:
    """Bounded synthetic FFN activation (tanh keeps the toy forward stable)."""
    return np.tanh(x)


def build_dump(fixture: dict) -> dict:
    L = mk.N_LAYERS
    N = mk.N_EXPERTS
    K = mk.TOP_K
    P = mk.N_PASSES
    R = mk.GATE_RANK
    C = mk.LOOP_CTRL_COUNT
    E = mk.EMB
    F = mk.FF
    T = mk.N_TOKENS
    start, end = mk.CORE_START, mk.CORE_END
    ffn = mk.CTRL_NAMES.index("ffn")
    n_loop = end - start

    logits = np.zeros((L, P, T, N), dtype=np.float32)
    topk = np.zeros((L, P, T, K), dtype=np.int32)
    sel13 = np.zeros((L, P, T, K, 2 * F, E), dtype=np.float32)
    sel2 = np.zeros((L, P, T, K, E, F), dtype=np.float32)
    expert_out = np.zeros((L, P, T, K, E), dtype=np.float32)
    message_mix = np.zeros((L, P, T, C, E), dtype=np.float32)
    layer_out = np.zeros((L, P, T, E), dtype=np.float32)
    traffic = np.zeros((L, P, N), dtype=np.int32)

    # Dequantize every expert once with the selected emulator.
    w13 = _nvfp4.dequantize_nvfp4(
        mk.NVFP4Tensor(fixture["expert_w13_codes"], fixture["expert_w13_scales"],
                       (L, N, 2 * F, E), mk.BLOCK)
    )
    w2 = _nvfp4.dequantize_nvfp4(
        mk.NVFP4Tensor(fixture["expert_w2_codes"], fixture["expert_w2_scales"],
                       (L, N, E, F), mk.BLOCK)
    )
    router_w = fixture["router_w"].astype(np.float64)
    x = fixture["x"].astype(np.float64)

    prelude = None                 # cur after the first block (layer CORE_START-1)
    prev_core_states = None        # previous pass block outputs for the message bank
    prev_bank_out = None           # previous pass output (last bank state)

    for p in range(P):
        cur = x.copy() if p == 0 else prelude.copy()
        core_states = []
        bank_out = None
        for il in range(L):
            if start <= il < end and p > 0:
                # Upstream mix(): softmax attention over the previous pass's
                # core block outputs + its output, using 4 learned queries.
                bank = prev_core_states[1:]  # layers start+1 .. end-1
                states = np.concatenate([bank, prev_bank_out[None, :, :]], axis=0)  # (n_states, E)
                states = rms_norm(states)
                q = fixture["msg_query"][il - start].astype(np.float64)  # (C, E)
                raw = np.sum(states[None, :, :, :] * q[:, None, None, :], axis=-1)  # (C, n_states, T)
                scores = softmax(raw / np.sqrt(E), axis=1)
                message = np.sum(scores[:, :, :, None] * states[None, :, :, :], axis=1)  # (C, T, E)
                message_mix[il, p] = np.transpose(message, (1, 0, 2)).astype(np.float32)

                # Upstream control(): (1-forget)*cur + gain*message, rank-r gates.
                c = il - start
                gate_in = np.concatenate([rms_norm(cur), rms_norm(message[ffn])], axis=-1)
                down = np.tanh(matvec(fixture["ctrl_down"][c, ffn].T, gate_in))        # (T, R)
                gain = np.tanh(matvec(fixture["ctrl_gain"][c, ffn].T, down))           # (T, E)
                forget = np.tanh(matvec(fixture["ctrl_forget"][c, ffn].T, down))       # (T, E)
                cur = (1.0 - forget) * cur + gain * message[ffn]

            z = rms_norm(cur)
            lg = matvec(router_w[il], z)  # (T, N)
            logits[il, p] = lg.astype(np.float32)
            ids = np.argsort(-lg, axis=-1, kind="stable")[:, :K]
            topk[il, p] = ids.astype(np.int32)
            probs = softmax(np.take_along_axis(lg, ids, axis=-1) * mk.ROUTE_SCALE, axis=-1)

            ysum = np.zeros_like(cur)
            w13_i = w13[il].astype(np.float64)
            w2_i = w2[il].astype(np.float64)
            for t in range(T):
                for k in range(K):
                    e = int(ids[t, k])
                    traffic[il, p, e] += 1
                    w1 = w13_i[e]
                    h = matvec(w1[:F], cur[t])
                    g = matvec(w1[F:], cur[t])
                    a = bounded_act(h) * bounded_act(g)
                    y = matvec(w2_i[e], a)
                    sel13[il, p, t, k] = w1.astype(np.float32)
                    sel2[il, p, t, k] = w2_i[e].astype(np.float32)
                    expert_out[il, p, t, k] = y.astype(np.float32)
                    ysum[t] += probs[t, k] * y

            cur = cur + mk.RESIDUAL_SCALE * ysum
            layer_out[il, p] = cur.astype(np.float32)

            if il < start:
                if il == start - 1:
                    prelude = cur.copy()
            elif il < end:
                core_states.append(cur.copy())
                if il == end - 1:
                    bank_out = cur.copy()
        prev_core_states = np.stack(core_states, axis=0)  # (end-start, T, E)
        prev_bank_out = bank_out

    return {
        "router_logits": logits,
        "topk_ids": topk,
        "selected_w13": sel13,
        "selected_w2": sel2,
        "expert_outputs": expert_out,
        "message_mix": message_mix,
        "layer_output": layer_out,
        "route_traffic": traffic,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True, help="fixture .npz from make_moe_fixture.py")
    ap.add_argument("--out", required=True, help="output dump .npz path")
    args = ap.parse_args()

    fixture = mk.load_npz(args.fixture)
    dump = build_dump(fixture)
    mk.save_npz(args.out, dump)
    print("nvfp4_source " + NVFP4_SOURCE)
    print("dump_sha256 " + mk.canonical_hash(dump))
    print("dump_file_sha256 " + mk.file_hash(args.out))
    print("wrote " + os.path.abspath(args.out))


if __name__ == "__main__":
    main()
