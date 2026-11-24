#!/usr/bin/env python3
"""Build a deterministic synthetic MoE fixture for the Inkling-2-Mini audit.

This is a *test vehicle*, not a model: every number comes from a seeded PRNG and
a local NVFP4 quantizer that is byte-for-byte identical to loopkit's emulator.
No Inkling weights are read, downloaded or embedded.

Usage:
    python make_moe_fixture.py --out FIXTURE.npz [--seed 0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Engine constants (source-labelled in loopkit/docs/INKLING_INTEGRATION.md).
# ---------------------------------------------------------------------------
N_LAYERS = 24          # physical layers
N_EXPERTS = 128        # experts per MoE layer
TOP_K = 8              # routed experts per token
N_PASSES = 3           # loop passes
GATE_RANK = 64         # rank of the gain/forget loop controllers
LOOP_CTRL_COUNT = 4    # LOOP_CTRL_Q, _K, _V, _FFN  (models.h:2536)
CORE_START = 6         # first looped physical layer
CORE_END = 18          # one past the last looped physical layer
N_SLOT = N_LAYERS + (N_PASSES - 1) * (CORE_END - CORE_START)  # 48

# Synthetic dimensions (kept small so the fixture/dump stay compact and fast).
EMB = 16               # n_embd
FF = 16                # expert intermediate size
N_TOKENS = 4           # tokens in the fixture
BLOCK = 16             # NVFP4 block size
ROUTE_SCALE = 8.0      # config: route_scale
RESIDUAL_SCALE = 0.1   # toy-forward residual gain (keeps the synthetic trace bounded)
CTRL_NAMES = ("q", "k", "v", "ffn")

FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Local NVFP4 emulator — identical to loopkit/src/loopkit/nvfp4.py.
# Kept here so fixture generation is numpy-only (no sibling-repo dependency).
# ---------------------------------------------------------------------------
FP4_LEVELS = np.array(
    [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)

_E4M3_LUT = np.zeros(256, dtype=np.float32)
_E4M3_VALID = np.zeros(256, dtype=bool)
for _c in range(256):
    _s = -1.0 if _c & 0x80 else 1.0
    _e = (_c >> 3) & 0x0F
    _m = _c & 0x07
    if _e == 0:
        if _m == 0:
            _E4M3_LUT[_c] = _s * 0.0
            _E4M3_VALID[_c] = True
        else:
            _E4M3_LUT[_c] = _s * (_m / 8.0) * (2.0 ** -6)
            _E4M3_VALID[_c] = True
    elif _e == 15 and _m == 7:
        continue  # NaN
    else:
        _E4M3_LUT[_c] = _s * (1.0 + _m / 8.0) * (2.0 ** (_e - 7))
        _E4M3_VALID[_c] = True
_VALID_CODES = np.flatnonzero(_E4M3_VALID)
_VALID_VALS = _E4M3_LUT[_VALID_CODES]


def e4m3_decode(code: np.ndarray) -> np.ndarray:
    return _E4M3_LUT[np.asarray(code, dtype=np.uint8)]


def e4m3_encode(values: np.ndarray) -> np.ndarray:
    """Nearest E4M3 code, ties-to-even, preserving signed zero."""
    v = np.asarray(values, dtype=np.float32)
    neg = np.signbit(v)
    a = np.abs(v).astype(np.float64)
    pos = _VALID_VALS >= 0
    tab = _VALID_VALS[pos]
    codes = _VALID_CODES[pos]
    idx = np.searchsorted(tab, a)
    idx = np.clip(idx, 1, len(tab) - 1)
    lo = np.abs(tab[idx - 1] - a)
    hi = np.abs(tab[idx] - a)
    pick = np.where(hi < lo, idx, np.where(lo < hi, idx - 1, idx % 2))
    out = codes[pick].astype(np.uint8)
    out[neg] |= 0x80
    out[a == 0] = np.where(neg[a == 0], 0x80, 0x00).astype(np.uint8)
    return out


@dataclass(frozen=True)
class NVFP4Tensor:
    codes: np.ndarray
    scales: np.ndarray
    shape: tuple[int, ...]
    block: int = 16


def _block_view(x: np.ndarray, block: int):
    last = x.shape[-1]
    pad = (-last) % block
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)]) if pad else x
    return xp.reshape(xp.shape[:-1] + (-1, block)), pad


def quantize_nvfp4(x: np.ndarray, block: int = 16) -> NVFP4Tensor:
    x = np.asarray(x, dtype=np.float32)
    blocks, _ = _block_view(x, block)
    absmax = np.max(np.abs(blocks), axis=-1)
    scale_codes = e4m3_encode(np.maximum(absmax / 6.0, 2.0 ** -9))
    scales = e4m3_decode(scale_codes)
    ratio = (blocks / scales[..., None]).astype(np.float64)
    codes = np.argmin(np.abs(ratio[..., None] - FP4_LEVELS), axis=-1).astype(np.uint8)
    codes = codes.reshape(x.shape[:-1] + (-1,))[..., : x.shape[-1]]
    return NVFP4Tensor(codes, scale_codes, x.shape, block)


def dequantize_nvfp4(t: NVFP4Tensor) -> np.ndarray:
    nb = (t.shape[-1] + t.block - 1) // t.block
    padded = np.zeros(t.shape[:-1] + (nb * t.block,), dtype=t.codes.dtype)
    padded[..., : t.shape[-1]] = t.codes.reshape(t.shape)
    vals = FP4_LEVELS[padded.reshape(t.shape[:-1] + (nb, t.block)).astype(np.intp)]
    out = (vals * e4m3_decode(t.scales)[..., None]).reshape(t.shape[:-1] + (-1,))[..., : t.shape[-1]]
    return out.reshape(t.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Deterministic helpers.
# ---------------------------------------------------------------------------
def constants() -> dict:
    return {
        "n_layers": N_LAYERS,
        "n_experts": N_EXPERTS,
        "top_k": TOP_K,
        "n_passes": N_PASSES,
        "gate_rank": GATE_RANK,
        "loop_ctrl_count": LOOP_CTRL_COUNT,
        "core_start": CORE_START,
        "core_end": CORE_END,
        "n_slot": N_SLOT,
        "emb": EMB,
        "ff": FF,
        "n_tokens": N_TOKENS,
        "nvfp4_block": BLOCK,
        "route_scale": ROUTE_SCALE,
        "residual_scale": RESIDUAL_SCALE,
        "ctrl_names": list(CTRL_NAMES),
        "format_version": FORMAT_VERSION,
    }


def canonical_hash(arrays: dict) -> str:
    """Content hash independent of .npz zip metadata / zlib version."""
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[key]))
        h.update(key.encode("utf-8") + b"\0")
        h.update(a.dtype.str.encode("ascii") + b"\0")
        h.update(str(a.shape).encode("ascii") + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_npz(path: str, arrays: dict) -> None:
    np.savez_compressed(path, **{k: arrays[k] for k in sorted(arrays)})


def load_npz(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in sorted(z.files)}


def build_fixture(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    x = rng.standard_normal((N_TOKENS, EMB)).astype(np.float32)
    router_w = (rng.standard_normal((N_LAYERS, N_EXPERTS, EMB)) * 0.5).astype(np.float32)

    n_loop = CORE_END - CORE_START
    msg_query = rng.standard_normal((n_loop, LOOP_CTRL_COUNT, EMB)).astype(np.float32)
    ctrl_down = rng.standard_normal((n_loop, LOOP_CTRL_COUNT, 2 * EMB, GATE_RANK)).astype(np.float32)
    ctrl_gain = (rng.standard_normal((n_loop, LOOP_CTRL_COUNT, GATE_RANK, EMB)) * 0.25).astype(np.float32)
    ctrl_forget = (rng.standard_normal((n_loop, LOOP_CTRL_COUNT, GATE_RANK, EMB)) * 0.25).astype(np.float32)

    c13 = np.empty((N_LAYERS, N_EXPERTS, 2 * FF, EMB), dtype=np.uint8)
    s13 = np.empty((N_LAYERS, N_EXPERTS, 2 * FF, EMB // BLOCK), dtype=np.uint8)
    c2 = np.empty((N_LAYERS, N_EXPERTS, EMB, FF), dtype=np.uint8)
    s2 = np.empty((N_LAYERS, N_EXPERTS, EMB, FF // BLOCK), dtype=np.uint8)
    for il in range(N_LAYERS):
        w13 = (rng.standard_normal((N_EXPERTS, 2 * FF, EMB)) * 0.4).astype(np.float32)
        w2 = (rng.standard_normal((N_EXPERTS, EMB, FF)) * 0.4).astype(np.float32)
        # loopkit's E4M3 encoder can emit a benign divide-by-zero warning when a
        # rare block rounds to a zero scale; values stay finite and identical.
        with np.errstate(divide="ignore", invalid="ignore"):
            t13 = quantize_nvfp4(w13, BLOCK)
            t2 = quantize_nvfp4(w2, BLOCK)
        c13[il], s13[il] = t13.codes, t13.scales
        c2[il], s2[il] = t2.codes, t2.scales

    return {
        "x": x,
        "router_w": router_w,
        "expert_w13_codes": c13,
        "expert_w13_scales": s13,
        "expert_w2_codes": c2,
        "expert_w2_scales": s2,
        "msg_query": msg_query,
        "ctrl_down": ctrl_down,
        "ctrl_gain": ctrl_gain,
        "ctrl_forget": ctrl_forget,
    }


def write_manifest(path: str, fixture: dict, seed: int, fixture_file_sha256: str) -> dict:
    manifest = {
        "generator": "make_moe_fixture.py",
        "seed": seed,
        "fixture_sha256": canonical_hash(fixture),
        "fixture_file_sha256": fixture_file_sha256,
        "constants": constants(),
        "arrays": {
            k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
            for k, v in sorted(fixture.items())
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed (default: 0)")
    ap.add_argument("--manifest", default=None, help="manifest path (default: OUT.manifest.json)")
    args = ap.parse_args()

    fixture = build_fixture(args.seed)
    save_npz(args.out, fixture)
    fhash = file_hash(args.out)
    manifest_path = args.manifest or (args.out + ".manifest.json")
    write_manifest(manifest_path, fixture, args.seed, fhash)
    print("fixture_sha256 " + canonical_hash(fixture))
    print("fixture_file_sha256 " + fhash)
    print("wrote " + os.path.abspath(args.out))
    print("wrote " + os.path.abspath(manifest_path))


if __name__ == "__main__":
    main()
