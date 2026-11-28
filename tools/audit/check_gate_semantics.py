#!/usr/bin/env python3
"""Offline, fail-closed gate-semantics check for the rank-64 Inkling2 loop controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
FACTS_PATH = AUDIT_DIR / "GATE_SEMANTICS.json"
SUMMARY_PATH = AUDIT_DIR / "M30_GATE_SEMANTICS_SUMMARY.json"
NOTE = "gate semantics only, no timing"
EPS = 1e-6
EXPECTED_CTRL_STREAMS = ["q", "k", "v", "ffn"]
TRUSTED_MARKERS = {
    "mix_formula": ("src/models/inkling2.cpp",
        "ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message))"),
    "tanh_gates": ("src/models/inkling2.cpp", "ggml_tanh"),
    "rms_norm_cur": ("src/models/inkling2.cpp", "ggml_rms_norm(ctx0, cur, 1e-6f)"),
    "rms_norm_message": ("src/models/inkling2.cpp", "ggml_rms_norm(ctx0, message, 1e-6f)"),
    "ctrl_down": ("src/models/inkling2.cpp", "ctrl_down"),
    "ctrl_gain": ("src/models/inkling2.cpp", "ctrl_gain"),
    "ctrl_forget": ("src/models/inkling2.cpp", "ctrl_forget"),
    "gate_rank": ("src/models/inkling2-loop.cpp", "LLM_KV_LOOP_GATE_RANK, gate_rank"),
}
TRUSTED_OP_SEQUENCE = [
    "rms_norm(cur, 1e-6) and rms_norm(message, 1e-6)",
    "concat(norm_cur, norm_message, axis=0)",
    "gate = down_weight @ concat",
    "gain = tanh(gain_weight @ gate) and forget = tanh(forget_weight @ gate)",
    "out = (1-forget)*cur + gain*message",
]
TRUSTED_CTRL_TENSORS = {
    "q": ["LOOP_Q_CTRL_DOWN", "LOOP_Q_CTRL_GAIN", "LOOP_Q_CTRL_FORGET"],
    "k": ["LOOP_K_CTRL_DOWN", "LOOP_K_CTRL_GAIN", "LOOP_K_CTRL_FORGET"],
    "v": ["LOOP_V_CTRL_DOWN", "LOOP_V_CTRL_GAIN", "LOOP_V_CTRL_FORGET"],
    "ffn": ["LOOP_FFN_CTRL_DOWN", "LOOP_FFN_CTRL_GAIN", "LOOP_FFN_CTRL_FORGET"],
}
FROZEN_SOURCE_SHA256 = {
    "src/models/inkling2.cpp": "c5b543ad965857efd7f9818d42d1f7a418954758b75ec68c82485ffbd341225f",
    "src/models/inkling2-loop.cpp": "a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab",
    "../loopkit/docs/INKLING_INTEGRATION.md": "69c587aba7ebbe3b866b7ca95107cfb891b28b3b30d3c54eeba698f878655432",
}
GATED_SOURCE_FILES = ["src/models/inkling2.cpp", "src/models/inkling2-loop.cpp"]
CHECK_NAMES = [
    "frozen_facts", "frozen_source_sha256", "gate_rank_fact", "live_source_sha256",
    "marker_mix_formula", "marker_tanh_gates", "marker_rms_norm_cur", "marker_rms_norm_message",
    "marker_ctrl_down", "marker_ctrl_gain", "marker_ctrl_forget", "marker_gate_rank",
    "zero_weights_identity", "tanh_bounded", "tanh_bounded_very_large",
    "scalar_gap_exceeds_tolerance", "forget_per_channel", "slow_reference_bitwise",
    "determinism", "negative_corrupt_shapes", "negative_unbounded_gate",
    "negative_scalar_substitution", "run_completed",
]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path):
    return path.read_text(encoding="utf-8")


def normalize_whitespace(text):
    return " ".join(text.split())


def validate_mix_shapes(cur, msg, down, gain_w, forget_w):
    arrays = (("cur", cur), ("msg", msg), ("down", down), ("gain_w", gain_w), ("forget_w", forget_w))
    for name, arr in arrays:
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError("%s must be a 2-D numpy array" % name)
    rank, two_d = down.shape
    if two_d < 2 or two_d % 2:
        raise ValueError("down second dimension must be even and positive")
    d = two_d // 2
    n = cur.shape[1]
    if d < 1 or rank < 1 or n < 1:
        raise ValueError("empty channel, rank or token dimension")
    if cur.shape != msg.shape:
        raise ValueError("cur and msg shapes must match")
    if cur.shape[0] != d:
        raise ValueError("cur channels must equal down second dimension / 2")
    expected = (d, rank)
    if gain_w.shape != expected or forget_w.shape != expected:
        raise ValueError("gain_w and forget_w shapes must be (%d, %d)" % expected)
    return d, rank, n


def rms_norm(x, eps):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or eps <= 0:
        raise ValueError("rms_norm needs a 2-D array and positive eps")
    scale = 1.0 / np.sqrt(np.mean(np.square(x), axis=0) + eps)
    return x * scale


def gate_preactivations(cur, msg, down, gain_w, forget_w):
    validate_mix_shapes(cur, msg, down, gain_w, forget_w)
    norm_cur = rms_norm(cur, EPS)
    norm_msg = rms_norm(msg, EPS)
    gate = np.column_stack([
        np.dot(down, np.concatenate((norm_cur[:, t], norm_msg[:, t])))
        for t in range(cur.shape[1])
    ])
    gain_pre = np.column_stack([np.dot(gain_w, gate[:, t]) for t in range(gate.shape[1])])
    forget_pre = np.column_stack([np.dot(forget_w, gate[:, t]) for t in range(gate.shape[1])])
    return gain_pre, forget_pre


def rank64_mix(cur, msg, down, gain_w, forget_w):
    gain_pre, forget_pre = gate_preactivations(cur, msg, down, gain_w, forget_w)
    gain = np.tanh(gain_pre)
    forget = np.tanh(forget_pre)
    return (1.0 - forget) * cur + gain * msg


def scalar_gamma_mix(cur, msg, gamma):
    return (1.0 - gamma) * cur + gamma * msg


def slow_reference_mix(cur, msg, down, gain_w, forget_w):
    d, rank, n = validate_mix_shapes(cur, msg, down, gain_w, forget_w)
    out = np.empty((d, n), dtype=np.float64)
    for t in range(n):
        cur_col = rms_norm(cur[:, t:t + 1], EPS)[:, 0]
        msg_col = rms_norm(msg[:, t:t + 1], EPS)[:, 0]
        joined = np.concatenate((cur_col, msg_col))
        gate = np.array([np.dot(down[r], joined) for r in range(rank)])
        for c in range(d):
            gain = np.tanh(np.dot(gain_w[c], gate))
            forget = np.tanh(np.dot(forget_w[c], gate))
            out[c, t] = (1.0 - forget) * cur[c, t] + gain * msg[c, t]
    return out


def best_scalar_gamma(out, cur, msg):
    delta = (msg - cur).ravel()
    denom = float(np.dot(delta, delta))
    if denom == 0.0:
        raise ValueError("msg equals cur; scalar gamma is undefined")
    gamma = float(np.dot((out - cur).ravel(), delta) / denom)
    scalar_out = scalar_gamma_mix(cur, msg, gamma)
    gap = float(np.max(np.abs(out - scalar_out)))
    return gamma, scalar_out, gap


def gates_strictly_bounded(gain, forget):
    return bool(
        np.all(np.isfinite(gain)) and np.all(np.isfinite(forget))
        and np.all(gain > -1.0) and np.all(gain < 1.0)
        and np.all(forget > -1.0) and np.all(forget < 1.0)
    )


def gates_match_tanh(gain_hat, forget_hat, gain_pre, forget_pre):
    return bool(
        np.allclose(gain_hat, np.tanh(gain_pre), rtol=1e-12, atol=1e-12)
        and np.allclose(forget_hat, np.tanh(forget_pre), rtol=1e-12, atol=1e-12)
    )


def recover_gates_from_mix(cur, msg, down, gain_w, forget_w):
    if not np.all(np.abs(cur) > 1e-12) or not np.all(np.abs(msg) > 1e-12):
        raise ValueError("gate recovery needs nonzero cur and msg")
    gain_pre, forget_pre = gate_preactivations(cur, msg, down, gain_w, forget_w)
    out_forget = rank64_mix(cur, msg, down, np.zeros_like(gain_w), forget_w)
    out_gain = rank64_mix(cur, msg, down, gain_w, np.zeros_like(forget_w))
    # zero gain gives out = (1-forget)*cur; zero forget gives out = cur + gain*msg
    forget_hat = 1.0 - out_forget / cur
    gain_hat = (out_gain - cur) / msg
    return gain_hat, forget_hat, gain_pre, forget_pre


def make_inputs(seed, n_embd, rank, n_tokens, down_scale, gate_weight_scale):
    rng = np.random.default_rng(seed)
    cur = rng.standard_normal((n_embd, n_tokens))
    msg = rng.standard_normal((n_embd, n_tokens))
    down = rng.standard_normal((rank, 2 * n_embd)) * down_scale
    gain_w = rng.standard_normal((n_embd, rank)) * gate_weight_scale
    forget_w = rng.standard_normal((n_embd, rank)) * gate_weight_scale
    return cur, msg, down, gain_w, forget_w


def check_zero_weights(cur, msg, down, gain_w, forget_w):
    zero_down = np.zeros_like(down)
    zero_gain = np.zeros_like(gain_w)
    zero_forget = np.zeros_like(forget_w)
    out = rank64_mix(cur, msg, zero_down, zero_gain, zero_forget)
    return bool(np.array_equal(out, cur))


def check_tanh_bounded(cur, msg, down, gain_w, forget_w):
    gain_hat, forget_hat, gain_pre, forget_pre = recover_gates_from_mix(
        cur, msg, down, gain_w, forget_w)
    exercised = bool(max(np.max(np.abs(gain_hat)), np.max(np.abs(forget_hat))) > 0.9)
    return bool(
        gates_match_tanh(gain_hat, forget_hat, gain_pre, forget_pre)
        and gates_strictly_bounded(gain_hat, forget_hat) and exercised
    )


def check_tanh_very_large(cur, msg, down, gain_w, forget_w, scale):
    scaled = tuple(arr * scale for arr in (down, gain_w, forget_w))
    gain_hat, forget_hat, gain_pre, forget_pre = recover_gates_from_mix(cur, msg, *scaled)
    return bool(
        gates_match_tanh(gain_hat, forget_hat, gain_pre, forget_pre)
        and np.all(np.isfinite(gain_hat)) and np.all(np.isfinite(forget_hat))
        and np.all(np.abs(gain_hat) <= 1.0 + 1e-12)
        and np.all(np.abs(forget_hat) <= 1.0 + 1e-12)
    )


def check_scalar_gap(cur, msg, down, gain_w, forget_w, tol):
    out = rank64_mix(cur, msg, down, gain_w, forget_w)
    _, _, gap = best_scalar_gamma(out, cur, msg)
    return bool(gap > tol)


def check_forget_per_channel(cur, msg, down, forget_w, tol):
    out = rank64_mix(cur, msg, down, np.zeros_like(forget_w), forget_w)
    _, _, gap = best_scalar_gamma(out, cur, msg)
    valid = np.abs(cur) > 1e-12
    ratios = out[valid] / cur[valid]
    spread = float(np.max(ratios) - np.min(ratios))
    return bool(not np.array_equal(out, cur) and gap > tol and spread > tol)


def check_slow_reference(cur, msg, down, gain_w, forget_w):
    fast = rank64_mix(cur, msg, down, gain_w, forget_w)
    slow = slow_reference_mix(cur, msg, down, gain_w, forget_w)
    return bool(np.array_equal(fast, slow))


def check_determinism(cfg):
    first = rank64_mix(*make_inputs(**cfg))
    second = rank64_mix(*make_inputs(**cfg))
    return bool(np.array_equal(first, second))


def check_corrupt_shapes(cur, msg, down, gain_w, forget_w):
    bad_cases = [
        (cur, msg, down[:, :-1], gain_w, forget_w),
        (cur, msg, down, gain_w[:-1], forget_w),
        (cur[:-1], msg[:-1], down, gain_w, forget_w),
        (cur, msg[:, :-1], down, gain_w, forget_w),
    ]
    for case in bad_cases:
        try:
            rank64_mix(*case)
        except ValueError:
            continue
        return False
    return True


def check_unbounded_gate_negative(cur, msg, down, gain_w, forget_w):
    gain_hat, forget_hat, gain_pre, forget_pre = recover_gates_from_mix(
        cur, msg, down, gain_w, forget_w)
    return bool(
        not gates_strictly_bounded(gain_pre, forget_pre)
        and gates_strictly_bounded(gain_hat, forget_hat)
        and gates_match_tanh(gain_hat, forget_hat, gain_pre, forget_pre)
    )


def check_scalar_substitution_negative(cur, msg, gamma=0.25):
    out = scalar_gamma_mix(cur, msg, gamma)
    _, _, gap = best_scalar_gamma(out, cur, msg)
    return bool(gap <= 1e-9)


def numeric_checks(facts):
    cfg = dict(facts["test_config"])
    input_cfg = {key: cfg[key] for key in (
        "seed", "n_embd", "rank", "n_tokens", "down_scale", "gate_weight_scale")}
    cur, msg, down, gain_w, forget_w = make_inputs(**input_cfg)
    tol = cfg["scalar_gap_tol"]
    checks = {
        "zero_weights_identity": check_zero_weights(cur, msg, down, gain_w, forget_w),
        "tanh_bounded": check_tanh_bounded(cur, msg, down, gain_w, forget_w),
        "tanh_bounded_very_large": check_tanh_very_large(
            cur, msg, down, gain_w, forget_w, cfg["very_large_scale"]),
        "scalar_gap_exceeds_tolerance": check_scalar_gap(cur, msg, down, gain_w, forget_w, tol),
        "forget_per_channel": check_forget_per_channel(cur, msg, down, forget_w, cfg["per_channel_tol"]),
        "slow_reference_bitwise": check_slow_reference(cur, msg, down, gain_w, forget_w),
        "determinism": check_determinism(input_cfg),
        "negative_corrupt_shapes": check_corrupt_shapes(cur, msg, down, gain_w, forget_w),
        "negative_unbounded_gate": check_unbounded_gate_negative(cur, msg, down, gain_w, forget_w),
        "negative_scalar_substitution": check_scalar_substitution_negative(cur, msg),
    }
    out = rank64_mix(cur, msg, down, gain_w, forget_w)
    gamma, _, gap = best_scalar_gamma(out, cur, msg)
    info = [
        "small test shape: d=%d rank=%d tokens=%d (frozen contract is d=%d rank=%d)"
        % (cfg["n_embd"], cfg["rank"], cfg["n_tokens"], facts["n_embd"], facts["gate_rank"]),
        "best scalar gamma %.6f leaves max abs gap %.6f (tol %.6f)" % (gamma, gap, tol),
        "tanh strict bound checked below float64 saturation; at scale %.0f gates are finite and <= 1 in abs"
        % cfg["very_large_scale"],
    ]
    return checks, info


def check_string_list(actual, trusted):
    if not isinstance(actual, list) or len(actual) != len(trusted):
        return False
    normalized = [normalize_whitespace(item) if isinstance(item, str) else item for item in actual]
    return normalized == [normalize_whitespace(item) for item in trusted]


def check_frozen_facts(facts):
    markers = facts.get("source_markers", [])
    names = [m.get("name") if isinstance(m, dict) else None for m in markers] if isinstance(markers, list) else []
    markers_ok = (
        isinstance(markers, list)
        and len(markers) == len(TRUSTED_MARKERS)
        and names == list(TRUSTED_MARKERS)
        and all(
            m.get("file") == TRUSTED_MARKERS[m["name"]][0]
            and isinstance(m.get("needle"), str)
            and normalize_whitespace(m["needle"]) == normalize_whitespace(TRUSTED_MARKERS[m["name"]][1])
            for m in markers if isinstance(m, dict) and m.get("name") in TRUSTED_MARKERS
        )
    )
    ctrl_tensors = facts.get("ctrl_tensors", {})
    tensors_ok = (
        isinstance(ctrl_tensors, dict)
        and sorted(ctrl_tensors) == sorted(TRUSTED_CTRL_TENSORS)
        and all(check_string_list(ctrl_tensors.get(stream), trusted)
                for stream, trusted in TRUSTED_CTRL_TENSORS.items())
    )
    return bool(
        facts.get("gate_rank") == 64
        and facts.get("n_embd") == 2304
        and facts.get("down_shape") == [64, 4608]
        and facts.get("gain_shape") == [2304, 64]
        and facts.get("forget_shape") == [2304, 64]
        and facts.get("dtype") == "F32"
        and facts.get("eps") == EPS
        and facts.get("formula") == "(1-forget)*cur + gain*message"
        and facts.get("ctrl_streams") == EXPECTED_CTRL_STREAMS
        and facts.get("loop") == {"core_start": 6, "core_end": 18, "num_passes": 3}
        and check_string_list(facts.get("op_sequence"), TRUSTED_OP_SEQUENCE)
        and markers_ok
        and tensors_ok
    )


def check_frozen_sha256(facts):
    return facts.get("provenance", {}).get("source_sha256") == FROZEN_SOURCE_SHA256


def read_markers(root, markers):
    texts = {}
    result = {}
    for marker in markers:
        rel = marker["file"]
        if rel not in texts:
            path = root / rel
            texts[rel] = normalize_whitespace(read_text(path)) if path.is_file() else ""
        needle = normalize_whitespace(marker["needle"]) if isinstance(marker.get("needle"), str) else ""
        result[marker["name"]] = bool(needle) and needle in texts[rel]
    return result


def run_check(root):
    checks = {name: False for name in CHECK_NAMES}
    info = []
    provenance = {}
    frozen_on = None
    frozen_facts_sha256 = None
    try:
        facts_path = root / "tools/audit/GATE_SEMANTICS.json"
        frozen_facts_sha256 = hashlib.sha256(facts_path.read_bytes()).hexdigest()
        facts = load_json(facts_path)
        frozen_on = facts["frozen_on"]
        provenance = dict(facts.get("provenance", {}))
        checks["frozen_facts"] = check_frozen_facts(facts)
        checks["frozen_source_sha256"] = check_frozen_sha256(facts)
        checks["gate_rank_fact"] = bool(facts.get("gate_rank") == 64)
        markers = read_markers(root, facts["source_markers"])
        for name, present in markers.items():
            checks["marker_" + name] = bool(present)
        live = {}
        for rel, digest in sorted(FROZEN_SOURCE_SHA256.items()):
            path = root / rel
            if path.is_file():
                live[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            info.append("sha256 %s frozen=%s live=%s (%s)"
                        % (rel, digest, live.get(rel, "missing"),
                           "gated" if rel in GATED_SOURCE_FILES else "INFO only; not gated"))
        info.append("frozen facts sha256 %s (INFO only; semantic facts are gated)" % frozen_facts_sha256)
        provenance["source_sha256_live"] = live
        checks["live_source_sha256"] = all(
            live.get(rel) == digest
            for rel, digest in FROZEN_SOURCE_SHA256.items() if rel in GATED_SOURCE_FILES
        )
        numeric, numeric_info = numeric_checks(facts)
        checks.update(numeric)
        info.extend(numeric_info)
        checks["run_completed"] = True
    except Exception as exc:
        info.append("exception: %s: %s" % (type(exc).__name__, exc))
    verdict = "PASS" if all(checks.values()) else "NO_GO"
    blocker = "" if verdict == "PASS" else ", ".join(name for name in sorted(checks) if not checks[name])
    summary = {
        "verdict": verdict,
        "blocker": blocker,
        "checks": checks,
        "info": info,
        "provenance": provenance,
        "frozen_source_sha256": FROZEN_SOURCE_SHA256,
        "frozen_facts_sha256": frozen_facts_sha256,
        "frozen_on": frozen_on,
        "note": NOTE,
        "speed_claim": None,
    }
    write_json(root / "tools/audit/M30_GATE_SEMANTICS_SUMMARY.json", summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    if summary["blocker"]:
        print("blocker: " + summary["blocker"])
    for name in sorted(summary["checks"]):
        print("  %-32s %s" % (name, "PASS" if summary["checks"][name] else "FAIL"))
    for line in summary["info"]:
        print("  INFO: " + line)
    print("note: " + summary["note"])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv=None):
    parse_args(argv)
    summary = run_check(REPO_ROOT)
    print_summary(summary)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
