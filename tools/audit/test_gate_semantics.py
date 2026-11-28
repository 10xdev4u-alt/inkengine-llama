"""Tests for the frozen rank-64 Inkling2 gate-semantics checker."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
sys.path.insert(0, str(AUDIT_DIR))

import check_gate_semantics as checker


@pytest.fixture()
def frozen_facts():
    return checker.load_json(AUDIT_DIR / "GATE_SEMANTICS.json")


def inputs_from(facts):
    cfg = facts["test_config"]
    keys = ("seed", "n_embd", "rank", "n_tokens", "down_scale", "gate_weight_scale")
    return checker.make_inputs(**{key: cfg[key] for key in keys})


def make_fixture_tree(tmp_path, facts, omit=None):
    audit = tmp_path / "tools/audit"
    audit.mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "GATE_SEMANTICS.json", audit / "GATE_SEMANTICS.json")
    for rel in checker.GATED_SOURCE_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, path)
    if omit is not None:
        marker = next(m for m in facts["source_markers"] if m["name"] == omit)
        path = tmp_path / marker["file"]
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(marker["needle"], "OMITTED_MARKER"), encoding="utf-8")
    return audit


def write_tampered_facts(audit, facts, mutate):
    payload = json.loads(json.dumps(facts))
    mutate(payload)
    (audit / "GATE_SEMANTICS.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_stale_pass(audit):
    checker.write_json(audit / "M30_GATE_SEMANTICS_SUMMARY.json", {"verdict": "PASS"})


def summary_path(tmp_path):
    return tmp_path / "tools/audit/M30_GATE_SEMANTICS_SUMMARY.json"


def test_frozen_json_shape(frozen_facts):
    facts = frozen_facts
    assert facts["schema_version"] == 1
    assert facts["frozen_on"] == "2026-11-28"
    assert facts["scope"] == "gate semantics only, no timing"
    assert facts["note"] == "gate semantics only, no timing"
    assert facts["speed_claim"] is None
    assert facts["gate_rank"] == 64
    assert facts["n_embd"] == 2304
    assert facts["down_shape"] == [64, 4608]
    assert facts["gain_shape"] == [2304, 64]
    assert facts["forget_shape"] == [2304, 64]
    assert facts["dtype"] == "F32"
    assert facts["eps"] == 1e-6
    assert facts["formula"] == "(1-forget)*cur + gain*message"
    assert len(facts["op_sequence"]) == 5
    assert facts["ctrl_streams"] == ["q", "k", "v", "ffn"]
    assert facts["loop"] == {"core_start": 6, "core_end": 18, "num_passes": 3}
    assert sorted(facts["ctrl_tensors"]) == ["ffn", "k", "q", "v"]
    for names in facts["ctrl_tensors"].values():
        assert len(names) == 3
    assert [m["name"] for m in facts["source_markers"]] == list(checker.TRUSTED_MARKERS)
    for marker in facts["source_markers"]:
        rel, trusted_needle = checker.TRUSTED_MARKERS[marker["name"]]
        assert marker["file"] == rel
        assert checker.normalize_whitespace(marker["needle"]) == checker.normalize_whitespace(trusted_needle)
    norm = checker.normalize_whitespace
    assert [norm(s) for s in facts["op_sequence"]] == [norm(s) for s in checker.TRUSTED_OP_SEQUENCE]
    assert facts["ctrl_tensors"] == checker.TRUSTED_CTRL_TENSORS
    assert "1e-6f" in checker.TRUSTED_MARKERS["rms_norm_cur"][1]
    assert "1e-6f" in checker.TRUSTED_MARKERS["rms_norm_message"][1]
    assert "ggml_mul(ctx0, gain, message)" in checker.TRUSTED_MARKERS["mix_formula"][1]
    assert "not a universal proof" in facts["scalar_comparison_scope"]
    assert facts["provenance"]["source_sha256"] == checker.FROZEN_SOURCE_SHA256
    assert facts["test_config"]["n_embd"] == 8
    assert facts["test_config"]["rank"] == 4


def test_real_fork_sources_match_frozen_markers(frozen_facts):
    facts = frozen_facts
    markers = checker.read_markers(REPO_ROOT, facts["source_markers"])
    assert markers and all(markers.values())
    for rel, digest in checker.FROZEN_SOURCE_SHA256.items():
        path = REPO_ROOT / rel
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert checker.check_frozen_facts(facts) is True
    assert checker.check_frozen_sha256(facts) is True


def test_rms_norm_definition():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((8, 3))
    expected = x / np.sqrt(np.mean(np.square(x), axis=0) + 1e-6)
    np.testing.assert_allclose(checker.rms_norm(x, 1e-6), expected, rtol=1e-15, atol=0)
    with pytest.raises(ValueError):
        checker.rms_norm(x.ravel(), 1e-6)


def test_zero_controller_weights_are_identity(frozen_facts):
    cur, msg, down, gain_w, forget_w = inputs_from(frozen_facts)
    out = checker.rank64_mix(
        cur, msg, np.zeros_like(down), np.zeros_like(gain_w), np.zeros_like(forget_w))
    assert np.array_equal(out, cur)
    assert checker.check_zero_weights(cur, msg, down, gain_w, forget_w) is True


def test_tanh_checks_recover_gates_from_rank64_mix(frozen_facts):
    cur, msg, down, gain_w, forget_w = inputs_from(frozen_facts)
    gain_pre, forget_pre = checker.gate_preactivations(cur, msg, down, gain_w, forget_w)
    gain_hat, forget_hat, _, _ = checker.recover_gates_from_mix(
        cur, msg, down, gain_w, forget_w)
    np.testing.assert_allclose(gain_hat, np.tanh(gain_pre), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(forget_hat, np.tanh(forget_pre), rtol=1e-12, atol=1e-12)
    assert checker.gates_match_tanh(gain_hat, forget_hat, gain_pre, forget_pre) is True
    assert checker.gates_strictly_bounded(gain_hat, forget_hat) is True
    assert max(np.max(np.abs(gain_pre)), np.max(np.abs(forget_pre))) > 1.0
    assert checker.check_tanh_bounded(cur, msg, down, gain_w, forget_w) is True
    assert checker.check_tanh_very_large(
        cur, msg, down, gain_w, forget_w, frozen_facts["test_config"]["very_large_scale"]) is True
    assert checker.check_unbounded_gate_negative(cur, msg, down, gain_w, forget_w) is True


def test_tanh_checks_fail_if_mix_drops_tanh(frozen_facts, monkeypatch):
    cur, msg, down, gain_w, forget_w = inputs_from(frozen_facts)

    def no_tanh_mix(cur, msg, down, gain_w, forget_w):
        gain_pre, forget_pre = checker.gate_preactivations(cur, msg, down, gain_w, forget_w)
        return (1.0 - forget_pre) * cur + gain_pre * msg

    monkeypatch.setattr(checker, "rank64_mix", no_tanh_mix)
    assert checker.check_tanh_bounded(cur, msg, down, gain_w, forget_w) is False
    assert checker.check_unbounded_gate_negative(cur, msg, down, gain_w, forget_w) is False
    assert checker.check_tanh_very_large(
        cur, msg, down, gain_w, forget_w, frozen_facts["test_config"]["very_large_scale"]) is False


def test_needle_whitespace_normalization():
    norm = checker.normalize_whitespace
    assert norm("ggml_rms_norm(ctx0,  cur,   1e-6f)") == norm("ggml_rms_norm(ctx0, cur, 1e-6f)")
    assert norm("ggml_rms_norm(ctx0, cur, 1e-5f)") != norm("ggml_rms_norm(ctx0, cur, 1e-6f)")


def test_scalar_gamma_cannot_reproduce_rank64(frozen_facts):
    cur, msg, down, gain_w, forget_w = inputs_from(frozen_facts)
    out = checker.rank64_mix(cur, msg, down, gain_w, forget_w)
    gamma, scalar_out, gap = checker.best_scalar_gamma(out, cur, msg)
    assert gap > frozen_facts["test_config"]["scalar_gap_tol"]
    assert not np.array_equal(out, scalar_out)
    assert checker.check_scalar_gap(
        cur, msg, down, gain_w, forget_w, frozen_facts["test_config"]["scalar_gap_tol"]) is True
    assert checker.check_scalar_substitution_negative(cur, msg) is True
    assert abs(gamma) < 10.0


def test_forget_only_changes_cur_per_channel(frozen_facts):
    cur, msg, down, gain_w, forget_w = inputs_from(frozen_facts)
    zero_gain = np.zeros_like(gain_w)
    out = checker.rank64_mix(cur, msg, down, zero_gain, forget_w)
    assert not np.array_equal(out, cur)
    ratios = out / cur
    assert float(np.max(ratios) - np.min(ratios)) > frozen_facts["test_config"]["per_channel_tol"]
    _, _, gap = checker.best_scalar_gamma(out, cur, msg)
    assert gap > frozen_facts["test_config"]["per_channel_tol"]
    assert checker.check_forget_per_channel(
        cur, msg, down, forget_w, frozen_facts["test_config"]["per_channel_tol"]) is True


def test_slow_reference_is_bitwise_identical(frozen_facts):
    cur, msg, down, gain_w, forget_w = inputs_from(frozen_facts)
    fast = checker.rank64_mix(cur, msg, down, gain_w, forget_w)
    slow = checker.slow_reference_mix(cur, msg, down, gain_w, forget_w)
    assert fast.shape == slow.shape
    assert int(np.count_nonzero(fast != slow)) == 0
    assert np.array_equal(fast, slow)
    assert checker.check_slow_reference(cur, msg, down, gain_w, forget_w) is True


def test_fixed_seed_determinism(frozen_facts):
    cfg = {key: frozen_facts["test_config"][key] for key in (
        "seed", "n_embd", "rank", "n_tokens", "down_scale", "gate_weight_scale")}
    assert checker.check_determinism(cfg) is True
    cur, msg, down, gain_w, forget_w = checker.make_inputs(**cfg)
    first = checker.rank64_mix(cur, msg, down, gain_w, forget_w)
    second = checker.rank64_mix(cur, msg, down, gain_w, forget_w)
    assert np.array_equal(first, second)


def test_corrupted_shapes_fail_closed(frozen_facts):
    cur, msg, down, gain_w, forget_w = inputs_from(frozen_facts)
    with pytest.raises(ValueError):
        checker.rank64_mix(cur, msg, down[:, :-1], gain_w, forget_w)
    with pytest.raises(ValueError):
        checker.rank64_mix(cur, msg, down, gain_w[:-1], forget_w)
    with pytest.raises(ValueError):
        checker.rank64_mix(cur[:-1], msg[:-1], down, gain_w, forget_w)
    with pytest.raises(ValueError):
        checker.rank64_mix(cur, msg[:, :-1], down, gain_w, forget_w)
    assert checker.check_corrupt_shapes(cur, msg, down, gain_w, forget_w) is True


def test_numeric_checks_all_pass(frozen_facts):
    checks, info = checker.numeric_checks(frozen_facts)
    assert all(isinstance(value, bool) for value in checks.values())
    assert all(checks.values())
    assert any("best scalar gamma" in line for line in info)


def test_run_check_writes_valid_summary(tmp_path, frozen_facts):
    audit = make_fixture_tree(tmp_path, frozen_facts)
    summary = checker.run_check(tmp_path)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == ""
    assert all(isinstance(value, bool) for value in summary["checks"].values())
    assert all(summary["checks"].values())
    assert summary["note"] == "gate semantics only, no timing"
    assert summary["speed_claim"] is None
    assert summary["frozen_facts_sha256"] == hashlib.sha256(
        (audit / "GATE_SEMANTICS.json").read_bytes()).hexdigest()
    live = summary["provenance"]["source_sha256_live"]
    assert sorted(live) == ["src/models/inkling2-loop.cpp", "src/models/inkling2.cpp"]
    for rel in checker.GATED_SOURCE_FILES:
        assert live[rel] == checker.FROZEN_SOURCE_SHA256[rel]
    assert summary["checks"]["live_source_sha256"] is True
    assert any("(gated)" in line for line in summary["info"])
    assert any("INFO only; not gated" in line for line in summary["info"])
    written = json.loads(summary_path(tmp_path).read_text(encoding="utf-8"))
    assert written == summary


def test_missing_marker_overwrites_stale_pass_summary(tmp_path, frozen_facts):
    audit = make_fixture_tree(tmp_path, frozen_facts, omit="ctrl_gain")
    stale = {"verdict": "PASS"}
    write_stale_pass(audit)
    summary = checker.run_check(tmp_path)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["marker_ctrl_gain"] is False
    assert "marker_ctrl_gain" in summary["blocker"]
    written = json.loads(summary_path(tmp_path).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_exception_overwrites_stale_pass_summary(tmp_path):
    audit = tmp_path / "tools/audit"
    audit.mkdir(parents=True)
    stale = {"verdict": "PASS"}
    write_stale_pass(audit)
    summary = checker.run_check(tmp_path)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["run_completed"] is False
    assert any("exception:" in line for line in summary["info"])
    written = json.loads(summary_path(tmp_path).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_cli_main_exit_codes(tmp_path, frozen_facts, monkeypatch, capsys):
    make_fixture_tree(tmp_path, frozen_facts)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    assert checker.main([]) == 0
    capsys.readouterr()
    assert json.loads(summary_path(tmp_path).read_text(encoding="utf-8"))["verdict"] == "PASS"
    (tmp_path / "src/models/inkling2.cpp").write_text("", encoding="utf-8")
    assert checker.main([]) == 1
    capsys.readouterr()
    assert json.loads(summary_path(tmp_path).read_text(encoding="utf-8"))["verdict"] == "NO_GO"


@pytest.mark.parametrize("mutation", [
    lambda f: f["source_markers"][2].__setitem__("needle", ""),
    lambda f: f.__setitem__("down_shape", [64, 4607]),
    lambda f: f["op_sequence"].__setitem__(1, "concat(norm_cur, norm_message, axis=1)"),
    lambda f: f["ctrl_tensors"]["q"].__setitem__(0, "LOOP_Q_CTRL_BAD"),
], ids=["needle", "down_shape", "op_sequence", "ctrl_tensors"])
def test_tampered_but_valid_facts_json_is_no_go(tmp_path, frozen_facts, mutation):
    audit = make_fixture_tree(tmp_path, frozen_facts)
    write_tampered_facts(audit, frozen_facts, mutation)
    stale = {"verdict": "PASS"}
    write_stale_pass(audit)
    summary = checker.run_check(tmp_path)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["frozen_facts"] is False
    assert summary["checks"]["run_completed"] is True
    assert "frozen_facts" in summary["blocker"]
    written = json.loads(summary_path(tmp_path).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_malformed_facts_json_is_no_go(tmp_path, frozen_facts):
    audit = make_fixture_tree(tmp_path, frozen_facts)
    (audit / "GATE_SEMANTICS.json").write_text("{not json\n", encoding="utf-8")
    stale = {"verdict": "PASS"}
    write_stale_pass(audit)
    summary = checker.run_check(tmp_path)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["run_completed"] is False
    assert any("exception:" in line for line in summary["info"])
    written = json.loads(summary_path(tmp_path).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_live_source_hash_mismatch_is_no_go(tmp_path, frozen_facts):
    audit = make_fixture_tree(tmp_path, frozen_facts)
    path = tmp_path / "src/models/inkling2.cpp"
    path.write_text(path.read_text(encoding="utf-8") + "\n// tamper\n", encoding="utf-8")
    stale = {"verdict": "PASS"}
    write_stale_pass(audit)
    summary = checker.run_check(tmp_path)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["live_source_sha256"] is False
    assert summary["checks"]["marker_mix_formula"] is True
    assert "live_source_sha256" in summary["blocker"]
    written = json.loads(summary_path(tmp_path).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale
