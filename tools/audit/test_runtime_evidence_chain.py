"""Behavior tests for the fail-closed runtime wiring evidence chain integrity gate."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CHECKER = AUDIT_DIR / "check_runtime_evidence_chain.py"
sys.path.insert(0, str(AUDIT_DIR))

import check_runtime_evidence_chain as checker

EXPECTED_TOTAL_CHECKS = 52
CONTRACT_SHA256 = "5a6cee68adfce706fd796c06e32bfaede59fc35609f31ead3725e020a8a2fced"
CHECKER_SHA256 = "bdb4be142fdf8eae9b5a9a7d2a99dac3a8dccf51ef0630444ae0baae0942f977"
CONTRACT_PATH = "tools/audit/RUNTIME_EVIDENCE_CHAIN.json"
SUMMARY_PATH = "tools/audit/M43_RUNTIME_EVIDENCE_CHAIN_SUMMARY.json"
NEW_FILES = [
    "RUNTIME_EVIDENCE_CHAIN.json",
    "check_runtime_evidence_chain.py",
    "test_runtime_evidence_chain.py",
    "RUNTIME_EVIDENCE_CHAIN.md",
    "M43_RUNTIME_EVIDENCE_CHAIN_SUMMARY.json",
]


def make_fixture_tree(tmp_path):
    """Copy checker inputs into an isolated root; the real repo is never edited."""
    root = tmp_path / "repo"
    (root / "tools/audit").mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "RUNTIME_EVIDENCE_CHAIN.json", root / CONTRACT_PATH)
    shutil.copy(REPO_ROOT / "DECISIONS.tsv", root / "DECISIONS.tsv")
    shutil.copy(REPO_ROOT / "BUILD_STATUS.md", root / "BUILD_STATUS.md")
    for rel in checker.FROZEN_SUMMARIES:
        shutil.copy(REPO_ROOT / rel, root / rel)
    for entry in checker.FROZEN_CONTRACTS.values():
        rel = entry["path"]
        shutil.copy(REPO_ROOT / rel, root / rel)
    return root


def run_cli(root, checker_path=None):
    return subprocess.run(
        [sys.executable, str(checker_path or CHECKER), "--root", str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def summary_path(root):
    return root / SUMMARY_PATH


def read_summary(root):
    return json.loads(summary_path(root).read_text(encoding="utf-8"))


def write_summary_file(root, rel, payload):
    (root / rel).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def tamper_summary(root, milestone, mutate):
    rel = [path for path in checker.FROZEN_SUMMARIES if checker.milestone_of(path) == milestone][0]
    payload = checker.load_json(root / rel)
    mutate(payload)
    write_summary_file(root, rel, payload)


def write_stale_pass(root):
    checker.write_json(summary_path(root), {"verdict": "PASS", "stale": True})


def write_contract(root, payload):
    (root / CONTRACT_PATH).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def tamper_contract(root, mutate):
    payload = checker.load_json(root / CONTRACT_PATH)
    mutate(payload)
    write_contract(root, payload)


def test_contract_sha_pin_matches_real_file():
    assert checker.sha256_bytes((AUDIT_DIR / "RUNTIME_EVIDENCE_CHAIN.json").read_bytes()) == CONTRACT_SHA256


def test_checker_sha_pin_matches_real_file():
    assert checker.sha256_bytes(CHECKER.read_bytes()) == CHECKER_SHA256


def test_contract_is_frozen_and_self_consistent():
    facts = checker.load_json(AUDIT_DIR / "RUNTIME_EVIDENCE_CHAIN.json")
    assert set(facts) == checker.EXPECTED_TOP_KEYS
    assert facts["schema_version"] == 1
    assert facts["frozen_on"] == "2026-11-28"
    assert facts["scope"] is None
    assert facts["note"] is None
    assert facts["speed_claim"] is None
    assert facts["no_weights"] is True
    assert facts["no_engine_run"] is True
    assert facts["milestones"] == ["M39", "M40", "M41", "M42"]
    assert facts["summaries"] == checker.FROZEN_SUMMARIES
    assert facts["contracts"] == checker.FROZEN_CONTRACTS
    assert facts["checker_sha256"] == CHECKER_SHA256
    assert checker.check_frozen_contract(facts) is True


def test_check_names_are_unique_and_complete():
    assert len(checker.CHECK_NAMES) == EXPECTED_TOTAL_CHECKS
    assert len(checker.CHECK_NAMES) == len(set(checker.CHECK_NAMES))
    assert set(checker.CHECK_NAMES) == (
        {"frozen_contract", "frozen_checker_sha256", "contracts_shape",
         "generated_summary_shape", "no_network", "run_completed"}
        | {"contract_" + name for name in checker.FROZEN_CONTRACTS}
        | {"decisions_" + m for m in checker.MILESTONES}
        | {"build_status_" + m for m in checker.MILESTONES}
        | {"summary_%s_%s" % (m, p) for m in checker.MILESTONES for p in checker.SUMMARY_PROPS}
    )
    assert len(checker.FROZEN_SUMMARIES) == 4
    assert len(checker.FROZEN_CONTRACTS) == 6
    assert [checker.milestone_of(rel) for rel in checker.FROZEN_SUMMARIES] == checker.MILESTONES


def test_frozen_summary_hashes_match_real_tree():
    for rel, expected in checker.FROZEN_SUMMARIES.items():
        path = REPO_ROOT / rel
        assert path.is_file(), rel
        assert checker.sha256_bytes(path.read_bytes()) == expected["sha256"], rel
        assert expected["verdict"] == "PASS"


def test_frozen_contract_pins_match_real_tree():
    for name, entry in checker.FROZEN_CONTRACTS.items():
        path = REPO_ROOT / entry["path"]
        assert path.is_file(), name
        assert checker.sha256_bytes(path.read_bytes()) == entry["sha256"], name


def test_decisions_and_build_status_real_tree():
    assert all(checker.decisions_checks(REPO_ROOT).values())
    assert all(checker.build_status_checks(REPO_ROOT).values())
    assert checker.check_no_network() is True


def test_summary_checks_helper_on_clean_and_missing_summary(tmp_path):
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M42"][0]
    dst = tmp_path / rel
    dst.parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / rel, dst)
    clean = checker.summary_checks(tmp_path, rel, checker.FROZEN_SUMMARIES[rel])
    assert all(clean.values())
    missing = checker.summary_checks(tmp_path, "tools/audit/ABSENT_SUMMARY.json", {"sha256": "0", "verdict": "PASS"})
    assert not any(missing.values())


def test_happy_path_cli_passes_with_exact_check_counts(tmp_path):
    root = make_fixture_tree(tmp_path)
    result = run_cli(root)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == []
    assert summary["check_counts"] == {"total": 52, "passed": 52, "failed": 0}
    assert all(value is True for value in summary["checks"].values())
    assert summary["speed_claim"] is None
    assert summary["note"] == "runtime wiring evidence integrity only, no timing"
    assert summary["no_weights"] is True
    assert summary["no_engine_run"] is True
    assert summary["checker_sha256"] == CHECKER_SHA256
    assert "sentinel" not in summary
    assert "verdict: PASS" in result.stdout
    assert "counts: 52/52 passed, 0 failed" in result.stdout


def test_missing_summary_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M40"][0]
    (root / rel).unlink()
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["summary_M40_exists"] is False
    assert summary["checks"]["summary_M40_sha256"] is False
    assert "summary_M40_exists" in summary["blocker"]


def test_wrong_summary_sha_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M41"][0]
    path = root / rel
    path.write_bytes(path.read_bytes() + b"\n")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["summary_M41_sha256"] is False
    assert summary["checks"]["summary_M41_json_object"] is True
    assert "summary_M41_sha256" in summary["blocker"]


def test_wrong_summary_verdict_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    tamper_summary(root, "M39", lambda payload: payload.__setitem__("verdict", "NO_GO"))
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["summary_M39_verdict"] is False
    assert "summary_M39_verdict" in summary["blocker"]


def test_non_null_summary_speed_claim_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    tamper_summary(root, "M42", lambda payload: payload.__setitem__("speed_claim", "2x faster"))
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["summary_M42_speed_claim"] is False
    assert "summary_M42_speed_claim" in summary["blocker"]


@pytest.mark.parametrize("value", ["blocked", ["blocked"]], ids=["string", "list"])
def test_nonempty_summary_blocker_is_no_go(tmp_path, value):
    root = make_fixture_tree(tmp_path)
    tamper_summary(root, "M40", lambda payload: payload.__setitem__("blocker", value))
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["summary_M40_blocker"] is False
    assert "summary_M40_blocker" in summary["blocker"]


def test_missing_summary_blocker_key_is_still_no_blocker(tmp_path):
    root = make_fixture_tree(tmp_path)
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M39"][0]
    payload = checker.load_json(root / rel)
    assert payload.get("blocker") == []
    payload.pop("blocker")
    write_summary_file(root, rel, payload)
    assert checker.summary_checks(root, rel, checker.FROZEN_SUMMARIES[rel])["blocker"] is True
    result = run_cli(root)
    assert result.returncode == 1
    assert read_summary(root)["checks"]["summary_M39_sha256"] is False


def test_nonzero_summary_failed_count_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    tamper_summary(root, "M39", lambda payload: payload["check_counts"].__setitem__("failed", 1))
    result = run_cli(root)
    assert result.returncode == 1
    assert read_summary(root)["checks"]["summary_M39_check_counts_failed"] is False


def test_summary_passed_not_equal_total_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    tamper_summary(root, "M41", lambda payload: payload["check_counts"].__setitem__("passed", 30))
    result = run_cli(root)
    assert result.returncode == 1
    assert read_summary(root)["checks"]["summary_M41_check_counts_passed_total"] is False


def test_malformed_summary_json_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M42"][0]
    (root / rel).write_text("{not json\n", encoding="utf-8")
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["summary_M42_json_object"] is False
    assert "summary_M42_json_object" in summary["blocker"]


def test_summary_json_array_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M41"][0]
    (root / rel).write_text("[]\n", encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    assert read_summary(root)["checks"]["summary_M41_json_object"] is False


def test_mutated_live_contract_file_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "tools/audit/THREEPASS_BINDING.json"
    path.write_bytes(path.read_bytes() + b"\n")
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["contract_binding_contract"] is False
    assert "contract_binding_contract" in summary["blocker"]


def test_missing_live_contract_file_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    (root / "tools/audit/BIND_TENSOR_LINK.json").unlink()
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["contract_runtime_contract"] is False
    assert "contract_runtime_contract" in summary["blocker"]


def test_tampered_contract_pin_sha_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    tamper_contract(root, lambda facts: facts["contracts"]["replay_contract"].__setitem__("sha256", "0" * 64))
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["frozen_contract"] is False
    assert summary["checks"]["contract_replay_contract"] is False


def test_tampered_contract_pin_path_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    tamper_contract(root, lambda facts: facts["contracts"]["evidence_chain"].__setitem__("path", "tools/audit/OTHER.json"))
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["frozen_contract"] is False
    assert summary["checks"]["contract_evidence_chain"] is False


@pytest.mark.parametrize("mutation", [
    lambda facts: facts["contracts"].pop("runtime_contract"),
    lambda facts: facts["contracts"].__setitem__("extra", {"path": "x", "sha256": "0" * 64}),
], ids=["missing", "extra"])
def test_contract_pin_key_drift_is_no_go(tmp_path, mutation):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    tamper_contract(root, mutation)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["contracts_shape"] is False
    assert summary["checks"]["frozen_contract"] is False
    assert "contracts_shape" in summary["blocker"]


@pytest.mark.parametrize("mutation", [
    lambda facts: facts.pop("contracts"),
    lambda facts: facts.pop("milestones"),
    lambda facts: facts.__setitem__("unexpected", 1),
], ids=["missing_contracts", "missing_milestones", "extra_key"])
def test_top_level_key_drift_is_no_go(tmp_path, mutation):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    tamper_contract(root, mutation)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["frozen_contract"] is False
    assert "frozen_contract" in summary["blocker"]


@pytest.mark.parametrize("mutation", [
    lambda f: f.__setitem__("scope", "timing"),
    lambda f: f.__setitem__("note", "speed"),
    lambda f: f.__setitem__("speed_claim", "2x"),
    lambda f: f.__setitem__("frozen_on", "2026-11-27"),
    lambda f: f.__setitem__("milestones", f["milestones"][:-1]),
    lambda f: f.__setitem__("no_weights", False),
    lambda f: f.__setitem__("no_engine_run", False),
    lambda f: f["summaries"]["tools/audit/M40_3PASS_BINDING_SUMMARY.json"].__setitem__("verdict", "NO_GO"),
], ids=["scope", "note", "speed_claim", "frozen_on", "milestones", "no_weights", "no_engine_run", "summary_verdict"])
def test_tampered_frozen_contract_field_is_no_go(tmp_path, mutation):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    tamper_contract(root, mutation)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["frozen_contract"] is False
    assert "frozen_contract" in summary["blocker"]


def test_contract_checker_sha_mismatch_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    tamper_contract(root, lambda facts: facts.__setitem__("checker_sha256", "0" * 64))
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["frozen_checker_sha256"] is False
    assert "frozen_checker_sha256" in summary["blocker"]
    assert summary["checks"]["frozen_contract"] is True


def test_tampered_checker_copy_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    copy = root / CHECKER.relative_to(REPO_ROOT)
    shutil.copy(CHECKER, copy)
    copy.write_text(copy.read_text(encoding="utf-8") + "\n# tamper\n", encoding="utf-8")
    result = run_cli(root, checker_path=copy)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["frozen_checker_sha256"] is False


def test_missing_decisions_row_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "DECISIONS.tsv"
    text = path.read_text(encoding="utf-8")
    assert "land M41" in text
    path.write_text(text.replace("land M41", "record M41"), encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["decisions_M41"] is False
    assert "decisions_M41" in summary["blocker"]


def test_decisions_substring_tamper_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "DECISIONS.tsv"
    text = path.read_text(encoding="utf-8")
    assert "land M41" in text
    path.write_text(text.replace("land M41", "land M410-CANCELLED"), encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["decisions_M41"] is False
    assert "decisions_M41" in summary["blocker"]


def test_missing_build_status_heading_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "BUILD_STATUS.md"
    text = path.read_text(encoding="utf-8")
    assert "## M40 " in text
    path.write_text(text.replace("## M40 ", "## MXX "), encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["build_status_M40"] is False
    assert "build_status_M40" in summary["blocker"]


def test_malformed_contract_json_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    (root / CONTRACT_PATH).write_text("{not json\n", encoding="utf-8")
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["frozen_contract"] is False
    assert summary["checks"]["run_completed"] is False
    assert any("exception:" in line for line in summary["info"])


def test_missing_contract_file_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    (root / CONTRACT_PATH).unlink()
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["run_completed"] is False
    assert summary["verdict"] == "NO_GO"


def test_seeded_stale_pass_is_overwritten_on_failure(tmp_path):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    stale = summary_path(root).read_text(encoding="utf-8")
    assert '"stale": true' in stale
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M42"][0]
    path = root / rel
    path.write_bytes(path.read_bytes() + b"\n")
    result = run_cli(root)
    assert result.returncode == 1
    written = summary_path(root).read_text(encoding="utf-8")
    assert written != stale
    assert '"stale": true' not in written
    assert read_summary(root)["verdict"] == "NO_GO"


def test_no_go_sentinel_is_written_before_checks(tmp_path, monkeypatch):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    original = checker.write_json
    seen = []

    def tracking(path, payload):
        result = original(path, payload)
        seen.append((path, json.loads(path.read_text(encoding="utf-8"))))
        return result

    monkeypatch.setattr(checker, "write_json", tracking)
    summary = checker.run_check(root)
    assert len(seen) == 2
    assert seen[0][0] == summary_path(root)
    assert seen[0][1]["verdict"] == "NO_GO"
    assert seen[0][1]["sentinel"] is True
    assert seen[0][1]["check_counts"]["passed"] == 0
    assert seen[1][1]["verdict"] == "PASS"
    assert summary == seen[1][1]


def test_sentinel_is_written_even_when_a_check_fails(tmp_path, monkeypatch):
    root = make_fixture_tree(tmp_path)
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M39"][0]
    (root / rel).unlink()
    original = checker.write_json
    seen = []

    def tracking(path, payload):
        result = original(path, payload)
        seen.append(json.loads(path.read_text(encoding="utf-8")))
        return result

    monkeypatch.setattr(checker, "write_json", tracking)
    summary = checker.run_check(root)
    assert len(seen) == 2
    assert seen[0]["verdict"] == "NO_GO"
    assert seen[0]["sentinel"] is True
    assert seen[1]["verdict"] == "NO_GO"
    assert "summary_M39_exists" in seen[1]["blocker"]
    assert summary == seen[1]


def test_atomic_write_failure_keeps_previous_summary_and_removes_temp(tmp_path, monkeypatch):
    target = tmp_path / "audit/M43_RUNTIME_EVIDENCE_CHAIN_SUMMARY.json"
    target.parent.mkdir(parents=True)
    old = '{"verdict": "NO_GO"}\n'
    target.write_text(old, encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("simulated write failure")

    monkeypatch.setattr(checker.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        checker.write_json(target, {"verdict": "PASS"})
    assert target.read_text(encoding="utf-8") == old
    assert not list(target.parent.glob(".*.tmp"))


def test_main_exits_nonzero_when_summary_write_fails(monkeypatch, capsys):
    def boom(root):
        raise OSError("simulated write failure")

    monkeypatch.setattr(checker, "run_check", boom)
    assert checker.main([]) == 1
    assert "could not write summary" in capsys.readouterr().err


def test_fail_closed_summary_shape():
    sentinel = checker.fail_closed_summary()
    assert sentinel["verdict"] == "NO_GO"
    assert sentinel["blocker"] == ["checker did not complete"]
    assert sentinel["sentinel"] is True
    assert all(value is False for value in sentinel["checks"].values())
    assert sentinel["check_counts"] == {
        "total": EXPECTED_TOTAL_CHECKS, "passed": 0, "failed": EXPECTED_TOTAL_CHECKS,
    }
    assert sentinel["speed_claim"] is None
    assert sentinel["no_weights"] is True
    assert sentinel["no_engine_run"] is True
    assert sentinel["note"] == "runtime wiring evidence integrity only, no timing"


def test_check_generated_summary_shape_helper():
    checks = {name: True for name in checker.CHECK_NAMES}
    summary = checker.build_summary(checks, CHECKER_SHA256, [], True)
    summary["checks"]["generated_summary_shape"] = True
    summary = checker.build_summary(summary["checks"], CHECKER_SHA256, [], True)
    assert checker.check_generated_summary_shape(summary) is True
    bad_checks = {name: True for name in checker.CHECK_NAMES}
    bad_checks["contract_evidence_chain"] = False
    bad_checks["generated_summary_shape"] = True
    bad = checker.build_summary(bad_checks, CHECKER_SHA256, [], True)
    assert checker.check_generated_summary_shape(bad) is True
    assert checker.check_generated_summary_shape({**summary, "verdict": "PASS", "blocker": ["x"]}) is False
    assert checker.check_generated_summary_shape({**summary, "sentinel": True}) is False


def test_is_sha256_helper():
    assert checker.is_sha256("a" * 64)
    assert not checker.is_sha256("A" * 64)
    assert not checker.is_sha256("a" * 63)
    assert not checker.is_sha256(None)
    assert checker.is_plain_int(3)
    assert not checker.is_plain_int(True)


@pytest.mark.parametrize("append", [
    "\nimport urllib.request\n",
    "\ndef probe():\n    urlopen('http://example.invalid')\n",
], ids=["network_import", "url_call"])
def test_static_no_network_check_detects_network_code(tmp_path, monkeypatch, append):
    assert checker.check_no_network() is True
    bad = tmp_path / "bad_checker.py"
    bad.write_text(CHECKER.read_text(encoding="utf-8") + append, encoding="utf-8")
    monkeypatch.setattr(checker, "__file__", str(bad))
    assert checker.check_no_network() is False


@pytest.mark.parametrize("rel", NEW_FILES)
def test_new_audit_files_are_ascii(rel):
    data = (AUDIT_DIR / rel).read_bytes()
    assert all(byte < 128 for byte in data)


@pytest.mark.parametrize("rel", NEW_FILES)
def test_new_audit_files_have_no_literal_tabs(rel):
    assert b"\t" not in (AUDIT_DIR / rel).read_bytes()
