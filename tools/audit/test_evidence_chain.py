"""Behavior tests for the fail-closed wiring evidence chain integrity gate."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CHECKER = AUDIT_DIR / "check_evidence_chain.py"
sys.path.insert(0, str(AUDIT_DIR))

import check_evidence_chain as checker

EXPECTED_TOTAL_CHECKS = 94
CONTRACT_PATH = "tools/audit/EVIDENCE_CHAIN.json"
SUMMARY_PATH = "tools/audit/M39_EVIDENCE_CHAIN_SUMMARY.json"
NEW_FILES = [
    "EVIDENCE_CHAIN.json",
    "check_evidence_chain.py",
    "test_evidence_chain.py",
    "EVIDENCE_CHAIN.md",
    "M39_EVIDENCE_CHAIN_SUMMARY.json",
]


def make_fixture_tree(tmp_path):
    """Copy checker inputs into an isolated root; the real repo is never edited."""
    root = tmp_path / "repo"
    (root / "tools/audit").mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "EVIDENCE_CHAIN.json", root / CONTRACT_PATH)
    shutil.copy(AUDIT_DIR / "UNION_WIRING.json", root / "tools/audit/UNION_WIRING.json")
    shutil.copy(REPO_ROOT / "DECISIONS.tsv", root / "DECISIONS.tsv")
    shutil.copy(REPO_ROOT / "BUILD_STATUS.md", root / "BUILD_STATUS.md")
    for rel in checker.FROZEN_SUMMARIES:
        shutil.copy(REPO_ROOT / rel, root / rel)
    for rel in checker.FROZEN_WIRING_SOURCES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, dst)
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


def test_contract_is_frozen_and_self_consistent():
    facts = checker.load_json(AUDIT_DIR / "EVIDENCE_CHAIN.json")
    assert set(facts) == checker.EXPECTED_TOP_KEYS
    assert facts["schema_version"] == 1
    assert facts["frozen_on"] == "2026-11-28"
    assert facts["scope"] is None
    assert facts["note"] is None
    assert facts["speed_claim"] is None
    assert facts["summaries"] == checker.FROZEN_SUMMARIES
    assert facts["milestones"] == checker.MILESTONES
    assert facts["wiring_contract"] == checker.WIRING_CONTRACT
    assert checker.is_sha256(facts["checker_sha256"])
    assert facts["no_weights"] is True
    assert facts["no_engine_run"] is True
    assert checker.check_frozen_contract(facts) is True


def test_check_names_are_unique_and_complete():
    assert len(checker.CHECK_NAMES) == EXPECTED_TOTAL_CHECKS
    assert len(checker.CHECK_NAMES) == len(set(checker.CHECK_NAMES))
    assert set(checker.CHECK_NAMES) == (
        {"frozen_contract", "frozen_checker_sha256", "wiring_contract_json",
         "wiring_sources_shape", "no_network", "run_completed"}
        | {"wiring_source_" + name for name in checker.WIRING_SOURCE_NAMES}
        | {"decisions_" + m for m in checker.MILESTONES}
        | {"build_status_" + m for m in checker.MILESTONES}
        | {"summary_%s_%s" % (m, p) for m in checker.SUMMARY_MILESTONES for p in checker.SUMMARY_PROPS}
    )
    assert len(checker.FROZEN_SUMMARIES) == 10
    assert len(checker.FROZEN_WIRING_SOURCES) == 8
    assert checker.SUMMARY_MILESTONES == checker.MILESTONES


def test_frozen_summary_hashes_match_real_tree():
    for rel, expected in checker.FROZEN_SUMMARIES.items():
        path = REPO_ROOT / rel
        assert path.is_file(), rel
        assert checker.sha256_bytes(path.read_bytes()) == expected["sha256"], rel
        assert expected["verdict"] == "PASS"


def test_frozen_wiring_sources_match_real_tree():
    for rel, digest in checker.FROZEN_WIRING_SOURCES.items():
        path = REPO_ROOT / rel
        assert path.is_file(), rel
        assert checker.sha256_bytes(path.read_bytes()) == digest, rel


def test_decisions_and_build_status_real_tree():
    assert all(checker.decisions_checks(REPO_ROOT).values())
    assert all(checker.build_status_checks(REPO_ROOT).values())
    assert checker.check_no_network() is True


def test_m29_older_summary_shape_has_no_blocker_key():
    rel = [path for path in checker.FROZEN_SUMMARIES if checker.milestone_of(path) == "M29"][0]
    payload = checker.load_json(REPO_ROOT / rel)
    assert "blocker" not in payload
    assert checker.summary_checks(REPO_ROOT, rel, checker.FROZEN_SUMMARIES[rel])["blocker"] is True


def test_happy_path_cli_passes_with_exact_check_counts(tmp_path):
    root = make_fixture_tree(tmp_path)
    result = run_cli(root)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == []
    assert summary["check_counts"] == {"total": 94, "passed": 94, "failed": 0}
    assert all(value is True for value in summary["checks"].values())
    assert summary["speed_claim"] is None
    assert summary["note"] == "wiring evidence integrity only, no timing"
    assert summary["no_weights"] is True
    assert summary["no_engine_run"] is True
    assert summary["wiring_contract"] == checker.WIRING_CONTRACT
    assert "sentinel" not in summary
    assert "verdict: PASS" in result.stdout
    assert "counts: 94/94 passed, 0 failed" in result.stdout


def test_missing_summary_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M30"][0]
    (root / rel).unlink()
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["summary_M30_exists"] is False
    assert summary["checks"]["summary_M30_sha256"] is False
    assert "summary_M30_exists" in summary["blocker"]


def test_wrong_summary_sha_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M31"][0]
    path = root / rel
    path.write_bytes(path.read_bytes() + b"\n")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["summary_M31_sha256"] is False
    assert summary["checks"]["summary_M31_json_object"] is True
    assert "summary_M31_sha256" in summary["blocker"]


def test_wrong_summary_verdict_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    tamper_summary(root, "M32", lambda payload: payload.__setitem__("verdict", "NO_GO"))
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["summary_M32_verdict"] is False
    assert "summary_M32_verdict" in summary["blocker"]


def test_non_null_summary_speed_claim_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    tamper_summary(root, "M33", lambda payload: payload.__setitem__("speed_claim", "2x faster"))
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["summary_M33_speed_claim"] is False
    assert "summary_M33_speed_claim" in summary["blocker"]


@pytest.mark.parametrize("value", ["blocked", ["blocked"]], ids=["string", "list"])
def test_nonempty_summary_blocker_is_no_go(tmp_path, value):
    root = make_fixture_tree(tmp_path)
    tamper_summary(root, "M34", lambda payload: payload.__setitem__("blocker", value))
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["summary_M34_blocker"] is False
    assert "summary_M34_blocker" in summary["blocker"]


def test_malformed_summary_json_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M35"][0]
    (root / rel).write_text("{not json\n", encoding="utf-8")
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["summary_M35_json_object"] is False
    assert "summary_M35_json_object" in summary["blocker"]


def test_mutated_live_wiring_source_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    rel = "src/llama-roret.h"
    path = root / rel
    path.write_text(path.read_text(encoding="utf-8") + "\n// tamper\n", encoding="utf-8")
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["wiring_source_llama-roret.h"] is False
    assert "wiring_source_llama-roret.h" in summary["blocker"]


def test_missing_live_wiring_source_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    (root / "src/llama-moe-cache-lru.h").unlink()
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["wiring_source_llama-moe-cache-lru.h"] is False


def test_tampered_wiring_contract_sources_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "tools/audit/UNION_WIRING.json"
    wiring = checker.load_json(path)
    wiring["sources"]["src/llama-graph.cpp"] = "0" * 64
    path.write_text(json.dumps(wiring, indent=2) + "\n", encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["wiring_sources_shape"] is False
    assert summary["checks"]["wiring_source_llama-graph.cpp"] is False


def test_missing_wiring_contract_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    (root / "tools/audit/UNION_WIRING.json").unlink()
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["wiring_contract_json"] is False
    assert summary["checks"]["wiring_sources_shape"] is False
    assert summary["checks"]["run_completed"] is True


def test_missing_decisions_row_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "DECISIONS.tsv"
    text = path.read_text(encoding="utf-8")
    assert "land M30" in text
    path.write_text(text.replace("land M30", "record M30"), encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["decisions_M30"] is False
    assert "decisions_M30" in summary["blocker"]


def test_missing_build_status_heading_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "BUILD_STATUS.md"
    text = path.read_text(encoding="utf-8")
    assert "## M30 " in text
    path.write_text(text.replace("## M30 ", "## MXX "), encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["build_status_M30"] is False
    assert "build_status_M30" in summary["blocker"]


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


def test_extra_contract_key_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    tamper_contract(root, lambda facts: facts.__setitem__("unexpected", 1))
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
    lambda f: f.__setitem__("wiring_contract", "tools/audit/OTHER.json"),
    lambda f: f.__setitem__("no_weights", False),
    lambda f: f["summaries"]["tools/audit/M36_UNION_WIRING_SUMMARY.json"].__setitem__("verdict", "NO_GO"),
], ids=["scope", "note", "speed_claim", "frozen_on", "milestones", "wiring_contract", "no_weights", "summary_verdict"])
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


def test_seeded_stale_pass_is_overwritten_on_failure(tmp_path):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    stale = summary_path(root).read_text(encoding="utf-8")
    assert '"stale": true' in stale
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M37"][0]
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


def test_atomic_write_failure_keeps_previous_summary_and_removes_temp(tmp_path, monkeypatch):
    target = tmp_path / "audit/M39_EVIDENCE_CHAIN_SUMMARY.json"
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


def test_summary_checks_helper_on_clean_and_missing_summary(tmp_path):
    rel = [p for p in checker.FROZEN_SUMMARIES if checker.milestone_of(p) == "M38"][0]
    src = REPO_ROOT / rel
    dst = tmp_path / rel
    dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)
    clean = checker.summary_checks(tmp_path, rel, checker.FROZEN_SUMMARIES[rel])
    assert all(clean.values())
    missing = checker.summary_checks(tmp_path, "tools/audit/ABSENT_SUMMARY.json", {"sha256": "0", "verdict": "PASS"})
    assert not any(missing.values())


def test_is_sha256_helper():
    assert checker.is_sha256("a" * 64)
    assert not checker.is_sha256("A" * 64)
    assert not checker.is_sha256("a" * 63)
    assert not checker.is_sha256(None)


@pytest.mark.parametrize("rel", NEW_FILES)
def test_new_audit_files_are_ascii(rel):
    data = (AUDIT_DIR / rel).read_bytes()
    assert all(byte < 128 for byte in data)
