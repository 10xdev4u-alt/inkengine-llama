"""Behavior tests for the fail-closed union-residency wiring contract gate."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CHECKER = AUDIT_DIR / "check_union_wiring.py"
sys.path.insert(0, str(AUDIT_DIR))

import check_union_wiring as checker

EXPECTED_TOTAL_CHECKS = 35
NEW_FILES = [
    "UNION_WIRING.json",
    "check_union_wiring.py",
    "test_union_wiring.py",
    "UNION_WIRING.md",
    "M36_UNION_WIRING_SUMMARY.json",
]


def make_fixture_tree(tmp_path):
    """Copy the checker inputs into an isolated root; the real repo is never edited."""
    root = tmp_path / "repo"
    (root / "tools/audit").mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "UNION_WIRING.json", root / "tools/audit/UNION_WIRING.json")
    for rel in checker.GATED_SOURCE_FILES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, dst)
    return root


def run_cli(root):
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def summary_path(root):
    return root / "tools/audit/M36_UNION_WIRING_SUMMARY.json"


def read_summary(root):
    return json.loads(summary_path(root).read_text(encoding="utf-8"))


def write_stale_pass(root):
    checker.write_json(summary_path(root), {"verdict": "PASS", "stale": True})


def write_contract(root, payload):
    (root / "tools/audit/UNION_WIRING.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def tamper_contract(root, mutate):
    payload = json.loads((root / "tools/audit/UNION_WIRING.json").read_text(encoding="utf-8"))
    mutate(payload)
    write_contract(root, payload)


def test_contract_json_is_frozen_and_self_consistent():
    facts = checker.load_json(AUDIT_DIR / "UNION_WIRING.json")
    assert facts["schema_version"] == 1
    assert facts["frozen_on"] == "2026-11-28"
    assert facts["scope"] == checker.NOTE
    assert facts["note"] == checker.NOTE
    assert facts["speed_claim"] is None
    assert facts["sources"] == checker.FROZEN_SOURCES
    assert facts["chain"] == checker.CHAIN_ITEMS
    assert facts["default_off"] == checker.DEFAULT_OFF_ITEMS
    assert facts["absent"] == checker.ABSENT_ITEMS
    assert facts["provenance"]["doc_path"] == checker.DOC_PATH
    assert facts["provenance"]["fork_head"] == checker.FORK_HEAD
    assert all(len(value) == 64 for value in facts["sources"].values())
    assert checker.check_frozen_contract(facts) is True


def test_check_names_are_unique_and_complete():
    assert len(checker.CHECK_NAMES) == EXPECTED_TOTAL_CHECKS
    assert len(checker.CHECK_NAMES) == len(set(checker.CHECK_NAMES))
    assert set(checker.CHECK_NAMES) == (
        {"frozen_contract", "frozen_source_sha256", "live_source_sha256", "chain_coverage", "run_completed"}
        | {"chain_" + entry["name"] for entry in checker.CHAIN_ITEMS}
        | {"default_off_" + entry["name"] for entry in checker.DEFAULT_OFF_ITEMS}
        | {"absent_" + entry["name"] for entry in checker.ABSENT_ITEMS}
    )
    assert len(checker.CHAIN_ITEMS) == 18
    assert len(checker.DEFAULT_OFF_ITEMS) == 7
    assert len(checker.ABSENT_ITEMS) == 5


def test_real_repo_sources_satisfy_frozen_contract():
    for rel, digest in checker.FROZEN_SOURCES.items():
        assert (REPO_ROOT / rel).is_file()
        assert checker.sha256_bytes((REPO_ROOT / rel).read_bytes()) == digest
    texts = {rel: checker.read_text(REPO_ROOT / rel) for rel in checker.GATED_SOURCE_FILES}
    for entry in checker.CHAIN_ITEMS + checker.DEFAULT_OFF_ITEMS:
        assert all(needle in texts[entry["source"]] for needle in entry["needles"]), entry["name"]
    for entry in checker.ABSENT_ITEMS:
        body = checker.scoped_text(texts[entry["source"]], entry["window"])
        assert body is not None
        assert entry["needle"] not in body, entry["name"]


def test_happy_path_cli_passes_with_exact_check_counts(tmp_path):
    root = make_fixture_tree(tmp_path)
    result = run_cli(root)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == ""
    assert summary["check_counts"] == {"total": 35, "passed": 35, "failed": 0}
    assert len(summary["checks"]) == EXPECTED_TOTAL_CHECKS
    assert all(value is True for value in summary["checks"].values())
    assert summary["speed_claim"] is None
    assert summary["no_weights"] is True
    assert summary["no_engine_run"] is True
    assert summary["scope"] == checker.NOTE
    assert summary["cycle_audit_required"] == (
        "NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory for any future number."
    )
    assert summary["provenance"]["source_sha256_live"] == checker.FROZEN_SOURCES
    assert "verdict: PASS" in result.stdout
    assert "counts: 35/35 passed, 0 failed" in result.stdout


def test_source_sha256_tamper_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "src/llama-moe-cache.cpp"
    path.write_text(path.read_text(encoding="utf-8") + "\n// tamper\n", encoding="utf-8")
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["live_source_sha256"] is False
    assert summary["checks"]["chain_bind_slot_map"] is True
    assert "live_source_sha256" in summary["blocker"]
    assert summary["check_counts"]["failed"] >= 1


def test_chain_needle_deletion_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "src/llama-moe-cache.cpp"
    text = path.read_text(encoding="utf-8")
    assert "roret->clear_layer(rb->il)" in text
    path.write_text(text.replace("roret->clear_layer(rb->il)", "REMOVED"), encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["chain_clear_pass0_union"] is False
    assert "chain_clear_pass0_union" in summary["blocker"]


def test_absent_needle_injection_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "src/llama-graph.cpp"
    text = path.read_text(encoding="utf-8")
    marker = "// RORET (opt-in): host-side bookkeeping only"
    at = text.index(marker) + len(marker)
    text = text[:at] + "\nselected_experts = nullptr;\n" + text[at:]
    path.write_text(text, encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["absent_no_selected_experts_assign_in_roret_block"] is False
    assert "absent_no_selected_experts_assign_in_roret_block" in summary["blocker"]


def test_default_off_needle_tamper_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    path = root / "src/llama-moe-cache.cpp"
    text = path.read_text(encoding="utf-8")
    assert "if (roret) {" in text
    path.write_text(text.replace("if (roret) {", "if (false) {"), encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["default_off_copy_extra_null"] is False
    assert "default_off_copy_extra_null" in summary["blocker"]


def test_malformed_contract_json_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    (root / "tools/audit/UNION_WIRING.json").write_text("{not json\n", encoding="utf-8")
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["run_completed"] is False
    assert any("exception:" in line for line in summary["info"])
    assert summary["check_counts"] == {"total": 35, "passed": 0, "failed": 35}


def test_missing_source_file_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    (root / "src/llama-moe-cache-lru.h").unlink()
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["live_source_sha256"] is False
    assert summary["checks"]["chain_lru_hint_contract"] is False
    assert "chain_lru_hint_contract" in summary["blocker"]


def test_missing_contract_file_is_no_go(tmp_path):
    root = make_fixture_tree(tmp_path)
    (root / "tools/audit/UNION_WIRING.json").unlink()
    write_stale_pass(root)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["run_completed"] is False
    assert any("exception:" in line for line in summary["info"])


def test_seeded_stale_pass_is_overwritten_on_tamper(tmp_path):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    stale = summary_path(root).read_text(encoding="utf-8")
    assert '"stale": true' in stale
    path = root / "src/models/inkling2.cpp"
    path.write_text(path.read_text(encoding="utf-8") + "\n// tamper\n", encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    written = summary_path(root).read_text(encoding="utf-8")
    assert written != stale
    assert '"stale": true' not in written
    assert read_summary(root)["verdict"] == "NO_GO"


def test_roret_header_hash_tamper_overwrites_stale_pass(tmp_path):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    path = root / checker.RORET_H
    path.write_text(path.read_text(encoding="utf-8") + "\n// tamper\n", encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["live_source_sha256"] is False
    assert "live_source_sha256" in summary["blocker"]
    assert '"stale": true' not in summary_path(root).read_text(encoding="utf-8")


def test_binding_header_needle_deletion_overwrites_stale_pass(tmp_path):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    path = root / checker.BINDING_H
    text = path.read_text(encoding="utf-8")
    needle = "by_lookup[lookup] = { il, pass };"
    assert needle in text
    path.write_text(text.replace(needle, "REMOVED"), encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["chain_binding_set_find"] is False
    assert "chain_binding_set_find" in summary["blocker"]
    assert '"stale": true' not in summary_path(root).read_text(encoding="utf-8")


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
    target = tmp_path / "audit/M36_UNION_WIRING_SUMMARY.json"
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


@pytest.mark.parametrize("mutation", [
    lambda f: f.__setitem__("speed_claim", "2x faster"),
    lambda f: f.__setitem__("frozen_on", "2026-11-27"),
    lambda f: f.__setitem__("scope", "timing"),
    lambda f: f["sources"].__setitem__("src/llama-graph.cpp", "0" * 64),
    lambda f: f["chain"][0]["needles"].__setitem__(0, "getenv(\"OTHER\")"),
    lambda f: f["provenance"].__setitem__("fork_head", "0" * 40),
], ids=["speed_claim", "frozen_on", "scope", "source_hash", "chain_needle", "fork_head"])
def test_tampered_frozen_contract_is_no_go(tmp_path, mutation):
    root = make_fixture_tree(tmp_path)
    write_stale_pass(root)
    tamper_contract(root, mutation)
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["frozen_contract"] is False
    assert "frozen_contract" in summary["blocker"]


@pytest.mark.parametrize("entry", checker.CHAIN_ITEMS, ids=[e["name"] for e in checker.CHAIN_ITEMS])
def test_each_chain_needle_is_enforced(tmp_path, entry):
    root = make_fixture_tree(tmp_path)
    path = root / entry["source"]
    text = path.read_text(encoding="utf-8")
    needle = entry["needles"][0]
    assert needle in text
    path.write_text(text.replace(needle, "REMOVED"), encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["chain_" + entry["name"]] is False
    assert "chain_" + entry["name"] in summary["blocker"]


@pytest.mark.parametrize("entry", checker.DEFAULT_OFF_ITEMS, ids=[e["name"] for e in checker.DEFAULT_OFF_ITEMS])
def test_each_default_off_needle_is_enforced(tmp_path, entry):
    root = make_fixture_tree(tmp_path)
    path = root / entry["source"]
    text = path.read_text(encoding="utf-8")
    needle = entry["needles"][0]
    assert needle in text
    path.write_text(text.replace(needle, "REMOVED"), encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["default_off_" + entry["name"]] is False
    assert "default_off_" + entry["name"] in summary["blocker"]


@pytest.mark.parametrize("entry", checker.ABSENT_ITEMS, ids=[e["name"] for e in checker.ABSENT_ITEMS])
def test_each_absent_needle_is_enforced(tmp_path, entry):
    root = make_fixture_tree(tmp_path)
    path = root / entry["source"]
    text = path.read_text(encoding="utf-8")
    if entry["window"] is None:
        at = len(text)
    else:
        at = text.index(entry["window"]["start"]) + len(entry["window"]["start"])
    path.write_text(text[:at] + "\n" + entry["needle"] + "\n" + text[at:], encoding="utf-8")
    result = run_cli(root)
    assert result.returncode == 1
    summary = read_summary(root)
    assert summary["checks"]["absent_" + entry["name"]] is False
    assert "absent_" + entry["name"] in summary["blocker"]


@pytest.mark.parametrize("rel", NEW_FILES)
def test_new_audit_files_are_ascii(rel):
    data = (AUDIT_DIR / rel).read_bytes()
    assert all(byte < 128 for byte in data)
