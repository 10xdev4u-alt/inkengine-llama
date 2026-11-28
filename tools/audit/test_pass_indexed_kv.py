"""Tests for the frozen pass-indexed KV slot contract checker."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
sys.path.insert(0, str(AUDIT_DIR))

import check_pass_indexed_kv as checker

FROZEN_SLOT_TRIPLES = [
    (p, ip, il)
    for p, base in ((0, 0), (1, 18), (2, 30))
    for ip, il in zip(range(6, 18), range(base + 6, base + 18))
]


@pytest.fixture()
def frozen_facts():
    return checker.load_json(AUDIT_DIR / "PASS_INDEXED_KV.json")


def make_fixture_tree(tmp_path, facts, omit=None, drop=None):
    """Build a self-contained root at tmp_path/repo from the live checker inputs."""
    root = tmp_path / "repo"
    audit = root / "tools/audit"
    audit.mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "PASS_INDEXED_KV.json", audit / "PASS_INDEXED_KV.json")
    for rel in checker.GATED_SOURCE_FILES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, dst)
    if omit is not None:
        entry = checker.SOURCE_NEEDLES[omit]
        path = root / entry["source"]
        text = path.read_text(encoding="utf-8")
        needle = entry["needles"][0]
        assert needle in text
        path.write_text(text.replace(needle, "OMITTED_" + omit.upper()), encoding="utf-8")
    if drop is not None:
        path = root / drop
        assert path.is_file()
        path.unlink()
    return root


def write_tampered_facts(root, facts, mutate):
    payload = json.loads(json.dumps(facts))
    mutate(payload)
    (root / "tools/audit/PASS_INDEXED_KV.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_stale_pass(root):
    checker.write_json(root / "tools/audit/M32_PASS_INDEXED_KV_SUMMARY.json", {"verdict": "PASS"})


def summary_path(root):
    return root / "tools/audit/M32_PASS_INDEXED_KV_SUMMARY.json"


def test_frozen_json_shape(frozen_facts):
    facts = frozen_facts
    assert facts["schema_version"] == 1
    assert facts["frozen_on"] == "2026-11-28"
    assert facts["scope"] == "pass-indexed KV slot contract only, no timing"
    assert facts["note"] == "pass-indexed KV slot contract only, no timing"
    assert facts["speed_claim"] is None
    assert facts["facts"] == checker.EXPECTED_FACTS
    assert facts["facts"]["n_phys"] == 24
    assert facts["facts"]["n_passes"] == 3
    assert facts["facts"]["core_start"] == 6
    assert facts["facts"]["core_end"] == 18
    assert facts["facts"]["n_slot"] == 48
    assert facts["facts"]["n_ctrl"] == 4
    assert facts["needles"] == checker.SOURCE_NEEDLES
    assert set(facts["needles"]) == set(checker.SOURCE_NEEDLES)
    assert facts["provenance"]["source_sha256"] == checker.FROZEN_SOURCE_SHA256
    for value in checker.FROZEN_SOURCE_SHA256.values():
        assert len(value) == 64
        int(value, 16)


def test_check_names_are_unique_and_complete():
    assert len(checker.CHECK_NAMES) == len(set(checker.CHECK_NAMES))
    assert len(checker.CHECK_NAMES) == 27
    assert set(checker.CHECK_NAMES) == (
        {"frozen_facts", "frozen_source_sha256", "live_source_sha256", "run_completed"}
        | set(checker.SOURCE_NEEDLES)
        | set(checker.DERIVED_CHECK_NAMES)
    )
    assert "slot_map_derived" in checker.CHECK_NAMES
    assert "slot_map_injective" in checker.CHECK_NAMES
    assert "slot_map_layer_phys_inverse" in checker.CHECK_NAMES
    assert "slot_map_coverage" in checker.CHECK_NAMES


def test_frozen_slot_map_has_all_36_triples(frozen_facts):
    slot_map = frozen_facts["slot_map"]
    assert len(slot_map) == 36
    assert [(e["pass"], e["ip"], e["il"]) for e in slot_map] == FROZEN_SLOT_TRIPLES
    assert slot_map[0] == {"pass": 0, "ip": 6, "il": 6}
    assert slot_map[11] == {"pass": 0, "ip": 17, "il": 17}
    assert slot_map[12] == {"pass": 1, "ip": 6, "il": 24}
    assert slot_map[23] == {"pass": 1, "ip": 17, "il": 35}
    assert slot_map[24] == {"pass": 2, "ip": 6, "il": 36}
    assert slot_map[-1] == {"pass": 2, "ip": 17, "il": 47}


def test_evaluate_slot_literals():
    assert checker.evaluate_slot(24, 3, 6, 18) == 48
    assert checker.evaluate_slot(24, 3, 6, 18) == checker.EXPECTED_FACTS["n_slot"]
    assert checker.evaluate_slot(24, 2, 6, 18) == 36
    assert checker.evaluate_slot(8, 2, 2, 6) == 12
    with pytest.raises(ValueError):
        checker.evaluate_slot(24, 0, 6, 18)
    with pytest.raises(ValueError):
        checker.evaluate_slot(24, 3, 18, 6)
    with pytest.raises(ValueError):
        checker.evaluate_slot(24, 3, 6, 25)
    with pytest.raises(ValueError):
        checker.evaluate_slot(24.0, 3, 6, 18)


def test_layer_phys_literals():
    f = checker.EXPECTED_FACTS
    args = (f["n_phys"], f["core_start"], f["core_end"])
    assert checker.layer_phys(0, *args) == 0
    assert checker.layer_phys(5, *args) == 5
    assert checker.layer_phys(6, *args) == 6
    assert checker.layer_phys(17, *args) == 17
    assert checker.layer_phys(18, *args) == 18
    assert checker.layer_phys(23, *args) == 23
    assert checker.layer_phys(24, *args) == 6
    assert checker.layer_phys(35, *args) == 17
    assert checker.layer_phys(36, *args) == 6
    assert checker.layer_phys(47, *args) == 17
    assert checker.layer_phys(48, *args) == 6
    with pytest.raises(ValueError):
        checker.layer_phys(-1, *args)
    with pytest.raises(ValueError):
        checker.layer_phys(True, *args)


def test_pass_slot_literals():
    f = checker.EXPECTED_FACTS
    args = (f["n_phys"], f["core_start"], f["core_end"])
    expected = {
        (0, 6): 6,
        (0, 17): 17,
        (1, 6): 24,
        (1, 12): 30,
        (1, 17): 35,
        (2, 6): 36,
        (2, 17): 47,
    }
    for (p, ip), il in expected.items():
        assert checker.pass_slot(p, ip, *args) == il
    with pytest.raises(ValueError):
        checker.pass_slot(-1, 6, *args)
    with pytest.raises(ValueError):
        checker.pass_slot(1, 5, *args)
    with pytest.raises(ValueError):
        checker.pass_slot(1, 18, *args)
    with pytest.raises(ValueError):
        checker.pass_slot(True, 6, *args)


def test_build_slot_map_matches_literal_triples():
    derived = checker.build_slot_map()
    assert [(e["pass"], e["ip"], e["il"]) for e in derived] == FROZEN_SLOT_TRIPLES
    assert all(set(e) == {"pass", "ip", "il"} for e in derived)


def test_build_slot_map_alternate_boundaries():
    derived = checker.build_slot_map(8, 2, 2, 6)
    assert [(e["pass"], e["ip"], e["il"]) for e in derived] == [
        (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5),
        (1, 2, 8), (1, 3, 9), (1, 4, 10), (1, 5, 11),
    ]
    with pytest.raises(ValueError):
        checker.build_slot_map(8, 0, 2, 6)


def test_check_slot_map_accepts_frozen(frozen_facts):
    slot_map = frozen_facts["slot_map"]
    f = checker.EXPECTED_FACTS
    assert checker.is_injective(slot_map) is True
    assert checker.layer_phys_inverse(slot_map, f["n_phys"], f["core_start"], f["core_end"]) is True
    assert checker.covers_core_slots(
        slot_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"]) is True
    assert checker.check_slot_map(
        slot_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"]) is True


def mutate_slot_map(facts, mutate):
    payload = json.loads(json.dumps(facts["slot_map"]))
    mutate(payload)
    return payload


def test_check_slot_map_rejects_duplicate_slot(frozen_facts):
    slot_map = mutate_slot_map(frozen_facts, lambda m: m[24].__setitem__("il", 6))
    f = checker.EXPECTED_FACTS
    assert checker.is_injective(slot_map) is False
    assert checker.check_slot_map(
        slot_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"]) is False


def test_check_slot_map_rejects_wrong_logical_layer(frozen_facts):
    slot_map = mutate_slot_map(frozen_facts, lambda m: m[24].__setitem__("ip", 7))
    f = checker.EXPECTED_FACTS
    assert checker.layer_phys_inverse(slot_map, f["n_phys"], f["core_start"], f["core_end"]) is False
    assert checker.check_slot_map(
        slot_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"]) is False


def test_check_slot_map_rejects_missing_entry(frozen_facts):
    slot_map = mutate_slot_map(frozen_facts, lambda m: m.pop())
    f = checker.EXPECTED_FACTS
    assert checker.covers_core_slots(
        slot_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"]) is False
    assert checker.check_slot_map(
        slot_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"]) is False


def test_check_slot_map_rejects_extra_entry(frozen_facts):
    slot_map = mutate_slot_map(frozen_facts, lambda m: m.append({"pass": 2, "ip": 18, "il": 48}))
    f = checker.EXPECTED_FACTS
    assert checker.covers_core_slots(
        slot_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"]) is False
    assert checker.check_slot_map(
        slot_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"]) is False


def test_check_slot_map_rejects_coda_slot_as_core(frozen_facts):
    slot_map = mutate_slot_map(frozen_facts, lambda m: m[12].__setitem__("il", 18))
    f = checker.EXPECTED_FACTS
    assert checker.covers_core_slots(
        slot_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"]) is False


@pytest.mark.parametrize("mutation", [
    lambda m: m.append({"pass": 0, "ip": 6}),
    lambda m: m[0].__setitem__("il", True),
    lambda m: m[0].__setitem__("ip", "6"),
    lambda m: m.__setitem__(0, [0, 6, 6]),
    lambda m: m.clear(),
], ids=["missing_key", "bool_value", "string_value", "list_entry", "empty"])
def test_check_slot_map_rejects_malformed_entries(frozen_facts, mutation):
    slot_map = mutate_slot_map(frozen_facts, mutation)
    f = checker.EXPECTED_FACTS
    assert checker.check_slot_map(
        slot_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"]) is False


def test_frozen_facts_and_hashes_validate(frozen_facts):
    assert checker.check_frozen_facts(frozen_facts) is True
    assert checker.check_frozen_sha256(frozen_facts) is True


def test_frozen_facts_rejects_tampered_needle(frozen_facts):
    payload = json.loads(json.dumps(frozen_facts))
    payload["needles"]["slot_formula"]["needles"][0] = "wrong needle"
    assert checker.check_frozen_facts(payload) is False


def test_frozen_facts_rejects_tampered_slot_map(frozen_facts):
    payload = json.loads(json.dumps(frozen_facts))
    payload["slot_map"][12]["il"] = 18
    assert checker.check_frozen_facts(payload) is False


def test_real_fork_sources_match_frozen_contract(frozen_facts):
    for rel, digest in checker.FROZEN_SOURCE_SHA256.items():
        path = REPO_ROOT / rel
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for entry in checker.SOURCE_NEEDLES.values():
        text = checker.read_text(REPO_ROOT / entry["source"])
        assert all(needle in text for needle in entry["needles"])
    assert checker.evaluate_slot(24, 3, 6, 18) == frozen_facts["facts"]["n_slot"]


def test_run_check_passes_on_real_tree(frozen_facts):
    summary = checker.run_check(REPO_ROOT)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == ""
    assert len(summary["checks"]) == len(checker.CHECK_NAMES)
    assert all(isinstance(value, bool) for value in summary["checks"].values())
    assert all(summary["checks"].values())
    assert summary["note"] == "pass-indexed KV slot contract only, no timing"
    assert summary["speed_claim"] is None
    assert summary["frozen_facts_sha256"] == hashlib.sha256(
        (AUDIT_DIR / "PASS_INDEXED_KV.json").read_bytes()).hexdigest()
    live = summary["provenance"]["source_sha256_live"]
    assert sorted(live) == checker.GATED_SOURCE_FILES
    for rel, digest in checker.FROZEN_SOURCE_SHA256.items():
        assert live[rel] == digest
    assert any("(gated)" in line for line in summary["info"])
    assert any("36 core applications, il 6..17 + 24..47" in line for line in summary["info"])
    written = json.loads((AUDIT_DIR / "M32_PASS_INDEXED_KV_SUMMARY.json").read_text(encoding="utf-8"))
    assert written == summary


def test_run_check_writes_valid_summary_on_fixture(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts)
    summary = checker.run_check(root)
    assert summary["verdict"] == "PASS"
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary


def test_cli_main_exit_codes(tmp_path, frozen_facts, monkeypatch, capsys):
    root = make_fixture_tree(tmp_path, frozen_facts)
    monkeypatch.setattr(checker, "REPO_ROOT", root)
    assert checker.main([]) == 0
    capsys.readouterr()
    assert json.loads(summary_path(root).read_text(encoding="utf-8"))["verdict"] == "PASS"
    (root / "src/models/inkling2-loop.cpp").write_text("", encoding="utf-8")
    assert checker.main([]) == 1
    capsys.readouterr()
    assert json.loads(summary_path(root).read_text(encoding="utf-8"))["verdict"] == "NO_GO"


def test_live_source_hash_mismatch_overwrites_stale_pass(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts)
    path = root / "src/models/inkling2.cpp"
    path.write_text(path.read_text(encoding="utf-8") + "\n// tamper\n", encoding="utf-8")
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["live_source_sha256"] is False
    assert summary["checks"]["converter_stack"] is True
    assert "live_source_sha256" in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_frozen_hash_mismatch_overwrites_stale_pass(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts)
    write_tampered_facts(root, frozen_facts, lambda f: f["provenance"]["source_sha256"].__setitem__(
        "src/models/inkling2.cpp", "0" * 64))
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["frozen_source_sha256"] is False
    assert "frozen_source_sha256" in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


@pytest.mark.parametrize("mutation", [
    lambda f: f["facts"].__setitem__("n_slot", 36),
    lambda f: f["facts"].__setitem__("n_passes", 2),
    lambda f: f["facts"].__setitem__("n_ctrl", 8),
    lambda f: f["facts"].__setitem__("core_end", 19),
    lambda f: f["facts"].__setitem__("slot_formula", "n_phys"),
    lambda f: f["facts"].__setitem__("layer_phys_formula", "il"),
    lambda f: f["facts"].__setitem__("pass_slot_formula", "ip"),
    lambda f: f.__setitem__("frozen_on", "2026-11-27"),
    lambda f: f.__setitem__("scope", "timing"),
    lambda f: f.__setitem__("speed_claim", "2x faster"),
], ids=[
    "n_slot", "n_passes", "n_ctrl", "core_end", "slot_formula", "layer_phys_formula",
    "pass_slot_formula", "frozen_on", "scope", "speed_claim",
])
def test_tampered_but_valid_facts_json_is_no_go(tmp_path, frozen_facts, mutation):
    root = make_fixture_tree(tmp_path, frozen_facts)
    write_tampered_facts(root, frozen_facts, mutation)
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["frozen_facts"] is False
    assert summary["checks"]["run_completed"] is True
    assert "frozen_facts" in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


@pytest.mark.parametrize("mutation", [
    lambda f: f["slot_map"][0].__setitem__("il", 7),
    lambda f: f["slot_map"][12].__setitem__("il", 18),
    lambda f: f["slot_map"][24].__setitem__("ip", 17),
    lambda f: f["slot_map"].pop(),
], ids=["duplicate_slot", "coda_slot", "wrong_logical_layer", "missing_entry"])
def test_tampered_slot_map_is_no_go(tmp_path, frozen_facts, mutation):
    root = make_fixture_tree(tmp_path, frozen_facts)
    write_tampered_facts(root, frozen_facts, mutation)
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["frozen_facts"] is False
    assert "frozen_facts" in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


@pytest.mark.parametrize("name", sorted(checker.SOURCE_NEEDLES))
def test_missing_source_needle_is_no_go(tmp_path, frozen_facts, name):
    root = make_fixture_tree(tmp_path, frozen_facts, omit=name)
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"][name] is False
    assert name in summary["blocker"]
    assert summary["checks"]["frozen_facts"] is True
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_malformed_facts_json_is_no_go(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts)
    (root / "tools/audit/PASS_INDEXED_KV.json").write_text("{not json\n", encoding="utf-8")
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["run_completed"] is False
    assert any("exception:" in line for line in summary["info"])
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_missing_source_file_is_no_go(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts, drop="src/models/inkling2.cpp")
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["live_source_sha256"] is False
    assert summary["checks"]["state_by_il"] is False
    assert "state_by_il" in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_exception_overwrites_stale_pass(tmp_path):
    root = tmp_path / "repo"
    (root / "tools/audit").mkdir(parents=True)
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["run_completed"] is False
    assert any("exception:" in line for line in summary["info"])
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


@pytest.mark.parametrize("rel", [
    "PASS_INDEXED_KV.json",
    "check_pass_indexed_kv.py",
    "test_pass_indexed_kv.py",
    "PASS_INDEXED_KV.md",
    "M32_PASS_INDEXED_KV_SUMMARY.json",
])
def test_new_audit_files_are_ascii(rel):
    data = (AUDIT_DIR / rel).read_bytes()
    assert all(byte < 128 for byte in data)
