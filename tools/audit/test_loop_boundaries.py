"""Tests for the frozen loop-boundary contract checker."""

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

import check_loop_boundaries as checker

STALE_EN_DASH_SAMPLE = (
    "Prelude (blocks 0\u20131, layers 0\u201311) runs **once**,\n"
    "   GPU1 owns middle block 3\n   (layers 18\u201323) + coda.\n"
    "Coda: blocks 0\u20131's tail layers once.\n"
)


@pytest.fixture()
def frozen_facts():
    return checker.load_json(AUDIT_DIR / "LOOP_BOUNDARIES.json")


def make_fixture_tree(tmp_path, facts, omit=None, stale_spec=None):
    """Build a self-contained root at tmp_path/repo with sibling docs at tmp_path."""
    root = tmp_path / "repo"
    audit = root / "tools/audit"
    audit.mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "LOOP_BOUNDARIES.json", audit / "LOOP_BOUNDARIES.json")

    for rel in checker.FROZEN_SOURCE_SHA256:
        src = REPO_ROOT / rel
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    for label, rel in checker.ENGINE_SPEC_PATHS.items():
        src = REPO_ROOT / rel
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        if stale_spec == label:
            text = dst.read_text(encoding="utf-8")
            dst.write_text(text + "\n" + STALE_EN_DASH_SAMPLE, encoding="utf-8")

    if omit is not None:
        path = root / "conversion/inkling2.py"
        text = path.read_text(encoding="utf-8")
        needle = checker.CONVERTER_NEEDLES[omit]
        assert needle in text
        path.write_text(text.replace(needle, "OMITTED_" + omit.upper()), encoding="utf-8")

    return root


def write_tampered_facts(root, facts, mutate):
    payload = json.loads(json.dumps(facts))
    mutate(payload)
    (root / "tools/audit/LOOP_BOUNDARIES.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_stale_pass(root):
    checker.write_json(root / "tools/audit/M31_LOOP_BOUNDARIES_SUMMARY.json", {"verdict": "PASS"})


def summary_path(root):
    return root / "tools/audit/M31_LOOP_BOUNDARIES_SUMMARY.json"


def ctrl_mapping_text(names, alias=True):
    lines = ["MODEL_TENSORS = {"]
    for name in names:
        lines.append("        MODEL_TENSOR.%s: (" % name)
        if alias:
            lines.append('            "model.layers.{bid}.controllers.attn.down_weight",')
        else:
            lines.append('            "model.layers.{bid}.q_controller.down_proj",')
        lines.append("        ),")
    lines.append("}")
    return "\n".join(lines) + "\n"


def test_frozen_json_shape(frozen_facts):
    facts = frozen_facts
    assert facts["schema_version"] == 1
    assert facts["frozen_on"] == "2026-11-28"
    assert facts["scope"] == "loop boundary contract only, no timing"
    assert facts["note"] == "loop boundary contract only, no timing"
    assert facts["speed_claim"] is None
    assert facts["facts"] == checker.EXPECTED_FACTS
    assert facts["derived"]["core_layers"] == list(range(6, 18))
    assert facts["derived"]["core_layer_count"] == 12
    assert facts["derived"]["prelude_layers"] == list(range(0, 6))
    assert facts["derived"]["coda_layers"] == list(range(18, 24))
    assert facts["derived"]["slot_formula"] == checker.EXPECTED_SLOT_FORMULA
    assert facts["loop_config_keys"] == ["core_start", "core_end", "num_passes", "gate_rank", "is_varloop"]
    assert set(facts["converter_needles"]) == set(checker.CONVERTER_NEEDLES)
    assert facts["converter_needles"] == checker.CONVERTER_NEEDLES
    assert facts["ctrl_tensor_count"] == 12
    assert facts["ctrl_alias_needle"] == "model.layers.{bid}.controllers."
    assert facts["engine_specs"] == checker.ENGINE_SPEC_PATHS
    assert facts["corrected_note_needles"] == checker.CORRECTED_NOTE_NEEDLES
    assert facts["stale_phrases"] == checker.STALE_PHRASES
    assert facts["provenance"]["source_sha256"] == checker.FROZEN_SOURCE_SHA256
    for value in checker.FROZEN_SOURCE_SHA256.values():
        assert len(value) == 64
        int(value, 16)


def test_derive_layers_is_zero_indexed_exclusive():
    core = checker.derive_layers(6, 18)
    assert len(core) == 12
    assert min(core) == 6
    assert max(core) == 17
    assert 18 not in core
    assert 6 in core
    assert checker.derive_layers(0, 6) == list(range(0, 6))
    assert checker.derive_layers(18, 24) == list(range(18, 24))
    with pytest.raises(ValueError):
        checker.derive_layers(18, 6)


def test_evaluate_slot_uses_core_span():
    assert checker.evaluate_slot(24, 3, 6, 18) == 48
    assert checker.evaluate_slot(24, 3, 6, 18) == checker.EXPECTED_FACTS["n_slot"]
    assert checker.evaluate_slot(24, 2, 6, 18) == 36
    with pytest.raises(ValueError):
        checker.evaluate_slot(24, 0, 6, 18)


def test_extract_slot_formula():
    text = "void load() {\n    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);\n}\n"
    assert checker.extract_slot_formula(text) == checker.EXPECTED_SLOT_FORMULA
    with pytest.raises(ValueError):
        checker.extract_slot_formula("void f() {}\n")
    with pytest.raises(ValueError):
        checker.extract_slot_formula(text + text)


def test_extract_ctrl_entries_counts_and_aliases():
    names = ["LOOP_Q_CTRL_DOWN", "LOOP_K_CTRL_GAIN", "LOOP_FFN_CTRL_FORGET"]
    entries = checker.extract_ctrl_entries(ctrl_mapping_text(names))
    assert [name for name, _ in entries] == names
    assert all(has_alias for _, has_alias in entries)
    bad = checker.extract_ctrl_entries(ctrl_mapping_text(names, alias=False))
    assert not any(has_alias for _, has_alias in bad)


def test_real_fork_sources_match_frozen_contract(frozen_facts):
    facts = frozen_facts
    for rel, digest in checker.FROZEN_SOURCE_SHA256.items():
        path = REPO_ROOT / rel
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    converter = checker.read_text(REPO_ROOT / "conversion/inkling2.py")
    assert all(needle in converter for needle in checker.CONVERTER_NEEDLES.values())
    mapping = checker.read_text(REPO_ROOT / "gguf-py/gguf/tensor_mapping.py")
    entries = checker.extract_ctrl_entries(mapping)
    assert len(entries) == 12
    assert all(has_alias for _, has_alias in entries)
    formula = checker.extract_slot_formula(checker.read_text(REPO_ROOT / "src/models/inkling2-loop.cpp"))
    assert checker.normalize_whitespace(formula) == checker.normalize_whitespace(checker.EXPECTED_SLOT_FORMULA)
    assert checker.check_frozen_facts(facts) is True
    assert checker.check_frozen_sha256(facts) is True


def test_corrected_engine_specs_are_present(frozen_facts):
    for rel in checker.ENGINE_SPEC_PATHS.values():
        text = checker.read_text(REPO_ROOT / rel)
        assert all(needle in text for needle in checker.CORRECTED_NOTE_NEEDLES)
        assert all(phrase not in text for phrase in checker.STALE_PHRASES)


def test_normalize_whitespace():
    norm = checker.normalize_whitespace
    assert norm("n_phys +  (size_t) (n_passes - 1)*(core_end - core_start)") == norm(
        checker.EXPECTED_SLOT_FORMULA)
    assert norm("core_end - core_start") != norm("core_end + core_start")


def test_normalize_doc_text_catches_en_dash_wrap():
    norm = checker.normalize_doc_text
    text = norm(STALE_EN_DASH_SAMPLE)
    assert "blocks 0-1, layers 0-11" in text
    assert "middle block 3 (layers 18-23)" in text
    assert "blocks 0-1's tail layers" in text


def test_run_check_writes_valid_summary(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts)
    summary = checker.run_check(root)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == ""
    assert all(isinstance(value, bool) for value in summary["checks"].values())
    assert all(summary["checks"].values())
    assert summary["note"] == "loop boundary contract only, no timing"
    assert summary["speed_claim"] is None
    assert summary["frozen_facts_sha256"] == hashlib.sha256(
        (root / "tools/audit/LOOP_BOUNDARIES.json").read_bytes()).hexdigest()
    live = summary["provenance"]["source_sha256_live"]
    assert sorted(live) == checker.GATED_SOURCE_FILES
    for rel in checker.GATED_SOURCE_FILES:
        assert live[rel] == checker.FROZEN_SOURCE_SHA256[rel]
    assert any("(gated)" in line for line in summary["info"])
    assert any("controller tensor entries: 12" in line for line in summary["info"])
    assert any("n_slot = 24 + (3-1)*(18-6) = 48" in line for line in summary["info"])
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary


def test_missing_converter_needle_overwrites_stale_pass(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts, omit="query_shape")
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["converter_query_shape"] is False
    assert "converter_query_shape" in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_wrong_controller_count_overwrites_stale_pass(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts)
    path = root / "gguf-py/gguf/tensor_mapping.py"
    text = path.read_text(encoding="utf-8")
    needle = '        MODEL_TENSOR.LOOP_FFN_CTRL_FORGET: ('
    assert needle in text
    path.write_text(text.replace(needle, "        # removed\n" + needle), encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    start = text.index("        # removed\n        MODEL_TENSOR.LOOP_FFN_CTRL_FORGET: (")
    end = text.index("\n        ),", start) + len("\n        ),")
    path.write_text(text[:start] + text[end:], encoding="utf-8")
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["tensor_ctrl_count"] is False
    assert "tensor_ctrl_count" in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_wrong_layer_set_overwrites_stale_pass(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts)
    write_tampered_facts(root, frozen_facts, lambda f: f["derived"].__setitem__(
        "core_layers", [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]))
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["frozen_facts"] is False
    assert summary["checks"]["derived_boundaries"] is False
    assert "frozen_facts" in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


@pytest.mark.parametrize("label", ["loopkit", "inkengine"])
def test_stale_engine_spec_overwrites_stale_pass(tmp_path, frozen_facts, label):
    root = make_fixture_tree(tmp_path, frozen_facts, stale_spec=label)
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["engine_spec_%s_no_stale" % label] is False
    assert "engine_spec_%s_no_stale" % label in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_exception_overwrites_stale_pass(tmp_path):
    root = tmp_path / "repo"
    audit = root / "tools/audit"
    audit.mkdir(parents=True)
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["run_completed"] is False
    assert any("exception:" in line for line in summary["info"])
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


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


@pytest.mark.parametrize("mutation", [
    lambda f: f["facts"].__setitem__("core_end", 19),
    lambda f: f["facts"]["is_varloop"].__setitem__("mini", True),
    lambda f: f["facts"].__setitem__("n_slot", 36),
    lambda f: f["derived"].__setitem__("coda_layers", [19, 20, 21, 22, 23]),
], ids=["core_end", "is_varloop", "n_slot", "coda_layers"])
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


def test_malformed_facts_json_is_no_go(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts)
    (root / "tools/audit/LOOP_BOUNDARIES.json").write_text("{not json\n", encoding="utf-8")
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["run_completed"] is False
    assert any("exception:" in line for line in summary["info"])
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale


def test_live_source_hash_mismatch_is_no_go(tmp_path, frozen_facts):
    root = make_fixture_tree(tmp_path, frozen_facts)
    path = root / "src/models/inkling2-loop.cpp"
    path.write_text(path.read_text(encoding="utf-8") + "\n// tamper\n", encoding="utf-8")
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["live_source_sha256"] is False
    assert summary["checks"]["converter_query_shape"] is True
    assert "live_source_sha256" in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale
