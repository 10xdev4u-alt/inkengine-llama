"""Tests for the frozen quant layout and Unsloth quant checker."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
sys.path.insert(0, str(AUDIT_DIR))

import check_quant_layout as checker

NEW_FILES = [
    "QUANT_LAYOUT.json",
    "check_quant_layout.py",
    "test_quant_layout.py",
    "QUANT_LAYOUT.md",
    "M33_QUANT_LAYOUT_SUMMARY.json",
]


@pytest.fixture()
def frozen_facts():
    return checker.load_json(AUDIT_DIR / "QUANT_LAYOUT.json")


@pytest.fixture(autouse=True)
def no_external_loopkit(monkeypatch):
    monkeypatch.delenv("LOOPKIT_ROOT", raising=False)


def make_fixture_tree(tmp_path):
    """Build a self-contained repo and sibling loopkit under tmp_path."""
    root = tmp_path / "repo"
    audit = root / "tools/audit"
    audit.mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "QUANT_LAYOUT.json", audit / "QUANT_LAYOUT.json")
    src = root / checker.CONVERTER_PY
    src.parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / checker.CONVERTER_PY, src)

    lk_root = tmp_path / "loopkit"
    docs = lk_root / "docs"
    docs.mkdir(parents=True)
    for rel in (checker.INTEGRATION_MD, checker.FORMAT_FACTS_MD):
        shutil.copy(REPO_ROOT.parent / "loopkit" / rel, docs / Path(rel).name)
    return root, lk_root


def needle_path(root, lk_root, name):
    spec = checker.SOURCE_NEEDLES[name]
    base = root if spec["root"] == "repo" else lk_root
    return base / spec["source"]


def write_tampered_facts(root, facts, mutate):
    payload = json.loads(json.dumps(facts))
    mutate(payload)
    (root / checker.FACTS_REL).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_stale_pass(root):
    checker.write_json(root / checker.SUMMARY_REL, {"verdict": "PASS"})


def summary_path(root):
    return root / checker.SUMMARY_REL


def assert_no_go(root, failed_check):
    stale = {"verdict": "PASS"}
    write_stale_pass(root)
    summary = checker.run_check(root)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"][failed_check] is False
    assert failed_check in summary["blocker"]
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale
    return summary


def test_frozen_json_shape(frozen_facts):
    facts = frozen_facts
    assert facts["schema_version"] == 1
    assert facts["frozen_on"] == "2026-11-28"
    assert facts["scope"] == "quant layout and Unsloth quant contract only, no timing"
    assert facts["note"] == "quant layout and Unsloth quant contract only, no timing"
    assert facts["speed_claim"] is None
    assert facts["facts"] == checker.EXPECTED_FACTS
    assert facts["needles"] == checker.SOURCE_NEEDLES
    assert facts["open_findings"] == checker.OPEN_FINDINGS
    assert facts["checkpoint_scale_parity"] == "UNVERIFIED"
    assert facts["provenance"]["source_sha256"] == checker.FROZEN_SOURCE_SHA256
    assert facts["provenance"]["header_probe"]["network_access_in_checker"] is False
    for value in checker.FROZEN_SOURCE_SHA256.values():
        assert len(value) == 64
        int(value, 16)


def test_check_names_are_unique_and_complete():
    assert len(checker.CHECK_NAMES) == 28
    assert len(checker.CHECK_NAMES) == len(set(checker.CHECK_NAMES))
    assert set(checker.CHECK_NAMES) == (
        {"frozen_facts", "frozen_source_sha256", "loopkit_present", "live_source_sha256", "run_completed"}
        | set(checker.NEEDLE_CHECK_NAMES)
        | set(checker.DERIVED_CHECK_NAMES)
    )
    assert len(checker.NEEDLE_CHECK_NAMES) == 17


def test_derive_elements_per_scale_literals():
    assert checker.derive_elements_per_scale(2304, 144) == 16
    assert checker.derive_elements_per_scale(1152, 72) == 16
    assert checker.derive_elements_per_scale(2304, 32) == 72
    assert checker.derive_elements_per_scale(16, 16) == 1
    for cols, scale_cols in ((0, 1), (2304, 0), (-2304, 144), (2304, True), (2304.0, 144), (2305, 144)):
        with pytest.raises(ValueError):
            checker.derive_elements_per_scale(cols, scale_cols)


def test_count_ggml_types_literal(frozen_facts):
    header = frozen_facts["facts"]["header"]
    assert checker.count_ggml_types(header["ggml_type_counts"]) == 744
    assert checker.count_ggml_types(header["ggml_type_counts"]) == header["n_tensors"]
    with pytest.raises(ValueError):
        checker.count_ggml_types({})
    with pytest.raises(ValueError):
        checker.count_ggml_types({"F32": -1})


def test_converter_block_semantics_accepts_frozen_and_rejects_tamper(frozen_facts):
    assert checker.check_converter_block_semantics(frozen_facts) is True
    before = copy.deepcopy(frozen_facts)
    payload = json.loads(json.dumps(frozen_facts))
    payload["facts"]["converter"]["nvfp4_block"] = 32
    assert checker.check_converter_block_semantics(payload) is False
    payload = json.loads(json.dumps(frozen_facts))
    payload["facts"]["converter"]["mxfp8_scale_dtype"] = "torch.float8_e4m3fn"
    assert checker.check_converter_block_semantics(payload) is False
    payload = json.loads(json.dumps(frozen_facts))
    payload["facts"]["converter"]["mxfp8_block_shape"] = [32, 32]
    assert checker.check_converter_block_semantics(payload) is False
    assert frozen_facts == before


def test_hf_shape_facts_accepts_frozen(frozen_facts):
    assert checker.check_hf_shape_facts(frozen_facts) is True


@pytest.mark.parametrize("mutate", [
    lambda f: f["facts"]["hf_shapes"]["nvfp4"]["w13_weight_scale"].__setitem__(2, 143),
    lambda f: f["facts"]["hf_shapes"]["nvfp4"].__setitem__("w13_weight", [128, 2304, 2304]),
    lambda f: f["facts"]["hf_shapes"]["nvfp4"].pop("w2_weight_amax"),
    lambda f: f["facts"]["hf_shapes"]["mxfp8"].__setitem__("w13_weight_amax", [128]),
    lambda f: f["facts"]["hf_shapes"]["mxfp8"].__setitem__("layers", [0, 3]),
    lambda f: f["facts"]["logical"].__setitem__("cols_w2", 1151),
], ids=["odd_scale_cols", "wrong_packed_w13", "missing_amax", "mxfp8_amax", "wrong_layers", "odd_logical_cols"])
def test_hf_shape_facts_rejects_tamper(frozen_facts, mutate):
    payload = json.loads(json.dumps(frozen_facts))
    mutate(payload)
    assert checker.check_hf_shape_facts(payload) is False


def test_header_facts_accepts_frozen_and_rejects_tamper(frozen_facts):
    assert checker.check_header_facts(frozen_facts) is True
    for path, value in (
        (("facts", "header", "kv", "general.architecture"), "llama"),
        (("facts", "header", "n_tensors"), 743),
        (("facts", "header", "loop_tensor_count"), 155),
        (("facts", "header", "first_16mib_sha256"), "0" * 64),
    ):
        payload = json.loads(json.dumps(frozen_facts))
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert checker.check_header_facts(payload) is False
    payload = json.loads(json.dumps(frozen_facts))
    payload["facts"]["header"]["loop_controllers"]["blk.6.loop_q_ctrl_down.weight"] = [64, 4608]
    assert checker.check_header_facts(payload) is False


def test_frozen_facts_and_hashes_validate(frozen_facts):
    assert checker.check_frozen_facts(frozen_facts) is True
    assert checker.check_frozen_sha256(frozen_facts) is True


@pytest.mark.parametrize("mutate", [
    lambda f: f.__setitem__("speed_claim", "2x faster"),
    lambda f: f.__setitem__("note", "timing"),
    lambda f: f.__setitem__("frozen_on", "2026-11-27"),
    lambda f: f.__setitem__("open_findings", []),
    lambda f: f.__setitem__("checkpoint_scale_parity", "PASS"),
    lambda f: f["needles"]["scheme_guard"]["needles"].__setitem__(0, "wrong needle"),
    lambda f: f["facts"]["converter"].__setitem__("mxfp8_block", 16),
    lambda f: f["facts"]["header"]["ggml_type_counts"].__setitem__("F32", 517),
    lambda f: f["provenance"]["source_sha256"].__setitem__(checker.CONVERTER_PY, "0" * 64),
], ids=["speed_claim", "note", "frozen_on", "no_findings", "parity_pass", "needle", "mxfp8_block", "type_counts", "frozen_hash"])
def test_frozen_facts_rejects_tamper(frozen_facts, mutate):
    payload = json.loads(json.dumps(frozen_facts))
    mutate(payload)
    assert checker.check_frozen_facts(payload) is False


def test_scale_layout_mismatch_open_is_enforced(frozen_facts):
    derived = checker.derive_elements_from_facts(frozen_facts)
    assert derived == {
        "nvfp4_w13": 16, "nvfp4_w2": 16, "mxfp8_w13": 16, "mxfp8_w2": 16,
    }
    assert checker.check_scale_layout_open(frozen_facts, derived) is True
    payload = json.loads(json.dumps(frozen_facts))
    payload["facts"]["converter"]["mxfp8_block"] = 16
    assert checker.check_scale_layout_open(payload, derived) is False
    payload = json.loads(json.dumps(frozen_facts))
    payload["checkpoint_scale_parity"] = "PASS"
    assert checker.check_scale_layout_open(payload, derived) is False


def test_pure_functions_do_not_mutate_inputs(frozen_facts):
    before = json.dumps(frozen_facts, sort_keys=True)
    checker.check_converter_block_semantics(frozen_facts)
    checker.check_hf_shape_facts(frozen_facts)
    checker.check_header_facts(frozen_facts)
    checker.check_frozen_facts(frozen_facts)
    checker.check_frozen_sha256(frozen_facts)
    checker.derive_elements_from_facts(frozen_facts)
    checker.check_scale_layout_open(frozen_facts, {"mxfp8_w13": 16})
    assert json.dumps(frozen_facts, sort_keys=True) == before


def test_real_fork_and_sibling_sources_match_frozen_contract(frozen_facts):
    for rel, digest in checker.FROZEN_SOURCE_SHA256.items():
        root = REPO_ROOT if checker.SOURCE_ROOTS[rel] == "repo" else REPO_ROOT.parent / "loopkit"
        path = root / rel
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for name, spec in checker.SOURCE_NEEDLES.items():
        root = REPO_ROOT if spec["root"] == "repo" else REPO_ROOT.parent / "loopkit"
        text = checker.read_text(root / spec["source"])
        assert all(needle in text for needle in spec["needles"]), name


def test_run_check_passes_on_real_tree(frozen_facts):
    summary = checker.run_check(REPO_ROOT)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == ""
    assert len(summary["checks"]) == len(checker.CHECK_NAMES)
    assert all(isinstance(value, bool) for value in summary["checks"].values())
    assert all(summary["checks"].values())
    assert summary["note"] == "quant layout and Unsloth quant contract only, no timing"
    assert summary["speed_claim"] is None
    assert summary["checkpoint_scale_parity"] == "UNVERIFIED"
    assert summary["open_findings"] == checker.OPEN_FINDINGS
    assert len(summary["open_findings"]) >= 1
    assert summary["derived"] == {
        "nvfp4_w13": 16, "nvfp4_w2": 16, "mxfp8_w13": 16, "mxfp8_w2": 16,
    }
    assert summary["frozen_facts_sha256"] == hashlib.sha256(
        (AUDIT_DIR / "QUANT_LAYOUT.json").read_bytes()).hexdigest()
    assert any("(gated)" in line for line in summary["info"])
    assert any("no network calls" in line for line in summary["info"])
    written = json.loads((AUDIT_DIR / "M33_QUANT_LAYOUT_SUMMARY.json").read_text(encoding="utf-8"))
    assert written == summary


def test_run_check_writes_valid_summary_on_fixture(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    summary = checker.run_check(root)
    assert summary["verdict"] == "PASS"
    written = json.loads(summary_path(root).read_text(encoding="utf-8"))
    assert written == summary


def test_cli_main_exit_codes(tmp_path, frozen_facts, monkeypatch, capsys):
    root, _ = make_fixture_tree(tmp_path)
    monkeypatch.setattr(checker, "REPO_ROOT", root)
    assert checker.main([]) == 0
    assert "verdict: PASS" in capsys.readouterr().out
    assert json.loads(summary_path(root).read_text(encoding="utf-8"))["verdict"] == "PASS"
    (root / checker.CONVERTER_PY).write_text("", encoding="utf-8")
    assert checker.main([]) == 1
    assert "verdict: NO_GO" in capsys.readouterr().out
    assert json.loads(summary_path(root).read_text(encoding="utf-8"))["verdict"] == "NO_GO"


def test_loopkit_env_override_can_point_to_moved_sibling(tmp_path, frozen_facts, monkeypatch):
    root, lk_root = make_fixture_tree(tmp_path)
    moved = tmp_path / "moved_loopkit"
    shutil.move(str(lk_root), str(moved))
    monkeypatch.setenv("LOOPKIT_ROOT", str(moved))
    assert checker.run_check(root)["verdict"] == "PASS"


def test_tampered_frozen_converter_fact_is_no_go(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    write_tampered_facts(root, frozen_facts, lambda f: f["facts"]["converter"].__setitem__("nvfp4_block", 32))
    assert_no_go(root, "frozen_facts")


def test_tampered_header_value_is_no_go(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    write_tampered_facts(root, frozen_facts, lambda f: f["facts"]["header"].__setitem__("n_tensors", 743))
    assert_no_go(root, "frozen_facts")


def test_tampered_needle_in_json_is_no_go(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    write_tampered_facts(root, frozen_facts, lambda f: f["needles"]["scheme_guard"]["needles"].__setitem__(0, "wrong"))
    assert_no_go(root, "frozen_facts")


def test_tampered_frozen_hash_is_no_go(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    write_tampered_facts(root, frozen_facts, lambda f: f["provenance"]["source_sha256"].__setitem__(
        checker.CONVERTER_PY, "0" * 64))
    assert_no_go(root, "frozen_source_sha256")


def test_cleared_open_findings_is_no_go(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    write_tampered_facts(root, frozen_facts, lambda f: f.__setitem__("open_findings", []))
    summary = assert_no_go(root, "frozen_facts")
    assert summary["open_findings"] == checker.OPEN_FINDINGS


def test_parity_pass_claim_is_no_go(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    write_tampered_facts(root, frozen_facts, lambda f: f.__setitem__("checkpoint_scale_parity", "PASS"))
    summary = assert_no_go(root, "frozen_facts")
    assert summary["checkpoint_scale_parity"] == "UNVERIFIED"


@pytest.mark.parametrize("name", [
    "scheme_guard",
    "stored_block_shape",
    "nvfp4_dequant",
    "mxfp8_dequant",
    "format_mxfp8_2d_tile",
    "integration_scale_warning",
], ids=["scheme", "block", "nvfp4", "mxfp8", "tile", "warning"])
def test_missing_source_needle_is_no_go(tmp_path, frozen_facts, name):
    root, lk_root = make_fixture_tree(tmp_path)
    path = needle_path(root, lk_root, name)
    text = path.read_text(encoding="utf-8")
    needle = checker.SOURCE_NEEDLES[name]["needles"][0]
    assert needle in text
    path.write_text(text.replace(needle, "OMITTED_" + name.upper()), encoding="utf-8")
    assert_no_go(root, "needle_" + name)


def test_live_converter_hash_mismatch_is_no_go(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    path = root / checker.CONVERTER_PY
    path.write_text(path.read_text(encoding="utf-8") + "\n# tamper\n", encoding="utf-8")
    assert_no_go(root, "live_source_sha256")


def test_live_doc_hash_mismatch_is_no_go(tmp_path, frozen_facts):
    root, lk_root = make_fixture_tree(tmp_path)
    path = lk_root / checker.INTEGRATION_MD
    path.write_text(path.read_text(encoding="utf-8") + "\ntamper\n", encoding="utf-8")
    assert_no_go(root, "live_source_sha256")


def test_missing_converter_source_is_no_go(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    (root / checker.CONVERTER_PY).unlink()
    summary = assert_no_go(root, "live_source_sha256")
    assert summary["checks"]["needle_scheme_guard"] is False


def test_missing_loopkit_sibling_is_no_go(tmp_path, frozen_facts):
    root, lk_root = make_fixture_tree(tmp_path)
    shutil.move(str(lk_root), str(tmp_path / "hidden_loopkit"))
    summary = assert_no_go(root, "loopkit_present")
    assert summary["checks"]["live_source_sha256"] is False


def test_missing_loopkit_env_override_is_no_go(tmp_path, frozen_facts, monkeypatch):
    root, _ = make_fixture_tree(tmp_path)
    monkeypatch.setenv("LOOPKIT_ROOT", str(tmp_path / "does_not_exist"))
    assert_no_go(root, "loopkit_present")


def test_malformed_facts_json_is_no_go(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    (root / checker.FACTS_REL).write_text("{not json\n", encoding="utf-8")
    summary = assert_no_go(root, "run_completed")
    assert any("exception:" in line for line in summary["info"])


def test_missing_facts_json_is_no_go(tmp_path, frozen_facts):
    root, _ = make_fixture_tree(tmp_path)
    (root / checker.FACTS_REL).unlink()
    summary = assert_no_go(root, "run_completed")
    assert summary["open_findings"] == checker.OPEN_FINDINGS


@pytest.mark.parametrize("rel", NEW_FILES)
def test_new_audit_files_are_ascii(rel):
    data = (AUDIT_DIR / rel).read_bytes()
    assert all(byte < 128 for byte in data)
