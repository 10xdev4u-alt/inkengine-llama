"""Tests for the frozen Inkling2-loop contract checker."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
sys.path.insert(0, str(AUDIT_DIR))

import check_upstream_loop_contract as checker


@pytest.fixture()
def frozen_facts():
    return checker.load_json(AUDIT_DIR / "UPSTREAM_LOOP_CONTRACT.json")


def constants_text(arch_int, tensor_names, arch_string="inkling2_loop"):
    lines = ["class MODEL_ARCH(IntEnum):"]
    for i in range(arch_int - 1):
        lines.append("    M%d = auto()" % i)
    lines.append("    INKLING2_LOOP = auto()")
    lines.append("")
    lines.append("MODEL_ARCH_NAMES = {")
    lines.append('    MODEL_ARCH.INKLING2_LOOP: "%s",' % arch_string)
    lines.append("}")
    lines.append("MODEL_TENSORS = {")
    lines.append("    MODEL_ARCH.INKLING2_LOOP: [")
    for name in tensor_names:
        lines.append("        MODEL_TENSOR.%s," % name)
    lines.append("    ],")
    lines.append("}")
    return "\n".join(lines) + "\n"


def cpp_text(keys, formula):
    lines = ["void load() {"]
    for key in keys:
        lines.append("    ml.get_key(%s, x);" % key)
    lines.append("    const size_t n_slot = %s;" % formula)
    lines.append("}")
    return "\n".join(lines) + "\n"


def extracted_from_facts(facts):
    return {
        "arch_int": facts["fork_arch_int"],
        "arch_string": facts["arch_string"],
        "tensor_names": list(facts["tensor_names"]),
        "loop_kv_keys": list(facts["loop_kv_keys"]),
        "slot_formula": facts["slot_formula"],
        "wiring": {marker["name"]: True for marker in facts["rorret_wiring"]},
    }


def test_frozen_json_shape(frozen_facts):
    facts = frozen_facts
    assert facts["schema_version"] == 1
    assert facts["frozen_on"] == "2026-11-28"
    assert facts["scope"] == "contract parity only, no timing"
    assert facts["speed_claim"] is None
    assert facts["arch_string"] == "inkling2_loop"
    assert facts["fork_arch_int"] == 179
    assert facts["upstream_arch_int"] == 181
    assert facts["tensor_count"] == 47
    assert len(facts["tensor_names"]) == 47
    assert len(set(facts["tensor_names"])) == 47
    assert len(facts["loop_kv_keys"]) == 5
    assert facts["slot_formula"] == "n_phys + (size_t) (n_passes - 1)*(core_end - core_start)"
    assert len(facts["rorret_wiring"]) == 5
    for marker in facts["rorret_wiring"]:
        assert set(marker) == {"name", "file", "needle"}
    hashes = facts["provenance"]["upstream_sha256"]
    for value in hashes.values():
        assert len(value) == 64
        int(value, 16)


def test_extract_arch_int_counts_auto_members():
    assert checker.extract_arch_int(constants_text(3, ["TOKEN_EMBD"])) == 3
    bad = "class MODEL_ARCH(IntEnum):\n    A = 1\n    INKLING2_LOOP = auto()\n"
    with pytest.raises(ValueError):
        checker.extract_arch_int(bad)


def test_extract_arch_string():
    assert checker.extract_arch_string(constants_text(179, ["A"])) == "inkling2_loop"
    with pytest.raises(ValueError):
        checker.extract_arch_string("x = 1\n")


def test_extract_tensor_names():
    text = constants_text(179, ["TOKEN_EMBD", "ATTN_Q"])
    assert checker.extract_tensor_names(text) == ["TOKEN_EMBD", "ATTN_Q"]
    with pytest.raises(ValueError):
        checker.extract_tensor_names(text + text)


def test_extract_loop_kv_keys():
    keys = ["LLM_KV_LOOP_START", "LLM_KV_LOOP_END", "LLM_KV_LOOP_COUNT", "LLM_KV_LOOP_GATE_RANK", "LLM_KV_LOOP_VARYING"]
    text = cpp_text(keys + ["LLM_KV_LOOP_START"], "n_phys")
    assert checker.extract_loop_kv_keys(text) == sorted(keys)
    with pytest.raises(ValueError):
        checker.extract_loop_kv_keys("void f() {}\n")


def test_extract_slot_formula():
    formula = "n_phys + (size_t) (n_passes - 1)*(core_end - core_start)"
    text = cpp_text(["LLM_KV_LOOP_START"], formula)
    assert checker.extract_slot_formula(text) == formula
    with pytest.raises(ValueError):
        checker.extract_slot_formula(cpp_text([], formula) + cpp_text([], formula))


def test_extract_facts_roundtrip(frozen_facts):
    facts = frozen_facts
    constants = constants_text(facts["fork_arch_int"], facts["tensor_names"], facts["arch_string"])
    loop = cpp_text(facts["loop_kv_keys"], facts["slot_formula"])
    extracted = checker.extract_facts(constants, loop)
    assert extracted["arch_int"] == facts["fork_arch_int"]
    assert extracted["arch_string"] == facts["arch_string"]
    assert extracted["tensor_names"] == facts["tensor_names"]
    assert extracted["loop_kv_keys"] == facts["loop_kv_keys"]
    assert extracted["slot_formula"] == facts["slot_formula"]


def test_check_wiring_markers():
    markers = [
        {"name": "a", "file": "a.cpp", "needle": "GATE"},
        {"name": "b", "file": "b.cpp", "needle": "PARAMS"},
    ]
    result = checker.check_wiring_markers({"a.cpp": "x GATE y", "b.cpp": "none"}, markers)
    assert result == {"a": True, "b": False}


def test_real_fork_files_match_frozen_contract(frozen_facts):
    facts = frozen_facts
    sources = facts["provenance"]["fork_sources"]
    constants = checker.read_text(REPO_ROOT / sources["constants"])
    loop = checker.read_text(REPO_ROOT / sources["loop_cpp"])
    extracted = checker.extract_facts(constants, loop)
    extracted["wiring"] = checker.check_wiring_markers(
        checker.read_wiring_sources(REPO_ROOT, facts["rorret_wiring"]), facts["rorret_wiring"]
    )
    result = checker.compare_contract(facts, extracted)
    assert result["overall"] == "PASS"
    assert all(check["result"] == "PASS" for check in result["checks"].values())


def test_unmutated_contract_passes(frozen_facts):
    result = checker.compare_contract(frozen_facts, extracted_from_facts(frozen_facts))
    assert result["overall"] == "PASS"
    assert all(check["result"] == "PASS" for check in result["checks"].values())


@pytest.mark.parametrize(
    "field,mutation,check_name",
    [
        ("tensor_names", lambda v: v[:-1], "tensor_names"),
        ("loop_kv_keys", lambda v: v[:-1], "loop_kv_keys"),
        ("arch_int", lambda v: v + 1, "fork_arch_int"),
        ("arch_string", lambda v: v + "_x", "arch_string"),
        ("slot_formula", lambda v: v + " + 1", "slot_formula"),
    ],
)
def test_mutated_contract_fields_are_no_go(frozen_facts, field, mutation, check_name):
    extracted = extracted_from_facts(frozen_facts)
    extracted[field] = mutation(extracted[field])
    result = checker.compare_contract(frozen_facts, extracted)
    assert result["overall"] == "NO_GO"
    assert result["checks"][check_name]["result"] == "FAIL"


def test_missing_wiring_marker_is_no_go(frozen_facts):
    extracted = extracted_from_facts(frozen_facts)
    extracted["wiring"]["gate"] = False
    result = checker.compare_contract(frozen_facts, extracted)
    assert result["overall"] == "NO_GO"
    assert result["checks"]["wiring_gate"]["result"] == "FAIL"


def test_missing_extracted_fields_fail_closed(frozen_facts):
    result = checker.compare_contract(frozen_facts, {})
    assert result["overall"] == "NO_GO"
    assert all(check["result"] == "FAIL" for check in result["checks"].values())


def test_upstream_arch_int_drift_is_info_only(frozen_facts):
    extracted = extracted_from_facts(frozen_facts)
    extracted["upstream_arch_int"] = frozen_facts["upstream_arch_int"] + 7
    result = checker.compare_contract(frozen_facts, extracted)
    assert result["overall"] == "PASS"
    assert all(check["result"] == "PASS" for check in result["checks"].values())
    assert any("INFO only" in line for line in result["info"])


def test_run_check_writes_valid_summary(tmp_path, frozen_facts):
    facts = frozen_facts
    audit = tmp_path / "tools/audit"
    audit.mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "UPSTREAM_LOOP_CONTRACT.json", audit / "UPSTREAM_LOOP_CONTRACT.json")
    (tmp_path / "gguf-py/gguf").mkdir(parents=True)
    (tmp_path / "src/models").mkdir(parents=True)
    constants = constants_text(facts["fork_arch_int"], facts["tensor_names"], facts["arch_string"])
    (tmp_path / "gguf-py/gguf/constants.py").write_text(constants, encoding="utf-8")
    for marker in facts["rorret_wiring"]:
        path = tmp_path / marker["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker["needle"] + "\n", encoding="utf-8")
    (tmp_path / "src/models/inkling2-loop.cpp").write_text(
        cpp_text(facts["loop_kv_keys"], facts["slot_formula"]) + "ip, (int32_t) pass\n",
        encoding="utf-8",
    )
    summary = checker.run_check(tmp_path)
    assert summary["verdict"] == "PASS"
    assert summary["speed_claim"] is None
    assert summary["note"] == "contract parity only, no timing"
    assert summary["mode"] == "frozen"
    written = json.loads((audit / "M29_UPSTREAM_CONTRACT_SUMMARY.json").read_text(encoding="utf-8"))
    assert written == summary


def test_extraction_failure_overwrites_stale_summary(tmp_path, frozen_facts):
    facts = frozen_facts
    audit = tmp_path / "tools/audit"
    audit.mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "UPSTREAM_LOOP_CONTRACT.json", audit / "UPSTREAM_LOOP_CONTRACT.json")
    stale = {"verdict": "PASS"}
    checker.write_json(audit / "M29_UPSTREAM_CONTRACT_SUMMARY.json", stale)
    sources = facts["provenance"]["fork_sources"]
    bad = tmp_path / sources["constants"]
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("class MODEL_ARCH(IntEnum):\n    A = auto()\n", encoding="utf-8")
    loop = tmp_path / sources["loop_cpp"]
    loop.parent.mkdir(parents=True, exist_ok=True)
    loop.write_text(
        cpp_text(facts["loop_kv_keys"], facts["slot_formula"]) + "ip, (int32_t) pass\n",
        encoding="utf-8",
    )
    for marker in facts["rorret_wiring"]:
        path = tmp_path / marker["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker["needle"] + "\n", encoding="utf-8")
    summary = checker.run_check(tmp_path)
    assert summary["verdict"] == "NO_GO"
    assert summary["checks"]["extraction"]["result"] == "FAIL"
    written = json.loads((audit / "M29_UPSTREAM_CONTRACT_SUMMARY.json").read_text(encoding="utf-8"))
    assert written == summary
    assert written != stale
