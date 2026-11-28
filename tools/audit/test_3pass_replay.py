"""Behavior tests for the fail-closed 3-pass loop replay / pass-to-slot gate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CHECKER = AUDIT_DIR / "check_3pass_replay.py"
CONTRACT = AUDIT_DIR / "THREEPASS_REPLAY.json"
SUMMARY_REL = "tools/audit/M41_3PASS_REPLAY_SUMMARY.json"
sys.path.insert(0, str(AUDIT_DIR))

import check_3pass_replay as checker

EXPECTED_TOTAL_CHECKS = 31
REAL_FIXTURE_PATH = Path("/tmp/opencode/synthetic.gguf")
REAL_FIXTURE_SHA256 = "cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617"
EXPECTED_CHECKER_SHA256 = "340d96ff2dd99020bdc029f48b2f2cdeb2ee58aac8e108bd6a6e78f7721e188f"
EXPECTED_CONTRACT_SHA256 = "00e7215637a42f08effaaa5a129846ffcbdb1297284f738b95c04df02ae4657b"
EXPECTED_PASS_TO_SLOT = {"0": 1, "1": 2, "2": 3}
EXPECTED_CORE_SLOTS = [1, 2, 3]
EXPECTED_NO_WEIGHTS_NOTE = (
    "No real weights and no speed claim; the engine runs only on a fixed-seed "
    "synthetic GGUF. This is not a no-engine-run gate."
)

FAKE_CLI = '''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

mode = os.environ.get("M41_FAKE_MODE", "")
on = os.environ.get("LLAMA_RORET") == "1"
dump = Path(os.environ["LLAMA_DUMP_TENSORS_DIR"])
dump.mkdir(parents=True, exist_ok=True)

if mode == "sentinel":
    summary = json.load(open(os.environ["M41_SUMMARY_PATH"], encoding="utf-8"))
    if summary.get("verdict") != "NO_GO" or summary.get("sentinel") is not True:
        sys.exit(3)

kinds = ("logits", "probs", "topk", "weights", "out")
rows = []
files = {}
for il in range(4):
    for kind in kinds:
        for occ in range(2):
            name = "ffn_moe_%s-%d" % (kind, il)
            row_name = name
            if mode == "bad_name" and name == "ffn_moe_logits-0" and occ == 0:
                row_name = "ffn_moe_logits-9"
            if mode == "bad_slot" and name == "ffn_moe_out-0" and occ == 0:
                row_name = "ffn_moe_out-9"
            if mode == "bad_occurrence" and name == "ffn_moe_probs-0" and occ == 1:
                continue
            filename = "%s__%d.bin" % (name, occ)
            if mode == "malformed_manifest" and name == "ffn_moe_topk-0" and occ == 0:
                rows.append("bad\\trow")
                continue
            fields = [row_name, "f32", "128", "2", "1", "1", "4", "512",
                      "1024", "1024", "1024", filename]
            rows.append("\\t".join(fields))
            blob = ("DATA:" + filename).encode("ascii")
            if mode == "bin_flip" and on and filename == "ffn_moe_logits-1__0.bin":
                blob = b"FLIPPED-BYTES"
            files[filename] = blob
(dump / "manifest.tsv").write_text("\\n".join(rows) + "\\n", encoding="utf-8")
for filename, blob in files.items():
    (dump / filename).write_bytes(blob)

print("PROMPT hello world")
if mode == "stdout_flip" and on:
    print("EXTRA ON ONLY LINE")
print("llama_moe_cache: LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache", file=sys.stderr)
if on:
    if mode != "enable_flip_on":
        print("llama_moe_cache: RORET union-resident MoE bookkeeping enabled", file=sys.stderr)
    passes = [0, 2] if mode == "missing_pass" else [0, 1, 2]
    for p in passes:
        ptr = 0x2 if mode == "slot_map_collapse" else 0x10 + p
        for i in range(12):
            print("RORET_BIND set lookup=0x%x slot_map=0x%x key_il=1 pass=%d" % (i + 1, ptr, p), file=sys.stderr)
    for i in range(12):
        print("RORET_BIND set lookup=0x%x slot_map=0x2 key_il=-1 pass=-1" % (i + 1), file=sys.stderr)
else:
    if mode == "bind_off":
        print("RORET_BIND set lookup=0x1 slot_map=0x2 key_il=1 pass=0", file=sys.stderr)
    if mode == "enable_flip_off":
        print("RORET union-resident MoE bookkeeping enabled", file=sys.stderr)
if mode == "nonzero":
    sys.exit(7)
sys.exit(0)
'''


def build_tree(tmp_path, mode=None):
    if not REAL_FIXTURE_PATH.is_file():
        pytest.skip("real synthetic fixture not present in this environment")
    root = tmp_path / "repo"
    (root / "tools/audit").mkdir(parents=True)
    out_dir = tmp_path / "logs"
    out_dir.mkdir()
    fixture = tmp_path / "fixture.gguf"
    shutil.copyfile(REAL_FIXTURE_PATH, fixture)
    (root / checker.CONTRACT_REL).write_bytes(CONTRACT.read_bytes())
    cli = tmp_path / "llama-debug"
    cli.write_text(FAKE_CLI, encoding="utf-8")
    cli.chmod(0o755)
    return root, cli, fixture, out_dir


def run_cli(root, cli, fixture, out_dir, mode=None):
    env = os.environ.copy()
    if mode:
        env["M41_FAKE_MODE"] = mode
    else:
        env.pop("M41_FAKE_MODE", None)
    return subprocess.run(
        [sys.executable, str(CHECKER),
         "--root", str(root), "--cli", str(cli), "--gguf", str(fixture),
         "--out-dir", str(out_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )


def read_summary(root):
    return json.loads((root / SUMMARY_REL).read_text(encoding="utf-8"))


def write_contract(root, payload):
    (root / checker.CONTRACT_REL).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def tamper_contract(root, mutate):
    payload = json.loads((root / checker.CONTRACT_REL).read_text(encoding="utf-8"))
    mutate(payload)
    write_contract(root, payload)


def assert_no_go(root, result, *failed_checks):
    assert result.returncode == 1, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    for name in failed_checks:
        assert name in summary["blocker"], (name, summary["blocker"])
        assert summary["checks"][name] is False
    assert summary["speed_claim"] is None
    compact = json.loads(result.stdout)
    assert compact["verdict"] == "NO_GO"
    assert compact["speed_claim"] is None
    return summary


def test_contract_is_frozen_and_self_consistent():
    facts = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert set(facts) == checker.EXPECTED_TOP_KEYS
    assert facts["verify"] == checker.CHECK_NAMES
    assert facts["schema_version"] == 1
    assert facts["frozen_on"] == "2026-11-28"
    assert facts["scope"] is None and facts["note"] is None
    assert facts["speed_claim"] is None
    assert facts["no_weights"] is True and facts["no_engine_run"] is False
    assert facts["no_weights_no_engine_run_note"] == EXPECTED_NO_WEIGHTS_NOTE
    assert facts["engine_run"] == "cpu_synthetic_only"
    assert "CycleAudit claim<=ci_low repeats>=20 loops=3" in facts["cycle_audit_required"]
    assert facts["expected_fixture_sha256"] == REAL_FIXTURE_SHA256
    assert facts["expected_fixture_bytes"] == 887392
    assert facts["expected_arch"] == "inkling2_loop"
    assert facts["loop_constants"] == checker.LOOP_CONSTANTS
    assert facts["expected_passes"] == [0, 1, 2]
    assert facts["expected_core_slots"] == EXPECTED_CORE_SLOTS
    assert facts["expected_il_slots"] == [0, 1, 2, 3]
    assert facts["pass_to_slot"] == EXPECTED_PASS_TO_SLOT
    assert facts["expected_tensor_kinds"] == ["logits", "probs", "topk", "weights", "out"]
    assert len(facts["expected_tensor_names"]) == 20
    assert facts["expected_occurrences_per_name"] == 2
    assert facts["expected_enable_needles"] == [
        "RORET union-resident MoE bookkeeping enabled"
    ]
    assert facts["informational_build_markers"] == [
        "LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache"
    ]
    assert facts["expected_trace_needles"] == ["RORET_BIND set"]
    assert facts["expected_manifest_fields"] == 12
    assert facts["expected_summary_path"] == SUMMARY_REL
    assert checker.sha256_bytes(CONTRACT.read_bytes()) == EXPECTED_CONTRACT_SHA256
    assert facts["checker_sha256"] == EXPECTED_CHECKER_SHA256
    assert checker.sha256_bytes(CHECKER.read_bytes()) == EXPECTED_CHECKER_SHA256
    assert checker.check_contract_keys(facts) is True
    assert checker.check_contract_values(facts) is True
    assert checker.check_checker_sha256(
        facts, checker.sha256_bytes(CHECKER.read_bytes())
    ) is True


def test_real_fixture_pin_matches_generator_output():
    path = Path("/tmp/opencode/synthetic.gguf")
    if not path.is_file():
        pytest.skip("real synthetic fixture not present in this environment")
    assert checker.sha256_file(path) == REAL_FIXTURE_SHA256
    assert path.stat().st_size == 887392


def test_check_names_are_unique_and_complete():
    assert len(checker.CHECK_NAMES) == EXPECTED_TOTAL_CHECKS
    assert len(checker.CHECK_NAMES) == len(set(checker.CHECK_NAMES))
    assert set(checker.CHECK_NAMES) == {
        "contract_json", "contract_keys", "contract_values", "checker_sha256",
        "no_network", "cli_exists", "cli_executable", "fixture_exists",
        "fixture_sha256", "fixture_bytes", "fixture_pin_authority",
        "fixture_arch", "off_exit_zero",
        "on_exit_zero", "off_absent_needles", "on_enable_needles",
        "on_trace_needles", "off_manifest_ok", "on_manifest_ok",
        "off_name_set", "on_name_set", "off_slot_set", "on_slot_set",
        "off_occurrences", "on_occurrences", "bind_pass_set",
        "bind_counts_equal_positive", "bin_parity", "stdout_parity",
        "slot_map_distinct_per_pass", "run_completed",
    }
    assert checker.check_no_network() is True


def test_derive_pass_to_slot():
    assert checker.derive_pass_to_slot(checker.LOOP_CONSTANTS) == {
        "pass_to_slot": EXPECTED_PASS_TO_SLOT,
        "core_slots": EXPECTED_CORE_SLOTS,
    }
    assert checker.PASS_TO_SLOT == EXPECTED_PASS_TO_SLOT
    assert checker.IL_SLOTS == [0, 1, 2, 3]
    assert len(checker.EXPECTED_NAMES) == 20
    bad = dict(checker.LOOP_CONSTANTS, CORE_END=3)
    with pytest.raises(ValueError):
        checker.derive_pass_to_slot(bad)
    with pytest.raises(ValueError):
        checker.derive_pass_to_slot({"N_LAYER": 2})


def test_pure_manifest_and_binding_helpers(tmp_path):
    assert checker.parse_tensor_name("ffn_moe_logits-1") == ("logits", 1)
    assert checker.parse_tensor_name("ffn_moe_out-3") == ("out", 3)
    assert checker.parse_tensor_name("other-1") is None
    path = tmp_path / "manifest.tsv"
    path.write_text(
        "ffn_moe_logits-0\tf32\t128\t2\t1\t1\t4\t512\t1024\t1024\t1024\tffn_moe_logits-0__0.bin\n"
        "ffn_moe_logits-0\tf32\t128\t2\t1\t1\t4\t512\t1024\t1024\t1024\tffn_moe_logits-0__1.bin\n"
        "bad\trow\n",
        encoding="utf-8",
    )
    problems = []
    rows = checker.parse_manifest(path, problems)
    assert len(rows) == 2 and len(problems) == 1
    summary = checker.summarize_rows(rows)
    assert summary["names"] == ["ffn_moe_logits-0"]
    assert summary["slots"] == [0]
    assert summary["occurrences"] == {"ffn_moe_logits-0": 2}
    text = (
        "RORET_BIND set lookup=0x1 slot_map=0x2 key_il=1 pass=0\n"
        "RORET_BIND set lookup=0x3 slot_map=0x2 key_il=1 pass=1\n"
        "RORET_BIND set lookup=0x4 slot_map=0x2 key_il=-1 pass=-1\n"
    )
    binds = checker.parse_bindings(text)
    assert binds == [(1, 0), (1, 1), (-1, -1)]
    assert checker.bind_counts(binds, 1) == {0: 1, 1: 1}
    assert checker.parse_slot_maps(text) == {0: ["0x2"], 1: ["0x2"]}
    assert checker.slot_maps_distinct_per_pass({0: ["0x2"], 1: ["0x3"], 2: ["0x4"]}) is True
    assert checker.slot_maps_distinct_per_pass({0: ["0x2"], 1: ["0x2"], 2: ["0x2"]}) is False
    assert checker.stdout_differing_bytes(b"abc", b"abc") == 0
    assert checker.stdout_differing_bytes(b"abc", b"axc") == 1


def test_happy_path_passes_with_fake_dump(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == []
    assert summary["check_counts"] == {
        "total": EXPECTED_TOTAL_CHECKS, "passed": EXPECTED_TOTAL_CHECKS, "failed": 0,
    }
    assert all(value is True for value in summary["checks"].values())
    assert summary["pass_to_slot"] == EXPECTED_PASS_TO_SLOT
    assert summary["expected_core_slots"] == EXPECTED_CORE_SLOTS
    assert summary["bind_passes_on"] == [0, 1, 2]
    assert summary["bind_counts_on"] == {"0": 12, "1": 12, "2": 12}
    assert summary["bind_sentinel_count_on"] == 12
    assert summary["bind_rows_off"] == 0
    assert summary["off_manifest_rows"] == 40 and summary["on_manifest_rows"] == 40
    assert summary["off_tensor_names"] == 20 and summary["on_tensor_names"] == 20
    assert summary["off_slots"] == [0, 1, 2, 3] and summary["on_slots"] == [0, 1, 2, 3]
    assert summary["off_occurrence_counts"] == [2] and summary["on_occurrence_counts"] == [2]
    assert summary["bin_files_compared"] == 40
    assert summary["bin_bytes_compared"] > 0
    assert summary["bin_differing_bytes"] == 0
    assert summary["stdout_differing_bytes"] == 0
    assert any("set-level co-occurrence" in line for line in summary["info"])
    assert any("unconditional build marker off=1 on=1" in line for line in summary["info"])
    assert "sentinel" not in summary
    compact = json.loads(result.stdout)
    assert compact["verdict"] == "PASS"
    assert compact["pass_to_slot"] == EXPECTED_PASS_TO_SLOT
    assert compact["speed_claim"] is None


def test_occurrence_index_is_not_a_pass_claim(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert any("occurrence index encodes graph eval count, not pass" in line
               for line in summary["info"])


def test_bad_tensor_name_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="bad_name")
    assert_no_go(root, result, "off_name_set", "on_name_set")


def test_bad_slot_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="bad_slot")
    assert_no_go(root, result, "on_slot_set", "on_name_set")


def test_bad_occurrence_count_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="bad_occurrence")
    assert_no_go(root, result, "off_occurrences", "on_occurrences")


def test_malformed_manifest_row_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="malformed_manifest")
    assert_no_go(root, result, "off_manifest_ok", "on_manifest_ok")


def test_missing_pass_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="missing_pass")
    summary = assert_no_go(root, result, "bind_pass_set", "bind_counts_equal_positive")
    assert summary["bind_passes_on"] == [0, 2]
    assert "slot_map_distinct_per_pass" in summary["blocker"]


def test_shared_slot_map_pointer_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    summary = assert_no_go(
        root, run_cli(root, cli, fixture, out_dir, mode="slot_map_collapse"),
        "slot_map_distinct_per_pass",
    )
    assert summary["slot_map_pointer_counts_on"] == {"0": 1, "1": 1, "2": 1}


def test_bind_rows_off_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="bind_off")
    summary = assert_no_go(root, result, "off_absent_needles")
    assert summary["bind_rows_off"] >= 1


def test_enable_marker_flip_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="enable_flip_on")
    assert_no_go(root, result, "on_enable_needles")
    root, cli, fixture, out_dir = build_tree(tmp_path / "second")
    result = run_cli(root, cli, fixture, out_dir, mode="enable_flip_off")
    assert_no_go(root, result, "off_absent_needles")


def test_byte_flip_in_bin_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="bin_flip")
    summary = assert_no_go(root, result, "bin_parity")
    assert summary["bin_differing_bytes"] >= 1


def test_stdout_mismatch_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="stdout_flip")
    summary = assert_no_go(root, result, "stdout_parity")
    assert summary["stdout_differing_bytes"] >= 1


def test_nonzero_engine_exit_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="nonzero")
    summary = assert_no_go(root, result, "off_exit_zero", "on_exit_zero")
    assert summary["off_exit"] == 7 and summary["on_exit"] == 7


def test_wrong_fixture_sha_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    tamper_contract(root, lambda facts: facts.update(expected_fixture_sha256="0" * 64))
    assert_no_go(root, run_cli(root, cli, fixture, out_dir),
                 "fixture_sha256", "fixture_pin_authority")


def test_wrong_fixture_bytes_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    tamper_contract(root, lambda facts: facts.update(expected_fixture_bytes=1))
    assert_no_go(root, run_cli(root, cli, fixture, out_dir),
                 "fixture_bytes", "fixture_pin_authority")


def test_checker_hash_drift_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    tamper_contract(root, lambda facts: facts.update(checker_sha256="1" * 64))
    assert_no_go(root, run_cli(root, cli, fixture, out_dir), "checker_sha256")


def test_contract_key_drift_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    tamper_contract(root, lambda facts: facts.update(surprise=True))
    assert_no_go(root, run_cli(root, cli, fixture, out_dir), "contract_keys")
    root, cli, fixture, out_dir = build_tree(tmp_path / "drop")
    def drop(facts):
        del facts["expected_arch"]
    tamper_contract(root, drop)
    assert_no_go(root, run_cli(root, cli, fixture, out_dir), "contract_keys")


def test_malformed_contract_json_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    (root / checker.CONTRACT_REL).write_text("{not json", encoding="utf-8")
    assert_no_go(root, run_cli(root, cli, fixture, out_dir), "contract_json")


def test_missing_cli_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    summary = assert_no_go(
        root, run_cli(root, tmp_path / "no-such-cli", fixture, out_dir),
        "cli_exists", "cli_executable", "off_exit_zero",
    )
    assert summary["fixture_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()


def test_sentinel_is_written_before_engine_runs(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="sentinel")
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_summary(root)["verdict"] == "PASS"


def test_stale_pass_is_overwritten(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    checker.write_json(root / SUMMARY_REL, {"verdict": "PASS", "stale": True})
    tamper_contract(root, lambda facts: facts.update(checker_sha256="2" * 64))
    summary = assert_no_go(root, run_cli(root, cli, fixture, out_dir), "checker_sha256")
    assert "stale" not in summary
    assert summary["check_counts"]["passed"] < EXPECTED_TOTAL_CHECKS


def test_atomic_write_failure_returns_one(tmp_path, monkeypatch):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    real_write = checker.write_json
    state = {"calls": 0}
    def flaky_write(path, payload):
        state["calls"] += 1
        if payload.get("sentinel") is not True:
            raise OSError("simulated final write failure")
        return real_write(path, payload)
    monkeypatch.setattr(checker, "write_json", flaky_write)
    rc = checker.main([
        "--root", str(root), "--cli", str(cli),
        "--gguf", str(fixture), "--out-dir", str(out_dir),
    ])
    assert rc == 1
    assert state["calls"] == 2
    payload = read_summary(root)
    assert payload["verdict"] == "NO_GO"
    assert payload["sentinel"] is True
