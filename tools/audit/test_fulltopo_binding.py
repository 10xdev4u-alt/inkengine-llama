"""Behavior and tamper tests for the fail-closed full-topology union-wiring gate."""

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
CHECKER = AUDIT_DIR / "check_fulltopo_binding.py"
CONTRACT = AUDIT_DIR / "FULLTOPO_BINDING.json"
GENERATOR = AUDIT_DIR / "make_fulltopo_gguf.py"
SUMMARY_REL = "tools/audit/M45_FULLTOPO_BINDING_SUMMARY.json"
sys.path.insert(0, str(AUDIT_DIR))

import check_fulltopo_binding as checker

EXPECTED_TOTAL_CHECKS = 35
REAL_FIXTURE_PATH = Path("/tmp/opencode/fulltopo-probe.gguf")
REAL_FIXTURE_SHA256 = "3ccf2fb4c6ef4fab60464d2a35a2ea42b24186db3e8b12cd68881f4fa873dff7"
REAL_FIXTURE_BYTES = 11941248
EXPECTED_CHECKER_SHA256 = "81d572430c3ba35b7a63a69c8f099b4f9facdcbed2baf0e492dbccb3a9b696cd"
EXPECTED_CONTRACT_SHA256 = "eae32687a5a68c35165158d650deb42f5385ab7dbc5a54384f0bfd2c7850b50c"
EXPECTED_CORE_IL = list(range(6, 18))
EXPECTED_PASSES = [0, 1, 2]
EXPECTED_PASS_TO_SLOT = {
    "0": list(range(6, 18)),
    "1": list(range(24, 36)),
    "2": list(range(36, 48)),
}
EXPECTED_NO_WEIGHTS_NOTE = (
    "No real weights and no speed claim; the engine runs only on a fixed-seed "
    "synthetic GGUF. This is not a no-engine-run gate."
)

FAKE_CLI = '''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

mode = os.environ.get("M45_FAKE_MODE", "")
on = os.environ.get("LLAMA_RORET") == "1"
dump = Path(os.environ["LLAMA_DUMP_TENSORS_DIR"])
dump.mkdir(parents=True, exist_ok=True)

if mode == "sentinel":
    summary = json.load(open(os.environ["M45_SUMMARY_PATH"], encoding="utf-8"))
    if summary.get("verdict") != "NO_GO" or summary.get("sentinel") is not True:
        sys.exit(3)

kinds = ("logits", "probs", "topk", "weights", "out")
rows = []
files = {}
for il in range(48):
    for kind in kinds:
        for occ in range(2):
            name = "ffn_moe_%s-%d" % (kind, il)
            row_name = name
            if mode == "mutated_slot" and name == "ffn_moe_out-6" and occ == 0:
                row_name = "ffn_moe_out-99"
            if mode == "bad_occurrence" and name == "ffn_moe_probs-6" and occ == 1:
                continue
            filename = "%s__%d.bin" % (name, occ)
            if mode == "malformed_manifest" and name == "ffn_moe_topk-6" and occ == 0:
                rows.append("bad\\trow")
                continue
            fields = [row_name, "f32", "128", "2", "1", "1", "4", "512",
                      "1024", "1024", "1024", filename]
            rows.append("\\t".join(fields))
            blob = ("DATA:" + filename).encode("ascii")
            if mode == "bin_flip" and on and filename == "ffn_moe_logits-6__0.bin":
                blob = b"FLIPPED-BYTES"
            files[filename] = blob
(dump / "manifest.tsv").write_text("\\n".join(rows) + "\\n", encoding="utf-8")
for filename, blob in files.items():
    (dump / filename).write_bytes(blob)

print("PROMPT hello world")
if mode == "stdout_flip" and on:
    print("EXTRA ON ONLY LINE")
print("[ Prompt: 1.00 t/s | Generation: 2.00 t/s ]")
print("llama_moe_cache: LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache", file=sys.stderr)
if on:
    if mode != "enable_flip_on":
        print("llama_moe_cache: RORET union-resident MoE bookkeeping enabled", file=sys.stderr)
    passes = [0, 2] if mode == "missing_pass" else [0, 1, 2]
    for p in passes:
        ptr = 0x100 + p
        for i in range(6, 18):
            key = 5 if (mode == "wrong_key_il" and i == 6) else i
            print("RORET_BIND set lookup=0x%x slot_map=0x%x key_il=%d pass=%d" %
                  (i + 1, ptr, key, p), file=sys.stderr)
    if mode == "rorret_pass":
        for p in passes:
            print("llama_moe_cache: RORET_PASS il=6 pass=%d plans=1 hits=0 "
                  "upload_ops=1 uploaded_bytes=1 residents=0" % p, file=sys.stderr)
else:
    if mode == "bind_off":
        print("RORET_BIND set lookup=0x1 slot_map=0x2 key_il=6 pass=0", file=sys.stderr)
    if mode == "enable_flip_off":
        print("RORET union-resident MoE bookkeeping enabled", file=sys.stderr)
if mode == "nonzero":
    sys.exit(7)
sys.exit(0)
'''


def build_tree(tmp_path, mode=None):
    if not REAL_FIXTURE_PATH.is_file():
        pytest.skip("full-topology fixture not present in this environment")
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
        env["M45_FAKE_MODE"] = mode
    else:
        env.pop("M45_FAKE_MODE", None)
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
    assert summary["no_weights"] is True
    assert summary["engine_run"] == "cpu_synthetic_only"
    compact = json.loads(result.stdout)
    assert compact["verdict"] == "NO_GO"
    assert compact["speed_claim"] is None
    return summary


def test_contract_is_frozen_and_self_consistent():
    facts = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert set(facts) == checker.EXPECTED_TOP_KEYS
    assert facts["verify"] == checker.CHECK_NAMES
    assert facts["schema_version"] == 1 and facts["frozen_on"] == "2026-11-28"
    assert facts["scope"] is None and facts["note"] is None and facts["speed_claim"] is None
    assert facts["no_weights"] is True and facts["no_engine_run"] is False
    assert facts["no_weights_no_engine_run_note"] == EXPECTED_NO_WEIGHTS_NOTE
    assert facts["engine_run"] == "cpu_synthetic_only"
    assert "CycleAudit claim<=ci_low repeats>=20 loops=3" in facts["cycle_audit_required"]
    assert facts["expected_fixture_sha256"] == REAL_FIXTURE_SHA256
    assert facts["expected_fixture_bytes"] == REAL_FIXTURE_BYTES
    assert facts["expected_arch"] == "inkling2_loop"
    assert facts["loop_constants"] == checker.LOOP_CONSTANTS
    assert facts["expected_core_il"] == EXPECTED_CORE_IL
    assert facts["expected_passes"] == EXPECTED_PASSES
    assert facts["pass_to_slot"] == EXPECTED_PASS_TO_SLOT
    assert facts["expected_tensor_kinds"] == ["logits", "probs", "topk", "weights", "out"]
    assert len(facts["expected_tensor_names"]) == 240
    assert facts["expected_occurrences"] == 2
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


def test_fixture_pin_matches_generator_output():
    path = REAL_FIXTURE_PATH
    if not path.is_file():
        pytest.skip("full-topology fixture not present in this environment")
    assert checker.sha256_file(path) == REAL_FIXTURE_SHA256
    assert path.stat().st_size == REAL_FIXTURE_BYTES
    gguf_py = REPO_ROOT / "gguf-py"
    sys.path.insert(0, str(AUDIT_DIR))
    sys.path.insert(0, str(gguf_py))
    try:
        import make_fulltopo_gguf as gen
    except ImportError:
        pytest.skip("numpy/gguf-py not importable in this environment")
    assert (gen.N_LAYER, gen.CORE_START, gen.CORE_END) == (24, 6, 18)
    assert (gen.N_PASSES, gen.GATE_RANK, gen.ATTN_RES_BLOCK) == (3, 64, 6)
    expected_swa = [i % 6 != 5 for i in range(24)]
    expected_swa[0] = False
    assert gen.IS_SWA == expected_swa
    assert gen.IS_SWA[0] is False
    assert [i for i, value in enumerate(gen.IS_SWA) if not value] == [0, 5, 11, 17, 23]


def test_check_names_are_unique_and_complete():
    assert len(checker.CHECK_NAMES) == EXPECTED_TOTAL_CHECKS
    assert len(checker.CHECK_NAMES) == len(set(checker.CHECK_NAMES))
    assert checker.check_no_network() is True
    assert "RORET_PASS" not in checker.CHECK_NAMES


def test_derive_pass_to_slot_and_expected_names():
    assert checker.derive_pass_to_slot(checker.LOOP_CONSTANTS) == {
        "pass_to_slot": EXPECTED_PASS_TO_SLOT,
        "core_il": EXPECTED_CORE_IL,
        "il_slots": list(range(48)),
    }
    assert checker.PASS_TO_SLOT == EXPECTED_PASS_TO_SLOT
    assert checker.CORE_IL == EXPECTED_CORE_IL
    assert checker.IL_SLOTS == list(range(48))
    assert len(checker.EXPECTED_NAMES) == 240
    assert checker.EXPECTED_EXPECTED_ROWS == 480
    with pytest.raises(ValueError):
        checker.derive_pass_to_slot({"N_LAYER": 24})
    with pytest.raises(ValueError):
        checker.derive_pass_to_slot(dict(checker.LOOP_CONSTANTS, CORE_END=25))


def test_pure_manifest_binding_and_normalize_helpers(tmp_path):
    assert checker.parse_tensor_name("ffn_moe_logits-6") == ("logits", 6)
    assert checker.parse_tensor_name("ffn_moe_out-47") == ("out", 47)
    assert checker.parse_tensor_name("other-1") is None
    path = tmp_path / "manifest.tsv"
    path.write_text(
        "ffn_moe_logits-6\tf32\t128\t2\t1\t1\t4\t512\t1024\t1024\t1024\t"
        "ffn_moe_logits-6__0.bin\n"
        "ffn_moe_logits-6\tf32\t128\t2\t1\t1\t4\t512\t1024\t1024\t1024\t"
        "ffn_moe_logits-6__1.bin\n"
        "bad\trow\n",
        encoding="utf-8",
    )
    problems = []
    rows = checker.parse_manifest(path, problems)
    assert len(rows) == 2 and len(problems) == 1
    summary = checker.summarize_rows(rows)
    assert summary["names"] == ["ffn_moe_logits-6"]
    assert summary["slots"] == [6]
    assert summary["occurrences"] == {"ffn_moe_logits-6": 2}
    text = (
        "RORET_BIND set lookup=0x1 slot_map=0x10 key_il=6 pass=0\n"
        "RORET_BIND set lookup=0x2 slot_map=0x11 key_il=6 pass=1\n"
        "RORET_BIND set lookup=0x3 slot_map=0x12 key_il=6 pass=2\n"
        "RORET_BIND set lookup=0x4 slot_map=0x13 key_il=-1 pass=-1\n"
    )
    binds = checker.parse_bindings(text)
    assert len(binds) == 4 and binds[0]["slot_map"] == "0x10"
    cov = checker.binding_coverage(binds)
    assert cov["ok"] is False and cov["valid_rows"] == 3
    assert checker.bind_counts(binds) == {0: 1, 1: 1, 2: 1}
    assert checker.parse_slot_maps(binds) == {0: ["0x10"], 1: ["0x11"], 2: ["0x12"]}
    assert checker.slot_maps_distinct_per_pass(checker.parse_slot_maps(binds)) is True
    assert checker.slot_maps_distinct_per_pass({0: ["0x10"], 1: ["0x10"], 2: ["0x12"]}) is False
    kept, dropped = checker.normalize_stdout(
        b"RORET_BIND set key_il=6 pass=0\n"
        b"[ Prompt: 1.00 t/s | Generation: 2.00 t/s ]\n"
        b"KEEP\n"
    )
    assert kept == b"KEEP\n" and dropped == 2
    assert checker.stdout_differing_bytes(b"abc", b"abc") == 0
    assert checker.stdout_differing_bytes(b"abc", b"axc") == 1


def test_happy_path_passes_with_fake_dump(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "PASS" and summary["blocker"] == []
    assert summary["check_counts"] == {
        "total": EXPECTED_TOTAL_CHECKS, "passed": EXPECTED_TOTAL_CHECKS, "failed": 0,
    }
    assert all(value is True for value in summary["checks"].values())
    assert summary["expected_bind_pairs"] == 36
    assert len(summary["bind_pairs_on"]) == 36
    assert summary["missing_bind_pairs"] == []
    assert summary["bind_counts_on"] == {"0": 12, "1": 12, "2": 12}
    assert summary["off_manifest_rows"] == 480 and summary["on_manifest_rows"] == 480
    assert summary["off_tensor_names"] == 240 and summary["on_tensor_names"] == 240
    assert len(summary["off_slots"]) == 48 and len(summary["on_slots"]) == 48
    assert summary["off_occurrence_counts"] == [2] and summary["on_occurrence_counts"] == [2]
    assert summary["bin_files_compared"] == 480
    assert summary["bin_bytes_compared"] > 0
    assert summary["bin_differing_bytes"] == 0
    assert summary["text_differing_bytes"] == 0
    assert summary["stdout_normalized_differing_bytes"] == 0
    assert summary["stdout_dropped_lines_off"] == summary["stdout_dropped_lines_on"] == 1
    assert summary["speed_claim"] is None
    assert "NOT any" in summary["scope"] or "NOT any" in str(summary["info"])
    assert any("core pair coverage=True" in line for line in summary["info"])
    assert "sentinel" not in summary
    compact = json.loads(result.stdout)
    assert compact["verdict"] == "PASS" and compact["speed_claim"] is None
    assert compact["bind_pairs_on"] == 36


def test_rorret_pass_rows_are_informational_only(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="rorret_pass")
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "PASS"
    assert summary["rorret_pass_rows_on_informational"] == 3
    assert any("informational only, never gated" in line for line in summary["info"])


def test_mutated_slot_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    assert_no_go(root, run_cli(root, cli, fixture, out_dir, mode="mutated_slot"),
                 "off_name_set", "on_name_set", "off_occurrences", "on_occurrences")


def test_missing_pass_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    summary = assert_no_go(
        root, run_cli(root, cli, fixture, out_dir, mode="missing_pass"),
        "bind_core_pair_coverage", "bind_pass_set", "bind_counts_equal_positive",
    )
    assert summary["bind_counts_on"] == {"0": 12, "1": 0, "2": 12}
    assert [6, 1] in summary["missing_bind_pairs"]


def test_wrong_key_il_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    summary = assert_no_go(
        root, run_cli(root, cli, fixture, out_dir, mode="wrong_key_il"),
        "bind_core_pair_coverage",
    )
    assert [6, 0] in summary["missing_bind_pairs"]


def test_enable_marker_flip_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    assert_no_go(root, run_cli(root, cli, fixture, out_dir, mode="enable_flip_on"),
                 "on_enable_needles")
    root, cli, fixture, out_dir = build_tree(tmp_path / "second")
    assert_no_go(root, run_cli(root, cli, fixture, out_dir, mode="enable_flip_off"),
                 "off_absent_needles")


def test_byte_flip_in_bin_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    summary = assert_no_go(root, run_cli(root, cli, fixture, out_dir, mode="bin_flip"),
                           "bin_parity")
    assert summary["bin_differing_bytes"] >= 1


def test_stdout_mismatch_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    summary = assert_no_go(
        root, run_cli(root, cli, fixture, out_dir, mode="stdout_flip"),
        "text_parity", "stdout_parity",
    )
    assert summary["text_differing_bytes"] >= 1
    assert summary["stdout_normalized_differing_bytes"] >= 1


def test_nonzero_engine_exit_is_no_go(tmp_path):
    root, cli, fixture, out_dir = build_tree(tmp_path)
    summary = assert_no_go(root, run_cli(root, cli, fixture, out_dir, mode="nonzero"),
                           "off_exit_zero", "on_exit_zero")
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
