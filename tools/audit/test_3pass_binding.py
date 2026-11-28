"""Behavior tests for the fail-closed 3-pass CPU runtime binding gate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CHECKER = AUDIT_DIR / "check_3pass_binding.py"
CONTRACT = AUDIT_DIR / "THREEPASS_BINDING.json"
SUMMARY_REL = "tools/audit/M40_3PASS_BINDING_SUMMARY.json"
sys.path.insert(0, str(AUDIT_DIR))

import check_3pass_binding as checker

EXPECTED_TOTAL_CHECKS = 21
REAL_FIXTURE_SHA256 = "cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617"
EXPECTED_CHECKER_SHA256 = "35357c44fdf72b95c240ad1eeefb940c97a28a8a10c38c2955bafd8a2f0eafcf"
EXPECTED_CONTRACT_SHA256 = "d7957f4809285c19f8e68d58ec2a12b5d7062127fef06fe693bcab10adeddd7e"
EXPECTED_CPU_TRAFFIC_SCOPE = (
    "RORET_PASS rows require the device slot-map copy and are NOT expected on "
    "the CPU fallback; see M21/M22. Informational only, never gated."
)

FAKE_CLI = """#!/bin/sh
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then
    shift
    out="$1"
  fi
  shift
done
mode="${M40_FAKE_MODE-}"
on=0
if [ "${LLAMA_RORET-}" = "1" ]; then
  on=1
fi
if [ "$mode" = "sentinel" ]; then
  python3 -c 'import json, os; d = json.load(open(os.environ["M40_SUMMARY_PATH"])); assert d.get("verdict") == "NO_GO" and d.get("sentinel") is True' || exit 3
fi
text="deterministic generated text"
if [ "$mode" = "parity_text" ] && [ "$on" = "1" ]; then
  text="DIFFERENT generated text"
fi
if [ -n "$out" ]; then
  printf '%s\\n' "$text" > "$out"
fi
printf 'PROMPT hello\\n'
if [ "$mode" = "parity_stdout" ] && [ "$on" = "1" ]; then
  printf 'EXTRA ON ONLY LINE\\n'
fi
if [ "$mode" = "stdout_trace_off" ] && [ "$on" = "0" ]; then
  printf '%s\\n' 'RORET_BIND set lookup=0x1 slot_map=0x2 key_il=1 pass=0'
fi
if [ "$mode" = "stdout_embedded_trace_on" ] && [ "$on" = "1" ]; then
  printf '%s\\n' 'XX RORET_BIND set lookup=0x1 slot_map=0x2 key_il=1 pass=0'
fi
printf '%s\\n' 'llama_moe_cache: LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache' >&2
if [ "$on" = "1" ]; then
  if [ "$mode" != "missing_enable" ]; then
    printf '%s\\n' 'llama_moe_cache: RORET union-resident MoE bookkeeping enabled' >&2
  fi
  if [ "$mode" = "missing_pass1" ]; then
    printf '%s\\n' 'RORET_BIND set lookup=0x1 slot_map=0x2 key_il=1 pass=0' >&2
    printf '%s\\n' 'RORET_BIND set lookup=0x3 slot_map=0x2 key_il=1 pass=2' >&2
  else
    printf '%s\\n' 'RORET_BIND set lookup=0x1 slot_map=0x2 key_il=1 pass=0' >&2
    printf '%s\\n' 'RORET_BIND set lookup=0x3 slot_map=0x2 key_il=1 pass=1' >&2
    printf '%s\\n' 'RORET_BIND set lookup=0x4 slot_map=0x2 key_il=1 pass=2' >&2
  fi
  if [ "$mode" = "with_pass_rows" ]; then
    printf '%s\\n' 'llama_moe_cache: RORET_PASS il=1 pass=0 plans=1 hits=1' >&2
    printf '%s\\n' 'llama_moe_cache: RORET_PASS il=1 pass=1 plans=1 hits=1' >&2
    printf '%s\\n' 'llama_moe_cache: RORET_PASS il=1 pass=2 plans=1 hits=1' >&2
  fi
  if [ "$mode" = "no_mulmat" ]; then
    printf '%s\\n' 'RORET_BIND no_mulmat layer_il=1 key_il=1 pass=1' >&2
  fi
elif [ "$mode" = "off_hook" ]; then
  printf '%s\\n' 'RORET_BIND set lookup=0x1 slot_map=0x2 key_il=1 pass=0' >&2
fi
if [ "$mode" = "nonzero" ]; then
  exit 7
fi
exit 0
"""


def fake_fixture_bytes():
    return b"GGUF\x03" + b"\x00" * 64 + b"inkling2_loop" + b"\x00" * 64


def build_tree(tmp_path, mode=None):
    root = tmp_path / "repo"
    (root / "tools/audit").mkdir(parents=True)
    out_dir = tmp_path / "logs"
    out_dir.mkdir()
    fixture = tmp_path / "fake.gguf"
    fixture.write_bytes(fake_fixture_bytes())
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["expected_fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    (root / checker.CONTRACT_REL).write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    cli = tmp_path / "llama-cli"
    cli.write_text(FAKE_CLI, encoding="utf-8")
    cli.chmod(0o755)
    return root, cli, fixture, out_dir, mode


def run_cli(root, cli, fixture, out_dir, mode=None, checker_path=None):
    env = os.environ.copy()
    if mode:
        env["M40_FAKE_MODE"] = mode
    else:
        env.pop("M40_FAKE_MODE", None)
    return subprocess.run(
        [sys.executable, str(checker_path or CHECKER),
         "--root", str(root), "--cli", str(cli), "--gguf", str(fixture),
         "--out-dir", str(out_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )


def read_summary(root):
    return json.loads((root / SUMMARY_REL).read_text(encoding="utf-8"))


def write_contract(root, payload):
    (root / checker.CONTRACT_REL).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def tamper_contract(root, mutate):
    payload = json.loads((root / checker.CONTRACT_REL).read_text(encoding="utf-8"))
    mutate(payload)
    write_contract(root, payload)


def assert_no_go(root, result, failed_check):
    assert result.returncode == 1, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "NO_GO"
    assert failed_check in summary["blocker"]
    assert summary["checks"][failed_check] is False
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
    assert facts["no_weights"] is True
    assert facts["engine_run"] == "cpu_synthetic_only"
    assert facts["expected_fixture_sha256"] == REAL_FIXTURE_SHA256
    assert facts["expected_arch"] == "inkling2_loop"
    assert facts["expected_key_il"] == 1
    assert facts["expected_passes"] == [0, 1, 2]
    assert facts["expected_enable_needles"] == [
        "RORET union-resident MoE bookkeeping enabled"
    ]
    assert facts["informational_build_markers"] == checker.EXPECTED_BUILD_MARKER
    assert facts["expected_trace_needles"] == ["RORET_BIND set"]
    assert facts["informational_needles"] == ["RORET_PASS"]
    assert facts["cpu_traffic_scope"] == EXPECTED_CPU_TRAFFIC_SCOPE
    assert "RORET_PASS" not in facts["expected_trace_needles"]
    assert "trace_parse_completed" in facts["verify"]
    assert "pass_rows_recorded" not in facts["verify"]
    assert checker.is_sha256(facts["checker_sha256"])
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
        "cli_exists", "cli_executable", "fixture_exists", "fixture_sha256",
        "fixture_arch", "off_exit_zero", "on_exit_zero", "off_hook_absent",
        "on_enable_needles", "on_trace_needles", "bind_set_passes",
        "no_mulmat_absent", "trace_parse_completed", "text_parity",
        "stdout_parity", "no_network", "run_completed",
    }
    assert checker.check_no_network() is True


def test_parse_bindings_and_normalize_stdout():
    text = (
        "RORET_BIND set lookup=0x1 slot_map=0x2 key_il=1 pass=0\n"
        "RORET_BIND set lookup=0x3 slot_map=0x2 key_il=1 pass=2\n"
        "RORET_BIND no_mulmat layer_il=1 key_il=1 pass=1\n"
        "llama_moe_cache: RORET_PASS il=1 pass=2 plans=19 hits=150\n"
    )
    parsed = checker.parse_bindings(text)
    assert checker.distinct_sorted(parsed["set"], 1) == [0, 2]
    assert checker.distinct_sorted(parsed["pass"], 1) == [2]
    assert parsed["no_mulmat"] == [(1, 1)]
    raw = b"PROMPT hello\nRORET_BIND set key_il=1 pass=0\nRORET_PASS il=1 pass=0\n"
    assert checker.normalize_stdout(raw) == (b"PROMPT hello\n", 2)
    paced = b"> hello\nfffa\n[ Prompt: 1349.5 t/s | Generation: 862.6 t/s ]\n"
    assert checker.normalize_stdout(paced) == (b"> hello\nfffa\n", 1)
    embedded = b"PROMPT hello\nXX RORET_BIND set key_il=1 pass=0\n"
    assert checker.normalize_stdout(embedded) == (embedded, 0)


def test_happy_path_passes_with_fake_traces(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == []
    assert summary["check_counts"] == {
        "total": EXPECTED_TOTAL_CHECKS, "passed": EXPECTED_TOTAL_CHECKS, "failed": 0,
    }
    assert all(value is True for value in summary["checks"].values())
    assert summary["bind_set_passes"] == [0, 1, 2]
    assert summary["pass_row_passes"] == []
    assert summary["pass_row_count"] == 0
    assert summary["checks"]["trace_parse_completed"] is True
    assert any(EXPECTED_CPU_TRAFFIC_SCOPE in line for line in summary["info"])
    assert any("unconditional build marker off=1 on=1" in line for line in summary["info"])
    assert any("dropped lines off=0 on=0" in line for line in summary["info"])
    assert summary["no_mulmat_total"] == 0
    assert summary["no_mulmat_rows"] == []
    assert summary["text_parity_bytes"] == len(b"deterministic generated text\n")
    assert summary["stdout_parity_bytes"] > 0
    assert summary["fixture_sha256"] == hashlib.sha256(fake_fixture_bytes()).hexdigest()
    assert summary["off_exit"] == 0 and summary["on_exit"] == 0
    assert "sentinel" not in summary
    compact = json.loads(result.stdout)
    assert compact["verdict"] == "PASS"
    assert compact["bind_set_passes"] == [0, 1, 2]
    assert compact["pass_row_passes"] == [] and compact["pass_row_count"] == 0
    assert set(compact) == {
        "verdict", "blocker", "counts", "fixture_sha256", "bind_set_passes",
        "pass_row_passes", "pass_row_count", "no_mulmat_total",
        "text_parity_bytes", "stdout_parity_bytes", "speed_claim", "note",
        "summary_path",
    }


def test_zero_ror_pass_rows_do_not_gate(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == []
    assert summary["pass_row_passes"] == []
    assert summary["pass_row_count"] == 0
    assert summary["checks"]["trace_parse_completed"] is True
    assert any("informational RORET_PASS rows passes=[] count=0" in line
               for line in summary["info"])


def test_informational_ror_pass_rows_do_not_gate(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="with_pass_rows")
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "PASS"
    assert summary["blocker"] == []
    assert summary["bind_set_passes"] == [0, 1, 2]
    assert summary["pass_row_passes"] == [0, 1, 2]
    assert summary["pass_row_count"] == 3
    assert all(value is True for value in summary["checks"].values())


def test_off_hook_present_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="off_hook")
    assert_no_go(root, result, "off_hook_absent")


def test_on_enable_needles_missing_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="missing_enable")
    assert_no_go(root, result, "on_enable_needles")


def test_missing_pass_1_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="missing_pass1")
    summary = assert_no_go(root, result, "bind_set_passes")
    assert summary["blocker"] == ["bind_set_passes"]
    assert summary["bind_set_passes"] == [0, 2]
    assert summary["pass_row_passes"] == [] and summary["pass_row_count"] == 0
    assert summary["checks"]["trace_parse_completed"] is True


def test_no_mulmat_for_key_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="no_mulmat")
    summary = assert_no_go(root, result, "no_mulmat_absent")
    assert summary["no_mulmat_rows"] == [[1, 1]]


def test_generated_text_parity_mismatch_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="parity_text")
    summary = assert_no_go(root, result, "text_parity")
    assert summary["text_parity_bytes"] == 0


def test_stdout_parity_mismatch_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="parity_stdout")
    summary = assert_no_go(root, result, "stdout_parity")
    assert summary["checks"]["text_parity"] is True


def test_dropped_line_count_mismatch_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="stdout_trace_off")
    summary = assert_no_go(root, result, "stdout_parity")
    assert any("dropped lines off=1 on=0" in line for line in summary["info"])


def test_embedded_trace_text_is_not_normalized_away(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="stdout_embedded_trace_on")
    summary = assert_no_go(root, result, "stdout_parity")
    assert any("dropped lines off=0 on=0" in line for line in summary["info"])


def test_wrong_fixture_sha_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    tamper_contract(root, lambda facts: facts.update(
        expected_fixture_sha256="0" * 64
    ))
    result = run_cli(root, cli, fixture, out_dir)
    assert_no_go(root, result, "fixture_sha256")


def test_extra_contract_key_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    tamper_contract(root, lambda facts: facts.update(surprise=True))
    result = run_cli(root, cli, fixture, out_dir)
    assert_no_go(root, result, "contract_keys")


def test_missing_contract_key_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    def drop(facts):
        del facts["expected_arch"]
    tamper_contract(root, drop)
    result = run_cli(root, cli, fixture, out_dir)
    assert_no_go(root, result, "contract_keys")


def test_checker_sha_mismatch_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    tamper_contract(root, lambda facts: facts.update(checker_sha256="1" * 64))
    result = run_cli(root, cli, fixture, out_dir)
    assert_no_go(root, result, "checker_sha256")


def test_malformed_contract_json_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    (root / checker.CONTRACT_REL).write_text("{not json", encoding="utf-8")
    result = run_cli(root, cli, fixture, out_dir)
    assert_no_go(root, result, "contract_json")


def test_missing_cli_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, tmp_path / "no-such-cli", fixture, out_dir)
    summary = assert_no_go(root, result, "cli_exists")
    assert "cli_executable" in summary["blocker"]
    assert "off_exit_zero" in summary["blocker"]


def test_nonzero_cli_exit_is_no_go(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="nonzero")
    summary = assert_no_go(root, result, "off_exit_zero")
    assert "on_exit_zero" in summary["blocker"]
    assert summary["off_exit"] == 7 and summary["on_exit"] == 7


def test_sentinel_is_written_before_engine_runs(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    result = run_cli(root, cli, fixture, out_dir, mode="sentinel")
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_summary(root)
    assert summary["verdict"] == "PASS"
    assert "sentinel" not in summary


def test_stale_pass_is_overwritten(tmp_path):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
    checker.write_json(root / SUMMARY_REL, {"verdict": "PASS", "stale": True})
    tamper_contract(root, lambda facts: facts.update(checker_sha256="2" * 64))
    result = run_cli(root, cli, fixture, out_dir)
    summary = assert_no_go(root, result, "checker_sha256")
    assert "stale" not in summary
    assert summary["check_counts"]["passed"] < EXPECTED_TOTAL_CHECKS


def test_atomic_write_failure_returns_one(tmp_path, monkeypatch):
    root, cli, fixture, out_dir, _ = build_tree(tmp_path)
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
