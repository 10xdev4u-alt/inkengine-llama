"""Tests for the upstream CLI stdout byte-compatibility gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

AUDIT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT_DIR))

import check_upstream_cli_compat as checker

REAL_CONTRACT = AUDIT_DIR / "UPSTREAM_CLI_COMPAT.json"
EXPECTED_SUMMARY_KEYS = frozenset(checker.SUMMARY_KEYS)
FAKE_PAYLOAD = "M44-CLI-COMPAT-TOKEN"
CONTRACT_SHA256 = "5fb7e01e63c526844231daa6905ea7e78c9f5f37d85fbc595a0286bbae6b2ae6"


def make_fixture(tmp_path):
    fixture = tmp_path / "synthetic.gguf"
    fixture.write_bytes(b"GGUF" + bytes(range(64)))
    return fixture, checker.sha256_file(fixture), fixture.stat().st_size


def shq(text):
    return "'" + text.replace("'", "'\\''") + "'"


def write_cli(path, stdout_text, stderr_text, exit_code, executable=True, embed_marker=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/bin/sh"]
    if stdout_text:
        lines.append("printf %s " + shq(stdout_text))
    if stderr_text:
        lines.append("printf %s " + shq(stderr_text) + " >&2")
    if embed_marker:
        lines.append(
            'if [ "$LLAMA_RORET" = "1" ]; then printf %s '
            + shq(checker.RORET_ENABLE_MARKER) + " >&2; fi"
        )
    lines.append("exit %d" % exit_code)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    path.chmod(0o755 if executable else 0o644)
    return path


def write_contract(tmp_path, fixture_sha, fixture_size, checksum_fork, checksum_upstream,
                   payload, checker_hash=None, mutate=None):
    facts = json.loads(REAL_CONTRACT.read_text(encoding="ascii"))
    facts["fixture"]["sha256"] = fixture_sha
    facts["fixture"]["size_bytes"] = fixture_size
    facts["fork_cli_sha256"] = checksum_fork
    facts["upstream_cli_sha256"] = checksum_upstream
    facts["stdout_contains"] = payload
    live = checker.sha256_bytes(Path(checker.__file__).read_bytes())
    facts["checker_sha256"] = live if checker_hash is None else checker_hash
    if mutate is not None:
        mutate(facts)
    audit = tmp_path / "repo/tools/audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "UPSTREAM_CLI_COMPAT.json").write_text(
        json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="ascii",
    )
    return tmp_path / "repo", facts


def baseline(tmp_path, *, out=None, fork_stdout=None, upstream_stdout=None,
             fork_stderr="", upstream_stderr="", fork_exit=0, upstream_exit=0,
             checker_hash=None, mutate=None, same_realpath=False,
             fork_executable=True, upstream_executable=True,
             fork_embed_marker=True, upstream_embed_marker=False,
             stdout_payload=FAKE_PAYLOAD):
    if fork_stdout is None:
        fork_stdout = FAKE_PAYLOAD + "\n"
    if upstream_stdout is None:
        upstream_stdout = FAKE_PAYLOAD + "\n"
    fixture, sha, size = make_fixture(tmp_path)
    audit = tmp_path / "repo/tools/audit"
    audit.mkdir(parents=True)
    fork_cli = write_cli(
        audit / "fork-cli", fork_stdout, fork_stderr, fork_exit,
        fork_executable, fork_embed_marker,
    )
    if same_realpath:
        upstream_cli = fork_cli
    else:
        upstream_cli = write_cli(
            audit / "upstream-cli", upstream_stdout, upstream_stderr,
            upstream_exit, upstream_executable, upstream_embed_marker,
        )
    root, facts = write_contract(
        tmp_path, sha, size, checker.sha256_file(fork_cli),
        checker.sha256_file(upstream_cli), stdout_payload, checker_hash, mutate,
    )
    if out is None:
        out = audit / "M44_UPSTREAM_CLI_COMPAT_SUMMARY.json"
    return {
        "root": root, "fork_cli": fork_cli, "upstream_cli": upstream_cli,
        "fixture": fixture, "out": Path(out), "facts": facts,
    }


def run_case(case, timeout=30):
    fixture = Path(case["fixture"])
    with pytest.MonkeyPatch.context() as mp:
        if fixture.is_file():
            mp.setattr(checker, "FIXTURE_SHA256", checker.sha256_file(fixture))
            mp.setattr(checker, "FIXTURE_SIZE_BYTES", fixture.stat().st_size)
        else:
            mp.setattr(checker, "FIXTURE_SHA256", "0" * 64)
            mp.setattr(checker, "FIXTURE_SIZE_BYTES", -1)
        return checker.run_check(
            case["root"], case["fork_cli"], case["upstream_cli"],
            case["fixture"], case["out"], timeout,
        )


def read_out(case):
    return json.loads(case["out"].read_text(encoding="ascii"))


def test_normalize_stdout_removes_only_build_banner():
    left = b"header\nbuild      : b1-abc\nmodel      : x\n> hello\ntoken\n"
    right = b"header\nbuild      : b2-def\nmodel      : x\n> hello\ntoken\n"
    assert checker.normalize_stdout(left) == checker.normalize_stdout(right)
    assert b"build" not in checker.normalize_stdout(left)
    assert b"model      : x" in checker.normalize_stdout(left)


def test_normalize_stdout_keeps_other_lines():
    data = b"build      : b1-abc\n> hello\n"
    assert checker.normalize_stdout(data) == b"> hello\n"


def test_diff_bytes_equal_inputs():
    assert checker.diff_bytes(b"abc", b"abc") == 0
    assert checker.diff_bytes(b"", b"") == 0


def test_diff_bytes_unequal_same_length():
    assert checker.diff_bytes(b"abc", b"abd") == 1
    assert checker.diff_bytes(b"AAA\n", b"BBB\n") == 3
    assert checker.diff_bytes(b"\x00\x01", b"\xff\x01") == 1


def test_diff_bytes_length_delta_included():
    assert checker.diff_bytes(b"abc", b"abcde") == 2
    assert checker.diff_bytes(b"abcde", b"abc") == 2
    assert checker.diff_bytes(b"", b"xyz") == 3


def test_diff_bytes_rejects_non_bytes():
    with pytest.raises(TypeError):
        checker.diff_bytes("abc", b"abc")
    with pytest.raises(TypeError):
        checker.diff_bytes(b"abc", None)


def test_sanitize_env_removes_debug_keys_and_keeps_path():
    env = {"PATH": "/usr/bin", "HOME": "/home/u"}
    env.update({key: "1" for key in checker.ENV_STRIP_KEYS})
    clean = checker.sanitize_env(env)
    assert clean == {"PATH": "/usr/bin", "HOME": "/home/u"}
    assert "PATH" in clean
    assert not set(clean) & set(checker.ENV_STRIP_KEYS)


def test_sanitize_env_removes_timing_key():
    assert "LLAMA_ARG_SHOW_TIMINGS" in checker.ENV_STRIP_KEYS
    clean = checker.sanitize_env({"PATH": "/bin", "LLAMA_ARG_SHOW_TIMINGS": "1"})
    assert clean == {"PATH": "/bin"}


def test_sanitize_env_does_not_mutate_input():
    env = {"PATH": "/bin", "LLAMA_RORET": "1"}
    clean = checker.sanitize_env(env)
    assert env["LLAMA_RORET"] == "1"
    assert clean is not env
    assert "LLAMA_RORET" not in clean


def test_final_verdict_all_checks_pass():
    checks = {"a": True, "b": True}
    assert checker.final_verdict(checks, 0, 0) == "PASS"


def test_final_verdict_partial_checks_fail():
    checks = {"a": True, "b": False}
    assert checker.final_verdict(checks, 0, 0) == "NO_GO"


def test_final_verdict_no_checks_fails_closed():
    assert checker.final_verdict({}, 0, 0) == "NO_GO"


def test_final_verdict_byte_delta_mismatch():
    assert checker.final_verdict({"a": True}, 2, 0) == "NO_GO"
    assert checker.final_verdict({"a": True}, 0, 2) == "NO_GO"


def test_load_contract_roundtrip(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text('{"a": 1}', encoding="ascii")
    assert checker.load_contract(path) == {"a": 1}



def test_frozen_fixture_constants_pinned():
    assert checker.FIXTURE_SHA256 == "cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617"
    assert checker.FIXTURE_SIZE_BYTES == 887392


def test_contract_sha256_pinned():
    assert checker.sha256_bytes(REAL_CONTRACT.read_bytes()) == CONTRACT_SHA256


def test_real_contract_has_exact_top_keys():
    facts = json.loads(REAL_CONTRACT.read_text(encoding="ascii"))
    assert set(facts) == {
        "frozen_on", "contract_version", "fixture", "upstream_base", "fork",
        "cli_args", "checker_sha256", "expect", "speed_claim",
        "fork_cli_sha256", "upstream_cli_sha256", "stdout_contains",
    }
    assert checker.check_contract_keys(facts)
    assert checker.check_contract_keys({}) is False
    bad = dict(facts)
    bad["extra"] = 1
    assert checker.check_contract_keys(bad) is False


def test_real_contract_fixture_and_pin_facts():
    facts = json.loads(REAL_CONTRACT.read_text(encoding="ascii"))
    assert facts["fixture"]["sha256"] == "cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617"
    assert facts["fixture"]["size_bytes"] == 887392
    assert facts["fixture"]["seed"] == 0
    assert facts["fixture"]["arch"] == "inkling2_loop"
    assert facts["upstream_base"]["commit"] == "bc641b516fe01f346b9f2bf70d69fb7afa3ac222"
    assert facts["expect"]["stderr_compared"] is False
    assert facts["speed_claim"] is None
    assert facts["checker_sha256"] == "08c9902d011024317debe5ccf296b08c7fa0e1ebfbdc8854b665b340c03d2289"
    assert facts["fork_cli_sha256"] == "0216eda9af0b0cf45457c9d2e87cc7b6cf09bc4b05735edd2bf8b517885dc323"
    assert facts["upstream_cli_sha256"] == "f1599b90d956aa75657f7b8b4c1e2dde45ca5549c210bf70ba61a28e105a2111"
    assert facts["stdout_contains"] == "fffa3fa3fa3fa397"
    assert "--no-show-timings" in facts["cli_args"]


def test_stdout_equal_is_pass(tmp_path):
    case = baseline(tmp_path)
    summary = run_case(case)
    assert summary["verdict"] == "PASS"
    assert summary["stdout_differing_bytes"] == 0
    assert summary["checks_passed"] == summary["checks_total"]
    assert summary["checks_failed"] == 0
    assert summary["blockers"] == []
    assert read_out(case) == summary


def test_summary_has_exact_field_set(tmp_path):
    case = baseline(tmp_path)
    summary = run_case(case)
    assert set(summary) == EXPECTED_SUMMARY_KEYS
    assert summary["speed_claim"] is None
    assert summary["no_weights"] is True
    assert summary["engine_run"] == "cpu_synthetic_only"
    assert "stdout-only" in summary["note"]
    assert "stderr" in summary["note"]


def test_summary_records_both_cli_and_stdout_hashes(tmp_path):
    case = baseline(tmp_path)
    summary = run_case(case)
    expected = (FAKE_PAYLOAD + "\n").encode("ascii")
    assert summary["fork_cli_sha256"] == checker.sha256_file(case["fork_cli"])
    assert summary["upstream_cli_sha256"] == checker.sha256_file(case["upstream_cli"])
    assert summary["fork_stdout_sha256"] == checker.sha256_bytes(expected)
    assert summary["upstream_stdout_sha256"] == checker.sha256_bytes(expected)
    assert summary["stdout_bytes_each"] == {"fork": len(expected), "upstream": len(expected)}
    assert summary["fork_exit"] == 0 and summary["upstream_exit"] == 0
    assert summary["fork_rort_enable_marker_absent"] is True
    assert summary["upstream_rort_enable_marker_absent"] is True
    assert summary["stdout_contains"] == FAKE_PAYLOAD
    assert summary["fork_stdout_contains_payload"] is True
    assert summary["upstream_stdout_contains_payload"] is True
    assert summary["fork_cli_hash_matches_pin"] is True
    assert summary["upstream_cli_hash_matches_pin"] is True
    assert summary["cli_hashes_distinct"] is True
    assert summary["fork_control_marker_present"] is True
    assert summary["upstream_control_marker_absent"] is True


def test_contract_missing_top_key_is_no_go(tmp_path):
    case = baseline(tmp_path, mutate=lambda facts: facts.pop("expect"))
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "contract_keys" in summary["blockers"]


def test_contract_extra_top_key_is_no_go(tmp_path):
    case = baseline(tmp_path, mutate=lambda facts: facts.update({"extra": 1}))
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "contract_keys" in summary["blockers"]


def test_contract_json_list_is_no_go(tmp_path):
    case = baseline(tmp_path)
    contract = case["root"] / "tools/audit/UPSTREAM_CLI_COMPAT.json"
    contract.write_text("[]", encoding="ascii")
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "contract_json" in summary["blockers"]


def test_contract_cli_args_drift_is_no_go(tmp_path):
    case = baseline(tmp_path, mutate=lambda facts: facts["cli_args"].append("--verbose"))
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "contract_values" in summary["blockers"]


def test_contract_cli_args_not_list_is_no_go(tmp_path):
    case = baseline(tmp_path, mutate=lambda facts: facts.update({"cli_args": "--temp 0"}))
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "contract_values" in summary["blockers"]
    assert "cli_args_usable" in summary["blockers"]


def test_checker_hash_pending_is_no_go(tmp_path):
    case = baseline(tmp_path, checker_hash="PENDING")
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "checker_sha256" in summary["blockers"]


def test_checker_hash_wrong_digest_is_no_go(tmp_path):
    case = baseline(tmp_path, checker_hash="0" * 64)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "checker_sha256" in summary["blockers"]


def test_wrong_fixture_sha_is_no_go(tmp_path):
    case = baseline(tmp_path, mutate=lambda facts: facts["fixture"].update({"sha256": "1" * 64}))
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fixture_sha256" in summary["blockers"]
    assert "contract_values" in summary["blockers"]


def test_wrong_fixture_size_is_no_go(tmp_path):
    case = baseline(tmp_path, mutate=lambda facts: facts["fixture"].update({"size_bytes": 999999}))
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fixture_size" in summary["blockers"]
    assert "contract_values" in summary["blockers"]


def test_fixture_path_is_directory_is_no_go(tmp_path):
    case = baseline(tmp_path)
    case["fixture"] = tmp_path
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fixture_exists" in summary["blockers"]


def test_missing_fork_cli_is_no_go(tmp_path):
    case = baseline(tmp_path)
    case["fork_cli"].unlink()
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fork_cli_exists" in summary["blockers"]


def test_missing_upstream_cli_is_no_go(tmp_path):
    case = baseline(tmp_path)
    case["upstream_cli"].unlink()
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "upstream_cli_exists" in summary["blockers"]


def test_non_executable_fork_cli_is_no_go(tmp_path):
    case = baseline(tmp_path, fork_executable=False)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fork_cli_executable" in summary["blockers"]


def test_non_executable_upstream_cli_is_no_go(tmp_path):
    case = baseline(tmp_path, upstream_executable=False)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "upstream_cli_executable" in summary["blockers"]


def test_same_realpath_for_both_cli_is_no_go(tmp_path):
    case = baseline(tmp_path, same_realpath=True)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "clis_distinct_realpath" in summary["blockers"]


def test_identical_binary_content_is_no_go(tmp_path):
    case = baseline(tmp_path)
    case["upstream_cli"].write_bytes(case["fork_cli"].read_bytes())
    case["upstream_cli"].chmod(0o755)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "clis_distinct_hash" in summary["blockers"]
    assert summary["cli_hashes_distinct"] is False


def test_missing_cli_hash_pin_is_no_go(tmp_path):
    case = baseline(tmp_path, mutate=lambda facts: facts.pop("fork_cli_sha256"))
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "contract_keys" in summary["blockers"]


def test_pending_cli_hash_pin_is_no_go(tmp_path):
    case = baseline(tmp_path, mutate=lambda facts: facts.update({"fork_cli_sha256": "PENDING"}))
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fork_cli_sha256_pinned" in summary["blockers"]


def test_wrong_cli_hash_pin_is_no_go(tmp_path):
    case = baseline(tmp_path, mutate=lambda facts: facts.update({"upstream_cli_sha256": "0" * 64}))
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "upstream_cli_sha256_pinned" in summary["blockers"]


def test_pending_stdout_payload_pin_is_no_go(tmp_path):
    case = baseline(tmp_path, stdout_payload="PENDING")
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fork_stdout_contains_payload" in summary["blockers"]
    assert "upstream_stdout_contains_payload" in summary["blockers"]


def test_same_stdout_stubs_without_payload_is_no_go(tmp_path):
    case = baseline(tmp_path, fork_stdout="hello\n", upstream_stdout="hello\n")
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert summary["stdout_differing_bytes"] == 0
    assert "fork_stdout_contains_payload" in summary["blockers"]
    assert "upstream_stdout_contains_payload" in summary["blockers"]


def test_stdout_payload_missing_from_fork_is_no_go(tmp_path):
    case = baseline(tmp_path, fork_stdout="hello\n")
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fork_stdout_contains_payload" in summary["blockers"]


def test_fork_control_without_rort_marker_is_no_go(tmp_path):
    case = baseline(tmp_path, fork_embed_marker=False)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fork_control_marker_present" in summary["blockers"]


def test_upstream_control_with_rort_marker_is_no_go(tmp_path):
    case = baseline(tmp_path, upstream_embed_marker=True)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "upstream_control_marker_absent" in summary["blockers"]


def test_nonzero_fork_exit_is_no_go(tmp_path):
    case = baseline(tmp_path, fork_exit=1)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fork_exit_zero" in summary["blockers"]
    assert summary["fork_exit"] == 1


def test_nonzero_upstream_exit_is_no_go(tmp_path):
    case = baseline(tmp_path, upstream_exit=2)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "upstream_exit_zero" in summary["blockers"]
    assert summary["upstream_exit"] == 2


def test_run_one_timeout_is_no_go(tmp_path, monkeypatch):
    case = baseline(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(["llama-cli"], 30)

    monkeypatch.setattr(checker.subprocess, "run", timeout)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fork_exit_zero" in summary["blockers"]
    assert summary["fork_exit"] is None


def test_run_one_oserror_is_no_go(tmp_path, monkeypatch):
    case = baseline(tmp_path)

    def failed(*args, **kwargs):
        raise OSError("failed to start")

    monkeypatch.setattr(checker.subprocess, "run", failed)
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fork_exit_zero" in summary["blockers"]
    assert summary["fork_exit"] is None


def test_rort_marker_in_fork_stderr_is_no_go(tmp_path):
    marker = checker.RORET_ENABLE_MARKER
    case = baseline(tmp_path, fork_stderr=marker + "\n")
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "fork_rort_marker_absent" in summary["blockers"]
    assert summary["fork_rort_enable_marker_absent"] is False


def test_rort_marker_in_upstream_stderr_is_no_go(tmp_path):
    marker = checker.RORET_ENABLE_MARKER
    case = baseline(tmp_path, upstream_stderr=marker + "\n")
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "upstream_rort_marker_absent" in summary["blockers"]
    assert summary["upstream_rort_enable_marker_absent"] is False


def test_stderr_text_is_not_compared(tmp_path):
    case = baseline(tmp_path, fork_stderr="timing banner\n", upstream_stderr="other banner\n")
    summary = run_case(case)
    assert summary["verdict"] == "PASS"
    assert summary["fork_rort_enable_marker_absent"] is True
    assert summary["upstream_rort_enable_marker_absent"] is True


def test_stdout_mismatch_is_no_go(tmp_path):
    case = baseline(
        tmp_path,
        fork_stdout=FAKE_PAYLOAD + "AAA\n",
        upstream_stdout=FAKE_PAYLOAD + "BBB\n",
    )
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert "stdout_diff_expected" in summary["blockers"]
    assert summary["stdout_differing_bytes"] == checker.diff_bytes(b"AAA\n", b"BBB\n") == 3


def test_stdout_length_mismatch_is_no_go(tmp_path):
    case = baseline(
        tmp_path,
        fork_stdout=FAKE_PAYLOAD + "hello\n",
        upstream_stdout=FAKE_PAYLOAD + "hello!\n",
    )
    summary = run_case(case)
    assert summary["verdict"] == "NO_GO"
    assert summary["stdout_differing_bytes"] == 2


def test_stale_pass_summary_overwritten_by_no_go(tmp_path):
    case = baseline(tmp_path)
    stale = {"verdict": "PASS", "blockers": []}
    checker.write_json(case["out"], stale)
    assert read_out(case) == stale
    failing = dict(case)
    failing["upstream_cli"] = case["fork_cli"]
    summary = run_case(failing)
    assert summary["verdict"] == "NO_GO"
    assert read_out(case) == summary
    assert read_out(case) != stale


def test_sentinel_written_before_checks(tmp_path, monkeypatch):
    case = baseline(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("forced check failure")

    monkeypatch.setattr(checker, "check_contract_values", boom)
    with pytest.raises(RuntimeError):
        run_case(case)
    written = read_out(case)
    assert written["verdict"] == "NO_GO"
    assert written["blockers"] == ["checker did not complete"]
    assert written["checks_passed"] == 0


def test_initial_sentinel_failure_removes_stale_pass(tmp_path, monkeypatch):
    case = baseline(tmp_path)
    stale_bytes = b'{\n  "verdict": "PASS"\n}\n'
    case["out"].write_bytes(stale_bytes)
    real_write = checker.write_json
    calls = {"n": 0}

    def flaky(path, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("forced sentinel write failure")
        return real_write(path, payload)

    monkeypatch.setattr(checker, "write_json", flaky)
    with pytest.raises(OSError):
        run_case(case)
    assert calls["n"] == 1
    assert not case["out"].exists()


def test_atomic_final_write_leaves_no_temp_files(tmp_path):
    case = baseline(tmp_path)
    summary = run_case(case)
    assert summary["verdict"] == "PASS"
    temps = list(case["out"].parent.glob(".*.tmp"))
    assert temps == []
    assert read_out(case) == summary


def test_final_write_failure_keeps_sentinel(tmp_path, monkeypatch):
    case = baseline(tmp_path)
    real_replace = checker.os.replace
    calls = {"n": 0}

    def counted(src, dst):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise OSError("forced final replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(checker.os, "replace", counted)
    with pytest.raises(OSError):
        run_case(case)
    assert calls["n"] == 2
    written = read_out(case)
    assert written["verdict"] == "NO_GO"
    assert written["blockers"] == ["checker did not complete"]


def test_no_network_module_in_checker_source():
    source = Path(checker.__file__).read_text(encoding="ascii")
    assert "urllib" not in source
    assert "socket" not in source
    assert "http.client" not in source
