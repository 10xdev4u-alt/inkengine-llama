#!/usr/bin/env python3
"""Offline, fail-closed stdout byte-compatibility gate for the fork GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CONTRACT_REL = "tools/audit/UPSTREAM_CLI_COMPAT.json"
DEFAULT_OUT_REL = "tools/audit/M44_UPSTREAM_CLI_COMPAT_SUMMARY.json"
DEFAULT_TIMEOUT = 180

FROZEN_ON = "2026-11-28"
CONTRACT_VERSION = 1
FIXTURE_SHA256 = "cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617"
FIXTURE_SIZE_BYTES = 887392
FIXTURE_GENERATOR = "tools/audit/make_synthetic_gguf.py"
FIXTURE_SEED = 0
FIXTURE_ARCH = "inkling2_loop"
UPSTREAM_BASE_COMMIT = "bc641b516fe01f346b9f2bf70d69fb7afa3ac222"
UPSTREAM_BASE_DATE = "2026-11-24"
UPSTREAM_BUILD_FLAGS = ["GGML_CUDA=OFF", "LLAMA_BUILD_TESTS=OFF", "LLAMA_CURL=OFF"]
FORK_BUILD_FLAGS = ["GGML_CUDA=OFF", "LLAMA_BUILD_TESTS=OFF", "LLAMA_CURL=OFF", "LLAMA_RORET_CPU_TEST=ON"]
CLI_ARGS = [
    "-m", "{fixture}", "-t", "1", "-n", "16", "-s", "0", "--temp", "0",
    "-p", "hello", "--no-perf", "--no-show-timings", "--no-warmup", "-st",
]
CONTROL_EXTRA_ARGS = ["-cmoe", "--moe-cache-mib", "16", "--verbose"]
EXPECT = {"both_exits_zero": True, "stdout_differing_bytes": 0, "stderr_compared": False}
ENV_STRIP_KEYS = (
    "LLAMA_RORET",
    "LLAMA_DUMP_TENSORS_DIR",
    "LLAMA_DUMP_TENSOR_PTR",
    "LLAMA_RORET_TRACE",
    "LLAMA_PIBS",
    "LLAMA_ODSL",
    "LLAMA_ARG_SHOW_TIMINGS",
)
RORET_ENABLE_MARKER = "RORET union-resident MoE bookkeeping enabled"
BUILD_LINE_PREFIX = b"build      : "
NOTE = ("stdout-only byte comparison after removing the build-info banner line; "
        "stderr timing lines are not compared")
ENGINE_RUN = "cpu_synthetic_only"

EXPECTED_TOP_KEYS = frozenset({
    "frozen_on", "contract_version", "fixture", "upstream_base", "fork",
    "cli_args", "checker_sha256", "expect", "speed_claim",
    "fork_cli_sha256", "upstream_cli_sha256", "stdout_contains",
})

CHECK_NAMES = [
    "contract_json",
    "contract_keys",
    "contract_values",
    "checker_sha256",
    "fixture_exists",
    "fixture_sha256",
    "fixture_size",
    "fork_cli_exists",
    "fork_cli_executable",
    "upstream_cli_exists",
    "upstream_cli_executable",
    "clis_distinct_realpath",
    "fork_cli_sha256_pinned",
    "upstream_cli_sha256_pinned",
    "clis_distinct_hash",
    "fork_control_marker_present",
    "upstream_control_marker_absent",
    "cli_args_usable",
    "fork_exit_zero",
    "upstream_exit_zero",
    "fork_rort_marker_absent",
    "upstream_rort_marker_absent",
    "fork_stdout_contains_payload",
    "upstream_stdout_contains_payload",
    "stdout_diff_expected",
    "run_completed",
]

SUMMARY_KEYS = (
    "verdict", "checks_total", "checks_passed", "checks_failed", "blockers",
    "fixture_sha256", "fork_cli_sha256", "upstream_cli_sha256",
    "fork_stdout_sha256", "upstream_stdout_sha256", "stdout_bytes_each",
    "stdout_differing_bytes", "fork_exit", "upstream_exit",
    "fork_rort_enable_marker_absent", "upstream_rort_enable_marker_absent",
    "stdout_contains", "fork_stdout_contains_payload", "upstream_stdout_contains_payload",
    "fork_cli_hash_matches_pin", "upstream_cli_hash_matches_pin", "cli_hashes_distinct",
    "fork_control_marker_present", "upstream_control_marker_absent",
    "speed_claim", "no_weights", "engine_run", "note",
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value
    )


def is_hex40(value):
    return isinstance(value, str) and len(value) == 40 and all(
        ch in "0123456789abcdef" for ch in value
    )


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_contract(path):
    """Load the frozen contract; malformed input fails closed in run_check."""
    return load_json(path)


def write_json(path, payload):
    """Write via a temp file in the same directory, then replace atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def diff_bytes(left, right):
    """Count differing bytes; each extra byte on either side counts once."""
    if not isinstance(left, (bytes, bytearray)) or not isinstance(right, (bytes, bytearray)):
        raise TypeError("diff_bytes needs bytes inputs")
    common = min(len(left), len(right))
    count = sum(1 for i in range(common) if left[i] != right[i])
    return count + abs(len(left) - len(right))


def normalize_stdout(data):
    """Drop the build-info banner line; it records build provenance, not GGUF output."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("normalize_stdout needs bytes input")
    return b"".join(
        line for line in bytes(data).splitlines(keepends=True)
        if not line.startswith(BUILD_LINE_PREFIX)
    )


def sanitize_env(env):
    """Return a new env without the routing and timing debug keys."""
    return {key: value for key, value in env.items() if key not in ENV_STRIP_KEYS}


def final_verdict(checks, differing_bytes, expected_differing_bytes):
    """PASS only when every check holds and the byte delta meets the frozen value."""
    if not checks or not all(checks.values()):
        return "NO_GO"
    if differing_bytes != expected_differing_bytes:
        return "NO_GO"
    return "PASS"


def check_contract_keys(facts):
    return isinstance(facts, dict) and set(facts) == EXPECTED_TOP_KEYS


def check_contract_values(facts):
    """Frozen facts are fixed; fixture identity is pinned twice and checked on disk."""
    if not isinstance(facts, dict):
        return False
    fixture = facts.get("fixture")
    upstream = facts.get("upstream_base")
    fork = facts.get("fork")
    expect = facts.get("expect")
    if not all(isinstance(value, dict) for value in (fixture, upstream, fork, expect)):
        return False
    if facts.get("frozen_on") != FROZEN_ON or facts.get("contract_version") != CONTRACT_VERSION:
        return False
    if facts.get("speed_claim") is not None:
        return False
    if set(fixture) != {"sha256", "size_bytes", "generator", "seed", "arch"}:
        return False
    if fixture.get("sha256") != FIXTURE_SHA256 or not is_sha256(fixture.get("sha256")):
        return False
    size = fixture.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        return False
    if size != FIXTURE_SIZE_BYTES:
        return False
    if fixture.get("generator") != FIXTURE_GENERATOR:
        return False
    if fixture.get("seed") != FIXTURE_SEED or fixture.get("arch") != FIXTURE_ARCH:
        return False
    if set(upstream) != {"commit", "date", "build_flags"}:
        return False
    if upstream.get("commit") != UPSTREAM_BASE_COMMIT or upstream.get("date") != UPSTREAM_BASE_DATE:
        return False
    if upstream.get("build_flags") != UPSTREAM_BUILD_FLAGS:
        return False
    if set(fork) != {"head_at_freeze", "build_flags", "rort_default_off_env_unset", "rort_enable_marker_absent"}:
        return False
    if not is_hex40(fork.get("head_at_freeze")):
        return False
    if fork.get("build_flags") != FORK_BUILD_FLAGS:
        return False
    if fork.get("rort_default_off_env_unset") is not True:
        return False
    if fork.get("rort_enable_marker_absent") is not True:
        return False
    if facts.get("cli_args") != CLI_ARGS:
        return False
    for key in ("fork_cli_sha256", "upstream_cli_sha256"):
        value = facts.get(key)
        if not isinstance(value, str):
            return False
        if value != "PENDING" and not is_sha256(value):
            return False
    if not isinstance(facts.get("stdout_contains"), str):
        return False
    if set(expect) != {"both_exits_zero", "stdout_differing_bytes", "stderr_compared"}:
        return False
    if expect.get("both_exits_zero") is not True:
        return False
    if expect.get("stdout_differing_bytes") != EXPECT["stdout_differing_bytes"]:
        return False
    if expect.get("stderr_compared") is not False:
        return False
    return True


def check_checker_sha256(facts, live_sha256):
    """Pinned hash must match; the PENDING placeholder is a hard failure."""
    return bool(
        isinstance(facts, dict)
        and is_sha256(live_sha256)
        and facts.get("checker_sha256") == live_sha256
    )


def run_one(cli, fixture, cli_args, env, timeout):
    """Run one CLI; capture stdout and stderr as bytes, never time the run."""
    cmd = [str(cli)] + [str(fixture) if arg == "{fixture}" else arg for arg in cli_args]
    result = {"exit": None, "stdout": b"", "stderr": b""}
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, timeout=timeout,
        )
        result["exit"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
    except subprocess.TimeoutExpired as exc:
        result["stdout"] = exc.stdout or b""
        result["stderr"] = (exc.stderr or b"") + b"\ntimeout"
    except OSError as exc:
        result["stderr"] = ("failed to start: %s: %s" % (type(exc).__name__, exc)).encode("utf-8")
    return result


def make_summary(
    verdict, checks_total, checks_passed, checks_failed, blockers,
    fixture_sha256, fork_cli_sha256, upstream_cli_sha256,
    fork_stdout_sha256, upstream_stdout_sha256, stdout_bytes_each,
    stdout_differing_bytes, fork_exit, upstream_exit,
    fork_rort_enable_marker_absent, upstream_rort_enable_marker_absent,
    stdout_contains, fork_stdout_contains_payload, upstream_stdout_contains_payload,
    fork_cli_hash_matches_pin, upstream_cli_hash_matches_pin, cli_hashes_distinct,
    fork_control_marker_present, upstream_control_marker_absent,
):
    return {
        "verdict": verdict,
        "checks_total": checks_total,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "blockers": list(blockers),
        "fixture_sha256": fixture_sha256,
        "fork_cli_sha256": fork_cli_sha256,
        "upstream_cli_sha256": upstream_cli_sha256,
        "fork_stdout_sha256": fork_stdout_sha256,
        "upstream_stdout_sha256": upstream_stdout_sha256,
        "stdout_bytes_each": dict(stdout_bytes_each),
        "stdout_differing_bytes": stdout_differing_bytes,
        "fork_exit": fork_exit,
        "upstream_exit": upstream_exit,
        "fork_rort_enable_marker_absent": fork_rort_enable_marker_absent,
        "upstream_rort_enable_marker_absent": upstream_rort_enable_marker_absent,
        "stdout_contains": stdout_contains,
        "fork_stdout_contains_payload": fork_stdout_contains_payload,
        "upstream_stdout_contains_payload": upstream_stdout_contains_payload,
        "fork_cli_hash_matches_pin": fork_cli_hash_matches_pin,
        "upstream_cli_hash_matches_pin": upstream_cli_hash_matches_pin,
        "cli_hashes_distinct": cli_hashes_distinct,
        "fork_control_marker_present": fork_control_marker_present,
        "upstream_control_marker_absent": upstream_control_marker_absent,
        "speed_claim": None,
        "no_weights": True,
        "engine_run": ENGINE_RUN,
        "note": NOTE,
    }


def sentinel_summary():
    """Startup NO_GO marker: never leave a stale PASS if the run stops early."""
    return make_summary(
        verdict="NO_GO",
        checks_total=len(CHECK_NAMES),
        checks_passed=0,
        checks_failed=len(CHECK_NAMES),
        blockers=["checker did not complete"],
        fixture_sha256=None,
        fork_cli_sha256=None,
        upstream_cli_sha256=None,
        fork_stdout_sha256=None,
        upstream_stdout_sha256=None,
        stdout_bytes_each={"fork": None, "upstream": None},
        stdout_differing_bytes=None,
        fork_exit=None,
        upstream_exit=None,
        fork_rort_enable_marker_absent=False,
        upstream_rort_enable_marker_absent=False,
        stdout_contains=None,
        fork_stdout_contains_payload=False,
        upstream_stdout_contains_payload=False,
        fork_cli_hash_matches_pin=False,
        upstream_cli_hash_matches_pin=False,
        cli_hashes_distinct=False,
        fork_control_marker_present=False,
        upstream_control_marker_absent=False,
    )


def run_check(root, fork_cli, upstream_cli, fixture, out, timeout=DEFAULT_TIMEOUT):
    root = Path(root)
    out = Path(out)
    try:
        write_json(out, sentinel_summary())
    except BaseException:
        try:
            out.unlink()
        except OSError:
            pass
        raise

    facts = {}
    checks = {name: False for name in CHECK_NAMES}
    fixture_sha256 = None
    fork_cli_sha256 = None
    upstream_cli_sha256 = None
    fork_stdout_sha256 = None
    upstream_stdout_sha256 = None
    stdout_bytes_each = {"fork": None, "upstream": None}
    differing_bytes = None
    fork_exit = None
    upstream_exit = None
    fork_marker_absent = False
    upstream_marker_absent = False
    stdout_contains = None
    fork_stdout_contains_payload = False
    upstream_stdout_contains_payload = False
    fork_cli_hash_matches_pin = False
    upstream_cli_hash_matches_pin = False
    cli_hashes_distinct = False
    fork_control_marker_present = False
    upstream_control_marker_absent = False

    checker_sha256 = sha256_bytes(Path(__file__).read_bytes())
    try:
        facts = load_contract(root / CONTRACT_REL)
    except (OSError, ValueError):
        facts = {}
    checks["contract_json"] = isinstance(facts, dict)
    checks["contract_keys"] = check_contract_keys(facts)
    checks["contract_values"] = check_contract_values(facts)
    checks["checker_sha256"] = check_checker_sha256(facts, checker_sha256)

    fixture_obj = facts.get("fixture") if isinstance(facts, dict) else None
    expected_sha = fixture_obj.get("sha256") if isinstance(fixture_obj, dict) else None
    expected_size = fixture_obj.get("size_bytes") if isinstance(fixture_obj, dict) else None
    fixture_path = Path(fixture)
    checks["fixture_exists"] = fixture_path.is_file()
    if checks["fixture_exists"]:
        fixture_sha256 = sha256_file(fixture_path)
        checks["fixture_sha256"] = fixture_sha256 == expected_sha == FIXTURE_SHA256
        checks["fixture_size"] = fixture_path.stat().st_size == expected_size == FIXTURE_SIZE_BYTES

    fork_path = Path(fork_cli)
    upstream_path = Path(upstream_cli)
    checks["fork_cli_exists"] = fork_path.is_file()
    checks["upstream_cli_exists"] = upstream_path.is_file()
    checks["fork_cli_executable"] = checks["fork_cli_exists"] and os.access(fork_path, os.X_OK)
    checks["upstream_cli_executable"] = checks["upstream_cli_exists"] and os.access(upstream_path, os.X_OK)
    checks["clis_distinct_realpath"] = fork_path.resolve() != upstream_path.resolve()

    pinned_fork_hash = facts.get("fork_cli_sha256") if isinstance(facts, dict) else None
    pinned_upstream_hash = facts.get("upstream_cli_sha256") if isinstance(facts, dict) else None
    stdout_contains = facts.get("stdout_contains") if isinstance(facts, dict) else None
    cli_args = facts.get("cli_args") if isinstance(facts, dict) else None
    checks["cli_args_usable"] = isinstance(cli_args, list)

    marker = RORET_ENABLE_MARKER.encode("ascii")
    if checks["fork_cli_exists"]:
        fork_cli_sha256 = sha256_file(fork_path)
    if checks["upstream_cli_exists"]:
        upstream_cli_sha256 = sha256_file(upstream_path)
    checks["fork_cli_sha256_pinned"] = is_sha256(pinned_fork_hash) and pinned_fork_hash == fork_cli_sha256
    checks["upstream_cli_sha256_pinned"] = is_sha256(pinned_upstream_hash) and pinned_upstream_hash == upstream_cli_sha256
    checks["clis_distinct_hash"] = (
        is_sha256(fork_cli_sha256) and is_sha256(upstream_cli_sha256)
        and fork_cli_sha256 != upstream_cli_sha256
    )

    can_run = (
        checks["fixture_exists"] and checks["fork_cli_executable"]
        and checks["upstream_cli_executable"] and checks["clis_distinct_realpath"]
        and checks["cli_args_usable"]
    )
    if can_run:
        env = sanitize_env(os.environ.copy())
        fork_run = run_one(fork_path, fixture_path, cli_args, env, timeout)
        upstream_run = run_one(upstream_path, fixture_path, cli_args, env, timeout)
        fork_exit = fork_run["exit"]
        upstream_exit = upstream_run["exit"]
        fork_stdout_sha256 = sha256_bytes(fork_run["stdout"])
        upstream_stdout_sha256 = sha256_bytes(upstream_run["stdout"])
        stdout_bytes_each = {
            "fork": len(fork_run["stdout"]),
            "upstream": len(upstream_run["stdout"]),
        }
        differing_bytes = diff_bytes(
            normalize_stdout(fork_run["stdout"]),
            normalize_stdout(upstream_run["stdout"]),
        )
        payload = b""
        if isinstance(stdout_contains, str) and stdout_contains and stdout_contains != "PENDING":
            payload = stdout_contains.encode("utf-8")
        fork_stdout_contains_payload = bool(payload) and payload in fork_run["stdout"]
        upstream_stdout_contains_payload = bool(payload) and payload in upstream_run["stdout"]
        fork_marker_absent = marker not in fork_run["stderr"]
        upstream_marker_absent = marker not in upstream_run["stderr"]

        control_env = dict(env)
        control_env["LLAMA_RORET"] = "1"
        control_args = cli_args + CONTROL_EXTRA_ARGS
        fork_control = run_one(fork_path, fixture_path, control_args, control_env, timeout)
        upstream_control = run_one(upstream_path, fixture_path, control_args, control_env, timeout)
        fork_control_marker_present = marker in fork_control["stderr"]
        upstream_control_marker_absent = marker not in upstream_control["stderr"]

    checks["fork_exit_zero"] = fork_exit == 0
    checks["upstream_exit_zero"] = upstream_exit == 0
    checks["fork_rort_marker_absent"] = fork_marker_absent
    checks["upstream_rort_marker_absent"] = upstream_marker_absent
    checks["fork_control_marker_present"] = fork_control_marker_present
    checks["upstream_control_marker_absent"] = upstream_control_marker_absent
    checks["fork_stdout_contains_payload"] = fork_stdout_contains_payload
    checks["upstream_stdout_contains_payload"] = upstream_stdout_contains_payload
    checks["stdout_diff_expected"] = differing_bytes == EXPECT["stdout_differing_bytes"]
    checks["run_completed"] = can_run and fork_exit is not None and upstream_exit is not None

    verdict = final_verdict(checks, differing_bytes, EXPECT["stdout_differing_bytes"])
    blockers = sorted(name for name, ok in checks.items() if not ok)
    passed = sum(1 for ok in checks.values() if ok)
    summary = make_summary(
        verdict=verdict,
        checks_total=len(checks),
        checks_passed=passed,
        checks_failed=len(checks) - passed,
        blockers=blockers,
        fixture_sha256=fixture_sha256,
        fork_cli_sha256=fork_cli_sha256,
        upstream_cli_sha256=upstream_cli_sha256,
        fork_stdout_sha256=fork_stdout_sha256,
        upstream_stdout_sha256=upstream_stdout_sha256,
        stdout_bytes_each=stdout_bytes_each,
        stdout_differing_bytes=differing_bytes,
        fork_exit=fork_exit,
        upstream_exit=upstream_exit,
        fork_rort_enable_marker_absent=fork_marker_absent,
        upstream_rort_enable_marker_absent=upstream_marker_absent,
        stdout_contains=stdout_contains,
        fork_stdout_contains_payload=fork_stdout_contains_payload,
        upstream_stdout_contains_payload=upstream_stdout_contains_payload,
        fork_cli_hash_matches_pin=checks["fork_cli_sha256_pinned"],
        upstream_cli_hash_matches_pin=checks["upstream_cli_sha256_pinned"],
        cli_hashes_distinct=checks["clis_distinct_hash"],
        fork_control_marker_present=fork_control_marker_present,
        upstream_control_marker_absent=upstream_control_marker_absent,
    )
    write_json(out, summary)
    return summary


def compact_summary(summary):
    return {key: summary[key] for key in (
        "verdict", "checks_total", "checks_passed", "checks_failed", "blockers",
        "fixture_sha256", "stdout_differing_bytes", "fork_exit", "upstream_exit",
        "fork_cli_hash_matches_pin", "upstream_cli_hash_matches_pin",
        "fork_stdout_contains_payload", "upstream_stdout_contains_payload",
        "fork_control_marker_present", "upstream_control_marker_absent",
        "speed_claim", "note",
    )}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="fork root to check")
    parser.add_argument("--fork-cli", type=Path, required=True, help="fork-built llama-cli")
    parser.add_argument("--upstream-cli", type=Path, required=True, help="upstream-base-built llama-cli")
    parser.add_argument("--fixture", type=Path, required=True, help="frozen synthetic inkling2_loop GGUF")
    parser.add_argument("--out", type=Path, default=None, help="summary JSON output path")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="per-run timeout in seconds")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out if args.out is not None else (args.root / DEFAULT_OUT_REL)
    try:
        summary = run_check(
            args.root, args.fork_cli, args.upstream_cli, args.fixture, out, args.timeout,
        )
    except Exception as exc:
        print("error: checker stopped: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print(json.dumps(compact_summary(summary), sort_keys=True, separators=(",", ":")))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
