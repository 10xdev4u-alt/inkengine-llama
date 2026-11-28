#!/usr/bin/env python3
"""Offline, fail-closed CPU-only runtime binding gate for the 3-pass loop fixture."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CONTRACT_REL = "tools/audit/THREEPASS_BINDING.json"
SUMMARY_REL = "tools/audit/M40_3PASS_BINDING_SUMMARY.json"
DEFAULT_CLI = Path("/tmp/opencode/ink-build/bin/llama-cli")
DEFAULT_GGUF = Path("/tmp/opencode/synthetic.gguf")
DEFAULT_OUT_DIR = Path("/tmp/opencode/m40-3pass-logs")

SCHEMA_VERSION = 1
FROZEN_ON = "2026-11-28"
NOTE = "CPU-only 3-pass runtime binding, no timing"
CYCLE_AUDIT_REQUIRED = (
    "NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains "
    "mandatory for any future number."
)
ENGINE_RUN = "cpu_synthetic_only"
EXPECTED_ARCH = "inkling2_loop"
EXPECTED_KEY_IL = 1
EXPECTED_PASSES = [0, 1, 2]
EXPECTED_ENABLE_NEEDLES = [
    "RORET union-resident MoE bookkeeping enabled",
]
EXPECTED_BUILD_MARKER = "LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache"
EXPECTED_TRACE_NEEDLES = ["RORET_BIND set"]
EXPECTED_INFORMATIONAL_NEEDLES = ["RORET_PASS"]
CPU_TRAFFIC_SCOPE = (
    "RORET_PASS rows require the device slot-map copy and are NOT expected on "
    "the CPU fallback; see M21/M22. Informational only, never gated."
)
EXPECTED_OFF_ABSENT_NEEDLES = [
    "RORET_BIND",
    "RORET_PASS",
    "RORET union-resident MoE bookkeeping enabled",
]
TIMEOUT_SECONDS = 300
CLI_FLAGS = [
    "-p", "hello", "-n", "4", "-t", "1", "-s", "0", "--temp", "0",
    "-cmoe", "--moe-cache-mib", "16", "--verbose",
]
DEFAULT_CPU_SHARE = "1.0"
NORMALIZE_NEEDLES = tuple(
    dict.fromkeys(EXPECTED_OFF_ABSENT_NEEDLES + [EXPECTED_BUILD_MARKER])
)
PERF_LINE_RX = re.compile(rb"^\[ Prompt: .* t/s \| Generation: .* t/s \]\r?\n?$")
NETWORK_MODULES = {
    "socket", "http", "urllib", "requests", "httpx", "ftplib", "smtplib",
    "telnetlib", "xmlrpc",
}
NETWORK_CALLS = {"urlopen", "urlretrieve", "socket", "connect"}

EXPECTED_TOP_KEYS = frozenset({
    "schema_version", "frozen_on", "scope", "note", "speed_claim",
    "no_weights", "engine_run", "expected_fixture_sha256", "expected_arch",
    "expected_key_il", "expected_passes", "expected_enable_needles",
    "informational_build_markers", "expected_trace_needles",
    "informational_needles", "cpu_traffic_scope",
    "expected_off_absent_needles", "expected_summary_path",
    "checker_sha256", "verify",
})

CHECK_NAMES = [
    "contract_json",
    "contract_keys",
    "contract_values",
    "checker_sha256",
    "cli_exists",
    "cli_executable",
    "fixture_exists",
    "fixture_sha256",
    "fixture_arch",
    "off_exit_zero",
    "on_exit_zero",
    "off_hook_absent",
    "on_enable_needles",
    "on_trace_needles",
    "bind_set_passes",
    "no_mulmat_absent",
    "trace_parse_completed",
    "text_parity",
    "stdout_parity",
    "no_network",
    "run_completed",
]

SET_RX = re.compile(r"RORET_BIND\s+set\b.*?\bkey_il=(-?\d+)\s+pass=(-?\d+)")
NOMUL_RX = re.compile(r"RORET_BIND\s+no_mulmat\b.*?\bkey_il=(-?\d+)\s+pass=(-?\d+)")
PASS_RX = re.compile(r"RORET_PASS\s+il=(-?\d+)\s+pass=(-?\d+)")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value
    )


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    """Write via a temp file in the same directory, then replace atomically."""
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


def fail_closed_summary():
    """Startup sentinel: never leave a stale PASS if the run stops early."""
    return {
        "verdict": "NO_GO",
        "blocker": ["checker did not complete"],
        "checks": {name: False for name in CHECK_NAMES},
        "check_counts": {
            "total": len(CHECK_NAMES), "passed": 0, "failed": len(CHECK_NAMES),
        },
        "info": ["NO_GO sentinel written at checker startup"],
        "note": NOTE,
        "speed_claim": None,
        "no_weights": True,
        "engine_run": ENGINE_RUN,
        "frozen_on": FROZEN_ON,
        "checker_sha256": None,
        "contract_path": CONTRACT_REL,
        "expected_summary_path": SUMMARY_REL,
        "sentinel": True,
    }


def check_contract_keys(facts):
    return isinstance(facts, dict) and set(facts) == EXPECTED_TOP_KEYS


def check_contract_values(facts):
    """The whole contract is frozen; any drift, including an extra key, fails closed."""
    if not isinstance(facts, dict):
        return False
    try:
        passes = facts["expected_passes"]
        enable = facts["expected_enable_needles"]
        build_marker = facts["informational_build_markers"]
        trace = facts["expected_trace_needles"]
        informational = facts["informational_needles"]
        absent = facts["expected_off_absent_needles"]
        verify = facts["verify"]
    except KeyError:
        return False
    return bool(
        facts.get("schema_version") == SCHEMA_VERSION
        and facts.get("frozen_on") == FROZEN_ON
        and facts.get("scope") is None
        and facts.get("note") is None
        and facts.get("speed_claim") is None
        and facts.get("no_weights") is True
        and facts.get("engine_run") == ENGINE_RUN
        and is_sha256(facts.get("expected_fixture_sha256"))
        and facts.get("expected_arch") == EXPECTED_ARCH
        and facts.get("expected_key_il") == EXPECTED_KEY_IL
        and isinstance(passes, list)
        and passes == EXPECTED_PASSES
        and all(isinstance(value, int) and not isinstance(value, bool) for value in passes)
        and enable == EXPECTED_ENABLE_NEEDLES
        and build_marker == EXPECTED_BUILD_MARKER
        and build_marker not in enable
        and trace == EXPECTED_TRACE_NEEDLES
        and informational == EXPECTED_INFORMATIONAL_NEEDLES
        and not set(informational) & set(trace)
        and facts.get("cpu_traffic_scope") == CPU_TRAFFIC_SCOPE
        and absent == EXPECTED_OFF_ABSENT_NEEDLES
        and isinstance(facts.get("expected_summary_path"), str)
        and facts["expected_summary_path"] == SUMMARY_REL
        and is_sha256(facts.get("checker_sha256"))
        and isinstance(verify, list)
        and verify == CHECK_NAMES
    )


def check_checker_sha256(facts, live_sha256):
    """The pin lives in the contract, so the checker source carries no self-hash."""
    return bool(
        isinstance(facts, dict)
        and is_sha256(live_sha256)
        and is_sha256(facts.get("checker_sha256"))
        and facts["checker_sha256"] == live_sha256
    )


def check_no_network():
    """Static self-check: no network imports or URL-fetch calls in this checker."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in NETWORK_MODULES for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in NETWORK_MODULES:
                return False
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in NETWORK_CALLS:
                return False
    return True


def run_phase(phase, cli, gguf, out_dir, summary_path, on, cpu_share):
    """Run one engine phase; capture stdout and stderr separately, never time it."""
    env = os.environ.copy()
    env.pop("LLAMA_RORET", None)
    env.pop("LLAMA_RORET_TRACE", None)
    if on:
        env["LLAMA_RORET"] = "1"
        env["LLAMA_RORET_TRACE"] = "1"
    env["M40_SUMMARY_PATH"] = str(summary_path)
    generated = out_dir / (phase + "_generated.txt")
    cmd = [
        str(cli), "-m", str(gguf), *CLI_FLAGS,
        "--moe-cache-cpu-share", str(cpu_share), "-o", str(generated),
    ]
    stdout_path = out_dir / (phase + "_stdout.log")
    stderr_path = out_dir / (phase + "_stderr.log")
    result = {
        "exit": None,
        "stdout": b"",
        "stderr": b"",
        "generated": b"",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "generated_path": str(generated),
    }
    try:
        with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
            proc = subprocess.run(cmd, stdout=so, stderr=se, env=env, timeout=TIMEOUT_SECONDS)
        result["exit"] = proc.returncode
    except subprocess.TimeoutExpired:
        result["exit"] = None
    except OSError as exc:
        result["exit"] = None
        result["stderr"] = ("failed to start: %s: %s" % (type(exc).__name__, exc)).encode("utf-8")
    for key, path in (("stdout", stdout_path), ("stderr", stderr_path), ("generated", generated)):
        try:
            result[key] = path.read_bytes()
        except OSError:
            result[key] = b""
    return result


def parse_bindings(text):
    """Extract (il, pass) pairs from the ON-phase RORET_BIND/RORET_PASS trace."""
    return {
        "set": [(int(il), int(pass_)) for il, pass_ in SET_RX.findall(text)],
        "no_mulmat": [(int(il), int(pass_)) for il, pass_ in NOMUL_RX.findall(text)],
        "pass": [(int(il), int(pass_)) for il, pass_ in PASS_RX.findall(text)],
    }


def normalize_stdout(data):
    """Drop anchored trace/enable lines plus perf lines; return (kept, dropped count)."""
    kept = []
    dropped = 0
    for line in data.splitlines(keepends=True):
        stripped = line.decode("utf-8", errors="replace").strip()
        if any(stripped.startswith(needle) for needle in NORMALIZE_NEEDLES):
            dropped += 1
            continue
        if PERF_LINE_RX.match(line):
            dropped += 1
            continue
        kept.append(line)
    return b"".join(kept), dropped


def distinct_sorted(rows, key_il):
    return sorted({pass_ for il, pass_ in rows if il == key_il})


def run_check(root, cli, gguf, out_dir, cpu_share):
    root = Path(root)
    summary_path = root / SUMMARY_REL
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = ["cpu_share=%s" % cpu_share]
    sentinel_ok = True

    try:
        summary_path.unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:
        info.append("sentinel_unlink_failed: %s: %s" % (type(exc).__name__, exc))
    try:
        write_json(summary_path, fail_closed_summary())
    except Exception as exc:
        sentinel_ok = False
        info.append("sentinel_write_failed: %s: %s" % (type(exc).__name__, exc))

    checks = {name: False for name in CHECK_NAMES}
    facts = {}
    checker_sha256 = None
    fixture_sha256 = None
    off = {"exit": None, "stdout": b"", "stderr": b"", "generated": b""}
    on = {"exit": None, "stdout": b"", "stderr": b"", "generated": b""}
    set_passes = []
    pass_rows = []
    pass_row_count = 0
    no_mulmat_rows = []
    no_mulmat_total = 0
    off_dropped = 0
    on_dropped = 0
    text_parity_bytes = 0
    stdout_parity_bytes = 0
    try:
        checker_sha256 = sha256_bytes(Path(__file__).read_bytes())
        try:
            facts = load_json(root / CONTRACT_REL)
            checks["contract_json"] = isinstance(facts, dict)
        except (OSError, ValueError) as exc:
            info.append("contract_load_failed: %s: %s" % (type(exc).__name__, exc))
        checks["contract_keys"] = check_contract_keys(facts)
        checks["contract_values"] = check_contract_values(facts)
        checks["checker_sha256"] = check_checker_sha256(facts, checker_sha256)
        checks["no_network"] = check_no_network()
        info.append("checker sha256 frozen=%s live=%s" % (
            facts.get("checker_sha256"), checker_sha256,
        ))

        cli_path = Path(cli)
        checks["cli_exists"] = cli_path.is_file()
        checks["cli_executable"] = checks["cli_exists"] and os.access(cli_path, os.X_OK)
        fixture = Path(gguf)
        checks["fixture_exists"] = fixture.is_file()
        if checks["fixture_exists"]:
            fixture_sha256 = sha256_file(fixture)
            expected_sha = facts.get("expected_fixture_sha256") if isinstance(facts, dict) else None
            checks["fixture_sha256"] = fixture_sha256 == expected_sha
            expected_arch = facts.get("expected_arch") if isinstance(facts, dict) else EXPECTED_ARCH
            arch_bytes = expected_arch.encode("ascii") if isinstance(expected_arch, str) else b""
            with fixture.open("rb") as handle:
                header = handle.read(1 << 20)
            checks["fixture_arch"] = bool(arch_bytes) and arch_bytes in header
        info.append("fixture sha256=%s size=%d" % (
            fixture_sha256, fixture.stat().st_size if checks["fixture_exists"] else -1,
        ))

        can_run = checks["cli_executable"] and checks["fixture_exists"]
        if can_run:
            off = run_phase("off", cli_path, fixture, out_dir, summary_path, False, cpu_share)
            on = run_phase("on", cli_path, fixture, out_dir, summary_path, True, cpu_share)
            info.append("off_exit=%s on_exit=%s" % (off["exit"], on["exit"]))
        checks["off_exit_zero"] = off["exit"] == 0
        checks["on_exit_zero"] = on["exit"] == 0

        off_text = (off["stdout"] + b"\n" + off["stderr"]).decode("utf-8", errors="replace")
        on_text = (on["stdout"] + b"\n" + on["stderr"]).decode("utf-8", errors="replace")
        absent = facts.get("expected_off_absent_needles") if isinstance(facts, dict) else EXPECTED_OFF_ABSENT_NEEDLES
        enable = facts.get("expected_enable_needles") if isinstance(facts, dict) else EXPECTED_ENABLE_NEEDLES
        trace = facts.get("expected_trace_needles") if isinstance(facts, dict) else EXPECTED_TRACE_NEEDLES
        key_il = facts.get("expected_key_il") if isinstance(facts, dict) else EXPECTED_KEY_IL
        expected_passes = facts.get("expected_passes") if isinstance(facts, dict) else EXPECTED_PASSES
        expected_set = set(expected_passes) if isinstance(expected_passes, list) else set()

        checks["off_hook_absent"] = can_run and bool(absent) and not any(
            needle in off_text for needle in absent
        )
        checks["on_enable_needles"] = bool(enable) and all(
            needle in on_text for needle in enable
        )
        checks["on_trace_needles"] = bool(trace) and all(
            needle in on_text for needle in trace
        )

        binding = parse_bindings(on_text)
        checks["trace_parse_completed"] = True
        set_passes = distinct_sorted(binding["set"], key_il)
        pass_rows = distinct_sorted(binding["pass"], key_il)
        pass_row_count = sum(1 for il, _ in binding["pass"] if il == key_il)
        no_mulmat_total = len(binding["no_mulmat"])
        no_mulmat_rows = [
            (il, pass_) for il, pass_ in binding["no_mulmat"]
            if il == key_il and pass_ in expected_set
        ]
        checks["bind_set_passes"] = bool(expected_set) and set(set_passes) == expected_set
        checks["no_mulmat_absent"] = can_run and not no_mulmat_rows
        info.append("bind set key_il=%s passes=%s count=%d" % (
            key_il, set_passes, sum(1 for il, _ in binding["set"] if il == key_il),
        ))
        info.append("no_mulmat total rows=%d scoped rows=%d (absence is the only observable CPU property)" % (
            no_mulmat_total, len(no_mulmat_rows),
        ))
        marker = facts.get("informational_build_markers") if isinstance(facts, dict) else EXPECTED_BUILD_MARKER
        off_markers = sum(1 for line in off_text.splitlines() if marker in line)
        on_markers = sum(1 for line in on_text.splitlines() if marker in line)
        info.append("unconditional build marker off=%d on=%d (not a gating enable needle)" % (
            off_markers, on_markers,
        ))
        scope = facts.get("cpu_traffic_scope") if isinstance(facts, dict) else CPU_TRAFFIC_SCOPE
        info.append("CPU traffic scope: %s; informational RORET_PASS rows passes=%s count=%d (not gated)" % (
            scope, pass_rows, pass_row_count,
        ))

        off_text_bytes = off["generated"]
        on_text_bytes = on["generated"]
        checks["text_parity"] = bool(off_text_bytes) and off_text_bytes == on_text_bytes
        if checks["text_parity"]:
            text_parity_bytes = len(off_text_bytes)
        off_normal, off_dropped = normalize_stdout(off["stdout"])
        on_normal, on_dropped = normalize_stdout(on["stdout"])
        checks["stdout_parity"] = bool(off_normal) and off_normal == on_normal and off_dropped == on_dropped
        if checks["stdout_parity"]:
            stdout_parity_bytes = len(off_normal)
        info.append("text parity bytes=%d stdout normalized bytes=%d dropped lines off=%d on=%d" % (
            text_parity_bytes, stdout_parity_bytes, off_dropped, on_dropped,
        ))
        checks["run_completed"] = True
    except Exception as exc:
        info.append("exception: %s: %s" % (type(exc).__name__, exc))

    failed_names = [name for name in sorted(checks) if not checks[name]]
    if not sentinel_ok:
        failed_names.append("sentinel_write_failed")
    verdict = "PASS" if not failed_names else "NO_GO"
    passed = sum(1 for value in checks.values() if value)
    summary = {
        "verdict": verdict,
        "blocker": [] if verdict == "PASS" else failed_names,
        "checks": checks,
        "check_counts": {
            "total": len(checks), "passed": passed, "failed": len(checks) - passed,
        },
        "info": info,
        "note": NOTE,
        "speed_claim": None,
        "no_weights": True,
        "engine_run": ENGINE_RUN,
        "frozen_on": FROZEN_ON,
        "checker_sha256": checker_sha256,
        "contract_path": CONTRACT_REL,
        "expected_summary_path": SUMMARY_REL,
        "fixture_sha256": fixture_sha256,
        "off_exit": off["exit"],
        "on_exit": on["exit"],
        "bind_set_passes": set_passes,
        "pass_row_passes": pass_rows,
        "pass_row_count": pass_row_count,
        "no_mulmat_total": no_mulmat_total,
        "no_mulmat_rows": [[il, pass_] for il, pass_ in no_mulmat_rows],
        "text_parity_bytes": text_parity_bytes,
        "stdout_parity_bytes": stdout_parity_bytes,
        "cycle_audit_required": CYCLE_AUDIT_REQUIRED,
    }
    write_json(summary_path, summary)
    return summary


def compact_summary(summary):
    return {
        "verdict": summary["verdict"],
        "blocker": summary["blocker"],
        "counts": summary["check_counts"],
        "fixture_sha256": summary["fixture_sha256"],
        "bind_set_passes": summary["bind_set_passes"],
        "pass_row_passes": summary["pass_row_passes"],
        "pass_row_count": summary["pass_row_count"],
        "no_mulmat_total": summary["no_mulmat_total"],
        "text_parity_bytes": summary["text_parity_bytes"],
        "stdout_parity_bytes": summary["stdout_parity_bytes"],
        "speed_claim": None,
        "note": NOTE,
        "summary_path": SUMMARY_REL,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to check")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI, help="built CPU-only llama-cli")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF, help="synthetic 3-pass GGUF")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="log directory")
    parser.add_argument("--cpu-share", default=DEFAULT_CPU_SHARE, help="moe-cache CPU share")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        summary = run_check(args.root, args.cli, args.gguf, args.out_dir, args.cpu_share)
    except Exception as exc:
        print("error: could not write summary: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print(json.dumps(compact_summary(summary), sort_keys=True, separators=(",", ":")))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
