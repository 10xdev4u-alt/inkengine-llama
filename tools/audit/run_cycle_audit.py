#!/usr/bin/env python3
"""Produce the M9 CycleAudit artifact from local CPU parity-harness timings.

This script is plumbing only. It records real wall-clock runs of the synthetic
GGUF with LLAMA_RORET unset and set, then drives the real loopkit audit CLI.
Local CPU timings can never support a publishable speed claim. The artifact
always records claim 0 and verdict NO_CLAIM. See AUDIT_PROTOCOL.md section 2.

Usage:
    python3 tools/audit/run_cycle_audit.py \
        --loopkit-src /path/to/loopkit/src \
        --fork-root /path/to/inkengine-llama \
        --build-dir /tmp/opencode/m9-build \
        --out /tmp/opencode/m9-audit

Standard library only. The script never builds, commits, or uploads anything.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_BOOTSTRAP_RESAMPLES = 2000
MIN_REPEATS = 20
GATE_TIMEOUT_SECONDS = 120
HOOK_LINE = "RORET union-resident MoE bookkeeping enabled"
PROMPT = "Hello my name is"
ENVIRONMENT = "local CPU, GGML_CUDA=OFF, LLAMA_RORET_CPU_TEST=ON"
REASON = (
    "Local CPU timings are plumbing only and ineligible for a publishable "
    "claim per tools/audit/AUDIT_PROTOCOL.md section 2, so no T4 speed "
    "evidence exists."
)
NEXT_BLOCKER = (
    "Required Kaggle 2xT4 CycleAudit run with repeats >= 20 and loops = 3; "
    "run at most one kernel and prefer logs over the output API. Only a "
    "claim <= ci_low from that run can support a speed number."
)
CLI_FLAGS = [
    "-p", PROMPT, "-s", "0", "-n", "16", "-t", "1", "-ngl", "0",
    "--moe-cache-mib", "1", "--moe-cache-cpu-share", "1.0",
    "--temp", "0", "--verbose",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def ensure_fixture(fork_root: Path, fixture: Path) -> str:
    fixture.parent.mkdir(parents=True, exist_ok=True)
    if not fixture.is_file():
        script = fork_root / "tools" / "audit" / "make_synthetic_gguf.py"
        if not script.is_file():
            raise RuntimeError(f"fixture generator missing: {script}")
        env = dict(os.environ)
        gguf_py = fork_root / "gguf-py"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(gguf_py), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        cmd = [sys.executable, str(script), "--seed", "0", "--out", str(fixture)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"fixture generation failed: {proc.stderr}")
    return sha256_file(fixture)


def locate_cli(build_dir: Path | None, cli_opt: Path | None) -> Path:
    if cli_opt is not None:
        cli = cli_opt.resolve()
    elif build_dir is not None:
        cli = (build_dir.resolve() / "bin" / "llama-cli")
    else:
        raise RuntimeError("pass --llama-cli or --build-dir")
    if not cli.is_file() or not os.access(cli, os.X_OK):
        raise RuntimeError(f"llama-cli not found or not executable: {cli}")
    return cli


def cli_version(cli: Path) -> str:
    proc = subprocess.run([str(cli), "--version"], capture_output=True,
                          text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli --version failed: {proc.stderr}")
    return (proc.stdout or proc.stderr).strip()


def run_once(cli: Path, fixture: Path, log: Path, text: Path,
             ror_on: bool) -> float:
    cmd = [str(cli), "-m", str(fixture), *CLI_FLAGS, "-o", str(text)]
    env = dict(os.environ)
    if ror_on:
        env["LLAMA_RORET"] = "1"
    else:
        env.pop("LLAMA_RORET", None)
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("wb") as sink:
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, env=env, stdout=sink,
                                  stderr=subprocess.STDOUT, timeout=120)
            elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"llama-cli timed out; see {log}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli exited {proc.returncode}; see {log}")
    return elapsed


def collect(cli: Path, fixture: Path, logs_dir: Path, tag: str, ror_on: bool,
            repeats: int, warmup: int) -> tuple[list[float], dict[str, Any]]:
    runs = logs_dir / tag
    runs.mkdir(parents=True, exist_ok=True)
    text = runs / "generated.txt"
    for i in range(warmup):
        run_once(cli, fixture, runs / f"warmup-{i}.log", text, ror_on)
    seconds: list[float] = []
    for i in range(repeats):
        seconds.append(run_once(cli, fixture, runs / f"run-{i:03d}.log",
                                text, ror_on))
    if not all(math.isfinite(t) and t > 0 for t in seconds):
        raise RuntimeError(f"non-finite timing in {tag}")
    evidence = {
        "log_dir": str(runs),
        "final_log": str(runs / f"run-{repeats - 1:03d}.log"),
        "generated_text": str(text),
        "generated_text_sha256": sha256_file(text),
        "hook_line_present": all(
            HOOK_LINE in (runs / f"run-{i:03d}.log").read_text(
                errors="replace") for i in range(repeats)),
    }
    return seconds, evidence


def write_timing(path: Path, name: str, seconds: list[float]) -> None:
    path.write_text(json.dumps({
        "name": name,
        "seconds": seconds,
        "cycles": len(seconds),
        "setup_seconds": 0,
    }, indent=2) + "\n")


def run_gate(claim: float, baseline: Path, candidate: Path, loops: int,
             loopkit_src: Path) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "loopkit.cli", "audit",
           "--claim", repr(float(claim)),
           "--baseline", str(baseline),
           "--candidate", str(candidate),
           "--loops", str(loops)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(loopkit_src), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=GATE_TIMEOUT_SECONDS)
        code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        code, stdout = -1, ""
        stderr = f"gate timed out after {GATE_TIMEOUT_SECONDS}s: {exc}"
    except OSError as exc:
        code, stdout = -1, ""
        stderr = f"gate could not start: {exc}"
    report: dict[str, Any] | None = None
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            report = parsed
    except (ValueError, TypeError):
        pass
    rendered = bool(report and "Traceback" not in stderr
                    and "timed out" not in stderr)
    return {
        "command": cmd,
        "claim": float(claim),
        "returncode": code,
        "report_present": report is not None,
        "rendered": rendered,
        "report": report,
        "stderr": stderr.strip(),
    }


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    default_fork = here.parents[2]
    default_workspace = here.parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loopkit-src", type=Path,
                    default=default_workspace / "loopkit" / "src")
    ap.add_argument("--fork-root", type=Path, default=default_fork)
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/opencode/m9-audit"))
    ap.add_argument("--fixture", type=Path, default=None)
    ap.add_argument("--llama-cli", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args(argv)

    fork_root = args.fork_root.resolve()
    loopkit_src = args.loopkit_src.resolve()
    out_dir = args.out.resolve()
    fixture = (args.fixture or out_dir / "synthetic.gguf").resolve()
    if args.repeats < MIN_REPEATS:
        raise RuntimeError(f"repeats={args.repeats} is below {MIN_REPEATS}")
    if not (loopkit_src / "loopkit" / "cli.py").is_file():
        raise RuntimeError(f"loopkit src missing: {loopkit_src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"

    fixture_sha = ensure_fixture(fork_root, fixture)
    cli = locate_cli(args.build_dir, args.llama_cli)
    version = cli_version(cli)

    baseline_seconds, off_evidence = collect(
        cli, fixture, logs_dir, "off", False, args.repeats, args.warmup)
    candidate_seconds, on_evidence = collect(
        cli, fixture, logs_dir, "on", True, args.repeats, args.warmup)

    if not on_evidence["hook_line_present"]:
        raise RuntimeError("ON logs do not contain the RORET hook line")
    if off_evidence["hook_line_present"]:
        raise RuntimeError("OFF logs contain the RORET hook line")
    text_off = Path(off_evidence["generated_text"]).read_bytes()
    text_on = Path(on_evidence["generated_text"]).read_bytes()
    if text_off != text_on:
        raise RuntimeError("generated text differs between OFF and ON")

    baseline_json = out_dir / "baseline.json"
    candidate_json = out_dir / "candidate.json"
    write_timing(baseline_json, "baseline-b1", baseline_seconds)
    write_timing(candidate_json, "candidate-b1", candidate_seconds)

    self_control = run_gate(1.01, baseline_json, baseline_json,
                            args.loops, loopkit_src)
    self_pass = (self_control["returncode"] == 1
                 and self_control["rendered"]
                 and self_control["report"] is not None
                 and self_control["report"].get("claim_valid") is False)
    if not self_pass:
        raise RuntimeError(f"self-comparison negative control failed: {self_control}")

    probe = run_gate(1.0, baseline_json, candidate_json,
                     args.loops, loopkit_src)
    if not probe["rendered"] or probe["report"] is None:
        raise RuntimeError(f"real comparison probe did not render: {probe}")
    probe_report = probe["report"]
    if int(probe_report.get("loops", -1)) != args.loops:
        raise RuntimeError(f"probe used wrong loops: {probe_report}")

    (out_dir / "baseline_seconds.json").write_text(
        json.dumps(baseline_seconds, separators=(",", ":")) + "\n")
    (out_dir / "candidate_seconds.json").write_text(
        json.dumps(candidate_seconds, separators=(",", ":")) + "\n")

    artifact = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fork": {
            "root": str(fork_root),
            "branch": git_output(fork_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": git_output(fork_root, ["rev-parse", "HEAD"]),
        },
        "tools": {
            "python": sys.version.split()[0],
            "loopkit_src": str(loopkit_src),
            "loopkit_head": git_output(loopkit_src.parent,
                                       ["rev-parse", "HEAD"]),
            "llama_cli": str(cli),
            "llama_cli_version": version,
            "loopkit_audit_defaults": {
                "loops": 3, "warmup": 5, "repeats": 50, "seed": 0,
            },
        },
        "fixture": {
            "path": str(fixture),
            "sha256": fixture_sha,
            "generator": "tools/audit/make_synthetic_gguf.py --seed 0",
        },
        "environment": ENVIRONMENT,
        "cli_flags": CLI_FLAGS,
        "loops": args.loops,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "seed": 0,
        "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "parity": {
            "off": off_evidence,
            "on": on_evidence,
            "generated_text_bytes_equal": True,
        },
        "negative_controls": {
            "self_comparison_claim_1_01": {
                "expected_claim_valid": False,
                "observed_claim_valid": self_control["report"].get("claim_valid"),
                "exit_code": self_control["returncode"],
                "pass": self_pass,
            },
        },
        "raw_cli_reports": {
            "self_comparison": self_control,
            "real_probe_claim_1_0": probe,
        },
        "cli_exit_codes": {
            "self_comparison": self_control["returncode"],
            "real_probe": probe["returncode"],
        },
        "ineligible_probe_point_estimates": {
            "ci_low": probe_report.get("ci_low"),
            "ci_high": probe_report.get("ci_high"),
            "speedup": probe_report.get("speedup"),
            "note": "Local CPU point estimates only. Not a claim and not evidence.",
        },
        "claim": 0,
        "claim_valid": False,
        "verdict": "NO_CLAIM",
        "reason": REASON,
        "gate_eligible": False,
        "next_blocker": NEXT_BLOCKER,
    }
    (out_dir / "CYCLE_AUDIT.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": artifact["verdict"],
        "ineligible_probe_ci_low": probe_report.get("ci_low"),
        "fixture_sha256": fixture_sha,
        "repeats": args.repeats,
        "loops": args.loops,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
