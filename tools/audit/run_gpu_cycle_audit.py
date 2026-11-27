#!/usr/bin/env python3
"""Run the M24 CycleAudit gate on real Kaggle 2xT4 llama-cli timings.

This script is plumbing plus a gate, not a speed claim generator. It times the
synthetic 3-pass MoE GGUF with RORET off (baseline) and on (candidate), checks
that the two configurations stay output-identical, drives the real loopkit
audit CLI, and requires both negative controls to fail closed. The verdict is
honest: CLAIM_SUPPORTED only when the gate's claim_valid is true and the
pre-registered claim is at or below ci_low; otherwise NO_CLAIM.

Run from a fresh Kaggle clone, for example:
    python3 tools/audit/run_gpu_cycle_audit.py \
        --llama-cli /kaggle/working/inkengine-llama/build/bin/llama-cli \
        --fixture /kaggle/working/synthetic.gguf \
        --out /kaggle/working/gpu-cycle-audit \
        --loopkit-src /kaggle/working/loopkit/src \
        --repeats 50 --warmup 5 --loops 3 --claim 1.0

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
RUN_TIMEOUT_SECONDS = 300
GATE_TIMEOUT_SECONDS = 120
HOOK_LINE = "RORET union-resident MoE bookkeeping enabled"
PROMPT = "Hello my name is"
SEED = 0
N_TOKENS = 16
THREADS = 1
NEXT_BLOCKER = (
    "Required Kaggle 2xT4 CycleAudit run with repeats >= 20 and loops = 3; "
    "only a pre-registered claim <= ci_low from that gate can support a speed "
    "number. Counts of MoE passes, bindings, hits, and uploads are not "
    "speedups."
)
SCOPE = (
    "Synthetic 3-pass MoE GGUF only, no real weights. RORET_PASS and MoE "
    "bookkeeping counts are counts, not speedups. Only the gate lower bound "
    "ci_low, with claim <= ci_low on a rendered gate with repeats >= 20 and "
    "loops = 3, may support a speed number. T4 (sm_75) has no FP4 or FP8 "
    "tensor cores."
)


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


def cli_version(cli: Path) -> str:
    proc = subprocess.run([str(cli), "--version"], capture_output=True,
                          text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli --version failed: {proc.stderr}")
    return (proc.stdout or proc.stderr).strip()


def build_cli_flags(cli: Path, fixture: Path, text: Path, args: argparse.Namespace
                    ) -> list[str]:
    return [
        str(cli),
        "-m", str(fixture),
        "-p", PROMPT,
        "-s", str(SEED),
        "-n", str(N_TOKENS),
        "-t", str(THREADS),
        "-ngl", str(args.ngl),
        "-cmoe",
        "--moe-cache-mib", str(args.moe_cache_mib),
        "--moe-cache-cpu-share", str(args.cpu_share),
        "--temp", "0",
        "--verbose",
        "-o", str(text),
    ]


def run_once(cmd: list[str], log: Path, ror_on: bool) -> float:
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
                                  stderr=subprocess.STDOUT,
                                  timeout=RUN_TIMEOUT_SECONDS)
            elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"llama-cli timed out; see {log}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli exited {proc.returncode}; see {log}")
    return elapsed


def collect(cmd: list[str], logs_dir: Path, tag: str, ror_on: bool,
            repeats: int, warmup: int) -> tuple[list[float], dict[str, Any]]:
    runs = logs_dir / tag
    runs.mkdir(parents=True, exist_ok=True)
    text = Path(cmd[-1])
    for i in range(warmup):
        run_once(cmd, runs / f"warmup-{i}.log", ror_on)
    seconds: list[float] = []
    hooks: list[bool] = []
    for i in range(repeats):
        log = runs / f"run-{i:03d}.log"
        seconds.append(run_once(cmd, log, ror_on))
        hooks.append(HOOK_LINE in log.read_text(errors="replace"))
    if not all(math.isfinite(t) and t > 0 for t in seconds):
        raise RuntimeError(f"non-finite or non-positive timing in {tag}")
    if not text.is_file() or text.stat().st_size == 0:
        raise RuntimeError(f"generated text missing or empty: {text}")
    evidence = {
        "log_dir": str(runs),
        "final_log": str(runs / f"run-{repeats - 1:03d}.log"),
        "hook_line": HOOK_LINE,
        "hook_present_all_runs": all(hooks),
        "hook_present_count": sum(hooks),
        "hook_absent_all_runs": not any(hooks),
        "per_run_hook_present": hooks,
        "generated_text": str(text),
        "generated_text_sha256": sha256_file(text),
        "generated_text_bytes": text.stat().st_size,
    }
    return seconds, evidence


def check_parity(off_evidence: dict[str, Any], on_evidence: dict[str, Any]
                 ) -> bool:
    if not on_evidence["hook_present_all_runs"]:
        raise RuntimeError("RORET hook line is absent from at least one ON log")
    if not off_evidence["hook_absent_all_runs"]:
        raise RuntimeError("RORET hook line is present in at least one OFF log")
    off_bytes = Path(off_evidence["generated_text"]).read_bytes()
    on_bytes = Path(on_evidence["generated_text"]).read_bytes()
    if off_bytes != on_bytes:
        raise RuntimeError("generated text bytes differ between OFF and ON")
    return True


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


def gate_result(gate: dict[str, Any], loops: int) -> dict[str, Any]:
    if not gate["rendered"] or gate["report"] is None:
        raise RuntimeError(f"gate did not render a JSON report: {gate}")
    report = gate["report"]
    if int(report.get("loops", -1)) != loops:
        raise RuntimeError(f"gate used wrong loops: {report}")
    for key in ("ci_low", "ci_high", "speedup", "claim_valid", "reason"):
        if key not in report:
            raise RuntimeError(f"gate report missing {key}: {report}")
    return report


def negative_control(gate: dict[str, Any], expected_exit: int = 1) -> bool:
    report = gate["report"] or {}
    return (gate["returncode"] == expected_exit
            and gate["rendered"]
            and report.get("claim_valid") is False)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    default_fork = here.parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama-cli", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--loopkit-src", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--claim", type=float, default=1.0,
                    help="pre-registered claim; must be <= ci_low")
    ap.add_argument("--cpu-share", default="0.0")
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--moe-cache-mib", default="16")
    ap.add_argument("--environment-label", default="Kaggle 2xT4 CUDA")
    args = ap.parse_args(argv)

    if args.repeats < MIN_REPEATS:
        raise RuntimeError(f"repeats={args.repeats} is below {MIN_REPEATS}")
    if args.loops != 3:
        raise RuntimeError(f"loops={args.loops} must be 3 for the M24 gate")
    if not math.isfinite(args.claim) or args.claim <= 0:
        raise RuntimeError(f"invalid pre-registered claim: {args.claim}")

    fork_root = default_fork.resolve()
    loopkit_src = args.loopkit_src.resolve()
    out_dir = args.out.resolve()
    fixture = args.fixture.resolve()
    cli = args.llama_cli.resolve()
    if not (loopkit_src / "loopkit" / "cli.py").is_file():
        raise RuntimeError(f"loopkit src missing: {loopkit_src}")
    if not fixture.is_file():
        raise RuntimeError(f"fixture missing: {fixture}")
    if not cli.is_file() or not os.access(cli, os.X_OK):
        raise RuntimeError(f"llama-cli not found or not executable: {cli}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"

    fixture_sha = sha256_file(fixture)
    version = cli_version(cli)
    off_text = logs_dir / "generated-off.txt"
    on_text = logs_dir / "generated-on.txt"
    off_cmd = build_cli_flags(cli, fixture, off_text, args)
    on_cmd = build_cli_flags(cli, fixture, on_text, args)

    baseline_seconds, off_evidence = collect(
        off_cmd, logs_dir, "off", False, args.repeats, args.warmup)
    candidate_seconds, on_evidence = collect(
        on_cmd, logs_dir, "on", True, args.repeats, args.warmup)
    text_bytes_equal = check_parity(off_evidence, on_evidence)

    if len(baseline_seconds) < MIN_REPEATS or len(candidate_seconds) < MIN_REPEATS:
        raise RuntimeError("fewer than 20 recorded runs; producer duty failed")
    baseline_json = out_dir / "baseline.json"
    candidate_json = out_dir / "candidate.json"
    write_timing(baseline_json, "baseline-b1", baseline_seconds)
    write_timing(candidate_json, "candidate-b1", candidate_seconds)
    (out_dir / "baseline_seconds.json").write_text(
        json.dumps(baseline_seconds, separators=(",", ":")) + "\n")
    (out_dir / "candidate_seconds.json").write_text(
        json.dumps(candidate_seconds, separators=(",", ":")) + "\n")

    self_control = run_gate(1.01, baseline_json, baseline_json,
                            args.loops, loopkit_src)
    self_pass = negative_control(self_control)
    if not self_pass:
        raise RuntimeError(f"self-comparison control did not fail closed: {self_control}")

    slow_json = out_dir / "slow.json"
    write_timing(slow_json, "candidate-slow", [2.0 * t for t in candidate_seconds])
    slow_control = run_gate(2.0, baseline_json, slow_json,
                            args.loops, loopkit_src)
    slow_pass = negative_control(slow_control)
    if not slow_pass:
        raise RuntimeError(f"slower-candidate control did not fail closed: {slow_control}")

    gate = run_gate(args.claim, baseline_json, candidate_json,
                    args.loops, loopkit_src)
    report = gate_result(gate, args.loops)
    claim_valid = bool(report["claim_valid"])
    ci_low = float(report["ci_low"])
    verdict = "CLAIM_SUPPORTED" if claim_valid and args.claim <= ci_low else "NO_CLAIM"

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
                "loops": 3, "warmup": 5, "repeats": 50, "seed": SEED,
            },
            "loopkit_cli_repeats_caveat": (
                "loopkit.cli audit sets only loops; repeats stays at the "
                "CycleAudit constructor default, so the >= 20 floor is a "
                "producer duty enforced in this script."
            ),
        },
        "fixture": {
            "path": str(fixture),
            "sha256": fixture_sha,
            "generator": "tools/audit/make_synthetic_gguf.py --seed 0",
        },
        "environment": args.environment_label,
        "cli_flags_off": off_cmd,
        "cli_flags_on": on_cmd,
        "cpu_share": args.cpu_share,
        "ngl": args.ngl,
        "moe_cache_mib": args.moe_cache_mib,
        "loops": args.loops,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "seed": SEED,
        "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "parity": {
            "off": off_evidence,
            "on": on_evidence,
            "hook_present_all_on_logs": on_evidence["hook_present_all_runs"],
            "hook_absent_all_off_logs": off_evidence["hook_absent_all_runs"],
            "generated_text_sha256_off": off_evidence["generated_text_sha256"],
            "generated_text_sha256_on": on_evidence["generated_text_sha256"],
            "generated_text_bytes_equal": text_bytes_equal,
        },
        "negative_controls": {
            "self_comparison_claim_1_01": {
                "expected_claim_valid": False,
                "observed_claim_valid": (self_control["report"] or {}).get("claim_valid"),
                "exit_code": self_control["returncode"],
                "pass": self_pass,
            },
            "slower_candidate_claim_2_0": {
                "input": "simulated: each measured candidate second multiplied by 2.0; not a run",
                "expected_claim_valid": False,
                "observed_claim_valid": (slow_control["report"] or {}).get("claim_valid"),
                "exit_code": slow_control["returncode"],
                "pass": slow_pass,
            },
            "all_pass": self_pass and slow_pass,
        },
        "gate": {
            "command": gate["command"],
            "rendered": gate["rendered"],
            "report_present": gate["report_present"],
            "exit_code": gate["returncode"],
            "ci_low": ci_low,
            "ci_high": float(report["ci_high"]),
            "speedup": float(report["speedup"]),
            "claim_valid": claim_valid,
            "reason": report["reason"],
            "raw_report": report,
            "stderr": gate["stderr"],
        },
        "raw_cli_reports": {
            "self_comparison": self_control,
            "slower_candidate": slow_control,
            "real_comparison": gate,
        },
        "claim": float(args.claim),
        "claim_valid": claim_valid,
        "verdict": verdict,
        "next_blocker": "" if verdict == "CLAIM_SUPPORTED" else NEXT_BLOCKER,
        "scope": SCOPE,
    }
    (out_dir / "GPU_CYCLE_AUDIT.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    summary = {
        "verdict": verdict,
        "claim": float(args.claim),
        "ci_low": ci_low,
        "ci_high": float(report["ci_high"]),
        "speedup": float(report["speedup"]),
        "claim_valid": claim_valid,
        "gate_exit_code": gate["returncode"],
        "repeats": args.repeats,
        "loops": args.loops,
        "fixture_sha256": fixture_sha,
        "fork_head": artifact["fork"]["head"],
        "llama_cli_version": version,
        "negative_controls_pass": self_pass and slow_pass,
    }
    print(f"CycleAudit gate rendered; verdict={verdict} "
          f"claim={args.claim} ci_low={ci_low:.6f} "
          f"claim_valid={claim_valid} exit={gate['returncode']}")
    print(f"repeats={args.repeats} loops={args.loops} "
          f"negative_controls_pass={summary['negative_controls_pass']} "
          f"text_bytes_equal={text_bytes_equal}")
    if verdict == "NO_CLAIM":
        print("NO_CLAIM is an honest result. " + NEXT_BLOCKER)
    print("M24_SUMMARY_JSON=" + json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
