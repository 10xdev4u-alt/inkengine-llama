"""Executable proof of the CycleAudit gate contract in AUDIT_PROTOCOL.md.

This is plumbing validation, not speed evidence. It generates deterministic
synthetic timing samples, drives the real `python -m loopkit.cli audit` command,
and checks that both negative controls fail closed. All sample timings are
fabricated on purpose: no wall-clock claim is made from them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

MIN_REPEATS = 20
DEFAULT_REPEATS = 50
DEFAULT_WARMUP = 5
DEFAULT_LOOPS = 3
DEFAULT_SEED = 0
SAMPLE_SEED = 7
GATE_TIMEOUT_SECONDS = 60


def default_loopkit_src() -> Path:
    env = os.environ.get("LOOPKIT_SRC")
    if env:
        return Path(env).resolve()
    # tools/audit/verify_gate.py -> parents[3] is the workspace root.
    return Path(__file__).resolve().parents[3] / "loopkit" / "src"


def make_timing(name: str, seconds: Any, *, setup_seconds: float = 0.0) -> dict:
    """Build one timing JSON payload, enforcing the repeats floor."""
    values = [float(t) for t in seconds]
    if len(values) < MIN_REPEATS:
        raise ValueError(f"repeats={len(values)} is below the {MIN_REPEATS} floor")
    if not all(math.isfinite(t) and t > 0 for t in values):
        raise ValueError("all seconds must be finite and positive")
    return {
        "name": name,
        "seconds": values,
        "cycles": len(values),
        "setup_seconds": float(setup_seconds),
    }


def synthetic_samples(repeats: int = DEFAULT_REPEATS, seed: int = SAMPLE_SEED) -> dict:
    """Fixed-seed samples. Candidate is faster, slow is slower, both synthetic."""
    if repeats < MIN_REPEATS:
        raise ValueError(f"repeats={repeats} is below the {MIN_REPEATS} floor")
    rng = np.random.default_rng(seed)
    base = np.clip(rng.normal(0.012, 0.0006, repeats), 1e-4, None)
    candidate = np.clip(base * 0.88 + rng.normal(0, 0.00015, repeats), 1e-4, None)
    slow = np.clip(base * 1.25 + rng.normal(0, 0.00015, repeats), 1e-4, None)
    return {
        "baseline": make_timing("baseline-b1", base),
        "candidate": make_timing("candidate-b1", candidate),
        "slow": make_timing("slow-b1", slow),
    }


def write_samples(out_dir: Path, samples: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, payload in samples.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths[name] = path
    return paths


def run_gate(claim: float, baseline: Path, candidate: Path, *,
             loops: int = DEFAULT_LOOPS, loopkit_src: Path | None = None,
             python: str | None = None,
             timeout: float = GATE_TIMEOUT_SECONDS) -> dict:
    """Drive the exact CLI command from AUDIT_PROTOCOL.md and parse its verdict."""
    src = Path(loopkit_src).resolve() if loopkit_src else default_loopkit_src()
    cmd = [
        python or sys.executable, "-m", "loopkit.cli", "audit",
        "--claim", repr(float(claim)),
        "--baseline", str(baseline),
        "--candidate", str(candidate),
        "--loops", str(loops),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=timeout)
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode, stdout = -1, ""
        stderr = f"gate timed out after {timeout}s: {exc}"
    except OSError as exc:
        returncode, stdout = -1, ""
        stderr = f"gate could not start: {exc}"

    report: dict | None = None
    try:
        report = json.loads(stdout)
    except (ValueError, TypeError):
        pass
    report_present = report is not None
    loops_ok = bool(report and int(report.get("loops", -1)) == loops)
    claim_valid = bool(report and report.get("claim_valid") is True)
    return {
        "claim": float(claim),
        "returncode": returncode,
        "claim_valid": claim_valid and loops_ok,
        "loops_ok": loops_ok,
        "report_present": report_present,
        "report": report,
        "stderr": stderr.strip(),
    }


def verify(out_dir: Path, *, loopkit_src: Path | None = None,
           seed: int = SAMPLE_SEED, repeats: int = DEFAULT_REPEATS,
           python: str | None = None) -> dict:
    """Run the four gate cases; return a report and write it as JSON."""
    out_dir = Path(out_dir).resolve()
    src = Path(loopkit_src).resolve() if loopkit_src else default_loopkit_src()
    kwargs = {"loopkit_src": src, "python": python}

    samples = synthetic_samples(repeats=repeats, seed=seed)
    paths = write_samples(out_dir / "samples", samples)

    probe = run_gate(0.0, paths["baseline"], paths["candidate"], **kwargs)
    ci_low = (probe["report"] or {}).get("ci_low")
    if not probe["claim_valid"] or not isinstance(ci_low, (int, float)):
        raise RuntimeError(f"gate probe failed; cannot read ci_low: {probe}")

    cases = {
        "self": {
            "expect_valid": False,
            "result": run_gate(1.01, paths["baseline"], paths["baseline"], **kwargs),
        },
        "slower": {
            "expect_valid": False,
            "result": run_gate(2.0, paths["baseline"], paths["slow"], **kwargs),
        },
        "at_bound": {
            "expect_valid": True,
            "result": run_gate(ci_low, paths["baseline"], paths["candidate"], **kwargs),
        },
        "inflated": {
            "expect_valid": False,
            "result": run_gate(ci_low + 0.05, paths["baseline"], paths["candidate"], **kwargs),
        },
    }
    cases = {
        name: {
            "expect_valid": case["expect_valid"],
            "claim_valid": case["result"]["claim_valid"],
            "returncode": case["result"]["returncode"],
            "claim": case["result"]["claim"],
            "report_present": case["result"]["report_present"],
            "stderr": case["result"]["stderr"],
            "ci_low": (case["result"]["report"] or {}).get("ci_low"),
            "ci_high": (case["result"]["report"] or {}).get("ci_high"),
        }
        for name, case in cases.items()
    }
    verdicts_match = all(c["claim_valid"] is c["expect_valid"]
                         for c in cases.values())
    # A crashed or hung CLI is not a fail-closed verdict.
    rendered = all(c["report_present"] and "Traceback" not in c["stderr"]
                   and "timed out" not in c["stderr"] for c in cases.values())
    passed = verdicts_match and rendered
    summary = {
        "verified": bool(passed and probe["loops_ok"] and repeats >= MIN_REPEATS),
        "gate": {
            "loops": DEFAULT_LOOPS,
            "warmup": DEFAULT_WARMUP,
            "min_repeats": MIN_REPEATS,
            "repeats": repeats,
            "sample_seed": seed,
            "bootstrap_seed": DEFAULT_SEED,
            "ci_low": ci_low,
            "note": "synthetic plumbing validation only; no speed claim",
        },
        "cases": cases,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_verification.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".", help="directory for gate_verification.json")
    ap.add_argument("--loopkit-src", default=None, help="path to loopkit/src")
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = ap.parse_args(argv)

    summary = verify(Path(args.out), loopkit_src=args.loopkit_src,
                     seed=args.seed, repeats=args.repeats)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
