#!/usr/bin/env python3
"""Offline, fail-closed integrity gate for the M29-M38 wiring evidence chain."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CONTRACT_REL = "tools/audit/EVIDENCE_CHAIN.json"
SUMMARY_REL = "tools/audit/M39_EVIDENCE_CHAIN_SUMMARY.json"
WIRING_CONTRACT = "tools/audit/UNION_WIRING.json"
DECISIONS_REL = "DECISIONS.tsv"
BUILD_STATUS_REL = "BUILD_STATUS.md"

FROZEN_ON = "2026-11-28"
SCHEMA_VERSION = 1
NOTE = "wiring evidence integrity only, no timing"
CYCLE_AUDIT_REQUIRED = "NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory for any future number."

MILESTONES = ["M29", "M30", "M31", "M32", "M33", "M34", "M35", "M36", "M37", "M38"]

FROZEN_SUMMARIES = {
    "tools/audit/M29_UPSTREAM_CONTRACT_SUMMARY.json": {
        "sha256": "d2df484e02cf2feb471968f3d8d6dd2e47251d5c218907b030851c9c8cac00a2",
        "verdict": "PASS",
    },
    "tools/audit/M30_GATE_SEMANTICS_SUMMARY.json": {
        "sha256": "b207299febeba25d959a1a804b0133b1c436aa5c2f9e44626c445dfe25375e67",
        "verdict": "PASS",
    },
    "tools/audit/M31_LOOP_BOUNDARIES_SUMMARY.json": {
        "sha256": "17c17dd3543f486fdbb579534253b524b8aba65b9b0f7a0b80983eab0bf386e7",
        "verdict": "PASS",
    },
    "tools/audit/M32_PASS_INDEXED_KV_SUMMARY.json": {
        "sha256": "7e924e6054ca3aa745d146796ef5b07546e1034fefb689bb986ca2543bb29595",
        "verdict": "PASS",
    },
    "tools/audit/M33_QUANT_LAYOUT_SUMMARY.json": {
        "sha256": "1b44752269ccfc3f26717ad9874a39e0b1c3f0511ca9da13a4e42b1b3ca5bd49",
        "verdict": "PASS",
    },
    "tools/audit/M34_MXFP8_SCALE_SUMMARY.json": {
        "sha256": "11234a800afb4c0fc632e96117cf4be95a2a3caa2f9e8c51eb95ab6ef9dc8413",
        "verdict": "PASS",
    },
    "tools/audit/M35_OUTLIER_PREMISE_SUMMARY.json": {
        "sha256": "19416074c6979e612132758a3830da3cbb3d3ea228fc178639748460e5f45c55",
        "verdict": "PASS",
    },
    "tools/audit/M36_UNION_WIRING_SUMMARY.json": {
        "sha256": "0f3c4f866fe9c043b3a97717eee060dd2d79bf3401ca6e98a146ed1c890f77f7",
        "verdict": "PASS",
    },
    "tools/audit/M37_GGUF_LOOP_SHAPES_SUMMARY.json": {
        "sha256": "693289c81e8d3e26affa58e5a3c22e167c56c2e7c2e49f9bed2d86fd849e1cfb",
        "verdict": "PASS",
    },
    "tools/audit/M38_INTEGRATION_DOC_CLOSURE_SUMMARY.json": {
        "sha256": "01b33a65a7e437d44fec8b9fb04be0c0ffcbb7734cc930e0b04f355da1b0e2ce",
        "verdict": "PASS",
    },
}

FROZEN_WIRING_SOURCES = {
    "src/models/inkling2-loop.cpp": "a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab",
    "src/models/models.h": "cb57cd8bf72891ac99ec4acc58818a011ea12fda62ca4fec63cc2c33ea5afb9e",
    "src/models/inkling2.cpp": "c5b543ad965857efd7f9818d42d1f7a418954758b75ec68c82485ffbd341225f",
    "src/llama-graph.cpp": "5f7728c91f4dd0dcf80cbf8ce49bde9b7883af2cfd158788edf541ed3a5fac5e",
    "src/llama-moe-cache.cpp": "5bab3cbf0a66698006f82188c47ceb8673e5a620888ba5e8e71d89c7559d10d4",
    "src/llama-moe-cache-lru.h": "064de659111601d43cd6ba0fbe8a68e526ac73f3b41df183a5cde7648fc80358",
    "src/llama-roret.h": "9018c1f5bb0a16e71743a7a452d5236761743ea7badd552a08717b02b9a89657",
    "src/llama-roret-binding.h": "56d42b8db6bd6229e1dfd7a02e2e5cdae437fd54e7ffb9e18c5aa3f3033791c2",
}

EXPECTED_TOP_KEYS = {
    "schema_version", "frozen_on", "scope", "note", "speed_claim", "summaries",
    "milestones", "wiring_contract", "checker_sha256", "no_weights", "no_engine_run",
}

SUMMARY_PROPS = ["exists", "json_object", "sha256", "verdict", "speed_claim", "blocker"]

NETWORK_MODULES = {"socket", "http", "urllib", "requests", "httpx", "ftplib", "smtplib", "telnetlib", "xmlrpc"}
NETWORK_CALLS = {"urlopen", "urlretrieve", "socket", "connect"}


def milestone_of(rel):
    return Path(rel).name.split("_")[0]


SUMMARY_MILESTONES = [milestone_of(rel) for rel in FROZEN_SUMMARIES]
WIRING_SOURCE_NAMES = sorted(Path(rel).name for rel in FROZEN_WIRING_SOURCES)

CHECK_NAMES = (
    ["frozen_contract", "frozen_checker_sha256", "wiring_contract_json", "wiring_sources_shape"]
    + ["wiring_source_" + name for name in WIRING_SOURCE_NAMES]
    + ["decisions_" + milestone for milestone in MILESTONES]
    + ["build_status_" + milestone for milestone in MILESTONES]
    + ["summary_%s_%s" % (milestone, prop) for milestone in SUMMARY_MILESTONES for prop in SUMMARY_PROPS]
    + ["no_network", "run_completed"]
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


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def fail_closed_summary():
    """Startup sentinel: never leave a stale PASS if the run is killed early."""
    return {
        "verdict": "NO_GO",
        "blocker": ["checker did not complete"],
        "checks": {name: False for name in CHECK_NAMES},
        "check_counts": {"total": len(CHECK_NAMES), "passed": 0, "failed": len(CHECK_NAMES)},
        "info": ["NO_GO sentinel written at checker startup"],
        "note": NOTE,
        "speed_claim": None,
        "no_weights": True,
        "no_engine_run": True,
        "frozen_on": FROZEN_ON,
        "checker_sha256": None,
        "wiring_contract": WIRING_CONTRACT,
        "sentinel": True,
    }


def check_frozen_contract(facts):
    """The whole contract is frozen; any drift, including an extra key, fails closed."""
    if not isinstance(facts, dict) or set(facts) != EXPECTED_TOP_KEYS:
        return False
    return bool(
        facts.get("schema_version") == SCHEMA_VERSION
        and facts.get("frozen_on") == FROZEN_ON
        and facts.get("scope") is None
        and facts.get("note") is None
        and facts.get("speed_claim") is None
        and facts.get("summaries") == FROZEN_SUMMARIES
        and facts.get("milestones") == MILESTONES
        and facts.get("wiring_contract") == WIRING_CONTRACT
        and is_sha256(facts.get("checker_sha256"))
        and facts.get("no_weights") is True
        and facts.get("no_engine_run") is True
    )


def check_checker_sha256(facts, live_sha256):
    """The pin lives in the contract, so the checker source carries no self-hash."""
    return bool(
        is_sha256(live_sha256)
        and is_sha256(facts.get("checker_sha256"))
        and facts["checker_sha256"] == live_sha256
    )


def summary_checks(root, rel, expected):
    """Return the six integrity booleans for one frozen summary."""
    out = {prop: False for prop in SUMMARY_PROPS}
    path = root / rel
    if not path.is_file():
        return out
    out["exists"] = True
    try:
        payload = load_json(path)
    except (OSError, ValueError):
        return out
    if not isinstance(payload, dict):
        return out
    out["json_object"] = True
    out["sha256"] = sha256_bytes(path.read_bytes()) == expected["sha256"]
    out["verdict"] = payload.get("verdict") == expected["verdict"]
    out["speed_claim"] = payload.get("speed_claim") is None
    # M29's older shape has no blocker key; M30-M38 use "". Both mean no blocker.
    # The sha256 check pins exact bytes, so any edit still fails closed.
    out["blocker"] = payload.get("blocker") in ("", [], None)
    return out


def check_wiring_sources_shape(sources):
    return isinstance(sources, dict) and sources == FROZEN_WIRING_SOURCES and all(
        is_sha256(value) for value in sources.values()
    )


def decisions_checks(root):
    """One 'land <milestone>' row per milestone; header and # comments are skipped."""
    out = {milestone: False for milestone in MILESTONES}
    path = root / DECISIONS_REL
    if not path.is_file():
        return out
    what_values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2 or fields[0] == "date":
            continue
        what_values.append(fields[1])
    for milestone in MILESTONES:
        needle = "land " + milestone
        out[milestone] = sum(1 for value in what_values if needle in value) >= 1
    return out


def build_status_checks(root):
    """One '## <milestone> ' section heading per milestone; append-only growth is fine."""
    out = {milestone: False for milestone in MILESTONES}
    path = root / BUILD_STATUS_REL
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8")
    for milestone in MILESTONES:
        out[milestone] = ("## " + milestone + " ") in text
    return out


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


def run_check(root):
    root = Path(root)
    checks = {name: False for name in CHECK_NAMES}
    info = []
    sentinel_ok = True
    summary_path = root / SUMMARY_REL
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

    checker_sha256 = None
    try:
        checker_sha256 = sha256_bytes(Path(__file__).read_bytes())
        facts = load_json(root / CONTRACT_REL)
        checks["frozen_contract"] = check_frozen_contract(facts)
        checks["frozen_checker_sha256"] = check_checker_sha256(facts, checker_sha256)
        info.append("checker sha256 frozen=%s live=%s" % (
            facts.get("checker_sha256"), checker_sha256,
        ))

        wiring = {}
        try:
            wiring = load_json(root / facts["wiring_contract"])
            checks["wiring_contract_json"] = isinstance(wiring, dict)
        except (OSError, ValueError):
            checks["wiring_contract_json"] = False
        sources = wiring.get("sources") if isinstance(wiring, dict) else None
        checks["wiring_sources_shape"] = check_wiring_sources_shape(sources)
        for rel, digest in FROZEN_WIRING_SOURCES.items():
            name = Path(rel).name
            path = root / rel
            live = sha256_bytes(path.read_bytes()) if path.is_file() else None
            checks["wiring_source_" + name] = (
                checks["wiring_sources_shape"] and sources.get(rel) == digest and live == digest
            )
            info.append("wiring sha256 %s frozen=%s live=%s" % (rel, digest, live))

        for milestone, ok in decisions_checks(root).items():
            checks["decisions_" + milestone] = ok
        for milestone, ok in build_status_checks(root).items():
            checks["build_status_" + milestone] = ok
        info.append("decisions rows: %d/%d; build status headings: %d/%d" % (
            sum(checks["decisions_" + m] for m in MILESTONES), len(MILESTONES),
            sum(checks["build_status_" + m] for m in MILESTONES), len(MILESTONES),
        ))

        for rel, expected in FROZEN_SUMMARIES.items():
            milestone = milestone_of(rel)
            for prop, value in summary_checks(root, rel, expected).items():
                checks["summary_%s_%s" % (milestone, prop)] = value
        info.append("frozen summaries: %d; properties each: %d" % (
            len(FROZEN_SUMMARIES), len(SUMMARY_PROPS),
        ))

        checks["no_network"] = check_no_network()
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
            "total": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
        },
        "info": info,
        "note": NOTE,
        "speed_claim": None,
        "no_weights": True,
        "no_engine_run": True,
        "frozen_on": FROZEN_ON,
        "checker_sha256": checker_sha256,
        "wiring_contract": WIRING_CONTRACT,
        "cycle_audit_required": CYCLE_AUDIT_REQUIRED,
    }
    write_json(summary_path, summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    if summary["blocker"]:
        print("blocker: " + ", ".join(summary["blocker"]))
    for name in sorted(summary["checks"]):
        print("  %-46s %s" % (name, "PASS" if summary["checks"][name] else "FAIL"))
    counts = summary["check_counts"]
    print("  counts: %d/%d passed, %d failed" % (counts["passed"], counts["total"], counts["failed"]))
    for line in summary["info"]:
        print("  INFO: " + line)
    print("note: " + summary["note"])
    print("cycle_audit_required: " + summary["cycle_audit_required"])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to check (default: this repo)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        summary = run_check(args.root)
    except Exception as exc:
        print("error: could not write summary: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print_summary(summary)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
