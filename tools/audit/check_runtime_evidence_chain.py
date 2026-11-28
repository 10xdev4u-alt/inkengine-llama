#!/usr/bin/env python3
"""Offline, fail-closed integrity gate for the M39-M42 runtime wiring evidence chain."""

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
CONTRACT_REL = "tools/audit/RUNTIME_EVIDENCE_CHAIN.json"
SUMMARY_REL = "tools/audit/M43_RUNTIME_EVIDENCE_CHAIN_SUMMARY.json"
DECISIONS_REL = "DECISIONS.tsv"
BUILD_STATUS_REL = "BUILD_STATUS.md"

FROZEN_ON = "2026-11-28"
SCHEMA_VERSION = 1
NOTE = "runtime wiring evidence integrity only, no timing"
CYCLE_AUDIT_REQUIRED = "NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory for any future number."

MILESTONES = ["M39", "M40", "M41", "M42"]

FROZEN_SUMMARIES = {
    "tools/audit/M39_EVIDENCE_CHAIN_SUMMARY.json": {
        "sha256": "4f6361b562a82ad2942f32b1b8f07bb5bf22417701f1ca940c0a1b837ae78ba3",
        "verdict": "PASS",
    },
    "tools/audit/M40_3PASS_BINDING_SUMMARY.json": {
        "sha256": "734658f8a2070dfa8798ceaaa08c4724e28cb8f8c678b9c38236303b097cd9ae",
        "verdict": "PASS",
    },
    "tools/audit/M41_3PASS_REPLAY_SUMMARY.json": {
        "sha256": "94e67830ba309791b2bd87abc5c8ee1225fbe964f0c691e5082d8a31ccb9b186",
        "verdict": "PASS",
    },
    "tools/audit/M42_BIND_TENSOR_LINK_SUMMARY.json": {
        "sha256": "953cfcc067aac769546a20792daac2ab8c86b51497fa5599e041f3919284c6d2",
        "verdict": "PASS",
    },
}

FROZEN_CONTRACTS = {
    "evidence_chain": {
        "path": "tools/audit/EVIDENCE_CHAIN.json",
        "sha256": "065311f38f1fff6f51771b57b0c6887775b0035af04a37f928ab304ccddae86b",
    },
    "evidence_chain_checker": {
        "path": "tools/audit/check_evidence_chain.py",
        "sha256": "aba235c2ae6291b48135843295045e564958541b85fc8aac20cf1333ad099e79",
    },
    "wiring_contract": {
        "path": "tools/audit/UNION_WIRING.json",
        "sha256": "c5e024c79b86a740e4dfb6123946610fbc74df0a58c70f0d1ae88ccde4b851da",
    },
    "binding_contract": {
        "path": "tools/audit/THREEPASS_BINDING.json",
        "sha256": "d7957f4809285c19f8e68d58ec2a12b5d7062127fef06fe693bcab10adeddd7e",
    },
    "replay_contract": {
        "path": "tools/audit/THREEPASS_REPLAY.json",
        "sha256": "00e7215637a42f08effaaa5a129846ffcbdb1297284f738b95c04df02ae4657b",
    },
    "runtime_contract": {
        "path": "tools/audit/BIND_TENSOR_LINK.json",
        "sha256": "f9628d0a755dab050f6f445ef2d5e3fa8eedec02069440b850d42cdd3cac4378",
    },
}

EXPECTED_TOP_KEYS = {
    "schema_version", "frozen_on", "scope", "note", "speed_claim", "milestones",
    "summaries", "contracts", "checker_sha256", "no_weights", "no_engine_run",
}
EXPECTED_CONTRACT_KEYS = set(FROZEN_CONTRACTS)
CONTRACT_ENTRY_KEYS = {"path", "sha256"}
SUMMARY_PROPS = [
    "exists", "json_object", "sha256", "verdict", "speed_claim", "blocker",
    "check_counts_failed", "check_counts_passed_total",
]
NETWORK_MODULES = {"socket", "http", "urllib", "requests", "httpx", "ftplib", "smtplib", "telnetlib", "xmlrpc"}
NETWORK_CALLS = {"urlopen", "urlretrieve", "socket", "connect"}

CHECK_NAMES = (
    ["frozen_contract", "frozen_checker_sha256", "contracts_shape"]
    + ["contract_" + name for name in sorted(FROZEN_CONTRACTS)]
    + ["decisions_" + milestone for milestone in MILESTONES]
    + ["build_status_" + milestone for milestone in MILESTONES]
    + ["summary_%s_%s" % (milestone, prop) for milestone in MILESTONES for prop in SUMMARY_PROPS]
    + ["generated_summary_shape", "no_network", "run_completed"]
)


def milestone_of(rel):
    return Path(rel).name.split("_")[0]


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


def is_plain_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


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
        "cycle_audit_required": CYCLE_AUDIT_REQUIRED,
        "sentinel": True,
    }


def check_contracts_shape(contracts):
    if not isinstance(contracts, dict) or set(contracts) != EXPECTED_CONTRACT_KEYS:
        return False
    return all(
        isinstance(entry, dict)
        and set(entry) == CONTRACT_ENTRY_KEYS
        and isinstance(entry["path"], str)
        and bool(entry["path"])
        and is_sha256(entry["sha256"])
        for entry in contracts.values()
    )


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
        and facts.get("milestones") == MILESTONES
        and facts.get("summaries") == FROZEN_SUMMARIES
        and facts.get("contracts") == FROZEN_CONTRACTS
        and check_contracts_shape(facts.get("contracts"))
        and is_sha256(facts.get("checker_sha256"))
        and facts.get("no_weights") is True
        and facts.get("no_engine_run") is True
    )


def check_checker_sha256(facts, live_sha256):
    """The pin lives in the contract, so the checker source carries no self-hash."""
    return bool(
        is_sha256(live_sha256)
        and isinstance(facts, dict)
        and is_sha256(facts.get("checker_sha256"))
        and facts["checker_sha256"] == live_sha256
    )


def contract_checks(root, facts):
    """One boolean per pinned contract: frozen entry equals live bytes at frozen path."""
    out = {name: False for name in FROZEN_CONTRACTS}
    contracts = facts.get("contracts") if isinstance(facts, dict) else None
    if not isinstance(contracts, dict):
        return out
    for name, expected in FROZEN_CONTRACTS.items():
        rel = expected["path"]
        path = root / rel
        live = sha256_bytes(path.read_bytes()) if path.is_file() else None
        out[name] = contracts.get(name) == expected and live == expected["sha256"]
    return out


def summary_checks(root, rel, expected):
    """Return the eight integrity booleans for one frozen summary."""
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
    # A missing blocker key means no blocker, exactly as in the M39 gate.
    # The byte-exact sha256 pin still detects any edit to these summaries.
    out["blocker"] = payload.get("blocker") in ("", [], None)
    counts = payload.get("check_counts")
    if isinstance(counts, dict):
        out["check_counts_failed"] = is_plain_int(counts.get("failed")) and counts["failed"] == 0
        total = counts.get("total")
        passed = counts.get("passed")
        out["check_counts_passed_total"] = bool(
            is_plain_int(total) and total > 0 and is_plain_int(passed) and passed == total
        )
    return out


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
        prefix = "land " + milestone
        # Anchor the milestone token: "land M410-CANCELLED" must not satisfy M41.
        out[milestone] = sum(
            1 for value in what_values
            if value == prefix or value.startswith(prefix + " ")
        ) >= 1
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


def check_generated_summary_shape(summary):
    """The artifact about to be written must have the promised fail-closed shape."""
    if not isinstance(summary, dict):
        return False
    checks = summary.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECK_NAMES):
        return False
    if not all(value is True or value is False for value in checks.values()):
        return False
    counts = summary.get("check_counts")
    if not isinstance(counts, dict) or set(counts) != {"total", "passed", "failed"}:
        return False
    total, passed, failed = counts["total"], counts["passed"], counts["failed"]
    if not (is_plain_int(total) and total == len(CHECK_NAMES)):
        return False
    if not (is_plain_int(passed) and is_plain_int(failed) and passed + failed == total):
        return False
    if failed != sum(1 for value in checks.values() if value is False):
        return False
    blocker = summary.get("blocker")
    if not isinstance(blocker, list) or not all(isinstance(item, str) for item in blocker):
        return False
    verdict = summary.get("verdict")
    if verdict not in ("PASS", "NO_GO"):
        return False
    if verdict != ("PASS" if not blocker else "NO_GO"):
        return False
    info = summary.get("info")
    if not isinstance(info, list) or not all(isinstance(line, str) for line in info):
        return False
    return bool(
        summary.get("note") == NOTE
        and summary.get("speed_claim") is None
        and summary.get("no_weights") is True
        and summary.get("no_engine_run") is True
        and summary.get("frozen_on") == FROZEN_ON
        and is_sha256(summary.get("checker_sha256"))
        and summary.get("cycle_audit_required") == CYCLE_AUDIT_REQUIRED
        and "sentinel" not in summary
    )


def build_summary(checks, checker_sha256, info, sentinel_ok):
    failed_names = [name for name in sorted(checks) if not checks[name]]
    if not sentinel_ok:
        failed_names.append("sentinel_write_failed")
    verdict = "PASS" if not failed_names else "NO_GO"
    passed = sum(1 for value in checks.values() if value)
    return {
        "verdict": verdict,
        "blocker": failed_names,
        "checks": dict(checks),
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
        "cycle_audit_required": CYCLE_AUDIT_REQUIRED,
    }


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
        checks["contracts_shape"] = check_contracts_shape(
            facts.get("contracts") if isinstance(facts, dict) else None
        )
        for name, ok in contract_checks(root, facts).items():
            checks["contract_" + name] = ok
            expected = FROZEN_CONTRACTS[name]
            path = root / expected["path"]
            live = sha256_bytes(path.read_bytes()) if path.is_file() else None
            info.append("contract %s path=%s frozen=%s live=%s" % (
                name, expected["path"], expected["sha256"], live,
            ))

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

    summary = build_summary(checks, checker_sha256, info, sentinel_ok)
    checks["generated_summary_shape"] = check_generated_summary_shape(summary)
    summary = build_summary(checks, checker_sha256, info, sentinel_ok)
    write_json(summary_path, summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    if summary["blocker"]:
        print("blocker: " + ", ".join(summary["blocker"]))
    for name in sorted(summary["checks"]):
        print("  %-52s %s" % (name, "PASS" if summary["checks"][name] else "FAIL"))
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
