#!/usr/bin/env python3
"""Offline, fail-closed closure gate for the loopkit integration doc markers."""

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
CONTRACT_REL = "tools/audit/INTEGRATION_DOC_CLOSURE.json"
SUMMARY_REL = "tools/audit/M38_INTEGRATION_DOC_CLOSURE_SUMMARY.json"
DOC_REL = "../loopkit/docs/INKLING_INTEGRATION.md"
DOC_SHA256 = "9b0993f68bd2169468c4beedb1d54522e7ce9bd3f1c3b1c94d4033120aa0affc"
M44_SUMMARY_REL = "tools/audit/M44_UPSTREAM_CLI_COMPAT_SUMMARY.json"
M44_SUMMARY_SHA256 = "38ab9c92f7c378e7f4ed23d834b2bd60367207b754e80f37e6f359c8a5ce8122"
NOTE = "documentation-contract closure only; no weights, no engine run, no parity, no timing"
DESCRIPTION = (
    "M38 closes the loopkit INKLING_INTEGRATION.md [?] markers using the M29-M37 "
    "audit artifacts; M46 adds M44 evidence for section-3 item (d); this is a "
    "documentation-contract result, not an engine measurement."
)
CYCLE_AUDIT_REQUIRED = "NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory for any future number."
FROZEN_ON = "2026-11-28"
SCHEMA_VERSION = 1

EXPECTED_TOP_KEYS = {
    "schema_version", "frozen_on", "scope", "note", "description", "doc_path",
    "doc_sha256", "checker_sha256", "resolved_needles", "honest_blockers",
    "closure_note", "evidence", "summary_expectations", "provenance",
    "cycle_audit_required", "speed_claim", "m44_summary_sha256",
}

EXPECTED_MILESTONES = ["M29", "M30", "M31", "M32", "M33", "M34", "M35", "M36", "M37", "M44"]
EXPECTED_SUMMARY_KEYS = {
    "verdict", "blocker", "checks", "check_counts", "info", "provenance", "note",
    "scope", "speed_claim", "cycle_audit_required", "no_weights", "no_engine_run",
    "frozen_on", "doc_sha256", "checker_sha256", "evidence_verdicts", "doc_path",
}


def item(name, evidence, needles):
    return {"name": name, "evidence": list(evidence), "needles": list(needles)}


RESOLVED_NEEDLES = [
    item("section2_gguf_shapes", ["M37"], [
        "`[F]` Exact per-tensor shapes resolved by M37 from a bounded range probe of",
        "87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76; GGUF v3, 744 tensors,",
        "`inkengine-llama/tools/audit/GGUF_LOOP_SHAPES.json`. Measurement/contract scope only:",
        "no weights, no engine run, no parity, no timing.",
    ]),
    item("section2_cache_hook", ["M36"], [
        "**Cache-hook plug-in.** `[F]` The wiring is implemented opt-in/default-off in",
        "`inkengine-llama/docs/INKLING_UPSTREAM_WIRING.md` and source-contract-gated by M36",
        "(`tools/audit/UNION_WIRING.json`); no runtime parity or timing has been measured.",
        "The d2027 FP32 side-lane should stay",
    ]),
    item("section4_item1", ["M29"], [
        "1. **Fork/upstream gap & enum identity** `[F]` (M29,",
        "`tools/audit/UPSTREAM_LOOP_CONTRACT.json` and `M29_UPSTREAM_CONTRACT_SUMMARY.json`:",
        "arch string, 47 tensor names, five loop KV keys and slot formula match upstream).",
    ]),
    item("section4_item2", ["M30"], [
        "**Gate semantics: rank-64 gain/forget vs our scalar `gamma`** `[F]` (M30,",
        "`tools/audit/GATE_SEMANTICS.json` and `M30_GATE_SEMANTICS_SUMMARY.json`: down",
        "[64,4608], gain/forget [2304,64] F32, `(1-forget)*cur + gain*message`).",
    ]),
    item("section4_item3", ["M31"], [
        "**Loop boundaries: layers 6",
        "** `[F]` (M31,",
        "`tools/audit/LOOP_BOUNDARIES.json` and `M31_LOOP_BOUNDARIES_SUMMARY.json`: core",
        "layers 6..17, prelude 0..5, coda 18..23, n_slot=48).",
    ]),
    item("section4_item5", ["M32"], [
        "**Pass-indexed KV: separate slots",
        "vs shared/cross-pass only** `[F]` (M32,",
        "`tools/audit/PASS_INDEXED_KV.json` and `M32_PASS_INDEXED_KV_SUMMARY.json`: 36",
        "distinct pass/core-layer slots, RORET key is (ip, pass), not the cache slot).",
    ]),
    item("section4_item6", ["M33", "M34"], [
        "**NVFP4/MXFP8 dequant + scale layout (16 vs 32), and Unsloth quant choice** `[F]`",
        "(M33/M34, `tools/audit/QUANT_LAYOUT.json`, `MXFP8_SCALE_HEADER.json` and their",
        "summaries: converter split-16 NVFP4 / flat-32 MXFP8 confirmed against the real",
        "HF header; Unsloth GGUF header shows IQ1_S/Q2_0 re-quant, not native NVFP4/MXFP8).",
    ]),
    item("section3_item_d", ["M44"], [
        "  MXFP8 layers; (d) RESOLVED by M44 (synthetic CPU stdout-byte compatibility only, evidence below);",
        "- `[F]` Section-3 item (d) is resolved by M44 on a synthetic CPU fixture only: 26/26 checks",
        "  PASS, 0 differing bytes at 918 bytes/side. Contract",
        "  `inkengine-llama/tools/audit/UPSTREAM_CLI_COMPAT.json` (sha256",
        "  5fb7e01e63c526844231daa6905ea7e78c9f5f37d85fbc595a0286bbae6b2ae6); summary",
        "  `M44_UPSTREAM_CLI_COMPAT_SUMMARY.json` (sha256",
        "  38ab9c92f7c378e7f4ed23d834b2bd60367207b754e80f37e6f359c8a5ce8122).",
        "  Compile/stdout-byte scope only: no real weights, no GPU, no tensor or logits parity, no timing,",
        "  NO SPEED CLAIM. Items (b) real-weight conversion parity and (e) tokenizer/chat-template",
        "  remain open; (a), (c) and (f) are unchanged.",
    ]),
]

HONEST_BLOCKERS = [
    item("section4_item4", ["M35"], [
        '4. **"Outlier at dim 2027" has no primary-source support** `[L]`. Experiment: audit where 2027 is',
        "   `[F]` (M35, cheap part only): classified LOCAL_ONLY_UNVERIFIED from primary-source",
        "NO_GO_BLOCKED without weights or hosted inference",
        "   (`tools/audit/OUTLIER_PREMISE.json`, `M35_OUTLIER_PREMISE_SUMMARY.json`).",
        "The d2027 FP32 side-lane should stay",
    ]),
]

CLOSURE_NOTE = item(
    "section5_closure",
    ["M29", "M30", "M31", "M32", "M33", "M34", "M35", "M36", "M37", "M44"],
    [
        "- **M38 closure note (2026-11-28):** section-4 items 1, 2, 3, 5 and 6 and both",
        "  M29-M37; item 4 stays LOCAL_ONLY_UNVERIFIED with real-trace falsification",
        "  NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory.",
        "- **M46 closure note (2026-11-28):** section-3 item (d) is PASS-scoped by M44: 26/26 checks,",
        "  `M44_UPSTREAM_CLI_COMPAT_SUMMARY.json`. Section-3 item (b) real-weight conversion parity and",
        "  no timing, NO SPEED CLAIM. CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory.",
    ],
)

EVIDENCE = {
    "M29": ["tools/audit/UPSTREAM_LOOP_CONTRACT.json", "tools/audit/M29_UPSTREAM_CONTRACT_SUMMARY.json"],
    "M30": ["tools/audit/GATE_SEMANTICS.json", "tools/audit/M30_GATE_SEMANTICS_SUMMARY.json"],
    "M31": ["tools/audit/LOOP_BOUNDARIES.json", "tools/audit/M31_LOOP_BOUNDARIES_SUMMARY.json"],
    "M32": ["tools/audit/PASS_INDEXED_KV.json", "tools/audit/M32_PASS_INDEXED_KV_SUMMARY.json"],
    "M33": ["tools/audit/QUANT_LAYOUT.json", "tools/audit/M33_QUANT_LAYOUT_SUMMARY.json"],
    "M34": ["tools/audit/MXFP8_SCALE_HEADER.json", "tools/audit/M34_MXFP8_SCALE_SUMMARY.json"],
    "M35": ["tools/audit/OUTLIER_PREMISE.json", "tools/audit/M35_OUTLIER_PREMISE_SUMMARY.json"],
    "M36": ["tools/audit/UNION_WIRING.json", "tools/audit/M36_UNION_WIRING_SUMMARY.json"],
    "M37": ["tools/audit/GGUF_LOOP_SHAPES.json", "tools/audit/M37_GGUF_LOOP_SHAPES_SUMMARY.json"],
    "M44": ["tools/audit/UPSTREAM_CLI_COMPAT.json", "tools/audit/M44_UPSTREAM_CLI_COMPAT_SUMMARY.json"],
}

SUMMARY_EXPECTATIONS = {
    "M35_classification": "LOCAL_ONLY_UNVERIFIED",
    "M35_real_trace_verdict": "NO_GO_BLOCKED",
    "M36_no_weights": True,
    "M37_no_weights": True,
    "M37_doc_note_history": (
        "loopkit docs/INKLING_INTEGRATION.md marker stays pending a coordinated two-repo doc update"
    ),
}

PROVENANCE = {
    "branch": "main",
    "doc_path": DOC_REL,
    "frozen_on": FROZEN_ON,
    "scope": NOTE,
    "no_network": True,
    "coordinated_two_repo_update": True,
    "supersedes": "M37 doc_note: marker stays pending a coordinated two-repo doc update",
    "milestones": [
        {
            "milestone": milestone,
            "contract": EVIDENCE[milestone][0],
            "summary": EVIDENCE[milestone][1],
            "verdict": "PASS",
        }
        for milestone in EXPECTED_MILESTONES
    ],
}

NEEDLE_NAMES = [entry["name"] for entry in RESOLVED_NEEDLES + HONEST_BLOCKERS + [CLOSURE_NOTE]]
CHECK_NAMES = [
    "frozen_contract", "frozen_checker_sha256", "live_doc_sha256", "no_open_markers",
    "evidence_pointers", "evidence_summaries", "m44_evidence_pins",
    "summary_expectations", "summary_shape", "no_network", "run_completed",
] + ["needle_" + name for name in NEEDLE_NAMES]

NETWORK_MODULES = {"socket", "http", "urllib", "requests", "httpx", "ftplib", "smtplib", "telnetlib", "xmlrpc"}
NETWORK_CALLS = {"urlopen", "urlretrieve", "socket", "connect"}


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
        "blocker": "checker did not complete",
        "checks": {name: False for name in CHECK_NAMES},
        "check_counts": {"total": len(CHECK_NAMES), "passed": 0, "failed": len(CHECK_NAMES)},
        "info": ["NO_GO sentinel written at checker startup"],
        "provenance": {},
        "note": NOTE,
        "scope": NOTE,
        "speed_claim": None,
        "cycle_audit_required": CYCLE_AUDIT_REQUIRED,
        "no_weights": True,
        "no_engine_run": True,
        "frozen_on": FROZEN_ON,
        "doc_sha256": None,
        "checker_sha256": None,
        "evidence_verdicts": {},
        "doc_path": DOC_REL,
        "sentinel": True,
    }


def check_frozen_contract(facts):
    if not isinstance(facts, dict) or set(facts) != EXPECTED_TOP_KEYS:
        return False
    return bool(
        facts.get("schema_version") == SCHEMA_VERSION
        and facts.get("frozen_on") == FROZEN_ON
        and facts.get("scope") == NOTE
        and facts.get("note") == NOTE
        and facts.get("description") == DESCRIPTION
        and facts.get("doc_path") == DOC_REL
        and facts.get("doc_sha256") == DOC_SHA256
        and facts.get("m44_summary_sha256") == M44_SUMMARY_SHA256
        and is_sha256(facts.get("checker_sha256"))
        and facts.get("resolved_needles") == RESOLVED_NEEDLES
        and facts.get("honest_blockers") == HONEST_BLOCKERS
        and facts.get("closure_note") == CLOSURE_NOTE
        and facts.get("evidence") == EVIDENCE
        and facts.get("summary_expectations") == SUMMARY_EXPECTATIONS
        and facts.get("provenance") == PROVENANCE
        and facts.get("cycle_audit_required") == CYCLE_AUDIT_REQUIRED
        and facts.get("speed_claim") is None
    )


def well_formed_entries(entries):
    return all(
        isinstance(entry, dict)
        and set(entry) == {"name", "evidence", "needles"}
        and isinstance(entry["name"], str)
        and entry["name"]
        and isinstance(entry["evidence"], list)
        and entry["evidence"]
        and all(isinstance(name, str) and name in EVIDENCE for name in entry["evidence"])
        and isinstance(entry["needles"], list)
        and entry["needles"]
        and all(isinstance(needle, str) and needle for needle in entry["needles"])
        for entry in entries
    )


def check_evidence_pointers(facts):
    evidence = facts.get("evidence")
    if evidence != EVIDENCE or not isinstance(evidence, dict):
        return False
    if set(evidence) != set(EXPECTED_MILESTONES):
        return False
    if not well_formed_entries(facts.get("resolved_needles", [])):
        return False
    if not well_formed_entries(facts.get("honest_blockers", [])):
        return False
    closure = facts.get("closure_note")
    if not well_formed_entries([closure]):
        return False
    all_entries = facts["resolved_needles"] + facts["honest_blockers"] + [closure]
    if sorted(entry["name"] for entry in all_entries) != sorted(NEEDLE_NAMES):
        return False
    for entry in all_entries:
        joined = " ".join(entry["needles"])
        if not any(milestone in joined for milestone in entry["evidence"]):
            return False
    return all(
        isinstance(paths, list)
        and len(paths) == 2
        and paths[0].startswith("tools/audit/")
        and paths[1].startswith("tools/audit/")
        and paths[1].endswith("_SUMMARY.json")
        for paths in evidence.values()
    )


def check_evidence_summaries(root, facts):
    """Every cited summary must be on disk, valid JSON and a clean PASS."""
    if not check_evidence_pointers(facts):
        return False
    for milestone in EXPECTED_MILESTONES:
        for rel in facts["evidence"][milestone]:
            path = root / rel
            if not path.is_file():
                return False
            try:
                payload = load_json(path)
            except (OSError, ValueError):
                return False
            if not isinstance(payload, dict):
                return False
            if "_SUMMARY.json" not in rel:
                continue
            if payload.get("verdict") != "PASS":
                return False
            if "blocker" in payload and payload["blocker"] != "":
                return False
            if "speed_claim" in payload and payload["speed_claim"] is not None:
                return False
            counts = payload.get("check_counts")
            if counts is not None:
                if not isinstance(counts, dict):
                    return False
                if counts.get("failed") != 0 or counts.get("passed") + counts.get("failed") != counts.get("total"):
                    return False
    return True


def check_m44_evidence_pins(root):
    """Byte-pin the M44 summary and validate every PASS-scoped field it claims."""
    try:
        raw = (root / M44_SUMMARY_REL).read_bytes()
    except OSError:
        return False
    if sha256_bytes(raw) != M44_SUMMARY_SHA256:
        return False
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return bool(
        payload.get("verdict") == "PASS"
        and payload.get("speed_claim") is None
        and payload.get("blockers") == []
        and payload.get("checks_passed") == 26
        and payload.get("checks_total") == 26
        and payload.get("checks_failed") == 0
        and payload.get("stdout_differing_bytes") == 0
        and payload.get("no_weights") is True
    )


def check_summary_expectations(root, facts):
    expected = facts.get("summary_expectations")
    if expected != SUMMARY_EXPECTATIONS:
        return False
    try:
        m35 = load_json(root / "tools/audit/M35_OUTLIER_PREMISE_SUMMARY.json")
        m36 = load_json(root / "tools/audit/M36_UNION_WIRING_SUMMARY.json")
        m37 = load_json(root / "tools/audit/M37_GGUF_LOOP_SHAPES_SUMMARY.json")
    except (OSError, ValueError):
        return False
    return bool(
        m35.get("classification") == expected["M35_classification"]
        and m35.get("real_trace_verdict") == expected["M35_real_trace_verdict"]
        and m36.get("no_weights") is expected["M36_no_weights"]
        and m37.get("no_weights") is expected["M37_no_weights"]
        and m37.get("doc_note") == expected["M37_doc_note_history"]
    )


def observed_evidence_verdicts(root, facts):
    """Read each cited summary verdict instead of hardcoding PASS."""
    verdicts = {}
    for milestone in EXPECTED_MILESTONES:
        verdict = None
        for rel in (facts.get("evidence") or {}).get(milestone, []):
            if not rel.endswith("_SUMMARY.json"):
                continue
            try:
                payload = load_json(root / rel)
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get("verdict") == "PASS":
                verdict = "PASS"
                break
        verdicts[milestone] = verdict
    return verdicts


def check_summary_shape(summary):
    if set(summary) != EXPECTED_SUMMARY_KEYS:
        return False
    total = len(CHECK_NAMES)
    return bool(
        summary["verdict"] == "PASS"
        and summary["blocker"] == ""
        and set(summary["checks"]) == set(CHECK_NAMES)
        and summary["check_counts"] == {"total": total, "passed": total, "failed": 0}
        and summary["speed_claim"] is None
        and summary["cycle_audit_required"] == CYCLE_AUDIT_REQUIRED
        and summary["no_weights"] is True
        and summary["no_engine_run"] is True
        and summary["doc_sha256"] == DOC_SHA256
        and summary["doc_path"] == DOC_REL
        and "sentinel" not in summary
    )


def check_no_network():
    """Static self-check: no network imports or URL-fetch calls in this checker."""
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
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
    provenance = {}
    doc_sha256 = None
    checker_sha256 = None
    facts = {}
    summary_path = root / SUMMARY_REL
    sentinel_ok = True
    try:
        write_json(summary_path, fail_closed_summary())
    except Exception as exc:
        sentinel_ok = False
        info.append("sentinel_write_failed: %s: %s" % (type(exc).__name__, exc))
    try:
        facts = load_json(root / CONTRACT_REL)
        provenance = dict(facts.get("provenance") or {})
        doc_sha256 = sha256_bytes((root / DOC_REL).read_bytes())
        checker_sha256 = sha256_bytes(Path(__file__).read_bytes())
        checks["frozen_contract"] = check_frozen_contract(facts)
        checks["frozen_checker_sha256"] = (
            is_sha256(facts.get("checker_sha256")) and facts["checker_sha256"] == checker_sha256
        )
        checks["live_doc_sha256"] = doc_sha256 == DOC_SHA256 and facts.get("doc_sha256") == DOC_SHA256
        doc_text = (root / DOC_REL).read_text(encoding="utf-8")
        checks["no_open_markers"] = '`[?]`.' not in doc_text and doc_text.count('`[?]`') == 1
        checks["evidence_pointers"] = check_evidence_pointers(facts)
        checks["evidence_summaries"] = check_evidence_summaries(root, facts)
        checks["m44_evidence_pins"] = check_m44_evidence_pins(root)
        checks["summary_expectations"] = check_summary_expectations(root, facts)
        checks["no_network"] = check_no_network()
        for entry in RESOLVED_NEEDLES + HONEST_BLOCKERS + [CLOSURE_NOTE]:
            checks["needle_" + entry["name"]] = bool(doc_text) and all(
                needle in doc_text for needle in entry["needles"]
            )
        provenance["doc_sha256_live"] = doc_sha256
        provenance["checker_sha256_live"] = checker_sha256
        info.append("doc sha256 frozen=%s live=%s (gated)" % (DOC_SHA256, doc_sha256))
        info.append(
            "closed markers: %d resolved, %d honest blocker, %d section-5 closure note"
            % (len(RESOLVED_NEEDLES), len(HONEST_BLOCKERS), 1)
        )
        info.append(
            "evidence: %d milestones, %d artifact paths, all PASS summaries"
            % (len(EXPECTED_MILESTONES), sum(len(paths) for paths in EVIDENCE.values()))
        )
        info.append(DOC_SHA256)
        checks["run_completed"] = True
    except Exception as exc:
        info.append("exception: %s: %s" % (type(exc).__name__, exc))

    # The shape check runs against a candidate with itself assumed true.
    checks["summary_shape"] = True
    failed_names = [name for name in sorted(checks) if not checks[name]]
    if not sentinel_ok:
        failed_names.append("sentinel_write_failed")
    verdict = "PASS" if not failed_names else "NO_GO"
    blocker = "" if verdict == "PASS" else ", ".join(failed_names)
    passed = sum(1 for value in checks.values() if value)
    summary = {
        "verdict": verdict,
        "blocker": blocker,
        "checks": checks,
        "check_counts": {
            "total": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
        },
        "info": info,
        "provenance": provenance,
        "note": NOTE,
        "scope": NOTE,
        "speed_claim": None,
        "cycle_audit_required": CYCLE_AUDIT_REQUIRED,
        "no_weights": True,
        "no_engine_run": True,
        "frozen_on": FROZEN_ON,
        "doc_sha256": doc_sha256,
        "checker_sha256": checker_sha256,
        "evidence_verdicts": observed_evidence_verdicts(root, facts),
        "doc_path": DOC_REL,
    }
    checks["summary_shape"] = check_summary_shape(summary)
    failed_names = [name for name in sorted(checks) if not checks[name]]
    if not sentinel_ok:
        failed_names.append("sentinel_write_failed")
    verdict = "PASS" if not failed_names else "NO_GO"
    blocker = "" if verdict == "PASS" else ", ".join(failed_names)
    passed = sum(1 for value in checks.values() if value)
    summary["verdict"] = verdict
    summary["blocker"] = blocker
    summary["checks"] = checks
    summary["check_counts"] = {
        "total": len(checks),
        "passed": passed,
        "failed": len(checks) - passed,
    }
    write_json(summary_path, summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    if summary["blocker"]:
        print("blocker: " + summary["blocker"])
    for name in sorted(summary["checks"]):
        print("  %-34s %s" % (name, "PASS" if summary["checks"][name] else "FAIL"))
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
