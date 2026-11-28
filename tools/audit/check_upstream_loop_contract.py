#!/usr/bin/env python3
"""Offline, fail-closed contract check for the 3-pass Inkling2-loop GGUF contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CONTRACT_PATH = AUDIT_DIR / "UPSTREAM_LOOP_CONTRACT.json"
SUMMARY_PATH = AUDIT_DIR / "M29_UPSTREAM_CONTRACT_SUMMARY.json"
NOTE = "contract parity only, no timing"

_ENUM_MEMBER = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*$")
_AUTO_MEMBER = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*auto\(\)\s*(?:#.*)?$")
_ARCH_STRING = re.compile(r'MODEL_ARCH\.INKLING2_LOOP\s*:\s*"([^"]+)"')
_TENSOR_ENTRY = re.compile(r"^MODEL_TENSOR\.([A-Z0-9_]+),$")
_KV_KEY = re.compile(r"LLM_KV_LOOP_[A-Z0-9_]+")
_SLOT = re.compile(r"const\s+size_t\s+n_slot\s*=\s*([^;]+);")


def class_block(text, name):
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("class " + name + "("):
            start = i
            break
    if start is None:
        raise ValueError("class " + name + " not found")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line[0].isspace():
            end = i
            break
    return lines[start:end]


def extract_arch_int(constants_text):
    members = []
    for line in class_block(constants_text, "MODEL_ARCH"):
        if not _ENUM_MEMBER.match(line):
            continue
        if not _AUTO_MEMBER.match(line):
            raise ValueError("MODEL_ARCH member is not auto(): " + line.strip())
        members.append(_AUTO_MEMBER.match(line).group(1))
    if "INKLING2_LOOP" not in members:
        raise ValueError("INKLING2_LOOP not found in MODEL_ARCH")
    return members.index("INKLING2_LOOP") + 1


def extract_arch_string(constants_text):
    matches = _ARCH_STRING.findall(constants_text)
    if len(matches) != 1:
        raise ValueError("expected exactly one INKLING2_LOOP arch string, got %d" % len(matches))
    return matches[0]


def extract_tensor_names(constants_text):
    lines = constants_text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "MODEL_ARCH.INKLING2_LOOP: ["]
    if len(starts) != 1:
        raise ValueError("expected exactly one MODEL_TENSORS INKLING2_LOOP block")
    names = []
    closed = False
    for line in lines[starts[0] + 1:]:
        stripped = line.strip()
        if stripped == "],":
            closed = True
            break
        match = _TENSOR_ENTRY.match(stripped)
        if match:
            names.append(match.group(1))
    if not closed or not names:
        raise ValueError("MODEL_TENSORS INKLING2_LOOP block is empty or unterminated")
    return names


def extract_loop_kv_keys(cpp_text):
    keys = sorted(set(_KV_KEY.findall(cpp_text)))
    if not keys:
        raise ValueError("no LLM_KV_LOOP_* keys found")
    return keys


def extract_slot_formula(cpp_text):
    matches = _SLOT.findall(cpp_text)
    if len(matches) != 1:
        raise ValueError("expected exactly one n_slot formula, got %d" % len(matches))
    formula = " ".join(matches[0].split())
    if not formula:
        raise ValueError("empty n_slot formula")
    return formula


def extract_facts(constants_text, loop_cpp_text):
    return {
        "arch_int": extract_arch_int(constants_text),
        "arch_string": extract_arch_string(constants_text),
        "tensor_names": extract_tensor_names(constants_text),
        "loop_kv_keys": extract_loop_kv_keys(loop_cpp_text),
        "slot_formula": extract_slot_formula(loop_cpp_text),
    }


def check_wiring_markers(source_texts, markers):
    return {marker["name"]: marker["needle"] in source_texts.get(marker["file"], "") for marker in markers}


def make_check(name, expected, actual):
    return {"name": name, "result": "PASS" if expected == actual else "FAIL", "expected": expected, "actual": actual}


def compare_contract(facts, extracted):
    checks = {}
    checks["arch_string"] = make_check("arch_string", facts["arch_string"], extracted.get("arch_string"))
    checks["tensor_names"] = make_check("tensor_names", list(facts["tensor_names"]), extracted.get("tensor_names"))
    checks["loop_kv_keys"] = make_check("loop_kv_keys", list(facts["loop_kv_keys"]), extracted.get("loop_kv_keys"))
    checks["slot_formula"] = make_check("slot_formula", facts["slot_formula"], extracted.get("slot_formula"))
    checks["fork_arch_int"] = make_check("fork_arch_int", facts["fork_arch_int"], extracted.get("arch_int"))
    wiring = extracted.get("wiring", {})
    for marker in facts.get("rorret_wiring", []):
        name = "wiring_" + marker["name"]
        checks[name] = make_check(name, True, bool(wiring.get(marker["name"], False)))
    info = ["frozen fork arch int %d vs frozen upstream arch int %d: benign upstream enum drift; gate keys on the fork value only"
            % (facts["fork_arch_int"], facts["upstream_arch_int"])]
    if "upstream_arch_int" in extracted:
        live = extracted["upstream_arch_int"]
        if live == facts["upstream_arch_int"]:
            info.append("live upstream arch int %d matches frozen upstream value" % live)
        else:
            info.append("live upstream arch int %d differs from frozen %d: INFO only, no gate failure"
                        % (live, facts["upstream_arch_int"]))
    overall = "PASS" if all(check["result"] == "PASS" for check in checks.values()) else "NO_GO"
    return {"checks": checks, "info": info, "overall": overall}


def force_fail(comparison, name, expected, actual):
    comparison["checks"][name] = make_check(name, expected, actual)
    comparison["overall"] = "NO_GO"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path):
    return path.read_text(encoding="utf-8")


def read_wiring_sources(root, markers):
    texts = {}
    for marker in markers:
        path = root / marker["file"]
        texts[marker["file"]] = read_text(path) if path.is_file() else ""
    return texts


def fetch_upstream_text(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def refresh_in_memory(facts, extracted):
    facts = dict(facts)
    urls = facts["provenance"]["upstream_urls"]
    constants_text = fetch_upstream_text(urls["constants"])
    loop_cpp_text = fetch_upstream_text(urls["loop_cpp"])
    facts["arch_string"] = extract_arch_string(constants_text)
    facts["tensor_names"] = extract_tensor_names(constants_text)
    facts["loop_kv_keys"] = extract_loop_kv_keys(loop_cpp_text)
    facts["slot_formula"] = extract_slot_formula(loop_cpp_text)
    facts["upstream_arch_int"] = extract_arch_int(constants_text)
    extracted["upstream_arch_int"] = facts["upstream_arch_int"]
    provenance = dict(facts["provenance"])
    provenance["upstream_sha256_live"] = {
        "constants": hashlib.sha256(constants_text.encode("utf-8")).hexdigest(),
        "loop_cpp": hashlib.sha256(loop_cpp_text.encode("utf-8")).hexdigest(),
    }
    return facts, provenance


def run_check(root, refresh=False):
    facts = load_json(root / "tools/audit/UPSTREAM_LOOP_CONTRACT.json")
    sources = facts["provenance"]["fork_sources"]
    extracted = {}
    extraction_error = None
    try:
        constants_text = read_text(root / sources["constants"])
        loop_cpp_text = read_text(root / sources["loop_cpp"])
        extracted = extract_facts(constants_text, loop_cpp_text)
        extracted["wiring"] = check_wiring_markers(read_wiring_sources(root, facts["rorret_wiring"]), facts["rorret_wiring"])
    except Exception as exc:
        extraction_error = "%s: %s" % (type(exc).__name__, exc)
    provenance = dict(facts["provenance"])
    mode = "frozen"
    fetch_error = None
    if refresh and extraction_error is None:
        mode = "refresh"
        try:
            facts, provenance = refresh_in_memory(facts, extracted)
        except Exception as exc:
            fetch_error = "%s: %s" % (type(exc).__name__, exc)
    comparison = compare_contract(facts, extracted)
    if extraction_error is not None:
        force_fail(comparison, "extraction", "successful source extraction", extraction_error)
    if fetch_error is not None:
        force_fail(comparison, "upstream_fetch", "successful upstream fetch", fetch_error)
    summary = {
        "verdict": comparison["overall"],
        "checks": comparison["checks"],
        "info": comparison["info"],
        "provenance": provenance,
        "mode": mode,
        "frozen_on": facts["frozen_on"],
        "note": NOTE,
        "speed_claim": None,
    }
    write_json(root / "tools/audit/M29_UPSTREAM_CONTRACT_SUMMARY.json", summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    for name in sorted(summary["checks"]):
        print("  %-26s %s" % (name, summary["checks"][name]["result"]))
    for line in summary["info"]:
        print("  INFO: " + line)
    print("note: " + summary["note"])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="fetch live upstream facts in memory; never rewrites the frozen JSON")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run_check(REPO_ROOT, refresh=args.refresh)
    print_summary(summary)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
