#!/usr/bin/env python3
"""Offline, fail-closed contract check for the opt-in RORET union-residency wiring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CONTRACT_PATH = AUDIT_DIR / "UNION_WIRING.json"
SUMMARY_NAME = "M36_UNION_WIRING_SUMMARY.json"
NOTE = "union-residency wiring source contract only, no timing, no weights, no engine run"
CYCLE_AUDIT_REQUIRED = "NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory for any future number."
DOC_PATH = "docs/INKLING_UPSTREAM_WIRING.md"
FORK_HEAD = "04a2c23537226816d5348541329026d8483c3f59"
FROZEN_ON = "2026-11-28"

LOOP_CPP = "src/models/inkling2-loop.cpp"
MODELS_H = "src/models/models.h"
INKLING_CPP = "src/models/inkling2.cpp"
GRAPH_CPP = "src/llama-graph.cpp"
MOE_CPP = "src/llama-moe-cache.cpp"
LRU_H = "src/llama-moe-cache-lru.h"
RORET_H = "src/llama-roret.h"
BINDING_H = "src/llama-roret-binding.h"

FROZEN_SOURCES = {
    LOOP_CPP: "a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab",
    MODELS_H: "cb57cd8bf72891ac99ec4acc58818a011ea12fda62ca4fec63cc2c33ea5afb9e",
    INKLING_CPP: "c5b543ad965857efd7f9818d42d1f7a418954758b75ec68c82485ffbd341225f",
    GRAPH_CPP: "5f7728c91f4dd0dcf80cbf8ce49bde9b7883af2cfd158788edf541ed3a5fac5e",
    MOE_CPP: "5bab3cbf0a66698006f82188c47ceb8673e5a620888ba5e8e71d89c7559d10d4",
    LRU_H: "064de659111601d43cd6ba0fbe8a68e526ac73f3b41df183a5cde7648fc80358",
    RORET_H: "9018c1f5bb0a16e71743a7a452d5236761743ea7badd552a08717b02b9a89657",
    BINDING_H: "56d42b8db6bd6229e1dfd7a02e2e5cdae437fd54e7ffb9e18c5aa3f3033791c2",
}
GATED_SOURCE_FILES = sorted(FROZEN_SOURCES)


def item(name, source, needles):
    return {"name": name, "source": source, "needles": list(needles)}


def absent_item(name, source, needle, start=None, end=None):
    window = None if start is None else {"start": start, "end": end}
    return {"name": name, "source": source, "needle": needle, "window": window}


CHAIN_ITEMS = [
    item("env_gate", MOE_CPP, ['getenv("LLAMA_RORET")']),
    item("loop_logical_key", LOOP_CPP, [
        "cur = build_layer(il, ip, cur, blocks",
        "ip, (int32_t) pass",
    ]),
    item("build_layer_defaults", MODELS_H, ["int32_t roret_il = -1, int32_t roret_pass = -1"]),
    item("forward_roret_args", INKLING_CPP, [
        "roret_pass",
        "roret_il); /* roret_key_il */",
    ]),
    item("graph_lookup_key", GRAPH_CPP, ["{ roret_key_il, roret_pass }"]),
    item("binding_lookup_key", BINDING_H, [
        "inline const T * llama_roret_bind_key(const T * slot_map, const T * lookup) {",
        "return lookup;",
    ]),
    item("binding_set_find", BINDING_H, [
        "std::unordered_map<const T *, llama_roret_binding> by_lookup;",
        "void set(const T * lookup, int32_t il, int32_t pass) {",
        "by_lookup[lookup] = { il, pass };",
        "const llama_roret_binding * find(const T * lookup) const {",
        "return it == by_lookup.end() ? nullptr : &it->second;",
    ]),
    item("bind_slot_map", MOE_CPP, [
        "pimpl->roret_of.set(llama_roret_bind_key(slot_map, slots), layer.roret_key.il, layer.roret_key.pass)",
    ]),
    item("resolve_binding", MOE_CPP, [
        "const llama_roret_binding * rb = roret_of.find(llama_roret_bind_key(src, lookup));",
    ]),
    item("copy_extra_defaults", MOE_CPP, [
        "const int32_t * extra   = nullptr",
        "n_extra = 0;",
    ]),
    item("clear_pass0_union", MOE_CPP, ["roret->clear_layer(rb->il)"]),
    item("union_state_and_clear", RORET_H, [
        "std::vector<uint64_t> union_bits;",
        "std::array<std::vector<uint64_t>, kMaxPass + 1> pass;",
        "mutable std::map<int32_t, layer_state> layers_;",
        "void clear_layer(int32_t il) const { layers_.erase(il); }",
    ]),
    item("union_add", RORET_H, [
        "bool add(const key & k, const int32_t * ids, size_t n_ids) const {",
        "layer.union_bits[w] |= b;",
    ]),
    item("register_real_ids", MOE_CPP, ["roret->add({ rb->il, rb->pass }, ids.data(), ids.size());"]),
    item("union_residents", RORET_H, [
        "std::vector<int32_t> residents(int32_t il) const {",
        "uint64_t bits = layer->union_bits[w];",
    ]),
    item("extra_from_residents", MOE_CPP, [
        "extra   = roret_residents.data()",
        "n_extra = roret_residents.size()",
    ]),
    item("plan_with_extra", MOE_CPP, [
        "g.lru.plan(il, ids.data(), ids.size(), dims.n_tokens, n_used, n_zero, n_cpu, fills, cpu, n_hit, extra, n_extra)",
    ]),
    item("lru_hint_contract", LRU_H, ["extra/n_extra are the opt-in RORET residents"]),
]

DEFAULT_OFF_ITEMS = [
    item("env_gate_explicit_values", MOE_CPP, [
        'if (const char * env = getenv("LLAMA_RORET"); env != nullptr) {',
        'if (on == "1" || on == "on" || on == "true") {',
    ]),
    item("default_off_comment", MOE_CPP, [
        "// RORET is strictly opt-in: without LLAMA_RORET the fields above stay empty and",
    ]),
    item("copy_extra_null", MOE_CPP, [
        "const int32_t * extra   = nullptr;",
        "if (roret) {",
    ]),
    item("default_build_layer_params", MODELS_H, ["int32_t roret_il = -1, int32_t roret_pass = -1"]),
    item("lru_default_params", LRU_H, ["const int32_t * extra = nullptr, size_t n_extra = 0) {"]),
    item("graph_block_guard", GRAPH_CPP, [
        "if (roret_cache != nullptr && roret_pass >= 0 && roret_selected_ids != nullptr && roret_n_selected > 0) {",
    ]),
    item("bind_guard", MOE_CPP, ["if (pimpl->roret) {"]),
]

GRAPH_RORET_START = "// RORET (opt-in): host-side bookkeeping only"
GRAPH_RORET_END = "if (arch == LLM_ARCH_GROVEMOE"
MOE_COPY_START = "const int32_t * extra   = nullptr;"
MOE_COPY_END = "// part of the misses is computed"
LRU_PLAN_START = "template <typename F>\n    bool plan("
LRU_PLAN_END = "if (uniq.size() > (size_t) n_slots) {"

ABSENT_ITEMS = [
    absent_item("no_selected_experts_assign_in_roret_block", GRAPH_CPP, "selected_experts =",
                GRAPH_RORET_START, GRAPH_RORET_END),
    absent_item("no_router_rerun_in_roret_block", GRAPH_CPP, "ggml_argsort_top_k",
                GRAPH_RORET_START, GRAPH_RORET_END),
    absent_item("no_residents_overwrite_ids", MOE_CPP, "ids = roret_residents",
                MOE_COPY_START, MOE_COPY_END),
    absent_item("no_extra_mutates_selected_ids", LRU_H, "ids[i] =",
                LRU_PLAN_START, LRU_PLAN_END),
    absent_item("no_forced_env_enable", MOE_CPP, 'setenv("LLAMA_RORET"'),
]

CHECK_NAMES = (
    ["frozen_contract", "frozen_source_sha256", "live_source_sha256"]
    + ["chain_" + entry["name"] for entry in CHAIN_ITEMS]
    + ["default_off_" + entry["name"] for entry in DEFAULT_OFF_ITEMS]
    + ["absent_" + entry["name"] for entry in ABSENT_ITEMS]
    + ["chain_coverage", "run_completed"]
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
        "frozen_on": None,
        "frozen_contract_sha256": None,
        "frozen_source_sha256": FROZEN_SOURCES,
        "doc_path": DOC_PATH,
        "sentinel": True,
    }


def read_text(path):
    return path.read_text(encoding="utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def check_frozen_contract(facts):
    provenance = facts.get("provenance") or {}
    return bool(
        facts.get("schema_version") == 1
        and facts.get("frozen_on") == FROZEN_ON
        and facts.get("scope") == NOTE
        and facts.get("note") == NOTE
        and facts.get("speed_claim") is None
        and facts.get("sources") == FROZEN_SOURCES
        and facts.get("chain") == CHAIN_ITEMS
        and facts.get("default_off") == DEFAULT_OFF_ITEMS
        and facts.get("absent") == ABSENT_ITEMS
        and provenance.get("doc_path") == DOC_PATH
        and provenance.get("branch") == "main"
        and provenance.get("fork_head") == FORK_HEAD
    )


def check_frozen_source_sha256(facts):
    return facts.get("sources") == FROZEN_SOURCES


def scoped_text(text, window):
    """Return the window body, or None when the window markers are missing."""
    if window is None:
        return text
    start = window.get("start")
    end = window.get("end")
    if not isinstance(start, str) or not isinstance(end, str) or not start or not end:
        return None
    if start not in text:
        return None
    begin = text.index(start) + len(start)
    if end not in text[begin:]:
        return None
    return text[begin:text.index(end, begin)]


def well_formed_needles(items):
    return all(
        isinstance(entry.get("name"), str)
        and entry["name"]
        and isinstance(entry.get("source"), str)
        and entry["source"] in FROZEN_SOURCES
        and isinstance(entry.get("needles"), list)
        and entry["needles"]
        and all(isinstance(needle, str) and needle for needle in entry["needles"])
        for entry in items
    )


def check_chain_coverage(chain, default_off, absent):
    return bool(
        well_formed_needles(chain)
        and well_formed_needles(default_off)
        and {entry["source"] for entry in chain} == set(GATED_SOURCE_FILES)
        and all(
            isinstance(entry.get("name"), str)
            and entry["name"]
            and isinstance(entry.get("source"), str)
            and entry["source"] in FROZEN_SOURCES
            and isinstance(entry.get("needle"), str)
            and entry["needle"]
            for entry in absent
        )
    )


def run_check(root):
    checks = {name: False for name in CHECK_NAMES}
    info = []
    provenance = {}
    live = {}
    frozen_on = None
    contract_sha256 = None
    summary_path = root / "tools/audit" / SUMMARY_NAME
    try:
        write_json(summary_path, fail_closed_summary())
    except Exception as exc:
        info.append("sentinel_write_failed: %s: %s" % (type(exc).__name__, exc))
    try:
        contract_path = root / "tools/audit/UNION_WIRING.json"
        contract_sha256 = sha256_bytes(contract_path.read_bytes())
        facts = load_json(contract_path)
        frozen_on = facts["frozen_on"]
        provenance = dict(facts.get("provenance") or {})
        checks["frozen_contract"] = check_frozen_contract(facts)
        checks["frozen_source_sha256"] = check_frozen_source_sha256(facts)

        for rel in GATED_SOURCE_FILES:
            digest = FROZEN_SOURCES[rel]
            path = root / rel
            if path.is_file():
                live[rel] = sha256_bytes(path.read_bytes())
            info.append("sha256 %s frozen=%s live=%s (gated)" % (rel, digest, live.get(rel, "missing")))
        provenance["source_sha256_live"] = live
        checks["live_source_sha256"] = all(
            live.get(rel) == digest for rel, digest in FROZEN_SOURCES.items()
        )
        info.append("frozen contract sha256 %s (INFO only; semantic facts are gated)" % contract_sha256)

        texts = {
            rel: read_text(root / rel) if (root / rel).is_file() else ""
            for rel in GATED_SOURCE_FILES
        }
        for entry in facts["chain"]:
            text = texts.get(entry["source"], "")
            checks["chain_" + entry["name"]] = bool(text) and all(
                needle in text for needle in entry["needles"]
            )
        for entry in facts["default_off"]:
            text = texts.get(entry["source"], "")
            checks["default_off_" + entry["name"]] = bool(text) and all(
                needle in text for needle in entry["needles"]
            )
        for entry in facts["absent"]:
            text = texts.get(entry["source"], "")
            body = scoped_text(text, entry.get("window"))
            checks["absent_" + entry["name"]] = bool(text) and body is not None and (
                entry["needle"] not in body
            )

        checks["chain_coverage"] = check_chain_coverage(
            facts["chain"], facts["default_off"], facts["absent"]
        )
        info.append(
            "chain: %d ordered checks over %d gated sources; default_off: %d; absent: %d"
            % (len(facts["chain"]), len(FROZEN_SOURCES), len(facts["default_off"]), len(facts["absent"]))
        )
        checks["run_completed"] = True
    except Exception as exc:
        info.append("exception: %s: %s" % (type(exc).__name__, exc))

    verdict = "PASS" if all(checks.values()) else "NO_GO"
    blocker = "" if verdict == "PASS" else ", ".join(
        name for name in sorted(checks) if not checks[name]
    )
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
        "frozen_on": frozen_on,
        "frozen_contract_sha256": contract_sha256,
        "frozen_source_sha256": FROZEN_SOURCES,
        "doc_path": DOC_PATH,
    }
    write_json(summary_path, summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    if summary["blocker"]:
        print("blocker: " + summary["blocker"])
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
