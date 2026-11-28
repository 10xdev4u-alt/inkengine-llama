#!/usr/bin/env python3
"""Offline, fail-closed loop-boundary contract check for the 3-pass Inkling2 loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
FACTS_PATH = AUDIT_DIR / "LOOP_BOUNDARIES.json"
SUMMARY_PATH = AUDIT_DIR / "M31_LOOP_BOUNDARIES_SUMMARY.json"
NOTE = "loop boundary contract only, no timing"

EXPECTED_FACTS = {
    "core_start": 6,
    "core_end": 18,
    "core_end_exclusive": True,
    "num_passes": 3,
    "gate_rank": 64,
    "is_varloop": {"mini": False, "varloop": True},
    "n_phys": 24,
    "n_slot": 48,
}
EXPECTED_CORE_LAYERS = list(range(6, 18))
EXPECTED_PRELUDE_LAYERS = list(range(0, 6))
EXPECTED_CODA_LAYERS = list(range(18, 24))
EXPECTED_SLOT_FORMULA = "n_phys + (size_t) (n_passes - 1)*(core_end - core_start)"
EXPECTED_LOOP_CONFIG_KEYS = ["core_start", "core_end", "num_passes", "gate_rank", "is_varloop"]
CONVERTER_NEEDLES = {
    "loop_config_keys": "set(self.loop) != {\"core_start\", \"core_end\", \"num_passes\", \"gate_rank\", \"is_varloop\"}",
    "range_check": "0 < start < end <= self.block_count",
    "query_shape": "2 * (self.loop[\"core_end\"] - start)",
    "add_loop_start": "self.gguf_writer.add_loop_start(",
    "add_loop_end": "self.gguf_writer.add_loop_end(",
    "add_loop_count": "self.gguf_writer.add_loop_count(",
    "add_loop_gate_rank": "self.gguf_writer.add_loop_gate_rank(",
    "add_loop_varying": "self.gguf_writer.add_loop_varying(",
}
EXPECTED_CTRL_TENSOR_COUNT = 12
CTRL_ALIAS_NEEDLE = "model.layers.{bid}.controllers."
ENGINE_SPEC_PATHS = {
    "loopkit": "../loopkit/docs/ENGINE_SPEC.md",
    "inkengine": "../inkengine/docs/ENGINE_SPEC.md",
}
CORRECTED_NOTE_NEEDLES = [
    "core layers 6..17 (core_start=6, core_end=18 exclusive)",
    "prelude layers 0..5",
    "coda layers 18..23",
    "6 prelude + 12 loop layers x 3 passes + 6 coda = 48 layer applications",
    "section 4 was corrected on 2026-11-28",
]
STALE_PHRASES = [
    "blocks 0-1, layers 0-11",
    "middle block 3 (layers 18-23)",
    "blocks 0-1's tail layers",
]
FROZEN_SOURCE_SHA256 = {
    "../loopkit/docs/INKLING_FORMAT_FACTS.md": "46f134e263d1f9889a22e556428f7bdc6144fa1c9068072c31794ea19c4071df",
    "conversion/inkling2.py": "aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7",
    "gguf-py/gguf/tensor_mapping.py": "c653a1993368f9967ef3788949f9e71c88b123938889d1180d13ac76b38aa463",
    "src/models/inkling2-loop.cpp": "a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab",
    "gguf-py/gguf/constants.py": "f31289cf677a5125cc37cd58d87228199d5f0a124f583deb95a93fcfd79e086b",
}
GATED_SOURCE_FILES = sorted(FROZEN_SOURCE_SHA256)

_SLOT = re.compile(r"const\s+size_t\s+n_slot\s*=\s*([^;]+);")
_CTRL_ENTRY = re.compile(r"MODEL_TENSOR\.(LOOP_[A-Z0-9_]*CTRL_[A-Z0-9_]*)\s*:\s*\(")

CHECK_NAMES = (
    ["frozen_facts", "frozen_source_sha256", "live_source_sha256"]
    + ["converter_" + name for name in CONVERTER_NEEDLES]
    + [
        "tensor_ctrl_count",
        "tensor_ctrl_aliases",
        "slot_formula",
        "slot_value",
        "core_layer_set",
        "derived_boundaries",
    ]
    + [
        "engine_spec_%s_%s" % (label, suffix)
        for label in ENGINE_SPEC_PATHS
        for suffix in ("corrected", "no_stale")
    ]
    + ["run_completed"]
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path):
    return path.read_text(encoding="utf-8")


def normalize_whitespace(text):
    return " ".join(text.split())


def normalize_doc_text(text):
    """Fold unicode dashes and line wraps so stale prose cannot hide from the gate."""
    for dash in ("\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(dash, "-")
    return normalize_whitespace(text)


def derive_layers(start, end):
    """0-indexed, exclusive end, matching upstream core_start/core_end."""
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        raise ValueError("invalid layer boundaries")
    return list(range(start, end))


def evaluate_slot(n_phys, n_passes, core_start, core_end):
    """Cache slots: n_phys + (n_passes - 1) * (core_end - core_start)."""
    if n_passes < 1 or core_end <= core_start or n_phys < 1:
        raise ValueError("invalid slot inputs")
    return n_phys + (n_passes - 1) * (core_end - core_start)


def extract_slot_formula(cpp_text):
    matches = _SLOT.findall(cpp_text)
    if len(matches) != 1:
        raise ValueError("expected exactly one n_slot formula, got %d" % len(matches))
    formula = " ".join(matches[0].split())
    if not formula:
        raise ValueError("empty n_slot formula")
    return formula


def extract_ctrl_entries(text):
    """Return (name, has_controllers_alias) for each LOOP_*_CTRL_* tensor mapping."""
    matches = list(_CTRL_ENTRY.finditer(text))
    entries = []
    for i, match in enumerate(matches):
        start = match.end()
        end = len(text) if i + 1 == len(matches) else matches[i + 1].start()
        close = re.search(r"\n\s*\),", text[start:end])
        block = text[start:start + close.start()] if close else text[start:end]
        entries.append((match.group(1), CTRL_ALIAS_NEEDLE in block))
    return entries


def check_frozen_facts(facts):
    derived = facts.get("derived", {})
    expected_derived = {
        "core_layers": EXPECTED_CORE_LAYERS,
        "core_layer_count": len(EXPECTED_CORE_LAYERS),
        "prelude_layers": EXPECTED_PRELUDE_LAYERS,
        "coda_layers": EXPECTED_CODA_LAYERS,
        "slot_formula": EXPECTED_SLOT_FORMULA,
    }
    return bool(
        facts.get("schema_version") == 1
        and facts.get("frozen_on") == "2026-11-28"
        and facts.get("scope") == NOTE
        and facts.get("note") == NOTE
        and facts.get("speed_claim") is None
        and facts.get("facts") == EXPECTED_FACTS
        and derived == expected_derived
        and facts.get("loop_config_keys") == EXPECTED_LOOP_CONFIG_KEYS
        and facts.get("converter_needles") == CONVERTER_NEEDLES
        and facts.get("ctrl_tensor_count") == EXPECTED_CTRL_TENSOR_COUNT
        and facts.get("ctrl_alias_needle") == CTRL_ALIAS_NEEDLE
        and facts.get("engine_specs") == ENGINE_SPEC_PATHS
        and facts.get("corrected_note_needles") == CORRECTED_NOTE_NEEDLES
        and facts.get("stale_phrases") == STALE_PHRASES
    )


def check_frozen_sha256(facts):
    return facts.get("provenance", {}).get("source_sha256") == FROZEN_SOURCE_SHA256


def run_check(root):
    checks = {name: False for name in CHECK_NAMES}
    info = []
    provenance = {}
    frozen_on = None
    frozen_facts_sha256 = None
    live = {}
    try:
        facts_path = root / "tools/audit/LOOP_BOUNDARIES.json"
        frozen_facts_sha256 = hashlib.sha256(facts_path.read_bytes()).hexdigest()
        facts = load_json(facts_path)
        frozen_on = facts["frozen_on"]
        provenance = dict(facts.get("provenance", {}))
        checks["frozen_facts"] = check_frozen_facts(facts)
        checks["frozen_source_sha256"] = check_frozen_sha256(facts)

        for rel in GATED_SOURCE_FILES:
            digest = FROZEN_SOURCE_SHA256[rel]
            path = root / rel
            if path.is_file():
                live[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            info.append("sha256 %s frozen=%s live=%s (gated)" % (rel, digest, live.get(rel, "missing")))
        provenance["source_sha256_live"] = live
        checks["live_source_sha256"] = all(live.get(rel) == digest for rel, digest in FROZEN_SOURCE_SHA256.items())
        info.append("frozen facts sha256 %s (INFO only; semantic facts are gated)" % frozen_facts_sha256)

        converter_text = read_text(root / "conversion/inkling2.py")
        for name, needle in CONVERTER_NEEDLES.items():
            checks["converter_" + name] = needle in converter_text

        mapping_text = read_text(root / "gguf-py/gguf/tensor_mapping.py")
        entries = extract_ctrl_entries(mapping_text)
        checks["tensor_ctrl_count"] = len(entries) == EXPECTED_CTRL_TENSOR_COUNT
        checks["tensor_ctrl_aliases"] = (
            len(entries) == EXPECTED_CTRL_TENSOR_COUNT and all(has_alias for _, has_alias in entries)
        )
        info.append("controller tensor entries: %d (expected %d)" % (len(entries), EXPECTED_CTRL_TENSOR_COUNT))

        loop_cpp = read_text(root / "src/models/inkling2-loop.cpp")
        formula = extract_slot_formula(loop_cpp)
        checks["slot_formula"] = (
            normalize_whitespace(formula) == normalize_whitespace(EXPECTED_SLOT_FORMULA)
            and "core_end - core_start" in formula
        )
        f = facts["facts"]
        slot_value = evaluate_slot(f["n_phys"], f["num_passes"], f["core_start"], f["core_end"])
        checks["slot_value"] = slot_value == EXPECTED_FACTS["n_slot"] and f["n_slot"] == EXPECTED_FACTS["n_slot"]
        info.append(
            "n_slot = %d + (%d-1)*(%d-%d) = %d (frozen %d)"
            % (f["n_phys"], f["num_passes"], f["core_end"], f["core_start"], slot_value, f["n_slot"])
        )

        core = derive_layers(f["core_start"], f["core_end"])
        checks["core_layer_set"] = (
            len(core) == len(EXPECTED_CORE_LAYERS)
            and min(core) == 6
            and max(core) == 17
            and 18 not in core
            and 6 in core
        )
        derived = facts["derived"]
        checks["derived_boundaries"] = (
            derived["core_layers"] == core == EXPECTED_CORE_LAYERS
            and derived["prelude_layers"] == EXPECTED_PRELUDE_LAYERS
            and derived["coda_layers"] == EXPECTED_CODA_LAYERS
            and derived["core_layer_count"] == len(EXPECTED_CORE_LAYERS)
        )

        for label, rel in ENGINE_SPEC_PATHS.items():
            path = root / rel
            text = normalize_doc_text(read_text(path)) if path.is_file() else ""
            checks["engine_spec_%s_corrected" % label] = (
                bool(text) and all(needle in text for needle in CORRECTED_NOTE_NEEDLES)
            )
            checks["engine_spec_%s_no_stale" % label] = (
                bool(text) and all(phrase not in text for phrase in STALE_PHRASES)
            )

        checks["run_completed"] = True
    except Exception as exc:
        info.append("exception: %s: %s" % (type(exc).__name__, exc))

    verdict = "PASS" if all(checks.values()) else "NO_GO"
    blocker = "" if verdict == "PASS" else ", ".join(name for name in sorted(checks) if not checks[name])
    summary = {
        "verdict": verdict,
        "blocker": blocker,
        "checks": checks,
        "info": info,
        "provenance": provenance,
        "frozen_source_sha256": FROZEN_SOURCE_SHA256,
        "frozen_facts_sha256": frozen_facts_sha256,
        "frozen_on": frozen_on,
        "note": NOTE,
        "speed_claim": None,
    }
    write_json(root / "tools/audit/M31_LOOP_BOUNDARIES_SUMMARY.json", summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    if summary["blocker"]:
        print("blocker: " + summary["blocker"])
    for name in sorted(summary["checks"]):
        print("  %-34s %s" % (name, "PASS" if summary["checks"][name] else "FAIL"))
    for line in summary["info"]:
        print("  INFO: " + line)
    print("note: " + summary["note"])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv=None):
    parse_args(argv)
    summary = run_check(REPO_ROOT)
    print_summary(summary)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
