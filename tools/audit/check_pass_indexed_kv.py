#!/usr/bin/env python3
"""Offline, fail-closed pass-indexed KV slot contract check for the 3-pass Inkling2 loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
SUMMARY_PATH = AUDIT_DIR / "M32_PASS_INDEXED_KV_SUMMARY.json"
NOTE = "pass-indexed KV slot contract only, no timing"

LOOP_CPP = "src/models/inkling2-loop.cpp"
GRAPH_CPP = "src/models/inkling2.cpp"
CONVERTER_PY = "conversion/inkling2.py"

EXPECTED_FACTS = {
    "n_phys": 24,
    "n_passes": 3,
    "core_start": 6,
    "core_end": 18,
    "n_slot": 48,
    "n_ctrl": 4,
    "slot_formula": "n_phys + (size_t) (n_passes - 1)*(core_end - core_start)",
    "layer_phys_formula": "core_start + (il - n_phys) % (core_end - core_start)",
    "pass_slot_formula": "m.n_phys + (pass - 1)*(end - start) + (ip - start)",
}

FROZEN_SOURCE_SHA256 = {
    CONVERTER_PY: "aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7",
    GRAPH_CPP: "c5b543ad965857efd7f9818d42d1f7a418954758b75ec68c82485ffbd341225f",
    LOOP_CPP: "a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab",
}
GATED_SOURCE_FILES = sorted(FROZEN_SOURCE_SHA256)

SOURCE_NEEDLES = {
    "slot_formula": {
        "source": LOOP_CPP,
        "needles": [
            "const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);",
        ],
    },
    "slot_guard": {
        "source": LOOP_CPP,
        "needles": [
            "if (n_slot > LLAMA_MAX_LAYERS) {",
            'throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));',
        ],
    },
    "n_layer_all": {
        "source": LOOP_CPP,
        "needles": ["hparams.n_layer_all = n_slot;"],
    },
    "layer_phys_formula": {
        "source": LOOP_CPP,
        "needles": ["return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);"],
    },
    "weight_share": {
        "source": LOOP_CPP,
        "needles": [
            "// share the physical weights, each cache slot keeps its own KV and conv state",
            "layers[il] = layers[layer_phys(il)];",
        ],
    },
    "slot_hparams_copy": {
        "source": LOOP_CPP,
        "needles": [
            "const int p = layer_phys(il);",
            "hparams.n_head_arr[il]        = hparams.n_head_arr[p];",
            "hparams.is_recr_impl[il]      = hparams.is_recr_impl[p];",
        ],
    },
    "loop_layers_resize": {
        "source": LOOP_CPP,
        "needles": ["loop_layers.resize(core_end - core_start);"],
    },
    "msg_query_shape": {
        "source": LOOP_CPP,
        "needles": [
            'l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);',
        ],
    },
    "pass_slot_expr": {
        "source": LOOP_CPP,
        "needles": [
            "const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);",
        ],
    },
    "pass0_slot_expr": {
        "source": LOOP_CPP,
        "needles": [
            "cur = build_layer(il, ip, cur, blocks, nullptr, nullptr, ip, (int32_t) pass);",
        ],
    },
    "build_layer_args": {
        "source": LOOP_CPP,
        "needles": ["build_layer(il, ip,", "ip, (int32_t) pass"],
    },
    "bank_previous_pass": {
        "source": LOOP_CPP,
        "needles": [
            "bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());",
            "bank_out = cur;",
        ],
    },
    "union_key_comment": {
        "source": LOOP_CPP,
        "needles": [
            "// the union is keyed by the logical core layer ip and the pass index; il stays",
            "// the physical cache slot used for cb names and per-slot hparams",
        ],
    },
    "state_by_il": {
        "source": GRAPH_CPP,
        "needles": ["recr->mctx->get_r_l(il)"],
    },
    "attn_by_il": {
        "source": GRAPH_CPP,
        "needles": [
            'cb(Qcur, "Qcur", il);',
            "Qcur, Kcur, Vcur, kq_b, sinks, 1.0f/sqrtf(n_embd_head), il);",
        ],
    },
    "residual_by_ip": {
        "source": GRAPH_CPP,
        "needles": ["if (ip % hparams.attn_res_block_size == 0) {"],
    },
    "roret_key_args": {
        "source": GRAPH_CPP,
        "needles": ["roret_pass, /* roret_pass */", "roret_il); /* roret_key_il */"],
    },
    "converter_query_shape": {
        "source": CONVERTER_PY,
        "needles": ['2 * (self.loop["core_end"] - start)'],
    },
    "converter_stack": {
        "source": CONVERTER_PY,
        "needles": ["torch.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]])"],
    },
}

DERIVED_CHECK_NAMES = [
    "slot_map_derived",
    "slot_map_injective",
    "slot_map_layer_phys_inverse",
    "slot_map_coverage",
]
CHECK_NAMES = (
    ["frozen_facts", "frozen_source_sha256", "live_source_sha256"]
    + sorted(SOURCE_NEEDLES)
    + DERIVED_CHECK_NAMES
    + ["run_completed"]
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path):
    return path.read_text(encoding="utf-8")


def _valid_boundaries(n_phys, n_passes, core_start, core_end):
    values = (n_phys, n_passes, core_start, core_end)
    return (
        all(isinstance(value, int) and not isinstance(value, bool) for value in values)
        and n_phys >= 1
        and n_passes >= 1
        and core_start >= 0
        and core_end > core_start
        and core_end <= n_phys
    )


def evaluate_slot(n_phys, n_passes, core_start, core_end):
    """Distinct cache slots: n_phys + (n_passes - 1) * (core_end - core_start)."""
    if not _valid_boundaries(n_phys, n_passes, core_start, core_end):
        raise ValueError("invalid slot inputs")
    return n_phys + (n_passes - 1) * (core_end - core_start)


def layer_phys(il, n_phys, core_start, core_end):
    """Inverse of the pass slot map: physical core layer for a cache slot."""
    if (
        not isinstance(il, int)
        or isinstance(il, bool)
        or il < 0
        or not _valid_boundaries(n_phys, 1, core_start, core_end)
    ):
        raise ValueError("invalid physical layer index")
    return il if il < n_phys else core_start + (il - n_phys) % (core_end - core_start)


def pass_slot(pass_index, ip, n_phys, core_start, core_end):
    """Cache slot used by one core layer application in one pass."""
    if (
        not isinstance(pass_index, int)
        or isinstance(pass_index, bool)
        or pass_index < 0
        or not isinstance(ip, int)
        or isinstance(ip, bool)
        or not _valid_boundaries(n_phys, 1, core_start, core_end)
        or not core_start <= ip < core_end
    ):
        raise ValueError("invalid pass or core layer")
    if pass_index == 0:
        return ip
    return n_phys + (pass_index - 1) * (core_end - core_start) + (ip - core_start)


def build_slot_map(n_phys=24, n_passes=3, core_start=6, core_end=18):
    """Derive every (pass, logical core layer, cache slot) triple, pass-major."""
    evaluate_slot(n_phys, n_passes, core_start, core_end)
    return [
        {
            "pass": p,
            "ip": ip,
            "il": pass_slot(p, ip, n_phys, core_start, core_end),
        }
        for p in range(n_passes)
        for ip in range(core_start, core_end)
    ]


def _well_formed(slot_map):
    if not isinstance(slot_map, list) or not slot_map:
        return False
    for entry in slot_map:
        if not isinstance(entry, dict) or set(entry) != {"pass", "ip", "il"}:
            return False
        if not all(isinstance(entry[key], int) and not isinstance(entry[key], bool) for key in entry):
            return False
    return True


def is_injective(slot_map):
    """No two core applications may share a KV/conv cache slot."""
    if not _well_formed(slot_map):
        return False
    slots = [entry["il"] for entry in slot_map]
    return len(set(slots)) == len(slots)


def layer_phys_inverse(slot_map, n_phys, core_start, core_end):
    """Each slot must map back to the logical core layer that wrote it."""
    if not _well_formed(slot_map):
        return False
    try:
        return all(
            layer_phys(entry["il"], n_phys, core_start, core_end) == entry["ip"]
            for entry in slot_map
        )
    except (TypeError, ValueError):
        return False


def covers_core_slots(slot_map, n_phys, n_passes, core_start, core_end):
    """The 36 core applications use slots 6..17 and 24..47 exactly.

    Slots 0..5 (prelude) and 18..23 (coda) are used once by the non-core layers,
    so the 48 layer applications fill all 48 cache slots with no overlap.
    """
    if not _well_formed(slot_map):
        return False
    try:
        n_slot = evaluate_slot(n_phys, n_passes, core_start, core_end)
        slots = [entry["il"] for entry in slot_map]
        expected = set(range(core_start, core_end)) | set(range(n_phys, n_slot))
        return len(slots) == n_passes * (core_end - core_start) and set(slots) == expected
    except (TypeError, ValueError):
        return False


def check_slot_map(slot_map, n_phys, n_passes, core_start, core_end):
    return (
        is_injective(slot_map)
        and layer_phys_inverse(slot_map, n_phys, core_start, core_end)
        and covers_core_slots(slot_map, n_phys, n_passes, core_start, core_end)
    )


def check_frozen_facts(facts):
    expected_map = build_slot_map(
        EXPECTED_FACTS["n_phys"],
        EXPECTED_FACTS["n_passes"],
        EXPECTED_FACTS["core_start"],
        EXPECTED_FACTS["core_end"],
    )
    return bool(
        facts.get("schema_version") == 1
        and facts.get("frozen_on") == "2026-11-28"
        and facts.get("scope") == NOTE
        and facts.get("note") == NOTE
        and facts.get("speed_claim") is None
        and facts.get("facts") == EXPECTED_FACTS
        and facts.get("needles") == SOURCE_NEEDLES
        and facts.get("slot_map") == expected_map
        and check_slot_map(
            facts.get("slot_map"),
            EXPECTED_FACTS["n_phys"],
            EXPECTED_FACTS["n_passes"],
            EXPECTED_FACTS["core_start"],
            EXPECTED_FACTS["core_end"],
        )
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
        facts_path = root / "tools/audit/PASS_INDEXED_KV.json"
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

        texts = {}
        for rel in GATED_SOURCE_FILES:
            texts[rel] = read_text(root / rel) if (root / rel).is_file() else ""
        for name in sorted(SOURCE_NEEDLES):
            needle = SOURCE_NEEDLES[name]
            text = texts.get(needle["source"], "")
            checks[name] = bool(text) and all(part in text for part in needle["needles"])

        f = facts["facts"]
        derived = build_slot_map(f["n_phys"], f["n_passes"], f["core_start"], f["core_end"])
        frozen_map = facts["slot_map"]
        checks["slot_map_derived"] = derived == frozen_map
        checks["slot_map_injective"] = is_injective(frozen_map)
        checks["slot_map_layer_phys_inverse"] = layer_phys_inverse(
            frozen_map, f["n_phys"], f["core_start"], f["core_end"])
        checks["slot_map_coverage"] = covers_core_slots(
            frozen_map, f["n_phys"], f["n_passes"], f["core_start"], f["core_end"])
        n_slot = evaluate_slot(f["n_phys"], f["n_passes"], f["core_start"], f["core_end"])
        info.append(
            "slot map: %d core applications, il %d..%d + %d..%d, n_slot %d"
            % (len(frozen_map), f["core_start"], f["core_end"] - 1, f["n_phys"], n_slot - 1, n_slot)
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
    write_json(root / "tools/audit/M32_PASS_INDEXED_KV_SUMMARY.json", summary)
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
