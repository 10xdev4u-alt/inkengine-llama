#!/usr/bin/env python3
"""Offline, fail-closed checker for the frozen GGUF loop-tensor shape contract."""

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
FACTS_REL = "tools/audit/GGUF_LOOP_SHAPES.json"
SUMMARY_REL = "tools/audit/M37_GGUF_LOOP_SHAPES_SUMMARY.json"
M33_REL = "tools/audit/QUANT_LAYOUT.json"
NOTE = "GGUF loop-tensor shape contract only; no timing or speed claim"
DESCRIPTION = "Frozen GGUF v3 loop and residual tensor shape contract measured from the first 16 MiB of Inkling-2-Mini-UD-IQ1_S.gguf; GGUF dims are reversed relative to HF shapes."
DOC_NOTE = "loopkit docs/INKLING_INTEGRATION.md marker stays pending a coordinated two-repo doc update"
FROZEN_ON = "2026-11-28"
SCHEMA_VERSION = 1
GGUF_SHA256 = "87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76"

CONVERTER_PY = "conversion/inkling2.py"
CONSTANTS_PY = "gguf-py/gguf/constants.py"
LOOP_CPP = "src/models/inkling2-loop.cpp"
INKLING_CPP = "src/models/inkling2.cpp"
FROZEN_SOURCES = {
    CONVERTER_PY: "aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7",
    CONSTANTS_PY: "f31289cf677a5125cc37cd58d87228199d5f0a124f583deb95a93fcfd79e086b",
    LOOP_CPP: "a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab",
    INKLING_CPP: "c5b543ad965857efd7f9818d42d1f7a418954758b75ec68c82485ffbd341225f",
}

BASE_DIMENSIONS = {
    "hidden": 2304,
    "gate_rank": 64,
    "n_ctrl_families": 4,
    "n_phys": 24,
    "n_passes": 3,
    "core_start": 6,
    "core_end": 18,
    "attn_res_block": 6,
    "attn_tail_rows": 3,
}


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def derive_dimensions(base):
    """Derive the graph dimensions from primary-source constants, not from a claim."""
    for key, value in base.items():
        if not is_int(value) or value <= 0:
            raise ValueError("dimension %s must be a positive integer" % key)
    hidden = base["hidden"]
    rank = base["gate_rank"]
    families = base["n_ctrl_families"]
    n_phys = base["n_phys"]
    passes = base["n_passes"]
    start = base["core_start"]
    end = base["core_end"]
    block = base["attn_res_block"]
    if families != len(("q", "k", "v", "ffn")):
        raise ValueError("loop controller families must be q, k, v and ffn")
    if not 0 < start < end <= n_phys:
        raise ValueError("core layers must lie inside the physical layers")
    if start % block or end % block or n_phys % block:
        raise ValueError("core and physical layers must align with the residual block")
    if passes < 2:
        raise ValueError("the loop needs at least two passes")
    if base["attn_tail_rows"] != 3:
        raise ValueError("the residual block tail has three rows")
    tail = [il for il in range(1, n_phys) if il % block == block - 1]
    if tail[0] != start - 1 or end - 1 not in tail:
        raise ValueError("residual block tails must bracket the loop core")
    return {
        "hidden": hidden,
        "gate_rank": rank,
        "n_ctrl_families": families,
        "n_phys": n_phys,
        "n_passes": passes,
        "core_start": start,
        "core_end": end,
        "attn_res_block": block,
        "attn_tail_rows": base["attn_tail_rows"],
        "down": 2 * hidden,
        "msg_rows": families,
        "attn_tail_layers": tail,
        "core_layers": list(range(start, end)),
        "n_slot": n_phys + (passes - 1) * (end - start),
        "expected_tensors": families * (end - start) * 3 + (end - start) + (n_phys - 1) + n_phys + 1,
    }


EXPECTED_DIMENSIONS = derive_dimensions(BASE_DIMENSIONS)
FAMILIES = ("q", "k", "v", "ffn")


def contract_from_dimensions(dims):
    """Build the compact per-family shape contract from graph dimensions."""
    hidden = dims["hidden"]
    rank = dims["gate_rank"]
    down = dims["down"]
    core = list(dims["core_layers"])
    tail = list(dims["attn_tail_layers"])
    return {
        "loop_msg_query": {
            "template": "blk.{bid}.loop_msg_query.weight",
            "layers": core,
            "shape": [hidden, dims["msg_rows"]],
            "dtype": "F32",
            "count": len(core),
        },
        "ctrl_down": {
            "template": "blk.{bid}.loop_{family}_ctrl_down.weight",
            "families": list(FAMILIES),
            "layers": core,
            "shape": [down, rank],
            "dtype": "F32",
            "count": len(FAMILIES) * len(core),
        },
        "ctrl_gain": {
            "template": "blk.{bid}.loop_{family}_ctrl_gain.weight",
            "families": list(FAMILIES),
            "layers": core,
            "shape": [rank, hidden],
            "dtype": "F32",
            "count": len(FAMILIES) * len(core),
        },
        "ctrl_forget": {
            "template": "blk.{bid}.loop_{family}_ctrl_forget.weight",
            "families": list(FAMILIES),
            "layers": core,
            "shape": [rank, hidden],
            "dtype": "F32",
            "count": len(FAMILIES) * len(core),
        },
        "attn_res_query": {
            "template": "blk.{bid}.attn_res_query.weight",
            "layers": list(range(1, dims["n_phys"])),
            "shape": [hidden],
            "special_layers": tail,
            "special_shape": [hidden, dims["attn_tail_rows"]],
            "dtype": "F32",
            "count": dims["n_phys"] - 1,
        },
        "ffn_res_query": {
            "template": "blk.{bid}.ffn_res_query.weight",
            "layers": list(range(dims["n_phys"])),
            "shape": [hidden],
            "dtype": "F32",
            "count": dims["n_phys"],
        },
        "output_res_query": {
            "template": "output_res_query.weight",
            "shape": [hidden],
            "dtype": "F32",
            "count": 1,
        },
    }


EXPECTED_CONTRACT = contract_from_dimensions(EXPECTED_DIMENSIONS)


def expand_contract(contract):
    """Expand the compact contract into an explicit name -> shape/dtype map."""
    if not isinstance(contract, dict):
        raise ValueError("contract must be an object")
    expanded = {}
    for spec in contract.values():
        template = spec["template"]
        layers = spec.get("layers", [])
        families = spec.get("families", [""])
        shape = spec["shape"]
        dtype = spec["dtype"]
        special = set(spec.get("special_layers", []))
        special_shape = spec.get("special_shape")
        count = spec["count"]
        made = 0
        if "{bid}" not in template and "{family}" not in template:
            if layers or len(families) != 1 or count != 1:
                raise ValueError("untemplated tensor must be a single named tensor")
            expanded[template] = {"shape": list(shape), "dtype": dtype}
            made = 1
        for il in layers:
            for family in families:
                name = template.replace("{bid}", str(il)).replace("{family}", family)
                if name in expanded:
                    raise ValueError("duplicate contract tensor " + name)
                chosen = special_shape if il in special else shape
                expanded[name] = {"shape": list(chosen), "dtype": dtype}
                made += 1
        if made != count:
            raise ValueError("contract count does not match expanded names")
    return expanded


REFERENCE_MAP = expand_contract(EXPECTED_CONTRACT)
REFERENCE_NAMES = sorted(REFERENCE_MAP)
EXPECTED_OBSERVED_NAMES = REFERENCE_NAMES
EXPECTED_SUMMARY = {
    "expected_tensors": 204,
    "loop_msg_query": 12,
    "ctrl": 144,
    "attn_res_query": 23,
    "ffn_res_query": 24,
    "output_res_query": 1,
    "other_matching_tensors": 0,
}

MEASURED_SHAPE_COUNTS = {
    "[2304]|F32": 44,
    "[2304,3]|F32": 4,
    "[2304,4]|F32": 12,
    "[4608,64]|F32": 48,
    "[64,2304]|F32": 96,
}

EXPECTED_PROVENANCE = {
    "url": "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-IQ1_S.gguf",
    "range": "bytes=0-16777215",
    "byte_start": 0,
    "byte_end": 16777215,
    "fetched_bytes": 16777216,
    "first_16mib_sha256": GGUF_SHA256,
    "magic": "GGUF",
    "version": 3,
    "n_tensors": 744,
    "n_kv": 63,
    "header_source_file": "/tmp/opencode/loopkit-m37/gguf_first16.gguf",
    "parsed_tensors_file": "/tmp/opencode/loopkit-m37/tensors.json",
    "parsed_tensors_sha256": "083d0c1a3ac4aa3e14a9ac818e90f495ab32890ed2d361443cf6b6bd44062a23",
    "network_access_in_checker": False,
    "note": "The first 16 MiB were fetched once out of band; this checker makes no network calls and reads no cache file.",
}


def needles(name, source, items):
    return {"name": name, "source": source, "needles": list(items)}


NEEDLE_SPECS = [
    needles("gguf_tensor_names", CONSTANTS_PY, [
        '"blk.{bid}.loop_msg_query"',
        '"blk.{bid}.loop_q_ctrl_down"',
        '"blk.{bid}.loop_ffn_ctrl_forget"',
        '"blk.{bid}.attn_res_query"',
        '"blk.{bid}.ffn_res_query"',
        '"output_res_query"',
    ]),
    needles("gguf_loop_config_keys", CONSTANTS_PY, [
        '"{arch}.loop.start"',
        '"{arch}.loop.end"',
        '"{arch}.loop.count"',
        '"{arch}.loop.gate_rank"',
        '"{arch}.loop.varying"',
    ]),
    needles("loop_tensor_shapes", LOOP_CPP, [
        'create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);',
        'create_tensor(tn(down[c],   "weight", il), {2*n_embd, rank}, 0);',
        'create_tensor(tn(gain[c],   "weight", il), {rank, n_embd}, 0);',
        'create_tensor(tn(forget[c], "weight", il), {rank, n_embd}, 0);',
    ]),
    needles("loop_config_keys", LOOP_CPP, [
        "ml.get_key(LLM_KV_LOOP_START,     core_start);",
        "ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);",
        "const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);",
    ]),
    needles("res_query_shapes", INKLING_CPP, [
        'output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);',
        'layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);',
        'layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);',
        "ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);",
    ]),
    needles("converter_loop_config", CONVERTER_PY, [
        'self.loop = self.hparams["loop_config"]',
        'if set(self.loop) != {"core_start", "core_end", "num_passes", "gate_rank", "is_varloop"}:',
        'self.gguf_writer.add_loop_gate_rank(self.loop["gate_rank"])',
        'if new_name.endswith(("loop_msg_query.weight", "_ctrl_down.weight", "_ctrl_gain.weight", "_ctrl_forget.weight")):',
    ]),
]

NEEDLE_CHECK_NAMES = ["needle_" + spec["name"] for spec in NEEDLE_SPECS]
CHECK_NAMES = [
    "frozen_facts",
    "frozen_provenance",
    "frozen_dimensions",
    "frozen_contract",
    "frozen_sources",
    "frozen_needles",
    "frozen_observed_names",
    "live_source_sha256",
    "derived_dimensions",
    "shape_consistency",
    "contract_coverage",
    "observed_names",
    "summary_counts",
    "m33_gguf_sha256",
    "no_network",
    "run_completed",
] + NEEDLE_CHECK_NAMES

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


def read_text(path):
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def fail_closed_summary():
    """Startup sentinel: never leave a stale PASS if the run is killed early."""
    return {
        "verdict": "NO_GO",
        "blocker": "checker did not complete",
        "checks": {name: False for name in CHECK_NAMES},
        "check_counts": {"total": len(CHECK_NAMES), "passed": 0, "failed": len(CHECK_NAMES)},
        "info": ["NO_GO sentinel written at checker startup"],
        "provenance": {},
        "dimensions": {},
        "note": NOTE,
        "scope": NOTE,
        "speed_claim": None,
        "no_weights": True,
        "no_engine_run": True,
        "frozen_on": None,
        "frozen_facts_sha256": None,
        "frozen_source_sha256": FROZEN_SOURCES,
        "doc_note": DOC_NOTE,
        "sentinel": True,
    }


def check_frozen_facts(facts):
    if not isinstance(facts, dict):
        return False
    expected_keys = {
        "schema_version", "frozen_on", "scope", "note", "description", "provenance",
        "dimensions", "contract", "sources", "needles", "observed_names", "summary",
        "measured_shape_counts", "speed_claim",
    }
    if set(facts) != expected_keys:
        return False
    return bool(
        facts.get("schema_version") == SCHEMA_VERSION
        and facts.get("frozen_on") == FROZEN_ON
        and facts.get("scope") == NOTE
        and facts.get("note") == NOTE
        and facts.get("speed_claim") is None
        and facts.get("description") == DESCRIPTION
        and facts.get("provenance") == EXPECTED_PROVENANCE
        and facts.get("dimensions") == EXPECTED_DIMENSIONS
        and facts.get("contract") == EXPECTED_CONTRACT
        and facts.get("sources") == FROZEN_SOURCES
        and facts.get("needles") == NEEDLE_SPECS
        and facts.get("observed_names") == EXPECTED_OBSERVED_NAMES
        and facts.get("summary") == EXPECTED_SUMMARY
        and facts.get("measured_shape_counts") == MEASURED_SHAPE_COUNTS
        and measured_counts_from_contract(facts.get("contract")) == MEASURED_SHAPE_COUNTS
    )


def check_live_source_sha256(root):
    for rel, digest in FROZEN_SOURCES.items():
        path = root / rel
        if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
            return False
    return True


def check_derived_dimensions(facts):
    try:
        derived = derive_dimensions(BASE_DIMENSIONS)
    except ValueError:
        return False
    return bool(
        derived == EXPECTED_DIMENSIONS
        and facts.get("dimensions") == derived
        and derived["down"] == 2 * derived["hidden"] == 4608
        and derived["msg_rows"] == derived["n_ctrl_families"] == 4
        and derived["attn_tail_rows"] == 3
        and derived["attn_tail_layers"] == [5, 11, 17, 23]
        and derived["core_layers"] == list(range(6, 18))
        and derived["n_phys"] == 24
        and derived["n_slot"] == 48 == 24 + 2 * (18 - 6)
        and derived["expected_tensors"] == 204
    )


def check_shape_consistency(facts):
    try:
        derived = contract_from_dimensions(facts["dimensions"])
    except (KeyError, TypeError, ValueError):
        return False
    return derived == facts.get("contract")


def check_contract_coverage(facts):
    try:
        expanded = expand_contract(facts["contract"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return expanded == REFERENCE_MAP and len(expanded) == 204


def check_observed_names(facts):
    names = facts.get("observed_names")
    return bool(
        isinstance(names, list)
        and len(names) == 204
        and len(set(names)) == 204
        and names == sorted(names)
        and names == REFERENCE_NAMES
    )


def check_summary_counts(facts):
    summary = facts.get("summary")
    if not isinstance(summary, dict):
        return False
    counts = {"loop_msg_query": 0, "ctrl": 0, "attn_res_query": 0, "ffn_res_query": 0, "output_res_query": 0}
    for name in REFERENCE_MAP:
        if "loop_msg_query" in name:
            counts["loop_msg_query"] += 1
        elif "_ctrl_" in name:
            counts["ctrl"] += 1
        elif "attn_res_query" in name:
            counts["attn_res_query"] += 1
        elif "ffn_res_query" in name:
            counts["ffn_res_query"] += 1
        elif "output_res_query" in name:
            counts["output_res_query"] += 1
    expected = dict(counts)
    expected["expected_tensors"] = 204
    expected["other_matching_tensors"] = 0
    return summary == expected


def measured_counts_from_contract(contract):
    """Shape/dtype histogram of the contract, compared with the frozen measurement."""
    expanded = expand_contract(contract)
    counts = {}
    for entry in expanded.values():
        key = json.dumps(entry["shape"], separators=(",", ":")) + "|" + entry["dtype"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def check_m33_sha256(root):
    path = root / M33_REL
    if not path.is_file():
        return False
    payload = load_json(path)
    frozen = payload["facts"]["header"]["first_16mib_sha256"]
    probe = payload["provenance"]["header_probe"]["first_16mib_sha256"]
    return bool(frozen == GGUF_SHA256 and probe == GGUF_SHA256)


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
    dimensions = {}
    facts_sha256 = None
    summary_path = root / SUMMARY_REL
    sentinel_ok = True
    try:
        write_json(summary_path, fail_closed_summary())
    except Exception as exc:
        sentinel_ok = False
        info.append("sentinel_write_failed: %s: %s" % (type(exc).__name__, exc))
    try:
        facts_path = root / FACTS_REL
        facts_sha256 = sha256_bytes(facts_path.read_bytes())
        facts = load_json(facts_path)
        provenance = dict(facts.get("provenance") or {})
        dimensions = dict(facts.get("dimensions") or {})
        checks["frozen_facts"] = check_frozen_facts(facts)
        checks["frozen_provenance"] = facts.get("provenance") == EXPECTED_PROVENANCE
        checks["frozen_dimensions"] = facts.get("dimensions") == EXPECTED_DIMENSIONS
        checks["frozen_contract"] = facts.get("contract") == EXPECTED_CONTRACT
        checks["frozen_sources"] = facts.get("sources") == FROZEN_SOURCES
        checks["frozen_needles"] = facts.get("needles") == NEEDLE_SPECS
        checks["frozen_observed_names"] = facts.get("observed_names") == EXPECTED_OBSERVED_NAMES
        checks["live_source_sha256"] = check_live_source_sha256(root)
        checks["derived_dimensions"] = check_derived_dimensions(facts)
        checks["shape_consistency"] = check_shape_consistency(facts)
        checks["contract_coverage"] = check_contract_coverage(facts)
        checks["observed_names"] = check_observed_names(facts)
        checks["summary_counts"] = check_summary_counts(facts)
        checks["m33_gguf_sha256"] = check_m33_sha256(root)
        checks["no_network"] = check_no_network()
        texts = {rel: read_text(root / rel) for rel in FROZEN_SOURCES}
        for spec in NEEDLE_SPECS:
            text = texts.get(spec["source"], "")
            checks["needle_" + spec["name"]] = bool(text) and all(
                needle in text for needle in spec["needles"]
            )
        provenance["source_sha256_live"] = {
            rel: sha256_bytes((root / rel).read_bytes()) if (root / rel).is_file() else "missing"
            for rel in FROZEN_SOURCES
        }
        info.append(
            "evidence: first 16 MiB of the GGUF, sha256 %s, %d tensors, %d KVs; checker makes no network calls"
            % (GGUF_SHA256, EXPECTED_PROVENANCE["n_tensors"], EXPECTED_PROVENANCE["n_kv"])
        )
        info.append(
            "derived dims: hidden %d, gate_rank %d, down %d = 2x hidden, msg rows %d, tail rows %d at %s"
            % (EXPECTED_DIMENSIONS["hidden"], EXPECTED_DIMENSIONS["gate_rank"],
               EXPECTED_DIMENSIONS["down"], EXPECTED_DIMENSIONS["msg_rows"],
               EXPECTED_DIMENSIONS["attn_tail_rows"], EXPECTED_DIMENSIONS["attn_tail_layers"])
        )
        info.append(
            "slots: n_phys %d, core [%d,%d), n_slot %d = %d + %d*(%d-%d)"
            % (EXPECTED_DIMENSIONS["n_phys"], EXPECTED_DIMENSIONS["core_start"],
               EXPECTED_DIMENSIONS["core_end"], EXPECTED_DIMENSIONS["n_slot"],
               EXPECTED_DIMENSIONS["n_phys"], EXPECTED_DIMENSIONS["n_passes"] - 1,
               EXPECTED_DIMENSIONS["core_end"], EXPECTED_DIMENSIONS["core_start"])
        )
        info.append("contract: %d distinct tensors; every name is shape/dtype bound by the frozen compact mapping" % len(REFERENCE_MAP))
        info.append("M33 QUANT_LAYOUT.json first_16mib_sha256 matches the frozen GGUF sha256")
        info.append(DOC_NOTE)
        info.append("frozen facts sha256 %s (INFO only; individual fields are gated)" % facts_sha256)
        checks["run_completed"] = True
    except Exception as exc:
        info.append("exception: %s: %s" % (type(exc).__name__, exc))

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
        "dimensions": dimensions,
        "note": NOTE,
        "scope": NOTE,
        "speed_claim": None,
        "no_weights": True,
        "no_engine_run": True,
        "frozen_on": FROZEN_ON,
        "frozen_facts_sha256": facts_sha256,
        "frozen_source_sha256": FROZEN_SOURCES,
        "doc_note": DOC_NOTE,
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
    print("doc: " + summary["doc_note"])


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
