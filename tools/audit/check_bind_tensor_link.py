#!/usr/bin/env python3
"""Offline, fail-closed CPU-only per-event binding-to-tensor linkage gate.

Drives an already built CPU-only llama-debug twice on the deterministic
synthetic Inkling2-loop GGUF: OFF (LLAMA_RORET unset) and ON (LLAMA_RORET=1,
LLAMA_RORET_TRACE=1, LLAMA_DUMP_TENSOR_PTR=1). The ON run writes a separate
manifest_ptr.tsv alongside manifest.tsv. The gate proves that every dumped
ffn_moe_slots-<il> tensor pointer is literally a RORET_BIND set lookup pointer,
and that the pass-to-slot map holds for each such event. Graph builds whose
lookup has no dumped tensor are counted but never claimed. No timing, no speed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CONTRACT_REL = "tools/audit/BIND_TENSOR_LINK.json"
SUMMARY_REL = "tools/audit/M42_BIND_TENSOR_LINK_SUMMARY.json"
DEFAULT_CLI = Path("/tmp/opencode/m41-probe/bin/llama-debug")
DEFAULT_GGUF = Path("/tmp/opencode/synthetic.gguf")
DEFAULT_OUT_DIR = Path("/tmp/opencode/m42-real")

SCHEMA_VERSION = 1
FROZEN_ON = "2026-11-28"
NOTE = "CPU-only per-event RORET binding pointer identity with dumped MoE slots tensors; no timing"
SCOPE = (
    "Per-event pointer identity between the registered RORET_BIND set lookup "
    "and the dumped ffn_moe_slots-<il> tensor, including the pass-to-slot map; "
    "NOT tensor-value correctness, callback resolution, or speed."
)
ENGINE_RUN = "cpu_synthetic_only"
NO_WEIGHTS_NOTE = (
    "No real weights and no speed claim; the engine runs only on a fixed-seed "
    "synthetic GGUF. This is not a no-engine-run gate."
)
CYCLE_AUDIT_REQUIRED = (
    "NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains "
    "mandatory for any future number."
)
FIXTURE_SHA256 = "cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617"
FIXTURE_BYTES = 887392
EXPECTED_ARCH = "inkling2_loop"
LOOP_CONSTANTS = {
    "N_LAYER": 2,
    "CORE_START": 1,
    "CORE_END": 2,
    "N_PASSES": 3,
    "GATE_RANK": 8,
}
TENSOR_KINDS = ("logits", "probs", "topk", "weights", "out", "slots")
EXPECTED_OCCURRENCES = 2
EXPECTED_KEY_IL = 1
EXPECTED_BIND_KEY_IL = 1
EXPECTED_BIND_SENTINEL = {"key_il": -1, "pass": -1}
EXPECTED_SET_ROWS_PER_PASS = 12
EXPECTED_SENTINEL_ROWS = 12
EXPECTED_SLOT_DUMPS_PER_PASS = 2
EXPECTED_SENTINEL_SLOT_DUMPS = 2
EXPECTED_MATCHED_SET_ROWS_PER_PASS = 2
EXPECTED_MATCHED_POINTERS_PER_PASS = 1
EXPECTED_ENABLE_NEEDLES = ["RORET union-resident MoE bookkeeping enabled"]
INFORMATIONAL_BUILD_MARKERS = [
    "LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache",
]
EXPECTED_TRACE_NEEDLES = ["RORET_BIND set"]
EXPECTED_OFF_ABSENT_NEEDLES = [
    "RORET_BIND",
    "RORET_PASS",
    "RORET union-resident MoE bookkeeping enabled",
]
EXPECTED_MANIFEST_FIELDS = 12
POINTER_MANIFEST_FILENAME = "manifest_ptr.tsv"
EXPECTED_POINTER_FIELDS = 2
EXPECTED_POINTER_FIELD_ORDER = ["filename", "ptr"]
CPU_SHARE = "1.0"
TENSOR_FILTER = r"^ffn_moe_(logits|probs|topk|weights|out|slots)-[0-9]+$"
CLI_FLAGS = [
    "-p", "hello world", "-n", "4", "-cmoe",
    "--moe-cache-mib", "16", "--moe-cache-cpu-share", CPU_SHARE,
    "--verbose", "--tensor-filter", TENSOR_FILTER,
]
TIMEOUT_SECONDS = 300

NETWORK_MODULES = {
    "socket", "http", "urllib", "requests", "httpx", "ftplib", "smtplib",
    "telnetlib", "xmlrpc",
}
NETWORK_CALLS = {"urlopen", "urlretrieve", "socket", "connect"}

CHECK_NAMES = [
    "contract_json",
    "contract_keys",
    "contract_values",
    "checker_sha256",
    "no_network",
    "cli_exists",
    "cli_executable",
    "fixture_exists",
    "fixture_sha256",
    "fixture_bytes",
    "fixture_pin_authority",
    "fixture_arch",
    "off_exit_zero",
    "on_exit_zero",
    "off_absent_needles",
    "on_enable_needles",
    "on_trace_needles",
    "off_manifest_ok",
    "on_manifest_ok",
    "off_name_set",
    "on_name_set",
    "off_slot_set",
    "on_slot_set",
    "off_occurrences",
    "on_occurrences",
    "bind_parse_ok",
    "bind_pass_set",
    "bind_counts_equal_positive",
    "bind_sentinel_count",
    "off_no_bind_no_ptr",
    "off_ptr_manifest_absent",
    "on_ptr_manifest_ok",
    "ptr_manifest_coverage",
    "slot_dump_names",
    "slot_dump_occurrences",
    "bind_slot_identity",
    "bind_slot_no_wrong_il",
    "slot_ptrs_all_bound",
    "sentinel_slot_ptr_bound",
    "slot_ptrs_disjoint_per_pass",
    "matched_counts_equal_per_pass",
    "matched_basename_unique_per_pass",
    "bin_parity",
    "stdout_parity",
    "run_completed",
]

EXPECTED_TOP_KEYS = frozenset({
    "schema_version", "frozen_on", "scope", "note", "speed_claim",
    "no_weights", "no_engine_run", "no_weights_no_engine_run_note",
    "engine_run", "cycle_audit_required", "expected_fixture_sha256",
    "expected_fixture_bytes", "expected_arch", "loop_constants",
    "expected_passes", "expected_core_slots", "expected_il_slots",
    "pass_to_slot", "expected_tensor_kinds", "expected_tensor_names",
    "expected_occurrences_per_name", "expected_enable_needles",
    "informational_build_markers", "expected_trace_needles",
    "expected_off_absent_needles", "expected_manifest_fields",
    "pointer_manifest_filename", "expected_pointer_fields",
    "expected_pointer_field_order", "expected_slot_tensor_kind",
    "expected_slot_tensor_names", "expected_slot_basename_per_pass",
    "expected_bind_key_il", "expected_bind_sentinel",
    "expected_set_rows_per_pass", "expected_sentinel_rows",
    "expected_slot_dumps_per_pass", "expected_sentinel_slot_dumps",
    "expected_matched_set_rows_per_pass",
    "expected_matched_pointers_per_pass", "expected_summary_path",
    "checker_sha256", "verify",
})

NAME_RX = re.compile(r"^ffn_moe_(logits|probs|topk|weights|out|slots)-(\d+)$")
OCC_RX = re.compile(r"__(\d+)\.bin$")
SET_LINE_RX = re.compile(r"RORET_BIND\s+set\b")
SET_RX = re.compile(
    r"RORET_BIND\s+set\b.*?\blookup=(0x[0-9a-fA-F]+)\s+"
    r"slot_map=(0x[0-9a-fA-F]+)\s+key_il=(-?\d+)\s+pass=(-?\d+)"
)
PTR_RX = re.compile(r"^0x[0-9a-fA-F]+$")


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def derive_pass_to_slot(constants):
    """Derive pass -> physical slot from frozen fixture loop constants."""
    required = ("N_LAYER", "CORE_START", "CORE_END", "N_PASSES", "GATE_RANK")
    if not isinstance(constants, dict) or set(constants) != set(required):
        raise ValueError("loop constants must have exactly the frozen key set")
    for key in required:
        if not is_int(constants[key]) or constants[key] <= 0:
            raise ValueError("loop constant %s must be a positive integer" % key)
    start = constants["CORE_START"]
    end = constants["CORE_END"]
    passes = constants["N_PASSES"]
    if not 0 < start < end <= constants["N_LAYER"]:
        raise ValueError("loop core must lie inside the physical layers")
    if end - start != 1:
        raise ValueError("this fixture has exactly one logical core layer")
    if passes < 2:
        raise ValueError("the loop needs at least two passes")
    core_slots = [start + p * (end - start) for p in range(passes)]
    return {
        "pass_to_slot": {str(p): slot for p, slot in enumerate(core_slots)},
        "core_slots": core_slots,
    }


DERIVED = derive_pass_to_slot(LOOP_CONSTANTS)
PASS_TO_SLOT = DERIVED["pass_to_slot"]
CORE_SLOTS = DERIVED["core_slots"]
IL_SLOTS = [0] + CORE_SLOTS
EXPECTED_PASSES = [int(p) for p in PASS_TO_SLOT]
EXPECTED_NAMES = sorted(
    "ffn_moe_%s-%d" % (kind, il) for il in IL_SLOTS for kind in TENSOR_KINDS
)
EXPECTED_SLOT_NAMES = sorted("ffn_moe_slots-%d" % il for il in IL_SLOTS)
EXPECTED_SLOT_BASENAME_PER_PASS = {
    str(p): "ffn_moe_slots-%d" % PASS_TO_SLOT[str(p)] for p in EXPECTED_PASSES
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value
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
    """Startup sentinel: never leave a stale PASS if the run stops early."""
    return {
        "verdict": "NO_GO",
        "blocker": ["checker did not complete"],
        "checks": {name: False for name in CHECK_NAMES},
        "check_counts": {
            "total": len(CHECK_NAMES), "passed": 0, "failed": len(CHECK_NAMES),
        },
        "info": ["NO_GO sentinel written at checker startup"],
        "note": NOTE,
        "speed_claim": None,
        "no_weights": True,
        "no_engine_run": False,
        "engine_run": ENGINE_RUN,
        "frozen_on": FROZEN_ON,
        "checker_sha256": None,
        "contract_path": CONTRACT_REL,
        "expected_summary_path": SUMMARY_REL,
        "sentinel": True,
    }


def check_contract_keys(facts):
    return isinstance(facts, dict) and set(facts) == EXPECTED_TOP_KEYS


def check_contract_values(facts):
    """The whole contract is frozen; any drift, including an extra key, fails closed."""
    if not isinstance(facts, dict):
        return False
    try:
        derived = derive_pass_to_slot(facts["loop_constants"])
        names = facts["expected_tensor_names"]
        kinds = facts["expected_tensor_kinds"]
        passes = facts["expected_passes"]
        core = facts["expected_core_slots"]
        il_slots = facts["expected_il_slots"]
        slot_names = facts["expected_slot_tensor_names"]
        slot_basenames = facts["expected_slot_basename_per_pass"]
        enable = facts["expected_enable_needles"]
        build = facts["informational_build_markers"]
        trace = facts["expected_trace_needles"]
        absent = facts["expected_off_absent_needles"]
        verify = facts["verify"]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        facts.get("schema_version") == SCHEMA_VERSION
        and facts.get("frozen_on") == FROZEN_ON
        and facts.get("scope") == SCOPE
        and facts.get("note") == NOTE
        and facts.get("speed_claim") is None
        and facts.get("no_weights") is True
        and facts.get("no_engine_run") is False
        and facts.get("no_weights_no_engine_run_note") == NO_WEIGHTS_NOTE
        and facts.get("engine_run") == ENGINE_RUN
        and facts.get("cycle_audit_required") == CYCLE_AUDIT_REQUIRED
        and is_sha256(facts.get("expected_fixture_sha256"))
        and facts["expected_fixture_sha256"] == FIXTURE_SHA256
        and is_int(facts.get("expected_fixture_bytes"))
        and facts["expected_fixture_bytes"] == FIXTURE_BYTES
        and facts.get("expected_arch") == EXPECTED_ARCH
        and facts.get("loop_constants") == LOOP_CONSTANTS
        and passes == EXPECTED_PASSES
        and all(is_int(value) for value in passes)
        and core == CORE_SLOTS
        and il_slots == IL_SLOTS
        and facts.get("pass_to_slot") == PASS_TO_SLOT
        and derived["pass_to_slot"] == PASS_TO_SLOT
        and derived["core_slots"] == CORE_SLOTS
        and kinds == list(TENSOR_KINDS)
        and names == EXPECTED_NAMES
        and facts.get("expected_occurrences_per_name") == EXPECTED_OCCURRENCES
        and facts.get("pointer_manifest_filename") == POINTER_MANIFEST_FILENAME
        and facts.get("expected_pointer_fields") == EXPECTED_POINTER_FIELDS
        and facts.get("expected_pointer_field_order") == EXPECTED_POINTER_FIELD_ORDER
        and facts.get("expected_slot_tensor_kind") == "slots"
        and slot_names == EXPECTED_SLOT_NAMES
        and slot_basenames == EXPECTED_SLOT_BASENAME_PER_PASS
        and facts.get("expected_bind_key_il") == EXPECTED_BIND_KEY_IL
        and facts.get("expected_bind_sentinel") == EXPECTED_BIND_SENTINEL
        and facts.get("expected_set_rows_per_pass") == EXPECTED_SET_ROWS_PER_PASS
        and facts.get("expected_sentinel_rows") == EXPECTED_SENTINEL_ROWS
        and facts.get("expected_slot_dumps_per_pass") == EXPECTED_SLOT_DUMPS_PER_PASS
        and facts.get("expected_sentinel_slot_dumps") == EXPECTED_SENTINEL_SLOT_DUMPS
        and facts.get("expected_matched_set_rows_per_pass") == EXPECTED_MATCHED_SET_ROWS_PER_PASS
        and facts.get("expected_matched_pointers_per_pass") == EXPECTED_MATCHED_POINTERS_PER_PASS
        and enable == EXPECTED_ENABLE_NEEDLES
        and build == INFORMATIONAL_BUILD_MARKERS
        and not set(build) & set(enable)
        and trace == EXPECTED_TRACE_NEEDLES
        and absent == EXPECTED_OFF_ABSENT_NEEDLES
        and facts.get("expected_manifest_fields") == EXPECTED_MANIFEST_FIELDS
        and facts.get("expected_summary_path") == SUMMARY_REL
        and is_sha256(facts.get("checker_sha256"))
        and isinstance(verify, list)
        and verify == CHECK_NAMES
    )


def check_checker_sha256(facts, live_sha256):
    """The pin lives in the contract, so the checker source carries no self-hash."""
    return bool(
        isinstance(facts, dict)
        and is_sha256(live_sha256)
        and is_sha256(facts.get("checker_sha256"))
        and facts["checker_sha256"] == live_sha256
    )


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


def parse_tensor_name(name):
    """Return (kind, il) for a dumped MoE tensor name, else None."""
    match = NAME_RX.match(name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def parse_manifest(path, problems):
    """Parse a 12-field manifest.tsv; report malformed or duplicate rows."""
    rows = []
    if not path.is_file():
        problems.append("missing manifest: " + str(path))
        return rows
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        problems.append("manifest read failed: %s: %s" % (type(exc).__name__, exc))
        return rows
    seen = set()
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != EXPECTED_MANIFEST_FIELDS:
            problems.append("bad manifest row %d" % lineno)
            continue
        name = fields[0]
        filename = fields[11]
        occ_match = OCC_RX.search(filename)
        if occ_match is None:
            problems.append("bad manifest filename %d" % lineno)
            continue
        key = (name, filename)
        if key in seen:
            problems.append("duplicate manifest row %d" % lineno)
            continue
        seen.add(key)
        rows.append({
            "line": lineno,
            "name": name,
            "filename": filename,
            "occurrence": int(occ_match.group(1)),
            "basename": filename[:occ_match.start()],
        })
    return rows


def parse_pointer_manifest(path, problems):
    """Parse manifest_ptr.tsv into filename -> normalized pointer."""
    pointers = {}
    if not path.is_file():
        return pointers
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        problems.append("pointer manifest read failed: %s: %s" % (type(exc).__name__, exc))
        return pointers
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != EXPECTED_POINTER_FIELDS:
            problems.append("bad pointer row %d" % lineno)
            continue
        filename, address = fields
        if OCC_RX.search(filename) is None or PTR_RX.match(address) is None:
            problems.append("bad pointer value %d" % lineno)
            continue
        if address.lower() == "0x0":
            problems.append("null pointer %d" % lineno)
            continue
        if filename in pointers:
            problems.append("duplicate pointer row %d" % lineno)
            continue
        pointers[filename] = address.lower()
    return pointers


def summarize_rows(rows):
    """Summarize name set, il slot set and per-name occurrence counts."""
    names = [row["name"] for row in rows]
    slots = set()
    for name in names:
        parsed = parse_tensor_name(name)
        if parsed is not None:
            slots.add(parsed[1])
    occurrences = {}
    for name in names:
        occurrences[name] = occurrences.get(name, 0) + 1
    return {
        "names": sorted(set(names)),
        "slots": sorted(slots),
        "occurrences": occurrences,
        "row_count": len(rows),
    }


def load_dump(dump_dir, problems):
    """Parse a phase dump into rows, pointer rows and raw .bin blobs."""
    rows = parse_manifest(dump_dir / "manifest.tsv", problems)
    pointers = parse_pointer_manifest(dump_dir / POINTER_MANIFEST_FILENAME, problems)
    blobs = {}
    for row in rows:
        path = dump_dir / row["filename"]
        if not path.is_file():
            problems.append("missing bin: " + row["filename"])
            continue
        try:
            blobs[row["filename"]] = path.read_bytes()
        except OSError as exc:
            problems.append("bin read failed: %s: %s" % (type(exc).__name__, exc))
    return rows, pointers, blobs


def parse_bindings(text):
    """Extract full RORET_BIND set rows from a phase log."""
    return [
        {
            "index": index,
            "lookup": address.lower(),
            "slot_map": slot_map.lower(),
            "key_il": int(il),
            "pass": int(pass_),
            "sentinel": int(il) == -1 and int(pass_) == -1,
        }
        for index, (address, slot_map, il, pass_) in enumerate(SET_RX.findall(text))
    ]


def bind_counts(bindings, key_il=EXPECTED_BIND_KEY_IL):
    """Count RORET_BIND set rows per pass for one key_il value."""
    counts = {}
    for row in bindings:
        if not row["sentinel"] and row["key_il"] == key_il:
            counts[row["pass"]] = counts.get(row["pass"], 0) + 1
    return counts


def stdout_differing_bytes(left, right):
    """Count differing byte positions; never inspects or records timing values."""
    if left == right:
        return 0
    return sum(1 for a, b in zip(left, right) if a != b) + abs(len(left) - len(right))


def run_phase(phase, cli, gguf, out_dir, summary_path, on):
    """Run one engine phase; capture stdout and stderr separately, never time it."""
    env = os.environ.copy()
    env.pop("LLAMA_RORET", None)
    env.pop("LLAMA_RORET_TRACE", None)
    env.pop("LLAMA_DUMP_TENSOR_PTR", None)
    if on:
        env["LLAMA_RORET"] = "1"
        env["LLAMA_RORET_TRACE"] = "1"
        env["LLAMA_DUMP_TENSOR_PTR"] = "1"
    env["M42_SUMMARY_PATH"] = str(summary_path)
    dump_dir = out_dir / phase
    try:
        if dump_dir.exists():
            shutil.rmtree(dump_dir)
    except OSError:
        pass
    env["LLAMA_DUMP_TENSORS_DIR"] = str(dump_dir)
    cmd = [str(cli), "-m", str(gguf), *CLI_FLAGS]
    stdout_path = out_dir / (phase + "_stdout.log")
    stderr_path = out_dir / (phase + "_stderr.log")
    result = {
        "exit": None,
        "stdout": b"",
        "stderr": b"",
        "dump_dir": dump_dir,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }
    try:
        with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
            proc = subprocess.run(cmd, stdout=so, stderr=se, env=env, timeout=TIMEOUT_SECONDS)
        result["exit"] = proc.returncode
    except subprocess.TimeoutExpired:
        result["exit"] = None
    except OSError as exc:
        result["exit"] = None
        result["stderr"] = ("failed to start: %s: %s" % (type(exc).__name__, exc)).encode("utf-8")
    for key, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        try:
            result[key] = path.read_bytes()
        except OSError:
            result[key] = b""
    return result


def run_check(root, cli, gguf, out_dir):
    root = Path(root)
    summary_path = root / SUMMARY_REL
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = []
    sentinel_ok = True

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

    checks = {name: False for name in CHECK_NAMES}
    facts = {}
    checker_sha256 = None
    fixture_sha256 = None
    fixture_bytes = None
    off = {"exit": None, "stdout": b"", "stderr": b"", "dump_dir": out_dir / "off"}
    on = {"exit": None, "stdout": b"", "stderr": b"", "dump_dir": out_dir / "on"}
    off_summary = {"names": [], "slots": [], "occurrences": {}, "row_count": 0}
    on_summary = {"names": [], "slots": [], "occurrences": {}, "row_count": 0}
    off_blobs = {}
    on_blobs = {}
    off_pointers = {}
    on_pointers = {}
    off_problems = []
    on_problems = []
    bindings_on = []
    bind_passes = []
    bind_counts_on = {}
    sentinel_bind_count = 0
    bind_lines_off = 0
    slot_events = 0
    sentinel_slot_events = 0
    slot_dumps_per_pass = {}
    matched_rows_per_pass = {}
    matched_pointers_per_pass = {}
    unmatched_set_rows = 0
    bin_files_compared = 0
    bin_bytes_compared = 0
    bin_differing = 0
    stdout_bytes = 0
    stdout_differing = 0
    try:
        checker_sha256 = sha256_bytes(Path(__file__).read_bytes())
        try:
            facts = load_json(root / CONTRACT_REL)
            checks["contract_json"] = isinstance(facts, dict)
        except (OSError, ValueError) as exc:
            info.append("contract_load_failed: %s: %s" % (type(exc).__name__, exc))
        checks["contract_keys"] = check_contract_keys(facts)
        checks["contract_values"] = check_contract_values(facts)
        checks["checker_sha256"] = check_checker_sha256(facts, checker_sha256)
        checks["no_network"] = check_no_network()
        info.append("checker sha256 frozen=%s live=%s" % (
            facts.get("checker_sha256"), checker_sha256,
        ))

        cli_path = Path(cli)
        checks["cli_exists"] = cli_path.is_file()
        checks["cli_executable"] = checks["cli_exists"] and os.access(cli_path, os.X_OK)
        fixture = Path(gguf)
        checks["fixture_exists"] = fixture.is_file()
        if checks["fixture_exists"]:
            fixture_sha256 = sha256_file(fixture)
            fixture_bytes = fixture.stat().st_size
            expected_sha = facts.get("expected_fixture_sha256") if isinstance(facts, dict) else None
            expected_bytes = facts.get("expected_fixture_bytes") if isinstance(facts, dict) else None
            expected_arch = facts.get("expected_arch") if isinstance(facts, dict) else EXPECTED_ARCH
            checks["fixture_sha256"] = fixture_sha256 == expected_sha
            checks["fixture_bytes"] = fixture_bytes == expected_bytes
            checks["fixture_pin_authority"] = bool(
                expected_sha == FIXTURE_SHA256 and expected_bytes == FIXTURE_BYTES
            )
            arch_bytes = expected_arch.encode("ascii") if isinstance(expected_arch, str) else b""
            with fixture.open("rb") as handle:
                header = handle.read(1 << 20)
            checks["fixture_arch"] = bool(arch_bytes) and arch_bytes in header
        info.append("fixture sha256=%s bytes=%s" % (fixture_sha256, fixture_bytes))

        can_run = checks["cli_executable"] and checks["fixture_exists"]
        if can_run:
            off = run_phase("off", cli_path, fixture, out_dir, summary_path, False)
            on = run_phase("on", cli_path, fixture, out_dir, summary_path, True)
            info.append("off_exit=%s on_exit=%s" % (off["exit"], on["exit"]))
        checks["off_exit_zero"] = off["exit"] == 0
        checks["on_exit_zero"] = on["exit"] == 0

        off_text = (off["stdout"] + b"\n" + off["stderr"]).decode("utf-8", errors="replace")
        on_text = (on["stdout"] + b"\n" + on["stderr"]).decode("utf-8", errors="replace")
        absent = facts.get("expected_off_absent_needles") if isinstance(facts, dict) else EXPECTED_OFF_ABSENT_NEEDLES
        enable = facts.get("expected_enable_needles") if isinstance(facts, dict) else EXPECTED_ENABLE_NEEDLES
        trace = facts.get("expected_trace_needles") if isinstance(facts, dict) else EXPECTED_TRACE_NEEDLES
        build = facts.get("informational_build_markers") if isinstance(facts, dict) else INFORMATIONAL_BUILD_MARKERS
        checks["off_absent_needles"] = can_run and bool(absent) and not any(
            needle in off_text for needle in absent
        )
        checks["on_enable_needles"] = can_run and bool(enable) and all(
            needle in on_text for needle in enable
        )
        checks["on_trace_needles"] = can_run and bool(trace) and all(
            needle in on_text for needle in trace
        )
        bind_lines_off = len(SET_LINE_RX.findall(off_text))
        info.append("bind set lines OFF=%d (must be 0)" % bind_lines_off)

        off_rows, off_pointers, off_blobs = load_dump(off["dump_dir"], off_problems)
        on_rows, on_pointers, on_blobs = load_dump(on["dump_dir"], on_problems)
        checks["off_manifest_ok"] = can_run and bool(off_rows) and not off_problems
        checks["on_manifest_ok"] = can_run and bool(on_rows) and not on_problems
        off_summary = summarize_rows(off_rows)
        on_summary = summarize_rows(on_rows)
        checks["off_name_set"] = off_summary["names"] == EXPECTED_NAMES
        checks["on_name_set"] = on_summary["names"] == EXPECTED_NAMES
        checks["off_slot_set"] = off_summary["slots"] == IL_SLOTS
        checks["on_slot_set"] = on_summary["slots"] == IL_SLOTS
        expected_occ = {name: EXPECTED_OCCURRENCES for name in EXPECTED_NAMES}
        checks["off_occurrences"] = off_summary["occurrences"] == expected_occ
        checks["on_occurrences"] = on_summary["occurrences"] == expected_occ
        info.append("manifest rows off=%d on=%d; names off=%d on=%d; slots off=%s on=%s" % (
            off_summary["row_count"], on_summary["row_count"],
            len(off_summary["names"]), len(on_summary["names"]),
            off_summary["slots"], on_summary["slots"],
        ))
        info.append("occurrence counts per name off=%s on=%s" % (
            sorted(set(off_summary["occurrences"].values())),
            sorted(set(on_summary["occurrences"].values())),
        ))

        checks["off_ptr_manifest_absent"] = can_run and not off_pointers and not (
            off["dump_dir"] / POINTER_MANIFEST_FILENAME
        ).exists()
        checks["off_no_bind_no_ptr"] = can_run and bind_lines_off == 0 and not off_pointers
        info.append("OFF pointer rows=%d (must be 0)" % len(off_pointers))

        checks["on_ptr_manifest_ok"] = can_run and bool(on_pointers) and not any(
            "pointer" in problem for problem in on_problems
        )
        manifest_filenames = {row["filename"] for row in on_rows}
        checks["ptr_manifest_coverage"] = bool(
            on_pointers and set(on_pointers) == manifest_filenames
        )
        info.append("pointer manifest rows ON=%d; manifest rows ON=%d; coverage=%s" % (
            len(on_pointers), len(on_rows), checks["ptr_manifest_coverage"],
        ))

        set_lines_on = len(SET_LINE_RX.findall(on_text))
        bindings_on = parse_bindings(on_text)
        checks["bind_parse_ok"] = can_run and len(bindings_on) == set_lines_on > 0
        counts = bind_counts(bindings_on)
        bind_passes = sorted(counts)
        bind_counts_on = {str(p): counts[p] for p in bind_passes}
        sentinel_bind_count = sum(1 for row in bindings_on if row["sentinel"])
        checks["bind_pass_set"] = bind_passes == EXPECTED_PASSES
        values = [counts.get(p, 0) for p in EXPECTED_PASSES]
        checks["bind_counts_equal_positive"] = bool(
            values and min(values) > 0 and len(set(values)) == 1
            and values == [EXPECTED_SET_ROWS_PER_PASS for _ in EXPECTED_PASSES]
        )
        checks["bind_sentinel_count"] = sentinel_bind_count == EXPECTED_SENTINEL_ROWS
        info.append("bind set key_il=%d passes=%s counts=%s sentinel(-1)=%d total=%d" % (
            EXPECTED_BIND_KEY_IL, bind_passes, bind_counts_on, sentinel_bind_count,
            len(bindings_on),
        ))

        slot_rows = [row for row in on_rows if parse_tensor_name(row["name"]) is not None
                     and parse_tensor_name(row["name"])[0] == "slots"]
        slot_events = len(slot_rows)
        slot_by_filename = {row["filename"]: row for row in slot_rows}
        slot_ptrs = {
            filename: on_pointers[filename]
            for filename in slot_by_filename if filename in on_pointers
        }
        checks["slot_dump_names"] = bool(slot_rows) and sorted(
            {row["name"] for row in slot_rows}
        ) == EXPECTED_SLOT_NAMES
        slot_occurrences = {}
        for row in slot_rows:
            slot_occurrences[row["name"]] = slot_occurrences.get(row["name"], 0) + 1
        checks["slot_dump_occurrences"] = slot_occurrences == {
            name: EXPECTED_OCCURRENCES for name in EXPECTED_SLOT_NAMES
        }
        info.append("slot dump rows=%d names=%d occurrence values=%s" % (
            slot_events, len(slot_occurrences), sorted(set(slot_occurrences.values())),
        ))

        ptr_to_events = {}
        for filename, address in slot_ptrs.items():
            ptr_to_events.setdefault(address, []).append(filename)
        set_rows_by_ptr = {}
        sentinel_rows_by_ptr = {}
        for row in bindings_on:
            if row["sentinel"]:
                sentinel_rows_by_ptr.setdefault(row["lookup"], []).append(row)
            else:
                set_rows_by_ptr.setdefault(row["lookup"], []).append(row)

        identity_ok = True
        wrong_il_ok = True
        all_bound_ok = True
        disjoint_ok = True
        basenames_by_pass = {str(p): set() for p in EXPECTED_PASSES}
        matched_indices_by_pass = {str(p): set() for p in EXPECTED_PASSES}
        match_ptrs_by_pass = {str(p): set() for p in EXPECTED_PASSES}
        slot_rows_by_pass = {str(p): 0 for p in EXPECTED_PASSES}
        for pass_str, basename in EXPECTED_SLOT_BASENAME_PER_PASS.items():
            slot_rows_by_pass[pass_str] = sum(
                1 for row in slot_rows if row["basename"] == basename
            )

        for filename, address in slot_ptrs.items():
            row = slot_by_filename[filename]
            matched = set_rows_by_ptr.get(address, [])
            binding_passes = {str(item["pass"]) for item in matched
                              if item["pass"] in EXPECTED_PASSES}
            if not matched:
                sentinel_matched = sentinel_rows_by_ptr.get(address, [])
                if row["basename"] == "ffn_moe_slots-0" and sentinel_matched:
                    sentinel_slot_events += 1
                    continue
                all_bound_ok = False
                continue
            if not binding_passes:
                all_bound_ok = False
                continue
            if len(binding_passes) != 1:
                disjoint_ok = False
                continue
            pass_str = next(iter(binding_passes))
            expected_basename = EXPECTED_SLOT_BASENAME_PER_PASS[pass_str]
            if row["basename"] != expected_basename:
                wrong_il_ok = False
            if any(item["key_il"] != EXPECTED_BIND_KEY_IL for item in matched):
                identity_ok = False
            if not all(item["pass"] == int(pass_str) for item in matched):
                disjoint_ok = False
            basenames_by_pass[pass_str].add(row["basename"])
            match_ptrs_by_pass[pass_str].add(address)
            matched_indices_by_pass[pass_str].update(item["index"] for item in matched)

        for row in bindings_on:
            if row["sentinel"] or row["pass"] not in EXPECTED_PASSES:
                continue
            if row["key_il"] != EXPECTED_BIND_KEY_IL:
                if row["lookup"] in ptr_to_events:
                    identity_ok = False
                continue
            if row["lookup"] not in ptr_to_events:
                unmatched_set_rows += 1
                continue
            expected_basename = EXPECTED_SLOT_BASENAME_PER_PASS[str(row["pass"])]
            for filename in ptr_to_events[row["lookup"]]:
                bound_basename = slot_by_filename[filename]["basename"]
                if bound_basename != expected_basename:
                    wrong_il_ok = False
                    basenames_by_pass[str(row["pass"])].add(bound_basename)

        for pass_str in EXPECTED_SLOT_BASENAME_PER_PASS:
            if basenames_by_pass[pass_str] != {EXPECTED_SLOT_BASENAME_PER_PASS[pass_str]}:
                wrong_il_ok = False
            if not match_ptrs_by_pass[pass_str]:
                identity_ok = False
        all_ptrs = [match_ptrs_by_pass[str(p)] for p in EXPECTED_PASSES]
        if any(not values for values in all_ptrs):
            disjoint_ok = False
        seen_ptrs = set()
        for values in all_ptrs:
            if seen_ptrs & values:
                disjoint_ok = False
            seen_ptrs |= values

        checks["bind_slot_identity"] = can_run and identity_ok
        checks["bind_slot_no_wrong_il"] = can_run and wrong_il_ok
        checks["slot_ptrs_all_bound"] = can_run and all_bound_ok
        checks["sentinel_slot_ptr_bound"] = (
            can_run and sentinel_slot_events == EXPECTED_SENTINEL_SLOT_DUMPS
        )
        checks["slot_ptrs_disjoint_per_pass"] = can_run and disjoint_ok
        slot_dumps_per_pass = {
            pass_: slot_rows_by_pass[str(pass_)] for pass_ in EXPECTED_PASSES
        }
        matched_rows_per_pass = {
            pass_: len(matched_indices_by_pass[str(pass_)]) for pass_ in EXPECTED_PASSES
        }
        matched_pointers_per_pass = {
            pass_: len(match_ptrs_by_pass[str(pass_)]) for pass_ in EXPECTED_PASSES
        }
        checks["matched_counts_equal_per_pass"] = can_run and bool(
            slot_dumps_per_pass
            and slot_dumps_per_pass == matched_rows_per_pass
            and all(value == EXPECTED_SLOT_DUMPS_PER_PASS
                    for value in slot_dumps_per_pass.values())
            and all(value == EXPECTED_MATCHED_POINTERS_PER_PASS
                    for value in matched_pointers_per_pass.values())
        )
        checks["matched_basename_unique_per_pass"] = can_run and all(
            basenames_by_pass[str(p)] == {EXPECTED_SLOT_BASENAME_PER_PASS[str(p)]}
            for p in EXPECTED_PASSES
        )
        info.append("matched slot events per pass=%s; slot dumps per pass=%s; "
                    "matched set rows per pass=%s" % (
                        matched_pointers_per_pass, slot_dumps_per_pass, matched_rows_per_pass,
                    ))
        info.append("set rows with no dumped slots tensor=%d (builds not dumped; not claimed)" % (
            unmatched_set_rows,
        ))
        info.append("sentinel-bound slot dump rows=%d (key_il=-1 pass=-1, not a pass event)" % (
            sentinel_slot_events,
        ))
        info.append("pass-to-slot map=%s verified per dumped event; addresses not recorded "
                    "(ASLR varies per run)" % PASS_TO_SLOT)

        off_keys = set(off_blobs)
        on_keys = set(on_blobs)
        common = off_keys & on_keys
        bin_files_compared = len(common)
        bin_bytes_compared = sum(len(off_blobs[key]) for key in common)
        bin_differing = sum(1 for key in common if off_blobs[key] != on_blobs[key])
        checks["bin_parity"] = bool(common) and off_keys == on_keys and bin_differing == 0
        stdout_bytes = len(off["stdout"])
        stdout_differing = stdout_differing_bytes(off["stdout"], on["stdout"])
        checks["stdout_parity"] = bool(off["stdout"]) and stdout_differing == 0
        info.append("bin parity files=%d bytes=%d differing=%d; stdout bytes=%d differing=%d" % (
            bin_files_compared, bin_bytes_compared, bin_differing,
            stdout_bytes, stdout_differing,
        ))
        for marker in build:
            off_count = sum(1 for line in off_text.splitlines() if marker in line)
            on_count = sum(1 for line in on_text.splitlines() if marker in line)
            info.append("unconditional build marker off=%d on=%d (informational only): %s" % (
                off_count, on_count, marker,
            ))
        info.append(CYCLE_AUDIT_REQUIRED)
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
            "total": len(checks), "passed": passed, "failed": len(checks) - passed,
        },
        "info": info,
        "note": NOTE,
        "scope": SCOPE,
        "speed_claim": None,
        "no_weights": True,
        "no_engine_run": False,
        "engine_run": ENGINE_RUN,
        "frozen_on": FROZEN_ON,
        "checker_sha256": checker_sha256,
        "contract_path": CONTRACT_REL,
        "expected_summary_path": SUMMARY_REL,
        "fixture_sha256": fixture_sha256,
        "fixture_bytes": fixture_bytes,
        "off_exit": off["exit"],
        "on_exit": on["exit"],
        "pass_to_slot": PASS_TO_SLOT,
        "expected_core_slots": CORE_SLOTS,
        "off_manifest_rows": off_summary["row_count"],
        "on_manifest_rows": on_summary["row_count"],
        "off_tensor_names": len(off_summary["names"]),
        "on_tensor_names": len(on_summary["names"]),
        "off_slots": off_summary["slots"],
        "on_slots": on_summary["slots"],
        "off_occurrence_counts": sorted(set(off_summary["occurrences"].values())),
        "on_occurrence_counts": sorted(set(on_summary["occurrences"].values())),
        "off_pointer_rows": len(off_pointers),
        "on_pointer_rows": len(on_pointers),
        "bind_passes_on": bind_passes,
        "bind_counts_on": bind_counts_on,
        "bind_sentinel_count_on": sentinel_bind_count,
        "bind_lines_off": bind_lines_off,
        "slot_dump_events_on": slot_events,
        "sentinel_slot_events_on": sentinel_slot_events,
        "slot_dumps_per_pass_on": slot_dumps_per_pass,
        "matched_slot_events_per_pass_on": matched_pointers_per_pass,
        "matched_set_rows_per_pass_on": matched_rows_per_pass,
        "unmatched_set_rows_on": unmatched_set_rows,
        "bin_files_compared": bin_files_compared,
        "bin_bytes_compared": bin_bytes_compared,
        "bin_differing_bytes": bin_differing,
        "stdout_bytes": stdout_bytes,
        "stdout_differing_bytes": stdout_differing,
        "cycle_audit_required": CYCLE_AUDIT_REQUIRED,
    }
    write_json(summary_path, summary)
    return summary


def compact_summary(summary):
    return {
        "verdict": summary["verdict"],
        "blocker": summary["blocker"],
        "counts": summary["check_counts"],
        "fixture_sha256": summary["fixture_sha256"],
        "fixture_bytes": summary["fixture_bytes"],
        "pass_to_slot": summary["pass_to_slot"],
        "bind_counts_on": summary["bind_counts_on"],
        "slot_dump_events_on": summary["slot_dump_events_on"],
        "sentinel_slot_events_on": summary["sentinel_slot_events_on"],
        "matched_slot_events_per_pass_on": summary["matched_slot_events_per_pass_on"],
        "unmatched_set_rows_on": summary["unmatched_set_rows_on"],
        "bin_files_compared": summary["bin_files_compared"],
        "bin_bytes_compared": summary["bin_bytes_compared"],
        "bin_differing_bytes": summary["bin_differing_bytes"],
        "stdout_bytes": summary["stdout_bytes"],
        "stdout_differing_bytes": summary["stdout_differing_bytes"],
        "speed_claim": None,
        "note": NOTE,
        "summary_path": SUMMARY_REL,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to check")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI, help="built CPU-only llama-debug")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF, help="synthetic 3-pass GGUF")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="run/dump directory")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        summary = run_check(args.root, args.cli, args.gguf, args.out_dir)
    except Exception as exc:
        print("error: could not write summary: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print(json.dumps(compact_summary(summary), sort_keys=True, separators=(",", ":")))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
