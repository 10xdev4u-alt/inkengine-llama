#!/usr/bin/env python3
"""Offline, fail-closed CPU-only full-topology union-wiring binding gate.

Drives an already built CPU-only llama-debug twice on a deterministic synthetic
GGUF carrying the real Inkling2 loop structure: 24 layers, logical core 6..17,
3 passes, rank-64 controllers, 48 physical cache slots. OFF leaves LLAMA_RORET
unset; ON sets LLAMA_RORET=1 and LLAMA_RORET_TRACE=1. Both phases dump MoE
tensor bytes. The gate requires a binding for all 36 (core il, pass) pairs,
pass-indexed slot separation, and OFF/ON byte parity. No timing, no speed.
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
CONTRACT_REL = "tools/audit/FULLTOPO_BINDING.json"
SUMMARY_REL = "tools/audit/M45_FULLTOPO_BINDING_SUMMARY.json"
DEFAULT_CLI = Path("/tmp/opencode/m44-probe/bin/llama-debug")
DEFAULT_GGUF = Path("/tmp/opencode/fulltopo-probe.gguf")
DEFAULT_OUT_DIR = Path("/tmp/opencode/m44-real")

SCHEMA_VERSION = 1
FROZEN_ON = "2026-11-28"
NOTE = "CPU-only full-topology union-wiring binding gate; no timing"
SCOPE = (
    "PASS means only that, on a deterministic synthetic GGUF carrying the real "
    "Inkling2 structural topology (24 layers, core 6-17, 3 passes, rank-64 "
    "controllers, 48 slots) on a CPU-only build, the union-resident MoE hook "
    "registers bindings for every core layer on passes 0/1/2 with disjoint "
    "pass-indexed slot maps and all mapped physical slots present; OFF/ON raw "
    "MoE tensors, logits and text are "
    "byte-identical. It is NOT reference tensor-value correctness, NOT "
    "real-weight parity, NOT GPU copy-callback hit/upload traffic, and NOT any "
    "timing/throughput/wall-clock result."
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
FIXTURE_SHA256 = "3ccf2fb4c6ef4fab60464d2a35a2ea42b24186db3e8b12cd68881f4fa873dff7"
FIXTURE_BYTES = 11941248
EXPECTED_ARCH = "inkling2_loop"
LOOP_CONSTANTS = {
    "N_LAYER": 24,
    "CORE_START": 6,
    "CORE_END": 18,
    "N_PASSES": 3,
    "GATE_RANK": 64,
}
TENSOR_KINDS = ("logits", "probs", "topk", "weights", "out")
EXPECTED_OCCURRENCES = 2
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
RORET_PASS_SCOPE = (
    "RORET_PASS rows require the device slot-map copy and are NOT expected on "
    "the CPU fallback; see M21/M22. Informational only, never gated."
)
EXPECTED_MANIFEST_FIELDS = 12
CPU_SHARE = "1.0"
TENSOR_FILTER = r"^ffn_moe_(logits|probs|topk|weights|out)-[0-9]+$"
CLI_FLAGS = [
    "-p", "hello world", "-n", "4", "-t", "1", "-cmoe",
    "--moe-cache-mib", "16", "--verbose", "--tensor-filter", TENSOR_FILTER,
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
    "bind_core_pair_coverage",
    "bind_pass_set",
    "bind_counts_equal_positive",
    "slot_map_distinct_per_pass",
    "pass_slot_map_subset_of_dump",
    "text_parity",
    "stdout_parity",
    "bin_parity",
    "run_completed",
]

EXPECTED_TOP_KEYS = frozenset({
    "schema_version", "frozen_on", "scope", "note", "speed_claim",
    "no_weights", "no_engine_run", "no_weights_no_engine_run_note",
    "engine_run", "cycle_audit_required", "expected_fixture_sha256",
    "expected_fixture_bytes", "expected_arch", "loop_constants",
    "pass_to_slot", "expected_core_il", "expected_passes",
    "expected_tensor_kinds", "expected_tensor_names", "expected_occurrences",
    "expected_enable_needles", "informational_build_markers",
    "expected_trace_needles", "expected_off_absent_needles",
    "expected_manifest_fields", "expected_summary_path",
    "checker_sha256", "verify",
})

NAME_RX = re.compile(r"^ffn_moe_(logits|probs|topk|weights|out)-(\d+)$")
OCC_RX = re.compile(r"__(\d+)\.bin$")
SET_LINE_RX = re.compile(r"RORET_BIND\s+set\b")
SET_RX = re.compile(
    r"RORET_BIND\s+set\b.*?\bslot_map=(0x[0-9a-fA-F]+)\s+"
    r"key_il=(-?\d+)\s+pass=(-?\d+)"
)
PASS_RX = re.compile(r"RORET_PASS\s+il=(-?\d+)\s+pass=(-?\d+)")
PERF_LINE_RX = re.compile(rb"^\[ Prompt: .* t/s \| Generation: .* t/s \]\r?\n?$")
NORMALIZE_NEEDLES = tuple(
    dict.fromkeys(EXPECTED_OFF_ABSENT_NEEDLES + INFORMATIONAL_BUILD_MARKERS)
)


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def derive_pass_to_slot(constants):
    """Derive pass -> physical slots from the frozen full-topology constants."""
    required = ("N_LAYER", "CORE_START", "CORE_END", "N_PASSES", "GATE_RANK")
    if not isinstance(constants, dict) or set(constants) != set(required):
        raise ValueError("loop constants must have exactly the frozen key set")
    for name in required:
        if not is_int(constants[name]) or constants[name] <= 0:
            raise ValueError("loop constant %s must be a positive integer" % name)
    n_layer = constants["N_LAYER"]
    start = constants["CORE_START"]
    end = constants["CORE_END"]
    passes = constants["N_PASSES"]
    if not 0 < start < end <= n_layer:
        raise ValueError("loop core must lie inside the physical layers")
    width = end - start
    if width < 1:
        raise ValueError("the loop core needs at least one logical layer")
    if passes < 2:
        raise ValueError("the loop needs at least two passes")
    n_slots = n_layer + (passes - 1) * width
    pass_to_slot = {}
    for p in range(passes):
        base = start if p == 0 else n_layer + (p - 1) * width
        pass_to_slot[str(p)] = list(range(base, base + width))
    return {
        "pass_to_slot": pass_to_slot,
        "core_il": list(range(start, end)),
        "il_slots": list(range(n_slots)),
    }


DERIVED = derive_pass_to_slot(LOOP_CONSTANTS)
PASS_TO_SLOT = DERIVED["pass_to_slot"]
CORE_IL = DERIVED["core_il"]
IL_SLOTS = DERIVED["il_slots"]
EXPECTED_PASSES = [int(p) for p in PASS_TO_SLOT]
EXPECTED_NAMES = sorted(
    "ffn_moe_%s-%d" % (kind, il) for il in IL_SLOTS for kind in TENSOR_KINDS
)
EXPECTED_EXPECTED_ROWS = len(EXPECTED_NAMES) * EXPECTED_OCCURRENCES


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
        "scope": SCOPE,
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
        core = facts["expected_core_il"]
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
        and facts.get("scope") is None
        and facts.get("note") is None
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
        and core == CORE_IL
        and facts.get("pass_to_slot") == PASS_TO_SLOT
        and derived["pass_to_slot"] == PASS_TO_SLOT
        and derived["core_il"] == CORE_IL
        and derived["il_slots"] == IL_SLOTS
        and kinds == list(TENSOR_KINDS)
        and names == EXPECTED_NAMES
        and facts.get("expected_occurrences") == EXPECTED_OCCURRENCES
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
            "type": fields[1],
            "ne": list(fields[2:6]),
            "nb": list(fields[6:10]),
            "nbytes": fields[10],
            "filename": filename,
            "occurrence": int(occ_match.group(1)),
        })
    return rows


def summarize_rows(rows):
    """Summarize name set, physical slot set and per-name occurrence counts."""
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
    """Parse a phase dump into rows and raw .bin blobs keyed by filename."""
    rows = parse_manifest(dump_dir / "manifest.tsv", problems)
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
    return rows, blobs


def parse_bindings(text):
    """Extract full RORET_BIND set rows from a phase log."""
    return [
        {
            "index": index,
            "slot_map": slot_map.lower(),
            "key_il": int(il),
            "pass": int(pass_),
            "sentinel": int(il) == -1 and int(pass_) == -1,
        }
        for index, (slot_map, il, pass_) in enumerate(SET_RX.findall(text))
    ]


def binding_coverage(bindings, core_il=CORE_IL, passes=EXPECTED_PASSES):
    """Require every (logical core layer, pass) pair exactly once as a set."""
    valid = [row for row in bindings if not row["sentinel"]]
    core = set(core_il)
    pass_set = set(passes)
    pairs = {(row["key_il"], row["pass"]) for row in valid}
    expected = {(il, p) for il in core for p in pass_set}
    scoped_ok = all(row["key_il"] in core and row["pass"] in pass_set for row in valid)
    ok = bool(valid) and scoped_ok and pairs == expected
    return {
        "ok": ok,
        "valid_rows": len(valid),
        "pairs": sorted(pairs),
        "missing": sorted(expected - pairs),
        "unexpected": sorted(pairs - expected),
    }


def bind_counts(bindings, passes=EXPECTED_PASSES):
    counts = {}
    for row in bindings:
        if not row["sentinel"] and row["pass"] in passes:
            counts[row["pass"]] = counts.get(row["pass"], 0) + 1
    return counts


def parse_slot_maps(bindings, passes=EXPECTED_PASSES):
    pointers = {p: set() for p in passes}
    for row in bindings:
        if not row["sentinel"] and row["pass"] in pointers:
            pointers[row["pass"]].add(row["slot_map"])
    return {p: sorted(values) for p, values in pointers.items()}


def slot_maps_distinct_per_pass(pointers, passes=EXPECTED_PASSES):
    """Require nonempty, nonzero, equally sized and pairwise disjoint pointer sets."""
    if sorted(pointers) != sorted(passes):
        return False
    per_pass = [set(pointers.get(p, [])) for p in passes]
    if any(not values or "0x0" in values for values in per_pass):
        return False
    if len({len(values) for values in per_pass}) != 1:
        return False
    seen = set()
    for values in per_pass:
        if seen & values:
            return False
        seen |= values
    return True


def normalize_stdout(data):
    """Drop anchored trace/enable lines plus perf lines; return (kept, dropped count)."""
    kept = []
    dropped = 0
    for line in data.splitlines(keepends=True):
        stripped = line.decode("utf-8", errors="replace").strip()
        if any(stripped.startswith(needle) for needle in NORMALIZE_NEEDLES):
            dropped += 1
            continue
        if PERF_LINE_RX.match(line):
            dropped += 1
            continue
        kept.append(line)
    return b"".join(kept), dropped


def stdout_differing_bytes(left, right):
    """Count differing byte positions; never inspects or records timing values."""
    if left == right:
        return 0
    return sum(1 for a, b in zip(left, right) if a != b) + abs(len(left) - len(right))


def run_phase(phase, cli, gguf, out_dir, summary_path, on, cpu_share):
    """Run one engine phase; capture stdout and stderr separately, never time it."""
    env = os.environ.copy()
    env.pop("LLAMA_RORET", None)
    env.pop("LLAMA_RORET_TRACE", None)
    env.pop("LLAMA_DUMP_TENSOR_PTR", None)
    if on:
        env["LLAMA_RORET"] = "1"
        env["LLAMA_RORET_TRACE"] = "1"
    env["M45_SUMMARY_PATH"] = str(summary_path)
    dump_dir = out_dir / phase
    problems = []
    try:
        if dump_dir.exists():
            shutil.rmtree(dump_dir)
    except OSError as exc:
        problems.append("dump_cleanup_failed: %s: %s" % (type(exc).__name__, exc))
    env["LLAMA_DUMP_TENSORS_DIR"] = str(dump_dir)
    cmd = [
        str(cli), "-m", str(gguf), *CLI_FLAGS,
        "--moe-cache-cpu-share", str(cpu_share),
    ]
    stdout_path = out_dir / (phase + "_stdout.log")
    stderr_path = out_dir / (phase + "_stderr.log")
    result = {
        "exit": None,
        "stdout": b"",
        "stderr": b"",
        "dump_dir": dump_dir,
        "problems": problems,
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


def run_check(root, cli, gguf, out_dir, cpu_share):
    root = Path(root)
    summary_path = root / SUMMARY_REL
    out_dir = Path(out_dir)
    info = ["cpu_share=%s" % cpu_share]
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
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        sentinel_ok = False
        info.append("out_dir_mkdir_failed: %s: %s" % (type(exc).__name__, exc))

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
    off_problems = []
    on_problems = []
    coverage = {"ok": False, "valid_rows": 0, "pairs": [], "missing": [], "unexpected": []}
    bind_counts_on = {}
    bind_sentinel_count = 0
    bind_lines_off = 0
    slot_pointers = {}
    pass_slot_ok = False
    rorret_pass_rows_on = 0
    bin_files_compared = 0
    bin_bytes_compared = 0
    bin_differing = 0
    text_differing = 0
    stdout_normal_differing = 0
    stdout_dropped_off = 0
    stdout_dropped_on = 0
    stdout_bytes = 0
    stdout_normal_bytes = 0
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
            off = run_phase("off", cli_path, fixture, out_dir, summary_path, False, cpu_share)
            on = run_phase("on", cli_path, fixture, out_dir, summary_path, True, cpu_share)
            off_problems.extend(off["problems"])
            on_problems.extend(on["problems"])
            info.append("off_exit=%s on_exit=%s" % (off["exit"], on["exit"]))
        checks["off_exit_zero"] = off["exit"] == 0
        checks["on_exit_zero"] = on["exit"] == 0

        off_text = (off["stdout"] + b"\n" + off["stderr"]).decode("utf-8", errors="replace")
        on_text = (on["stdout"] + b"\n" + on["stderr"]).decode("utf-8", errors="replace")
        absent = facts.get("expected_off_absent_needles") if isinstance(facts, dict) else EXPECTED_OFF_ABSENT_NEEDLES
        enable = facts.get("expected_enable_needles") if isinstance(facts, dict) else EXPECTED_ENABLE_NEEDLES
        trace = facts.get("expected_trace_needles") if isinstance(facts, dict) else EXPECTED_TRACE_NEEDLES
        build = facts.get("informational_build_markers") if isinstance(facts, dict) else INFORMATIONAL_BUILD_MARKERS
        off_lines = [line.strip() for line in off_text.splitlines()]
        on_lines = [line.strip() for line in on_text.splitlines()]
        def anchored(lines, needle):
            return any(line == needle or line.endswith(" " + needle) for line in lines)

        checks["off_absent_needles"] = can_run and bool(absent) and not any(
            anchored(off_lines, needle) for needle in absent
        )
        checks["on_enable_needles"] = can_run and bool(enable) and all(
            anchored(on_lines, needle) for needle in enable
        )
        checks["on_trace_needles"] = can_run and bool(trace) and all(
            any(needle in line for line in on_lines) for needle in trace
        )
        bind_lines_off = len(SET_LINE_RX.findall(off_text))
        info.append("bind set lines OFF=%d (must be 0)" % bind_lines_off)

        off_rows, off_blobs = load_dump(off["dump_dir"], off_problems)
        on_rows, on_blobs = load_dump(on["dump_dir"], on_problems)
        checks["off_manifest_ok"] = can_run and bool(off_rows) and not off_problems
        checks["on_manifest_ok"] = can_run and bool(on_rows) and not on_problems
        off_summary = summarize_rows(off_rows)
        on_summary = summarize_rows(on_rows)
        expected_occ = {name: EXPECTED_OCCURRENCES for name in EXPECTED_NAMES}
        checks["off_name_set"] = off_summary["names"] == EXPECTED_NAMES
        checks["on_name_set"] = on_summary["names"] == EXPECTED_NAMES
        checks["off_slot_set"] = off_summary["slots"] == IL_SLOTS
        checks["on_slot_set"] = on_summary["slots"] == IL_SLOTS
        checks["off_occurrences"] = (
            off_summary["occurrences"] == expected_occ
            and off_summary["row_count"] == EXPECTED_EXPECTED_ROWS
        )
        checks["on_occurrences"] = (
            on_summary["occurrences"] == expected_occ
            and on_summary["row_count"] == EXPECTED_EXPECTED_ROWS
        )
        info.append("manifest rows off=%d on=%d (expected %d); names off=%d on=%d; "
                    "slots off=%d on=%d" % (
                        off_summary["row_count"], on_summary["row_count"], EXPECTED_EXPECTED_ROWS,
                        len(off_summary["names"]), len(on_summary["names"]),
                        len(off_summary["slots"]), len(on_summary["slots"]),
                    ))
        info.append("occurrence counts per name off=%s on=%s" % (
            sorted(set(off_summary["occurrences"].values())),
            sorted(set(on_summary["occurrences"].values())),
        ))

        set_lines_on = len(SET_LINE_RX.findall(on_text))
        bindings_on = parse_bindings(on_text)
        checks["bind_parse_ok"] = can_run and len(bindings_on) == set_lines_on > 0
        coverage = binding_coverage(bindings_on)
        counts = bind_counts(bindings_on)
        bind_counts_on = {str(p): counts.get(p, 0) for p in EXPECTED_PASSES}
        bind_sentinel_count = sum(1 for row in bindings_on if row["sentinel"])
        passes_seen = sorted(counts)
        checks["bind_core_pair_coverage"] = can_run and coverage["ok"]
        checks["bind_pass_set"] = passes_seen == EXPECTED_PASSES
        values = [counts.get(p, 0) for p in EXPECTED_PASSES]
        checks["bind_counts_equal_positive"] = bool(
            values and min(values) > 0 and len(set(values)) == 1
        )
        info.append("bind set rows=%d valid=%d sentinel(-1)=%d; passes=%s counts=%s" % (
            set_lines_on, coverage["valid_rows"], bind_sentinel_count,
            passes_seen, bind_counts_on,
        ))
        info.append("core pair coverage=%s pairs=%d/%d missing=%s unexpected=%s" % (
            coverage["ok"], len(coverage["pairs"]), len(CORE_IL) * len(EXPECTED_PASSES),
            coverage["missing"], coverage["unexpected"],
        ))

        slot_pointers = parse_slot_maps(bindings_on)
        checks["slot_map_distinct_per_pass"] = can_run and slot_maps_distinct_per_pass(slot_pointers)
        dumped_slots = set(on_summary["slots"])
        pass_slot_ok = checks["on_slot_set"] and all(
            set(PASS_TO_SLOT[str(p)]) <= dumped_slots for p in EXPECTED_PASSES
        )
        checks["pass_slot_map_subset_of_dump"] = can_run and pass_slot_ok
        info.append("pass-to-slot map=%s; physical slots in ON dump=%d; all mapped "
                    "slots present=%s (implied by on_slot_set; set-level, not "
                    "per-event linkage)" % (
                        {p: "%d-%d" % (values[0], values[-1]) for p, values in PASS_TO_SLOT.items()},
                        len(dumped_slots), pass_slot_ok,
                    ))
        info.append("slot_map pointer counts per pass=%s disjoint=%s (addresses not "
                    "recorded: ASLR varies per run)" % (
                        {str(p): len(slot_pointers.get(p, [])) for p in EXPECTED_PASSES},
                        slot_maps_distinct_per_pass(slot_pointers),
                    ))

        rorret_pass_rows_on = len(PASS_RX.findall(on_text))
        info.append("RORET_PASS rows ON=%d (informational only, never gated): %s" % (
            rorret_pass_rows_on, RORET_PASS_SCOPE,
        ))
        for marker in build:
            off_count = sum(1 for line in off_text.splitlines() if marker in line)
            on_count = sum(1 for line in on_text.splitlines() if marker in line)
            info.append("unconditional build marker off=%d on=%d (informational only): %s" % (
                off_count, on_count, marker,
            ))

        off_keys = set(off_blobs)
        on_keys = set(on_blobs)
        common = off_keys & on_keys
        bin_files_compared = len(common)
        bin_bytes_compared = sum(len(off_blobs[key]) for key in common)
        bin_differing = sum(1 for key in common if off_blobs[key] != on_blobs[key])
        checks["bin_parity"] = bool(common) and off_keys == on_keys and bin_differing == 0
        text_differing = stdout_differing_bytes(off["stdout"], on["stdout"])
        checks["text_parity"] = bool(off["stdout"]) and text_differing == 0
        stdout_bytes = len(off["stdout"])
        off_normal, stdout_dropped_off = normalize_stdout(off["stdout"])
        on_normal, stdout_dropped_on = normalize_stdout(on["stdout"])
        stdout_normal_differing = stdout_differing_bytes(off_normal, on_normal)
        stdout_normal_bytes = len(off_normal)
        checks["stdout_parity"] = bool(off_normal) and stdout_normal_differing == 0 and (
            stdout_dropped_off == stdout_dropped_on
        )
        info.append("bin parity files=%d bytes=%d differing=%d; text bytes=%d differing=%d" % (
            bin_files_compared, bin_bytes_compared, bin_differing,
            stdout_bytes, text_differing,
        ))
        info.append("normalized stdout bytes=%d differing=%d dropped lines off=%d on=%d" % (
            stdout_normal_bytes, stdout_normal_differing,
            stdout_dropped_off, stdout_dropped_on,
        ))
        info.append("occurrence index encodes graph eval count, not pass")
        info.append(SCOPE)
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
        "expected_core_il": CORE_IL,
        "expected_passes": EXPECTED_PASSES,
        "expected_rows": EXPECTED_EXPECTED_ROWS,
        "off_manifest_rows": off_summary["row_count"],
        "on_manifest_rows": on_summary["row_count"],
        "off_tensor_names": len(off_summary["names"]),
        "on_tensor_names": len(on_summary["names"]),
        "off_slots": off_summary["slots"],
        "on_slots": on_summary["slots"],
        "off_occurrence_counts": sorted(set(off_summary["occurrences"].values())),
        "on_occurrence_counts": sorted(set(on_summary["occurrences"].values())),
        "bind_rows_off": bind_lines_off,
        "bind_valid_rows_on": coverage["valid_rows"],
        "bind_pairs_on": coverage["pairs"],
        "expected_bind_pairs": len(CORE_IL) * len(EXPECTED_PASSES),
        "missing_bind_pairs": coverage["missing"],
        "unexpected_bind_pairs": coverage["unexpected"],
        "bind_counts_on": bind_counts_on,
        "bind_sentinel_count_on": bind_sentinel_count,
        "slot_map_pointer_counts_on": {
            str(p): len(slot_pointers.get(p, [])) for p in EXPECTED_PASSES
        },
        "pass_slot_map_subset_of_dump": pass_slot_ok,
        "rorret_pass_rows_on_informational": rorret_pass_rows_on,
        "rorret_pass_scope": RORET_PASS_SCOPE,
        "bin_files_compared": bin_files_compared,
        "bin_bytes_compared": bin_bytes_compared,
        "bin_differing_bytes": bin_differing,
        "text_bytes": stdout_bytes,
        "text_differing_bytes": text_differing,
        "stdout_normalized_bytes": stdout_normal_bytes,
        "stdout_normalized_differing_bytes": stdout_normal_differing,
        "stdout_dropped_lines_off": stdout_dropped_off,
        "stdout_dropped_lines_on": stdout_dropped_on,
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
        "core_il": summary["expected_core_il"],
        "pass_to_slot": summary["pass_to_slot"],
        "bind_pairs_on": len(summary["bind_pairs_on"]),
        "expected_bind_pairs": summary["expected_bind_pairs"],
        "missing_bind_pairs": summary["missing_bind_pairs"],
        "off_manifest_rows": summary["off_manifest_rows"],
        "on_manifest_rows": summary["on_manifest_rows"],
        "bin_files_compared": summary["bin_files_compared"],
        "bin_bytes_compared": summary["bin_bytes_compared"],
        "bin_differing_bytes": summary["bin_differing_bytes"],
        "text_differing_bytes": summary["text_differing_bytes"],
        "stdout_normalized_differing_bytes": summary["stdout_normalized_differing_bytes"],
        "speed_claim": None,
        "scope": SCOPE,
        "summary_path": SUMMARY_REL,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to check")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI, help="built CPU-only llama-debug")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF, help="synthetic full-topology GGUF")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="run/dump directory")
    parser.add_argument("--cpu-share", default=CPU_SHARE, help="moe-cache CPU share")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        summary = run_check(args.root, args.cli, args.gguf, args.out_dir, args.cpu_share)
    except Exception as exc:
        print("error: could not write summary: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print(json.dumps(compact_summary(summary), sort_keys=True, separators=(",", ":")))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
