#!/usr/bin/env python3
"""Offline, fail-closed gate for the local d2027 outlier premise."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
FACTS_REL = "tools/audit/OUTLIER_PREMISE.json"
SUMMARY_REL = "tools/audit/M35_OUTLIER_PREMISE_SUMMARY.json"
FACTS_PATH = AUDIT_DIR / "OUTLIER_PREMISE.json"
SUMMARY_PATH = AUDIT_DIR / SUMMARY_REL
NOTE = "outlier premise classification only; no timing"
FROZEN_ON = "2026-11-28"
FROZEN_FACTS_SHA256 = "0d73a0c2de1e0e15632f55cc0024cdf4ce1b57d323671e38a6b23db7424533a6"
INLINE_VERIFICATION_URL = "https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json"
OUT_OF_BAND_CAVEAT = (
    "the two HTML primary-source zero-hit records are out-of-band assertions: this offline "
    "checker cannot re-fetch them and cannot recompute their digests; only the small inline "
    "config.json is recomputed here"
)
SCHEMA_VERSION = 1
CLASSIFICATION = "LOCAL_ONLY_UNVERIFIED"
REAL_TRACE_VERDICT = "NO_GO_BLOCKED"
SIDE_LANE_ENABLED = False
RESOLUTION = (
    "The d2027 activation outlier is a LOCAL_ONLY_UNVERIFIED premise: absent from all three "
    "frozen primary sources, present only in local config/docs/tests. hidden_size 2304 makes "
    "index 2027 plausible but is not evidence. The speculative FP32 side-lane is NOT enabled in "
    "the fork engine path and must stay behind a default-off flag. Real-trace falsification remains "
    "NO_GO_BLOCKED without weights or hosted inference."
)
NO_GO_RESOLUTION = (
    "Gate did not complete: the d2027 premise stays UNVERIFIED_GATE_FAILED. No classification, "
    "side-lane or parity claim is made. Real-trace falsification remains NO_GO_BLOCKED."
)
OPEN_FINDINGS = [
    "Real-trace falsification of the d2027 premise is NO_GO_BLOCKED: recording the "
    "max-|activation| dimension with a forward hook needs real weights or hosted inference; "
    "neither is available here, so the premise stays LOCAL_ONLY_UNVERIFIED."
]

EXPECTED_PROVENANCE = [
    {
        "url": "https://thinkingmachines.ai/news/inkling-2-mini/",
        "byte_count": 448727,
        "sha256": "1301b66c36ff9112dd9c731cc5c83eba1e6b6c401c6364301fbc933d40f7249c",
        "occurrences_2027": 0,
    },
    {
        "url": "https://thinkingmachines.ai/model-card/inkling-2-mini/",
        "byte_count": 222597,
        "sha256": "2fd49bae2efdbef1914314280ebc2d93742f89e903f66354c30b02b95f5efe28",
        "occurrences_2027": 0,
    },
    {
        "url": "https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json",
        "byte_count": 2722,
        "sha256": "a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2",
        "occurrences_2027": 0,
    },
]
PROVENANCE_KEYS = frozenset({"url", "byte_count", "sha256", "occurrences_2027"})
EXPECTED_HIDDEN_SIZE_INFO = {
    "hidden_size": 2304,
    "num_hidden_layers": 24,
    "model_type": "inkling2_loop",
    "source": "https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json",
    "status": "INFO_PLAUSIBLE_INDEX_NOT_EVIDENCE",
}

SOURCE_ROOTS = {
    "configs/inkling2mini.toml": "loopkit",
    "src/loopkit/model.py": "loopkit",
    "docs/INKLING_INTEGRATION.md": "loopkit",
    "README.md": "repo",
    "tools/odsl/README.md": "repo",
    "tools/odsl/LAYOUT_DESIGN.md": "repo",
}
FROZEN_SOURCE_SHA256 = {
    "configs/inkling2mini.toml": "8be0e650f90cb490af15057768fbe1d7cc9538f23f36edeb11ccd7c31d3a89eb",
    "src/loopkit/model.py": "523003de85142d9f4c1792a5698abf393ee3b620ec085057621e53b19d6a39bf",
    "docs/INKLING_INTEGRATION.md": "9b0993f68bd2169468c4beedb1d54522e7ce9bd3f1c3b1c94d4033120aa0affc",
    "README.md": "4d3d09f39975bfbaca2c3b757bb1c25d23957b8b24ab7d4a78d7e19cae68c293",
    "tools/odsl/README.md": "09b068e2f4bf36d266c1f2d27345846a577bb52e328794fc848d825e6faf2d74",
    "tools/odsl/LAYOUT_DESIGN.md": "23625beb8d9283572e824b29499ee3c1135cbc2a74876e1d06fccdd4e6604302",
}

SOURCE_NEEDLES = {
    "config_outlier_dim": {
        "root": "loopkit",
        "source": "configs/inkling2mini.toml",
        "needles": ["outlier_dim = 2027"],
    },
    "model_unverified_hypothesis": {
        "root": "loopkit",
        "source": "src/loopkit/model.py",
        "needles": [
            "``outlier_dim`` is an UNVERIFIED local hypothesis (current value 2027). It is",
            "absent from every primary source -- TML's release post and model card -- and is",
            "kept only as a local test knob. It is not a stated model property.",
            "outlier_dim: int | None",
        ],
    },
    "integration_local_only": {
        "root": "loopkit",
        "source": "docs/INKLING_INTEGRATION.md",
        "needles": [
            "**Known outlier.** `[L]` \"activation outlier at dimension 2027\" appears only in our `README.md`,",
            "`configs/inkling2mini.toml` (`outlier_dim=2027`), `docs/ENGINE_SPEC.md` and tests. **none of the TML/HF",
            "primary sources mention it.",
        ],
    },
    "integration_side_lane_flag": {
        "root": "loopkit",
        "source": "docs/INKLING_INTEGRATION.md",
        "needles": [
            "The d2027 FP32 side-lane should stay",
            "**disabled behind a flag** until risk #4 is resolved.",
        ],
    },
    "integration_roadmap_item4": {
        "root": "loopkit",
        "source": "docs/INKLING_INTEGRATION.md",
        "needles": [
            "4. **\"Outlier at dim 2027\" has no primary-source support** `[L]`. Experiment: audit where 2027 is",
            "requires weights or hosted inference. Until then, flag side-lane as",
        ],
    },
    "fork_readme_unverified": {
        "root": "repo",
        "source": "README.md",
        "needles": [
            "The \"activation outlier at dimension",
            "2027\" mentioned in some local docs is an **unverified local hypothesis; not in primary sources**",
        ],
    },
    "odsl_readme_unverified": {
        "root": "repo",
        "source": "tools/odsl/README.md",
        "needles": [
            "The d2027 outlier is an **unverified local hypothesis;",
            "not in primary sources** (absent from TML's release post/model card; present only in our",
            "an FP32 side-lane for the hypothetical d2027 outlier **[unverified local",
            "hypothesis; not in primary sources]**",
            "No bit-exactness is claimed for the FP16 path",
        ],
    },
    "odsl_layout_default_off": {
        "root": "repo",
        "source": "tools/odsl/LAYOUT_DESIGN.md",
        "needles": [
            "The FP32 side-lane path is explicitly labeled LOWER-DRIFT but",
            "NOT BIT-EXACT. It is a cast/numeric model, not a correctness gate.",
            "Runtime gate: environment variable `LLAMA_ODSL`, enabled only for",
            "`ODSL_LAYOUT`, default OFF.",
            "Side-lane sub-gate: `LLAMA_ODSL_SIDE_LANE`, also default off.",
            "The d2027 outlier is an UNVERIFIED LOCAL HYPOTHESIS and is NOT in",
            "No speedup claim is publishable without a CycleAudit lower bound",
        ],
    },
}

EXPECTED_ENGINE_SCOPE = {
    "roots": ["src", "common", "ggml"],
    "loop_path_markers": ["inkling", "roret", "pibs", "moe-cache", "odsl"],
    "loop_content_markers": ["inkling2", "llama_roret", "llama_pibs"],
    "global_forbidden_tokens": [
        "outlier_dim",
        "d2027",
        "LLAMA_ODSL",
        "ODSL_LAYOUT",
        "outlier_drift",
        "side_lane",
    ],
    "scoped_literal": "2027",
    "required_loop_files": [
        "src/models/inkling2.cpp",
        "src/models/inkling2-loop.cpp",
        "src/llama-roret.h",
    ],
    "min_loop_files": 3,
}
ENGINE_ROOTS = tuple(EXPECTED_ENGINE_SCOPE["roots"])
LOOP_PATH_MARKERS = tuple(EXPECTED_ENGINE_SCOPE["loop_path_markers"])
LOOP_CONTENT_MARKERS = tuple(EXPECTED_ENGINE_SCOPE["loop_content_markers"])
GLOBAL_FORBIDDEN_TOKENS = tuple(EXPECTED_ENGINE_SCOPE["global_forbidden_tokens"])
SCOPED_LITERAL = EXPECTED_ENGINE_SCOPE["scoped_literal"]
REQUIRED_LOOP_FILES = tuple(EXPECTED_ENGINE_SCOPE["required_loop_files"])
MIN_LOOP_FILES = EXPECTED_ENGINE_SCOPE["min_loop_files"]
DEFAULT_OFF_NEEDLES = (
    "Runtime gate: environment variable `LLAMA_ODSL`, enabled only for",
    "`ODSL_LAYOUT`, default OFF.",
    "Side-lane sub-gate: `LLAMA_ODSL_SIDE_LANE`, also default off.",
)
ENABLE_TOKENS = ("llama_odsl_side_lane", "llama_odsl", "odsl_layout")

EXPECTED_FACTS = {
    "schema_version": SCHEMA_VERSION,
    "frozen_on": FROZEN_ON,
    "scope": NOTE,
    "note": NOTE,
    "speed_claim": None,
    "classification": CLASSIFICATION,
    "real_trace_verdict": REAL_TRACE_VERDICT,
    "side_lane_enabled": SIDE_LANE_ENABLED,
    "provenance": EXPECTED_PROVENANCE,
    "hidden_size_info": EXPECTED_HIDDEN_SIZE_INFO,
    "source_roots": SOURCE_ROOTS,
    "source_sha256": FROZEN_SOURCE_SHA256,
    "needles": SOURCE_NEEDLES,
    "engine_scope": EXPECTED_ENGINE_SCOPE,
    "open_findings": OPEN_FINDINGS,
    "resolution": RESOLUTION,
}

NEEDLE_CHECK_NAMES = ["needle_" + name for name in sorted(SOURCE_NEEDLES)]
CHECK_NAMES = [
    "frozen_facts",
    "frozen_facts_digest",
    "frozen_provenance",
    "frozen_engine_scope",
    "frozen_classification",
    "frozen_hidden_size_info",
    "frozen_local_sha256",
    "frozen_needle_specs",
    "provenance_zero_hits",
    "provenance_self_consistency",
    "provenance_inline_verified",
    "open_findings_recorded",
    "live_local_sha256",
    "engine_inventory_nonempty",
    "engine_no_forbidden_token",
    "engine_no_d2027_literal",
    "side_lane_default_off",
    "no_network",
    "run_completed",
] + NEEDLE_CHECK_NAMES

NETWORK_MODULES = {"socket", "http", "urllib", "requests", "httpx", "ftplib", "smtplib", "telnetlib", "xmlrpc"}
NETWORK_CALLS = {"urlopen", "urlretrieve", "socket", "connect"}
HEX_DIGITS = frozenset("0123456789abcdef")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path):
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in HEX_DIGITS for char in value)


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def text_key(root_name, source):
    return root_name + ":" + source


def loopkit_root(repo_root):
    """Sibling loopkit by default; no network and no loopkit imports."""
    return (Path(repo_root).resolve().parent / "loopkit").resolve()


def source_roots(repo_root, lk_root):
    return {"repo": Path(repo_root).resolve(), "loopkit": Path(lk_root).resolve()}


def check_frozen_facts(facts):
    """Frozen JSON must match every module constant; inline bytes are gated separately."""
    if not isinstance(facts, dict):
        return False
    return {key: value for key, value in facts.items() if key != "inline_verification"} == EXPECTED_FACTS


def check_facts_digest(digest):
    return digest == FROZEN_FACTS_SHA256


def check_provenance(facts):
    if not isinstance(facts, dict):
        return False
    return facts.get("provenance") == EXPECTED_PROVENANCE


def check_engine_scope(facts):
    if not isinstance(facts, dict):
        return False
    return facts.get("engine_scope") == EXPECTED_ENGINE_SCOPE


def check_classification(facts):
    if not isinstance(facts, dict):
        return False
    return bool(
        facts.get("classification") == CLASSIFICATION
        and facts.get("real_trace_verdict") == REAL_TRACE_VERDICT
        and facts.get("side_lane_enabled") is SIDE_LANE_ENABLED
        and facts.get("speed_claim") is None
        and facts.get("scope") == NOTE
        and facts.get("note") == NOTE
    )


def check_hidden_size_info(facts):
    if not isinstance(facts, dict):
        return False
    info = facts.get("hidden_size_info")
    if info != EXPECTED_HIDDEN_SIZE_INFO:
        return False
    return any(record["url"] == info["source"] for record in EXPECTED_PROVENANCE)


def check_frozen_local_sha256(facts):
    if not isinstance(facts, dict):
        return False
    hashes = facts.get("source_sha256")
    return bool(
        isinstance(hashes, dict)
        and hashes == FROZEN_SOURCE_SHA256
        and set(hashes) == set(SOURCE_ROOTS)
        and all(is_sha256(value) for value in hashes.values())
    )


def needle_specs_valid(specs):
    if not isinstance(specs, dict) or set(specs) != set(SOURCE_NEEDLES):
        return False
    return all(
        isinstance(spec, dict)
        and spec.get("root") == SOURCE_ROOTS.get(spec.get("source"))
        and isinstance(spec.get("needles"), list)
        and bool(spec["needles"])
        and all(isinstance(needle, str) and needle for needle in spec["needles"])
        for spec in specs.values()
    )


def check_frozen_needle_specs(facts):
    if not isinstance(facts, dict):
        return False
    return needle_specs_valid(facts.get("needles"))


def check_provenance_zero_hits(facts):
    if not isinstance(facts, dict):
        return False
    records = facts.get("provenance")
    return bool(
        isinstance(records, list)
        and bool(records)
        and all(
            isinstance(record, dict) and record.get("occurrences_2027") == 0
            for record in records
        )
    )


def check_provenance_self_consistency(facts):
    """Structural check of the frozen records, independent of their literal values."""
    if not isinstance(facts, dict):
        return False
    records = facts.get("provenance")
    if not isinstance(records, list) or len(records) != 3:
        return False
    urls = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != PROVENANCE_KEYS:
            return False
        url = record["url"]
        size = record["byte_count"]
        digest = record["sha256"]
        hits = record["occurrences_2027"]
        if not isinstance(url, str) or not url.startswith("https://") or url in urls:
            return False
        urls.add(url)
        if not _positive_int(size) or not is_sha256(digest):
            return False
        if isinstance(hits, bool) or not isinstance(hits, int) or hits < 0:
            return False
    info = facts.get("hidden_size_info")
    return isinstance(info, dict) and info.get("source") in urls


def check_provenance_inline(facts):
    """Recompute length, sha256, zero-hit count and hidden_size from the frozen inline bytes."""
    try:
        inline = facts["inline_verification"]
        content = inline["content"]
        if inline["url"] != INLINE_VERIFICATION_URL:
            return False
        if not isinstance(content, str) or not content.isascii():
            return False
        data = content.encode("ascii")
        record = next(item for item in facts["provenance"] if item["url"] == inline["url"])
        config = json.loads(content)
        text_config = config.get("text_config", {})
        return bool(
            len(data) == record["byte_count"]
            and hashlib.sha256(data).hexdigest() == record["sha256"]
            and data.count(SCOPED_LITERAL.encode("ascii")) == record["occurrences_2027"]
            and config.get("model_type") == EXPECTED_HIDDEN_SIZE_INFO["model_type"]
            and text_config.get("hidden_size") == EXPECTED_HIDDEN_SIZE_INFO["hidden_size"]
            and text_config.get("num_hidden_layers") == EXPECTED_HIDDEN_SIZE_INFO["num_hidden_layers"]
        )
    except (KeyError, TypeError, ValueError, StopIteration, AttributeError):
        return False


def check_open_findings(facts):
    if not isinstance(facts, dict):
        return False
    findings = facts.get("open_findings")
    if not isinstance(findings, list) or not findings:
        return False
    if not all(isinstance(finding, str) and finding for finding in findings):
        return False
    return any(
        "real-trace" in finding.lower()
        and "weights" in finding
        and "hosted inference" in finding
        for finding in findings
    )


def check_live_local_hashes(live_hashes):
    return bool(live_hashes) and all(
        live_hashes.get(source) == digest for source, digest in FROZEN_SOURCE_SHA256.items()
    )


def check_needle(texts, spec):
    try:
        key = text_key(spec["root"], spec["source"])
        needles = spec["needles"]
    except (KeyError, TypeError):
        return False
    text = texts.get(key, "")
    return bool(text) and isinstance(needles, list) and bool(needles) and all(
        isinstance(needle, str) and needle in text for needle in needles
    )


def is_loop_source(rel, text):
    """Loop-specific engine file: path marker or Inkling2/RORET content marker."""
    low_rel = rel.lower()
    low_text = text.lower()
    if any(marker in low_rel for marker in LOOP_PATH_MARKERS):
        return True
    return any(marker in low_text for marker in LOOP_CONTENT_MARKERS)


def check_engine_inventory(loop_paths):
    return bool(loop_paths) and len(loop_paths) >= MIN_LOOP_FILES and all(
        path in loop_paths for path in REQUIRED_LOOP_FILES
    )


def scan_engine_texts(texts, loop_paths):
    """Pure scan: global tokens are forbidden anywhere; bare 2027 is gated by loop scope."""
    global_hits = []
    literal_hits = []
    for rel in sorted(texts):
        text = texts[rel]
        low = text.lower()
        for token in GLOBAL_FORBIDDEN_TOKENS:
            if token.lower() in low:
                global_hits.append({"path": rel, "token": token})
        if rel in loop_paths and SCOPED_LITERAL in text:
            literal_hits.append({"path": rel, "token": SCOPED_LITERAL})
    return {"global": global_hits, "literal": literal_hits}


def check_side_lane_default_off(texts):
    doc = texts.get(text_key("repo", "tools/odsl/LAYOUT_DESIGN.md"), "")
    return bool(doc) and all(needle in doc for needle in DEFAULT_OFF_NEEDLES)


def read_engine_texts(root):
    """Read every engine file as text; undecodable bytes still yield readable ASCII tokens."""
    texts = {}
    for engine in ENGINE_ROOTS:
        base = Path(root) / engine
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                try:
                    text = path.read_bytes().decode("utf-8", errors="replace")
                except OSError:
                    continue
                texts[path.relative_to(root).as_posix()] = text
    return texts


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
    root = Path(root).resolve()
    checks = {name: False for name in CHECK_NAMES}
    info = []
    live_hashes = {}
    engine_hits = {"global": [], "literal": []}
    frozen_facts_sha256 = None
    try:
        facts_path = root / FACTS_REL
        frozen_facts_sha256 = sha256_file(facts_path)
        facts = load_json(facts_path)
        checks["frozen_facts"] = check_frozen_facts(facts)
        checks["frozen_facts_digest"] = check_facts_digest(frozen_facts_sha256)
        checks["frozen_provenance"] = check_provenance(facts)
        checks["frozen_engine_scope"] = check_engine_scope(facts)
        checks["frozen_classification"] = check_classification(facts)
        checks["frozen_hidden_size_info"] = check_hidden_size_info(facts)
        checks["frozen_local_sha256"] = check_frozen_local_sha256(facts)
        checks["frozen_needle_specs"] = check_frozen_needle_specs(facts)
        checks["provenance_zero_hits"] = check_provenance_zero_hits(facts)
        checks["provenance_self_consistency"] = check_provenance_self_consistency(facts)
        checks["provenance_inline_verified"] = check_provenance_inline(facts)
        checks["open_findings_recorded"] = check_open_findings(facts)

        lk_root = loopkit_root(root)
        info.append("loopkit root ../loopkit")
        roots = source_roots(root, lk_root)
        texts = {}
        for source in sorted(SOURCE_ROOTS):
            path = roots[SOURCE_ROOTS[source]] / source
            if path.is_file():
                live_hashes[source] = sha256_file(path)
            texts[text_key(SOURCE_ROOTS[source], source)] = read_text(path)
        checks["live_local_sha256"] = check_live_local_hashes(live_hashes)

        for name, spec in sorted(facts.get("needles", {}).items()):
            check_name = "needle_" + name
            if check_name in checks:
                checks[check_name] = check_needle(texts, spec)

        engine_texts = read_engine_texts(root)
        loop_paths = {rel for rel, text in engine_texts.items() if is_loop_source(rel, text)}
        checks["engine_inventory_nonempty"] = check_engine_inventory(loop_paths)
        engine_hits = scan_engine_texts(engine_texts, loop_paths)
        checks["engine_no_forbidden_token"] = not engine_hits["global"]
        checks["engine_no_d2027_literal"] = not engine_hits["literal"]
        checks["side_lane_default_off"] = (
            check_side_lane_default_off(texts) and not engine_hits["global"]
        )
        checks["no_network"] = check_no_network()

        for record in EXPECTED_PROVENANCE:
            info.append(
                "provenance %s: %d bytes, sha256 %s, occurrences_2027 %d"
                % (record["url"], record["byte_count"], record["sha256"], record["occurrences_2027"])
            )
        info.append(
            "hidden_size %d with 24 layers and model_type inkling2_loop is INFO only, not d2027 evidence"
            % EXPECTED_HIDDEN_SIZE_INFO["hidden_size"]
        )
        if engine_hits["global"] or engine_hits["literal"]:
            info.append(
                "engine inventory: %d loop-specific files under src/common/ggml; "
                "%d forbidden-token hits and %d scoped 2027 hits recorded in engine_hits"
                % (len(loop_paths), len(engine_hits["global"]), len(engine_hits["literal"]))
            )
        else:
            info.append(
                "engine inventory: %d loop-specific files under src/common/ggml; no d2027 symbol, "
                "no scoped 2027 literal, no ODSL side-lane enable" % len(loop_paths)
            )
        info.append(OUT_OF_BAND_CAVEAT)
        info.append(
            "frozen facts sha256 %s (pinned, semantic facts and inline bytes are gated)"
            % frozen_facts_sha256
        )
        checks["run_completed"] = True
    except Exception as exc:
        info.append("exception: %s: %s" % (type(exc).__name__, exc))

    verdict = "PASS" if all(checks.values()) and len(checks) == len(CHECK_NAMES) else "NO_GO"
    blocker = "" if verdict == "PASS" else ", ".join(
        name for name in sorted(checks) if not checks[name]
    )
    summary = {
        "verdict": verdict,
        "blocker": blocker,
        "checks": checks,
        "info": info,
        "classification": CLASSIFICATION if verdict == "PASS" else "UNVERIFIED_GATE_FAILED",
        "real_trace_verdict": REAL_TRACE_VERDICT,
        "side_lane_enabled": any(
            hit["token"].lower() in ENABLE_TOKENS for hit in engine_hits["global"]
        ),
        "open_findings": list(OPEN_FINDINGS),
        "note": NOTE,
        "speed_claim": None,
        "frozen_on": FROZEN_ON,
        "frozen_facts_sha256": frozen_facts_sha256,
        "live_source_sha256": live_hashes,
        "engine_hits": engine_hits,
        "resolution": RESOLUTION if verdict == "PASS" else NO_GO_RESOLUTION,
    }
    write_json(root / SUMMARY_REL, summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    if summary["blocker"]:
        print("blocker: " + summary["blocker"])
    for name in sorted(summary["checks"]):
        print("  %-32s %s" % (name, "PASS" if summary["checks"][name] else "FAIL"))
    for line in summary["info"]:
        print("  INFO: " + line)
    for finding in summary["open_findings"]:
        print("  OPEN: " + finding)
    print("  classification: " + summary["classification"])
    print("  real_trace_verdict: " + summary["real_trace_verdict"])
    print("note: " + summary["note"])


def main(argv=None):
    summary = run_check(REPO_ROOT)
    print_summary(summary)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
