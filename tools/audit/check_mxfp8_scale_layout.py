#!/usr/bin/env python3
"""Offline, fail-closed MXFP8 expert scale-layout parity gate."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
FACTS_REL = "tools/audit/MXFP8_SCALE_HEADER.json"
SUMMARY_REL = "tools/audit/M34_MXFP8_SCALE_SUMMARY.json"
M33_FACTS_REL = "tools/audit/QUANT_LAYOUT.json"
FACTS_PATH = AUDIT_DIR / FACTS_REL
SUMMARY_PATH = AUDIT_DIR / SUMMARY_REL
M33_FACTS_PATH = AUDIT_DIR / "QUANT_LAYOUT.json"
NOTE = "MXFP8 scale-layout parity only; no timing or speed claim"
FROZEN_ON = "2026-11-28"
SCHEMA_VERSION = 1
CONVERTER_PY = "conversion/inkling2.py"
FROZEN_CONVERTER_SHA256 = "aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7"
CHECKPOINT_SCALE_PARITY = "VERIFIED"
RESOLUTION = "MXFP8 stored per-scale layout is (1,32) with 32 elements per scale, matching the converter; NVFP4 is (1,16); the published [32,32] tile is config metadata and is INFO only."

EXPECTED_PROVENANCE = {
    "url": "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/model-00001-of-00001.safetensors",
    "byte_range": "bytes=0-148415",
    "byte_start": 0,
    "byte_end": 148415,
    "fetched_bytes": 148416,
    "sha256": "a154b6ce2ead7032e9c0edcdcf020bf819a35f5078a36cce90938abe4d2c5247",
    "header_len": 148408,
    "n_tensors": 1245,
    "converter_path": CONVERTER_PY,
    "converter_sha256": FROZEN_CONVERTER_SHA256,
    "network_access_in_checker": False,
    "note": "The header bytes were fetched once out of band. This checker makes no network calls.",
}

_MXFP8_ENTRIES = {
    "w13_weight": {"dtype": "F8_E4M3", "shape": [128, 2304, 2304]},
    "w13_weight_scale": {"dtype": "U8", "shape": [128, 2304, 72]},
    "w2_weight": {"dtype": "F8_E4M3", "shape": [128, 2304, 1152]},
    "w2_weight_scale": {"dtype": "U8", "shape": [128, 2304, 36]},
}
_NVFP4_ENTRIES = {
    "w13_weight": {"dtype": "U8", "shape": [128, 2304, 1152]},
    "w13_weight_scale": {"dtype": "F8_E4M3", "shape": [128, 2304, 144]},
    "w13_weight_amax": {"dtype": "F32", "shape": [128]},
    "w13_weight_outer_scale_denominator": {"dtype": "F32", "shape": [1]},
    "w2_weight": {"dtype": "U8", "shape": [128, 2304, 576]},
    "w2_weight_scale": {"dtype": "F8_E4M3", "shape": [128, 2304, 72]},
    "w2_weight_amax": {"dtype": "F32", "shape": [128]},
    "w2_weight_outer_scale_denominator": {"dtype": "F32", "shape": [1]},
}

EXPECTED_FACTS = {
    "measured": {
        "mxfp8": {
            "layers": [20, 21, 22, 23],
            "layers_detail": {str(layer): json.loads(json.dumps(_MXFP8_ENTRIES)) for layer in (20, 21, 22, 23)},
            "w13_weight": {"dtype": "F8_E4M3", "shape": [128, 2304, 2304]},
            "w13_weight_scale": {"dtype": "U8", "shape": [128, 2304, 72]},
            "w2_weight": {"dtype": "F8_E4M3", "shape": [128, 2304, 1152]},
            "w2_weight_scale": {"dtype": "U8", "shape": [128, 2304, 36]},
            "extra_companions": False,
        },
        "nvfp4": {
            "layers": [19],
            "layers_detail": {"19": json.loads(json.dumps(_NVFP4_ENTRIES))},
            "w13_weight": {"dtype": "U8", "shape": [128, 2304, 1152]},
            "w13_weight_scale": {"dtype": "F8_E4M3", "shape": [128, 2304, 144]},
            "w13_weight_amax": {"dtype": "F32", "shape": [128]},
            "w13_weight_outer_scale_denominator": {"dtype": "F32", "shape": [1]},
            "w2_weight": {"dtype": "U8", "shape": [128, 2304, 576]},
            "w2_weight_scale": {"dtype": "F8_E4M3", "shape": [128, 2304, 72]},
            "w2_weight_amax": {"dtype": "F32", "shape": [128]},
            "w2_weight_outer_scale_denominator": {"dtype": "F32", "shape": [1]},
        },
    },
    "logical": {
        "experts": 128,
        "rows_w13": 2304,
        "rows_w2": 2304,
        "cols_w13": 2304,
        "cols_w2": 1152,
    },
    "derived": {
        "mxfp8": {"w13_elements_per_scale": 32, "w2_elements_per_scale": 32, "stored_block": [1, 32]},
        "nvfp4": {"w13_elements_per_scale": 16, "w2_elements_per_scale": 16, "stored_block": [1, 16]},
    },
    "converter": {
        "mxfp8_block": 32,
        "mxfp8_dequant_reshape": 32,
        "nvfp4_block": 16,
        "nvfp4_dequant_reshape": 16,
        "mxfp8_weight_dtype": "torch.float8_e4m3fn",
        "mxfp8_scale_dtype": "torch.uint8",
        "nvfp4_weight_dtype": "torch.uint8",
        "nvfp4_scale_dtype": "torch.float8_e4m3fn",
    },
    "published_config": {
        "source": "hf_quant_config.json",
        "weight_block_shape": [32, 32],
        "elements_per_scale": 1024,
        "status": "INFO_CONFIG_VS_CHECKPOINT",
    },
    "superseded_m33": {
        "source": M33_FACTS_REL,
        "status": "SUPERSEDED",
        "wrong_mxfp8_shapes": {
            "w13_weight_scale": [128, 2304, 144],
            "w2_weight_scale": [128, 2304, 72],
        },
        "wrong_elements_per_scale": 16,
        "checkpoint_scale_parity": "UNVERIFIED",
        "open_finding": "MXFP8 scale layout is OPEN: HF scale_cols=144 over 2304 elements gives 16 elements per scale; converter dequant groups a flat 32; published weight_block_shape [32,32] gives 1024 elements per scale. These three do not agree. Real checkpoint scale-layout parity is UNVERIFIED without weights or header inspection.",
        "reason": "The M33 MXFP8 scale shapes were mislabeled from NVFP4 layer 19. The MXFP8 header shows scale cols 72 and 36, which give 32 elements per scale.",
    },
}

SOURCE_NEEDLES = {
    "mxfp8_stored_block": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "block = (rows // ss[1], cols // ss[2])",
            "if block != (1, 16 if nvfp4 else 32):",
            'raise ValueError(f"Inkling2: {name}_scale: unsupported stored block shape {block}")',
        ],
    },
    "mxfp8_dequant_reshape": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "block_scale = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)",
            "values.reshape(values.shape[0], -1, 32)[:] *= block_scale[:, :, None]",
        ],
    },
    "nvfp4_dequant_reshape": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "values.reshape(values.shape[0], -1, 16)[:] *= block_scale[:, :, None]",
        ],
    },
    "weight_scale_dtypes": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "weight_dtype, scale_dtype = (torch.uint8, torch.float8_e4m3fn) if nvfp4 else (torch.float8_e4m3fn, torch.uint8)",
        ],
    },
    "packed_shape_divisibility": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "if min(shape) <= 0 or cols % (16 if nvfp4 else 32) or tuple(tensor.shape) != (experts, rows, cols // (2 if nvfp4 else 1)):",
        ],
    },
    "config_block_shape_warns": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            'if list(block) != config.get("weight_block_shape"):',
            'logger.warning("Inkling2: %s: hf_quant_config.json weight_block_shape %s conflicts with stored weight/scale shapes %s/%s; using %s", name, config.get("weight_block_shape"), tuple(tensor.shape), ss, list(block))',
        ],
    },
    "nvfp4_scale_format": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            'if nvfp4 and config.get("weight_scale_format") != "amax_denominator":',
        ],
    },
}

NEEDLE_CHECK_NAMES = ["needle_" + name for name in sorted(SOURCE_NEEDLES)]
CHECK_NAMES = [
    "frozen_facts",
    "frozen_provenance",
    "frozen_converter_sha256",
    "live_converter_sha256",
    "derived_elements_per_scale",
    "stored_block_parity",
    "converter_semantics",
    "published_config_info",
    "m33_superseded_detected",
    "no_network",
    "run_completed",
] + NEEDLE_CHECK_NAMES

NETWORK_MODULES = {"socket", "http", "urllib", "requests", "httpx", "ftplib", "smtplib", "telnetlib", "xmlrpc"}
NETWORK_CALLS = {"urlopen", "urlretrieve", "socket", "connect"}
PACKED_DIVISOR = {"mxfp8": 1, "nvfp4": 2}
REPRESENTATIVE_KEYS = {
    "mxfp8": sorted(_MXFP8_ENTRIES),
    "nvfp4": sorted(_NVFP4_ENTRIES),
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path):
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def derive_elements_per_scale(cols, scale_cols):
    """Logical weight elements covered by one stored scale."""
    if not _positive_int(cols) or not _positive_int(scale_cols):
        raise ValueError("cols and scale_cols must be positive integers")
    if cols % scale_cols:
        raise ValueError("scale_cols must divide cols")
    return cols // scale_cols


def check_frozen_facts(facts):
    """Frozen JSON must match every module constant."""
    if not isinstance(facts, dict):
        return False
    return bool(
        facts.get("schema_version") == SCHEMA_VERSION
        and facts.get("frozen_on") == FROZEN_ON
        and facts.get("scope") == NOTE
        and facts.get("note") == NOTE
        and facts.get("speed_claim") is None
        and facts.get("facts") == EXPECTED_FACTS
        and facts.get("needles") == SOURCE_NEEDLES
        and facts.get("provenance") == EXPECTED_PROVENANCE
        and facts.get("checkpoint_scale_parity") == CHECKPOINT_SCALE_PARITY
        and facts.get("resolution") == RESOLUTION
    )


def check_frozen_provenance(facts):
    if not isinstance(facts, dict):
        return False
    return facts.get("provenance") == EXPECTED_PROVENANCE


def check_frozen_converter_sha256(facts):
    if not isinstance(facts, dict):
        return False
    provenance = facts.get("provenance", {})
    return isinstance(provenance, dict) and provenance.get("converter_sha256") == FROZEN_CONVERTER_SHA256


def check_live_converter_sha256(root):
    path = root / CONVERTER_PY
    if not path.is_file():
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_CONVERTER_SHA256


def check_converter_needles(root):
    text = read_text(root / CONVERTER_PY)
    if not text:
        return False
    return all(
        spec.get("root") == "repo"
        and spec.get("source") == CONVERTER_PY
        and isinstance(spec.get("needles"), list)
        and bool(spec["needles"])
        and all(needle in text for needle in spec["needles"])
        for spec in SOURCE_NEEDLES.values()
    )


def derive_from_facts(facts):
    """Derive the stored scale layout from measured header shapes, not from a claim."""
    frozen = facts["facts"]
    logical = frozen["logical"]
    measured = frozen["measured"]
    claimed = frozen["derived"]
    derived = {}
    for precision in ("mxfp8", "nvfp4"):
        group = measured[precision]
        divisor = PACKED_DIVISOR[precision]
        if group["layers"] != ([20, 21, 22, 23] if precision == "mxfp8" else [19]):
            raise ValueError("unexpected layer list for " + precision)
        keys = REPRESENTATIVE_KEYS[precision]
        if set(group["layers_detail"]) != {str(layer) for layer in group["layers"]}:
            raise ValueError("layer detail must cover every measured layer")
        for detail in group["layers_detail"].values():
            if set(detail) != set(keys) or any(detail[key] != group[key] for key in keys):
                raise ValueError("layer detail disagrees with representative shapes")
        derived[precision] = {}
        for name in ("w13", "w2"):
            weight = group[name + "_weight"]
            scale = group[name + "_weight_scale"]
            rows = logical["rows_" + name]
            cols = logical["cols_" + name]
            experts = logical["experts"]
            if weight["shape"] != [experts, rows, cols // divisor]:
                raise ValueError("weight shape disagrees with logical shape")
            scale_shape = scale["shape"]
            if len(scale_shape) != 3 or scale_shape[0] != experts:
                raise ValueError("scale shape must start with experts")
            if rows % scale_shape[1] or cols % scale_shape[2]:
                raise ValueError("scale shape must divide logical shape")
            elements = derive_elements_per_scale(cols, scale_shape[2])
            block = [rows // scale_shape[1], elements]
            derived[precision][name] = {
                "elements_per_scale": elements,
                "stored_block": block,
                "weight_shape": weight["shape"],
                "scale_shape": scale_shape,
                "weight_dtype": weight["dtype"],
                "scale_dtype": scale["dtype"],
            }
        expected = claimed[precision]
        e13 = derived[precision]["w13"]["elements_per_scale"]
        e2 = derived[precision]["w2"]["elements_per_scale"]
        if expected["w13_elements_per_scale"] != e13 or expected["w2_elements_per_scale"] != e2:
            raise ValueError("claimed elements per scale disagree with measured shapes")
        if e13 != e2:
            raise ValueError("w13 and w2 use different scale groups")
        if expected["stored_block"] != derived[precision]["w13"]["stored_block"]:
            raise ValueError("claimed stored block disagrees with measured shapes")
        if expected["stored_block"] != derived[precision]["w2"]["stored_block"]:
            raise ValueError("w13 and w2 use different stored blocks")
    if group_extra(frozen):
        raise ValueError("MXFP8 must not carry amax or outer_scale_denominator")
    for name in ("w13", "w2"):
        for companion in ("amax", "outer_scale_denominator"):
            entry = measured["nvfp4"][name + "_weight_" + companion]
            if entry["dtype"] != "F32" or not entry["shape"]:
                raise ValueError("NVFP4 companion must be F32 with a shape")
    return derived


def group_extra(frozen):
    mx = frozen["measured"]["mxfp8"]
    return mx.get("extra_companions") is not False or any(
        "amax" in key or "denominator" in key for key in mx
    )


def check_derived_elements(facts, derived):
    return bool(
        derived
        and derived["mxfp8"]["w13"]["elements_per_scale"] == 32
        and derived["mxfp8"]["w2"]["elements_per_scale"] == 32
        and derived["nvfp4"]["w13"]["elements_per_scale"] == 16
        and derived["nvfp4"]["w2"]["elements_per_scale"] == 16
    )


def check_stored_block_parity(facts, derived):
    return bool(
        derived
        and derived["mxfp8"]["w13"]["stored_block"] == [1, 32]
        and derived["mxfp8"]["w2"]["stored_block"] == [1, 32]
        and derived["nvfp4"]["w13"]["stored_block"] == [1, 16]
        and derived["nvfp4"]["w2"]["stored_block"] == [1, 16]
        and facts["facts"]["converter"]["mxfp8_block"] == 32
        and facts["facts"]["converter"]["mxfp8_dequant_reshape"] == 32
        and facts["facts"]["converter"]["nvfp4_block"] == 16
        and facts["facts"]["converter"]["nvfp4_dequant_reshape"] == 16
    )


def check_converter_semantics(facts):
    frozen = facts["facts"]
    converter = frozen["converter"]
    measured = frozen["measured"]
    return bool(
        converter.get("mxfp8_weight_dtype") == "torch.float8_e4m3fn"
        and converter.get("mxfp8_scale_dtype") == "torch.uint8"
        and converter.get("nvfp4_weight_dtype") == "torch.uint8"
        and converter.get("nvfp4_scale_dtype") == "torch.float8_e4m3fn"
        and measured["mxfp8"]["w13_weight"]["dtype"] == "F8_E4M3"
        and measured["mxfp8"]["w13_weight_scale"]["dtype"] == "U8"
        and measured["nvfp4"]["w13_weight"]["dtype"] == "U8"
        and measured["nvfp4"]["w13_weight_scale"]["dtype"] == "F8_E4M3"
    )


def check_published_config(facts):
    published = facts["facts"]["published_config"]
    return bool(
        published.get("source") == "hf_quant_config.json"
        and published.get("weight_block_shape") == [32, 32]
        and published.get("elements_per_scale") == 1024
        and published.get("status") == "INFO_CONFIG_VS_CHECKPOINT"
    )


def check_m33_superseded(root, info):
    """Detect the frozen M33 wrong fact and record it as SUPERSEDED, never as a failure."""
    path = root / M33_FACTS_REL
    if not path.is_file():
        return False
    payload = load_json(path)
    mx = payload["facts"]["hf_shapes"]["mxfp8"]
    superseded = EXPECTED_FACTS["superseded_m33"]
    wrong = superseded["wrong_mxfp8_shapes"]
    if mx.get("w13_weight_scale") != wrong["w13_weight_scale"]:
        return False
    if mx.get("w2_weight_scale") != wrong["w2_weight_scale"]:
        return False
    if payload.get("checkpoint_scale_parity") != superseded["checkpoint_scale_parity"]:
        return False
    finding = " ".join(payload.get("open_findings", []))
    if "OPEN" not in finding or "UNVERIFIED" not in finding:
        return False
    if derive_elements_per_scale(2304, 144) != superseded["wrong_elements_per_scale"]:
        return False
    if derive_elements_per_scale(1152, 72) != superseded["wrong_elements_per_scale"]:
        return False
    info.append(
        "M33 QUANT_LAYOUT.json froze MXFP8 scale cols 144/72 (16 per scale) from NVFP4 layer 19; "
        "detected and marked SUPERSEDED (INFO, not a failure)"
    )
    return True


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
    derived = {}
    frozen_facts_sha256 = None
    converter_sha256_live = None
    try:
        facts_path = root / FACTS_REL
        frozen_facts_sha256 = hashlib.sha256(facts_path.read_bytes()).hexdigest()
        facts = load_json(facts_path)
        checks["frozen_facts"] = check_frozen_facts(facts)
        checks["frozen_provenance"] = check_frozen_provenance(facts)
        checks["frozen_converter_sha256"] = check_frozen_converter_sha256(facts)
        checks["live_converter_sha256"] = check_live_converter_sha256(root)
        converter_path = root / CONVERTER_PY
        if converter_path.is_file():
            converter_sha256_live = hashlib.sha256(converter_path.read_bytes()).hexdigest()
        text = read_text(converter_path)
        for name in sorted(SOURCE_NEEDLES):
            spec = SOURCE_NEEDLES[name]
            checks["needle_" + name] = bool(text) and all(n in text for n in spec["needles"])
        derived = derive_from_facts(facts)
        checks["derived_elements_per_scale"] = check_derived_elements(facts, derived)
        checks["stored_block_parity"] = check_stored_block_parity(facts, derived)
        checks["converter_semantics"] = check_converter_semantics(facts)
        checks["published_config_info"] = check_published_config(facts)
        checks["m33_superseded_detected"] = check_m33_superseded(root, info)
        checks["no_network"] = check_no_network()
        info.append("evidence: header bytes 0-148415, 148416 bytes, sha256 %s, 1245 tensors; checker makes no network calls" % EXPECTED_PROVENANCE["sha256"])
        info.append("MXFP8 elements per scale from measured header: w13 %d, w2 %d; NVFP4 contrast: w13 %d, w2 %d" % (
            derived["mxfp8"]["w13"]["elements_per_scale"], derived["mxfp8"]["w2"]["elements_per_scale"],
            derived["nvfp4"]["w13"]["elements_per_scale"], derived["nvfp4"]["w2"]["elements_per_scale"]))
        info.append("published hf_quant_config.json weight_block_shape [32,32] = 1024 elements per scale is config metadata; stored per-scale layout is (1,32); INFO only, not a blocker")
        info.append("stored MXFP8 block (1,32) matches converter check and dequant reshape; no conversion edit was needed")
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
        "derived": derived,
        "checkpoint_scale_parity": CHECKPOINT_SCALE_PARITY if verdict == "PASS" else "UNVERIFIED",
        "speed_claim": None,
        "note": NOTE,
        "frozen_on": FROZEN_ON,
        "frozen_facts_sha256": frozen_facts_sha256,
        "converter_sha256_live": converter_sha256_live,
        "resolution": RESOLUTION,
    }
    write_json(root / SUMMARY_REL, summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    if summary["blocker"]:
        print("blocker: " + summary["blocker"])
    for name in sorted(summary["checks"]):
        print("  %-34s %s" % (name, "PASS" if summary["checks"][name] else "FAIL"))
    for line in summary["info"]:
        print("  INFO: " + line)
    print("  checkpoint_scale_parity: " + summary["checkpoint_scale_parity"])
    print("note: " + summary["note"])


def main(argv=None):
    summary = run_check(REPO_ROOT)
    print_summary(summary)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
