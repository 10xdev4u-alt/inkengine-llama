#!/usr/bin/env python3
"""Offline, fail-closed NVFP4/MXFP8 scale layout and Unsloth quant contract check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
FACTS_REL = "tools/audit/QUANT_LAYOUT.json"
SUMMARY_REL = "tools/audit/M33_QUANT_LAYOUT_SUMMARY.json"
FACTS_PATH = AUDIT_DIR / "QUANT_LAYOUT.json"
SUMMARY_PATH = AUDIT_DIR / "M33_QUANT_LAYOUT_SUMMARY.json"
NOTE = "quant layout and Unsloth quant contract only, no timing"
FROZEN_ON = "2026-11-28"
SCHEMA_VERSION = 1

CONVERTER_PY = "conversion/inkling2.py"
INTEGRATION_MD = "docs/INKLING_INTEGRATION.md"
FORMAT_FACTS_MD = "docs/INKLING_FORMAT_FACTS.md"
SOURCE_ROOTS = {
    CONVERTER_PY: "repo",
    INTEGRATION_MD: "loopkit",
    FORMAT_FACTS_MD: "loopkit",
}

FROZEN_SOURCE_SHA256 = {
    CONVERTER_PY: "aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7",
    INTEGRATION_MD: "69c587aba7ebbe3b866b7ca95107cfb891b28b3b30d3c54eeba698f878655432",
    FORMAT_FACTS_MD: "46f134e263d1f9889a22e556428f7bdc6144fa1c9068072c31794ea19c4071df",
}
GATED_SOURCE_FILES = sorted(FROZEN_SOURCE_SHA256)

EXPECTED_FACTS = {
    "converter": {
        "schemes": ["nvfp4_nearest", "mxfp8_nearest"],
        "nvfp4_block": 16,
        "mxfp8_block": 32,
        "nvfp4_block_shape": [1, 16],
        "mxfp8_block_shape": [1, 32],
        "nvfp4_weight_dtype": "torch.uint8",
        "nvfp4_scale_dtype": "torch.float8_e4m3fn",
        "mxfp8_weight_dtype": "torch.float8_e4m3fn",
        "mxfp8_scale_dtype": "torch.uint8",
        "nvfp4_scale_format": "amax_denominator",
        "nvfp4_amax_shape": [128],
        "nvfp4_denominator_shape": [1],
        "mxfp8_extra_companions": False,
        "ue8m0_nan_byte": 255,
        "nvfp4_dequant_reshape": 16,
        "mxfp8_dequant_reshape": 32,
        "nvfp4_scale_table": "_E4M3",
        "mxfp8_scale_formula": "ldexp(1, scale - 127)",
        "published_nvfp4_activation_block": 16,
        "published_mxfp8_activation_block": 32,
        "published_mxfp8_weight_tile": [32, 32],
        "published_mxfp8_elements_per_scale": 1024,
        "published_mxfp8_derived_w13_scale_shape": [128, 72, 72],
        "published_mxfp8_verdict": "CONTRADICTED / INCOMPLETE",
    },
    "logical": {
        "experts": 128,
        "rows_w13": 2304,
        "rows_w2": 2304,
        "cols_w13": 2304,
        "cols_w2": 1152,
    },
    "hf_shapes": {
        "nvfp4": {
            "layers": [0, 19],
            "w13_weight": [128, 2304, 1152],
            "w2_weight": [128, 2304, 576],
            "w13_weight_scale": [128, 2304, 144],
            "w2_weight_scale": [128, 2304, 72],
            "w13_weight_amax": [128],
            "w13_weight_outer_scale_denominator": [1],
            "w2_weight_amax": [128],
            "w2_weight_outer_scale_denominator": [1],
        },
        "mxfp8": {
            "layers": [20, 23],
            "w13_weight": [128, 2304, 2304],
            "w2_weight": [128, 2304, 1152],
            "w13_weight_scale": [128, 2304, 144],
            "w2_weight_scale": [128, 2304, 72],
        },
    },
    "header": {
        "file": "Inkling-2-Mini-UD-IQ1_S.gguf",
        "repo": "unsloth/Inkling-2-Mini-GGUF",
        "url": "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-IQ1_S.gguf",
        "range": "bytes=0-16777215",
        "first_16mib_sha256": "87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76",
        "magic": "GGUF",
        "version": 3,
        "n_tensors": 744,
        "n_kv": 63,
        "kv": {
            "general.architecture": "inkling2_loop",
            "general.name": "Inkling-2-Mini",
            "general.file_type": 24,
            "general.file_type_name": "MOSTLY_IQ1_S except 1D tensors",
            "general.quantization_version": 2,
            "general.size_label": "128x1.8B",
        },
        "loop_tensor_count": 156,
        "loop_controllers": {
            "blk.6.loop_q_ctrl_down.weight": [4608, 64],
            "blk.6.loop_q_ctrl_gain.weight": [64, 2304],
            "blk.6.loop_q_ctrl_forget.weight": [64, 2304],
        },
        "loop_controller_dtype": "F32",
        "ffn_tensor_count": 220,
        "ffn_tensors": {
            "blk.0.ffn_gate_up_exps.weight": [2304, 2304, 128],
            "blk.0.ffn_gate_up_exps.weight.ggml_type": 19,
            "blk.0.ffn_gate_up_exps.weight.ggml_type_name": "IQ1_S",
            "blk.0.ffn_down_exps.weight": [1152, 2304, 128],
            "blk.0.ffn_down_exps.weight.ggml_type": 42,
            "blk.0.ffn_down_exps.weight.ggml_type_name": "Q2_0",
            "blk.0.ffn_gate_inp.weight": [2304, 128],
            "blk.0.ffn_gate_inp.weight.dtype": "F32",
            "ffn_expert_scale": [128],
            "ffn_expert_scale_dtype": "F32",
        },
        "ggml_type_counts": {
            "F32": 518,
            "Q8_0": 85,
            "BF16": 32,
            "Q6_K": 26,
            "Q2_0(42)": 23,
            "Q5_K": 16,
            "Q4_K": 14,
            "IQ2_XXS": 11,
            "IQ1_S": 6,
            "IQ2_XS": 5,
            "IQ3_XXS": 3,
            "IQ4_XS": 2,
            "IQ2_S": 2,
            "NVFP4(40)": 1,
        },
        "interpretation": "Mixed re-quant, not native NVFP4/MXFP8 expert tensors. Expert weights are IQ1_S/Q2_0; the arch string matches inkling2_loop; rank-64 loop controllers are F32 as frozen in M30.",
    },
}

SOURCE_NEEDLES = {
    "scheme_guard": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": ['if scheme not in ("nvfp4_nearest", "mxfp8_nearest"):'],
    },
    "nvfp4_scale_format": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": ['if nvfp4 and config.get("weight_scale_format") != "amax_denominator":'],
    },
    "mxfp8_no_amax": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            'raise ValueError(f"Inkling2: {name}: MXFP8 must not have amax or outer_scale_denominator")'
        ],
    },
    "weight_scale_dtypes": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "weight_dtype, scale_dtype = (torch.uint8, torch.float8_e4m3fn) if nvfp4 else (torch.float8_e4m3fn, torch.uint8)"
        ],
    },
    "packed_shape_divisibility": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "if min(shape) <= 0 or cols % (16 if nvfp4 else 32) or tuple(tensor.shape) != (experts, rows, cols // (2 if nvfp4 else 1)):"
        ],
    },
    "stored_block_shape": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "block = (rows // ss[1], cols // ss[2])",
            "if block != (1, 16 if nvfp4 else 32):",
            'raise ValueError(f"Inkling2: {name}_scale: unsupported stored block shape {block}")',
        ],
    },
    "config_block_shape_only_warns": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            'if list(block) != config.get("weight_block_shape"):',
            'logger.warning("Inkling2: %s: hf_quant_config.json weight_block_shape %s conflicts with stored weight/scale shapes %s/%s; using %s", name, config.get("weight_block_shape"), tuple(tensor.shape), ss, list(block))',
        ],
    },
    "nvfp4_dequant": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "block_scale = _E4M3[scales]",
            "values.reshape(values.shape[0], -1, 16)[:] *= block_scale[:, :, None]",
            "values *= np.float32(a) / np.float32(d)",
        ],
    },
    "mxfp8_dequant": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "block_scale = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)",
            "values.reshape(values.shape[0], -1, 32)[:] *= block_scale[:, :, None]",
        ],
    },
    "ue8m0_nan_byte": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "if np.any(scales == 255):",
            'raise ValueError(f"Inkling2: {name}_scale expert {expert}: UE8M0 byte 255 is NaN")',
        ],
    },
    "amax_denominator_shapes": {
        "root": "repo",
        "source": CONVERTER_PY,
        "needles": [
            "for companion, gen, expected in ((companions[1], amax, (experts,)), (companions[2], denominator, (1,))):"
        ],
    },
    "integration_hf_shapes": {
        "root": "loopkit",
        "source": INTEGRATION_MD,
        "needles": [
            "`w13_weight` [128,2304,1152], `w2_weight` [128,2304,576], `w13_weight_scale` [128,2304,144],",
            "`w13_weight` [128,2304,2304], `w2_weight` [128,2304,1152],",
        ],
    },
    "integration_scale_warning": {
        "root": "loopkit",
        "source": INTEGRATION_MD,
        "needles": [
            "Scale shape 144 = 2304/16 even for MXFP8",
            "do not assume a block size without checking converter dequant",
        ],
    },
    "integration_section4_item6": {
        "root": "loopkit",
        "source": INTEGRATION_MD,
        "needles": [
            "**NVFP4/MXFP8 dequant + scale layout (16 vs 32), and Unsloth quant choice**",
            "which quant an Unsloth GGUF actually uses (native NVFP4/MXFP8 vs",
        ],
    },
    "format_nvfp4_row_block": {
        "root": "loopkit",
        "source": FORMAT_FACTS_MD,
        "needles": [
            '"block_size": 16',
            '"weight_block_shape": [1, 16]',
            '"scale_dtype": "float8_e4m3fn"',
        ],
    },
    "format_mxfp8_2d_tile": {
        "root": "loopkit",
        "source": FORMAT_FACTS_MD,
        "needles": [
            '"block_size": 32',
            '"weight_block_shape": [32, 32]',
            "one scale per 1024 weight elements",
            '"scale_dtype": "ue8m0"',
        ],
    },
    "format_mxfp8_verdict": {
        "root": "loopkit",
        "source": FORMAT_FACTS_MD,
        "needles": [
            "`w13_weight_scale` ~ [128, 72, 72]",
            "CONTRADICTED / INCOMPLETE",
            "A flat 32-element grouping does not describe the checkpoint's weight scales.",
        ],
    },
}

OPEN_FINDINGS = [
    "MXFP8 scale layout is OPEN: HF scale_cols=144 over 2304 elements gives 16 elements per scale; converter dequant groups a flat 32; published weight_block_shape [32,32] gives 1024 elements per scale. These three do not agree. Real checkpoint scale-layout parity is UNVERIFIED without weights or header inspection.",
]
CHECKPOINT_SCALE_PARITY = "UNVERIFIED"

NEEDLE_CHECK_NAMES = ["needle_" + name for name in sorted(SOURCE_NEEDLES)]
DERIVED_CHECK_NAMES = [
    "converter_block_semantics",
    "hf_shape_divisibility",
    "elements_per_scale_derived",
    "header_facts",
    "header_type_counts_sum",
    "scale_layout_mismatch_open",
]
CHECK_NAMES = (
    ["frozen_facts", "frozen_source_sha256", "loopkit_present", "live_source_sha256"]
    + NEEDLE_CHECK_NAMES
    + DERIVED_CHECK_NAMES
    + ["run_completed"]
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
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


def count_ggml_types(type_counts):
    """Total tensor count from a frozen ggml type histogram."""
    if not isinstance(type_counts, dict) or not type_counts:
        raise ValueError("ggml type counts must be a nonempty dict")
    if not all(_positive_int(value) for value in type_counts.values()):
        raise ValueError("ggml type counts must be positive integers")
    return sum(type_counts.values())


def check_converter_block_semantics(facts):
    """NVFP4 uses [1,16] E4M3 scales; MXFP8 uses a flat [1,32] UE8M0 scale."""
    converter = facts.get("facts", {}).get("converter", {})
    expected = EXPECTED_FACTS["converter"]
    return bool(
        converter.get("nvfp4_block") == expected["nvfp4_block"] == 16
        and converter.get("mxfp8_block") == expected["mxfp8_block"] == 32
        and converter.get("nvfp4_block_shape") == [1, 16]
        and converter.get("mxfp8_block_shape") == [1, 32]
        and converter.get("nvfp4_weight_dtype") == "torch.uint8"
        and converter.get("nvfp4_scale_dtype") == "torch.float8_e4m3fn"
        and converter.get("mxfp8_weight_dtype") == "torch.float8_e4m3fn"
        and converter.get("mxfp8_scale_dtype") == "torch.uint8"
        and converter.get("nvfp4_scale_format") == "amax_denominator"
        and converter.get("mxfp8_extra_companions") is False
        and converter.get("ue8m0_nan_byte") == 255
        and converter.get("nvfp4_dequant_reshape") == 16
        and converter.get("mxfp8_dequant_reshape") == 32
    )


def check_group_shapes(group, logical, packed_divisor, companions):
    """Scale shapes must divide logical rows and cols, at 16 elements per scale."""
    if not isinstance(group, dict) or not isinstance(logical, dict):
        return False
    for name in ("w13", "w2"):
        rows = logical.get("rows_" + name)
        cols = logical.get("cols_" + name)
        weight = group.get(name + "_weight")
        scale = group.get(name + "_weight_scale")
        if not all(_positive_int(value) for value in (rows, cols)):
            return False
        if not isinstance(weight, list) or not isinstance(scale, list) or len(scale) != 3:
            return False
        if weight != [logical.get("experts"), rows, cols // packed_divisor]:
            return False
        if scale[0] != logical.get("experts") or rows % scale[1] or cols % scale[2]:
            return False
        try:
            if derive_elements_per_scale(cols, scale[2]) != 16:
                return False
        except ValueError:
            return False
        amax_name = name + "_weight_amax"
        den_name = name + "_weight_outer_scale_denominator"
        if companions:
            if group.get(amax_name) != [128] or group.get(den_name) != [1]:
                return False
        elif amax_name in group or den_name in group:
            return False
    return True


def check_hf_shape_facts(facts):
    """HF header shapes divide the logical dims and give 16 elements per scale."""
    frozen = facts.get("facts", {}) if isinstance(facts, dict) else {}
    logical = frozen.get("logical", {})
    shapes = frozen.get("hf_shapes", {})
    try:
        return bool(
            shapes.get("nvfp4", {}).get("layers") == [0, 19]
            and shapes.get("mxfp8", {}).get("layers") == [20, 23]
            and check_group_shapes(shapes.get("nvfp4", {}), logical, 2, True)
            and check_group_shapes(shapes.get("mxfp8", {}), logical, 1, False)
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def check_header_facts(facts):
    """Observed GGUF header facts: arch, type counts, loop controller shapes."""
    header = facts.get("facts", {}).get("header", {}) if isinstance(facts, dict) else {}
    expected = EXPECTED_FACTS["header"]
    return bool(
        header.get("file") == expected["file"]
        and header.get("repo") == expected["repo"]
        and header.get("url") == expected["url"]
        and header.get("range") == expected["range"]
        and header.get("first_16mib_sha256") == expected["first_16mib_sha256"]
        and header.get("magic") == "GGUF"
        and header.get("version") == 3
        and header.get("n_tensors") == 744
        and header.get("n_kv") == 63
        and header.get("kv") == expected["kv"]
        and header.get("loop_tensor_count") == 156
        and header.get("loop_controllers") == expected["loop_controllers"]
        and header.get("loop_controller_dtype") == "F32"
        and header.get("ffn_tensor_count") == 220
        and header.get("ffn_tensors") == expected["ffn_tensors"]
        and header.get("ggml_type_counts") == expected["ggml_type_counts"]
        and header.get("interpretation") == expected["interpretation"]
    )


def check_frozen_facts(facts):
    """Frozen JSON must match every module constant byte for semantic content."""
    if not isinstance(facts, dict):
        return False
    provenance = facts.get("provenance", {})
    return bool(
        facts.get("schema_version") == SCHEMA_VERSION
        and facts.get("frozen_on") == FROZEN_ON
        and facts.get("scope") == NOTE
        and facts.get("note") == NOTE
        and facts.get("speed_claim") is None
        and facts.get("facts") == EXPECTED_FACTS
        and facts.get("needles") == SOURCE_NEEDLES
        and facts.get("open_findings") == OPEN_FINDINGS
        and facts.get("checkpoint_scale_parity") == CHECKPOINT_SCALE_PARITY
        and isinstance(provenance, dict)
        and provenance.get("source_sha256") == FROZEN_SOURCE_SHA256
    )


def check_frozen_sha256(facts):
    """Frozen source hashes in the JSON must match the checker constants."""
    if not isinstance(facts, dict):
        return False
    provenance = facts.get("provenance", {})
    return isinstance(provenance, dict) and provenance.get("source_sha256") == FROZEN_SOURCE_SHA256


def derive_elements_from_facts(facts):
    """Compute elements per scale from HF shapes and logical dims, not from a claim."""
    frozen = facts["facts"]
    logical = frozen["logical"]
    shapes = frozen["hf_shapes"]
    derived = {}
    for precision in ("nvfp4", "mxfp8"):
        for name in ("w13", "w2"):
            scale = shapes[precision][name + "_weight_scale"]
            cols = logical["cols_" + name]
            derived[precision + "_" + name] = derive_elements_per_scale(cols, scale[2])
    return derived


def check_scale_layout_open(facts, derived):
    """The three scale layouts must stay visibly different, never declared equal."""
    converter = facts["facts"]["converter"]
    tile = converter["published_mxfp8_elements_per_scale"]
    flat = converter["mxfp8_block"]
    hf = derived["mxfp8_w13"]
    finding = " ".join(facts.get("open_findings", []))
    return bool(
        hf != flat and flat != tile and hf != tile
        and facts.get("checkpoint_scale_parity") == CHECKPOINT_SCALE_PARITY
        and all(token in finding for token in ("16", "32", "1024", "UNVERIFIED"))
    )


def loopkit_root(repo_root=REPO_ROOT):
    """Sibling loopkit by default, LOOPKIT_ROOT overrides; never fetch the network."""
    override = os.environ.get("LOOPKIT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (repo_root.parent / "loopkit").resolve()


def source_roots(repo_root, lk_root):
    return {"repo": Path(repo_root), "loopkit": Path(lk_root)}


def run_check(root):
    checks = {name: False for name in CHECK_NAMES}
    info = []
    provenance = {}
    frozen_on = None
    frozen_facts_sha256 = None
    derived = {}
    live = {}
    try:
        facts_path = root / FACTS_REL
        frozen_facts_sha256 = hashlib.sha256(facts_path.read_bytes()).hexdigest()
        facts = load_json(facts_path)
        frozen_on = facts["frozen_on"]
        provenance = dict(facts.get("provenance", {}))
        checks["frozen_facts"] = check_frozen_facts(facts)
        checks["frozen_source_sha256"] = check_frozen_sha256(facts)

        lk_root = loopkit_root(root)
        checks["loopkit_present"] = lk_root.is_dir()
        info.append("loopkit root %s (%s)" % (lk_root, "env" if os.environ.get("LOOPKIT_ROOT") else "default"))
        roots = source_roots(root, lk_root)
        for rel in GATED_SOURCE_FILES:
            path = roots[SOURCE_ROOTS[rel]] / rel
            if path.is_file():
                live[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            info.append("sha256 %s frozen=%s live=%s (gated)"
                        % (rel, FROZEN_SOURCE_SHA256[rel], live.get(rel, "missing")))
        provenance["source_sha256_live"] = live
        checks["live_source_sha256"] = all(
            live.get(rel) == digest for rel, digest in FROZEN_SOURCE_SHA256.items()
        )
        info.append("frozen facts sha256 %s (INFO only; semantic facts are gated)" % frozen_facts_sha256)

        texts = {rel: read_text(roots[SOURCE_ROOTS[rel]] / rel) for rel in GATED_SOURCE_FILES}
        for name in sorted(SOURCE_NEEDLES):
            spec = SOURCE_NEEDLES[name]
            text = texts.get(spec["source"], "")
            checks["needle_" + name] = bool(text) and all(n in text for n in spec["needles"])

        checks["converter_block_semantics"] = check_converter_block_semantics(facts)
        checks["hf_shape_divisibility"] = check_hf_shape_facts(facts)
        checks["header_facts"] = check_header_facts(facts)
        header = facts["facts"]["header"]
        try:
            checks["header_type_counts_sum"] = count_ggml_types(header["ggml_type_counts"]) == header["n_tensors"]
        except (KeyError, TypeError, ValueError):
            checks["header_type_counts_sum"] = False
        derived = derive_elements_from_facts(facts)
        checks["elements_per_scale_derived"] = (
            derived["nvfp4_w13"] == 16 and derived["nvfp4_w2"] == 16
            and derived["mxfp8_w13"] == 16 and derived["mxfp8_w2"] == 16
        )
        checks["scale_layout_mismatch_open"] = check_scale_layout_open(facts, derived)
        info.append("elements per scale from HF shapes: nvfp4 %d/%d, mxfp8 %d/%d"
                    % (derived["nvfp4_w13"], derived["nvfp4_w2"],
                       derived["mxfp8_w13"], derived["mxfp8_w2"]))
        info.append("converter flat blocks 16/32; published MXFP8 tile 32x32 = 1024 elements per scale")
        info.append("header probe: one bounded range request was recorded; checker makes no network calls")
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
        "open_findings": list(OPEN_FINDINGS),
        "checkpoint_scale_parity": CHECKPOINT_SCALE_PARITY,
        "provenance": provenance,
        "frozen_source_sha256": FROZEN_SOURCE_SHA256,
        "frozen_facts_sha256": frozen_facts_sha256,
        "frozen_on": frozen_on,
        "note": NOTE,
        "speed_claim": None,
    }
    write_json(root / SUMMARY_REL, summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    if summary["blocker"]:
        print("blocker: " + summary["blocker"])
    for name in sorted(summary["checks"]):
        print("  %-36s %s" % (name, "PASS" if summary["checks"][name] else "FAIL"))
    for line in summary["info"]:
        print("  INFO: " + line)
    for finding in summary["open_findings"]:
        print("  OPEN: " + finding)
    print("  checkpoint_scale_parity: " + summary["checkpoint_scale_parity"])
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
