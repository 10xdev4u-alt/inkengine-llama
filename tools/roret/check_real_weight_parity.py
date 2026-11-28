#!/usr/bin/env python3
"""Fail-closed readiness gate for real-weight intermediate MoE parity.

The synthetic fixture is only for self-tests of this gate. No real weights are
held, so the gate returns NO_GO locally by design. NO SPEED CLAIM: this tool
measures correctness only. Any future number still requires a CycleAudit with
claim <= ci_low, repeats >= 20 and loops = 3.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from check_intermediate_parity import compare_share, sha256_file, version_string

SYNTHETIC_SHAS = (
    "cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617",
    "fff0e60fa4f55ba717d1e0bdc0e89695a0d35c85df680a15fd57ed1a0ac959ff",
)
MIN_SIZE = 10_000_000
REAL_ARCHES = frozenset(("inkling2", "inkling2_loop"))
MIN_EXPERTS = 16
REQUIRED_SHARES = ("0.0", "1.0")
HEX_DIGITS = frozenset("0123456789abcdef")
UNKNOWN_INFO = {
    "arch": "", "expert_count": None, "expert_used_count": None,
    "block_count": None,
}
SUMMARY_NAME = "M28_REAL_WEIGHT_SUMMARY.json"


def classify_weights(info):
    """Classify weights from plain data; fail closed on any missing evidence."""
    reasons = []
    if info.get("attested") is not True:
        reasons.append("--real-weights-attestation not set")
    if info.get("exists") is not True:
        reasons.append("GGUF file does not exist")
    sha = info.get("sha256", "")
    if not isinstance(sha, str) or len(sha) != 64 or not set(sha) <= HEX_DIGITS:
        reasons.append("GGUF sha256 missing or malformed")
    elif sha in SYNTHETIC_SHAS:
        reasons.append("GGUF sha256 is a frozen synthetic fixture")
    if not isinstance(info.get("size"), int) or info["size"] < MIN_SIZE:
        reasons.append("GGUF size below real-weight minimum")
    if info.get("arch", "") not in REAL_ARCHES:
        reasons.append("architecture not inkling2 or inkling2_loop")
    count = info.get("expert_count")
    if not isinstance(count, int) or count < MIN_EXPERTS:
        reasons.append("expert_count missing or below minimum")
    if reasons:
        return "synthetic_or_unknown", reasons
    return "real", []


def final_verdict(weights_class, parity_ok):
    """Only real weights plus clean parity can yield PASS."""
    if weights_class == "real" and parity_ok is True:
        return "PASS"
    return "NO_GO"


def shares_covered(cpu_shares):
    """Both historical cpu_share values are required for a PASS."""
    return set(REQUIRED_SHARES).issubset(set(cpu_shares))


def shares_ok(records, problems):
    """Mirror the M26/M27 per-share gate; empty records fail closed."""
    if not records or problems:
        return False
    return all(
        record.get("off_exit") == 0
        and record.get("on_exit") == 0
        and record.get("off_hook") is False
        and record.get("on_hook") is True
        and record.get("tensors", 0) > 0
        and record.get("bytes_compared", 0) > 0
        and record.get("differing_bytes", -1) == 0
        for record in records
    )


def field_value(field):
    """Return a scalar metadata value, or None when the field is unusable."""
    parts = getattr(field, "parts", None)
    if not parts:
        return None
    try:
        return parts[-1][0]
    except (IndexError, TypeError):
        return None


def field_int(reader, key):
    value = field_value(getattr(reader, "fields", {}).get(key))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def field_text(reader, key):
    parts = getattr(getattr(reader, "fields", {}).get(key), "parts", None)
    if not parts:
        return ""
    raw = parts[-1]
    try:
        data = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)
        return data.decode("utf-8", errors="replace").rstrip("\x00")
    except (TypeError, ValueError):
        return ""


def read_gguf_info(path):
    """Read Inkling2 metadata with vendored gguf-py; failures fail closed."""
    gguf_dir = REPO_ROOT / "gguf-py"
    if str(gguf_dir) not in sys.path:
        sys.path.insert(0, str(gguf_dir))
    try:
        import gguf
        reader = gguf.GGUFReader(str(path))
        arch = field_text(reader, "general.architecture")
        prefix = arch + "." if arch else ""
        return {
            "arch": arch,
            "expert_count": field_int(reader, prefix + "expert_count"),
            "expert_used_count": field_int(reader, prefix + "expert_used_count"),
            "block_count": field_int(reader, prefix + "block_count"),
        }
    except Exception:
        return dict(UNKNOWN_INFO)


def file_state(path):
    """Return (exists, sha256, size) with safe defaults for a missing file."""
    if not path.is_file():
        return False, "", 0
    return True, sha256_file(str(path)), path.stat().st_size


def parity_args(args):
    """Build the SimpleNamespace that compare_share reads."""
    return SimpleNamespace(
        llama_debug=args.llama_debug,
        model=args.gguf,
        prompt=args.prompt,
        n=args.n,
        ngl=args.ngl,
        threads=args.threads,
        expect_names=args.expect_names,
    )


def write_summary(out_dir, summary):
    path = out_dir / SUMMARY_NAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def report(summary, summary_path):
    for record in summary["shares"]:
        print(
            "share=%s off_exit=%d on_exit=%d off_hook=%s on_hook=%d tensors=%d bytes=%d diff=%d"
            % (
                record["cpu_share"], record["off_exit"], record["on_exit"],
                record["off_hook"], record["on_hook"], record["tensors"],
                record["bytes_compared"], record["differing_bytes"],
            )
        )
    for blocker in summary["blockers"]:
        print("BLOCKER: " + blocker)
    for problem in summary["problems"][:20]:
        print("FAIL: " + problem)
    if len(summary["problems"]) > 20:
        print("FAIL: %d more (see %s)" % (len(summary["problems"]) - 20, summary_path))
    print("OVERALL: %s" % summary["overall_verdict"])
    print("summary: " + str(summary_path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", required=True, help="path to a real Inkling2 GGUF")
    parser.add_argument("--llama-debug", required=True, help="path to llama-debug binary")
    parser.add_argument("--out", required=True, help="output directory for dumps and summary")
    parser.add_argument("--real-weights-attestation", action="store_true",
                        help="assert that --gguf is a real pretrained model")
    parser.add_argument("--expect-names", nargs="+", default=[],
                        help="tensor base names required in both OFF and ON manifests")
    parser.add_argument("--prompt", default="hello world")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--ngl", type=int, default=0, help="GPU layers; 0 keeps the CPU cmd")
    parser.add_argument("--threads", type=int, default=0, help="thread count; 0 leaves default")
    parser.add_argument("--cpu-share", nargs="+", default=["0.0", "1.0"])
    args = parser.parse_args(argv)
    if args.real_weights_attestation and not args.expect_names:
        parser.exit(1, "error: --expect-names is required with --real-weights-attestation\n")
    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gguf_path = Path(args.gguf)
    info = read_gguf_info(gguf_path)
    exists, sha, size = file_state(gguf_path)
    decision_info = {
        "attested": args.real_weights_attestation,
        "exists": exists,
        "sha256": sha,
        "size": size,
        "arch": info["arch"],
        "expert_count": info["expert_count"],
    }
    weights_class, blockers = classify_weights(decision_info)

    summary = {
        "gguf_path": args.gguf,
        "gguf_sha256": sha,
        "gguf_size": size,
        "gguf_metadata": info,
        "attested": args.real_weights_attestation,
        "weights_class": weights_class,
        "blockers": blockers,
        "shares": [],
        "problems": [],
        "parity_ok": False,
        "overall_verdict": "NO_GO",
    }

    # No engine run without real weights: the gate must stay NO_GO locally.
    if weights_class != "real":
        summary_path = write_summary(out_dir, summary)
        report(summary, summary_path)
        return 1

    # The documented evidence rule requires both historical cpu_share values.
    if not shares_covered(args.cpu_share):
        summary["problems"] = ["--cpu-share must include both 0.0 and 1.0"]
        summary_path = write_summary(out_dir, summary)
        report(summary, summary_path)
        return 1

    share_args = parity_args(args)
    problems = []
    version = version_string(args.llama_debug)
    if version is None:
        problems.append("llama-debug --version failed")
    else:
        summary["llama_debug_version"] = version
    for share in args.cpu_share:
        record, share_problems = compare_share(share, share_args, out_dir)
        summary["shares"].append(record)
        problems.extend(share_problems)

    summary["problems"] = problems
    summary["parity_ok"] = shares_ok(summary["shares"], problems)
    summary["overall_verdict"] = final_verdict(weights_class, summary["parity_ok"])
    summary_path = write_summary(out_dir, summary)
    report(summary, summary_path)
    return 0 if summary["overall_verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
