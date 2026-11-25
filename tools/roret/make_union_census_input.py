#!/usr/bin/env python3
"""Build the fixture-derived top-k id include for the RORET union census.

Regenerates the deterministic synthetic fixture (seed 0) with
tools/audit/make_moe_fixture.py and the F32 dump with tools/audit/dump_f32.py,
then writes tools/roret/union_census_ids.inc with the real top-k ids of the
12 looped core layers (ip = 6..17) and passes 0, 1, 2.

The generated include is ASCII-only and byte-identical on every run; it embeds
only the ids the census needs, not the rest of the dump.

Usage:
    python3 tools/roret/make_union_census_input.py [--work-dir DIR]

    # optional: prefer loopkit's NVFP4 emulator over the byte-identical fallback
    PYTHONPATH=/path/to/loopkit/src python3 tools/roret/make_union_census_input.py

Printed hashes pin the provenance: fixture_sha256 and dump_sha256 are content
hashes (independent of .npz zip metadata); the *_file_sha256 lines are the
hashes of the regenerated .npz files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
AUDIT = ROOT / "tools" / "audit"
sys.path.insert(0, str(AUDIT))

import dump_f32 as du  # noqa: E402  (audit dir inserted above)
import make_moe_fixture as mk  # noqa: E402

SEED = 0
DEFAULT_WORK_DIR = Path("/tmp/opencode/m11-union-census")


def emit_include(ids: np.ndarray, meta: dict[str, str]) -> str:
    """Return the ASCII C++ include text for the core-layer top-k ids."""
    n_ip = mk.CORE_END - mk.CORE_START
    n_pass = mk.N_PASSES
    n_tok = mk.N_TOKENS
    top_k = mk.TOP_K

    expected = (mk.N_LAYERS, mk.N_PASSES, mk.N_TOKENS, mk.TOP_K)
    if ids.shape != expected:
        raise ValueError("top-k id shape %s != %s" % (ids.shape, expected))
    if ids.dtype.kind not in "iu":
        raise ValueError("top-k ids must be integers, got %s" % ids.dtype)
    if int(ids.min()) < 0 or int(ids.max()) >= mk.N_EXPERTS:
        raise ValueError("top-k id outside [0, %d)" % mk.N_EXPERTS)
    if (mk.CORE_START, mk.CORE_END, n_pass, top_k) != (6, 18, 3, 8):
        raise ValueError("unexpected fixture constants; update the C++ test")
    if n_ip != 12 or n_tok != 4:
        raise ValueError("unexpected core/token counts; update the C++ test")

    out = [
        "// Generated file - do not edit by hand.",
        "// Producer: tools/roret/make_union_census_input.py (seed 0).",
        "// Upstream: tools/audit/make_moe_fixture.py -> tools/audit/dump_f32.py.",
        "// fixture_sha256 " + meta["fixture_sha256"],
        "// dump_sha256 " + meta["dump_sha256"],
        "// Dump key: topk_ids, shape [24][3][4][8] = [L][P][T][K].",
        "// Only ip = %d..%d and pass = 0..2 are embedded." % (mk.CORE_START, mk.CORE_END - 1),
        "",
        "static constexpr int32_t kCensusNExperts = %d;" % mk.N_EXPERTS,
        "static constexpr int32_t kCensusTopK = %d;" % top_k,
        "static constexpr int32_t kCensusNPass = %d;" % n_pass,
        "static constexpr int32_t kCensusNTokens = %d;" % n_tok,
        "static constexpr int32_t kCensusNCoreLayers = %d;" % n_ip,
        "static constexpr int32_t kCensusIpFirst = %d;" % mk.CORE_START,
        "",
        "// [ip - kCensusIpFirst][pass][token][k]",
        "static constexpr int32_t kCensusIds[%d][%d][%d][%d] = {" % (n_ip, n_pass, n_tok, top_k),
    ]
    for c in range(n_ip):
        ip = mk.CORE_START + c
        out.append("    { // ip %d" % ip)
        for p in range(n_pass):
            out.append("        { // pass %d" % p)
            for t in range(n_tok):
                row = ids[ip, p, t]
                cells = ", ".join("%3d" % int(v) for v in row)
                out.append("            { %s }, // token %d" % (cells, t))
            out.append("        },")
        out.append("    },")
    out.append("};")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                    help="directory for the regenerated .npz files (default: %s)" % DEFAULT_WORK_DIR)
    ap.add_argument("--out", type=Path, default=HERE / "union_census_ids.inc",
                    help="generated C++ include (default: tools/roret/union_census_ids.inc)")
    args = ap.parse_args()

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    fixture_path = work / ("fixture-seed%d.npz" % SEED)
    dump_path = work / "dump-f32.npz"

    fixture = mk.build_fixture(SEED)
    mk.save_npz(str(fixture_path), fixture)
    dump = du.build_dump(fixture)
    mk.save_npz(str(dump_path), dump)

    ids = dump.get("topk_ids", dump.get("topk"))
    if ids is None:
        raise ValueError("dump has neither 'topk_ids' nor 'topk'")

    meta = {
        "fixture_sha256": mk.canonical_hash(fixture),
        "dump_sha256": mk.canonical_hash(dump),
    }
    text = emit_include(np.asarray(ids), meta)
    out = args.out.resolve()
    with open(out, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    text.encode("ascii")  # fail loudly if a non-ASCII byte slipped in

    print("fixture_sha256 " + meta["fixture_sha256"])
    print("fixture_file_sha256 " + mk.file_hash(str(fixture_path)))
    print("dump_sha256 " + meta["dump_sha256"])
    print("dump_file_sha256 " + mk.file_hash(str(dump_path)))
    print("nvfp4_source " + du.NVFP4_SOURCE)
    print("wrote " + str(out))
    print("ids_shape " + str(tuple(np.asarray(ids).shape)))


if __name__ == "__main__":
    main()
