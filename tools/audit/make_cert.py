#!/usr/bin/env python3
"""Freeze the bit-exact baseline certificate for the synthetic MoE fixture.

Drives the real loopkit CLI for certify, nvfp4-roundtrip and fdm-check, adds a
fail-closed negative control, and writes a deterministic FREEZE.json. This is
plumbing validation only: no speed claim is made or implied.

Usage:
    python3 tools/audit/make_cert.py --loopkit-src <abs path to loopkit/src> \
        --out <output dir> [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dump_f32 as du  # noqa: E402  (script dir inserted above)
import make_moe_fixture as mk  # noqa: E402

MODEL = "inkling2mini"
SUBPROCESS_TIMEOUT = 60
EXPECTED_PASSES = mk.N_PASSES
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# The CLI loads a whole directory of pass*.npy as ONE state, so its cert.json
# always reports n_passes=1. Loopkit's own certify_loop takes one state per
# pass; this snippet runs it under PYTHONPATH=<loopkit-src> to get the 3-pass
# certificate documented in AUDIT_PROTOCOL.md section 7.
LOOP_CERT_CODE = r"""
import dataclasses
import json
import sys
import numpy as np
from loopkit.bitloop import certify_loop

def load_states(directory, n_passes):
    states = []
    for p in range(n_passes):
        rec = np.load(directory + "/pass" + str(p) + ".npy", allow_pickle=False)
        states.append({k: np.asarray(rec[k]) for k in rec.dtype.names})
    return states

cert = certify_loop(load_states(sys.argv[1], int(sys.argv[4])),
                    load_states(sys.argv[2], int(sys.argv[4])),
                    model=sys.argv[3])
print(json.dumps(dataclasses.asdict(cert)))
"""

PASS_STRING_RE = re.compile(
    r"PassCertificate\(pass_index=(?P<pass_index>\d+), layer='(?P<layer>[^']*)', "
    r"tensor='(?P<tensor>[^']*)', n_elements=(?P<n_elements>\d+), "
    r"n_mismatch=(?P<n_mismatch>\d+), max_abs_err=(?P<max_abs_err>[^,]+), "
    r"exact=(?P<exact>True|False), digest='(?P<digest>[0-9a-f]{64})'\)"
)


def _environment(src: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def _run(cmd: list[str], *, cwd: Path, src: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), env=_environment(src),
                          capture_output=True, text=True,
                          timeout=SUBPROCESS_TIMEOUT)
    return proc.returncode, proc.stdout, proc.stderr


def _cli(args: list[str], *, cwd: Path, src: Path) -> tuple[int, str, str]:
    return _run([sys.executable, "-m", "loopkit.cli", *args], cwd=cwd, src=src)


def _stdout_json(stdout: str, label: str) -> dict:
    # A failing certify prints its JSON verdict, then a plain-text assertion
    # line. Parse the first line; the trailing text is not part of the verdict.
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{label}: empty stdout")
    try:
        parsed = json.loads(lines[0])
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"{label}: cannot parse stdout as JSON: {stdout!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label}: stdout JSON is not an object: {lines[0]!r}")
    return parsed


def _parse_cert_json(path: Path) -> dict:
    """Read the cert.json written by the CLI and extract per-pass digests."""
    data = json.loads(path.read_text(encoding="utf-8"))
    passes = []
    for entry in data.get("passes", []):
        if isinstance(entry, dict):
            passes.append(entry)
            continue
        match = PASS_STRING_RE.match(str(entry))
        if match is None:
            raise RuntimeError(f"cannot parse pass certificate in {path}: {entry!r}")
        found = match.groupdict()
        passes.append({
            "pass_index": int(found["pass_index"]),
            "layer": found["layer"],
            "tensor": found["tensor"],
            "n_elements": int(found["n_elements"]),
            "n_mismatch": int(found["n_mismatch"]),
            "max_abs_err": float(found["max_abs_err"]),
            "exact": found["exact"] == "True",
            "digest": found["digest"],
        })
    return {
        "model": data["model"],
        "n_passes": int(data["n_passes"]),
        "all_exact": bool(data["all_exact"]),
        "root_digest": data["root_digest"],
        "passes": passes,
    }


def _save_pass_state(path: Path, dump: dict, pass_index: int) -> None:
    """Pack every dump array for one pass into a single structured .npy file."""
    sliced = {key: np.ascontiguousarray(np.asarray(value)[:, pass_index, ...])
              for key, value in sorted(dump.items())}
    dtype = [(key, value.dtype, value.shape) for key, value in sliced.items()]
    record = np.zeros((), dtype=dtype)
    for key, value in sliced.items():
        record[key] = value
    np.save(path, record, allow_pickle=False)


def _loop_certificate(src: Path, refdir: Path, canddir: Path, *,
                      cwd: Path) -> dict:
    cmd = [sys.executable, "-c", LOOP_CERT_CODE, str(refdir), str(canddir),
           MODEL, str(EXPECTED_PASSES)]
    code, stdout, stderr = _run(cmd, cwd=cwd, src=src)
    if code != 0:
        raise RuntimeError(f"loopkit certify_loop failed (exit {code}): {stderr}")
    return _stdout_json(stdout, "certify_loop")


def _fixture_record(out: Path, name: str, seed: int, fixture: dict) -> dict:
    npz_path = out / f"{name}_fixture.npz"
    mk.save_npz(str(npz_path), fixture)
    file_sha256 = mk.file_hash(str(npz_path))
    manifest_path = out / f"{name}_fixture.npz.manifest.json"
    manifest = mk.write_manifest(str(manifest_path), fixture, seed, file_sha256)
    return {
        "npz": npz_path.name,
        "manifest": manifest_path.name,
        "canonical_sha256": manifest["fixture_sha256"],
        "file_sha256": file_sha256,
        "_fixture": fixture,
    }


def _negative_control(out: Path, src: Path) -> dict:
    neg = out / "negative_control"
    neg.mkdir(exist_ok=True)
    shutil.copytree(out / "ref", neg / "ref", dirs_exist_ok=True)
    shutil.copytree(out / "cand", neg / "cand", dirs_exist_ok=True)

    flipped = neg / "cand" / f"pass{1}.npy"
    record = np.load(flipped, allow_pickle=False).copy()
    raw = np.ascontiguousarray(record["layer_output"]).view(np.uint8)
    raw.flat[0] ^= 1
    np.save(flipped, record, allow_pickle=False)

    code, stdout, stderr = _cli(
        ["certify", "--ref", str(neg / "ref"), "--cand", str(neg / "cand"),
         "--model", MODEL], cwd=neg, src=src)
    printed = _stdout_json(stdout, "negative control certify")
    cert = _parse_cert_json(neg / "cert.json")
    if code == 0 or printed.get("all_exact") is True or cert["all_exact"] is True:
        raise RuntimeError(
            "negative control FAILED CLOSED: flipped pass 1 yet certify "
            f"returned exit {code} and all_exact={cert['all_exact']}")
    mismatch = sum(int(p["n_mismatch"]) for p in cert["passes"])

    three_pass = _loop_certificate(src, neg / "ref", neg / "cand", cwd=neg)
    bad_passes = [p for p in three_pass["passes"] if not p["exact"]]
    if three_pass["all_exact"] or not any(
            p["pass_index"] == 1 and p["n_mismatch"] > 0 for p in bad_passes):
        raise RuntimeError(
            "negative control 3-pass certificate did not detect the flip: "
            f"{three_pass}")
    return {
        "all_exact": cert["all_exact"],
        "exit_code": code,
        "root_digest": cert["root_digest"],
        "n_mismatch": mismatch,
        "flip": "cand/pass1.npy layer_output byte 0 XOR 1",
        "three_pass": {
            "all_exact": three_pass["all_exact"],
            "root_digest": three_pass["root_digest"],
            "n_mismatch": sum(int(p["n_mismatch"]) for p in three_pass["passes"]),
            "passes": three_pass["passes"],
        },
    }


def make_freeze(out: Path, src: Path, seed: int) -> dict:
    """Build fixtures, drive the real CLI, and write FREEZE.json."""
    out = out.resolve()
    src = src.resolve()
    if not (src / "loopkit" / "cli.py").is_file():
        raise RuntimeError(f"loopkit CLI not found under {src}")
    out.mkdir(parents=True, exist_ok=True)

    records = {name: _fixture_record(out, name, seed, mk.build_fixture(seed))
               for name in ("ref", "cand")}
    dumps = {name: du.build_dump(records[name]["_fixture"])
             for name in ("ref", "cand")}
    dump_sha256 = {name: mk.canonical_hash(dumps[name]) for name in ("ref", "cand")}

    for name in ("ref", "cand"):
        state_dir = out / name
        state_dir.mkdir(exist_ok=True)
        for p in range(EXPECTED_PASSES):
            _save_pass_state(state_dir / f"pass{p}.npy", dumps[name], p)
    np.save(out / "x.npy", records["ref"]["_fixture"]["x"], allow_pickle=False)

    cert = _loop_certificate(src, out / "ref", out / "cand", cwd=out)
    if (cert["n_passes"] != EXPECTED_PASSES or cert["all_exact"] is not True
            or not HEX64.match(cert["root_digest"])):
        raise RuntimeError(f"3-pass certificate failed: {cert}")

    code, stdout, stderr = _cli(
        ["certify", "--ref", str(out / "ref"), "--cand", str(out / "cand"),
         "--model", MODEL], cwd=out, src=src)
    printed = _stdout_json(stdout, "certify")
    if code != 0 or printed.get("all_exact") is not True:
        raise RuntimeError(f"certify failed (exit {code}): {stdout} {stderr}")
    cli_cert = _parse_cert_json(out / "cert.json")
    if cli_cert["root_digest"] != printed.get("root_digest"):
        raise RuntimeError("cert.json root_digest disagrees with CLI stdout")

    negative = _negative_control(out, src)

    code, stdout, stderr = _cli(
        ["nvfp4-roundtrip", "--file", "x.npy", "--block", "16"], cwd=out, src=src)
    nvfp4 = _stdout_json(stdout, "nvfp4-roundtrip")
    if code != 0 or nvfp4.get("lossless") is not True:
        raise RuntimeError(f"nvfp4-roundtrip failed (exit {code}): {stdout} {stderr}")

    code, stdout, stderr = _cli(["fdm-check"], cwd=out, src=src)
    fdm = _stdout_json(stdout, "fdm-check")
    if code != 0 or fdm.get("max_abs_err_vs_ref") != 0:
        raise RuntimeError(f"fdm-check failed (exit {code}): {stdout} {stderr}")

    fixture_info = {name: {k: v for k, v in records[name].items()
                           if not k.startswith("_")}
                    for name in ("ref", "cand")}
    freeze = {
        "generator": "make_cert.py",
        "seed": int(seed),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "loopkit_src": "loopkit/src",
        "model": MODEL,
        "fixture": fixture_info,
        "dump_sha256": dump_sha256,
        "cert": {k: cert[k] for k in ("model", "n_passes", "all_exact",
                                      "root_digest", "passes")},
        "cli_cert": cli_cert,
        "negative_control": negative,
        "nvfp4": {
            "lossless": bool(nvfp4["lossless"]),
            "ratio": nvfp4["ratio"],
            "ans_bytes": nvfp4["ans_bytes"],
            "tensor_bytes": nvfp4["tensor_bytes"],
        },
        "fdm": {
            "path": fdm["path"],
            "max_abs_err_vs_ref": fdm["max_abs_err_vs_ref"],
        },
        "notes": [
            "cli certify bundles all pass*.npy files into one state (n_passes=1)",
            "cert is loopkit.bitloop.certify_loop over the same three pass states",
            "negative control proves a single flipped bit invalidates the certificate",
            "no speedup is claimed or measured",
        ],
    }
    (out / "FREEZE.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return freeze


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loopkit-src", required=True, help="absolute path to loopkit/src")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed (default: 0)")
    args = ap.parse_args(argv)

    freeze = make_freeze(Path(args.out), Path(args.loopkit_src), args.seed)
    summary = {
        "all_exact": freeze["cert"]["all_exact"],
        "n_passes": freeze["cert"]["n_passes"],
        "root_digest": freeze["cert"]["root_digest"],
        "cli_n_passes": freeze["cli_cert"]["n_passes"],
        "ref_fixture_sha256": freeze["fixture"]["ref"]["canonical_sha256"],
        "cand_fixture_sha256": freeze["fixture"]["cand"]["canonical_sha256"],
        "negative_control_all_exact": freeze["negative_control"]["all_exact"],
        "negative_control_n_mismatch": freeze["negative_control"]["n_mismatch"],
        "nvfp4_lossless": freeze["nvfp4"]["lossless"],
        "nvfp4_ratio": freeze["nvfp4"]["ratio"],
        "fdm_max_abs_err_vs_ref": freeze["fdm"]["max_abs_err_vs_ref"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
