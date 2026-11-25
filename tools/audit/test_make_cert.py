"""Public-CLI tests for the bit-exact baseline freeze producer.

Each test drives make_cert.py through subprocess, exactly as an operator would.
Run: python -m pytest -q tools/audit/test_make_cert.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PRODUCER = HERE / "make_cert.py"
DEFAULT_LOOPKIT_SRC = HERE.parents[2] / "loopkit" / "src"
LOOPKIT_SRC = Path(os.environ.get("LOOPKIT_SRC", DEFAULT_LOOPKIT_SRC)).resolve()
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TIMEOUT = 120

pytestmark = pytest.mark.skipif(
    not (LOOPKIT_SRC / "loopkit" / "cli.py").is_file(),
    reason="loopkit src not found; set LOOPKIT_SRC to loopkit/src",
)


def _produce(out: Path) -> tuple[subprocess.CompletedProcess, dict]:
    proc = subprocess.run(
        [sys.executable, str(PRODUCER), "--loopkit-src", str(LOOPKIT_SRC),
         "--out", str(out), "--seed", "0"],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    assert proc.returncode == 0, f"producer failed:\n{proc.stdout}\n{proc.stderr}"
    freeze = json.loads((out / "FREEZE.json").read_text(encoding="ascii"))
    return proc, freeze


@pytest.fixture(scope="module")
def first(tmp_path_factory):
    out = tmp_path_factory.mktemp("freeze-first")
    proc, freeze = _produce(out)
    return out, proc, freeze


@pytest.fixture(scope="module")
def second(tmp_path_factory):
    out = tmp_path_factory.mktemp("freeze-second")
    proc, freeze = _produce(out)
    return out, proc, freeze


def test_producer_exits_zero_and_freezes_three_pass_cert(first):
    out, proc, freeze = first
    summary = json.loads(proc.stdout.strip())
    assert summary["all_exact"] is True
    cert = freeze["cert"]
    assert cert["model"] == "inkling2mini"
    assert cert["n_passes"] == 3
    assert cert["all_exact"] is True
    assert HEX64.match(cert["root_digest"])
    assert len(cert["passes"]) == 3
    assert [p["pass_index"] for p in cert["passes"]] == [0, 1, 2]
    assert all(HEX64.match(p["digest"]) and p["exact"] is True
               for p in cert["passes"])
    # Independently recompute loopkit's documented root: sha256 over the
    # concatenation of the three pass digests, not a self-reported constant.
    joined = "".join(p["digest"] for p in cert["passes"]).encode("ascii")
    assert hashlib.sha256(joined).hexdigest() == cert["root_digest"]

    cli_cert = freeze["cli_cert"]
    written = json.loads((out / "cert.json").read_text(encoding="utf-8"))
    assert cli_cert["root_digest"] == written["root_digest"]
    assert HEX64.match(cli_cert["root_digest"])
    # The CLI loads a directory as one state; the 3-pass certificate is
    # produced by loopkit.bitloop.certify_loop instead.
    assert cli_cert["n_passes"] == 1


def test_negative_control_fails_closed(first):
    _, _, freeze = first
    negative = freeze["negative_control"]
    assert negative["all_exact"] is False
    assert negative["exit_code"] == 1
    assert negative["n_mismatch"] > 0
    assert HEX64.match(negative["root_digest"])
    three_pass = negative["three_pass"]
    assert three_pass["all_exact"] is False
    assert three_pass["n_mismatch"] == 1  # exactly one bit flipped on pass 1
    assert three_pass["root_digest"] != freeze["cert"]["root_digest"]
    # A single flipped bit must invalidate pass 1 and leave the other two exact.
    bad = [p for p in three_pass["passes"] if not p["exact"]]
    assert [p["pass_index"] for p in bad] == [1]
    assert [p["pass_index"] for p in three_pass["passes"] if p["exact"]] == [0, 2]


def test_nvfp4_and_fdm_checks(first):
    _, _, freeze = first
    nvfp4 = freeze["nvfp4"]
    assert nvfp4["lossless"] is True
    assert nvfp4["ratio"] > 0
    assert nvfp4["ans_bytes"] > 0
    assert nvfp4["tensor_bytes"] > 0
    assert freeze["fdm"]["path"]
    assert freeze["fdm"]["max_abs_err_vs_ref"] == 0


def test_two_runs_produce_identical_digest_and_hashes(first, second):
    _, _, a = first
    _, _, b = second
    assert a["cert"]["root_digest"] == b["cert"]["root_digest"]
    for side in ("ref", "cand"):
        for field in ("canonical_sha256", "file_sha256"):
            assert a["fixture"][side][field] == b["fixture"][side][field]
    assert a["dump_sha256"] == b["dump_sha256"]
    left = {k: v for k, v in a.items() if k != "loopkit_src"}
    right = {k: v for k, v in b.items() if k != "loopkit_src"}
    assert left == right
