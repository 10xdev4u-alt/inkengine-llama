"""Determinism and shape checks for the synthetic MoE fixture + F32 dump.

All heavy objects are module-scoped so the whole file stays well under 20 s.
Run: python -m pytest -q tools/audit/test_fixture.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_moe_fixture as mk  # noqa: E402
import dump_f32 as du  # noqa: E402


@pytest.fixture(scope="module")
def fix0():
    return mk.build_fixture(0)


@pytest.fixture(scope="module")
def dump0(fix0):
    return du.build_dump(fix0)


def test_same_seed_twice_identical_sha256():
    assert mk.canonical_hash(mk.build_fixture(7)) == mk.canonical_hash(mk.build_fixture(7))


def test_different_seed_different_hash(fix0):
    assert mk.canonical_hash(fix0) != mk.canonical_hash(mk.build_fixture(1))


def test_topk_ids_in_range_and_unique(dump0):
    ids = dump0["topk_ids"]
    assert ids.shape == (mk.N_LAYERS, mk.N_PASSES, mk.N_TOKENS, mk.TOP_K)
    assert ids.min() >= 0 and ids.max() < mk.N_EXPERTS
    assert all(len(np.unique(row)) == mk.TOP_K for row in ids.reshape(-1, mk.TOP_K))


def test_dump_rerun_identical_hash(fix0, dump0, tmp_path):
    again = du.build_dump(fix0)
    assert mk.canonical_hash(again) == mk.canonical_hash(dump0)
    p1, p2 = tmp_path / "d1.npz", tmp_path / "d2.npz"
    mk.save_npz(str(p1), dump0)
    mk.save_npz(str(p2), again)
    assert mk.file_hash(str(p1)) == mk.file_hash(str(p2))


def test_fixture_has_three_passes_and_128_experts(fix0):
    c = mk.constants()
    assert c["n_passes"] == 3
    assert c["n_experts"] == 128
    assert fix0["expert_w13_codes"].shape[:2] == (mk.N_LAYERS, mk.N_EXPERTS)
    assert dump_shape_ok(fix0)


def dump_shape_ok(fix):
    return fix["x"].shape == (mk.N_TOKENS, mk.EMB) and fix["msg_query"].shape[1] == mk.LOOP_CTRL_COUNT
