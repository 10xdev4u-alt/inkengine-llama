"""Tests for the executable CycleAudit gate verification.

Run with:
  PYTHONPATH=tools/audit python3 -m pytest -q tools/audit/test_verify_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_gate as vg  # noqa: E402


WORKSPACE = Path(__file__).resolve().parents[3]
LOOPKIT_SRC = WORKSPACE / "loopkit" / "src"


@pytest.fixture(scope="module")
def samples():
    return vg.synthetic_samples(repeats=50, seed=7)


def test_synthetic_samples_have_frozen_shape(samples):
    assert len(samples["baseline"]["seconds"]) == 50
    assert all(len(s["seconds"]) >= 20 for s in samples.values())
    for name, s in samples.items():
        assert s["cycles"] == len(s["seconds"])
        assert s["name"] == f"{name}-b1"
        assert all(t > 0 for t in s["seconds"])


def test_timing_guard_rejects_too_few_repeats():
    with pytest.raises(ValueError, match="repeats"):
        vg.make_timing("short", [0.1, 0.2, 0.3])


def test_self_comparison_fails_closed(tmp_path, samples):
    paths = vg.write_samples(tmp_path, samples)
    result = vg.run_gate(1.01, paths["baseline"], paths["baseline"],
                         loopkit_src=LOOPKIT_SRC)
    assert result["claim_valid"] is False
    assert result["returncode"] == 1
    assert result["report"]["ci_low"] < 1.01


def test_slower_candidate_fails_closed(tmp_path, samples):
    paths = vg.write_samples(tmp_path, samples)
    result = vg.run_gate(2.0, paths["baseline"], paths["slow"],
                         loopkit_src=LOOPKIT_SRC)
    assert result["claim_valid"] is False
    assert result["returncode"] == 1


def test_claim_at_ci_low_passes_and_inflated_claim_fails(tmp_path, samples):
    paths = vg.write_samples(tmp_path, samples)
    probe = vg.run_gate(0.0, paths["baseline"], paths["candidate"],
                        loopkit_src=LOOPKIT_SRC)
    ci_low = probe["report"]["ci_low"]
    assert ci_low > 1.0

    at_bound = vg.run_gate(ci_low, paths["baseline"], paths["candidate"],
                           loopkit_src=LOOPKIT_SRC)
    assert at_bound["claim_valid"] is True
    assert at_bound["returncode"] == 0

    inflated = vg.run_gate(ci_low + 0.05, paths["baseline"], paths["candidate"],
                           loopkit_src=LOOPKIT_SRC)
    assert inflated["claim_valid"] is False
    assert inflated["returncode"] == 1


def test_verify_runs_all_controls_and_is_deterministic(tmp_path):
    first = vg.verify(tmp_path, loopkit_src=LOOPKIT_SRC, seed=7)
    second = vg.verify(tmp_path, loopkit_src=LOOPKIT_SRC, seed=7)
    assert first["verified"] is True
    assert first == second
    assert set(first["cases"]) == {"self", "slower", "at_bound", "inflated"}
    assert first["cases"]["self"]["claim_valid"] is False
    assert first["cases"]["slower"]["claim_valid"] is False
    assert first["cases"]["at_bound"]["claim_valid"] is True
    assert first["cases"]["inflated"]["claim_valid"] is False
    assert first["gate"]["loops"] == 3
    assert first["gate"]["min_repeats"] >= 20
    assert (tmp_path / "gate_verification.json").is_file()
    written = json.loads((tmp_path / "gate_verification.json").read_text())
    assert written == first
