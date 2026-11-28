"""Unit tests for the pure decision logic of the M28 real-weight gate.

Run: PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest -q tools/audit
"""

import importlib.util
import sys
from pathlib import Path

import pytest

GATE_PATH = Path(__file__).resolve().parents[1] / "roret" / "check_real_weight_parity.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("check_real_weight_parity", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def good_info(**overrides):
    info = {
        "attested": True,
        "exists": True,
        "sha256": "a" * 64,
        "size": 20_000_000,
        "arch": "inkling2_loop",
        "expert_count": 128,
    }
    info.update(overrides)
    return info


@pytest.mark.parametrize("sha", [
    "cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617",
    "fff0e60fa4f55ba717d1e0bdc0e89695a0d35c85df680a15fd57ed1a0ac959ff",
])
def test_frozen_synthetic_sha_rejected(gate, sha):
    kind, reasons = gate.classify_weights(good_info(sha256=sha))
    assert kind == "synthetic_or_unknown"
    assert any("sha256" in reason for reason in reasons)


def test_size_too_small_rejected(gate):
    kind, reasons = gate.classify_weights(good_info(size=887_392))
    assert kind == "synthetic_or_unknown"
    assert any("size" in reason for reason in reasons)


def test_wrong_arch_rejected(gate):
    kind, reasons = gate.classify_weights(good_info(arch="llama"))
    assert kind == "synthetic_or_unknown"
    assert any("architecture" in reason for reason in reasons)


def test_missing_attestation_rejected(gate):
    kind, reasons = gate.classify_weights(good_info(attested=False))
    assert kind == "synthetic_or_unknown"
    assert any("attestation" in reason for reason in reasons)


def test_missing_file_rejected(gate):
    kind, reasons = gate.classify_weights(good_info(exists=False))
    assert kind == "synthetic_or_unknown"
    assert any("does not exist" in reason for reason in reasons)


@pytest.mark.parametrize("count", [None, 8])
def test_expert_count_missing_or_low_rejected(gate, count):
    kind, reasons = gate.classify_weights(good_info(expert_count=count))
    assert kind == "synthetic_or_unknown"
    assert any("expert_count" in reason for reason in reasons)


def test_all_good_is_real(gate):
    assert gate.classify_weights(good_info()) == ("real", [])


@pytest.mark.parametrize("kind,parity_ok,expected", [
    ("real", False, "NO_GO"),
    ("synthetic_or_unknown", True, "NO_GO"),
    ("real", True, "PASS"),
])
def test_final_verdict(gate, kind, parity_ok, expected):
    assert gate.final_verdict(kind, parity_ok) == expected


@pytest.mark.parametrize("sha", ["", "abc", "A" * 64, "g" * 64, None])
def test_sha256_required_and_well_formed(gate, sha):
    kind, reasons = gate.classify_weights(good_info(sha256=sha))
    assert kind == "synthetic_or_unknown"
    assert any("sha256" in reason for reason in reasons)


def good_record(**overrides):
    record = {
        "cpu_share": "0.0", "off_exit": 0, "on_exit": 0,
        "off_hook": False, "on_hook": True, "tensors": 40,
        "bytes_compared": 30080, "differing_bytes": 0,
    }
    record.update(overrides)
    return record


def test_shares_ok_empty_records_fail_closed(gate):
    assert gate.shares_ok([], []) is False


def test_shares_ok_problems_fail_closed(gate):
    assert gate.shares_ok([good_record()], ["one problem"]) is False


def test_shares_ok_rejects_byte_difference(gate):
    assert gate.shares_ok([good_record(differing_bytes=3)], []) is False


def test_shares_ok_accepts_clean_records(gate):
    assert gate.shares_ok([good_record(), good_record(cpu_share="1.0")], []) is True


@pytest.mark.parametrize("shares,expected", [
    (["0.0", "1.0"], True),
    (["0.0", "1.0", "0.5"], True),
    (["0.0"], False),
    (["1.0"], False),
    (["0.5"], False),
    ([], False),
])
def test_shares_covered(gate, shares, expected):
    assert gate.shares_covered(shares) is expected


@pytest.mark.parametrize("path", [GATE_PATH, GATE_PATH.parent / "does-not-exist.gguf"])
def test_read_gguf_info_fails_closed(gate, path):
    info = gate.read_gguf_info(path)
    assert info["arch"] == ""
    assert info["expert_count"] is None
    assert info["expert_used_count"] is None
    assert info["block_count"] is None
