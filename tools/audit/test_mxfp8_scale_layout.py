"""Tests for the offline MXFP8 scale-layout parity gate."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
sys.path.insert(0, str(AUDIT_DIR))

import check_mxfp8_scale_layout as checker


def make_fixture_tree(tmp):
    root = Path(tmp) / "repo"
    (root / "tools" / "audit").mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "MXFP8_SCALE_HEADER.json", root / checker.FACTS_REL)
    shutil.copy(AUDIT_DIR / "QUANT_LAYOUT.json", root / checker.M33_FACTS_REL)
    src = root / checker.CONVERTER_PY
    src.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / checker.CONVERTER_PY, src)
    return root


def write_facts(root, facts):
    (root / checker.FACTS_REL).write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")


def read_summary(root):
    return json.loads((root / checker.SUMMARY_REL).read_text(encoding="utf-8"))


class GateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = make_fixture_tree(self._tmp.name)
        self.facts = checker.load_json(AUDIT_DIR / "MXFP8_SCALE_HEADER.json")

    def tearDown(self):
        self._tmp.cleanup()

    def assert_no_go(self, *failed):
        stale = {"verdict": "PASS"}
        checker.write_json(self.root / checker.SUMMARY_REL, stale)
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "NO_GO")
        for name in failed:
            self.assertIs(summary["checks"][name], False, name)
            self.assertIn(name, summary["blocker"])
        written = read_summary(self.root)
        self.assertEqual(written, summary)
        self.assertNotEqual(written, stale)
        self.assertIsNone(written["speed_claim"])
        self.assertEqual(written["note"], checker.NOTE)
        return summary

    def test_frozen_json_validity(self):
        facts = self.facts
        self.assertEqual(facts["schema_version"], 1)
        self.assertEqual(facts["frozen_on"], "2026-11-28")
        self.assertEqual(facts["scope"], checker.NOTE)
        self.assertEqual(facts["note"], checker.NOTE)
        self.assertIsNone(facts["speed_claim"])
        self.assertEqual(facts["facts"], checker.EXPECTED_FACTS)
        self.assertEqual(facts["needles"], checker.SOURCE_NEEDLES)
        self.assertEqual(facts["provenance"], checker.EXPECTED_PROVENANCE)
        self.assertEqual(facts["checkpoint_scale_parity"], "VERIFIED")
        self.assertIn("(1,32)", facts["resolution"])
        self.assertEqual(facts["provenance"]["fetched_bytes"], 148416)
        self.assertEqual(facts["provenance"]["header_len"], 148408)
        self.assertEqual(facts["provenance"]["n_tensors"], 1245)
        self.assertFalse(facts["provenance"]["network_access_in_checker"])
        for key in ("sha256", "converter_sha256"):
            value = facts["provenance"][key]
            self.assertEqual(len(value), 64)
            int(value, 16)

    def test_check_names_unique_and_complete(self):
        self.assertEqual(len(checker.CHECK_NAMES), 18)
        self.assertEqual(len(checker.CHECK_NAMES), len(set(checker.CHECK_NAMES)))
        self.assertEqual(len(checker.NEEDLE_CHECK_NAMES), 7)
        self.assertEqual(
            set(checker.CHECK_NAMES),
            {
                "frozen_facts", "frozen_provenance", "frozen_converter_sha256",
                "live_converter_sha256", "derived_elements_per_scale",
                "stored_block_parity", "converter_semantics",
                "published_config_info", "m33_superseded_detected",
                "no_network", "run_completed",
            }
            | set(checker.NEEDLE_CHECK_NAMES),
        )

    def test_derive_elements_per_scale_literals(self):
        self.assertEqual(checker.derive_elements_per_scale(2304, 72), 32)
        self.assertEqual(checker.derive_elements_per_scale(1152, 36), 32)
        self.assertEqual(checker.derive_elements_per_scale(2304, 144), 16)
        self.assertEqual(checker.derive_elements_per_scale(1152, 72), 16)
        self.assertEqual(checker.derive_elements_per_scale(2304, 32), 72)
        for cols, scale_cols in ((0, 1), (2304, 0), (-2304, 144), (2304, True), (2304.0, 144), (2305, 144)):
            with self.assertRaises(ValueError):
                checker.derive_elements_per_scale(cols, scale_cols)

    def test_derived_layout_from_frozen_shapes(self):
        derived = checker.derive_from_facts(self.facts)
        self.assertEqual(derived["mxfp8"]["w13"]["elements_per_scale"], 32)
        self.assertEqual(derived["mxfp8"]["w2"]["elements_per_scale"], 32)
        self.assertEqual(derived["nvfp4"]["w13"]["elements_per_scale"], 16)
        self.assertEqual(derived["nvfp4"]["w2"]["elements_per_scale"], 16)
        self.assertEqual(derived["mxfp8"]["w13"]["stored_block"], [1, 32])
        self.assertEqual(derived["nvfp4"]["w13"]["stored_block"], [1, 16])
        self.assertTrue(checker.check_derived_elements(self.facts, derived))
        self.assertTrue(checker.check_stored_block_parity(self.facts, derived))
        self.assertTrue(checker.check_converter_semantics(self.facts))
        self.assertTrue(checker.check_published_config(self.facts))

    def test_derive_rejects_tampered_shapes(self):
        for mutate in (
            lambda f: f["facts"]["measured"]["mxfp8"]["w13_weight_scale"]["shape"].__setitem__(2, 144),
            lambda f: f["facts"]["measured"]["mxfp8"]["w2_weight_scale"]["shape"].__setitem__(2, 72),
            lambda f: f["facts"]["measured"]["mxfp8"]["layers_detail"]["21"]["w13_weight_scale"].__setitem__("shape", [128, 2304, 144]),
            lambda f: f["facts"]["derived"]["mxfp8"].__setitem__("w13_elements_per_scale", 16),
            lambda f: f["facts"]["measured"]["mxfp8"]["layers"].__setitem__(0, 21),
            lambda f: f["facts"]["measured"]["mxfp8"].__setitem__("extra_companions", True),
        ):
            payload = json.loads(json.dumps(self.facts))
            mutate(payload)
            with self.assertRaises((ValueError, KeyError, TypeError)):
                checker.derive_from_facts(payload)

    def test_pass_summary_fields_and_info(self):
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual(summary["blocker"], "")
        self.assertTrue(all(summary["checks"].values()))
        self.assertEqual(summary["checkpoint_scale_parity"], "VERIFIED")
        self.assertIsNone(summary["speed_claim"])
        self.assertEqual(summary["note"], "MXFP8 scale-layout parity only; no timing or speed claim")
        self.assertEqual(summary["frozen_on"], "2026-11-28")
        self.assertEqual(summary["converter_sha256_live"], checker.FROZEN_CONVERTER_SHA256)
        self.assertEqual(summary["derived"]["mxfp8"]["w13"]["elements_per_scale"], 32)
        self.assertEqual(summary["derived"]["nvfp4"]["w2"]["elements_per_scale"], 16)
        info = " ".join(summary["info"])
        self.assertIn("SUPERSEDED", info)
        self.assertIn("INFO, not a failure", info)
        self.assertIn("INFO only, not a blocker", info)
        self.assertIn("checker makes no network calls", info)
        self.assertEqual(read_summary(self.root), summary)

    def test_cli_exit_codes(self):
        with mock.patch.object(checker, "REPO_ROOT", self.root):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main([]), 0)
            converter = self.root / checker.CONVERTER_PY
            with converter.open("a", encoding="utf-8") as handle:
                handle.write("# tampered\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main([]), 1)

    def test_mutate_frozen_mxfp8_shapes_forces_no_go(self):
        payload = json.loads(json.dumps(self.facts))
        payload["facts"]["measured"]["mxfp8"]["w13_weight_scale"]["shape"] = [128, 2304, 144]
        payload["facts"]["measured"]["mxfp8"]["w2_weight_scale"]["shape"] = [128, 2304, 72]
        write_facts(self.root, payload)
        self.assert_no_go("frozen_facts", "derived_elements_per_scale", "stored_block_parity")

    def test_mutate_converter_reshape_needle_forces_no_go(self):
        path = self.root / checker.CONVERTER_PY
        text = path.read_text(encoding="utf-8")
        self.assertIn("values.reshape(values.shape[0], -1, 32)[:] *= block_scale[:, :, None]", text)
        path.write_text(
            text.replace(
                "values.reshape(values.shape[0], -1, 32)[:] *= block_scale[:, :, None]",
                "values.reshape(values.shape[0], -1, 16)[:] *= block_scale[:, :, None]",
            ),
            encoding="utf-8",
        )
        self.assert_no_go("needle_mxfp8_dequant_reshape")

    def test_mutate_converter_validation_needle_forces_no_go(self):
        path = self.root / checker.CONVERTER_PY
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("else 32):", "else 16):"), encoding="utf-8")
        self.assert_no_go("needle_mxfp8_stored_block")

    def test_mutate_converter_dtype_needle_forces_no_go(self):
        path = self.root / checker.CONVERTER_PY
        text = path.read_text(encoding="utf-8")
        old = "weight_dtype, scale_dtype = (torch.uint8, torch.float8_e4m3fn) if nvfp4 else (torch.float8_e4m3fn, torch.uint8)"
        self.assertIn(old, text)
        path.write_text(text.replace(old, old.replace("torch.uint8)", "torch.int8)")), encoding="utf-8")
        self.assert_no_go("needle_weight_scale_dtypes")

    def test_wrong_live_converter_hash_forces_no_go(self):
        with (self.root / checker.CONVERTER_PY).open("a", encoding="utf-8") as handle:
            handle.write("# tampered\n")
        self.assert_no_go("live_converter_sha256")

    def test_wrong_frozen_converter_hash_forces_no_go(self):
        payload = json.loads(json.dumps(self.facts))
        payload["provenance"]["converter_sha256"] = "0" * 64
        write_facts(self.root, payload)
        self.assert_no_go("frozen_converter_sha256", "frozen_facts", "frozen_provenance")

    def test_missing_m33_file_forces_no_go(self):
        (self.root / checker.M33_FACTS_REL).unlink()
        self.assert_no_go("m33_superseded_detected")

    def test_malformed_facts_json_forces_no_go(self):
        (self.root / checker.FACTS_REL).write_text("{not json", encoding="utf-8")
        self.assert_no_go("run_completed", "frozen_facts")

    def test_malformed_m33_json_forces_no_go(self):
        (self.root / checker.M33_FACTS_REL).write_text("{not json", encoding="utf-8")
        self.assert_no_go("run_completed", "m33_superseded_detected")

    def test_checker_has_no_network_access(self):
        source = Path(checker.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], checker.NETWORK_MODULES)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.assertNotIn(node.module.split(".")[0], checker.NETWORK_MODULES)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"urlopen", "urlretrieve"})
        for token in ("import socket", "import requests", "urlopen(", "urlretrieve("):
            self.assertNotIn(token, source)
        self.assertFalse(self.facts["provenance"]["network_access_in_checker"])
        self.assertEqual(hashlib.sha256((self.root / checker.CONVERTER_PY).read_bytes()).hexdigest(), checker.FROZEN_CONVERTER_SHA256)

    def test_new_audit_files_are_ascii_only(self):
        for name in ("MXFP8_SCALE_HEADER.json", "check_mxfp8_scale_layout.py", "test_mxfp8_scale_layout.py", "MXFP8_SCALE_HEADER.md", "M34_MXFP8_SCALE_SUMMARY.json"):
            raw = (AUDIT_DIR / name).read_bytes()
            self.assertTrue(all(byte < 128 for byte in raw), name)


if __name__ == "__main__":
    unittest.main()
