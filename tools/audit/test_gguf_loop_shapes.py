"""Behavior tests for the fail-closed GGUF loop-tensor shape contract gate."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CHECKER = AUDIT_DIR / "check_gguf_loop_shapes.py"
sys.path.insert(0, str(AUDIT_DIR))

import check_gguf_loop_shapes as checker

EXPECTED_TOTAL_CHECKS = 22
NEW_FILES = [
    "GGUF_LOOP_SHAPES.json",
    "check_gguf_loop_shapes.py",
    "test_gguf_loop_shapes.py",
    "GGUF_LOOP_SHAPES.md",
    "M37_GGUF_LOOP_SHAPES_SUMMARY.json",
]
CACHE_TENSORS = Path("/tmp/opencode/loopkit-m37/tensors.json")


def make_fixture_tree(tmp):
    root = Path(tmp) / "repo"
    (root / "tools" / "audit").mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "GGUF_LOOP_SHAPES.json", root / checker.FACTS_REL)
    shutil.copy(AUDIT_DIR / "QUANT_LAYOUT.json", root / checker.M33_REL)
    for rel in checker.FROZEN_SOURCES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, dst)
    return root


def write_facts(root, payload):
    (root / checker.FACTS_REL).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def tamper_facts(root, mutate):
    payload = checker.load_json(root / checker.FACTS_REL)
    mutate(payload)
    write_facts(root, payload)


def tamper_m33(root, mutate):
    path = root / checker.M33_REL
    payload = checker.load_json(path)
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_summary(root):
    return json.loads((root / checker.SUMMARY_REL).read_text(encoding="utf-8"))


def run_cli(root):
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


class GateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = make_fixture_tree(self._tmp.name)
        self.facts = checker.load_json(AUDIT_DIR / "GGUF_LOOP_SHAPES.json")

    def tearDown(self):
        self._tmp.cleanup()

    def assert_no_go(self, *failed):
        stale = {"verdict": "PASS", "stale": True}
        checker.write_json(self.root / checker.SUMMARY_REL, stale)
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "NO_GO")
        self.assertNotEqual(summary["blocker"], "")
        for name in failed:
            self.assertIs(summary["checks"][name], False, name)
            self.assertIn(name, summary["blocker"])
        written = read_summary(self.root)
        self.assertEqual(written, summary)
        self.assertNotEqual(written, stale)
        self.assertIsNone(written["speed_claim"])
        self.assertTrue(written["no_weights"])
        self.assertTrue(written["no_engine_run"])
        self.assertGreaterEqual(written["check_counts"]["failed"], 1)
        return summary

    def test_frozen_json_validity(self):
        facts = self.facts
        self.assertEqual(facts["schema_version"], 1)
        self.assertEqual(facts["frozen_on"], "2026-11-28")
        self.assertEqual(facts["scope"], checker.NOTE)
        self.assertEqual(facts["note"], checker.NOTE)
        self.assertIsNone(facts["speed_claim"])
        self.assertEqual(facts["description"], checker.DESCRIPTION)
        self.assertEqual(facts["provenance"], checker.EXPECTED_PROVENANCE)
        self.assertEqual(facts["dimensions"], checker.EXPECTED_DIMENSIONS)
        self.assertEqual(facts["contract"], checker.EXPECTED_CONTRACT)
        self.assertEqual(facts["sources"], checker.FROZEN_SOURCES)
        self.assertEqual(facts["needles"], checker.NEEDLE_SPECS)
        self.assertEqual(facts["observed_names"], checker.EXPECTED_OBSERVED_NAMES)
        self.assertEqual(facts["summary"], checker.EXPECTED_SUMMARY)
        self.assertEqual(facts["measured_shape_counts"], checker.MEASURED_SHAPE_COUNTS)
        self.assertEqual(
            checker.measured_counts_from_contract(facts["contract"]),
            checker.MEASURED_SHAPE_COUNTS,
        )
        self.assertEqual(sum(checker.MEASURED_SHAPE_COUNTS.values()), 204)
        self.assertTrue(checker.check_frozen_facts(facts))
        self.assertEqual(facts["provenance"]["first_16mib_sha256"], checker.GGUF_SHA256)
        self.assertEqual(len(facts["observed_names"]), 204)
        for rel, digest in facts["sources"].items():
            self.assertEqual(len(digest), 64)
            int(digest, 16)
            self.assertTrue((REPO_ROOT / rel).is_file(), rel)

    def test_check_names_unique_and_complete(self):
        self.assertEqual(len(checker.CHECK_NAMES), EXPECTED_TOTAL_CHECKS)
        self.assertEqual(len(checker.CHECK_NAMES), len(set(checker.CHECK_NAMES)))
        self.assertEqual(len(checker.NEEDLE_CHECK_NAMES), 6)
        self.assertEqual(
            set(checker.CHECK_NAMES),
            {
                "frozen_facts", "frozen_provenance", "frozen_dimensions", "frozen_contract",
                "frozen_sources", "frozen_needles", "frozen_observed_names",
                "live_source_sha256", "derived_dimensions", "shape_consistency",
                "contract_coverage", "observed_names", "summary_counts",
                "m33_gguf_sha256", "no_network", "run_completed",
            }
            | set(checker.NEEDLE_CHECK_NAMES),
        )

    def test_derived_dimension_literals(self):
        dims = checker.derive_dimensions(checker.BASE_DIMENSIONS)
        self.assertEqual(dims, checker.EXPECTED_DIMENSIONS)
        self.assertEqual(dims["hidden"], 2304)
        self.assertEqual(dims["gate_rank"], 64)
        self.assertEqual(dims["down"], 4608)
        self.assertEqual(dims["down"], 2 * dims["hidden"])
        self.assertEqual(dims["msg_rows"], 4)
        self.assertEqual(dims["msg_rows"], dims["n_ctrl_families"])
        self.assertEqual(dims["attn_tail_rows"], 3)
        self.assertEqual(dims["attn_tail_layers"], [5, 11, 17, 23])
        self.assertEqual(dims["core_layers"], list(range(6, 18)))
        self.assertEqual(dims["n_phys"], 24)
        self.assertEqual(dims["n_slot"], 48)
        self.assertEqual(dims["n_slot"], 24 + 2 * (18 - 6))
        self.assertEqual(dims["expected_tensors"], 204)
        self.assertTrue(checker.check_derived_dimensions(self.facts))

    def test_derive_dimensions_rejects_bad_inputs(self):
        bad = [
            {"hidden": True},
            {"hidden": 0},
            {"gate_rank": -1},
            {"core_start": 5},
            {"core_end": 19},
            {"core_end": 17},
            {"n_passes": 1},
            {"attn_res_block": 5},
            {"attn_tail_rows": 4},
            {"n_ctrl_families": 5},
        ]
        for change in bad:
            base = dict(checker.BASE_DIMENSIONS)
            base.update(change)
            with self.assertRaises(ValueError):
                checker.derive_dimensions(base)

    def test_expanded_contract_has_exact_204_tensor_shapes(self):
        expanded = checker.expand_contract(checker.EXPECTED_CONTRACT)
        self.assertEqual(len(expanded), 204)
        self.assertEqual(sorted(expanded), checker.EXPECTED_OBSERVED_NAMES)
        self.assertEqual(
            expanded["blk.6.loop_msg_query.weight"], {"shape": [2304, 4], "dtype": "F32"}
        )
        self.assertEqual(
            expanded["blk.17.loop_ffn_ctrl_forget.weight"], {"shape": [64, 2304], "dtype": "F32"}
        )
        self.assertEqual(
            expanded["blk.6.loop_q_ctrl_down.weight"], {"shape": [4608, 64], "dtype": "F32"}
        )
        self.assertEqual(
            expanded["blk.5.attn_res_query.weight"], {"shape": [2304, 3], "dtype": "F32"}
        )
        self.assertEqual(
            expanded["blk.6.attn_res_query.weight"], {"shape": [2304], "dtype": "F32"}
        )
        self.assertEqual(
            expanded["blk.23.ffn_res_query.weight"], {"shape": [2304], "dtype": "F32"}
        )
        self.assertEqual(
            expanded["output_res_query.weight"], {"shape": [2304], "dtype": "F32"}
        )
        self.assertTrue(checker.check_contract_coverage(self.facts))
        self.assertTrue(checker.check_observed_names(self.facts))
        self.assertTrue(checker.check_summary_counts(self.facts))
        self.assertTrue(checker.check_shape_consistency(self.facts))

    def test_expand_contract_rejects_malformed_specs(self):
        bad = [
            {"bogus": {"template": "blk.0.bogus.weight", "shape": [2304], "dtype": "F32", "count": 2}},
            {"bogus": {"template": "output_res_query.weight", "shape": [2304], "dtype": "F32", "count": 2}},
            {"bogus": {"template": "blk.{bid}.bogus.weight", "layers": [0, 0], "shape": [2304], "dtype": "F32", "count": 2}},
        ]
        for contract in bad:
            with self.assertRaises(ValueError):
                checker.expand_contract(contract)

    @unittest.skipUnless(CACHE_TENSORS.is_file(), "cached GGUF tensor list is absent")
    def test_contract_matches_cached_gguf_measurement(self):
        measured = {
            entry["name"]: {"shape": entry["shape"], "dtype": entry["type"]}
            for entry in json.loads(CACHE_TENSORS.read_text(encoding="utf-8"))
        }
        self.assertEqual(checker.REFERENCE_MAP, {
            name: measured[name] for name in checker.REFERENCE_MAP
        })
        markers = ("loop", "ctrl", "attn_res", "ffn_res", "output_res")
        extra = [
            name for name in measured
            if any(token in name for token in markers) and name not in checker.REFERENCE_MAP
        ]
        self.assertEqual(extra, [])

    def test_real_tree_cli_passes_with_all_checks_true(self):
        summary_path = REPO_ROOT / checker.SUMMARY_REL
        original = summary_path.read_bytes()
        try:
            result = run_cli(REPO_ROOT)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = read_summary(REPO_ROOT)
            self.assertEqual(summary["verdict"], "PASS")
            self.assertEqual(summary["blocker"], "")
            self.assertEqual(summary["check_counts"], {"total": 22, "passed": 22, "failed": 0})
            self.assertTrue(all(value is True for value in summary["checks"].values()))
            self.assertIsNone(summary["speed_claim"])
            self.assertTrue(summary["no_weights"])
            self.assertTrue(summary["no_engine_run"])
            self.assertIn("coordinated two-repo doc update", " ".join(summary["info"]))
            self.assertIn("verdict: PASS", result.stdout)
        finally:
            summary_path.write_bytes(original)

    def test_fixture_cli_passes_with_exact_check_counts(self):
        result = run_cli(self.root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = read_summary(self.root)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual(summary["check_counts"], {"total": 22, "passed": 22, "failed": 0})
        self.assertEqual(summary["provenance"]["source_sha256_live"], checker.FROZEN_SOURCES)
        self.assertIn("counts: 22/22 passed, 0 failed", result.stdout)
        facts_bytes = (self.root / checker.FACTS_REL).read_bytes()
        self.assertEqual(hashlib.sha256(facts_bytes).hexdigest(), summary["frozen_facts_sha256"])

    def test_checker_source_is_pinned(self):
        digest = hashlib.sha256(CHECKER.read_bytes()).hexdigest()
        self.assertEqual(digest, "20ced5e91e9e4f016fa1cdf710ab0a93d98d9ec8e0dcf4af6bc3ba773e0c2482")

    def test_frozen_shape_counts_tamper_forces_no_go(self):
        tamper_facts(
            self.root,
            lambda f: f["measured_shape_counts"].__setitem__("[2304,4]|F32", 11),
        )
        self.assert_no_go("frozen_facts")

    def test_extra_top_level_fact_key_forces_no_go(self):
        tamper_facts(self.root, lambda f: f.__setitem__("extra_unknown_key", "hello"))
        self.assert_no_go("frozen_facts")

    def test_sentinel_write_failure_forces_no_go(self):
        stale = {"verdict": "PASS", "stale": True}
        checker.write_json(self.root / checker.SUMMARY_REL, stale)
        real_write = checker.write_json
        calls = []

        def fail_first(path, payload):
            calls.append(path)
            if len(calls) == 1:
                raise OSError("read-only")
            return real_write(path, payload)

        with mock.patch.object(checker, "write_json", side_effect=fail_first):
            summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "NO_GO")
        self.assertIn("sentinel_write_failed", summary["blocker"])

    def test_live_source_sha256_tamper_forces_no_go(self):
        path = self.root / checker.LOOP_CPP
        path.write_text(path.read_text(encoding="utf-8") + "\n// tamper\n", encoding="utf-8")
        self.assert_no_go("live_source_sha256")

    def test_frozen_source_sha256_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f["sources"].__setitem__(checker.LOOP_CPP, "0" * 64))
        self.assert_no_go("frozen_sources", "frozen_facts")

    def test_description_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f.__setitem__("description", "timing result"))
        self.assert_no_go("frozen_facts")

    def test_provenance_sha_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f["provenance"].__setitem__("first_16mib_sha256", "0" * 64))
        self.assert_no_go("frozen_provenance", "frozen_facts")

    def test_shape_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f["contract"]["ctrl_down"]["shape"].__setitem__(0, 4609))
        self.assert_no_go("frozen_contract", "shape_consistency", "contract_coverage", "frozen_facts")

    def test_count_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f["contract"]["ctrl_gain"].__setitem__("count", 47))
        self.assert_no_go("frozen_contract", "contract_coverage", "frozen_facts")

    def test_special_layers_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f["contract"]["attn_res_query"].__setitem__("special_layers", [5, 11, 17]))
        self.assert_no_go("frozen_contract", "shape_consistency", "contract_coverage", "frozen_facts")

    def test_special_shape_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f["contract"]["attn_res_query"].__setitem__("special_shape", [2304, 4]))
        self.assert_no_go("frozen_contract", "shape_consistency", "contract_coverage", "frozen_facts")

    def test_dimension_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f["dimensions"].__setitem__("hidden", 2305))
        self.assert_no_go("frozen_dimensions", "derived_dimensions", "shape_consistency", "frozen_facts")

    def test_extra_tensor_contract_forces_no_go(self):
        def add(f):
            f["contract"]["bogus"] = {
                "template": "blk.0.bogus.weight",
                "layers": [0],
                "shape": [2304],
                "dtype": "F32",
                "count": 1,
            }
        tamper_facts(self.root, add)
        self.assert_no_go("frozen_contract", "shape_consistency", "contract_coverage", "frozen_facts")

    def test_observed_name_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f["observed_names"].__setitem__(0, "blk.0.extra.weight"))
        self.assert_no_go("frozen_observed_names", "observed_names", "frozen_facts")

    def test_summary_count_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f["summary"].__setitem__("expected_tensors", 205))
        self.assert_no_go("summary_counts", "frozen_facts")

    def test_frozen_needle_tamper_forces_no_go(self):
        tamper_facts(self.root, lambda f: f["needles"][0]["needles"].__setitem__(0, "REMOVED"))
        self.assert_no_go("frozen_needles", "frozen_facts")

    def test_live_needle_deletion_forces_no_go(self):
        path = self.root / checker.CONSTANTS_PY
        needle = '"blk.{bid}.loop_msg_query"'
        self.assertIn(needle, path.read_text(encoding="utf-8"))
        path.write_text(path.read_text(encoding="utf-8").replace(needle, "REMOVED"), encoding="utf-8")
        self.assert_no_go("needle_gguf_tensor_names", "live_source_sha256")

    def test_m33_sha_tamper_forces_no_go(self):
        tamper_m33(self.root, lambda f: f["provenance"]["header_probe"].__setitem__("first_16mib_sha256", "0" * 64))
        self.assert_no_go("m33_gguf_sha256")

    def test_malformed_facts_json_forces_no_go(self):
        (self.root / checker.FACTS_REL).write_text("{not json\n", encoding="utf-8")
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "NO_GO")
        self.assertIs(summary["checks"]["run_completed"], False)
        self.assertTrue(any("exception:" in line for line in summary["info"]))
        self.assertEqual(summary["check_counts"]["passed"], 0)

    def test_missing_facts_file_forces_no_go(self):
        (self.root / checker.FACTS_REL).unlink()
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "NO_GO")
        self.assertIs(summary["checks"]["run_completed"], False)
        self.assertIs(summary["checks"]["frozen_facts"], False)

    def test_missing_source_file_forces_no_go(self):
        (self.root / checker.INKLING_CPP).unlink()
        self.assert_no_go("live_source_sha256", "needle_res_query_shapes")

    def test_missing_m33_file_forces_no_go(self):
        (self.root / checker.M33_REL).unlink()
        self.assert_no_go("m33_gguf_sha256")

    def test_m33_malformed_json_forces_no_go(self):
        (self.root / checker.M33_REL).write_text("{not json\n", encoding="utf-8")
        self.assert_no_go("m33_gguf_sha256", "run_completed")

    def test_no_go_sentinel_is_written_before_final_verdict(self):
        stale = {"verdict": "PASS", "stale": True}
        checker.write_json(self.root / checker.SUMMARY_REL, stale)
        original = checker.write_json
        seen = []

        def tracking(path, payload):
            result = original(path, payload)
            seen.append((path, json.loads(path.read_text(encoding="utf-8"))))
            return result

        with mock.patch.object(checker, "write_json", tracking):
            summary = checker.run_check(self.root)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][0], self.root / checker.SUMMARY_REL)
        self.assertEqual(seen[0][1]["verdict"], "NO_GO")
        self.assertIs(seen[0][1]["sentinel"], True)
        self.assertEqual(seen[0][1]["check_counts"]["passed"], 0)
        self.assertEqual(seen[1][1]["verdict"], "PASS")
        self.assertNotIn("sentinel", seen[1][1])
        self.assertEqual(summary, seen[1][1])

    def test_atomic_write_failure_keeps_previous_summary_and_removes_temp(self):
        target = self.root / "tools/audit/M37_GGUF_LOOP_SHAPES_SUMMARY.json"
        old = '{"verdict": "NO_GO"}\n'
        target.write_text(old, encoding="utf-8")

        def fail_replace(src, dst):
            raise OSError("simulated write failure")

        with mock.patch.object(checker.os, "replace", fail_replace):
            with self.assertRaisesRegex(OSError, "simulated"):
                checker.write_json(target, {"verdict": "PASS"})
        self.assertEqual(target.read_text(encoding="utf-8"), old)
        self.assertEqual(list(target.parent.glob(".*.tmp")), [])

    def test_main_exits_nonzero_when_summary_write_fails(self):
        def boom(root):
            raise OSError("simulated write failure")

        with mock.patch.object(checker, "run_check", boom):
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(checker.main([]), 1)
        self.assertIn("could not write summary", err.getvalue())

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
        self.assertTrue(checker.check_no_network())
        self.assertIs(self.facts["provenance"]["network_access_in_checker"], False)

    def test_new_audit_files_are_ascii_only(self):
        for name in NEW_FILES:
            raw = (AUDIT_DIR / name).read_bytes()
            self.assertTrue(all(byte < 128 for byte in raw), name)


if __name__ == "__main__":
    unittest.main()
