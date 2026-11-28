"""Behavior tests for the fail-closed integration-doc closure gate."""

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
CHECKER = AUDIT_DIR / "check_integration_doc_closure.py"
sys.path.insert(0, str(AUDIT_DIR))

import check_integration_doc_closure as checker

EXPECTED_TOTAL_CHECKS = 19
EXPECTED_CHECKER_SHA256 = "7b2232149c4c6292beface76184c53bd92b8de9cc30ab633db79dafb827532b8"
NEW_FILES = [
    "INTEGRATION_DOC_CLOSURE.json",
    "check_integration_doc_closure.py",
    "test_integration_doc_closure.py",
    "INTEGRATION_DOC_CLOSURE.md",
    "M38_INTEGRATION_DOC_CLOSURE_SUMMARY.json",
]


def make_fixture_tree(tmp):
    root = Path(tmp) / "repo"
    (root / "tools" / "audit").mkdir(parents=True)
    doc = root / checker.DOC_REL
    doc.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / checker.DOC_REL, doc)
    shutil.copy(AUDIT_DIR / "INTEGRATION_DOC_CLOSURE.json", root / checker.CONTRACT_REL)
    for paths in checker.EVIDENCE.values():
        for rel in paths:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(AUDIT_DIR / Path(rel).name, dst)
    return root


def write_contract(root, payload):
    (root / checker.CONTRACT_REL).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def tamper_contract(root, mutate):
    payload = checker.load_json(root / checker.CONTRACT_REL)
    mutate(payload)
    write_contract(root, payload)


def write_summary(root, payload):
    (root / checker.SUMMARY_REL).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_summary(root):
    return json.loads((root / checker.SUMMARY_REL).read_text(encoding="utf-8"))


def tamper_summary(root, rel, mutate):
    path = root / rel
    payload = checker.load_json(path)
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
        self.contract = checker.load_json(AUDIT_DIR / "INTEGRATION_DOC_CLOSURE.json")

    def tearDown(self):
        self._tmp.cleanup()

    def assert_no_go(self, *failed):
        stale = {"verdict": "PASS", "stale": True}
        write_summary(self.root, stale)
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

    def test_contract_structure_and_pins(self):
        c = self.contract
        self.assertEqual(set(c), checker.EXPECTED_TOP_KEYS)
        self.assertEqual(c["schema_version"], 1)
        self.assertEqual(c["frozen_on"], "2026-11-28")
        self.assertEqual(c["scope"], checker.NOTE)
        self.assertEqual(c["note"], checker.NOTE)
        self.assertEqual(c["description"], checker.DESCRIPTION)
        self.assertEqual(c["doc_path"], "../loopkit/docs/INKLING_INTEGRATION.md")
        self.assertEqual(c["doc_sha256"], checker.DOC_SHA256)
        self.assertEqual(
            c["checker_sha256"],
            hashlib.sha256(CHECKER.read_bytes()).hexdigest(),
        )
        self.assertEqual(c["cycle_audit_required"], checker.CYCLE_AUDIT_REQUIRED)
        self.assertIsNone(c["speed_claim"])
        self.assertTrue(checker.check_frozen_contract(c))

    def test_needle_entries_are_ascii_and_present_in_live_doc(self):
        doc = (REPO_ROOT / checker.DOC_REL).read_text(encoding="utf-8")
        entries = checker.RESOLVED_NEEDLES + checker.HONEST_BLOCKERS + [checker.CLOSURE_NOTE]
        self.assertEqual(len(checker.RESOLVED_NEEDLES), 7)
        self.assertEqual(len(checker.HONEST_BLOCKERS), 1)
        self.assertEqual(len(entries), 9)
        for entry in entries:
            self.assertTrue(entry["needles"])
            for needle in entry["needles"]:
                self.assertTrue(needle.isascii(), needle)
                self.assertIn(needle, doc, (entry["name"], needle))

    def test_check_names_unique_and_complete(self):
        self.assertEqual(len(checker.CHECK_NAMES), EXPECTED_TOTAL_CHECKS)
        self.assertEqual(len(checker.CHECK_NAMES), len(set(checker.CHECK_NAMES)))
        self.assertEqual(len(checker.NEEDLE_NAMES), 9)
        self.assertEqual(
            set(checker.CHECK_NAMES),
            {
                "frozen_contract", "frozen_checker_sha256", "live_doc_sha256",
                "no_open_markers", "evidence_pointers", "evidence_summaries",
                "summary_expectations", "summary_shape", "no_network", "run_completed",
            }
            | {"needle_" + name for name in checker.NEEDLE_NAMES},
        )

    def test_checker_source_is_pinned(self):
        digest = hashlib.sha256(CHECKER.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_CHECKER_SHA256)
        self.assertEqual(digest, self.contract["checker_sha256"])

    def test_fixture_cli_passes_with_exact_counts(self):
        result = run_cli(self.root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = read_summary(self.root)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual(summary["blocker"], "")
        self.assertEqual(
            summary["check_counts"],
            {"total": EXPECTED_TOTAL_CHECKS, "passed": EXPECTED_TOTAL_CHECKS, "failed": 0},
        )
        self.assertTrue(all(value is True for value in summary["checks"].values()))
        self.assertEqual(summary["doc_sha256"], checker.DOC_SHA256)
        self.assertEqual(summary["checker_sha256"], EXPECTED_CHECKER_SHA256)
        self.assertIsNone(summary["speed_claim"])
        self.assertEqual(summary["evidence_verdicts"], {m: "PASS" for m in checker.EXPECTED_MILESTONES})
        self.assertIn("verdict: PASS", result.stdout)
        self.assertIn("counts: 19/19 passed, 0 failed", result.stdout)

    def test_real_tree_cli_passes_and_restores_summary_bytes(self):
        summary_path = REPO_ROOT / checker.SUMMARY_REL
        original = summary_path.read_bytes()
        try:
            result = run_cli(REPO_ROOT)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = read_summary(REPO_ROOT)
            self.assertEqual(summary["verdict"], "PASS")
            self.assertEqual(summary["blocker"], "")
            self.assertEqual(summary["check_counts"]["failed"], 0)
            self.assertTrue(all(value is True for value in summary["checks"].values()))
            self.assertEqual(summary["doc_sha256"], checker.DOC_SHA256)
        finally:
            summary_path.write_bytes(original)
        self.assertEqual(summary_path.read_bytes(), original)

    def test_fail_closed_summary_is_sentinel_shaped(self):
        sentinel = checker.fail_closed_summary()
        self.assertEqual(sentinel["verdict"], "NO_GO")
        self.assertTrue(sentinel["sentinel"])
        self.assertEqual(set(sentinel["checks"]), set(checker.CHECK_NAMES))
        self.assertFalse(any(sentinel["checks"].values()))
        self.assertEqual(
            sentinel["check_counts"],
            {"total": EXPECTED_TOTAL_CHECKS, "passed": 0, "failed": EXPECTED_TOTAL_CHECKS},
        )
        self.assertIsNone(sentinel["speed_claim"])

    def test_hash_mismatch_forces_no_go_and_overwrites_stale_pass(self):
        path = self.root / checker.DOC_REL
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assert_no_go("live_doc_sha256")

    def test_deleted_needle_forces_no_go(self):
        path = self.root / checker.DOC_REL
        needle = "no weights, no engine run, no parity, no timing."
        self.assertIn(needle, path.read_text(encoding="utf-8"))
        path.write_text(path.read_text(encoding="utf-8").replace(needle, "REMOVED"), encoding="utf-8")
        self.assert_no_go("needle_section2_gguf_shapes", "live_doc_sha256")

    def test_missing_doc_forces_no_go(self):
        (self.root / checker.DOC_REL).unlink()
        self.assert_no_go("live_doc_sha256")

    def test_malformed_contract_forces_no_go(self):
        (self.root / checker.CONTRACT_REL).write_text("{not json\n", encoding="utf-8")
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "NO_GO")
        self.assertIs(summary["checks"]["run_completed"], False)
        self.assertTrue(any("exception:" in line for line in summary["info"]))

    def test_missing_contract_forces_no_go(self):
        (self.root / checker.CONTRACT_REL).unlink()
        self.assert_no_go("frozen_contract")

    def test_extra_contract_key_forces_no_go(self):
        tamper_contract(self.root, lambda c: c.__setitem__("unexpected_key", 1))
        self.assert_no_go("frozen_contract")

    def test_frozen_checker_sha_tamper_forces_no_go(self):
        tamper_contract(self.root, lambda c: c.__setitem__("checker_sha256", "0" * 64))
        self.assert_no_go("frozen_checker_sha256")

    def test_doc_sha_tamper_forces_no_go(self):
        tamper_contract(self.root, lambda c: c.__setitem__("doc_sha256", "0" * 64))
        self.assert_no_go("frozen_contract", "live_doc_sha256")

    def test_evidence_pointers_tamper_forces_no_go(self):
        tamper_contract(
            self.root,
            lambda c: c["evidence"]["M37"].__setitem__(1, "tools/audit/MISSING_SUMMARY.json"),
        )
        self.assert_no_go("frozen_contract", "evidence_pointers", "evidence_summaries")

    def test_milestone_summary_verdict_tamper_forces_no_go(self):
        rel = checker.EVIDENCE["M37"][1]
        tamper_summary(self.root, rel, lambda s: s.__setitem__("verdict", "NO_GO"))
        self.assert_no_go("evidence_summaries")

    def test_milestone_summary_counts_tamper_forces_no_go(self):
        rel = checker.EVIDENCE["M37"][1]
        tamper_summary(self.root, rel, lambda s: s["check_counts"].__setitem__("failed", 2))
        self.assert_no_go("evidence_summaries")

    def test_malformed_milestone_summary_forces_no_go(self):
        (self.root / checker.EVIDENCE["M30"][1]).write_text("{not json\n", encoding="utf-8")
        self.assert_no_go("evidence_summaries")

    def test_missing_milestone_summary_forces_no_go(self):
        (self.root / checker.EVIDENCE["M36"][1]).unlink()
        self.assert_no_go("evidence_summaries", "summary_expectations")

    def test_summary_expectations_tamper_forces_no_go(self):
        tamper_contract(
            self.root,
            lambda c: c["summary_expectations"].__setitem__("M35_classification", "VERIFIED"),
        )
        self.assert_no_go("frozen_contract", "summary_expectations")

    def test_live_m35_classification_tamper_forces_no_go(self):
        rel = checker.EVIDENCE["M35"][1]
        tamper_summary(self.root, rel, lambda s: s.__setitem__("classification", "VERIFIED"))
        self.assert_no_go("summary_expectations")

    def test_open_marker_regression_forces_no_go(self):
        path = self.root / checker.DOC_REL
        path.write_text(path.read_text(encoding="utf-8") + "\n`[?]`. regression\n", encoding="utf-8")
        self.assert_no_go("no_open_markers", "live_doc_sha256")

    def test_no_go_sentinel_is_written_before_final_pass(self):
        stale = {"verdict": "PASS", "stale": True}
        write_summary(self.root, stale)
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
        self.assertEqual(seen[1][1]["verdict"], "PASS")
        self.assertNotIn("sentinel", seen[1][1])
        self.assertEqual(summary, seen[1][1])

    def test_sentinel_write_failure_forces_no_go(self):
        stale = {"verdict": "PASS", "stale": True}
        write_summary(self.root, stale)
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

    def test_atomic_write_failure_keeps_old_summary_and_removes_temp(self):
        target = self.root / checker.SUMMARY_REL
        old = '{"verdict": "NO_GO"}\n'
        target.write_text(old, encoding="utf-8")

        def fail_replace(src, dst):
            raise OSError("simulated write failure")

        with mock.patch.object(checker.os, "replace", fail_replace):
            with self.assertRaisesRegex(OSError, "simulated"):
                checker.write_json(target, {"verdict": "PASS"})
        self.assertEqual(target.read_text(encoding="utf-8"), old)
        self.assertEqual(list(target.parent.glob(".*.tmp")), [])

    def test_main_exits_nonzero_when_run_check_raises(self):
        def boom(root):
            raise OSError("simulated write failure")

        with mock.patch.object(checker, "run_check", boom):
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(checker.main([]), 1)
        self.assertIn("could not write summary", err.getvalue())

    def test_checker_has_no_network_access(self):
        source = CHECKER.read_text(encoding="utf-8")
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
        self.assertIs(self.contract["provenance"]["no_network"], True)

    def test_summary_shape_check_rejects_wrong_counts(self):
        summary = checker.run_check(self.root)
        self.assertTrue(checker.check_summary_shape(summary))
        bad = json.loads(json.dumps(summary))
        bad["check_counts"]["passed"] -= 1
        self.assertFalse(checker.check_summary_shape(bad))
        bad = json.loads(json.dumps(summary))
        bad["speed_claim"] = "1.2x"
        self.assertFalse(checker.check_summary_shape(bad))

    def test_scope_has_no_speed_claim_and_keeps_cycle_audit(self):
        summary = read_summary(self.root) if (self.root / checker.SUMMARY_REL).is_file() else checker.run_check(self.root)
        self.assertIsNone(summary["speed_claim"])
        self.assertTrue(summary["no_weights"])
        self.assertTrue(summary["no_engine_run"])
        self.assertIn("claim<=ci_low", summary["cycle_audit_required"])
        self.assertIn("repeats>=20", summary["cycle_audit_required"])
        self.assertIn("loops=3", summary["cycle_audit_required"])

    def test_new_audit_files_are_ascii_only(self):
        for name in NEW_FILES:
            raw = (AUDIT_DIR / name).read_bytes()
            self.assertTrue(all(byte < 128 for byte in raw), name)


if __name__ == "__main__":
    unittest.main()
