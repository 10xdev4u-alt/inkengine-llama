"""Tests for the offline d2027 outlier-premise gate."""

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
LOOPKIT_ROOT = REPO_ROOT.parent / "loopkit"
sys.path.insert(0, str(AUDIT_DIR))

import check_outlier_premise as checker

FIXTURE_SOURCES = tuple(sorted(checker.SOURCE_ROOTS))
FIXTURE_ENGINE_FILES = {
    "src/models/inkling2.cpp": "// inkling2 engine file\n",
    "src/models/inkling2-loop.cpp": "// inkling2 loop graph\n",
    "src/llama-roret.h": "// roret wiring\n",
    "ggml/src/ggml-et/et-kernels/src/glu_f32.c": "const float a4 = -1.453152027f;\n",
    "common/chat-tts-norm.cpp": "{0x2010, 0x2027},\n",
}


def make_fixture_tree(tmp):
    root = Path(tmp) / "repo"
    lk_root = Path(tmp) / "loopkit"
    (root / "tools" / "audit").mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "OUTLIER_PREMISE.json", root / checker.FACTS_REL)
    for source in FIXTURE_SOURCES:
        root_name = checker.SOURCE_ROOTS[source]
        parent = lk_root if root_name == "loopkit" else root
        source_path = LOOPKIT_ROOT / source if root_name == "loopkit" else REPO_ROOT / source
        target = parent / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, target)
    for source, text in FIXTURE_ENGINE_FILES.items():
        target = root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


def write_facts(root, facts):
    (root / checker.FACTS_REL).write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")


def read_summary(root):
    return json.loads((root / checker.SUMMARY_REL).read_text(encoding="utf-8"))


def copy_facts(facts):
    return json.loads(json.dumps(facts))


class GateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = make_fixture_tree(self._tmp.name)
        self.facts = checker.load_json(AUDIT_DIR / "OUTLIER_PREMISE.json")

    def tearDown(self):
        self._tmp.cleanup()

    def assert_no_go(self, *failed):
        stale = {"verdict": "PASS"}
        checker.write_json(self.root / checker.SUMMARY_REL, stale)
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "NO_GO")
        self.assertTrue(summary["blocker"])
        for name in failed:
            self.assertIs(summary["checks"][name], False, name)
            self.assertIn(name, summary["blocker"])
        written = read_summary(self.root)
        self.assertEqual(written, summary)
        self.assertNotEqual(written, stale)
        self.assertEqual(written["classification"], "UNVERIFIED_GATE_FAILED")
        self.assertEqual(written["real_trace_verdict"], checker.REAL_TRACE_VERDICT)
        self.assertEqual(written["resolution"], checker.NO_GO_RESOLUTION)
        self.assertIsNone(written["speed_claim"])
        self.assertEqual(written["note"], checker.NOTE)
        return summary

    def test_frozen_json_validity(self):
        facts = self.facts
        self.assertEqual(
            {key: value for key, value in facts.items() if key != "inline_verification"},
            checker.EXPECTED_FACTS,
        )
        self.assertEqual(facts["inline_verification"]["url"], checker.INLINE_VERIFICATION_URL)
        self.assertEqual(facts["schema_version"], 1)
        self.assertEqual(facts["frozen_on"], "2026-11-28")
        self.assertEqual(facts["scope"], checker.NOTE)
        self.assertEqual(facts["note"], checker.NOTE)
        self.assertIsNone(facts["speed_claim"])
        self.assertEqual(facts["classification"], "LOCAL_ONLY_UNVERIFIED")
        self.assertEqual(facts["real_trace_verdict"], "NO_GO_BLOCKED")
        self.assertIs(facts["side_lane_enabled"], False)
        self.assertEqual(facts["resolution"], checker.RESOLUTION)
        self.assertEqual(len(facts["provenance"]), 3)
        for record in facts["provenance"]:
            self.assertEqual(len(record["sha256"]), 64)
            int(record["sha256"], 16)
            self.assertGreater(record["byte_count"], 0)
            self.assertEqual(record["occurrences_2027"], 0)
        self.assertEqual(facts["hidden_size_info"]["hidden_size"], 2304)
        self.assertEqual(facts["hidden_size_info"]["status"], "INFO_PLAUSIBLE_INDEX_NOT_EVIDENCE")
        self.assertEqual(len(facts["source_sha256"]), 6)
        self.assertTrue(all(checker.is_sha256(v) for v in facts["source_sha256"].values()))

    def test_check_names_unique_and_complete(self):
        self.assertEqual(len(checker.CHECK_NAMES), 27)
        self.assertEqual(len(checker.CHECK_NAMES), len(set(checker.CHECK_NAMES)))
        self.assertEqual(len(checker.NEEDLE_CHECK_NAMES), 8)
        self.assertEqual(
            set(checker.CHECK_NAMES),
            {
                "frozen_facts", "frozen_facts_digest", "frozen_provenance", "frozen_engine_scope",
                "frozen_classification", "frozen_hidden_size_info",
                "frozen_local_sha256", "frozen_needle_specs",
                "provenance_zero_hits", "provenance_self_consistency",
                "provenance_inline_verified", "open_findings_recorded", "live_local_sha256",
                "engine_inventory_nonempty", "engine_no_forbidden_token",
                "engine_no_d2027_literal", "side_lane_default_off",
                "no_network", "run_completed",
            }
            | set(checker.NEEDLE_CHECK_NAMES),
        )

    def test_provenance_helpers_pure(self):
        self.assertTrue(checker.check_provenance(self.facts))
        self.assertTrue(checker.check_provenance_zero_hits(self.facts))
        self.assertTrue(checker.check_provenance_self_consistency(self.facts))
        hits = copy_facts(self.facts)
        hits["provenance"][2]["occurrences_2027"] = 1
        self.assertFalse(checker.check_provenance_zero_hits(hits))
        self.assertTrue(checker.check_provenance_self_consistency(hits))
        bad_sha = copy_facts(self.facts)
        bad_sha["provenance"][0]["sha256"] = "abc"
        self.assertFalse(checker.check_provenance_self_consistency(bad_sha))
        bad_size = copy_facts(self.facts)
        bad_size["provenance"][1]["byte_count"] = 0
        self.assertFalse(checker.check_provenance_self_consistency(bad_size))
        dupe = copy_facts(self.facts)
        dupe["provenance"][1]["url"] = dupe["provenance"][0]["url"]
        self.assertFalse(checker.check_provenance_self_consistency(dupe))
        extra_key = copy_facts(self.facts)
        extra_key["provenance"][0]["extra"] = 1
        self.assertFalse(checker.check_provenance_self_consistency(extra_key))
        self.assertFalse(checker.check_provenance({}))

    def test_inline_provenance_recomputation(self):
        self.assertTrue(checker.check_provenance_inline(self.facts))
        self.assertTrue(
            checker.check_facts_digest(
                hashlib.sha256((AUDIT_DIR / "OUTLIER_PREMISE.json").read_bytes()).hexdigest()
            )
        )
        self.assertFalse(checker.check_facts_digest("0" * 64))
        bad_bytes = copy_facts(self.facts)
        bad_bytes["inline_verification"]["content"] = bad_bytes["inline_verification"]["content"][:-1]
        self.assertFalse(checker.check_provenance_inline(bad_bytes))
        bad_hits = copy_facts(self.facts)
        bad_hits["provenance"][2]["occurrences_2027"] = 1
        self.assertFalse(checker.check_provenance_inline(bad_hits))
        bad_url = copy_facts(self.facts)
        bad_url["inline_verification"]["url"] = "https://example.invalid/config.json"
        self.assertFalse(checker.check_provenance_inline(bad_url))
        bad_json = copy_facts(self.facts)
        bad_json["inline_verification"]["content"] = "{not json"
        self.assertFalse(checker.check_provenance_inline(bad_json))
        self.assertFalse(checker.check_provenance_inline({}))

    def test_inline_tamper_no_go(self):
        payload = copy_facts(self.facts)
        payload["inline_verification"]["content"] = payload["inline_verification"]["content"].replace(
            '"hidden_size": 2304', '"hidden_size": 2027'
        )
        write_facts(self.root, payload)
        self.assert_no_go("frozen_facts_digest", "provenance_inline_verified")

    def test_classification_and_hidden_size_pure(self):
        self.assertTrue(checker.check_classification(self.facts))
        self.assertTrue(checker.check_hidden_size_info(self.facts))
        for mutate in (
            lambda f: f.__setitem__("classification", "VERIFIED"),
            lambda f: f.__setitem__("real_trace_verdict", "GO"),
            lambda f: f.__setitem__("side_lane_enabled", True),
            lambda f: f.__setitem__("speed_claim", "1.0x"),
            lambda f: f.__setitem__("note", "timing"),
        ):
            payload = copy_facts(self.facts)
            mutate(payload)
            self.assertFalse(checker.check_classification(payload))
        bad_size = copy_facts(self.facts)
        bad_size["hidden_size_info"]["hidden_size"] = 2027
        self.assertFalse(checker.check_hidden_size_info(bad_size))
        bad_source = copy_facts(self.facts)
        bad_source["hidden_size_info"]["source"] = "https://example.invalid/config.json"
        self.assertFalse(checker.check_hidden_size_info(bad_source))

    def test_frozen_facts_detects_mutations(self):
        self.assertTrue(checker.check_frozen_facts(self.facts))
        self.assertFalse(checker.check_frozen_facts({}))
        for mutate in (
            lambda f: f["engine_scope"]["global_forbidden_tokens"].append("new_token"),
            lambda f: f["source_roots"].__setitem__("README.md", "loopkit"),
            lambda f: f.__setitem__("schema_version", 2),
            lambda f: f.__setitem__("resolution", "changed"),
            lambda f: f.__setitem__("open_findings", []),
        ):
            payload = copy_facts(self.facts)
            mutate(payload)
            self.assertFalse(checker.check_frozen_facts(payload))

    def test_local_sha_and_needle_specs_pure(self):
        self.assertTrue(checker.check_frozen_local_sha256(self.facts))
        self.assertTrue(checker.needle_specs_valid(self.facts["needles"]))
        self.assertTrue(checker.check_frozen_needle_specs(self.facts))
        bad_hash = copy_facts(self.facts)
        bad_hash["source_sha256"]["README.md"] = "0" * 64
        self.assertFalse(checker.check_frozen_local_sha256(bad_hash))
        missing = copy_facts(self.facts)
        del missing["source_sha256"]["README.md"]
        self.assertFalse(checker.check_frozen_local_sha256(missing))
        bad_root = copy_facts(self.facts)
        bad_root["needles"]["fork_readme_unverified"]["root"] = "loopkit"
        self.assertFalse(checker.needle_specs_valid(bad_root["needles"]))
        empty = copy_facts(self.facts)
        empty["needles"]["config_outlier_dim"]["needles"] = []
        self.assertFalse(checker.needle_specs_valid(empty["needles"]))

    def test_is_loop_source_pure(self):
        self.assertTrue(checker.is_loop_source("src/models/inkling2.cpp", ""))
        self.assertTrue(checker.is_loop_source("src/llama-roret.h", ""))
        self.assertTrue(checker.is_loop_source("some/file.cpp", "inkling2_loop"))
        self.assertTrue(checker.is_loop_source("some/file.cpp", "// INKLING2 MARKER"))
        self.assertTrue(checker.is_loop_source("some/file.cpp", "// LLAMA_PIBS"))
        self.assertFalse(checker.is_loop_source("ggml/src/kernels/glu_f32.c", "-1.453152027f"))
        self.assertFalse(checker.is_loop_source("common/chat-tts-norm.cpp", "0x2027"))

    def test_scan_engine_texts_pure(self):
        texts = {
            "src/models/inkling2.cpp": "// 2027 outlier_dim side_lane\n",
            "ggml/src/glu_f32.c": "const float a4 = -1.453152027f;\n",
            "common/chat-tts-norm.cpp": "{0x2010, 0x2027},\n",
        }
        loop_paths = {"src/models/inkling2.cpp"}
        hits = checker.scan_engine_texts(texts, loop_paths)
        self.assertEqual(len(hits["global"]), 2)
        self.assertEqual(len(hits["literal"]), 1)
        self.assertEqual(hits["literal"][0]["path"], "src/models/inkling2.cpp")
        clean = checker.scan_engine_texts(
            {"src/models/inkling2.cpp": "// inkling2\n", "ggml/src/glu_f32.c": "1.453152027f\n"},
            {"src/models/inkling2.cpp"},
        )
        self.assertEqual(clean, {"global": [], "literal": []})

    def test_check_engine_inventory_pure(self):
        required = set(checker.REQUIRED_LOOP_FILES)
        self.assertTrue(checker.check_engine_inventory(required))
        self.assertFalse(checker.check_engine_inventory(set(list(required)[:-1])))
        self.assertFalse(checker.check_engine_inventory({"src/models/inkling2.cpp"}))
        self.assertFalse(checker.check_engine_inventory(set()))

    def test_check_side_lane_default_off_pure(self):
        doc = (self.root / "tools" / "odsl" / "LAYOUT_DESIGN.md").read_text(encoding="utf-8")
        key = checker.text_key("repo", "tools/odsl/LAYOUT_DESIGN.md")
        self.assertTrue(checker.check_side_lane_default_off({key: doc}))
        self.assertFalse(checker.check_side_lane_default_off({key: doc.replace("also default off.", "enabled.")}))
        self.assertFalse(checker.check_side_lane_default_off({}))

    def test_needle_helpers_pure(self):
        spec = self.facts["needles"]["config_outlier_dim"]
        key = checker.text_key("loopkit", "configs/inkling2mini.toml")
        text = (self.root.parent / "loopkit" / "configs" / "inkling2mini.toml").read_text(encoding="utf-8")
        self.assertTrue(checker.check_needle({key: text}, spec))
        self.assertFalse(checker.check_needle({}, spec))
        self.assertFalse(checker.check_needle({key: text}, {"root": "repo", "source": "x"}))
        self.assertFalse(checker.check_needle({key: text}, {"root": "loopkit", "source": "configs/inkling2mini.toml"}))

    def test_pass_summary_fields_and_info(self):
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual(summary["blocker"], "")
        self.assertEqual(len(summary["checks"]), 27)
        self.assertTrue(all(summary["checks"].values()))
        self.assertEqual(summary["resolution"], checker.RESOLUTION)
        self.assertEqual(summary["classification"], "LOCAL_ONLY_UNVERIFIED")
        self.assertEqual(summary["real_trace_verdict"], "NO_GO_BLOCKED")
        self.assertIs(summary["side_lane_enabled"], False)
        self.assertIsNone(summary["speed_claim"])
        self.assertEqual(summary["note"], "outlier premise classification only; no timing")
        self.assertEqual(summary["frozen_on"], "2026-11-28")
        self.assertEqual(len(summary["live_source_sha256"]), 6)
        self.assertEqual(summary["engine_hits"], {"global": [], "literal": []})
        self.assertTrue(any("weights or hosted inference" in f for f in summary["open_findings"]))
        info = " ".join(summary["info"])
        self.assertIn("occurrences_2027 0", info)
        self.assertIn("INFO only, not d2027 evidence", info)
        self.assertIn("no d2027 symbol", info)
        self.assertIn("cannot re-fetch them", info)
        self.assertEqual(read_summary(self.root), summary)

    def test_cli_exit_codes(self):
        with mock.patch.object(checker, "REPO_ROOT", self.root):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(checker.main([]), 0)
            self.assertIn("verdict: PASS", out.getvalue())
            engine = self.root / "src" / "models" / "inkling2.cpp"
            with engine.open("a", encoding="utf-8") as handle:
                handle.write("// LLAMA_ODSL_SIDE_LANE=1\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main([]), 1)
        self.assertEqual(read_summary(self.root)["verdict"], "NO_GO")

    def test_mutate_primary_sha_no_go(self):
        payload = copy_facts(self.facts)
        payload["provenance"][0]["sha256"] = "0" * 64
        write_facts(self.root, payload)
        self.assert_no_go("frozen_facts", "frozen_provenance")

    def test_mutate_primary_bytes_no_go(self):
        payload = copy_facts(self.facts)
        payload["provenance"][1]["byte_count"] = 1
        write_facts(self.root, payload)
        self.assert_no_go("frozen_facts", "frozen_provenance")

    def test_mutate_primary_hits_no_go(self):
        payload = copy_facts(self.facts)
        payload["provenance"][2]["occurrences_2027"] = 1
        write_facts(self.root, payload)
        self.assert_no_go("frozen_facts", "frozen_provenance", "provenance_zero_hits")

    def test_mutate_needle_no_go(self):
        payload = copy_facts(self.facts)
        payload["needles"]["config_outlier_dim"]["needles"] = ["outlier_dim = 2028"]
        write_facts(self.root, payload)
        self.assert_no_go("frozen_facts", "needle_config_outlier_dim")

    def test_live_hash_mismatch_no_go(self):
        config = self.root.parent / "loopkit" / "configs" / "inkling2mini.toml"
        with config.open("a", encoding="utf-8") as handle:
            handle.write("# tampered\n")
        self.assert_no_go("live_local_sha256")

    def test_missing_local_file_no_go(self):
        (self.root.parent / "loopkit" / "configs" / "inkling2mini.toml").unlink()
        self.assert_no_go("live_local_sha256", "needle_config_outlier_dim")

    def test_enabled_side_lane_no_go(self):
        engine = self.root / "src" / "models" / "inkling2.cpp"
        with engine.open("a", encoding="utf-8") as handle:
            handle.write("// LLAMA_ODSL_SIDE_LANE=1 side_lane enabled\n")
        summary = self.assert_no_go("engine_no_forbidden_token", "side_lane_default_off")
        self.assertIs(summary["side_lane_enabled"], True)

    def test_d2027_literal_in_loop_source_no_go(self):
        engine = self.root / "src" / "models" / "inkling2-loop.cpp"
        with engine.open("a", encoding="utf-8") as handle:
            handle.write("// 2027\n")
        self.assert_no_go("engine_no_d2027_literal")

    def test_unrelated_2027_literals_stay_pass(self):
        unrelated = self.root / "ggml" / "src" / "sycl-2027.cpp"
        unrelated.write_text("// oneDNN until 2027.0; U+2027; -1.453152027f\n", encoding="utf-8")
        seed = {"verdict": "FAIL"}
        checker.write_json(self.root / checker.SUMMARY_REL, seed)
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual(summary["engine_hits"], {"global": [], "literal": []})
        self.assertEqual(read_summary(self.root), summary)

    def test_malformed_facts_json_no_go(self):
        (self.root / checker.FACTS_REL).write_text("{not json", encoding="utf-8")
        self.assert_no_go("run_completed", "frozen_facts")

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
        for token in ("import socket", "import requests", "urlopen(", "urlretrieve(", "import loopkit"):
            self.assertNotIn(token, source)
        self.assertTrue(checker.check_no_network())

    def test_new_audit_files_are_ascii_only(self):
        for name in (
            "OUTLIER_PREMISE.json", "check_outlier_premise.py",
            "test_outlier_premise.py", "OUTLIER_PREMISE.md",
            "M35_OUTLIER_PREMISE_SUMMARY.json",
        ):
            raw = (AUDIT_DIR / name).read_bytes()
            self.assertTrue(all(byte < 128 for byte in raw), name)
        self.assertEqual(
            hashlib.sha256((AUDIT_DIR / "OUTLIER_PREMISE.json").read_bytes()).hexdigest(),
            checker.run_check(self.root)["frozen_facts_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
