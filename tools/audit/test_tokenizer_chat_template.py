"""Behavior tests for the fail-closed tokenizer and chat-template contract gate."""

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
CHECKER = AUDIT_DIR / "check_tokenizer_chat_template.py"
sys.path.insert(0, str(AUDIT_DIR))

import check_tokenizer_chat_template as checker

EXPECTED_TOTAL_CHECKS = 26
EXPECTED_CHECKER_SHA256 = "e187bb9ac6914987b40bee59452633a585984dc65d0aecb4ef3766ab79ee0e3f"
NEW_FILES = [
    "TOKENIZER_CHAT_TEMPLATE.json",
    "check_tokenizer_chat_template.py",
    "test_tokenizer_chat_template.py",
    "TOKENIZER_CHAT_TEMPLATE.md",
    "M47_TOKENIZER_CHAT_TEMPLATE_SUMMARY.json",
]


def make_fixture_tree(tmp):
    root = Path(tmp) / "repo"
    (root / "tools" / "audit").mkdir(parents=True)
    shutil.copy(AUDIT_DIR / "TOKENIZER_CHAT_TEMPLATE.json", root / checker.CONTRACT_REL)
    shutil.copy(AUDIT_DIR / "GGUF_LOOP_SHAPES.json", root / checker.M37_REL)
    dst = root / checker.CONVERTER_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / checker.CONVERTER_REL, dst)
    return root


def write_contract(root, payload):
    (root / checker.CONTRACT_REL).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def tamper_contract(root, mutate):
    payload = checker.load_json(root / checker.CONTRACT_REL)
    mutate(payload)
    write_contract(root, payload)


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
        self.contract = checker.load_json(AUDIT_DIR / "TOKENIZER_CHAT_TEMPLATE.json")

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

    def test_contract_structure_and_pins(self):
        c = self.contract
        self.assertEqual(set(c), checker.EXPECTED_TOP_KEYS)
        self.assertEqual(c["schema_version"], 1)
        self.assertEqual(c["frozen_on"], "2026-11-28")
        self.assertEqual(c["scope"], checker.NOTE)
        self.assertEqual(c["note"], checker.NOTE)
        self.assertEqual(c["description"], checker.DESCRIPTION)
        self.assertEqual(c["summary_path"], checker.SUMMARY_REL)
        self.assertEqual(c["checker_sha256"], EXPECTED_CHECKER_SHA256)
        self.assertEqual(c["cycle_audit_required"], checker.CYCLE_AUDIT_REQUIRED)
        self.assertEqual(c["doc_note"], checker.DOC_NOTE)
        self.assertIsNone(c["speed_claim"])
        self.assertTrue(checker.check_frozen_contract(c))
        self.assertEqual(checker.strip_embedded_content(c)["hf"], checker.EXPECTED_HF)
        for digest in (c["hf"]["chat_template"]["sha256"], c["gguf"]["kv"]["tokenizer.chat_template"]["sha256"]):
            self.assertTrue(checker.is_sha256(digest))
            self.assertEqual(digest, checker.CHAT_TEMPLATE_SHA256)

    def test_check_names_unique_and_complete(self):
        self.assertEqual(len(checker.CHECK_NAMES), EXPECTED_TOTAL_CHECKS)
        self.assertEqual(len(checker.CHECK_NAMES), len(set(checker.CHECK_NAMES)))
        self.assertEqual(
            set(checker.CHECK_NAMES),
            {
                "frozen_contract", "frozen_checker_sha256", "contract_summary_path",
                "embedded_chat_template_bytes", "embedded_chat_template_ascii",
                "embedded_chat_template_needles", "embedded_special_tokens_bytes",
                "embedded_special_tokens_json", "tokenizer_config_facts",
                "tokenizer_json_facts", "frozen_template_digest_consistency",
                "frozen_template_length_consistency", "eos_pad_id_mapping",
                "gguf_tokenizer_kvs", "gguf_special_id_kvs", "gguf_vocab_consistency",
                "gguf_header_provenance", "gguf_no_cache_read",
                "converter_live_sha256", "converter_needles", "m37_gguf_sha256",
                "summary_expectations", "no_network", "sentinel_write",
                "run_completed", "summary_shape",
            },
        )
        self.assertEqual(len(checker.EXPECTED_SUMMARY_KEYS), 18)

    def test_embedded_chat_template_matches_pinned_primary_source_digest(self):
        section = self.contract["hf"]["chat_template"]
        raw = section["content"].encode("ascii")
        self.assertEqual(len(raw), 6868)
        self.assertEqual(checker.sha256_bytes(raw), checker.CHAT_TEMPLATE_SHA256)
        self.assertEqual(section["bytes"], len(raw))
        self.assertEqual(section["sha256"], checker.sha256_bytes(raw))
        self.assertEqual(
            self.contract["gguf"]["kv"]["tokenizer.chat_template"]["sha256"],
            checker.sha256_bytes(raw),
        )
        self.assertTrue(checker.check_embedded_chat_template(self.contract))
        self.assertTrue(checker.check_chat_template_ascii(self.contract))
        self.assertTrue(checker.check_chat_template_needles(self.contract))

    def test_embedded_special_tokens_match_pinned_primary_source_digest(self):
        section = self.contract["hf"]["special_tokens_map"]
        raw = section["content"].encode("ascii")
        self.assertEqual(len(raw), 529)
        self.assertEqual(checker.sha256_bytes(raw), checker.SPECIAL_MAP_SHA256)
        self.assertEqual(section["bytes"], len(raw))
        self.assertEqual(section["sha256"], checker.sha256_bytes(raw))
        self.assertTrue(checker.check_embedded_special_tokens(self.contract))
        self.assertTrue(checker.check_special_tokens_json(self.contract))

    def test_chat_template_pin_literals(self):
        self.assertEqual(checker.CHAT_TEMPLATE_BYTES, 6868)
        self.assertEqual(
            checker.CHAT_TEMPLATE_SHA256,
            "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc",
        )
        self.assertEqual(checker.CHAT_TEMPLATE_SHA256, checker.GGUF_CHAT_TEMPLATE_SHA256)
        content = self.contract["hf"]["chat_template"]["content"]
        for needle in checker.CHAT_TEMPLATE_NEEDLES:
            self.assertIn(needle, content)
        self.assertTrue(content.startswith(checker.CHAT_TEMPLATE_NEEDLES[0]))

    def test_tokenizer_fact_literals(self):
        self.assertEqual(checker.SPECIAL_MAP_BYTES, 529)
        self.assertEqual(checker.TOKENIZER_CONFIG_BYTES, 13235)
        self.assertEqual(checker.TOKENIZER_JSON_BYTES, 26178678)
        self.assertEqual(
            checker.TOKENIZER_JSON_SHA256,
            "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6",
        )
        self.assertEqual(checker.VOCAB_ENTRIES, 199998)
        self.assertEqual(checker.ADDED_TOKEN_IDS["200006"], checker.EOS_TOKEN)
        self.assertEqual(checker.ADDED_TOKEN_IDS["199999"], checker.PAD_TOKEN)
        self.assertTrue(checker.check_tokenizer_config_facts(self.contract))
        self.assertTrue(checker.check_tokenizer_json_facts(self.contract))

    def test_no_inline_chat_template_in_tokenizer_config(self):
        config = self.contract["hf"]["tokenizer_config"]
        self.assertNotIn("chat_template", config["keys"])
        self.assertIs(config["inline_chat_template_key"], False)
        self.assertEqual(config["tokenizer_class"], "PreTrainedTokenizerFast")
        self.assertEqual(config["model_max_length"], 1048576)
        self.assertTrue(checker.check_tokenizer_config_facts(self.contract))

    def test_special_token_id_mapping_literals(self):
        self.assertEqual(checker.EOS_TOKEN, "<|content_model_end_sampling|>")
        self.assertEqual(checker.PAD_TOKEN, "<|endoftext|>")
        self.assertEqual(checker.EOS_TOKEN_ID, 200006)
        self.assertEqual(checker.PAD_TOKEN_ID, 199999)
        self.assertTrue(checker.check_eos_pad_id_mapping(self.contract))
        inverse = {v: int(k) for k, v in checker.ADDED_TOKEN_IDS.items()}
        self.assertEqual(inverse[checker.EOS_TOKEN], checker.EOS_TOKEN_ID)
        self.assertEqual(inverse[checker.PAD_TOKEN], checker.PAD_TOKEN_ID)

    def test_gguf_kv_and_header_pin_literals(self):
        self.assertEqual(checker.GGUF_SHA256, "87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76")
        kv = self.contract["gguf"]["kv"]
        self.assertEqual(kv["tokenizer.ggml.model"], "gpt2")
        self.assertEqual(kv["tokenizer.ggml.pre"], "inkling2")
        self.assertIs(kv["tokenizer.ggml.add_bos_token"], False)
        self.assertIs(kv["tokenizer.ggml.add_eos_token"], False)
        self.assertIs(kv["tokenizer.ggml.clean_spaces"], False)
        self.assertEqual(kv["tokenizer.ggml.eos_token_id"], kv["tokenizer.ggml.eot_token_id"])
        self.assertEqual(self.contract["gguf"]["n_tensors"], 744)
        self.assertEqual(self.contract["gguf"]["n_kv"], 63)
        self.assertTrue(checker.check_gguf_tokenizer_kvs(self.contract))
        self.assertTrue(checker.check_gguf_special_id_kvs(self.contract))
        self.assertTrue(checker.check_gguf_vocab_consistency(self.contract))
        self.assertTrue(checker.check_gguf_header_provenance(self.contract))

    def test_vocab_ordering_and_added_id_bounds(self):
        kv = self.contract["gguf"]["kv"]
        self.assertGreater(kv["inkling2_loop.vocab_size"], kv["inkling2_loop.vocab_size_unpadded"])
        self.assertGreater(kv["inkling2_loop.vocab_size_unpadded"], checker.EOS_TOKEN_ID)
        self.assertGreater(checker.EOS_TOKEN_ID, checker.PAD_TOKEN_ID)
        self.assertGreater(checker.UNPADDED_VOCAB_SIZE, max(int(k) for k in checker.ADDED_TOKEN_IDS))
        self.assertGreater(checker.UNPADDED_VOCAB_SIZE, checker.VOCAB_ENTRIES)

    def test_converter_needles_present_in_live_source(self):
        text = (REPO_ROOT / checker.CONVERTER_REL).read_text(encoding="utf-8")
        self.assertEqual(len(checker.CONVERTER_NEEDLES), 13)
        for needle in checker.CONVERTER_NEEDLES:
            self.assertIn(needle, text)
        self.assertTrue(checker.check_converter_needles(REPO_ROOT))
        self.assertTrue(checker.check_live_converter_sha256(REPO_ROOT))

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
        self.assertEqual(summary["checker_sha256"], EXPECTED_CHECKER_SHA256)
        self.assertEqual(
            hashlib.sha256((self.root / checker.CONTRACT_REL).read_bytes()).hexdigest(),
            summary["contract_sha256"],
        )
        self.assertIn("counts: 26/26 passed, 0 failed", result.stdout)

    def test_real_tree_cli_passes_and_restores_summary_bytes(self):
        summary_path = REPO_ROOT / checker.SUMMARY_REL
        original = summary_path.read_bytes()
        try:
            result = run_cli(REPO_ROOT)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = read_summary(REPO_ROOT)
            self.assertEqual(summary["verdict"], "PASS")
            self.assertEqual(summary["blocker"], "")
            self.assertTrue(all(value is True for value in summary["checks"].values()))
            self.assertEqual(summary["checker_sha256"], EXPECTED_CHECKER_SHA256)
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
        self.assertTrue(sentinel["no_weights"])
        self.assertTrue(sentinel["no_engine_run"])

    def test_tampered_template_bytes_force_no_go(self):
        def mutate(c):
            section = c["hf"]["chat_template"]
            section["content"] = section["content"].replace("emit_thinking_effort", "emit_thinking_efforx")
        tamper_contract(self.root, mutate)
        self.assert_no_go(
            "embedded_chat_template_bytes", "embedded_chat_template_needles",
            "frozen_template_digest_consistency",
        )

    def test_template_length_tamper_forces_no_go(self):
        tamper_contract(
            self.root,
            lambda c: c["hf"]["chat_template"].__setitem__("bytes", 6869),
        )
        self.assert_no_go("frozen_contract", "frozen_template_length_consistency")

    def test_tampered_template_digest_forces_no_go(self):
        tamper_contract(
            self.root,
            lambda c: c["hf"]["chat_template"].__setitem__("sha256", "0" * 64),
        )
        summary = self.assert_no_go("frozen_contract", "frozen_template_digest_consistency")
        self.assertIn("template parity check FAILED", summary["info"])
        self.assertFalse(any("byte-identical" in line for line in summary["info"]))

    def test_wrong_special_token_id_forces_no_go(self):
        tamper_contract(
            self.root,
            lambda c: c["gguf"]["kv"].__setitem__("tokenizer.ggml.eos_token_id", 200007),
        )
        self.assert_no_go("frozen_contract", "eos_pad_id_mapping", "gguf_special_id_kvs")

    def test_wrong_pad_id_forces_no_go(self):
        tamper_contract(
            self.root,
            lambda c: c["gguf"]["kv"].__setitem__("tokenizer.ggml.padding_token_id", 199998),
        )
        self.assert_no_go("frozen_contract", "eos_pad_id_mapping", "gguf_special_id_kvs")

    def test_wrong_gguf_kv_forces_no_go(self):
        tamper_contract(
            self.root,
            lambda c: c["gguf"]["kv"].__setitem__("tokenizer.ggml.add_bos_token", True),
        )
        self.assert_no_go("frozen_contract", "gguf_tokenizer_kvs")

    def test_wrong_gguf_header_fact_forces_no_go(self):
        tamper_contract(self.root, lambda c: c["gguf"].__setitem__("n_kv", 62))
        self.assert_no_go("frozen_contract", "gguf_header_provenance")

    def test_missing_converter_needle_forces_no_go(self):
        path = self.root / checker.CONVERTER_REL
        text = path.read_text(encoding="utf-8")
        needle = checker.CONVERTER_NEEDLES[0]
        self.assertIn(needle, text)
        path.write_text(text.replace(needle, "REMOVED"), encoding="utf-8")
        self.assert_no_go("converter_live_sha256", "converter_needles")

    def test_contract_needle_tamper_forces_no_go(self):
        tamper_contract(
            self.root,
            lambda c: c["converter"]["needles"].__setitem__(0, "REMOVED"),
        )
        self.assert_no_go("frozen_contract")

    def test_live_converter_sha_mismatch_forces_no_go(self):
        path = self.root / checker.CONVERTER_REL
        path.write_text(path.read_text(encoding="utf-8") + "\n# tamper\n", encoding="utf-8")
        self.assert_no_go("converter_live_sha256")

    def test_checker_hash_drift_forces_no_go(self):
        tamper_contract(self.root, lambda c: c.__setitem__("checker_sha256", "0" * 64))
        self.assert_no_go("frozen_checker_sha256")

    def test_extra_top_level_key_forces_no_go(self):
        tamper_contract(self.root, lambda c: c.__setitem__("extra_unknown_key", "hello"))
        self.assert_no_go("frozen_contract")

    def test_malformed_contract_forces_no_go(self):
        (self.root / checker.CONTRACT_REL).write_text("{not json\n", encoding="utf-8")
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "NO_GO")
        self.assertIs(summary["checks"]["run_completed"], False)
        self.assertTrue(any("exception:" in line for line in summary["info"]))
        self.assertTrue(checker.is_sha256(summary["contract_sha256"]))

    def test_missing_contract_forces_no_go(self):
        (self.root / checker.CONTRACT_REL).unlink()
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "NO_GO")
        self.assertIs(summary["checks"]["run_completed"], False)
        self.assertIs(summary["checks"]["frozen_contract"], False)

    def test_missing_m37_contract_forces_no_go(self):
        (self.root / checker.M37_REL).unlink()
        self.assert_no_go("m37_gguf_sha256")

    def test_m37_gguf_sha_tamper_forces_no_go(self):
        path = self.root / checker.M37_REL
        payload = checker.load_json(path)
        payload["provenance"]["first_16mib_sha256"] = "0" * 64
        checker.write_json(path, payload)
        self.assert_no_go("m37_gguf_sha256")

    def test_missing_converter_forces_no_go(self):
        (self.root / checker.CONVERTER_REL).unlink()
        self.assert_no_go("converter_live_sha256", "converter_needles")

    def test_stale_pass_is_overwritten_with_no_go(self):
        stale = {"verdict": "PASS", "stale": True}
        checker.write_json(self.root / checker.SUMMARY_REL, stale)
        tamper_contract(self.root, lambda c: c.__setitem__("schema_version", 2))
        summary = checker.run_check(self.root)
        self.assertEqual(summary["verdict"], "NO_GO")
        written = read_summary(self.root)
        self.assertNotEqual(written, stale)
        self.assertEqual(written, summary)
        self.assertIs(written["checks"]["frozen_contract"], False)
        self.assertIs(written["checks"]["summary_shape"], True)
        self.assertEqual(written["check_counts"]["failed"], 1)

    def test_no_go_sentinel_is_written_before_checks(self):
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
        self.assertIn("sentinel_write", summary["blocker"])
        self.assertIs(summary["checks"]["sentinel_write"], False)

    def test_atomic_write_failure_keeps_previous_summary_and_removes_temp(self):
        target = self.root / "tools/audit/M47_TOKENIZER_CHAT_TEMPLATE_SUMMARY.json"
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
                self.assertNotIn(node.func.id, {"urlopen", "urlretrieve", "open"})
        self.assertTrue(checker.check_no_network())
        self.assertIs(self.contract["gguf"]["network_access_in_checker"], False)
        self.assertIs(self.contract["hf"]["network_access_in_checker"], False)

    def test_checker_never_reads_the_cached_gguf(self):
        source = CHECKER.read_text(encoding="utf-8")
        lookup = "[" + chr(34) + "header_cache_file" + chr(34) + "]"
        self.assertNotIn(lookup, source)
        self.assertTrue(checker.check_no_cache_read())
        self.assertIs(self.contract["gguf"]["checker_reads_cache"], False)
        self.assertIs(self.contract["gguf"]["long_arrays_fully_validated"], False)

    def test_summary_shape_checks_structure_not_verdict(self):
        checker.run_check(self.root)
        summary = read_summary(self.root)
        self.assertTrue(checker.check_summary_shape(summary))
        no_go = json.loads(json.dumps(summary))
        no_go["verdict"] = "NO_GO"
        no_go["blocker"] = "frozen_contract"
        no_go["checks"]["frozen_contract"] = False
        no_go["check_counts"] = {"total": EXPECTED_TOTAL_CHECKS, "passed": 25, "failed": 1}
        self.assertTrue(checker.check_summary_shape(no_go))
        bad = json.loads(json.dumps(no_go))
        bad["verdict"] = "UNKNOWN"
        self.assertFalse(checker.check_summary_shape(bad))
        bad = json.loads(json.dumps(no_go))
        bad["checks"]["unexpected"] = True
        self.assertFalse(checker.check_summary_shape(bad))
        bad = json.loads(json.dumps(no_go))
        bad["checks"]["summary_shape"] = "yes"
        self.assertFalse(checker.check_summary_shape(bad))
        bad = json.loads(json.dumps(no_go))
        bad["check_counts"]["failed"] = "1"
        self.assertFalse(checker.check_summary_shape(bad))
        bad = json.loads(json.dumps(no_go))
        bad["contract_sha256"] = "xyz"
        self.assertFalse(checker.check_summary_shape(bad))
        bad = json.loads(json.dumps(no_go))
        bad["speed_claim"] = "2x"
        self.assertFalse(checker.check_summary_shape(bad))
        bad = json.loads(json.dumps(no_go))
        bad["sentinel"] = True
        self.assertFalse(checker.check_summary_shape(bad))

    def test_real_no_go_summary_still_passes_shape(self):
        tamper_contract(self.root, lambda c: c.__setitem__("schema_version", 2))
        summary = self.assert_no_go("frozen_contract")
        self.assertEqual(summary["check_counts"]["failed"], 1)
        self.assertIs(summary["checks"]["summary_shape"], True)
        self.assertTrue(checker.check_summary_shape(summary))

    def test_doc_note_and_cycle_audit_are_in_summary(self):
        checker.run_check(self.root)
        summary = read_summary(self.root)
        self.assertIn("item (b)", summary["doc_note"])
        self.assertIn("NO_GO_BLOCKED", summary["doc_note"])
        self.assertIn("coordinated two-repo doc update", summary["doc_note"])
        self.assertIn("CycleAudit", summary["cycle_audit_required"])
        self.assertIn("claim<=ci_low", summary["cycle_audit_required"])
        self.assertIn("repeats>=20", summary["cycle_audit_required"])
        self.assertIn("loops=3", summary["cycle_audit_required"])

    def test_new_audit_files_are_ascii_only_and_tab_free(self):
        for name in NEW_FILES:
            raw = (AUDIT_DIR / name).read_bytes()
            self.assertTrue(all(byte < 128 for byte in raw), name)
            self.assertNotIn(b"\t", raw, name)


if __name__ == "__main__":
    unittest.main()
