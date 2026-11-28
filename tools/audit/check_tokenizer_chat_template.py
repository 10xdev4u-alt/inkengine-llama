#!/usr/bin/env python3
"""Offline, fail-closed checker for the frozen tokenizer and chat-template contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
CONTRACT_REL = "tools/audit/TOKENIZER_CHAT_TEMPLATE.json"
SUMMARY_REL = "tools/audit/M47_TOKENIZER_CHAT_TEMPLATE_SUMMARY.json"
M37_REL = "tools/audit/GGUF_LOOP_SHAPES.json"
CONVERTER_REL = "conversion/inkling2.py"
NOTE = "tokenizer and chat-template contract only; no weights, no engine run, no timing"
DESCRIPTION = (
    "M47 freezes the thinkingmachines/Inkling-2-Mini chat template, tokenizer "
    "special tokens and bounded tokenizer facts. an out-of-band bounded probe "
    "measured byte equality between the GGUF tokenizer.chat_template KV and "
    "the HF chat_template.jinja; the offline checker verifies the frozen "
    "digests and lengths are internally consistent and source-pinned, and "
    "cannot observe live GGUF drift. it closes section-3 item (e) as a "
    "contract gate only."
)
DOC_NOTE = (
    "loopkit docs/INKLING_INTEGRATION.md item (e) marker stays pending a "
    "coordinated two-repo doc update; item (b) real-weight conversion parity "
    "remains NO_GO_BLOCKED"
)
CYCLE_AUDIT_REQUIRED = (
    "NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains "
    "mandatory for any future number."
)
FROZEN_ON = "2026-11-28"
SCHEMA_VERSION = 1

CHAT_TEMPLATE_BYTES = 6868
CHAT_TEMPLATE_SHA256 = "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc"
GGUF_CHAT_TEMPLATE_SHA256 = "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc"
SPECIAL_MAP_BYTES = 529
SPECIAL_MAP_SHA256 = "c16f44b3594563d82743ddb0417a2be1bfe3e2000dfad2da4b8f32554593e1a0"
TOKENIZER_CONFIG_BYTES = 13235
TOKENIZER_CONFIG_SHA256 = "bf09342d285139fd37ce10c70878af1d52cb390e9901e10248654293a80ee636"
TOKENIZER_JSON_BYTES = 26178678
TOKENIZER_JSON_SHA256 = "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6"
GGUF_SHA256 = "87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76"
CONVERTER_SHA256 = "aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7"
EOS_TOKEN = "<|content_model_end_sampling|>"
PAD_TOKEN = "<|endoftext|>"
EOS_TOKEN_ID = 200006
PAD_TOKEN_ID = 199999
VOCAB_SIZE = 201024
UNPADDED_VOCAB_SIZE = 200058
VOCAB_ENTRIES = 199998

CHAT_TEMPLATE_NEEDLES = [
    '{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}',
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "emit_thinking_effort",
]
ROLE_TOKENS = [
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
]
ADDITIONAL_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_image|>",
    "<|content_model_end_sampling|>",
    "<|content_thinking|>",
    "<|end_message|>",
    "<|content_tool_error|>",
    "<|content_xml|>",
    "<|begin_of_text|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
]
ADDED_TOKEN_IDS = {
    "199999": PAD_TOKEN,
    "200000": "<|message_user|>",
    "200001": "<|message_model|>",
    "200002": "<|message_system|>",
    "200003": "<|message_tool|>",
    "200006": EOS_TOKEN,
}
TOKENIZER_CONFIG_KEYS = [
    "added_tokens_decoder",
    "clean_up_tokenization_spaces",
    "eos_token",
    "extra_special_tokens",
    "fix_mistral_regex",
    "model_max_length",
    "pad_token",
    "response_template",
    "tokenizer_class",
]

CONVERTER_NEEDLES = [
    "tokens, types, pre = self.get_vocab_base()",
    "return tokens[:n], types[:n], pre",
    'self.gguf_writer.add_tokenizer_model("gpt2")',
    "self.gguf_writer.add_tokenizer_pre(pre)",
    "self.gguf_writer.add_token_list(tokens)",
    "self.gguf_writer.add_token_types(types)",
    "special = gguf.SpecialVocab(self.dir_model, load_merges=True)",
    'eos = self.hparams["eos_token_id"]',
    'special.special_token_ids["eos"] = eos',
    'special.special_token_ids["eot"] = eos',
    'special.add_special_token["bos"] = False',
    'w.add_vocab_size(hp["vocab_size"])',
    'w.add_vocab_size_unpadded(hp["unpadded_vocab_size"])',
]

EXPECTED_HF = {
    "repo": "thinkingmachines/Inkling-2-Mini",
    "commit": "c9d028302cd0f035fdf8e7e91f089a4ce94d2195",
    "fetched_out_of_band": True,
    "network_access_in_checker": False,
    "chat_template": {
        "filename": "chat_template.jinja",
        "bytes": CHAT_TEMPLATE_BYTES,
        "sha256": CHAT_TEMPLATE_SHA256,
        "needles": CHAT_TEMPLATE_NEEDLES,
    },
    "tokenizer_config": {
        "filename": "tokenizer_config.json",
        "bytes": TOKENIZER_CONFIG_BYTES,
        "sha256": TOKENIZER_CONFIG_SHA256,
        "keys": TOKENIZER_CONFIG_KEYS,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 1048576,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "inline_chat_template_key": False,
    },
    "special_tokens_map": {
        "filename": "special_tokens_map.json",
        "bytes": SPECIAL_MAP_BYTES,
        "sha256": SPECIAL_MAP_SHA256,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "role_tokens": ROLE_TOKENS,
        "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS,
    },
    "tokenizer_json": {
        "filename": "tokenizer.json",
        "bytes": TOKENIZER_JSON_BYTES,
        "sha256": TOKENIZER_JSON_SHA256,
        "model_type": "BPE",
        "vocab_entries": VOCAB_ENTRIES,
        "added_token_ids": ADDED_TOKEN_IDS,
    },
}

EXPECTED_GGUF = {
    "url": "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-IQ1_S.gguf",
    "range": "bytes=0-16777215",
    "byte_start": 0,
    "byte_end": 16777215,
    "fetched_bytes": 16777216,
    "first_16mib_sha256": GGUF_SHA256,
    "magic": "GGUF",
    "version": 3,
    "n_tensors": 744,
    "n_kv": 63,
    "header_cache_file": "/tmp/opencode/loopkit-m37/gguf_first16.gguf",
    "network_access_in_checker": False,
    "checker_reads_cache": False,
    "long_arrays_fully_validated": False,
    "kv": {
        "tokenizer.chat_template": {
            "bytes": CHAT_TEMPLATE_BYTES,
            "sha256": CHAT_TEMPLATE_SHA256,
            "utf8": True,
        },
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "inkling2",
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_eos_token": False,
        "tokenizer.ggml.clean_spaces": False,
        "tokenizer.ggml.eos_token_id": EOS_TOKEN_ID,
        "tokenizer.ggml.eot_token_id": EOS_TOKEN_ID,
        "tokenizer.ggml.padding_token_id": PAD_TOKEN_ID,
        "inkling2_loop.vocab_size": VOCAB_SIZE,
        "inkling2_loop.vocab_size_unpadded": UNPADDED_VOCAB_SIZE,
    },
    "note": (
        "The first 16 MiB were fetched once out of band. This checker makes no "
        "network calls and never opens the cached GGUF file. The long token, "
        "token-type and merge arrays are not claimed to be fully validated."
    ),
}

EXPECTED_CONVERTER = {
    "path": CONVERTER_REL,
    "sha256": CONVERTER_SHA256,
    "needles": CONVERTER_NEEDLES,
}

EXPECTED_CROSS_ARTIFACT = {
    "m37_contract": M37_REL,
    "m37_first_16mib_sha256": GGUF_SHA256,
}

EXPECTED_SUMMARY_EXPECTATIONS = {
    "verdict": "PASS",
    "no_weights": True,
    "no_engine_run": True,
    "speed_claim": None,
}

EXPECTED_CONTRACT_BODY = {
    "schema_version": SCHEMA_VERSION,
    "frozen_on": FROZEN_ON,
    "scope": NOTE,
    "note": NOTE,
    "description": DESCRIPTION,
    "summary_path": SUMMARY_REL,
    "hf": EXPECTED_HF,
    "gguf": EXPECTED_GGUF,
    "converter": EXPECTED_CONVERTER,
    "cross_artifact": EXPECTED_CROSS_ARTIFACT,
    "summary_expectations": EXPECTED_SUMMARY_EXPECTATIONS,
    "doc_note": DOC_NOTE,
    "cycle_audit_required": CYCLE_AUDIT_REQUIRED,
    "speed_claim": None,
}
EXPECTED_TOP_KEYS = set(EXPECTED_CONTRACT_BODY) | {"checker_sha256"}

EXPECTED_SUMMARY_KEYS = {
    "verdict", "blocker", "checks", "check_counts", "info", "note", "scope",
    "speed_claim", "cycle_audit_required", "no_weights", "no_engine_run",
    "frozen_on", "summary_path", "checker_sha256", "contract_sha256",
    "hf_chat_template_sha256", "gguf_chat_template_sha256", "doc_note",
}

CHECK_NAMES = [
    "frozen_contract",
    "frozen_checker_sha256",
    "contract_summary_path",
    "embedded_chat_template_bytes",
    "embedded_chat_template_ascii",
    "embedded_chat_template_needles",
    "embedded_special_tokens_bytes",
    "embedded_special_tokens_json",
    "tokenizer_config_facts",
    "tokenizer_json_facts",
    "frozen_template_digest_consistency",
    "frozen_template_length_consistency",
    "eos_pad_id_mapping",
    "gguf_tokenizer_kvs",
    "gguf_special_id_kvs",
    "gguf_vocab_consistency",
    "gguf_header_provenance",
    "gguf_no_cache_read",
    "converter_live_sha256",
    "converter_needles",
    "m37_gguf_sha256",
    "summary_expectations",
    "no_network",
    "sentinel_write",
    "run_completed",
    "summary_shape",
]
NETWORK_MODULES = {"socket", "http", "urllib", "requests", "httpx", "ftplib", "smtplib", "telnetlib", "xmlrpc"}
NETWORK_CALLS = {"urlopen", "urlretrieve", "socket", "connect"}
HEX_DIGITS = frozenset("0123456789abcdef")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    """Write via a temp file in the same directory, then replace atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_text(path):
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in HEX_DIGITS for char in value)


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def strip_embedded_content(facts):
    """Project the two byte-bearing contract sections down to their metadata."""
    if not isinstance(facts, dict):
        return facts
    out = dict(facts)
    hf = dict(out.get("hf") or {})
    for key in ("chat_template", "special_tokens_map"):
        if isinstance(hf.get(key), dict):
            section = dict(hf[key])
            section.pop("content", None)
            hf[key] = section
    out["hf"] = hf
    return out


def fail_closed_summary():
    """Startup sentinel: never leave a stale PASS if the run is killed early."""
    return {
        "verdict": "NO_GO",
        "blocker": "checker did not complete",
        "checks": {name: False for name in CHECK_NAMES},
        "check_counts": {"total": len(CHECK_NAMES), "passed": 0, "failed": len(CHECK_NAMES)},
        "info": ["NO_GO sentinel written at checker startup"],
        "note": NOTE,
        "scope": NOTE,
        "speed_claim": None,
        "cycle_audit_required": CYCLE_AUDIT_REQUIRED,
        "no_weights": True,
        "no_engine_run": True,
        "frozen_on": FROZEN_ON,
        "summary_path": SUMMARY_REL,
        "checker_sha256": None,
        "contract_sha256": None,
        "hf_chat_template_sha256": None,
        "gguf_chat_template_sha256": None,
        "doc_note": DOC_NOTE,
        "sentinel": True,
    }


def check_frozen_contract(facts):
    if not isinstance(facts, dict) or set(facts) != EXPECTED_TOP_KEYS:
        return False
    public = strip_embedded_content(facts)
    public.pop("checker_sha256", None)
    return public == EXPECTED_CONTRACT_BODY


def check_embedded_chat_template(facts):
    try:
        content = facts["hf"]["chat_template"]["content"]
        if not isinstance(content, str):
            return False
        data = content.encode("ascii")
    except (KeyError, TypeError, AttributeError, UnicodeEncodeError):
        return False
    return len(data) == CHAT_TEMPLATE_BYTES and sha256_bytes(data) == CHAT_TEMPLATE_SHA256


def check_chat_template_ascii(facts):
    try:
        content = facts["hf"]["chat_template"]["content"]
    except (KeyError, TypeError):
        return False
    return isinstance(content, str) and content.isascii() and "\t" not in content


def check_chat_template_needles(facts):
    try:
        content = facts["hf"]["chat_template"]["content"]
        needles = facts["hf"]["chat_template"]["needles"]
    except (KeyError, TypeError):
        return False
    return bool(content) and needles == CHAT_TEMPLATE_NEEDLES and all(
        isinstance(needle, str) and needle in content for needle in needles
    )


def check_embedded_special_tokens(facts):
    try:
        content = facts["hf"]["special_tokens_map"]["content"]
        if not isinstance(content, str) or not content.isascii():
            return False
        data = content.encode("ascii")
    except (KeyError, TypeError, AttributeError):
        return False
    return len(data) == SPECIAL_MAP_BYTES and sha256_bytes(data) == SPECIAL_MAP_SHA256


def check_special_tokens_json(facts):
    try:
        section = facts["hf"]["special_tokens_map"]
        parsed = json.loads(section["content"])
    except (KeyError, TypeError, ValueError):
        return False
    if not isinstance(parsed, dict):
        return False
    return bool(
        set(parsed) == {"additional_special_tokens", "eos_token", "pad_token"}
        and parsed["eos_token"] == section["eos_token"] == EOS_TOKEN
        and parsed["pad_token"] == section["pad_token"] == PAD_TOKEN
        and parsed["additional_special_tokens"] == ADDITIONAL_SPECIAL_TOKENS
        and all(token in ADDITIONAL_SPECIAL_TOKENS for token in ROLE_TOKENS)
    )


def check_tokenizer_config_facts(facts):
    try:
        section = facts["hf"]["tokenizer_config"]
    except (KeyError, TypeError):
        return False
    return bool(
        section["keys"] == TOKENIZER_CONFIG_KEYS
        and "chat_template" not in section["keys"]
        and section["inline_chat_template_key"] is False
        and section["tokenizer_class"] == "PreTrainedTokenizerFast"
        and section["model_max_length"] == 1048576
        and section["eos_token"] == EOS_TOKEN
        and section["pad_token"] == PAD_TOKEN
        and section["bytes"] == TOKENIZER_CONFIG_BYTES
        and section["sha256"] == TOKENIZER_CONFIG_SHA256
    )


def check_tokenizer_json_facts(facts):
    try:
        section = facts["hf"]["tokenizer_json"]
    except (KeyError, TypeError):
        return False
    added = section["added_token_ids"]
    ids = {int(token_id) for token_id in added}
    return bool(
        section["model_type"] == "BPE"
        and section["vocab_entries"] == VOCAB_ENTRIES
        and section["bytes"] == TOKENIZER_JSON_BYTES
        and section["sha256"] == TOKENIZER_JSON_SHA256
        and added == ADDED_TOKEN_IDS
        and max(ids) < UNPADDED_VOCAB_SIZE
        and len(added) == 6
    )


def check_frozen_template_digest_consistency(facts):
    try:
        hf = facts["hf"]["chat_template"]
        gguf = facts["gguf"]["kv"]["tokenizer.chat_template"]
        digest = sha256_bytes(hf["content"].encode("ascii"))
    except (KeyError, TypeError, AttributeError, UnicodeEncodeError):
        return False
    return bool(
        digest == CHAT_TEMPLATE_SHA256
        == hf["sha256"] == GGUF_CHAT_TEMPLATE_SHA256 == gguf["sha256"]
    )


def check_frozen_template_length_consistency(facts):
    try:
        hf = facts["hf"]["chat_template"]
        gguf = facts["gguf"]["kv"]["tokenizer.chat_template"]
    except (KeyError, TypeError):
        return False
    return bool(
        hf["bytes"] == gguf["bytes"] == CHAT_TEMPLATE_BYTES
        and gguf["utf8"] is True
        and len(hf["content"].encode("ascii")) == gguf["bytes"]
    )


def check_eos_pad_id_mapping(facts):
    try:
        added = facts["hf"]["tokenizer_json"]["added_token_ids"]
        special = facts["hf"]["special_tokens_map"]
        kv = facts["gguf"]["kv"]
        inverse = {token: int(token_id) for token_id, token in added.items()}
        eos = inverse[special["eos_token"]]
        pad = inverse[special["pad_token"]]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        eos != pad
        and eos == kv["tokenizer.ggml.eos_token_id"] == kv["tokenizer.ggml.eot_token_id"] == EOS_TOKEN_ID
        and pad == kv["tokenizer.ggml.padding_token_id"] == PAD_TOKEN_ID
    )


def check_gguf_tokenizer_kvs(facts):
    try:
        kv = facts["gguf"]["kv"]
    except (KeyError, TypeError):
        return False
    return bool(
        kv["tokenizer.ggml.model"] == "gpt2"
        and kv["tokenizer.ggml.pre"] == "inkling2"
        and kv["tokenizer.ggml.add_bos_token"] is False
        and kv["tokenizer.ggml.add_eos_token"] is False
        and kv["tokenizer.ggml.clean_spaces"] is False
    )


def check_gguf_special_id_kvs(facts):
    try:
        kv = facts["gguf"]["kv"]
    except (KeyError, TypeError):
        return False
    return bool(
        kv["tokenizer.ggml.eos_token_id"] == EOS_TOKEN_ID
        and kv["tokenizer.ggml.eot_token_id"] == EOS_TOKEN_ID
        and kv["tokenizer.ggml.padding_token_id"] == PAD_TOKEN_ID
        and is_int(kv["tokenizer.ggml.eos_token_id"])
        and is_int(kv["tokenizer.ggml.padding_token_id"])
    )


def check_gguf_vocab_consistency(facts):
    try:
        kv = facts["gguf"]["kv"]
        vocab = kv["inkling2_loop.vocab_size"]
        unpadded = kv["inkling2_loop.vocab_size_unpadded"]
        eos = kv["tokenizer.ggml.eos_token_id"]
        pad = kv["tokenizer.ggml.padding_token_id"]
    except (KeyError, TypeError):
        return False
    return bool(
        all(is_int(value) and value > 0 for value in (vocab, unpadded, eos, pad))
        and vocab == VOCAB_SIZE
        and unpadded == UNPADDED_VOCAB_SIZE
        and vocab > unpadded > eos > pad
        and vocab - unpadded == 966
        and unpadded > VOCAB_ENTRIES
    )


def check_gguf_header_provenance(facts):
    gguf = facts.get("gguf")
    if not isinstance(gguf, dict):
        return False
    return bool(
        gguf["first_16mib_sha256"] == GGUF_SHA256
        and gguf["magic"] == "GGUF"
        and gguf["version"] == 3
        and gguf["n_tensors"] == 744
        and gguf["n_kv"] == 63
        and gguf["byte_start"] == 0
        and gguf["byte_end"] == 16777215
        and gguf["fetched_bytes"] == 16777216
        and gguf["range"] == "bytes=0-16777215"
        and gguf["network_access_in_checker"] is False
        and gguf["checker_reads_cache"] is False
        and gguf["long_arrays_fully_validated"] is False
        and is_sha256(gguf["first_16mib_sha256"])
    )


def check_no_cache_read():
    """Static self-check: the frozen cache path is never looked up or opened."""
    source = Path(__file__).read_text(encoding="utf-8")
    lookup = "[" + chr(34) + "header_cache_file" + chr(34) + "]"
    if lookup in source or lookup.replace(chr(34), "'") in source:
        return False
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                return False
            if isinstance(node.func, ast.Attribute) and node.func.attr == "mmap":
                return False
    return True


def check_live_converter_sha256(root):
    path = root / CONVERTER_REL
    return path.is_file() and sha256_bytes(path.read_bytes()) == CONVERTER_SHA256


def check_converter_needles(root):
    text = read_text(root / CONVERTER_REL)
    return bool(text) and all(needle in text for needle in CONVERTER_NEEDLES)


def check_m37_gguf_sha256(root):
    path = root / M37_REL
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
    except (OSError, ValueError):
        return False
    return payload["provenance"]["first_16mib_sha256"] == GGUF_SHA256


def check_summary_expectations(facts):
    return facts.get("summary_expectations") == EXPECTED_SUMMARY_EXPECTATIONS


def check_no_network():
    """Static self-check: no network imports or URL-fetch calls in this checker."""
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in NETWORK_MODULES for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in NETWORK_MODULES:
                return False
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in NETWORK_CALLS:
                return False
    return True


def check_summary_shape(summary):
    """Check structure only; verdict and individual check outcomes are not shape."""
    if not isinstance(summary, dict) or set(summary) != EXPECTED_SUMMARY_KEYS:
        return False
    if summary["verdict"] not in ("PASS", "NO_GO"):
        return False
    checks = summary["checks"]
    if not isinstance(checks, dict) or set(checks) != set(CHECK_NAMES):
        return False
    if not all(isinstance(value, bool) for value in checks.values()):
        return False
    counts = summary["check_counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != {"total", "passed", "failed"}
        or not all(is_int(value) for value in counts.values())
    ):
        return False
    string_keys = ("blocker", "note", "scope", "doc_note", "cycle_audit_required", "frozen_on", "summary_path")
    for key in string_keys:
        if not isinstance(summary[key], str):
            return False
    for key in ("no_weights", "no_engine_run"):
        if not isinstance(summary[key], bool):
            return False
    if summary["speed_claim"] is not None:
        return False
    for key in ("checker_sha256", "contract_sha256"):
        if summary[key] is not None and not is_sha256(summary[key]):
            return False
    for key in ("hf_chat_template_sha256", "gguf_chat_template_sha256"):
        if not is_sha256(summary[key]):
            return False
    return "sentinel" not in summary


def run_check(root):
    root = Path(root)
    checks = {name: False for name in CHECK_NAMES}
    info = []
    checker_sha256 = None
    contract_sha256 = None
    summary_path = root / SUMMARY_REL
    sentinel_ok = True
    try:
        write_json(summary_path, fail_closed_summary())
    except Exception as exc:
        sentinel_ok = False
        info.append("sentinel_write_failed: %s: %s" % (type(exc).__name__, exc))
    try:
        contract_path = root / CONTRACT_REL
        contract_sha256 = sha256_bytes(contract_path.read_bytes())
        facts = load_json(contract_path)
        checker_sha256 = sha256_bytes(Path(__file__).read_bytes())
        checks["frozen_contract"] = check_frozen_contract(facts)
        checks["frozen_checker_sha256"] = (
            is_sha256(facts.get("checker_sha256"))
            and facts["checker_sha256"] == checker_sha256
        )
        checks["contract_summary_path"] = facts.get("summary_path") == SUMMARY_REL
        checks["embedded_chat_template_bytes"] = check_embedded_chat_template(facts)
        checks["embedded_chat_template_ascii"] = check_chat_template_ascii(facts)
        checks["embedded_chat_template_needles"] = check_chat_template_needles(facts)
        checks["embedded_special_tokens_bytes"] = check_embedded_special_tokens(facts)
        checks["embedded_special_tokens_json"] = check_special_tokens_json(facts)
        checks["tokenizer_config_facts"] = check_tokenizer_config_facts(facts)
        checks["tokenizer_json_facts"] = check_tokenizer_json_facts(facts)
        checks["frozen_template_digest_consistency"] = check_frozen_template_digest_consistency(facts)
        checks["frozen_template_length_consistency"] = check_frozen_template_length_consistency(facts)
        checks["eos_pad_id_mapping"] = check_eos_pad_id_mapping(facts)
        checks["gguf_tokenizer_kvs"] = check_gguf_tokenizer_kvs(facts)
        checks["gguf_special_id_kvs"] = check_gguf_special_id_kvs(facts)
        checks["gguf_vocab_consistency"] = check_gguf_vocab_consistency(facts)
        checks["gguf_header_provenance"] = check_gguf_header_provenance(facts)
        checks["gguf_no_cache_read"] = check_no_cache_read()
        checks["converter_live_sha256"] = check_live_converter_sha256(root)
        checks["converter_needles"] = check_converter_needles(root)
        checks["m37_gguf_sha256"] = check_m37_gguf_sha256(root)
        checks["summary_expectations"] = check_summary_expectations(facts)
        checks["no_network"] = check_no_network()
        info.append(
            "evidence: HF %s at %s, chat_template.jinja %d bytes sha256 %s; "
            "checker makes no network calls"
            % (EXPECTED_HF["repo"], EXPECTED_HF["commit"], CHAT_TEMPLATE_BYTES, CHAT_TEMPLATE_SHA256)
        )
        if checks["frozen_template_digest_consistency"] and checks["frozen_template_length_consistency"]:
            info.append(
                "template consistency: out-of-band bounded probe measured byte equality; "
                "frozen digests and lengths internally consistent (%d bytes sha256 %s)"
                % (CHAT_TEMPLATE_BYTES, CHAT_TEMPLATE_SHA256)
            )
        else:
            info.append("template parity check FAILED")
        info.append(
            "GGUF first 16 MiB sha256 %s, %d tensors, %d KVs; cached header is never opened"
            % (GGUF_SHA256, 744, 63)
        )
        info.append(
            "special tokens: eos %s id %d, pad %s id %d, eos==eot, add_bos/add_eos/clean_spaces false"
            % (EOS_TOKEN, EOS_TOKEN_ID, PAD_TOKEN, PAD_TOKEN_ID)
        )
        info.append(
            "vocab: padded %d > unpadded %d > eos %d > pad %d; long arrays are not fully validated"
            % (VOCAB_SIZE, UNPADDED_VOCAB_SIZE, EOS_TOKEN_ID, PAD_TOKEN_ID)
        )
        info.append("contract sha256 %s; checker sha256 %s" % (contract_sha256, checker_sha256))
        info.append(DOC_NOTE)
        info.append(CYCLE_AUDIT_REQUIRED)
        checks["run_completed"] = True
    except Exception as exc:
        info.append("exception: %s: %s" % (type(exc).__name__, exc))

    checks["sentinel_write"] = sentinel_ok
    checks["summary_shape"] = True

    def verdict_counts():
        failed = [name for name in sorted(checks) if not checks[name]]
        passed = len(checks) - len(failed)
        verdict = "PASS" if not failed else "NO_GO"
        return verdict, ("" if not failed else ", ".join(failed)), {
            "total": len(checks), "passed": passed, "failed": len(failed),
        }

    verdict, blocker, counts = verdict_counts()
    summary = {
        "verdict": verdict,
        "blocker": blocker,
        "checks": dict(checks),
        "check_counts": counts,
        "info": info,
        "note": NOTE,
        "scope": NOTE,
        "speed_claim": None,
        "cycle_audit_required": CYCLE_AUDIT_REQUIRED,
        "no_weights": True,
        "no_engine_run": True,
        "frozen_on": FROZEN_ON,
        "summary_path": SUMMARY_REL,
        "checker_sha256": checker_sha256,
        "contract_sha256": contract_sha256,
        "hf_chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "gguf_chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "doc_note": DOC_NOTE,
    }
    checks["summary_shape"] = check_summary_shape(summary)
    verdict, blocker, counts = verdict_counts()
    summary["verdict"] = verdict
    summary["blocker"] = blocker
    summary["checks"] = dict(checks)
    summary["check_counts"] = counts
    write_json(summary_path, summary)
    return summary


def print_summary(summary):
    print("verdict: " + summary["verdict"])
    if summary["blocker"]:
        print("blocker: " + summary["blocker"])
    for name in sorted(summary["checks"]):
        print("  %-34s %s" % (name, "PASS" if summary["checks"][name] else "FAIL"))
    counts = summary["check_counts"]
    print("  counts: %d/%d passed, %d failed" % (counts["passed"], counts["total"], counts["failed"]))
    for line in summary["info"]:
        print("  INFO: " + line)
    print("note: " + summary["note"])
    print("doc: " + summary["doc_note"])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to check (default: this repo)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        summary = run_check(args.root)
    except Exception as exc:
        print("error: could not write summary: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print_summary(summary)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
