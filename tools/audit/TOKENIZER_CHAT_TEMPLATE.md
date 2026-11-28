# M47 - offline tokenizer and chat-template contract gate

## Scope

This gate freezes the tokenizer and chat-template contract for Inkling-2-Mini
and closes section-3 item (e) of `loopkit/docs/INKLING_INTEGRATION.md`,
"that tokenizer / chat template (`tml-renderers`) matches", as a contract
result only. The loopkit document is not edited by this milestone: M30, M33,
M35 and M38 pin its SHA-256, so a coordinated marker update belongs to a
separate milestone.

The checker is stdlib-only, offline and fail-closed. It reads only the frozen
contract, the live `conversion/inkling2.py` source, the M37 contract and its
own source. It makes no network calls and never opens the out-of-band GGUF
header cache.

## Commands

Run the gate from the repo root:

    python3 tools/audit/check_tokenizer_chat_template.py

Run the tests:

    PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit/test_tokenizer_chat_template.py -q
    PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit -q

An isolated copy can be checked with `--root <dir>`. The checker writes a
`NO_GO` sentinel to `M47_TOKENIZER_CHAT_TEMPLATE_SUMMARY.json` before it
starts and atomically replaces it with the final verdict. Exit code 0 means
PASS; any failed check or exception leaves NO_GO and exit code 1. A seeded
stale PASS is always overwritten.

## What the gate proves

26 checks must all be true for PASS:

- the frozen contract still matches every checker constant, including exact
  top-level key sets, and the checker's own SHA-256 matches the contract pin;
- the embedded HF `chat_template.jinja` is 6868 ASCII bytes with SHA-256
  `53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc`, and
  the embedded `special_tokens_map.json` is 529 ASCII bytes with SHA-256
  `c16f44b3594563d82743ddb0417a2be1bfe3e2000dfad2da4b8f32554593e1a0`;
- the frozen HF and GGUF `tokenizer.chat_template` digests and lengths are
  internally consistent and source-pinned (`frozen_template_digest_consistency`
  and `frozen_template_length_consistency`); the out-of-band bounded probe
  measured byte equality, and the offline checker cannot observe live GGUF
  drift;
- the HF tokenizer config has no inline `chat_template` key: the template is
  the separate `chat_template.jinja` file. Its tokenizer class is
  `PreTrainedTokenizerFast`, `model_max_length` is 1048576, and eos/pad are
  `<|content_model_end_sampling|>` and `<|endoftext|>`;
- the four role tokens `<|message_user|>`, `<|message_model|>`,
  `<|message_system|>` and `<|message_tool|>` and the
  `emit_thinking_effort` macro appear in the frozen template, whose first
  line defines the `effort_map`;
- the HF added-token ids map eos to 200006 and pad to 199999, matching the
  GGUF `eos_token_id`, `eot_token_id` and `padding_token_id`;
- the GGUF tokenizer KVs are `model=gpt2`, `pre=inkling2`, with
  `add_bos_token`, `add_eos_token` and `clean_spaces` all false; padded vocab
  201024 exceeds unpadded vocab 200058, which exceeds eos 200006 and pad
  199999;
- the live converter still has its frozen SHA-256 and all 13 vocabulary and
  special-token needles; the M37 contract still carries the same GGUF hash;
- the checker source parses as Python with no network import or fetch call
  and no runtime lookup of the cached header path.

A failure to write the startup sentinel also forces NO_GO even if every other
check passes. The test suite pins the checker source SHA-256, so editing the
gate itself is a detectable change.

## What was measured out of band

The HF files came from `thinkingmachines/Inkling-2-Mini` at commit
`c9d028302cd0f035fdf8e7e91f089a4ce94d2195` in one bounded fetch:

- `chat_template.jinja`: 6868 bytes, SHA-256
  `53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc`;
- `tokenizer_config.json`: 13235 bytes, SHA-256
  `bf09342d285139fd37ce10c70878af1d52cb390e9901e10248654293a80ee636`;
- `special_tokens_map.json`: 529 bytes, SHA-256
  `c16f44b3594563d82743ddb0417a2be1bfe3e2000dfad2da4b8f32554593e1a0`;
- `tokenizer.json`: 26178678 bytes, SHA-256
  `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6`,
  BPE model, 199998 base vocab entries, added ids 199999, 200000, 200001,
  200002, 200003 and 200006 for the special tokens.

The GGUF evidence came from one bounded probe of
`unsloth/Inkling-2-Mini-GGUF` `Inkling-2-Mini-UD-IQ1_S.gguf`, byte range
`bytes=0-16777215`, SHA-256
`87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76`.
That header is GGUF v3 with 744 tensors and 63 KVs. Its
`tokenizer.chat_template` is a UTF-8 string of 6868 bytes with the same
SHA-256 as the HF template; the out-of-band bounded probe compared the two
byte strings and found them equal. The offline checker verifies only that the
frozen digests and lengths are internally consistent and source-pinned; it
cannot observe live GGUF drift. The cache path is recorded as provenance only
and is never read by the checker. The long token, token-type and merge arrays
are not claimed to be fully validated, and no full-vocabulary assertion is
made.

## What the gate does not prove

- No weights are loaded or read, and no engine is run.
- No tokenization golden test runs against real weights. The out-of-band probe
  measured template byte equality and the offline gate checks frozen digests,
  lengths and bounded tokenizer metadata; neither proves that a particular
  runtime renders identical token ids.
- Section-3 item (b), real-weight conversion parity, stays NO_GO_BLOCKED.
- No timing, throughput, latency or speed measurement is made. NO SPEED
  CLAIM. Any future number must satisfy CycleAudit
  `claim<=ci_low repeats>=20 loops=3`.
- The loopkit marker is not changed here; the coordinated two-repo doc update
  is a separate milestone.
