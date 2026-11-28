# M26 - Intermediate per-pass MoE tensor parity

This page records the method, the gate and the measured CPU-test result for
intermediate MoE tensor parity with RORET off vs on.

## Method

Run `llama-debug` twice per `cpu_share` on the synthetic fixture. The OFF run
has no `LLAMA_RORET`; the ON run sets `LLAMA_RORET=1`. Both runs set
`LLAMA_DUMP_TENSORS_DIR` and use `--tensor-filter` to dump selected MoE
tensors. `common/debug.cpp` writes raw tensor bytes plus `manifest.tsv` for
each run. `tools/roret/check_intermediate_parity.py` compares manifest rows
by tensor name and occurrence, then compares the raw bytes. The tool writes
`M26_SUMMARY.json` and fails closed on any mismatch.

## Tensor names and why they matter

- `ffn_moe_logits-<il>`: router logits, F32 in this fixture.
- `ffn_moe_probs-<il>`: router probabilities after softmax.
- `ffn_moe_topk-<il>`: selected expert ids, I32.
- `ffn_moe_weights-<il>`: selected expert weights, F32.
- `ffn_moe_out-<il>`: combined MoE layer output, F32.

Here `<il>` is the graph layer index. In the synthetic 3-pass fixture,
`il=1,2,3` are the three loop passes. These tensors are the intermediate
values between expert dispatch and the final output, so byte parity here
checks more than final text parity.

## Scope limits

- CPU-test mode only (`GGML_CUDA=OFF`, `LLAMA_RORET_CPU_TEST=ON`).
- Synthetic fixed-seed random-weight fixture only; no real model weights.
- No GPU, no CUDA path, no real-weight parity claim.
- No wall-clock or speed claim of any kind.

## Gate rules

The verdict is PASS only when all of the following hold for every cpu_share:

- both subprocess exits are 0;
- the RORET hook is ABSENT in OFF stderr and ACTIVE in ON stderr;
- each phase's dump directory is freshly cleared;
- every expected tensor base name appears in both manifests with equal
  occurrence counts, and no unexpected name appears;
- manifest row sets match, with identical type, ne, nb and nbytes;
- every referenced `.bin` file exists in both runs and its size matches the
  manifest `nbytes`;
- differing bytes equal 0, with at least one tensor and one byte compared.

Any violation gives NO_GO and a nonzero tool exit.

## Measured result (2026-11-28, CPU-test mode)

Built `llama-debug` from branch `main` HEAD `6082fce` with the M26 working-tree
change applied, using `GGML_CUDA=OFF`, `LLAMA_RORET_CPU_TEST=ON` in the existing
build dir `/tmp/opencode/m23-llama-build`. Build exit 0 with no warnings after
removing a redundant null check on the `ggml_tensor::name` array.

| cpu_share | exits (off/on) | hook (off/on) | tensors | bytes compared | differing bytes | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0 / 0 | ABSENT / ACTIVE | 40 | 30080 | 0 | PASS |
| 1.0 | 0 / 0 | ABSENT / ACTIVE | 40 | 30080 | 0 | PASS |

Overall verdict: PASS (CPU-test scope, synthetic fixture only). Fixture sha256
`cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`.
The final hardened run reports `llama-debug --version` as
`1.0.0-dev (build 148, commit 94a4712)`; the binary was built from the working
tree, so the commit string names the pushed HEAD at build time, not the SHA of
the later hardening commit. Evidence:
`/tmp/opencode/m26-parity-h/M26_SUMMARY.json` (sha256
`8ef1e0c9c12d1e24720286ed9b34e47ec4bd5dedbccfe25ccc6b9d68724b8a4a`).
Each share has 20 unique tensor names (5 kinds x `il=0..3`) across two graph
evaluations, hence 40 dumped `.bin` files per phase.

Gate hardening after independent review: the tool now requires the exact 20
expected tensor names in both manifests (`--expect-names`, default listed in
`check_intermediate_parity.py`), requires equal occurrence counts per name,
refuses to proceed when a stale dump directory cannot be removed, and checks
each `.bin` size against the manifest `nbytes` before counting compared bytes.
The C++ dump logs and skips the manifest row when a file write fails instead of
silently recording an incomplete file. The reviewer's claimed blocking issue
("--cmoe is invalid") was a misread: the checked-in tool passes `-cmoe`, which
is a registered alias, and both real runs exit 0.

Fail-closed negative controls with the checked-in tool: `--llama-debug /bin/true`
(exits 0, dumps nothing) returned exit 1 with missing-manifest and hook-ABSENT
failures; `/bin/false` (nonzero exits) returned exit 1. Neither could be read
as PASS. A second full run to a fresh output directory produced a byte-identical
`M26_SUMMARY.json` (same sha256 above), so the evidence is repeatable.

Local suites this pass: roret ctest 10/10, pibs 2/2, odsl 1/1, `tools/audit`
pytest 17 passed, loopkit fast pytest 41 passed. No `src/` edit, no Kaggle
kernel spent, no speed claim. GPU intermediate parity and real-weight parity
remain unmeasured; see M24 for the mandatory CycleAudit gate before any number.
