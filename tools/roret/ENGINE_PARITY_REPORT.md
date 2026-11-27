# Engine-level logits parity on Kaggle 2xT4

Current status: **PASS (scoped)** as of M18 on 2026-11-28. Raw F32 final logits
from `llama-debug` have 0 differing bytes with RORET off vs on for both
`--moe-cache-cpu-share 0.0` and `1.0` on the deterministic synthetic GGUF.
This supersedes the M17 NO_GO recorded below, which was caused by a self-inflicted
build timeout, not by a parity failure. Scope: raw F32 FINAL logits only, not
intermediate per-pass MoE tensors; synthetic fixture only; no real weights;
NO SPEED CLAIM.

## M18 - corrected notebook retry (2026-11-28)

The corrected checked-in notebook was pushed under a fresh kernel id so the
M17 timeout bug could not be rerun:

* Kernel `jprinceraj/inkengine-cuda-engine-parity` v1, private,
  `machine_shape = NvidiaTeslaT4x2`. The title is `Inkengine CUDA engine parity`
  so it resolves to the id (the old title resolved to the M17 slug and Kaggle
  returned 409). Pushing auto-started the kernel; status reached `COMPLETE`.
* Notebook `tools/roret/kaggle-engine-parity/engine_parity.ipynb`,
  sha256 `157d8b444ced8affa00833b2c674594a1d36dd93fd0422eeac6ecdb96f753617`.
  Logs read with `kaggle kernels logs`, not the output API: raw log
  sha256 `29fdc682e37828b03c85e085db6b9daf80ea60e3e463ca8a26b8077e03ec2901`,
  reconstructed stdout sha256
  `b1bbe8175b005bdca4ae6af1f8ccee91f4d4ef76253f4d95efe0a00fcf15ea99`.
* Clone `-b main` HEAD `a6424630e7d78b4eecf817282eba369a2f6e4dcb`;
  nvcc 13.0.88; configure with `-DGGML_CUDA_NO_VMM=ON` exit 0;
  `cmake --build build --target llama-cli llama-debug -j4` exit 0,
  `build_ok = true`. Configure, build and GGUF generation were uncapped;
  only the eight binary runs kept `timeout = 300`.
* Fixture sha256
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`,
  matches M15/M16/M17.
* All 8 binary runs (2 shares x off/on x cli/debug) exited 0. Hook ABSENT off
  and ACTIVE on in both binaries. Zero `GGML_ASSERT`/`Assertion` lines.
* Each share produced non-empty `llamacpp-synthetic.bin` (logits) and
  `llamacpp-synthetic-tokens.bin` (tokens) for off and on. Off/on differing
  bytes: logits 0, tokens 0, generated text 0 (text sha256
  `cf915b84cf60abfd20ced6ac8ec7e69a72c2f0c4b0e2f6abbe8c9c9130bbec3d`
  for all four runs).
* `RORET_PASS` rows from `llama-cli`: 2 total, one per share on the `on` run,
  `il=1 pass=2 plans=19 hits=150 upload_ops=78 uploaded_bytes=153600
  residents=812`. These are count evidence that the copy path ran, not timing.
* Summary: `logits_parity_verdict = LOGITS_PARITY: 0 differing bytes
  (cpu_share=0.0 and 1.0)`, `text_parity_verdict` likewise,
  `engine_verdict = ENGINE_COPY_PATH_EXERCISED: YES`, `blocker = ""`,
  `overall_verdict = PASS`. Parsed summary saved at
  `tools/roret/kaggle-engine-parity/M18_SUMMARY.json`, sha256
  `7eb2230cb0727cc5240b1b973c4f022ab4009a7c5b0adfb61fd613b5a151d2a6`.
* Local tests this pass: roret ctest 7/7, pibs 2/2, odsl 1/1;
  `tools/audit` pytest 17 passed; loopkit fast pytest 40 passed, 1 deselected.
  No `src/` edit.

NO SPEED CLAIM. Any speed number still needs CycleAudit `claim <= ci_low`
with `repeats >= 20` and `loops = 3`.

## M17 attempt (history - NO_GO, build timeout)

## What was attempted

A new 9-cell notebook, `tools/roret/kaggle-engine-parity/engine_parity.ipynb`,
to compare raw F32 final logits from `llama-debug --save-logits` with RORET
off vs on, on the deterministic synthetic GGUF, for
`--moe-cache-cpu-share 0.0` and `1.0`. `llama-cli` runs provide generated-text
cross-checks and `RORET_PASS` count rows. The hook must be ABSENT off and
ACTIVE on for both binaries; any crash, assert, missing/empty bin file, or
byte difference fails closed.

## Kaggle result (measured)

* Kernel `jprinceraj/inkengine-cuda-roret-engine-logits-parity` v1,
  `NvidiaTeslaT4x2`, status COMPLETE. Logs were read with `kaggle kernels
  logs`, not the output API.
* Notebook actually run: sha256
  `abaf970dbbbbaf77cd1cfc471413ee04ccc5c094a0befa3aaa7103148b09ba9d`.
  Kernel log sha256 `6a7b77b41aac5fdc6b6549edc1b2e6efd6461849432f954b92d64e5616dd15f3`;
  stdout extract sha256
  `33198a5b70b91638b67d2f2e96c9822e2abd48cfa2e5947211952986662ded55`.
* Environment: python 3.13.16, nvcc 13.0.88, 2x Tesla T4, torch
  2.11.0+cu130. `git clone -b main` exit 0 at the HEAD above.
* CUDA configure with `-DGGML_CUDA_NO_VMM=ON`: **exit 0** (6.3 s), including
  "Configuring done" and "Generating done".
* `cmake --build build --target llama-cli llama-debug -j4`: **exit 124**
  (TIMEOUT after 300 s) at about 28-30 percent of the CUDA objects. The log
  prints `TIMEOUT: command exceeded 300 seconds`, then `build_ok False` and
  `BLOCKER: CUDA build failed; later stages skipped`.
* No GGUF generated, no `llama-cli` or `llama-debug` runs, no logits files,
  no `RORET_PASS` rows, no off/on comparison.
* Summary: `logits_parity_verdict=NOT_RUN`, `text_parity_verdict=NOT_RUN`,
  `engine_verdict=NO_GO: pipeline blocked before runs`,
  `overall_verdict=NO_GO`, `speed_claim=NO SPEED CLAIM`.

## Root cause and committed fix

The review pass correctly asked for a timeout on the eight binary runs so a
hang would still produce a summary. The fix was applied inside the shared
`run_logged()` helper, which also wraps configure, build and GGUF generation.
The build therefore inherited a 300 s cap and was killed before completion.
The checked-in notebook now takes `timeout=None` by default in `run_logged()`
and passes `timeout=300` only at the two binary-run call sites. Configure,
build and generation are uncapped again. This is a tooling fix, not a source
change; it has not been run on Kaggle yet (one-kernel budget spent).

## Local supporting evidence (CPU test mode, PASS-SCOPED)

Build `/tmp/opencode/m15-llama-cpu`, `GGML_CUDA=OFF`,
`LLAMA_RORET_CPU_TEST=ON`, targets `llama-cli` and `llama-debug`:
`1.0.0-dev (build 131, commit 6fb5786)`.

Synthetic GGUF sha256
`cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`.
`llama-debug --verbose --save-logits` off/on, `-cmoe`,
`--moe-cache-mib 16`, for both cpu_share values:

| cpu_share | hook off | hook on | off/on logits sha256 | differing bytes |
|---|---|---|---|---|
| 0.0 | ABSENT | ACTIVE | `8e1385e1e1da124308d84bbc6f2f5e2d16f38496c580fadd203561da0a67131c` (both) | 0 |
| 1.0 | ABSENT | ACTIVE | `0eb9c00a2c6a841abc2508def5b9e34d241059be11a22816f4a33bb9764c888e` (both) | 0 |

Token-id `.bin` files were byte-identical in both shares. This is test-mode
CPU evidence on the synthetic fixture, not GPU evidence.

## Local test suite (this pass)

* roret ctest 7/7, pibs ctest 2/2, odsl ctest 1/1.
* `tools/audit` pytest 17 passed; loopkit fast pytest 40 passed, 1 deselected.
* `UNION_WIRING.patch` sha256 `a9186808...ca1e84` and `PIBS_WIRING.patch`
  sha256 `f69543d2...371804` unchanged. No `src/` edit, no issue or PR filed.

### Verdict and next step (M17 history, superseded by M18 above)

* Kaggle engine-level GPU logits parity: **NO_GO (build timeout, no runs)**.
* Engine-level GPU parity issue #4 and T4 traffic issue #5 remain NO_GO.
* Local CPU-test logits parity stays PASS-SCOPED.
* Next run: push the corrected checked-in notebook with a fresh kernel id/title
  and require `build_ok`, both exits 0, hook ABSENT off / ACTIVE on, non-empty
  logits bins, and 0 differing bytes before any parity claim. Keep the
  per-run 300 s timeout only on the binary runs.
* NO SPEED CLAIM. Any speed number needs CycleAudit `claim <= ci_low`,
  `repeats >= 20`, `loops = 3`.
