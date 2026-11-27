# M17 - Engine-level logits parity attempt on Kaggle 2xT4 (2026-11-27)

Status: **NO_GO on Kaggle**. One kernel was spent; the CUDA build was cut off
by a 300 s timeout before any engine run. The timeout was introduced by this
milestone's own review fix and applied to the build step as well as the binary
runs. No engine-level GPU parity number was measured.

Fork: `10xdev4u-alt/inkengine-llama`, branch `main`, HEAD
`6fb578632aa6b5ab6b9b6f136c55c6a1f78e10bf`.

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

## Verdict and next step

* Kaggle engine-level GPU logits parity: **NO_GO (build timeout, no runs)**.
* Engine-level GPU parity issue #4 and T4 traffic issue #5 remain NO_GO.
* Local CPU-test logits parity stays PASS-SCOPED.
* Next run: push the corrected checked-in notebook with a fresh kernel id/title
  and require `build_ok`, both exits 0, hook ABSENT off / ACTIVE on, non-empty
  logits bins, and 0 differing bytes before any parity claim. Keep the
  per-run 300 s timeout only on the binary runs.
* NO SPEED CLAIM. Any speed number needs CycleAudit `claim <= ci_low`,
  `repeats >= 20`, `loops = 3`.
