# M24 - Kaggle 2xT4 CycleAudit

Status: **GATE RAN WITH INTEGRITY; VERDICT NO_CLAIM.** No speed number is
published. The measured 95% bootstrap interval lies below 1.0 on the synthetic
16-token fixture, so the pre-registered claim 1.0 fails closed.

## Pre-registered configuration (frozen before the run)

- Requested kernel id `jprinceraj/inkengine-cuda-cycle-audit`; the Kaggle CLI
  resolved the title to slug `jprinceraj/inkengine-cuda-cycleaudit`, which is
  the URL to use for status/logs (private, T4x2, internet).
- One kernel only; logs via `kaggle kernels logs`, not the output API.
- Claim 1.0: valid only if claim_valid true AND claim <= ci_low.
- Repeats 50 (>=20), warmup 5, loops 3, seed 0, 2000 bootstrap resamples.
- Synthetic 3-pass MoE GGUF from `tools/audit/make_synthetic_gguf.py --seed 0`.
- RORET OFF vs ON, cpu_share 0.0, ngl 99, moe-cache-mib 16, real loopkit gate.
- Negative controls must fail closed: self 1.01, slower 2.0.
- Verdict: CLAIM_SUPPORTED only if claim_valid and claim <= ci_low, else NO_CLAIM.

## Tooling landed before the kernel (commit 9c975ae)

- `tools/audit/run_gpu_cycle_audit.py`: stdlib-only producer. Times 5 warmups
  and 50 recorded `llama-cli` runs per phase with `perf_counter`, enforces
  hook ABSENT off / ACTIVE on, checks generated-text byte parity using
  separate `-o` files per phase, writes `baseline.json`/`candidate.json`,
  drives the real `python -m loopkit.cli audit` gate, and requires two
  fail-closed negatives (self 1.01, simulated 2x-slower candidate 2.0).
  Verdict is `CLAIM_SUPPORTED` only when `claim_valid` and `claim <= ci_low`,
  otherwise the honest `NO_CLAIM` (still exit 0).
- `tools/roret/kaggle-cycle-audit/`: notebook plus kernel metadata, modeled on
  the proven M18/M23 T4 build pattern (`-DGGML_CUDA_NO_VMM=ON`, sm_75, `-j4`,
  no self-inflicted build timeout). The notebook clones fork `main` and
  loopkit, builds, generates the fixture, runs the producer, and prints a
  summary JSON into notebook logs for `kaggle kernels logs`.
- Local validation before push: script `py_compile` OK; notebook and metadata
  JSON load OK; all new files ASCII-only; stub-CLI end-to-end smoke test
  rendered the real gate with both negatives fail-closed; fresh ctest roret
  10/10, pibs 2/2, odsl 1/1; audit pytest 17 passed; loopkit fast pytest 40
  passed, 1 deselected; no `src/` edits.

## Scope

Counts (RORET_PASS rows, hits, uploads) are counts, not speedups. T4 sm_75 has
no FP4/FP8 tensor cores. Only ci_low may support a speed number. PASS means the
gate ran with integrity, not that a speed claim exists.

## Kaggle 2xT4 kernel result (logs, not the output API)

One kernel: `jprinceraj/inkengine-cuda-cycleaudit` v1, private, `COMPLETE`.
Kernel log sha256
`4d8d11a7146e2538018febe36e56416c9e9f607ed0e7ae0e5271664f02d5550d`;
decoded stdout sha256
`32edd02b472f992345909f4c31d23b73a00755b233676201ea6cbae0baa0bf40`.
Evidence is also saved in
`tools/roret/kaggle-cycle-audit/M24_SUMMARY.json`.

- Fork HEAD `9c975ae68f11e848ce7ecc0292e57e3ffe156ff6`; loopkit HEAD
  `0597f5ea0ffe310bc1f96aea75352cb3c91a75ee` (default branch).
- nvcc 13.0 V13.0.88; CUDA configure exit 0; `llama-cli+llama-debug` build
  exit 0, `build_ok=true`; `llama-cli` version 1.0.0-dev (build 12755,
  commit 9c975ae68).
- Fixture sha256
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`.
- Producer exit 0; `repeats=50`, `warmup=5`, `loops=3`, seed 0, 2000 bootstrap
  resamples; hook ABSENT off / ACTIVE on; generated text bytes equal.
- Negative controls both failed closed (self claim 1.01 and simulated
  2x-slower candidate claim 2.0 both exit 1 with `claim_valid=false`).
- Gate: `claim=1.0`, `speedup=0.9999176055055583`, `ci_low=0.9977116162363447`,
  `ci_high=0.9999835093546772`, `claim_valid=false`, gate exit 1.
- Notebook `overall_verdict=PASS` means gate integrity only, not a speed claim.

## Interpretation and next step

The entire 95% interval is below 1.0, so on this synthetic 16-token fixture
RORET ON was marginally **slower** than OFF; no speedup is supported. The
24-to-union load-count reduction from the 100k-token probe and the 6 per-pass
`RORET_PASS` traffic rows from M23 are counts, not wall-clock savings, and this
run does not convert them into a speed number. The result is scoped to the
fixed-seed random-weight fixture, not a real model.

Next bounded work: the remaining correctness gap is intermediate per-pass MoE
tensor parity and real-weight parity (both unmeasured); any future speed number
still requires a fresh CycleAudit with a claim at or below its lower 95% bound.
