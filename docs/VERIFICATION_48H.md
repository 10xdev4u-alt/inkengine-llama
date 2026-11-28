# Verification 48h - M10 consolidation

Date: 2026-11-25
Fork: `10xdev4u-alt/inkengine-llama`, branch `main`, verified at HEAD
`57da28a067664b50d0b2648ae4e387576846ec35`.
Status: HONEST / DOC-ONLY MILESTONE. This pass adds only this document; no source
file, issue or PR was created or modified. No speed claim is made. No engine-level
GPU parity is claimed. No Kaggle kernel was run in this pass.

Scope note: the roadmap's "Fresh Kaggle CPU rebuild" requirement is only PARTIALLY
MET. A fresh CPU rebuild was done locally (details in section 4); the prior Kaggle
CPU parity kernel evidence is recorded, and no new Kaggle run was attempted.

## M25 addendum (2026-11-28) - supersedes stale M10-era status below

This pass is docs-only. No Kaggle kernel was run in this pass, no source file was
edited, and no issue or PR was created. This addendum propagates the
already-landed M24 Kaggle 2xT4 CycleAudit result into this honesty/status
document. The M10 text below is kept intact as historical record; where it
conflicts with this addendum, this addendum is current.

### M24 Kaggle 2xT4 CycleAudit (already landed, not re-run here)

- Kernel `jprinceraj/inkengine-cuda-cycleaudit` v1, T4x2, status COMPLETE.
  Started 2026-11-27T23:36:27Z, finished ~2026-11-28T00:18:19Z. Logs were
  read with `kaggle kernels logs`, not the output API. Kernel log sha256
  `4d8d11a7146e2538018febe36e56416c9e9f607ed0e7ae0e5271664f02d5550d`;
  decoded stdout sha256
  `32edd02b472f992345909f4c31d23b73a00755b233676201ea6cbae0baa0bf40`.
- Fork HEAD `9c975ae68f11e848ce7ecc0292e57e3ffe156ff6`; loopkit HEAD
  `0597f5ea0ffe310bc1f96aea75352cb3c91a75ee`; nvcc 13.0 V13.0.88; configure
  exit 0; build exit 0; `llama-cli` 1.0.0-dev build 12755, commit `9c975ae68`.
- Fixture sha256
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`;
  `repeats=50`, `warmup=5`, `loops=3`, seed 0, 2000 bootstrap resamples. Hook
  ABSENT off / ACTIVE on; generated text bytes equal. Negative controls (self
  claim 1.01 and simulated 2x-slower claim 2.0) both exited 1 with
  `claim_valid=false`.
- Gate: `claim=1.0`, `ci_low=0.9977116162363447`,
  `ci_high=0.9999835093546772`, point-estimate ratio `0.9999176055055583`,
  `claim_valid=false`, gate exit 1, verdict `NO_CLAIM`. Notebook
  `overall_verdict=PASS` means gate integrity only, NOT a speed claim.
- Evidence: [M24_CYCLE_AUDIT.md](../tools/roret/M24_CYCLE_AUDIT.md) and
  [M24_SUMMARY.json](../tools/roret/kaggle-cycle-audit/M24_SUMMARY.json).

### Interpretation - NO SPEED CLAIM

The whole 95% interval is below 1.0 on the synthetic 16-token fixture: no
speedup is supported and the point estimate `0.9999176055055583` must NOT be
published as a speed number. This result is scoped to the fixed-seed
random-weight fixture, not a real model. M18 recorded engine-level raw F32 FINAL
logits parity PASS-SCOPED on 2xT4 for the synthetic 3-pass GGUF at both
`cpu_share 0.0` and `cpu_share 1.0`. M23 recorded 6 `RORET_PASS` rows (3 passes
per share); these are COUNTS, not wall-clock savings.

### Superseded M10-era lines (still present below as historical record)

- Section 1, M9 row: still describes only the ineligible local CPU probe
  (`ci_low=0.9714442869532246`). It is not the M24 gate.
- Section 5, claim-integrity row (row 2): now points here; an eligible gate has
  run with integrity and returned `NO_CLAIM`.
- Section 6: "No Kaggle 2xT4 CycleAudit was run this pass" is true only of the
  M10 pass. The M24 kernel did run and is reported above.
- Section 7: blockers #2, #4 and #5 are CLOSED as of 2026-11-28; the "no
  eligible claim exists" sentence is superseded by the M24 gate above.

### Issue and local-test status as of 2026-11-28

- All GitHub issues #1 through #8 in `10xdev4u-alt/inkengine-llama` are CLOSED
  (verified today). In particular #2, #4 and #5 are closed.
- Local tests on 2026-11-28: ctest roret 10/10, pibs 2/2, odsl 1/1; audit
  pytest 17 passed; loopkit fast pytest 40 passed, 1 deselected.

## 1. Artifact ledger (M1-M9)

| Milestone | Artifact | Status | One-line evidence |
|---|---|---|---|
| M1 | [AUDIT_PROTOCOL.md](../tools/audit/AUDIT_PROTOCOL.md) | PASS | Gate recipe frozen: `claim <= ci_low`, `repeats >= 20`, `loops >= 3`; negative control claim 1.01 returned `claim_valid=false`, exit 1; audit pytest 13 passed today. |
| M2 | [make_moe_fixture.py](../tools/audit/make_moe_fixture.py) | PASS-SCOPED | Seed-0 fixture/dump hashes reproduced (`fixture_sha256 fff0e60f...`); deterministic Python model, not engine output; engine-loadable synthetic GGUF came later from `make_synthetic_gguf.py`. |
| M3 | [UNION_WIRING.patch](../tools/roret/UNION_WIRING.patch) | PASS-SCOPED | `git apply --check` + `git apply` PASS on base `5035bdf`, 8/8 file hashes match `fd28da0`; default-off `LLAMA_RORET`; engine bit-exact parity was NOT established by this patch. |
| M4 | [BITEXACT_REPORT.md](../tools/roret/BITEXACT_REPORT.md) | NO_GO | Engine-level GPU parity NO_GO (no GPU build or model run in that gate); host/tile parity PASS-SCOPED: 0 differing bytes out of 24576 compared. See PARITY_REPORT row for the CPU test path. |
| M5 | [TRAFFIC_REPORT.md](../tools/roret/TRAFFIC_REPORT.md) | PASS-SCOPED | LOAD-COUNT MODEL ONLY: OFF 24 uploads/layer vs ON 8.25-11.59 mean union; no tok/s, no measured HBM bytes, no T4 traffic counters. |
| M6 | [SCHEDULER_DESIGN.md](../tools/pibs/SCHEDULER_DESIGN.md) | PASS-SCOPED | Batch 8: 8 steps / 104 wasted -> 3 steps / 24 wasted; batch 1 regression 13 -> 45 wasted; all counts, enabling condition `batch >= 4`, default-off proposal. |
| M7 | [PIBS_WIRING.patch](../tools/pibs/PIBS_WIRING.patch) | PASS-SCOPED | `git apply --check` + `git apply` PASS on base `a05f501`, 2/2 file hashes match `f8ed8e4`; pibs ctest 2/2; decision/instrumentation seam only, no engine node reorder, no off/on F32 engine parity. |
| M8 | [LAYOUT_DESIGN.md](../tools/odsl/LAYOUT_DESIGN.md) | PASS-SCOPED | Spec-only and default-off; 0/16000 dequant mismatches, 76.428x drift model, 512 B vs 2048 B sector model are counts/estimates, not speed or bandwidth; odsl ctest 1/1. |
| M9 | [CYCLE_AUDIT.json](../tools/audit/CYCLE_AUDIT.json) | NO_CLAIM | `claim=0`, `claim_valid=false`, `verdict=NO_CLAIM`, `gate_eligible=false`; `repeats=50`, `loops=3`, 2000 bootstrap resamples; `ci_low=0.9714442869532246` is an INELIGIBLE local CPU probe value, not evidence. |
| Supporting | [PARITY_REPORT.md](../tools/audit/PARITY_REPORT.md) | PASS-SCOPED | CPU test path (`LLAMA_RORET_CPU_TEST=ON`, synthetic GGUF): generated text and raw F32 logits 0 differing bytes, hook absent off / active on; re-measured today. NOT GPU engine-level parity. |

### Reconciliation of the two parity statements

Both statements are kept, each with its own scope; neither is generalized:

- [BITEXACT_REPORT.md](../tools/roret/BITEXACT_REPORT.md) verdict is **NO_GO** for
  engine-level parity. At that gate there was no GPU build and no `ffn_moe_*` /
  logits off-vs-on comparison. Its measured PASS was only host/tile level
  (real LRU planner, synthetic tiles, 0 differing bytes out of 24576).
- [PARITY_REPORT.md](../tools/audit/PARITY_REPORT.md) reports **MEASURED PASS** for
  generated text and raw F32 logits (0 differing bytes, sha `cf915b84...` and
  `8e1385e1...`) with `LLAMA_RORET` unset vs `=1`, on the synthetic GGUF with the
  test-only `LLAMA_RORET_CPU_TEST=ON` CPU-test build. Hook line absent off, active on.
- These do not conflict: the NO_GO scope is GPU engine-level parity; the PASS scope
  is the CPU test path on the synthetic looped graph. Fresh local CPU-test numbers
  were re-measured this pass (section 4). No GPU engine-level parity, GPU tile
  residency or HBM traffic measurement exists, so that scope stays NO_GO.

## 2. Honest constraints (verbatim)

Copied verbatim from [ROADMAP_48H.md](ROADMAP_48H.md), section
"Honest constraints (binding on every milestone)". The wording below is inherited
from the roadmap, including its references to the planning pass.

```text
1. **No speedup claims without a CycleAudit lower bound.** A published number must be `<= ci_low`
   from `loopkit` with `repeats >= 20` and `loops >= 3`; load counts, wasted-slot counts, drift
   ratios, and sector models are explicitly not speedups. If the gate fails, the claim is NO_CLAIM.
2. **No weights are held.** Every executable check uses a deterministic synthetic random MoE
   fixture or the vendored vectors. Real-model router drift (NDIF / Spectral Health Checker)
   cannot be measured in this window; the union precondition remains a synthetic-model result
   until real readouts exist, and any real-model speed claim stays blocked.
3. **T4 (sm_75) has FP16 tensor cores, no FP4/FP8 tensor cores.** NVFP4 and NVFP4 scale layouts
   are storage/loading optimizations only; ODSL's 4x figure is a sector-fetch model, not measured
   bandwidth, and its FP16 side-lane result is drift reduction, not bit-exactness.
4. **No issue/PR spam.** At most 2 real issues per hour, check `gh issue list` before posting,
   and no upstream PRs from an autonomous agent. `10xdev4u-alt/inkengine-llama` currently has
   `hasIssuesEnabled: false`, so zero issues were created in this planning pass; the five
   prepared drafts below are for manual filing only if issues are enabled later.
```

Notes on the verbatim text: the `hasIssuesEnabled: false` sentence describes the
roadmap planning pass, not this run. `gh issue list -R 10xdev4u-alt/inkengine-llama
--state open` succeeded here and returned issues #2, #4 and #5; those IDs are fork
tracker items, not upstream llama.cpp issues. This pass created zero issues and
zero PRs.

## 3. Gate results (all measured locally this pass; LOCAL CPU, not Kaggle)

| Gate | Exact command | Exact observed output | Exit |
|---|---|---|---|
| Fresh llama-cli build | `cmake -S <fork> -B /tmp/opencode/m10-build -DGGML_CUDA=OFF -DLLAMA_RORET_CPU_TEST=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF` | `Configuring done (4.2s)`, `Generating done (0.3s)`, `Build files have been written to: /tmp/opencode/m10-build` | 0 |
| Fresh llama-cli build | `cmake --build /tmp/opencode/m10-build --target llama-cli -j4` | `[100%] Built target llama-cli` | 0 |
| Binary version | `/tmp/opencode/m10-build/bin/llama-cli --version` | `version: 1.0.0-dev (build 121, commit 57da28a)` / `built with GNU 16.2.1 for Linux x86_64` | 0 |
| roret ctest | `cmake -S tools/roret -B /tmp/opencode/m10-ctest-roret && cmake --build /tmp/opencode/m10-ctest-roret -j4 && ctest --test-dir /tmp/opencode/m10-ctest-roret --output-on-failure` | `1/2 Test #1: test_union_cache ... Passed`, `2/2 Test #2: test_roret_tile_bitexact ... Passed`, `100% tests passed out of 2`, `Total Test time (real) = 0.01 sec` | 0 |
| pibs ctest | `cmake -S tools/pibs -B /tmp/opencode/m10-ctest-pibs && cmake --build /tmp/opencode/m10-ctest-pibs -j4 && ctest --test-dir /tmp/opencode/m10-ctest-pibs --output-on-failure` | `1/2 Test #1: pibs_model ... Passed`, `2/2 Test #2: pibs_wiring ... Passed`, `100% tests passed out of 2`, `Total Test time (real) = 0.01 sec` | 0 |
| odsl ctest | `cmake -S tools/odsl -B /tmp/opencode/m10-ctest-odsl && cmake --build /tmp/opencode/m10-ctest-odsl -j4 && ctest --test-dir /tmp/opencode/m10-ctest-odsl --output-on-failure` | `1/1 Test #1: odsl_numeric_layout_model ... Passed`, `100% tests passed out of 1`, `Total Test time (real) = 0.02 sec` | 0 |
| audit pytest | `PYTHONPATH=/tmp/build/loopkit/src python3 -m pytest -q tools/audit/` | `............. [100%]`, `13 passed in 9.02s` | 0 |
| loopkit fast pytest | `PYTHONPATH=/tmp/build/loopkit/src python3 -m pytest -q -m "not kaggle"` (run in the sibling `loopkit` checkout at `0597f5e`) | `........................................ [100%]`, `40 passed, 1 deselected in 1.58s` | 0 |
| M3 patch apply | clone at base `5035bdf`: `git apply --check <UNION_WIRING.patch>` then `git apply <UNION_WIRING.patch>` | `UNION apply --check: PASS (exit 0)`, `UNION apply: PASS (exit 0)`; 8/8 touched files hash-match `fd28da0`; patch sha256 `a91868081fbb9d7657946dfff1c4513171b08131a38a1099934728212bca1e84` (603 lines, 26215 B) | 0 |
| M7 patch apply | same clone at base `a05f501`: `git apply --check <PIBS_WIRING.patch>` then `git apply <PIBS_WIRING.patch>` | `PIBS apply --check: PASS (exit 0)`, `PIBS apply: PASS (exit 0)`; 2/2 touched files hash-match `f8ed8e4`; patch sha256 `f69543d2ba51bf16a5af2fd5c17b3ccdc64c16d252ac844186fe1cd294371804` (131 lines, 5054 B) | 0 |
| CPU-test text parity | `ROOT=<fork> BUILD=/tmp/opencode/m10-build MODEL=/tmp/opencode/m10-parity/synthetic.gguf OUT=/tmp/opencode/m10-parity/out NGL=0 MOE_MIB=1 CPU_SHARE=1.0 N=16 bash tools/audit/parity_run.sh` | `TEXT_PARITY: 0 differing bytes`; off/on text sha both `cf915b84cf60abfd20ced6ac8ec7e69a72c2f0c4b0e2f6abbe8c9c9130bbec3d`; `off: RORET_HOOK_ABSENT`, `on: RORET_HOOK_ACTIVE` | 0 |
| CPU-test raw F32 logits parity | same harness, `llama-debug` built from the fresh tree | `LOGITS_PARITY: 0 differing bytes`; off/on logits bin sha both `8e1385e1e1da124308d84bbc6f2f5e2d16f38496c580fadd203561da0a67131c`; token-id bin sha both `31760d32cbb5458f74281f26cf8c4308e79b7db3c48a5285296904e79c61dee8` | 0 |

Every result above is a build, count, byte comparison or gate result on the
synthetic fixture or vendored vectors. None is a speed measurement. The `ci_low`
recorded in M9 comes from an ineligible local CPU probe and is not reused here.

## 4. Default-off gating and clean source tree

Explicit confirmation, checked against the current tree at HEAD `57da28a`:

- M3 / [UNION_WIRING.patch](../tools/roret/UNION_WIRING.patch) remains default-off.
  `src/llama-moe-cache.cpp:151-153` constructs the union cache only when
  `getenv("LLAMA_RORET")` is exactly `1`, `on` or `true`; unset or any other value
  leaves the upstream path unchanged. The patch is an artifact, not an un-gated edit.
- `LLAMA_RORET_CPU_TEST` is a test-only CMake option, default `OFF`
  (`CMakeLists.txt:15-17`). It only allows the CPU backend for the MoE cache in
  test builds (`src/llama-moe-cache.cpp:168-184`). This pass set it to `ON`
  deliberately in a scratch build dir; production builds are unchanged.
- M7 / [PIBS_WIRING.patch](../tools/pibs/PIBS_WIRING.patch) remains default-off.
  `src/llama-pibs.h:22-28` enables the gate only for exact `1`, `on` or `true`, and
  `src/llama-pibs.h:78-80` interleaves only when the gate is on AND
  `n_tokens >= kMinBatch` (4); batches 1 and 2 stay sequential. The hook in
  `src/models/inkling2-loop.cpp:87-97` computes and logs a plan only; the pass loop
  and every tensor are unchanged.
- M8 / [LAYOUT_DESIGN.md](../tools/odsl/LAYOUT_DESIGN.md) is spec-only and
  default-off. Its proposed `LLAMA_ODSL`, `ODSL_LAYOUT` and `LLAMA_ODSL_SIDE_LANE`
  gates are not present in this tree's `CMakeLists.txt` or `src/`; no patch and no
  source edit were shipped. It is independent of M3/M7 gating.
- No scratch source edits: `git status --short -- src/` produced zero bytes
  (empty output). Before this document was created, `git status --short` for the
  whole fork was also empty; afterwards its only entry is
  `?? docs/VERIFICATION_48H.md`.

Observed build warnings, recorded honestly (non-fatal, exit 0): three
`-Wunused-variable` warnings in `src/llama-pibs.h:84-86` where `s0`, `s1`, `s2`
are used only inside `assert()` under `-DNDEBUG`, plus pre-existing
`-Wdeprecated-enum-enum-conversion` warnings in `tools/mtmd` headers. CMake also
provisioned UI assets from huggingface.co during the build; no model weights were
downloaded and no weight-dependent measurement was made.

## 5. STOP conditions at hour 48

Verbatim from [ROADMAP_48H.md](ROADMAP_48H.md), section "STOP conditions":

```text
1. **Correctness stop:** any nonzero byte difference in M4 or M7 on/off F32 dumps, or any
   evidence that union ids reached `selected_experts_in` / `mm_ids`. Pause all optimization
   milestones, revert to the known-good default-off path, and do not restart until the root
   cause is recorded in `BITEXACT_REPORT.md`.
2. **Claim-integrity stop:** any speed statement in any artifact without a CycleAudit report
   satisfying `claim <= ci_low`, or two consecutive failed audits. Pause all reporting and the
   hourly loop's outward-facing output; continue only correctness and documentation work until
   a valid audit exists or the claim is deleted.
3. **Budget/policy stop:** remaining Kaggle GPU quota below 6 h, a GPU kernel failing twice
   consecutively, an external write rate above 2 issues/hour, or a task attempting a
   weight-dependent measurement without weights. Pause all external writes and GPU scheduling;
   fall back to local CPU models and re-enter only at the next 6 h checkpoint with evidence.
```

| # | Hour-48 value | Evidence |
|---|---|---|
| 1. Correctness | NOT TRIGGERED | Host/tile level: 0 differing bytes out of 24576 compared (`BITEXACT_REPORT.md`). Fresh local CPU test path this pass: text 0 differing bytes and raw F32 logits 0 differing bytes with hook absent off / active on. No byte flip observed and no evidence that union ids reached `selected_experts_in` / `mm_ids` (diff inspection only). No GPU engine-level parity was measured, so that scope remains NO_GO; missing evidence is not a byte flip. |
| 2. Claim-integrity | NOT TRIGGERED | `CYCLE_AUDIT.json`: `claim=0`, `claim_valid=false`, `gate_eligible=false`, `verdict=NO_CLAIM`; self-control claim 1.01 failed closed (exit 1). All other fork-authored reported figures are counts, models or drift/layout estimates. The claim-integrity audit is scoped to fork-authored artifacts; inherited upstream llama.cpp docs (for example `docs/moe-cache.md:10,22` and `docs/speculative.md:3,5`) predate this fork and make no claim about this fork's work. No fork speed statement is published and no eligible audit has twice failed (M10-era status). SUPERSEDED by the M25 addendum above: an eligible Kaggle 2xT4 CycleAudit gate has now run with integrity and returned verdict `NO_CLAIM` (`claim=1.0`, `ci_low=0.9977116162363447`, `ci_high=0.9999835093546772`); see the addendum for current claim-integrity status. |
| 3. Budget/policy | NOT TRIGGERED | No Kaggle kernel run in this pass and no GPU quota consumed; no GPU kernel failed; zero issues and zero PRs created this run; all checks used the synthetic fixture or vendored vectors, so no weight-dependent measurement was attempted. |

## 6. Fresh rebuild evidence and Kaggle status

Fresh LOCAL CPU rebuild, not a Kaggle build, completed this pass:

- Clean configure from HEAD `57da28a` with `GGML_CUDA=OFF`,
  `LLAMA_RORET_CPU_TEST=ON`, `LLAMA_BUILD_TESTS=OFF`, `LLAMA_CURL=OFF`:
  exit 0. Build of `llama-cli` reached `[100%] Built target llama-cli`, exit 0.
- Binary: `version: 1.0.0-dev (build 121, commit 57da28a)`, GNU 16.2.1 x86_64.
- `llama-debug` was built incrementally from the same tree for the raw F32 logits
  dump. The synthetic GGUF was regenerated with seed 0 and sha256
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`, matching
  the fixture recorded in `CYCLE_AUDIT.json`.
- Three standalone ctest suites were run in separate `/tmp/opencode/m10-*`
  build dirs: roret 2/2, pibs 2/2, odsl 1/1, all exit 0. Audit pytest 13 passed,
  loopkit fast pytest 40 passed / 1 deselected, both exit 0.
- CPU-test off/on parity was re-measured on this fresh build: 0 differing bytes for
  generated text and for raw F32 logits. This is the CPU test path only.

Roadmap requirement "Fresh Kaggle CPU rebuild of fork `main` + applied gated
patches is green": PARTIALLY MET. Prior Kaggle evidence is recorded, but no new
Kaggle kernel was run in this pass:

- [PARITY_REPORT.md](../tools/audit/PARITY_REPORT.md) section "Kaggle reproduction":
  kernel `jprinceraj/inkengine-cpu-parity` (CPU, fork `main`, CPU-test build) was
  **COMPLETE / PASS** with `TEXT_PARITY: 0 differing bytes`,
  `off: RORET_HOOK_ABSENT`, `on: RORET_HOOK_ACTIVE`,
  `LOGITS_PARITY: 0 differing bytes`, `KAGGLE CPU PARITY PASS`.
- [BUILD_STATUS.md](../BUILD_STATUS.md) records earlier Kaggle fork builds green
  with `GGML_CUDA=OFF` (component ctests rc 0 each).
- Neither prior result is a GPU engine-level parity or speed result. No Kaggle 2xT4
  CycleAudit was run this pass.

## 7. Remaining blockers

Historical M10-era blocker list, kept as record. All entries below are
superseded by the M25 addendum above as of 2026-11-28; in particular #2, #4 and
#5 are CLOSED and an eligible Kaggle 2xT4 CycleAudit gate has run.

- Issue #2 - deterministic fixture / F32 dump harness: the artifacts now exist
  ([make_moe_fixture.py](../tools/audit/make_moe_fixture.py),
  [make_synthetic_gguf.py](../tools/audit/make_synthetic_gguf.py),
  `dump_f32.py`, [parity_run.sh](../tools/audit/parity_run.sh)) and reproduce stable
  hashes, but the issue was open at the M10 gate; `cert.json` `root_digest` /
  `all_exact` were still PENDING in
  [AUDIT_PROTOCOL.md](../tools/audit/AUDIT_PROTOCOL.md) then.
  SUPERSEDED 2026-11-28: issue #2 is CLOSED (verified today); see the M25
  addendum above.
- Issue #4 - engine-level GPU parity was NO_GO at the M10 gate. At that gate no
  GPU build, GPU tile residency check, or raw `ffn_moe_*` off/on comparison had
  been measured; the CPU-test PASS did not close this.
  SUPERSEDED 2026-11-28: issue #4 is CLOSED (verified today). M18 recorded
  engine-level raw F32 FINAL logits parity PASS-SCOPED on 2xT4 for the synthetic
  3-pass GGUF at both `cpu_share 0.0` and `cpu_share 1.0`. T4 tile-residency
  counters remain unmeasured. See the M25 addendum above.
- Issue #5 - T4 traffic counters and the CycleAudit claim were open at the M10
  gate. At that gate no T4 `n_hit` / uploaded-byte measurement existed and no
  eligible claim existed.
  SUPERSEDED 2026-11-28: issue #5 is CLOSED (verified today). An eligible Kaggle
  2xT4 CycleAudit gate has now run with integrity and returned verdict
  `NO_CLAIM`. T4 `n_hit` / uploaded-byte counters remain unmeasured and the 6
  M23 `RORET_PASS` rows are counts, not wall-clock savings. See the M25 addendum
  above.
- No speed number is publishable until a Kaggle 2xT4 CycleAudit with
  `repeats >= 20` and `loops = 3` supports `claim <= ci_low`. The current 40-pass
  plus NO_CLAIM record is the honest handoff state.
  UPDATED 2026-11-28: the Kaggle 2xT4 CycleAudit ran with `repeats=50` and
  `loops=3`, but the whole 95% interval is below 1.0, so NO SPEED CLAIM is
  publishable. The M24 `NO_CLAIM` record is the honest handoff state; see the
  M25 addendum above.

## 8. Discrepancies found

- [BITEXACT_REPORT.md](../tools/roret/BITEXACT_REPORT.md) NO_GO vs
  [PARITY_REPORT.md](../tools/audit/PARITY_REPORT.md) MEASURED PASS: reconciled by
  scope in section 1 (GPU engine-level vs CPU test path); both statements are kept.
- [BUILD_STATUS.md](../BUILD_STATUS.md) local re-verification line records pibs
  ctest 1/1; the current tree has 2/2 because M7 added `pibs_wiring`. The 2/2
  measured today is the current value.
- `CYCLE_AUDIT.json` records `fork.head = 2848d5a` and an environment
  `local CPU ... LLAMA_RORET_CPU_TEST=ON`; that is the head at the M9 run, while
  this verification is at `57da28a`. This is provenance, not a contradiction.
- The loopkit fast pytest first failed at collection without `PYTHONPATH`; with
  `PYTHONPATH=/tmp/build/loopkit/src` it passed 40 passed / 1 deselected. The
  `.py` sources in `/tmp/build/loopkit/src` and the sibling `loopkit/src` are
  identical (only `.pyc` caches differ).
- The roadmap says "Fresh Kaggle CPU rebuild"; only a fresh local CPU rebuild was
  performed this pass. Recorded as PARTIALLY MET in section 6, not as a Kaggle PASS.
