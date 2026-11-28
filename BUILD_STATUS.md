# Fork build status — verified on Kaggle CPUs (2026-11-24)

Kernel `jprinceraj/inkengine-fork-build`, fork `main` @ `646c1a5`, GCC 13.3,
CMake with `GGML_CUDA=OFF`, `LLAMA_BUILD_TESTS=OFF`, `LLAMA_CURL=OFF`.

| Check | Result |
|---|---|
| Clone `-b main` and component presence | PASS |
| CMake configure + build target `llama-cli` | **BUILD OK** |
| `llama-cli --version` | `1.0.0-dev (build 1, commit 646c1a5)` |
| `tools/roret` ctest | PASS (1/1 ctest: test_union_cache) |
| `tools/pibs` ctest | rc 0 |
| `tools/odsl` ctest | rc 0 |
| Overall kernel | **COMPLETE / KAGGLE FORK BUILD PASS** |

Fixes required along the way (all committed):
- kernel cloned upstream default branch `master` instead of our `main` → `-b main`;
- ODSL depended on the sibling `inkengine` repo → LUT and vectors vendored here;
- upstream `*.bin` ignore hid the vectors → force-added; local build dirs untracked.

`tools/roret/test_union_cache.cpp` is now compiled and registered with ctest:
clean default and Release builds both report 1/1 real test, and `-UNDEBUG`
keeps its `assert()` calls active in Release (Release would otherwise define
`NDEBUG` and silently disable the test oracle).

The fork is buildable and all loop-specific prototypes are independently
verifiable on a fresh machine. No speedup is claimed yet; see each tool's
RESULTS.md for the honest numeric models.

## Re-verified 2026-11-24 (kernel v6, HEAD 149a963)

- Full `llama-cli` build: OK
- roret / pibs / odsl ctest: rc 0 each
- audit fixture + gate verifier: 13 passed in 10.30s
- Overall: **COMPLETE / KAGGLE FORK BUILD PASS**
- Note: audit tests require loopkit cloned at `/tmp/loopkit/src`; the kernel now does that.

## Local re-verification 2026-11-25 (M4 increment, branch `main`)

- Full `GGML_CUDA=OFF` `llama-cli` build: OK (100%).
- roret ctest: **2/2** (`test_union_cache`, `test_roret_tile_bitexact`); pibs 1/1; odsl 1/1.
- audit pytest: 13 passed. Standalone bit-exact tests clean under C++17 and C++20 with `-Werror`.
- `tools/roret/test_roret_tile_bitexact.cpp` exercises the real LRU planner and real
  RORET header on synthetic tiles: 24 configs, 0 differing bytes. Host/tile-level
  parity only; engine-level `ffn_moe_*`/logits parity remains NO_GO (see
  `tools/roret/BITEXACT_REPORT.md`). No speed claim.

## M24/M25 re-verification 2026-11-28 (branch main)

- Gate: `claim=1.0`, `ci_low=0.9977116162363447`,
  `ci_high=0.9999835093546772`, point-estimate ratio `0.9999176055055583`,
  `claim_valid=false`, gate exit 1, verdict `NO_CLAIM`. Notebook
  `overall_verdict=PASS` means gate integrity only, NOT a speed claim.
- No speed number is supported: the whole 95% interval is below 1.0 on the
  synthetic 16-token fixture, scoped to a fixed-seed random-weight fixture, not
  a real model. This is a NO SPEED CLAIM state; the point estimate must NOT be
  published as a speed number.
- Evidence: `tools/roret/M24_CYCLE_AUDIT.md` and
  `tools/roret/kaggle-cycle-audit/M24_SUMMARY.json`.
- Local tests on 2026-11-28: ctest roret 10/10, pibs 2/2, odsl 1/1; audit
  pytest 17 passed; loopkit fast pytest 40 passed, 1 deselected.

## M26 re-verification 2026-11-28 (branch main)

- New default-off raw intermediate-tensor dump in `common/debug.cpp`, gated by
  `LLAMA_DUMP_TENSORS_DIR` plus an explicit `--tensor-filter`; no `src/` edit.
- New `tools/roret/check_intermediate_parity.py` and report
  `tools/roret/M26_INTERMEDIATE_PARITY.md`.
- Incremental `GGML_CUDA=OFF` `LLAMA_RORET_CPU_TEST=ON` `llama-debug` rebuild:
  exit 0, no warnings.
- CPU-test parity: 40 tensors per cpu_share, 30080 bytes compared, 0 differing
  bytes at both 0.0 and 1.0; hook ABSENT off / ACTIVE on; both exits 0; verdict
  PASS at CPU-test scope only. Three fail-closed negative controls returned 1
  (empty stub, nonzero-exit stub, deliberately wrong expected-name set).
- Post-review hardening: exact expected tensor-name set and equal occurrence
  counts required in both phases, stale dump dir removal is strict, `.bin`
  sizes are checked against manifest `nbytes`, and C++ write failures are
  logged and skip the manifest row. Re-run after hardening: still PASS with the
  same 40/30080/0 figures.
- Local tests this pass: ctest roret 10/10, pibs 2/2, odsl 1/1; audit pytest 17
  passed; loopkit fast pytest 41 passed.
- No GPU run, no real weights, no speed claim; GPU intermediate parity remains
  unmeasured. Any future number still needs CycleAudit `claim <= ci_low`,
  `repeats >= 20`, `loops = 3`.

## M27 re-verification 2026-11-28 (Kaggle 2xT4, branch main)

- Engine-level intermediate per-pass MoE tensor parity (RORET off vs on) now
  measured on device: `cpu_share` 0.0 and 1.0 each give 40 tensors, 30080 bytes
  compared, 0 differing bytes; hook ABSENT off / ACTIVE on; harness exit 0.
- Kernel `jprinceraj/inkengine-cuda-intermediate-parity` v1 COMPLETE; clone
  `-b main` HEAD `6353cf4`, CUDA configure with `-DGGML_CUDA_NO_VMM=ON` exit 0,
  `llama-debug` build exit 0, fixture sha256 matches the frozen value.
- `tools/roret/kaggle-intermediate-parity/M27_SUMMARY.json`:
  `overall_verdict=PASS`, blocker empty, 20/20 gate keys true.
- The harness `--ngl`/`--threads` flags are default-off; the CPU-test command
  is unchanged and its parity regression still passes 40/30080/0 both shares.
- Local suites this pass: ctest roret 10/10, pibs 2/2, odsl 1/1; audit pytest
  17 passed; loopkit fast pytest 40 passed, 1 deselected.
- Correctness scope only. No real weights, no timing. NO SPEED CLAIM; any future
  number still requires CycleAudit `claim <= ci_low`, `repeats >= 20`, `loops = 3`.

## M28 real-weight parity readiness gate (branch main)

- No real weights are held (roadmap constraint 2), so the only honest local
  result is NO_GO. The gate refuses to classify the frozen synthetic fixture
  as real and does not run the engine when classification fails.
- Added `tools/roret/check_real_weight_parity.py`: fail-closed classification
  from attestation, existence, sha256, size, architecture and expert_count,
  plus reuse of the M26/M27 `compare_share` harness for OFF/ON raw byte parity.
  The verdict is written to `M28_REAL_WEIGHT_SUMMARY.json`; exit 0 only on PASS.
- Added `tools/audit/test_real_weight_gate.py` (pure-function tests), and
  `tools/roret/REAL_WEIGHT_PARITY.md` with the command and metadata-derived
  `--expect-names` recipe for when a real Inkling2 GGUF exists.
- Local smoke check on the frozen synthetic fixture with attestation: exit 1,
  weights_class `synthetic_or_unknown`, two blockers (frozen sha, size),
  no engine run. Local suite: `tools/audit` pytest 29 passed (17 existing plus
  12 new). No `src/` edit, no Kaggle kernel, no speed claim.

## M29 upstream loop-contract parity (branch main)

- First unchecked cheap-verification experiment in `loopkit/docs/INKLING_INTEGRATION.md`
  section 4 item 1, now executable offline via `tools/audit/check_upstream_loop_contract.py`.
- Frozen facts in `tools/audit/UPSTREAM_LOOP_CONTRACT.json` (upstream raw
  `constants.py` sha256 `d904f1dc06317f950c51f82425593790e6bc0f6179b9f6c7b2861ba4c8420bfb`,
  `inkling2-loop.cpp` sha256
  `b5983c6ef955b54d35187933fb0295988997385a12d608c762eb67d06b473772`).
- Checker exit 0, verdict PASS, 10/10 checks: arch string `inkling2_loop`;
  47 tensor names identical to upstream; 5 `LLM_KV_LOOP_*` keys; slot formula
  `n_phys + (size_t) (n_passes - 1)*(core_end - core_start)`; fork arch int 179
  and 5 additive/default-off RORET wiring markers. Upstream arch int 181 is
  recorded as benign version drift (INFO only).
- `M29_UPSTREAM_CONTRACT_SUMMARY.json` sha256
  `d2df484e02cf2feb471968f3d8d6dd2e47251d5c218907b030851c9c8cac00a2`;
  `speed_claim: null`, note "contract parity only, no timing".
- Independent review (0 CRITICAL/HIGH, 1 MEDIUM) hardening: extraction errors
  now force a written NO_GO summary instead of leaving a stale PASS artifact;
  regression test added. Five fail-closed negative controls verified.
- Local tests this pass: new audit tests 20 passed, full `tools/audit` 66 passed,
  loopkit fast pytest 40 passed 1 deselected, 5 standalone C++ RORET tests
  compiled under C++17 `-Werror -UNDEBUG` and ran rc 0. No `src/` or CMake edit,
  so no engine rebuild was needed. No Kaggle kernel, no speed claim.
