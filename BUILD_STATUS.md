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

## M30 rank-64 gate semantics gate (branch main)

- Roadmap section 4 item 2 (next unchecked after M29): the upstream graph uses
  rank-64 per-channel gain/forget controllers, not the local scalar `gamma`.
- New `tools/audit/check_gate_semantics.py` mirrors the exact op sequence at
  `src/models/inkling2.cpp:236-244` and writes
  `tools/audit/M30_GATE_SEMANTICS_SUMMARY.json`: verdict PASS, 23/23 boolean
  checks PASS, blocker empty, `speed_claim: null`, note "gate semantics only,
  no timing". Fails closed on tampered facts, marker needles, op sequence,
  ctrl tensors or live `src/` sha256; stale PASS summaries are overwritten.
- Fixed-seed synthetic instance: best scalar gamma 0.349170 leaves max abs gap
  3.013654 (tol 0.010000); zero controller weights give `out == cur` exactly.
  This is one small synthetic case, not a universal proof, no weights, no graph
  execution and no timing.
- Local tests this pass: new audit suite 23 passed, full `tools/audit` 89 passed,
  ASCII-only new files, `git diff --check` clean. No `src/` or CMake edit, no
  engine rebuild, no Kaggle kernel, NO SPEED CLAIM.

## M31 loop-boundary metadata gate (branch main)

- Roadmap section 4 item 3 (next unchecked after M30). Reconciled primary-source
  boundaries are now executable: `core_start=6`, `core_end=18` (exclusive), so the
  loop core is layers 6..17 (12 layers), prelude 0..5, coda 18..23, and
  `n_slot = 24 + 2*12 = 48`.
- New `tools/audit/check_loop_boundaries.py` with frozen facts in
  `LOOP_BOUNDARIES.json`, tests, report `LOOP_BOUNDARIES.md`, and generated
  `M31_LOOP_BOUNDARIES_SUMMARY.json`: verdict PASS, 22/22 checks true, blocker
  empty, `speed_claim: null`, note "loop boundary contract only, no timing".
- Five live source hashes are gated (format facts doc, converter, tensor mapping,
  loop graph source, GGUF constants); failure of any extraction, needle, hash or
  stale-doc check forces NO_GO and overwrites the summary instead of leaving a
  stale PASS. Independent review found one HIGH (en-dash/line-wrap fail-open in
  the stale-phrase check) and one LOW (`12x3=48` shorthand); both fixed, and the
  original `git show HEAD` wording is now caught by the gate.
- Corrected the stale section 4 premises in both `loopkit/docs/ENGINE_SPEC.md` and
  `inkengine/docs/ENGINE_SPEC.md` (byte-identical, sha256
  `6557926f895811e326c1d64ce635b4721470dac3c9d423dc80a07eb616737d7d`); changed
  lines ASCII-only, `git diff --check` clean in all three repos.
- Local tests this pass: checker exit 0, `tools/audit` pytest 112 passed, loopkit
  fast pytest 40 passed 1 deselected. No `src/` or CMake edit, no engine rebuild,
  no Kaggle kernel. Metadata/contract scope only: no weights, no graph execution,
  no timing, NO SPEED CLAIM. Any future number still needs CycleAudit
  `claim <= ci_low`, `repeats >= 20`, `loops = 3`.

## M32 pass-indexed KV slot gate (branch main)

- Roadmap section 4 item 5 (next unchecked cheap-verification item). New offline,
  fail-closed gate `tools/audit/check_pass_indexed_kv.py` with frozen facts in
  `tools/audit/PASS_INDEXED_KV.json`, report `PASS_INDEXED_KV.md` and generated
  `M32_PASS_INDEXED_KV_SUMMARY.json`: verdict PASS, blocker empty, 27/27 checks true,
  `speed_claim: null`, note "pass-indexed KV slot contract only, no timing".
- Frozen facts: `n_phys=24`, `n_passes=3`, `core_start=6`, `core_end=18`, `n_slot=48`,
  `n_ctrl=4=LOOP_CTRL_COUNT`; the three live formula strings and 19 source needles
  are frozen too. Derived slot map has 36 entries: pass 0 il=ip=6..17, pass 1 il=24..35,
  pass 2 il=36..47, all distinct and inverted by `layer_phys(il)`. Coda slots 18..23
  and prelude slots 0..5 are each used once, so 48 applications fill 48 slots.
  RORET routing is keyed by `(roret_il=ip, roret_pass=pass)`, not by the cache slot.
- Three gated source hashes (frozen equals live): `conversion/inkling2.py` sha256
  `aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7`,
  `src/models/inkling2-loop.cpp` sha256
  `a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab`,
  `src/models/inkling2.cpp` sha256
  `c5b543ad965857efd7f9818d42d1f7a418954758b75ec68c82485ffbd341225f`.
- Checker exit 0. Fail-closed on any frozen fact/needle/hash mutation, malformed JSON,
  missing source or extraction exception; a pre-seeded stale PASS is always overwritten.
- Local tests this pass: new audit suite 69 passed, full `tools/audit` 181 passed,
  loopkit fast pytest 40 passed 1 deselected, `git diff --check` clean, all new files
  ASCII-only. No `src/` or CMake edit, no engine rebuild, no Kaggle kernel.
  Contract/metadata scope only: no weights, no graph execution, no timing, NO SPEED CLAIM.
  Any future number still needs CycleAudit `claim <= ci_low`, `repeats >= 20`, `loops = 3`.

## M33 quant-layout and Unsloth quant contract gate (branch main)

- Roadmap section 4 item 6 (next unchecked cheap-verification item after M32;
  item 4 stays blocked without weights or hosted inference). New offline,
  fail-closed gate `tools/audit/check_quant_layout.py` with frozen facts in
  `QUANT_LAYOUT.json`, report `QUANT_LAYOUT.md` and generated
  `M33_QUANT_LAYOUT_SUMMARY.json`: verdict PASS, 28/28 checks true, blocker
  empty, `speed_claim: null`, note "quant layout and Unsloth quant contract
  only, no timing". Checker exit 0; no network calls from the checker.
- Frozen converter semantics from `conversion/inkling2.py` (sha256
  `aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7`): NVFP4
  dequant groups a flat 16 per row with E4M3 block scales and FP32
  amax/outer-scale denominator (`scheme nvfp4_nearest`, `weight_scale_format
  amax_denominator`); MXFP8 groups a flat 32 per row with UE8M0 scales and must
  not carry amax/denominator (`scheme mxfp8_nearest`). Divisibility and
  stored-block guards are frozen as 17 needle groups.
- One bounded HTTP range probe (bytes 0-16777215) of
  `unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ1_S.gguf`, first-16MiB sha256
  `87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76`, GGUF v3,
  744 tensors, 63 KVs: `general.architecture inkling2_loop`, `general.file_type
  24` (MOSTLY_IQ1_S except 1d tensors), 156 loop tensors with F32 rank-64
  controllers, expert weights IQ1_S / Q2_0, not native NVFP4/MXFP8. The checker
  never re-fetches.
- OPEN finding recorded, not resolved: HF scale shapes give 16 elements/scale
  for both precisions (2304/144, 1152/72), the converter uses a flat 32 for
  MXFP8, and published `weight_block_shape [32,32]` gives 1024. These three do
  not agree; `checkpoint_scale_parity: UNVERIFIED` without weights or a full
  header. PASS means only that frozen facts and live files agree; no speed
  claim.
- Independent Hy4 Flash review: 0 CRITICAL/HIGH/MEDIUM; 3 LOW non-blocking
  notes (summary write outside try, weak warning-body needle, heading wording).
  The needle and wording notes were applied; the write-path note is an
  unwritable-directory case that no fail-closed code can overwrite. 15 tamper
  scenarios in an isolated copy all forced NO_GO and overwrote a seeded PASS.
- Local tests this pass: new audit suite 54 passed, full `tools/audit` 235
  passed, loopkit fast pytest 41 passed, `git diff --check` clean, all new files
  ASCII-only. No `src/`/CMake edit, no engine rebuild, no Kaggle kernel.
  Contract/metadata scope only: no weights, no graph execution, no timing,
  NO SPEED CLAIM. Any future number still needs CycleAudit
  `claim <= ci_low`, `repeats >= 20`, `loops = 3`.

## M34 MXFP8 expert scale-layout gate (2026-11-28, branch main)

- Resolves GitHub issue #9 with a new offline, fail-closed gate
  `tools/audit/check_mxfp8_scale_layout.py`, frozen facts in
  `tools/audit/MXFP8_SCALE_HEADER.json`, report `MXFP8_SCALE_HEADER.md` and
  generated `M34_MXFP8_SCALE_SUMMARY.json`: verdict PASS, blocker empty,
  18/18 checks true, `speed_claim: null`, note "MXFP8 scale-layout parity only;
  no timing or speed claim". Checker exit 0; no network calls in the checker.
- One out-of-band bounded header fetch (bytes 0-148415, 148416 bytes, sha256
  `a154b6ce2ead7032e9c0edcdcf020bf819a35f5078a36cce90938abe4d2c5247`,
  header len 148408, 1245 tensors) gives MXFP8 layers 20-23 scale shapes
  [128,2304,72] and [128,2304,36], i.e. 32 elements per scale and stored block
  (1,32). NVFP4 layer 19 scales are [128,2304,144] and [128,2304,72], i.e. 16
  per scale. The converter (sha256
  `aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7`) checks
  `block != (1, 16 if nvfp4 else 32)` and reshapes dequant to 32 for MXFP8:
  the converter is correct and needs no change.
- Three-way resolution: 16 was the mislabeled NVFP4 shape (WRONG for MXFP8);
  32 is the real stored layout and converter behavior (CORRECT);
  published `hf_quant_config.json` `weight_block_shape [32,32]` = 1024
  elements per scale is config metadata and is recorded as INFO, not a
  correctness blocker. M33's OPEN finding and UNVERIFIED parity are explicitly
  SUPERSEDED in the new facts; no M33 file or sibling loopkit doc was edited.
- Fails closed on frozen shape/hash/provenance mutation, converter needle or
  live-hash tamper, missing or malformed M33 facts, malformed JSON, and any
  extraction exception; a seeded stale PASS is always overwritten with NO_GO.
- Local tests this pass: checker exit 0; new audit suite 18 passed; full
  `tools/audit` 253 passed; loopkit fast pytest 41 passed with
  `PYTHONPATH=../loopkit/src`; `git diff --check` clean; all new files
  ASCII-only. No `src/`, `conversion/`, CMake or sibling-repo edit, no engine
  rebuild, no Kaggle kernel. Metadata/header scope only: no weights loaded, no
  engine run, no timing, NO SPEED CLAIM. Any future number still needs
  CycleAudit `claim <= ci_low`, `repeats >= 20`, `loops = 3`.

## M35 d2027 outlier-premise gate (2026-11-28, branch main)

- Adds an offline, fail-closed gate `tools/audit/check_outlier_premise.py`,
  frozen facts `tools/audit/OUTLIER_PREMISE.json`, report `OUTLIER_PREMISE.md`
  and generated `M35_OUTLIER_PREMISE_SUMMARY.json`: verdict PASS, blocker empty,
  27/27 checks true, `classification: LOCAL_ONLY_UNVERIFIED`,
  `real_trace_verdict: NO_GO_BLOCKED`, `side_lane_enabled: false`,
  `speed_claim: null`, note "outlier premise classification only; no timing".
  Checker exit 0; no network calls and no loopkit imports.
- Three out-of-band primary-source records are frozen: TML release post 448727
  bytes sha256 `1301b66c36ff9112dd9c731cc5c83eba1e6b6c401c6364301fbc933d40f7249c`,
  TML model card 222597 bytes sha256
  `2fd49bae2efdbef1914314280ebc2d93742f89e903f66354c30b02b95f5efe28`, HF
  `config.json` 2722 bytes sha256
  `a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2`; all
  have zero "2027" hits. The 2722-byte `config.json` is embedded verbatim in
  the facts and the offline checker recomputes its byte count, sha256 and
  zero-hit count; the two HTML records remain out-of-band assertions the
  offline checker cannot re-fetch or recompute. `hidden_size = 2304` is
  recorded as INFO only, not evidence. Six live local files (loopkit config/model/integration doc, fork
  README, two ODSL docs) are hash-gated with needles labeling d2027 an
  unverified local hypothesis and the side-lane default-off.
- Engine scope `src/`, `common/`, `ggml/`: 24 loop-specific files, no
  `outlier_dim`, no `d2027`, no scoped 2027 literal and no ODSL side-lane
  enable. Unrelated hits (`0x2027`, `2027.0`, `-1.453152027f`) are excluded by
  scope and are not reported as evidence. The speculative FP32 side-lane is
  NOT enabled in the fork engine path.
- Fails closed on frozen primary sha/bytes/hits mutation, inline-content
  mutation, facts-digest mismatch, needle or live-hash mismatch, missing file,
  malformed JSON, extraction exception, and any enabled side-lane; a seeded
  stale PASS is always overwritten with NO_GO. Path and content markers are
  case-insensitive, so a new engine file cannot evade the scoped `2027` check
  by uppercase marker text; a concrete bypass probe returned NO_GO.
- Local tests this pass: checker exit 0; new audit suite 27 passed; full
  `tools/audit` 280 passed with `PYTHONPATH=../loopkit/src:gguf-py`; loopkit
  fast pytest 41 passed; `git diff --check` clean; all new files ASCII-only.
  No `src/`, CMake, engine or sibling-repo edit, no engine rebuild, no Kaggle
  kernel. Metadata/contract scope only: no weights, no forward-hook trace, no
  timing, NO SPEED CLAIM. Real-trace falsification stays NO_GO_BLOCKED because
  it needs weights or hosted inference. Any future number still needs
  CycleAudit `claim <= ci_low`, `repeats >= 20`, `loops = 3`.

## M36 union-wiring contract gate (2026-11-28, branch main)

- Adds five files under `tools/audit/`: `UNION_WIRING.json` (frozen source sha256
  and needle contract), `check_union_wiring.py` (stdlib-only, offline, fail-closed
  CLI with optional `--root`), `test_union_wiring.py` (58 tests: 47 invoke the
  checker CLI through subprocess, 11 are direct assertion/unit tests),
  `UNION_WIRING.md` (scope doc), and generated
  `M36_UNION_WIRING_SUMMARY.json`.
- Gate covers eight gated sources with 18 ordered chain checks, 7 default-off
  checks and 5 absent checks, plus contract/hash/coverage/completion checks:
  35 booleans total. The chain now gates the union-cache implementation in
  `src/llama-roret.h` (add, residents, clear_layer, internal union bits/state)
  and the lookup-keyed set/find binding semantics in
  `src/llama-roret-binding.h`. Every requested needle was verified verbatim;
  no substitution was needed. Generated summary: verdict PASS, blocker empty,
  35/35 checks true, `check_counts` total 35 passed 35 failed 0,
  `speed_claim: null`, `no_weights: true`, `no_engine_run: true`.
- Fails closed on live or frozen sha256 mismatch, chain/default-off needle
  deletion, absent-needle injection, frozen-field tamper, malformed or missing
  contract, missing source, and any exception; a seeded stale PASS is always
  overwritten with NO_GO. Tamper tests use isolated temp copies of the sources,
  including dedicated hash/needle tamper cases for the two new headers.
- Summary writes are robust: a NO_GO sentinel is written before checks and the
  final result is written atomically (temp file plus `os.replace`); a write
  failure returns exit 1 and leaves the last written file unchanged. Tests cover
  the sentinel ordering, atomic write failure cleanup, and the non-zero CLI exit.
- Local checks this pass: checker exit 0; new suite 58 passed; full
  `tools/audit` 338 passed with `PYTHONPATH=../loopkit/src:gguf-py`; loopkit
  fast pytest 41 passed from the loopkit root with `PYTHONPATH=src`;
  `git diff --check` clean; all new files ASCII-only. No `src/`, CMake or
  sibling-repo edit, no engine run, no Kaggle kernel, no weights. This is a
  source-contract gate only: no graph execution, no bit-exact or tensor parity,
  no timing and no speedup. NO SPEED CLAIM; CycleAudit `claim <= ci_low`,
  `repeats >= 20`, `loops = 3` remains mandatory for any future number.

## M37 GGUF loop-tensor shape contract gate (2026-11-28, branch main)

- Resolves the last unchecked `[?]` item in
  `loopkit/docs/INKLING_INTEGRATION.md` section 2: exact per-tensor GGUF shapes
  beyond the HF shapes. That doc is intentionally NOT edited because M35
  hash-gates it; its marker stays pending a coordinated two-repo doc update.
- One bounded out-of-band range probe of
  `unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ1_S.gguf` bytes
  0-16777215 (16777216 bytes, sha256
  `87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76`,
  GGUF v3, 744 tensors, 63 KVs) is frozen in
  `tools/audit/GGUF_LOOP_SHAPES.json`. The same hash is cross-checked against
  M33 `QUANT_LAYOUT.json`; the offline checker never fetches or reads a cache.
- Exact 204-tensor contract: 12 `loop_msg_query` [2304,4], 144 controllers
  (4 families x 3 roles on core layers 6..17; down [4608,64], gain/forget
  [64,2304]), 23 `attn_res_query` ([2304], 4 global-tail layers 5/11/17/23
  are [2304,3]), 24 `ffn_res_query` [2304], 1 `output_res_query` [2304],
  all F32. GGUF dims are reversed vs HF dims.
- New `tools/audit/check_gguf_loop_shapes.py`: stdlib-only, offline,
  fail-closed CLI with `--root`; NO_GO sentinel before checks and atomic final
  write; verdict PASS, blocker empty, 22/22 checks true, `speed_claim: null`,
  `no_weights: true`, `no_engine_run: true`. Summary sha256
  `693289c81e8d3e26affa58e5a3c22e167c56c2e7c2e49f9bed2d86fd849e1cfb`.
  It binds four live source hashes plus 6 needle groups, derives dimensions
  from primary dims, compares the measured shape/dtype histogram to the
  contract histogram, and rejects extra top-level facts keys. A sentinel
  write failure now forces NO_GO.
- Independent Hy4 Flash adversarial review: 0 CRITICAL/HIGH, 2 MEDIUM and
  3 LOW. Fixes applied: frozen measured-shape histogram anchors the contract
  to the header measurement, exact top-level fact key set, sentinel failure
  forces NO_GO, checker source SHA-256 pinned in tests, and the real-tree CLI
  test restores the summary bytes it overwrites.
- Local tests this pass: checker exit 0; new suite 38 passed; full
  `tools/audit` 376 passed with `PYTHONPATH=../loopkit/src:gguf-py`; loopkit
  fast pytest 41 passed; `git diff --check` clean; all new files ASCII-only.
  No `src/`, CMake or sibling-repo edit, no engine run, no Kaggle kernel,
  no weights. Metadata/contract scope only: no graph execution, no tensor or
  logits parity, no timing. NO SPEED CLAIM; CycleAudit `claim <= ci_low`,
  `repeats >= 20`, `loops = 3` remains mandatory for any future number.
