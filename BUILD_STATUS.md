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

## M38 coordinated integration-doc closure (2026-11-28, branch main)

- Resolved the stale markup in `loopkit/docs/INKLING_INTEGRATION.md`: section-2
  GGUF loop-tensor shapes and cache-hook plug-in, section-4 items 1, 2, 3, 5 and 6,
  plus the section-5 dated closure note, now carry evidence-labelled `[F]` markers
  pointing at the M29-M37 audit artifacts. Item 4 stays `[L]` with the cheap M35
  classification recorded and real-trace falsification still NO_GO_BLOCKED.
  Doc sha256 old `69c587aba7ebbe3b866b7ca95107cfb891b28b3b30d3c54eeba698f878655432`
  -> new `054188036cb75d373b3a1740051f873e3e5646f5cb8c47810426a7f2f930e4b0`.
  All pinned source needles (d2027 wording, side-lane flag, quant shapes and
  section-4 headings) are byte-identical; existing non-ASCII bytes are untouched.
- Refreshed the new hash in exactly these files: `tools/audit/GATE_SEMANTICS.json`,
  `check_gate_semantics.py`, `GATE_SEMANTICS.md`, `QUANT_LAYOUT.json`,
  `check_quant_layout.py`, `QUANT_LAYOUT.md`, `OUTLIER_PREMISE.json`,
  `check_outlier_premise.py`. `OUTLIER_PREMISE.md` had no hash pin and was not edited.
  The M35 checker's `FROZEN_FACTS_SHA256` also had to move from
  `fbbcd51af241476b0a4e7755dd62ac50bcadefffe69972790a21ad5f6d074c55` to
  `46c7260707371d83a7f662f4c7fad1420736b6f38469c512b835fb74956d6917` because
  editing `OUTLIER_PREMISE.json` changes that whole-file digest; without it the M35
  gate could not return PASS. M29/M36/M37 contracts, checkers and summaries,
  `DECISIONS.tsv`, `PASS_INDEXED_KV.json`, `ENGINE_SPEC.md`, `src/`, CMake and
  the engine were not touched.
- Regenerated summaries from the inkengine-llama root with
  `PYTHONPATH=../loopkit/src:gguf-py`: M30 sha256
  `f5b0f0367379430706ab044f0963cd296dcbd04c4aa66ff6cdd522e6093e2de8`, M33 sha256
  `81f7d4cd842b62b503993b60c1a6481775cdcd5d8e979dc3990de8dc77c4ebf9`, M35 sha256
  `add7b5ccf99d0c329eae837eb4b360cfe0623e18b655da24f4b3dece25e257b3`; all three
  checkers exit 0 with verdict PASS and each summary contains the new doc hash and
  no old-hash occurrence.
- New fail-closed closure gate (5 ASCII-only files): `INTEGRATION_DOC_CLOSURE.json`
  (checker sha256 `7b2232149c4c6292beface76184c53bd92b8de9cc30ab633db79dafb827532b8`,
  frozen_on 2026-11-28), `check_integration_doc_closure.py` (stdlib-only, offline,
  `--root`, NO_GO sentinel before checks, atomic `os.replace` final write),
  `test_integration_doc_closure.py`, `INTEGRATION_DOC_CLOSURE.md` and generated
  `M38_INTEGRATION_DOC_CLOSURE_SUMMARY.json` (sha256
  `0eda87b02afb183772bab4b2833f4613014242f154d5a74b7979f5fb6abf5ec5`, verdict PASS,
  blocker empty, 19/19 checks, speed_claim null). The gate checks the exact contract
  key set, the live doc and checker hashes, every resolved/blocker/closure needle,
  the nine M29-M37 evidence artifact pairs, each cited summary verdict and count,
  the M35 classification and blocked verdict, the M37 history note, and the
  generated summary shape.
- Tests: full `tools/audit` suite 407 passed (376 baseline + 31 new) with
  `PYTHONPATH=../loopkit/src:gguf-py`; new closure suite 31 passed; loopkit fast
  suite 40 passed, 1 deselected (the brief's stated 41-pass baseline was 40 in this
  checkout); `git diff --check` clean in both repos; no new non-ASCII code points
  (two modified doc lines retain pre-existing U+2013 and U+00D7 bytes). An
  adversarial review found no CRITICAL/HIGH issue and three LOW nits, all fixed:
  the ASCII claim above was imprecise, `check_summary_expectations` now guards
  summary loads so independent checks still run on a missing evidence summary,
  and `evidence_verdicts` is now observed from the loaded summaries instead of
  hardcoded PASS. No Kaggle kernel, no network call, no weights, no engine rebuild,
  no speed claim.
  M37's "marker pending" `doc_note` remains as history; M38 supersedes it here.
- Scope: documentation-contract closure only. No weights, no engine run, no graph
  execution, no tensor or logits parity, no timing. NO SPEED CLAIM; CycleAudit
  `claim<=ci_low`, `repeats>=20`, `loops=3` remains mandatory for any future number.
  The d2027 premise stays LOCAL_ONLY_UNVERIFIED and its real-trace falsification
  stays NO_GO_BLOCKED without weights or hosted inference.

## M39 - wiring evidence chain integrity gate (2026-11-28, branch main)

- Adds a single offline, fail-closed command that re-verifies the accumulated
  M29-M38 evidence chain: `tools/audit/check_evidence_chain.py`. Five new
  ASCII-only files under `tools/audit/`: `EVIDENCE_CHAIN.json` (frozen contract,
  sha256 `065311f38f1fff6f51771b57b0c6887775b0035af04a37f928ab304ccddae86b`),
  `check_evidence_chain.py` (sha256
  `aba235c2ae6291b48135843295045e564958541b85fc8aac20cf1333ad099e79`),
  `test_evidence_chain.py` (sha256
  `076c3553401c7326d6052d30cf9fe846ead200f3f5a6343bc8e4cc0f19620923`),
  `EVIDENCE_CHAIN.md` (sha256
  `ad5ccafaa13a332f860ff5a851114f5d9afc7c877e8a2bab6ecf622263cd0e95`), and
  the generated `M39_EVIDENCE_CHAIN_SUMMARY.json`.
- The checker source hash is pinned only in `EVIDENCE_CHAIN.json`, not inside
  the checker itself, so the pin is not self-referential. `EVIDENCE_CHAIN.md`
  records that a future legitimate checker edit must update that pin openly.
- Frozen contract: `schema_version` 1, `frozen_on` 2026-11-28, `scope`/`note`/
  `speed_claim` null, `no_weights` true, `no_engine_run` true, ten
  M29-M38 summary paths each with sha256 and expected verdict PASS, milestones
  `["M29", ..., "M38"]`, and `wiring_contract: tools/audit/UNION_WIRING.json`.
  Any missing, malformed or extra top-level key fails closed.
- Measured checker result from the inkengine-llama root: exit 0, verdict PASS,
  `blocker` empty list, `check_counts` total 94 passed 94 failed 0,
  `speed_claim: null`, `no_weights: true`, `no_engine_run: true`, note "wiring
  evidence integrity only, no timing". Generated summary sha256
  `4f6361b562a82ad2942f32b1b8f07bb5bf22417701f1ca940c0a1b837ae78ba3`.
- The 94 booleans cover: exact contract and checker-hash match; eight pinned
  live union-wiring sources from `UNION_WIRING.json` with frozen equals live
  sha256; ten `land <milestone>` rows in `DECISIONS.tsv`; ten
  `## <milestone> ` headings in `BUILD_STATUS.md`; six integrity properties
  per frozen summary (exists, JSON object, sha256, verdict, no speed claim,
  empty blocker); a static no-network self-check; and run completion.
- Honest shape finding: the M29 summary predates the `blocker` field and has no
  `blocker` key at all, while M30-M38 use `""`. Both are read as "no blocker".
  The byte-exact sha256 pin still detects any edit to M29, so the allowance
  does not weaken the gate. No summary was regenerated and no frozen hash was
  changed to make this gate pass.
- Any previous summary is unlinked first, then a NO_GO sentinel is written
  before checks, and the final verdict uses an atomic temp file plus
  `os.replace`. This closes a dual write failure path found by independent
  Hy4 Flash review: even if both writes fail, no stale PASS remains on disk.
  A final write failure exits 1. Tests cover sentinel ordering, atomic-write
  failure and non-zero CLI exit, plus missing/wrong-sha/wrong-verdict/
  speed-claim/blocker summaries, malformed JSON, mutated live wiring source,
  missing DECISIONS row, missing BUILD_STATUS heading, extra contract key,
  contract field tamper, checker-hash mismatch, and stale-PASS overwrite.
  The reviewer's second LOW note (one test ran against the real tree and
  restored the generated summary bytes) was fixed by dropping that test;
  tamper tests use isolated temp trees only.
- Local tests this pass: new suite 45 passed; full `tools/audit` suite 452
  passed (407 baseline plus 45 new) with
  `PYTHONPATH=../loopkit/src:gguf-py`; `git diff --check` clean; all five new
  files have 0 non-ASCII bytes. No `src/`, `common/`, `ggml/`, CMake or
  `DECISIONS.tsv` edit, no sibling-repo edit, no engine rebuild, no Kaggle
  kernel, no network call, no weights.
- Scope: evidence-chain integrity only. No engine run, no graph execution, no
  tensor or logits parity, no timing. NO SPEED CLAIM; CycleAudit
  `claim<=ci_low`, `repeats>=20`, `loops=3` remains mandatory for any future
  number. Real-weight parity stays blocked without weights.

## M40 - CPU-only 3-pass runtime binding gate (2026-11-28, branch main)

- Adds five new ASCII-only files under `tools/audit/`: `THREEPASS_BINDING.json`
  (sha256 `d7957f4809285c19f8e68d58ec2a12b5d7062127fef06fe693bcab10adeddd7e`),
  `check_3pass_binding.py` (sha256
  `35357c44fdf72b95c240ad1eeefb940c97a28a8a10c38c2955bafd8a2f0eafcf`),
  `test_3pass_binding.py` (sha256
  `efb72aa26936055afed0170b6d7cde7d58a5977d40545ddedaee5ea246d9956e`),
  `THREEPASS_BINDING.md` (sha256
  `de3192af6ab25b84255a969d321d60ac4ff4c0edef2074b30e738e6b5ec0571f`), and
  the generated `M40_3PASS_BINDING_SUMMARY.json` (sha256
  `734658f8a2070dfa8798ceaaa08c4724e28cb8f8c678b9c38236303b097cd9ae`).
- Exact commands from the fork root with the parent-built binary:
  `cmake -S . -B /tmp/opencode/ink-build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DLLAMA_RORET_CPU_TEST=ON`
  and `cmake --build /tmp/opencode/ink-build --target llama-cli -j4`;
  `PYTHONPATH=gguf-py python3 tools/audit/make_synthetic_gguf.py --out /tmp/opencode/synthetic.gguf`;
  `python3 tools/audit/check_3pass_binding.py --cli /tmp/opencode/ink-build/bin/llama-cli --gguf /tmp/opencode/synthetic.gguf --out-dir /tmp/opencode/m40-3pass-logs-final`;
  `python3 -m pytest tools/audit/test_3pass_binding.py -q`;
  `PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit -q`;
  `git diff --check`.
- The checker is stdlib-only, offline and fail-closed. It pins the exact
  contract key set and its own sha256 in the contract (tests additionally pin
  the contract sha256), verifies the 887392-byte synthetic fixture sha256
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`
  and its `inkling2_loop` arch string, then runs the same CPU-only `llama-cli`
  command with `LLAMA_RORET` unset and with `LLAMA_RORET=1 LLAMA_RORET_TRACE=1`.
  It writes a NO_GO sentinel before checks and the final verdict atomically.
  The only required ON enable needle is
  `RORET union-resident MoE bookkeeping enabled`.
  `LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache` appears in OFF as
  well as ON, so it is an unconditional build marker counted in `info`
  (measured off=2, on=2), never a gating enable needle.
- Honest measured result on the real binary: checker exit 0, verdict **PASS**,
  21/21 checks true, empty blocker. Both phases exited 0 at
  `--moe-cache-cpu-share 1.0`; the sibling value `0` aborted during warmup in
  both phases and was rejected as unusable. The ON trace contained 42
  `RORET_BIND set` rows for `key_il=1` with the distinct pass set
  `[0, 1, 2]`; `no_mulmat_total` was 0 and the scoped `no_mulmat` count was 0,
  which is the only observable CPU property for that control. OFF had no hook
  or RORET trace lines. Generated text was byte-identical at 30 bytes and
  normalized stdout at 906 bytes, with 1 dropped line in each phase.
  Normalization drops only lines whose stripped form starts with a trace or
  enable needle, plus llama-cli's non-deterministic
  `[ Prompt: ... t/s | Generation: ... t/s ]` metrics line; an embedded
  occurrence is not dropped, and unequal dropped-line counts fail closed.
  No timing value is compared or recorded.
- Disclosed scope correction: `RORET_PASS` traffic-row gating was REMOVED from
  the required checks because the CPU fallback cannot emit device slot-map
  copy traffic (M21/M22 root cause). Parse completion is recorded by the
  `trace_parse_completed` liveness check; the observed `pass_row_passes` is
  `[]` and `pass_row_count` is 0. These rows are informational only, are
  reported in the summary and in the contract's `cpu_traffic_scope` string,
  and are not treated as traffic evidence. Two of the 21 checks
  (`trace_parse_completed` and `run_completed`) are liveness flags, not
  evidence; the remaining 19 gates carry the binding, hook, contract and
  parity evidence.
- Local tests this pass: new suite 25 passed; full `tools/audit` suite 477
  passed (452 baseline plus 25 new) with `PYTHONPATH=../loopkit/src:gguf-py`;
  `git diff --check` clean; all new files have 0 non-ASCII bytes. No `src/`,
  `common/`, `ggml/`, CMake, `DECISIONS.tsv` or sibling-repo edit, no network
  call, no Kaggle kernel, no real weights.
- Scope: CPU-only runtime binding on a fixed-seed synthetic fixture. No speed,
  timing, throughput or wall-clock win is claimed. NO SPEED CLAIM; CycleAudit
  `claim<=ci_low`, `repeats>=20`, `loops=3` remains mandatory for any future
  number. GPU traffic and real-weight parity stay outside this gate.

## M41 - fail-closed runtime 3-pass loop replay / pass-to-slot mapping gate (2026-11-28, branch main)

- Adds five new ASCII-only files under `tools/audit/`: `THREEPASS_REPLAY.json`
  (sha256 `00e7215637a42f08effaaa5a129846ffcbdb1297284f738b95c04df02ae4657b`),
  `check_3pass_replay.py` (sha256
  `340d96ff2dd99020bdc029f48b2f2cdeb2ee58aac8e108bd6a6e78f7721e188f`),
  `test_3pass_replay.py` (sha256
  `c3e0f3ed9170234d578baf2ce671c77d257537e845f972ffb4d0f7a52bd32f8c`),
  `THREEPASS_REPLAY.md` (sha256
  `be1f1a5ca71873f22b37ebd503f9e2e2f82d7d1554a4af7a423943263ed57b58`), and
  the generated `M41_3PASS_REPLAY_SUMMARY.json` (sha256
  `94e67830ba309791b2bd87abc5c8ee1225fbe964f0c691e5082d8a31ccb9b186`).
- Exact commands from the fork root with the parent-built binary:
  `cmake -S . -B /tmp/opencode/m41-probe -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DLLAMA_RORET_CPU_TEST=ON -DCMAKE_BUILD_TYPE=Release`
  and `cmake --build /tmp/opencode/m41-probe --target llama-debug -j4`;
  `PYTHONPATH=gguf-py python3 tools/audit/make_synthetic_gguf.py --out /tmp/opencode/synthetic.gguf`;
  `python3 tools/audit/check_3pass_replay.py --root . --cli /tmp/opencode/m41-probe/bin/llama-debug --gguf /tmp/opencode/synthetic.gguf --out-dir /tmp/opencode/m41-real`;
  `python3 -m pytest tools/audit/test_3pass_replay.py -q`;
  `PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit -q`;
  `PYTHONPATH=../loopkit/src python3 -m pytest ../loopkit/tests -q -k 'not slow'`;
  `git diff --check`. The suggested `PYTHONPATH=src` variant fails at
  collection with `ModuleNotFoundError: No module named 'loopkit'`; the
  sibling source path `../loopkit/src` is required.
- The checker is stdlib-only, offline and fail-closed. It pins the exact
  contract key set and its own sha256 in the contract (tests additionally pin
  the contract sha256); the fixture sha256 and byte count are pinned both in
  the contract and as literals in the checker (check
  `fixture_pin_authority`). It verifies the 887392-byte synthetic fixture sha256
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`,
  its byte count and its `inkling2_loop` arch string, then runs the same
  CPU-only `llama-debug` command with `LLAMA_RORET`/`LLAMA_RORET_TRACE`
  unset and with both set to `1`. `LLAMA_DUMP_TENSORS_DIR` gives each phase a
  private dump directory; the tensor filter is
  `^ffn_moe_(logits|probs|topk|weights|out)-[0-9]+$`. It writes a NO_GO
  sentinel before checks and the final verdict atomically. The only required
  ON enable needle is `RORET union-resident MoE bookkeeping enabled`;
  `LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache` is an unconditional
  build marker counted in `info` (measured off=2, on=2), never a gating needle.
- Honest measured result on the real binary: checker exit 0, verdict **PASS**,
  31/31 checks true, empty blocker. Both phases exited 0. OFF and ON each
  wrote 40 manifest rows, 20 names (5 kinds x il=0..3), slots `{0,1,2,3}`
  and exactly 2 occurrences per name (`__0`, `__1`). The ON trace had 48
  `RORET_BIND set` rows: `key_il=1` at passes 0, 1 and 2 with 12 rows each,
  plus 12 `key_il=-1 pass=-1` sentinel rows; OFF had 0 bind rows. Core slots
  `{1,2,3}` were derived from the frozen fixture constants
  (`N_LAYER=2, CORE_START=1, CORE_END=2, N_PASSES=3, GATE_RANK=8`), giving the
  recorded pass-to-slot map `{0:1, 1:2, 2:3}`. Raw dump parity was 40 files,
  30080 bytes, 0 differing bytes; stdout was byte-identical at 20516 bytes.
  These values match the independent feasibility probe with no deviation.
- The pass set is read from the ON trace and the core slot set from the same
  ON manifest, then compared through the derived map. The checker also extracts
  each bind row's `slot_map=` pointer and requires the per-pass pointer sets to
  be equally sized, nonzero and pairwise disjoint (two pointers per pass here,
  one per graph eval); a shared pointer across passes fails closed. This is
  observed slot-map separation, not a tautology from the frozen map. Raw
  addresses are not recorded (ASLR varies them); the summary stores only
  per-pass pointer counts and the disjointness verdict, and two consecutive
  real runs produced byte-identical summaries. It remains set-level
  co-occurrence, not a per-event pointer linkage; the occurrence
  index encodes graph-eval count, not pass, and is never used as a pass
  identifier. (Superseded by M42 below for the pointer-identity scope only:
  per-event binding-to-dumped-slot linkage is now gated; the eval-index rule
  and the disjoint-slot-map evidence still stand.)
- Tamper coverage in isolated temp trees: mutated manifest name, mutated
  slot, wrong occurrence count, malformed manifest row, missing pass, bind
  rows OFF, enable marker flip in both directions, byte flip in a `.bin`,
  stdout mismatch, nonzero engine exit, wrong fixture sha256, wrong fixture
  byte count, checker hash drift, extra/missing contract key, malformed
  contract JSON, missing CLI, stale PASS overwrite, sentinel-before-engine
  check and atomic-write-failure exit 1. Also: a shared `slot_map` pointer
  across all three passes is NO_GO, and a forged fixture plus matching contract
  sha/byte pin still fails `fixture_pin_authority`. No tamper test touches the
  real tree.
- Local tests this pass: new suite 27 passed; full `tools/audit` suite 504
  passed (477 baseline plus 27 new) with
  `PYTHONPATH=../loopkit/src:gguf-py`; loopkit fast suite 41 passed with
  `PYTHONPATH=../loopkit/src`; `git diff --check` clean; M39 evidence-chain
  checker still PASS 94/94; all
  five new files have 0 non-ASCII bytes. No `src/`, `common/`, `ggml/`,
  CMake, `DECISIONS.tsv` or sibling-repo edit, no network call, no Kaggle
  kernel and no real weights.
- Scope: CPU-only synthetic runtime replay, disjoint per-pass slot-map pointer
  sets and set-level pass/slot co-occurrence. No GPU, no per-event
  binding-to-tensor linkage, no tensor-value reference and no timing,
  throughput or wall-clock win. NO SPEED CLAIM;
  CycleAudit `claim<=ci_low`, `repeats>=20`, `loops=3` remains mandatory for
  any future number. (Superseded by M42 below for the per-event linkage scope.)

## M42 re-verification 2026-11-28 (branch main, HEAD bbbd992 + M42 worktree)

- M41 left one explicit gap: set-level pass/slot co-occurrence and disjoint
  slot-map pointers, but no per-event proof that a particular `RORET_BIND set`
  row points at a particular tensor. M42 closes the observable part of that gap
  on the CPU synthetic fixture.
- Only source edit: `common/debug.cpp` adds a default-off `LLAMA_DUMP_TENSOR_PTR`
  flag that appends a separate `manifest_ptr.tsv` (`filename<TAB>ptr`) alongside
  the existing dump. The 12-field `manifest.tsv` schema is unchanged. With the
  flag unset, no pointer manifest is written and M41 behavior is unchanged.
- New files under `tools/audit/`: `BIND_TENSOR_LINK.json` (frozen contract),
  `check_bind_tensor_link.py` (stdlib-only checker), `test_bind_tensor_link.py`,
  `BIND_TENSOR_LINK.md`, and generated `M42_BIND_TENSOR_LINK_SUMMARY.json`.
  The tensor filter is widened to include `ffn_moe_slots-<il>`; all other CLI
  flags match M41, including `cpu_share=1.0`.
- Honest measured result on the real binary: checker exit 0, verdict **PASS**,
  45/45 checks true, empty blocker. Both engine phases exited 0. The ON trace
  had 48 `RORET_BIND set` rows (12 each for passes 0/1/2 at `key_il=1`, plus 12
  `key_il=-1 pass=-1` sentinels) and the ON pointer manifest had 48 rows;
  OFF had 0 bind rows and 0 pointer rows. Eight `ffn_moe_slots` dumps were
  observed (slots 0..3, two occurrences each): six matched a `key_il=1` set
  lookup for pass 0/1/2 at slots 1/2/3 (two occurrences each, one per graph
  eval), and two matched the sentinel at slots-0. Per pass the checker requires
  exactly one distinct matched pointer, two matched set rows, and the basename
  set equal to the frozen pass-to-slot slot; swapped `il`, a shared pointer
  across passes, a wrong `key_il`, or an unbound slot pointer fail closed.
- The remaining 30 non-sentinel set rows have no evaluated slot dump in this
  CPU build and are recorded as `unmatched_set_rows_on=30`; they are NOT
  claimed as linked. Raw bytes: 48 `.bin` files / 30720 bytes compared, 0
  differing; stdout 23892 bytes, 0 differing. Two consecutive real runs gave
  byte-identical summaries; no raw pointer or address is written to the
  summary (ASLR varies them).
- Fail-closed controls verified in isolated temp trees: mutated/missing/duplicate
  pointer row, pointer bound to the wrong `il`, pointer shared across passes,
  missing pass, bind rows in OFF, pointer manifest in OFF, enable-marker flip
  in both directions, byte flip in a `.bin`, stdout mismatch, nonzero engine
  exit, wrong fixture sha256/bytes, checker hash drift, contract key drift,
  malformed contract JSON, missing CLI, stale PASS overwrite,
  sentinel-before-engine check and atomic-write-failure exit 1. No tamper test
  touches the real tree.
- Default-off regression: rerunning M41's checker with the newly built binary
  gave 31/31 PASS and a byte-identical M41 summary with no pointer manifest.
- Local tests this pass: new suite 28 passed; full `tools/audit` suite 532
  passed (504 prior plus 28 new) with `PYTHONPATH=../loopkit/src:gguf-py`;
  loopkit fast suite 40 passed, 1 deselected with `PYTHONPATH=src`;
  `git diff --check` clean; all new lines ASCII-only with no literal tabs in
  source. Only `common/debug.cpp` was edited outside `tools/audit/`.
- Scope: per-event pointer identity between the registered RORET binding and
  the dumped `ffn_moe_slots-<il>` tensor on the fixed-seed synthetic 3-pass
  fixture, including the pass-to-slot map. NOT tensor-value correctness, NOT
  tensor-value reference, no copy-callback linkage (the CPU log emits zero
  `resolve`/`hit` rows), no real weights, no GPU, no timing, throughput or
  wall-clock win. NO SPEED CLAIM; CycleAudit `claim<=ci_low`, `repeats>=20`,
  `loops=3` remains mandatory for any future number.

## M43 runtime wiring evidence chain gate (2026-11-28, branch main)

- Gap closed: M39's evidence-chain gate pins M29-M38 summaries only. The three
  runtime wiring gates M40 (CPU 3-pass binding), M41 (3-pass replay and
  pass-to-slot map) and M42 (per-event binding-to-dumped-slot pointer identity),
  plus M39 itself, had no independent byte-level integrity gate. M43 adds that
  gate without editing any M39-M42 artifact.
- Five new ASCII-only files under `tools/audit/` with sha256:
  `RUNTIME_EVIDENCE_CHAIN.json` fe39a5d642af5e13d6271ff16d8994db845800e413fd2a6b76d13a383af7cec2,
  `check_runtime_evidence_chain.py` 87aac891e0a3267e0bb5e7aebef52c67521fe8b85f609fdecce9b03425102436,
  `test_runtime_evidence_chain.py` 0ecd2b3e46b7fe278bf7af001387def7fdeecfa94d764093a4faeb98680b34c6,
  `RUNTIME_EVIDENCE_CHAIN.md` 70794bebbf08567fab12bfb7b7cd3e0a2123839d28b87a022881038aa3c7f04b,
  and generated `M43_RUNTIME_EVIDENCE_CHAIN_SUMMARY.json` 44aaff0822561677d97afa3f392393c7066bc36f1ac9fb68e91a27d03f0cc76b.
- Measured checker run: `python3 tools/audit/check_runtime_evidence_chain.py
  --root .` exit 0, verdict PASS, blocker empty, check_counts 52/52 passed and
  0 failed; note "runtime wiring evidence integrity only, no timing";
  speed_claim null, no_weights true, no_engine_run true. The generated summary
  sha256 is 44aaff0822561677d97afa3f392393c7066bc36f1ac9fb68e91a27d03f0cc76b,
  matching the pin recorded above.
- Commands used from the fork root, with no network and no gh write:
  `python3 tools/audit/check_runtime_evidence_chain.py --root .`;
  `python3 -m pytest -q tools/audit/test_runtime_evidence_chain.py`;
  `PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest -q tools/audit`;
  `PYTHONPATH=../loopkit/src python3 -m pytest -q ../loopkit/tests -m 'not kaggle'`;
  `python3 tools/audit/check_evidence_chain.py --root .`;
  `git diff --check` and `sha256sum` on the five new files.
- Test results: new suite 64 passed; full `tools/audit` 596 passed (532 baseline
  plus 64 new); loopkit fast suite 40 passed, 1 deselected; M39 gate exit 0,
  verdict PASS with 94/94 checks and its summary bytes unchanged (sha256
  4f6361b562a82ad2942f32b1b8f07bb5bf22417701f1ca940c0a1b837ae78ba3). The full
  suite regenerated `M33_QUANT_LAYOUT_SUMMARY.json` with a machine-specific path
  line; it was restored with `git checkout --
  tools/audit/M33_QUANT_LAYOUT_SUMMARY.json`, so the final diff has only the five
  new files plus this section.
- Two real weaknesses were fixed before the pins were frozen. A negative
  control during implementation showed the first draft of
  `check_frozen_contract` did not deep-compare the `contracts` object, so a
  tampered pin could leave `frozen_contract` true; it now compares the whole
  object. An independent Hy4 Flash adversarial review (0 CRITICAL/HIGH, 5 LOW)
  showed the DECISIONS row check used substring matching, so a row reading
  `land M410-CANCELLED` still satisfied M41; it now requires an exact
  `land M<xx>` value or that token followed by a space, and a dedicated negative
  control covers the bypass. The other four LOW notes are inherited from the
  M39 pattern and are recorded honestly: a fabricated BUILD_STATUS heading can
  satisfy the presence check, a coordinated checker-plus-contract edit is
  detected only by the pytest literal pins (so CI must run the suite, not just
  the checker), an unlink failure combined with two write failures can leave a
  stale PASS file while still exiting 1, and the generated summary cannot
  self-pin. The checker, contract, test and summary hashes above were recomputed
  after the fixes. All tamper tests use isolated temp trees; no test runs the
  checker on the real tree.
- Scope: evidence-chain integrity only. No engine run by this gate, no graph
  execution, no weights, no tensor or logits parity, no pointer re-derivation
  and no timing. NO SPEED CLAIM; CycleAudit `claim<=ci_low`, `repeats>=20`,
  `loops=3` remains mandatory for any future number. A legitimate future pin
  change must be explained openly here, not made silently to force PASS.

## M44 fork/upstream `llama-cli` GGUF byte-compatibility gate (2026-11-28, branch `main`)

- Gap closed: `INKLING_INTEGRATION.md` section 3 item (d), "a GGUF produced by the
  fork is byte-compatible with upstream `llama-cli`", was the last unchecked
  cheap-verification item. M29-M43 verified contracts, evidence chains and
  runtime bindings inside the fork, but never ran an unmodified upstream build
  on the same frozen fixture.
- Five new ASCII-only files under `tools/audit/` with sha256:
  `UPSTREAM_CLI_COMPAT.json` 5fb7e01e63c526844231daa6905ea7e78c9f5f37d85fbc595a0286bbae6b2ae6,
  `check_upstream_cli_compat.py` 08c9902d011024317debe5ccf296b08c7fa0e1ebfbdc8854b665b340c03d2289,
  `test_upstream_cli_compat.py` 8152a3690391a8603f86fa8d5086e5ab3afb4c30913328d271e82398991b1a2b,
  `UPSTREAM_CLI_COMPAT.md` 403f591fba0e1db10c3e6ade2476fdcde9499627d1f4171c58b13cca79cc2ec2,
  and generated `M44_UPSTREAM_CLI_COMPAT_SUMMARY.json` 38ab9c92f7c378e7f4ed23d834b2bd60367207b754e80f37e6f359c8a5ce8122.
- Method: two distinct local `GGML_CUDA=OFF` builds, upstream base
  `bc641b516fe01f346b9f2bf70d69fb7afa3ac222` and fork HEAD
  `5039437e312ecc0bf37674e24b89c7e2e8ee7a99` (fork also
  `LLAMA_RORET_CPU_TEST=ON`), both run on the frozen synthetic `inkling2_loop`
  fixture sha256 `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`
  with `-t 1 -n 16 -s 0 --temp 0 -p hello --no-perf --no-show-timings
  --no-warmup -st`. Seven `LLAMA_*` env keys are stripped for both runs. Both
  CLI files are hash-pinned in the contract and required to have different
  hashes. A positive control invokes each binary with `LLAMA_RORET=1 -cmoe
  --moe-cache-mib 16 --verbose` and requires the marker on the fork stderr and
  its absence on the upstream stderr; control exit codes are not gated because
  the CPU-test `-cmoe` path can abort during teardown after printing.
- The `build : <build-info>` banner line is removed from both stdout streams
  before byte comparison; it records build provenance (build number and
  commit), not GGUF or graph output, and necessarily differs between the two
  builds. All other stdout bytes, including prompt echo and generated tokens,
  are compared. Stderr is used only for the enable-marker checks.
- Real checker run: exit 0, verdict PASS, 26/26 checks, blockers empty,
  `stdout_differing_bytes` 0, 918 stdout bytes on each side, both main exits 0,
  both main enable markers absent, positive control present/absent as required.
  Raw stdout differed only in the banner line; normalized stdout was identical.
  Generated-token payload `fffa3fa3fa3fa397` was present in both stdout streams.
- Real negative control with the two CLI paths swapped exited 1 with verdict
  NO_GO and four failing checks (both CLI hash pins and both control marker
  checks), writing to a separate output path; the PASS summary was unchanged.
- Two Hy4 Flash subagents (implementation plus independent adversarial review,
  two passes) found and fixed vacuous-PASS paths, nondeterministic stdout from
  `--no-perf` alone, a vacuous marker check, a stale-PASS path on failed initial
  sentinel write, and weak fixture pinning. Local tests after fixes: new suite
  60 passed, full `tools/audit` 656 passed, loopkit fast suite 40 passed with 1
  deselected, `git diff --check` clean, all five new files ASCII-only with no
  literal tabs. The full audit regenerated `M33_QUANT_LAYOUT_SUMMARY.json` with
  a machine-specific path; it was restored with `git checkout --`, so the final
  diff has only the five new files plus this section.
- Scope: byte compatibility of one frozen synthetic GGUF on two CPU builds with
  union routing compiled in but default-off. No real weights, no GPU, no tensor
  or logits reference, no timing, throughput or wall-clock result. NO SPEED
  CLAIM; CycleAudit `claim<=ci_low`, `repeats>=20`, `loops=3` remains mandatory
  for any future number. No issue or PR filed; issues #1-#9 remain closed.

## M45 - CPU-only full-topology 3-pass union-wiring gate (2026-11-28, branch main)

- Closes the last locally reachable wiring-shaped gap: M40-M42 exercised the union-resident MoE hook only on a toy fixture (2 physical layers, 1 core layer, rank 8, 4 cache slots). M45 runs the real Inkling2 loop structure on a deterministic synthetic GGUF: 24 physical layers, logical core 6..17, 3 passes, rank-64 controllers, `attn_res.block_size=6` and 48 cache slots. No real weights and no Kaggle kernel.
- Six new ASCII-only, tab-free files under `tools/audit/` with sha256: generator `make_fulltopo_gguf.py` fb375e727f6dc866321cddf743ff33f62a66e1c18f0f6510a2caadeb0cef3c6a, contract `FULLTOPO_BINDING.json` eae32687a5a68c35165158d650deb42f5385ab7dbc5a54384f0bfd2c7850b50c, checker `check_fulltopo_binding.py` 81d572430c3ba35b7a63a69c8f099b4f9facdcbed2baf0e492dbccb3a9b696cd, tests `test_fulltopo_binding.py` 4b846258aba9ab28103f3d97d9bb354a87481f2038e0c8e4414350c05ba3e011, scope doc `FULLTOPO_BINDING.md` 097dcd2e23fcb89f256fa21b07dda161fd764090809e15af0468b91c579f6fd6, generated `M45_FULLTOPO_BINDING_SUMMARY.json` e5b16a03b32334e776c590eb19d5aa40e11050a01921c1f6af1a71b48ad5f5f3.
- Fixture: 11941248 bytes, sha256 `3ccf2fb4c6ef4fab60464d2a35a2ea42b24186db3e8b12cd68881f4fa873dff7`, 708 tensors, arch `inkling2_loop`, seed 0; the generator reproduces it byte-for-byte. The first probe used the simplified block size 1; before pins were frozen this was corrected to the upstream value 6, the fixture regenerated, and the real binary re-probed with identical binding coverage. The pin change is recorded here openly, not silently to force PASS.
- Commands from the fork root: `cmake -S . -B /tmp/opencode/m44-probe -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DLLAMA_RORET_CPU_TEST=ON -DCMAKE_BUILD_TYPE=Release`, `cmake --build /tmp/opencode/m44-probe --target llama-debug -j4`, `PYTHONPATH=gguf-py python3 tools/audit/make_fulltopo_gguf.py --out /tmp/opencode/fulltopo-final.gguf`, `python3 tools/audit/check_fulltopo_binding.py --root . --cli /tmp/opencode/m44-probe/bin/llama-debug --gguf /tmp/opencode/fulltopo-final.gguf --out-dir /tmp/opencode/m44-real-final2`, `python3 -m pytest -q tools/audit/test_fulltopo_binding.py`, `PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest -q tools/audit`, `PYTHONPATH=src python3 -m pytest -q ../loopkit/tests -m 'not kaggle'`, both evidence-chain checkers and `git diff --check`.
- Measured result on the real CPU-only binary: checker exit 0, verdict **PASS**, blocker empty, 35/35 checks true. OFF and ON exits 0/0. ON `RORET_BIND set`: 576 rows = 144 sentinel rows plus 144 rows for each pass 0/1/2; 36 real `(key_il, pass)` pairs covering every core layer 6..17 at every pass, no missing and no unexpected pair. Dumps: 480 manifest rows, 240 names (5 kinds x 48 slots), exactly 2 occurrences each, physical slots 0..47; 480 `.bin` files, 360960 bytes compared, 0 differing; text 245927 bytes, 0 differing; normalized stdout 0 differing with 0 dropped lines in both phases. Per-pass `slot_map` pointer sets are nonzero, equally sized (24 each) and pairwise disjoint. `RORET_PASS` rows are 0 on this CPU fallback and are informational only, never gated. No timing was measured or recorded.
- Fail-closed design: NO_GO sentinel is written before the engine runs and before output-directory creation; the final verdict uses atomic `os.replace`, so a killed run or write failure cannot leave a stale PASS. Checker sha256 is pinned only in the contract with test literals as a second witness; fixture sha256/bytes are also literal checker pins; exact top-level contract key set; static no-network self-check; anchored end-of-line needle matching; dump-cleanup failure now fails the manifest checks. Tamper tests use isolated temp trees only.
- Independent Hy4 Flash adversarial review: 0 CRITICAL/HIGH. Fix applied from the MEDIUM findings: this pin record and the corrected stdout byte count in the scope doc. LOW fixes applied: scope wording now says disjoint pass-indexed slot-map sets rather than claiming per-slot binding; enable-needle matching anchored to end of line; sentinel moved before `out_dir.mkdir`; dump-cleanup errors surfaced; the implied-by-`on_slot_set` subset check is disclosed in `info`; the stale `/tmp` fixture sidecar was regenerated.
- Local tests this pass: new suite 23 passed; full `tools/audit` suite 619 passed (the M33 summary path line regenerated by the full suite was restored with `git checkout --`); loopkit fast suite 40 passed, 1 deselected; M39 wiring evidence-chain gate PASS 94/94 and M43 runtime evidence-chain gate PASS 52/52 after the new row and heading were appended; ctest roret 10/10, pibs 2/2, odsl 1/1; `git diff --check` clean; all six new files have 0 non-ASCII bytes and 0 tabs.
- Scope: set-level binding, topology and byte-parity evidence on a fixed-seed synthetic fixture with real loop structure. NOT per-event binding-to-tensor pointer linkage, NOT reference tensor-value correctness, NOT real-weight parity, NOT GPU copy-callback hit/upload traffic, and NOT timing, throughput or wall-clock gain. NO SPEED CLAIM; the eligible M24 Kaggle CycleAudit placed the whole 95% interval below 1.0. Any future number still needs `claim<=ci_low`, `repeats>=20`, `loops=3`.


## M46 - coordinated section-3 integration-doc closure (2026-11-28, branch main)

- Closes issue #10: loopkit/docs/INKLING_INTEGRATION.md section 3 item (d) is marked RESOLVED with M44 evidence; items (b) real-weight conversion parity and (e) tokenizer/chat-template remain open, and the other section-3 items are unchanged.
- Loopkit doc sha256 054188036cb75d373b3a1740051f873e3e5646f5cb8c47810426a7f2f930e4b0 -> 9b0993f68bd2169468c4beedb1d54522e7ce9bd3f1c3b1c94d4033120aa0affc; only sections 3 and 5 changed, exactly one literal [?] remains (the label legend).
- M44 evidence: tools/audit/UPSTREAM_CLI_COMPAT.json sha256 5fb7e01e63c526844231daa6905ea7e78c9f5f37d85fbc595a0286bbae6b2ae6 and tools/audit/M44_UPSTREAM_CLI_COMPAT_SUMMARY.json sha256 38ab9c92f7c378e7f4ed23d834b2bd60367207b754e80f37e6f359c8a5ce8122; 26/26 checks PASS, 0 differing bytes at 918 bytes/side, synthetic CPU fixture and compile/stdout-byte scope only.
- Refreshed the current doc-hash pins in check_gate_semantics.py, check_quant_layout.py, check_outlier_premise.py, check_integration_doc_closure.py, GATE_SEMANTICS.json, QUANT_LAYOUT.json, OUTLIER_PREMISE.json, INTEGRATION_DOC_CLOSURE.json, GATE_SEMANTICS.md, QUANT_LAYOUT.md and INTEGRATION_DOC_CLOSURE.md; OUTLIER_PREMISE.json digest 46c7260707371d83a7f662f4c7fad1420736b6f38469c512b835fb74956d6917 -> 0d73a0c2de1e0e15632f55cc0024cdf4ce1b57d323671e38a6b23db7424533a6 and that new digest is pinned in check_outlier_premise.py. Historical hash records in this file and DECISIONS.tsv were not rewritten.
- M38 closure gate extended with M44 in EXPECTED_MILESTONES, EVIDENCE, the section3_item_d needle entry and PROVENANCE; DESCRIPTION mentions M44; 8 resolved needles (was 7), 10 needle names (was 9), 21 checks (was 19). INTEGRATION_DOC_CLOSURE.json pins m44_summary_sha256 and check_integration_doc_closure.py adds the dedicated gated check m44_evidence_pins: it byte-pins the M44 summary and validates verdict PASS, speed_claim null, blockers empty, checks_passed == checks_total == 26, checks_failed 0, stdout_differing_bytes 0 and no_weights true. Its negative-control test rewrites the fixture summary to 0/26 checks, 918 differing bytes and a non-empty blockers list while keeping verdict PASS, and must get NO_GO. Checker sha256 da7e4a5e67ffe926a9405ad28f05711306b297dce7498328328d10ef7cb49d46 is pinned in INTEGRATION_DOC_CLOSURE.json and test_integration_doc_closure.py.
- Machine-specific summaries fixed: check_quant_layout.py and check_outlier_premise.py now emit the location-independent INFO line 'loopkit root ../loopkit' instead of an absolute checkout path. Re-running the two checkers from a copied checkout at a different path produced byte-identical M33/M35 summaries. All gated sha256 checks and verdicts are unchanged; only the INFO line changed.
- Regenerated from the fork root, each exit 0 verdict PASS blocker empty speed_claim null: M30_GATE_SEMANTICS_SUMMARY.json sha256 b207299febeba25d959a1a804b0133b1c436aa5c2f9e44626c445dfe25375e67 (unchanged); M33_QUANT_LAYOUT_SUMMARY.json sha256 1b44752269ccfc3f26717ad9874a39e0b1c3f0511ca9da13a4e42b1b3ca5bd49; M35_OUTLIER_PREMISE_SUMMARY.json sha256 19416074c6979e612132758a3830da3cbb3d3ea228fc178639748460e5f45c55; M38_INTEGRATION_DOC_CLOSURE_SUMMARY.json sha256 01b33a65a7e437d44fec8b9fb04be0c0ffcbb7734cc930e0b04f355da1b0e2ce.
- M39 chain gate refreshed: EVIDENCE_CHAIN.json sha256 39a40b9b60913e566423cd4329d1a4f892e20b17365141ed14a728e415f0ff89, check_evidence_chain.py sha256 5fd2d7a2e6976e4971a2ea84b9b68ba5edd240ff6b9969b71f555cc27cb1e63a, regenerated M39_EVIDENCE_CHAIN_SUMMARY.json sha256 a8efa53efc0f2223c67b17a8db260fba7140c01e298760011906f3f461da09be; exit 0 PASS 94/94.
- M43 chain gate refreshed: RUNTIME_EVIDENCE_CHAIN.json sha256 5a6cee68adfce706fd796c06e32bfaede59fc35609f31ead3725e020a8a2fced, check_runtime_evidence_chain.py sha256 bdb4be142fdf8eae9b5a9a7d2a99dac3a8dccf51ef0630444ae0baae0942f977, regenerated M43_RUNTIME_EVIDENCE_CHAIN_SUMMARY.json sha256 e0600a2df69aa5080c7c49905e4cdd96bd3a6dd00db5e373b3466a50d1368510; exit 0 PASS 52/52.
- Test counts: targeted six audit test files 245 passed; full tools/audit suite 680 passed; loopkit fast suite 40 passed 1 deselected; gate checks M38 21/21, M39 94/94, M43 52/52. M39 and M43 gates were re-run after this section and the DECISIONS row were appended and still PASS.
- Independent Hy4 Flash adversarial review: 0 CRITICAL/HIGH. Both MEDIUM findings are fixed (the fail-closed M44 evidence pin and the machine-specific summary info lines); both LOW prose findings are fixed (INTEGRATION_DOC_CLOSURE.md scope says M29-M37 plus M44 and this section no longer claims added lines are tab-free).
- Verification: all six checkers exit 0 PASS in dependency order; git diff --check clean in both repos; added lines are ASCII-only with tabs used only as DECISIONS.tsv field separators; no src/, CMake, engine or network/Kaggle changes; no commit, push, issue or PR.
- Scope: synthetic CPU byte compatibility only. No real weights, no GPU, no tensor or logits parity, no timing, no throughput or wall-clock result. NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory for any future number. Items (b) and (e) remain open.

## M47 - offline tokenizer and chat-template contract gate (2026-11-28, branch main)

- Closes the last cheap weight-free section-3 item (e) of `loopkit/docs/INKLING_INTEGRATION.md`, "that tokenizer / chat template (`tml-renderers`) matches", as a frozen-contract gate. No edit was made to that doc: M30/M33/M35/M38 hash-pin its bytes, so the marker update must be a coordinated two-repo milestone. Item (b) real-weight conversion parity remains NO_GO_BLOCKED.
- Five new ASCII-only, tab-free files under `tools/audit/` with sha256: contract `TOKENIZER_CHAT_TEMPLATE.json` 998c58d244e66d5b2dd8b09fcd33ebee9e187b38ea95adf66779463186b2d9e7, checker `check_tokenizer_chat_template.py` e187bb9ac6914987b40bee59452633a585984dc65d0aecb4ef3766ab79ee0e3f, tests `test_tokenizer_chat_template.py` a26ae3ffcac9c7c84da618cddfa816023c02193fa7c256e8ea0224e9ad1521be, scope doc `TOKENIZER_CHAT_TEMPLATE.md` 297b036834e109958c9612759c42730d663cb5346b41165d1c6da9b0d4d2664b, generated `M47_TOKENIZER_CHAT_TEMPLATE_SUMMARY.json` 3fdc5c088c4a49008bee34f7cd386b021dd89aa3e01bbdd2d35d6f06a6dc6f20.
- Out-of-band primary-source evidence re-measured locally: HF `thinkingmachines/Inkling-2-Mini` commit c9d028302cd0f035fdf8e7e91f089a4ce94d2195; `chat_template.jinja` 6868 bytes sha256 53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc, `tokenizer_config.json` 13235 bytes sha256 bf09342d285139fd37ce10c70878af1d52cb390e9901e10248654293a80ee636 (no inline `chat_template` key), `special_tokens_map.json` 529 bytes sha256 c16f44b3594563d82743ddb0417a2be1bfe3e2000dfad2da4b8f32554593e1a0, `tokenizer.json` 26178678 bytes sha256 7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6 (BPE, 199998 base vocab entries). The bounded GGUF probe of `Inkling-2-Mini-UD-IQ1_S.gguf` bytes 0-16777215 has sha256 87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76, GGUF v3, 744 tensors, 63 KVs; the out-of-band bounded probe measured byte equality for `tokenizer.chat_template` (6868 bytes, same sha256). The offline checker verifies only that the frozen digests and lengths are internally consistent and source-pinned and cannot observe live GGUF drift. Long token/token-type/merge arrays are not claimed to be fully validated.
- Real checker run from the fork root: exit 0, verdict PASS, blocker empty, 26/26 checks, speed_claim null, no_weights true, no_engine_run true. Frozen facts: eos `<|content_model_end_sampling|>` id 200006 equals eot, pad `<|endoftext|>` id 199999, add_bos/add_eos/clean_spaces false, padded vocab 201024 > unpadded 200058 > eos > pad; converter sha256 aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7 and all 13 needles verified live.
- Fail-closed design: NO_GO sentinel written before any check, atomic `os.replace` final write, exact contract and summary key sets, checker self-hash pinned in the contract and a second literal pin in the tests, static no-network and no-cache-read self-checks, negative controls in isolated temp trees (tampered template bytes/digest, wrong token id and GGUF KV, missing converter needle, live source and checker hash drift, malformed or missing contract, stale PASS overwrite, sentinel-before-checks and atomic-write failure). The real-tree test restores the summary bytes it writes. The template checks are named `frozen_template_digest_consistency` and `frozen_template_length_consistency`, the summary `info` line says `template parity check FAILED` on parity failure instead of claiming byte identity, and `summary_shape` checks structure only so it stays true on NO_GO without adding a blocker. The two primary-source byte tests assert the contract-embedded sha256 and length against the pinned digests with no `/tmp` reads.
- Verification: new suite 43 passed; full `PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest -q tools/audit` 723 passed; `git diff --check` clean. Only the five new files plus this section are present in the diff; no src/, CMake, gguf-py or loopkit file was changed, no commit or push.
- Scope: out-of-band measured byte equality of the chat template and offline-verified consistency of bounded tokenizer metadata only. The offline checker cannot observe live GGUF drift. No weights, no engine run, no tokenization golden test on real weights, no timing or throughput. NO SPEED CLAIM; CycleAudit `claim<=ci_low repeats>=20 loops=3` remains mandatory for any future number. The section-3 marker update and item (b) both stay open.
## M48 - coordinated section-3 item (e) closure (2026-11-28, branch main)

- Closes the last unchecked cheap item in `loopkit/docs/INKLING_INTEGRATION.md` section 3: item (e), "that tokenizer / chat template (`tml-renderers`) matches", is marked RESOLVED by M47 as a frozen tokenizer/chat-template contract gate only. Item (b) real-weight conversion parity remains NO_GO_BLOCKED; items (a), (c), (d) and (f) are unchanged.
- Loopkit doc sha256 9b0993f68bd2169468c4beedb1d54522e7ce9bd3f1c3b1c94d4033120aa0affc -> 95f7595401bf534fa7e3d007423979e7246151678eb67655cd35d9f3660d07d3. Only the section-3 marker/paragraph and the section-5 notes changed; exactly one literal `[?]` remains (the label legend); added lines are ASCII-only and no new non-ASCII bytes were introduced.
- M47 evidence: contract `tools/audit/TOKENIZER_CHAT_TEMPLATE.json` sha256 998c58d244e66d5b2dd8b09fcd33ebee9e187b38ea95adf66779463186b2d9e7 and summary `M47_TOKENIZER_CHAT_TEMPLATE_SUMMARY.json` sha256 3fdc5c088c4a49008bee34f7cd386b021dd89aa3e01bbdd2d35d6f06a6dc6f20; 26/26 checks PASS, blocker empty, speed_claim null. The out-of-band bounded probe measured byte equality of `tokenizer.chat_template` (6868 bytes, sha256 53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc) with HF `chat_template.jinja`; the offline checker verifies frozen-digest/length consistency only and cannot observe live GGUF drift.
- Refreshed every current doc-hash pin in check_gate_semantics.py, check_quant_layout.py, check_outlier_premise.py, check_integration_doc_closure.py, GATE_SEMANTICS.json, GATE_SEMANTICS.md, QUANT_LAYOUT.json, QUANT_LAYOUT.md, OUTLIER_PREMISE.json, INTEGRATION_DOC_CLOSURE.json and INTEGRATION_DOC_CLOSURE.md (New doc sha256 line only). Historical hash records in this file, DECISIONS.tsv and the INTEGRATION_DOC_CLOSURE.md "Old doc sha256" line were left intact. OUTLIER_PREMISE.json digest 0d73a0c2de1e0e15632f55cc0024cdf4ce1b57d323671e38a6b23db7424533a6 -> f2dd8abb5592920602e6a8c15cfdb0dbcef94e3506285a9abc8a84af810c3935, and that new digest is pinned in check_outlier_premise.py.
- M38 closure gate extended in lockstep in check_integration_doc_closure.py and INTEGRATION_DOC_CLOSURE.json: M47 added to EXPECTED_MILESTONES, EVIDENCE and provenance milestones; new top-level m47_summary_sha256 and checker constant M47_SUMMARY_SHA256; new gated check m47_evidence_pins byte-pins the M47 summary and validates verdict PASS, empty blocker, 26/26 checks with 0 failed, speed_claim null, no_weights/no_engine_run true and the historical doc_note; new section3_item_e needle group; M46 and M48 closure-note needles; DESCRIPTION mentions M47/M48. Now 9 resolved needles, 11 needle groups and 23 checks. Checker sha256 0db6db1437a0b12896d59388c50bbd96b135fd476e8b666ecdfb259cb24d41a7 is pinned in the contract and as a second literal in the tests; contract sha256 afb2566341b2e95adaf75c8bdd164734f6b191051b227e8ca399dd9495a62cc4 is pinned in the tests. New negative controls tamper the M47 summary fields/bytes, the M47 contract pin and the section-3 item (e) needle; all force NO_GO.
- Regenerated from the fork root in dependency order, each exit 0 verdict PASS blocker empty speed_claim null: M30_GATE_SEMANTICS_SUMMARY.json sha256 a8273ed0cc601b177d2892cea64ab080b34224cc244ee456787ec4e585c3bf4c (23/23); M33_QUANT_LAYOUT_SUMMARY.json sha256 9501a87f9e1d937f613fff9b362d7d1a1dd459af7e0da69ee1ea662248dd6462 (28/28); M35_OUTLIER_PREMISE_SUMMARY.json sha256 97dda997409491e1057234206ca7fb9bf871945d952a8cb56603f95d341d6861 (27/27); M38_INTEGRATION_DOC_CLOSURE_SUMMARY.json sha256 e4cc7af59c18e3b64addb3aa4f9464ec360d837281a1d3c7cc97067214fbb1aa (23/23).
- Evidence-chain pins refreshed: EVIDENCE_CHAIN.json sha256 1e0300c7a8cab53602b565094386712fd9cafa1751d6148e7301af6da6f0d276 and check_evidence_chain.py sha256 ca57106ce641511d05aabe701021307369fe9c36f3c6a0e96b0902b55672e29b; regenerated M39_EVIDENCE_CHAIN_SUMMARY.json sha256 fb7ec46ee0cdd98e5c1bbc78af69bd6aa5a3dbb1c245e2a7de15bacb3cbfd6a6, exit 0 PASS 94/94. RUNTIME_EVIDENCE_CHAIN.json sha256 6cc30cab6acade6c98ffb52c3325cc1e5dca93998c20b0137a910be032b73b56 and check_runtime_evidence_chain.py sha256 4d90a692ebdf250ddac4f3666e506bc3a7de548e7a66527aae80d3182801e1aa; regenerated M43_RUNTIME_EVIDENCE_CHAIN_SUMMARY.json sha256 d6023ed61144bc84fb5eabbaf46cd5d4475cc139ac8ba1626df2b92e7271e637, exit 0 PASS 52/52. Both chain test literal pins were updated.
- Verification commands from the fork root: `PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest -q tools/audit` -> 727 passed; `PYTHONPATH=src python3 -m pytest -q tests -m 'not kaggle'` from the loopkit root -> 40 passed, 1 deselected; the six regenerated checkers -> PASS with counts 23/23, 28/28, 27/27, 23/23, 94/94 and 52/52; both chains re-run after this section and the DECISIONS row were appended and still PASS; `git diff --check` clean in both repos; added lines ASCII-only with tabs only as DECISIONS.tsv field separators; no src/, CMake, gguf-py or engine files changed; no commit, push, issue or PR.
- Post-review fail-open fix (2026-11-28): independent adversarial review found that `check_m44_evidence_pins` and `check_m47_evidence_pins` byte-pinned only the summary files, so tampering the live M44/M47 contract could leave the gate PASS; both checks now hash the live `tools/audit/UPSTREAM_CLI_COMPAT.json` (sha256 5fb7e01e63c526844231daa6905ea7e78c9f5f37d85fbc595a0286bbae6b2ae6, pinned to the checker constant because the M44 summary carries no `contract_sha256` field) and `tools/audit/TOKENIZER_CHAT_TEMPLATE.json` (sha256 998c58d244e66d5b2dd8b09fcd33ebee9e187b38ea95adf66779463186b2d9e7, also required to equal the M47 summary's `contract_sha256`), with new isolated-tree negative controls that tamper each contract and require CLI exit 1 with a seeded PASS summary overwritten to NO_GO. Cascaded shas: closure checker 838d0c34eba0de7e0269d1bf905e4b02b3f07caa27f709073788c36a855c2c4d, closure contract 57cff9649902f9385edaf83efe19a075e4f2652102425e5e1be507bbd6e299e1, M38 summary e9c71e3c375e925204fa95fe2538a329c648489dd25f569453105044697c1f99, EVIDENCE_CHAIN.json 890313d7df78900c1369401752241b2d959c98b6fe3d0906ddcc2b794a1ab68d and checker 9de42b929607ac0369c6049c565cf3df38ddecea5d591389be883e198922bdef, M39 summary c83073b57ab412af541bf622c6fe69e60e015c7a723856b175f23ef3be44d8fe, RUNTIME_EVIDENCE_CHAIN.json 6196de81800bfddd859aed241100138f910e84786cfdf2e74bcda3b885df66fc and checker b6df6594b0a4dee61d4d8d714de37454cc9ae5310e651f2e1b602fd49e06cfbe, M43 summary 444a8876a161122d2c47344ee2866632d652f6937872f5829efe34fa49703339; fresh verification: all six checkers exit 0 PASS (M30 23/23, M33 28/28, M35 27/27, M38 23/23, M39 94/94, M43 52/52), full tools/audit pytest 729 passed, loopkit fast suite 40 passed 1 deselected, both contract-tamper negative controls fail closed, git diff --check clean in both repos; no DECISIONS.tsv row was added.
- Scope: documentation-contract closure only. No real weights, no engine run, no tokenization golden test, no GPU, no tensor/logits parity and no timing. NO SPEED CLAIM; CycleAudit `claim<=ci_low`, `repeats>=20`, `loops=3` remains mandatory before any future number. Item (b) real-weight conversion parity stays NO_GO_BLOCKED.
