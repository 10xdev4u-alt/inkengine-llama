# PIBS wiring report (M7) - 2026-11-25

Status: **DECISION/INSTRUMENTATION SEAM LANDED; FULL ENGINE REORDER NO_GO**.
Repo: `inkengine-llama`, branch `main`, HEAD `a05f501918ef85c8801a4c817385fd722a0316b0`.
No commit, no push, no branch change. No file outside this repo was modified.

## What the patch does

`tools/pibs/PIBS_WIRING.patch` adds two small src changes:

1. New header `src/llama-pibs.h` (92 lines): a pure, header-only, standard-library-only
   decision and count helper. `gate_enabled(const char *)` parses the opt-in gate with
   the exact-value rule `1`, `on`, `true` (mirrors `src/llama-moe-cache.cpp:449-455`).
   `decide(n_tokens, gate_on)` returns `interleave = gate_on && n_tokens >= 4` plus
   modelled `steps`, `padded_total` and `wasted_total` from the M6 m16 formulas.
   `model(n_tokens, interleave)` exposes both schedules so the batch-1/2 regression
   stays visible. `seq_of()` encodes the global order of `(token, pass)`.
   Asserts encode `M <= batch`, `M != batch*3`, `seq(t,0) < seq(t,1) < seq(t,2)`.
2. `src/models/inkling2-loop.cpp`: includes the helper and `llama-impl.h`, reads
   `INKLING2_PIBS`, and emits one `LLAMA_LOG_DEBUG` line with the planned step count.
   The pass loop (`for (uint32_t pass = 0; pass < m.n_passes; ++pass)`), every
   `build_layer` call, and every tensor are **untouched**.

Because the wiring only computes and logs a plan, gate off and gate on produce
bit-identical graphs by construction: no node is reordered, fused, split or
retargeted. Existing behavior is unchanged byte-for-byte when the gate is off.

## What the patch does NOT do

- It does **not** reorder ggml nodes across ubatches. The standalone PIBS
  `(token, pass)` work-item abstraction is not translated into engine node order.
  SCHEDULER_DESIGN.md section 6 calls this an OPEN DESIGN QUESTION; it is still open.
- It does **not** change the router, argsort, weights, `selected_experts_in`, or any
  F32 tensor. It does not add a `M = batch*3` GEMM.
- It does **not** produce an engine-level off/on F32 comparison. The modelled
  `8 -> 3` steps is a COUNT from the pure helper and the standalone model, not a
  measured kernel launch count in the engine.
- It makes **no speed claim**: no tok/s, no wall-clock, no speedup.

## Reproduced counts (all COUNTS / numeric model results, never speedups)

| batch | scheduler | steps | padded | wasted | efficiency |
|---|---|---|---|---|---|
| 1 | sequential | 1 | 16 | 13 | 18.75% |
| 1 | interleaved (model only, gate never selects it) | 3 | 48 | 45 | 6.25% |
| 2 | sequential | 2 | 32 | 26 | 18.75% |
| 2 | interleaved (model only, gate never selects it) | 3 | 48 | 42 | 12.50% |
| 4 | sequential | 4 | 64 | 52 | 18.75% |
| 4 | interleaved | 3 | 48 | 36 | 25.00% |
| 8 | sequential | 8 | 128 | 104 | 18.75% |
| 8 | interleaved | 3 | 48 | 24 | 50.00% |

- Batch 8: **8 -> 3 steps**, **104 -> 24 wasted tile slots**.
- Batch 1 regression: sequential 13 wasted vs interleaved 45 (the gate forces sequential).
- Batch 2 regression: sequential 26 wasted vs interleaved 42 (the gate forces sequential).
- Breakeven: interleaved first wins at batch **4** (36 vs 52).
- These match `tools/pibs/RESULTS.md`, `RESULTS_EXTENDED.md` and SCHEDULER_DESIGN.md
  section 3. They are arithmetic on M with `kWmmaM = 16`, not timings.

## Verification: exact commands and results

All commands run from the repo root unless noted. `<repo>` is
`/home/princetheprogrammerbtw/.t3/scratch/2026-11-24-autonomous-continuation-for-the-loopkit-9dec44a4/inkengine-llama`.

### 1. Patch artifact

```sh
git add src/llama-pibs.h src/models/inkling2-loop.cpp
git diff HEAD -- src/llama-pibs.h src/models/inkling2-loop.cpp > tools/pibs/PIBS_WIRING.patch
sha256sum tools/pibs/PIBS_WIRING.patch
wc -c -l tools/pibs/PIBS_WIRING.patch
```

Result:

| Property | Value |
|---|---|
| sha256 | `f69543d2ba51bf16a5af2fd5c17b3ccdc64c16d252ac844186fe1cd294371804` |
| bytes | 5054 |
| lines | 131 |
| diffstat | 2 files changed, 109 insertions(+), 0 deletions(-) |

### 2. Fresh worktree apply + hash match

```sh
git worktree add /tmp/pibs-verify HEAD
cd /tmp/pibs-verify
git apply --check <repo>/tools/pibs/PIBS_WIRING.patch
git apply <repo>/tools/pibs/PIBS_WIRING.patch
sha256sum src/llama-pibs.h src/models/inkling2-loop.cpp
```

Result: `git apply --check` PASS, `git apply` PASS, 2/2 file hashes match the working tree:

| File | sha256 (both worktree and working tree) |
|---|---|
| `src/llama-pibs.h` | `176602854a9eedea32a5927587518b08e25a15c5be7d6088849af3f10cce5e53` |
| `src/models/inkling2-loop.cpp` | `a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab` |

Cleanup: `git worktree remove --force /tmp/pibs-verify` (the only changes in that
temporary worktree were the two files applied by this patch; force was required
because `git apply` leaves them modified/untracked). `git worktree list` then showed
only the main repo at `a05f501`. No user work was deleted.

### 3. Full fork build (CPU-only)

```sh
cmake -S . -B /tmp/pibs-build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build /tmp/pibs-build --target llama-cli -j4
```

Result: **BUILD OK / 100%**. Configure ended with `Configuring done (6.2s)`,
`Generating done (1.4s)`, `ggml commit a05f501`. Build ended with
`[100%] Built target llama-cli`, exit code 0. The first foreground invocation
hit the harness 900000 ms cap while compiling common sources (no compiler error);
the resumed background invocation completed the same target successfully.
Binary: `/tmp/pibs-build/bin/llama-cli`, `--version` reported
`version: 1.0.0-dev (build 115, commit a05f501)` and
`built with GNU 16.2.1 for Linux x86_64`. This is a CPU-only build
(`GGML_CUDA=OFF`), so the MoE cache is not constructed and no runtime PIBS hook
is exercised; it proves compilation only.

### 4. ctest suites (asserts active via `-UNDEBUG`)

```sh
cmake -S tools/pibs -B /tmp/pibs-ctest-pibs && cmake --build /tmp/pibs-ctest-pibs -j4 && ctest --test-dir /tmp/pibs-ctest-pibs --output-on-failure
cmake -S tools/roret -B /tmp/pibs-ctest-roret && cmake --build /tmp/pibs-ctest-roret -j4 && ctest --test-dir /tmp/pibs-ctest-roret --output-on-failure
cmake -S tools/odsl -B /tmp/pibs-ctest-odsl && cmake --build /tmp/pibs-ctest-odsl -j4 && ctest --test-dir /tmp/pibs-ctest-odsl --output-on-failure
```

Results:

| Suite | Result |
|---|---|
| pibs | **2/2 PASS** (`pibs_model`, `pibs_wiring`), 100% tests passed |
| roret | **2/2 PASS** (`test_union_cache`, `test_roret_tile_bitexact`), 100% tests passed |
| odsl | **1/1 PASS** (`odsl_numeric_layout_model`), 100% tests passed |

`tools/pibs/CMakeLists.txt` registers `test_pibs_wiring`; its compile flags include
`-UNDEBUG`, and the test itself has `#error` if `NDEBUG` is defined, so its
assert-based oracle cannot be silently disabled in a Release build.

New test binary output (direct invocation):

```text
batch=1 gate=unset interleave=0 steps=1 padded=16 wasted=13
batch=2 gate=unset interleave=0 steps=2 padded=32 wasted=26
batch=4 gate=unset interleave=0 steps=4 padded=64 wasted=52
batch=8 gate=unset interleave=0 steps=8 padded=128 wasted=104
batch=8 gate=0 interleave=0 steps=8 padded=128 wasted=104
batch=1 gate=1 interleave=0 steps=1 padded=16 wasted=13
batch=2 gate=1 interleave=0 steps=2 padded=32 wasted=26
batch=4 gate=1 interleave=1 steps=3 padded=48 wasted=36
batch=8 gate=1 interleave=1 steps=3 padded=48 wasted=24
test_pibs_wiring: PASS (gate default-off, batch>=4 only, 8->3 steps, 104->24 wasted, invariants)
```

The test checks: unset/empty/`0`/`off`/`TRUE` are all off; `1`/`on`/`true` enable;
gate off is sequential for batches 1/2/4/8; gate on interleaves only at `>= 4`;
batch 8 gives 8 -> 3 steps and 104 -> 24 wasted; batch 1/2 stay sequential;
`M == batch` per interleaved step (never `batch*3`); and
`seq(t,0) < seq(t,1) < seq(t,2)` for every token in both schedules.

Additional standalone compile checks:

```sh
g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -I. tools/pibs/test_pibs_wiring.cpp -o /tmp/pibs-wiring17 && /tmp/pibs-wiring17
g++ -std=c++20 -Wall -Wextra -Wpedantic -Werror -I. tools/pibs/test_pibs_wiring.cpp -o /tmp/pibs-wiring20 && /tmp/pibs-wiring20
```

Result: both PASS (exit 0). GNU 16.2.1, clean under `-Werror`.

### 5. loopkit audit pytest

```sh
cd /tmp/build/loopkit && PYTHONPATH=src python3 -m pytest -q
```

Result: **41 passed in 4.97s** (41 collected). numpy 2.5.4 is present; scipy is
absent and not needed by the audit tests.

### 6. Diff inspection

```sh
grep -inE 'logits|probs|argsort|weights|selected_experts_in' tools/pibs/PIBS_WIRING.patch
grep -nE 'gate_enabled|INKLING2_PIBS|kMinBatch|n_tokens >= ' tools/pibs/PIBS_WIRING.patch
```

Result: the first grep returned **no matches** - no changed line touches router
logits, probs, argsort, weights or `selected_experts_in`. The patch hunks in
`src/models/inkling2-loop.cpp` add only two includes and one logging block after
`build_inputs()`; the `for (uint32_t pass ...)` loop and all `build_layer` calls
are context-only, with zero changed lines. Default-off conditions confirmed:
`gate_enabled()` requires the exact strings `1`/`on`/`true`; `decide()` sets
`interleave = gate_on && n_tokens >= kMinBatch` with `kMinBatch = 4`; unset gate
and batches 1/2 resolve to the untouched sequential path.

### 7. src hygiene and ASCII

```sh
git status --short
LC_ALL=C grep -nP '[^\x00-\x7F]' src/llama-pibs.h src/models/inkling2-loop.cpp tools/pibs/test_pibs_wiring.cpp tools/pibs/CMakeLists.txt
```

Result: under `src/` there are exactly two commit-ready tracked entries
(`A  src/llama-pibs.h`, `M  src/models/inkling2-loop.cpp`) and no scratch or
untracked files. ASCII check returned no matches. `tools/pibs/CMakeLists.txt` is
modified and `tools/pibs/PIBS_WIRING.patch` / `tools/pibs/test_pibs_wiring.cpp` /
`tools/pibs/WIRING_REPORT.md` are new; no commit was made.

## NO_GO items

1. **Full engine-level node reordering: NO_GO / OPEN DESIGN QUESTION.** The patch
   is a default-off decision and instrumentation seam. It does not reorder ggml
   nodes or regroup ubatch tokens. The M6 design's open question is unclosed.
2. **Engine-level F32 parity with the gate actually changing the graph: NO_GO.**
   No GPU is available (T4 sm_75 has FP16 tensor cores, no FP4/FP8), no weights
   are held, and the `GGML_CUDA=OFF` build cannot construct the MoE cache. No
   `ffn_moe_topk` / `ffn_moe_weights` / `ffn_moe_out` / final-logits off-vs-on
   `memcmp` was measured. Bit-identical off/on here follows from the fact that
   the graph is literally unchanged, not from a measured tensor comparison.
3. **Any speed claim: NO_GO.** No tok/s, wall-clock, or speedup figure is
   published. Steps and wasted slots are COUNTS from a numeric model.
4. M4's engine-level parity was advanced concurrently on 2026-11-25 through
   the default-off `LLAMA_RORET_CPU_TEST` build flag and
   `tools/audit/PARITY_REPORT.md` (reported MEASURED PASS on the synthetic
   Inkling2-loop GGUF with RORET active). `tools/roret/BITEXACT_REPORT.md`
   still records the earlier host/tile-scoped result. This milestone does not
   change either verdict; issue #4 remains open in the tracker.

## Mandatory gate for any future number

Any performance claim born from this work must pass a loopkit CycleAudit result
with `claim <= ci_low`, `repeats >= 20` and `loops = 3`. If the gate fails or no
audit exists, the claim is NO_CLAIM. Load counts, step counts and wasted-slot
counts are never promoted to speedups.

## References

- `docs/ROADMAP_48H.md` (M7 acceptance, honest constraints, STOP conditions)
- `tools/pibs/SCHEDULER_DESIGN.md` (M6 design, sections 4-7)
- `tools/pibs/RESULTS.md`, `RESULTS_EXTENDED.md` (authoritative counts)
- `tools/roret/UNION_WIRING.patch`, `tools/roret/BITEXACT_REPORT.md` (artifact and
  honesty pattern)
- `src/llama-pibs.h`, `src/models/inkling2-loop.cpp`, `tools/pibs/PIBS_WIRING.patch`
