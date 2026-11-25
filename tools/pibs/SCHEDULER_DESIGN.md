# PIBS scheduler design (M6)

Status: design only. No source files were modified. No patch was written, no
commit was made, and no `src/` file was touched. This document turns the PIBS
tile-count model into a gated scheduler design for M7. It is a COUNT and
SCHEDULING MODEL, not a measured speedup. There is no tok/s figure, no
wall-clock figure, and no claim of real engine speedup here.

## 1. Scope and honest constraints

- The model counts logical GEMM steps and m16 tile slots. A "step" is one
  modelled fused GEMM in the model, not a measured kernel launch in the engine.
- Wasted-slot counts and efficiency ratios are arithmetic on `M`, not timings.
- At batch 8 the 104 -> 24 drop in wasted slots, and the 24 wasted slots of the
  interleaved row, are tile-slot counts. They are not a speedup.
- Any future performance number from this work must pass the loopkit CycleAudit
  gate: `claim <= ci_low`, with `repeats >= 20` and `loops = 3`. If the gate is
  not passed, the claim is NO_CLAIM. Load counts are never promoted to speedups.
- No weights are held. All checks use the standalone model or the synthetic
  fixture; this document does not measure a real model.

## 2. Model and definitions

Work items are `(token_id, pass_index, ready)` triples, `n_tokens x 3` of them
(`kPasses = 3`). Dependency: item `(t, p+1)` becomes ready only after `(t, p)`
is done. There are no cross-token dependencies, so at most one pass of a token
is ready at a time.

A scheduling step folds the currently ready items into one logical GEMM:

- `M` = number of folded items.
- `padded_m = ceil(M/16)*16`; the model uses `kWmmaM = 16` (T4 FP16 WMMA m16).
- `wasted = padded_m - M` for that step.
- `efficiency = sum(M) / sum(padded_m)` over all steps in a run.

Schedulers in `tools/pibs/pibs.cpp`:

| scheduler | rule | M per step | steps |
| --- | --- | --- | --- |
| sequential | finish all 3 passes of token i before token i+1; passes fused in one step, order kept | 3 | n_tokens |
| interleaved | one pass level across all tokens per step | n_tokens | 3 |
| bucketed | fill a bucket of size `ceil(ready/16)*16` from the lowest ready pass level; released next passes may join the same bucket | up to `ceil(3*batch/16)*16` | varies |

Batches exercised are `{1, 2, 4, 8, 16, 32, 64}`.

## 3. Reproduced results

Authoritative rows from `tools/pibs/RESULTS_EXTENDED.md` (and `RESULTS.md`).
Wasted is summed over steps; efficiency is `sum(M)/sum(padded_m)`.

| batch | scheduler | steps | mean M | padded | wasted | efficiency |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | sequential | 1 | 3 | 16 | 13 | 18.75% |
| 1 | interleaved | 3 | 1 | 48 | 45 | 6.25% |
| 1 | bucketed | 1 | 3 | 16 | 13 | 18.75% |
| 2 | sequential | 2 | 3 | 32 | 26 | 18.75% |
| 2 | interleaved | 3 | 2 | 48 | 42 | 12.50% |
| 2 | bucketed | 1 | 6 | 16 | 10 | 37.50% |
| 4 | sequential | 4 | 3 | 64 | 52 | 18.75% |
| 4 | interleaved | 3 | 4 | 48 | 36 | 25.00% |
| 4 | bucketed | 1 | 12 | 16 | 4 | 75.00% |
| 8 | sequential | 8 | 3 | 128 | 104 | 18.75% |
| 8 | interleaved | 3 | 8 | 48 | 24 | 50.00% |
| 8 | bucketed | 2 | 12 | 32 | 8 | 75.00% |
| 16 | sequential | 16 | 3 | 256 | 208 | 18.75% |
| 16 | interleaved | 3 | 16 | 48 | 0 | 100.00% |
| 16 | bucketed | 3 | 16 | 48 | 0 | 100.00% |
| 32 | sequential | 32 | 3 | 512 | 416 | 18.75% |
| 32 | interleaved | 3 | 32 | 96 | 0 | 100.00% |
| 32 | bucketed | 3 | 32 | 96 | 0 | 100.00% |
| 64 | sequential | 64 | 3 | 1024 | 832 | 18.75% |
| 64 | interleaved | 3 | 64 | 192 | 0 | 100.00% |
| 64 | bucketed | 3 | 64 | 192 | 0 | 100.00% |

Batch 8 (target result): sequential is 8 steps, 104 wasted slots, 18.75%
efficiency; interleaved is 3 steps, 24 wasted slots, 50% efficiency. Each
sequential step is M=3 padded to 16 (13 wasted). Each interleaved step is M=8
padded to 16 (8 wasted), three such steps.

Batch 1 (regression, recorded honestly): sequential wastes 13 slots in one
step; interleaved wastes 45 slots in 3 steps of M=1 (15 wasted each). At
batch 1 interleaving is strictly worse on wasted slots. Batch 2 is also worse
(42 vs 26). The model does not hide either case.

Breakeven (first batch where the challenger wastes fewer slots than sequential):
interleaved = 4, bucketed = 2.

## 4. Correctness invariants

- `M <= batch`. Because a token has at most one ready pass at a time, a step
  can hold at most one item per token. The step size cannot exceed `n_tokens`.
- Intra-token pass order is preserved: `seq(t,0) < seq(t,1) < seq(t,2)` for
  every token. This holds for sequential, interleaved, and bucketed schedules.
- A single GEMM with `M = batch*3` is FORBIDDEN. It would require all three
  passes of every token to be ready at once, which relaxes the pass-order
  dependency. The correctness assertions explicitly reject that relaxation.
  Interleaving changes row count per step (batch instead of 3) and step count
  (3 instead of n_tokens); it never changes the total work (3*n_tokens).
- `verify()` in `pibs.cpp` asserts: exactly `3*n_tokens` items executed, each
  item executed exactly once, all items done, and pass order per token as above.
  These assertions pass for every (batch, scheduler) run in the results above.
- `run()` also rejects a non-ready item, a scheduler stall, and a dependency
  released after its successor is done.

## 5. Enabling condition and default-off gating (proposal)

- Enabling condition: `batch >= 4`. This is the first batch where interleaved
  wastes fewer slots than sequential (36 vs 52 at batch 4). Breakeven is 4.
- Default-OFF. The interleaved path is a candidate, not a default. M7 must keep
  the current sequential behavior when the gate is off.
- Batches 1 and 2 always keep the sequential path. At those sizes interleaving
  wastes more slots (45 vs 13 and 42 vs 26), so there is no reason to enable it.
- Proposed gate mechanism (NOT implemented, name is a proposal only): an
  opt-in flag, for example an environment knob `INKLING2_PIBS=1` or an
  equivalent build/run option, that enables interleaved ordering only when the
  effective batch is `>= 4`. Any off value, unset flag, or batch `< 4` resolves
  to the untouched sequential path. The exact flag location and name are an M7
  implementation decision.
- The gate must not change router logits, argsort, weights, or
  `selected_experts_in`. This mirrors the default-off discipline used by the
  RORET union wiring milestone.

## 6. Proposed M7 integration points and open questions (PROPOSAL ONLY)

Nothing in this section is implemented or proven. These are candidate seams
for a future M7 patch:

- `src/models/inkling2-loop.cpp:90-112`: the pass loop is
  `for (uint32_t pass = 0; pass < m.n_passes; ++pass)`, and each core layer is
  built via `build_layer(il, ip, ..., ip, (int32_t) pass)`. The slot count is
  `n_slot = n_phys + (n_passes-1)*(core_end-core_start) = 24 + 2*12 = 48`.
  A pass-interleaved ordering would have to restructure how these nodes are
  emitted, not just relabel them.
- `src/models/inkling2.cpp:393-414`: `build_moe_ffn` is called with
  `roret_pass` set, but `roret_cache` and `roret_selected_ids` are currently
  `nullptr`. Any interleave-aware routing must be added without disturbing
  those existing arguments or the default path.
- `src/llama-graph.cpp:2498-2510`: the RORET graph-side hook is guarded by
  `roret_cache != nullptr && roret_pass >= 0 && roret_selected_ids != nullptr
  && roret_n_selected > 0`. With the current nullptr arguments this hook is a
  no-op. It is host-side bookkeeping only and never substitutes device ids.

OPEN DESIGN QUESTION for M7: the standalone PIBS model uses a simplified
`(token, pass)` work-item abstraction. Translating that abstraction into real
ggml node ordering and ubatch token grouping is NOT solved by this document.
This doc does not claim the wiring already works, and it does not claim that a
fusion of three passes into one GEMM is possible in the engine. Those are M7
implementation and correctness questions, gated by the bit-exact check.

## 7. What would close M7

- Instrumentation on the synthetic fixture reports 8 -> 3 steps at batch 8.
- F32 dumps (top-k, weights, expert output, logits) are identical with the gate
  off and on: `memcmp` reports zero differing bytes on all three passes.
  Any nonzero difference is a correctness stop, not a tunable tradeoff.
- Batch 1 and batch 2 stay on the sequential path.
- Pass order per token stays intact; no `M = batch*3` GEMM is introduced.
- Full fork build is green, and all three ctest suites (roret, pibs, odsl) pass.
- The gate remains default-off; no scratch source edits are left under `src/`.
- Any performance number born in M7 or later must satisfy the CycleAudit rule
  `claim <= ci_low` with `repeats >= 20` and `loops = 3`. Without a passing
  audit the result is NO_CLAIM, and step/waste counts stay labelled as counts.

## 8. References

- `tools/pibs/RESULTS.md` - recorded model output, batch 8 detail.
- `tools/pibs/RESULTS_EXTENDED.md` - authoritative JSON rows for all batches
  and schedulers; breakeven interleaved 4, bucketed 2.
- `tools/pibs/README.md` - model, dependency, `M <= n_tokens`, correctness.
- `tools/pibs/pibs.cpp` - schedulers, `kWmmaM = 16`, `kBatches`, `verify()`.
- `docs/ROADMAP_48H.md` - M6 acceptance and M7 unblock; honest constraints.
- `src/models/inkling2-loop.cpp`, `src/models/inkling2.cpp`,
  `src/llama-graph.cpp` - proposed M7 seams (read only, unmodified).
