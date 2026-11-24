# 48-hour execution roadmap: inkengine-llama fork

Status: planning deliverable only. No source files were modified in this pass.
Baseline: fork `10xdev4u-alt/inkengine-llama`, branch `main`, commit `5035bdf`.
Budget: 48 h wall clock, ~29.7 h Kaggle GPU quota (~10 h scheduled below), hourly autonomous task.
Evidence already on record: `SPEC_RISK.md` (fork decision), `tools/roret/KAGGLE_RESULTS.md`
(100k-token probe, union 8.25-11.58 experts vs 24 loads, 61-66% load reduction, exact
route-once NO_GO), `tools/pibs/RESULTS.md` (batch 8: 104 -> 24 wasted tile slots),
`tools/odsl/RESULTS.md` (0/16000 bit mismatches, 76.4x drift reduction, 4x scale-load
reduction), `BUILD_STATUS.md` (fresh Kaggle fork build green).

The sequence is deliberately correctness-first: every performance milestone sits behind
a bit-exact gate and a CycleAudit gate. A failed gate stops the chain; it is never
traded for a number.

## Milestone summary

| # | Title | Owner role | Deliverable file | Est. | Unblocks |
|---|---|---|---|---|---|
| M1 | Freeze bit-exact baseline and CycleAudit gate contract | Hy4 Flash subagent | `tools/audit/AUDIT_PROTOCOL.md` | 2 h | M2, M9, M10 |
| M2 | Generate deterministic synthetic MoE GGUF fixture and F32 dump harness | Hy4 Flash subagent | `tools/audit/make_moe_fixture.py` | 3 h | M3, M4, M5 |
| M3 | Wire union-resident MoE cache behind a default-off gate | integration | `tools/roret/UNION_WIRING.patch` | 5 h | M4, M5 |
| M4 | Prove bit-exact parity of union wiring on three loop passes | integration | `tools/roret/BITEXACT_REPORT.md` | 4 h | M5, M9, M10 |
| M5 | Measure MoE tile traffic with union residency on Kaggle T4 | Kaggle run | `tools/roret/TRAFFIC_REPORT.md` | 4 h GPU | M9, M10 |
| M6 | Convert PIBS tile model into a gated scheduler design | Hy4 Flash subagent | `tools/pibs/SCHEDULER_DESIGN.md` | 3 h | M7 |
| M7 | Implement pass-interleaved batching with preserved pass order | integration | `tools/pibs/PIBS_WIRING.patch` | 5 h | M9, M10 |
| M8 | Specify bit-exact block-major scale layout integration | Hy4 Flash subagent | `tools/odsl/LAYOUT_DESIGN.md` | 3 h | M9, M10 |
| M9 | Run end-to-end CycleAudit on combined candidate | Kaggle run | `tools/audit/CYCLE_AUDIT.json` | 6 h GPU | M10 |
| M10 | Consolidate verification, claims, and handoff state | integration | `docs/VERIFICATION_48H.md` | 4 h | handoff |

Total planned work: 39 h. Reserve: 9 h for gate failures, kernel retries, and checkpoint correction.
GPU planned: 10 h of 29.7 h; 19.7 h stays unspent.

## M1 - Freeze bit-exact baseline and CycleAudit gate contract

- Owner: Hy4 Flash subagent
- Deliverable: `tools/audit/AUDIT_PROTOCOL.md`
- Acceptance: the document fixes a deterministic audit recipe (`python -m loopkit.cli audit
  --claim C --baseline b.json --candidate c.json --loops 3`), requires `repeats >= 20`, and a
  negative-control replay with `C` above `ci_low` must return `claim_valid: false` and exit 1.
  It also records the rule that a claim is publishable only when `claim <= ci_low`.
- Unblocks: M2, M9, M10. Sets the vocabulary so no downstream report can call a load count
  or a numeric-model ratio a speedup.

## M2 - Generate deterministic synthetic MoE GGUF fixture and F32 dump harness

- Owner: Hy4 Flash subagent
- Deliverable: `tools/audit/make_moe_fixture.py`
- Acceptance: generates a small random-weight (no pretrained weights) 128-expert/top-8 MoE GGUF
  with three loop passes and a fixed seed; dumps F32 `ffn_moe_topk`, `ffn_moe_weights`,
  `ffn_moe_out`, and final logits. Two consecutive runs produce byte-identical dumps with the
  same SHA-256. Runs on Kaggle CPU without network access.
- Unblocks: M3, M4, M5. This is the only test vehicle available while no weights are held.

## M3 - Wire union-resident MoE cache behind a default-off gate

- Owner: integration
- Deliverable: `tools/roret/UNION_WIRING.patch`
- Acceptance: artifact applies cleanly to `5035bdf`; follows `tools/roret/PATCH_build_moe_ffn.md`
  insertion points exactly; `pass_index = -1` or empty `host_selected_ids` leaves all four
  insertion points no-ops; `GGML_CUDA=OFF` build of `llama-cli` succeeds and all three ctest
  suites (roret, pibs, odsl) pass. Router logits, argsort, weights, and `selected_experts_in`
  are unchanged by inspection of the diff.
- Unblocks: M4, M5. Deliberately produces a patch artifact, not an un-gated source edit.

## M4 - Prove bit-exact parity of union wiring on three loop passes

- Owner: integration
- Deliverable: `tools/roret/BITEXACT_REPORT.md`
- Acceptance: with the M2 fixture, `memcmp` on F32 dumps shows 0 differing bytes for all three
  passes, feature off vs on; `pass_index = -1` path is identical to the untouched baseline;
  verified at `cpu_share = 0` and then `cpu_share > 0`; if CPU-share flips a bit, the report
  records a scoped fallback (pin device-resident layers only) or verdict NO_GO. Any nonzero
  difference stops M5-M9.
- Unblocks: M5, M9, M10. This is the hard correctness gate for the whole chain.

## M5 - Measure MoE tile traffic with union residency on Kaggle T4

- Owner: Kaggle run
- Deliverable: `tools/roret/TRAFFIC_REPORT.md`
- Acceptance: logs `n_hit` and uploaded `bytes` from `llama-moe-cache.cpp` for off/on runs on
  the M2 fixture; distinct uploaded tiles move from ~24 toward the measured union size
  (8.25-11.58) and steady-state union growth per token is bounded because `clear_layer` runs
  per token. The report states "load count, not wall-clock" in its first paragraph and makes
  no tok/s claim.
- Unblocks: M9, M10. Converts the 100k-token synthetic precondition into an engine-side
  traffic counter without needing real weights.

## M6 - Convert PIBS tile model into a gated scheduler design

- Owner: Hy4 Flash subagent
- Deliverable: `tools/pibs/SCHEDULER_DESIGN.md`
- Acceptance: reproduces batch 8 (sequential 8 steps / 104 wasted slots / 18.75% efficiency
  vs interleaved 3 steps / 24 slots / 50%) and records the batch-1 regression (13 -> 45
  wasted slots); documents the invariant `M <= batch` and that intra-token pass order stays
  intact (no `M = batch*3` GEMM); defines the enabling condition `batch >= 4` and default-off
  behavior.
- Unblocks: M7. Prevents an implementation that silently relaxes correctness for a better tile count.

## M7 - Implement pass-interleaved batching with preserved pass order

- Owner: integration
- Deliverable: `tools/pibs/PIBS_WIRING.patch`
- Acceptance: applies to `5035bdf` plus M3/M4-gated state; instrumentation on the M2 fixture
  reports 8 -> 3 steps at batch 8 with identical F32 dumps to sequential ordering; batch 1
  and batch 2 keep the sequential path; full fork build and all three ctest suites stay green.
- Unblocks: M9, M10. Adds a scheduler change only after the MoE numerics are proven.

## M8 - Specify bit-exact block-major scale layout integration

- Owner: Hy4 Flash subagent
- Deliverable: `tools/odsl/LAYOUT_DESIGN.md`
- Acceptance: reproduces 0/16000 mismatches from vendored vectors and the 76.4x FP16 drift
  reduction with the FP32 side lane; explicitly labels the side-lane path as lower-drift but
  NOT bit-exact; labels 512 B vs 2048 B as a sector-fetch model, not measured bandwidth;
  states T4 (sm_75) has FP16 tensor cores and no FP4/FP8 tensor cores, so NVFP4 is storage only.
  Design is default-off and independent of M3/M7 gating.
- Unblocks: M9, M10. Ensures ODSL enters the audit only as a clearly classified candidate.

## M9 - Run end-to-end CycleAudit on combined candidate

- Owner: Kaggle run
- Deliverable: `tools/audit/CYCLE_AUDIT.json`
- Acceptance: `python -m loopkit.cli audit` with `repeats >= 20`, `loops = 3`, 2000 bootstrap
  resamples, baseline and candidate raw second arrays saved alongside the JSON; any claim in
  the report is `<= ci_low`. If `claim_valid` is false, the JSON records `claim: 0` and
  verdict NO_CLAIM rather than a rounded-up number. No weight-dependent measurement is
  attempted; the run uses the M2 fixture.
- Unblocks: M10. This is the only place a speed number may be born.

## M10 - Consolidate verification, claims, and handoff state

- Owner: integration
- Deliverable: `docs/VERIFICATION_48H.md`
- Acceptance: links M1-M9 artifacts, restates all four honest constraints verbatim, lists
  every gate result as PASS/NO_GO, confirms all patches remain default-off and no scratch
  source edits are left untracked in `src/`, and records the three STOP conditions with their
  hour-48 values. Fresh Kaggle CPU rebuild of fork `main` + applied gated patches is green.
- Unblocks: handoff to the next scheduled hourly run; no new milestone starts before this file exists.

## Checkpoints

**Hour 6 (measurable state):** M1 and M2 complete (2 h + 3 h), M3 started.
`tools/audit/AUDIT_PROTOCOL.md` exists and its negative-control replay returns
`claim_valid: false` with exit 1. `tools/audit/make_moe_fixture.py` has produced two identical
dump SHA-256 values from a fixed seed. `git status` shows no changes under `src/`. If either
artifact is missing, the loop spends reserve time here and does not start M3.

**Hour 24 (measurable state):** M1-M6 complete (21 h), M7 started.
`tools/roret/BITEXACT_REPORT.md` reports 0 differing bytes on all three passes and both
CPU-share settings; `tools/roret/TRAFFIC_REPORT.md` reports distinct uploads moving toward
8.25-11.58 from ~24 with `clear_layer` per token; `tools/pibs/SCHEDULER_DESIGN.md` reproduces
104 -> 24 at batch 8 and the batch-1 regression. No GitHub issue or PR has been created while
issues are disabled; the 2-per-hour cap has not been approached.

**Hour 48 (measurable state):** all 10 milestones complete by hour 39, 9 h reserve intact or
logged as consumed. `tools/audit/CYCLE_AUDIT.json` exists with `claim <= ci_low` or an explicit
NO_CLAIM; `docs/VERIFICATION_48H.md` lists 10/10 artifacts with PASS/NO_GO, fork build green,
and all patches default-off. Any unverified claim found in any doc is removed before handoff.

## Honest constraints (binding on every milestone)

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

## STOP conditions

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

## Appendix: prepared issue drafts (NOT created - issues disabled)

Preflight evidence: `gh issue list --repo 10xdev4u-alt/inkengine-llama --state all --limit 30`
returned "the repository has disabled issues"; `gh api repos/... --jq .has_issues` returned
`false` while authenticated as `10xdev4u-alt`. Zero issues created, below the 5-issue and
2-per-hour caps. Intended labels: `engine`, `roadmap` (create them first if enabled).

1. **Freeze bit-exact baseline and CycleAudit gate contract**
   Establish the single audit recipe before any optimization is wired: reproducible bootstrap
   timing via `python -m loopkit.cli audit`, `repeats >= 20`, `loops = 3`, and a machine-checked
   rule that a speed claim is publishable only when it is at or below the lower 95% confidence
   bound. A negative control must fail closed. Deliver `tools/audit/AUDIT_PROTOCOL.md`. This
   prevents load counts and numeric-model ratios from being reported as speedups later.

2. **Generate deterministic synthetic MoE GGUF fixture and F32 dump harness**
   Because no pretrained weights are available, all executable verification needs one small
   fixed-seed random-weight 128-expert/top-8 MoE fixture with three loop passes and a harness
   that dumps F32 top-k, weights, expert output, and logits with a stable SHA-256 across runs.
   Deliver `tools/audit/make_moe_fixture.py`. This is the test vehicle for bit-exactness, cache
   traffic, and CycleAudit; any real-model drift study stays out of scope until weights exist.

3. **Wire union-resident MoE cache behind a default-off gate**
   Apply the insertion points in `tools/roret/PATCH_build_moe_ffn.md` as a patch artifact,
   keeping the router, argsort, weights, and `selected_experts` untouched and making the
   feature a no-op for `pass_index = -1` or empty host ids. Deliver
   `tools/roret/UNION_WIRING.patch`, verified by a clean `GGML_CUDA=OFF` build and all three
   ctest suites. This is residency only: it targets the 24-loads-to-8.25-11.58 union found in
   the 100k-token probe and makes no top-k reduction claim.

4. **Prove bit-exact parity of union wiring on three loop passes**
   Before any traffic or timing claim, dump F32 tensors with the feature off and on and require
   `memcmp` to report zero differing bytes on all three passes, including the default path and
   both `cpu_share = 0` and `cpu_share > 0`. Deliver `tools/roret/BITEXACT_REPORT.md`, with an
   explicit NO_GO or a device-only pinning fallback if any byte flips. This gate is the hard
   precondition for every later performance milestone.

5. **Measure MoE tile traffic with union residency on Kaggle T4**
   Instrument `llama-moe-cache.cpp` `n_hit` and uploaded bytes on the synthetic fixture and
   check that distinct uploads move from about 24 toward the measured union of 8.25-11.58
   experts, with per-token `clear_layer` preventing union growth to all experts. Deliver
   `tools/roret/TRAFFIC_REPORT.md`, labelled as a load count rather than wall-clock. This turns
   the synthetic precondition into an engine-side observation without requiring weights or a
   speed claim.
