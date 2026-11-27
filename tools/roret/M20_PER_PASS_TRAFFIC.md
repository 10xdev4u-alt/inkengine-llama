# M20 - Kaggle 2xT4 per-pass RORET traffic rerun (2026-11-28)

Status: **NO_GO for the per-pass traffic table; PASS-scoped for final-logits parity.**
The engine still emits only one `(il=1, pass=2)` RORET_PASS row per cpu_share
after M19. The new fail-closed gate rejected the table. No speed claim.

## Why this run was done

M18 measured engine-level off/on F32 final-logits parity but produced only
2 RORET_PASS rows, both `(il=1, pass=2)`. M19 changed the RORET binding key from
the shared slot-map tensor to the per-graph GET_ROWS lookup so that each pass
could be attributed separately (`src/llama-roret-binding.h`, commit `74c0e82`).
DECISIONS.tsv row 39 made the next step explicit: rerun the corrected CUDA parity
notebook at the new HEAD and require three distinct RORET_PASS rows before any
per-pass comparison. This run spent this pass's single Kaggle kernel on that test.

## What was run

* Kernel `jprinceraj/inkengine-cuda-per-pass-traffic` v1, private,
  `NvidiaTeslaT4x2`, status COMPLETE. Logs were read with `kaggle kernels logs`,
  not the output API.
* Notebook actually executed:
  `tools/roret/kaggle-engine-parity/engine_parity.ipynb` sha256
  `e41cc0ba53afb976f8476b8a7fbe60b75d91c309e0bb616143e1eb590fbc0aa2`
  (commit `de2f41d`). Kernel log sha256
  `88d269e783b7f8f9e8f86c240577ec409677d31dff0e5d08cd439d32dbc4b10d`;
  reconstructed stdout sha256
  `4e1aa559abbaf2b0fd6256840b55f72b885fd9621cf654d49caf5036dca4f739`.
* Clone `-b main` HEAD `de2f41d0db95b0c7c3b601a5ecb4ba508dba7690`.
  nvcc 13.0.88, CUDA host compiler GNU 13.3.0, 2x Tesla T4.
* CUDA configure with `-DGGML_CUDA_NO_VMM=ON` exit 0; build of `llama-cli` and
  `llama-debug` exit 0; `build_ok = true`.
* Synthetic GGUF sha256
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`
  (unchanged from M15-M18).
* All 8 binary runs exited 0. Hook ABSENT off / ACTIVE on in both cli and
  debug, for both cpu_share values. Zero `GGML_ASSERT` / `Assertion` lines.

## Measured result

| cpu_share | il | pass | plans | hits | upload_ops | uploaded_bytes | residents |
|---|---|---|---|---|---|---|---|
| 0.0 | 1 | 2 | 19 | 150 | 78 | 153600 | 812 |
| 1.0 | 1 | 2 | 19 | 150 | 78 | 153600 | 812 |

Aggregate row printed by the run's gate:
`PER_PASS il=1 pass=2 plans=38 hits=300 upload_ops=156 uploaded_bytes=307200 residents=1624`.

Gate outcome: `per_pass_gate_ok = false`,
`per_pass_verdict = NO_GO: fewer than 3 distinct (il,pass) pairs; pass values
do not include 0,1,2`, `overall_verdict = NO_GO`. The aggregate gate requires
at least 3 distinct `(il,pass)` pairs and pass values 0,1,2. Only pass 2 was
observed, so it failed closed.

Off/on parity remained intact: F32 final logits, token ids and generated text
all had 0 differing bytes for cpu_share 0.0 and 1.0
(`logits_parity_verdict` and `text_parity_verdict` both PASS-scoped). This does
not extend to intermediate per-pass MoE tensors.

## Interpretation and blocker

M19's per-graph lookup key did **not** change the engine-level row count: the
result is byte-for-byte the same pass-2-only table as M18. The hypothesis that
the old slot-map binding collapsed earlier passes into `(1,2)` is therefore
**disproven at engine level** by this run. Source inspection shows the graph
builds all three passes with logical key `(ip=1, pass=0/1/2)`
(`src/models/inkling2-loop.cpp:107-128`, `src/models/inkling2.cpp:411-414`) and
registers one binding per pass (`src/llama-moe-cache.cpp:780-784`), but only one
binding is observed in `roret_traffic.rows()`
(`src/llama-moe-cache.cpp:684-685,708-715`) during execution. The root cause of
the missing pass-0 and pass-1 rows is **not established** by this run. Plausible
structural candidates, both unverified: the minimal 2-layer synthetic fixture
maps later passes to physical cache slots that have no MoE-cache slot map
(`src/llama-moe-cache.cpp:468-479`), or the scheduler's copy callback resolves
only the last pass's lookup node. Do not treat either as a finding.

BLOCKER for issue #5: no full per-pass T4 traffic table exists. A future run
needs an engine-level trace of `roret_of.find(...)` resolutions and
`roret_traffic.add(...)` keys on the CUDA graph, or a fixture with enough
physical layers for all three pass slots, before any per-pass comparison is
defensible. One Kaggle kernel was spent this run; no second kernel was used.

## Gate hardening after the run

An independent Hy4 Flash review found that the aggregate gate could be
satisfied by two partial tables (passes 0,1 on one cpu_share plus pass 2 on the
other) and could hide a zero-plans row by summing. The checked-in notebook was
therefore tightened after the kernel was pushed (the running kernel kept its
snapshot): commit `40c8c5a`, sha256
`79d5a3bb78f1f60d2a34ea90366c603fbab1d80a3226ecd2106f5cd68679f1a0`. The
strict gate requires **each** cpu_share on-run to show one `il` with passes
0,1,2 and `plans > 0` on every contributing row; counters are never summed
across cache policies. Simulations: M18 collapsed rows -> NO_GO, partial tables
across shares -> NO_GO, both shares with passes 0,1,2 -> PASS. Original parity
gates, hook checks, exit checks, assert checks and timeouts were not weakened.

## Local supporting evidence

* Fresh ctest: roret 8/8, pibs 2/2, odsl 1/1.
* `tools/audit` pytest: 17 passed. loopkit fast pytest: 40 passed, 1 deselected.
* Fresh `GGML_CUDA=OFF LLAMA_RORET_CPU_TEST=ON` build:
  `1.0.0-dev (build 135, commit de2f41d)`, GNU 16.2.1. No `src/` edit this pass.
* All new report text is ASCII-only.

## Honest constraints

These RORET_PASS figures are engine COUNTS, not wall-clock, not `tok/s`, not a
measured device bandwidth, and not a speedup. The 8.25-11.58 union range is a
numeric-model probe, not a measured engine statistic, so this table neither
confirms nor refutes it. No real weights were used. NO SPEED CLAIM. Any future
performance claim must come from a CycleAudit run with `claim <= ci_low`,
`repeats >= 20` and `loops = 3`; no such lower bound exists here.
