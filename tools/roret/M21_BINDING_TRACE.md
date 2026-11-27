# M21 - Engine-level RORET binding trace on Kaggle 2xT4 (2026-11-28)

Status: **ROOT CAUSE OF THE SINGLE-ROW SYMPTOM ESTABLISHED (proximate).**
The deeper question of why `build_lookup` never reaches a binding for pass 0
and pass 1 remains open and is now instrumented for the next run. No speed
claim. Issue #5 stays BLOCKED until a full per-pass table exists.

## What was done

M20 recorded the exact next step: run an engine-level trace of
`roret_of.find(...)` resolutions and `roret_traffic.add(...)` keys on the CUDA
graph. M21 added a default-off `LLAMA_RORET_TRACE=1` diagnostic to
`src/llama-moe-cache.cpp` that logs binding `set`, `resolve`, `hit`, `miss`,
`add`, `no_slot_map`, and (added after the run) `no_experts`, `no_mulmat` and
`no_getrows` early returns. The trace is read-only INFO logging: no tensor,
router, graph-node or control-flow line changed. A host-side test
(`test_roret_binding_trace.cpp`) exercises the real binding class lifecycle.

One Kaggle kernel was spent this run:
`jprinceraj/inkengine-cuda-per-pass-traffic` v2, private, `NvidiaTeslaT4x2`,
status COMPLETE. Logs were read with `kaggle kernels logs` (not the output API)
for the notebook summary; one output download was used to read the four raw
`*-cli.log` files that contain the INFO lines. Clone `-b main` HEAD
`618721eacd9cae87dc10a2e7b43b6f28970fb9a6`, nvcc 13.0.88, CUDA configure with
`-DGGML_CUDA_NO_VMM=ON` exit 0, `build_ok = true`, fixture sha256
`cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`.
Notebook sha256 `187967c82792abc09d9ad58a1fca8b0f48d350142021d5f97fa9abde2cd159c2`.
Kernel log sha256 `84f880bcd80d74dd4604c345b9ccb9621044f0dbd9b8bd81f9a3807d7150fc1d`.
`0.0-on-cli.log` sha256 `767cf8aeb9e96b232c53434d98e52d3c1f63e69271f7b42ba57c70a4e6980d6c`;
142-line trace extract sha256 `ee460f4d9aa9f4cd27446a3962651ad0997f631c91f18809148c90759e02d31f`.

## Measured evidence (identical for cpu_share 0.0 and 1.0)

| RORET_BIND line kind | count per ON run |
|---|---|
| `set` | 28 |
| `no_slot_map` | 16 |
| `resolve` | 38 |
| `hit` | 19 |
| `miss` | 19 |
| `add` | 19 |

Keys observed:

| Line kind | distinct keys (key_il, pass) |
|---|---|
| `set` | 14 x `(-1, -1)`, 14 x `(1, 2)` |
| `no_slot_map` (layer_il -> key) | 4 x `0 -> (-1,-1)`, 4 x `1 -> (1,0)`, 4 x `2 -> (1,1)`, 4 x `3 -> (1,2)` |
| `hit` | 19 x `(1, 2)` |
| `add` | 19 x `(1, 2)` |

Engine row (each share): `RORET_PASS il=1 pass=2 plans=19 hits=150
upload_ops=78 uploaded_bytes=153600 residents=812`. The notebook's strict
per-share gate still failed closed (`per_pass_verdict NO_GO`,
`overall_verdict NO_GO`) because pass values 0 and 1 are absent.

## Root cause (proximate, evidence-backed)

Across 28 binding registrations there is **no `set` with pass 0 or pass 1**.
The only loop key ever registered is `(1, 2)`. Consequently
`roret_of.find(...)` can only hit `(1, 2)`, and `roret_traffic.add(...)` can
only receive `(1, 2)`, which is exactly the single row observed. The missing
pass-0 and pass-1 rows are therefore **not** a logging, aggregation or
post-processing artifact: those passes never produce a binding in the engine.

The `no_slot_map` lines show one concrete reason for some builds: layers 1 and
2 (the physical cache slots used by pass 0 and pass 1 in this 2-layer fixture)
return `nullptr` from `get_slot_map` and exit `build_lookup` before registration.
Not every pass-0/pass-1 build produced a `no_slot_map` line, so at least one
other pre-binding early return (`get_experts == nullptr`, unsupported
`mul_mat_id`, or unsupported `get_rows`) is also taken. Those three branches
were **not** logged in the kernel that ran; they were instrumented immediately
after the run and remain for the next CUDA run to distinguish. This is the
open deeper question, not a finding.

M19's hypothesis (the per-graph lookup key collapsed earlier passes into
`(1,2)`) is refined, not rescued: the key is fine, but no binding for an
earlier pass is ever inserted.

## Honesty and next step

All numbers above are **event counts** from INFO logs. No wall-clock, tok/s or
speedup is claimed. Any future speed number still requires CycleAudit
`claim <= ci_low` with `repeats >= 20` and `loops >= 3`.

Issue #5 remains BLOCKED: no full per-pass T4 traffic table exists. Next run:
spend one Kaggle kernel on the now-instrumented HEAD, require the raw
`no_experts` / `no_mulmat` / `no_getrows` counts for pass 0 and pass 1, and only
then decide whether the 2-layer synthetic fixture can ever exercise three
distinct pass bindings or must be replaced by a fixture with a real multi-layer
loop core. The notebook parser was fixed to match INFO lines with timestamp
prefixes (`"RORET_BIND " in ln` instead of `startswith`) so the trace is
captured in the summary JSON on that run.
