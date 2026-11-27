# M22 - Deeper root cause of missing pass-0/pass-1 RORET bindings on Kaggle 2xT4 (2026-11-28)

Status: **DEEPER ROOT CAUSE ESTABLISHED (build-time cache-eligibility failure).**
The pass-0/pass-1 MoE slot layers are dropped by the MUL_MAT_ID backend-support
probe in `build_lookup` before any binding registration. No speed claim.
Issue #5 stays OPEN/BLOCKED: no full per-pass T4 traffic table exists.

## What was run

M21 established the proximate cause: across 28 registrations there was no
`set` for pass 0 or pass 1, only keys `(-1,-1)` and `(1,2)`. It instrumented
three more pre-binding early returns (`no_experts`, `no_mulmat`, `no_getrows`)
and asked for one Kaggle run at the new HEAD to separate them. M22 is that run.

One Kaggle kernel was spent: `jprinceraj/inkengine-cuda-per-pass-traffic` v3,
private, `NvidiaTeslaT4x2`, status COMPLETE. Logs were read with
`kaggle kernels logs` (not the output API, so no 429). The notebook run was
`tools/roret/kaggle-engine-parity/engine_parity.ipynb` at commit
`369bdd30c827509d7b237147656ee27bcbe43bbf`, file sha256
`343aafd1f691e7fd52ce32e49301c2819df2db2b3a5a39705b214f5090476c13`; it clones
`-b main` at that HEAD. Kernel log sha256
`1d5b4be984f593f3f2e4de080da1a76fba1c932c44dba5896fd093520657f770`;
reconstructed stdout sha256
`bb68ccb54fcdaf70209e765c922ff5b6b2046e6501f12bc20508844e860ce188`. An
additive notebook cell summarized the `RORET_BIND INFO` lines; the extracted
summary is saved as `tools/roret/kaggle-engine-parity/M22_SUMMARY.json` sha256
`a0907f0c49cb0d36da3594221737a4cdd89f515d0020ea8b9317e7c8a4411198`.

Toolchain: nvcc 13.0.88, GNU 13.3.0, 2x Tesla T4. CUDA configure with
`-DGGML_CUDA_NO_VMM=ON` exit 0; build llama-cli + llama-debug exit 0,
`build_ok=true`. Fixture sha256
`cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`.

## Measured evidence

All 8/8 binary exits were 0. The RORET hook was ABSENT for off and ACTIVE for
on in both binaries. There were zero `GGML_ASSERT` lines. Logits, token ids and
generated text all showed 0 differing bytes for cpu_share 0.0 and 1.0;
`logits_parity_verdict` and `text_parity_verdict` are both PASS-scoped.

Bind summary (additive INFO cell). Counts were IDENTICAL for `0.0-on` and
`1.0-on`, with 167 captured lines each; both off runs had 0 lines because
RORET gating was off. All values are event counts, not time or tok/s.

| RORET_BIND line kind | count per ON run | distinct keys (key_il, pass) |
|---|---|---|
| `set` | 28 | 14 x `(-1,-1)`, 14 x `(1,2)` |
| `no_slot_map` | 16 | 4 x `(-1,-1)`, 4 x `(1,0)`, 4 x `(1,1)`, 4 x `(1,2)` |
| `resolve` | 38 | (not bucketed in summary) |
| `hit` | 19 | 19 x `(1,2)` |
| `miss` | 19 | (not bucketed in summary) |
| `add` | 19 | 19 x `(1,2)` |
| `no_experts` | 0 | none |
| `no_mulmat` | 28 | 14 x `(1,0)`, 14 x `(1,1)` |
| `no_getrows` | 0 | none |

Engine row from llama-cli, identical for both shares:

    il=1 pass=2 plans=19 hits=150 upload_ops=78 uploaded_bytes=153600 residents=812

Only 1 `RORET_PASS` row per share (2 total). Distinct pass values were `[2]`
for each share. `per_share_gate_ok` was `0.0=False`, `1.0=False`;
`per_pass_gate_ok=False`; `overall_verdict=NO_GO`; the saved summary has no
`blocker` key (the gate failed closed on missing passes, not on a crash or
parity error).

Raw trace lines, with the identical pattern in both ON runs:

    RORET_BIND no_mulmat layer_il=1 key_il=1 pass=0
    RORET_BIND no_mulmat layer_il=2 key_il=1 pass=1

Each appeared once per graph build (14 builds per run per key), while
`set ... key_il=1 pass=2` and `set ... key_il=-1 pass=-1` succeeded.

## Root cause

The source is `src/llama-moe-cache.cpp` `build_lookup()`, around lines 780-792.
After `get_slot_map` succeeds, each expert weight `w` is probed with
`ggml_backend_supports_op(pimpl->backend(il), ggml_mul_mat_id(ctx0, experts, b,
selected_experts))`. When that probe is false, the code logs `no_mulmat` and
returns `{}` before the `set` registration at about line 820.

Pass-0 and pass-1 MoE slot layers (physical `layer_il` 1 and 2, keys `(1,0)`
and `(1,1)`) therefore fail the MUL_MAT_ID backend-support probe on the
assigned backend. They are reported as `no_mulmat` 28 times (14 each) and never
reach binding registration. Pass 2 (`layer_il=3`, key `(1,2)`) passes the probe
and is the only loop key ever registered, which is why `roret_of.find(...)` can
only hit `(1,2)` and `roret_traffic.add(...)` can only receive `(1,2)`, giving
exactly the single observed row.

The other two instrumented branches are ruled out: `no_experts` (about line
782) and `no_getrows` (about line 807) were both 0. So the deeper cause is not
a missing expert tensor and not an unsupported `get_rows` op.

This refines M21: the missing rows are a **cache-eligibility failure at graph
build time**, not a logging/aggregation artifact and not a key-collapse
problem. The per-graph lookup key introduced in M19 is fine; the earlier-pass
binding is simply never inserted.

## No_slot_map is secondary

`no_slot_map` produced 16 lines per ON run, spread over all four `layer_il`
values including pass 2 and the `(-1,-1)` prelude (4 each). It occurs in some
graph builds and is a separate, minor branch. It is NOT the dominant cause of
the missing pass-0/pass-1 bindings: the `no_mulmat` branch accounts for 28
events, one per pass-0/pass-1 build per key, with no corresponding `set`. Even
in builds where `get_slot_map` succeeded, the MUL_MAT_ID probe still dropped
those two layers. Treating `no_slot_map` as the explanation would be wrong.

## Consequence for issue #5

The 2-layer synthetic Inkling2-loop fixture uses `CORE_START=1`, `CORE_END=2`,
`N_PASSES=3`, giving slot layers `il=1/2/3` for passes 0/1/2. It cannot
currently yield three distinct pass bindings on this CUDA configuration,
because the pass-0/pass-1 slot layers fail the MUL_MAT_ID support probe. No
full per-pass T4 traffic table exists, so issue #5 stays OPEN/BLOCKED.

Options for a future bounded milestone are listed as OPEN; none is claimed to
work:

- (a) make the pass-0/pass-1 slot layers use a backend/op configuration that
  passes `ggml_backend_supports_op` for `ggml_mul_mat_id`;
- (b) regenerate the synthetic fixture with a real multi-layer loop core so
  the pass slot layers have supported expert layouts;
- (c) scope the per-pass table to passes that are actually bindable and record
  the others as structurally ineligible.

Any such change keeps RORET default-off and must re-run the logits/text parity
gate before any further claim.

## Honesty / scope

Final logits/text parity remains PASS-scoped on the synthetic GGUF only
(0 differing bytes at both cpu_share values). Intermediate per-pass MoE tensor
parity is still NOT measured, and real-weight parity is not measured either.
Issue #4 stays closed with its explicit scope note; issue #5 stays OPEN/BLOCKED
with one real engine-side traffic row recorded. All figures above are event
counts or byte-comparison results, not time or throughput. Linked artifacts:
`M21_BINDING_TRACE.md`, `M20_PER_PASS_TRAFFIC.md`, and
`kaggle-engine-parity/M22_SUMMARY.json`, all relative to `tools/roret/`.

No new issue or PR was filed, and no source file was changed by this report.

NO SPEED CLAIM - CycleAudit claim <= ci_low with repeats >= 20 and loops >= 3 remains mandatory.
