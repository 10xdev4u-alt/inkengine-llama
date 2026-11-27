# M23 - Shared cache group for loop slots (2026-11-28)

Status: **FIX VERIFIED ON KAGGLE 2xT4 - FULL PER-PASS TABLE OBTAINED.**
No speed claim. The per-pass T4 table that blocked issue #5 now exists.

## Root cause (refines M22)

M22 established that pass-0/pass-1 slot layers are dropped before binding
registration because `ggml_backend_supports_op` rejects the `MUL_MAT_ID` probe.
The deeper mechanism is a device/group mismatch:

- `src/models/inkling2-loop.cpp::post_arch_tensors` makes loop slot layers
  `il=1/2/3` share the same physical host expert tensors.
- `src/llama-moe-cache.cpp` builds one group per (device, expert layout). With
  `-ngl 99` on 2x T4, the three slots can be assigned to different devices by
  `model.dev_layer(il)`.
- `bindings` is keyed by host expert tensor pointer. Later groups overwrite
  earlier entries, so `get_experts()` returns the **last** group's bank view
  for all three slots.
- But `backend(il)` and `copy()` (`group & g = groups[l.ig]`,
  `GGML_ASSERT(backend == devices[g.id].backend)`) use each slot's **own**
  group. The probe at `ggml-cuda.cu:6381-6388` rejects a CUDA source buffer on
  a different device, so only the slot whose own group matches the bank group
  (pass 2) ever reaches `roret_of.set(...)`.

Changing `backend(il)` alone is not safe: `copy()` would then abort on the
assert because `l.ig` is unchanged. An earlier M23 attempt was reverted for
exactly this reason after independent review.

## Change landed

Commit `48cb7e0` on `main`: four files, 77 insertions / 4 deletions.

- New `src/llama-moe-cache-group.h` with the pure helper
  `llama_moe_cache_shared_group(own_id, shared_id)`.
- `src/llama-moe-cache.cpp`: a local `expert_group` map records the device
  group id that first registered each host expert tensor. A later layer whose
  `experts.front()` is already registered reuses that group id through the
  helper instead of re-resolving `model.dev_layer(il)`. All loop slots over one
  physical layer therefore join one group, so bank view, `backend(il)` and
  `copy()` agree on one device.
- New `tools/roret/test_roret_shared_group.cpp` and its CMake registration
  pin the decision model: old own-group rule binds 1 of 3 loop slots, the
  shared-group helper binds 3 of 3. The helper is called by the engine at
  `llama-moe-cache.cpp:231`, so the test is not detached from production code.

No router, weight, argsort, `selected_experts_in`, mm-id, or graph-node line
was changed. `LLAMA_RORET` gating stays default-off.

## Local evidence

- Two Hy4 Flash subagents produced the root-cause analysis and the patch; two
  independent read-only reviews followed. The first review caught and blocked
  the unsafe `backend(il)`-only approach before it was pushed. The second
  review of the landed change returned SAFE TO PUSH with no CRITICAL/HIGH
  findings.
- Fresh `cmake -S tools/roret -B /tmp/opencode/m23b-roret-build` + build +
  ctest: **10/10** passed, including `roret_shared_group`.
- `g++ -std=c++17 -Werror` and `-std=c++20 -Werror` standalone builds of the
  new test: clean, byte-identical stdout
  `shared group decision: old binds 1/3 loop slots, new binds 3/3`.
- `g++ -fsyntax-only -std=c++17 -I src -I include -I ggml/include
  -DLLAMA_RORET_CPU_TEST src/llama-moe-cache.cpp`: OK.
- Fresh `GGML_CUDA=OFF -DLLAMA_RORET_CPU_TEST=ON` build of `llama-cli` and
  `llama-debug`: 100%, version `1.0.0-dev (build 141, commit 525c791)`.
- Local CPU-test parity on the synthetic GGUF (sha256
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`),
  `-ngl 99 -cmoe --moe-cache-mib 16 --temp 0`, `cpu_share` 0.0 and 1.0:
  final raw F32 logits and token ids 0 differing bytes; generated text 0
  differing bytes at share 1.0; hook ABSENT with `LLAMA_RORET` unset and
  ACTIVE with `LLAMA_RORET=1`. At share 0.0 `llama-cli` segfaults during
  warmup in BOTH off and on runs of the CPU-only harness, while `llama-debug`
  exits 0; this is a harness limitation of the local CPU path, not a RORET
  difference, and no speed or engine-traffic claim is made from it.
- pibs ctest 2/2; odsl ctest 1/1; audit pytest 17 passed; loopkit fast
  pytest 41 passed.

## Kaggle 2xT4 engine verification (M23)

One kernel was spent: `jprinceraj/inkengine-cuda-shared-group-passes` v1,
private, `NvidiaTeslaT4x2`, status COMPLETE. Logs were read with
`kaggle kernels logs`, not the output API. Kernel log sha256
`b9dded3147a6d0886ca9f1e4b77c391b29993520c64e6f9855e576ba1545eb98`;
notebook sha256
`343aafd1f691e7fd52ce32e49301c2819df2db2b3a5a39705b214f5090476c13`.
The notebook cloned `-b main` at HEAD
`48cb7e0d6fdf0ba2a14a70056f5c9fc0ced5f3a5a39705b214f5090476c13`.
nvcc 13.0.88, configure with `-DGGML_CUDA_NO_VMM=ON` exit 0,
`build_ok=true`, fixture sha256
`cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`.

All 8/8 binary runs exited 0. Hook ABSENT for off and ACTIVE for on in both
binaries, with 0 assert lines. Final raw F32 logits, token ids and generated
text all showed **0 differing bytes** for `cpu_share` 0.0 and 1.0.

`RORET_PASS` rows from `llama-cli` (COUNTS, not time):

| cpu_share | il | pass | plans | hits | upload_ops | uploaded_bytes | residents |
|---|---|---|---|---|---|---|---|
| 0.0 | 1 | 0 | 19 | 149 | 80 | 156672 | 0 |
| 0.0 | 1 | 1 | 19 | 150 | 80 | 156672 | 200 |
| 0.0 | 1 | 2 | 19 | 150 | 80 | 156672 | 207 |
| 1.0 | 1 | 0 | 19 | 149 | 80 | 156672 | 0 |
| 1.0 | 1 | 1 | 19 | 150 | 80 | 156672 | 200 |
| 1.0 | 1 | 2 | 19 | 150 | 80 | 156672 | 207 |

`RORET_BIND` summary, identical for both ON runs (281 lines each; both OFF
runs had 0 lines): `set` 56 with 14 each for `(-1,-1)`, `(1,0)`, `(1,1)`,
`(1,2)`; `add` 57 with 19 each for all three passes; `hit` 57 likewise;
`resolve` 76; `no_slot_map` 16 (secondary branch, unchanged); `no_mulmat` **0**;
`no_experts` 0; `no_getrows` 0. This is the direct evidence that the pass-0 and
pass-1 `no_mulmat` failures from M21/M22 are gone.

Strict gate: `per_share_gate_ok` true for both shares, `per_pass_gate_ok` true,
`per_pass_verdict` "PER_PASS_GATE: PASS (counts only, per cpu_share)",
`blocker` empty, `overall_verdict` **PASS**.

## Scope

The ctest is a CPU-only decision model, not an engine measurement. The
synthetic GGUF is a random-weight fixture, not a real model. Final logits and
text parity are PASS-scoped to that fixture. Intermediate per-pass MoE tensor
parity and real-weight parity remain unmeasured. All counts are counts, not
time or throughput.

NO SPEED CLAIM - CycleAudit `claim <= ci_low` with `repeats >= 20` and
`loops = 3` remains mandatory for any published number.
