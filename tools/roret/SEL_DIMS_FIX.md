# M16: device-agnostic selected-expert dims in the MoE cache copy callback

## Root cause

`llama_moe_cache::copy()` found the `GET_ROWS` node whose `src[0]` is the slot
map and took `sel = lookup->src[1]`. It asserted that `sel` is a `RESHAPE` and
read `n_used` and `n_tokens` from `sel->src[0]->ne[]`. On a CUDA scheduler split
the op chain reaching the callback can differ: the kernel below observed 68
`op=NONE` and 68 `op=RESHAPE` reads on 2x T4. The `RESHAPE`-only assert aborted
the `-cmoe` warmup before any RORET or union code ran, with RORET both off and
on. The shape is known at graph-build time, not at execution time, so reading
it from the runtime op chain was the wrong source.

## Fix

`build_lookup()` stores `[n_expert_used, n_tokens]` of `selected_experts` in
`impl::sel_dims`, keyed by the slot-map tensor and the flat id count. The flat
count is part of the key because a slot map is shared by graphs with different
ubatch sizes: a decode graph with `n_tokens=1` must not overwrite the dims of a
prefill graph with `n_tokens=N`. `copy()` resolves the dims from that store and
validates `n_used*n_tokens == ggml_nelements(sel)` with the pure helper in
`src/llama-moe-cache-sel.h`; a non-`RESHAPE` `sel` now uses
`ggml_backend_tensor_get()` instead of the async path plus backend synchronize.
The old `sel->src[0]` dims remain only as a fallback for manually built graphs
whose `sel` is a `RESHAPE`. A debug log prints the observed op, shape and
resolved dims. RORET gating, router, logits, probs, argsort, weights and graph
nodes are unchanged.

## Local verification (2026-11-27, all passed)

```sh
cmake -S tools/roret -B /tmp/opencode/m16-roret-parent
cmake --build /tmp/opencode/m16-roret-parent -j4
ctest --test-dir /tmp/opencode/m16-roret-parent --output-on-failure
# 7/7: roret sel_dims, traffic, tile_bitexact, union_census, pass_traffic, fixture_parity, union_cache

g++ -std=c++17 -Werror -UNDEBUG -Isrc tools/roret/test_roret_sel_dims.cpp -o /tmp/opencode/m16-sel17
g++ -std=c++20 -Werror -UNDEBUG -Isrc tools/roret/test_roret_sel_dims.cpp -o /tmp/opencode/m16-sel20
# both: 13/13 cases, SUMMARY_FNV1A64=fa9d5039ee3e6568
```

Also: pibs ctest 2/2, odsl ctest 1/1, `tools/audit` pytest 17 passed, loopkit
fast pytest 40 passed 1 deselected, incremental `GGML_CUDA=OFF` `llama-cli`
build 100 percent (version 1.0.0-dev build 129 commit 8ad8a1d). All new lines
are ASCII.

## Kaggle 2xT4 CUDA smoke test (2026-11-27, DONE)

One kernel this pass: `jprinceraj/inkengine-cuda-sel-dims-probe` v1, new slug
because the title changed. Notebook
`/tmp/opencode/kaggle-cuda-sel-dims/cuda_sel_dims.ipynb` sha256
`042c3605d1eca57f04bca2b0e4ec3fd7db44050cc180cef6e3003ccbe5db4188`. Logs were
read with `kaggle kernels logs`, not the output API. Kernel log sha256
`5af929dbbd2506e9b7965d2a0c7ac35b33a7a1f0d4520a924772542e9aa38496`; stdout
extract sha256 `392521c126ddbd5a6441fa7f8ea701630d1dbccdad4656924340e2b5015146e7`.

* clone `-b main` HEAD `ca1f4f18622f1869adde435e8fd576920dde429b`, exit 0.
* configure with `-DGGML_CUDA_NO_VMM=ON`: exit 0; `cmake --build ... --target
  llama-cli -j4`: exit 0; `build_ok = true`; nvcc 13.0.88, GNU 13.3.0, 2x T4.
* Synthetic GGUF generator exit 0, fixture sha256
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`.
* `llama-cli` off and on (same prompt, seed, `-n 16`, `-t 1`, `-ngl 99`,
  `-cmoe`, `--moe-cache-mib 16`, `--temp 0`, `--verbose`): `off_exit = 0`,
  `on_exit = 0`. Hook ABSENT off, ACTIVE on. Zero assert/abort lines in both.
* Generated text: `off.txt` and `on.txt` sha256 both
  `cf915b84cf60abfd20ced6ac8ec7e69a72c2f0c4b0e2f6abbe8c9c9130bbec3d`;
  `text_parity_bytes_diff = 0`. Scope: generated text on this synthetic GGUF,
  not tensor or logit parity.
* `RORET_PASS` rows 1: `il=1 pass=2 plans=19 hits=150 upload_ops=78
  uploaded_bytes=153600 residents=812`. `engine_copy_exercised = true`,
  `engine_verdict = ENGINE_COPY_PATH_EXERCISED: YES`.
* Debug evidence for the root cause: 136 `sel il=...` lines across off and on,
  68 `op=NONE` and 68 `op=RESHAPE`, all `ok=1`, with shapes including
  `ne=[8,1,1]` (`n_tokens=1` warmup), `ne=[16,1,1]` (`n_tokens=2`),
  `ne=[32,1,1]` and `ne=[80,1,1]`. The `op=NONE` reads confirm the old
  `GGML_ASSERT(sel->op == GGML_OP_RESHAPE)` was the wrong precondition; they
  now take the device-agnostic readback path.

This resolves the blocker recorded in issue #8: the engine completes a CUDA
`-cmoe` run with RORET both off and on. It does not by itself prove F32 tensor
or logit parity (issue #4) or a full off/on traffic table (issue #5); those
remain NO_GO at this increment; M18 later closed the engine final-logits part
of issue #4 as PASS-scoped on 2xT4, with intermediate tensors still NOT measured.
`RORET_PASS` figures are counts, not time or tok/s.

## Standing constraints

* NO SPEED CLAIM. `RORET_PASS` rows are counts, not time or tok/s.
* Any future performance claim still requires a real CycleAudit result with
  `claim <= ci_low`, `repeats >= 20`, `loops = 3`.
* No weights are held; every check here uses the deterministic synthetic
  random-weight MoE fixture.
