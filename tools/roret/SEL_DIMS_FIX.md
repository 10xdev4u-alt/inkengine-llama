# M16: device-agnostic selected-expert dims in the MoE cache copy callback

## Root cause

`llama_moe_cache::copy()` found the `GET_ROWS` node whose `src[0]` is the slot
map and took `sel = lookup->src[1]`. It asserted that `sel` is a `RESHAPE` and
read `n_used` and `n_tokens` from `sel->src[0]->ne[]`. On a CUDA scheduler split
the op chain reaching the callback can differ (graph splitting/optimization, or
a reshape that collapsed), so the assert aborted in the `-cmoe` warmup before
any RORET or union code ran. The shape is known at graph-build time, not at
execution time, so reading it from the runtime op chain was the wrong source.

## Fix

`build_lookup()` stores `[n_expert_used, n_tokens]` of `selected_experts` in
`impl::sel_dims`, keyed by the slot-map tensor passed to the callback as `src`
and the flat id count. A slot map is shared by every ubatch graph of a layer,
so the flat id count keeps a decode graph from overwriting the dims of a
prefill graph. `copy()` resolves the dims from that map and validates `n_used*n_tokens ==
ggml_nelements(sel)` with the pure helper in `src/llama-moe-cache-sel.h`; a
non-RESHAPE `sel` now uses `ggml_backend_tensor_get()` instead of the async
path plus backend synchronize. The old `sel->src[0]` dims remain only as a
fallback for manually built graphs whose `sel` is a `RESHAPE`. A debug log
prints the observed op, shape and resolved dims. RORET gating, router, logits,
probs, argsort, weights and graph nodes are unchanged.

## Local verification (all must pass)

```sh
cd <repo root>
cmake -S tools/roret -B /tmp/opencode/m16-roret-build
cmake --build /tmp/opencode/m16-roret-build -j4
ctest --test-dir /tmp/opencode/m16-roret-build --output-on-failure
# expect 7/7 tests passed, including roret_sel_dims

g++ -std=c++17 -Werror -UNDEBUG -Isrc tools/roret/test_roret_sel_dims.cpp -o /tmp/opencode/m16-sel17 && /tmp/opencode/m16-sel17
g++ -std=c++20 -Werror -UNDEBUG -Isrc tools/roret/test_roret_sel_dims.cpp -o /tmp/opencode/m16-sel20 && /tmp/opencode/m16-sel20
```

## Kaggle smoke test - PENDING, NOT RUN

The fix is not verified on GPU. Generate the synthetic MoE GGUF, build with
`GGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON`, then run the same binary, prompt, seed,
`-n`, `-ngl 99` and `--moe-cache-mib` with RORET off and on:

```sh
MODEL=/path/to/inkling2-loop-synthetic.gguf
CLI=<cuda-build>/bin/llama-cli
env -u LLAMA_RORET "$CLI" -m "$MODEL" -p "Hello my name is" -s 0 -n 16 \
  -ngl 99 --moe-cache-mib 256 -v > /tmp/opencode/m16-off.log 2>&1
LLAMA_RORET=1 "$CLI" -m "$MODEL" -p "Hello my name is" -s 0 -n 16 \
  -ngl 99 --moe-cache-mib 256 -v > /tmp/opencode/m16-on.log 2>&1
grep 'RORET_PASS' /tmp/opencode/m16-on.log
grep 'sel il=' /tmp/opencode/m16-on.log
```

Both runs must exit 0, print non-empty generated text, and only the on run may
print `RORET_PASS`. Capture the `sel il=... op=... ne=[...] ...` debug lines.
No result is claimed here: off/on exits, text and rows are PENDING.

## Standing constraints

* NO SPEED CLAIM. `RORET_PASS` rows are counts, not time or tok/s.
* Any future performance claim still requires a real CycleAudit result with
  `claim <= ci_low`, `repeats >= 20`, `loops = 3`.
