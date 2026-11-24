# PIBS — pass-interleaved batch scheduler model

A standalone C++20 scheduling model for Inkling 2 Mini's middle-block loop
(12 layers repeated 3 times per token). No llama.cpp dependency, no GEMM is
actually executed.

## Model

`n_tokens` × 3 work items `(token_id, pass_index, ready)`. Dependency:
`(t, p+1)` becomes ready only after `(t, p)` is done. There are no cross-token
dependencies.

A scheduling step folds the selected ready items into one logical GEMM:

- `M` = number of folded items
- `padded_m16 = ceil(M/16)*16` (T4 FP16 WMMA m16 tiles)
- `wasted = padded_m16 - M`
- `efficiency = M / padded_m16`

Schedulers:

| scheduler | rule | M per step | steps |
| --- | --- | --- | --- |
| `sequential` | finish all 3 passes of token `i` before token `i+1`; the 3 passes are folded into one fused step (pass order kept inside the step) | 3 | `n_tokens` |
| `interleaved` | one pass level across all tokens per step | `n_tokens` | 3 |

The table reports total wasted tile slots (summed over steps) and aggregate
efficiency `sum(M) / sum(padded_m16)`. For these two schedules `M` is uniform,
so aggregate efficiency equals the per-step `M / ceil(M/16)*16`.

## Honest scope

This is a scheduling model, not a measured speedup. Under the stated
intra-token pass dependency, a token can have at most one pass ready at a
time, so `M <= n_tokens`: the "batch × 3" rows in a single GEMM would require
relaxing that dependency and is *not* modelled here. What interleaving does
change is row count per GEMM (batch instead of 3) and step count (3 instead of
`n_tokens`). At `batch=1` interleaving is strictly worse on wasted slots
(3 steps of M=1 vs one step of M=3); the model does not claim otherwise.

Correctness assertions check both schedulers execute exactly `3*n_tokens`
items, execute each item exactly once, and maintain `seq(t,0) < seq(t,1) <
seq(t,2)` for every token.

## Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/pibs
# or: ctest --test-dir build --output-on-failure
```

`RESULTS.md` contains actual output from this build.
