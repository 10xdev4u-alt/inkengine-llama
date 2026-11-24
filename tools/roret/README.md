# RORET — Route Once, pass-Resident Expert Tiles

Inkling 2 Mini runs the middle 12 layers three times per token. The cross-loop
gate `gamma` starts at 0, so pass 1 is a non-looping forward; passes 2–3 add a
learned message. **Hypothesis:** the top-8 expert set is largely unchanged across
the three passes. If true, we route once and keep the expert tiles resident,
removing redundant routing and weight activations.

## Precondition probe (this directory)

Standalone, builds without llama.cpp:

```bash
cmake -S tools/roret -B /tmp/roret-build && cmake --build /tmp/roret-build -j
/tmp/roret-build/roret_probe --tokens 2000 --gamma2 0.3 --gamma3 0.6
```

Outputs both `exact_rate_all` (identical top-8 on all three passes) and
`mean_union_experts` (size of the union across passes). See [RESULTS.md](RESULTS.md):
exact route-once is **NO_GO**, union-resident routing is **GO** with a 61–66%
reduction in expert loads. The synthetic router is a conservative stand-in:
real cross-loop messages derive from the same residual stream, so drift should
be smaller — but this must be verified on the real model.

## Integration points in this llama.cpp fork

1. `src/llama-moe-cache.h` — already keeps recently/repeatedly used experts in a
   device buffer with a per-layer slot map. RORET reuses **selected expert IDs**
   from the first pass, so the cache hits on passes 2–3 instead of re-routing.
2. `src/llama-graph.cpp:2320` and `:2362` — `llm_graph_context::build_moe_ffn`.
   Add an optional `selected_experts_override` input; when present, skip the
   router top-k and use the supplied IDs.
3. Model graph for the looped architecture: build blocks 2–3 once, then replay
   the same graph three times with pass-indexed KV slots and a message tensor.
   The cache lookup from step 1 must be keyed by `(il, pass_index)`.

## Honesty

Nothing here is a speedup yet. The probe measures a precondition; the result
must be reproduced on real router activations (via `loopkit`'s Spectral Health
Checker or NDIF hooks) before touching the graph. On T4, any speed claim must be
verified with CycleAudit and reported below its lower 95% confidence bound.
