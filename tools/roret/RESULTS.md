# RORET probe results — 2026-11-24

Standalone C++ probe in this directory, 3,000 synthetic tokens per row,
128 experts / top-8, dim 256, γ = [0, 0.3, 0.6], deterministic seed 20261124.

| msg_scale | exact top-8 agreement (all 3 passes) | mean top-8 overlap (1→2) | mean union experts | load reduction vs 24 | pass-2/3 hit rate from pass-1 | verdict (exact) | verdict (union) |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.760 | 0.985 | 8.25 | 65.6% | 97.7% | NO_GO | **GO** |
| 0.10 | 0.578 | 0.969 | 8.47 | 64.7% | 95.7% | NO_GO | **GO** |
| 0.30 | 0.165 | 0.918 | 9.28 | 61.3% | 88.2% | NO_GO | **GO** |
| 1.00 | 0.003 | 0.761 | 11.58 | 51.8% | 66.9% | NO_GO | NO_GO |

## Conclusion

The original RORET hypothesis — route once and reuse the identical top-8 set —
is **false** under this model: even with a tiny cross-loop message only ~76% of
tokens keep the exact same experts across all three passes. The reason is that
top-k boundaries are unstable even when the underlying logits barely move.

The refined design is **union-resident routing**: still route on every pass, but
keep the *union* of the three top-8 sets resident in llama.cpp's existing
`llama_moe_cache`. With realistic message scales the union is only 8.3–9.3
experts versus 24 re-routed expert activations — a **~61–66% reduction in expert
loads**. Pass-2/3 hit rates against pass-1's set are 88–98%, so the cache is
mostly warm even before the union is built.

At `msg_scale = 1.0` (unrealistically large relative to the learned gate) the
union grows to 11.6 and the win drops to 52%. This is the stress case and the
honest lower bound.

## Caveats

- The router here is synthetic: random Gaussian prelude and an independent
  random message. Real cross-loop messages are a function of the same residual
  stream, so drift should be smaller — but this must be verified with NDIF /
  Spectral Health Checker readouts on the real model before claiming a speedup.
- This is a load-count result, not wall-clock. No tok/s number is claimed.
- Any eventual speed claim must be verified with loopkit's CycleAudit and be
  below its lower 95% confidence bound.

## Next

Wire the union-resident lookup into `llm_graph_context::build_moe_ffn`
(`src/llama-graph.cpp:2320`, `:2362`) keyed by `(il, pass_index)`, using the
existing `llama_moe_cache` slot map (`src/llama-moe-cache.h`).
