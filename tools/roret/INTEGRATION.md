# RORET integration — opt-in union-resident MoE wiring

> **Scope note (2026-11-24 fact-check).** Upstream llama.cpp already provides the Inkling2 loop graph
> and converter (`src/models/inkling2-loop.cpp`, `src/models/inkling2.cpp`,
> `conversion/inkling2.py`): the loop uses rank-r gain/forget controllers
> (`LLM_TENSOR_LOOP_*_CTRL_{DOWN,GAIN,FORGET}`, rank 64) and
> `n_slot = n_phys + (n_passes-1)*(core_end-core_start)`. Our fork's contribution is union-resident
> MoE routing on top of upstream, not building the loop graph from scratch. Any scalar `gamma` in
> local docs was a simplification; the "outlier at dimension 2027" elsewhere is an unverified local
> hypothesis; not in primary sources.

Status: **applied, wired into the Inkling2 loop graph, and building**. No commit was made.

The router runs on every pass. RORET adds host-side bookkeeping plus an opt-in
residency hint to the MoE cache; it does not alter the router, the selected-experts
tensor, any weight, any matmul, or the graph node order. The feature is off by
default: without `LLAMA_RORET` (or when the MoE cache is not constructed), no union
cache exists and every path is the upstream path.

## Files changed

| File | Change |
|------|--------|
| `src/llama-roret.h` | **new**, header-only C++17 union bookkeeping, no ggml/llama includes; unchanged by the loop wiring |
| `src/llama-graph.h` | forward declaration + 4 trailing defaulted params on both `build_moe_ffn` declarations, later a 5th `roret_key_il`; `#include <cstddef>` |
| `src/llama-graph.cpp` | `#include "llama-roret.h"`; params on both definitions, short-overload forwarding, gated registration block, logical-key-aware `build_lookup` call |
| `src/models/models.h` | `build_layer` gains trailing defaulted `roret_il`/`roret_pass` |
| `src/models/inkling2.cpp` | `build_layer` definition accepts and forwards them to `build_moe_ffn` |
| `src/models/inkling2-loop.cpp` | core passes pass `(ip, pass)` as the logical RORET key |
| `src/llama-moe-cache.h` | `llama_moe_cache_layer` gains a defaulted `roret_key` |
| `src/llama-moe-cache.cpp` | opt-in union cache, slot-map → `(ip, pass)` binding, runtime registration from real host ids, default-off protected-resident hint in `plan()` |

`CMakeLists.txt` was not changed (`llama-roret.h` is header-only).

### Why the API differs from `tools/roret/union_cache.hpp`

This fork builds with `cxx_std_17` (`src/CMakeLists.txt:70`, "don't bump"), so
`std::span` and `<bit>` (`std::countr_zero` / `std::popcount`) are unavailable.
`src/llama-roret.h` is a C++17 port of `union_cache.hpp`: ids are passed as
`const int32_t * + size_t`, and the bit scans are portable loops. The class is
named `llama_roret_union_cache` and lives in the global namespace to match
`llama_moe_cache` and `llm_graph_context` (`llama-graph.h` declares no namespace).
Mutating methods are `const` with `mutable` internal state, so the const
`build_moe_ffn` can use a `const llama_roret_union_cache *` as specified.

## Exact new API

```cpp
// src/llama-roret.h
struct llama_roret_union_cache::key {
    int32_t il   = -1;
    int32_t pass = -1;   // 0..kMaxPass (4), negative disables
};

class llama_roret_union_cache {
public:
    static constexpr int32_t kMaxPass = 4;
    llama_roret_union_cache(int n_experts, int n_top_k);

    int  n_experts() const;
    int  n_top_k()   const;

    bool add(const key & k, const int32_t * ids, size_t n_ids) const;      // true iff union grew
    std::vector<int32_t> residents(int32_t il) const;                     // union, ascending
    double hit_rate(const key & k, const int32_t * ids, size_t n_ids) const; // before add()
    void   clear_layer(int32_t il) const;                                 // mandatory between tokens
    size_t total_residents(int32_t il) const;
};
```

```cpp
// both declarations in src/llama-graph.h, appended after `post_exp_norm_eps`:
       float          post_exp_norm_eps = 1e-5f,
const  llama_roret_union_cache * roret_cache = nullptr,
       int            roret_pass = -1,
const  int32_t *       roret_selected_ids = nullptr,
       size_t         roret_n_selected = 0,
       int32_t        roret_key_il = -1
```

All existing `build_moe_ffn` call sites compile unchanged. The short overload
forwards all five values to the long overload.

`build_layer(int il, int ip, ..., ggml_tensor * messages = nullptr,
const loop_layer * loop = nullptr, int32_t roret_il = -1,
int32_t roret_pass = -1)` forwards `roret_pass` and `roret_key_il = roret_il` to
`build_moe_ffn`. `il` stays the physical cache slot; only the RORET key uses `ip`.

## Reachability and runtime behaviour

1. `llama_model_inkling2_loop::graph::graph()` builds every pass of the loop in one
   graph. For each core layer it calls `build_layer(il, ip, ..., ip, pass)`; `il`
   is the physical slot, `ip` the logical layer, `pass` the loop index.
2. `build_moe_ffn` passes `{roret_key_il, roret_pass}` to
   `llama_moe_cache::build_lookup` in `llama_moe_cache_layer::roret_key`. When RORET
   is enabled, `build_lookup` records `slot_map → (ip, pass)`; no graph node is
   added or reordered.
3. At execution time the upstream copy callback already reads the real selected
   expert ids from the device (`ggml_backend_tensor_get_async` + sync). With RORET
   enabled, `impl::copy()` uses them to `clear_layer(ip)` on pass 0, `add({ip,
   pass}, ids)`, and — for passes > 0 — reads `residents(ip)` and passes them to
   `moe_cache_lru::plan()` as protected hints.
4. In `plan()`, protected hints are deduplicated against the real ids, capped so
   they can never displace a real selection, marked as selected so their slots
   cannot be evicted, and uploaded if missing (never sent through the CPU-miss
   path). `selected_experts` and `mm_id` remain the only source of expert ids for
   the graph's matmuls.

The union cache is owned by `llama_moe_cache::impl`, not by `llm_graph_context`:
the latter is destroyed at the end of `llama_model::build_graph`, before the copy
callback runs, so it would be a dangling pointer.

Enable with:

```
LLAMA_RORET=1 llama-cli -m <inkling2-loop.gguf> --moe-cache-size <bytes> ...
```

`LLAMA_RORET` accepts `1`, `on` or `true`; anything else leaves the hook off. A MoE
cache (GPU backend, `moe_cache_size > 0`) is required because the copy callback is
the producer of host ids. The older build-time hook in `build_moe_ffn` still works
for any driver that supplies `roret_cache` plus host ids directly; the loop driver
does not fabricate ids.

## Bit-exactness checklist

### Verified

1. **Compile safety.** Requested commands were run against `/tmp/ink-verify3`:
   configure OK and build reached **100% `Built target llama-cli`**. Only the
   pre-existing `tools/mtmd` enum-conversion warnings appeared. An earlier attempt
   with a duplicated trailing argument failed; the call was fixed and the exact
   configure + build command passed afterwards.
2. **Default path unchanged by inspection.** With no `LLAMA_RORET`, no union cache
   is constructed, `roret_of` is empty, `extra == nullptr` in every `copy()`, and
   `plan()` runs the original algorithm. Every `build_moe_ffn` / `build_layer`
   default keeps other models on the upstream path.
3. **Router/math untouched by inspection.** No line in `ffn_moe_logits`,
   `ffn_moe_probs`, group masking, `ggml_argsort_top_k`, `ffn_moe_weights`,
   `norm_w`, `w_scale`, `ggml_get_rows(probs, selected_experts)`, `*_b` paths,
   `build_experts`, or `n_expert_used_il` logic was modified. `selected_experts_in`
   is not populated from union ids; hints only affect which tiles are resident.
4. **Header logic.** A standalone C++17 smoke test (`-Wall -Wextra -Werror`)
   exercised multi-word bitmaps, ascending `residents`, hit-rate-before-add
   ordering, idempotent `add`, invalid ids/passes, and `clear_layer`; it passed.

### Not verified — bit-exact parity is NOT measured

1. **No runtime model comparison.** No `memcmp` of `ffn_moe_topk`,
   `ffn_moe_weights` or `ffn_moe_out` across passes 0/1/2 with RORET off vs on, and
   no end-to-end logits comparison was performed. Outputs are expected to be
   identical because the graph is identical, but this has **not been measured**.
2. **No real Inkling2 model run.** No real GGUF was executed, so no tok/s
   improvement or upload-traffic reduction is claimed.
3. **Scheduler split / CPU miss path with RORET on.** `cpu_share = 0` and
   `cpu_share > 0` were not exercised with a model; the protected-hint path was
   reviewed and compiles, but PATCH doc section 5 items 4–7 still need to be run
   before any performance claim.
4. **Cross-layer eviction.** Hints protect slots only for the layer being planned;
   another layer's `plan()` in the same shared bank can still evict them. The union
   is re-applied on every pass, but a tile is not guaranteed to survive an
   unrelated layer's plan. No tok/s or upload-reduction result is claimed.

### Fallback

If any of the unverified items fails once a real model is available, remove the
RORET block from `impl::copy()`, the `extra`/`n_extra` parameters from
`moe_cache_lru::plan()`, and set `LLAMA_RORET` handling to no-op. The non-union
`llama_moe_cache` path is the known-good bit-exact baseline.

## M3 - UNION_WIRING.patch verification

Artifact: `tools/roret/UNION_WIRING.patch`. Generated (no prose header) with
`git diff 5035bdf HEAD -- src/llama-roret.h src/llama-graph.h src/llama-graph.cpp src/llama-moe-cache.h src/llama-moe-cache.cpp src/models/models.h src/models/inkling2.cpp src/models/inkling2-loop.cpp`.
Result: 603 lines, 26215 bytes, sha256
`a91868081fbb9d7657946dfff1c4513171b08131a38a1099934728212bca1e84`, exactly 8
`diff --git` headers, all under `src/`.

Base apply (current HEAD `149a9631e`, `$REPO` = repo root):

```
git worktree add --detach /tmp/opencode/m3-base 5035bdf                    [exit 0]
git -C /tmp/opencode/m3-base apply --check $REPO/tools/roret/UNION_WIRING.patch   [exit 0]
git -C /tmp/opencode/m3-base apply $REPO/tools/roret/UNION_WIRING.patch           [exit 0]
```

`sha256sum` of all 8 files in the applied worktree matched the current HEAD
copies byte for byte. `git -C /tmp/opencode/m3-base diff --stat HEAD -- <8 files>`
was empty (7 tracked files; the new header is untracked after `git apply` and was
checked by hash). `git worktree remove --force /tmp/opencode/m3-base` [exit 0];
main checkout still at `149a9631e`, only the patch file untracked.

Router immutability (patch inspection):

```
grep -E '^[+-]' tools/roret/UNION_WIRING.patch | grep -vE '^(\+\+\+|---)' \
  | grep -E 'ffn_moe_logits|ffn_moe_probs|ggml_argsort_top_k|ffn_moe_weights|selected_experts_in|ggml_get_rows|ggml_add_id|mm_id|selection_probs|ffn_moe_topk|norm_w|w_scale|build_experts\('
+            nullptr,  /* selected_experts_in */
```

That single hit makes the base default explicit: `git show 5035bdf:src/models/inkling2.cpp`
ends the call with `nullptr, layer.ffn_gate_up_exps);`, and the base signature has
`selected_experts_in = nullptr`. No changed line touches logits, gating, group
mask, argsort, weights, `ggml_get_rows(probs, selected_experts)`, `*_b` paths,
`build_experts`, `mm_id`, `n_expert_used_il`, or graph node order.
`post_exp_norm` and `post_exp_norm_eps` keep their defaults.

No-op guards (patch line numbers):

```
63:  if (roret_cache != nullptr && roret_pass >= 0 && roret_selected_ids != nullptr && roret_n_selected > 0)
181: const int32_t * extra = nullptr, size_t n_extra = 0
190: if (extra != nullptr)
267: if (const char * env = getenv("LLAMA_RORET"); env != nullptr)
288: if (roret)
289: const auto ir = roret_of.find(src)
324: if (pimpl->roret && layer.roret_key.il >= 0 && layer.roret_key.pass >= 0)
```

- `pass_index = -1`: the graph block needs `roret_pass >= 0`, build_lookup
  registration needs `roret_key.pass >= 0`, and `valid_pass()` rejects `pass < 0`.
  `build_layer` and `build_moe_ffn` defaults are `-1`. All insertion points no-op.
- Empty ids: the graph block needs a non-null pointer and `roret_n_selected > 0`;
  `add()` rejects `ids == nullptr`, `hit_rate()` returns 0 for `n_ids == 0`. No
  registration and no protected hints. The loop graph passes no fabricated host
  ids; the runtime copy callback supplies the real ids instead.
- Without `LLAMA_RORET` in `{1, on, true}`, `roret` is null, `roret_of` is empty,
  `extra == nullptr`, `n_extra == 0`, and `plan()` runs the original algorithm.
  Hints are appended after the CPU split (never a CPU miss) and capped so they
  cannot evict a real selection; `n_hit` counts real ids only.

Insertion-point mapping to `PATCH_build_moe_ffn.md`: 2.1 -> bookkeeping block
right after `cb(selected_experts, "ffn_moe_topk", il)`; 2.2 -> the `build_lookup`
call in `build_moe_ffn`; 3.1/3.2 -> registration in `build_lookup` plus the
`plan()`/`copy()` hint path. Plumbing is the C++17 port: global
`llama_roret_union_cache`, a defaulted `roret_key` on `llama_moe_cache_layer`,
and extra hint parameters on `plan()` instead of `std::span` arguments or a
separate `pin()` call. Same insertion points; no graph nodes added or reordered.

This is a patch artifact only. No tensor-level off-vs-on parity was measured here;
`BITEXACT_REPORT.md` remains NO_GO.
