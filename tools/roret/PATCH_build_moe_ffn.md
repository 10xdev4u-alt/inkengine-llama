# Patch instructions — union-resident expert tiles (RORET)

> **Scope note (2026-11-24 fact-check).** Upstream llama.cpp already provides the Inkling2 loop graph
> and converter (`src/models/inkling2-loop.cpp`, `src/models/inkling2.cpp`,
> `conversion/inkling2.py`): the loop uses rank-r gain/forget controllers
> (`LLM_TENSOR_LOOP_*_CTRL_{DOWN,GAIN,FORGET}`, rank 64) and
> `n_slot = n_phys + (n_passes-1)*(core_end-core_start)`. Our fork's contribution is union-resident
> MoE routing on top of upstream, not building the loop graph from scratch. Any scalar `gamma` in
> local docs was a simplification; the "outlier at dimension 2027" elsewhere is an unverified local
> hypothesis; not in primary sources.


Status: **instructions only, not applied to `src/`**. Only new files are
`tools/roret/union_cache.hpp`, `tools/roret/test_union_cache.cpp`, and this
document. Apply the changes below in a later pass, on a branch where you can
run the full decode comparison in section 6.

## Hard scope constraint — read first

The router **still runs on every loop pass**. This patch does **not** implement
the old "route once, reuse top-k" idea (RESULTS.md: route-once is NO_GO). On
each pass, `build_moe_ffn` must still compute `ffn_moe_logits`, `ffn_moe_probs`,
the group mask (if any), `ggml_argsort_top_k`, `ffn_moe_weights`, and all 8
expert matmuls. `selected_experts_in` stays `nullptr` on every pass; union ids
must **never** be fed into that parameter.

The only thing that changes is **residency**: the distinct expert weight tiles
kept in the `llama_moe_cache` device buffer is the union of the three passes'
top-k sets (8.25–11.58 experts) instead of 24 freshly uploaded tiles. This
saves host→device tile uploads / cache thrash; it does not reduce the number of
top-k or matmul operations, and it is not yet a tok/s claim.

## 0. New header, no build-system change needed

`tools/roret/union_cache.hpp` is header-only, C++20, has no ggml/llama includes
and is not thread-safe (graph build is single-threaded). Include it from
`src/llama-graph.cpp` with a path relative to the source file:

```cpp
// src/llama-graph.cpp, with the other includes
#include "../../tools/roret/union_cache.hpp" // ink::roret::UnionCache
```

Add `#include <span>` where needed (`src/llama-graph.h` and
`src/llama-moe-cache.h`) if it is not already transitively available.

## 1. Context state and the two `build_moe_ffn` signatures

`src/llama-graph.h`: forward-declare the type and add a context member plus the
two new trailing parameters on **both** `build_moe_ffn` declarations. The
`const` build method may mutate through the pointer; that is intentional.

```cpp
// near the top of llama-graph.h
namespace ink::roret { class UnionCache; }

// in struct llm_graph_context
std::shared_ptr<ink::roret::UnionCache> roret_union_cache; // null = feature off
```

`src/llama-graph.cpp:2320` — short overload: add the same two trailing params
and forward them to the long overload at `:2362`:

```cpp
ggml_tensor * llm_graph_context::build_moe_ffn(
         /* ... existing args ... */
         ggml_tensor * post_exp_norm,
               float   post_exp_norm_eps,
              int8_t   pass_index = -1,                    // NEW: -1 = non-looped
  std::span<const int32_t> host_selected_ids = {}) const { // NEW: host-observed top-k
    return build_moe_ffn(
        /* ... existing args ... */
        post_exp_norm,
        post_exp_norm_eps,
        pass_index,           // NEW
        host_selected_ids     // NEW
    );
}
```

`src/llama-graph.cpp:2362` — long overload: add the same two params after
`post_exp_norm_eps`. All existing call sites keep compiling because both have
defaults; the feature is off unless the caller passes `pass_index >= 0`.

`host_selected_ids` is the id list **observed on the host** for this pass
(e.g. supplied by the loop driver from a router readback). It is bookkeeping
input only: it is never substituted for the device `selected_experts` tensor.
When it is empty, every new block below is skipped and the old path is exact.

## 2. Where `add` / `hit_rate` and `residents` go in `build_moe_ffn`

### 2.1 After the router top-k, before the weights — `src/llama-graph.cpp:2472-2478`

Current code:

```cpp
    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);
```

Insert **immediately after** line 2478 and before `ggml_get_rows` at `:2489`.
Order matters: `residents()` and `hit_rate()` are read **before** `add()`, so
pass 1 sees pass 0's union, pass 2 sees passes 0–1, etc. Reading them after
`add()` would trivially return a 100% hit rate and would invalidate telemetry.

```cpp
    // RORET: host-side union bookkeeping only. The router above ran normally;
    // selected_experts and all weights below are unchanged.
    const ink::roret::UnionKey ukey{il, pass_index};
    std::vector<int32_t> resident_before;
    double union_hit = 0.0;
    if (roret_union_cache != nullptr && pass_index >= 0 && !host_selected_ids.empty()) {
        resident_before = roret_union_cache->residents(ukey);          // BEFORE add
        union_hit       = roret_union_cache->hit_rate(ukey, host_selected_ids);
        (void) roret_union_cache->add(ukey, host_selected_ids);        // OR into union
        LLAMA_LOG_DEBUG("roret: il=%d pass=%d hit=%.3f union=%zu\n",
                il, pass_index, union_hit, roret_union_cache->total_residents(il));
    }
```

`hit_rate` is the observable from RESULTS.md (expected ~0.88–0.98 for
passes 2–3 at realistic message scales). `add` returning `true` means the union
grew; in steady state after warm-up it should be `false` most passes. Keying is
`UnionKey{il, pass_index}` with `pass_index` in `0..4` (Inkling 2 Mini uses
`0,1,2`). The union itself is per-layer, so `residents({il,0}) ==
residents({il,2})` once all three passes are registered — that is intentional.

### 2.2 Feed the union to the cache lookup — `src/llama-graph.cpp:2547-2549`

Current code:

```cpp
    const llama_moe_cache_lookup cache = moe_cache == nullptr ? llama_moe_cache_lookup{} :
        moe_cache->build_lookup(*this, { il, up_exps, gate_exps, down_exps, gate_up_exps, weight_before_ffn }, selected_experts);
```

Change only the trailing argument (see section 3 for the new parameter):

```cpp
    const llama_moe_cache_lookup cache = moe_cache == nullptr ? llama_moe_cache_lookup{} :
        moe_cache->build_lookup(*this,
            { il, up_exps, gate_exps, down_exps, gate_up_exps, weight_before_ffn },
            selected_experts,
            pass_index >= 0 ? std::span<const int32_t>(resident_before) : std::span<const int32_t>{});
```

`resident_before` lives until the end of `build_moe_ffn`, so the span is valid
across both `build_lookup` and `build_experts`. `selected_experts` remains the
router output; the cache lookup still indexes the slot map with those ids.

### 2.3 `build_experts` — `src/llama-graph.cpp:2711-2716`: **no change**

Keep this call exactly as it is:

```cpp
    ggml_tensor * experts = cache.slots == nullptr ?
        build_experts(cur, selected_experts, nullptr, true) :
        moe_cache->build_experts(*this, cache, cur, selected_experts,
                [](ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * mm_ids, bool cached, void * user_data) {
                    return (*static_cast<decltype(build_experts) *>(user_data))(cur, ids, mm_ids, cached);
                }, &build_experts);
```

In particular, `mm_ids = cache.slots` stays derived from the **real** selected
experts. Do not pass `resident_before` as `ids` or `mm_ids` here: `ids` drives
`ggml_add_id` for biases/scales and would change numerics.

## 3. MoE cache API — `src/llama-moe-cache.h:22-27, :49-54`

Extend the lookup struct with a 4th field and `build_lookup` with a 4th
argument. `build_experts` keeps its current signature (`:53-54`).

```cpp
// src/llama-moe-cache.h:22
struct llama_moe_cache_lookup {
    int32_t       il      = -1;
    ggml_tensor * slots   = nullptr;
    ggml_tensor * cpu_ids = nullptr;
    std::vector<int32_t> resident_ids; // NEW: union ids to pin/prefetch; empty = old behavior
};

// src/llama-moe-cache.h:49
llama_moe_cache_lookup build_lookup(const llm_graph_context & g,
        const llama_moe_cache_layer & layer,
        ggml_tensor * selected_experts,
        std::span<const int32_t> resident_ids = {}) const; // NEW

// :53-54 — UNCHANGED
ggml_tensor * build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
        ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const;
```

### 3.1 `build_lookup` — `src/llama-moe-cache.cpp:891-947`

Keep every existing validity check and the `ggml_get_rows(ctx0, slot_map, ids)`
at `:929` exactly as-is. Only change the return at `:946` to carry the union
ids, and register them as "pin these experts" in the cache pimpl (a new
`void set_pinned(int32_t il, std::span<const int32_t> ids)` on `impl`, stored
per layer; `pimpl` is writable from this `const` method):

```cpp
    // after the slot_map != nullptr check at :902-905
    pimpl->set_pinned(il, resident_ids); // empty span clears pins -> old behavior

    // ... existing checks :906-945 unchanged; slot lookup still uses `ids` ...

    // :946, was: return { il, ggml_reshape_2d(...), cpu_ids };
    std::vector<int32_t> keep(resident_ids.begin(), resident_ids.end());
    return { il, ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]), cpu_ids, std::move(keep) };
```

The slot map is still a host weight and the copy callback still runs on its
copy; the pinned ids only make the copy callback upload the union experts
*before* they are needed, so the three passes hit instead of re-loading.

### 3.2 Copy callback — `src/llama-moe-cache.cpp:790-822`

At `:820`, `g.lru.plan(il, ids.data(), ids.size(), ...)` must keep receiving the
**actual** `sel` ids (the router output) — never `resident_ids`. Add the pinned
set as pre-resident state for this layer before planning:

```cpp
        g.lru.pin(il, l.pinned);                    // NEW: union tiles resident
        if (!g.lru.plan(il, ids.data(), ids.size(), sel->src[0]->ne[1],
                        n_used, n_zero, n_cpu, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
```

`n_hit` / `bytes` at `:852-857` become the traffic counters to compare against
RESULTS.md (expect roughly 24 distinct uploads to drop to the union size).
If your LRU has no `pin` operation, an acceptable alternative is a warm-up
lookup built by the loop driver whose ids tensor is the union ids, executed
before pass 1; the real per-pass lookups stay unchanged. In that case do not
also add the `pin` call — pick one mechanism.

### 3.3 `build_experts` — `src/llama-moe-cache.cpp:949-963`: **no functional change**

Keep `build(cur, selected_experts, lookup.slots, true, ...)` at `:953` and the
cached+CPU merge at `:958` byte-for-byte. Residency is decided in
`build_lookup`/the copy callback; the expert matmuls and their `mm_ids` stay
the same.

## 4. Loop driver: keying and reset

Key every registration by `(il, pass_index)`; `pass_index` is `0,1,2` for the
three Inkling 2 Mini loop passes, `3,4` reserved, `-1` disables the feature.
Gate the patch on the looped arch so every other model takes the original path.

```cpp
ink::roret::UnionCache union_cache(n_expert, n_expert_used); // e.g. 128, 8
ctx.roret_union_cache = &union_cache;

for (int32_t token = 0; token < n_tokens; ++token) {
    for (int8_t pass = 0; pass < 3; ++pass) {
        // host-observed top-k for THIS pass; if unavailable, pass {} and the
        // whole patch is a no-op for this pass (safe fallback).
        auto ids = host_topk_ids_for(pass);
        build_looped_layers(ctx, /*pass_index =*/ pass, /*host_selected_ids =*/ ids);
        decode(ctx);
    }
    // Mandatory: otherwise the union accumulates across tokens and grows to
    // all n_experts, destroying the win.
    for (int32_t il : looped_layers) {
        union_cache.clear_layer(il);
    }
}
```

`llama_decode` does not normally expose router ids on the host. If the loop
driver cannot supply them, leave `host_selected_ids` empty: no `add`, no pins,
bit-exact old behavior. Do not fake the ids by reading `selected_experts` at
graph-build time — it is a device tensor produced by `ggml_argsort_top_k`.

## 5. Bit-exactness checklist (all required before claiming anything)

1. **Router unchanged.** Diff-check `src/llama-graph.cpp:2391-2478`: logits,
   gating function, `exp_probs_b`, group masking, `ggml_argsort_top_k`,
   `probs`, `weights`, `norm_w`, `w_scale` are untouched. No call passes union
   ids via `selected_experts_in`.
2. **All three passes still compute.** Dump `ffn_moe_topk`, `ffn_moe_weights`
   and `ffn_moe_out` tensors for passes 0/1/2 with the feature on and off;
   they must be byte-identical F32 (`memcmp`, 0 ULP), not "close".
3. **Ids never replaced.** `ggml_get_rows(ctx0, probs, selected_experts)` at
   `:2489`, `ggml_add_id(..., down_exps_b, selected_experts)` at `:2719`, and
   every `*_b` path (`:2571`, `:2586`, `:2598`) still use the real router ids.
   Union ids may only reach the LRU pinning / warm-up lookup.
4. **Slot map still consistent.** For identical router ids, dump `slot_map`
   before and after pinning. Pins may change which experts are resident, but
   `ggml_get_rows(slot_map, ids)` must return the slot holding that same
   expert; `copy` at `:792-861` must not map a different expert into `slots`.
5. **CPU miss path.** First verify with `cpu_share = 0` (no `cpu_ids` merge at
   `:958`). Then enable `cpu_share > 0`: `lookup.cpu_ids` must still encode the
   same misses, and the cached+CPU `ggml_add` sum order must be identical. If
   pinning changes the CPU/cached split and flips a bit, restrict pinning to
   device-resident layers and leave `cpu_part` untouched.
6. **Scheduler splits.** `llama-moe-cache.cpp:792-798` aborts if the selected
   ids are computed in the same split as the lookup. Any warm-up/pin node must
   be in an earlier split and must not move `sel` or change `gf` node order for
   the router or weights (`ggml_build_forward_expand` at `:2528`, `:2545`,
   `:2735`, `:2749`, `:2758`).
7. **Graph shape/precision.** `n_expert_used_il` view loop at `:2745-2759`,
   `post_exp_norm`, `use_amax` / `prec_policy` at `:2692-2699`, and the
   `moe_random_routing` benchmark path at `:2530-2542` unchanged. No new
   tensor shapes; union ids must not be mixed with random-routing ids.
8. **End-to-end.** Same seed/prompt/ubatch, `memcmp` on F32 output logits and
   on `ffn_moe_out` dumps, feature off vs on: expect 0 differing bytes. Also
   run with `pass_index = -1` to confirm the default path is unchanged.
9. **Traffic claim (separate from correctness).** Compare `bytes` uploaded
   (`:836-856`) and `n_hit` (`:819`) with the union off/on. Confirm the
   distinct uploaded tiles drop from ~24 toward the measured union size
   (8.25–11.58). This is a load-count result, not wall-clock; any speed claim
   still needs real-model router readouts and CycleAudit's lower 95% bound.

## 6. Fallback

If any item in section 5 fails, revert the four insertion points (2.1, 2.2,
3.1, 3.2) and keep `roret_union_cache = nullptr`. The existing non-union
`llama_moe_cache` path is the known-good bit-exact baseline.
