# RORET integration — compile-safe union-resident MoE bookkeeping

Status: **applied and building**. No commit was made.

Router still runs on every pass. This change adds only host-side bookkeeping to
`build_moe_ffn`; it does not alter the router, the selected-experts tensor, any
weight, any matmul, or the graph node order. The feature is **off by default**:
when `roret_cache == nullptr` (or `roret_pass < 0`, or no host ids), the added
block is skipped and the original code path is byte-identical.

## Files changed

| File | Change |
|------|--------|
| `src/llama-roret.h` | **new**, header-only C++17 union bookkeeping, no ggml/llama includes |
| `src/llama-graph.h` | forward declaration + 4 trailing defaulted params on **both** `build_moe_ffn` declarations; `#include <cstddef>` |
| `src/llama-graph.cpp` | `#include "llama-roret.h"`; 4 params on both definitions; short overload forwards them; one gated registration block after the top-k `cb` |

`src/llama-moe-cache.h/.cpp` were **not** changed. `CMakeLists.txt` was not
changed (the new header is header-only, so no build-system edit is needed).

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
// both declarations in src/llama-graph.h (appended after `post_exp_norm_eps`):
       float          post_exp_norm_eps = 1e-5f,
const  llama_roret_union_cache * roret_cache = nullptr,
       int            roret_pass = -1,
const  int32_t *       roret_selected_ids = nullptr,
       size_t         roret_n_selected = 0
```

All existing `build_moe_ffn` call sites compile unchanged. The short overload
(`src/llama-graph.cpp:2321`) forwards all four values to the long overload
(`src/llama-graph.cpp:2363`).

## Exact call sites

- `src/llama-graph.cpp:2493-2505` — registration block, inserted immediately
  after `cb(selected_experts, "ffn_moe_topk", il);` and **before** the GROVEMoE
  id remap and `ggml_get_rows(ctx0, probs, selected_experts)`. Order is
  intentional: `hit_rate()` and `residents()` are read **before** `add()`, so
  pass N observes only earlier passes. Nothing in this block writes to any
  ggml tensor or changes `selected_experts`.
- `src/llama-graph.cpp:2340-2343, 2364-2367` — short overload signature and
  forwarding call.
- `src/llama-graph.cpp:2395-2398` — long overload signature.
- `src/llama-graph.h:1284-1287, 1313-1316` — declarations.
- The values `roret_selected_ids` must be **host-observed** top-k ids supplied
  by the loop driver. The device tensor `selected_experts` is produced by
  `ggml_argsort_top_k` and cannot be read on the host at graph-build time; it is
  never substituted or faked. Empty ids mean the whole patch is a no-op.

To use it, a looped model driver constructs one `llama_roret_union_cache`,
passes `&cache`, the pass index (`0..4`) and the host top-k ids to each pass's
`build_moe_ffn`, then calls `clear_layer(il)` for the looped layers at the end
of each token. `cache.residents(il)` is the union available to a residency /
warm-up mechanism on subsequent passes.

## Bit-exactness checklist — verified vs not

### Verified

1. **Compile safety.** Exact commands requested were run:
   - `cmake -S /tmp/build/inkengine-llama -B /tmp/ink-verify -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF` — configure OK.
   - `cmake --build /tmp/ink-verify --target llama-cli -j4` — **100%, built target llama-cli**, no errors attributable to this change (only pre-existing `-Wdeprecated-enum-enum-conversion` warnings in `tools/mtmd`).
2. **Default path unchanged by inspection.** With the default arguments, all four `if` conditions are false, so no cache method is called and no tensor is touched.
3. **Router/math untouched by inspection.** The diff adds only declarations, parameter forwarding, and one gated bookkeeping block. No line in the router (`ffn_moe_logits`, `ffn_moe_probs`, group mask, `ggml_argsort_top_k`, `ffn_moe_weights`, `norm_w`, `w_scale`), in `ggml_get_rows(probs, selected_experts)`, in `*_b` paths, in `build_experts`, or in the `n_expert_used_il` / precision / random-routing paths was modified. `selected_experts_in` remains `nullptr` on every pass; union ids never enter it.
4. **Header logic.** A standalone C++17 smoke test (`-Wall -Wextra -Werror`) exercised multi-word bitmaps, ascending `residents`, hit-rate-before-add ordering, idempotent `add`, invalid ids/passes, and `clear_layer`; it passed.

### Not verified (required before any performance or bit-exactness claim)

1. **No runtime model comparison.** No `memcmp` of `ffn_moe_topk`, `ffn_moe_weights`, or `ffn_moe_out` across passes 0/1/2 with the feature off vs on. No end-to-end logits comparison. The graph is expected to be identical, but this was not measured.
2. **No residency wiring.** `llama_moe_cache::build_lookup`, the copy callback / LRU `plan`, and `build_experts` are unchanged. `residents(il)` is exposed but no code pins or prefetches those tiles, so **no upload-traffic reduction or tok/s win exists yet**. Wiring it (PATCH doc sections 3.1–3.2, or a warm-up lookup) is a separate change and must be done with the bit-exactness checks below.
3. **Scheduler splits / CPU miss path / slot-map consistency** were not exercised, since the graph is unchanged and no pins are applied. If residency wiring is added later, re-run PATCH doc section 5 items 4–7, including `cpu_share = 0` first and then `cpu_share > 0`.
4. **No real Inkling 2 Mini router readback.** The loop driver that supplies host top-k ids is not part of this change; until it exists, the feature is structurally opt-in but has no producer of ids in-tree.

### Fallback

If any of the unverified items fails once residency wiring exists, follow
PATCH doc section 6: remove the block at `src/llama-graph.cpp:2493-2505`,
drop the four params, and keep `roret_cache = nullptr`. The non-union
`llama_moe_cache` path is the known-good bit-exact baseline.
