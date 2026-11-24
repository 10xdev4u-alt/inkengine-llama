# Inkling2 loop → RORET union-cache wiring: IMPLEMENTED (opt-in)

Status: **wired, building, default-off**. No commit was made and nothing was pushed.

Branch note: the checkout is on local `master` (`origin/master` only); there is no
local or remote `main` branch. Work stayed on the checked-out `master`.

## Verification

Exact commands requested:

```
cmake -S /tmp/build/inkengine-llama -B /tmp/ink-verify3 \
  -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build /tmp/ink-verify3 --target llama-cli -j4
```

Result: **configure OK, build 100% `Built target llama-cli`**. The only warnings are
the pre-existing `-Wdeprecated-enum-enum-conversion` warnings in `tools/mtmd`.

One intermediate build attempt failed because the new trailing arguments were passed
to `build_moe_ffn` twice (`roret_pass` appeared both in its original slot and at the
end). The call was corrected to 23 arguments and the command above was re-run from a
fresh configure; it passed. The fork was never left unbuildable.

## What is wired

```
llama_model_inkling2_loop::graph::graph()          inkling2-loop.cpp
  build_layer(il, ip, ..., ip, pass)                inkling2-loop.cpp (core passes 0..N-1)
    -> llama_model_inkling2::graph::build_layer()   inkling2.cpp
       -> build_moe_ffn(..., roret_key_il=ip,
                        roret_pass=pass)            llama-graph.cpp
          -> moe_cache->build_lookup(...,
                 {logical ip, pass})                llama-graph.cpp
             -> roret_of[slot_map] = {ip, pass}     llama-moe-cache.cpp
  ... graph executes ...
    -> llama_moe_cache::impl::copy()                llama-moe-cache.cpp
       reads the real selected ids back from the device (already done upstream)
       -> roret->clear_layer(ip) on pass 0
       -> roret->add({ip, pass}, ids)
       -> pass > 0: roret->residents(ip) are fed to lru.plan() as protected
                    hints so earlier passes' experts stay resident
```

Nothing in this chain writes to `selected_experts`, changes the router, changes gate
tensors or weights, or adds/reorders graph nodes. `il` remains the physical cache
slot used for `cb(...)` names and per-slot hparams; the logical `ip` is a separate
value carried only in the RORET key.

## Files changed (source)

| File | Change |
|------|--------|
| `src/models/models.h` | `build_layer` gains two trailing defaulted params `int32_t roret_il = -1, int32_t roret_pass = -1` |
| `src/models/inkling2.cpp` | definition accepts them; forwards them to `build_moe_ffn` as `roret_pass` + `roret_key_il` (all earlier args unchanged, including physical `il`) |
| `src/models/inkling2-loop.cpp` | core passes (including pass 0) pass `ip` and `pass`; prelude/epilogue layers keep the defaults |
| `src/llama-graph.h` | `build_moe_ffn` (both overloads) gains trailing `int32_t roret_key_il = -1` |
| `src/llama-graph.cpp` | both definitions and the short-overload forwarding updated; registration block keys on `roret_key_il` instead of the physical slot; `build_lookup` receives `{roret_key_il, roret_pass}` |
| `src/llama-moe-cache.h` | `llama_moe_cache_layer` gains a defaulted `roret_key`; header includes `llama-roret.h` |
| `src/llama-moe-cache.cpp` | opt-in union cache owned by the cache impl (lifetime spans graph execution), `slot_map → (ip, pass)` binding, registration from real host-observed ids in `copy()`, and a default-off `plan()` hook that protects/uploads the union on passes > 0 |

`src/llama-roret.h` is unchanged (the existing `key{il, pass}`, `add`, `residents`,
`clear_layer` API is sufficient).

## Default-off contract

* Without `LLAMA_RORET` set to `1`/`on`/`true`, `llama_moe_cache::impl::roret` is
  null: no union cache exists, `roret_of` stays empty, `copy()` passes
  `extra = nullptr`, and `plan()` executes the original upstream algorithm.
* Loop graphs still pass the logical key, but it is ignored when the cache is off.
* All new parameters are trailing and defaulted; every other model and every
  existing `build_moe_ffn`/`build_layer` caller compiles and behaves unchanged.
* The pre-existing build-time registration hook in `build_moe_ffn` still requires
  an explicitly supplied `roret_cache` + host ids. The loop driver does not supply
  them; it uses the real runtime readback in the MoE cache copy callback instead.

## What exists at runtime, and what does not

**Registration exists (opt-in).** The upstream MoE cache copy callback already
reads the selected expert ids from the device with `ggml_backend_tensor_get_async`
plus a sync. With RORET enabled, those real ids are registered under `(logical ip,
pass)`. That is a genuine host-observed producer; no fabricated ids are used.

**Residency reuse exists (opt-in).** On passes > 0 the union for that logical layer
is passed to `lru.plan()` as protected hints. Resident union experts are promoted
in the LRU and cannot be evicted by that plan; missing union experts are uploaded
like ordinary misses (never routed to the CPU path). Real selected ids are still
the only ids the graph reads through `mm_id`.

**Not measured.** No real Inkling2 model run, no `memcmp` of `ffn_moe_topk`,
`ffn_moe_weights`, `ffn_moe_out`, no logits comparison, and no tok/s benchmark was
performed. **Bit-exact parity is NOT measured.** Correctness of the residency hook
is established by inspection and compile verification only.

## Activation

```
LLAMA_RORET=1 llama-cli -m <inkling2-loop.gguf> --moe-cache-size <bytes> ...
```

The MoE cache must be enabled (`moe_cache_size > 0` and a GPU backend present),
since the copy callback is the producer of host ids. On CPU-only builds the hook
is inert: the MoE cache is not constructed.

## Safety notes

* The union cache is owned by `llama_moe_cache::impl`, not by `llm_graph_context`,
  because `llm_graph_context` is destroyed at the end of `llama_model::build_graph`,
  before graph execution. This makes the pointer used by the copy callback safe.
* `clear_layer(ip)` on pass 0 resets the union per token; pass 0 always executes
  before later passes in the loop graph.
* Protected hints are capped to the group's slot count; real selections are never
  dropped to make room for a hint, so the original "cache too small" failure mode
  is unchanged.
* The hook does not change `seen_gen`-based victim selection for any layer when
  disabled.
* Hints protect a slot only within the layer being planned. The shared bank's LRU
  can still evict another layer's resident when that other layer plans its own
  misses, so the mechanism guarantees the union is considered on each pass, not
  that a tile survives every intervening layer's plan. No throughput win is
  claimed.
