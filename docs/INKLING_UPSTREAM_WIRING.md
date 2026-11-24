# Inkling2 loop → RORET union-cache wiring: BLOCKED (scope limit)

Status: **not wired**. No source file was edited for this attempt. The tree still
contains only the earlier opt-in `build_moe_ffn` hook
(`src/llama-roret.h`, `src/llama-graph.h`, `src/llama-graph.cpp`).

Note on branch: the checkout is on `master` (`origin/master` only); there is no
local or remote `main` branch. Work was done on the checked-out `master`.

## Verification run (unchanged tree)

```
cmake -S /tmp/build/inkengine-llama -B /tmp/ink-verify2 \
  -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build /tmp/ink-verify2 --target llama-cli -j4
```

Result: **configure OK, build 100% `Built target llama-cli`**, no errors.
Only pre-existing `-Wdeprecated-enum-enum-conversion` warnings in `tools/mtmd`.
This verifies the pre-existing hook compiles; it does **not** verify any loop
wiring, because none was applied.

## Exact reachability chain

The looped graph is built in `src/models/inkling2-loop.cpp:76-115`:

```
graph::graph()                         inkling2-loop.cpp:76
  build_layer(il, ip, ...)             inkling2-loop.cpp:84,103,107
    -> llama_model_inkling2::graph::build_layer()   inkling2.cpp:270-415  (NOT ALLOWED TO EDIT)
       -> build_moe_ffn(...)           inkling2.cpp:393-406             (NOT ALLOWED TO EDIT)
          -> gated RORET block         llama-graph.cpp:2497-2505        (already present)
```

`build_layer` is **declared** at `src/models/models.h:2567-2568` (also not
allowed to edit), is **protected and non-virtual**, and its definition has a
single hard-coded `build_moe_ffn` call with 17 positional arguments ending in
`layer.ffn_gate_up_exps` (`inkling2.cpp:406`). There is no override point, no
context/callback parameter, and no member on `llama_graph_context` that carries
RORET state into it.

## Blocker 1 — the hook's parameters cannot be forwarded to MoE

`build_moe_ffn`'s four opt-in parameters (`roret_cache`, `roret_pass`,
`roret_selected_ids`, `roret_n_selected`, `llama-graph.h:1284-1287` and
`1313-1316`) are only reachable by the caller of `build_moe_ffn`. The only caller
on the looped path is `build_layer` in `src/models/inkling2.cpp`, which does not
accept or forward them. The declaration/definition of `build_layer` and the call
site all live in files outside the permitted set (`src/models/models.h`,
`src/models/inkling2.cpp`).

A side-channel workaround (ambient RORET fields on `llm_graph_context` that
`build_moe_ffn` reads when its explicit args are default) was rejected: it makes
the documented "default = no-op" contract ambiguous, silently activates the
hook for every MoE layer built while the ambient fields are set, and still
cannot supply real host ids (Blocker 3). That would be an unsafe edit, so I
stopped instead.

## Blocker 2 — key mismatch: physical slot vs logical layer

The task requires the union key `(il = logical layer ip, pass = p)`, but
`build_layer` deliberately passes the **physical cache-slot index** `il` into
`build_moe_ffn` (`inkling2-loop.cpp:101,103,107`; `inkling2.cpp:393-406` uses it
for `cb(...)` labels and per-slot hparams). The RORET registration block keys on
`build_moe_ffn`'s `il` (`llama-graph.cpp:2498`), i.e. the physical slot. Simply
substituting `ip` for `il` would change `cb` names and hparams array indexing,
which risks changing graph node naming/order and is explicitly disallowed. A
separate `roret_key_il` argument is needed, and it must be forwarded through
`build_layer` (Blocker 1).

## Blocker 3 — no producer of host-observed top-k ids

`selected_experts` is produced at graph-build time by `ggml_argsort_top_k`
inside `build_moe_ffn` and lives on the device; it cannot be read back as host
`int32_t` ids during graph construction. No in-tree loop driver supplies
`roret_selected_ids`. Adding one requires changes to the public/runtime layer
(`src/llama.h`, `src/llama.cpp` and/or `src/llama-model.cpp`) to carry ids into
`llm_graph_params`, which are outside the permitted set. `INTEGRATION.md`
already records this as unverified item 4. Passing fabricated ids would violate
"do not change selected_experts / router semantics" and was not done.

## Blocker 4 — no consumer of `residents()`, so passes 2..N cannot actually reuse

`llama_roret_union_cache::residents(il)` is exposed, but tile residency is owned
by `llama_moe_cache::build_lookup` / the copy callback / LRU `plan` /
`build_experts` in `src/llama-moe-cache.h/.cpp` — outside the permitted set.
Until those consume `residents(il)`, registering ids has no effect on what is
resident, so "passes 2..N reuse residents" cannot be delivered by this change
alone.

## Minimal additional files/edits needed

1. `src/models/models.h:2567-2568` — extend `llama_model_inkling2::graph::build_layer`
   with trailing defaulted RORET params (cache pointer, pass, logical key `ip`,
   host ids pointer, count). All existing callers compile unchanged.
2. `src/models/inkling2.cpp:270-271,393-406` — accept those params in the
   definition and forward them to `build_moe_ffn`; keep the existing physical
   `il` argument to `build_moe_ffn` untouched for `cb`/hparams, and pass `ip`
   only as the separate RORET key.
3. `src/llama-graph.h:1265-1316` / `src/llama-graph.cpp:2321-2367,2497-2505` —
   add the `roret_key_il` parameter (already-permitted files) so the cache key
   is `(ip, pass)` rather than the physical slot.
4. `src/models/inkling2-loop.cpp:90-109` — set/pass the per-pass cache, pass
   index and key `ip` on each `build_layer` call (allowed; currently unchanged).
5. Runtime id producer: `src/llama.h`, `src/llama.cpp`, `src/llama-model.cpp`
   (or a dedicated loop driver) to read host top-k ids from the previous pass
   and put them in `llm_graph_params`. Required for real ids; not optional.
6. Actual reuse: `src/llama-moe-cache.h/.cpp` to pin/prefetch
   `residents(ip)` tiles on passes 2..N.

After (1)-(6), the bit-exactness checks in `tools/roret/INTEGRATION.md` become
mandatory: compare `ffn_moe_topk`, `ffn_moe_weights`, `ffn_moe_out` and logits
for passes 0/1/2 with the feature off vs on. **Bit-exact parity has NOT been
measured in this attempt** (and no runtime wiring exists to measure yet).
