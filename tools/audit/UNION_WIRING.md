# M36 - Union-residency wiring contract gate (source scope only)

## Commands

From the inkengine-llama root:

```
python3 tools/audit/check_union_wiring.py
PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit/test_union_wiring.py -q
```

Stdlib only for the checker, offline, fail closed. Exit 0 only on verdict PASS.
Each run first writes a NO_GO sentinel to `tools/audit/M36_UNION_WIRING_SUMMARY.json`,
then replaces it atomically (temp file plus `os.replace`) with the final verdict.
A run killed after startup therefore leaves NO_GO, not a stale PASS. If the sentinel
write itself fails (for example an unwritable audit directory), that guarantee does
not apply; a final write failure returns exit 1 and leaves the last written file
unchanged. `--root <dir>` checks an isolated copy of the repo; the tamper tests use
that flag against temp trees and never edit the real sources. No network calls, no
weights, no engine run.

## What the gate proves

- The live bytes of eight gated source files still match their frozen sha256 values.
- 18 ordered chain needles are still present, covering the whole opt-in route from
  the `LLAMA_RORET` env gate through the logical `(ip, pass)` key, the defaulted
  `build_layer` parameters, the `build_moe_ffn` forwarding, the `{roret_key_il,
  roret_pass}` lookup key, the lookup-keyed `set`/`find` binding semantics, the
  `slot_map` binding, the host-id registration, the union-cache `add`/`residents`/
  `clear_layer` implementations and their internal union set state, the
  `extra/n_extra` hints, `lru.plan(...)`, and the LRU hint contract.
- 7 default-off needles are still present: explicit opt-in values, defaulted
  parameters, null `extra`, the `if (roret)` / `if (pimpl->roret)` guards, and the
  `roret_cache` guard in `build_moe_ffn`.
- 5 negative needles are still absent, including no assignment to
  `selected_experts` and no router re-run inside the RORET bookkeeping block, no
  overwrite of real ids with residents, no mutation of `ids` inside `plan()`, and no
  code forcing the env gate on.
- All 35 checks are booleans; verdict is PASS only when every check is true.

## Full chain diagram

```
llama_moe_cache::impl ctor                    src/llama-moe-cache.cpp
  getenv("LLAMA_RORET") == "1"/"on"/"true" -> roret cache exists
  otherwise roret stays null, extra = nullptr, n_extra = 0
       |
       v
llama_model_inkling2_loop::graph::graph()     src/models/inkling2-loop.cpp
  build_layer(il, ip, ..., ip, (int32_t) pass)
       |
       v
llama_model_inkling2::graph::build_layer()    src/models/models.h + inkling2.cpp
  trailing defaults roret_il = -1, roret_pass = -1
  forwards roret_pass and roret_il as roret_key_il to build_moe_ffn
       |
       v
build_moe_ffn()                               src/llama-graph.cpp
  moe_cache->build_lookup(..., {roret_key_il, roret_pass})
  RORET block does host bookkeeping only; router and selected_experts untouched
       |
       v
binding identity and map                      src/llama-roret-binding.h
  llama_roret_bind_key(slot_map, slots) -> slots
  by_lookup keyed by that lookup pointer; set(il, pass) and find(lookup)
       |
       v
moe cache binding and registration            src/llama-moe-cache.cpp
  roret_of.set(bind_key(slot_map, slots), layer.roret_key.il, layer.roret_key.pass)
  copy() resolves the binding, clear_layer(roret_key_il) on pass 0
  roret->add({ip, pass}, host-observed ids)
  pass > 0: extra = residents.data(), n_extra = residents.size()
       |
       v
union-cache implementation                    src/llama-roret.h
  union_bits per layer OR-ed from each pass; clear_layer(roret_key_il) erases it
  residents(roret_key_il) returns the ascending union of all registered passes
       |
       v
g.lru.plan(..., ids, ..., extra, n_extra)     src/llama-moe-cache-lru.h
  residents are hints: never evict a real selection, never become CPU misses,
  never change the expert ids the graph reads
```

## Frozen source sha256

| file | sha256 |
|---|---|
| src/models/inkling2-loop.cpp | a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab |
| src/models/models.h | cb57cd8bf72891ac99ec4acc58818a011ea12fda62ca4fec63cc2c33ea5afb9e |
| src/models/inkling2.cpp | c5b543ad965857efd7f9818d42d1f7a418954758b75ec68c82485ffbd341225f |
| src/llama-graph.cpp | 5f7728c91f4dd0dcf80cbf8ce49bde9b7883af2cfd158788edf541ed3a5fac5e |
| src/llama-moe-cache.cpp | 5bab3cbf0a66698006f82188c47ceb8673e5a620888ba5e8e71d89c7559d10d4 |
| src/llama-moe-cache-lru.h | 064de659111601d43cd6ba0fbe8a68e526ac73f3b41df183a5cde7648fc80358 |
| src/llama-roret.h | 9018c1f5bb0a16e71743a7a452d5236761743ea7badd552a08717b02b9a89657 |
| src/llama-roret-binding.h | 56d42b8db6bd6229e1dfd7a02e2e5cdae437fd54e7ffb9e18c5aa3f3033791c2 |

## Verdict

`M36_UNION_WIRING_SUMMARY.json`: verdict PASS, blocker empty, 35/35 checks true,
`check_counts` total 35 passed 35 failed 0, `speed_claim: null`, `no_weights: true`,
`no_engine_run: true`.

## What the gate does NOT prove

- No engine execution: no graph build, no forward pass, no Kaggle run.
- No real weights and no host/device expert-id trace were collected.
- No bit-exact or tensor parity is measured for `ffn_moe_topk`, `ffn_moe_weights`,
  `ffn_moe_out`, or logits.
- No timing is measured, and no throughput, latency, cache-hit or speedup number is
  claimed. A source needle is not a runtime observation.
- A future source edit that changes semantics without changing these bytes and
  needles is outside this contract.

NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory for any future number.
