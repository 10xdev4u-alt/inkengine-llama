# M32 - Pass-indexed KV slots (frozen contract)

## Command

From the inkengine-llama root:

```
python3 tools/audit/check_pass_indexed_kv.py
```

Stdlib only, offline, fail closed. Exit 0 only on verdict PASS. Every run rewrites
`tools/audit/M32_PASS_INDEXED_KV_SUMMARY.json`, so a stale PASS is never left behind on a
failure. Mirrors the M31 (`check_loop_boundaries.py`) gate.

## What was checked

- Frozen facts: `n_phys=24`, `n_passes=3`, `core_start=6`, `core_end=18`, `n_slot=48`,
  `n_ctrl=4`, and the three live formula strings.
- Frozen and live sha256 of all three gated sources; any mismatch forces NO_GO.
- 19 live source needles covering: slot formula and the `LLAMA_MAX_LAYERS` guard,
  `n_layer_all`, `layer_phys` inverse, per-slot hparam copy, weight sharing with private
  KV/conv state, loop control tensor shapes, pass slot assignment, previous-pass bank,
  the union routing comment, `build_layer(il, ip, ..., ip, pass)` arguments, state/attn
  keyed by `il`, residual boundary keyed by `ip`, RORET key args `(roret_il, roret_pass)`,
  and the converter's 4-row `(q, kv, kv, q)` message stack.
- Four derived checks on the 36-entry slot map: derived from the pure functions, injective,
  `layer_phys` inverse, and exact coverage.

## Slot map (3 passes x 12 core layers = 36 distinct cache slots)

| pass | logical core layer `ip` | cache slot `il` |
|---|---|---|
| 0 | 6 | 6 |
| 0 | 7 | 7 |
| 0 | 8 | 8 |
| 0 | 9 | 9 |
| 0 | 10 | 10 |
| 0 | 11 | 11 |
| 0 | 12 | 12 |
| 0 | 13 | 13 |
| 0 | 14 | 14 |
| 0 | 15 | 15 |
| 0 | 16 | 16 |
| 0 | 17 | 17 |
| 1 | 6 | 24 |
| 1 | 7 | 25 |
| 1 | 8 | 26 |
| 1 | 9 | 27 |
| 1 | 10 | 28 |
| 1 | 11 | 29 |
| 1 | 12 | 30 |
| 1 | 13 | 31 |
| 1 | 14 | 32 |
| 1 | 15 | 33 |
| 1 | 16 | 34 |
| 1 | 17 | 35 |
| 2 | 6 | 36 |
| 2 | 7 | 37 |
| 2 | 8 | 38 |
| 2 | 9 | 39 |
| 2 | 10 | 40 |
| 2 | 11 | 41 |
| 2 | 12 | 42 |
| 2 | 13 | 43 |
| 2 | 14 | 44 |
| 2 | 15 | 45 |
| 2 | 16 | 46 |
| 2 | 17 | 47 |

`il = ip` on pass 0; `il = n_phys + (pass-1)*(core_end-core_start) + (ip-core_start)` on later passes.
`layer_phys(il)` inverts every row. The 36 core slots are 6..17 and 24..47; the prelude
slots 0..5 and coda slots 18..23 are each used once, so all 48 layer applications use
48 distinct cache slots. RORET routing uses `(roret_il=ip, roret_pass=pass)`, not the
physical slot `il`, so the union is keyed by logical core layer and pass even though KV
and conv state live in separate slots.

## Frozen source sha256

| file | sha256 |
|---|---|
| conversion/inkling2.py | aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7 |
| src/models/inkling2-loop.cpp | a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab |
| src/models/inkling2.cpp | c5b543ad965857efd7f9818d42d1f7a418954758b75ec68c82485ffbd341225f |

## Verdict

`M32_PASS_INDEXED_KV_SUMMARY.json`: verdict PASS, blocker empty, 27/27 checks true,
`speed_claim: null`, note "pass-indexed KV slot contract only, no timing".

## Scope limits

Contract only. No weights, no graph execution, no timing. NO SPEED CLAIM. Any future
performance number still needs a CycleAudit measured claim with `claim <= ci_low`,
`repeats >= 20`, and `loops = 3`. This gate does not substitute for that measurement.
