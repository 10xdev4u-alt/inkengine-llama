# M31 - Loop boundaries (frozen contract)

## Command

From the inkengine-llama root:

```
python tools/audit/check_loop_boundaries.py
```

Stdlib only, offline, fail closed. Exit 0 only on verdict PASS. Every run rewrites
`tools/audit/M31_LOOP_BOUNDARIES_SUMMARY.json`, so a stale PASS is never left behind on a
failure. Mirror of the M29 (`check_upstream_loop_contract.py`) and M30
(`check_gate_semantics.py`) gates.

## Frozen facts (2026-11-28)

| fact | value |
|---|---|
| core_start | 6 |
| core_end | 18 (0-indexed, exclusive) |
| num_passes | 3 |
| gate_rank | 64 |
| is_varloop | false (mini) / true (varloop) |
| n_phys | 24 |
| n_slot | 48 |

Slot formula: `n_slot = n_phys + (n_passes-1)*(core_end-core_start) = 24 + 2*12 = 48`.

Source of truth: `loopkit/docs/INKLING_FORMAT_FACTS.md` section 3 (reconciled primary
sources S2/S4/S8).

## Derived layer set

- core layers: 6,7,8,9,10,11,12,13,14,15,16,17 (12 layers)
- prelude: 0,1,2,3,4,5 (block 0, runs once)
- coda: 18,19,20,21,22,23 (block 3, runs once)
- 48 layer applications from 24 stored layers: 6 prelude + (12 loop layers x 3 passes) + 6 coda

## Evidence

Frozen source sha256 (all five gated, live equals frozen at freeze time):

| file | sha256 |
|---|---|
| ../loopkit/docs/INKLING_FORMAT_FACTS.md | 46f134e263d1f9889a22e556428f7bdc6144fa1c9068072c31794ea19c4071df |
| conversion/inkling2.py | aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7 |
| gguf-py/gguf/tensor_mapping.py | c653a1993368f9967ef3788949f9e71c88b123938889d1180d13ac76b38aa463 |
| src/models/inkling2-loop.cpp | a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab |
| gguf-py/gguf/constants.py | f31289cf677a5125cc37cd58d87228199d5f0a124f583deb95a93fcfd79e086b |

Checks (22, all true in the generated summary):

- frozen facts and frozen hashes, plus live hashes of all five gated sources
- converter needles: loop_config key set, `0 < start < end <= self.block_count`,
  `2 * (self.loop["core_end"] - start)` cross-pass query shape, and the five
  `add_loop_start/end/count/gate_rank/varying` metadata writes
- exactly 12 `MODEL_TENSOR.LOOP_*_CTRL_*` entries in tensor_mapping.py, each with an
  alias containing `model.layers.{bid}.controllers.`
- slot formula uses `core_end - core_start` and evaluates to 48
- derived core set: len 12, min 6, max 17, excludes 18, includes 6
- both `../loopkit/docs/ENGINE_SPEC.md` and `../inkengine/docs/ENGINE_SPEC.md` carry
  the corrected section-4 note and contain none of the three stale ASCII premises

## Scope statement

Metadata only. No weights, no graph execution, no timing. NO SPEED CLAIM. Any future
performance number still needs a CycleAudit measured claim with `claim <= ci_low`,
`repeats >= 20`, and `loops = 3`. This gate does not substitute for that measurement.
