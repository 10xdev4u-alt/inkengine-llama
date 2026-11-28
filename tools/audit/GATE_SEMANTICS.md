# M30: Rank-64 gate semantics check

This closes section 4 item 2 of [INKLING_INTEGRATION.md](../../loopkit/docs/INKLING_INTEGRATION.md):
a synthetic rank-64 mixing test with random tensors, grounded in the fork's real graph code.
No weights, no Kaggle, no graph execution and no timing are involved.

## What was verified

The frozen facts live in [GATE_SEMANTICS.json](GATE_SEMANTICS.json) (frozen on 2026-11-28).
The op sequence is taken from `src/models/inkling2.cpp:236-244`, not invented:

    gate = concat(rms_norm(cur, 1e-6), rms_norm(message, 1e-6), axis=0)
    gate = down_weight @ gate
    gain   = tanh(gain_weight @ gate)
    forget = tanh(forget_weight @ gate)
    out = (1 - forget) * cur + gain * message

The recorded contract is `gate_rank=64`, `n_embd=2304`, `down_weight` [64,4608],
`gain_weight`/`forget_weight` [2304,64], all F32 on export, `eps=1e-6`, four controller
streams (q, k, v, ffn), controllers only on layers 6-17 (`core_start=6`, `core_end=18`)
and `num_passes=3`. The local scalar `gamma` in `docs/ENGINE_SPEC.md` is only a
simplification.

The two `src/*.cpp` live sha256 values are gated against the frozen values. The
loopkit doc hash and the frozen facts JSON hash are recorded as INFO; the facts are
gated by trusted needles, the trusted op sequence, trusted controller tensor names
and the frozen shapes instead of by byte identity. Needle and marker matching
collapses whitespace runs so harmless reformatting is not flagged while value edits
such as `1e-6` -> `1e-5` are. The `gate_rank_fact` check only reads the frozen fact;
the numeric exercise uses rank 4, disclosed in the summary INFO lines.

    src/models/inkling2.cpp                 c5b543ad965857efd7f9818d42d1f7a418954758b75ec68c82485ffbd341225f
    src/models/inkling2-loop.cpp            a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab
    ../loopkit/docs/INKLING_INTEGRATION.md  9b0993f68bd2169468c4beedb1d54522e7ce9bd3f1c3b1c94d4033120aa0affc

## Checks

[check_gate_semantics.py](check_gate_semantics.py) uses a fixed seed with d=8, rank=4,
3 tokens for speed, while asserting the real shapes above as the frozen contract. It
checks, fail-closed:

- zero controller weights give exactly `cur` (the non-looping start),
- gates are recovered from `rank64_mix` output (`forget_hat = 1 - out/cur` with
  zero gain weights, `gain_hat = (out-cur)/msg` with zero forget weights) and checked
  against `tanh` of the pre-activations; they stay strictly inside (-1,1) for large
  weights, and at scale 1000 stay finite and within [-1,1] up to float64 rounding
  (tanh rounds to exactly 1 above about 19, so the strict check stays below that),
- in one fixed-seed synthetic instance the rank-64 output differs from the best
  least-squares scalar gamma by a clear gap; this is not a universal proof,
- a forget-only controller changes `cur` per channel in a way scalar gamma cannot express,
- the vectorized mixer is bitwise identical to an explicit per-channel Python loop,
- two fixed-seed runs are bit-identical,
- negative controls (corrupt shapes, a non-tanh gate, a scalar-gamma substitution)
  each fail their check as expected.

Missing source markers, a wrong frozen fact, a changed live source hash, an exception
or any failed check produce a NO_GO summary; a stale PASS summary is always overwritten.

## Commands and exact output

From the repo root:

    PYTHONPATH=tools/audit python3 -m pytest tools/audit/test_gate_semantics.py -q
    23 passed in 0.22s

    PYTHONPATH=../loopkit/src:gguf-py:tools/audit python3 -m pytest tools/audit -q
    89 passed in 13.09s

    python3 tools/audit/check_gate_semantics.py
    verdict: PASS
    (all checks PASS, listed in M30_GATE_SEMANTICS_SUMMARY.json)
    note: gate semantics only, no timing

Exit code was 0. The generated
[M30_GATE_SEMANTICS_SUMMARY.json](M30_GATE_SEMANTICS_SUMMARY.json) has `verdict: PASS`,
an empty `blocker`, `speed_claim: null`, `note: "gate semantics only, no timing"`, all
checks as booleans, and the frozen source sha256 values.

## What PASS means and does not mean

PASS means the frozen facts match the recorded contract, the real source files still
carry the behavioral markers, and the synthetic rank-64 math has the documented
identity, bounded-gate, per-channel and non-scalar properties, with byte-identical
reference and deterministic output.

PASS does NOT mean any real GGUF was loaded, any real gate weights were read, the
compute graph was executed, anything was timed, or any speedup exists. It makes no
claim about model quality or performance. If real weights arrive later, the next step
is to hook one core layer (6-17) and compare its gate output against this reference.

## How to rerun

    python3 tools/audit/check_gate_semantics.py
    PYTHONPATH=tools/audit python3 -m pytest tools/audit/test_gate_semantics.py -q
    PYTHONPATH=../loopkit/src:gguf-py:tools/audit python3 -m pytest tools/audit -q
