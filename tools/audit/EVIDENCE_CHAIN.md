# M39 - Wiring evidence chain integrity gate

## Commands

From the inkengine-llama root:

```
python3 tools/audit/check_evidence_chain.py
PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit/test_evidence_chain.py -q
PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit -q
```

The checker is stdlib only and offline. It makes no network calls and never
runs the engine. Exit 0 means verdict PASS; exit 1 means NO_GO.

Each run first unlinks any previous
`tools/audit/M39_EVIDENCE_CHAIN_SUMMARY.json`, writes a NO_GO sentinel, then
replaces it atomically (temp file plus `os.replace`) with the final verdict.
A run killed after startup therefore leaves NO_GO, not a stale PASS. The
unlink runs before the sentinel so that a dual write failure (for example an
unwritable audit directory or a full disk) leaves no artifact rather than a
stale PASS. A final write failure returns exit 1. `--root <dir>` checks an
isolated copy of the repo; the tamper tests use that flag against temp trees
and never edit real sources.

## Frozen contract

`tools/audit/EVIDENCE_CHAIN.json` is frozen with `schema_version` 1,
`frozen_on` 2026-11-28, `scope`/`note`/`speed_claim` null, `no_weights` true and
`no_engine_run` true. It pins:

- ten `summaries`, one per M29-M38 artifact path, each with its exact sha256
  and expected verdict PASS;
- the `milestones` list `["M29", ..., "M38"]`;
- `wiring_contract: tools/audit/UNION_WIRING.json`, the source of the eight
  frozen live source hashes that M36 uses for the union-residency chain;
- `checker_sha256`, the sha256 of `check_evidence_chain.py` itself.

The whole top-level key set is exact: a missing, malformed or extra top-level
key fails closed. The checker source is deliberately not given its own hash in
code, because that would be self-referential. It computes its own sha256 at
runtime and compares it with the pin in the contract. A future legitimate
checker edit must update `checker_sha256` in the contract in the same change;
the contract change is then visible in the diff like any other evidence
update.

## What is proved

- Every frozen M29-M38 summary still exists, parses as a JSON object, has the
  exact pinned bytes, records verdict PASS, carries no speed claim, and has no
  blocker.
  - M29 is an older gate shape: it has no `blocker` key at all, while M30-M38
    use `""`. Both are treated as "no blocker". The byte-exact sha256 pin still
    detects any edit to M29, so this shape allowance does not weaken the gate.
- The eight live sources named by `UNION_WIRING.json` still exist with their
  frozen sha256 values. This re-proves the union-residency MoE routing source
  chain (env gate, loop logical key, binding, union cache, LRU hints) is
  intact at the byte level.
- `DECISIONS.tsv` still has at least one row per frozen milestone whose
  `what` column contains `land <milestone>`. Header and `#` comment lines are
  skipped; append-only growth is allowed.
- `BUILD_STATUS.md` still contains a `## <milestone> ` section heading for
  each frozen milestone.
- The checker source still matches the hash pinned in the contract.
- A static self-check confirms the checker imports no network module and makes
  no URL-fetch call.

All checks are booleans. Verdict is PASS only when every check is true. On
NO_GO the `blocker` list names every failed check.

## What is NOT proved

- No engine run: no graph build, no forward pass, no Kaggle kernel.
- No weights are loaded or verified. Real-weight parity stays NO_GO.
- No tensor, activation or logits parity is measured.
- No timing is measured. No throughput, latency, cache-hit or speedup number
  is claimed. NO SPEED CLAIM. CycleAudit `claim<=ci_low`, `repeats>=20`,
  `loops=3` remains mandatory for any future number.
- This is evidence-chain integrity only: it proves the frozen artifacts and
  gated sources have not drifted, not that the earlier gates were correct
  beyond their own stated scopes.
- A future edit that changes semantics without changing these bytes is outside
  this contract.

## Honest regeneration rule

A legitimate future regeneration of any M29-M38 summary, source or doc will
change a pinned hash or heading. That is expected. The regeneration must
update `EVIDENCE_CHAIN.json` transparently in the same change, with a clear
note in `BUILD_STATUS.md` explaining why the evidence moved. A silent update
that just makes this gate pass again is exactly the drift this gate exists to
prevent.
