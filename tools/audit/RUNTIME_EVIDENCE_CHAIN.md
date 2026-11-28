# M43 - Runtime wiring evidence chain integrity gate

## Commands

From the inkengine-llama root:

```
python3 tools/audit/check_runtime_evidence_chain.py --root .
python3 -m pytest tools/audit/test_runtime_evidence_chain.py -q
PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit -q
```

The checker is stdlib only and offline. It makes no network calls, loads no
weights, runs no engine and measures no time. Exit 0 means verdict PASS; exit 1
means NO_GO.

Each run first unlinks any previous
`tools/audit/M43_RUNTIME_EVIDENCE_CHAIN_SUMMARY.json`, writes a NO_GO sentinel,
then replaces it atomically (temp file plus `os.replace`) with the final
verdict. A run killed after startup therefore leaves NO_GO, not a stale PASS.
The unlink runs before the sentinel so a dual write failure leaves no artifact
rather than a stale PASS. A final write failure returns exit 1. `--root <dir>`
checks an isolated copy of the repo; every tamper test uses that flag on a
temp tree and never edits real sources.

## Why this gate exists

M39's `check_evidence_chain.py` pins evidence summaries for M29-M38 only. The
three runtime wiring gates M40 (CPU 3-pass binding), M41 (3-pass replay and
pass-to-slot map) and M42 (per-event binding-to-dumped-slot pointer identity),
plus M39 itself, had no independent evidence-chain integrity gate. M43 closes
that gap. It is a new gate; it does not edit or re-derive any M39-M42 artifact.

## Frozen contract

`tools/audit/RUNTIME_EVIDENCE_CHAIN.json` is frozen with `schema_version` 1,
`frozen_on` 2026-11-28, `scope`/`note`/`speed_claim` null, `no_weights` true and
`no_engine_run` true. Its exact top-level key set is rejected if any key is
missing or added. It pins:

- four `summaries`, one per M39-M42 artifact path, each with its exact sha256
  and expected verdict PASS;
- the `milestones` list `["M39", "M40", "M41", "M42"]`;
- six `contracts`, each with an exact `path` and sha256:
  `evidence_chain` (`EVIDENCE_CHAIN.json`), `evidence_chain_checker`
  (`check_evidence_chain.py`), `wiring_contract` (`UNION_WIRING.json`),
  `binding_contract` (`THREEPASS_BINDING.json`), `replay_contract`
  (`THREEPASS_REPLAY.json`) and `runtime_contract` (`BIND_TENSOR_LINK.json`);
- `checker_sha256`, the sha256 of `check_runtime_evidence_chain.py` itself.

The checker source deliberately carries no self-hash: it computes its own
sha256 at runtime and compares it with the pin in the contract. A future
legitimate checker edit must update `checker_sha256` in the contract in the
same change. The tests pin the contract sha256 and the checker sha256 as
literals, so all three artifacts move together or the suite fails.

## What is proved

- The frozen contract has exactly the promised key set and values; its
  `contracts` object is compared byte-for-byte against the checker's frozen
  constants, and each of its six pinned files still exists at the frozen path
  with the frozen sha256.
- The checker source still hashes to the value pinned in the contract.
- Every frozen M39-M42 summary exists, parses as a JSON object, has the exact
  pinned bytes, records verdict PASS, carries no speed claim, has no blocker,
  and still reports `check_counts.failed == 0` with `passed == total`.
  A missing `blocker` key counts as no blocker exactly as in the M39 gate;
  the byte-exact sha256 pin still detects any edit to these summaries.
- `DECISIONS.tsv` still has at least one row per frozen milestone whose
  `what` column contains `land <milestone>`. Header and `#` comment lines are
  skipped; append-only growth is allowed.
- `BUILD_STATUS.md` still contains a `## <milestone> ` section heading for
  each frozen milestone.
- The generated summary has the promised shape: complete check set, boolean
  check values, consistent `check_counts`, a PASS verdict only with an empty
  blocker, no sentinel marker, and the required no-timing fields.
- A static self-check confirms the checker imports no network module and makes
  no URL-fetch call.

All checks are booleans. Verdict is PASS only when every check is true and the
sentinel write succeeded. On NO_GO the `blocker` list names every failed check.

## What is NOT proved

- No engine run by this gate: no graph build, no forward pass, no CPU or GPU
  fixture run, no Kaggle kernel. The CPU synthetic runs behind M40-M42 are
  historical evidence, not re-executed here.
- No weights are loaded or verified. Real-weight parity stays NO_GO.
- No tensor, activation or logits parity is measured. Pointer identity and
  pass-to-slot mapping from M42 are not re-derived, only pinned as evidence.
- No timing is measured. No throughput, latency, cache-hit or speedup number
  is claimed. NO SPEED CLAIM. CycleAudit `claim<=ci_low`, `repeats>=20`,
  `loops=3` remains mandatory for any future number.
- This is evidence-chain integrity only. It proves the frozen artifacts and
  contracts have not drifted, not that the earlier gates were correct beyond
  their own stated scopes.
- A future edit that changes semantics without changing these bytes is outside
  this contract.

## Honest regeneration rule

A legitimate future regeneration of any M39-M42 summary, contract or doc will
change a pinned hash or heading. That is expected. The regeneration must
update `RUNTIME_EVIDENCE_CHAIN.json` transparently in the same change, and the
test literals for the contract and checker shas with it, with a clear note in
`BUILD_STATUS.md` explaining why the evidence moved. A silent update that just
makes this gate pass again is exactly the drift this gate exists to prevent.
