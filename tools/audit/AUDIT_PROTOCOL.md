# Audit protocol — bit-exact baseline and CycleAudit gate contract

Status: freeze before optimization. Branch: `main` (fork of llama.cpp). Gate source of truth: `loopkit/src/loopkit/audit.py`, `loopkit/src/loopkit/cli.py`, `loopkit/README.md`.

## 1. Purpose

Freeze two things BEFORE any optimization is measured: (1) the bit-exact baseline, where fast tests pass and all three loop passes certify to a stable `root_digest`; (2) the CycleAudit contract, one recipe, one verdict rule, and two negative controls that fail closed. The work this gates is not yet measured: union-resident MoE wiring (`DECISIONS.tsv`, `tools/roret/RESULTS.md`) and the 3-pass looped graph. Both stay models or plans until this contract is applied to a real T4 timing. No speedup is asserted here.

## 2. Environment and setup

Python 3.11+. From the workspace root, either `export PYTHONPATH=<loopkit>/src` or `pip install -e loopkit`. Dependencies: `numpy`, `scipy`. Fast test command, run from the `loopkit` checkout (CI uses the same one):

```bash
pytest -q -m "not kaggle"
```

The two `kaggle`-marked geometry tests are excluded locally and run on the T4 notebook. Only Kaggle 2xT4 (sm_75) timings are eligible for a publishable claim. Local CPU numbers are mechanical plumbing validation, never speed evidence.

## 3. Timing JSON schema

One JSON file per configuration. The CLI reads `name` and `seconds`; keep `cycles` and `setup_seconds` as provenance (the CLI infers `cycles` from array length and defaults `setup_seconds` to 0).

```json
{
  "name": "baseline-b1",
  "seconds": [0.0123, 0.0119, 0.0121, 0.0124, 0.0118],
  "cycles": 5,
  "setup_seconds": 0.031
}
```

- `repeats >= 20`; use the default 50. The five values above are truncated for shape only — a real file has 50 entries and `"cycles": 50`.
- `warmup = 5` iterations discarded before the first recorded value.
- `cycles` must equal the number of `seconds` entries; `loops = 3`, seed 0.
- Put batch size in `name` (`baseline-b1`, `baseline-b8`, `candidate-b1`, `candidate-b8`); use separate files per batch size.
- Defaults are set once in `CycleAudit.__init__` in `loopkit/src/loopkit/audit.py`: `loops=3, warmup=5, repeats=50, seed=0`.

## 4. The gate command

```bash
python -m loopkit.cli audit --claim X --baseline BASE.json --candidate CAND.json --loops 3
```

It prints the `CycleReport` dict (`baseline`, `candidate`, `loops`, `speedup`, `ci_low`, `ci_high`, `claim_valid`, `reason`) plus `claim_valid`. `ci_low` is the 2.5th percentile of 2000 bootstrap `b / c` ratios; `ci_high` is the 97.5th, a 95% interval on the ratio rather than on one run. `speedup` (`base.median() / cand.median()`) is an informational point estimate only. Exit 0 means `claim_valid` true: the claim passed. Exit 1 means false: no claim. A failed gate is not a near miss and is not rounded up.

## 5. The verdict rule

From `verdict()` in `loopkit/src/loopkit/audit.py`:

```python
if self.repeats < 20 or self.loops < 1:
    return False
return claim_speedup <= report.ci_low
```

A claim is publishable ONLY IF `claim <= ci_low` (lower 95% bootstrap bound), `repeats >= 20`, AND `loops >= 1`. All three must hold. Never round up to meet the bound. If `ci_low < 1.0` there is no claim at all, regardless of the point estimate. The bound, not the median, is the gate.

## 6. Negative controls (must fail closed)

Run both before any real comparison and record the output in section 10. Self-comparison:

```bash
python -m loopkit.cli audit --claim 1.01 --baseline BASE.json --candidate BASE.json --loops 3
```

Expect `claim_valid` false, exit 1. `compare()` resamples the same array independently for `b` and `c`, so ratios scatter around 1.0 and the 2.5th percentile sits just below 1.0. A 1.01 claim exceeds that lower bound and must be rejected. This proves the gate cannot manufacture a win from noise. Slower candidate:

```bash
python -m loopkit.cli audit --claim 2.0 --baseline BASE.json --candidate SLOW.json --loops 3
```

Expect `claim_valid` false, exit 1; a slower candidate cannot validate a 2x claim. If either control exits 0, stop: the gate is broken and no result may be published.

`verify_gate.py` runs both controls plus two positive-side checks (claim exactly at
`ci_low` passes, `ci_low + 0.05` fails) against deterministic fabricated samples:

```bash
python3 tools/audit/verify_gate.py --out /tmp/auditcheck --loopkit-src <loopkit>/src
python3 -m pytest -q tools/audit/test_verify_gate.py
```

It writes `/tmp/auditcheck/gate_verification.json` (`"verified": true`) and exits 0
only when all four expectations hold, `loops == 3`, and `repeats >= 20`. The samples
are synthetic: this proves the plumbing, never a speedup.

## 7. Bit-exact baseline freeze

```bash
pytest -q -m "not kaggle"
python -m loopkit.cli certify --ref REF --cand CAND --model inkling2mini
```

`certify` writes `cert.json` with `root_digest` and `all_exact`. The root digest is `sha256(concat(pass digests))`; one flipped bit in any pass invalidates it. The two guarantees from `loopkit/README.md` are `nvfp4-roundtrip` (`ans_decode(ans_encode(t)) == t` on codes, scales, shape, block) and `fdm-check` (`fdm_dot(a, b) == reference_dot(a, b)`, bit-exact).

| Artifact | Field | Value |
|---|---|---|
| `cert.json` | `root_digest` | PENDING |
| `cert.json` | `all_exact` | PENDING |
| REF fixture | sha256 | PENDING |
| CAND fixture | sha256 | PENDING |
| `nvfp4-roundtrip` | lossless | PENDING |
| `fdm-check` | max_abs_err_vs_ref | PENDING |

Any later wiring change (union-resident MoE, 3-pass looped graph) must keep `root_digest` identical on all 3 passes; a changed digest voids the audit. Per `ENGINE_SPEC.md` section 3, stored representation and dequantized FP32 operands are bit-exact; FP16 tensor-core output is measured drift, never labelled bit-exact.

## 8. Honesty rules

- T4 (sm_75) has no FP4 or FP8 tensor cores. FP4/MXFP8 are storage formats here.
- The union-resident MoE result in `tools/roret/RESULTS.md` is a load-count model versus 24 re-routed activations, not wall-clock. It claims no `tok/s`.
- Numeric models are models. `DECISIONS.tsv` marks the b8 step change and the pass-interleaved prototype "model only; no speed claim".
- No speedup from a local CPU run, a single warm run, or a point estimate without a bootstrap bound.
- The alternative to a win is "no measured advantage" and stopping: STOP criterion 6 in `ENGINE_SPEC.md` says if `ci_low < 1.0` against llama.cpp Q4_K_M, report no advantage and stop; do not publish a rounded-up claim.

## 9. What this protocol does NOT allow

- Rounded-up claims, including rounding `speedup` into the claim.
- Claims with fewer than 20 repeats, with `loops < 1`, from a single warm run, or from a local CPU timing.
- Treating a load-count reduction as a wall-clock speedup.
- Using `ci_high` instead of `ci_low`, or ignoring `claim_valid` / the exit code.
- Labelling FP16 tensor-core output bit-exact.

## 10. Results (fill in, stay honest)

| Item | Result |
|---|---|
| Fast tests (`pytest -q -m "not kaggle"`) | 27 passed, 1 deselected (local CPU, 2026-11-24) |
| Gate unit tests (`test_verify_gate.py`) | 6 passed |
| Executable gate check (`verify_gate.py`) | `verified: true` on synthetic samples only |
| Negative control self (claim 1.01, expect exit 1) | PASS: `claim_valid=false`, exit 1, `ci_low=0.9832` |
| Negative control slower (claim 2.0, expect exit 1) | PASS: `claim_valid=false`, exit 1, `ci_low=0.7869` |
| Claim at `ci_low` / `ci_low + 0.05` | PASS / FAIL as expected (synthetic samples) |
| Certify `root_digest` / `all_exact` | PENDING (fixture not generated; issue #2) |
| Real T4 baseline b1 (`ci_low`, repeats, loops) | PENDING |
| Real T4 baseline b8 (`ci_low`, repeats, loops) | PENDING |
| Publishable claim vs llama.cpp Q4_K_M (claim <= ci_low) | PENDING |
