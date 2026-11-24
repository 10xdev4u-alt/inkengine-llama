# Kaggle RORET probe — 100,000 tokens per setting

> **Source-status note (2026-11-24).** The γ values below are synthetic scalar probe knobs standing
> in for the upstream rank-r gain/forget loop controllers, not measured model parameters; our
> earlier scalar-gamma model was a simplification. Upstream already provides the loop graph and
> converter; this fork's contribution is union-resident MoE routing on top of it.

Kernel: `jprinceraj/roret-probe-cpu`, built from fork `main` (commit f8a63c2),
GCC 13.3 on Kaggle CPU. γ = [0, 0.3, 0.6] **[synthetic scalar gate stand-ins; upstream gate is
rank-r gain/forget controllers]**, 128 experts / top-8, dim 256.

| msg_scale | exact all-3 agreement | mean overlap 1→2 | union experts | load reduction | pass-2/3 hit from pass-1 | exact verdict | union verdict |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.7608 | 0.984 | 8.25 | 65.6% | 97.7% | NO_GO | **GO** |
| 0.10 | 0.5685 | 0.969 | 8.48 | 64.7% | 95.6% | NO_GO | **GO** |
| 0.30 | 0.1637 | 0.917 | 9.29 | 61.3% | 88.1% | NO_GO | **GO** |
| 1.00 | 0.0015 | 0.758 | 11.59 | 51.7% | 66.7% | NO_GO | NO_GO |

These reproduce the 3,000-token local results in `RESULTS.md` to within
~0.5%, confirming the synthetic precondition at 33× the sample size.

**Decision:** drop exact route-once. Implement union-resident routing using
`union_cache.hpp` and the insertion points in `PATCH_build_moe_ffn.md`.

Kernel output is large because the clone lives in `/kaggle/working`; the numeric
results are in the kernel logs. Future kernels should clone into `/tmp`.
