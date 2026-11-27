# M24 - Kaggle 2xT4 CycleAudit (PENDING_KERNEL)

Status: PENDING_KERNEL. No kernel pushed or run; no result below is measured.

## Pre-registered configuration (frozen before the run)

- Kernel slug: `jprinceraj/inkengine-cuda-cycle-audit` (private, T4x2, internet).
- One kernel only; logs via `kaggle kernels logs`, not the output API.
- Claim 1.0: valid only if claim_valid true AND claim <= ci_low.
- Repeats 50 (>=20), warmup 5, loops 3, seed 0, 2000 bootstrap resamples.
- Synthetic 3-pass MoE GGUF from `tools/audit/make_synthetic_gguf.py --seed 0`.
- RORET OFF vs ON, cpu_share 0.0, ngl 99, moe-cache-mib 16, real loopkit gate.
- Negative controls must fail closed: self 1.01, slower 2.0.
- Verdict: CLAIM_SUPPORTED only if claim_valid and claim <= ci_low, else NO_CLAIM.

## Scope

Counts (RORET_PASS rows, hits, uploads) are counts, not speedups. T4 sm_75 has
no FP4/FP8 tensor cores. Only ci_low may support a speed number. PASS means the
gate ran with integrity, not that a speed claim exists.

## Placeholder - fill from kernel logs after the run

- Kernel URL/version/status and fork/loopkit HEADs: PENDING
- nvcc, configure exit, build exit, fixture sha256, cli version: PENDING
- M24_SUMMARY_JSON line verbatim from kernel logs: PENDING
- ci_low / ci_high / speedup / claim_valid / gate exit code: PENDING
- Negative controls (self 1.01, slower 2.0): PENDING
- Final verdict and next blocker: PENDING
