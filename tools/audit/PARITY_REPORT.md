# Issue #4 — parity audit: MEASURED PASS (CPU test mode)

**Decision: bit-exact parity of union-resident wiring is MEASURED on the
synthetic Inkling2-loop GGUF.** Generated text and raw F32 logits are
byte-identical with `LLAMA_RORET` unset vs `=1`, while the log proves the RORET
hook is active only in the `=1` run.

## What changed since the earlier NO_GO

The earlier NO_GO was correct for the then-available setup: with
`--moe-cache-mib 0` the MoE cache (and therefore the RORET hook) is never
constructed, so matching output proved nothing. Two changes made the hook
exercisable without a CUDA toolkit:

1. Default-off build flag `LLAMA_RORET_CPU_TEST` (`CMakeLists.txt`,
   `src/llama-moe-cache.cpp`) lets the MoE cache use the CPU backend **only**
   in test builds. Production builds are unchanged.
2. Harness `tools/audit/parity_run.sh` now passes `--moe-cache-mib`,
   `--moe-cache-cpu-share` and `--verbose`, so the hook-activation log line is
   captured.

Kaggle images have no `nvcc` and no `CUDA::cuda_driver`, so a CUDA build is
impossible there. This CPU test path is the honest substitute.

## Evidence (local, 2026-11-25)

Build: fork `main` @ `be7f771`, CMake `-DLLAMA_RORET_CPU_TEST=ON`,
`-DGGML_CUDA=OFF`, targets `llama-cli` and `llama-debug`.
GGUF: `tools/audit/make_synthetic_gguf.py --seed 0`, sha256
`cf72921a…`, 65 tensors, 128 experts, top-8, 3 passes, gate rank 8.
Run: `NGL=0 MOE_MIB=1 CPU_SHARE=1.0 N=16`.

| Check | Result |
|---|---|
| `off` log contains RORET hook line | **ABSENT** (correct) |
| `on` log contains RORET hook line | **ACTIVE** |
| Generated text sha256 | `cf915b84…` (identical for both) |
| Raw F32 logits sha256 | `8e1385e1…` (identical for both) |
| Byte comparison, text and logits | **0 differing bytes** |

## Scope and limits

- This proves the opt-in wiring does not perturb outputs on the synthetic
  looped graph with the hook active. It does **not** prove anything about
  real Inkling weights (none available), GPU tile residency, HBM traffic,
  or tok/s.
- `LLAMA_RORET_CPU_TEST` is a test-only flag, never enabled in a normal build.
- Any speed claim still requires a T4 CycleAudit run with `claim <= ci_low`
  (`tools/audit/AUDIT_PROTOCOL.md` §5).

## Reproduce

```bash
cmake -S . -B build -DGGML_CUDA=OFF -DLLAMA_RORET_CPU_TEST=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-cli llama-debug -j4
PYTHONPATH=gguf-py python3 tools/audit/make_synthetic_gguf.py --out synthetic.gguf
ROOT=$PWD BUILD=$PWD/build MODEL=$PWD/synthetic.gguf OUT=/tmp/parity \
  NGL=0 MOE_MIB=1 CPU_SHARE=1.0 N=16 bash tools/audit/parity_run.sh
```

## Kaggle reproduction

Kernel `jprinceraj/inkengine-cpu-parity` (CPU, fork `main`, CPU-test build): **COMPLETE / PASS** — `TEXT_PARITY: 0 differing bytes`, `off: RORET_HOOK_ABSENT`, `on: RORET_HOOK_ACTIVE`, `LOGITS_PARITY: 0 differing bytes`, `KAGGLE CPU PARITY PASS`.
