# M14 - Kaggle 2xT4 CUDA build probe (2026-11-27)

Kernel: `jprinceraj/inkengine-cuda-roret-probe` version 1 (private, `NvidiaTeslaT4x2`).
Status: **COMPLETE with build blocker**. One kernel was run this pass, which is the
per-run limit, so the retry was not attempted here.

## What was measured

The notebook (`/tmp/opencode/kaggle-cuda-probe/cuda_probe.ipynb`) printed these facts
to the kernel log (saved locally as `/tmp/opencode/m14-kaggle-logs.json`):

* Python 3.13.16; `nvcc` release 13.0, V13.0.88; nvidia-smi 580.178.04;
  CUDA Version 13.0; two Tesla T4 (15360 MiB each); torch 2.11.0+cu130,
  `device_count 2`.
* `git clone -b main https://github.com/10xdev4u-alt/inkengine-llama.git`
  exit 0, HEAD `5facded0833449b7dd2993a542b70628f1b21f3a`.
* Configure command:
  `cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF`
  -> **exit 1**. CUDAToolkit 13.0.88 and the CUDA compiler were found, then:
  `CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:184 (target_link_libraries):
   Target "ggml-cuda" links to: CUDA::cuda_driver
   but the target was not found.` The generate step failed.
* The build step still ran against stale files:
  `cmake --build build --target llama-cli -j4` -> **exit 2**, ending in
  `/usr/bin/ld: cannot find -lCUDA::cuda_driver: No such file or directory`.

An earlier local note said Kaggle images had no `nvcc`. That premise is wrong:
this kernel has nvcc 13.0 and a working CUDA compiler. The blocker is a missing
CMake imported target (`CUDA::cuda_driver`), not compiler availability.

## What was NOT measured

* No synthetic GGUF was generated (the stage was skipped after `build_ok=False`).
* No `llama-cli` run, off or on, so no generated-text comparison and no hook
  absent/active assertion.
* No `RORET_PASS` rows. The engine-side copy path remains **unexercised on GPU**.
* No speed number of any kind.

## Local verification done in this pass

* Fresh `cmake -S tools/roret -B /tmp/opencode/m14-roret` + build + ctest:
  **6/6** passed.
* Fresh `tools/pibs` ctest: **2/2**; fresh `tools/odsl` ctest: **1/1**.
* `tools/audit` pytest: **17 passed**.
* `loopkit` fast pytest (`-m "not kaggle"`): **40 passed, 1 deselected**.
* Fresh local CPU build
  (`GGML_CUDA=OFF`, `LLAMA_RORET_CPU_TEST=ON`): 100%,
  `llama-cli --version` = `1.0.0-dev (build 127, commit 5facded)`.
* UNION_WIRING.patch sha256 `a9186808...ca1e84` and PIBS_WIRING.patch
  sha256 `f69543d2...371804` unchanged at HEAD `5facded`.

## Blocker and next step

BLOCKER: the Kaggle CUDA 13.0 image does not provide the CMake imported target
`CUDA::cuda_driver` that `ggml/src/ggml-cuda/CMakeLists.txt:184` links by default,
so a default CUDA configure of this fork fails before any engine run.

Candidate retry (UNVERIFIED - not run this pass because the one-kernel budget was
spent): add `-DGGML_CUDA_NO_VMM=ON` to the configure command. In-tree source
shows that this branch skips the `CUDA::cuda_driver` link. A future run should
retry the same probe with that flag and keep the same fail-closed gates: require
`build_ok`, both CLI exits 0, hook absent off / active on, non-empty text files,
and only then interpret `RORET_PASS` rows as engine-side counts.

## Standing verdicts

* Issue #4 (engine-level GPU bit-exact parity): **NO_GO**, unchanged.
* Issue #5 (T4 per-pass MoE traffic): **NO_GO**, unchanged; no traffic rows exist.
* No speed claim. Any future number still needs a CycleAudit result with
  `claim <= ci_low`, `repeats >= 20` and `loops = 3`.
* `RORET_PASS` counts, if a later run obtains them, are counts, not time or tok/s.

## M15 - Kaggle NO_VMM retry (2026-11-27): build PASS, engine still NO_GO

Kernel: `jprinceraj/inkengine-cuda-roret-probe-v2` (private, `NvidiaTeslaT4x2`),
v1, **COMPLETE**. Note: changing the notebook title in `kernel-metadata.json`
made the Kaggle CLI create a NEW kernel slug (`...-probe-v2`) rather than a new
version of `jprinceraj/inkengine-cuda-roret-probe`; this is one kernel run for
this pass and the logs were read from the new slug with `kaggle kernels logs`,
not the output API.

One kernel was run this pass (the per-run limit). Notebook source:
`/tmp/opencode/kaggle-cuda-probe-v2/cuda_probe_v2.ipynb` sha256
`1da667919392cb74e4f68e78cbb77264e9615bf0dad0d844156310668a2fbad1`.
It is v1 plus `-DGGML_CUDA_NO_VMM=ON` and four informational summary fields
(`configure_command`, `configure_exit`, `build_exit`, `configure_log_tail`);
all fail-closed gates and the "NO SPEED CLAIM" text are unchanged. An
independent Hy4 Flash review found no CRITICAL/HIGH/MEDIUM issue.

### Measured on Kaggle

* git clone `-b main`, HEAD `7b29215b8647e5c6276e55ae2298108ffcb930c2`, exit 0.
* `cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DGGML_CUDA_NO_VMM=ON`:
  **configure exit 0** (5.5 s), "Configuring done", "Generating done". The
  v1 `CUDA::cuda_driver` blocker is resolved; the in-tree NO_VMM branch at
  `ggml/src/ggml-cuda/CMakeLists.txt:181-185` was the correct escape hatch.
* `cmake --build build --target llama-cli -j4`: **build exit 0**.
  `build_ok = true`. nvcc 13.0.88, GNU 13.3.0 host compiler, 2x Tesla T4.
* Synthetic GGUF generator exit 0, file sha256
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`.
* `llama-cli` off run: exit `-6` (SIGABRT); hook ABSENT.
  `llama-cli` on run (`LLAMA_RORET=1`): exit `-6`; hook ACTIVE.
  Both abort during warmup with the same assertion:
  `src/llama-moe-cache.cpp:577: GGML_ASSERT(sel->op == GGML_OP_RESHAPE) failed`,
  reached via `llama_context::sched_copy_experts` -> `llama_moe_cache::copy()`.
* No text files were produced; `text_parity_bytes_diff = null`, verdict
  `TEXT_PARITY: NO_GO`. No `RORET_PASS` rows, `engine_copy_exercised = false`,
  verdict `NO_GO: RORET_PASS table empty`. No speed number.

### What this means

The build blocker in issue #7 is fixed by `GGML_CUDA_NO_VMM=ON`. A NEW blocker
now prevents any engine run: on the CUDA scheduler, the selected-experts tensor
that `copy()` reads is not a `GGML_OP_RESHAPE` view as the assertion assumes,
so the `-cmoe` MoE-cache path aborts during warmup. This fires with RORET
OFF as well as ON, i.e. before any union-resident routing code is reached.
Issues #4 (engine GPU bit-exact parity) and #5 (T4 per-pass traffic) stay
NO_GO; no hook state can be trusted until the CLI runs to completion.

Next bounded step (NOT done this pass): inspect the actual op/shape of
`lookup->src[1]` in the CUDA graph split and replace the assert with a
device-agnostic extraction of the [n_expert_used, n_tokens] ids, with an
off/on CLI smoke test on the same fixture before any RORET_PASS claim.
Any future speed number still requires CycleAudit
`claim <= ci_low`, `repeats >= 20`, `loops = 3`.

### Local verification in this pass (fresh)

* ctest roret **6/6**, pibs **2/2**, odsl **1/1**.
* `tools/audit` pytest **17 passed**; loopkit fast pytest **40 passed,
  1 deselected**.
* Fresh local CPU build `GGML_CUDA=OFF` + `LLAMA_RORET_CPU_TEST=ON`:
  `llama-cli --version` = `1.0.0-dev (build 128, commit 7b29215)`.
* No source file was changed by this milestone; the Kaggle kernel and the
  report/ledger are the only artifacts.
