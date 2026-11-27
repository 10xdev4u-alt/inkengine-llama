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
