# M27 - GPU intermediate per-pass MoE tensor parity (RORET off vs on)

This page records the method, gate and measured 2xT4 result for engine-level
intermediate MoE tensor parity with RORET off vs on.

## Method

The M26 harness `tools/roret/check_intermediate_parity.py` is extended with two
default-off engine options, `--ngl` and `--threads`. With both at 0 the command
it sends to `llama-debug` is byte-for-byte the historical CPU-only command, so
the M26 CPU evidence is unchanged. With `--ngl 99 --threads 1` the same harness
runs the full graph on the device.

For each `cpu_share` (0.0 and 1.0) the harness runs `llama-debug` twice on the
synthetic fixture: OFF with no `LLAMA_RORET`, ON with `LLAMA_RORET=1`. Both runs
set `LLAMA_DUMP_TENSORS_DIR` and an explicit `--tensor-filter`. The eval callback
in `common/debug.cpp` copies device tensors to the host with
`ggml_backend_tensor_get` before writing raw bytes plus `manifest.tsv`. The
harness requires the exact expected tensor-name set with equal occurrence
counts in both phases, checks each `.bin` size against the manifest `nbytes`,
then compares raw bytes. It fails closed on any exit, hook, manifest or byte
mismatch.

The Kaggle notebook `tools/roret/kaggle-intermediate-parity/intermediate_parity.ipynb`
clones fork `main`, configures CUDA with `-DGGML_CUDA_NO_VMM=ON`, builds
`llama-debug`, regenerates the fixture, runs the harness for both shares, and
writes `M27_SUMMARY.json` on every path. The CUDA build is not given a short
per-run timeout. Kernel status was read with `kaggle kernels status` and kernel
logs with `kaggle kernels logs`; the text log was read before any output API
call.

## Gate rules

`overall_verdict` is PASS only when all of the following hold:

- clone and `git rev-parse HEAD` succeed, CUDA configure and build exit 0, and
  `build/bin/llama-debug` exists and is executable;
- fixture sha256 equals the frozen M26 value
  `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`;
- for both shares: harness exit 0, harness verdict PASS, OFF and ON exits 0,
  RORET hook ABSENT in OFF and ACTIVE in ON, at least one tensor and one byte
  compared, and exactly 0 differing bytes.

A missing or crashed run cannot pass: the per-share record defaults to exit -1,
hook off, and zero tensors and bytes, and the harness verdict is read from its
own summary JSON rather than parsed from stdout.

## Measured result (2026-11-28, Kaggle 2xT4)

Kernel `jprinceraj/inkengine-cuda-intermediate-parity` v1, `NvidiaTeslaT4x2`,
status COMPLETE. Kernel log sha256
`cf91edc9588504d1a8d9c99aac8d5a9b8f5d3c378a3da3ae2c6672c6fc2c7e46`.
Checked-in notebook sha256
`cc5a2d7e6e36616b72e255c168bc21d66ad8cdca1676b1402c4bd2f0222d81d7`.
Clone `-b main` at HEAD `6353cf489d536013831ccb949d540737046b1358`.
CUDA configure exit 0 (6.2 s), `llama-debug` build exit 0, `build_ok=true`.
Fixture sha256 matches the frozen value above.

| cpu_share | harness exit | hook off/on | tensors | bytes compared | differing bytes | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0 | ABSENT/ACTIVE | 40 | 30080 | 0 | PASS |
| 1.0 | 0 | ABSENT/ACTIVE | 40 | 30080 | 0 | PASS |

`M27_SUMMARY.json` sha256
`5e94f17e9ef865214b8e997788ec945ce88bbe7d7e3e3af3b2308bbcfcb2e1ca`:
`overall_verdict=PASS`, `blocker` empty, all 20 gate keys true. Each share has
20 unique tensor names (5 kinds x `il=0..3`) across two graph evaluations, so
40 dumped `.bin` files per phase.

Local checks this pass: roret ctest 10/10, pibs 2/2, odsl 1/1; `tools/audit`
pytest 17 passed; loopkit fast pytest 40 passed, 1 deselected. Re-running the
CPU-test harness at default flags still gives 40 tensors, 30080 bytes and 0
differing bytes at both shares, so the new flags did not regress M26.

## Scope limits

- Synthetic fixed-seed random-weight fixture only. No real model weights.
- 0 differing bytes on raw intermediate MoE tensors at engine level, for both
  `cpu_share` values, on the 2xT4 runner. Real-weight parity is still unmeasured.
- This is a correctness result only. NO SPEED CLAIM. No timing or throughput
  was measured here; any future number still needs CycleAudit `claim <= ci_low`
  with `repeats >= 20` and `loops = 3`.
