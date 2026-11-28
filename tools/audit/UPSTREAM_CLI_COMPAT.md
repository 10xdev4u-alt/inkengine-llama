# Upstream CLI byte compatibility (M44)

## What item (d) is

`loopkit/docs/INKLING_INTEGRATION.md`, section 3, item (d) asks to verify "that a GGUF produced by the fork is byte-compatible with upstream `llama-cli`". This gate closes that item. It does not use real model weights. It runs two built `llama-cli` binaries on the same frozen synthetic `inkling2_loop` GGUF and compares the generated stdout bytes.

The claim is narrow: with union-resident MoE routing compiled into the fork but OFF by default, a frozen GGUF produces identical stdout bytes on the upstream-base build and the fork build. The two binaries must have different content and different realpaths. A vacuous same-binary comparison, or two identical-content copies at different paths, fails closed via the distinct-hash check and the CLI hash pins.

## Exact method

1. Write a NO_GO sentinel to `--out` before any check.
2. Load `UPSTREAM_CLI_COMPAT.json`. Reject any missing or extra top-level key. Reject drift in any frozen fact.
3. Verify the checker self-hash: `sha256(Path(__file__).read_bytes())` must equal the pinned `checker_sha256`. A `PENDING` placeholder is a hard failure, not a skip.
4. Verify the frozen fixture SHA-256 and byte size. The contract values must match the checker's hardcoded constants, and the live fixture must match both.
5. Verify both CLI paths exist, are files, are executable, and resolve to different realpaths.
6. Verify both CLI binaries hash to their pinned `fork_cli_sha256` and `upstream_cli_sha256`, and that those two hashes differ. `PENDING` fails closed.
7. Run a third, control invocation of each pinned binary with `LLAMA_RORET=1` and the extra arguments `-cmoe --moe-cache-mib 16 --verbose`. The fork control must print the RORET enable marker on stderr; the upstream control must not. Control exit codes are not gated: the CPU-test `-cmoe` path can abort during teardown after the marker is printed. This is a runtime activation control, stronger than a raw binary-string scan.
8. Build the child env from the parent env with these seven keys removed for BOTH runs: `LLAMA_RORET`, `LLAMA_DUMP_TENSORS_DIR`, `LLAMA_DUMP_TENSOR_PTR`, `LLAMA_RORET_TRACE`, `LLAMA_PIBS`, `LLAMA_ODSL`, `LLAMA_ARG_SHOW_TIMINGS`. This makes the comparison genuinely default-off and hides timing state.
9. Run each binary with the frozen `cli_args`, substituting the fixture path for the `{fixture}` token. Require a list from the contract, never a checker fallback. Capture stdout and stderr as bytes. Require exit 0 from both.
10. Require the contract-pinned `stdout_contains` payload to occur in both captured stdout byte streams. `PENDING` or an empty payload fails closed.
11. Require the enable marker `RORET union-resident MoE bookkeeping enabled` to be ABSENT from both stderrs in the two main runs. Absence alone is necessary but not sufficient, because the child env strips `LLAMA_RORET`. Step 7 supplies the positive control: the same pinned fork binary does print the marker when `LLAMA_RORET=1`, and the pinned upstream binary does not. Together the two controls show the fork hook is activatable and that the compared runs left it off.
12. Remove the `build      : <build-info>` banner line from both stdout streams, then compare the remaining bytes and count differing bytes. The banner records build provenance (build number and commit), not GGUF or graph output; it necessarily differs between the upstream-base and fork builds. A length difference counts once per extra byte. All other stdout bytes, including the prompt echo and generated tokens, are compared.
13. Verdict is PASS only when every check passes and the differing-byte count equals `expect.stdout_differing_bytes` (0).
14. Write the final summary atomically: temp file in the same directory, then `os.replace`.

There are no network calls. Stdout is the only compared stream, and only its build-info banner line is removed. Stderr is used solely to check for the RORET enable marker; timing lines and build banners on stderr are not compared.

## Contract fields

- `frozen_on`: freeze date, `2026-11-28`.
- `contract_version`: integer 1.
- `fixture`: `sha256`, `size_bytes`, `generator`, `seed`, `arch`. Frozen synthetic file made by `tools/audit/make_synthetic_gguf.py --seed 0`. SHA-256 `cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`, size 887392 bytes, arch `inkling2_loop`.
- `upstream_base`: `commit` `bc641b516fe01f346b9f2bf70d69fb7afa3ac222` (llama.cpp master at 2026-11-24, before the first fork commit), `date`, `build_flags` (`GGML_CUDA=OFF`, `LLAMA_BUILD_TESTS=OFF`, `LLAMA_CURL=OFF`).
- `fork`: `head_at_freeze`, `build_flags` (same three plus `LLAMA_RORET_CPU_TEST=ON`), `rort_default_off_env_unset`, `rort_enable_marker_absent`.
- `cli_args`: `-m {fixture} -t 1 -n 16 -s 0 --temp 0 -p hello --no-perf --no-show-timings --no-warmup -st`. Deterministic. `--no-perf` disables perf output; `--no-show-timings` is the flag tied to `LLAMA_ARG_SHOW_TIMINGS` (`common/arg.cpp`) and removes the timing lines.
- `checker_sha256`: pinned checker digest. Set to `PENDING` at author time; fill with the real digest before the first real run.
- `fork_cli_sha256`, `upstream_cli_sha256`: pinned digests of the two built binaries. `PENDING` at author time. They must be valid 64-hex, match the live files, and differ from each other.
- `stdout_contains`: payload that both stdout streams must contain. `PENDING` at author time. Fill with a deterministic token present in both builds' output.
- `expect`: `both_exits_zero` true, `stdout_differing_bytes` 0, `stderr_compared` false.
- `speed_claim`: null.

The summary is written to `tools/audit/M44_UPSTREAM_CLI_COMPAT_SUMMARY.json` by default. Its fields are `verdict`, `checks_total`, `checks_passed`, `checks_failed`, `blockers`, `fixture_sha256`, `fork_cli_sha256`, `upstream_cli_sha256`, `fork_stdout_sha256`, `upstream_stdout_sha256`, `stdout_bytes_each`, `stdout_differing_bytes`, `fork_exit`, `upstream_exit`, `fork_rort_enable_marker_absent`, `upstream_rort_enable_marker_absent`, `stdout_contains`, `fork_stdout_contains_payload`, `upstream_stdout_contains_payload`, `fork_cli_hash_matches_pin`, `upstream_cli_hash_matches_pin`, `cli_hashes_distinct`, `fork_control_marker_present`, `upstream_control_marker_absent`, `speed_claim`, `no_weights`, `engine_run`, `note`.

## Fail-closed controls

- Sentinel first: a stale PASS summary cannot survive an early stop. If the initial sentinel write fails, the checker unlinks the stale output before re-raising.
- Exact key set and frozen facts: extra, missing, or changed fields cause NO_GO.
- Self-hash pin: a checker edit without a contract update causes NO_GO. `PENDING` also causes NO_GO.
- Fixture pinned in both the contract and the checker constants, then matched against the live file.
- Distinct realpaths and distinct CLI hashes: one binary, or two identical copies, cannot pass as both sides.
- Binary-content check: the fork must carry the compiled RORET hook and the upstream base must not.
- Stdout payload check: two streams without the pinned payload cannot pass even if their bytes match.
- Both exits must be zero; any timeout or start failure fails closed.
- Enable marker absence checked on both main stderrs, with a positive control showing the pinned fork binary can print the marker and the pinned upstream binary cannot.
- Nonzero byte delta fails closed, including a length-only difference.
- Contract `cli_args` must be a list; no checker fallback list is used.
- Atomic final write; a failed replace leaves the sentinel in place.

## Scope limits

Synthetic fixture only. No real weights. No GPU. No timing and no speed claim from this gate. Stdout bytes only; stderr is inspected solely for the enable marker, never compared. NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory for any future number.

## How to rerun

Build both binaries from the frozen sources with the flags in the contract. Fill the four `PENDING` pins. Then run:

```sh
python3 tools/audit/check_upstream_cli_compat.py \
  --fork-cli /path/to/fork/bin/llama-cli \
  --upstream-cli /path/to/upstream/bin/llama-cli \
  --fixture /path/to/synthetic.gguf
```

Optional flags: `--root` (fork root, defaults to repo root), `--out` (summary path), `--timeout` (per-run seconds, default 180). The process exits 0 on PASS and 1 on NO_GO.

Tests run only in isolated `tmp_path` trees with fake CLI scripts and never run the checker on the real fork tree:

```sh
python3 -m pytest -q tools/audit/test_upstream_cli_compat.py
```

## Measured result

Real run on 2026-11-28, fork `main` HEAD `5039437e312ecc0bf37674e24b89c7e2e8ee7a99` and upstream base `bc641b516fe01f346b9f2bf70d69fb7afa3ac222`, both built locally with `GGML_CUDA=OFF`, `LLAMA_BUILD_TESTS=OFF`, `LLAMA_CURL=OFF` (fork also `LLAMA_RORET_CPU_TEST=ON`): checker exit 0, verdict PASS, 26/26 checks passed, blockers empty, `stdout_differing_bytes` 0. Stdout was 918 bytes on each side; raw stdout differed only in the `build : <build-info>` line, which the gate removes before comparison. `fork_stdout_sha256` `faa87c2b22949dbab87e032a5c73257577f3c566aa0847ef38a3d0e08b91feb1`, `upstream_stdout_sha256` `6e07a87cc3613a4c82e62f62e6597e81975156afd7600e8f2ebdb01aa394397f`, `fork_cli_sha256` `0216eda9af0b0cf45457c9d2e87cc7b6cf09bc4b05735edd2bf8b517885dc323`, `upstream_cli_sha256` `f1599b90d956aa75657f7b8b4c1e2dde45ca5549c210bf70ba61a28e105a2111`. Both main exits were 0 and both enable markers were absent; the positive control printed the marker for the fork binary and not for the upstream binary. Saved as `M44_UPSTREAM_CLI_COMPAT_SUMMARY.json` sha256 `38ab9c92f7c378e7f4ed23d834b2bd60367207b754e80f37e6f359c8a5ce8122`. A real negative control with the two CLI paths swapped exited 1 with verdict NO_GO and four failing checks (both CLI hash pins and both control marker checks); it wrote to a separate output path and left the PASS summary unchanged. Scope: synthetic fixture only, no real weights, no GPU, no timing. NO SPEED CLAIM; CycleAudit `claim<=ci_low`, `repeats>=20`, `loops=3` remains mandatory for any future number.
