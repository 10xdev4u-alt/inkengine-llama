# M40 - CPU-only 3-pass runtime binding gate

This is an offline, stdlib-only, fail-closed gate. It drives an already built
CPU-only `llama-cli` twice on the deterministic synthetic Inkling2-loop GGUF:
once with `LLAMA_RORET` unset (OFF) and once with `LLAMA_RORET=1` plus
`LLAMA_RORET_TRACE=1` (ON). The optional `--cpu-share` flag defaults to `1.0`:
the sibling value `0` aborts during warmup in both OFF and ON phases on this
CPU-only build, so it cannot yield a meaningful binding measurement.
It parses the default-off RORET trace and writes
`tools/audit/M40_3PASS_BINDING_SUMMARY.json`. Exit 0 means verdict PASS;
exit 1 means NO_GO. No network access, no build step, no timing and no weights.

## Commands

From the inkengine-llama root:

```
python3 tools/audit/check_3pass_binding.py --cli /tmp/opencode/ink-build/bin/llama-cli --gguf /tmp/opencode/synthetic.gguf
PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit/test_3pass_binding.py -q
PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit -q
```

The fixture is regenerated, when needed, with the default seed:

```
PYTHONPATH=gguf-py python3 tools/audit/make_synthetic_gguf.py --out /tmp/opencode/synthetic.gguf
```

It is 887392 bytes with sha256
`cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`. The
checker verifies that sha256 and the `inkling2_loop` architecture string in
the GGUF header before it runs the engine. A wrong fixture is a NO_GO.

The engine command is identical in both phases except for the environment:

```
llama-cli -m FIXTURE -p hello -n 4 -t 1 -s 0 --temp 0 -cmoe \
  --moe-cache-mib 16 --moe-cache-cpu-share 1.0 --verbose -o GENERATED.txt
```

`-s 0` and `--temp 0` remove sampling randomness. `-o` captures the generated
text block as a separate file. Each run has a 300 second subprocess timeout;
the timeout is a guard only and is never recorded or used as a measurement.

## What is proved

All checks are booleans. Verdict is PASS only when every check is true. The
contract `tools/audit/THREEPASS_BINDING.json` freezes the exact top-level key
set, schema version, date, scope, fixture sha256, architecture, key_il, the
three expected passes, the single discriminating enable needle, the
unconditional `informational_build_markers` string, the required trace
needle, the informational needles, the CPU traffic scope string, the
OFF-absent needles, the summary path, the ordered `verify` list and this
checker's own sha256. The pin lives only in the contract, never inside the
checker source. Any missing, malformed or extra contract key fails closed.
Tests also pin the contract's own sha256, so a coordinated edit of both
contract and checker is visible in two places.

The required scope is exactly three-pass `RORET_BIND set` binding, hook
inertness when OFF, and OFF/ON output parity. `RORET_PASS` traffic rows
are `informational_needles`: they are parsed and reported, never gated.
Two of the 21 checks (`trace_parse_completed` and `run_completed`) are
liveness flags, not evidence. Parse completion with zero traffic rows is
recorded, not treated as traffic evidence.

- OFF phase: exit 0 and no line containing `RORET_BIND`, `RORET_PASS` or
  `RORET union-resident MoE bookkeeping enabled` in combined stdout/stderr.
- ON phase: exit 0 and the one discriminating enable needle present -
  `RORET union-resident MoE bookkeeping enabled`. The
  `LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache` line is emitted
  unconditionally by the test-only CPU build, appears in both OFF and ON
  stderr, and is recorded as a non-gating build marker with per-phase counts
  in the summary `info` list. It is not an enable signal, a device claim or a
  speed claim.
- ON trace: at least one `RORET_BIND set` row for `key_il=1` at each pass in
  `{0, 1, 2}`. The distinct pass set must be exactly `{0, 1, 2}`.
- ON trace: no `RORET_BIND no_mulmat` row for `key_il=1` with a pass in
  `{0, 1, 2}`. This is the CPU-only form of the M21/M22 early-return control.
  The total `no_mulmat` row count, regardless of `il` and `pass`, is reported
  as `no_mulmat_total`. On the CPU path zero rows is the only observable
  property; a passing check does not measure device-side matmul behavior.
- ON trace parse completion: the `trace_parse_completed` liveness check is
  true whenever the RORET trace parse succeeds. Parsed `RORET_PASS il=1 pass=P` rows are
  reported in `pass_row_passes` and `pass_row_count`, but they are
  informational only and are never a gate. They are engine counts, not time
  or throughput. On the CPU fallback the count is expected to be zero:
  `RORET_PASS` requires the device slot-map copy, which the `GGML_CUDA=OFF`
  scheduler never performs. This is the known GPU-only gap from M21/M22 and
  is recorded verbatim in the contract's `cpu_traffic_scope` string and in
  the summary `info` list.
- OFF/ON parity: the generated-text files written by `-o` are non-empty and
  byte-identical. Separately, full stdout is compared byte-for-byte after a
  normalization that drops only lines whose stripped form *starts with* one
  of these needles: `RORET_BIND`, `RORET_PASS`,
  `RORET union-resident MoE bookkeeping enabled`, and
  `LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache`. An embedded
  occurrence later in a line is not dropped and therefore fails parity.
  It also drops llama-cli's non-deterministic metrics line
  `[ Prompt: ... t/s | Generation: ... t/s ]`; no timing value is compared or
  recorded. The number of dropped lines must be equal in OFF and ON; a
  mismatch fails closed. All other stdout bytes, including order and line
  endings, must match. Dropped-line counts are reported in `info`. Trace
  lines are read from both streams, but normalization applies to stdout only;
  stderr carries INFO traffic that is allowed to differ.
- A static AST self-check confirms this checker imports no network module and
  makes no URL-fetch call.

Each run unlinks any previous summary, writes a NO_GO sentinel before any
check, then writes the final verdict through a temp file plus `os.replace`.
A killed or interrupted run therefore leaves NO_GO rather than a stale PASS.
A final write failure exits 1. `--root` checks an isolated copy; the tamper
tests use it against temp trees and never touch the real generated summary.

## What is NOT proved

- No real weights. The fixture is a fixed-seed random-weight synthetic model.
- No GPU, no CUDA build, no Kaggle kernel and no hosted inference.
- No timing, throughput, latency, cache-hit saving or speedup number.
  NO SPEED CLAIM. CycleAudit `claim<=ci_low`, `repeats>=20`, `loops=3`
  remains mandatory for any future number.
- No tensor, activation or logits parity; only generated text and normalized
  stdout are compared here. M26 covered intermediate tensors separately.
- Engine traffic rows are not measured: `RORET_PASS` needs the device slot-map
  copy and is a known GPU-only gap (M21/M22). Any rows seen here are
  informational counts of bookkeeping events, not evidence of a faster path.
- A future source or engine change that keeps the same trace bytes but changes
  semantics is outside this contract.

## Honest regeneration rule

A legitimate engine, flag or fixture change can move the measured pass set,
generated text or checker hash. Regenerate the summary openly by rerunning the
checker, update this contract's pins in the same change, and explain why in
`BUILD_STATUS.md`. A silent pin update that merely makes the gate pass again
is exactly the drift this gate exists to prevent.
