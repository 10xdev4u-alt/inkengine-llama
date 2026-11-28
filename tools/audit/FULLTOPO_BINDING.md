# M45 - CPU-only full-topology 3-pass union-wiring gate

This is an offline, stdlib-only, fail-closed gate. It drives an already built
CPU-only `llama-debug` twice on a deterministic synthetic GGUF that carries the
real Inkling2 loop structure: 24 physical layers, logical core 6-17, 3 passes,
rank-64 loop controllers, `attn_res.block_size=6` and a 48-slot loop cache.
Only the weights and the small per-layer dimensions are synthetic. OFF leaves
`LLAMA_RORET` unset; ON sets `LLAMA_RORET=1` and `LLAMA_RORET_TRACE=1`. Both
phases dump MoE tensors into phase-private directories. The gate writes
`tools/audit/M45_FULLTOPO_BINDING_SUMMARY.json`. Exit 0 means PASS; exit 1 means
NO_GO. No network, no build step, no timing and no weights.

**PASS means only that, on a deterministic synthetic GGUF carrying the real
Inkling2 structural topology (24 layers, core 6-17, 3 passes, rank-64
controllers, 48 slots) on a CPU-only build, the union-resident MoE hook
registers bindings for every core layer on passes 0/1/2 with disjoint
pass-indexed slot maps and all mapped physical slots present; OFF/ON raw MoE
tensors, logits and text are byte-identical. It is NOT reference tensor-value correctness, NOT real-weight
parity, NOT GPU copy-callback hit/upload traffic, and NOT any
timing/throughput/wall-clock result.**

**NO SPEED CLAIM: M24 eligible CycleAudit placed the whole 95% CI below 1.0.
Any future number still requires claim<=ci_low, repeats>=20, loops=3.**

No weights are used and there is no Kaggle kernel.

## Commands

From the fork root, with the built binary and generated fixture:

```
PYTHONPATH=gguf-py python3 tools/audit/make_fulltopo_gguf.py \
  --out /tmp/opencode/fulltopo-probe.gguf
python3 tools/audit/check_fulltopo_binding.py --root . \
  --cli /tmp/opencode/m44-probe/bin/llama-debug \
  --gguf /tmp/opencode/fulltopo-probe.gguf \
  --out-dir /tmp/opencode/m44-real
python3 -m pytest tools/audit/test_fulltopo_binding.py -q
```

The binary was built out of tree with:

```
cmake -S . -B /tmp/opencode/m44-probe -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DLLAMA_RORET_CPU_TEST=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/m44-probe --target llama-debug -j4
```

The fixture is NOT a real model. Every weight is fixed-seed random f32 data.
The default seed 0 gives 11941248 bytes with sha256
`3ccf2fb4c6ef4fab60464d2a35a2ea42b24186db3e8b12cd68881f4fa873dff7`, 708
tensors and arch `inkling2_loop`. The checker verifies sha256, byte count and
arch string before it runs the engine. A wrong fixture is a NO_GO. The
structural constants are the upstream ones: 24 layers, core 6..17, 3 passes,
rank-64 controllers, block size 6 and 48 cache slots. Only the per-layer
dimensions and the weights are simplified.

The engine command is identical in both phases except for the environment:

```
llama-debug -m FIXTURE -p 'hello world' -n 4 -t 1 -cmoe \
  --moe-cache-mib 16 --moe-cache-cpu-share 1.0 --verbose \
  --tensor-filter '^ffn_moe_(logits|probs|topk|weights|out)-[0-9]+$'
```

`LLAMA_DUMP_TENSORS_DIR` points at a phase-private dump directory that is
cleared before each run. `LLAMA_RORET` and `LLAMA_RORET_TRACE` are unset in OFF
and set to `1` in ON. `--cpu-share` overrides the share in both phases. The
300-second subprocess timeout is a guard only; it is never measured or
recorded. `llama-debug` has no `-o` flag, so the generated text is the textual
stdout, which includes the tensor printout and prompt echo.

## Measured values (real binary, real fixture)

| Quantity | OFF | ON |
| --- | --- | --- |
| Engine exit | 0 | 0 |
| Manifest rows | 480 | 480 |
| Distinct tensor names | 240 | 240 |
| Physical slots from names | 0-47 | 0-47 |
| Occurrences per name | 2 | 2 |
| `RORET_BIND set` rows, `key_il=6..17`, pass 0 / 1 / 2 | 0 | 144 / 144 / 144 |
| Distinct `(key_il, pass)` pairs | 0 | 36 |
| `RORET_BIND set` sentinel rows `key_il=-1 pass=-1` | 0 | 144 |
| `slot_map` pointers per pass | 0 | 24 / 24 / 24, pairwise disjoint |
| `RORET_PASS` rows | 0 | 0 (informational) |
| Enable marker | absent | present |
| Build marker `LLAMA_RORET_CPU_TEST...` | 2 | 2 |

Other measured values: 480 `.bin` files compared, 360960 raw bytes compared,
0 differing bytes; 245927 stdout text bytes identical, 0 differing bytes;
normalized stdout 0 differing bytes and 0 dropped lines in either phase. The
summary verdict is PASS with 35/35 checks true and an empty blocker. These
values were reproduced with the independent feasibility binary and the final
CPU-only build at `/tmp/opencode/m44-probe/bin/llama-debug`.

## What is proved

All checks are booleans. Verdict is PASS only when every check is true. The
contract `tools/audit/FULLTOPO_BINDING.json` freezes the exact top-level key
set, schema version, date, null scope/note/speed claim, fixture sha256 and
bytes, architecture, loop constants, the derived pass-to-slot map, the 12
logical core layers, the three passes, the five tensor kinds, all 240 expected
tensor names, the occurrence count, the discriminating enable needle, the
unconditional build marker, the trace needle, the OFF-absent needles, the
manifest field count, the summary path, the ordered `verify` list and this
checker's own sha256. The pin lives only in the contract, never inside the
checker source. Tests additionally pin the checker and contract sha256, so a
coordinated edit of both is visible in two places. Any missing, malformed or
extra contract key fails closed.

The pass-to-slot map is derived from the frozen constants, not from the dump:
pass 0 uses physical slots 6-17, pass 1 uses 24-35 and pass 2 uses 36-47.
Slots 18-23 exist in the 48-slot cache but are not loop-core slots. The ON run
must show the exact 240-name set across physical slots 0-47, exactly two
occurrences per name and 480 manifest rows, and every one of the 36
`(key_il, pass)` pairs for `key_il` in 6-17 and pass in 0-2. No non-sentinel
binding may use another `key_il` or pass. Binding rows per pass must be equal
and positive; OFF must have no binding row, no `RORET_PASS` row and no enable
needle. The trace also carries a `slot_map=` pointer per bind row: the checker
requires the per-pass pointer sets to be non-empty, nonzero, equally sized and
pairwise disjoint, which is observed separation of slot maps rather than a
tautology from the frozen map. The mapped physical slots must all appear in the
ON dump. This is set-level co-occurrence, not per-event binding-to-tensor
pointer linkage; raw addresses are not recorded because ASLR varies them per
run. The occurrence suffix (`__0`, `__1`) is a graph-eval index, not a pass
index.

Raw `.bin` bytes from OFF and ON are compared for the same filenames; any
differing file is a NO_GO. Generated text (raw stdout) is compared
byte-for-byte. A second stdout comparison drops only lines whose stripped form
starts with one of the anchored trace/enable needles, plus llama.cpp's
non-deterministic `[ Prompt: ... t/s | Generation: ... t/s ]` line; the number
of dropped lines must be equal in OFF and ON. No timing value is compared,
recorded or inferred.

`RORET_PASS` rows from ON are parsed and reported as `rorret_pass_rows_on_informational`
but are never a gate. They require the device slot-map copy and are not
expected on this CPU fallback; see M21/M22. The unconditional
`LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache` line is likewise
informational, with per-phase counts in `info`.

Each run unlinks any previous summary, writes a NO_GO sentinel before any
check, then writes the final verdict through a temp file plus `os.replace`. A
killed or interrupted run therefore leaves NO_GO rather than a stale PASS. A
final write failure exits 1. `--root` checks an isolated copy; all tamper tests
use isolated temp trees and never touch the real generated summary.

## What is NOT proved

- No real weights. The fixture is fixed-seed random f32 data with real
  structure only.
- No GPU, no CUDA build, no Kaggle kernel and no hosted inference.
- No reference tensor-value correctness: OFF/ON byte parity says nothing about
  whether either value matches an external model.
- No GPU copy-callback hit/upload traffic. `RORET_PASS` is informational and
  the CPU fallback does not produce the device traffic rows.
- No timing, throughput, latency, cache-hit saving or speedup number.
- No per-event binding-to-tensor pointer linkage; only set-level slot-map
  separation plus the frozen pass-to-slot map and the dump slot set.
- A future engine change that keeps the same trace and dump bytes but changes
  semantics is outside this contract.

## Honest regeneration rule

A legitimate engine, flag or fixture change can move the measured bind counts,
manifest bytes or checker hash. Regenerate the summary openly by rerunning the
checker, update the contract pins in the same change, and explain why in
`BUILD_STATUS.md`. A silent pin update that merely makes the gate pass again
is exactly the drift this gate exists to prevent.
