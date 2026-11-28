# M41 - fail-closed runtime 3-pass loop replay / pass-to-slot mapping gate

This is an offline, stdlib-only, fail-closed gate. It drives an already built
CPU-only `llama-debug` twice on the deterministic synthetic Inkling2-loop GGUF:
once with `LLAMA_RORET` and `LLAMA_RORET_TRACE` unset (OFF) and once with both
set to `1` (ON). Each phase dumps the selected MoE tensors through
`LLAMA_DUMP_TENSORS_DIR` and writes a `manifest.tsv` plus per-occurrence raw
`.bin` files. The gate ties the three ON runtime passes to the three distinct
physical core cache slots of the same ON run. It writes
`tools/audit/M41_3PASS_REPLAY_SUMMARY.json`. Exit 0 means verdict PASS; exit 1
means NO_GO. No network, no build step, no timing and no weights.

## Commands

From the fork root, with the already built binary and fixture:

```
python3 tools/audit/check_3pass_replay.py --root . \
  --cli /tmp/opencode/m41-probe/bin/llama-debug \
  --gguf /tmp/opencode/synthetic.gguf \
  --out-dir /tmp/opencode/m41-real
python3 -m pytest tools/audit/test_3pass_replay.py -q
PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit -q
```

The binary was built out of tree with:

```
cmake -S . -B /tmp/opencode/m41-probe -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DLLAMA_RORET_CPU_TEST=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/m41-probe --target llama-debug -j4
```

The fixture is regenerated, when needed, with the default seed:

```
PYTHONPATH=gguf-py python3 tools/audit/make_synthetic_gguf.py \
  --out /tmp/opencode/synthetic.gguf
```

It is 887392 bytes with sha256
`cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617` and arch
`inkling2_loop`. The checker verifies the sha256, byte count and arch string
before it runs the engine. A wrong fixture is a NO_GO.

The engine command is identical in both phases except for the environment:

```
llama-debug -m FIXTURE -p 'hello world' -n 4 -cmoe \
  --moe-cache-mib 16 --moe-cache-cpu-share 1.0 --verbose \
  --tensor-filter '^ffn_moe_(logits|probs|topk|weights|out)-[0-9]+$'
```

`LLAMA_DUMP_TENSORS_DIR` points at a phase-private dump directory which is
cleared before each run. OFF leaves `LLAMA_RORET` and `LLAMA_RORET_TRACE`
unset; ON sets both to `1`. Each run has a 300 second subprocess timeout used
only as a guard; it is never recorded or used as a measurement.

## Measured values (real binary, real fixture)

| Quantity | OFF | ON |
| --- | --- | --- |
| Engine exit | 0 | 0 |
| Manifest rows | 40 | 40 |
| Distinct tensor names | 20 | 20 |
| Physical slots from names | 0,1,2,3 | 0,1,2,3 |
| Occurrences per name | 2 | 2 |
| `RORET_BIND set` rows, `key_il=1`, pass 0 / 1 / 2 | 0 | 12 / 12 / 12 |
| `RORET_BIND set` sentinel rows `key_il=-1 pass=-1` | 0 | 12 |
| Total `RORET_BIND set` rows | 0 | 48 |
| Enable marker `RORET union-resident MoE bookkeeping enabled` | absent | present |
| Build marker `LLAMA_RORET_CPU_TEST...` | 2 | 2 |

Other measured values: 40 `.bin` files compared, 30080 raw bytes compared, 0
differing bytes; 20516 stdout bytes identical, 0 differing bytes. The summary
verdict is PASS with 31/31 checks true and an empty blocker. These values match
the independent feasibility probe recorded under `/tmp/opencode/m41-probe`;
there was no deviation.

The 20 expected names are `ffn_moe_<kind>-<il>` for `kind` in `logits`,
`probs`, `topk`, `weights`, `out` and `il` in `0,1,2,3`. Slot `il=0` is the
prelude; the loop core slots are `1,2,3`. Each name occurs twice (`__0`, `__1`)
because there are two graph evals.

## What is proved

All checks are booleans. Verdict is PASS only when every check is true. The
contract `tools/audit/THREEPASS_REPLAY.json` freezes the exact top-level key
set, schema version, date, scope, fixture sha256/bytes/arch, loop constants,
expected passes, core and full slot sets, the pass-to-slot map, tensor kinds,
names and occurrence count, the single discriminating enable needle, the
unconditional build marker, the required trace needle, the OFF-absent needles,
the manifest field count, the summary path, the ordered `verify` list and this
checker's own sha256. The pin lives only in the contract, never inside the
checker source. Tests additionally pin the contract sha256 and the checker
sha256, so a coordinated edit of both is visible in two places. Any missing,
malformed or extra contract key fails closed.

The core slots are derived from the frozen fixture constants
`N_LAYER=2, CORE_START=1, CORE_END=2, N_PASSES=3, GATE_RANK=8`, not from the
dump. With one logical core layer, pass `p` maps to physical slot
`CORE_START + p`, giving `{0:1, 1:2, 2:3}`. The ON run must show the exact
manifest name set and slot set, exactly two occurrences per name, exactly the
pass set `{0,1,2}` for `key_il=1`, equal positive bind counts in all three
passes, zero bind rows OFF, correct enable marker polarity, and raw `.bin` and
stdout byte parity between OFF and ON. Wrong or missing manifest rows, missing
files, a nonzero engine exit and any of the above mismatches fail closed.

The cross-correlation is performed inside the same ON run: the pass set is read
from the ON trace and the core slot set is read from the ON manifest, then
compared through the derived pass-to-slot map. The trace also carries a
`slot_map=` pointer per bind row; the checker now extracts those pointers and
requires the per-pass pointer sets to be non-empty, equally sized, nonzero and
pairwise disjoint (two pointers per pass here, one per graph eval). A shared
`slot_map` pointer across passes fails closed, so this is observed separation
of slot maps rather than a tautology from the frozen map. Raw addresses are
intentionally not written to the summary because ASLR varies them per run;
only per-pass pointer counts and the disjointness verdict are recorded, which
makes the generated summary byte-identical across runs. It is still **set-level
co-occurrence**, not a per-event pointer linkage: the checker does not claim
that a particular `RORET_BIND set` row points at a particular dumped tensor.
The occurrence index (`__0`, `__1`) encodes graph-eval count, not pass; it is
never used as a pass identifier. The fixture sha256 and byte count are pinned
both in the contract and as literals in the checker, so editing both a forged
fixture and its contract pin still fails closed.

`LLAMA_RORET_CPU_TEST: using CPU backend for MoE cache` is emitted
unconditionally by the test-only CPU build in both phases. It is recorded as a
non-gating informational build marker with per-phase counts, never as an enable
needle or a device or speed claim.

Each run unlinks any previous summary, writes a NO_GO sentinel before any
check, then writes the final verdict through a temp file plus `os.replace`. A
killed or interrupted run therefore leaves NO_GO rather than a stale PASS. A
final write failure exits 1. `--root` checks an isolated copy; all tamper tests
use isolated temp trees and never touch the real generated summary.

## What is NOT proved

- No real weights. The fixture is a fixed-seed random-weight synthetic model.
- No GPU, no CUDA build, no Kaggle kernel and no hosted inference.
- No timing, throughput, latency, cache-hit saving or speedup number.
- No per-event binding-to-tensor linkage; only disjoint per-pass slot-map
  pointer sets plus set-level pass/slot co-occurrence.
- No claim that the occurrence suffix is a pass index; it is an eval index.
- No tensor-value equality to a reference; only OFF/ON byte parity.
- A future engine change that keeps the same trace and manifest bytes but
  changes semantics is outside this contract.

NO SPEED CLAIM. CycleAudit `claim<=ci_low`, `repeats>=20`, `loops=3` remains
mandatory for any future number.

## Honest regeneration rule

A legitimate engine, flag or fixture change can move the measured bind counts,
manifest bytes or checker hash. Regenerate the summary openly by rerunning the
checker, update the contract pins in the same change, and explain why in
`BUILD_STATUS.md`. A silent pin update that merely makes the gate pass again is
exactly the drift this gate exists to prevent.
