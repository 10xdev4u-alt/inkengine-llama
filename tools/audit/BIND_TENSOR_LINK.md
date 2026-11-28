# M42 - fail-closed per-event binding-to-tensor pointer linkage gate

This closes the gap M41 left explicitly: M41 proved only set-level co-occurrence
between bind rows and the physical slots seen in the dump. M42 proves per-event
pointer identity. It drives an already built CPU-only `llama-debug` twice on the
deterministic synthetic Inkling2-loop GGUF: OFF (`LLAMA_RORET` unset) and ON
(`LLAMA_RORET=1`, `LLAMA_RORET_TRACE=1`, `LLAMA_DUMP_TENSOR_PTR=1`). The ON run
writes a separate two-column `manifest_ptr.tsv` next to `manifest.tsv`. The
checker writes `tools/audit/M42_BIND_TENSOR_LINK_SUMMARY.json`. Exit 0 means
PASS; exit 1 means NO_GO. No network, no build step, no timing and no weights.

## Source change (default off)

`common/debug.cpp` reads `LLAMA_DUMP_TENSOR_PTR` once, next to the existing
`LLAMA_DUMP_TENSORS_DIR` static. Only `1`, `on` and `true` enable it; everything
else is off. When enabled, every dumped tensor appends `filename<TAB>ptr` to
`manifest_ptr.tsv` using `(const void *) t` and `%p`. The 12-field
`manifest.tsv` schema is unchanged, so the M41 pin still holds. A write failure
logs an error but does not abort the engine. With the flag unset, no
`manifest_ptr.tsv` is created and the dump bytes are unchanged. No other source
file was changed.

## Commands

From the fork root, with the already built binary and fixture:

```
cmake --build /tmp/opencode/m41-probe --target llama-debug -j4
python3 tools/audit/check_bind_tensor_link.py --root . \
  --cli /tmp/opencode/m41-probe/bin/llama-debug \
  --gguf /tmp/opencode/synthetic.gguf \
  --out-dir /tmp/opencode/m42-real
python3 -m pytest tools/audit/test_bind_tensor_link.py -q
PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit -q
```

The fixture is regenerated, when needed, with the default seed:

```
PYTHONPATH=gguf-py python3 tools/audit/make_synthetic_gguf.py \
  --out /tmp/opencode/synthetic.gguf
```

It is 887392 bytes with sha256
`cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617` and arch
`inkling2_loop`. The checker verifies sha256, byte count and arch before it runs
the engine. A wrong fixture is a NO_GO.

The engine command is identical in both phases except for the environment:

```
llama-debug -m FIXTURE -p 'hello world' -n 4 -cmoe \
  --moe-cache-mib 16 --moe-cache-cpu-share 1.0 --verbose \
  --tensor-filter '^ffn_moe_(logits|probs|topk|weights|out|slots)-[0-9]+$'
```

`LLAMA_DUMP_TENSORS_DIR` points at a phase-private dump directory which is
cleared before each run. OFF explicitly unsets all three RORET/pointer flags;
ON sets them. Each run has a 300 second subprocess timeout used only as a guard;
it is never recorded or used as a measurement.

## Measured values (real binary, real fixture)

| Quantity | OFF | ON |
| --- | --- | --- |
| Engine exit | 0 | 0 |
| Manifest rows | 48 | 48 |
| Distinct tensor names | 24 | 24 |
| Physical slots from names | 0,1,2,3 | 0,1,2,3 |
| Occurrences per name | 2 | 2 |
| Pointer manifest rows | 0 (file absent) | 48 |
| `RORET_BIND set` rows, `key_il=1`, pass 0 / 1 / 2 | 0 | 12 / 12 / 12 |
| `RORET_BIND set` sentinel rows `key_il=-1 pass=-1` | 0 | 12 |
| Total `RORET_BIND set` rows | 0 | 48 |
| Dumped slot tensor rows | 0 pointer rows | 8 (4 names x 2) |
| Pass-bound slot pointer events | 0 | 6 (2 per pass) |
| Sentinel-bound slot pointer events | 0 | 2 |
| Set rows with no dumped slots tensor | n/a | 30 |

Other values: 48 `.bin` files compared, 30720 raw bytes compared, 0 differing
bytes; 23892 stdout bytes identical, 0 differing bytes. The real-run verdict is
PASS with 45/45 checks true and an empty blocker. Two consecutive real runs
produced byte-identical summaries. These numbers come from the real binary and
real fixture, not from the fake engine used by the unit tests.

## What is proved

The contract `tools/audit/BIND_TENSOR_LINK.json` freezes the exact top-level key
set, the scope string, schema, date, note, fixture pins, loop constants,
pass-to-slot map, tensor kinds/names/occurrences, pointer-manifest field names,
expected slot basenames per pass, expected bind/sentinel counts, expected
matched counts, enable and trace needles, OFF-absent needles, the summary path,
the ordered `verify` list and this checker's own sha256. The pin lives only in
the contract. Tests additionally pin the contract sha256 and the checker
sha256, so a coordinated edit of both is visible in two places. Any missing,
malformed or extra contract key fails closed.

The core slots and the pass-to-slot map `{0:1, 1:2, 2:3}` are derived from the
frozen fixture constants, not from the dump. The checker parses the full
`RORET_BIND set lookup=<ptr> slot_map=<ptr> key_il=<int> pass=<int>` rows. For
every dumped `ffn_moe_slots-<il>__N.bin` row it requires the pointer in
`manifest_ptr.tsv` to be literally one of the `RORET_BIND set` lookup pointers,
with `key_il=1` and the pass whose derived slot is `<il>`. The `ffn_moe_slots-0`
rows are instead matched to the `key_il=-1 pass=-1` sentinel rows and are never
counted as pass events. Per pass, the matched slot basenames must be exactly one
expected basename, the number of bound set rows must equal the number of dumped
slot rows, and the matched pointer sets must be pairwise disjoint across passes.
A pointer placed on the wrong `il` basename, shared across passes, mutated,
duplicated or missing fails closed.

The summary records only counts, booleans and basenames. Raw addresses are
never written to it because ASLR varies them per run. This is what makes two
runs byte-identical while still proving the identity: the comparison happens
inside one ON run between its own trace and its own pointer manifest. The
pointer identity was also inspected directly in the real ON artifacts before
this gate was accepted: the dumped `ffn_moe_slots-1`, `-2` and `-3` pointers
each equaled a `key_il=1` set lookup for passes 0, 1 and 2 respectively, and the
`ffn_moe_slots-0` pointer equaled a sentinel set lookup.

Only some graph builds produce a dumped tensor: the 30 non-sentinel set rows
whose lookup has no dumped slots row are explicitly counted as `unmatched_set_rows_on`
and are not claimed. The occurrence suffix `__0`/`__1` encodes graph-eval
count, not pass, and is never used as a pass identifier. OFF must have zero bind
rows, zero pointer-manifest rows and zero pointer-bearing slot dumps, and raw
`.bin` and stdout bytes must match ON exactly.

Each run unlinks any previous summary, writes a NO_GO sentinel before any check
or engine run, then writes the final verdict through a temp file plus
`os.replace`. A killed run therefore leaves NO_GO rather than a stale PASS. A
final write failure exits 1. `--root` checks an isolated copy; all tamper tests
use isolated temp trees and never touch the real generated summary.

## What is NOT proved

- No real weights. The fixture is a fixed-seed random-weight synthetic model.
- No GPU, no CUDA build, no Kaggle kernel and no hosted inference.
- No timing, throughput, latency, cache-hit saving or speedup number.
- No tensor-value equality to a reference; only OFF/ON byte parity.
- No per-event linkage from a binding row to the copy callback: this CPU log
  has zero `resolve`/`hit` rows, so that step is out of scope here.
- No claim that all 36 non-sentinel set rows are dumped; 30 have no dumped
  slots tensor and are reported, not silently treated as evidence.
- A future engine change that keeps the same trace and manifest bytes but
  changes semantics is outside this contract.

NO SPEED CLAIM. CycleAudit `claim<=ci_low`, `repeats>=20`, `loops=3` remains
mandatory for any future number.

## Honest regeneration rule

A legitimate engine, flag or fixture change can move the measured counts,
manifest bytes or checker hash. Regenerate the summary openly by rerunning the
checker, update the contract pins in the same change, and explain why in
`BUILD_STATUS.md`. A silent pin update that merely makes the gate pass again is
exactly the drift this gate exists to prevent.
