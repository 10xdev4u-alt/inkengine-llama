# RORET fixture parity (M13): host/tile bit-exact gate with real fixture ids

**Verdict: HOST/TILE-level PASS. Engine-level GPU final-logits parity is
PASS-scoped as of M18 (it was NO_GO at this M13 increment); intermediate
per-pass MoE tensors and real weights remain NOT measured.**

This milestone extends the chain from `union_census_ids.inc` (real seed-0
fixture ids) and the per-pass traffic replay to a bit-exact tile comparison over
all 12 loop layers. It adds `tools/roret/test_roret_fixture_parity.cpp`,
registers it in `tools/roret/CMakeLists.txt`, and records the evidence here.
No file under `src/` was edited. No commit was made. Nothing in `DECISIONS.tsv`
was edited.

## 1. Provenance pinned by the checked-in generated file

`tools/roret/union_census_ids.inc` is a checked-in generated artifact; no Python
or loopkit code runs when the new test executes. Its header records the exact
producer hashes:

| Artifact | SHA-256 |
|---|---|
| `fixture_sha256` | `fff0e60fa4f55ba717d1e0bdc0e89695a0d35c85df680a15fd57ed1a0ac959ff` |
| `dump_sha256` | `c85a8629c4b45a6f81bc505f5959dea469249f13e18029c47d13513fb686f5ec` |

Embedded shape: `[ip-6][pass][token][k] = [12][3][4][8]`, 128 experts, top-8,
ip 6..17. The test asserts these constants at compile time.

## 2. What the test measures

For every `(ip, token)` a real `llama_roret_union_cache` timeline supplies the
union of earlier passes in the bookkeeping order of `llama_moe_cache::copy()`:
`clear_layer()` on pass 0, `residents()` sampled before `add()` for the counter,
then this pass's ids registered. `copy()` samples the *hints* after `add()`
(`src/llama-moe-cache.cpp:612-615`); the two are equivalent here because the
current pass's ids are already in `ids` and `plan()` deduplicates them. Pass 0
has empty hints; pass p>0 has the union of passes 0..p-1. The test checks
pass-0 hints are empty, hints contain every earlier pass's ids, and the union
is monotone across passes for each `(ip, token)`.

For every `(ip, pass, token)` two fresh runners use the real
`moe_cache_lru::plan()` copied verbatim in `src/llama-moe-cache-lru.h`:

* **OFF**: `plan(ids, ..., no hints)`; earlier passes replayed with no hints.
* **ON**: `plan(ids, ..., extra = earlier-pass union, n_extra = its size)`;
  earlier passes replayed with their own earlier-pass hints, so a hint that
  harms a later pass is observable.

Both scheduler splits are run: `cpu_share = 0` (`n_zero = 0`) and
`cpu_share > 0` (`n_zero = kCensusTopK = 8`). The host-miss share is the
`(n_miss+1)/2` stress model from the existing tile test, not the engine's
cache-full fixed-point `cpu_share` accumulator; the `n_zero` settings match
the existing tile test. Every cached slot byte is compared against that expert's deterministic
F32 tile (16 floats / 64 bytes) and every CPU-miss zero slot is checked to be
all zeros. The gathered tile bytes for all 8 real selections are compared OFF
vs ON. Zero differing bytes are required.

A CPU-missed expert is recomputed on the host from its real tile, so the
comparison stream uses that tile in place of the device's zero placeholder.
This matters: before this modeling was added, 9 of 288 target events (all pass
2 with `n_zero = 8`) showed raw device-gather byte differences because hints
moved experts from a CPU zero slot to a cached slot. `hit_on >= hit_off` held in
every event. Comparing raw zero-slot bytes would report a false tile mismatch;
the test still asserts the zero slots themselves are all-zero. In the final run
15 events had different hit counts and 1093 of 4608 gathered tile reads came
from host recomputation, yet every compared tile byte matched.

## 3. Commands and measured output

All commands were run from the repository root at `HEAD = b85edf4`, branch
`main`, with GCC 16.2.1 and CMake 4.4.4.

### Fresh cmake build and ctest (6/6)

```text
$ rm -rf /tmp/opencode/m13-roret-build
$ cmake -S tools/roret -B /tmp/opencode/m13-roret-build
-- Configuring done (0.5s)
-- Generating done (0.0s)
-- Build files have been written to: /tmp/opencode/m13-roret-build
$ cmake --build /tmp/opencode/m13-roret-build -j
[100%] Built target test_roret_fixture_parity
$ ctest --test-dir /tmp/opencode/m13-roret-build --output-on-failure
Test project /tmp/opencode/m13-roret-build
    Start 1: test_union_cache
1/6 Test #1: test_union_cache .................   Passed    0.00 sec
    Start 2: test_roret_traffic
2/6 Test #2: test_roret_traffic ...............   Passed    0.00 sec
    Start 3: test_roret_tile_bitexact
3/6 Test #3: test_roret_tile_bitexact .........   Passed    0.01 sec
    Start 4: roret_union_census
4/6 Test #4: roret_union_census ...............   Passed    0.00 sec
    Start 5: roret_pass_traffic
5/6 Test #5: roret_pass_traffic ...............   Passed    0.00 sec
    Start 6: roret_fixture_parity
6/6 Test #6: roret_fixture_parity .............   Passed    0.06 sec

100% tests passed out of 6

Total Test time (real) =   0.08 sec
```

### Standalone C++17 and C++20, `-Werror`

```text
$ g++ -std=c++17 -Wall -Wextra -Werror -I. tools/roret/test_roret_fixture_parity.cpp -o /tmp/opencode/m13p17
$ /tmp/opencode/m13p17 > /tmp/opencode/m13-p17.out 2> /tmp/opencode/m13-p17.err
exit 0; stderr empty

$ g++ -std=c++20 -Wall -Wextra -Werror -I. tools/roret/test_roret_fixture_parity.cpp -o /tmp/opencode/m13p20
$ /tmp/opencode/m13p20 > /tmp/opencode/m13-p20.out 2> /tmp/opencode/m13-p20.err
exit 0; stderr empty

$ cmp /tmp/opencode/m13-p17.out /tmp/opencode/m13-p20.out
stdout byte-identical (exit 0)
$ /tmp/opencode/m13p17 > /tmp/opencode/m13-p17-run2.out
$ cmp /tmp/opencode/m13-p17.out /tmp/opencode/m13-p17-run2.out
stdout byte-identical across two runs (exit 0)
$ sha256sum /tmp/opencode/m13-p17.out /tmp/opencode/m13-p20.out /tmp/opencode/m13-p17-run2.out
9c89e98c10c13a4d80c9d5d1567edbcaa16ae503187681c2efea963430fc6699  /tmp/opencode/m13-p17.out
9c89e98c10c13a4d80c9d5d1567edbcaa16ae503187681c2efea963430fc6699  /tmp/opencode/m13-p20.out
9c89e98c10c13a4d80c9d5d1567edbcaa16ae503187681c2efea963430fc6699  /tmp/opencode/m13-p17-run2.out
```

### Summary line and pinned hash

```text
SUMMARY plans=1152 events=288 tiles_compared=2304 bytes_compared=147456 differing_bytes=0 oracle_tiles_checked=4608 cache_tiles=3515 cpu_recomputed_tiles=1093 hit_changed_events=15 pass0_hint_checks=96 monotone_checks=192 failures=0
SUMMARY_FNV1A64=9fda93418b02d1ae
test_roret_fixture_parity: PASS
```

Counts: 288 target events (12 layers x 3 passes x 4 tokens x 2 cpu_share
settings), 1152 successful `plan()` calls including same-mode replays of
earlier passes, 2304 tile streams (each 8 tiles), 147456 tile bytes compared
OFF vs ON, 4608 oracle tile checks, 0 differing bytes. 96 pass-0 empty-hint
checks and 192 union monotonicity checks held. The hash is FNV-1a 64-bit over
the entire printed table plus the `SUMMARY ...` line, excluding the hash line
and final PASS line. The test asserts the hash equals the pinned value
`9fda93418b02d1ae`; a negative control compiled from a copy with the pinned
value zeroed exited 1 with `FAIL ...: hash == kPinnedHash`, confirming the gate
actually fires.

## 4. Honest scope and standing gates

This is **HOST/TILE-level parity only**, measured on synthetic fixture top-k
ids with the real `moe_cache_lru::plan()` and the real `llama_roret_union_cache`.
It does not exercise `llama_moe_cache::copy()` in the engine, does not run a
CUDA build or a GGUF, and does not compare `ffn_moe_topk`, `ffn_moe_weights`,
`ffn_moe_out`, device bytes, or final logits. No device residency or HBM
traffic is measured. The deterministic F32 tiles are stand-ins, not model
weights.

* Issue #4 (engine GPU bit-exact parity): **NO_GO at M13**, superseded at the
  engine final-logits level by M18; intermediate tensors still NOT measured.
* Issue #5 (T4 per-pass MoE traffic): **NO_GO**, unchanged.
* No speed claim is made. Tile byte counts are not time or `tok/s`.

Any published speed number still requires a real CycleAudit result with
`claim <= ci_low`, `repeats >= 20` and `loops = 3`. Nothing here changes that
mandatory gate.
