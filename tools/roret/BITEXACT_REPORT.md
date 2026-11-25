# RORET bit-exact parity gate — status 2026-11-24

**Decision: NO_GO** for the next milestone (any bit-exact-parity or performance claim).
The wiring is opt-in, builds, and its header contract now has an executable gate, but no
engine-level off-vs-on comparison has been measured. No real or synthetic GGUF was run,
and no throughput number exists. See "Evidence command" below for what would change
the decision.

New files in this pass: `tools/roret/test_roret_bitexact.cpp` and this report. Nothing
under `src/` or any other doc was modified by this pass.

## What HAS been verified

1. **Header-level input immutability (executable, this pass).**
   `tools/roret/test_roret_bitexact.cpp` includes `src/llama-roret.h` and runs three
   cases: (a) three synthetic top-8 passes `{0..7}`, `{0..5,8,9}`,
   `{0,1,2,3,6,7,8,10}` with known overlap (union `{0..10}`, pre-add pass-2
   hit rate `7/8 = 0.875`); (b) negative-pass default-off; (c) an independent
   second layer whose ids straddle the 63/64 bitmap word boundary. Before every
   mutation/read the test takes a raw byte snapshot of the caller-owned id arrays
   and key structs (the stand-in for a `selected_experts` tensor payload) and
   compares it afterwards. It also checks that `residents()`/`hit_rate()` do not
   change the union and that `clear_layer()` on one layer leaves the other intact.
   The header's own bitmap **is** mutated by `add()` by design; it is bookkeeping,
   not tensor data, and is not claimed immutable.

   ```text
   g++ -std=c++17 -Wall -Wextra -Werror -I. tools/roret/test_roret_bitexact.cpp \
       -o /tmp/opencode/test_roret_bitexact17 && /tmp/opencode/test_roret_bitexact17
   test_roret_bitexact: PASS (3 cases, input bytes unchanged)      [exit 0]

   g++ -std=c++20 -Wall -Wextra -Werror -I. tools/roret/test_roret_bitexact.cpp \
       -o /tmp/opencode/test_roret_bitexact20 && /tmp/opencode/test_roret_bitexact20
   test_roret_bitexact: PASS (3 cases, input bytes unchanged)      [exit 0]
   ```
   GCC 16.2.1, clean under `-Wall -Wextra -Werror`; no `assert`, so `NDEBUG`
   cannot silently disable checks.

2. **Build with the RORET wiring (reported earlier, binary present).**
   `INTEGRATION.md` and `docs/INKLING_UPSTREAM_WIRING.md` record configure OK and
   `Built target llama-cli` at 100% via
   `cmake -S /tmp/build/inkengine-llama -B /tmp/ink-verify3 -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF`
   and `cmake --build /tmp/ink-verify3 --target llama-cli -j4`. The binary exists at
   `/tmp/ink-verify3/bin/llama-cli`. That build is CPU-only (`GGML_CUDA=OFF`), so the
   MoE cache is not constructed there and the runtime hook is inert; it proves
   compilation, not behaviour.

3. **Upstream graph untouched at tensor level — by inspection only.**
   The wiring adds trailing defaulted parameters and a host-side union cache; no
   change is described to `ffn_moe_logits`, `ffn_moe_probs`, group masking,
   `ggml_argsort_top_k`, `ffn_moe_weights`, `norm_w`, `w_scale`,
   `ggml_get_rows(probs, selected_experts)`, `build_experts`, graph node order, or
   `selected_experts_in`. `il` stays physical; the logical `ip`/`pass` live only in
   the RORET key. This is a source review, **not** a measured tensor comparison.
   The env gate is default-off (`LLAMA_RORET` must be exactly `1`, `on` or `true`),
   and the header test covers only the negative-pass part of that contract.

4. **Synthetic fixture hashes in `tools/audit` (reproduced this pass).**
   Regenerated with seed 0 via `tools/audit/make_moe_fixture.py` and
   `tools/audit/dump_f32.py`:

   | Artifact | Hash |
   |---|---|
   | fixture content (`fixture_sha256`) | `fff0e60fa4f55ba717d1e0bdc0e89695a0d35c85df680a15fd57ed1a0ac959ff` |
   | fixture file (`fixture_file_sha256`) | `5b752043e81ed2aefc19c13c9df8df28f8fce6b20a21e626025b0db1cfa7787b` |
   | F32 dump content (`dump_sha256`, local NVFP4 fallback) | `c85a8629c4b45a6f81bc505f5959dea469249f13e18029c47d13513fb686f5ec` |

   These match the committed generator output and prove the Python harness is
   deterministic. They are **not** engine outputs and do not measure the fork's
   router or MoE tensors.

## What has NOT been verified

- **No F32 tensor or logits parity off vs on.** No `memcmp`/hash comparison of
  `ffn_moe_topk`, `ffn_moe_weights`, `ffn_moe_out` or final logits for passes 0/1/2
  with `LLAMA_RORET` unset vs enabled. The graph is expected to be identical; that
  expectation is unmeasured.
- **No GGUF run at all**, synthetic or real. The synthetic fixture is a Python
  stand-in, not a model the fork executes. `cert.json` `root_digest`/`all_exact`
  remain PENDING in `tools/audit/AUDIT_PROTOCOL.md` (fixture not yet certified).
- **No traffic, upload-count or `tok/s` measurement.** `tools/roret/RESULTS.md`
  and `KAGGLE_RESULTS.md` report a load-count model on a synthetic router, not
  wall-clock, and are not evidence for the wired engine.
- **Runtime RORET paths unexercised.** The `LLAMA_RORET` env gate, copy-callback
  readback, `roret_of` slot binding, protected-hint `plan()` path, both scheduler
  splits (`cpu_share = 0` and `> 0`), cross-layer eviction and pass-0 `clear_layer`
  have not run against a model. The CPU-only build cannot exercise them.

## Evidence command (prerequisites: GPU-enabled build + synthetic GGUF)

The current build is CPU-only, where the MoE cache is not constructed, so parity
must be measured on a GPU build. Build the raw-F32 logits dumper (real flags:
`--moe-cache-mib`, `--save-logits`, `--logits-output-dir`):

```bash
cmake -S /tmp/build/inkengine-llama -B /tmp/ink-verify-cuda \
  -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build /tmp/ink-verify-cuda --target llama-debug -j4

MODEL=/path/to/inkling2-loop-synthetic.gguf
PROMPT="Hello my name is"
env -u LLAMA_RORET -u LLAMA_ARG_MOE_CACHE_MIB \
  /tmp/ink-verify-cuda/bin/llama-debug -m "$MODEL" -p "$PROMPT" -s 0 -n 16 \
  -ngl 99 --moe-cache-mib 256 --save-logits \
  --logits-output-dir /tmp/opencode/roret-off > /tmp/opencode/roret-off.log
LLAMA_RORET=1 env -u LLAMA_ARG_MOE_CACHE_MIB \
  /tmp/ink-verify-cuda/bin/llama-debug -m "$MODEL" -p "$PROMPT" -s 0 -n 16 \
  -ngl 99 --moe-cache-mib 256 --save-logits \
  --logits-output-dir /tmp/opencode/roret-on > /tmp/opencode/roret-on.log
sha256sum /tmp/opencode/roret-off/*.bin /tmp/opencode/roret-on/*.bin
for f in /tmp/opencode/roret-off/*.bin; do
  cmp "$f" "/tmp/opencode/roret-on/$(basename "$f")" || exit 1
done
```

`llama-debug --save-logits` writes raw F32 logits (`.bin`) and token IDs, so
identical hashes are a genuine bit-exact check of the model output, not a text
comparison. It does **not** dump the intermediate MoE tensors: the normal debug
callback prints values at `%12.4f`, which is insufficient for bit-exactness. A
raw-byte dump hook for `ffn_moe_topk`/`ffn_moe_weights`/`ffn_moe_out` still needs
to be added before claiming tensor-level parity. Off/on must also be checked with
identical prompt, seed, `-n`, `-ngl`, `--moe-cache-mib` and build; a mismatch in
any of those invalidates the comparison.

For any later throughput claim, the gate from `tools/audit/AUDIT_PROTOCOL.md`
additionally requires `claim <= ci_low`, at least 20 repeats and `loops = 3` on a
real T4 run; a load-count reduction is not a speedup.

## GO criteria for the next milestone

Change to GO only when all of the following exist:

1. Raw F32 logits and token IDs match bit-for-bit between the two runs above on
   a GPU build with the same synthetic GGUF, seed and prompt, for the full
   three-pass graph; per-pass `ffn_moe_*` intermediates must also match once a
   raw-byte dump hook exists.
2. The union hook is confirmed active in the `LLAMA_RORET=1` run and absent in the
   unset run (the info log is emitted only in the enabled case).
3. For a performance claim, a real CycleAudit result with `claim <= ci_low`,
   `repeats >= 20`, `loops = 3` — not a point estimate or a load-count model.

Until then: **NO_GO**. Absence of measured parity is not evidence of parity.

## 2026-11-25 increment - real-LRU tile-level parity measured on the host

This section adds executable measurements. It does not replace the 2026-11-24
evidence above, and it does not change the top-level decision: the engine-level
gate is still NO_GO.

### Files changed in this increment

- `src/llama-moe-cache-lru.h` (new): the `moe_cache_lru` struct copied verbatim
  from `src/llama-moe-cache.cpp` (anonymous namespace, `#pragma once`, standard
  includes only). A mechanical byte comparison against `git show
  HEAD:src/llama-moe-cache.cpp` confirmed the 11435-byte struct body (from
  `struct moe_cache_lru {` to its closing `};`) is identical, so this is a
  test seam, not a logic change.
- `src/llama-moe-cache.cpp`: includes the new header after `"ggml-cpp.h"`;
  the inlined struct was removed. All other code is unchanged.
- `tools/roret/test_roret_tile_bitexact.cpp` (new), registered with ctest in
  `tools/roret/CMakeLists.txt`.
- `tools/audit/make_moe_gguf.py` (new): copied from a prior run's uncommitted
  WIP in the sibling `work/` clone; its docstring credits that origin. The
  `work/` original was not modified. No loadability claim is made in the code.
- Nothing else was edited. `DECISIONS.tsv`, `BUILD_STATUS.md`,
  `docs/ROADMAP_48H.md` and the generator under `work/` were left untouched.

### Commands run and exit codes

All commands were run from the top-level clone at `HEAD = fd28da0`, branch
`main`. Paths below are literal, with `<repo>` standing for
`/home/princetheprogrammerbtw/.t3/scratch/2026-11-24-autonomous-continuation-for-the-loopkit-9dec44a4/inkengine-llama`.

| Command | Exit | Key output |
|---|---|---|
| `cmake -S <repo> -B /tmp/opencode/m4-build -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF` | 0 | `Configuring done`, `Generating done` |
| `cmake --build /tmp/opencode/m4-build --target llama-cli -j4` | 0 (second call) | `[100%] Built target llama-cli`; first call was terminated at 59% by the harness 900000 ms cap with no compiler error, then resumed and completed |
| `cmake -S tools/roret -B /tmp/opencode/m4-ctest-roret && cmake --build /tmp/opencode/m4-ctest-roret -j4 && ctest --test-dir /tmp/opencode/m4-ctest-roret --output-on-failure` | 0 | `2/2 Test #2: test_roret_tile_bitexact ... Passed`, `100% tests passed out of 2` |
| `cmake -S tools/pibs -B /tmp/opencode/m4-ctest-pibs && cmake --build /tmp/opencode/m4-ctest-pibs -j4 && ctest --test-dir /tmp/opencode/m4-ctest-pibs --output-on-failure` | 0 | `1/1 Test #1: pibs_model ... Passed` |
| `cmake -S tools/odsl -B /tmp/opencode/m4-ctest-odsl && cmake --build /tmp/opencode/m4-ctest-odsl -j4 && ctest --test-dir /tmp/opencode/m4-ctest-odsl --output-on-failure` | 0 | `1/1 Test #1: odsl_numeric_layout_model ... Passed` |
| `g++ -std=c++17 -Wall -Wextra -Werror -I. tools/roret/test_roret_tile_bitexact.cpp -o /tmp/opencode/test_roret_tile_bitexact && /tmp/opencode/test_roret_tile_bitexact` | 0 | see counts below |
| `g++ -std=c++20 -Wall -Wextra -Werror -Wpedantic -I. tools/roret/test_roret_tile_bitexact.cpp -o /tmp/opencode/test_roret_tile_bitexact20 && /tmp/opencode/test_roret_tile_bitexact20` | 0 | same counts, also clean under C++20 |
| `g++ -std=c++17 -Wall -Wextra -Werror -I. tools/roret/test_roret_bitexact.cpp ...` and the same with `-std=c++20` | 0 both | `test_roret_bitexact: PASS (3 cases, input bytes unchanged)` |
| `cd tools/audit && python3 make_moe_fixture.py --out /tmp/opencode/m4-fixture.npz --seed 0` | 0 | `fixture_sha256 fff0e60fa4f55ba717d1e0bdc0e89695a0d35c85df680a15fd57ed1a0ac959ff` |
| `cd tools/audit && python3 dump_f32.py --fixture /tmp/opencode/m4-fixture.npz --out /tmp/opencode/m4-dump.npz` | 0 | `dump_sha256 c85a8629c4b45a6f81bc505f5959dea469249f13e18029c47d13513fb686f5ec` (`nvfp4_source local-fallback`) |
| `cd tools/audit && python3 make_moe_gguf.py --out /tmp/opencode/m4-fixture.gguf --seed 0` | 0 | `gguf_sha256 7e8566245dc91c54f5a3f78b3bff002d91ce587f2a1f4f6d112edff44a1072bf`, 987008 bytes |
| same command with `--out /tmp/opencode/m4-fixture-run2.gguf --seed 0` | 0 | identical `gguf_sha256`; `sha256sum` of both files gives `7e8566245dc91c54f5a3f78b3bff002d91ce587f2a1f4f6d112edff44a1072bf` |
| gguf-py reader check against the manifest (in-tree `gguf-py` on `sys.path`) | 0 | `114/114` tensor names match; `114/114` shapes match the manifest after reversing GGUF's stored dimension order; 0 missing, 0 extra |

### Tile-level byte comparison counts

The test uses the real `llama_roret_union_cache` and the real `moe_cache_lru::plan()`.
The top-k ids are copied from the seed-0 Python dump, layer 6, 4 tokens x 3
passes x 8 ids. Expert tiles are deterministic F32 arrays of 16 floats (64
bytes) indexed by expert id; no ggml tensor is involved. For every
(token, pass) the three runners start from the same replayed LRU state: BASE
calls `plan()` with no trailing arguments, OFF passes explicit `nullptr, 0`,
ON passes `extra = residents(il)`. The gathered byte stream is the concatenation
of the 8 tile byte reads the device would perform for the real selections,
resolved through each plan's slot map (cached slot, or the zero slot assigned
to a CPU miss).

```text
test_roret_tile_bitexact: PASS
configs=24 (3 passes x 2 cpu_share x 4 tokens)
tiles_compared=192 bytes_compared=12288 differing_bytes=0
hint_configs=16 residency_changed_configs=7 hint_checks=24 clear_layer_checks=8 failures=0
```

- Configurations: 3 passes x 2 `cpu_share` settings (`n_zero = 0` and
  `n_zero = 8`) x 4 tokens = 24.
- Tile reads compared per stream: 24 x 8 = 192 tiles = 12288 F32 bytes.
- Two comparisons against OFF were made (BASE vs OFF and OFF vs ON), so 384
  tile-read comparisons / 24576 byte comparisons in total.
- Differing bytes: 0 in both comparisons and in every individual configuration.
- **Effective hint coverage:** 16/24 configurations actually passed non-empty
  hints (pass 1 and 2 only; pass 0 correctly has none, matching production's
  `residents()`-before-`add()` order). In 7/24 configurations the hint changed
  the planner's residency decision (`fills_on != fills_off`); the other 9 were
  already cached. The earlier single-step version overstated this coverage;
  it has been replaced.
- **Absolute oracle:** every gathered cached-slot byte stream is now `memcmp`ed
  against the source tile of the expert that slot maps to, and every CPU-miss
  zero slot is checked to be all zeros. A gather/fill bug shared by all runners
  can no longer make the relative comparison pass vacuously.
- **Cumulative ON trajectory:** earlier passes are replayed with hints in the ON
  runner (production applies hints on every pass > 0), so a hint that evicts a
  real expert needed on a later pass is observable, not hidden by an OFF-only
  replay.
- `n_hit` for real ids never decreased under hints (24/24 configurations);
  BASE hit counts equalled OFF hit counts (24/24). The set of real ids assigned
  a cached slot was identical between OFF and ON (24/24).
- `clear_layer()` reset checks: 8/8 (once per token per `cpu_share` setting).
- `cpu_share > 0` here is a synthetic stress setting: CPU misses are modelled as
  zero device slots, not a recomputed CPU tile. It exercises the scheduler's
  zero-slot split, not production CPU recomputation; see the NO_GO scope below.

### Verdict split

1. **HOST / TILE-LEVEL parity: PASS, scoped.** With the real LRU planner and
   real RORET header, at both `cpu_share = 0` and `cpu_share > 0`, and for the
   default-off path, the gathered F32 tile bytes for the real (token, k)
   selections are byte-identical off vs on across all three passes: 0 differing
   bytes out of 24576 compared. Each slot's bytes are also checked against the
   absolute source tile (or zero for a CPU-miss slot), the hint timeline follows
   production's `residents()`-before-`add()` order, and the ON runner replays
   earlier hinted passes. This is a host-side residency/byte-gather result on
   synthetic tiles, not a model of CPU recomputation. It does not prove engine
   tensor or logit equality.
2. **ENGINE-LEVEL parity: NO_GO, unchanged.** No GPU build was made, no model
   was run, and no raw `ffn_moe_topk` / `ffn_moe_weights` / `ffn_moe_out` /
   final-logits comparison off vs on was measured. The build used here is
   `GGML_CUDA=OFF`, where the MoE cache is not constructed. Absence of that
   measurement is not evidence of parity.
3. **Synthetic GGUF loadability: NOT LOADED.** Exact command:
   `/tmp/opencode/m4-build/bin/llama-cli -m /tmp/opencode/m4-fixture.gguf -p "Hello" -n 4 -ngl 0 > /tmp/opencode/m4-gguf-load.log 2>&1`
   Exit code 134 (SIGABRT). First error line:
   `Loading model... |/home/princetheprogrammerbtw/.t3/scratch/2026-11-24-autonomous-continuation-for-the-loopkit-9dec44a4/inkengine-llama/ggml/src/ggml-backend.cpp:534: GGML_ASSERT(buf != NULL && "tensor buffer not set") failed`
   (the `|` is the CLI spinner).
   The GGUF itself is valid to gguf-py (114/114 tensors), but the engine did
   not load it. This is recorded as evidence, not as a defect in the copied
   generator. A concurrent change on `main` added the loadable
   `tools/audit/make_synthetic_gguf.py` and explains the failure precisely in
   `tools/audit/PARITY_REPORT.md`: an all-SWA pattern leaves `tau` unused, so
   `inp_pos` is pruned and never allocated. The WIP generator here is marked
   SUPERSEDED and must not be used for future parity runs.

### Standing constraints

No speed claim is made in this increment. These are load counts, tile byte
counts and model results. Any future speed number requires a CycleAudit result
with `claim <= ci_low`, `repeats >= 20` and `loops = 3`. The Python harness in
`tools/audit` is a deterministic **model**, not engine output; matching model
hashes prove only that the model is reproducible. If a later host or engine
check shows a byte flip, that configuration is NO_GO and the scoped fallback is
to pin only device-resident layers (disable hints for any layer whose tiles can
be computed on the host), not to relax the byte comparison.
