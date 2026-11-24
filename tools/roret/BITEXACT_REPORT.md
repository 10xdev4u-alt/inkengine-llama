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
