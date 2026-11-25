# Synthetic MoE fixture + F32 dump harness

This directory is a **test vehicle for bit-exactness and traffic counts**, not a model.

It manufactures a deterministic, fixed-seed MoE fixture whose *structural* constants match
our engine's Inkling-2-Mini model - 24 layers, 128 experts/layer, top-8 routing, 3 loop
passes, rank-64 gain/forget loop controllers, NVFP4 codes with E4M3 block scales for expert
weights, and a 4-vector cross-loop message (`q, k, v, ffn`, per upstream `LOOP_CTRL_COUNT`).
Per-expert weight matrices and the token count are deliberately **small synthetic values** so
the artifacts stay compact and the tests run in seconds. The F32 dump exercises the same
quantization/dequantization and routing/mixing math the real engine uses, so a wiring change
can be checked for bit-exactness (`root_digest`-style content hashes) and for expert-load
traffic without needing any checkpoint.

**No real Inkling weights are involved.** Nothing is downloaded, read from a model file, or
embedded here: every value comes from `numpy`'s seeded PCG64 PRNG plus a local NVFP4 quantizer
that is byte-for-byte identical to `loopkit/src/loopkit/nvfp4.py`.

## Files

- `make_moe_fixture.py` - generates the fixture (numpy-only) and its `manifest.json`.
- `dump_f32.py` - reads the fixture and dumps F32 intermediates; prints the dump SHA-256.
- `test_fixture.py` - pytest checks for seed/dump determinism, top-k validity and constants.
- `AUDIT_PROTOCOL.md`, `verify_gate.py`, `test_verify_gate.py` - the CycleAudit gate contract.

## Regenerate

From anywhere (all paths are explicit):

```bash
cd tools/audit
python make_moe_fixture.py --out /tmp/fixture.npz --seed 0
# writes /tmp/fixture.npz and /tmp/fixture.npz.manifest.json
# prints: fixture_sha256 <hex>          (content hash)
# prints: fixture_file_sha256 <hex>     (raw .npz file hash)

python dump_f32.py --fixture /tmp/fixture.npz --out /tmp/dump.npz
# prints: nvfp4_source <loopkit|local-fallback>
# prints: dump_sha256 <hex>              (content hash)
# prints: dump_file_sha256 <hex>         (raw .npz file hash)
```

The manifest records both fixture hashes plus every model and quantization constant. The
primary `*_sha256` is a **content hash** over sorted arrays (name, dtype, shape and raw bytes),
not over the `.npz` zip container, so zip metadata and zlib versions cannot change it. The
`*_file_sha256` is provided for `sha256sum` convenience and is stable with the same numpy/zlib
build.

`dump_f32.py` imports `loopkit`'s NVFP4 emulator when it is on `PYTHONPATH`, e.g.:

```bash
PYTHONPATH=/tmp/build/loopkit/src python dump_f32.py --fixture /tmp/fixture.npz --out /tmp/dump.npz
```

Otherwise it uses the identical local fallback in `make_moe_fixture.py`. Both paths yield the
same dump hash.

### Dump contents

For every `(layer, pass, token)` the dump stores F32 `router_logits`, `topk_ids` (int32),
`selected_w13` / `selected_w2` (dequantized expert weights), `expert_outputs`, `message_mix`
(the 4 mixed cross-loop vectors per token; zero on pass 0 and non-core layers), and
`layer_output`.
`route_traffic` is an int32 count of how many tokens selected each expert per `(layer, pass)`
- the hook for traffic-count checks.

The synthetic forward pass is a simplified FFN-only path: the rank-r controller is applied to
the FFN input, the router is a linear projection of the RMS-normalized state, and experts are
combined with a softmax over the selected logits scaled by `route_scale`. This is enough to
exercise bit-exactness and routing counts; it is not claimed to reproduce model outputs.

## Tests

```bash
python -m pytest -q tools/audit/test_fixture.py
```

Five tests cover: same-seed hash equality, different-seed hash inequality, top-k ids in range
and unique per row, dump hash reproducibility, and the 3-pass / 128-expert constants. The suite
completes in a few seconds.

## Honesty notes

- A matching hash proves the harness is deterministic and the stored/dequantized operands are
  bit-exact; it does **not** validate or measure the real model.
- Local CPU values are plumbing checks. A publishable speed claim still needs the T4 CycleAudit
  gate in `AUDIT_PROTOCOL.md`, with `claim <= ci_low` and no rounded-up numbers.
- The "activation outlier at dimension 2027" idea is an unverified local hypothesis, not a
  property asserted by this fixture.

## B1 bit-exact certificate freeze

`make_cert.py` freezes the bit-exact baseline over the deterministic fixture and
`test_make_cert.py` checks it through the producer's public CLI. The committed
`FREEZE.json` is the full record; `cert.json` is the CLI's single-bundled-state
certificate (`n_passes=1`) and is **not** the 3-pass gate digest.

```bash
python3 tools/audit/make_cert.py --loopkit-src <loopkit>/src \
  --out /tmp/opencode/b1-freeze --seed 0
python3 -m pytest -q tools/audit/test_make_cert.py \
  tools/audit/test_fixture.py tools/audit/test_verify_gate.py
```

Frozen 2026-11-25 (seed 0): 3-pass `root_digest`
`22b36c0a9335e5d7ad9a31316c45d5de5cdcc896771aec204c165da6b60dfb27`,
`all_exact=true`, three passes of 625920 elements with 0 mismatches; REF and
CAND fixtures both have canonical sha256
`fff0e60fa4f55ba717d1e0bdc0e89695a0d35c85df680a15fd57ed1a0ac959ff`
and F32 dump sha256
`c85a8629c4b45a6f81bc505f5959dea469249f13e18029c47d13513fb686f5ec`;
`nvfp4-roundtrip` lossless with ratio 4.7656; `fdm-check` max absolute error 0.0.
A one-bit flip on pass 1 fails closed: `all_exact=false`, exit 1, one mismatch.
Two producer runs produce the same digest and hashes. The producer writes the
large per-pass state arrays and fixture files only under `--out`; only the two
small JSON files are committed. No speedup is claimed or measured.

Because `loopkit.cli certify` loads a whole directory as one state, the
authoritative 3-pass certificate is computed with loopkit's own
`certify_loop`; the test independently recomputes
`sha256(concat(pass digests))` from the frozen pass digests. See
[AUDIT_PROTOCOL.md](AUDIT_PROTOCOL.md) section 7 for the scope caveat.

## M9 CycleAudit artifact

`CYCLE_AUDIT.json` is the local CPU producer output. `run_cycle_audit.py` records
50 real wall-clock runs per RORET config on the synthetic GGUF (5 warmups
discarded), verifies RORET log state and text parity, then drives the real
`loopkit.cli audit` gate with loops 3 and 2000 bootstrap resamples. Raw arrays
are saved in `baseline_seconds.json` and `candidate_seconds.json`.

Regenerate with the build done separately by the operator (the script never builds):

```bash
cmake -S <fork> -B /tmp/opencode/m9-build -DGGML_CUDA=OFF \
  -DLLAMA_RORET_CPU_TEST=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build /tmp/opencode/m9-build --target llama-cli -j4
python3 tools/audit/run_cycle_audit.py \
  --loopkit-src <workspace>/loopkit/src --fork-root <fork> \
  --build-dir /tmp/opencode/m9-build --out /tmp/opencode/m9-audit
cp /tmp/opencode/m9-audit/CYCLE_AUDIT.json \
   /tmp/opencode/m9-audit/baseline_seconds.json \
   /tmp/opencode/m9-audit/candidate_seconds.json tools/audit/
```

Verdict: NO_CLAIM with `claim: 0`. Local CPU timings are plumbing only under
`AUDIT_PROTOCOL.md` section 2. Any `ci_low` in the artifact is an ineligible
point estimate from a probe, never a publishable speed number. Blocker: one
Kaggle 2xT4 CycleAudit run (repeats >= 20, loops 3, logs preferred over the
output API) with `claim <= ci_low`.
