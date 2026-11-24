# Synthetic MoE fixture + F32 dump harness

This directory is a **test vehicle for bit-exactness and traffic counts**, not a model.

It manufactures a deterministic, fixed-seed MoE fixture whose *structural* constants match
our engine's Inkling-2-Mini model — 24 layers, 128 experts/layer, top-8 routing, 3 loop
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

- `make_moe_fixture.py` — generates the fixture (numpy-only) and its `manifest.json`.
- `dump_f32.py` — reads the fixture and dumps F32 intermediates; prints the dump SHA-256.
- `test_fixture.py` — pytest checks for seed/dump determinism, top-k validity and constants.
- `AUDIT_PROTOCOL.md`, `verify_gate.py`, `test_verify_gate.py` — the CycleAudit gate contract.

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
— the hook for traffic-count checks.

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
