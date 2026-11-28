# M37 - offline GGUF loop-tensor shape contract gate

## Scope

This gate freezes the exact per-tensor shape and dtype contract for the
loop and residual-query tensors in the upstream Inkling-2 GGUF graph. It
resolves the last unchecked `[?]` item in section 2 of
`loopkit/docs/INKLING_INTEGRATION.md`: exact per-tensor shapes beyond the HF
shapes. The loopkit document is not edited here; its marker stays pending a
coordinated two-repo doc update.

No weights are loaded, no engine is run, and no timing or speed claim is made.
The checker is stdlib-only and makes no network calls.

## Commands

Run the gate from the repo root:

    python3 tools/audit/check_gguf_loop_shapes.py

Run the tests:

    PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit/test_gguf_loop_shapes.py -q
    PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest tools/audit -q

An isolated copy can be checked with `--root <dir>`. The checker writes a
`NO_GO` sentinel to `M37_GGUF_LOOP_SHAPES_SUMMARY.json` before it starts and
atomically replaces it with the final verdict. Exit code 0 means PASS; any
failed check or exception leaves NO_GO and exit code 1. A seeded stale PASS is
always overwritten.

## What the gate proves

22 checks must all be true for PASS:

- the frozen facts JSON still matches every checker constant (provenance,
  dimensions, compact shape contract, source hashes, needles, observed names
  and counts);
- all four live source files still match their frozen SHA-256 values and
  contain the required tensor-name and loop-config needles;
- the graph dimensions are internally consistent with primary-source dims:
  hidden 2304, gate rank 64, down 4608 = 2 x 2304, message rows 4, residual
  tail rows 3 at global layers 5, 11, 17, 23, loop core layers 6..17,
  n_phys 24 and n_slot 48 = 24 + 2 x (18 - 6);
- the compact contract expands to exactly 204 distinct tensor names with no
  missing or unexpected shape, dtype or count;
- M33 `QUANT_LAYOUT.json` still carries the same first-16-MiB GGUF hash;
- the frozen shape/dtype histogram measured from the header (44 x [2304],
  4 x [2304,3], 12 x [2304,4], 48 x [4608,64], 96 x [64,2304]) equals the
  histogram derived from the compact contract, so the shapes are anchored to
  the measurement and not only to the derived dimensions;
- the checker source imports no network module and makes no fetch call.

A failure to write the startup sentinel forces NO_GO even if every other check
passes. The test suite pins the checker source SHA-256, so editing the gate
itself is a detectable change.

## Frozen shape table

GGUF dims are reversed relative to HF dims. All 204 tensors are F32.

| Family / tensor name template | Global layers | Shape | Count |
| --- | --- | --- | --- |
| `blk.{bid}.loop_msg_query.weight` | 6..17 | [2304, 4] | 12 |
| `blk.{bid}.loop_q_ctrl_down.weight` | 6..17 | [4608, 64] | 12 |
| `blk.{bid}.loop_k_ctrl_down.weight` | 6..17 | [4608, 64] | 12 |
| `blk.{bid}.loop_v_ctrl_down.weight` | 6..17 | [4608, 64] | 12 |
| `blk.{bid}.loop_ffn_ctrl_down.weight` | 6..17 | [4608, 64] | 12 |
| `blk.{bid}.loop_q_ctrl_gain.weight` | 6..17 | [64, 2304] | 12 |
| `blk.{bid}.loop_k_ctrl_gain.weight` | 6..17 | [64, 2304] | 12 |
| `blk.{bid}.loop_v_ctrl_gain.weight` | 6..17 | [64, 2304] | 12 |
| `blk.{bid}.loop_ffn_ctrl_gain.weight` | 6..17 | [64, 2304] | 12 |
| `blk.{bid}.loop_q_ctrl_forget.weight` | 6..17 | [64, 2304] | 12 |
| `blk.{bid}.loop_k_ctrl_forget.weight` | 6..17 | [64, 2304] | 12 |
| `blk.{bid}.loop_v_ctrl_forget.weight` | 6..17 | [64, 2304] | 12 |
| `blk.{bid}.loop_ffn_ctrl_forget.weight` | 6..17 | [64, 2304] | 12 |
| `blk.{bid}.attn_res_query.weight` | 1..23 except 5, 11, 17, 23 | [2304] | 19 |
| `blk.{bid}.attn_res_query.weight` | 5, 11, 17, 23 | [2304, 3] | 4 |
| `blk.{bid}.ffn_res_query.weight` | 0..23 | [2304] | 24 |
| `output_res_query.weight` | global | [2304] | 1 |

Down controllers total 48, gain controllers 48 and forget controllers 48,
so the controller families total 144. Total matching tensors:
12 + 144 + 23 + 24 + 1 = 204. No other tensor name containing `loop`,
`ctrl`, `attn_res`, `ffn_res` or `output_res` exists in the measured header.

## Provenance

The measurement was taken once, out of band, from
`https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-IQ1_S.gguf`,
byte range `bytes=0-16777215` (16 MiB), SHA-256
`87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76`.
The cached file is `/tmp/opencode/loopkit-m37/gguf_first16.gguf`; the parsed
tensor list is `/tmp/opencode/loopkit-m37/tensors.json` with SHA-256
`083d0c1a3ac4aa3e14a9ac818e90f495ab32890ed2d361443cf6b6bd44062a23`.
The header is GGUF version 3 with 744 tensors and 63 key/value pairs. These
cache paths and hashes are frozen as facts; the checker does not read the cache
and never fetches anything.

Frozen live source hashes:

- `conversion/inkling2.py`: `aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7`
- `gguf-py/gguf/constants.py`: `f31289cf677a5125cc37cd58d87228199d5f0a124f583deb95a93fcfd79e086b`
- `src/models/inkling2-loop.cpp`: `a3fb8b2d6883a1977a283e2769f738273d45321fd612fbe4949fbcec0df0dfab`
- `src/models/inkling2.cpp`: `c5b543ad965857efd7f9818d42d1f7a418954758b75ec68c82485ffbd341225f`

## What the gate does not prove

- No engine run, no inference and no kernel behavior are tested.
- No weights are loaded or read; only a frozen header measurement and source
  text are used.
- No timing, throughput, latency or speed claim is made or implied.
- The loopkit `[?]` marker is not changed by this gate. It stays pending a
  coordinated two-repo doc update.
