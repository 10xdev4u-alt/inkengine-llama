# M33 - NVFP4/MXFP8 dequant scale layout and Unsloth quant (frozen facts)

## Command

From the inkengine-llama root:

```
python3 tools/audit/check_quant_layout.py
```

Stdlib only, offline, fail closed. Exit 0 only on verdict PASS. Every run rewrites
`tools/audit/M33_QUANT_LAYOUT_SUMMARY.json`, so a stale PASS is never left behind on a
failure. Mirrors the M29-M32 gates.

The checker must find a sibling loopkit at `REPO_ROOT.parent / "loopkit"`. Set
`LOOPKIT_ROOT` to override that path. A missing or wrong sibling forces NO_GO. The
checker never opens a network connection: the GGUF header facts below come from one
bounded range probe that was already recorded, and only its numbers are frozen.

## What is frozen

- Converter semantics in `conversion/inkling2.py`, gated by sha256 AND 11 live needle
  groups: scheme guard `nvfp4_nearest`/`mxfp8_nearest`, NVFP4 `amax_denominator`
  requirement, MXFP8 must not carry amax or outer scale denominator, weight/scale
  dtypes, packed shape divisibility (`cols % 16` for NVFP4, `% 32` for MXFP8 and
  `cols // 2` for packed NVFP4 bytes), stored block `(1, 16 if nvfp4 else 32)`,
  `_E4M3` NVFP4 scales followed by `amax/denominator`, flat `reshape(..., 32)` MXFP8
  scales from `ldexp(1, scale - 127)`, UE8M0 byte 255 is NaN, and the warning (not a
  rejection) when `weight_block_shape` in config differs from the stored block.
- Derived converter contract: `nvfp4_block=16`, `mxfp8_block=32`, both per row
  (`block[0] == 1`); NVFP4 scales are `torch.float8_e4m3fn` with per-expert F32 amax
  `[128]` and outer scale denominator `[1]`; MXFP8 scales are `torch.uint8` UE8M0
  only. The published format facts add activation blocks 16/32 and a MXFP8 *weight*
  tile `[32,32]`, one scale per 1024 elements, with derived scale shape
  `[128,72,72]` for MXFP8 w13 and the published verdict
  `CONTRADICTED / INCOMPLETE` for a flat 1-D MXFP8 weight group.
- HF header shapes from section 1 of `INKLING_INTEGRATION.md`: NVFP4 layers 0-19
  `w13_weight [128,2304,1152]`, `w2_weight [128,2304,576]`, scales `[128,2304,144]`
  and `[128,2304,72]`, amax `[128]`, denominator `[1]`; MXFP8 layers 20-23
  `w13_weight [128,2304,2304]`, `w2_weight [128,2304,1152]`, scales `[128,2304,144]`
  and `[128,2304,72]`. Section 1 explicitly warns that 144 = 2304/16 even for MXFP8.
  Section 4 item 6 asks for the comparison made here.
- Observed GGUF header from `unsloth/Inkling-2-Mini-GGUF`, file
  `Inkling-2-Mini-UD-IQ1_S.gguf`, one HTTP range probe over bytes 0-16777215;
  first-16MiB sha256 `87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76`.
  Frozen numbers: magic GGUF, version 3, `n_tensors=744`, `n_kv=63`,
  `general.architecture=inkling2_loop`, `general.name=Inkling-2-Mini`,
  `general.file_type=24`, `general.quantization_version=2`,
  `general.size_label=128x1.8B`, 156 loop tensors and 220 ffn tensors. The rank-64
  loop controllers on block 6 are F32: `loop_q_ctrl_down.weight [4608,64]`,
  `loop_q_ctrl_gain.weight [64,2304]`, `loop_q_ctrl_forget.weight [64,2304]` in
  GGUF dimension order (HF logical order in M30 is the transpose; the gate records
  the header literally). `blk.0.ffn_gate_up_exps.weight [2304,2304,128]` is ggml
  type 19 (IQ1_S), `blk.0.ffn_down_exps.weight [1152,2304,128]` is ggml type 42
  (Q2_0), `blk.0.ffn_gate_inp.weight [2304,128]` is F32, expert scale `[128]` is F32.
  Type histogram: F32 518, Q8_0 85, BF16 32, Q6_K 26, Q2_0(42) 23, Q5_K 16,
  Q4_K 14, IQ2_XXS 11, IQ1_S 6, IQ2_XS 5, IQ3_XXS 3, IQ4_XS 2, IQ2_S 2,
  NVFP4(40) 1. The histogram sums to 744 and that sum is gated.
- Frozen source hashes: all three inputs are sha256 gated. The two docs are gated by
  hash and by 6 live needle groups. Any mismatch, malformed JSON, missing loopkit,
  or tampered frozen fact forces NO_GO and overwrites the summary.

## Frozen source sha256

| file (relative to its root) | sha256 |
|---|---|
| conversion/inkling2.py | aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7 |
| docs/INKLING_INTEGRATION.md (loopkit) | 69c587aba7ebbe3b866b7ca95107cfb891b28b3b30d3c54eeba698f878655432 |
| docs/INKLING_FORMAT_FACTS.md (loopkit) | 46f134e263d1f9889a22e556428f7bdc6144fa1c9068072c31794ea19c4071df |

## Derived arithmetic (HF shapes divided at runtime; other values frozen)

`derive_elements_per_scale(cols, scale_cols)` divides logical w13 cols 2304 by
scale cols 144 and logical w2 cols 1152 by scale cols 72. All four cases give 16:
NVFP4 w13 16, NVFP4 w2 16, MXFP8 w13 16, MXFP8 w2 16. NVFP4 agrees with the
converter's flat 16. MXFP8 does not: 16 from the HF scale shape, 32 from the
converter's flat dequant grouping, and 1024 from the published 2-D `[32,32]`
weight tile. The gate checks that these numbers stay different and refuses to
mark parity as resolved.

## What PASS means

PASS means all 28 boolean checks are true: frozen facts and hashes match the
checker constants, live sources still match their frozen sha256 and needles, the
sibling loopkit is present, converter block semantics and scale dtypes match the
frozen contract, HF shapes divide the logical dims, the observed header facts and
type-count sum match, the derived elements-per-scale values are 16/16/16/16, and
the MXFP8 discrepancy is still recorded as an OPEN finding. PASS does **not** mean
the MXFP8 scale discrepancy is resolved. `checkpoint_scale_parity` stays
`UNVERIFIED`, `open_findings` stays nonempty, `speed_claim` is null, and the note
is "quant layout and Unsloth quant contract only, no timing".

## What stays unverified

- Real checkpoint scale-layout parity. No weights and no full header inspection
  were available. The Unsloth header was read once by bounded range request; its
  scale tensors were not inspected. The three MXFP8 elements-per-scale numbers do
  not agree, so no claim of agreement is made.
- The Unsloth quant is a **mixed re-quant**: `general.file_type=24` means
  MOSTLY_IQ1_S except 1D tensors, and the expert weights are IQ1_S/Q2_0. These are
  not native NVFP4/MXFP8 expert tensors. `general.architecture` does match the
  fork's `inkling2_loop`, and the rank-64 loop controllers remain F32 as frozen in
  M30. Accuracy delta versus the native checkpoint is not measured here.
- No graph execution, no dequantized weight parity, no accuracy claim, no timing,
  no speed claim. A future performance number still needs a CycleAudit measured
  claim; this gate does not substitute for one.
