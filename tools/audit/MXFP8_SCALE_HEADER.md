# MXFP8 expert scale-layout header evidence (M34, 2026-11-28)

This report resolves GitHub issue #9. The converter is correct and was not changed.
The earlier claim of "16 elements per scale for MXFP8" was a mislabel of NVFP4
shapes. The gate `check_mxfp8_scale_layout.py` is offline and makes no network
calls; the header bytes were fetched once out of band.

## Evidence

Source: `https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/model-00001-of-00001.safetensors`

| Item | Measured value |
| --- | --- |
| Byte range | 0-148415 (148416 bytes fetched) |
| SHA256 of fetched bytes | `a154b6ce2ead7032e9c0edcdcf020bf819a35f5078a36cce90938abe4d2c5247` |
| Safetensors header length | 148408 bytes (8-byte LE length prefix + JSON) |
| Tensor count | 1245 |
| Converter | `conversion/inkling2.py`, sha256 `aace332cd7a8b13fe5ba69abeb59d06fef4f8166aad4232a7f2ef7ae14bdf3d7` |

Measured expert tensor shapes (all four MXFP8 layers are identical):

| Layer | Tensor | Dtype | Shape | Elements/scale | Stored block |
| --- | --- | --- | --- | --- | --- |
| 20 | w13_weight | F8_E4M3 | [128, 2304, 2304] | - | - |
| 20 | w13_weight_scale | U8 | [128, 2304, 72] | 2304/72 = 32 | (1, 32) |
| 20 | w2_weight | F8_E4M3 | [128, 2304, 1152] | - | - |
| 20 | w2_weight_scale | U8 | [128, 2304, 36] | 1152/36 = 32 | (1, 32) |
| 21 | w13_weight | F8_E4M3 | [128, 2304, 2304] | - | - |
| 21 | w13_weight_scale | U8 | [128, 2304, 72] | 32 | (1, 32) |
| 21 | w2_weight | F8_E4M3 | [128, 2304, 1152] | - | - |
| 21 | w2_weight_scale | U8 | [128, 2304, 36] | 32 | (1, 32) |
| 22 | w13_weight | F8_E4M3 | [128, 2304, 2304] | - | - |
| 22 | w13_weight_scale | U8 | [128, 2304, 72] | 32 | (1, 32) |
| 22 | w2_weight | F8_E4M3 | [128, 2304, 1152] | - | - |
| 22 | w2_weight_scale | U8 | [128, 2304, 36] | 32 | (1, 32) |
| 23 | w13_weight | F8_E4M3 | [128, 2304, 2304] | - | - |
| 23 | w13_weight_scale | U8 | [128, 2304, 72] | 32 | (1, 32) |
| 23 | w2_weight | F8_E4M3 | [128, 2304, 1152] | - | - |
| 23 | w2_weight_scale | U8 | [128, 2304, 36] | 32 | (1, 32) |

NVFP4 contrast, layer 19 (the shapes that were mislabeled as MXFP8):

| Tensor | Dtype | Shape | Elements/scale | Stored block |
| --- | --- | --- | --- | --- |
| w13_weight | U8 | [128, 2304, 1152] | - | - |
| w13_weight_scale | F8_E4M3 | [128, 2304, 144] | 2304/144 = 16 | (1, 16) |
| w2_weight | U8 | [128, 2304, 576] | - | - |
| w2_weight_scale | F8_E4M3 | [128, 2304, 72] | 1152/72 = 16 | (1, 16) |
| w13_weight_amax | F32 | [128] | - | - |
| w13_weight_outer_scale_denominator | F32 | [1] | - | - |

## Three-way resolution

| Number | Where it came from | Verdict |
| --- | --- | --- |
| 16 | NVFP4 layer 19 scale shapes 144 and 72 columns. M33 copied these into its MXFP8 facts. | WRONG for MXFP8. It is the NVFP4 stored layout only. |
| 32 | The actual MXFP8 header (scale columns 72 and 36) and the live converter: `if block != (1, 16 if nvfp4 else 32):` plus `values.reshape(values.shape[0], -1, 32)[:] *= block_scale[:, :, None]`. | CORRECT. The live stored layout is (1, 32) and the converter dequantizes 32 elements per scale. Flat-32 is not a bug. |
| 1024 | Published `hf_quant_config.json` `weight_block_shape [32,32]`, i.e. one 32x32 tile per scale. | INFO only. It is config metadata that disagrees with the stored per-scale layout; the converter logs a warning and uses the stored layout. Not a correctness blocker for converter parity. |

M33 froze MXFP8 `w13_weight_scale [128, 2304, 144]` and
`w2_weight_scale [128, 2304, 72]` (16 per scale) and set
`checkpoint_scale_parity: UNVERIFIED` with an OPEN finding. That finding is
SUPERSEDED by this measured header evidence. The M33 files are untouched: other
gates freeze their hashes, so M34 records corrected facts in
`MXFP8_SCALE_HEADER.json` instead of editing them. No converter change was
needed or made.

## Scope and limits

No weights were loaded and no engine was run. This is a header-metadata and
source-needle check only. There is NO SPEED CLAIM. Any future speed number still
needs CycleAudit evidence with `claim <= ci_low`, `repeats >= 20` and `loops = 3`.
