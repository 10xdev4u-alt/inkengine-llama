# M8 - Bit-exact block-major scale layout integration design

Status: specification only. This pass adds exactly one file, this document.
No file under src/ was edited. No commit and no push. No speed, bandwidth,
throughput, or tok/s claim is made anywhere in this document.

Baseline: fork branch main, HEAD f8ed8e4 at the time of writing.
Roadmap anchor: docs/ROADMAP_48H.md, section "M8 - Specify bit-exact
block-major scale layout integration" (lines 109-118).

## Scope

This document specifies how the ODSL numeric/layout prototype could enter
the fork as a default-off integration candidate. It applies no integration.
It reproduces the existing evidence, names real seams in this tree, defines
a default-off gate independent of M3/M7, and gives a verification recipe.
M8 unblocks M9/M10: ODSL may enter CycleAudit only as a clearly classified
candidate (bit-exact dequant count, numeric drift model, layout model),
never as a measured speed result.

## Acceptance criteria (checklist)

- [x] 1. Reproduce 0/16000 mismatches from the vendored vectors:
  `scalar_reference` vs expected f32 = 0/16000, `block_major_simd` vs
  expected f32 = 0/16000, and the two paths vs each other = 0/16000.
  Reproduce the 76.4x FP16 drift reduction with the FP32 side lane:
  relative L2 error 0.000233148161 with no side lane ->
  3.05057452e-06 with side lane, factor 76.428x, ref L2 norm
  27046.3666647. All are counts or numeric-model values, not speeds.
- [x] 2. The FP32 side-lane path is explicitly labeled LOWER-DRIFT but
  NOT BIT-EXACT. It is a cast/numeric model, not a correctness gate.
- [x] 3. The 512 B (block-major) vs 2048 B (row-major) per 1024 elements
  is explicitly labeled a SECTOR-FETCH MODEL, not measured bandwidth.
  Useful payload is 64 B in both layouts; only fetched sectors differ.
- [x] 4. T4 (sm_75) is stated to have FP16 tensor cores and NO FP4/FP8
  tensor cores, so NVFP4 is storage only in this context.
- [x] 5. The design is default-off and independent of M3/M7 gating; it
  does not depend on UNION_WIRING.patch or PIBS_WIRING.patch state.
- [x] 6. Honesty note restated: the d2027 outlier is an UNVERIFIED LOCAL
  HYPOTHESIS, not in primary sources; the 16x weight / 20x activation
  spike is injected synthetically; no pretrained weights are held.

## Reproduced evidence

Every value below is a COUNT, a NUMERIC MODEL, or a LAYOUT MODEL
ESTIMATE. None is a measured speed or a measured memory-transaction
count on hardware.

### Dequant bit-exactness (COUNT)

| Comparison | Mismatches |
|---|---|
| `scalar_reference` vs expected f32 | 0 / 16000 |
| `block_major_simd` vs expected f32 | 0 / 16000 |
| `block_major_simd` vs `scalar_reference` | 0 / 16000 |

Bitwise mismatch count between the two dequant paths: 0 (COUNT).
Sources: tools/odsl/RESULTS.md:21-31; tools/odsl/odsl.cpp:187-221
(`scalar_reference`, `block_major_simd`), :371-396 (counters and
assertions), :404-415 (printed counts). The block-major path stages
scales into a contiguous plane, issues one scale load per 16-element
block, and uses two 8-wide unrolled groups, with no change in output
bits. `bitwise_equal` (odsl.cpp:51-56) does the u32 bit comparison.

### FP16 drift model (NUMERIC MODEL, not bit-exact)

| Quantity | Value |
|---|---|
| Full-f32 reference L2 norm | 27046.3666647 |
| Relative L2 error, FP16 main, no side lane | 0.000233148161 |
| Relative L2 error, FP16 main, with FP32 side lane | 3.05057452e-06 |
| Improvement factor | 76.428x |

This is a NUMERIC MODEL over 256 emulated rows of 16-wide dot products
(kModelRows = 256, a model parameter), with FP32 accumulation throughout
and only the operands cast through round-to-nearest-even FP16. Sources:
tools/odsl/RESULTS.md:34-53; tools/odsl/odsl.cpp:255-295
(`outlier_drift`), :398-400 (side-lane assertion), :417-426 (printed
values). The binary prints the L2 norm as 27046.3667 with `%.9g`
(odsl.cpp:421); the fuller 27046.3666647 is from RESULTS.md:44 and the
independent cross-check at RESULTS.md:73. The FP16 path is a CAST of
bit-exact f32 operands and is NOT
bit-exact; these numbers are measured drift in a model, not a
correctness certificate. The side-lane contribution is an exact f32
product, so the injected dimension 2027 lane is exact by construction
in this model.

### Scale-load estimate (LAYOUT MODEL / ESTIMATE, not measured bandwidth)

Model inputs: 1024 elements = 64 NVFP4 blocks, 4 rows per warp,
32-byte sector granularity (odsl.cpp constants kRowsPerWarp=4,
kSectorBytes=32, kElementsPerReport=1024; see also odsl.cpp:35-44).

| Layout | Scale fetches (model) | Fetched bytes / 1024 elements (model) |
|---|---|---|
| Row-major (strided 1 B per row) | 64 x 1 B, 64 sectors | 2048 B |
| Block-major (contiguous 4 B per warp) | 16 x 4 B, 16 sectors | 512 B |

Reduction in the model: 1536 B per 1024 elements = 75.0% = 4.00x fewer
sector fetches (LAYOUT MODEL ESTIMATE). Useful payload is 64 B in both
layouts; the saving is in fetched sectors, not in useful scale bytes.
This is the sector model from ENGINE_SPEC.md section 3 (in the sibling
inkengine repo), not measured bandwidth and not a memory-transaction count
on T4. Sources:
tools/odsl/RESULTS.md:55-67; tools/odsl/odsl.cpp:315-327
(`scale_load_estimate`), :428-436 (printed model values).

### Independent cross-check (NUMERIC MODEL)

A separate Python/NumPy reimplementation reproduced the drift values
(ref_l2 = 27046.366664737336, rel_no = 0.00023314816067569288,
rel_with = 3.0505745212217188e-06) to every printed digit, and the
portable FP16 converter passed signed-zero, 2^-24, 2^-25 tie-to-even,
and 65504 checks. Source: tools/odsl/RESULTS.md:69-77. These are
cross-checks of a model, not engine measurements.

## Integration seams (real symbols and files in this fork)

### Prototype side

- tools/odsl/odsl.cpp:187 `scalar_reference` - level x E4M3 scale per
  element, scale read from the interleaved record; matches the vendor
  LUT path. This is the default bit-exact form.
- tools/odsl/odsl.cpp:202 `block_major_simd` - one scale load per
  16-element block from a staged plane, two 8-wide unrolled groups.
  Integration seam for a future block-major dequant loader; must keep
  the 0/16000 bit-exact count before any gate can open.
- tools/odsl/odsl.cpp:366 `scale_plane` - local staging vector in
  `main`, filled from `ink::kE4M3[blocks[b].scale_code]`. This is the
  contiguous-plane layout a future exporter would emit; it is a local
  variable, not a standalone function, and lives only in this prototype.
- tools/odsl/odsl.cpp:315 `scale_load_estimate` - returns struct
  `ScaleLoads` (blocks, warps, row_major_bytes, block_major_bytes,
  saved_bytes, reduction_pct, factor). This is the layout-model seam;
  it is arithmetic on constants, not a hardware counter.
- tools/odsl/odsl.cpp:255 `outlier_drift` - returns struct `Drift`
  (ref_l2, err_no_side_l2, err_with_side_l2, rel_no_side,
  rel_with_side). This is the side-lane numeric-model seam.
- tools/odsl/odsl.cpp:51 `bitwise_equal`, :146 `read_vectors`,
  :58 `f32_to_f16_bits`, :89 `f16_bits_to_f32` - supporting helpers
  for the bit-exact and drift gates.
- tools/odsl/ink_lut.hpp - read-only LUT header included at
  odsl.cpp:20; provides `ink::kFp4Levels` and `ink::kE4M3`. Never
  edited by this prototype.
- tools/odsl/vectors.bin - vendored bit-exact vectors, registered at
  tools/odsl/CMakeLists.txt:13 and copied to the build dir at :20.
  Format: uint32 LE count, then records of 16 codes, 1 E4M3 scale
  code, and 16 expected f32 values (odsl.cpp:136-139, :146-180).

### GGML / llama.cpp NVFP4 side (verified by reading this tree)

- ggml/include/ggml.h:259 - `GGML_TYPE_NVFP4 = 40`, with
  `GGML_FTYPE_MOSTLY_NVFP4 = 26` at :316.
- ggml/src/ggml-common.h:221-227 - `QK_NVFP4` (64),
  `QK_NVFP4_SUB` (16), and `block_nvfp4` with
  `uint8_t d[QK_NVFP4/QK_NVFP4_SUB]` (4 UE4M3 scales) and
  `uint8_t qs[QK_NVFP4/2]` (32 packed E2M1 bytes).
- ggml/src/ggml-quants.c:381 `quantize_row_nvfp4_ref`, :592
  `dequantize_row_nvfp4`, :2318 `quantize_nvfp4`. These are the real
  quantize/dequantize paths a future layout work would face.
- ggml/src/ggml.c:769-775 - `GGML_TYPE_NVFP4` type traits:
  `.to_float = (ggml_to_float_t) dequantize_row_nvfp4` and
  `.from_float_ref = (ggml_from_float_t) quantize_row_nvfp4_ref`;
  dispatch to `quantize_nvfp4` at :8172.
- ggml/src/ggml-quants.c:402-404 and :603 - conversions
  `ggml_fp32_to_ue4m3` / `ggml_ue4m3_to_fp32` used by the NVFP4
  paths.

Grain note: ODSL's 16-element block with one scale matches the GGML
sub-block grain `QK_NVFP4_SUB = 16`, not the full `QK_NVFP4 = 64`
block. Caution: ODSL scales come from `ink::kE4M3`, while GGML NVFP4
uses UE4M3. Do not assume those encodings are identical. A future
patch must prove scale-bit equivalence or keep the prototype inside
its own LUT domain; this document claims neither.

No symbol above was invented. Any future integration must re-verify
these names against the tree at the commit it targets.

## Proposed default-off gate

The gate is ODSL-only and does not read M3 or M7 state.

- Runtime gate: environment variable `LLAMA_ODSL`, enabled only for
  values {1, on, true} (case-insensitive), matching the fork's
  existing gate convention. Unset or any other value is a no-op:
  dequant stays on `scalar_reference`, no `scale_plane` staging is
  emitted, and no side lane runs.
- Build gate: optional CMake/cache option `ODSL_LAYOUT`, default OFF.
  When OFF, no layout-specific code is compiled into any engine target.
- Side-lane sub-gate: `LLAMA_ODSL_SIDE_LANE`, also default off. It is
  separate because the side lane is LOWER-DRIFT but NOT BIT-EXACT and
  can never satisfy a bit-exact gate.
- Independence: no check of `LLAMA_RORET` (M3) or `INKLING2_PIBS`
  (M7); no dependency on UNION_WIRING.patch or PIBS_WIRING.patch,
  present or absent. The design holds on the bare fork at f8ed8e4.
- Default behavior is unchanged: with all gates off, output bits are
  the same as the untouched baseline by inspection of the design;
  engine-level F32 parity remains a NO_GO gate until actually run.
- Future integration should land as a patch artifact, following the
  M3/M7 pattern, not as an un-gated src/ edit. This pass ships no
  patch and no source edit.

## Verification recipe

Prototype (expected: odsl ctest 1/1):

```sh
cd tools/odsl
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/odsl
```

Expected printed counts: 0/16000 on all three dequant comparisons,
`side-lane error <= no-side-lane : PASS`, dequant mismatch count 0,
factor 76.428x, and the 512 B / 2048 B layout-model rows. A Debug
build (where `assert()` is active) should print identical numbers.

Gate-off check for a future patch: rerun the same recipe with
`LLAMA_ODSL` unset and `ODSL_LAYOUT=OFF`; output must be unchanged
bit-for-bit from the baseline.

Optional engine build (cheap enough on this fork, no weights needed):

```sh
cmake -S . -B /tmp/odsl-llama-build -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build /tmp/odsl-llama-build --target llama-cli -j
```

Expected: `llama-cli` build reaches 100%, as recorded in
BUILD_STATUS.md:30-46. This checks that the design does not break the
fork build; it does not measure speed.

Verification limits: the prototype reads only vendored vectors and
the read-only LUT; it uses no GPU and no tensor cores. No CycleAudit
claim is made here; any future speed statement still needs
`claim <= ci_low` with `repeats >= 20` and `loops = 3`.

## Honest constraints

1. The d2027 outlier is an UNVERIFIED LOCAL HYPOTHESIS and is NOT in
   primary sources (absent from TML's release post and model card).
   The 16x weight / 20x activation spike is injected synthetically
   (loopkit `tests/test_shc.py` convention) and is not read from the
   test vectors or from model weights. No pretrained weights are held;
   every check here uses vendored vectors or a synthetic fixture.
2. The FP32 side-lane path is lower-drift but NOT bit-exact. The FP16
   path is a cast of f32 operands; its sequential f32 reduction is not
   claimed to match a WMMA tile's or NumPy's reduction order.
3. The 512 B vs 2048 B figures are a SECTOR-FETCH MODEL, not measured
   bandwidth. Useful payload is 64 B in both layouts.
4. T4 (sm_75) has FP16 tensor cores and NO FP4/FP8 tensor cores;
   NVFP4 here is storage only. Nothing in this document measures speed,
   bandwidth, wall clock, or tok/s.
5. No speedup claim is publishable without a CycleAudit lower bound
   (`claim <= ci_low`, `repeats >= 20`, `loops = 3`). Counts, drift
   ratios, and sector models are explicitly not speedups.
6. No issue/PR spam: at most 2 real issues per hour, check
   `gh issue list` first, and no upstream PRs from an autonomous
   agent. This pass created no issue and no PR.
