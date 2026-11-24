# RESULTS — ODSL numeric/layout prototype

Measured on 2026-11-24 (UTC). This is a **numeric and layout model, not a kernel
benchmark**. No GPU, no tensor cores, no wall-clock or bandwidth measurements.

## Environment

- GCC 16.2.1, CMake 4.4.4, C++20, `-DCMAKE_BUILD_TYPE=Release`
- Vectors: `/tmp/build/inkengine/tests/vectors.bin` (1,000 blocks / 16,000 f32 values)
- LUTs: read-only `/tmp/build/inkengine/include/ink_lut.hpp`
- Build/run: `cmake -S . -B build && cmake --build build -j && ./build/odsl`
- `ctest --test-dir build --output-on-failure`: 1/1 passed
- A Debug build (where `assert()` is active) produced identical numbers.

## 1. Dequant — bit-exact results

| Comparison | Mismatches |
|---|---|
| `scalar_reference` vs expected f32 | **0 / 16,000** |
| `block_major_simd` vs expected f32 | **0 / 16,000** |
| `block_major_simd` vs `scalar_reference` | **0 / 16,000** |

**Bit-exact mismatch count between the two dequant paths: 0.**
`bitwise_equal` on all 16,000 values: PASS. The block-major path stages scales
into a contiguous float plane, issues one scale load per 16-element block, and
uses 8-wide manual unrolling — with no change in output bits.

## 2. FP16 drift model — d2027 side lane

256 rows, 16-wide dot products, weights = bit-exact dequantized f32 from the
vectors. d2027 = global dim 2027 = block-local lane 11; injected 16× weight and
20× activation spike (loopkit `tests/test_shc.py` convention). FP32 accumulation
throughout; only the operands are cast through round-to-nearest-even FP16.

| Quantity | Value |
|---|---|
| Full-f32 reference L2 norm | 27046.3666647 |
| Relative L2 error, FP16 main, **no** side lane | **0.000233148161** |
| Relative L2 error, FP16 main, **with** FP32 side lane | **3.05057452e-06** |
| Improvement factor | 76.428× |

Assertion `side-lane error <= no-side-lane error`: **PASS**.
The FP16 path is **not** bit-exact; these numbers are measured drift, not a
correctness certificate. The side-lane contribution itself is an exact f32
product (same operation as in the reference), so d2027 is exact by construction.

## 3. Scale-load estimate — bytes per 1024 elements

1024 elements = 64 NVFP4 blocks; 4 rows/warp; 32-byte sector granularity.

| Layout | Scale fetches | Fetched bytes / 1024 elements |
|---|---|---|
| Row-major (strided 1 B per row) | 64 × 1 B, 64 sectors | **2048 B** |
| Block-major (contiguous 4 B per warp) | 16 × 4 B, 16 sectors | **512 B** |

**Reduction: 1536 B per 1024 elements = 75.0% = 4.00× fewer sector fetches.**
Useful payload is 64 B in both layouts; the saving is in fetched sectors, not
in the number of useful scale bytes. This is an estimate from the sector model
in `ENGINE_SPEC.md` §3, not a measured memory-transaction count on T4.

## 4. Independent cross-check

The FP16 drift model was reimplemented independently in Python/NumPy
(`numpy.float16`, float32 arithmetic, same splitmix64 activations). It reproduced
all three drift values: `ref_l2 = 27046.366664737336`, `rel_no =
0.00023314816067569288`, `rel_with = 3.0505745212217188e-06` — matching the C++
output to every printed digit. The portable bit-manipulation FP16 converter also
passed fixed-vector checks including signed zero, 2^-24, 2^-25 tie-to-even, and
65504.

## Honesty

- T4 (sm_75) has FP16 tensor cores but **no FP4/FP8 tensor cores**; FP4/FP8 here
  are storage formats. Nothing in this prototype measures speed.
- The d2027 spike magnitudes are injected to emulate a known model property,
  not read from the random test vectors.
- Sequential float32 accumulation here is not claimed to match NumPy's or a
  WMMA tile's exact reduction order.
- No end-to-end, kernel, or tok/s claims are made.
