# ODSL — dequant / layout / outlier numeric prototype

> **Source-status note (2026-11-24).** The d2027 outlier is an **unverified local hypothesis;
> not in primary sources** (absent from TML's release post/model card; present only in our
> config/tests/docs). The spike below is injected synthetically; it is not read from model weights.
> Upstream already provides the loop graph and converter; this fork's contribution is
> union-resident MoE routing on top of upstream.

Standalone C++20 prototype for **ODSL** (dequant-to-FP16 WMMA with a block-major
scale plane and an FP32 side-lane for the hypothetical d2027 outlier **[unverified local
hypothesis; not in primary sources]**) from Inkling 2 Mini.
It is a **numeric and layout model, not a kernel benchmark** and not a CUDA or
tensor-core implementation. No llama.cpp build is needed.

## What it measures

1. **Bit-exactness of two dequant forms.** Reads the existing 1,000-block
   bit-exact vectors from `/tmp/build/inkengine/tests/vectors.bin` (16 E2M1
   codes + 1 E4M3 scale code + 16 expected float32 values per block, `uint32`
   LE count header). Two paths must produce byte-identical float32 output:
   - `scalar_reference`: `kFp4Levels[code] * kE4M3[scale]`, scale read from the
     interleaved record (matches `ink::dequant_block` exactly).
   - `block_major_simd`: scales staged into a contiguous plane, **one** scale
     load per 16-element block, two 8-wide manually unrolled groups. Plain C++,
     no intrinsics.

2. **FP16 drift + side lane.** For each of 256 rows, a 16-wide dot product is
   computed three ways: full float32 (oracle), FP16-cast operands accumulated
   in float32 (no side lane), and FP16-cast for 15 lanes plus an exact float32
   product for the d2027 lane (with side lane). The hypothesized d2027 energy spike
   **[unverified local hypothesis; not in primary sources]** is injected using the loopkit
   convention (16× weight, 20× activation); global dim 2027 maps to block-local lane 11
   (`2027 = 126*16 + 11`). The float16 cast
   is round-to-nearest-even and implemented in portable bit manipulation.
   **No bit-exactness is claimed for the FP16 path — the drift is measured.**

3. **Scale-load estimate.** Per 1024 elements (64 NVFP4 blocks), under a
   32-byte-sector model with 4 rows per warp: row-major scales force 64 strided
   single-byte fetches; block-major scales use 16 contiguous 4-byte fetches.
   Useful payload is 64 bytes in both cases — the saving is in fetched sectors.

## Build and run

```bash
cmake -S . -B build
cmake --build build -j
./build/odsl
# or: ctest --test-dir build --output-on-failure
```

The LUT header is read **read-only** from the sibling anchor at
`/tmp/build/inkengine/include/ink_lut.hpp`. Override with
`-DODSL_INK_INCLUDE=...`; override the vectors with `-DODSL_VECTORS=...`.

## Honesty notes

- FP4/FP8 tensor cores do not exist on T4 (sm_75); FP16 is the only tensor-core
  path. This prototype does not measure runtime, bandwidth, or tok/s.
- The scale-load figures are a sector-granularity **estimate**, not a measured
  memory-transaction count on hardware.
- The d2027 spike magnitudes are injected to emulate an unverified local hypothesis; not in
  primary sources (loopkit `tests/test_shc.py`); they are not read from the test vectors or model
  weights.
- FP32 accumulation mirrors `mma.sync` with f16 inputs and an f32 accumulator;
  reduction order here is simple sequential and is not claimed to match
  numpy's or a WMMA tile's exact reduction order.

Actual measured numbers are in [RESULTS.md](RESULTS.md).
