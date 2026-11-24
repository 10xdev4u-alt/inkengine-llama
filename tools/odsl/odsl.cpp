// odsl.cpp -- standalone C++20 numeric/layout prototype for ODSL.
//
// Scope: this is a numeric + layout MODEL, not a kernel benchmark and not a
// CUDA/tensor-core implementation. The bit-exact LUTs are the read-only ones
// from the sibling anchor /tmp/build/inkengine/include/ink_lut.hpp.
//
// It answers three questions over the real NVFP4 test vectors:
//   1. Do two different dequant loop/layout forms produce byte-identical f32?
//      - scalar_reference: level * scale per element, scale read from the
//        interleaved vector record.
//      - block_major_simd: scales staged into a contiguous plane, ONE scale
//        load per 16-element block, 8-wide manual unrolling.
//   2. How much FP16 drift does the known d2027 outlier introduce, and how
//      much of that drift is removed by an FP32 side-lane?  FP16 is a CAST of
//      the bit-exact f32 operands; accumulation stays in f32 (as in mma.sync
//      f16 inputs, f32 accumulator). No bit-exactness is claimed for FP16.
//   3. Under a 32-byte-sector model, how many scale-load bytes per 1024
//      elements does a block-major scale plane save vs row-major?

#include "ink_lut.hpp"  // read-only: ink::kFp4Levels, ink::kE4M3

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace odsl {

constexpr int kBlockElems = 16;
constexpr int kOutlierDim = 2027;
// 2027 = 126 * 16 + 11, so d2027 lands on local lane 11 of its NVFP4 block.
constexpr int kOutlierLocal = kOutlierDim % kBlockElems;
constexpr int kRowsPerWarp = 4;      // per ENGINE_SPEC.md section 3
constexpr int kSectorBytes = 32;     // L1/L2 fetch granularity on sm_75
constexpr int kElementsPerReport = 1024;
constexpr int kModelRows = 256;      // rows in the emulated FP16 accumulator fragment
constexpr float kWeightSpike = 16.0f;      // loopkit tests/test_shc.py convention
constexpr float kActivationSpike = 20.0f;  // loopkit tests/test_shc.py convention

// ---------------------------------------------------------------------------
// Bitwise comparison and a portable, deterministic FP32<->FP16 conversion
// (round-to-nearest-even). No dependency on compiler __fp16 / std::float16_t.
// ---------------------------------------------------------------------------

inline bool bitwise_equal(float a, float b) {
    uint32_t ua = 0, ub = 0;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    return ua == ub;
}

inline uint16_t f32_to_f16_bits(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp_field = (x >> 23) & 0xffu;
    const uint32_t mant = x & 0x7fffffu;
    int exp = static_cast<int>(exp_field) - 127 + 15;

    if (exp_field == 0xffu) {  // Inf / NaN
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1f) {         // overflow to Inf
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (exp <= 0) {            // zero or half-subnormal
        if (exp < -10) return static_cast<uint16_t>(sign);
        const uint32_t m = mant | 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);  // 14..24
        uint32_t h = m >> shift;
        const uint32_t half = 1u << (shift - 1);
        const uint32_t rem = m & ((1u << shift) - 1u);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return static_cast<uint16_t>(sign | h);
    }
    // Normal: round the 23-bit mantissa to 10 bits, ties to even.
    uint32_t h = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return static_cast<uint16_t>(sign | (h & 0x7fffu));
}

inline float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Half subnormal = mant * 2^-24; normalize to 1.frac * 2^e.
            int shift = 0;
            while ((mant & 0x400u) == 0) { mant <<= 1; ++shift; }
            const uint32_t frac = mant & 0x3ffu;
            const int e = -14 - shift;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (frac << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);  // Inf / NaN
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Cheap fixed-vector self-check so a broken converter cannot silently
// invalidate the drift numbers.
bool half_converter_selfcheck() {
    struct Pair { float f; uint16_t h; };
    const Pair cases[] = {
        {1.0f, 0x3c00u}, {-1.0f, 0xbc00u}, {0.5f, 0x3800u},
        {2.0f, 0x4000u}, {0.0f, 0x0000u}, {-0.0f, 0x8000u},
        {65504.0f, 0x7bffu}, {5.960464477539063e-8f, 0x0001u},
        {2.98023223876953125e-8f, 0x0000u},
    };
    for (const auto& c : cases) {
        if (f32_to_f16_bits(c.f) != c.h) return false;
        if (!bitwise_equal(f16_bits_to_f32(c.h), c.f)) {
            // 0x0000 round-trip of 2^-25 is zero, which is expected; skip only
            // the deliberate round-to-zero case.
            if (c.h != 0x0000u) return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Vector file: uint32 LE count, then `count` records of
//   16 codes (u8) | 1 E4M3 scale code (u8) | 16 expected f32 (LE).
// ---------------------------------------------------------------------------

struct Block {
    std::array<uint8_t, kBlockElems> codes{};
    uint8_t scale_code = 0;
};

bool read_vectors(const std::string& path, std::vector<Block>& blocks,
                  std::vector<std::array<float, kBlockElems>>& expected,
                  std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    if (data.size() < 4) { err = "file shorter than header"; return false; }
    const uint32_t n = static_cast<uint32_t>(data[0]) |
                       (static_cast<uint32_t>(data[1]) << 8) |
                       (static_cast<uint32_t>(data[2]) << 16) |
                       (static_cast<uint32_t>(data[3]) << 24);
    const size_t need = 4 + static_cast<size_t>(n) * (16 + 1 + 16 * 4);
    if (data.size() != need) {
        err = "bad size: got " + std::to_string(data.size()) +
              " want " + std::to_string(need);
        return false;
    }
    blocks.resize(n);
    expected.resize(n);
    size_t off = 4;
    for (uint32_t b = 0; b < n; ++b) {
        for (int i = 0; i < 16; ++i) blocks[b].codes[i] = data[off++];
        blocks[b].scale_code = data[off++];
        for (int i = 0; i < 16; ++i) {
            uint32_t bits = static_cast<uint32_t>(data[off + 4 * i]) |
                            (static_cast<uint32_t>(data[off + 4 * i + 1]) << 8) |
                            (static_cast<uint32_t>(data[off + 4 * i + 2]) << 16) |
                            (static_cast<uint32_t>(data[off + 4 * i + 3]) << 24);
            std::memcpy(&expected[b][i], &bits, 4);
        }
        off += 16 * 4;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Path 1 -- scalar reference. Level * E4M3 scale, one element at a time,
// scale read from the interleaved record. Matches ink::dequant_block exactly.
// ---------------------------------------------------------------------------

std::array<float, kBlockElems> scalar_reference(const Block& b) {
    const float scale = ink::kE4M3[b.scale_code];
    std::array<float, kBlockElems> out{};
    for (int i = 0; i < kBlockElems; ++i) {
        out[i] = ink::kFp4Levels[b.codes[i]] * scale;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Path 2 -- block-major SIMD-style. The E4M3 scale is staged once into a
// contiguous plane; this function issues a SINGLE scale load for the whole
// 16-element block and then processes two 8-wide unrolled groups.
// ---------------------------------------------------------------------------

std::array<float, kBlockElems> block_major_simd(
        const uint8_t* codes, const float* scale_plane_at_block) {
    const float scale = *scale_plane_at_block;  // one load per 16-element block
    std::array<float, kBlockElems> out{};
    for (int base = 0; base < kBlockElems; base += 8) {
        const float v0 = ink::kFp4Levels[codes[base + 0]] * scale;
        const float v1 = ink::kFp4Levels[codes[base + 1]] * scale;
        const float v2 = ink::kFp4Levels[codes[base + 2]] * scale;
        const float v3 = ink::kFp4Levels[codes[base + 3]] * scale;
        const float v4 = ink::kFp4Levels[codes[base + 4]] * scale;
        const float v5 = ink::kFp4Levels[codes[base + 5]] * scale;
        const float v6 = ink::kFp4Levels[codes[base + 6]] * scale;
        const float v7 = ink::kFp4Levels[codes[base + 7]] * scale;
        out[base + 0] = v0; out[base + 1] = v1;
        out[base + 2] = v2; out[base + 3] = v3;
        out[base + 4] = v4; out[base + 5] = v5;
        out[base + 6] = v6; out[base + 7] = v7;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Outlier side-lane model.
//
// Each of kModelRows rows is a 16-wide dot product. Weights are the bit-exact
// dequantized f32 values from the vectors. Activations are deterministic,
// seeded pseudo-random values. To emulate the known d2027 energy concentration
// we apply the loopkit 16x weight / 20x activation spike at local lane 11
// (global dim 2027). Three results per row:
//   ref      : full f32 products, f32 accumulation  (the oracle)
//   no_side  : every operand cast f32->f16->f32, f32 accumulation
//   with_side: 15 non-outlier lanes as above, but the d2027 lane keeps its
//              exact f32 product in a separate side lane added in the epilogue
// The f16 cast is the only approximation; accumulation is f32, mirroring
// mma.sync with f16 inputs and an f32 accumulator.
// ---------------------------------------------------------------------------

struct Drift {
    double ref_l2 = 0.0;
    double err_no_side_l2 = 0.0;
    double err_with_side_l2 = 0.0;
    double rel_no_side = 0.0;
    double rel_with_side = 0.0;
};

uint32_t next_u32(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<uint32_t>((z ^ (z >> 31)) >> 32);
}

Drift outlier_drift(const std::vector<std::array<float, kBlockElems>>& expected) {
    uint64_t rng = 0x243f6a8885a308d3ULL;
    double sum_ref2 = 0.0, sum_err0 = 0.0, sum_err1 = 0.0;
    for (int r = 0; r < kModelRows; ++r) {
        std::array<float, kBlockElems> w = expected[static_cast<size_t>(r) % expected.size()];
        std::array<float, kBlockElems> a{};
        for (int i = 0; i < kBlockElems; ++i) {
            const uint32_t u = next_u32(rng);
            const float unit = (static_cast<float>(u) / 4294967296.0f) * 2.0f - 1.0f;
            a[i] = unit * 10.0f;
        }
        // Known physical property of this model, injected onto the d2027 lane.
        w[kOutlierLocal] *= kWeightSpike;
        a[kOutlierLocal] *= kActivationSpike;

        float ref = 0.0f, no_side = 0.0f, with_side = 0.0f;
        for (int i = 0; i < kBlockElems; ++i) {
            const float exact = w[i] * a[i];
            const float hw = f16_bits_to_f32(f32_to_f16_bits(w[i]));
            const float ha = f16_bits_to_f32(f32_to_f16_bits(a[i]));
            const float rounded = hw * ha;
            ref += exact;
            no_side += rounded;
            // The side lane keeps the outlier product exact, in f32.
            with_side += (i == kOutlierLocal) ? exact : rounded;
        }
        const double dref = static_cast<double>(ref);
        const double e0 = static_cast<double>(no_side) - dref;
        const double e1 = static_cast<double>(with_side) - dref;
        sum_ref2 += dref * dref;
        sum_err0 += e0 * e0;
        sum_err1 += e1 * e1;
    }
    Drift d;
    d.ref_l2 = std::sqrt(sum_ref2);
    d.err_no_side_l2 = std::sqrt(sum_err0);
    d.err_with_side_l2 = std::sqrt(sum_err1);
    d.rel_no_side = d.err_no_side_l2 / d.ref_l2;
    d.rel_with_side = d.err_with_side_l2 / d.ref_l2;
    return d;
}

// ---------------------------------------------------------------------------
// Scale-load estimate. A warp covers 4 rows x one 16-code block. Row-major
// scales are strided by N, so each of the 4 scale bytes fetches its own 32-B
// sector. Block-major scales are contiguous, so the same 4 bytes share one
// sector. Useful payload bytes are identical (64 B per 1024 elements); the
// saving is in fetched sectors, which is what the memory pipeline pays for.
// ---------------------------------------------------------------------------

struct ScaleLoads {
    int blocks = 0;
    int warps = 0;
    int row_major_bytes = 0;
    int block_major_bytes = 0;
    int saved_bytes = 0;
    double reduction_pct = 0.0;
    double factor = 1.0;
};

ScaleLoads scale_load_estimate() {
    ScaleLoads s;
    s.blocks = kElementsPerReport / kBlockElems;                       // 64
    s.warps = kElementsPerReport / (kRowsPerWarp * kBlockElems);       // 16
    s.row_major_bytes = s.blocks * kSectorBytes;   // 64 strided 1-B fetches
    s.block_major_bytes = s.warps * kSectorBytes;  // 16 contiguous 4-B fetches
    s.saved_bytes = s.row_major_bytes - s.block_major_bytes;
    s.reduction_pct = 100.0 * static_cast<double>(s.saved_bytes) /
                      static_cast<double>(s.row_major_bytes);
    s.factor = static_cast<double>(s.row_major_bytes) /
               static_cast<double>(s.block_major_bytes);
    return s;
}

}  // namespace odsl

// ---------------------------------------------------------------------------

#ifndef ODSL_VECTORS
#define ODSL_VECTORS "vectors.bin"
#endif

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "ASSERTION FAILED: %s\n", msg);            \
            return 3;                                                       \
        }                                                                   \
    } while (0)

int main() {
    using namespace odsl;

    if (!half_converter_selfcheck()) {
        std::fprintf(stderr, "FP16 converter self-check failed\n");
        return 2;
    }

    std::vector<Block> blocks;
    std::vector<std::array<float, kBlockElems>> expected;
    std::string err;
    if (!read_vectors(ODSL_VECTORS, blocks, expected, err)) {
        std::fprintf(stderr, "vector read failed: %s\n", err.c_str());
        return 2;
    }

    const size_t nblocks = blocks.size();
    const size_t nvalues = nblocks * kBlockElems;

    // Stage E4M3 scales into a contiguous plane, as the block-major layout
    // would in inkbin_export.py. Both dequant paths then consume the same LUT.
    std::vector<float> scale_plane(nblocks);
    for (size_t b = 0; b < nblocks; ++b) {
        scale_plane[b] = ink::kE4M3[blocks[b].scale_code];
    }

    size_t mismatch_scalar = 0;
    size_t mismatch_block = 0;
    size_t mismatch_between_paths = 0;
    for (size_t b = 0; b < nblocks; ++b) {
        const auto a = scalar_reference(blocks[b]);
        const auto c = block_major_simd(blocks[b].codes.data(), &scale_plane[b]);
        for (int i = 0; i < kBlockElems; ++i) {
            const bool ok_ref = bitwise_equal(a[i], expected[b][i]);
            const bool ok_blk = bitwise_equal(c[i], expected[b][i]);
            const bool ok_both = bitwise_equal(a[i], c[i]);
            if (!ok_ref) ++mismatch_scalar;
            if (!ok_blk) ++mismatch_block;
            if (!ok_both) ++mismatch_between_paths;
            // Literal assertion over all 16000 values (compiled out only under
            // NDEBUG; the explicit counters below still enforce the gate).
            assert(bitwise_equal(a[i], expected[b][i]));
            assert(bitwise_equal(c[i], expected[b][i]));
            assert(bitwise_equal(a[i], c[i]));
        }
    }

    // Required assertions.
    CHECK(mismatch_scalar == 0, "scalar_reference differs from expected f32");
    CHECK(mismatch_block == 0, "block_major_simd differs from expected f32");
    CHECK(mismatch_between_paths == 0, "the two dequant paths are not bit-exact");
    CHECK(nvalues == 16000, "unexpected total value count");

    const Drift d = outlier_drift(expected);
    CHECK(d.rel_with_side <= d.rel_no_side,
          "side-lane error is not <= no-side-lane error");

    const ScaleLoads sl = scale_load_estimate();

    std::printf("ODSL numeric/layout prototype (numeric model, NOT a kernel benchmark)\n");
    std::printf("vectors: %zu blocks / %zu f32 values (%s)\n", nblocks, nvalues, ODSL_VECTORS);
    std::printf("fp16 converter self-check: PASS\n\n");

    std::printf("[dequant -- bit-exact]\n");
    std::printf("  scalar_reference  vs expected f32 : mismatches = %zu / %zu\n",
                mismatch_scalar, nvalues);
    std::printf("  block_major_simd  vs expected f32 : mismatches = %zu / %zu\n",
                mismatch_block, nvalues);
    std::printf("  block_major_simd  vs scalar_ref  : mismatches = %zu / %zu\n",
                mismatch_between_paths, nvalues);
    std::printf("  bitwise_equal on all %zu values    : PASS\n\n", nvalues);

    std::printf("[fp16 drift model -- d2027, 16x weight / 20x activation, %d rows]\n",
                kModelRows);
    std::printf("  outlier lane = global dim %d = block-local lane %d\n",
                kOutlierDim, kOutlierLocal);
    std::printf("  full-f32 reference L2 norm                       = %.9g\n", d.ref_l2);
    std::printf("  relative L2 error, FP16 main, NO side lane       = %.9g\n", d.rel_no_side);
    std::printf("  relative L2 error, FP16 main, WITH FP32 side lane= %.9g\n", d.rel_with_side);
    std::printf("  side-lane improvement factor                     = %.3fx\n",
                d.rel_no_side / d.rel_with_side);
    std::printf("  note: FP16 path is a CAST of bit-exact f32 operands; not bit-exact.\n\n");

    std::printf("[scale-load estimate -- %d elements, %d NVFP4 blocks, %d-B sectors, %d rows/warp]\n",
                kElementsPerReport, sl.blocks, kSectorBytes, kRowsPerWarp);
    std::printf("  row-major   : %d blocks x strided 1 B = %d sectors x %d B = %d B fetched\n",
                sl.blocks, sl.blocks, kSectorBytes, sl.row_major_bytes);
    std::printf("  block-major : %d warps x contiguous 4 B = %d sectors x %d B = %d B fetched\n",
                sl.warps, sl.warps, kSectorBytes, sl.block_major_bytes);
    std::printf("  reduction   : %d B per 1024 elements (%.1f%%, %.2fx fewer sector fetches)\n",
                sl.saved_bytes, sl.reduction_pct, sl.factor);
    std::printf("  useful payload bytes are 64 B in both layouts; saving is in sectors.\n\n");

    std::printf("[assertions]\n");
    std::printf("  bitwise_equal on all 16000 values : PASS\n");
    std::printf("  side-lane error <= no-side-lane   : PASS\n");
    std::printf("  dequant-path bit-exact mismatches : %zu\n", mismatch_between_paths);
    return 0;
}
