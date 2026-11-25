// Tile-level bit-exactness gate for the RORET union-resident cache.
//
// This test includes the real src/llama-roret.h and the real LRU planner
// extracted verbatim into src/llama-moe-cache-lru.h. It does not include any
// ggml header: GGML_ASSERT/GGML_ABORT are defined below so the planner can be
// compiled standalone. CHECK is a real check, independent of assert/NDEBUG.
//
// The top-k ids are copied from the seed-0 Python fixture dump
// (tools/audit/make_moe_fixture.py + dump_f32.py), layer 6, tokens 0..3,
// passes 0..2. They are model output, not engine output.
//
// Build and run (from the repository root):
//   g++ -std=c++17 -Wall -Wextra -Werror -I. tools/roret/test_roret_tile_bitexact.cpp -o /tmp/opencode/test_roret_tile_bitexact
//   /tmp/opencode/test_roret_tile_bitexact

#include "src/llama-roret.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define GGML_ASSERT(cond)                                                             \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "GGML_ASSERT failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::abort();                                                             \
        }                                                                             \
    } while (false)

#define GGML_ABORT(msg)                                                               \
    do {                                                                              \
        std::fprintf(stderr, "GGML_ABORT %s:%d: %s\n", __FILE__, __LINE__, (msg));     \
        std::abort();                                                                 \
    } while (false)

#include "src/llama-moe-cache-lru.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++g_failures;                                                             \
        }                                                                             \
    } while (false)

constexpr int32_t kIl       = 6;
constexpr int32_t kExperts  = 128;
constexpr int32_t kTopK     = 8;
constexpr int32_t kSlots    = 64;
constexpr int32_t kTokens   = 4;
constexpr int32_t kPasses   = 3;
constexpr size_t  kTileFloats = 16;
constexpr size_t  kTileBytes  = kTileFloats * sizeof(float);

// topk_ids[pass][token][k] from tools/audit/dump_f32.py, seed 0, layer 6.
constexpr int32_t kFixtureIds[kPasses][kTokens][kTopK] = {
    {
        { 107, 117, 38, 15, 46, 91, 53, 50 },
        {  22, 107,  1, 53, 10, 63, 15, 78 },
        {  48,  97, 93, 108, 75, 121, 83, 28 },
        {  11,  28, 67, 37, 61, 73, 38, 74 },
    },
    {
        { 11, 50, 62, 27, 70, 19, 92, 81 },
        {  6, 68, 108, 106, 8, 81, 27, 70 },
        { 48, 93, 121, 65, 104, 97, 68, 45 },
        { 28, 15, 74, 12, 4, 11, 61, 88 },
    },
    {
        { 62, 19, 50, 81, 91, 11, 27, 117 },
        {  1, 78, 35, 107, 104, 120, 50, 123 },
        { 56, 48, 121, 45, 58, 6, 97, 57 },
        { 68, 28, 73, 61, 67, 74, 126, 12 },
    },
};

// Deterministic F32 tile per expert: the stand-in for an expert weight tile.
std::vector<unsigned char> make_tiles() {
    std::vector<unsigned char> tiles(static_cast<size_t>(kExperts) * kTileBytes);
    for (int32_t e = 0; e < kExperts; ++e) {
        auto * t = tiles.data() + static_cast<size_t>(e) * kTileBytes;
        for (size_t j = 0; j < kTileFloats; ++j) {
            const float v = static_cast<float>((e * 37 + static_cast<int32_t>(j) * 11) % 1000) * 0.125f + 0.5f;
            std::memcpy(t + j * sizeof(float), &v, sizeof(float));
        }
    }
    return tiles;
}

std::vector<int32_t> pass_ids(int32_t pass, int32_t token) {
    const int32_t * row = kFixtureIds[pass][token];
    return std::vector<int32_t>(row, row + kTopK);
}

// One LRU instance plus the host slot map and the modelled cache bank.
struct runner {
    explicit runner(int32_t n_zero) : n_zero(n_zero) {
        lru.init(kIl + 1, kExperts, kSlots);
        slots.assign(kExperts, kSlots);
        lru.slot_map[kIl] = slots.data();
        // Tiles after kSlots are the zero slots used by CPU-missed experts.
        bank.assign(static_cast<size_t>(kSlots + n_zero + 1) * kTileBytes, 0);
    }

    bool plan(const std::vector<int32_t> & ids, bool with_hints,
              const int32_t * extra, size_t n_extra,
              size_t & n_hit, std::vector<moe_cache_lru::fill> & fills,
              std::vector<moe_cache_lru::fill> & cpu, const std::vector<unsigned char> & tiles) {
        auto n_cpu = [](size_t n_miss) -> size_t { return (n_miss + 1) / 2; };
        const bool ok = with_hints
            ? lru.plan(kIl, ids.data(), ids.size(), 1, kTopK, n_zero, n_cpu, fills, cpu, n_hit, extra, n_extra)
            : lru.plan(kIl, ids.data(), ids.size(), 1, kTopK, n_zero, n_cpu, fills, cpu, n_hit);
        if (!ok) {
            return false;
        }
        for (const auto & f : fills) {
            CHECK(f.slot >= 0 && f.slot < kSlots);
            CHECK(f.expert >= 0 && f.expert < kExperts);
            std::memcpy(bank.data() + static_cast<size_t>(f.slot) * kTileBytes,
                        tiles.data() + static_cast<size_t>(f.expert) * kTileBytes, kTileBytes);
        }
        return true;
    }

    // The bytes the device reads for every real (token, k) selection.
    // Absolute oracle: a cached slot must hold exactly that expert's tile, and
    // a CPU-miss zero slot must hold zeros. Without this, a gather/fill bug
    // shared by all runners could make the relative comparison pass vacuously.
    std::vector<unsigned char> gather(const std::vector<int32_t> & ids,
                                      const std::vector<unsigned char> & tiles) const {
        std::vector<unsigned char> out;
        out.reserve(ids.size() * kTileBytes);
        for (int32_t e : ids) {
            const int32_t s = slots[e];
            CHECK(s >= 0 && s < kSlots + n_zero + 1);
            if (s < 0 || s >= kSlots + n_zero + 1) {
                continue;
            }
            const unsigned char * src = bank.data() + static_cast<size_t>(s) * kTileBytes;
            if (s < kSlots) {
                CHECK(lru.key_of[s] == lru.make_key(kIl, e));
                CHECK(std::memcmp(src, tiles.data() + static_cast<size_t>(e) * kTileBytes,
                                  kTileBytes) == 0);
            } else {
                for (size_t b = 0; b < kTileBytes; ++b) {
                    CHECK(src[b] == 0);
                }
            }
            out.insert(out.end(), src, src + kTileBytes);
        }
        return out;
    }

    int32_t n_zero;
    moe_cache_lru lru;
    std::vector<int32_t> slots;
    std::vector<unsigned char> bank;
};

size_t count_differing_bytes(const std::vector<unsigned char> & a,
                             const std::vector<unsigned char> & b) {
    CHECK(a.size() == b.size());
    if (a.size() != b.size()) {
        return a.size() + b.size();
    }
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++n;
        }
    }
    return n;
}

void check_cpu_setting(int32_t n_zero, const std::vector<unsigned char> & tiles,
                       size_t & tiles_compared, size_t & bytes_compared,
                       size_t & differing_bytes, size_t & hint_checks,
                       size_t & clear_checks, size_t & hint_configs,
                       size_t & residency_changed) {
    for (int32_t token = 0; token < kTokens; ++token) {
        // Production reads residents() BEFORE add() on each pass: pass p sees
        // the union of passes 0..p-1. Pass 0 therefore has no hints at all.
        std::vector<std::vector<int32_t>> hints_before(kPasses);
        llama_roret_union_cache timeline(kExperts, kTopK);
        for (int32_t q = 0; q < kPasses; ++q) {
            hints_before[q] = timeline.residents(kIl);
            const std::vector<int32_t> prior = pass_ids(q, token);
            (void) timeline.add({ kIl, q }, prior.data(), prior.size());
        }

        llama_roret_union_cache cache(kExperts, kTopK);
        cache.clear_layer(kIl);
        CHECK(cache.residents(kIl).empty());
        CHECK(cache.total_residents(kIl) == 0);
        ++clear_checks;

        for (int32_t pass = 0; pass < kPasses; ++pass) {
            const std::vector<int32_t> ids = pass_ids(pass, token);
            const std::vector<int32_t> & hints = hints_before[pass];
            (void) cache.add({ kIl, pass }, ids.data(), ids.size());

            runner off(n_zero);
            runner base(n_zero);
            runner on(n_zero);

            // Replay earlier passes in the SAME mode each runner will use at
            // pass p. ON accumulates hinted passes, so a hint that evicts a
            // real expert needed on a later pass is observable here.
            std::vector<moe_cache_lru::fill> filler;
            std::vector<moe_cache_lru::fill> cpu_ids;
            size_t replay_hit = 0;
            for (int32_t q = 0; q < pass; ++q) {
                const std::vector<int32_t> prior = pass_ids(q, token);
                const std::vector<int32_t> & prior_hints = hints_before[q];
                const bool replay_on = !prior_hints.empty();
                CHECK(off.plan(prior, false, nullptr, 0, replay_hit, filler, cpu_ids, tiles));
                CHECK(base.plan(prior, false, nullptr, 0, replay_hit, filler, cpu_ids, tiles));
                CHECK(on.plan(prior, replay_on, prior_hints.data(), prior_hints.size(),
                              replay_hit, filler, cpu_ids, tiles));
            }

            CHECK(hints.size() <= static_cast<size_t>(pass) * kTopK);
            for (int32_t q = 0; q < pass; ++q) {
                for (int32_t e : pass_ids(q, token)) {
                    bool present = false;
                    for (int32_t h : hints) {
                        if (h == e) {
                            present = true;
                            break;
                        }
                    }
                    CHECK(present);
                }
            }

            const bool use_hints = !hints.empty();
            hint_configs += use_hints ? 1 : 0;
            size_t hit_off = 0, hit_base = 0, hit_on = 0;
            std::vector<moe_cache_lru::fill> fills_off, fills_base, fills_on;
            std::vector<moe_cache_lru::fill> cpu_off, cpu_base, cpu_on;
            CHECK(off.plan(ids, false, nullptr, 0, hit_off, fills_off, cpu_off, tiles));
            CHECK(base.plan(ids, false, nullptr, 0, hit_base, fills_base, cpu_base, tiles));
            CHECK(on.plan(ids, use_hints, hints.data(), hints.size(), hit_on, fills_on, cpu_on, tiles));

            // A hint may never reduce the hits of a real selection.
            CHECK(hit_on >= hit_off);
            CHECK(hit_base == hit_off);
            if (fills_on.size() != fills_off.size()) {
                ++residency_changed;
            }
            ++hint_checks;

            for (int32_t e : ids) {
                const bool off_cached = off.slots[e] < kSlots;
                const bool on_cached  = on.slots[e]  < kSlots;
                CHECK(off_cached == on_cached);
            }

            const auto bytes_off  = off.gather(ids, tiles);
            const auto bytes_base = base.gather(ids, tiles);
            const auto bytes_on   = on.gather(ids, tiles);
            CHECK(bytes_off.size() == ids.size() * kTileBytes);
            tiles_compared += ids.size();
            bytes_compared += bytes_off.size();
            differing_bytes += count_differing_bytes(bytes_off, bytes_base);
            differing_bytes += count_differing_bytes(bytes_off, bytes_on);
        }
    }
}

} // namespace

int main() {
    const std::vector<unsigned char> tiles = make_tiles();

    size_t tiles_compared = 0;
    size_t bytes_compared = 0;
    size_t differing_bytes = 0;
    size_t hint_checks = 0;
    size_t clear_checks = 0;
    size_t hint_configs = 0;
    size_t residency_changed = 0;

    // cpu_share = 0 (no zero slots) and cpu_share > 0 (8 zero slots, 8 top-k).
    check_cpu_setting(0, tiles, tiles_compared, bytes_compared, differing_bytes,
                      hint_checks, clear_checks, hint_configs, residency_changed);
    check_cpu_setting(kTopK, tiles, tiles_compared, bytes_compared, differing_bytes,
                      hint_checks, clear_checks, hint_configs, residency_changed);

    std::printf("test_roret_tile_bitexact: %s\n", g_failures == 0 ? "PASS" : "FAIL");
    std::printf("configs=24 (3 passes x 2 cpu_share x 4 tokens)\n");
    std::printf("tiles_compared=%zu bytes_compared=%zu differing_bytes=%zu\n",
                tiles_compared, bytes_compared, differing_bytes);
    std::printf("hint_configs=%zu residency_changed_configs=%zu hint_checks=%zu clear_layer_checks=%zu failures=%d\n",
                hint_configs, residency_changed, hint_checks, clear_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
