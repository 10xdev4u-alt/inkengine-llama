// Fixture-wide bit-exact tile parity gate for the RORET union-resident cache.
//
// This test includes the real src/llama-roret.h and the real LRU planner
// extracted verbatim into src/llama-moe-cache-lru.h. It defines GGML_ASSERT and
// GGML_ABORT below so the planner compiles standalone. CHECK is a real check,
// independent of assert/NDEBUG.
//
// The top-k ids come from the checked-in union_census_ids.inc, generated from
// the seed-0 Python fixture dump: 12 loop layers (ip 6..17) x 3 passes x 4
// tokens x top-8. No Python or loopkit code runs here.
//
// For each (ip, token) a real llama_roret_union_cache timeline supplies the
// union of earlier passes: pass 0 has empty hints, pass p>0 has the union of
// passes 0..p-1, sampled before this pass's ids are registered. Each target
// (ip, pass, token) is replayed twice from the same empty LRU state: OFF plans
// with no hints, ON plans with that earlier-pass union as extra hints. Earlier
// passes are replayed in the same mode on each runner, so a hint that harms a
// later pass is visible. The gathered F32 tile bytes are compared OFF vs ON
// and against an absolute per-expert tile oracle.
//
// Build and run (from the repository root):
//   g++ -std=c++17 -Wall -Wextra -Werror -I. tools/roret/test_roret_fixture_parity.cpp -o /tmp/opencode/m13p17
//   /tmp/opencode/m13p17

#include "src/llama-roret.h"

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
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
#include "union_census_ids.inc"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++g_failures;                                                             \
        }                                                                             \
    } while (false)

using Key = llama_roret_union_cache::key;

constexpr int32_t kSlots      = 64;
constexpr size_t  kTileFloats = 16;
constexpr size_t  kTileBytes  = kTileFloats * sizeof(float);

// Pinned FNV-1a 64-bit hash of the printed table/summary text. A changed result
// fails this check even if every individual CHECK above stays silent.
constexpr uint64_t kPinnedHash = 0x9fda93418b02d1aeULL;

// Deterministic F32 tile per expert: the stand-in for an expert weight tile.
std::vector<unsigned char> make_tiles() {
    std::vector<unsigned char> tiles(static_cast<size_t>(kCensusNExperts) * kTileBytes);
    for (int32_t e = 0; e < kCensusNExperts; ++e) {
        auto * t = tiles.data() + static_cast<size_t>(e) * kTileBytes;
        for (size_t j = 0; j < kTileFloats; ++j) {
            const float v = static_cast<float>((e * 37 + static_cast<int32_t>(j) * 11) % 1000) * 0.125f + 0.5f;
            std::memcpy(t + j * sizeof(float), &v, sizeof(float));
        }
    }
    return tiles;
}

std::vector<int32_t> pass_ids(int32_t layer, int32_t pass, int32_t token) {
    const int32_t * row = kCensusIds[layer][pass][token];
    return std::vector<int32_t>(row, row + kCensusTopK);
}

// One LRU instance plus the host slot map and the modelled cache bank.
struct runner {
    runner(int32_t il, int32_t n_zero) : il(il), n_zero(n_zero) {
        lru.init(il + 1, kCensusNExperts, kSlots);
        slots.assign(kCensusNExperts, kSlots);
        lru.slot_map[il] = slots.data();
        bank.assign(static_cast<size_t>(kSlots + n_zero + 1) * kTileBytes, 0);
    }

    bool plan(const std::vector<int32_t> & ids, bool with_hints,
              const int32_t * extra, size_t n_extra,
              size_t & n_hit, std::vector<moe_cache_lru::fill> & fills,
              std::vector<moe_cache_lru::fill> & cpu, const std::vector<unsigned char> & tiles) {
        // Stress model inherited from test_roret_tile_bitexact.cpp: half the misses
        // go to the CPU, without the engine's cache-full fixed-point accumulator.
        auto n_cpu = [](size_t n_miss) -> size_t { return (n_miss + 1) / 2; };
        const bool ok = with_hints
            ? lru.plan(il, ids.data(), ids.size(), 1, kCensusTopK, n_zero, n_cpu, fills, cpu, n_hit, extra, n_extra)
            : lru.plan(il, ids.data(), ids.size(), 1, kCensusTopK, n_zero, n_cpu, fills, cpu, n_hit);
        if (!ok) {
            return false;
        }
        for (const auto & f : fills) {
            CHECK(f.slot >= 0 && f.slot < kSlots);
            CHECK(f.expert >= 0 && f.expert < kCensusNExperts);
            std::memcpy(bank.data() + static_cast<size_t>(f.slot) * kTileBytes,
                        tiles.data() + static_cast<size_t>(f.expert) * kTileBytes, kTileBytes);
        }
        return true;
    }

    // The tile bytes consumed for every real (token, k) selection.
    // Absolute oracle: a cached slot must hold exactly that expert's tile, and
    // a CPU-miss zero slot must hold zeros on the device. A CPU-missed expert is
    // recomputed on the host from that same tile, so the comparison stream uses
    // the recomputed tile rather than the zero placeholder. Without this, a
    // hint that moves an expert to the CPU path would look like a tile mismatch.
    std::vector<unsigned char> gather(const std::vector<int32_t> & ids,
                                      const std::vector<unsigned char> & tiles,
                                      size_t & cache_tiles, size_t & cpu_tiles) const {
        std::vector<unsigned char> out;
        out.reserve(ids.size() * kTileBytes);
        for (int32_t e : ids) {
            const int32_t s = slots[e];
            CHECK(s >= 0 && s < kSlots + n_zero + 1);
            if (s < 0 || s >= kSlots + n_zero + 1) {
                continue;
            }
            const unsigned char * tile = tiles.data() + static_cast<size_t>(e) * kTileBytes;
            const unsigned char * src = nullptr;
            if (s < kSlots) {
                src = bank.data() + static_cast<size_t>(s) * kTileBytes;
                CHECK(lru.key_of[s] == lru.make_key(il, e));
                CHECK(std::memcmp(src, tile, kTileBytes) == 0);
                ++cache_tiles;
            } else {
                const unsigned char * zero = bank.data() + static_cast<size_t>(s) * kTileBytes;
                for (size_t b = 0; b < kTileBytes; ++b) {
                    CHECK(zero[b] == 0);
                }
                src = tile; // host recomputation of the CPU-missed expert
                ++cpu_tiles;
            }
            out.insert(out.end(), src, src + kTileBytes);
        }
        return out;
    }

    int32_t il;
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

struct totals_t {
    uint64_t plans             = 0;
    uint64_t events            = 0;
    uint64_t tiles             = 0;
    uint64_t bytes             = 0;
    uint64_t differing         = 0;
    uint64_t oracle_tiles      = 0;
    uint64_t cache_tiles       = 0;
    uint64_t cpu_tiles         = 0;
    uint64_t hits_changed      = 0;
    uint64_t pass0_hint_checks = 0;
    uint64_t monotone_checks   = 0;
};

struct row_t {
    uint64_t events    = 0;
    uint64_t hints_min = std::numeric_limits<uint64_t>::max();
    uint64_t hints_max = 0;
    uint64_t union_min = std::numeric_limits<uint64_t>::max();
    uint64_t union_max = 0;
    uint64_t tiles     = 0;
    uint64_t bytes     = 0;
    uint64_t differing = 0;
};

bool run_plan(runner & r, const std::vector<int32_t> & ids, bool with_hints,
              const int32_t * extra, size_t n_extra, size_t & n_hit,
              std::vector<moe_cache_lru::fill> & fills,
              std::vector<moe_cache_lru::fill> & cpu,
              const std::vector<unsigned char> & tiles, totals_t & totals) {
    const bool ok = r.plan(ids, with_hints, extra, n_extra, n_hit, fills, cpu, tiles);
    CHECK(ok);
    totals.plans += ok ? 1 : 0;
    return ok;
}

void check_cpu_setting(int32_t n_zero, const std::vector<unsigned char> & tiles,
                       totals_t & totals,
                       row_t rows[kCensusNCoreLayers][kCensusNPass]) {
    for (int32_t layer = 0; layer < kCensusNCoreLayers; ++layer) {
        const int32_t ip = kCensusIpFirst + layer;
        for (int32_t token = 0; token < kCensusNTokens; ++token) {
            // copy() order: clear on pass 0, sample the earlier-pass union,
            // then register this pass's ids.
            llama_roret_union_cache timeline(kCensusNExperts, kCensusTopK);
            std::vector<std::vector<int32_t>> hints_before(kCensusNPass);
            size_t union_after[kCensusNPass] = {};

            for (int32_t p = 0; p < kCensusNPass; ++p) {
                if (p == 0) {
                    timeline.clear_layer(ip);
                }
                hints_before[p] = timeline.residents(ip);
                const size_t before = hints_before[p].size();
                CHECK(timeline.total_residents(ip) == before);
                CHECK(before <= static_cast<size_t>(p) * kCensusTopK);
                if (p == 0) {
                    CHECK(before == 0);
                    ++totals.pass0_hint_checks;
                }
                for (int32_t q = 0; q < p; ++q) {
                    for (int32_t e : pass_ids(layer, q, token)) {
                        bool present = false;
                        for (int32_t h : hints_before[p]) {
                            if (h == e) {
                                present = true;
                                break;
                            }
                        }
                        CHECK(present);
                    }
                }

                const std::vector<int32_t> ids = pass_ids(layer, p, token);
                timeline.add(Key{ip, p}, ids.data(), ids.size());
                union_after[p] = timeline.residents(ip).size();
                CHECK(union_after[p] >= before);
                CHECK(timeline.total_residents(ip) == union_after[p]);
                if (p > 0) {
                    CHECK(union_after[p] >= union_after[p - 1]);
                    ++totals.monotone_checks;
                }
            }

            for (int32_t pass = 0; pass < kCensusNPass; ++pass) {
                const std::vector<int32_t> ids = pass_ids(layer, pass, token);
                const std::vector<int32_t> & hints = hints_before[pass];

                runner off(ip, n_zero);
                runner on(ip, n_zero);

                // Replay earlier passes in the SAME mode each runner uses at
                // the target pass, from an identical empty LRU state.
                std::vector<moe_cache_lru::fill> filler;
                std::vector<moe_cache_lru::fill> cpu_ids;
                size_t replay_hit = 0;
                for (int32_t q = 0; q < pass; ++q) {
                    const std::vector<int32_t> prior = pass_ids(layer, q, token);
                    const std::vector<int32_t> & prior_hints = hints_before[q];
                    if (!run_plan(off, prior, false, nullptr, 0, replay_hit,
                                  filler, cpu_ids, tiles, totals) ||
                        !run_plan(on, prior, !prior_hints.empty(), prior_hints.data(),
                                  prior_hints.size(), replay_hit, filler, cpu_ids,
                                  tiles, totals)) {
                        return;
                    }
                }

                const bool use_hints = !hints.empty();
                size_t hit_off = 0, hit_on = 0;
                std::vector<moe_cache_lru::fill> fills_off, fills_on;
                std::vector<moe_cache_lru::fill> cpu_off, cpu_on;
                if (!run_plan(off, ids, false, nullptr, 0, hit_off, fills_off, cpu_off,
                              tiles, totals) ||
                    !run_plan(on, ids, use_hints, hints.data(), hints.size(), hit_on,
                              fills_on, cpu_on, tiles, totals)) {
                    return;
                }
                CHECK(hit_on >= hit_off);
                if (hit_on != hit_off) {
                    ++totals.hits_changed;
                }

                size_t cache_off = 0, cpu_off_tiles = 0, cache_on = 0, cpu_on_tiles = 0;
                const auto bytes_off = off.gather(ids, tiles, cache_off, cpu_off_tiles);
                const auto bytes_on  = on.gather(ids, tiles, cache_on, cpu_on_tiles);
                CHECK(cache_off + cpu_off_tiles == ids.size());
                CHECK(cache_on + cpu_on_tiles == ids.size());
                totals.cache_tiles += cache_off + cache_on;
                totals.cpu_tiles += cpu_off_tiles + cpu_on_tiles;
                CHECK(bytes_off.size() == ids.size() * kTileBytes);
                CHECK(bytes_off.size() == bytes_on.size());
                const size_t diff = count_differing_bytes(bytes_off, bytes_on);
                CHECK(diff == 0);

                ++totals.events;
                totals.tiles += ids.size();
                totals.bytes += bytes_off.size();
                totals.differing += diff;
                totals.oracle_tiles += 2 * ids.size();

                row_t & row = rows[layer][pass];
                ++row.events;
                row.hints_min = std::min<uint64_t>(row.hints_min, hints.size());
                row.hints_max = std::max<uint64_t>(row.hints_max, hints.size());
                // These columns report union_after (post-add), not the hint union.
                // The engine samples hints after add() too (llama-moe-cache.cpp:612-615).
                row.union_min = std::min<uint64_t>(row.union_min, union_after[pass]);
                row.union_max = std::max<uint64_t>(row.union_max, union_after[pass]);
                row.tiles += ids.size();
                row.bytes += bytes_off.size();
                row.differing += diff;
            }
        }
    }
}

void append_line(std::string & text, const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    CHECK(n > 0 && static_cast<size_t>(n) < sizeof(buf));
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(buf)) {
        std::abort();
    }
    text.append(buf, static_cast<size_t>(n));
}

uint64_t fnv1a64(const std::string & text) {
    uint64_t h = 14695981039346656037ULL; // FNV-1a 64-bit offset basis
    for (unsigned char c : text) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace

int main() {
    static_assert(kCensusNExperts == 128 && kCensusTopK == 8 && kCensusNPass == 3 &&
                  kCensusNTokens == 4 && kCensusNCoreLayers == 12 && kCensusIpFirst == 6,
                  "union_census_ids.inc does not match the seed-0 fixture constants");
    static_assert(kCensusNPass <= llama_roret_union_cache::kMaxPass,
                  "pass index outside the runtime union cache's valid range");

    const std::vector<unsigned char> tiles = make_tiles();
    totals_t totals;
    row_t rows[kCensusNCoreLayers][kCensusNPass] = {};

    // cpu_share = 0 (n_zero = 0) and cpu_share > 0 (n_zero = top-k).
    check_cpu_setting(0, tiles, totals, rows);
    check_cpu_setting(kCensusTopK, tiles, totals, rows);

    const uint64_t expected_events = static_cast<uint64_t>(kCensusNCoreLayers) *
                                     kCensusNPass * kCensusNTokens * 2;
    const uint64_t expected_plans = static_cast<uint64_t>(kCensusNCoreLayers) *
                                    kCensusNTokens * 2 *
                                    (2 * kCensusNPass + kCensusNPass * (kCensusNPass - 1));
    CHECK(totals.events == expected_events);
    CHECK(totals.plans == expected_plans);
    CHECK(totals.differing == 0);
    CHECK(totals.pass0_hint_checks == static_cast<uint64_t>(kCensusNCoreLayers) *
                                      kCensusNTokens * 2);
    CHECK(totals.monotone_checks == static_cast<uint64_t>(kCensusNCoreLayers) *
                                   kCensusNTokens * 2 * (kCensusNPass - 1));

    std::string text;
    append_line(text, "RORET fixture bit-exact tile parity (HOST/TILE level; real LRU planner and union cache)\n");
    append_line(text, " ip pass events hints_min hints_max union_min union_max tiles bytes differing\n");
    for (int32_t layer = 0; layer < kCensusNCoreLayers; ++layer) {
        for (int32_t pass = 0; pass < kCensusNPass; ++pass) {
            const row_t & row = rows[layer][pass];
            append_line(text, "%3d %4d %6llu %10llu %10llu %9llu %9llu %6llu %6llu %9llu\n",
                        kCensusIpFirst + layer, pass,
                        static_cast<unsigned long long>(row.events),
                        static_cast<unsigned long long>(row.hints_min),
                        static_cast<unsigned long long>(row.hints_max),
                        static_cast<unsigned long long>(row.union_min),
                        static_cast<unsigned long long>(row.union_max),
                        static_cast<unsigned long long>(row.tiles),
                        static_cast<unsigned long long>(row.bytes),
                        static_cast<unsigned long long>(row.differing));
        }
    }
    append_line(text, "SUMMARY plans=%llu events=%llu tiles_compared=%llu bytes_compared=%llu differing_bytes=%llu oracle_tiles_checked=%llu cache_tiles=%llu cpu_recomputed_tiles=%llu hit_changed_events=%llu pass0_hint_checks=%llu monotone_checks=%llu failures=%d\n",
                static_cast<unsigned long long>(totals.plans),
                static_cast<unsigned long long>(totals.events),
                static_cast<unsigned long long>(totals.tiles),
                static_cast<unsigned long long>(totals.bytes),
                static_cast<unsigned long long>(totals.differing),
                static_cast<unsigned long long>(totals.oracle_tiles),
                static_cast<unsigned long long>(totals.cache_tiles),
                static_cast<unsigned long long>(totals.cpu_tiles),
                static_cast<unsigned long long>(totals.hits_changed),
                static_cast<unsigned long long>(totals.pass0_hint_checks),
                static_cast<unsigned long long>(totals.monotone_checks),
                g_failures);

    const uint64_t hash = fnv1a64(text);
    CHECK(hash == kPinnedHash);
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::printf("SUMMARY_FNV1A64=%016llx\n", static_cast<unsigned long long>(hash));
    std::printf("test_roret_fixture_parity: %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
