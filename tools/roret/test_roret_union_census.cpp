// Host-side union census driven by the real fixture top-k ids.
//
// Mirrors the state-mutating sequence of the RORET bookkeeping in
// llama_moe_cache::copy(): pass 0 clears the layer, every pass reads residents
// BEFORE add(), then registers its ids and reads residents again. copy() does
// not call hit_rate(); the read-only hit_rate() sample below comes from the
// separate graph-side site in llama-graph.cpp and cannot change the union.
// The union cache is the real src/llama-roret.h used by the engine.
//
// This is a COUNT census. It measures how many resident experts a union would
// keep, not GPU traffic, not residency in a device buffer, and not time.
#include "../../src/llama-roret.h"

#include "union_census_ids.inc"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

#ifdef NDEBUG
#error "roret union census requires assert(); build with -UNDEBUG"
#endif

using Key = llama_roret_union_cache::key;

namespace {

struct pass_stats {
    uint64_t tokens            = 0; // copy() calls modelled for this (ip, pass)
    uint64_t ids               = 0; // top-k ids registered
    double   hits_before_add   = 0.0; // sum of hit counts: hit_rate() * top_k
    double   hit_rate_sum      = 0.0;
    uint64_t adds_that_grew    = 0;
    size_t   union_sum_after   = 0;
    size_t   union_min         = std::numeric_limits<size_t>::max();
    size_t   union_max         = 0;
};

} // namespace

int main() {
    static_assert(kCensusNExperts == 128 && kCensusTopK == 8 && kCensusNPass == 3 &&
                  kCensusNTokens == 4 && kCensusNCoreLayers == 12 && kCensusIpFirst == 6,
                  "union_census_ids.inc does not match the seed-0 fixture constants");
    static_assert(kCensusNPass <= llama_roret_union_cache::kMaxPass,
                  "pass index outside the runtime union cache's valid range");

    llama_roret_union_cache cache(kCensusNExperts, kCensusTopK);
    assert(cache.n_experts() == kCensusNExperts);
    assert(cache.n_top_k() == kCensusTopK);

    pass_stats stats[kCensusNCoreLayers][kCensusNPass] = {};
    size_t token_union[kCensusNCoreLayers][kCensusNTokens][kCensusNPass] = {};
    uint64_t off_loads_layer[kCensusNCoreLayers] = {};
    uint64_t on_loads_layer[kCensusNCoreLayers] = {};
    uint64_t off_loads = 0;
    uint64_t on_loads = 0;
    uint64_t fresh_unions = 0;

    for (int t = 0; t < kCensusNTokens; ++t) {
        for (int c = 0; c < kCensusNCoreLayers; ++c) {
            const int32_t ip = kCensusIpFirst + c;
            size_t after_pass[kCensusNPass] = {};

            for (int p = 0; p < kCensusNPass; ++p) {
                const int32_t * ids = kCensusIds[c][p][t];

                // Exact order of llama_moe_cache::copy(): clear on pass 0,
                // sample the earlier-pass union, then add this pass's ids.
                if (p == 0) {
                    cache.clear_layer(ip);
                    ++fresh_unions;
                }
                const size_t before = cache.residents(ip).size();
                const double hit = cache.hit_rate(Key{ip, p}, ids, kCensusTopK);
                if (p == 0) {
                    // (a) a fresh union has no earlier-pass residents
                    assert(hit == 0.0);
                    assert(before == 0);
                }
                const bool grew = cache.add(Key{ip, p}, ids, kCensusTopK);
                const size_t after = cache.residents(ip).size();

                assert(after >= before);
                assert(cache.total_residents(ip) == after);
                // (b) top_k <= union size <= passes * top_k
                assert(after >= static_cast<size_t>(kCensusTopK));
                assert(after <= static_cast<size_t>(kCensusNPass * kCensusTopK));

                pass_stats & s = stats[c][p];
                s.tokens += 1;
                s.ids += kCensusTopK;
                s.hits_before_add += hit * kCensusTopK;
                s.hit_rate_sum += hit;
                s.adds_that_grew += grew ? 1 : 0;
                s.union_sum_after += after;
                if (after < s.union_min) { s.union_min = after; }
                if (after > s.union_max) { s.union_max = after; }
                after_pass[p] = after;
                token_union[c][t][p] = after;
            }

            // (c) the union cannot shrink as later passes register their ids
            assert(after_pass[1] >= after_pass[0]);
            assert(after_pass[2] >= after_pass[1]);

            // (d) COUNTS: OFF loads three distinct top-8 id sets per token;
            //     ON loads the final union for that token.
            const uint64_t off = static_cast<uint64_t>(kCensusNPass * kCensusTopK);
            const uint64_t on  = static_cast<uint64_t>(after_pass[kCensusNPass - 1]);
            assert(off == 24);
            assert(on >= static_cast<uint64_t>(kCensusTopK) && on <= off);
            off_loads += off;
            on_loads += on;
            off_loads_layer[c] += off;
            on_loads_layer[c] += on;
        }
    }

    // Cross-check monotonicity using the retained per-token values.
    for (int c = 0; c < kCensusNCoreLayers; ++c) {
        for (int t = 0; t < kCensusNTokens; ++t) {
            assert(token_union[c][t][1] >= token_union[c][t][0]);
            assert(token_union[c][t][2] >= token_union[c][t][1]);
        }
    }

    std::puts("RORET union census (COUNTS only; no traffic, residency or time):");
    std::puts(" ip pass tokens ids hits_before_add grew union_min union_max union_mean");
    for (int c = 0; c < kCensusNCoreLayers; ++c) {
        for (int p = 0; p < kCensusNPass; ++p) {
            const pass_stats & s = stats[c][p];
            const double mean = static_cast<double>(s.union_sum_after) / static_cast<double>(s.tokens);
            std::printf("%3d %4d %6llu %3llu %15.1f %4llu %9zu %9zu %10.3f\n",
                    kCensusIpFirst + c, p,
                    static_cast<unsigned long long>(s.tokens),
                    static_cast<unsigned long long>(s.ids),
                    s.hits_before_add,
                    static_cast<unsigned long long>(s.adds_that_grew),
                    s.union_min, s.union_max, mean);
        }
    }

    std::puts("\nlayer off_loads_count on_loads_count mean_union");
    for (int c = 0; c < kCensusNCoreLayers; ++c) {
        const double mean = static_cast<double>(on_loads_layer[c]) / static_cast<double>(kCensusNTokens);
        std::printf("%4d %16llu %15llu %11.3f\n",
                kCensusIpFirst + c,
                static_cast<unsigned long long>(off_loads_layer[c]),
                static_cast<unsigned long long>(on_loads_layer[c]), mean);
    }

    const double overall = static_cast<double>(on_loads) /
            static_cast<double>(kCensusNCoreLayers * kCensusNTokens);
    std::printf("TOTAL off_loads_count=%llu on_loads_count=%llu overall_mean_union=%.6f\n",
            static_cast<unsigned long long>(off_loads),
            static_cast<unsigned long long>(on_loads), overall);
    std::printf("fresh_pass0_unions=%llu PASS: union census assertions held\n",
            static_cast<unsigned long long>(fresh_unions));
    return 0;
}
