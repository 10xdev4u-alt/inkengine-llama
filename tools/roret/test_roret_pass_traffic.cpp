// Host-side replay of the per-pass traffic counters with the real fixture ids.
//
// Drives the real roret_traffic aggregator and the real llama_roret_union_cache
// in the bookkeeping order of llama_moe_cache::copy(): clear on pass 0, sample
// residents and hit_rate() before add(), then register the ids. hit_rate() is a
// separate read-only graph-side sample; copy() only reads residents.
//
// COUNT replay only: no device tensor, no residency measurement, no timing.
#include "../../src/llama-roret.h"
#include "../../src/llama-roret-traffic.h"

#include "union_census_ids.inc"

#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

#ifdef NDEBUG
#error "roret pass traffic replay requires assert(); build with -UNDEBUG"
#endif

using Key = llama_roret_union_cache::key;

namespace {

void append_line(std::string & text, const char * fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    assert(n > 0 && static_cast<size_t>(n) < sizeof(buf));
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

    llama_roret_union_cache cache(kCensusNExperts, kCensusTopK);
    roret_traffic traffic;
    assert(cache.n_experts() == kCensusNExperts);
    assert(cache.n_top_k() == kCensusTopK);

    size_t union_after[kCensusNCoreLayers][kCensusNTokens][kCensusNPass] = {};
    uint64_t events = 0;
    uint64_t sum_hits = 0;
    uint64_t sum_upload_ops = 0;
    uint64_t sum_residents = 0;

    for (int t = 0; t < kCensusNTokens; ++t) {
        for (int c = 0; c < kCensusNCoreLayers; ++c) {
            const int32_t ip = kCensusIpFirst + c;
            for (int p = 0; p < kCensusNPass; ++p) {
                const int32_t * ids = kCensusIds[c][p][t];
                if (p == 0) {
                    cache.clear_layer(ip);
                    assert(cache.residents(ip).empty());
                }

                // copy() order: residents before add(); hit_rate() is read-only.
                const size_t before = cache.residents(ip).size();
                const double hit = cache.hit_rate(Key{ip, p}, ids, kCensusTopK);
                const double hit_count_real = hit * static_cast<double>(kCensusTopK);
                assert(hit_count_real >= 0.0 &&
                       hit_count_real <= static_cast<double>(kCensusTopK));
                const uint64_t hits_count = static_cast<uint64_t>(hit_count_real);
                assert(hit_count_real == static_cast<double>(hits_count));
                const uint64_t upload_ops = static_cast<uint64_t>(kCensusTopK) - hits_count;
                assert(hits_count <= static_cast<uint64_t>(kCensusTopK));
                assert(upload_ops <= static_cast<uint64_t>(kCensusTopK));
                assert(hits_count + upload_ops == static_cast<uint64_t>(kCensusTopK));
                if (p == 0) {
                    assert(hit == 0.0);
                    assert(before == 0);
                    assert(hits_count == 0);
                }

                const bool grew = cache.add(Key{ip, p}, ids, kCensusTopK);
                const size_t after = cache.residents(ip).size();
                assert(after >= before);
                assert(cache.total_residents(ip) == after);
                assert(after >= static_cast<size_t>(kCensusTopK));
                assert(after <= static_cast<size_t>(kCensusNPass * kCensusTopK));
                if (p == 0) {
                    assert(grew);
                }
                union_after[c][t][p] = after;

                // Upper bound: one upload per not-yet-resident id, no coalescing.
                // No device tensor exists here, so uploaded_bytes is zero.
                traffic.add(ip, p, hits_count, upload_ops, 0, before);
                ++events;
                sum_hits += hits_count;
                sum_upload_ops += upload_ops;
                sum_residents += static_cast<uint64_t>(before);
            }
            assert(union_after[c][t][1] >= union_after[c][t][0]);
            assert(union_after[c][t][2] >= union_after[c][t][1]);
        }
    }

    const auto & rows = traffic.rows();
    assert(rows.size() == static_cast<size_t>(kCensusNCoreLayers * kCensusNPass));

    uint64_t total_plans = 0;
    uint64_t total_hits = 0;
    uint64_t total_upload_ops = 0;
    uint64_t total_uploaded_bytes = 0;
    uint64_t total_residents = 0;
    const uint64_t row_cap = static_cast<uint64_t>(kCensusNTokens * kCensusTopK);
    const uint64_t resident_cap = static_cast<uint64_t>(kCensusNTokens * kCensusNPass * kCensusTopK);

    for (const auto & kv : rows) {
        const roret_pass_stats & s = kv.second;
        assert(kv.first.first >= kCensusIpFirst);
        assert(kv.first.first < kCensusIpFirst + kCensusNCoreLayers);
        assert(kv.first.second >= 0 && kv.first.second < kCensusNPass);
        assert(s.plans == static_cast<uint64_t>(kCensusNTokens));
        assert(s.hits <= row_cap && s.upload_ops <= row_cap);
        assert(s.residents <= resident_cap);
        assert(s.uploaded_bytes == 0);
        assert(s.hits + s.upload_ops == row_cap);
        if (kv.first.second == 0) {
            assert(s.hits == 0);
            assert(s.residents == 0);
        }
        total_plans += s.plans;
        total_hits += s.hits;
        total_upload_ops += s.upload_ops;
        total_uploaded_bytes += s.uploaded_bytes;
        total_residents += s.residents;
    }

    assert(events == 144);
    assert(total_plans == 144);
    assert(total_hits == sum_hits && total_upload_ops == sum_upload_ops);
    assert(total_hits + total_upload_ops == 144 * 8);
    assert(total_uploaded_bytes == 0);
    assert(total_residents == sum_residents);
    assert(total_residents == 1086);

    std::string table;
    append_line(table, "RORET host-side per-pass traffic replay (COUNTS only; no device traffic, no time)\n");
    append_line(table, " ip pass plans hits upload_ops uploaded_bytes residents\n");
    for (const auto & kv : rows) {
        const roret_pass_stats & s = kv.second;
        append_line(table, "%3d %4d %5llu %4llu %10llu %15llu %9llu\n",
                kv.first.first, kv.first.second,
                static_cast<unsigned long long>(s.plans),
                static_cast<unsigned long long>(s.hits),
                static_cast<unsigned long long>(s.upload_ops),
                static_cast<unsigned long long>(s.uploaded_bytes),
                static_cast<unsigned long long>(s.residents));
    }
    append_line(table, "TOTAL plans=%llu hits=%llu upload_ops=%llu uploaded_bytes=%llu residents=%llu hits_plus_upload_ops=%llu\n",
            static_cast<unsigned long long>(total_plans),
            static_cast<unsigned long long>(total_hits),
            static_cast<unsigned long long>(total_upload_ops),
            static_cast<unsigned long long>(total_uploaded_bytes),
            static_cast<unsigned long long>(total_residents),
            static_cast<unsigned long long>(total_hits + total_upload_ops));

    const uint64_t table_hash = fnv1a64(table);
    assert(table_hash == 0xf3eaaa28ac25472fULL);
    std::fwrite(table.data(), 1, table.size(), stdout);
    std::printf("TABLE_FNV1A64=%016llx\n",
            static_cast<unsigned long long>(table_hash));
    std::puts("PASS: roret traffic counters replayed with real union cache and fixture ids");
    return 0;
}
