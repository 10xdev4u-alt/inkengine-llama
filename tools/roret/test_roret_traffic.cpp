#include "llama-roret-traffic.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

int main() {
    roret_traffic t;

    // add rows out of order; rows() must present them sorted by (il, pass)
    t.add(7, 2, 3, 4, 40, 8);
    t.add(6, 1, 1, 2, 20, 6);
    t.add(7, 0, 5, 6, 60, 0);
    t.add(6, 0, 2, 3, 30, 0);
    t.add(6, 2, 4, 5, 50, 7);
    t.add(7, 1, 2, 3, 30, 5);

    // repeated keys accumulate into a single row
    t.add(6, 0, 1, 1, 10, 0);
    t.add(7, 2, 1, 1, 10, 1);

    const auto & rows = t.rows();
    assert(rows.size() == 6);

    const roret_pass_stats & s60 = rows.at({ 6, 0 });
    assert(s60.plans == 2 && s60.hits == 3 && s60.upload_ops == 4 && s60.uploaded_bytes == 40 && s60.residents == 0);

    const roret_pass_stats & s61 = rows.at({ 6, 1 });
    assert(s61.plans == 1 && s61.hits == 1 && s61.upload_ops == 2 && s61.uploaded_bytes == 20 && s61.residents == 6);

    const roret_pass_stats & s62 = rows.at({ 6, 2 });
    assert(s62.plans == 1 && s62.hits == 4 && s62.upload_ops == 5 && s62.uploaded_bytes == 50 && s62.residents == 7);

    const roret_pass_stats & s70 = rows.at({ 7, 0 });
    assert(s70.plans == 1 && s70.hits == 5 && s70.upload_ops == 6 && s70.uploaded_bytes == 60 && s70.residents == 0);

    const roret_pass_stats & s71 = rows.at({ 7, 1 });
    assert(s71.plans == 1 && s71.hits == 2 && s71.upload_ops == 3 && s71.uploaded_bytes == 30 && s71.residents == 5);

    const roret_pass_stats & s72 = rows.at({ 7, 2 });
    assert(s72.plans == 2 && s72.hits == 4 && s72.upload_ops == 5 && s72.uploaded_bytes == 50 && s72.residents == 9);

    uint64_t total_plans = 0;
    for (const auto & kv : rows) {
        total_plans += kv.second.plans;
    }
    assert(total_plans == 8);

    // deterministic order: std::map sorts by il, then pass
    const std::vector<std::pair<int32_t, int32_t>> expected = {
        { 6, 0 }, { 6, 1 }, { 6, 2 }, { 7, 0 }, { 7, 1 }, { 7, 2 },
    };
    size_t i = 0;
    for (const auto & kv : rows) {
        assert(i < expected.size());
        assert(kv.first == expected[i]);
        ++i;
    }
    assert(i == expected.size());

    t.clear();
    assert(t.rows().empty());

    std::puts("PASS: roret traffic counters accumulate and sort deterministically");
    return 0;
}
