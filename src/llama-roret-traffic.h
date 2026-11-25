#pragma once

// Per-pass (logical layer il, loop pass) load-count and byte counters for the
// opt-in RORET union-resident MoE route. Header-only, C++17, standard library
// only. Not thread-safe: the copy callback runs on the graph execution thread.
// These are load counts and byte counters, not wall-clock or speed measurements.

#include <cstdint>
#include <map>
#include <utility>

struct roret_pass_stats {
    uint64_t plans          = 0;
    uint64_t hits           = 0;
    uint64_t upload_ops     = 0;
    uint64_t uploaded_bytes = 0;
    uint64_t residents      = 0;
};

class roret_traffic {
public:
    void add(int32_t il, int32_t pass, uint64_t hits, uint64_t upload_ops, uint64_t uploaded_bytes, uint64_t residents) {
        roret_pass_stats & s = m_rows[std::make_pair(il, pass)];
        ++s.plans;
        s.hits           += hits;
        s.upload_ops     += upload_ops;
        s.uploaded_bytes += uploaded_bytes;
        s.residents      += residents;
    }

    const std::map<std::pair<int32_t, int32_t>, roret_pass_stats> & rows() const {
        return m_rows;
    }

    void clear() {
        m_rows.clear();
    }

private:
    std::map<std::pair<int32_t, int32_t>, roret_pass_stats> m_rows;
};
