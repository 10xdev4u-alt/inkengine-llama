#pragma once

// Opt-in, host-side union bookkeeping for union-resident MoE routing (RORET).
// Records the host-observed top-k expert ids selected for each (layer, pass) and
// exposes their per-layer union so a loop driver can keep that union resident
// across the passes of a looped model.
//
// This is bookkeeping only: it never runs the router, never selects experts and
// never changes any graph math. The router still runs on every pass; only the
// set of tiles a driver chooses to keep resident changes.
//
// Header-only, C++17, standard library only (no ggml/llama dependencies).
// Not thread-safe: graph build is single-threaded.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

class llama_roret_union_cache {
public:
    struct key {
        int32_t il   = -1;
        int32_t pass = -1; // 0..kMaxPass, negative disables the feature
    };

    static constexpr int32_t kMaxPass = 4;

    llama_roret_union_cache(int n_experts, int n_top_k)
        : n_experts_(n_experts > 0 ? n_experts : 0),
          n_top_k_(n_top_k > 0 ? n_top_k : 0),
          words_(static_cast<size_t>((n_experts_ + 63) / 64)) {}

    int n_experts() const { return n_experts_; }
    int n_top_k()   const { return n_top_k_;   }

    // OR the host-observed ids selected for (il, pass) into that layer's union.
    // Out-of-range passes and ids are ignored. Returns true iff the union grew.
    // Const because build_moe_ffn is const and receives a const pointer; the
    // internal state is mutable and graph build is single-threaded.
    bool add(const key & k, const int32_t * ids, size_t n_ids) const {
        if (!valid_pass(k.pass) || words_ == 0 || ids == nullptr) {
            return false;
        }

        layer_state & layer = layer_for(k.il);
        std::vector<uint64_t> & pass_bits = layer.pass[k.pass];
        if (pass_bits.empty()) {
            pass_bits.assign(words_, 0);
        }

        bool added = false;
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t id = ids[i];
            if (!valid_id(id)) {
                continue;
            }
            const size_t   w = static_cast<size_t>(id) >> 6;
            const uint64_t b = uint64_t{1} << (static_cast<unsigned>(id) & 63u);
            pass_bits[w] |= b;
            if ((layer.union_bits[w] & b) == 0) {
                layer.union_bits[w] |= b;
                added = true;
            }
        }
        return added;
    }

    // Experts to keep resident for a layer: the union of all registered passes,
    // ascending. The residency set is deliberately shared across passes, so the
    // value is identical for every pass of that layer once all are registered.
    std::vector<int32_t> residents(int32_t il) const {
        std::vector<int32_t> out;
        const layer_state * layer = find_layer(il);
        if (layer == nullptr) {
            return out;
        }
        out.reserve(static_cast<size_t>(n_top_k_) * static_cast<size_t>(kMaxPass + 1));
        for (size_t w = 0; w < words_; ++w) {
            uint64_t bits = layer->union_bits[w];
            while (bits != 0) {
                const unsigned bit = countr_zero(bits);
                out.push_back(static_cast<int32_t>(w * 64 + bit));
                bits &= bits - 1;
            }
        }
        return out;
    }

    // Fraction of `ids` already present in the layer union *before* add().
    // Invalid ids are counted in the denominator but never as hits.
    double hit_rate(const key & k, const int32_t * ids, size_t n_ids) const {
        if (ids == nullptr || n_ids == 0) {
            return 0.0;
        }
        const layer_state * layer = find_layer(k.il);
        if (layer == nullptr) {
            return 0.0;
        }
        size_t hits = 0;
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t id = ids[i];
            if (!valid_id(id)) {
                continue;
            }
            const size_t   w = static_cast<size_t>(id) >> 6;
            const uint64_t b = uint64_t{1} << (static_cast<unsigned>(id) & 63u);
            if ((layer->union_bits[w] & b) != 0) {
                ++hits;
            }
        }
        return static_cast<double>(hits) / static_cast<double>(n_ids);
    }

    // Mandatory between tokens: otherwise the union accumulates across tokens
    // and grows toward all n_experts.
    void clear_layer(int32_t il) const { layers_.erase(il); }

    size_t total_residents(int32_t il) const {
        const layer_state * layer = find_layer(il);
        if (layer == nullptr) {
            return 0;
        }
        size_t n = 0;
        for (uint64_t bits : layer->union_bits) {
            while (bits != 0) {
                bits &= bits - 1;
                ++n;
            }
        }
        return n;
    }

private:
    struct layer_state {
        std::vector<uint64_t> union_bits;                             // [words_]
        std::array<std::vector<uint64_t>, kMaxPass + 1> pass;         // per-pass contributions
    };

    static unsigned countr_zero(uint64_t x) {
        unsigned n = 0;
        while ((x & 1) == 0) {
            x >>= 1;
            ++n;
        }
        return n;
    }

    bool valid_id(int32_t id) const { return id >= 0 && id < n_experts_; }

    static bool valid_pass(int32_t pass) { return pass >= 0 && pass <= kMaxPass; }

    layer_state & layer_for(int32_t il) const {
        auto it = layers_.find(il);
        if (it == layers_.end()) {
            it = layers_.emplace(il, layer_state{}).first;
            it->second.union_bits.assign(words_, 0);
        }
        return it->second;
    }

    const layer_state * find_layer(int32_t il) const {
        auto it = layers_.find(il);
        return it == layers_.end() ? nullptr : &it->second;
    }

    int    n_experts_ = 0;
    int    n_top_k_   = 0;
    size_t words_     = 0;
    mutable std::map<int32_t, layer_state> layers_;
};
