#pragma once

// Union-resident routing bookkeeping for the three loop passes of Inkling 2 Mini.
// Records the top-k expert ids selected for each (layer, pass) and exposes their
// union. This is host-side metadata only: it never selects experts, never runs
// the router and has no ggml / llama.cpp dependencies.
//
// C++20, header-only. Not thread-safe (graph build is single-threaded).

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace ink::roret {

struct UnionKey {
    int32_t il;
    int8_t  pass; // 0..4
};

class UnionCache {
public:
    static constexpr int8_t kMaxPass = 4;

    explicit UnionCache(int n_experts, int n_top_k)
        : n_experts_(n_experts > 0 ? n_experts : 0),
          n_top_k_(n_top_k > 0 ? n_top_k : 0),
          words_(static_cast<size_t>((n_experts_ + 63) / 64)) {}

    int n_experts() const { return n_experts_; }
    int n_top_k() const { return n_top_k_; }

    // OR the ids selected for (layer, pass) into that layer's union.
    // Out-of-range passes and ids are ignored.
    // Returns true iff at least one expert was newly added to the layer union.
    bool add(const UnionKey & key, std::span<const int32_t> ids) {
        if (!valid_pass(key.pass) || words_ == 0) {
            return false;
        }

        Layer & layer = layer_for(key.il);
        std::vector<uint64_t> & pass_bits = layer.pass[key.pass];
        if (pass_bits.empty()) {
            pass_bits.assign(words_, 0);
        }

        bool added = false;
        for (int32_t id : ids) {
            if (!valid_id(id)) {
                continue;
            }
            const size_t w   = static_cast<size_t>(id) >> 6;
            const uint64_t b = uint64_t{1} << (static_cast<unsigned>(id) & 63u);
            pass_bits[w] |= b;
            if ((layer.union_bits[w] & b) == 0) {
                layer.union_bits[w] |= b;
                added = true;
            }
        }
        return added;
    }

    // Experts to keep resident for (layer, pass): the union of all registered
    // passes of that layer, ascending. `pass` identifies the registration slot;
    // the residency set is deliberately shared across passes of the same layer.
    std::vector<int32_t> residents(const UnionKey & key) const {
        std::vector<int32_t> out;
        const Layer * layer = find_layer(key.il);
        if (layer == nullptr) {
            return out;
        }
        out.reserve(static_cast<size_t>(n_top_k_) * static_cast<size_t>(kMaxPass + 1));
        for (size_t w = 0; w < words_; ++w) {
            uint64_t bits = layer->union_bits[w];
            while (bits != 0) {
                const int bit = std::countr_zero(bits);
                out.push_back(static_cast<int32_t>(w * 64 + static_cast<size_t>(bit)));
                bits &= bits - 1;
            }
        }
        return out;
    }

    // Fraction of `ids` already present in the layer union *before* add().
    // Invalid ids are counted in the denominator but never as hits.
    // Returns 0.0 for an empty span: there is no observed id to claim a hit for.
    double hit_rate(const UnionKey & key, std::span<const int32_t> ids) const {
        if (ids.empty()) {
            return 0.0;
        }
        const Layer * layer = find_layer(key.il);
        if (layer == nullptr) {
            return 0.0;
        }
        size_t hits = 0;
        for (int32_t id : ids) {
            if (!valid_id(id)) {
                continue;
            }
            const size_t w   = static_cast<size_t>(id) >> 6;
            const uint64_t b = uint64_t{1} << (static_cast<unsigned>(id) & 63u);
            if ((layer->union_bits[w] & b) != 0) {
                ++hits;
            }
        }
        return static_cast<double>(hits) / static_cast<double>(ids.size());
    }

    void clear_layer(int32_t il) { layers_.erase(il); }

    size_t total_residents(int32_t il) const {
        const Layer * layer = find_layer(il);
        if (layer == nullptr) {
            return 0;
        }
        size_t n = 0;
        for (uint64_t bits : layer->union_bits) {
            n += static_cast<size_t>(std::popcount(bits));
        }
        return n;
    }

private:
    struct Layer {
        std::vector<uint64_t> union_bits;                     // [words_]
        std::array<std::vector<uint64_t>, kMaxPass + 1> pass; // per-pass contributions
    };

    bool valid_id(int32_t id) const { return id >= 0 && id < n_experts_; }

    static bool valid_pass(int8_t pass) {
        return pass >= 0 && pass <= kMaxPass;
    }

    Layer & layer_for(int32_t il) {
        auto [it, inserted] = layers_.try_emplace(il);
        if (inserted) {
            it->second.union_bits.assign(words_, 0);
        }
        return it->second;
    }

    const Layer * find_layer(int32_t il) const {
        auto it = layers_.find(il);
        return it == layers_.end() ? nullptr : &it->second;
    }

    int    n_experts_ = 0;
    int    n_top_k_   = 0;
    size_t words_     = 0;
    std::map<int32_t, Layer> layers_;
};

} // namespace ink::roret
