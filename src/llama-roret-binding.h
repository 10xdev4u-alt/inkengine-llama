#pragma once

#include <cstdint>
#include <unordered_map>

struct llama_roret_binding {
    int32_t il   = -1;
    int32_t pass = -1;
};

// A slot map is shared by all loop passes of a layer; the GET_ROWS lookup is built
// once per graph, so it is the only tensor that identifies a pass.
template <typename T>
inline const T * llama_roret_bind_key(const T * slot_map, const T * lookup) {
    (void) slot_map;
    return lookup;
}

// Only real loop passes leave a binding; negative passes and unknown lookups resolve to none.
template <typename T>
class llama_roret_bindings {
    std::unordered_map<const T *, llama_roret_binding> by_lookup;

public:
    void set(const T * lookup, int32_t il, int32_t pass) {
        if (il >= 0 && pass >= 0) {
            by_lookup[lookup] = { il, pass };
        } else {
            by_lookup.erase(lookup); // graph tensors can reuse an address from an earlier graph
        }
    }

    const llama_roret_binding * find(const T * lookup) const {
        const auto it = by_lookup.find(lookup);
        return it == by_lookup.end() ? nullptr : &it->second;
    }
};
