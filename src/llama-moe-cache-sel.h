#pragma once

#include <cstddef>
#include <cstdint>

struct llama_moe_cache_sel_dims {
    int32_t n_used   = 0;
    int32_t n_tokens = 0;
    bool    ok       = false;
};

inline llama_moe_cache_sel_dims llama_moe_cache_sel_dims_resolve(bool has_stored, int32_t stored_used, int32_t stored_tokens, size_t n_ids) {
    llama_moe_cache_sel_dims res;
    // divide instead of multiplying so large dims cannot wrap size_t
    if (has_stored && stored_used > 0 && stored_tokens > 0 &&
        n_ids/(size_t) stored_tokens == (size_t) stored_used &&
        n_ids%(size_t) stored_tokens == 0) {
        res.n_used   = stored_used;
        res.n_tokens = stored_tokens;
        res.ok       = true;
    }
    return res;
}
