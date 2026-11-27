// CPU-only decision check for shared host expert group assignment. No ggml, no device.
// assert() is the oracle; build with -UNDEBUG.
#include "llama-moe-cache-group.h"

#include <cassert>
#include <cstdio>

#ifdef NDEBUG
#error "roret shared group test requires assert(); build with -UNDEBUG"
#endif

static bool bindable(int32_t chosen_id, int32_t bank_id) {
    return chosen_id == bank_id;
}

int main() {
    // prelude: no shared bank yet, the own group wins
    assert(llama_moe_cache_shared_group(0, -1) == 0);

    // three loop slots reuse the pass-0 bank
    const int32_t own[3]    = {0, 1, 1};
    const int32_t shared[3] = {0, 0, 0};
    assert(llama_moe_cache_shared_group(own[0], shared[0]) == 0);
    assert(llama_moe_cache_shared_group(own[1], shared[1]) == 0);
    assert(llama_moe_cache_shared_group(own[2], shared[2]) == 0);

    int old_bindable = 0;
    int new_bindable = 0;
    for (int i = 0; i < 3; ++i) {
        if (bindable(own[i], shared[i])) ++old_bindable;
        if (bindable(llama_moe_cache_shared_group(own[i], shared[i]), shared[i])) ++new_bindable;
    }
    assert(old_bindable == 1);
    assert(new_bindable == 3);

    // fallback: no shared bank, keep the own group
    assert(llama_moe_cache_shared_group(2, -1) == 2);

    std::printf("shared group decision: old binds %d/3 loop slots, new binds %d/3\n",
            old_bindable, new_bindable);
    return 0;
}
