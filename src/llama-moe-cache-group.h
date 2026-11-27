#pragma once

#include <cstdint>

// Loop slots share host expert tensors. They must be cached in one group so the
// bank view, the cache backend and the copy callback agree on one device.
static inline int32_t llama_moe_cache_shared_group(int32_t own_id, int32_t shared_id) {
    return shared_id >= 0 ? shared_id : own_id;
}
