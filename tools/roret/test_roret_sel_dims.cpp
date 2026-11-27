// Standalone test of the selected-expert dims resolver used by the MoE cache copy callback.
// No ggml, no device, no timing. assert() is the oracle; build with -UNDEBUG.
#include "llama-moe-cache-sel.h"

#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

#ifdef NDEBUG
#error "sel dims test requires assert(); build with -UNDEBUG"
#endif

namespace {

struct sel_case {
    const char * name;
    bool         has_stored;
    int32_t      used;
    int32_t      tokens;
    size_t       n_ids;
    bool         ok;
};

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
    const int32_t i32max = std::numeric_limits<int32_t>::max();
    const size_t  smax   = std::numeric_limits<size_t>::max();

    const sel_case cases[] = {
        { "stored_8x12",       true,  8,       12,      96,  true  },
        { "warmup_1x96",       true,  1,       96,      96,  true  },
        { "warmup_8x1",        true,  8,       1,       8,   true  },
        { "product_mismatch",  true,  8,       12,      95,  false },
        { "not_stored",        false, 8,       12,      96,  false },
        { "zero_used",         true,  0,       12,      0,   false },
        { "zero_tokens",       true,  8,       0,       0,   false },
        { "zero_zero",         true,  0,       0,       0,   false },
        { "negative_used",     true,  -1,      96,      96,  false },
        { "negative_tokens",   true,  8,       -1,      96,  false },
        { "negative_both",     true,  -8,      -12,     96,  false },
        { "large_dims",        true,  i32max,  i32max,  96,  false },
        { "large_dims_smax",   true,  i32max,  i32max,  smax, false },
    };

    std::string text;
    size_t passed = 0;
    size_t failed = 0;
    append_line(text, "case has_stored used tokens n_ids ok n_used n_tokens\n");
    for (const sel_case & c : cases) {
        const llama_moe_cache_sel_dims d = llama_moe_cache_sel_dims_resolve(c.has_stored, c.used, c.tokens, c.n_ids);
        append_line(text, "%s %d %d %d %zu %d %d %d\n", c.name, (int) c.has_stored, c.used, c.tokens, c.n_ids,
                (int) d.ok, d.n_used, d.n_tokens);
        const bool good = d.ok == c.ok && (!d.ok || (d.n_used == c.used && d.n_tokens == c.tokens));
        assert(good);
        if (good) {
            ++passed;
        } else {
            ++failed;
        }
    }
    append_line(text, "SUMMARY passed=%zu failed=%zu\n", passed, failed);
    assert(failed == 0 && passed == sizeof(cases)/sizeof(cases[0]));

    const uint64_t hash = fnv1a64(text);
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::printf("SUMMARY_FNV1A64=%016llx\n", static_cast<unsigned long long>(hash));
    std::printf("test_roret_sel_dims: %s\n", failed == 0 ? "PASS" : "FAIL");
    return failed == 0 ? 0 : 1;
}
