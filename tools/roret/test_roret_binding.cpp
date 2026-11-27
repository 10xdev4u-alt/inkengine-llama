// Host-side replay of the RORET per-pass binding key used by llama_moe_cache.
// No ggml, no device, no timing. assert() is the oracle; build with -UNDEBUG.
#include "llama-roret-binding.h"

#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#ifdef NDEBUG
#error "roret binding test requires assert(); build with -UNDEBUG"
#endif

namespace {

struct fake_tensor {};

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
    // One physical layer, three loop builds: one shared slot map, one lookup per pass.
    fake_tensor slot_map;
    fake_tensor lookup0;
    fake_tensor lookup1;
    fake_tensor lookup_neg;
    fake_tensor lookup_bad_il;
    fake_tensor recycled;
    fake_tensor unknown;

    assert(llama_roret_bind_key(&slot_map, &lookup0) == &lookup0);
    assert(llama_roret_bind_key(&slot_map, &lookup1) == &lookup1);
    assert(llama_roret_bind_key(&slot_map, &lookup0) != llama_roret_bind_key(&slot_map, &lookup1));
    assert(llama_roret_bind_key(&slot_map, &lookup0) != &slot_map);

    llama_roret_bindings<fake_tensor> bindings;
    bindings.set(llama_roret_bind_key(&slot_map, &lookup0), 6, 0);
    bindings.set(llama_roret_bind_key(&slot_map, &lookup1), 6, 1);
    bindings.set(llama_roret_bind_key(&slot_map, &lookup_neg), 6, -1);
    bindings.set(llama_roret_bind_key(&slot_map, &lookup_bad_il), -1, 2);
    // a recycled address re-registered with a negative pass must not keep its old binding
    bindings.set(&recycled, 6, 2);
    assert(bindings.find(&recycled) != nullptr);
    bindings.set(&recycled, 6, -1);

    const llama_roret_binding * b0 = bindings.find(&lookup0);
    const llama_roret_binding * b1 = bindings.find(&lookup1);
    assert(b0 != nullptr && b0->il == 6 && b0->pass == 0);
    assert(b1 != nullptr && b1->il == 6 && b1->pass == 1);
    assert(b0 != b1);
    assert(bindings.find(&lookup_neg) == nullptr);
    assert(bindings.find(&recycled) == nullptr);
    assert(bindings.find(&lookup_bad_il) == nullptr);
    assert(bindings.find(&unknown) == nullptr);
    assert(bindings.find(&slot_map) == nullptr);

    std::string text;
    append_line(text, "key0=%d key1=%d key_shared=%d\n",
            llama_roret_bind_key(&slot_map, &lookup0) == &lookup0,
            llama_roret_bind_key(&slot_map, &lookup1) == &lookup1,
            llama_roret_bind_key(&slot_map, &lookup0) == &slot_map);
    append_line(text, "p0 il=%d pass=%d p1 il=%d pass=%d neg=%d recycled=%d bad_il=%d unknown=%d shared=%d\n",
            b0->il, b0->pass, b1->il, b1->pass,
            bindings.find(&lookup_neg) == nullptr,
            bindings.find(&recycled) == nullptr,
            bindings.find(&lookup_bad_il) == nullptr,
            bindings.find(&unknown) == nullptr,
            bindings.find(&slot_map) == nullptr);

    const uint64_t hash = fnv1a64(text);
    assert(hash == 0xe47415a4d1ddd37fULL && "unexpected per-pass binding summary");
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::printf("SUMMARY_FNV1A64=%016llx\n", static_cast<unsigned long long>(hash));
    std::puts("PASS: roret per-pass bindings keyed by the lookup, not the shared slot map");
    return 0;
}
