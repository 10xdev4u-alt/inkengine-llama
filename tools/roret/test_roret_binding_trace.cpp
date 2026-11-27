// Host-side lifecycle check of the real RORET binding class. No ggml, no device, no timing.
// assert() is the oracle; build with -UNDEBUG. The summary counters are incremented at the
// call sites so the printed line reflects executed operations, not a hardcoded literal.
#include "llama-roret-binding.h"

#include <cassert>
#include <cstdio>

#ifdef NDEBUG
#error "roret binding trace test requires assert(); build with -UNDEBUG"
#endif

struct trace_dummy {};

int main() {
    llama_roret_bindings<trace_dummy> bindings;
    trace_dummy look0, look1, look2, unknown;

    int sets = 0;
    int finds = 0;
    int overwrites = 0;
    int erases = 0;
    int misses = 0;
    int failures = 0;

    // Three distinct lookups, one per pass of the same logical layer.
    bindings.set(&look0, 1, 0); ++sets;
    bindings.set(&look1, 1, 1); ++sets;
    bindings.set(&look2, 1, 2); ++sets;

    const llama_roret_binding * b0 = bindings.find(&look0); ++finds;
    const llama_roret_binding * b1 = bindings.find(&look1); ++finds;
    const llama_roret_binding * b2 = bindings.find(&look2); ++finds;
    if (!(b0 != nullptr && b0->il == 1 && b0->pass == 0)) ++failures;
    if (!(b1 != nullptr && b1->il == 1 && b1->pass == 1)) ++failures;
    if (!(b2 != nullptr && b2->il == 1 && b2->pass == 2)) ++failures;
    assert(b0 != nullptr && b0->il == 1 && b0->pass == 0);
    assert(b1 != nullptr && b1->il == 1 && b1->pass == 1);
    assert(b2 != nullptr && b2->il == 1 && b2->pass == 2);

    // A reused address rebinds to a new pass.
    bindings.set(&look0, 1, 2); ++sets; ++overwrites;
    const llama_roret_binding * reused = bindings.find(&look0); ++finds;
    if (!(reused != nullptr && reused->il == 1 && reused->pass == 2)) ++failures;
    assert(reused != nullptr && reused->il == 1 && reused->pass == 2);

    // A negative pass erases the binding at that address.
    bindings.set(&look0, 1, -1); ++sets; ++erases;
    if (bindings.find(&look0) != nullptr) ++failures;
    assert(bindings.find(&look0) == nullptr);

    // An unregistered address resolves to none; the other passes stay intact.
    if (bindings.find(&unknown) != nullptr) ++failures;
    ++misses;
    assert(bindings.find(&unknown) == nullptr);
    const llama_roret_binding * still1 = bindings.find(&look1); ++finds;
    const llama_roret_binding * still2 = bindings.find(&look2); ++finds;
    if (!(still1 != nullptr && still1->pass == 1)) ++failures;
    if (!(still2 != nullptr && still2->pass == 2)) ++failures;
    assert(still1 != nullptr && still1->pass == 1);
    assert(still2 != nullptr && still2->pass == 2);

    assert(failures == 0);
    std::printf("SUMMARY sets=%d finds=%d overwrite=%d erase=%d misses=%d failures=%d\n",
            sets, finds, overwrites, erases, misses, failures);
    return failures == 0 ? 0 : 1;
}
