// PIBS wiring gate test. Exercises the real src/llama-pibs.h decision helper.
// All numbers are model counts, not measured speedups.

#ifdef NDEBUG
#error "test_pibs_wiring requires asserts enabled (build with -UNDEBUG)"
#endif

#include "src/llama-pibs.h"

#include <cstddef>
#include <iostream>

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

struct expected {
    bool        interleave;
    int         steps;
    std::size_t padded;
    std::size_t wasted;
};

void check_decision(int batch, const char * gate_value, const expected & want) {
    const bool gate_on = llama_pibs::gate_enabled(gate_value);
    const auto plan    = llama_pibs::decide(batch, gate_on);
    check(plan.interleave == want.interleave, "interleave decision");
    check(plan.c.steps == want.steps, "step count");
    check(plan.c.padded_total == want.padded, "padded total");
    check(plan.c.wasted_total == want.wasted, "wasted total");
    std::cout << "batch=" << batch << " gate=" << (gate_value ? gate_value : "unset")
              << " interleave=" << (plan.interleave ? 1 : 0)
              << " steps=" << plan.c.steps
              << " padded=" << plan.c.padded_total
              << " wasted=" << plan.c.wasted_total << "\n";
}

}  // namespace

int main() {
    using namespace llama_pibs;

    // Gate parsing mirrors src/llama-moe-cache.cpp: only 1/on/true enable.
    check(!gate_enabled(nullptr), "unset gate is off");
    check(!gate_enabled(""), "empty gate is off");
    check(!gate_enabled("0"), "0 is off");
    check(!gate_enabled("off"), "off is off");
    check(!gate_enabled("TRUE"), "TRUE is off (case sensitive)");
    check(gate_enabled("1"), "1 enables");
    check(gate_enabled("on"), "on enables");
    check(gate_enabled("true"), "true enables");

    // Default-off: unset gate is sequential for every tested batch.
    check_decision(1, nullptr, {false, 1, 16, 13});
    check_decision(2, nullptr, {false, 2, 32, 26});
    check_decision(4, nullptr, {false, 4, 64, 52});
    check_decision(8, nullptr, {false, 8, 128, 104});

    // An explicit off value is identical to unset.
    check_decision(8, "0", {false, 8, 128, 104});

    // Gate on: interleave only at batch >= 4; batches 1 and 2 stay sequential.
    check_decision(1, "1", {false, 1, 16, 13});
    check_decision(2, "1", {false, 2, 32, 26});
    check_decision(4, "1", {true,  3, 48, 36});
    check_decision(8, "1", {true,  3, 48, 24});

    // Interleaved model at batch 1/2 would waste 45/42 slots; the gate keeps
    // those batches sequential by design. This is a recorded regression.
    check(model(1, true).wasted_total == 45, "batch1 interleaved model wastes 45");
    check(model(2, true).wasted_total == 42, "batch2 interleaved model wastes 42");
    check(model(8, false).steps == 8 && model(8, true).steps == 3,
          "batch8 steps 8 -> 3");
    check(model(8, false).wasted_total == 104 && model(8, true).wasted_total == 24,
          "batch8 wasted 104 -> 24");
    check(model(4, false).wasted_total == 52 && model(4, true).wasted_total == 36,
          "breakeven interleaved at batch 4");

    // M <= batch and no M = batch*3 GEMM in the interleaved path.
    for (int batch : {1, 2, 4, 8, 16}) {
        const counts c = model(batch, true);
        check(c.steps == kPasses, "interleaved has 3 steps");
        const int m_per_step = static_cast<int>(c.padded_total / kPasses)
                             - static_cast<int>(c.wasted_total / kPasses);
        check(m_per_step == batch, "interleaved M == batch (never batch*3)");
        for (int t = 0; t < batch; ++t) {
            check(seq_of(batch, true, t, 0) < seq_of(batch, true, t, 1) &&
                  seq_of(batch, true, t, 1) < seq_of(batch, true, t, 2),
                  "interleaved pass order");
            check(seq_of(batch, false, t, 0) < seq_of(batch, false, t, 1) &&
                  seq_of(batch, false, t, 1) < seq_of(batch, false, t, 2),
                  "sequential pass order");
        }
    }

    if (failures != 0) {
        std::cerr << "test_pibs_wiring: FAIL (" << failures << " checks)\n";
        return 1;
    }
    std::cout << "test_pibs_wiring: PASS (gate default-off, batch>=4 only, "
                 "8->3 steps, 104->24 wasted, invariants)\n";
    return 0;
}
