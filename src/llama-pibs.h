#pragma once

// Pass-interleaved batching (PIBS) decision and count helper.
// Pure, header-only, standard library only. No ggml/llama dependency.
//
// This is a DECISION/INSTRUMENTATION seam, not a node reorder. It plans a
// schedule and counts modelled m16 tile slots. It never reorders ggml nodes,
// changes a tensor, or relaxes pass order. Real engine-level interleave
// reordering across ubatches remains an open design question (NO_GO locally).

#include <cassert>
#include <cstddef>
#include <string>

namespace llama_pibs {

constexpr int kPasses   = 3;
constexpr int kWmmaM    = 16;
constexpr int kMinBatch = 4;

// Mirror src/llama-moe-cache.cpp: only exact 1, on or true enables the feature.
inline bool gate_enabled(const char * value) {
    if (value == nullptr) {
        return false;
    }
    const std::string on(value);
    return on == "1" || on == "on" || on == "true";
}

inline int padded_to_wmma(int m) {
    return ((m + kWmmaM - 1) / kWmmaM) * kWmmaM;
}

struct counts {
    int         steps        = 0;  // modelled fused GEMM steps
    std::size_t padded_total = 0;  // sum of ceil(M/16)*16
    std::size_t wasted_total = 0;  // sum of padded - M
};

// Counts for an explicit scheduler choice. false is the sequential path
// (M = kPasses per step); true is interleaved (M = n_tokens per step).
inline counts model(int n_tokens, bool interleave) {
    assert(n_tokens >= 1);
    if (!interleave) {
        const int m     = kPasses;
        const int pad   = padded_to_wmma(m);
        const int steps = n_tokens;
        return counts{steps, static_cast<std::size_t>(pad) * steps,
                      static_cast<std::size_t>(pad - m) * steps};
    }
    const int m = n_tokens;  // one pass level, at most one item per token
    assert(m <= n_tokens);              // M <= batch
    assert(m != n_tokens * kPasses);    // no M = batch*3 GEMM
    const int pad   = padded_to_wmma(m);
    const int steps = kPasses;
    return counts{steps, static_cast<std::size_t>(pad) * steps,
                  static_cast<std::size_t>(pad - m) * steps};
}

// Global execution order of (token, pass) for the two model schedules.
// Interleaved groups by pass level; sequential groups by token.
inline std::size_t seq_of(int n_tokens, bool interleave, int token, int pass) {
    assert(n_tokens >= 1);
    assert(token >= 0 && token < n_tokens);
    assert(pass >= 0 && pass < kPasses);
    if (interleave) {
        return static_cast<std::size_t>(pass) * n_tokens + token;
    }
    return static_cast<std::size_t>(token) * kPasses + pass;
}

struct decision {
    bool   interleave = false;  // true only when gate on AND n_tokens >= 4
    counts c;
};

// Default-off decision: batches 1 and 2 always stay sequential.
inline decision decide(int n_tokens, bool gate_on) {
    assert(n_tokens >= 1);
    const bool interleave = gate_on && n_tokens >= kMinBatch;
    const counts c = model(n_tokens, interleave);
    // Intra-token pass order: seq(t,0) < seq(t,1) < seq(t,2) for every token.
    for (int t = 0; t < n_tokens; ++t) {
        const std::size_t s0 = seq_of(n_tokens, interleave, t, 0);
        const std::size_t s1 = seq_of(n_tokens, interleave, t, 1);
        const std::size_t s2 = seq_of(n_tokens, interleave, t, 2);
        assert(s0 < s1 && s1 < s2);
    }
    return decision{interleave, c};
}

}  // namespace llama_pibs
