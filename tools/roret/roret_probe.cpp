// RORET precondition probe: measure cross-pass MoE routing agreement.
// Standalone C++20, no llama.cpp build required.
//
// Model of one looped MoE layer (Inkling 2 Mini: 128 experts, top-8):
//   pass 1: x = prelude                      (gate gamma = 0)
//   pass p: x = prelude + gamma_p * msg_{p-1}
//   router logits = W_r @ x,  selected = top8(logits)
// RORET is viable iff the selected sets agree across passes with high probability.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kDim = 256;

struct Args {
    int tokens = 10000;
    double gamma2 = 0.30;
    double gamma3 = 0.60;
    double msg_scale = 1.0;
    uint32_t seed = 20261124;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> double { return i + 1 < argc ? std::atof(argv[++i]) : 0.0; };
        if (k == "--tokens") a.tokens = int(next());
        else if (k == "--gamma2") a.gamma2 = next();
        else if (k == "--gamma3") a.gamma3 = next();
        else if (k == "--msg-scale") a.msg_scale = next();
        else if (k == "--seed") a.seed = uint32_t(next());
    }
    return a;
}

std::array<int, kTopK> topk(const std::array<double, kExperts>& logits) {
    std::array<int, kExperts> idx{};
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + kTopK, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::array<int, kTopK> out{};
    for (int i = 0; i < kTopK; ++i) out[i] = idx[i];
    return out;
}

int overlap(const std::array<int, kTopK>& a, const std::array<int, kTopK>& b) {
    int n = 0;
    for (int x : a) for (int y : b) n += (x == y);
    return n;
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    std::mt19937 rng(a.seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    std::vector<std::array<double, kExperts>> W(kDim);
    for (auto& row : W) for (auto& v : row) v = norm(rng) / std::sqrt(double(kDim));

    long long exact_1_2 = 0, exact_1_3 = 0, exact_all = 0, sum_ov_1_2 = 0, sum_ov_1_3 = 0;
    long long sum_union = 0, sum_hits = 0;

    for (int t = 0; t < a.tokens; ++t) {
        std::array<double, kDim> prelude{}, msg{};
        for (int d = 0; d < kDim; ++d) { prelude[d] = norm(rng); msg[d] = norm(rng) * a.msg_scale; }

        std::array<std::array<double, kExperts>, 3> logits{};
        const double gamma[3] = {0.0, a.gamma2, a.gamma3};
        for (int p = 0; p < 3; ++p) {
            std::array<double, kDim> x{};
            for (int d = 0; d < kDim; ++d) x[d] = prelude[d] + gamma[p] * msg[d];
            for (int e = 0; e < kExperts; ++e) {
                double s = 0.0;
                for (int d = 0; d < kDim; ++d) s += W[d][e] * x[d];
                logits[p][e] = s;
            }
        }
        auto s1 = topk(logits[0]), s2 = topk(logits[1]), s3 = topk(logits[2]);
        int o12 = overlap(s1, s2), o13 = overlap(s1, s3);
        exact_1_2 += (o12 == kTopK);
        exact_1_3 += (o13 == kTopK);
        exact_all += (o12 == kTopK && o13 == kTopK);
        sum_ov_1_2 += o12;
        sum_ov_1_3 += o13;

        // union-resident variant: keep the union of the three top-8 sets loaded
        bool present[kExperts] = {};
        for (int e : s1) present[e] = true;
        for (int e : s2) present[e] = true;
        for (int e : s3) present[e] = true;
        int un = 0;
        for (bool b : present) un += b;
        sum_union += un;
        int hits = 0;
        for (int p = 1; p < 3; ++p) {
            for (int e : topk(logits[p])) {
                bool in1 = false;
                for (int f : s1) in1 |= (f == e);
                hits += in1;
            }
        }
        sum_hits += hits;  // out of 16 pass-2/3 slots
    }

    const double n = double(a.tokens);
    std::printf("{\"probe\":\"roret\",\"tokens\":%d,\"gamma\":[0.0,%.3f,%.3f],"
                "\"msg_scale\":%.3f,\"exact_rate_1_2\":%.4f,\"exact_rate_1_3\":%.4f,"
                "\"exact_rate_all\":%.4f,\"mean_overlap_1_2\":%.3f,\"mean_overlap_1_3\":%.3f,"
                "\"mean_union_experts\":%.2f,\"load_reduction_vs_24\":%.3f,"
                "\"pass23_hit_rate_from_pass1\":%.3f,"
                "\"verdict_exact\":\"%s\",\"verdict_union\":\"%s\"}\n",
                a.tokens, a.gamma2, a.gamma3, a.msg_scale,
                double(exact_1_2) / n, double(exact_1_3) / n, double(exact_all) / n,
                double(sum_ov_1_2) / n / kTopK, double(sum_ov_1_3) / n / kTopK,
                double(sum_union) / n, 1.0 - double(sum_union) / n / 24.0,
                double(sum_hits) / n / 16.0,
                double(exact_all) / n >= 0.95 ? "GO" : "NO_GO",
                double(sum_union) / n <= 10.0 ? "GO" : "NO_GO");
    return 0;
}
