// Pass-interleaved batch scheduler (PIBS) — standalone scheduling model.
// No llama.cpp dependencies. Models n_tokens x 3 dependent pass items, two
// schedulers, and WMMA m16 tile utilization. Not a measured kernel speedup.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pibs {

constexpr int kPasses = 3;
constexpr std::size_t kWmmaM = 16;

struct WorkItem {
    int token_id = 0;
    int pass_index = 0;
    bool ready = false;
    bool done = false;
    std::size_t seq = 0;  // global execution order, assigned when done
};

// ---------------------------------------------------------------------------
// Queue: n_tokens * kPasses work items with in-token pass dependencies.
// Item (t, p+1) becomes ready only after item (t, p) is done.
// ---------------------------------------------------------------------------
class Queue {
public:
    explicit Queue(int n_tokens) : n_tokens_(n_tokens) {
        if (n_tokens < 1) {
            throw std::invalid_argument("n_tokens must be >= 1");
        }
        items_.reserve(static_cast<std::size_t>(n_tokens) * kPasses);
        for (int t = 0; t < n_tokens_; ++t) {
            for (int p = 0; p < kPasses; ++p) {
                items_.push_back(WorkItem{t, p, p == 0, false, 0});
            }
        }
    }

    int n_tokens() const { return n_tokens_; }
    std::size_t size() const { return items_.size(); }

    std::size_t index(int t, int p) const {
        return static_cast<std::size_t>(t) * kPasses + static_cast<std::size_t>(p);
    }

    const WorkItem& item(std::size_t idx) const { return items_.at(idx); }
    WorkItem& item(std::size_t idx) { return items_.at(idx); }

    bool is_ready(int t, int p) const { return items_[index(t, p)].ready; }
    bool is_done(int t, int p) const { return items_[index(t, p)].done; }

    bool all_done() const {
        return std::all_of(items_.begin(), items_.end(),
                           [](const WorkItem& w) { return w.done; });
    }

    void mark_done_and_release(std::size_t idx) {
        WorkItem& w = items_.at(idx);
        w.done = true;
        if (w.pass_index + 1 < kPasses) {
            WorkItem& next = items_[index(w.token_id, w.pass_index + 1)];
            if (next.done) {
                throw std::logic_error("dependency released after successor already done");
            }
            next.ready = true;
        }
    }

private:
    int n_tokens_;
    std::vector<WorkItem> items_;
};

// ---------------------------------------------------------------------------
// A scheduling step: the ready (token, pass) pairs folded into one GEMM.
// ---------------------------------------------------------------------------
struct Step {
    std::vector<std::size_t> indices;  // execution order inside the fused GEMM
    std::size_t m = 0;
    std::size_t padded_m = 0;
    std::size_t wasted = 0;
    double efficiency = 0.0;
};

class Scheduler {
public:
    virtual ~Scheduler() = default;
    virtual std::string_view name() const = 0;
    // Return the items to fold this step, in intra-step execution order.
    virtual std::vector<std::size_t> next_step(const Queue& q) const = 0;
};

// Finish all kPasses of a token before starting the next token.
// One step folds that token's 3 passes into one GEMM: M = 3.
class SequentialScheduler final : public Scheduler {
public:
    std::string_view name() const override { return "sequential"; }

    std::vector<std::size_t> next_step(const Queue& q) const override {
        for (int t = 0; t < q.n_tokens(); ++t) {
            bool unfinished = false;
            for (int p = 0; p < kPasses; ++p) {
                unfinished = unfinished || !q.is_done(t, p);
            }
            if (!unfinished) {
                continue;
            }
            std::vector<std::size_t> chosen;
            chosen.reserve(kPasses);
            // Pass order is preserved inside the fused step.
            for (int p = 0; p < kPasses; ++p) {
                if (!q.is_done(t, p)) {
                    chosen.push_back(q.index(t, p));
                }
            }
            return chosen;
        }
        return {};
    }
};

// One pass level across all tokens per step. After step p every (t, p)
// is done, so step p+1 has all tokens ready: M = n_tokens.
class InterleavedScheduler final : public Scheduler {
public:
    std::string_view name() const override { return "interleaved"; }

    std::vector<std::size_t> next_step(const Queue& q) const override {
        int lowest_ready_pass = kPasses;
        for (int p = 0; p < kPasses; ++p) {
            for (int t = 0; t < q.n_tokens(); ++t) {
                if (!q.is_done(t, p) && q.is_ready(t, p)) {
                    lowest_ready_pass = std::min(lowest_ready_pass, p);
                }
            }
        }
        if (lowest_ready_pass == kPasses) {
            return {};
        }
        std::vector<std::size_t> chosen;
        for (int t = 0; t < q.n_tokens(); ++t) {
            const std::size_t idx = q.index(t, lowest_ready_pass);
            if (!q.is_done(t, lowest_ready_pass) && q.is_ready(t, lowest_ready_pass)) {
                chosen.push_back(idx);
            }
        }
        return chosen;
    }
};

struct Result {
    std::vector<Step> steps;
    std::size_t executed = 0;
    std::size_t padded_total = 0;
    std::size_t wasted_total = 0;
    double mean_m = 0.0;
    double efficiency = 0.0;  // sum(M) / sum(padded M)
};

std::size_t padded_to_wmma(std::size_t m) {
    return ((m + kWmmaM - 1) / kWmmaM) * kWmmaM;
}

Result run(const Scheduler& scheduler, Queue& q) {
    Result result;
    std::size_t seq = 0;
    while (!q.all_done()) {
        const std::vector<std::size_t> chosen = scheduler.next_step(q);
        if (chosen.empty()) {
            throw std::logic_error("scheduler stalled with unfinished items");
        }

        Step step;
        for (std::size_t idx : chosen) {
            WorkItem& w = q.item(idx);
            if (w.done || !w.ready) {
                throw std::logic_error("scheduler selected a non-ready item");
            }
            w.seq = seq++;
            q.mark_done_and_release(idx);
            step.indices.push_back(idx);
        }
        step.m = step.indices.size();
        step.padded_m = padded_to_wmma(step.m);
        step.wasted = step.padded_m - step.m;
        step.efficiency = static_cast<double>(step.m) / static_cast<double>(step.padded_m);

        result.executed += step.m;
        result.padded_total += step.padded_m;
        result.wasted_total += step.wasted;
        result.steps.push_back(step);
    }
    result.mean_m = static_cast<double>(result.executed) / result.steps.size();
    result.efficiency = static_cast<double>(result.executed) / result.padded_total;
    return result;
}

// Correctness assertions: exact item count, single execution, pass dependencies.
void verify(const Queue& q, const Result& result) {
    assert(result.executed == q.size());
    if (result.executed != q.size()) {
        throw std::logic_error("wrong number of executed items");
    }

    std::vector<int> executions(q.size(), 0);
    for (const Step& step : result.steps) {
        for (std::size_t idx : step.indices) {
            ++executions.at(idx);
            if (executions[idx] != 1) {
                throw std::logic_error("item executed more than once");
            }
        }
    }
    assert(std::all_of(executions.begin(), executions.end(),
                       [](int n) { return n == 1; }));

    for (int t = 0; t < q.n_tokens(); ++t) {
        assert(q.is_done(t, 0) && q.is_done(t, 1) && q.is_done(t, 2));
        const std::size_t s0 = q.item(q.index(t, 0)).seq;
        const std::size_t s1 = q.item(q.index(t, 1)).seq;
        const std::size_t s2 = q.item(q.index(t, 2)).seq;
        assert(s0 < s1 && s1 < s2);
        if (!(s0 < s1 && s1 < s2)) {
            throw std::logic_error("pass dependency violated for token " + std::to_string(t));
        }
    }
}

struct Row {
    int batch = 0;
    std::string scheduler;
    std::size_t steps = 0;
    double mean_m = 0.0;
    std::size_t wasted = 0;
    double efficiency = 0.0;
};

Row summarise(int batch, const Scheduler& scheduler, Queue& q) {
    const Result result = run(scheduler, q);
    verify(q, result);
    return Row{batch, std::string(scheduler.name()), result.steps.size(),
               result.mean_m, result.wasted_total, result.efficiency};
}

void print_table(const std::vector<Row>& rows) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "batch  scheduler     steps  mean M  wasted tile slots  efficiency\n";
    std::cout << "-----  ----------  ------  ------  -----------------  ----------\n";
    for (const Row& r : rows) {
        std::cout << std::setw(5) << r.batch << "  " << std::left << std::setw(10)
                  << r.scheduler << std::right << "  " << std::setw(6) << r.steps
                  << "  " << std::setw(6) << r.mean_m << "  " << std::setw(17)
                  << r.wasted << "  " << std::setw(9)
                  << (r.efficiency * 100.0) << "%\n";
    }
}

void print_step_detail(int batch, const Scheduler& scheduler, Queue& q) {
    const Result result = run(scheduler, q);
    verify(q, result);
    std::cout << "\nstep detail, batch=" << batch << ", " << scheduler.name() << "\n";
    for (std::size_t i = 0; i < result.steps.size(); ++i) {
        const Step& s = result.steps[i];
        std::cout << "  step " << i << ": M=" << s.m << " padded_m16=" << s.padded_m
                  << " wasted=" << s.wasted << " efficiency="
                  << static_cast<int>(s.efficiency * 100.0) << "% [";
        for (std::size_t j = 0; j < s.indices.size(); ++j) {
            if (j != 0) std::cout << ",";
            const WorkItem& w = q.item(s.indices[j]);
            std::cout << "t" << w.token_id << "p" << w.pass_index;
        }
        std::cout << "]\n";
    }
}

}  // namespace pibs

int main() {
    using namespace pibs;
    const std::vector<int> batches{1, 4, 8, 16};
    const SequentialScheduler sequential;
    const InterleavedScheduler interleaved;

    std::vector<Row> rows;
    for (int batch : batches) {
        Queue q1(batch);
        Queue q2(batch);
        rows.push_back(summarise(batch, sequential, q1));
        rows.push_back(summarise(batch, interleaved, q2));
    }
    print_table(rows);

    Queue q1(8);
    Queue q2(8);
    print_step_detail(8, sequential, q1);
    print_step_detail(8, interleaved, q2);

    std::cout << "\ncorrectness: all suites executed exactly 3*n_tokens items in "
                 "pass order (assertions passed)\n";
    return 0;
}
