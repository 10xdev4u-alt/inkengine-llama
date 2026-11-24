// Standalone bit-exactness guard for the RORET union-cache header.
//
// Scope: src/llama-roret.h is pure host-side bookkeeping. It contains no ggml
// tensor, no router and no graph node, so this test asserts the strongest
// property the header itself can offer: the caller's bytes (the synthetic
// selected-expert arrays and the keys, i.e. the stand-in for the
// `selected_experts` tensor payload) are byte-identical after add(),
// residents() and hit_rate(). The cache's own mutable bitmap state is expected
// to change on add(); it is not a tensor and is not part of the snapshot.
//
// Build and run (from the repository root):
//   g++ -std=c++17 -Wall -Wextra -Werror -I. tools/roret/test_roret_bitexact.cpp -o /tmp/test_roret_bitexact
//   /tmp/test_roret_bitexact
// C++20 also compiles unchanged. Do not define NDEBUG: CHECK is a real check,
// independent of assert().

#include "src/llama-roret.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (false)

// Raw byte snapshot of a caller-owned buffer (the "tensor" bytes in this
// header-only context).
struct byte_snapshot {
    explicit byte_snapshot(const void * p, size_t n) : bytes(n) {
        if (n != 0) {
            std::memcpy(bytes.data(), p, n);
        }
    }

    bool equals(const void * p, size_t n) const {
        return bytes.size() == n &&
               (n == 0 || std::memcmp(bytes.data(), p, n) == 0);
    }

    std::vector<unsigned char> bytes;
};

template <typename T>
byte_snapshot snapshot(const T & value) {
    return byte_snapshot(&value, sizeof(T));
}

template <typename T>
byte_snapshot snapshot_ids(const T * ids, size_t n) {
    return byte_snapshot(ids, sizeof(T) * n);
}

void check_unchanged(const char * what, const byte_snapshot & before,
                     const void * p, size_t n) {
    if (!before.equals(p, n)) {
        std::fprintf(stderr, "FAIL %s: input bytes changed\n", what);
        ++g_failures;
    }
}

template <typename T, size_t N>
size_t id_bytes(const std::array<T, N> & ids) {
    return sizeof(T) * ids.size();
}

// (a) Three passes with known overlap. The union of {0..7}, {0..5,8,9} and
// {0,1,2,3,6,7,8,10} is exactly {0..10}: 11 residents, hit-rate 7/8 before
// pass 2 is registered. Every call is followed by a byte comparison of all
// inputs, and the read-only calls are checked for cache-state stability too.
void test_three_passes_inputs_unchanged() {
    constexpr int32_t il = 7;
    llama_roret_union_cache cache(128, 8);

    const std::array<int32_t, 8> p0 = { 0, 1, 2, 3, 4, 5, 6,  7 };
    const std::array<int32_t, 8> p1 = { 0, 1, 2, 3, 4, 5, 8,  9 };
    const std::array<int32_t, 8> p2 = { 0, 1, 2, 3, 6, 7, 8, 10 };
    const llama_roret_union_cache::key k0{il, 0};
    const llama_roret_union_cache::key k1{il, 1};
    const llama_roret_union_cache::key k2{il, 2};

    const byte_snapshot b_p0 = snapshot_ids(p0.data(), p0.size());
    const byte_snapshot b_p1 = snapshot_ids(p1.data(), p1.size());
    const byte_snapshot b_p2 = snapshot_ids(p2.data(), p2.size());
    const byte_snapshot b_k0 = snapshot(k0);
    const byte_snapshot b_k1 = snapshot(k1);
    const byte_snapshot b_k2 = snapshot(k2);

    CHECK(cache.residents(il).empty());
    CHECK(cache.total_residents(il) == 0);

    CHECK(cache.add(k0, p0.data(), p0.size()));
    check_unchanged("p0 after add", b_p0, p0.data(), id_bytes(p0));
    check_unchanged("k0 after add", b_k0, &k0, sizeof(k0));

    CHECK(cache.add(k1, p1.data(), p1.size()));
    check_unchanged("p1 after add", b_p1, p1.data(), id_bytes(p1));
    check_unchanged("k1 after add", b_k1, &k1, sizeof(k1));

    const double hit2 = cache.hit_rate(k2, p2.data(), p2.size());
    CHECK(hit2 == 0.875);
    check_unchanged("p2 after hit_rate", b_p2, p2.data(), id_bytes(p2));
    check_unchanged("k2 after hit_rate", b_k2, &k2, sizeof(k2));

    // hit_rate() and residents() must not change the union.
    const std::vector<int32_t> before_read = cache.residents(il);
    const size_t before_total = cache.total_residents(il);
    (void) cache.hit_rate(k2, p2.data(), p2.size());
    (void) cache.residents(il);
    CHECK(cache.residents(il) == before_read);
    CHECK(cache.total_residents(il) == before_total);
    check_unchanged("p2 after reads", b_p2, p2.data(), id_bytes(p2));

    CHECK(cache.add(k2, p2.data(), p2.size()));
    check_unchanged("p2 after add", b_p2, p2.data(), id_bytes(p2));
    check_unchanged("k2 after add", b_k2, &k2, sizeof(k2));

    const std::vector<int32_t> expected = { 0,  1,  2,  3,  4,  5,
                                            6,  7,  8,  9, 10 };
    const std::vector<int32_t> r0 = cache.residents(il);
    const std::vector<int32_t> r1 = cache.residents(il);
    const std::vector<int32_t> r2 = cache.residents(il);
    CHECK(r0 == expected);
    CHECK(r0 == r1);
    CHECK(r1 == r2);
    CHECK(cache.total_residents(il) == 11);
    for (size_t i = 1; i < r2.size(); ++i) {
        CHECK(r2[i - 1] < r2[i]);
    }

    // Re-registering the same pass is idempotent and still leaves inputs alone.
    CHECK(!cache.add(k2, p2.data(), p2.size()));
    check_unchanged("p2 after idempotent add", b_p2, p2.data(), id_bytes(p2));
    CHECK(cache.residents(il) == expected);
}

// (b) Header-level default-off contract: a negative pass is ignored even on a
// fresh cache and even after another pass has registered ids. Once a real
// model exists, the LLAMA_RORET env gate is a separate runtime check in
// llama-moe-cache.cpp; this header never reads the environment.
void test_default_off_negative_pass() {
    llama_roret_union_cache cache(128, 8);
    const int32_t il = 7;
    const std::array<int32_t, 8> ids   = { 11, 12, 13, 14, 15, 16, 17, 18 };
    const std::array<int32_t, 8> other = { 90, 91, 92, 93, 94, 95, 96, 97 };
    const llama_roret_union_cache::key off{il, -1};
    const llama_roret_union_cache::key off2{il, -2};
    const llama_roret_union_cache::key past_max{il, llama_roret_union_cache::kMaxPass + 1};
    const llama_roret_union_cache::key on{il, 0};

    const byte_snapshot b_ids = snapshot_ids(ids.data(), ids.size());
    const byte_snapshot b_other = snapshot_ids(other.data(), other.size());
    const byte_snapshot b_off = snapshot(off);

    CHECK(cache.residents(il).empty());
    CHECK(cache.total_residents(il) == 0);
    CHECK(cache.hit_rate(off, ids.data(), ids.size()) == 0.0);

    CHECK(!cache.add(off, ids.data(), ids.size()));
    CHECK(!cache.add(off2, ids.data(), ids.size()));
    CHECK(!cache.add(past_max, ids.data(), ids.size()));
    CHECK(!cache.add(on, nullptr, 0));
    CHECK(cache.residents(il).empty());
    CHECK(cache.total_residents(il) == 0);
    check_unchanged("ids after disabled adds", b_ids, ids.data(), id_bytes(ids));
    check_unchanged("off key after disabled adds", b_off, &off, sizeof(off));

    CHECK(cache.add(on, ids.data(), ids.size()));
    CHECK(!cache.add(off, other.data(), other.size()));
    const std::vector<int32_t> expected(ids.begin(), ids.end());
    CHECK(cache.residents(il) == expected);
    CHECK(cache.hit_rate(on, ids.data(), ids.size()) == 1.0);
    check_unchanged("other after disabled add", b_other, other.data(), id_bytes(other));
}

// (c) A second logical layer has an independent union. Layer A deliberately
// straddles the 63/64 bitmap word boundary; layer B does not. clear_layer(A)
// must not disturb B.
void test_second_layer_independent() {
    constexpr int32_t il_a = 7;
    constexpr int32_t il_b = 9;
    llama_roret_union_cache cache(128, 8);

    const std::array<int32_t, 8> a = { 60, 61, 62, 63, 64, 65, 66, 67 };
    const std::array<int32_t, 8> b = { 0, 8, 16, 24, 32, 40, 48, 56 };
    const llama_roret_union_cache::key ka{il_a, 0};
    const llama_roret_union_cache::key kb{il_b, 0};

    const byte_snapshot b_a = snapshot_ids(a.data(), a.size());
    const byte_snapshot b_b = snapshot_ids(b.data(), b.size());

    const std::vector<int32_t> expected_a(a.begin(), a.end());
    const std::vector<int32_t> expected_b(b.begin(), b.end());

    CHECK(cache.add(ka, a.data(), a.size()));
    CHECK(cache.add(kb, b.data(), b.size()));
    CHECK(cache.residents(il_a) == expected_a);
    CHECK(cache.residents(il_b) == expected_b);
    CHECK(cache.total_residents(il_a) == 8);
    CHECK(cache.total_residents(il_b) == 8);
    CHECK(cache.hit_rate(ka, b.data(), b.size()) == 0.0);
    CHECK(cache.hit_rate(kb, a.data(), a.size()) == 0.0);

    cache.clear_layer(il_a);
    CHECK(cache.residents(il_a).empty());
    CHECK(cache.total_residents(il_a) == 0);
    CHECK(cache.residents(il_b) == expected_b);
    CHECK(cache.hit_rate(kb, b.data(), b.size()) == 1.0);

    CHECK(cache.add(ka, a.data(), a.size()));
    CHECK(cache.residents(il_a) == expected_a);
    CHECK(cache.residents(il_b) == expected_b);
    check_unchanged("a after all layer ops", b_a, a.data(), id_bytes(a));
    check_unchanged("b after all layer ops", b_b, b.data(), id_bytes(b));
}

} // namespace

int main() {
    test_three_passes_inputs_unchanged();
    test_default_off_negative_pass();
    test_second_layer_independent();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_roret_bitexact: FAIL (%d check(s))\n", g_failures);
        return 1;
    }
    std::puts("test_roret_bitexact: PASS (3 cases, input bytes unchanged)");
    return 0;
}
