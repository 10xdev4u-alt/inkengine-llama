#include "union_cache.hpp"

#include <array>
#include <cassert>
#include <cstdio>
#include <vector>

using ink::roret::UnionCache;
using ink::roret::UnionKey;

int main() {
    constexpr int32_t il = 7;

    // (a) three passes of 8 ids with ~70% overlap: union 8..12, pass-2 hit > 0.5
    {
        UnionCache cache(128, 8);
        const std::array<int32_t, 8> p0 = { 0, 1, 2, 3, 4, 5, 6,  7 };
        const std::array<int32_t, 8> p1 = { 0, 1, 2, 3, 4, 5, 8,  9 };
        const std::array<int32_t, 8> p2 = { 0, 1, 2, 3, 6, 7, 8, 10 };

        assert(cache.add({ il, 0 }, p0));
        assert(cache.add({ il, 1 }, p1));

        const double hit2 = cache.hit_rate({ il, 2 }, p2); // before add
        assert(hit2 > 0.5);
        assert(cache.add({ il, 2 }, p2));

        const size_t union_size = cache.total_residents(il);
        assert(union_size >= 8 && union_size <= 12);
        assert(cache.residents({ il, 0 }) == cache.residents({ il, 2 }));
        const auto r2 = cache.residents({ il, 2 });
        assert(r2.size() == union_size);
        for (size_t i = 1; i < r2.size(); ++i) {
            assert(r2[i - 1] < r2[i]);
        }
        std::printf("case (a): hit_rate(pass2)=%.3f union_size=%zu\n", hit2, union_size);
    }

    // (b) clear_layer resets the layer without touching other layers
    {
        UnionCache cache(128, 8);
        const std::array<int32_t, 8> ids = { 11, 12, 13, 14, 15, 16, 17, 18 };
        assert(cache.add({ il, 0 }, ids));
        assert(cache.add({ il + 1, 0 }, ids));

        cache.clear_layer(il);
        assert(cache.total_residents(il) == 0);
        assert(cache.residents({ il, 1 }).empty());
        assert(cache.hit_rate({ il, 0 }, ids) == 0.0);
        assert(cache.total_residents(il + 1) == 8);

        assert(cache.add({ il, 0 }, ids)); // reusable after clear
        assert(cache.total_residents(il) == 8);
    }

    // (c) disjoint passes give the full 24-expert union
    {
        UnionCache cache(128, 8);
        for (int8_t pass = 0; pass < 3; ++pass) {
            std::array<int32_t, 8> ids{};
            for (size_t i = 0; i < ids.size(); ++i) {
                ids[i] = static_cast<int32_t>(pass * 8 + static_cast<int8_t>(i));
            }
            assert(cache.add({ il, pass }, ids));
        }
        assert(cache.total_residents(il) == 24);
        assert(cache.residents({ il, 2 }).size() == 24);
        std::printf("case (c): union_size=%zu\n", cache.total_residents(il));
    }

    // (d) re-registering the same ids adds nothing, and invalid inputs are ignored
    {
        UnionCache cache(128, 8);
        const std::array<int32_t, 8> ids = { 20, 21, 22, 23, 24, 25, 26, 27 };
        assert(cache.add({ il, 0 }, ids));
        assert(!cache.add({ il, 0 }, ids));
        assert(!cache.add({ il, 1 }, ids));
        assert(!cache.add({ il, 2 }, ids));
        assert(cache.total_residents(il) == 8);

        const std::array<int32_t, 2> invalid = { -1, 128 };
        assert(!cache.add({ il, 5 }, invalid)); // pass out of range
        assert(!cache.add({ il, 0 }, invalid)); // ids out of range
        assert(cache.total_residents(il) == 8);
    }

    std::puts("all union cache tests passed");
    return 0;
}
