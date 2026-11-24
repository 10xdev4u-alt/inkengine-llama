# PIBS model results

Built with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug` (GNU 16.2.1, C++20)
and ran `./build/pibs`. `ctest --test-dir build` passed (1/1).
Assertions passed for all eight (batch, scheduler) runs.

## Comparison

```
batch  scheduler     steps  mean M  wasted tile slots  efficiency
-----  ----------  ------  ------  -----------------  ----------
    1  sequential       1    3.00                 13      18.75%
    1  interleaved       3    1.00                 45       6.25%
    4  sequential       4    3.00                 52      18.75%
    4  interleaved       3    4.00                 36      25.00%
    8  sequential       8    3.00                104      18.75%
    8  interleaved       3    8.00                 24      50.00%
   16  sequential      16    3.00                208      18.75%
   16  interleaved       3   16.00                  0     100.00%
```

`wasted tile slots` = `sum over steps of (ceil(M/16)*16 - M)`.
`efficiency` = `sum(M) / sum(ceil(M/16)*16)`; M is uniform within each run so
this equals the per-step `M / ceil(M/16)*16`.

## batch = 8

- sequential: 8 steps, mean M = 3, 104 wasted tile slots, 18.75% efficiency
- interleaved: 3 steps, mean M = 8, 24 wasted tile slots, 50.00% efficiency

**Yes, at batch=8 interleaving reduces wasted tile slots: 104 → 24 (~77%
fewer) and cuts steps from 8 to 3.** It is worse at batch=1 (45 vs 13 wasted
slots) because three M=1 steps each waste 15 slots; it wins from batch=4 on.

Step detail for batch=8: sequential has 8 steps of `[t_i p0,p1,p2]`, M=3,
padded=16, waste=13 each. Interleaved has 3 steps, each folding all 8 tokens
at one pass level, M=8, padded=16, waste=8 each.

This is a scheduling model only. It does not measure kernel runtime, and
under the modelled dependency `M <= batch` — a single GEMM with `M = batch*3`
would require relaxing intra-token pass order, which the correctness
assertions explicitly forbid.
