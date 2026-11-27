# Traffic-count model — RORET union-resident MoE (issue #5)

**Status: LOAD-COUNT MODEL ONLY.** This report is an analytic count of expert-tile
uploads derived from `tools/roret/RESULTS.md` and `tools/roret/KAGGLE_RESULTS.md`.
It is **not** wall-clock, not `tok/s`, not a measured HBM byte count, and not a
performance claim. No code was changed and no commit was made for this report.

---

## 0. Verified inputs and provenance

| `msg_scale` | mean union experts | load reduction vs 24 | pass-2/3 hit rate from pass-1 (100k) | union verdict |
|---|---|---|---|---|
| 0.05 | 8.25 | 65.6% | 97.7% | GO |
| 0.10 | 8.48 | 64.7% | 95.6% | GO |
| 0.30 | 9.29 | 61.3% | 88.1% | GO |
| 1.00 | 11.59 | 51.7% | 66.7% | NO_GO |

* Source of the 100k-token figures: `tools/roret/KAGGLE_RESULTS.md`. They reproduce the
  3,000-token local figures in `tools/roret/RESULTS.md` to ~0.5%.
* Reduction is `(24 - union) / 24`; e.g. `(24 - 8.25)/24 = 65.625% -> 65.6%`.
  The small differences from the 3k table (8.47/8.28/9.28/11.58) are sampling;
  this report uses the 100k numbers listed above as verified inputs.
* **Source-status caveat (carried from both RESULTS files):** the `msg_scale` / gamma
  knobs are *synthetic scalar probe stand-ins* for the upstream rank-r gain/forget
  loop controllers (`LLM_TENSOR_LOOP_*_CTRL_{DOWN,GAIN,FORGET}`, rank 64). Upstream
  already supplies the loop graph and converter; this fork's contribution is
  union-resident MoE routing on top of it. So these are model inputs, not measured
  model parameters.
* Structured constants from `tools/audit/README.md` and
  `tools/audit/make_moe_fixture.py`: 24 layers, 128 experts/layer, top-8, 3 loop
  passes, 4-vector cross-loop message. `N_SLOT = N_LAYERS + (N_PASSES-1) *
  (CORE_END-CORE_START) = 24 + 2*12 = 48` — **48 upstream cache slots**, hence
  **12 looped (core) layers**.
* Fixture content SHA-256 `fff0e60fa4f55ba717d1e0bdc0e89695a0d35c85df680a15fd57ed1a0ac959ff`;
  F32 dump content SHA-256 `c85a8629c4b45a6f81bc505f5959dea469249f13e18029c47d13513fb686f5ec`
  (from `tools/roret/BITEXACT_REPORT.md`). These pin the fixture used for bit-exactness
  and traffic-count checks; they do not validate the real model.

---

## 1. Load-count model: OFF vs ON

### Formulas

```
uploads_off = passes * top_k = 3 * 8 = 24          (per looped layer, per token)
uploads_on  = |union(top-8 on pass 0, 1, 2)|        (per looped layer, per token)

reduction   = (uploads_off - uploads_on) / uploads_off = (24 - union) / 24
```

`uploads_on` assumes a tile stays resident across the three passes once uploaded,
and that the union is **cleared per token** via `clear_layer(ip)` on pass 0 so it
does not grow across tokens. In the implementation this is
`roret->clear_layer(rb.il)` when `rb.pass == 0` in
`src/llama-moe-cache.cpp::impl::copy()` (lines 875–878), and the union is built by
`roret->add({il, pass}, ids)` on every pass (line 879).

The union is a **mean over 100k synthetic tokens**, so `uploads_on` is an expected
value; a single token's union is an integer between 8 and 24.

### Per looped layer, and all 12 looped layers per token

| `msg_scale` | OFF uploads/layer | ON uploads/layer (mean union) | reduction | OFF uploads/token (×12 layers) | ON uploads/token (×12) |
|---|---|---|---|---|---|
| 0.05 | 24 | 8.25 | 65.6% | 288 | 99.00 |
| 0.10 | 24 | 8.48 | 64.7% | 288 | 101.76 |
| 0.30 | 24 | 9.29 | 61.3% | 288 | 111.48 |
| 1.00 | 24 | 11.59 | 51.7% | 288 | 139.08 |

**Answer to the four counts (per looped layer):** OFF = **24** at every scale;
ON = **8.25 / 8.48 / 9.29 / 11.59** at `msg_scale` 0.05 / 0.10 / 0.30 / 1.00.

Capacity sanity: the largest mean union (11.59) is well below the 48 slots
available per group, so the union fits. This is a capacity *check*, not a guarantee
of residency — see §4.

---

## 2. Memory-traffic estimate — ESTIMATE, not a measurement

**Explicit assumption (not measured).** Model one expert tile as a single
`d_model × d_ff` matrix stored at FP16, i.e. **2 bytes/element**:

```
bytes_per_tile = s * d_model * d_ff,   s = 2 B/element (FP16)
```

The task sketch suggested the convention `d_model*d_ff/2` bytes at FP16; that
would be `s = 0.5 B/element`, which is not FP16. The general formula above lets a
reader substitute either convention; this report uses the physically correct
`s = 2`. Two further honest qualifications:

* A real FFN expert has **two** matrices (`w13` = gate/up, shape `2*d_ff × d_model`,
  and `w2` = down, shape `d_model × d_ff`), so a full expert is up to
  `3*d_model*d_ff` elements = `6*d_model*d_ff` FP16 bytes — up to **3×** the
  single-tile figure used here.
* The audit fixture stores expert weights as **NVFP4 codes with E4M3 block
  scales** (`tools/audit/README.md`), not FP16. FP16 here is a deliberately
  simple illustrative assumption for the traffic model.

**Nothing in this section is a measured HBM byte count.** The `st.bytes` counter
in `src/llama-moe-cache.cpp` measures actual `ggml_backend_tensor_set_async`
payload sizes once a real model runs on a GPU; until then all byte figures below
are estimates.

### Tile-size range for plausible `d_ff` (with `d_model = 256`, the probe dim)

| `d_ff` | `bytes_per_tile = 2*256*d_ff` | per looped layer OFF (×24) | per token OFF (×12 layers) |
|---|---|---|---|
| 512  | 262,144 B (256 KiB) | 6 MiB | 72 MiB |
| 1024 | 524,288 B (512 KiB) | 12 MiB | 144 MiB |
| 2048 | 1,048,576 B (1 MiB)  | 24 MiB | 288 MiB |

### Estimated bytes/token at the representative `d_ff = 1024` (tile = 0.5 MiB)

| `msg_scale` | ON bytes/layer | ON bytes/token (×12) | OFF bytes/token | estimated saved bytes/token |
|---|---|---|---|---|
| 0.05 | 4.125 MiB | 49.50 MiB | 144 MiB | 94.50 MiB |
| 0.10 | 4.240 MiB | 50.88 MiB | 144 MiB | 93.12 MiB |
| 0.30 | 4.645 MiB | 55.74 MiB | 144 MiB | 88.26 MiB |
| 1.00 | 5.795 MiB | 69.54 MiB | 144 MiB | 74.46 MiB |

Scale these linearly for other `d_ff` (×0.5 at 512, ×2 at 2048) and by up to ×3
if all three FFN matrices are counted. Again: **estimates**, not measured traffic.

---

## 3. Instrumentation that would turn this model into a measurement

The counters already exist in `src/llama-moe-cache.cpp`; the missing piece is
attribution to `(logical il, pass)`.

1. **`n_hit`** — local `size_t n_hit = 0` at line 895, passed by reference into
   `g.lru.plan(...)` at line 896, incremented at line 245 when a *real* selected
   expert is already resident (`slots[e] < n_slots` and `i < n_real`). Resident
   hints are deliberately excluded from `n_hit` (line 244 guard and the comment
   at lines 224–227: "n_hit counts real ids only"). Accumulated into
   `stats_small.hits` / `stats_large.hits` at line 929.
2. **Uploaded bytes** — local `size_t bytes = 0` at line 912; the actual upload
   is `ggml_backend_tensor_set_async(...)` at line 922, one call per coalesced run
   of consecutive `(expert, slot)` pairs, adding `n*expert_size` at line 923
   where `expert_size = w->nb[2]` (line 916). Summed into `st.bytes` at line 932,
   with `st.misses += fills.size() + cpu.size()` at line 930. `log_stats()`
   (lines 940–947) prints `hits`, `misses`, `hit rate`, and `uploaded MiB` for
   "ubatch <= 8" and "ubatch > 8".
3. **`copy_experts`** — `impl::copy_experts()` (lines 785–818) coalesces
   consecutive resident experts into one async device→device copy; it records
   `stats_copy.hits += n` and `stats_copy.bytes += ggml_nbytes(src_view)` at
   lines 814–815, logged at line 952. This is the large-batch cache-read traffic
   counter, distinct from the host→device uploads above.

**What to add (minimal, per-pass attribution):** carry `rb.il` / `rb.pass` from
the RORET binding (lines 872–885) into the stats update at lines 928–932, and
count (a) `upload_ops` = number of `tensor_set_async` calls at line 922, (b)
`uploaded_bytes` = their byte total, (c) `residents` = `roret->residents(il).size()`
at pass > 0, and (d) `n_hit` already keyed by pass. That yields a direct
off-vs-on table of uploads and bytes per `(layer, pass)` to compare against §1/§2.
Aggregated `log_stats()` output is not enough by itself because it pools all
passes and both ubatch classes.

**Kaggle T4 run that would exercise them, once a GGUF exists:**

* Build with `GGML_CUDA=ON` targeting `sm_75` (T4) — note `BUILD_STATUS.md`
  records CPU-only (`GGML_CUDA=OFF`) builds so far.
* Supply a real `inkling2-loop.gguf` with 12 looped (core) layers, 128 experts,
  top-8, rank-64 gain/forget controllers, and `--moe-cache-size <bytes>` large
  enough to hold the union (`48` slots/group upstream).
* Run a fixed prompt/token count twice at batch **b1** and **b8**:
  `llama-cli -m <inkling2-loop.gguf> --moe-cache-size <bytes>` with RORET off, then
  `LLAMA_RORET=1 llama-cli ...` with RORET on. Capture `log_stats()` plus the
  new per-pass counters. Run the two CycleAudit negative controls before any
  comparison (`tools/audit/AUDIT_PROTOCOL.md` §6).
* Feed 50 recorded runs (5 warmup discarded), `loops=3`, `seed=0`, into
  `python -m loopkit.cli audit --claim X --baseline BASE.json --candidate CAND.json --loops 3`
  on the T4; publish only if `X <= ci_low` (§5). T4 has no FP4/FP8 tensor cores —
  NVFP4 here is a storage format, not a compute path.

---

## 4. Honest: what is NOT proven, and GO/NO_GO

Not proven by this report or any current artifact:

* **No GGUF.** No real Inkling2-loop model file was executed; the wiring is
  compile- and inspection-verified only (`docs/INKLING_UPSTREAM_WIRING.md`).
* **No GPU run.** All builds to date are `GGML_CUDA=OFF` on Kaggle CPU
  (`BUILD_STATUS.md`). No T4 timing exists.
* **No `tok/s`, no wall-clock speedup.** The 51.7–65.6% figure is a reduction in
  *upload counts*, not time. `AUDIT_PROTOCOL.md` §8–§9 explicitly forbids
  treating a load-count reduction as wall-clock.
* **Bit-exact parity was NOT measured at that point.** No `memcmp` of
  `ffn_moe_topk`, `ffn_moe_weights` or `ffn_moe_out`, and no logits comparison,
  had been performed; `tools/roret/BITEXACT_REPORT.md` was NO_GO then. M18 later
  measured raw F32 final-logits parity PASS-scoped; the per-pass tensors above
  remain unmeasured.
* **Cross-layer LRU eviction.** Protected hints apply only to the layer being
  planned; another layer's `plan()` in the shared bank can still evict a resident
  union tile. The union is re-applied every pass, but survival across intervening
  layers is not guaranteed. §1's "resident across passes" is an assumption.
* **Synthetic gate.** The `msg_scale` knobs stand in for rank-r gain/forget
  controllers; real cross-loop messages derive from the same residual stream, so
  drift may be smaller — unverified. The `1.00` row (51.7%, union 11.59) is the
  honest stress-case lower bound and is a NO_GO even at the load-count level.
* **Byte figures are estimates** under an FP16, single-matrix assumption; real
  weights are NVFP4/E4M3 and full experts are up to 3× larger.

**GO / NO_GO for the next milestone:**

* **GO** — proceed to the next milestone: add the per-pass counters of §3 and
  run the T4 CycleAudit measurement once a GGUF exists. The load-count model is
  internally consistent and the three realistic scales (0.05/0.10/0.30) clear
  the union verdict.
* **NO_GO** — for any published performance claim, tok/s number, or "measured
  byte reduction" until a real T4 run validates `claim <= ci_low` with the
  negative controls passing. The 1.00 stress case is **NO_GO** on load count
  alone. STOP criterion 6 (`ENGINE_SPEC.md`, via `AUDIT_PROTOCOL.md` §8): if
  `ci_low < 1.0` vs llama.cpp Q4_K_M, report no advantage and stop.

This is a conditional GO: authorization covers instrumentation and measurement,
not publication of a speedup.

---

## 5. CycleAudit lower-bound rule (authority: `tools/audit/AUDIT_PROTOCOL.md`)

From `AUDIT_PROTOCOL.md` §5 (`verdict()` in `loopkit/src/loopkit/audit.py`):

```
if self.repeats < 20 or self.loops < 1:
    return False
return claim_speedup <= report.ci_low
```

* `ci_low` is the **2.5th percentile of 2000 bootstrap `b/c` ratios** (95%
  interval on the ratio); `speedup = base.median()/cand.median()` is informational
  only. The bound, not the median, is the gate.
* A claim is publishable only if `claim <= ci_low` **and** `repeats >= 20` **and**
  `loops >= 1`. Never round up to meet the bound. If `ci_low < 1.0` there is no
  claim regardless of the point estimate.
* Only Kaggle 2xT4 (`sm_75`) timings are eligible; local CPU numbers are plumbing
  validation, never speed evidence (§2, §8–§9). The CLI uses the constructor
  default `repeats=50` and ignores the JSON array length, so the 20-run floor is
  the producer's responsibility (§5 scope caveat).
* `AUDIT_PROTOCOL.md` §9 forbids rounding up claims, single warm runs, sub-20
  repeats, local CPU timings, treating load-count reduction as wall-clock, using
  `ci_high`, and labelling FP16 tensor-core output bit-exact. This report's §2
  byte figures are therefore explicitly **estimates**, and §1 explicitly a
  **load-count model**, so neither is submitted as a CycleAudit claim.

---

## Engine-side per-pass counters (CPU-test path, synthetic fixture)

This section records engine-side instrumentation added after sections 1-5. It
does not change the load-count model; it makes those quantities directly visible
in the engine.

The counters are opt-in: the per-pass table is populated only when LLAMA_RORET=1,
the same switch that creates the union-resident cache. With LLAMA_RORET unset the
bookkeeping path is inert. Each row is attributed to a (logical layer il, loop
pass) pair from the RORET binding, so the pooled log_stats() output no longer has
to mix all passes together.

A CPU-test build is used because Kaggle images have no nvcc, so a CUDA build is
not available there. tools/roret/pass_traffic.sh uses -DLLAMA_RORET_CPU_TEST=ON
and -DGGML_CUDA=OFF and runs the same synthetic looped GGUF twice, with
LLAMA_RORET unset and with LLAMA_RORET=1, then prints the table below.

Field meanings:

- n_hit: real selected experts already resident in the MoE cache for that pass;
  resident hints are not counted.
- upload_ops: number of coalesced host-to-buffer tensor uploads for that pass.
- uploaded_bytes: byte total of those uploads.
- residents: earlier-pass union size (union of passes 0..p-1, sampled before this
  pass's ids are added); 0 on pass 0, which has no prior passes. This is a counter
  only and does not affect the hint set handed to the planner.

These are load counts and byte counters. They are NOT HBM bandwidth, NOT
wall-clock time, NOT tok/s, and no speed claim is made from them. No CycleAudit
result is implied or reported here.

### Blocker: the CPU-test path does not invoke the copy callback

Measured on 2026-11-25 with fork `main`, a fresh `-DGGML_CUDA=OFF
-DLLAMA_RORET_CPU_TEST=ON` build and the synthetic GGUF (sha256 `cf72921a…`).
`LLAMA_RORET=1` constructs the cache and logs the hook, and the graph contains
`moe_cache.slot_map_cpu` lookups and `MUL_MAT_ID` nodes, but the scheduler never
copies the slot map between host and device buffers, so
`llama_moe_cache::copy()` is never invoked. Instrumenting the very first line of
`copy()` produced zero calls, and `log_stats()` reported an empty counter map.
The per-pass counters are therefore UNEXERCISED locally: this is absence of a
measurement, not a parity failure and not a speed claim.

`tools/roret/pass_traffic.sh` reports `VERDICT: NO_GO` in this case rather than
printing an empty table as a pass. Producing the table requires a CUDA build
where the slot-map copy fires; Kaggle images have no `nvcc`, so that GPU run is
out of reach in this environment. No T4 traffic measurement is claimed here.

<!-- PASS_TRAFFIC_TABLE -->

---

## Host-side replay of the per-pass traffic counters (M12)

The new ctest `roret_pass_traffic` (`tools/roret/test_roret_pass_traffic.cpp`)
drives the REAL `roret_traffic` aggregator from `src/llama-roret-traffic.h` and
the REAL `llama_roret_union_cache` from `src/llama-roret.h`, using the REAL seed-0
fixture ids in `tools/roret/union_census_ids.inc`, at HOST level. It replays the
bookkeeping order of `llama_moe_cache::copy()` for ip=6..17, pass=0..2,
token=0..3: pass 0 clears the layer, residents and the read-only hit_rate()
sample are taken before `add()`, then the pass ids are registered.

This does NOT exercise `llama_moe_cache::copy()`. It does NOT measure device
residency or bytes, and it is NOT engine-side traffic. The blocker above still
stands and the engine-side table placeholder above is still pending: the table
needs a CUDA build where the slot-map copy fires.

Field meanings in the table below:

- plans: four token events per (ip, pass) row.
- hits: integer count of earlier-pass top-k ids already in the union before
  `add()` (hit_rate() multiplied by top-k, then rounded to an exact integer).
- upload_ops: 8 minus hits, an UPPER BOUND of one upload per not-yet-resident
  id. No coalescing is modelled. These are replay-model proxy values, not the
  engine's coalesced upload count from `ggml_backend_tensor_set_async`: union
  membership and LRU slot residency are independent, so this column must not be
  read as an engine-side counter.
- uploaded_bytes: zero by construction; this replay has no device tensor.
- residents: sum over the four tokens of the earlier-pass union size sampled
  before `add()`; zero on pass 0.

Measured table from the test stdout (COUNTS only):

```
RORET host-side per-pass traffic replay (COUNTS only; no device traffic, no time)
 ip pass plans hits upload_ops uploaded_bytes residents
  6    0     4    0         32               0         0
  6    1     4    9         23               0        32
  6    2     4   21         11               0        55
  7    0     4    0         32               0         0
  7    1     4    8         24               0        32
  7    2     4   12         20               0        56
  8    0     4    0         32               0         0
  8    1     4    3         29               0        32
  8    2     4    8         24               0        61
  9    0     4    0         32               0         0
  9    1     4    4         28               0        32
  9    2     4   10         22               0        60
 10    0     4    0         32               0         0
 10    1     4    7         25               0        32
 10    2     4    7         25               0        57
 11    0     4    0         32               0         0
 11    1     4    7         25               0        32
 11    2     4    9         23               0        57
 12    0     4    0         32               0         0
 12    1     4    6         26               0        32
 12    2     4    5         27               0        58
 13    0     4    0         32               0         0
 13    1     4    7         25               0        32
 13    2     4    6         26               0        57
 14    0     4    0         32               0         0
 14    1     4    5         27               0        32
 14    2     4    9         23               0        59
 15    0     4    0         32               0         0
 15    1     4    1         31               0        32
 15    2     4    6         26               0        63
 16    0     4    0         32               0         0
 16    1     4    5         27               0        32
 16    2     4    8         24               0        59
 17    0     4    0         32               0         0
 17    1     4    4         28               0        32
 17    2     4    9         23               0        60
TOTAL plans=144 hits=176 upload_ops=976 uploaded_bytes=0 residents=1086 hits_plus_upload_ops=1152
TABLE_FNV1A64=f3eaaa28ac25472f
```

The hash is FNV-1a 64-bit over the printed table text (offset basis
14695981039346656037, prime 1099511628211) up to and including the TOTAL line,
excluding the hash line itself. It is a byte-comparison aid only, and the test
asserts it equals `f3eaaa28ac25472f` so a changed table fails ctest.

Assertions in the test hold: 36 rows, 4 plans per row, pass-0 hits and residents
zero, union monotone across passes per (layer, token), hits + upload_ops = 8 per
event and 32 per row, 144 plans, 1152 hits + upload_ops, residents=1086
independently summed from the pre-add union sizes, and no negative or overflow
values. Fresh `cmake -S tools/roret -B /tmp/opencode/m12-roret-build`
plus build plus `ctest --output-on-failure` gave 5/5 passing, including the new
test. Standalone compilation of the new source with `-std=c++17 -Werror` and
`-std=c++20 -Werror` (with `../../src` on the include path) passed; two runs of
the binary produced byte-identical stdout including the hash.

These are COUNTS only, with zero uploaded bytes by construction. NO speed claim
is made here. CycleAudit claim<=ci_low with repeats>=20 and loops=3 remains
mandatory for any published performance claim. The engine-side NO_GO above was
current at M12; the M18 section below supersedes it at the engine final-logits
level only.

## Engine-side RORET_PASS row on Kaggle 2xT4 (M18, 2026-11-28)

This supersedes the engine-side NO_GO statements in section 4 and the M12
counter section above at the engine final-logits level. It does not change the
NO_GO status for intermediate per-pass MoE tensors or real weights.

The older no-nvcc premise in section 3 was disproved in M14: Kaggle GPU images
ship nvcc 13.0.88. The M18 kernel `jprinceraj/inkengine-cuda-engine-parity`
built the CUDA fork with `-DGGML_CUDA_NO_VMM=ON` and ran the synthetic GGUF
with RORET off and on for `cpu_share = 0.0` and `1.0`. The paired `llama-cli`
logs produced one real `RORET_PASS` row per share on the `on` run:

| cpu_share | il | pass | plans | hits | upload_ops | uploaded_bytes | residents |
|---|---|---|---|---|---|---|---|
| 0.0 | 1 | 2 | 19 | 150 | 78 | 153600 | 812 |
| 1.0 | 1 | 2 | 19 | 150 | 78 | 153600 | 812 |

Evidence: `tools/roret/ENGINE_PARITY_REPORT.md` and
`tools/roret/kaggle-engine-parity/M18_SUMMARY.json`. These are engine COUNTS
from a single pass of a synthetic fixture, not a full per-pass traffic table
and not wall-clock. The 8.25-11.59 union target from the 100k-token table in section 1 is a
numeric-model probe on `msg_scale`, not a measured engine statistic (the 3k
table tops out at 11.58), so this
row neither confirms nor refutes it. NO speed claim; CycleAudit
`claim <= ci_low` with `repeats >= 20` and `loops = 3` stays mandatory.
