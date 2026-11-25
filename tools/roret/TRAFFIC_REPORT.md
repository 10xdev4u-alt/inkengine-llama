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
* **Bit-exact parity is NOT measured.** No `memcmp` of `ffn_moe_topk`,
  `ffn_moe_weights` or `ffn_moe_out`, and no logits comparison, was performed;
  `tools/roret/BITEXACT_REPORT.md` remains NO_GO.
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
