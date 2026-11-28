# M35 d2027 outlier-premise gate

Offline, fail-closed classification of the "activation outlier at dimension
2027" premise. This is a metadata/contract result only. It is not a correctness
measurement, a measured activation trace, a timing result, or a speed claim.

## What was checked

- Three frozen primary-source records were fetched once out of band, before
  this gate. The checker makes no network calls and does not import loopkit.
  Each record stores url, byte count, sha256, and `occurrences_2027 = 0`. The
  small 2722-byte `config.json` is embedded verbatim in the facts and its byte
  count, sha256 and zero-hit count are recomputed by the offline checker. The
  two HTML records are out-of-band assertions: the offline checker cannot
  re-fetch them and cannot recompute their digests.
- `config.json` reports `hidden_size = 2304`, 24 layers, and
  `model_type = inkling2_loop`. The hidden size makes index 2027 plausible but
  is INFO only, not evidence for the premise.
- Six local files whose sha256 values are frozen are checked live: the loopkit
  config, model module and integration doc, the fork README, and the two ODSL
  docs. Needles assert that each file labels d2027 an unverified local
  hypothesis, that the roadmap item 4 says the trace needs weights or hosted
  inference, and that the side-lane is lower-drift but NOT bit-exact and is
  behind `LLAMA_ODSL` / `LLAMA_ODSL_SIDE_LANE`, both default off.
- The engine source inventory scans `src/`, `common/` and `ggml/`. Bare `2027`
  is checked only in loop-specific files (path or Inkling2/RORET content
  markers), so unrelated hits such as `0x2027`, `2027.0` and the erf constant
  `-1.453152027f` are not reported as d2027 evidence. Symbols
  `outlier_dim`, `d2027`, `LLAMA_ODSL`, `ODSL_LAYOUT`, `outlier_drift` and
  `side_lane` are forbidden anywhere in that engine scope.

## Frozen provenance

| Source | Bytes | sha256 | "2027" hits |
| --- | --- | --- | --- |
| TML release post `https://thinkingmachines.ai/news/inkling-2-mini/` | 448727 | `1301b66c36ff9112dd9c731cc5c83eba1e6b6c401c6364301fbc933d40f7249c` | 0 |
| TML model card `https://thinkingmachines.ai/model-card/inkling-2-mini/` | 222597 | `2fd49bae2efdbef1914314280ebc2d93742f89e903f66354c30b02b95f5efe28` | 0 |
| HF `.../raw/main/config.json` | 2722 | `a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2` | 0 |

Frozen on 2026-11-28. Facts: `tools/audit/OUTLIER_PREMISE.json`. Generated
result: `tools/audit/M35_OUTLIER_PREMISE_SUMMARY.json`.

## PASS scope and blocker

Checker exit 0, verdict PASS, blocker empty, 27/27 boolean checks true,
`classification = LOCAL_ONLY_UNVERIFIED`, `real_trace_verdict = NO_GO_BLOCKED`,
`side_lane_enabled = false`, `speed_claim = null`, and note "outlier premise
classification only; no timing". The engine inventory found 24 loop-specific
files and no d2027 symbol, no scoped 2027 literal, and no ODSL side-lane
enable. The speculative FP32 side-lane is NOT enabled in the fork engine path.

The falsification half stays NO_GO_BLOCKED: recording the max-|activation|
dimension with a forward hook on a real prompt needs real weights or hosted
inference, and neither is available. This gate records that blocker honestly;
it does not claim the premise was observed or disproved.

NO SPEED CLAIM. Any future number must satisfy CycleAudit `claim <= ci_low`,
`repeats >= 20`, `loops = 3`. Counts, drift ratios and this classification are
not speedups.
