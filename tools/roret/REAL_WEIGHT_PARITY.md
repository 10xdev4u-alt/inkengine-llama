# M28 - Real-weight parity readiness gate

## Blocker

No real pretrained Inkling2 weights are held (roadmap constraint 2). The gate
`tools/roret/check_real_weight_parity.py` therefore returns NO_GO locally by
design: it refuses to classify a frozen synthetic fixture as real and does not
run the engine when classification fails. This is a readiness gate, not a
measurement.

## Recipe for when a real Inkling2 GGUF is available

Read `general.architecture` plus `<arch>.loop.start` and `<arch>.loop.end` from
the GGUF. Build `--expect-names` as `ffn_moe_<kind>-<il>` for each graph layer
index `il` in the loop range inclusive, for kinds `logits`, `probs`, `topk`,
`weights`, `out`. The gate never guesses these names; the operator must supply
them from the model metadata. Example for loop range 1..2:

```sh
python3 tools/roret/check_real_weight_parity.py \
  --gguf /path/to/real-inkling2.gguf \
  --llama-debug /path/to/build/bin/llama-debug \
  --out /tmp/m28-real \
  --real-weights-attestation \
  --expect-names \
    ffn_moe_logits-1 ffn_moe_probs-1 ffn_moe_topk-1 ffn_moe_weights-1 ffn_moe_out-1 \
    ffn_moe_logits-2 ffn_moe_probs-2 ffn_moe_topk-2 ffn_moe_weights-2 ffn_moe_out-2 \
  --cpu-share 0.0 1.0
```

Replace the example names with the model's own loop range. Add `--ngl` and
`--threads` only when the engine run needs them; defaults keep the CPU command.

The gate classifies weights as real only when attestation is set, the file
exists, its sha256 is not one of the two frozen synthetic shas, its size is at
least 10000000 bytes, the architecture is `inkling2` or `inkling2_loop`, and
expert_count is at least 16. It then reuses the M26/M27 `compare_share`
harness. Required evidence for PASS: OFF/ON raw byte parity at `cpu_share` 0.0
and 1.0, both engine exits 0, RORET hook ABSENT in OFF and ACTIVE in ON,
tensors and bytes compared greater than zero, and zero differing bytes with no
harness problems. The verdict is written to
`<out>/M28_REAL_WEIGHT_SUMMARY.json`; exit is 0 only on PASS.

## Scope

Correctness scope only. NO SPEED CLAIM. No timing is measured here, and any
future number still requires CycleAudit `claim <= ci_low` with `repeats >= 20`
and `loops = 3`. This gate complements M18 (final logits), M27 (GPU
intermediate tensors) and M23 (per-pass traffic counts): it covers the missing
real-weight case of the intermediate raw-byte check, and it does not replace
the other three scopes.
