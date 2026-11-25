# Issue #4 — synthetic-GGUF parity audit (bounded experiment)

**Decision: NO_GO for union-resident wiring parity.** A loadable, fully synthetic
GGUF was produced and a CPU off/on comparison was run with 0 differing bytes,
but the RORET union code path is not constructed in a CPU-only build. The
measured equality is therefore **not** evidence that union-resident wiring is
bit-exact. No real weights or checkpoint downloads were used. No `src/` file or
existing test was modified.

## Synthetic GGUF (feasibility: YES)

`tools/audit/make_synthetic_gguf.py` emits a deterministic F32 GGUF using the
upstream `gguf-py` writer API. Seed 0, sha256
`cf72921a76518bff78f969227be8b036a9dc869ca8c0eb5e78a66c89a81e9617`,
887392 bytes, 65 tensors, 45 metadata keys. Regeneration at the same seed is
byte-identical. The tokenizer is the built-in `test` type (hashed token ids),
so no 256-entry byte table or BPE merges are needed.

Minimal metadata (exact keys read by the loader):

| Key | Value | Reader |
|---|---|---|
| `general.architecture` | `inkling2_loop` | `llama_model_loader` |
| `inkling2_loop.context_length` | 64 | `llama_model::load_hparams` |
| `inkling2_loop.embedding_length` | 32 | same |
| `inkling2_loop.block_count` | 2 | same |
| `inkling2_loop.vocab_size` / `vocab_size_unpadded` | 16 / 16 | `llama_model_inkling2::load_arch_hparams` |
| `inkling2_loop.expert_count` / `expert_used_count` | 128 / 8 | `llama_model::load_hparams` |
| `inkling2_loop.expert_feed_forward_length` | 8 | `llama_model_inkling2::load_arch_hparams` |
| `inkling2_loop.attention.head_count` / `head_count_kv` | 2 / 1 | `llama_model::load_hparams` |
| `inkling2_loop.attention.key_length_swa` / `value_length_swa` | 16 / 16 | same |
| `inkling2_loop.rope.dimension_count` | 0 | same |
| `inkling2_loop.attention.layer_norm_rms_epsilon` / `qk_clip` | 1e-5 / 10 | `llama_model_inkling2::load_arch_hparams` |
| `inkling2_loop.attn_res.block_size` | 1 | same |
| `inkling2_loop.shortconv.l_cache` | 2 | same |
| `inkling2_loop.attention.relative_length` / `relative_extent` / `relative_clip` | 4 / 8 / 10 | same |
| `inkling2_loop.attention.log_scale_floor` / `log_scale_alpha` | 1 / 0.1 | same |
| `inkling2_loop.attention.sliding_window` / `sliding_window_pattern` | 8 / [false, true] | same |
| `inkling2_loop.activation.situ_beta` / `situ_linear_beta` | 1 / 1 | same |
| `inkling2_loop.expert_weights_scale` / `final_logit_softcapping` | 1 / 0 | same |
| `inkling2_loop.loop.start` / `end` / `count` / `gate_rank` / `varying` | 1 / 2 / 3 / 8 / false | `llama_model_inkling2_loop::load_arch_hparams` |
| `tokenizer.ggml.model` | `test` | `llama_vocab` |
| `tokenizer.ggml.tokens` / `token_type` / `bos_token_id` / `eos_token_id` | 16 / 16 / 0 / 1 | `llama_vocab` |

Two details mattered when making the GGUF actually runnable:

* Layer 0 must be full attention (`sliding_window_pattern[0] = false`). With all
  layers SWA, `tau` is unused by the graph, `inp_pos` is pruned and never
  allocated, and `llm_graph_result::set_inputs` aborts at
  `ggml-backend.cpp:534` ("tensor buffer not set").
* The tokenizer id keys must be `uint32`, not `int32`, or the vocab loader
  rejects the file.

Minimal tensor set: `token_embd`, `token_embd_norm`, `output`, `output_norm`,
`output_res_query`; per physical layer the attention tensors (with the
full-attention input norms and conv projections on layer 0 and `attn_norm` on
layer 1), `ffn_norm`, `ffn_gate_inp`, `ffn_gate_up_exps`, `ffn_down_exps`,
`ffn_post_conv`, `ffn_res_query` (layer 1), optional `exp_probs_b`; and on core
layer 1 the 13 loop tensors (`loop_msg_query` and `q/k/v/ffn` ×
`down/gain/forget`). Shapes come from `llama_model_inkling2::load_arch_tensors`
and `llama_model_inkling2_loop::load_arch_tensors`. Expert count stays at the
real 128 with top-8; only intermediate dims are reduced.

## Runtime comparison (CPU-only build)

Build:

```sh
cmake -S /tmp/build/inkengine-llama -B /tmp/ink-parity \
  -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build /tmp/ink-parity --target llama-cli -j4
cmake --build /tmp/ink-parity --target llama-debug -j4
```

Harness: `tools/audit/parity_run.sh` runs the same prompt (`Hello my name is`),
seed 0, `-n 16`, `-t 1`, `-ngl 0`, `--temp 0` twice: `LLAMA_RORET` unset vs
`LLAMA_RORET=1`, comparing `-o` text and `llama-debug --save-logits` raw F32
logits byte-for-byte. Outputs are under `/tmp/ink-parity/parity-out/`.

Measured result (2026-11-25, both exits 0):

| Artifact | off | on | Result |
|---|---|---|---|
| generated text (53 B) | `cf915b84cf60abfd20ced6ac8ec7e69a72c2f0c4b0e2f6abbe8c9c9130bbec3d` | same | **0 differing bytes** |
| raw F32 logits (64 B = 16 vocab floats) | `78d3eea22c090e0abca7b494adb37d4fa11af1ce412cbd388989a2b4b9065685` | same | **0 differing bytes** |
| token id file (16 B) | `31760d32cbb5458f74281f26cf8c4308e79b7db3c48a5285296904e79c61dee8` | same | **0 differing bytes** |
| `RORET union-resident MoE bookkeeping enabled` in log | absent | **absent** | hook never enabled |

This is a genuine byte comparison of engine output on a 3-pass `inkling2_loop`
graph, but it is a comparison of two runs in which the RORET hook is inactive.

## Why this is NOT union-wiring parity

1. `LLAMA_RORET` is read only in `llama_moe_cache::impl::impl`
   (`src/llama-moe-cache.cpp:449-455`). The union object, slot-map binding and
   copy-callback registration live only inside `llama_moe_cache`
   (`src/llama-moe-cache.cpp:871-885, 985-987`).
2. `llama_moe_cache` is constructed only when `moe_cache_size > 0`
   (`src/llama-context.cpp:618-620`), and its constructor requires at least one
   GPU/iGPU backend, throwing otherwise (`src/llama-moe-cache.cpp:457-467`).
   The requested build is `GGML_CUDA=OFF`; this machine has no `nvidia-smi`, no
   CUDA toolkit and no Vulkan headers, so no GPU backend exists. `--moe-cache-mib`
   cannot be used, and the RORET object is never created in either run — hence
   `RORET_HOOK_ABSENT` in both logs.
3. `llama_model_inkling2::graph::build_layer` passes `roret_cache = nullptr`,
   `roret_selected_ids = nullptr` and `roret_n_selected = 0` to `build_moe_ffn`
   (`src/models/inkling2.cpp:410-414`), so the graph-side hook at
   `src/llama-graph.cpp:2500-2510` is also inert in both runs.

Consequently, 0 differing bytes here proves only that an inert environment
variable does not change output. It does not exercise host-observed top-k
readback, `roret_of` slot binding, the protected `plan()` path, cross-pass
union residency, or per-pass `ffn_moe_*` tensors.

## Static gate check

`LLAMA_RORET` is default-off: the code enables it only when the variable is
exactly `1`, `on` or `true` (`src/llama-moe-cache.cpp:449-455`); unset or any
other value leaves `roret == nullptr`. This is inspection, not runtime
measurement of the union path.

## What would close issue #4

A GPU-enabled build plus the synthetic GGUF above, run twice with identical
prompt/seed/`-n`/`-ngl`/`--moe-cache-mib`, `LLAMA_RORET` unset vs `=1`,
comparing raw F32 logits byte-for-byte and confirming the info log appears
only in the enabled run. Per-pass `ffn_moe_*` intermediate parity additionally
needs a raw-byte dump hook; the normal debug callback prints only `%12.4f`.
