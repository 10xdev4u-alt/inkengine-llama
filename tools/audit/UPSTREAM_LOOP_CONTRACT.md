# M29: Upstream Inkling2-loop contract parity check

This closes the first cheap-verification experiment in section 4 item 1 of
[INKLING_INTEGRATION.md](../../loopkit/docs/INKLING_INTEGRATION.md).
It is a contract parity check only. It does not measure anything and makes no speed claim.

## What was checked

The frozen facts live in [UPSTREAM_LOOP_CONTRACT.json](UPSTREAM_LOOP_CONTRACT.json)
(frozen on 2026-11-28). The checker
[check_upstream_loop_contract.py](check_upstream_loop_contract.py) re-extracts the same
facts from the fork's `gguf-py/gguf/constants.py` and `src/models/inkling2-loop.cpp`,
then compares them with the frozen facts:

- `MODEL_ARCH.INKLING2_LOOP` arch string: `inkling2_loop`.
- The `MODEL_TENSORS[MODEL_ARCH.INKLING2_LOOP]` tensor-name list.
- The five loop metadata keys read by the loop model.
- The cache-slot formula.
- The fork's frozen arch integer, 179, as a hard gate.
- Five additive RORET wiring markers, all required to be present.

Upstream provenance is recorded as two raw-file URLs and two sha256 digests:
`d904f1dc06317f950c51f82425593790e6bc0f6179b9f6c7b2861ba4c8420bfb` for
`constants.py` and `b5983c6ef955b54d35187933fb0295988997385a12d608c762eb67d06b473772`
for `src/models/inkling2-loop.cpp`. The checker runs offline from the frozen JSON;
it does not need the snapshot files in `/tmp`.

## Commands and exact output

From the repo root:

    PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest -q tools/audit/test_upstream_loop_contract.py
    ...................                                                      [100%]
    19 passed in 0.09s

    python3 tools/audit/check_upstream_loop_contract.py
    verdict: PASS
      arch_string                PASS
      fork_arch_int              PASS
      loop_kv_keys               PASS
      slot_formula               PASS
      tensor_names               PASS
      wiring_build_lookup_key    PASS
      wiring_defaulted_params    PASS
      wiring_forwarding          PASS
      wiring_gate                PASS
      wiring_logical_key_loop    PASS
      INFO: frozen fork arch int 179 vs frozen upstream arch int 181: benign upstream enum drift; gate keys on the fork value only
    note: contract parity only, no timing

Exit code was 0. The generated
[M29_UPSTREAM_CONTRACT_SUMMARY.json](M29_UPSTREAM_CONTRACT_SUMMARY.json) has
`verdict: PASS`, all ten checks PASS, one INFO line, `speed_claim: null`, and
`note: "contract parity only, no timing"`.

Regression run of the existing audit suite:

    PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest -q tools/audit
    .................................................................        [100%]
    65 passed in 12.48s

## Honest findings

- The `MODEL_TENSORS[INKLING2_LOOP]` tensor list is text-identical between the fork
  and the frozen upstream snapshot. Programmatic extraction yields 47 tensor names
  in each. The milestone brief said 49; the extracted source is authoritative and
  the frozen JSON records the actual 47 entries.
- The five loop metadata keys are identical in both files:
  `LLM_KV_LOOP_START`, `LLM_KV_LOOP_END`, `LLM_KV_LOOP_COUNT`,
  `LLM_KV_LOOP_GATE_RANK`, `LLM_KV_LOOP_VARYING`.
- The slot formula is identical in both files:
  `n_phys + (size_t) (n_passes - 1)*(core_end - core_start)`.
- The arch string is identical: `inkling2_loop`.
- The fork's `MODEL_ARCH.INKLING2_LOOP` integer is 179; upstream master is 181.
  Upstream added two enum members after the fork base. This is benign version
  drift, recorded as INFO. The gate fails on a change to the fork's value, not
  on upstream advancement.
- The whole loop `.cpp` files are not identical, and that is expected. The fork
  carries additive, default-off union-resident MoE (RORET) wiring; upstream also
  has later changes the fork does not carry, including DSpark visit tracking and
  `TENSOR_NO_SCALE`. Whole-file equality is not part of this contract.
- All five RORET wiring markers are present. The gate is default-off:
  `LLAMA_RORET` must be enabled explicitly, and the new `build_layer` parameters
  default to -1.

Fail-closed negative controls in
[test_upstream_loop_contract.py](test_upstream_loop_contract.py) mutate the tensor
list, the loop keys, the fork arch integer, the arch string, the slot formula,
and a wiring marker; each yields NO_GO. Upstream arch-integer drift alone stays
PASS with INFO.

## Scope limits

No weights, no model downloads, no Kaggle, no graph execution, no timing, and no
speedup or performance claim of any kind. This checks source-level contract facts
only. It cannot prove that a real GGUF loads or that any RORET path is faster.

## How to rerun

Offline, from the repo root:

    python3 tools/audit/check_upstream_loop_contract.py
    PYTHONPATH=../loopkit/src:gguf-py python3 -m pytest -q tools/audit/test_upstream_loop_contract.py

With a network connection, to compare against live upstream master in memory
without rewriting the frozen JSON:

    python3 tools/audit/check_upstream_loop_contract.py --refresh

`--refresh` replaces the upstream-derived facts in memory and prints/updates the
same summary file. A failed fetch is a NO_GO. A changed upstream arch integer is
INFO only; a changed arch string, tensor list, loop keys, or slot formula is a
NO_GO.
