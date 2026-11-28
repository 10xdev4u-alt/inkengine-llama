# M38 integration-doc closure gate (scope only)

This gate closes the `[?]` markers in `loopkit/docs/INKLING_INTEGRATION.md` against
the M29-M37 plus M44 and M47 audit artifacts; M48 closes the section-3 item (e)
marker in the loopkit doc. It is a documentation-contract result only.

Old doc sha256: `69c587aba7ebbe3b866b7ca95107cfb891b28b3b30d3c54eeba698f878655432`.
New doc sha256: `95f7595401bf534fa7e3d007423979e7246151678eb67655cd35d9f3660d07d3`.

## Frozen contract

`INTEGRATION_DOC_CLOSURE.json` pins the new doc hash, the checker hash, nine
resolved-marker needle groups (section-2 GGUF shapes, section-2 cache hook, section-3
items (d) and (e), and section-4 items 1, 2, 3, 5 and 6), one honest-blocker group
(section-4 item 4), the section-5 closure note, and the eleven milestone artifact
pairs M29-M37 plus M44 and M47. `frozen_on` is 2026-11-28.

## Checker

`check_integration_doc_closure.py` is stdlib-only and offline. It accepts an
optional `--root`. It writes a NO_GO sentinel to
`M38_INTEGRATION_DOC_CLOSURE_SUMMARY.json` before any check, then writes the final
summary atomically with `os.replace`. Any failure leaves NO_GO, never a stale PASS.
It gates the exact contract key set, the live doc hash, the live checker hash,
every needle, the evidence pointers, every cited summary verdict and count, the
byte-pinned M44 and M47 summary fields, the M35 classification/blocked verdict, the M37
history note, and the summary shape.
No network import or fetch call is present; this is enforced by a static AST check.

## Scope limits

- No weights are held, fetched or loaded.
- No engine run, no graph execution, no GPU kernel and no Kaggle kernel.
- No tensor, per-pass MoE or logits parity is measured; no hosted inference.
- No timing, no wall-clock number and NO SPEED CLAIM.
- CycleAudit `claim<=ci_low`, `repeats>=20`, `loops=3` remains mandatory before
  any future speed number.

The d2027 premise stays LOCAL_ONLY_UNVERIFIED and its real-trace falsification
stays NO_GO_BLOCKED. The M37 summary's "marker pending" `doc_note` is preserved
as history; M38 supersedes it in `BUILD_STATUS.md`.
