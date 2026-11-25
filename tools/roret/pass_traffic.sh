#!/usr/bin/env bash
# Issue #5 / M5: per-pass RORET load-count table on the synthetic looped GGUF.
# Runs the CPU-test build with LLAMA_RORET unset vs =1 and reports expert load
# counts and byte counters only. No wall-clock, no tok/s, no speed claim.
#
# KNOWN LIMIT: on a GGML_CUDA=OFF / LLAMA_RORET_CPU_TEST=ON build the scheduler
# never copies the slot map between host and device buffers, so the MoE cache
# copy callback (llama_moe_cache::copy) is never invoked and no per-pass rows can
# exist. This harness reports NO_GO in that case instead of a false PASS. A CUDA
# build with a real host-to-device slot-map copy is required for the table.
set -euo pipefail

ROOT=${ROOT:-.}
BIN=${BIN:-$ROOT/build/bin/llama-debug}
MODEL=${MODEL:-$ROOT/synthetic.gguf}
OUT=${OUT:-/tmp/roret-pass-traffic}
N=${N:-16}
MOE_MIB=${MOE_MIB:-1}
CPU_SHARE=${CPU_SHARE:-1.0}
NGL=${NGL:-0}

HOOK_LINE="RORET union-resident MoE bookkeeping enabled"
PROMPT="Hello my name is"
SEED=0

echo "LOAD COUNTS and byte counters only: not wall-clock, not tok/s, not a speed claim."

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: BIN is not executable: $BIN" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: MODEL not found: $MODEL" >&2
    exit 1
fi

mkdir -p "$OUT"
rm -f "$OUT/off.log" "$OUT/on.log" "$OUT/pass_lines.txt" "$OUT/pass_rows.txt"

run() {
    local tag=$1; shift
    "$@" "$BIN" -m "$MODEL" -p "$PROMPT" -s "$SEED" -n "$N" -t 1 -ngl "$NGL" \
        --moe-cache-mib "$MOE_MIB" --moe-cache-cpu-share "$CPU_SHARE" --temp 0 --verbose \
        > "$OUT/$tag.log" 2>&1
}

run off env -u LLAMA_RORET
run on env LLAMA_RORET=1

if grep -q "$HOOK_LINE" "$OUT/off.log"; then
    echo "ERROR: OFF log contains the RORET hook line" >&2
    exit 1
fi
if ! grep -q "$HOOK_LINE" "$OUT/on.log"; then
    echo "ERROR: ON log is missing the RORET hook line" >&2
    exit 1
fi

grep "RORET_PASS il=" "$OUT/on.log" > "$OUT/pass_lines.txt" || true
sed -n 's/.*RORET_PASS il=\([0-9-]*\) pass=\([0-9-]*\) plans=\([0-9]*\) hits=\([0-9]*\) upload_ops=\([0-9]*\) uploaded_bytes=\([0-9]*\) residents=\([0-9]*\).*/\1 \2 \3 \4 \5 \6 \7/p' \
    "$OUT/pass_lines.txt" > "$OUT/pass_rows.txt"

if [[ ! -s "$OUT/pass_rows.txt" ]]; then
    echo "VERDICT: NO_GO - no RORET_PASS rows. On this CPU-test build the scheduler"
    echo "never copies the slot map, so llama_moe_cache::copy is never invoked and"
    echo "the per-pass counters are unexercised. A CUDA build (host -> device slot-map"
    echo "copy) is required. No load count is reported; no speed claim is made."
    exit 0
fi

echo "| il | pass | plans | hits | upload_ops | uploaded_bytes | residents |"
echo "|---|------|-------|------|------------|----------------|-----------|"
while read -r il pass plans hits ops bytes residents; do
    echo "| $il | $pass | $plans | $hits | $ops | $bytes | $residents |"
done < "$OUT/pass_rows.txt"

echo "PASS: OFF log has no hook line, ON log has the hook and per-pass load counters."
