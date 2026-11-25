#!/usr/bin/env bash
# Issue #4 parity harness: same synthetic GGUF, same prompt/seed, LLAMA_RORET
# unset vs =1. Compares generated text bytes and (when llama-debug is available)
# raw F32 logits bytes. No downloads, no real weights, no src/ changes.
set -euo pipefail

ROOT=${ROOT:-/tmp/build/inkengine-llama}
BUILD=${BUILD:-/tmp/ink-parity}
MODEL=${MODEL:-/tmp/ink-parity/synthetic.gguf}
OUT=${OUT:-/tmp/ink-parity/parity-out}
PROMPT=${PROMPT:-Hello my name is}
SEED=${SEED:-0}
N=${N:-16}
NGL=${NGL:-0}
MOE_MIB=${MOE_MIB:-0}
CPU_SHARE=${CPU_SHARE:-1.0}

CLI="$BUILD/bin/llama-cli"
DBG="$BUILD/bin/llama-debug"

mkdir -p "$OUT"
rm -f "$OUT"/*.txt "$OUT"/*.log "$OUT"/*.bin "$OUT"/*.sha256

run_cli() {
    local tag=$1; shift
    "$@" "$CLI" -m "$MODEL" -p "$PROMPT" -s "$SEED" -n "$N" -t 1 -ngl "$NGL" \
        --moe-cache-mib "$MOE_MIB" --moe-cache-cpu-share "$CPU_SHARE" --temp 0 --verbose -o "$OUT/$tag.txt" > "$OUT/$tag.log" 2>&1
}

echo "=== llama-cli: LLAMA_RORET unset ==="
run_cli off env -u LLAMA_RORET
echo "=== llama-cli: LLAMA_RORET=1 ==="
run_cli on env LLAMA_RORET=1

echo "=== compare generated text ==="
sha256sum "$OUT/off.txt" "$OUT/on.txt" | tee "$OUT/text.sha256"
if cmp -s "$OUT/off.txt" "$OUT/on.txt"; then
    echo "TEXT_PARITY: 0 differing bytes"
else
    echo "TEXT_PARITY: MISMATCH"
    cmp "$OUT/off.txt" "$OUT/on.txt" || true
    exit 1
fi

echo "=== RORET info log presence ==="
for tag in off on; do
    if grep -q "RORET union-resident MoE bookkeeping enabled" "$OUT/$tag.log"; then
        echo "$tag: RORET_HOOK_ACTIVE"
    else
        echo "$tag: RORET_HOOK_ABSENT"
    fi
done

if [[ -x "$DBG" ]]; then
    echo "=== llama-debug: raw F32 logits ==="
    for tag in off on; do
        mkdir -p "$OUT/$tag-logits"
        rm -f "$OUT/$tag-logits"/*.bin
        if [[ $tag == off ]]; then
            env -u LLAMA_RORET "$DBG" -m "$MODEL" -p "$PROMPT" -s "$SEED" -n "$N" \
                -t 1 -ngl "$NGL" --moe-cache-mib "$MOE_MIB" --temp 0 --save-logits \
                --logits-output-dir "$OUT/$tag-logits" > "$OUT/$tag-debug.log" 2>&1
        else
            env LLAMA_RORET=1 "$DBG" -m "$MODEL" -p "$PROMPT" -s "$SEED" -n "$N" \
                -t 1 -ngl "$NGL" --moe-cache-mib "$MOE_MIB" --temp 0 --save-logits \
                --logits-output-dir "$OUT/$tag-logits" > "$OUT/$tag-debug.log" 2>&1
        fi
    done
    sha256sum "$OUT"/off-logits/*.bin "$OUT"/on-logits/*.bin | tee "$OUT/logits.sha256"
    ok=1
    for f in "$OUT"/off-logits/*.bin; do
        base=$(basename "$f")
        if ! cmp -s "$f" "$OUT/on-logits/$base"; then
            echo "LOGITS_PARITY: MISMATCH ($base)"
            ok=0
        fi
    done
    if [[ $ok -eq 1 ]]; then
        echo "LOGITS_PARITY: 0 differing bytes"
    else
        exit 1
    fi
else
    echo "LOGITS_PARITY: SKIPPED (llama-debug not built)"
fi

echo "=== done ==="
