#!/usr/bin/env python3
"""Check byte parity of intermediate MoE tensors with RORET OFF vs ON.

CPU-test mode only. Synthetic fixture only.
This is NOT a GPU test and NOT a speed claim.
"""

import argparse
import filecmp
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HOOK_TEXT = "RORET union-resident MoE bookkeeping enabled"
TENSOR_FILTER = r"^ffn_moe_(logits|probs|topk|weights|out)-[0-9]+$"
TIMEOUT_S = 600


def slug(value):
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path):
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def version_string(llama_debug):
    try:
        proc = subprocess.run(
            [llama_debug, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = proc.stdout.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def read_manifest(path, problems):
    rows = {}
    if not path.is_file():
        problems.append("missing manifest: " + str(path))
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 12:
                problems.append("bad manifest row %s:%d" % (path, lineno))
                continue
            name, type_name = fields[0], fields[1]
            ne = tuple(fields[2:6])
            nb = tuple(fields[6:10])
            nbytes, filename = fields[10], fields[11]
            key = (name, filename)
            if key in rows:
                problems.append("duplicate manifest key %s" % (repr(key),))
                continue
            rows[key] = {"type": type_name, "ne": ne, "nb": nb, "nbytes": nbytes, "filename": filename}
    return rows


def run_phase(phase, share, args, out_dir, problems):
    dump_dir = out_dir / ("%s_%s" % (phase, share))
    shutil.rmtree(dump_dir, ignore_errors=True)
    env = os.environ.copy()
    env.pop("LLAMA_RORET", None)
    if phase == "on":
        env["LLAMA_RORET"] = "1"
    env["LLAMA_DUMP_TENSORS_DIR"] = str(dump_dir)

    tag = "%s_%s" % (phase, slug(share))
    stdout_path = out_dir / (tag + "_stdout.log")
    stderr_path = out_dir / (tag + "_stderr.log")
    cmd = [
        args.llama_debug,
        "-m", args.model,
        "-p", args.prompt,
        "-n", str(args.n),
        "-cmoe",
        "--moe-cache-mib", "16",
        "--moe-cache-cpu-share", share,
        "--verbose",
        "--tensor-filter", TENSOR_FILTER,
    ]
    exit_code = -1
    try:
        with open(stdout_path, "wb") as stdout_handle, open(stderr_path, "wb") as stderr_handle:
            proc = subprocess.run(cmd, stdout=stdout_handle, stderr=stderr_handle, env=env, timeout=TIMEOUT_S)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        problems.append("%s timed out" % phase)
    except OSError as exc:
        problems.append("%s failed to start: %s" % (phase, exc))
    if exit_code != 0:
        problems.append("%s exit %d" % (phase, exit_code))

    hook = HOOK_TEXT in read_text(stderr_path)
    return dump_dir, exit_code, hook


def compare_share(share, args, out_dir):
    problems = []
    record = {
        "cpu_share": share,
        "off_exit": -1,
        "on_exit": -1,
        "off_hook": False,
        "on_hook": False,
        "tensors": 0,
        "bytes_compared": 0,
        "differing_bytes": 0,
        "per_file_sha256": {},
    }

    dirs = {}
    for phase in ("off", "on"):
        dump_dir, exit_code, hook = run_phase(phase, share, args, out_dir, problems)
        dirs[phase] = dump_dir
        record[phase + "_exit"] = exit_code
        record[phase + "_hook"] = hook
    if record["off_hook"]:
        problems.append("hook ACTIVE in OFF")
    if not record["on_hook"]:
        problems.append("hook ABSENT in ON")

    rows = {}
    for phase in ("off", "on"):
        rows[phase] = read_manifest(dirs[phase] / "manifest.tsv", problems)

    off_keys = set(rows["off"])
    on_keys = set(rows["on"])
    if off_keys != on_keys:
        problems.append("manifest row sets differ: off_only=%d on_only=%d" % (len(off_keys - on_keys), len(on_keys - off_keys)))

    for phase in ("off", "on"):
        for key, row in rows[phase].items():
            bin_path = dirs[phase] / row["filename"]
            if not bin_path.is_file():
                problems.append("missing bin %s %s" % (phase, row["filename"]))
                continue
            hashes = record["per_file_sha256"].setdefault(row["filename"], {})
            hashes[phase] = sha256_file(bin_path)

    for key in sorted(off_keys & on_keys):
        off_row = rows["off"][key]
        on_row = rows["on"][key]
        record["tensors"] += 1
        for field in ("type", "ne", "nb", "nbytes"):
            if off_row[field] != on_row[field]:
                problems.append("manifest %s differs for %s" % (field, repr(key)))

        off_path = dirs["off"] / off_row["filename"]
        on_path = dirs["on"] / on_row["filename"]
        if not off_path.is_file() or not on_path.is_file():
            continue
        if filecmp.cmp(off_path, on_path, shallow=False):
            record["bytes_compared"] += os.path.getsize(off_path)
            continue

        problems.append("bytes differ for %s" % repr(key))
        off_bytes = off_path.read_bytes()
        on_bytes = on_path.read_bytes()
        total = max(len(off_bytes), len(on_bytes))
        record["bytes_compared"] += total
        record["differing_bytes"] += sum(
            1 for i in range(total)
            if i >= len(off_bytes) or i >= len(on_bytes) or off_bytes[i] != on_bytes[i]
        )

    return record, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llama-debug", required=True, help="path to llama-debug binary")
    parser.add_argument("--model", required=True, help="path to synthetic GGUF fixture")
    parser.add_argument("--out", required=True, help="output directory for dumps and summary")
    parser.add_argument("--prompt", default="hello world")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--cpu-share", nargs="+", default=["0.0", "1.0"])
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_problems = []
    records = []
    for share in args.cpu_share:
        record, problems = compare_share(share, args, out_dir)
        records.append(record)
        all_problems.extend(problems)

    verdict = "NO_GO"
    if records and not all_problems:
        if all(
            r["off_exit"] == 0 and r["on_exit"] == 0
            and not r["off_hook"] and r["on_hook"]
            and r["tensors"] > 0 and r["bytes_compared"] > 0
            and r["differing_bytes"] == 0
            for r in records
        ):
            verdict = "PASS"

    summary = {
        "fixture_sha256": sha256_file(args.model),
        "shares": records,
        "overall_verdict": verdict,
        "problems": all_problems,
    }
    version = version_string(args.llama_debug)
    if version:
        summary["llama_debug_version"] = version

    summary_path = out_dir / "M26_SUMMARY.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for record in records:
        print(
            "share=%s off_exit=%d on_exit=%d off_hook=%s on_hook=%d tensors=%d bytes=%d diff=%d"
            % (
                record["cpu_share"], record["off_exit"], record["on_exit"],
                record["off_hook"], record["on_hook"], record["tensors"],
                record["bytes_compared"], record["differing_bytes"],
            )
        )
    print("OVERALL: %s" % verdict)
    for problem in all_problems[:20]:
        print("FAIL: " + problem)
    if len(all_problems) > 20:
        print("FAIL: %d more (see %s)" % (len(all_problems) - 20, summary_path))
    print("summary: " + str(summary_path))

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
