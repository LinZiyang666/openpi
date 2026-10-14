"""Aggregate native supply under MPS; mixed X = 1/(h/X_hit + (1-h)/X_call)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


def shares_arg(value):
    shares = [float(v) for v in value.replace(",", " ").split()]
    if not shares or any(not math.isfinite(h) or not 0 <= h <= 1 for h in shares):
        raise argparse.ArgumentTypeError("hit shares must be finite numbers in 0..1")
    return shares


def summarize(directory):
    samples = []
    with (directory / "smi.csv").open() as stream:
        for row in csv.reader(stream):
            try:
                timestamp = datetime.fromisoformat(row[0].strip().replace("/", "-")).timestamp()
                samples.append((timestamp, float(row[1]), float(row[2]) / 1024))
            except (ValueError, IndexError):  # Header or unsupported nvidia-smi fields.
                continue
    groups = defaultdict(list)
    for path in sorted(directory.glob("R*/replica*.jsonl")):
        replicas = int(path.parent.name[1:])
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups[(row["model"], row["steps"], replicas, row["B"], row["phase"])].append(row)
    if not groups:
        raise ValueError("no replica phase rows found")
    summaries = {}
    for key, rows in sorted(groups.items()):
        if len(rows) != key[2] or len({r["pid"] for r in rows}) != key[2]:
            raise ValueError(f"incomplete or duplicate replica rows for {key}")
        # Account for late starts and the full completion time of the last batch.
        begin = min(r["begin"] for r in rows)
        end = max(r["begin"] + r["wall_s"] for r in rows)
        requests = sum(r["requests"] for r in rows)
        # Sample only where every replica is in this slot, excluding drain/gap.
        common_begin = max(r["begin"] for r in rows)
        common_end = min(r["begin"] + r["wall_s"] for r in rows)
        common_end = min(common_end, min(r["scheduled_begin"] + r["phase_s"] for r in rows))
        measured = [s for s in samples if common_begin <= s[0] < common_end]
        memories = [r["mem_gib"] for r in rows]
        summaries[key] = dict(x=requests / (end - begin) if end > begin else 0,
                              util=statistics.mean(s[1] for s in measured) if measured else None,
                              gpu_mem=max(s[2] for s in measured) if measured else None,
                              samples=len(measured), memory=memories)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--hit-shares", type=shares_arg, default=shares_arg("0.82,0.5,0"))
    args = parser.parse_args()
    summaries = summarize(args.directory)
    print("model    K   R   B phase    req/s   GPU% samples GPU-GiB  reserved GiB/proc min/mean/max")
    for (model, steps, replicas, size, phase), row in summaries.items():
        util = f'{row["util"]:.1f}' if row["util"] is not None else "n/a"
        gpu_mem = f'{row["gpu_mem"]:.2f}' if row["gpu_mem"] is not None else "n/a"
        memory = row["memory"]
        print(f'{model:<8} {steps:3} {replicas:3} {size:3} {phase:<5} {row["x"]:8.2f} '
              f'{util:>6} {row["samples"]:7} {gpu_mem:>7}  '
              f'{min(memory):.2f}/{statistics.mean(memory):.2f}/{max(memory):.2f}')
    print("\nMixed req/s (request hit share h; harmonic estimate, same R/B/K)")
    for (model, steps, replicas, size, phase), row in summaries.items():
        if phase != "hit":
            continue
        call = summaries.get((model, steps, replicas, size, "call"))
        if call is None:
            raise ValueError(f"missing call phase for {(model, steps, replicas, size)}")
        for share in args.hit_shares:
            costs = [(share, row["x"]), (1 - share, call["x"])]
            mixed = 0 if any(weight and x <= 0 for weight, x in costs) else 1 / sum(weight / x for weight, x in costs if weight)
            print(f'{model} K={steps} R={replicas} B={size} h={share:g}: {mixed:.2f}')


if __name__ == "__main__":
    main()
