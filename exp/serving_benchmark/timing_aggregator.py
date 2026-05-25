"""Aggregate SystemTimer CSVs from multiple concurrent worker connections.

Each WebSocket connection on the server runs its own InferenceInterceptor +
SystemTimer (line of work). On task end, SystemTimer dumps one CSV per
task to ``cache_config.timer.output_csv_dir`` containing rows
``timestamp,task_id,name,elapsed_ms[,gpu.util_pct,gpu.mem_used_mb,
cpu.proc_pct,cpu.rss_mb]``.

This script:

1. Globs every CSV under ``--csv-root`` (the server-side directory).
2. Tags each row with the source filename = a unique connection id.
3. Bins rows by ``timestamp`` (configurable window, default 5 s).
4. Per-window emits per-probe (``name``) summary: count,
   concurrent_workers (distinct task_ids in the window), mean/p50/p95
   latency, plus per-window mean GPU util + GPU mem + CPU proc%.

Usage::

    # On the client side after pulling the CSV directory back:
    python -m exp.serving_benchmark.timing_aggregator \\
        --csv-root /tmp/timing \\
        --window-s 5 \\
        --out timing_concurrent_breakdown.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean
from statistics import median


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(p * (len(s) - 1))))
    return s[idx]


def _read_csv(path: Path, conn_id: str) -> list[dict]:
    out = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["timestamp"])
                el = float(row["elapsed_ms"])
            except (KeyError, ValueError):
                continue
            entry = {
                "ts": ts,
                "conn_id": conn_id,
                "task_id": row.get("task_id", "0"),
                "name": row.get("name", "?"),
                "elapsed_ms": el,
            }
            for k, v in row.items():
                if k in ("timestamp", "task_id", "name", "elapsed_ms"):
                    continue
                try:
                    entry[k] = float(v) if v not in ("", None) else None
                except (TypeError, ValueError):
                    pass
            out.append(entry)
    return out


def _aggregate(rows: list[dict], window_s: float) -> list[dict]:
    # Bin by floor(ts / window_s).
    bins: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        win_idx = int(r["ts"] // window_s)
        bins[(win_idx, r["name"])].append(r)

    summary_rows = []
    for (win_idx, name), bucket in sorted(bins.items()):
        win_start = win_idx * window_s
        elapsed = [b["elapsed_ms"] for b in bucket]
        conn_ids = {b["conn_id"] for b in bucket}
        # Aggregate resource columns if present.
        gpu_util = [b.get("gpu.util_pct") for b in bucket if b.get("gpu.util_pct") is not None]
        gpu_mem = [b.get("gpu.mem_used_mb") for b in bucket if b.get("gpu.mem_used_mb") is not None]
        cpu_proc = [b.get("cpu.proc_pct") for b in bucket if b.get("cpu.proc_pct") is not None]
        cpu_rss = [b.get("cpu.rss_mb") for b in bucket if b.get("cpu.rss_mb") is not None]
        summary_rows.append({
            "window_start": win_start,
            "probe": name,
            "concurrent_workers": len(conn_ids),
            "count": len(elapsed),
            "mean_ms": mean(elapsed),
            "p50_ms": median(elapsed),
            "p95_ms": _percentile(elapsed, 0.95),
            "max_ms": max(elapsed),
            "gpu_util_mean": mean(gpu_util) if gpu_util else "",
            "gpu_mem_mb_mean": mean(gpu_mem) if gpu_mem else "",
            "cpu_proc_pct_mean": mean(cpu_proc) if cpu_proc else "",
            "cpu_rss_mb_mean": mean(cpu_rss) if cpu_rss else "",
        })
    return summary_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv-root", required=True, help="directory containing per-task SystemTimer CSV files")
    p.add_argument("--window-s", type=float, default=5.0,
                   help="aggregation window in seconds (default 5)")
    p.add_argument("--out", default="",
                   help="optional output CSV path (default: stdout)")
    args = p.parse_args()

    root = Path(args.csv_root)
    if not root.is_dir():
        raise SystemExit(f"--csv-root {root} not a directory")

    rows: list[dict] = []
    for csv_path in sorted(root.glob("**/*.csv")):
        conn_id = csv_path.stem
        rows.extend(_read_csv(csv_path, conn_id))

    if not rows:
        raise SystemExit(f"no CSV rows under {root}")

    summary = _aggregate(rows, args.window_s)
    headers = [
        "window_start", "probe", "concurrent_workers", "count",
        "mean_ms", "p50_ms", "p95_ms", "max_ms",
        "gpu_util_mean", "gpu_mem_mb_mean", "cpu_proc_pct_mean", "cpu_rss_mb_mean",
    ]
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in summary:
                w.writerow(r)
        print(f"wrote {len(summary)} rows to {args.out}")
    else:
        w = csv.DictWriter(__import__("sys").stdout, fieldnames=headers)
        w.writeheader()
        for r in summary:
            w.writerow(r)


if __name__ == "__main__":
    main()
