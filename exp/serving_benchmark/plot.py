"""Plotting helpers — generate throughput / latency / Pareto figures.

Reads ``data/<run_id>/sweep_summary.csv`` (produced by ``sweep.py``) plus
``data/<run_id>/gpu_microbench.csv`` (Mode 0 output) and writes PNGs under
``analysis/<run_id>/``.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger("serving_benchmark.plot")


def _read_summary(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_pareto(summary_csv: Path, out_png: Path, title: str = "throughput vs latency") -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    rows = _read_summary(summary_csv)
    if not rows:
        logger.warning("empty summary: %s", summary_csv)
        return
    xs = [float(r["throughput_rps"]) for r in rows]
    ys = [float(r["p95_ms"]) for r in rows]
    labels = [r["cell_id"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=60, alpha=0.75)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("throughput (req/s)")
    ax.set_ylabel("latency p95 (ms)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_png)


def plot_gpu_microbench(csv_path: Path, out_png: Path) -> None:
    """Plot Mode 0 GPU microbench: throughput / latency vs batch_size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Aggregate per batch_size: median latency.
    by_bs: dict[int, list[float]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            bs = int(row["batch_size"])
            by_bs.setdefault(bs, []).append(float(row["latency_ms"]))

    batch_sizes = sorted(by_bs)
    p50 = [sorted(by_bs[bs])[len(by_bs[bs]) // 2] for bs in batch_sizes]
    throughput = [bs * 1000.0 / p50[i] for i, bs in enumerate(batch_sizes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(batch_sizes, p50, marker="o")
    ax1.set_xlabel("batch_size")
    ax1.set_ylabel("p50 latency (ms)")
    ax1.set_title("Mode 0 — latency vs batch_size")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)

    ax2.plot(batch_sizes, throughput, marker="o", color="tab:orange")
    ax2.set_xlabel("batch_size")
    ax2.set_ylabel("throughput (req/s)")
    ax2.set_title("Mode 0 — throughput vs batch_size")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_png)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--data-root", default="exp/serving_benchmark/data")
    p.add_argument("--out-root", default="exp/serving_benchmark/analysis")
    p.add_argument("--mode", default="pareto", choices=["pareto", "gpu_microbench"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data = Path(args.data_root) / args.run_id
    out = Path(args.out_root) / args.run_id
    if args.mode == "pareto":
        plot_pareto(data / "sweep_summary.csv", out / "pareto.png")
    elif args.mode == "gpu_microbench":
        plot_gpu_microbench(data / "gpu_microbench.csv", out / "gpu_microbench.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
