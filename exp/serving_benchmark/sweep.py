"""Sweep automation — fans out driver runs across a grid of configurations.

Reads a YAML file declaring per-cell parameters and runs ``driver.run_driver``
on each cell. Per-cell data lands under ``data/<run_id>/<cell_id>/``; an
aggregate ``sweep_summary.csv`` row is appended after each cell completes.

Usage:
    python -m exp.serving_benchmark.sweep \
        --config configs/sparse_to_dense.yaml \
        --run-id 2026-05-23_baseline

YAML schema (minimal):
    mode: "sparse_to_dense" | "freq_sweep" | "yaml_density" | "batch_window"
    server:
      host: 127.0.0.1
      port: 8000
    duration_s: 30
    warmup_s: 2
    cells:
      - id: w1
        num_workers: 1
        request_hz: 2.0
        bundle_id: default
      - id: w2
        num_workers: 2
        ...
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import yaml

from exp.serving_benchmark.driver import DriverConfig, run_driver

logger = logging.getLogger("serving_benchmark.sweep")


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(p * (len(s) - 1))))
    return s[idx]


def run_sweep(config_yaml: str, run_id: str, data_root: str = "data") -> str:
    cfg = yaml.safe_load(Path(config_yaml).read_text())
    server = cfg.get("server", {})
    duration_s = float(cfg.get("duration_s", 30))
    warmup_s = float(cfg.get("warmup_s", 2))

    out_root = Path(data_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "sweep_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cell_id", "num_workers", "request_hz", "bundle_id",
            "throughput_rps", "p50_ms", "p95_ms", "p99_ms",
        ])
        for cell in cfg.get("cells", []):
            cell_id = cell["id"]
            cell_dir = out_root / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            dc = DriverConfig(
                host=server.get("host", "127.0.0.1"),
                port=int(server.get("port", 8000)),
                num_workers=int(cell["num_workers"]),
                request_hz=float(cell.get("request_hz", 0.0)),
                duration_s=duration_s,
                warmup_s=warmup_s,
                bundle_id=str(cell.get("bundle_id", "default")),
                latency_csv=str(cell_dir / "latency.csv"),
            )
            logger.info("sweep cell %s: num_workers=%d hz=%.2f",
                        cell_id, dc.num_workers, dc.request_hz)
            summary = run_driver(dc)
            lats = summary["latencies_ms"]
            w.writerow([
                cell_id, dc.num_workers, dc.request_hz, dc.bundle_id,
                f"{summary['throughput_rps']:.2f}",
                f"{_percentile(lats, 0.50):.2f}",
                f"{_percentile(lats, 0.95):.2f}",
                f"{_percentile(lats, 0.99):.2f}",
            ])
            f.flush()
    logger.info("sweep complete: %s", summary_path)
    return str(summary_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to sweep YAML")
    p.add_argument("--run-id", required=True)
    p.add_argument("--data-root", default="data")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_sweep(args.config, args.run_id, args.data_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
