"""Metrics aggregator — collates per-cell driver output with GPU / CPU samples.

Runs alongside ``sweep.py``: spawn this in a separate shell before kicking
off the sweep, point it at the same ``data/<run_id>/`` directory, and it
will (a) tail ``nvidia-smi dmon`` into ``gpu.log``, (b) sample CPU
utilisation into ``cpu.csv``, and (c) merge them with the per-cell
``latency.csv`` files into a master ``cell_metrics.csv`` after the sweep
finishes.

Lightweight by design: no torch / openpi imports — just ``psutil`` +
subprocess. This keeps the metrics process minimal so it does not steal
CPU from the actual benchmark.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import signal
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("serving_benchmark.collect")


def start_nvidia_smi_dmon(output_log: str | Path) -> subprocess.Popen | None:
    """Background ``nvidia-smi dmon -s u`` writer. Returns None when nvidia-smi is missing."""
    if shutil.which("nvidia-smi") is None:
        logger.warning("nvidia-smi not found; skipping GPU sampling")
        return None
    out = Path(output_log)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w")
    return subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
        stdout=f, stderr=subprocess.DEVNULL,
    )


def sample_cpu_loop(output_csv: str | Path, stop_path: str | Path, interval_s: float = 1.0) -> None:
    """Append per-second CPU utilisation rows until ``stop_path`` exists."""
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed; skipping CPU sampling")
        return

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = Path(stop_path)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        n_cores = psutil.cpu_count(logical=True)
        w.writerow(["ts"] + [f"core{i}" for i in range(n_cores)])
        while not stop.exists():
            per_cpu = psutil.cpu_percent(interval=interval_s, percpu=True)
            w.writerow([time.time()] + per_cpu)
            f.flush()


def collate(run_id: str, data_root: str = "data") -> None:
    """Merge per-cell latency.csv files into a master ``cell_metrics.csv``."""
    root = Path(data_root) / run_id
    summary = root / "sweep_summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"sweep_summary.csv not found at {summary}")
    out = root / "cell_metrics.csv"
    with open(summary) as src, open(out, "w", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        writer.writerow(header + ["n_requests", "n_ok"])
        for row in reader:
            cell_id = row[0]
            latency = root / cell_id / "latency.csv"
            if not latency.exists():
                writer.writerow(row + ["0", "0"])
                continue
            n = 0
            n_ok = 0
            with open(latency) as lf:
                latrows = csv.reader(lf)
                next(latrows, None)
                for r in latrows:
                    n += 1
                    if len(r) >= 4 and r[3] == "ok":
                        n_ok += 1
            writer.writerow(row + [n, n_ok])
    logger.info("collated metrics: %s", out)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="Start GPU/CPU sampling (long-lived)")
    s.add_argument("--run-id", required=True)
    s.add_argument("--data-root", default="data")
    s.add_argument("--stop-file", default=".sb_stop")

    c = sub.add_parser("collate", help="Merge per-cell metrics after sweep")
    c.add_argument("--run-id", required=True)
    c.add_argument("--data-root", default="data")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.data_root) / args.run_id
    if args.cmd == "sample":
        root.mkdir(parents=True, exist_ok=True)
        stop = root / args.stop_file
        if stop.exists():
            stop.unlink()
        gpu_proc = start_nvidia_smi_dmon(root / "gpu.log")
        try:
            sample_cpu_loop(root / "cpu.csv", stop, interval_s=1.0)
        except KeyboardInterrupt:
            pass
        finally:
            if gpu_proc is not None:
                gpu_proc.send_signal(signal.SIGINT)
                try:
                    gpu_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    gpu_proc.kill()
    elif args.cmd == "collate":
        collate(args.run_id, args.data_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
