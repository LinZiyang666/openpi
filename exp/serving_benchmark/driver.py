"""Client driver — issues concurrent inference requests to a running server.

Spawns ``num_workers`` worker threads. Each worker maintains its own
WebSocket connection to the server and submits requests at a configurable
rate. Each request emits one CSV row in ``--latency-csv``.

Usage (called by ``sweep.py`` per cell):
    python -m exp.serving_benchmark.driver \
        --host 127.0.0.1 --port 8000 \
        --num-workers 4 --request-hz 2.0 --duration-s 30 \
        --bundle-id default \
        --latency-csv data/<run_id>/latency.csv

The driver is intentionally minimal: it reproduces the obs shape the
configured policy expects (currently LIBERO's pose+image+prompt shape)
with zero tensors, since we are timing end-to-end wire+forward latency
rather than measuring action correctness. Pair with ``collect.py`` for
GPU / CPU / cache-hit metrics that run alongside the driver.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import threading
import time

logger = logging.getLogger("serving_benchmark.driver")


@dataclass
class DriverConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    num_workers: int = 4
    request_hz: float = 2.0      # per-worker request rate; 0 = max-rate
    duration_s: float = 30.0
    bundle_id: str = "default"
    latency_csv: str = ""
    warmup_s: float = 2.0


def _make_dummy_obs() -> dict:
    """Construct a minimal LIBERO-shaped obs dict (zeros). Server only times
    the round-trip; values are irrelevant. Keys must match
    ``examples/libero/main.py`` so the data transforms find them."""
    import numpy as np

    return {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "benchmark dummy",
    }


def _worker_loop(
    worker_id: int,
    cfg: DriverConfig,
    stop_event: threading.Event,
    rows: list,
    rows_lock: threading.Lock,
) -> None:
    """One worker: connect, optionally select_bundle, drive requests until stop."""
    # Local import so the module can be imported on machines without the
    # client lib installed (e.g. doc-generation hosts).
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(
        host=cfg.host, port=cfg.port,
    )
    # M2 BundleDispatcher: bind this worker's connection to ``cfg.bundle_id``.
    # Errors are surfaced to the caller (no swallow) so Mode 3 multi-bundle
    # sweeps fail loudly when a yaml hasn't been ``load_cache_config``'d.
    if cfg.bundle_id and cfg.bundle_id != "default":
        client.select_bundle(cfg.bundle_id)

    interval = 0.0 if cfg.request_hz <= 0 else 1.0 / cfg.request_hz
    next_send = time.monotonic()
    obs = _make_dummy_obs()
    while not stop_event.is_set():
        now = time.monotonic()
        if now < next_send:
            time.sleep(min(interval, next_send - now))
            continue
        t0 = time.perf_counter()
        try:
            client.infer(obs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            row = (worker_id, time.time(), latency_ms, "ok")
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            row = (worker_id, time.time(), latency_ms, f"err:{exc}")
        with rows_lock:
            rows.append(row)
        next_send = max(next_send + interval, now)


def run_driver(cfg: DriverConfig) -> dict:
    """Synchronous driver runner. Returns a small summary dict.

    Writes per-request latency rows to ``cfg.latency_csv`` (if set).
    Honors ``cfg.warmup_s`` by discarding requests that arrive before the
    warmup deadline.
    """
    rows: list = []
    rows_lock = threading.Lock()
    stop_event = threading.Event()

    start = time.monotonic()
    start_wall = time.time()  # epoch clock, to match per-row time.time() stamps
    end_deadline = start + cfg.warmup_s + cfg.duration_s

    threads = [
        threading.Thread(
            target=_worker_loop,
            args=(i, cfg, stop_event, rows, rows_lock),
            name=f"sb-worker-{i}",
            daemon=True,
        )
        for i in range(cfg.num_workers)
    ]
    for t in threads:
        t.start()

    # Block until end_deadline.
    try:
        while time.monotonic() < end_deadline:
            time.sleep(0.1)
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=5.0)

    # Drop warm-up rows by TIMESTAMP: keep only requests that completed after
    # the warmup deadline (row[1] is the epoch completion time). Robust to
    # non-integer windows, startup jitter, max-rate mode, request errors, and
    # any warmup-vs-measurement rate shift — unlike a count-ratio slice.
    warmup_cutoff = start_wall + cfg.warmup_s
    measured = [r for r in rows if r[1] >= warmup_cutoff]

    if cfg.latency_csv:
        out = Path(cfg.latency_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["worker_id", "ts", "latency_ms", "status"])
            for r in rows:
                w.writerow(r)

    summary = {
        "total_requests": len(rows),
        "measured_requests": len(measured),
        "duration_s": cfg.duration_s,
        "throughput_rps": len(measured) / max(cfg.duration_s, 1e-6),
        "latencies_ms": [r[2] for r in measured],
    }
    return summary


def _parse_args() -> DriverConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--request-hz", type=float, default=2.0,
                   help="Per-worker request rate; 0 = max-rate (back-to-back)")
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--warmup-s", type=float, default=2.0)
    p.add_argument("--bundle-id", default="default")
    p.add_argument("--latency-csv", default="")
    args = p.parse_args()
    return DriverConfig(
        host=args.host, port=args.port,
        num_workers=args.num_workers, request_hz=args.request_hz,
        duration_s=args.duration_s, warmup_s=args.warmup_s,
        bundle_id=args.bundle_id, latency_csv=args.latency_csv,
    )


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    summary = run_driver(_parse_args())
    print(f"throughput_rps={summary['throughput_rps']:.2f} "
          f"requests={summary['measured_requests']}/{summary['total_requests']}")
