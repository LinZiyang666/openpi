"""Synthetic websocket client for trace-mode server smoke runs (plan §12-K).

Drives one or more connections against a running policy server with the
LIBERO wire observation shape (``observation/image`` 256x256, wrist image,
8-D state, prompt) or the GR00T LIBERO shape (the same keys; the adapter
reshapes them), sending ``episode_start`` / ``episode_end`` control frames so
the trace writer opens and commits one file per episode. Prints the
``__hit_meta__`` summary of every step and a per-connection tally of executed
arms, which is the evidence the end-to-end gate reads.

This is a smoke driver, not a benchmark or an evaluation: the observations
are random, so the only claims it supports are protocol / commit / batching
ones (files exist and audit clean, buckets formed, verdict arms served).

Usage::

    uv run python exp/common/trace_smoke_client.py --host 127.0.0.1 --port 8000 \\
        --connections 3 --episodes 2 --steps 6 --experiment smoke --resolution 256
"""

from __future__ import annotations

import argparse
import collections
import json
import threading
import time

import numpy as np
from openpi_client import websocket_client_policy


def _observation(rng: np.random.Generator, resolution: int, prompt: str) -> dict:
    frame = rng.integers(0, 256, size=(resolution, resolution, 3), dtype=np.uint8)
    return {
        "observation/image": frame,
        "observation/wrist_image": rng.integers(0, 256, size=(resolution, resolution, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(8).astype(np.float32),
        "prompt": prompt,
    }


def run_connection(idx: int, args: argparse.Namespace, results: dict, barrier: threading.Barrier | None) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    rng = np.random.default_rng(1000 + idx)
    tally: collections.Counter = collections.Counter()
    verdicts: collections.Counter = collections.Counter()
    latencies: list[float] = []
    try:
        for ep in range(args.episodes):
            episode_id = idx * 1000 + ep
            client.episode_start(
                experiment=args.experiment,
                task=args.prompt,
                episode_id=episode_id,
                episode_name=f"c{idx}_ep{ep}",
                extra_metadata={"task_uid": f"smoke:{idx}:{ep}", "attempt": 1},
            )
            for step in range(args.steps):
                if barrier is not None:
                    try:
                        barrier.wait(timeout=30)
                    except threading.BrokenBarrierError:
                        pass
                t0 = time.perf_counter()
                out = client.infer(_observation(rng, args.resolution, args.prompt))
                latencies.append((time.perf_counter() - t0) * 1000.0)
                meta = out.get("__hit_meta__") or {}
                verdicts[str(meta.get("hit_type"))] += 1
                trace = meta.get("trace") or {}
                tally[str(trace.get("executed_arm"))] += 1
                if args.verbose:
                    print(f"[c{idx} ep{ep} s{step}] {json.dumps(meta, default=str)}", flush=True)
            client.episode_end(success=True)
    finally:
        results[idx] = {
            "arms": dict(tally),
            "verdicts": dict(verdicts),
            "steps": len(latencies),
            "p50_ms": float(np.median(latencies)) if latencies else None,
            "max_ms": float(np.max(latencies)) if latencies else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--connections", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--experiment", default="trace_smoke")
    parser.add_argument("--prompt", default="pick up the bowl")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--lockstep", action="store_true", help="Barrier every step across connections (bucket forming).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results: dict = {}
    barrier = threading.Barrier(args.connections) if args.lockstep and args.connections > 1 else None
    threads = [threading.Thread(target=run_connection, args=(i, args, results, barrier)) for i in range(args.connections)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    print(json.dumps({"wall_s": round(wall, 2), "connections": results}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
