"""Event-driven experiment runner for Phase 7 GPU-memory attribution.

Instead of blind ``sleep N`` between snapshots, this driver:

1. Holds an open WS connection to the server.
2. Calls ``dump_metrics`` periodically to read the in-memory ``util``
   and ``batch`` buffers.
3. Decides when each phase has reached **steady state** based on
   monitor events:
   * util buffer's ``torch_alloc_mb`` stops growing (Δ < ``--alloc-tol-mb``
     for ``--stable-windows`` consecutive 5 s samples), AND
   * the batch event count is non-zero recently (the server saw real
     work, not just idle).
4. Calls ``snapshot_mem`` to anchor a labeled memory snapshot.
5. Moves on to the next phase.

Phases are declared on the CLI; this script does NOT start workers —
that responsibility stays with whatever drives the LIBERO sweep (e.g.
``tether exec timan107 -- tmux new ...``). The runner only:
  - takes a snapshot at the start of a phase (``--start-label LABEL``)
  - waits for steady-state on the next phase's load
  - takes another snapshot when the next phase is steady

Typical usage (Stage 1 SNAPSHOT ramp)::

    python experiment_runner.py \\
        --host 149.165.151.106 --port 8000 \\
        --phase baseline:0 \\
        --phase n1:1 \\
        --phase n2:2 \\
        --phase n4:4 \\
        --phase cleanup:0 \\
        --worker-start-cmd 'tether exec timan107 -- bash -lc ...' \\
        --worker-stop-cmd 'tether exec timan107 -- tmux kill-session -t drv_{N}' \\
        --out-dir logs/phase7_mem_attribution/stage1/

The ``--phase NAME:N`` argument says "label this phase NAME and bring
the active connection count to N before snapshotting". The runner uses
the user-supplied start/stop commands to scale connections; it does
not assume any particular sim harness.

The simpler entry point ``--manual`` skips command invocation entirely
and waits on stdin between phases — use it when you want to drive the
sim manually from a tmux session and just trigger snapshots.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pickle
import subprocess
import sys
import time

from openpi_client import msgpack_numpy
import websockets.sync.client

logger = logging.getLogger("exp_runner")


# ---------------------------------------------------------------------------
# WebSocket helper
# ---------------------------------------------------------------------------


def _connect(host: str, port: int):
    uri = f"ws://{host}:{port}"
    conn = websockets.sync.client.connect(uri, compression=None, max_size=None)
    _ = conn.recv()  # drain metadata frame
    return conn


def _ctrl(conn, payload: dict) -> dict:
    packer = msgpack_numpy.Packer()
    conn.send(packer.pack(payload))
    raw = conn.recv()
    if isinstance(raw, str):
        raise RuntimeError(f"server error: {raw}")
    return msgpack_numpy.unpackb(raw)


# ---------------------------------------------------------------------------
# Steady-state detection
# ---------------------------------------------------------------------------


def wait_steady(
    conn,
    *,
    poll_interval_s: float = 3.0,
    alloc_tol_mb: float = 200.0,
    stable_windows: int = 3,
    min_batches_since_last: int = 1,
    max_wait_s: float = 600.0,
    label_hint: str = "?",
) -> dict:
    """Block until the server's monitor reports a steady state.

    Polls ``dump_metrics`` (without ``include_mem``/``clear``) every
    ``poll_interval_s``. Steady state := the last ``stable_windows``
    consecutive util samples have ``|Δ torch_alloc_mb| < alloc_tol_mb``
    AND the batch event count has advanced by at least
    ``min_batches_since_last`` since the previous poll.

    Returns a dict with the final reading so the caller can log it.
    """
    start = time.monotonic()
    last_batch_total = -1
    alloc_history: list[float] = []
    while True:
        if time.monotonic() - start > max_wait_s:
            raise TimeoutError(
                f"wait_steady({label_hint}) timed out after {max_wait_s:.0f}s"
            )
        resp = _ctrl(conn, {
            "__ctrl__": "dump_metrics",
            "include_mem": False,
            "include_timing": False,
            "clear": False,
        })
        util = resp.get("util", [])
        batch = resp.get("batch", [])
        appended = resp.get("appended_totals", {}) or resp.get("stats", {}).get(
            "appended_totals", {}
        )
        batch_total = appended.get("batch", len(batch))

        if util:
            cur_alloc = float(util[-1].get("torch_alloc_mb", 0.0))
            cur_gpu = float(util[-1].get("gpu_util_pct", 0.0))
            cur_q = (
                int(util[-1].get("q1", 0))
                + int(util[-1].get("q2", 0))
                + int(util[-1].get("q3", 0))
            )
        else:
            cur_alloc, cur_gpu, cur_q = 0.0, 0.0, 0
        alloc_history.append(cur_alloc)
        if len(alloc_history) > stable_windows + 1:
            alloc_history.pop(0)
        recent_batch_delta = batch_total - max(last_batch_total, 0)
        last_batch_total = batch_total

        stable = False
        if len(alloc_history) >= stable_windows + 1:
            deltas = [
                abs(alloc_history[i + 1] - alloc_history[i])
                for i in range(stable_windows)
            ]
            stable = all(d < alloc_tol_mb for d in deltas)
        active_enough = recent_batch_delta >= min_batches_since_last

        logger.info(
            "[%s] t=%.0fs alloc_mb=%.0f gpu_util=%.0f%% q=%d "
            "batch_total=%d (Δ%d) stable=%s active=%s",
            label_hint, time.monotonic() - start,
            cur_alloc, cur_gpu, cur_q, batch_total, recent_batch_delta,
            stable, active_enough,
        )

        # "cleanup" / "baseline" phases want no activity — the active check
        # is inverted there. Caller handles via min_batches_since_last=0.
        if stable and (active_enough or min_batches_since_last == 0):
            return {
                "torch_alloc_mb": cur_alloc,
                "gpu_util_pct": cur_gpu,
                "queue_depth": cur_q,
                "batch_total": batch_total,
                "elapsed_s": time.monotonic() - start,
            }
        time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# Phase driver
# ---------------------------------------------------------------------------


def parse_phase(s: str) -> tuple[str, int]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"--phase requires NAME:N, got {s!r}"
        )
    name, n_str = s.split(":", 1)
    return name.strip(), int(n_str)


def _spawn_worker(cmd_template: str, worker_idx: int) -> None:
    cmd = cmd_template.replace("{N}", str(worker_idx)).replace(
        "{GPU}", str(worker_idx % 8)
    )
    logger.info("[spawn] %s", cmd)
    subprocess.run(cmd, shell=True, check=False)


def _stop_worker(cmd_template: str, worker_idx: int) -> None:
    cmd = cmd_template.replace("{N}", str(worker_idx))
    logger.info("[stop ] %s", cmd)
    subprocess.run(cmd, shell=True, check=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--phase", action="append", type=parse_phase, default=[],
        help="Phase declaration NAME:N (active worker count). Repeat.",
    )
    p.add_argument(
        "--worker-start-cmd", default="",
        help=(
            "Shell command to start worker {N}. Will substitute {N} (worker "
            "index, 0-based) and {GPU} (={N} mod 8). Leave empty with --manual."
        ),
    )
    p.add_argument(
        "--worker-stop-cmd", default="",
        help="Shell command to stop worker {N}. Optional.",
    )
    p.add_argument(
        "--manual", action="store_true",
        help="Don't run start/stop cmds; pause on stdin between phases.",
    )
    p.add_argument("--out-dir", default=".")
    p.add_argument("--poll-interval-s", type=float, default=3.0)
    p.add_argument("--alloc-tol-mb", type=float, default=200.0)
    p.add_argument("--stable-windows", type=int, default=3)
    p.add_argument("--min-batches-per-poll", type=int, default=1)
    p.add_argument("--max-wait-s", type=float, default=600.0)
    p.add_argument(
        "--warmup-s", type=float, default=15.0,
        help="Fixed initial sleep after spawning a worker, before "
             "starting steady-state polling. Avoids latch-on at idle.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    if not args.phase:
        print("ERR: at least one --phase NAME:N required", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = _connect(args.host, args.port)
    logger.info("connected to ws://%s:%d", args.host, args.port)

    # Sanity: server must be at level >= SNAPSHOT for the snapshot ctrls.
    probe = _ctrl(conn, {"__ctrl__": "dump_metrics", "include_mem": False,
                         "include_batch": False, "include_util": False,
                         "include_timing": False})
    server_level = probe.get("level", "?")
    logger.info("server OPENPI_MONITOR_LEVEL=%s", server_level)
    if server_level not in ("SNAPSHOT", "HISTORY"):
        logger.warning(
            "server level is %s (<SNAPSHOT) — snapshot_mem will be rejected. "
            "Restart server with OPENPI_MONITOR_LEVEL=snapshot or higher.",
            server_level,
        )

    running_workers: set[int] = set()
    phase_records: list[dict] = []

    for phase_idx, (name, target_n) in enumerate(args.phase):
        logger.info("=== PHASE %d: %s (target_n=%d) ===", phase_idx, name, target_n)
        cur_n = len(running_workers)

        # Scale workers up.
        while len(running_workers) < target_n:
            idx = len(running_workers)
            if args.manual:
                input(f"[manual] start worker idx={idx} then press ENTER: ")
            elif args.worker_start_cmd:
                _spawn_worker(args.worker_start_cmd, idx)
            running_workers.add(idx)

        # Scale workers down.
        while len(running_workers) > target_n:
            idx = max(running_workers)
            if args.manual:
                input(f"[manual] stop worker idx={idx} then press ENTER: ")
            elif args.worker_stop_cmd:
                _stop_worker(args.worker_stop_cmd, idx)
            running_workers.discard(idx)

        # Warmup gap so the first batch hits the server before we measure.
        if target_n != cur_n and args.warmup_s > 0:
            logger.info("warmup sleep %.0fs", args.warmup_s)
            time.sleep(args.warmup_s)

        # Wait for steady state. "baseline" / "cleanup" / target=0 phases
        # don't need batch activity, so flip min_batches_since_last=0.
        min_batches = 0 if target_n == 0 else args.min_batches_per_poll
        steady = wait_steady(
            conn,
            poll_interval_s=args.poll_interval_s,
            alloc_tol_mb=args.alloc_tol_mb,
            stable_windows=args.stable_windows,
            min_batches_since_last=min_batches,
            max_wait_s=args.max_wait_s,
            label_hint=name,
        )
        logger.info("[%s] STEADY %s", name, steady)

        # Anchor with a labeled snapshot.
        snap = _ctrl(conn, {"__ctrl__": "snapshot_mem", "label": name})
        if snap.get("__ack__") == "error":
            logger.error("snapshot_mem rejected: %s", snap.get("msg"))
        else:
            logger.info(
                "[%s] SNAPSHOT segments=%d active_mb=%.1f inactive_mb=%.1f",
                name, snap.get("n_segments", 0),
                snap.get("active_mb", 0.0), snap.get("inactive_mb", 0.0),
            )
        phase_records.append({
            "name": name, "target_n": target_n,
            "steady": steady, "snapshot": snap,
        })

    # Final bulk dump for offline analysis.
    logger.info("=== final dump_metrics ===")
    final = _ctrl(conn, {
        "__ctrl__": "dump_metrics",
        "include_util": True, "include_batch": True,
        "include_mem": True, "include_timing": True,
        "clear": False,
    })
    # Pre-decode each snapshot pickle for downstream tools.
    decoded = 0
    for entry in final.get("mem_snapshots", []):
        blob = entry.pop("snapshot_pickle", None)
        if blob:
            try:
                entry["snapshot"] = pickle.loads(blob)
                decoded += 1
            except Exception as exc:
                entry["snapshot"] = None
                entry["_decode_error"] = str(exc)
    out_path = out_dir / f"experiment_dump_{int(time.time())}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"phase_records": phase_records, "final": final}, f, protocol=4)
    logger.info(
        "wrote %s (util_n=%d batch_n=%d timing_n=%d snap_n=%d decoded=%d)",
        out_path,
        len(final.get("util", [])), len(final.get("batch", [])),
        len(final.get("timing", [])), len(final.get("mem_snapshots", [])),
        decoded,
    )
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
