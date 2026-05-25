"""Client driver for the Phase 7 GPU-memory monitoring ctrl endpoints.

Drives the five new WS handlers added in ``websocket_policy_server.py``:

    snapshot_mem        — capture torch.cuda.memory_snapshot() under <label>
    dump_metrics        — pull all in-memory ring buffers (util / batch /
                          timing / mem_snapshots) back to a local pickle
    mem_history_start   — turn on torch.cuda.memory._record_memory_history
                          (server must be started with OPENPI_MONITOR_LEVEL=HISTORY)
    mem_history_stop    — turn off recording
    clear               — wipe selected buffers on the server

Examples
--------
::

    # Capture three snapshots under labels baseline / n1 / n4 over a sweep:
    python dump_mem.py --host 149.165.151.106 --port 8000 snapshot --label baseline
    # ... start 1 worker, wait for steady state ...
    python dump_mem.py --host 149.165.151.106 --port 8000 snapshot --label n1
    # ... ramp to 4 workers, wait for steady state ...
    python dump_mem.py --host 149.165.151.106 --port 8000 snapshot --label n4

    # Pull EVERYTHING (snapshots + timeseries) into a single pickle:
    python dump_mem.py --host 149.165.151.106 --port 8000 dump \
        --out phase7_a100_ramp.pkl

    # Or only the snapshots:
    python dump_mem.py --host 149.165.151.106 --port 8000 dump \
        --skip-util --skip-batch --skip-timing --out phase7_snapshots.pkl

The pickle contains the dict returned by the server, with each
``mem_snapshots[i]["snapshot_pickle"]`` pre-decoded back into a Python
object for direct use with ``exp/serving_benchmark/memory_diagnose.py``.

Exit code is 0 on success, 1 on any server-side error (the error message
is printed to stderr).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import time

from openpi_client import msgpack_numpy
import websockets.sync.client


def _connect(uri: str):
    """Open a WS connection + drain the initial metadata frame."""
    conn = websockets.sync.client.connect(uri, compression=None, max_size=None)
    # Server sends metadata immediately after handshake — discard it.
    _ = conn.recv()
    return conn


def _send_ctrl(conn, payload: dict) -> dict:
    packer = msgpack_numpy.Packer()
    conn.send(packer.pack(payload))
    raw = conn.recv()
    if isinstance(raw, str):
        raise RuntimeError(f"server returned string (error): {raw}")
    return msgpack_numpy.unpackb(raw)


def _cmd_snapshot(conn, args) -> int:
    label = args.label or f"snap_{time.strftime('%Y%m%d_%H%M%S')}"
    resp = _send_ctrl(conn, {"__ctrl__": "snapshot_mem", "label": label})
    if resp.get("__ack__") == "error":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        return 1
    print(
        f"snapshot OK label={resp['label']} "
        f"segments={resp['n_segments']} "
        f"active_mb={resp['active_mb']:.1f} "
        f"inactive_mb={resp['inactive_mb']:.1f}"
    )
    return 0


def _cmd_history_start(conn, args) -> int:
    payload = {"__ctrl__": "mem_history_start"}
    if args.max_entries:
        payload["max_entries"] = args.max_entries
    resp = _send_ctrl(conn, payload)
    if resp.get("__ack__") == "error":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        return 1
    print(f"history_start OK {resp}")
    return 0


def _cmd_history_stop(conn, args) -> int:
    resp = _send_ctrl(conn, {"__ctrl__": "mem_history_stop"})
    if resp.get("__ack__") == "error":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        return 1
    print(f"history_stop OK {resp}")
    return 0


def _cmd_dump(conn, args) -> int:
    payload = {
        "__ctrl__": "dump_metrics",
        "include_util": not args.skip_util,
        "include_batch": not args.skip_batch,
        "include_mem": not args.skip_mem,
        "include_timing": not args.skip_timing,
        "clear": args.clear,
    }
    resp = _send_ctrl(conn, payload)
    if resp.get("__ack__") == "error":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        return 1
    # Pre-decode every snapshot_pickle entry so the downstream analyzer
    # does not need to know about the wire-level encoding.
    snaps = resp.get("mem_snapshots", [])
    decoded = 0
    for entry in snaps:
        blob = entry.pop("snapshot_pickle", None)
        if blob:
            try:
                entry["snapshot"] = pickle.loads(blob)
                decoded += 1
            except Exception as exc:
                entry["snapshot"] = None
                entry["_decode_error"] = str(exc)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(resp, f, protocol=4)
    stats = resp.get("stats", {})
    print(
        f"dump OK -> {out_path}\n"
        f"  level={resp.get('level')}\n"
        f"  util_n={len(resp.get('util', []))}\n"
        f"  batch_n={len(resp.get('batch', []))}\n"
        f"  timing_n={len(resp.get('timing', []))}\n"
        f"  mem_snapshots_n={len(snaps)} (decoded={decoded})\n"
        f"  stats={stats}"
    )
    return 0


def _cmd_set_batch(conn, args) -> int:
    """Hot-set coordinator batch params."""
    payload = {"__ctrl__": "set_batch_params"}
    if args.max_batch_size is not None: payload["max_batch_size"] = args.max_batch_size
    if args.max_wait_ms is not None: payload["max_wait_ms"] = args.max_wait_ms
    resp = _send_ctrl(conn, payload)
    if resp.get("__ack__") == "error":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr); return 1
    print(f"set_batch_params OK: max_batch_size={resp['max_batch_size']} max_wait_ms={resp['max_wait_ms']}")
    return 0


def _cmd_summary(conn, args) -> int:
    """Compute throughput / batching / GPU util summary from server buffers."""
    payload = {"__ctrl__": "throughput_summary", "clear": args.clear}
    resp = _send_ctrl(conn, payload)
    if resp.get("__ack__") == "error":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        return 1
    if resp.get("empty"):
        print("server buffers empty (no batch events recorded)")
        return 0
    if resp.get("insufficient_window"):
        print("insufficient window: too few samples to compute a rate "
              f"(util_events={resp.get('n_util_events', '?')}, "
              f"batch_events={resp.get('n_batch_events', '?')}). "
              "Let the server serve longer before requesting a summary.")
        return 0
    print(f"=== Throughput Summary ({resp['elapsed_s']:.1f}s window) ===")
    if resp.get("partial"):
        print(f"  WARNING: PARTIAL — only {resp.get('n_contributing')}/{resp.get('n_replicas')} "
              "replicas contributed; rate below is survivor-only, NOT whole-host.")
    elif resp.get("n_replicas"):
        print(f"  (aggregated over {resp.get('n_contributing')}/{resp.get('n_replicas')} replicas)")
    print(f"  inference rate (stage=1):  {resp['stage1_throughput_inf_per_s']:.2f} inf/s")
    print(f"  peak GPU util:             {resp['peaks']['gpu_util_pct']}%")
    print(f"  peak torch_alloc:          {resp['peaks']['torch_alloc_mb']:.0f} MB")
    print(f"  peak queue depth:          {resp['peaks']['queue_depth']}")
    print(f"  peak cgroup RAM:           {resp['peaks']['cgroup_ram_used_mb']:.0f} MB")
    print(f"  avg proc CPU%:             {resp['averages']['cpu_proc_pct']:.1f}%")
    print(f"  avg sys CPU%:              {resp['averages']['sys_cpu_pct']:.1f}%")
    print(f"  util samples:              {resp['n_util_samples']}")
    print(f"  batch events:              {resp['n_batch_events']}")
    for stage, st in resp.get("stages", {}).items():
        if st.get("count", 0) == 0:
            continue
        print(f"  stage={stage}: count={st['count']} infs={st['inferences']}  "
              f"avg_size={st['avg_size']:.2f} max={st['max_size']}  "
              f"avg_wait={st['avg_wait_ms']:.1f}ms run={st['avg_run_ms']:.1f}ms "
              f"spread={st['avg_spread_ms']:.1f}ms")
    if "cleared" in resp:
        print(f"  cleared: {resp['cleared']}")
    return 0


def _cmd_clear(conn, args) -> int:
    """Convenience: dump_metrics with empty payload but clear=True kinds."""
    payload = {
        "__ctrl__": "dump_metrics",
        "include_util": args.util,
        "include_batch": args.batch,
        "include_mem": args.mem,
        "include_timing": args.timing,
        "clear": True,
    }
    resp = _send_ctrl(conn, payload)
    if resp.get("__ack__") == "error":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        return 1
    print(f"cleared: {resp.get('cleared', {})}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--scheme", default="ws", choices=["ws", "wss"])

    sub = p.add_subparsers(dest="cmd", required=True)

    s_snap = sub.add_parser("snapshot", help="capture one memory snapshot")
    s_snap.add_argument("--label", default=None)

    s_dump = sub.add_parser("dump", help="pull all metrics to a local pickle")
    s_dump.add_argument("--out", required=True)
    s_dump.add_argument("--skip-util", action="store_true")
    s_dump.add_argument("--skip-batch", action="store_true")
    s_dump.add_argument("--skip-mem", action="store_true")
    s_dump.add_argument("--skip-timing", action="store_true")
    s_dump.add_argument("--clear", action="store_true",
                        help="also clear buffers after dumping")

    s_hs = sub.add_parser("history-start", help="enable torch memory history recording")
    s_hs.add_argument("--max-entries", type=int, default=None)

    sub.add_parser("history-stop", help="disable torch memory history recording")

    s_clr = sub.add_parser("clear", help="clear server-side buffers")
    s_clr.add_argument("--util", action="store_true")
    s_clr.add_argument("--batch", action="store_true")
    s_clr.add_argument("--mem", action="store_true")
    s_clr.add_argument("--timing", action="store_true")

    s_sum = sub.add_parser("summary",
                           help="print throughput / batching summary from in-memory buffers")
    s_sum.add_argument("--clear", action="store_true",
                       help="also clear util+batch buffers after reading (for next-ramp baseline)")

    s_bp = sub.add_parser("set-batch", help="hot-set BatchingCoordinator params")
    s_bp.add_argument("--max-batch-size", type=int, default=None)
    s_bp.add_argument("--max-wait-ms", type=float, default=None)

    args = p.parse_args()
    if args.cmd == "clear" and not any([args.util, args.batch, args.mem, args.timing]):
        p.error("clear requires at least one of --util / --batch / --mem / --timing")

    uri = f"{args.scheme}://{args.host}:{args.port}"
    conn = _connect(uri)
    try:
        if args.cmd == "snapshot":
            return _cmd_snapshot(conn, args)
        if args.cmd == "dump":
            return _cmd_dump(conn, args)
        if args.cmd == "history-start":
            return _cmd_history_start(conn, args)
        if args.cmd == "history-stop":
            return _cmd_history_stop(conn, args)
        if args.cmd == "clear":
            return _cmd_clear(conn, args)
        if args.cmd == "summary":
            return _cmd_summary(conn, args)
        if args.cmd == "set-batch":
            return _cmd_set_batch(conn, args)
        p.error(f"unknown command: {args.cmd}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
