"""Offline analyzer for Phase 7 GPU-memory dumps from ``dump_mem.py``.

Consumes one or more pickle files produced by

    python dump_mem.py dump --out <file>.pkl

and emits four reports:

1. **Snapshot inventory** — for each snapshot label in each input file:
   number of segments / active / inactive / fragmentation ratio.

2. **Diff table** — comparing two snapshots (``--baseline LABEL`` and
   ``--target LABEL``): which (size, top_frame) bucket grew, by how much,
   net bytes delta. Answers "what did N=4 add over N=1?".

3. **Redundancy table** — for the target snapshot: group blocks by
   ``(size, top_frame_filename:line)``, sort by total bytes descending.
   The top rows are exactly the call sites that allocated the same-sized
   tensor most times — strong signal of per-connection replicas.

4. **Age distribution** — when ``device_traces`` are present (i.e. the
   server was started with ``OPENPI_MONITOR_LEVEL=HISTORY``), bucket every
   active block by ``now - alloc_time`` and report counts + total bytes
   per bucket. Long-tail (>60 s, never freed) blocks are the zombie
   candidates.

Each report is also dumped to CSV if ``--csv-out DIR`` is given.

Usage
-----
::

    # Single-file redundancy report on the latest snapshot:
    python memory_diagnose.py phase7_a100_ramp.pkl

    # Diff N=4 vs N=1, write CSVs:
    python memory_diagnose.py phase7_a100_ramp.pkl \\
        --baseline n1 --target n4 --csv-out ./reports/

    # Cross-file (compare a100 vs jupyter at the same concurrency):
    python memory_diagnose.py a100_n4.pkl jupyter_n4.pkl --target n4
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
import csv
from pathlib import Path
import pickle
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _load_dump(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _iter_snapshots(dump: dict, source: str) -> Iterable[dict]:
    """Yield ``{"source", "label", "ts", "segments", "traces", "summary"}``
    tuples for every snapshot entry in the dump."""
    for entry in dump.get("mem_snapshots", []):
        snap = entry.get("snapshot")
        if snap is None:
            continue
        if isinstance(snap, dict):
            segments = snap.get("segments", [])
            traces = snap.get("device_traces", [])
        else:
            segments = list(snap)
            traces = []
        yield {
            "source": source,
            "label": entry.get("label", "?"),
            "ts": entry.get("ts_epoch", 0.0),
            "segments": segments,
            "traces": traces,
            "summary": entry.get("summary_text", ""),
        }


# ---------------------------------------------------------------------------
# Per-block iteration
# ---------------------------------------------------------------------------


def _format_frame(frame: Any) -> str:
    """PyTorch frame entries may be dicts or tuples; render as 'file:line:fn'."""
    if isinstance(frame, dict):
        fn = frame.get("filename", "?")
        ln = frame.get("line", 0)
        nm = frame.get("name", "?")
        return f"{fn}:{ln}:{nm}"
    if isinstance(frame, (list, tuple)) and len(frame) >= 3:
        return f"{frame[0]}:{frame[1]}:{frame[2]}"
    return str(frame)


def _top_frame(block: dict) -> str:
    frames = block.get("frames") or []
    if not frames:
        return "<no-history>"
    return _format_frame(frames[0])


def _iter_blocks(segments: list[dict]) -> Iterable[tuple[dict, dict]]:
    for seg in segments:
        for block in seg.get("blocks", ()):
            yield seg, block


# ---------------------------------------------------------------------------
# Report 1 — inventory
# ---------------------------------------------------------------------------


def report_inventory(snapshots: list[dict]) -> list[dict]:
    rows = []
    for snap in snapshots:
        n_seg = len(snap["segments"])
        active_b = inactive_b = pending_b = 0
        block_count = 0
        for _, block in _iter_blocks(snap["segments"]):
            size = int(block.get("size", 0))
            state = block.get("state", "")
            block_count += 1
            if "active_allocated" in state:
                active_b += size
            elif "pending_free" in state:
                pending_b += size
            else:
                inactive_b += size
        reserved_b = active_b + inactive_b + pending_b
        rows.append({
            "source": snap["source"],
            "label": snap["label"],
            "n_segments": n_seg,
            "n_blocks": block_count,
            "active_mb": active_b / (1024 * 1024),
            "pending_free_mb": pending_b / (1024 * 1024),
            "inactive_mb": inactive_b / (1024 * 1024),
            "reserved_mb": reserved_b / (1024 * 1024),
            "frag_ratio": (inactive_b / reserved_b) if reserved_b else 0.0,
            "n_traces": len(snap["traces"]),
        })
    return rows


def print_inventory(rows: list[dict]) -> None:
    print("\n=== SNAPSHOT INVENTORY ===")
    if not rows:
        print("(no snapshots in any input)")
        return
    hdr = ("source", "label", "n_seg", "n_blk",
           "active_mb", "pend_mb", "inactive_mb", "reserved_mb", "frag",
           "traces")
    print("  {:30} {:18} {:>6} {:>7} {:>10} {:>9} {:>11} {:>11} {:>6} {:>7}".format(*hdr))
    print("  " + "-" * 120)
    for r in rows:
        print(
            "  {source:30} {label:18} "
            "{n_segments:>6} {n_blocks:>7} "
            "{active_mb:>10.1f} {pending_free_mb:>9.1f} "
            "{inactive_mb:>11.1f} {reserved_mb:>11.1f} "
            "{frag_ratio:>6.2f} {n_traces:>7}".format(**r)
        )


# ---------------------------------------------------------------------------
# Report 2 — redundancy (per target snapshot)
# ---------------------------------------------------------------------------


def report_redundancy(snap: dict, top_k: int = 30) -> list[dict]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for _, block in _iter_blocks(snap["segments"]):
        if "active_allocated" not in block.get("state", ""):
            continue
        size = int(block.get("size", 0))
        site = _top_frame(block)
        groups[(size, site)].append(size)
    rows = []
    for (size, site), sizes in groups.items():
        n = len(sizes)
        total = sum(sizes)
        rows.append({
            "size_kb": size / 1024.0,
            "count": n,
            "total_mb": total / (1024 * 1024),
            "call_site": site,
        })
    rows.sort(key=lambda r: r["total_mb"], reverse=True)
    return rows[:top_k]


def print_redundancy(rows: list[dict], target_label: str) -> None:
    print(f"\n=== REDUNDANCY (target='{target_label}', top {len(rows)} by total bytes) ===")
    if not rows:
        print("(no active blocks)")
        return
    print("  {:>10} {:>7} {:>11}  {}".format(
        "size_kb", "count", "total_mb", "top_frame"
    ))
    print("  " + "-" * 110)
    for r in rows:
        print("  {size_kb:>10.1f} {count:>7d} {total_mb:>11.2f}  {call_site}".format(**r))


# ---------------------------------------------------------------------------
# Report 3 — diff (target - baseline)
# ---------------------------------------------------------------------------


def _bucket(snap: dict) -> dict[tuple, dict]:
    """Group active blocks by (size, top_frame) and return per-bucket stats."""
    out: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "total_bytes": 0})
    for _, block in _iter_blocks(snap["segments"]):
        if "active_allocated" not in block.get("state", ""):
            continue
        size = int(block.get("size", 0))
        site = _top_frame(block)
        bucket = out[(size, site)]
        bucket["count"] += 1
        bucket["total_bytes"] += size
    return out


def report_diff(baseline: dict, target: dict, top_k: int = 30) -> list[dict]:
    b = _bucket(baseline)
    t = _bucket(target)
    keys = set(b) | set(t)
    rows = []
    for key in keys:
        size, site = key
        b_count = b.get(key, {}).get("count", 0)
        t_count = t.get(key, {}).get("count", 0)
        b_bytes = b.get(key, {}).get("total_bytes", 0)
        t_bytes = t.get(key, {}).get("total_bytes", 0)
        rows.append({
            "size_kb": size / 1024.0,
            "baseline_count": b_count,
            "target_count": t_count,
            "delta_count": t_count - b_count,
            "delta_mb": (t_bytes - b_bytes) / (1024 * 1024),
            "call_site": site,
        })
    rows.sort(key=lambda r: r["delta_mb"], reverse=True)
    return rows[:top_k]


def print_diff(rows: list[dict], baseline_label: str, target_label: str) -> None:
    print(f"\n=== DIFF ({target_label} - {baseline_label}, top {len(rows)} by Δmb) ===")
    if not rows:
        print("(no diff buckets)")
        return
    print("  {:>10} {:>9} {:>9} {:>9} {:>9}  {}".format(
        "size_kb", "base_n", "tgt_n", "Δn", "Δmb", "top_frame"
    ))
    print("  " + "-" * 120)
    for r in rows:
        print(
            "  {size_kb:>10.1f} {baseline_count:>9d} {target_count:>9d} "
            "{delta_count:>9d} {delta_mb:>9.2f}  {call_site}".format(**r)
        )


# ---------------------------------------------------------------------------
# Report 4 — age (requires device_traces)
# ---------------------------------------------------------------------------


def report_age(snap: dict, buckets_s: tuple[float, ...] = (1, 5, 30, 60, 300)) -> list[dict]:
    traces = snap.get("traces", [])
    if not traces:
        return []
    # Build alloc-time index: address → time_us of latest alloc event.
    alloc_time_us: dict[int, int] = {}
    last_time_us = 0
    for stream_events in traces:
        for ev in stream_events:
            t = ev.get("time_us", 0)
            last_time_us = max(last_time_us, t)
            action = ev.get("action", "")
            addr = ev.get("addr", 0)
            if action in ("alloc", "segment_alloc"):
                alloc_time_us[addr] = t
            elif action in ("free", "segment_free", "free_completed"):
                alloc_time_us.pop(addr, None)
    # Per-block age, only for active_allocated.
    ages_us: list[int] = []
    sizes: list[int] = []
    for _, block in _iter_blocks(snap["segments"]):
        if "active_allocated" not in block.get("state", ""):
            continue
        addr = int(block.get("address", 0))
        t = alloc_time_us.get(addr)
        if t is None:
            continue
        ages_us.append(last_time_us - t)
        sizes.append(int(block.get("size", 0)))
    # Bucket.
    bucket_edges_us = tuple(int(s * 1_000_000) for s in buckets_s)
    bucket_labels = (
        [f"≤{buckets_s[0]:g}s"]
        + [f"{buckets_s[i-1]:g}-{buckets_s[i]:g}s" for i in range(1, len(buckets_s))]
        + [f">{buckets_s[-1]:g}s"]
    )
    counts = [0] * (len(buckets_s) + 1)
    bytes_per = [0] * (len(buckets_s) + 1)
    for age_us, sz in zip(ages_us, sizes):
        for i, edge in enumerate(bucket_edges_us):
            if age_us <= edge:
                counts[i] += 1
                bytes_per[i] += sz
                break
        else:
            counts[-1] += 1
            bytes_per[-1] += sz
    return [
        {"bucket": bucket_labels[i], "count": counts[i], "total_mb": bytes_per[i] / (1024 * 1024)}
        for i in range(len(counts))
    ]


def print_age(rows: list[dict], target_label: str) -> None:
    print(f"\n=== AGE DISTRIBUTION (target='{target_label}', active_allocated only) ===")
    if not rows:
        print("(no device_traces — server probably not started with OPENPI_MONITOR_LEVEL=HISTORY)")
        return
    print("  {:<10} {:>10} {:>12}".format("bucket", "count", "total_mb"))
    print("  " + "-" * 36)
    for r in rows:
        print("  {bucket:<10} {count:>10d} {total_mb:>12.2f}".format(**r))


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("inputs", nargs="+", help="pickle files from dump_mem.py dump")
    p.add_argument("--baseline", default=None,
                   help="snapshot label to use as diff baseline; omit for redundancy-only mode")
    p.add_argument("--target", default=None,
                   help="snapshot label to use as diff target / redundancy + age subject; "
                        "defaults to the last snapshot found")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--csv-out", default=None,
                   help="directory to write CSV reports into")
    args = p.parse_args()

    snapshots: list[dict] = []
    for inp in args.inputs:
        path = Path(inp)
        if not path.is_file():
            print(f"WARN: not a file: {path}", file=sys.stderr)
            continue
        dump = _load_dump(path)
        for s in _iter_snapshots(dump, source=path.name):
            snapshots.append(s)

    if not snapshots:
        print("no snapshots found", file=sys.stderr)
        return 1

    # Inventory always runs.
    inv = report_inventory(snapshots)
    print_inventory(inv)

    # Pick target.
    target_label = args.target or snapshots[-1]["label"]
    targets = [s for s in snapshots if s["label"] == target_label]
    if not targets:
        print(f"\nERROR: target label {target_label!r} not found", file=sys.stderr)
        return 2
    # If multiple files have the same label, prefer the last one read.
    target = targets[-1]

    red = report_redundancy(target, top_k=args.top_k)
    print_redundancy(red, target_label)

    diff_rows: list[dict] = []
    if args.baseline:
        baselines = [s for s in snapshots if s["label"] == args.baseline]
        if not baselines:
            print(f"\nWARN: baseline label {args.baseline!r} not found; skipping diff", file=sys.stderr)
        else:
            diff_rows = report_diff(baselines[-1], target, top_k=args.top_k)
            print_diff(diff_rows, args.baseline, target_label)

    age_rows = report_age(target)
    print_age(age_rows, target_label)

    if args.csv_out:
        out_dir = Path(args.csv_out)
        _write_csv(inv, out_dir / "inventory.csv")
        _write_csv(red, out_dir / f"redundancy_{target_label}.csv")
        if diff_rows:
            _write_csv(diff_rows, out_dir / f"diff_{args.baseline}_to_{target_label}.csv")
        if age_rows:
            _write_csv(age_rows, out_dir / f"age_{target_label}.csv")
        print(f"\nCSV reports written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
