"""Parse server log for BatchingCoordinator dispatch events.

The coordinator now logs one ``[batch_dispatch] stage=<n> batch_size=<k>``
INFO line per ``_run_batch`` invocation. This script tails or scans the
log, emits per-window summary statistics (count, mean, p50, p95, max)
broken down by stage.

Usage::

    # one-shot summary over the whole log:
    python batch_stats.py --log /tmp/server_srv1.log

    # rolling 60 s windows, append CSV per window:
    python batch_stats.py --log /tmp/server_srv1.log \\
        --tail --window-s 60 --out batch_stats.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re
from statistics import median
import time

DISPATCH_RE = re.compile(r"\[batch_dispatch\] stage=(\d+) batch_size=(\d+)")


def _summarize(events: list[tuple[int, int]]) -> dict[int, dict]:
    """events: list of (stage_id, batch_size). Returns per-stage stats."""
    by_stage: dict[int, list[int]] = defaultdict(list)
    for stage, bs in events:
        by_stage[stage].append(bs)
    out: dict[int, dict] = {}
    for stage, sizes in sorted(by_stage.items()):
        s = sorted(sizes)
        if not s:
            continue
        out[stage] = {
            "count": len(s),
            "mean": sum(s) / len(s),
            "p50": median(s),
            "p95": s[int(0.95 * (len(s) - 1))],
            "max": s[-1],
            "histogram": _histogram(s),
        }
    return out


def _histogram(sizes: list[int], max_bin: int = 16) -> dict[int, int]:
    h: dict[int, int] = defaultdict(int)
    for x in sizes:
        h[min(max_bin, x)] += 1
    return dict(sorted(h.items()))


def _parse_lines(lines) -> list[tuple[int, int]]:
    events = []
    for ln in lines:
        m = DISPATCH_RE.search(ln)
        if m:
            events.append((int(m.group(1)), int(m.group(2))))
    return events


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="server log path")
    p.add_argument("--tail", action="store_true",
                   help="follow log + emit windowed summary continuously")
    p.add_argument("--window-s", type=float, default=60.0)
    p.add_argument("--out", default="", help="optional CSV path for windowed summary")
    args = p.parse_args()

    path = Path(args.log)
    if not args.tail:
        # one-shot
        events = _parse_lines(path.read_text(errors="ignore").splitlines())
        stats = _summarize(events)
        print(f"=== batch dispatch summary ({len(events)} events) ===")
        for stage, st in stats.items():
            print(f"stage{stage}: count={st['count']} mean={st['mean']:.2f} "
                  f"p50={st['p50']} p95={st['p95']} max={st['max']} hist={st['histogram']}")
        return

    # tail mode
    csv_fh = None
    if args.out:
        new = not Path(args.out).exists()
        csv_fh = open(args.out, "a", newline="")
        w = csv.writer(csv_fh)
        if new:
            w.writerow(["window_end_ts", "stage", "count", "mean", "p50", "p95", "max"])
    with path.open() as f:
        f.seek(0, 2)
        buf: list[tuple[int, int]] = []
        win_start = time.time()
        while True:
            line = f.readline()
            if line:
                m = DISPATCH_RE.search(line)
                if m:
                    buf.append((int(m.group(1)), int(m.group(2))))
            else:
                time.sleep(0.1)
            if time.time() - win_start >= args.window_s:
                stats = _summarize(buf)
                ts = time.strftime("%H:%M:%S")
                for stage, st in stats.items():
                    line_out = (
                        f"[{ts}] stage{stage}: n={st['count']} mean={st['mean']:.2f} "
                        f"p50={st['p50']} p95={st['p95']} max={st['max']} hist={st['histogram']}"
                    )
                    print(line_out, flush=True)
                    if csv_fh:
                        w.writerow([ts, stage, st["count"], f"{st['mean']:.2f}",
                                    st["p50"], st["p95"], st["max"]])
                if csv_fh:
                    csv_fh.flush()
                buf.clear()
                win_start = time.time()


if __name__ == "__main__":
    main()
