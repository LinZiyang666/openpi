"""Aggregate the weighted-sum search cell summaries into one ranking table.

Reads every ``summary_ws1-*.json`` under the per-teacher journal dir (written
by ``run_ws_search`` at the end of each cell), joins the cell's weight vector
from the emitted ``index.json``, and writes a ranked CSV + a top-K listing.
Pure local aggregation — safe to run any time mid-sweep; incomplete cells
simply have fewer scored episodes (``n_scored`` column makes that visible).

Usage::

    python exp/robocasa365/summarize_ws_search.py --teacher groot_tp
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

FIELDS = ("vision_0", "vision_1", "vision_2", "robot_state")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True, choices=("groot_tp", "pi05"))
    ap.add_argument("--data-dir", default="", help="default: exp/robocasa365/data/ws_search/<teacher>")
    ap.add_argument("--index", default="", help="default: exp/robocasa365/config/ws_search/<teacher>/index.json")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--run-prefix", default="ws1", help="round tag the summaries were written under")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else root / "data" / "ws_search" / args.teacher
    index_path = pathlib.Path(args.index) if args.index else (
        root / "config" / "ws_search" / args.teacher / "index.json")
    index = json.loads(index_path.read_text())

    rows = []
    for path in sorted(data_dir.glob(f"summary_{args.run_prefix}-*.json")):
        s = json.loads(path.read_text())
        cid = s["cid"]
        weights = index.get(cid, {}).get("weights", {})
        n_scored = sum(v["n_scored"] for v in s["tasks"].values())
        rows.append({
            "cid": cid,
            "macro_sr": s["macro_sr"],
            "n_scored": n_scored,
            "n_err": s["n_err"],
            "n_missing": s.get("n_missing", 0),
            "complete": s.get("complete", False),
            **{f"w_{f}": weights.get(f, 0.0) for f in FIELDS},
            **{f"sr_{t}": v["sr"] for t, v in s["tasks"].items()},
        })
    rows.sort(key=lambda r: (r["macro_sr"] is not None, r["macro_sr"]), reverse=True)

    out_csv = data_dir / f"{args.run_prefix}_search_results.csv"
    if rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # A cell that ended early still leaves a summary file, so counting files
    # would report a full sweep that is not one; the drain gate is the
    # *complete* count, and any incomplete cid needs an --only re-run.
    incomplete = [r["cid"] for r in rows if not r["complete"]]
    missing = sorted(set(index) - {r["cid"] for r in rows})
    print(f"{len(rows) - len(incomplete)}/{len(index)} cells COMPLETE "
          f"({len(rows)} summary files, {len(incomplete)} incomplete, {len(missing)} never run) -> {out_csv}")
    for cid in incomplete + missing:
        why = "incomplete" if cid in incomplete else "never run"
        print(f"  RERUN [{why}] {cid}")
    if incomplete or missing:
        print("  --only " + ",".join(incomplete + missing))
    for r in rows[: args.top]:
        ws = "/".join(f"{r[f'w_{f}']:.3f}" for f in FIELDS)
        sr = "None" if r["macro_sr"] is None else f"{r['macro_sr']:.3f}"
        flag = "" if r["complete"] else "  [INCOMPLETE]"
        print(f"  {sr}  n={r['n_scored']:>3} err={r['n_err']:>2} miss={r['n_missing']:>2}  "
              f"w(v0/v1/v2/rs)={ws}  {r['cid']}{flag}")


if __name__ == "__main__":
    main()
