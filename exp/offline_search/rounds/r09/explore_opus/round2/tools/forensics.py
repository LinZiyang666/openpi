"""Discovery-only (inits 0-29) extraction of the R8 profile pack (forensics episode labels, stage truth).

RULE 1: the CSVs contain all 500 pairs; ``stream_csv`` parses the ``init`` column of each row and drops the row
before any other field is kept when init >= 30.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

from .common import RUNS, DERIVED, allowed

csv.field_size_limit(sys.maxsize)
PROFILE = RUNS / "r08_main" / "profile"


def stream_csv(path, keep=None):
    rows = []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        ii = header.index("init")
        cols = [header.index(c) for c in keep] if keep else list(range(len(header)))
        names = [header[c] for c in cols]
        for r in rd:
            try:
                init = int(float(r[ii]))
            except ValueError:
                continue
            if not allowed(init):
                continue                     # RULE 1
            rows.append([r[c] for c in cols])
    return pd.DataFrame(rows, columns=names)


EP_COLS = ["arm", "task_id", "init", "success", "label", "onset_control", "onset_decision", "n_controls",
           "active_controls", "subgoals", "subgoals_done", "wrong_objects", "tail_motion", "tail_net", "disturbed",
           "termination_reason", "predicate_detail"]


def episodes(arm):
    out = DERIVED / "forensics" / f"{arm}_episodes.parquet"
    if out.exists():
        return pd.read_parquet(out)
    df = stream_csv(PROFILE / "forensics" / arm / "episodes_forensics.csv", EP_COLS)
    for c in ["task_id", "init", "n_controls", "active_controls", "subgoals", "subgoals_done"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["onset_control", "onset_decision", "tail_motion", "tail_net"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["success"] = df["success"].map({"True": True, "False": False, "true": True, "false": False})
    assert df["init"].max() < 30
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def main():
    arms = sorted(p.name for p in (PROFILE / "forensics").iterdir() if p.is_dir())
    for a in arms:
        df = episodes(a)
        print(a, len(df), df.success.mean().round(3))


if __name__ == "__main__":
    main()
