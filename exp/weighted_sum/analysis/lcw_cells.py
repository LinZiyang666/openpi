"""Collect closed-loop weight-search cells into one JSON shape per suite.

Each source (the pi0.5 weighted-sum phase-2 CSV and the GR00T ws_search round
summaries) is reduced to ``{"cells": [{"w": [v0, v1, rs], "sr": .., "n": ..}]}``
so the offline statistics script can compare any weight vector against the
closed-loop landscape without knowing where the numbers came from. pi0.5 cells
repeated across rounds are pooled (successes summed, n summed); the ``__norm2``
control rows and the non-spatial_16 keybuilders are dropped.

Usage:
  python exp/weighted_sum/analysis/lcw_cells.py pi05 --csv <all_results.csv> --out <json>
  python exp/weighted_sum/analysis/lcw_cells.py groot --summaries <r1.json> <r2.json> --out <json>
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re

_KEY = re.compile(r"v0@(\d+)_v1@(\d+)_rs@(\d+)")


def _pi05(csv_path: pathlib.Path, keybuilder: str) -> list[dict]:
    pooled: dict[tuple[float, float, float], list[int]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["keybuilder"] != keybuilder or row["normalizer"] != "zscore":
                continue
            w = (float(row["v0"]), float(row["v1"]), float(row["rs"]))
            n = int(row["n"])
            succ = round(float(row["success_rate"]) * n)
            acc = pooled.setdefault(w, [0, 0])
            acc[0] += succ
            acc[1] += n
    cells = []
    for w, (succ, n) in sorted(pooled.items()):
        total = sum(w)
        cells.append({"w": [x / total for x in w], "sr": succ / n, "n": n, "raw": list(w)})
    return cells


_YAML_ID = re.compile(r"^(?P<kb>[a-z0-9_]+?)__(?:grid3?|iso)_(?P<fields>[a-z_0-9@]+?)(?P<suffix>__d\d+|__norm2)?$")


def _merge_pi05_json(cells: list[dict], json_path: pathlib.Path, keybuilder: str) -> list[dict]:
    """Add depth-1 cells from a summarize.py results JSON (yaml_id -> success_rate/n)."""
    pooled: dict[tuple[float, float, float], list[int]] = {
        tuple(c["raw"]): [round(c["sr"] * c["n"]), c["n"]] for c in cells
    }
    for yaml_id, rec in json.loads(json_path.read_text()).items():
        m = _YAML_ID.match(yaml_id)
        if m is None or m.group("kb") != keybuilder:
            continue
        if m.group("suffix") not in (None, "__d1"):
            continue
        w = {"vision_0": 0.0, "vision_1": 0.0, "robot_state": 0.0}
        for part in m.group("fields").split("_vision_1@") if False else re.findall(r"(vision_0|vision_1|robot_state)@(\d+)", m.group("fields")):
            w[part[0]] = int(part[1]) / 100.0
        if "iso" in yaml_id:
            field = m.group("fields")
            w = {k: (1.0 if k == field else 0.0) for k in w}
        key = (w["vision_0"], w["vision_1"], w["robot_state"])
        acc = pooled.setdefault(key, [0, 0])
        acc[0] += int(rec["n_success"])
        acc[1] += int(rec["n"])
    out = []
    for w, (succ, n) in sorted(pooled.items()):
        total = sum(w)
        out.append({"w": [x / total for x in w], "sr": succ / n, "n": n, "raw": list(w)})
    return out


def _groot(summaries: list[pathlib.Path]) -> list[dict]:
    cells = []
    for path in summaries:
        d = json.loads(path.read_text())
        n = int(d["episodes_per_cell"])
        for key, sr in d["sr"].items():
            m = _KEY.fullmatch(key)
            if m is None:
                raise ValueError(key)
            raw = [int(g) for g in m.groups()]
            total = sum(raw)
            cells.append({"w": [x / total for x in raw], "sr": float(sr), "n": n, "raw": raw,
                          "round": path.stem})
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("pi05")
    p.add_argument("--csv", type=pathlib.Path, required=True)
    p.add_argument("--keybuilder", default="cp1_spatial_pool_16")
    p.add_argument("--merge-json", type=pathlib.Path, nargs="*", default=[],
                   help="summarize.py results JSON files whose depth-1 cells are pooled in")
    p.add_argument("--out", type=pathlib.Path, required=True)
    g = sub.add_parser("groot")
    g.add_argument("--summaries", type=pathlib.Path, nargs="+", required=True)
    g.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if args.kind == "pi05":
        cells = _pi05(args.csv, args.keybuilder)
        for path in args.merge_json:
            cells = _merge_pi05_json(cells, path, args.keybuilder)
    else:
        cells = _groot(args.summaries)
    leader = max(cells, key=lambda c: c["sr"])
    out = {"cells": cells, "leader": leader, "n_cells": len(cells)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{args.out}: {len(cells)} cells, leader w={leader['w']} sr={leader['sr']:.3f} n={leader['n']}")


if __name__ == "__main__":
    main()
