"""Insert the reduced-step teacher-alone points into the four published RIT figures.

Reads the ``nfe_<policy>_<suite>.json`` specs written by ``aggregate_nfe`` (in
``exp/nfe_baseline/data/figures``) and adds / replaces one series (same key) in
``exp/rit_pareto/analysis/figures/{pareto,groot}_libero_{10,spatial}.json`` so the
figure editor shows and edits the points in place. Figure script: never committed.

Owner rulings (2026-09-15, corrected 17:10): the pi0.5 libero_10 measured k points get +0.04 on the success rate (GR00T untouched)
(kept as ``y_measured`` / ``y_shift`` on each point); the N-step endpoint is kept
for the pi0.5 figures (no teacher-only point drawn there) and dropped for the
GR00T figures (their ``anchors`` series already draws teacher-only at IR 100).
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "exp/nfe_baseline/data/figures"
DST = ROOT / "exp/rit_pareto/analysis/figures"
KEY = "teacher alone, k denoising steps (no cache)"
STYLE = {"color": "#d81b60", "marker": "o", "marker_size": 42, "alpha": 0.95, "point_zorder": 6,
         "linestyle": "-", "linewidth": 2.2, "line_zorder": 5, "annotate_fontsize": 7}
PLAN = [  # (source spec, target figure, keep N-step endpoint, y shift)
    ("nfe_pi05_libero_spatial", "pareto_libero_spatial", True, 0.0),
    ("nfe_pi05_libero_10", "pareto_libero_10", True, 0.04),
    ("nfe_groot_libero_spatial", "groot_libero_spatial", False, 0.0),
    ("nfe_groot_libero_10", "groot_libero_10", False, 0.0),
]


def main() -> None:
    for src_id, dst_id, keep_anchor, shift in PLAN:
        src = json.loads((SRC / f"{src_id}.json").read_text())
        dst_path = DST / f"{dst_id}.json"
        dst = json.loads(dst_path.read_text())
        pts = []
        for p in src["series"][0]["points"]:
            if p["id"].endswith("_anchor") and not keep_anchor:
                continue
            q = {k: v for k, v in p.items()}
            if shift and not p["id"].endswith("_anchor"):
                q["y_measured"] = p["y"]
                q["y_shift"] = shift
                q["y"] = round(p["y"] + shift, 6)
            pts.append(q)
        n_steps = src["num_steps"] if "num_steps" in src else None
        label = src["series"][0]["key"]
        series = {"key": KEY, "style": STYLE, "show_points": True, "show_frontier": True, "annotate": True,
                  "scatter_label": f"{label}: points ({{n_arms}} x 500 ep)",
                  "frontier_label": f"{label}: Pareto frontier ({{n_front}} non-dominated)",
                  "points": pts, "hidden": False}
        dst["series"] = [s for s in dst["series"] if s.get("key") != KEY] + [series]
        dst_path.write_text(json.dumps(dst, indent=2) + "\n")
        print(dst_id, "<-", src_id, f"{len(pts)} pts", f"shift={shift}", [(p['label'], p['y']) for p in pts])


if __name__ == "__main__":
    main()
