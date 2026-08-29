"""Descriptive diagnostics on a calibration table (read-only, no fitting).

Two jobs, both purely descriptive and neither touching a frozen criterion:

  1. Distribution acceptance -- the checks that would have caught the retrieval
     width collapse (v all-None, k_eff pinned to 1) before it reached the fit.
  2. The numbers that decide whether the surface can beat the threshold
     frontier. T1-T3 cut on `s` alone; S0's deviation-calibrated cut lies in the
     same family as a rate cut on a monotone transform of `s`. So the surface's
     only structural advantage over the frontier is `v`, and the question is how
     much `v` still says once `s` is known -- reported as a partial rank
     correlation plus a within-s-decile stratification.

Also prints `p90(y10) - p10(y10)`, the spread the frozen delta grid spans. When
that spread is smaller than the fit's OOF safety offset, no delta on the grid
can satisfy `q_hat + offset <= delta` and stop-loss A is forced regardless of
surface quality -- so this one number previews the fit's fate. The offset itself
is only known after the fit runs; this tool does not compute or alter it.

Usage:
  uv run python -m exp.dispatch_surface.analysis.table_diagnostics \
      --table exp/dispatch_surface/data/dispatch_table_fresh.jsonl [--out d.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np
from scipy import stats


def _rank_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residual of ranks(a) after linear regression on ranks(b)."""
    design = np.c_[np.ones_like(b), b]
    return a - design @ np.linalg.lstsq(design, a, rcond=None)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--split", default="fit", help="which split to profile")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.table)]
    out: dict = {"table": args.table, "n_rows": len(rows)}

    # -- 1. distribution acceptance ------------------------------------
    out["acceptance"] = {
        "v_none": sum(1 for r in rows if r.get("v") is None),
        "s_none": sum(1 for r in rows if r.get("s") is None),
        "y7_none": sum(1 for r in rows if r.get("y_tau7") is None),
        "y10_none": sum(1 for r in rows if r.get("y_tau10") is None),
        "k_eff_counts": dict(sorted(collections.Counter(r["k_eff"] for r in rows).items())),
        "ref_modes": sorted({r["ref_mode"] for r in rows}),
        "tasks": len({r["task_id"] for r in rows}),
        "episodes": len({r["episode_id"] for r in rows}),
        "split_rows": dict(collections.Counter(r["split"] for r in rows)),
        "split_episodes": {
            sp: len({r["episode_id"] for r in rows if r["split"] == sp})
            for sp in sorted({r["split"] for r in rows})
        },
    }

    sub = [r for r in rows if r["split"] == args.split]
    if not sub:
        raise SystemExit(f"no rows with split={args.split!r}")
    y10 = np.array([r["y_tau10"] for r in sub], dtype=float)
    y7 = np.array([r["y_tau7"] for r in sub], dtype=float)
    s = np.array([r["s"] for r in sub], dtype=float)
    v = np.array([r["v"] for r in sub], dtype=float)

    pct = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    out["deviation"] = {
        "split": args.split,
        "n": len(sub),
        "percentiles": {p: dict(zip(map(str, pct), np.percentile(a, pct).round(4).tolist()))
                        for p, a in (("y_tau10", y10), ("y_tau7", y7))},
        # The frozen delta grid spans exactly p10..p90 of y10 on this split.
        "delta_grid_span": {
            "p10": float(np.percentile(y10, 10)),
            "p90": float(np.percentile(y10, 90)),
            "spread": float(np.percentile(y10, 90) - np.percentile(y10, 10)),
            "y10_floor": float(y10.min()),
        },
    }

    # -- 2. does v still say anything once s is known ------------------
    rs, ry, rv = (stats.rankdata(a) for a in (s, y10, v))
    out["signal"] = {
        "spearman_y10_s": float(stats.spearmanr(y10, s).statistic),
        "spearman_y10_v": float(stats.spearmanr(y10, v).statistic),
        "spearman_y7_s": float(stats.spearmanr(y7, s).statistic),
        "spearman_y7_v": float(stats.spearmanr(y7, v).statistic),
        "spearman_s_v": float(stats.spearmanr(s, v).statistic),
        "partial_y10_v_given_s": float(
            stats.pearsonr(_rank_residual(ry, rs), _rank_residual(rv, rs)).statistic
        ),
        "partial_y10_s_given_v": float(
            stats.pearsonr(_rank_residual(ry, rv), _rank_residual(rs, rv)).statistic
        ),
    }

    edges = np.quantile(s, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(s, edges[1:-1]), 0, 9)
    deciles, gaps = [], []
    for i in range(10):
        m = bins == i
        if m.sum() < 20:
            continue
        vmed = np.median(v[m])
        lo, hi = m & (v <= vmed), m & (v > vmed)
        gap = float(np.median(y10[hi]) - np.median(y10[lo]))
        gaps.append(gap)
        deciles.append({
            "decile": i, "n": int(m.sum()), "s_upper": float(edges[i + 1]),
            "median_y10": float(np.median(y10[m])),
            "median_y10_v_low": float(np.median(y10[lo])),
            "median_y10_v_high": float(np.median(y10[hi])),
            "v_gap": gap,
        })
    out["s_deciles"] = deciles
    out["v_increment"] = {
        "mean_within_decile_gap": float(np.mean(gaps)),
        "positive_deciles": int(sum(g > 0 for g in gaps)),
        "n_deciles": len(gaps),
        # s's own range across deciles, as the yardstick the v gap is read against
        "s_cross_decile_range": float(deciles[0]["median_y10"] - deciles[-1]["median_y10"]),
    }

    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
