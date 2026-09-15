"""Budget-vs-regret simulation for cheap weight-selection protocols on the closed-loop landscapes.

The truth model for a weight vector is the kernel-smoothed closed-loop success
rate over the grid-search cells (same smoother as lcw_offline_stats); an
episode is a Bernoulli draw from it. Protocols compared, each returning one
weight vector:

* ``lda_only``      the zero-cost LDA prior, no rollouts;
* ``uniform_only``  equal weights, no rollouts;
* ``confirm3``      {LDA, LDA with the field of highest offline stage-confusion
                    rate dropped, uniform}, B episodes each, pick the best;
* ``halving28``     successive halving on the 1/6 simplex grid (28 cells):
                    B1 each, keep half, ... (the "blind" cheap search);
* ``memo9``         nine 1/6-grid cells nearest the Phase-1 J-ratio point,
                    halving 9 -> 4 -> 2 (the §3u plan-2 shape).

Regret is the smoothed-landscape maximum minus the truth of the chosen vector.

Usage:
  python exp/weighted_sum/analysis/lcw_confirm_sim.py --cells <json> --stats <json> --fits <json> \
      --J a,b,c --output <dir> [--sims 4000]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from lcw_offline_stats import FIELDS, l1, normalize_w, simplex, smoothed_sr


def run_protocol(rng, truth, arms: list, budgets: list[int]) -> float:
    """Successive halving over ``arms`` with per-round per-arm budgets; returns chosen truth."""
    alive = list(range(len(arms)))
    wins = np.zeros(len(arms))
    n = np.zeros(len(arms))
    for b in budgets:
        for a in alive:
            wins[a] += rng.binomial(b, truth[a])
            n[a] += b
        if len(alive) == 1:
            break
        keep = max(1, len(alive) // 2)
        alive = sorted(alive, key=lambda a: -(wins[a] / n[a]))[:keep]
    return truth[alive[0]]


def confirm_with_margin(rng, truth, b: int, z: float) -> float:
    """Keep the prior (arm 0) unless a challenger beats it by z standard errors."""
    hits = np.array([rng.binomial(b, t) for t in truth]) / b
    se = np.sqrt(max(hits[0] * (1 - hits[0]), 0.05) / b)
    best = int(np.argmax(hits))
    return truth[best] if hits[best] - hits[0] >= z * se else truth[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=pathlib.Path, required=True)
    ap.add_argument("--stats", type=pathlib.Path, required=True)
    ap.add_argument("--fits", type=pathlib.Path, required=True)
    ap.add_argument("--J", required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--lda-key", default="lda@0.05")
    ap.add_argument("--bandwidth", type=float, default=0.15)
    ap.add_argument("--sims", type=int, default=4000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260913)

    cells = json.loads(args.cells.read_text())["cells"]
    stats = json.loads(args.stats.read_text())["stats"]
    fits = json.loads(args.fits.read_text())["fits"]
    truth_of = lambda w: smoothed_sr(w, cells, args.bandwidth)  # noqa: E731
    grid6 = [list(w) for w in simplex(6)]
    best = max(truth_of(w) for w in grid6 + [c["w"] for c in cells])

    lda = fits[args.lda_key]["w"]
    conf = {f: stats[f]["top1_phase_gt_0.3"] for f in FIELDS}
    flagged = max(conf, key=conf.get)
    dropped = normalize_w([0.0 if f == flagged else lda[k] for k, f in enumerate(FIELDS)])
    uniform = [1 / 3] * 3
    jw = normalize_w([float(x) for x in args.J.split(",")])
    memo_cells = sorted(grid6, key=lambda w: l1(w, jw))[:9]

    arms3 = [lda, dropped, uniform]
    t3 = [truth_of(w) for w in arms3]
    t28 = [truth_of(w) for w in grid6]
    t9 = [truth_of(w) for w in memo_cells]
    report = {
        "smoothed_max": best, "lda": lda, "lda_truth": t3[0], "flagged_field": flagged,
        "stage_confusion": conf, "dropped": dropped, "dropped_truth": t3[1], "uniform_truth": t3[2],
        "grid6_truth_max": max(t28), "memo9_truth_max": max(t9), "protocols": {},
    }
    print(f"smoothed max {best:.3f}; LDA {np.round(lda, 3).tolist()} -> {t3[0]:.3f}; "
          f"flagged {flagged} ({conf[flagged]:.4f}) dropped -> {t3[1]:.3f}; uniform -> {t3[2]:.3f}")
    print(f"{'protocol':14s} {'episodes':>9s} {'mean regret pp':>15s} {'P(regret>2pp)':>14s} {'P(regret>5pp)':>14s}")

    def record(name, episodes, chosen):
        r = 100 * (best - np.asarray(chosen))
        report["protocols"][f"{name}@{episodes}"] = {
            "episodes": episodes, "mean_regret_pp": float(r.mean()),
            "p_gt2": float((r > 2).mean()), "p_gt5": float((r > 5).mean())}
        print(f"{name:14s} {episodes:9d} {r.mean():15.2f} {(r > 2).mean():14.2f} {(r > 5).mean():14.2f}")

    record("lda_only", 0, [t3[0]] * args.sims)
    record("uniform_only", 0, [t3[2]] * args.sims)
    for b in (50, 100, 200, 500):
        record("confirm3", 3 * b, [run_protocol(rng, t3, arms3, [b]) for _ in range(args.sims)])
    for b in (100, 200, 500):
        record("confirm3_z1", 3 * b, [confirm_with_margin(rng, t3, b, 1.0) for _ in range(args.sims)])
    interior = [c for c in cells if min(c["w"]) > 1e-9]
    sm = np.array([truth_of(c["w"]) for c in interior])
    report["flatness"] = {"interior_cells": len(interior),
                          "within_2pp": float((sm >= best - 0.02).mean()),
                          "within_3pp": float((sm >= best - 0.03).mean()),
                          "within_5pp": float((sm >= best - 0.05).mean()),
                          "smoothed_min_interior": float(sm.min())}
    print("flatness (smoothed, interior cells):", json.dumps(report["flatness"]))
    for b1 in (25, 50, 100):
        budgets = [b1, 2 * b1, 4 * b1, 10 * b1]
        eps = 28 * b1 + 14 * 2 * b1 + 7 * 4 * b1 + 3 * 10 * b1
        record("halving28", eps, [run_protocol(rng, t28, grid6, budgets) for _ in range(args.sims)])
    for b1 in (50, 100):
        budgets = [b1, 2 * b1, 5 * b1]
        eps = 9 * b1 + 4 * 2 * b1 + 2 * 5 * b1
        record("memo9", eps, [run_protocol(rng, t9, memo_cells, budgets) for _ in range(args.sims)])
    (args.output / "lcw_confirm_sim.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
