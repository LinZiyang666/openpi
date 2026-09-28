"""Sequential randomized excursion effects, reusing K5 task/init cluster draws.

Outcome = terminal success, future policy = the same randomized profiler. This
does NOT identify a single MISS followed by pure A. Anchor-weighted estimand;
repeated terminal outcomes within episodes are NOT independent observations.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from exp.offline_search.rounds.r04.k5_rand.estimate import cluster_draws


def effect(rows, boot=2000, seed=603):
    eligible = [r for r in rows if str(r["assignment.eligible"]) in ("True", "1")
                and 0 < float(r["assignment.propensity"]) < 1]
    if not eligible:
        raise ValueError("no randomized anchors with positivity")
    groups = {str(r["arm"]) for r in eligible}
    # Caller pools only comparable model/suite/library cells. Different
    # replicate arm names are fine; output always lists exactly what was pooled.
    identities = [dict(task_id=int(r["task_id"]), init=int(r["init"])) for r in eligible]
    cid, weights = cluster_draws(identities, boot, seed)
    p = np.array([float(r["assignment.propensity"]) for r in eligible])
    a = np.array([str(r["assignment.assigned_call"]) in ("True", "1") for r in eligible])
    y = np.array([float(r["Y"]) for r in eligible])
    # Fixed centering reduces HT noise without learning from held-out outcomes.
    score = (a - p) / (p * (1 - p)) * (y - .5)
    k = weights.shape[1]
    numerator, denominator = np.zeros(k), np.zeros(k)
    np.add.at(numerator, cid, score)
    np.add.at(denominator, cid, 1.)
    draws = (weights @ numerator) / (weights @ denominator)
    return dict(arms=sorted(groups), n_anchors=len(eligible), n_clusters=k, calls=int(a.sum()),
                delta=float(score.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                bootstrap=boot, seed=seed, supported=bool(k >= 2 and a.any() and (~a).any()),
                estimand="CALL-minus-CACHE at visited anchors; randomized p continuation; anchor weighted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--arms", nargs="+", required=True, help="replicates of ONE model/suite/library cell")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with a.csv.open() as f:
        rows = [r for r in csv.DictReader(f) if r["arm"] in a.arms]
    result = effect(rows, a.boot)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
