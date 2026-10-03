"""E3 R8 callback: summarize debug.tools.physical.grasp_audit outputs by arm and issuing source.

Reads /tmp/r8cb_E3_failure_forensics/grasp_*/grasp_attempts.csv. Descriptive only (no threshold is chosen here).
"""
import collections
import csv
import glob
import json
import sys

import numpy as np

csv.field_size_limit(sys.maxsize)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    rows = []
    for f in sorted(glob.glob("/tmp/r8cb_E3_failure_forensics/grasp_*/grasp_attempts.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r.get("status") != "available" or r.get("lift_within_window") in ("", None):
                    continue
                rows.append(r)
    eps = collections.defaultdict(set)
    for r in rows:
        eps[r["arm"]].add(r["episode_key"])
    print("== grasp attempts (close onsets near a goal object, observed 40-control outcome) ==")
    print("arm  src  attempts  lift_rate  xy_offset_med(cm)  ref_xyz_err_med(cm)  attempts/episode-with-attempt")
    for arm in sorted({r["arm"] for r in rows}):
        a = [r for r in rows if r["arm"] == arm]
        for src in ["ALL"] + sorted({r["src"] for r in a}):
            s = a if src == "ALL" else [r for r in a if r["src"] == src]
            if len(s) < 10:
                continue
            lift = np.mean([r["lift_within_window"] == "True" for r in s])
            xy = np.median([float(r["xy_offset"]) for r in s if r["xy_offset"]]) * 100
            ref = [float(r["reference_xyz_error"]) for r in s if r.get("reference_xyz_error") not in ("", None)]
            refm = np.median(ref) * 100 if ref else float("nan")
            print(f"{arm.replace('r8_', ''):<22} {src:<12} {len(s):5d} {lift:.3f} {xy:6.2f} {refm:6.2f} "
                  f"{len(s) / max(1, len({r['episode_key'] for r in s})):.2f}")
    print("\n== does reference error (object-frame pose vs P10 grasp on the same task/init) predict a failed lift? ==")
    for arm in sorted({r["arm"] for r in rows}):
        s = [r for r in rows if r["arm"] == arm and r.get("reference_xyz_error") not in ("", None)]
        if len(s) < 30:
            continue
        bad = [float(r["reference_xyz_error"]) for r in s if r["lift_within_window"] != "True"]
        ok = [float(r["reference_xyz_error"]) for r in s if r["lift_within_window"] == "True"]
        xb = [float(r["xy_offset"]) for r in s if r["lift_within_window"] != "True"]
        xo = [float(r["xy_offset"]) for r in s if r["lift_within_window"] == "True"]
        print(f"{arm.replace('r8_', ''):<22} n={len(s)} fail={len(bad)} AUROC ref_err {auroc(bad, ok):.3f} "
              f"| AUROC xy_offset {auroc(xb, xo):.3f} | ref_err med fail/ok {np.median(bad) * 100 if bad else float('nan'):.2f}/"
              f"{np.median(ok) * 100:.2f} cm")


if __name__ == "__main__":
    main()
