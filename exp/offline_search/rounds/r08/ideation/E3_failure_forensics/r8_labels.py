"""E3 R8 callback: pool forensics episodes_forensics.csv across tool runs; per-arm failure modes, paired rescue tables.

Reads /tmp/r8cb_E3_failure_forensics/forensics_*/episodes_forensics.csv (debug.tools.physical.forensics outputs, all
run with the same frozen near-radius calibration from P10 discovery successes). Descriptive arm-level tables use all
500 pairs; nothing here selects a threshold or segmentation.
"""
import collections
import csv
import glob
import json
import sys

import numpy as np

csv.field_size_limit(sys.maxsize)
BASE = "/tmp/r8cb_E3_failure_forensics"
KEEP = ("arm", "task_id", "init", "success", "label", "onset_control", "n_controls", "active_controls",
        "subgoals_done", "subgoals", "tail_motion", "termination_reason", "status", "reference_control")
MODES = ["grasp_miss", "drop", "drop_regrasp_fail", "misplace", "unknown_release", "held_not_placed", "undone",
         "fixture_not_done", "never_reached", "wrong_object", "all_preds_true_but_fail"]


def load():
    rows = {}
    for f in sorted(glob.glob(f"{BASE}/forensics_*/episodes_forensics.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r = {k: r.get(k) for k in KEEP}
                rows[(r["arm"], int(r["task_id"]), int(r["init"]))] = r
    return rows


def cell_of(arm):
    p = arm.split("_")  # r8_<model>_<suite>[_<lib>]_<variant>
    return p[1], p[2], (p[3] if len(p) == 5 else None), p[-1]


def paired_ci(a, b, draws=2000, seed=0):
    """Mean of b-a with task/init cluster bootstrap (one pair per cluster)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(draws, len(d)))
    m = d[idx].mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    rows = load()
    arms = sorted({k[0] for k in rows})
    by_arm = collections.defaultdict(dict)
    for (arm, t, i), r in rows.items():
        by_arm[arm][(t, i)] = r
    print("arms with forensics:", len(arms))
    print("\n== per-arm failure modes (all 500 pairs; label of first failing goal) ==")
    hdr = ["arm", "n", "SR"] + MODES + ["unavail"]
    print("\t".join(h[:10] for h in hdr))
    for arm in arms:
        d = by_arm[arm]
        c = collections.Counter(r["label"] for r in d.values() if r["status"] == "available")
        un = sum(1 for r in d.values() if r["status"] != "available")
        sr = np.mean([r["success"] == "True" for r in d.values()])
        print("\t".join([arm.replace("r8_", ""), str(len(d)), f"{sr:.3f}"] + [str(c.get(m, 0)) for m in MODES] + [str(un)]))
    # grouped shares
    print("\n== grouped failure shares by variant (L10 / Spatial, 50-lib cells + P10) ==")
    groups = collections.defaultdict(collections.Counter)
    for arm in arms:
        model, suite, lib, var = cell_of(arm)
        if lib not in (None, "50"):
            continue
        for r in by_arm[arm].values():
            if r["success"] == "False" and r["status"] == "available":
                groups[(suite, var)][r["label"]] += 1
    for k in sorted(groups):
        c = groups[k]
        n = sum(c.values())
        gm = c["grasp_miss"]
        dr = c["drop"] + c["drop_regrasp_fail"]
        pl = c["misplace"] + c["unknown_release"] + c["held_not_placed"] + c["undone"]
        print(f"{k[0]:<8}{k[1]:<6} failures {n:4d} | grasp_miss {gm:3d} ({gm / n:.0%}) | drop {dr:3d} ({dr / n:.0%}) | "
              f"place-side {pl:3d} ({pl / n:.0%}) | other {n - gm - dr - pl:3d}")
    # paired rescue vs A in the same cell
    print("\n== paired vs A (same cell, same task/init/seed): SR diff [95% cluster CI]; rescue of A failures by A label ==")
    for arm in arms:
        model, suite, lib, var = cell_of(arm)
        if var == "A":
            continue
        refs = [f"r8_{model}_{suite}_{lib}_A"] if lib else [f"r8_{model}_{suite}_50_A", f"r8_{model}_{suite}_500_A"]
        for ref in refs:
            if ref not in by_arm:
                continue
            keys = sorted(set(by_arm[arm]) & set(by_arm[ref]))
            a = [by_arm[ref][k]["success"] == "True" for k in keys]
            b = [by_arm[arm][k]["success"] == "True" for k in keys]
            m, lo, hi = paired_ci(a, b)
            resc = collections.Counter()
            tot = collections.Counter()
            for k in keys:
                ra = by_arm[ref][k]
                if ra["success"] == "False":
                    tot[ra["label"]] += 1
                    if by_arm[arm][k]["success"] == "True":
                        resc[ra["label"]] += 1
            lost = sum(1 for k in keys if by_arm[ref][k]["success"] == "True" and by_arm[arm][k]["success"] == "False")
            gained = sum(resc.values())
            detail = " ".join(f"{lab}:{resc[lab]}/{tot[lab]}" for lab in ("grasp_miss", "drop", "misplace",
                              "unknown_release", "held_not_placed") if tot[lab])
            print(f"{arm.replace('r8_', ''):<22} vs {ref.replace('r8_', ''):<18} n={len(keys)} dSR={m:+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] rescued {gained} lost {lost} | {detail}")
    # onsets and waste
    print("\n== onset (decisive event) and controls after onset, failures with onset ==")
    for arm in arms:
        f = [r for r in by_arm[arm].values() if r["success"] == "False" and r["onset_control"] not in ("", None)]
        if len(f) < 5:
            continue
        on = np.array([float(r["onset_control"]) for r in f])
        n = np.array([float(r["n_controls"]) for r in f])
        print(f"{arm.replace('r8_', ''):<24} fail_with_onset {len(f):3d} onset med {np.median(on):5.0f} "
              f"after-onset med {np.median(n - on):4.0f} ({np.median((n - on) / n):.0%})")


if __name__ == "__main__":
    main()
