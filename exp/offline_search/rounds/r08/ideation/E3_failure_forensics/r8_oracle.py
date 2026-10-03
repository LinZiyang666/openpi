"""E3 R8 callback: oracle grasp-window arms (O5a/O5b) - call delivery, timing and paired outcome by A's label.

Reads server decisions_*.jsonl of the O5 arms (read-only) and the pooled forensics labels
(/tmp/r8cb_E3_failure_forensics/forensics_*/episodes_forensics.csv). Descriptive only.
"""
import collections
import csv
import glob
import json
import sys

import numpy as np

csv.field_size_limit(sys.maxsize)
RUN = "/home/weiland/trace_runs/os_closed_loop/r08_main/runs"
ARMS = ["r8_pi05_l10_50_O5a", "r8_pi05_l10_50_O5b", "r8_groot_l10_50_O5a", "r8_groot_l10_50_O5b",
        "r8_pi05_spatial_50_O5b", "r8_groot_spatial_50_O5b"]


def labels():
    out = {}
    for f in glob.glob("/tmp/r8cb_E3_failure_forensics/forensics_*/episodes_forensics.csv"):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                out[(r["arm"], int(r["task_id"]), int(r["init"]))] = (r["success"] == "True", r["label"])
    return out


def main():
    lab = labels()
    for arm in ARMS:
        calls = collections.defaultdict(list)   # (task, init) -> list of (decision_seq, distance)
        anchors = collections.Counter()
        inwin_anchor = collections.Counter()
        n_dec = collections.Counter()
        for f in glob.glob(f"{RUN}/{arm}/debug/server_*/decisions*.jsonl"):
            with open(f) as fh:
                for line in fh:
                    if not line.endswith("\n"):
                        continue
                    j = json.loads(line)
                    key = (int(j["task_id"]), int(j["init"]))
                    n_dec[key] += 1
                    if j.get("fresh"):
                        anchors[key] += 1
                        o = j.get("oracle")
                        if isinstance(o, str):
                            try:
                                o = json.loads(o)
                            except ValueError:
                                o = None
                        if isinstance(o, dict) and o.get("in_window"):
                            inwin_anchor[key] += 1
                    if j.get("override") == "oracle_window" and j.get("src") == "policy":
                        o = j.get("oracle") if isinstance(j.get("oracle"), dict) else {}
                        calls[key].append((int(j["decision_seq"]), o.get("distance")))
        keys = sorted(n_dec)
        ncalls = np.array([len(calls[k]) for k in keys])
        frac_win = np.array([inwin_anchor[k] / max(1, anchors[k]) for k in keys])
        print(f"\n== {arm}: episodes {len(keys)}, oracle calls/episode mean {ncalls.mean():.2f}, "
              f"episodes with >=1 call {np.mean(ncalls > 0):.3f}, in-window anchors share {frac_win.mean():.3f}")
        dist = [d for k in keys for _, d in calls[k] if d is not None]
        if dist:
            print(f"   call distance to object centre: median {np.median(dist) * 100:.1f} cm, p10/p90 "
                  f"{np.percentile(dist, 10) * 100:.1f}/{np.percentile(dist, 90) * 100:.1f} cm")
        model, suite = arm.split("_")[1], arm.split("_")[2]
        ref = f"r8_{model}_{suite}_50_A"
        tab = collections.defaultdict(lambda: [0, 0, 0, 0])  # A label -> [n, rescued, n_with_call, rescued_with_call]
        for k in keys:
            if (ref, *k) not in lab or (arm, *k) not in lab:
                continue
            a_s, a_l = lab[(ref, *k)]
            o_s, o_l = lab[(arm, *k)]
            g = "A_success" if a_s else a_l
            t = tab[g]
            t[0] += 1
            t[1] += int(o_s)
            t[2] += int(len(calls[k]) > 0)
            t[3] += int(o_s and len(calls[k]) > 0)
        for g, (n, r, nc, rc) in sorted(tab.items(), key=lambda x: -x[1][0]):
            if n < 3:
                continue
            print(f"   A={g:<18} n={n:3d} O5 success {r:3d} ({r / n:.0%}) | with>=1 oracle call {nc:3d}, "
                  f"success among them {rc}/{nc}")
        # oracle-arm failure labels by whether any call happened
        fl = collections.Counter()
        for k in keys:
            if (arm, *k) in lab and not lab[(arm, *k)][0]:
                fl[(lab[(arm, *k)][1], len(calls[k]) > 0)] += 1
        print("   O5 failures by (label, had_call):", dict(sorted(fl.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
