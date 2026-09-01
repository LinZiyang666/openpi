"""Offline H-CRD parameter pre-screen on the frozen dev table (no rollouts).

Replays each dev episode's (s, v) sequence (force-MISS fresh trajectories from
``dispatch_table_fresh.jsonl``) through the exact H-CRD decision rules of a CRD
artifact, with optional parameter overrides, and reports the tier mix, the
analytic cost per decision and the RECOVERY stay statistics.

APPROXIMATION, NOT A MEASUREMENT: the replayed trajectory was produced under
always-MISS execution, so cache-heavy stretches drift away from what the
controller would really see. The MISS/RECOVERY regime (where every step IS a
full inference) is well approximated -- which is exactly the regime that ate
51.6 % of batch D's cost -- and stay lengths calibrate against batch D via the
default-parameter rows. Use this to RANK candidate parameter sets before
spending episodes; never to quote cost or SR.
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from openpi.cache.components.crd_judge import CumulativeRiskJudge

COST = {"FULL": 10.260266, "WARM": 46.82, "MISS": 67.518595}


def replay(judge, episodes, *, gamma=None, j_bad=None, dwell=None, reopen_mult=1.0):
    g = judge.gamma if gamma is None else gamma
    b = judge.beta
    j = judge.j_bad if j_bad is None else j_bad
    fuse_l = judge.l_max
    dw = judge.min_recovery_misses if dwell is None else dwell
    dr = judge.delta_reopen * reopen_mult
    delta = judge.delta
    edges = judge.artifact.v_bin_edges
    n_v = len(edges) - 1
    mix = collections.Counter()
    stays = []
    for task_id, rows in episodes:
        scale = judge.task_scale[int(task_id)]
        D, mode, bad, fh, rec, cur = 0.0, "A", 0, 0, 0, 0
        for s, v in rows:
            vb = min(max(int(np.searchsorted(edges, v, side="right")) - 1, 0), n_v - 1)
            u_f = judge._cell(judge._u, 1, s, vb)
            u_w = judge._cell(judge._u, 0, s, vb)
            d_f = judge._cell(judge._d, 1, s, vb) / scale
            d_w = judge._cell(judge._d, 0, s, vb) / scale
            big_f, big_w = g * D + d_f, g * D + d_w
            if mode == "R":
                cur += 1
                if rec >= dw and u_f <= dr and big_f <= b:
                    mix["FULL"] += 1
                    D, fh, mode = big_f, 1, "A"
                    stays.append(cur)
                    cur = 0
                elif rec >= dw and u_w <= dr and big_w <= b:
                    mix["WARM"] += 1
                    D, fh, mode = big_w, 0, "A"
                    stays.append(cur)
                    cur = 0
                else:
                    mix["MISS"] += 1
                    mix["rec_miss"] += 1
                    D = 0.0
                    rec += 1
                continue
            if fuse_l is not None and fh >= fuse_l:
                mix["MISS"] += 1
                mix["fuse"] += 1
                D, fh, bad = 0.0, 0, 0
            elif u_f <= delta and big_f <= b:
                mix["FULL"] += 1
                D, bad = big_f, 0
                fh += 1
            elif u_w <= delta and big_w <= b:
                mix["WARM"] += 1
                D, fh, bad = big_w, 0, 0
            else:
                mix["MISS"] += 1
                D, fh = 0.0, 0
                if u_f <= delta or u_w <= delta:
                    mix["debt"] += 1
                    bad = 0
                else:
                    mix["region"] += 1
                    bad += 1
                    if j is not None and bad >= j:
                        mode, bad, rec, cur = "R", 0, 0, 0
        if cur:
            stays.append(cur)
    n = mix["FULL"] + mix["WARM"] + mix["MISS"]
    cost = (mix["FULL"] * COST["FULL"] + mix["WARM"] * COST["WARM"] + mix["MISS"] * COST["MISS"]) / n
    st = np.array(stays) if stays else np.array([0])
    return {"n": n, "cost": cost, "full": 100 * mix["FULL"] / n, "warm": 100 * mix["WARM"] / n,
            "miss": 100 * mix["MISS"] / n, "rec_miss": 100 * mix["rec_miss"] / n, "fuse": mix["fuse"],
            "p50": float(np.median(st)), "p90": float(np.quantile(st, 0.9)), "max": int(st.max())}


def load_episodes(table_path):
    by = collections.defaultdict(list)
    task = {}
    for line in open(table_path):
        r = json.loads(line)
        if r.get("ref_mode") != "fresh" or r.get("v") is None:
            continue
        by[r["episode_id"]].append((int(r["step_idx"]), float(r["s"]), float(r["v"])))
        task[r["episode_id"]] = r["task_id"]
    return [(task[e], [(s, v) for _, s, v in sorted(rows)]) for e, rows in sorted(by.items())]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--artifact", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--reopen-mults", default="1.0")
    ap.add_argument("--j-bads", default="3")
    ap.add_argument("--gammas", default="1.0")
    ap.add_argument("--dwells", default="2")
    args = ap.parse_args()
    episodes = load_episodes(args.table)
    print(f"dev episodes: {len(episodes)}  decisions: {sum(len(r) for _, r in episodes)}")
    print(f"{'artifact':9s} {'reopen':>6s} {'J':>4s} {'gam':>4s} {'dw':>3s} | {'cost':>6s} {'full%':>6s} "
          f"{'warm%':>6s} {'miss%':>6s} {'recM%':>6s} {'fuse':>5s} {'stay p50/p90/max':>16s}")
    for spec in args.artifact:
        name, path = spec.split("=", 1)
        judge = CumulativeRiskJudge(path)
        for rm in [float(x) for x in args.reopen_mults.split(",")]:
            for jb in [None if x == "inf" else int(x) for x in args.j_bads.split(",")]:
                for gm in [float(x) for x in args.gammas.split(",")]:
                    for dwl in [int(x) for x in args.dwells.split(",")]:
                        r = replay(judge, episodes, gamma=gm, j_bad=jb, dwell=dwl, reopen_mult=rm)
                        print(f"{name:9s} {rm:6.2f} {str(jb):>4s} {gm:4.2f} {dwl:3d} | {r['cost']:6.2f} "
                              f"{r['full']:6.1f} {r['warm']:6.1f} {r['miss']:6.1f} {r['rec_miss']:6.1f} "
                              f"{r['fuse']:5d} {r['p50']:5.0f}/{r['p90']:4.0f}/{r['max']:4d}")


if __name__ == "__main__":
    main()
