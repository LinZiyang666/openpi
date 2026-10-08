"""R12 opus: tau selection grid and knob interplay for the wrist-first consistency cascade (library only).

For each cell-size: the full (deployed) sequence and the cascade at a grid of tau, with the periodic layer-4 knob
off and at gap caps of 4, 3 and 2 looks (a policy call at least every 8 / 6 / 4 decisions). Owner IR per slot uses
v = .505. Selection rule (frozen before any closed-loop result, library-only): per cell-size, the largest tau in the
grid whose knob-off retrieval-quality cost dq (mean over looks of e10(served half) - e10(F)) is <= .002.
"""
from __future__ import annotations

import argparse

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.cascade_sim import Cell
from exp.offline_search.rounds.r12.opus.tools.sim2 import simulate

TAUS = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3)
KNOBS = (0, 4, 3, 2)
DQ_MAX = 0.002


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pi05:l10:50,pi05:spatial:50,groot:l10:50,groot:spatial:50,"
                                       "pi05:l10:200,groot:l10:200,pi05:spatial:200,groot:spatial:200,"
                                       "pi05:l10:500,groot:l10:500,pi05:spatial:500,groot:spatial:500")
    a = ap.parse_args(argv)
    res = {}
    for spec in a.cells.split(","):
        m, s, n = spec.split(":")
        cell = Cell(m, s, int(n))
        key = f"{m}_{s}_{n}"
        res[key] = {}
        for K in KNOBS:
            r = {"full": simulate(cell, "full", knob=K), "all_own": simulate(cell, "all", "W", guard="own", knob=K)}
            for t in TAUS:
                r[f"cw{t}"] = simulate(cell, "cw", "W", thr=t, guard="own", knob=K)
            res[key][f"K{K}"] = r
        off = res[key]["K0"]
        ok = [t for t in TAUS if off[f"cw{t}"]["dq"] <= DQ_MAX]
        tau = max(ok) if ok else None
        res[key]["tau_selected"] = tau
        print(f"\n== {key}: selected tau = {tau}")
        print(f"{'K':3s} {'rule':8s} {'half':>5s} {'comp':>5s} {'g':>6s} {'irO':>6s} {'dIR_O':>7s} {'dIR_F':>7s} {'dq':>7s} "
              f"{'dq_se':>6s} {'de_eff':>7s}")
        for K in KNOBS:
            r = res[key][f"K{K}"]
            for name in ["full", "all_own"] + [f"cw{t}" for t in TAUS]:
                x = r[name]
                print(f"{K:<3d} {name:8s} {x['half']:5.2f} {x['comp']:5.2f} {x['g']:6.3f} {x['ir_owner']:6.3f} "
                      f"{x['ir_owner'] - x['irF_owner']:+7.4f} {x['ir_fair'] - x['irF_fair']:+7.4f} {x['dq']:+7.4f} "
                      f"{x['dq_se']:6.4f} {x['de_eff']:+7.4f}")
        cm.save_json(res, cm.OUT / "sim3.json")


if __name__ == "__main__":
    main()
