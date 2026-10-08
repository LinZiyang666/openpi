"""R12 opus: summarize the k-fold replay (full vs single-camera retrieval). Library only.

Per cell-size and variant, over held-out rows at look steps (even steps, the deployed ten-control cadence):
  e10     motion error of the served (uncorrected) chunk vs the row's own stored policy chunk, sigma units (R10/R11
          yardstick, 10 steps x 6 motion channels)
  de10    paired e10 - e10(F)
  grip    gripper sign mismatch over 10 steps vs the policy chunk
  dF      RMS difference of the served chunk vs F's served chunk (motion, sigma units): size of the perturbation
  top1    share of looks whose top-1 row equals F's top-1
  ovl     kernel-weight mass of the variant's members that are also F members
  g       open-loop guard rate of the frozen only-no-progress rule (top-1 progress must advance > .5 library steps
          between consecutive looks), replayed per held-out episode on even steps
  gagree  share of looks whose guard flag equals F's
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm

PROG_EPS = 0.5


def load(model, suite, size):
    fs = sorted(glob.glob(str(cm.OUT / "replay" / f"{model}_{suite}_{size}_f*.npz")))
    if len(fs) != 5:
        return None
    zs = [np.load(f) for f in fs]
    meta = json.loads(str(zs[0]["meta_json"]))
    out = {}
    for k in zs[0].files:
        if k == "meta_json":
            continue
        out[k] = np.concatenate([z[k] for z in zs])
    out["meta"] = meta
    out["sig"] = np.asarray(meta["sig_r"])
    for extra in ("replay2", "replay3"):
        f2 = sorted(glob.glob(str(cm.OUT / extra / f"{model}_{suite}_{size}_f*.npz")))
        if len(f2) != 5:
            continue
        z2 = [np.load(f) for f in f2]
        if not np.array_equal(np.concatenate([z["row"] for z in z2]), out["row"]):
            raise ValueError(f"{extra} row order differs")
        m2 = json.loads(str(z2[0]["meta_json"]))
        for k in z2[0].files:
            if "|" in k:
                out[k] = np.concatenate([z[k] for z in z2])
        meta["variants"] = list(meta["variants"]) + list(m2["variants"])
    return out


def guard_flags(Z, v, prog_all, eplen_all):
    """Open-loop guard per row (only meaningful on look rows = even steps)."""
    top1 = Z[f"{v}|mem"][:, 0]
    p = prog_all[top1]
    L = np.maximum(eplen_all[top1] - 1, 1)
    flags = np.zeros(len(top1), bool)
    order = np.lexsort((Z["step"], Z["ep"]))
    prev = {}
    for i in order:
        if Z["step"][i] % 2:
            continue
        e = int(Z["ep"][i])
        if e in prev:
            j = prev[e]
            flags[i] = (p[i] - p[j]) * L[i] <= PROG_EPS
        prev[e] = i
    return flags


def summarize(Z, C, variants):
    sig = Z["sig"]
    look = Z["step"] % 2 == 0
    act = Z["act"]
    out = {}
    eF = cm.motion_err(act, Z["F|served"], sig)
    fF = guard_flags(Z, "F", C["prog"], C["ep_len"])
    for v in variants:
        s = Z[f"{v}|served"]
        e = cm.motion_err(act, s, sig)
        gm = cm.grip_mis(act, s)
        dF = cm.motion_err(Z["F|served"], s, sig)
        t1 = Z[f"{v}|mem"][:, 0] == Z["F|mem"][:, 0]
        memF = Z["F|mem"]
        ovl = np.array([Z[f"{v}|w"][i][np.isin(Z[f"{v}|mem"][i], memF[i])].sum() for i in range(len(s))])
        fl = guard_flags(Z, v, C["prog"], C["ep_len"])
        nz = look & (Z["step"] > 0)
        out[v] = dict(e10=float(e[nz].mean()), de10=float((e - eF)[nz].mean()),
                      de10_se=float((e - eF)[nz].std() / np.sqrt(nz.sum())), grip=float(gm[nz].mean()),
                      dF=float(dF[nz].mean()), dF_med=float(np.median(dF[nz])), top1=float(t1[nz].mean()),
                      ovl=float(ovl[nz].mean()), g=float(fl[nz].mean()), gagree=float((fl == fF)[nz].mean()),
                      g_extra=float((fl & ~fF)[nz].mean()), g_missed=float((~fl & fF)[nz].mean()), n=int(nz.sum()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="")
    a = ap.parse_args(argv)
    from exp.offline_search.rounds.r12.opus.tools.replay import CELLS, VARIANTS
    cells = CELLS if not a.cells else [(s.split(":")[0], s.split(":")[1], int(s.split(":")[2]))
                                       for s in a.cells.split(",")]
    res = {}
    Cs = {}
    for m, s, n in cells:
        Z = load(m, s, n)
        if Z is None:
            print(f"{m}_{s}_{n}: incomplete")
            continue
        if (m, s) not in Cs:
            Cs[(m, s)] = cm.load_cell(m, s)
        r = summarize(Z, Cs[(m, s)], Z["meta"]["variants"])
        res[f"{m}_{s}_{n}"] = r
        print(f"\n== {m}_{s}_{n} (look rows n={r['F']['n']})")
        print(f"{'var':6s} {'e10':>6s} {'de10':>7s} {'grip':>6s} {'dF':>6s} {'dFmed':>6s} {'top1':>6s} {'ovl':>6s} "
              f"{'g':>6s} {'gagr':>6s} {'g+':>6s} {'g-':>6s}")
        for v, x in r.items():
            print(f"{v:6s} {x['e10']:6.3f} {x['de10']:+7.3f} {x['grip']:6.3f} {x['dF']:6.3f} {x['dF_med']:6.3f} "
                  f"{x['top1']:6.3f} {x['ovl']:6.3f} {x['g']:6.3f} {x['gagree']:6.3f} {x['g_extra']:6.3f} "
                  f"{x['g_missed']:6.3f}")
    cm.save_json(res, cm.OUT / "replay_summary.json")


if __name__ == "__main__":
    main()
