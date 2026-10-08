"""R12 opus: which online signals predict that the one-camera answer differs from the two-camera answer?

Library replay only. For every held-out look row (even step >= 2) with the previous look served by F:
targets
  big_dF    served-chunk difference |H - F| (motion RMS, sigma) above the cell's 75th percentile
  harm      e10(H) - e10(F) > .05 (the one-camera chunk is clearly further from the policy's own chunk)
  gdis      guard-flag disagreement between H and F (H's own flag vs F's)
candidate scores (all computable without the missing camera)
  dcont     |H served - continuation of the previous kernel| + first-5-step gripper disagreement
  dWWL      |W served - WL served| (does the stale agentview change the one-camera answer?)
  dWR       |W served - R served|  (does the wrist view change the state-only answer?)
  d1rel     H's nearest distance / the cell median
  disp      kernel dispersion of H's members' chunks around H's served chunk
  gvote     1 - |weighted gripper vote of H's members at step 0|
  evt       H's member mass near a gripper event (R7 stage table)
Reports ROC AUC per (cell, score, target).
"""
from __future__ import annotations

import argparse

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.cascade_sim import Cell


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1.0
    # average ties
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def features(cell, H):
    Z, sig, S = cell.Z, cell.sig, cell.S
    act = Z["act"]
    pos = {}
    for e, looks in cell.eps.items():
        for k, i in enumerate(looks):
            if k:
                pos[i] = looks[k - 1]
    idx = np.array(sorted(pos))
    prev = np.array([pos[i] for i in idx])
    d1all = Z[f"{H}|d1"][Z["step"] > 0]
    med = np.median(d1all)
    F = {k: [] for k in ("dcont", "dWWL", "dWR", "d1rel", "disp", "gvote", "evt", "dF", "de10", "gH", "gF")}
    for i, j in zip(idx, prev):
        memH, wH = Z[f"{H}|mem"][i], Z[f"{H}|w"][i].astype(np.float64)
        sH = Z[f"{H}|served"][i]
        contc, adv, ok = cell.cont(Z["F|mem"][j], Z["F|w"][j].astype(np.float64))
        F["dcont"].append(float(cm.motion_err(sH, contc, sig)) + float(np.mean(np.sign(sH[:5, 6]) != np.sign(contc[:5, 6]))))
        F["dWWL"].append(float(cm.motion_err(Z["W|served"][i], Z["WL|served"][i], sig)))
        F["dWR"].append(float(cm.motion_err(Z["W|served"][i], Z["R|served"][i], sig)))
        F["d1rel"].append(float(Z[f"{H}|d1"][i] / med))
        ch = cell.act_parent[memH]
        F["disp"].append(float(np.sqrt(np.einsum("k,ktc->", wH, ((ch[:, :10, :6] - sH[None, :10, :6]) / sig[:6]) ** 2) / 60)))
        F["gvote"].append(1 - abs(float(wH @ np.sign(ch[:, 0, 6]))))
        F["evt"].append(float(wH @ S["event"][memH]))
        F["dF"].append(float(cm.motion_err(Z["F|served"][i], sH, sig)))
        F["de10"].append(float(cm.motion_err(act[i], sH, sig) - cm.motion_err(act[i], Z["F|served"][i], sig)))
        pp = cell.prog[Z["F|mem"][j][0]]
        F["gH"].append((cell.prog[memH[0]] - pp) * max(cell.eplen[memH[0]] - 1, 1) <= 0.5)
        F["gF"].append((cell.prog[Z["F|mem"][i][0]] - pp) * max(cell.eplen[Z["F|mem"][i][0]] - 1, 1) <= 0.5)
    return {k: np.asarray(v) for k, v in F.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pi05:l10:50,pi05:spatial:50,groot:l10:50,groot:spatial:50,pi05:l10:500,groot:l10:500")
    ap.add_argument("--H", default="W")
    a = ap.parse_args(argv)
    out = {}
    for spec in a.cells.split(","):
        m, s, n = spec.split(":")
        cell = Cell(m, s, int(n))
        X = features(cell, a.H)
        tg = {"big_dF": X["dF"] > np.quantile(X["dF"], .75), "harm": X["de10"] > .05, "gdis": X["gH"] != X["gF"]}
        key = f"{m}_{s}_{n}"
        out[key] = {}
        print(f"\n== {key} H={a.H} n={len(X['dF'])} harm-rate={tg['harm'].mean():.3f} gdis={tg['gdis'].mean():.3f}")
        print(f"{'score':7s} " + " ".join(f"{t:>7s}" for t in tg))
        for sc in ("dcont", "dWWL", "dWR", "d1rel", "disp", "gvote", "evt"):
            r = {t: auc(X[sc], y) for t, y in tg.items()}
            out[key][sc] = r
            print(f"{sc:7s} " + " ".join(f"{r[t]:7.3f}" for t in tg))
        # combined: rank-average of dcont and dWWL
        from scipy.stats import rankdata
        comb = rankdata(X["dcont"]) + rankdata(X["dWWL"])
        r = {t: auc(comb, y) for t, y in tg.items()}
        out[key]["dcont+dWWL"] = r
        print(f"{'c+WWL':7s} " + " ".join(f"{r[t]:7.3f}" for t in tg))
    cm.save_json(out, cm.OUT / f"doubt_auc_{a.H}.json")


if __name__ == "__main__":
    main()
