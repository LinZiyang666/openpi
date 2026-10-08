"""R12 opus (directive 2): learned "would the two-camera guard call here?" predictor from one-camera information.

Logistic regression (numpy IRLS, ridge 1e-2) on features available at a wrist-first look before any agentview
encode: progress deltas of W / WL2 top-1 vs the previous served top-1 (clipped), their min, previous-look call,
robot-state and wrist-code motion since the previous look, W and WL2 nearest distances (task-normalised by the
median of the arm), |W - WL2| consistency score, step. Evaluated leave-one-arm-out over the closed-loop dev arms
(same model): threshold set on the training arms for 98 % recall; reported recall and the one-camera share of
non-call looks on the held-out arm.
"""
from __future__ import annotations

import glob
from collections import defaultdict

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.dev_eval import load, score


def table(d, meta):
    sig = np.asarray(meta["sig_r"])
    by = defaultdict(list)
    for i, u in enumerate(d["uid"]):
        by[u].append(i)
    X, y = [], []
    medW = np.median(d["W_d1"]); medL = np.median(d["WL2_d1"])
    for u, idx in by.items():
        idx = sorted(idx, key=lambda i: d["step"][i])
        for a, b in zip(idx[:-1], idx[1:]):
            pp = float(d["F_top1_prog"][a])
            dW = (float(d["W_top1_prog"][b]) - pp) * max(int(d["W_top1_eplen"][b]) - 1, 1)
            dL = (float(d["WL2_top1_prog"][b]) - pp) * max(int(d["WL2_top1_eplen"][b]) - 1, 1)
            sc = score(d["S_W"][b], d["S_WL2"][b], sig)
            X.append([np.clip(dW, -5, 8), np.clip(dL, -5, 8), np.clip(min(dW, dL), -5, 8), float(dW <= 0.5),
                      float(dL <= 0.5), float(not d["hit"][a]), d["drs"][b], d["dP1"][b] / 10,
                      d["W_d1"][b] / medW, d["WL2_d1"][b] / medL, sc, d["step"][b] / 100])
            y.append(not bool(d["hit"][b]))
    return np.nan_to_num(np.asarray(X, float)), np.asarray(y, bool)


def fit_logit(X, y, lam=1e-2, it=50):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + lam * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + lam * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return lambda Xn: 1 / (1 + np.exp(-(np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w)))


def main():
    arms = {}
    for p in sorted(glob.glob(str(cm.OUT / "dev" / "*.npz"))):
        d, meta = load(p)
        arms[f"{meta['model']}_{meta['suite']}_{meta['size']}"] = table(d, meta)
    out = {}
    for model in ("pi05", "groot"):
        keys = [k for k in arms if k.startswith(model)]
        for hold in keys:
            Xtr = np.concatenate([arms[k][0] for k in keys if k != hold])
            ytr = np.concatenate([arms[k][1] for k in keys if k != hold])
            f = fit_logit(Xtr, ytr)
            ptr = f(Xtr)
            res = {}
            for target in (0.95, 0.98, 0.99):
                thr = np.quantile(ptr[ytr], 1 - target)
                Xte, yte = arms[hold]
                pte = f(Xte)
                flag = pte >= thr
                res[str(target)] = dict(recall=float(flag[yte].mean()), kept_nocall=float((~flag[~yte]).mean()))
            out[hold] = res
            print(hold, " | ".join(f"target {t}: recall {r['recall']:.3f} kept {r['kept_nocall']:.3f}" for t, r in res.items()))
    cm.save_json(out, cm.OUT / "call_predict2.json")


if __name__ == "__main__":
    main()
