"""R12 opus (guard task): agentview-free guards judged on their own merit, at matched call rate, on closed-loop B-dev
looks (out/guard/*.npz).

Guards (all decide at a look from the episode's own history; none needs the agentview unless noted):
  G0   deployed no-progress rule on the two-camera top-1 (reference; = recorded guard calls)
  G1   the same rule on the wrist-only top-1 (naive look-half guard)
  G2   learned risk: logistic model of P(episode fails) from [wrist-only distance to the library, elapsed step,
       lag of the wrist top-1 behind its demo, recent xyz speed, xyz direction reversals, action-vs-state mismatch,
       previous look called, wrist-only progress advance]; fitted leave-one-arm-out on the other dev arms of the
       same model; calls at looks whose risk exceeds the arm's matched-rate quantile
  G3   G1 OR risk above a threshold (union, same total rate)
  G2c  G2 plus the |W - WL2| consistency score (needs the remembered agentview: look-half setting)
Metrics at the deployed guard's call rate on each held-out arm:
  D_at_calls      mean cache-vs-policy divergence at the flagged looks (50L arms; what a call would change)
  in_failed       share of flagged looks in episodes that failed
  rescue_cov      share of rescuable failed episodes (pure policy succeeds from that init) with >= 1 flag in their
                  first 60 % of looks
  succ_flags      flags per look in successful episodes (IR spent where the cache succeeded anyway)
"""
from __future__ import annotations

import glob
import json

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm

FEAT = ["W_d1rel", "step", "lagrel", "speed2", "osc", "mismatch", "prevcall", "W_adv"]
FEAT_C = FEAT + ["cons"]


def X_of(z, feats):
    cols = []
    for f in feats:
        v = z[f].astype(float)
        if f == "W_adv":
            v = np.clip(np.nan_to_num(v, nan=5.0), -5, 8)
        elif f == "step":
            v = v / 100.0
        elif f == "mismatch":
            v = np.log1p(np.nan_to_num(v, nan=1.0))
        cols.append(np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(v).any() else 0.0))
    return np.stack(cols, 1)


def fit_logit(X, y, lam=1e-2, it=40):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + lam * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + lam * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return (lambda Xn: 1 / (1 + np.exp(-(np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w)))), dict(
        mu=mu.tolist(), sd=sd.tolist(), w=w.tolist())


def metrics(z, flag):
    fail, pure, D = z["fail"], z["pure_ok"], z["D"]
    uid = z["uid"]
    out = dict(rate=float(flag.mean()), in_failed=float(fail[flag].mean()) if flag.any() else np.nan,
               succ_flags=float(flag[~fail].mean()))
    if np.isfinite(D).any():
        out["D_at_calls"] = float(np.nanmean(D[flag]))
    resc = []
    for u in np.unique(uid[fail & pure]):
        m = np.flatnonzero(uid == u)
        k = max(1, int(np.ceil(0.6 * len(m))))
        resc.append(bool(flag[m[:k]].any()))
    out["rescue_cov"] = float(np.mean(resc)) if resc else np.nan
    return out


def at_rate(score, rate):
    thr = np.quantile(score, 1 - rate)
    return score >= thr


def main():
    arms = {p.split("/")[-1][:-4]: dict(np.load(p)) for p in sorted(glob.glob(str(cm.OUT / "guard" / "*.npz")))}
    res, models = {}, {}
    for model in ("pi05", "groot"):
        keys = [k for k in arms if f"_{model}_" in k]
        for hold in keys:
            z = arms[hold]
            g0 = z["called"] & (z["os_reason"] == 4)
            rate = float(g0.mean())
            g1 = np.nan_to_num(z["W_adv"].astype(float), nan=9.0) <= 0.5
            tr = [k for k in keys if k != hold]
            out = {"G0": metrics(z, g0), "G1": metrics(z, g1)}
            for name, feats in (("G2", FEAT), ("G2c", FEAT_C)):
                Xtr = np.concatenate([X_of(arms[k], feats) for k in tr])
                ytr = np.concatenate([arms[k]["fail"] for k in tr]).astype(float)
                f, coef = fit_logit(Xtr, ytr)
                risk = f(X_of(z, feats))
                out[name] = metrics(z, at_rate(risk, rate))
                models[f"{hold}|{name}"] = dict(feats=feats, **coef)
                if name == "G2":
                    # union with the naive one-camera stall rule at the same total rate
                    for share in (0.5,):
                        r2 = risk.copy(); r2[g1] = np.inf
                        flag = at_rate(r2, rate) if g1.mean() < rate else g1
                        out["G3"] = metrics(z, flag)
            rnd = np.random.default_rng(0).random(len(g0)) < rate
            out["random"] = metrics(z, rnd)
            res[hold] = out
            print(f"\n== {hold} (deployed guard rate {rate:.3f})")
            for g, m in out.items():
                print(f"  {g:7s} rate {m['rate']:.3f} in_failed {m['in_failed']:.2f} rescue_cov {m['rescue_cov']:.2f} "
                      f"succ_flags {m['succ_flags']:.3f}" + (f" D_at_calls {m['D_at_calls']:.3f}" if "D_at_calls" in m else ""))
    cm.save_json(dict(results=res, models=models), cm.OUT / "guard_design.json")


if __name__ == "__main__":
    main()
