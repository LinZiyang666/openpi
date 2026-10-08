"""R12 opus: can the half/full choice be made BEFORE the look (no completion-for-retrieval path)? Library only.

Score s_i = |W - WL(lag 2)| + first-5-step gripper-sign disagreement at look i (both computable after look i).
Pre-look rule: look i is wrist-only iff s_{i-1} <= tau (persistence of the previous look's consistency).
Reported vs the post-look cascade at the same tau (half iff s_i <= tau): half share, dq = mean over looks of
(e10(W) - e10(F)) on half looks, and the AUC of s_{i-1} for predicting s_i > tau.
"""
import numpy as np
from scipy.stats import rankdata
from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.analyze_replay import load

out = {}
for m, s, n in [("pi05", "l10", 50), ("pi05", "spatial", 50), ("groot", "l10", 50), ("groot", "spatial", 50),
                ("pi05", "l10", 500), ("groot", "l10", 500)]:
    Z = load(m, s, n)
    sig, act = Z["sig"], Z["act"]
    sW, sWL, sF = Z["W|served"], Z["WL|served"], Z["F|served"]
    sc = cm.motion_err(sW, sWL, sig) + np.mean(np.sign(sW[:, :5, 6]) != np.sign(sWL[:, :5, 6]), axis=1)
    dq = cm.motion_err(act, sW, sig) - cm.motion_err(act, sF, sig)
    order = np.lexsort((Z["step"], Z["ep"]))
    prev = np.full(len(sc), -1)
    last = {}
    for i in order:
        if Z["step"][i] % 2:
            continue
        e = int(Z["ep"][i])
        if e in last:
            prev[i] = last[e]
        last[e] = i
    ok = (prev >= 0) & (Z["step"] >= 4)
    res = {}
    for t in (0.1, 0.15, 0.2):
        post = ok & (sc <= t)
        pre = ok & (sc[np.maximum(prev, 0)] <= t)
        y = sc[ok] > t
        r = rankdata(sc[np.maximum(prev, 0)][ok])
        auc = float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))
        res[str(t)] = dict(post_half=float(post.sum() / ok.sum()), post_dq=float(dq[post].sum() / ok.sum()),
                           pre_half=float(pre.sum() / ok.sum()), pre_dq=float(dq[pre].sum() / ok.sum()), auc_prev=auc)
    out[f"{m}_{s}_{n}"] = res
    print(f"{m}_{s}_{n}: " + " | ".join(f"tau {t}: post {d['post_half']:.2f}/{d['post_dq']:+.4f} pre {d['pre_half']:.2f}/{d['pre_dq']:+.4f} auc {d['auc_prev']:.2f}" for t, d in res.items()))
cm.save_json(out, cm.OUT / "prelook.json")
