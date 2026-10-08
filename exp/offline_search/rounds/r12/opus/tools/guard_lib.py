"""R12 opus (guard task): a LIBRARY-fitted divergence guard, tested on closed-loop B-dev looks.

Library side (k-fold LOEO replay outputs of replay.py, 50-episode subsets; library rows only): for every held-out
look row (even step >= 4) the same agentview-free features as guard_table.py (wrist-only distance to the library,
elapsed step, lag of the wrist top-1 behind its demo, recent xyz speed, direction reversals, action-vs-state
mismatch, wrist-only progress advance) and the target D_lib = divergence between the two-camera cache chunk and the
row's own stored policy chunk (what a call would change). A logistic model of "D in the top quarter" is fitted per
model on the two 50-episode suites (L10 + Spatial). Then, on the closed-loop dev arms of the same model, it is used
as a guard at the deployed guard's call rate and compared with the deployed guard (G0) and the dev-fitted failure
guard (G2 from guard_design.py).
"""
from __future__ import annotations

import glob
import json

import numpy as np
from scipy.stats import rankdata

from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.analyze_replay import load as load_replay
from exp.offline_search.rounds.r12.opus.tools.guard_design import X_of, fit_logit, metrics, at_rate
from exp.offline_search.rounds.r12.opus.tools.guard_table import mismatch_map
from exp.offline_search.rounds.r12.opus.tools.dev_eval import score

FEAT = ["W_d1rel", "step", "lagrel", "speed2", "osc", "mismatch", "W_adv"]


def lib_table(model, suite, size=50):
    Z = load_replay(model, suite, size)
    C = cm.load_cell(model, suite)
    B, scale = mismatch_map(model, suite)
    sig = Z["sig"]
    pos = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(Z["ep"], Z["step"]))}
    med = np.median(Z["W|d1"][Z["step"] > 0])
    R = {k: [] for k in FEAT + ["D", "cons"]}
    for i in range(len(Z["row"])):
        e, s = int(Z["ep"][i]), int(Z["step"][i])
        if s % 2 or s < 4:
            continue
        j2, j4, j6 = pos.get((e, s - 2)), pos.get((e, s - 4)), pos.get((e, s - 6))
        if j2 is None or j4 is None:
            continue
        rs = lambda k: C["rs"][Z["row"][k]]
        top = Z["W|mem"][i][0]
        el = int(C["ep_len"][top]); prog = float(C["prog"][top])
        progF2 = float(C["prog"][Z["F|mem"][j2][0]])
        dv = [np.linalg.norm(rs(i)[:3] - rs(j2)[:3]), np.linalg.norm(rs(j2)[:3] - rs(j4)[:3])]
        vec = [rs(i)[:3] - rs(j2)[:3], rs(j2)[:3] - rs(j4)[:3]] + ([rs(j4)[:3] - rs(j6)[:3]] if j6 is not None else [])
        j1 = pos.get((e, s - 1))
        a_prev = Z["act"][j2][:5, :3].sum(0)
        a_mid = Z["act"][j1][:5, :3].sum(0) if j1 is not None else a_prev
        x = np.r_[1.0, a_prev, a_mid, rs(j2)]
        R["W_d1rel"].append(float(Z["W|d1"][i] / med)); R["step"].append(s)
        R["lagrel"].append((s - prog * max(el - 1, 1)) / max(el, 1))
        R["speed2"].append(float(np.mean(dv)))
        R["osc"].append(sum(float(np.dot(vec[k], vec[k + 1]) < 0) for k in range(len(vec) - 1)))
        R["mismatch"].append(float(np.linalg.norm(rs(i)[:3] - rs(j2)[:3] - x @ B)) / scale)
        R["W_adv"].append((prog - progF2) * max(el - 1, 1))
        R["cons"].append(score(Z["W|served"][i], Z["WL|served"][i], sig))
        R["D"].append(float(cm.motion_err(Z["act"][i], Z["F|served"][i], sig)))
    return {k: np.asarray(v, float) for k, v in R.items()}


def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    res = {}
    dev = {p.split("/")[-1][:-4]: dict(np.load(p)) for p in sorted(glob.glob(str(cm.OUT / "guard" / "*.npz")))}
    for model in ("pi05", "groot"):
        T = [lib_table(model, s) for s in ("l10", "spatial")]
        L = {k: np.concatenate([t[k] for t in T]) for k in T[0]}
        y = L["D"] > np.quantile(L["D"], .75)
        f, coef = fit_logit(X_of(L, FEAT), y.astype(float))
        res[f"{model}|lib_auc_in_sample"] = auc(f(X_of(L, FEAT)), y)
        for arm, z in dev.items():
            if f"_{model}_" not in arm:
                continue
            g0 = z["called"] & (z["os_reason"] == 4)
            risk = f(X_of(z, FEAT))
            out = {"G0": metrics(z, g0), "Glib": metrics(z, at_rate(risk, float(g0.mean())))}
            if np.isfinite(z["D"]).any():
                Dd = z["D"]; ok = np.isfinite(Dd)
                out["auc_Dhi_dev"] = auc(risk[ok], Dd[ok] > np.quantile(Dd[ok], .75))
            out["auc_fail_dev"] = auc(risk, z["fail"].astype(bool))
            res[arm] = out
            print(f"{arm}: G0 in_failed {out['G0']['in_failed']:.2f} succ_flags {out['G0']['succ_flags']:.3f}"
                  + (f" D {out['G0']['D_at_calls']:.3f}" if 'D_at_calls' in out['G0'] else "")
                  + f" | Glib in_failed {out['Glib']['in_failed']:.2f} succ_flags {out['Glib']['succ_flags']:.3f}"
                  + (f" D {out['Glib']['D_at_calls']:.3f}" if 'D_at_calls' in out['Glib'] else "")
                  + f" | AUC fail {out['auc_fail_dev']:.2f}" + (f" D_hi {out['auc_Dhi_dev']:.2f}" if 'auc_Dhi_dev' in out else ""))
        res[f"{model}|coef"] = dict(feats=FEAT, **coef)
        print(model, "library in-sample AUC", round(res[f'{model}|lib_auc_in_sample'], 3))
    cm.save_json(res, cm.OUT / "guard_lib.json")


if __name__ == "__main__":
    main()
