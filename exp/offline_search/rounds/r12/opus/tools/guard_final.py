"""R12 opus (guard task): the frozen library risk guard ("Glib") with ONE threshold per model, alone and combined with
look-half, replayed on the closed-loop B-dev looks.

Risk model: logistic, library-fitted (guard_lib.py coefficients, B library LOEO replay, 50-episode subsets of both
suites), features need no agentview. Threshold theta per model = median over that model's knob-off dev arms of the
risk quantile that matches the deployed guard's call rate (dev-calibrated, labelled).
Per arm:
  rate, in_failed, succ_flags, D_at_calls for the deployed guard (G0) and for Glib at the frozen theta;
  look-half with Glib as the guard (M2): every look that Glib does not call and that is not a knob call is
  wrist-first; it stays one-camera if the agentview memory is <= 8 decisions old and |W - WL_lag| + grip <= .10;
  one-camera share, vision dIR (compute share), retrieval cost vs shadow chunks (50L arms).
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.dev_eval import load as load_dev, score
from exp.offline_search.rounds.r12.opus.tools.guard_design import X_of, metrics

import os
PER_CELL = os.environ.get("PER_CELL", "1") == "1"
V = 0.505
HALF = {"pi05": 0.076, "groot": 0.074}


def risk_fn(coef):
    mu, sd, w = (np.asarray(coef[k]) for k in ("mu", "sd", "w"))
    return lambda X: 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), (X - mu) / sd] @ w)))


def main():
    G = json.loads((cm.OUT / "guard_lib.json").read_text())
    dev = {p.split("/")[-1][:-4]: dict(np.load(p)) for p in sorted(glob.glob(str(cm.OUT / "guard" / "*.npz")))}
    devf = {p.split("__")[-1][:-4]: p for p in glob.glob(str(cm.OUT / "dev" / "*.npz"))}
    out = {}
    for model in ("pi05", "groot"):
        coef = G[f"{model}|coef"]
        f = risk_fn(coef)
        arms = [a for a in dev if f"_{model}_" in a]
        qs = []
        for a in arms:
            if "devknob" in a:
                continue
            z = dev[a]
            g0 = z["called"] & (z["os_reason"] == 4)
            el = z["step"] >= 4
            qs.append(float(np.quantile(f(X_of(z, coef["feats"]))[el], 1 - g0.sum() / el.sum())))
        theta_med = float(np.median(qs))
        out[f"{model}|theta_median"] = theta_med
        th_arm = {}
        for a in arms:
            if "devknob" in a:
                continue
            z = dev[a]
            g0 = z["called"] & (z["os_reason"] == 4)
            el = z["step"] >= 4                                   # the risk guard needs two looks of history
            th_arm[a] = float(np.quantile(f(X_of(z, coef["feats"]))[el], 1 - g0.sum() / el.sum()))
        out[f"{model}|theta_per_cell"] = th_arm
        for a in arms:
            z = dev[a]
            base = a.replace("devknob", "dev").replace("_periodic_ir32", "")
            theta = th_arm[base] if PER_CELL else theta_med
            g0 = z["called"] & (z["os_reason"] == 4)
            risk = f(X_of(z, coef["feats"]))
            gl = (risk >= theta) & (z["step"] >= 4)
            r = {"G0": metrics(z, g0), "Glib": metrics(z, gl)}
            # look-half with Glib as the guard
            d, meta = load_dev(devf[a])
            sig = np.asarray(meta["sig_r"])
            key = {(u, int(s)): i for i, (u, s) in enumerate(zip(d["uid"], d["step"]))}
            knob = z["called"] & (z["os_reason"] == 62)
            by = defaultdict(list)
            for j, u in enumerate(z["uid"]):
                by[u].append(j)
            half, dq = [], []
            for u, js in by.items():
                js = sorted(js, key=lambda j: z["step"][j])
                lag = 0
                for k, j in enumerate(js):
                    lag = 0 if k == 0 else lag + 2
                    i = key[(u, int(z["step"][j]))]
                    if k == 0 or gl[j] or knob[j] or lag > 8:
                        lag = 0
                        half.append(False); dq.append(0.0)
                        continue
                    L = min(8, max(2, lag))
                    ok = score(d["S_W"][i], d[f"S_WL{L}"][i], sig) <= 0.10
                    half.append(ok)
                    if ok:
                        sh = d["shadow"][i]
                        dq.append(float(cm.motion_err(sh, d[f"S_WL{L}"][i], sig) - cm.motion_err(sh, d["S_F"][i], sig))
                                  if np.isfinite(sh).all() else np.nan)
                    else:
                        lag = 0
                        dq.append(0.0)
            half = np.asarray(half); dq = np.asarray(dq, float)
            r["M2"] = dict(half=float(half.mean()), dIR_vision=-V * HALF[model] * float(half.mean()),
                           dIR_calls=V * 0.85 * float(gl.mean() - g0.mean()),
                           dq_retrieval=float(np.mean(dq)) if np.isfinite(dq).all() else None)
            out[a] = r
            print(f"{a}: G0 rate {r['G0']['rate']:.3f} inF {r['G0']['in_failed']:.2f} succF {r['G0']['succ_flags']:.3f}"
                  + (f" D {r['G0']['D_at_calls']:.3f}" if 'D_at_calls' in r['G0'] else "")
                  + f" | Glib rate {r['Glib']['rate']:.3f} inF {r['Glib']['in_failed']:.2f} succF {r['Glib']['succ_flags']:.3f}"
                  + (f" D {r['Glib']['D_at_calls']:.3f}" if 'D_at_calls' in r['Glib'] else "")
                  + f" | M2 half {r['M2']['half']:.2f} dIRv {r['M2']['dIR_vision']:+.4f} dIRcalls {r['M2']['dIR_calls']:+.4f}"
                  + (f" dq {r['M2']['dq_retrieval']:+.4f}" if r['M2']['dq_retrieval'] is not None else ""))
        print(model, "theta per cell", {k: round(v, 4) for k, v in th_arm.items()})
    cm.save_json(out, cm.OUT / ("guard_final.json" if PER_CELL else "guard_final_single_theta.json"))


if __name__ == "__main__":
    main()
