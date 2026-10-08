"""R12 opus (guard task): hybrid agentview-free guards, rate-matched to the deployed guard on each dev arm.
H1: one-camera stall (wrist-only top-1 progress advance <= .5) gated by risk >= its eligible-look quantile qa.
H2: H1 OR risk >= qb (qb chosen so the total rate equals the deployed guard's).  Risk = library divergence model.
Reported: rate, share in failed episodes, flags per look in successful episodes, D at calls, median first-flag step
in rescuable failed episodes (deployed guard G0 for reference)."""
import glob, json
from collections import defaultdict
import numpy as np
from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.guard_final import risk_fn
from exp.offline_search.rounds.r12.opus.tools.guard_design import X_of, metrics

G = json.loads((cm.OUT / "guard_lib.json").read_text())


def first_step(z, flag):
    by = defaultdict(list)
    for j, u in enumerate(z["uid"]):
        by[u].append(j)
    fs = []
    for u, js in by.items():
        js = sorted(js, key=lambda j: z["step"][j])
        if z["fail"][js[0]] and z["pure_ok"][js[0]]:
            idx = np.flatnonzero(flag[js])
            fs.append(int(z["step"][js[idx[0]]]) if len(idx) else 999)
    return float(np.median(fs)) if fs else np.nan


out = {}
for p in sorted(glob.glob(str(cm.OUT / "guard" / "r11_dev_*.npz"))):
    arm = p.split("/")[-1][:-4]
    z = dict(np.load(p)); model = "pi05" if "_pi05_" in arm else "groot"
    coef = G[f"{model}|coef"]
    risk = risk_fn(coef)(X_of(z, coef["feats"]))
    el = z["step"] >= 4
    g0 = z["called"] & (z["os_reason"] == 4)
    stall = (np.nan_to_num(z["W_adv"].astype(float), nan=9.0) <= 0.5) & el
    res = {"G0": dict(metrics(z, g0), first=first_step(z, g0))}
    for qa in (0.25, 0.5, 0.75):
        tha = np.quantile(risk[el], qa)
        h1 = stall & (risk >= tha)
        res[f"H1_q{qa}"] = dict(metrics(z, h1), first=first_step(z, h1))
        need = g0.sum() - h1.sum()
        if need > 0:
            rest = el & ~h1
            thb = np.sort(risk[rest])[::-1][min(int(need), rest.sum()) - 1]
            h2 = h1 | (rest & (risk >= thb))
        else:
            h2 = h1
        res[f"H2_q{qa}"] = dict(metrics(z, h2), first=first_step(z, h2))
    out[arm] = res
    print(f"\n== {arm}")
    for g, m in res.items():
        print(f"  {g:9s} rate {m['rate']:.3f} inF {m['in_failed']:.2f} succF {m['succ_flags']:.3f} first {m['first']:.0f}"
              + (f" D {m['D_at_calls']:.3f}" if 'D_at_calls' in m else ""))
cm.save_json(out, cm.OUT / "guard_hybrid.json")
