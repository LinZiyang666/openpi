"""R12 opus (guard task): WHEN do the deployed guard and the risk guard fire in failing episodes? (dev, rate-matched)"""
import glob, json
from collections import defaultdict
import numpy as np
from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.guard_final import risk_fn
from exp.offline_search.rounds.r12.opus.tools.guard_design import X_of

G = json.loads((cm.OUT / "guard_lib.json").read_text())
F = json.loads((cm.OUT / "guard_final.json").read_text())
out = {}
for p in sorted(glob.glob(str(cm.OUT / "guard" / "r11_dev_*.npz"))):
    arm = p.split("/")[-1][:-4]
    z = dict(np.load(p)); model = "pi05" if "_pi05_" in arm else "groot"
    coef = G[f"{model}|coef"]; theta = F[f"{model}|theta_per_cell"][arm]
    gl = (risk_fn(coef)(X_of(z, coef["feats"])) >= theta) & (z["step"] >= 4)
    g0 = z["called"] & (z["os_reason"] == 4)
    by = defaultdict(list)
    for j, u in enumerate(z["uid"]):
        by[u].append(j)
    R = {g: dict(first=[], early_share=[], n_fail_eps=0) for g in ("G0", "Glib")}
    for u, js in by.items():
        js = sorted(js, key=lambda j: z["step"][j])
        if not (z["fail"][js[0]] and z["pure_ok"][js[0]]):
            continue
        for g, fl in (("G0", g0), ("Glib", gl)):
            f = fl[js]
            R[g]["n_fail_eps"] += 1
            idx = np.flatnonzero(f)
            R[g]["first"].append(int(z["step"][js[idx[0]]]) if len(idx) else 999)
            half = len(js) // 2
            R[g]["early_share"].append(float(f[:half].sum() / max(f.sum(), 1)))
    out[arm] = {g: dict(median_first_step=float(np.median(v["first"])), mean_early_share=float(np.mean(v["early_share"])),
                        eps=v["n_fail_eps"]) for g, v in R.items()}
    print(arm, {g: {k: round(x, 2) for k, x in v.items()} for g, v in out[arm].items()})
cm.save_json(out, cm.OUT / "guard_timing.json")
