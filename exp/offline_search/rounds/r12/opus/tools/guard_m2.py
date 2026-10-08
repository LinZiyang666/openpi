"""R12 opus (guard task): frozen guard "H2" (one-camera stall gated by risk, OR high risk) with per-cell dev-matched
thresholds, alone (as a replacement of the deployed guard on full looks) and as the guard of look-half (M2).
theta_a = 25th percentile of the risk over eligible looks (step >= 4) of the cell's knob-off dev arm;
theta_b = the risk level that makes H2's rate equal the deployed guard's rate on that arm (dev-calibrated).
M2: a look that H2 does not call and that is not a knob call is wrist-first; it stays one-camera when the agentview
memory is <= 8 decisions old and |W - WL_lag| + grip <= .10; otherwise it is completed."""
import glob, json
from collections import defaultdict
import numpy as np
from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.guard_final import risk_fn
from exp.offline_search.rounds.r12.opus.tools.guard_design import X_of, metrics
from exp.offline_search.rounds.r12.opus.tools.guard_hybrid import first_step
from exp.offline_search.rounds.r12.opus.tools.dev_eval import load as load_dev, score

G = json.loads((cm.OUT / "guard_lib.json").read_text())
V = 0.505
HALF = {"pi05": 0.076, "groot": 0.074}
dev = {p.split("/")[-1][:-4]: dict(np.load(p)) for p in sorted(glob.glob(str(cm.OUT / "guard" / "*.npz")))}
devf = {p.split("__")[-1][:-4]: p for p in glob.glob(str(cm.OUT / "dev" / "*.npz"))}


def h2(z, risk, ta, tb):
    el = z["step"] >= 4
    stall = (np.nan_to_num(z["W_adv"].astype(float), nan=9.0) <= 0.5) & el
    return (stall & (risk >= ta)) | (el & (risk >= tb))


thetas = {}
for arm, z in dev.items():
    if "devknob" in arm:
        continue
    model = "pi05" if "_pi05_" in arm else "groot"
    coef = G[f"{model}|coef"]
    risk = risk_fn(coef)(X_of(z, coef["feats"]))
    el = z["step"] >= 4
    ta = float(np.quantile(risk[el], 0.25))
    g0 = z["called"] & (z["os_reason"] == 4)
    cand = np.sort(np.unique(risk[el]))[::-1]
    lo, hi = 0, len(cand) - 1
    while lo < hi:                                # smallest tb (largest rate) with rate <= g0 rate
        mid = (lo + hi + 1) // 2
        if h2(z, risk, ta, cand[mid]).sum() <= g0.sum():
            lo = mid
        else:
            hi = mid - 1
    thetas[arm] = dict(theta_a=ta, theta_b=float(cand[lo]))
out = {"thetas": thetas}
for arm, z in dev.items():
    model = "pi05" if "_pi05_" in arm else "groot"
    base = arm.replace("devknob", "dev").replace("_periodic_ir32", "")
    th = thetas[base]
    coef = G[f"{model}|coef"]
    risk = risk_fn(coef)(X_of(z, coef["feats"]))
    g0 = z["called"] & (z["os_reason"] == 4)
    gh = h2(z, risk, th["theta_a"], th["theta_b"])
    knob = z["called"] & (z["os_reason"] == 62)
    d, meta = load_dev(devf[arm])
    sig = np.asarray(meta["sig_r"])
    key = {(u, int(s)): i for i, (u, s) in enumerate(zip(d["uid"], d["step"]))}
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
            if k == 0 or gh[j] or knob[j] or lag > 8:
                lag = 0; half.append(False); dq.append(0.0); continue
            L = min(8, max(2, lag))
            ok = score(d["S_W"][i], d[f"S_WL{L}"][i], sig) <= 0.10
            half.append(ok)
            if ok:
                sh = d["shadow"][i]
                dq.append(float(cm.motion_err(sh, d[f"S_WL{L}"][i], sig) - cm.motion_err(sh, d["S_F"][i], sig))
                          if np.isfinite(sh).all() else np.nan)
            else:
                lag = 0; dq.append(0.0)
    half = np.asarray(half); dq = np.asarray(dq, float)
    r = dict(G0=dict(metrics(z, g0), first=first_step(z, g0)), H2=dict(metrics(z, gh), first=first_step(z, gh)),
             M2=dict(half=float(half.mean()), dIR_vision=-V * HALF[model] * float(half.mean()),
                     dIR_calls=V * (0.848 if model == "pi05" else 0.852) * float(gh.mean() - g0.mean()),
                     dq_retrieval=float(np.mean(dq)) if np.isfinite(dq).all() else None))
    out[arm] = r
    print(f"{arm}: G0 rate {r['G0']['rate']:.3f} inF {r['G0']['in_failed']:.2f} succF {r['G0']['succ_flags']:.3f} first {r['G0']['first']:.0f}"
          + (f" D {r['G0']['D_at_calls']:.3f}" if 'D_at_calls' in r['G0'] else "")
          + f" | H2 rate {r['H2']['rate']:.3f} inF {r['H2']['in_failed']:.2f} succF {r['H2']['succ_flags']:.3f} first {r['H2']['first']:.0f}"
          + (f" D {r['H2']['D_at_calls']:.3f}" if 'D_at_calls' in r['H2'] else "")
          + f" | M2 half {r['M2']['half']:.2f} dIRv {r['M2']['dIR_vision']:+.4f} dIRcalls {r['M2']['dIR_calls']:+.4f}"
          + (f" dq {r['M2']['dq_retrieval']:+.4f}" if r['M2']['dq_retrieval'] is not None else ""))
cm.save_json(out, cm.OUT / "guard_m2.json")
print(json.dumps(thetas, indent=0))
