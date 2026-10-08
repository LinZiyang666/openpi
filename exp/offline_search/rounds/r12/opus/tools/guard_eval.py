"""R12 opus (guard task): how useful is the deployed no-progress guard, and which agentview-free signals carry the
same information? Closed-loop B-dev looks (out/guard/*.npz)."""
import glob, json
import numpy as np
from scipy.stats import rankdata
from exp.offline_search.rounds.r12.opus.tools import common as cm


def auc(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(s); n1 = y.sum(); n0 = (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


SIG = ["speed", "speed2", "rot", "gripw", "dgrip", "mismatch", "osc", "cmdgrip_incons", "W_adv", "W_d1rel", "lagrel",
       "cons", "step", "prevcall", "F_adv"]
NEG = {"speed", "speed2", "rot", "W_adv", "F_adv"}            # low value = suspicious
out = {}
for p in sorted(glob.glob(str(cm.OUT / "guard" / "*.npz"))):
    z = np.load(p)
    arm = p.split("/")[-1][:-4]
    called, fail, pure = z["called"], z["fail"], z["pure_ok"]
    guard = called & (z["os_reason"] == 4)
    D = z["D"]
    first = np.r_[True, z["uid"][1:] != z["uid"][:-1]]
    res = dict(looks=int(len(called)), eps=int(first.sum()), fail_eps=float(fail[first].mean()),
               rescuable_eps=float((fail & pure)[first].mean()), call_rate=float(called.mean()),
               guard_rate=float(guard.mean()), guard_in_failed=float(fail[guard].mean()) if guard.any() else None,
               looks_in_failed=float(fail.mean()))
    if np.isfinite(D).any():
        hi = D > np.nanquantile(D, .75)
        res.update(D_mean=float(np.nanmean(D)), D_at_guard=float(np.nanmean(D[guard])), D_at_nonguard=float(np.nanmean(D[~called])),
                   D_failed=float(np.nanmean(D[fail])), D_success=float(np.nanmean(D[~fail])))
    a = {}
    for s in SIG:
        v = z[s].astype(float) * (-1 if s in NEG else 1)
        a[s] = dict(fail=auc(v, fail), guard=auc(v, guard))
        if np.isfinite(D).any():
            a[s]["D_hi"] = auc(v, hi)
    res["auc"] = a
    out[arm] = res
    print(f"\n== {arm}: {res['eps']} eps, fail {res['fail_eps']:.2f} (rescuable {res['rescuable_eps']:.2f}), calls {res['call_rate']:.3f}, "
          f"guard {res['guard_rate']:.3f}, guard calls in failed eps {res['guard_in_failed']}, looks in failed eps {res['looks_in_failed']:.2f}")
    if "D_mean" in res:
        print(f"   D mean {res['D_mean']:.3f} | at guard calls {res['D_at_guard']:.3f} | at non-calls {res['D_at_nonguard']:.3f} | failed eps {res['D_failed']:.3f} | success eps {res['D_success']:.3f}")
    print("   AUC (fail / guard" + (" / D_hi)" if "D_mean" in res else ")"))
    print("   " + "  ".join(f"{s}:{v['fail']:.2f}/{v['guard']:.2f}" + (f"/{v['D_hi']:.2f}" if 'D_hi' in v else "") for s, v in a.items()))
cm.save_json(out, cm.OUT / "guard_eval.json")
