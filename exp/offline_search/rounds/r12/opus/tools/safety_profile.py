"""R12 opus: where along an episode is a one-camera look safe? (library only)

Safe look (post-look test, lag-2 agentview): |W - WL| + first-5-step gripper-sign disagreement <= tau(N),
tau(N) = .15 * sqrt(50 / N). Profiled over held-out look rows by normalised progress (fifths), by the query's own
gripper mode (R7 two-means segmentation of its own policy chunk) and by distance to its own next/previous gripper
change. Also the excess error of W over F on safe vs unsafe looks.
"""
import numpy as np
from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.analyze_replay import load

out = {}
for m, s, n in [("pi05", "l10", 50), ("groot", "l10", 50), ("pi05", "spatial", 50), ("groot", "spatial", 50),
                ("pi05", "l10", 500), ("groot", "l10", 500)]:
    tau = 0.15 * np.sqrt(50 / n)
    Z = load(m, s, n)
    S = cm.stage_arrays(m, s)
    sig, act = Z["sig"], Z["act"]
    look = (Z["step"] % 2 == 0) & (Z["step"] > 0)
    sW, sWL, sF = Z["W|served"], Z["WL|served"], Z["F|served"]
    sc = cm.motion_err(sW, sWL, sig) + np.mean(np.sign(sW[:, :5, 6]) != np.sign(sWL[:, :5, 6]), axis=1)
    safe = sc <= tau
    dq = cm.motion_err(act, sW, sig) - cm.motion_err(act, sF, sig)
    prog = Z["prog"]
    rows = Z["row"]
    mode = S["mode"][rows]
    # distance (rows) to the nearest own gripper change
    ev = np.full(len(rows), 99)
    for e in np.unique(Z["ep"]):
        idx = np.flatnonzero(Z["ep"] == e)
        idx = idx[np.argsort(Z["step"][idx])]
        md = mode[idx]
        ch = np.flatnonzero(np.diff(md.astype(int)) != 0) + 1
        if len(ch) and (md >= 0).all():
            ev[idx] = np.abs(np.arange(len(idx))[:, None] - ch[None]).min(1)
    r = {}
    for k in range(5):
        msk = look & (prog >= k / 5) & (prog < (k + 1) / 5 + (k == 4))
        r[f"prog{k}"] = dict(n=int(msk.sum()), safe=float(safe[msk].mean()))
    for name, msk in (("open/mode0", look & (mode == 0)), ("closed/mode1", look & (mode == 1)),
                      ("unknown(failed ep)", look & (mode < 0)), ("near_event<=2", look & (ev <= 2)),
                      ("far_event>2", look & (ev > 2) & (ev < 99))):
        r[name] = dict(n=int(msk.sum()), safe=float(safe[msk].mean()) if msk.any() else None)
    r["dq_safe"] = float(dq[look & safe].mean())
    r["dq_unsafe"] = float(dq[look & ~safe].mean())
    r["safe_share"] = float(safe[look].mean())
    r["tau"] = float(tau)
    out[f"{m}_{s}_{n}"] = r
    print(f"\n{m}_{s}_{n} tau={tau:.3f} safe={r['safe_share']:.2f} dq safe {r['dq_safe']:+.4f} unsafe {r['dq_unsafe']:+.4f}")
    print("  " + "  ".join(f"{k}:{v['safe']:.2f}(n{v['n']})" for k, v in r.items() if isinstance(v, dict) and v['safe'] is not None))
cm.save_json(out, cm.OUT / "safety_profile.json")
