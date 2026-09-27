"""Threshold table for the deployable judge V7-on-AWM (confidence = -predicted err, one scale for all regimes).
For each model x suite and V7 run dir: pooled AURC, mixture fixed-point thresholds at h = .3/.5/.7 (after-HIT
states weighted h, after-MISS 1-h), realized stale / fresh acceptance, accepted err / bad, with and without the
V6 guard recomputed from V7's extras (stuck_n, terminal, overtime, lag; ot_still clause)."""
import json, sys, pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent
RUNS = {"cur": OUT / "v7awm/V7dc_iso__AWM_joint_cur_fcur_kr5", "big": OUT / "v7awm/V7dc_iso__AWM_joint_big_fbig"}
MS = ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
HR = [0.3, 0.5, 0.7, 0.8]
BIN = ["early", "mid", "late"]


def aurc(conf, err):
    o = np.lexsort((np.arange(conf.size), -conf)); e = err[o]
    return float((np.cumsum(e) / np.arange(1, e.size + 1)).mean())


def wq(conf, w, h):
    o = np.argsort(-conf, kind="stable"); cw = np.cumsum(w[o])
    k = min(int(np.searchsorted(cw, h, side="left")), conf.size - 1)
    return float(conf[o][k])


def load(d, cell):
    z = np.load(d / f"{cell}.npz", allow_pickle=True); j = json.load(open(d / f"{cell}.json"))
    p90 = np.asarray([j["floor"]["p90"][b] for b in BIN])[z["bin"].astype(int)]
    g = lambda k: z[f"x_{k}"] if f"x_{k}" in z.files else np.full(z["err"].shape, np.nan)
    sn, term, ot, lag = g("stuck_n"), g("terminal"), g("overtime"), g("lag")
    guard = (sn >= 2) | (term == 1) | ((ot > 1) & (lag > 5) & (sn >= 1))
    return {"conf": z["confidence"], "err": z["err"], "bad": z["err"] > p90, "grip": z["grip_mis"], "step": z["step"],
            "pred": -g("pred_err"), "guard": guard, "reg": g("regime"), "term": term == 1, "sn": sn, "ot": ot}


rows = []
for tag, d in RUNS.items():
    if not d.exists():
        continue
    for ms in MS:
        try:
            I, C = load(d, f"{ms}_inf"), load(d, f"{ms}_cache")
        except FileNotFoundError:
            continue
        st, fr = C["reg"] > 0, I["reg"] > 0
        r = {"lib": tag, "ms": ms, "aurc_stale": aurc(C["conf"][st], C["err"][st]), "aurc_fresh": aurc(I["conf"][fr], I["err"][fr]),
             "guard_stale": float(C["guard"][st].mean()), "guard_fresh": float(I["guard"][fr].mean()),
             "guard_err_stale": float(C["err"][st & C["guard"]].mean()), "unguard_err_stale": float(C["err"][st & ~C["guard"]].mean()),
             "term_stale": float(C["term"][st].mean()), "stuck2_stale": float((C["sn"][st] >= 2).mean())}
        # pooled (p=.5) AURC with one scale
        pc = np.r_[I["conf"][fr], C["conf"][st]]; pe = np.r_[I["err"][fr], C["err"][st]]
        wgt = np.r_[np.full(fr.sum(), 0.5 / fr.sum()), np.full(st.sum(), 0.5 / st.sum())]
        r["aurc_pooled"] = aurc(pc, pe)
        for gname, gi, gc in (("nog", np.zeros(I["conf"].size, bool), np.zeros(C["conf"].size, bool)), ("g", I["guard"], C["guard"])):
            for h in HR:
                ci = np.where(gi, -1e9, I["conf"]); cc = np.where(gc, -1e9, C["conf"])
                w = np.r_[np.full(ci.size, (1 - h) / ci.size), np.full(cc.size, h / cc.size)]
                tau = wq(np.r_[ci, cc], w, h)
                ai, ac = (ci >= tau) & ~gi, (cc >= tau) & ~gc
                wi, wc = (1 - h) * ai.mean(), h * ac.mean()
                mix = lambda k: float((wi * I[k][ai].mean() + wc * C[k][ac].mean()) / max(wi + wc, 1e-12))
                r[f"{gname}_h{h}"] = {"tau_pred_err": -tau, "rate_stale": float(ac.mean()), "rate_fresh": float(ai.mean()),
                                      "rate": float(wi + wc), "err": mix("err"), "bad": mix("bad"), "grip": mix("grip"),
                                      "stale_err": float(C["err"][ac].mean()), "fresh_err": float(I["err"][ai].mean()),
                                      "stale_bad": float(C["bad"][ac].mean()), "fresh_bad": float(I["bad"][ai].mean())}
        rows.append(r)

json.dump(rows, open(OUT / "v7_thresholds.json", "w"), indent=1)
for r in rows:
    print(f"\n[{r['lib']}] {r['ms']}: AURC stale {r['aurc_stale']:.3f} fresh {r['aurc_fresh']:.3f} pooled(1-scale) {r['aurc_pooled']:.3f} | guard stale {r['guard_stale']:.3f} (err {r['guard_err_stale']:.2f} vs {r['unguard_err_stale']:.2f}; terminal {r['term_stale']:.3f}, stuck2 {r['stuck2_stale']:.3f}) fresh {r['guard_fresh']:.3f}")
    for gname in ("nog", "g"):
        for h in HR:
            x = r[f"{gname}_h{h}"]
            print(f"   {gname:3s} h={h}: tau(pred_err)={x['tau_pred_err']:.3f} rate {x['rate']:.2f} (stale {x['rate_stale']:.2f} / fresh {x['rate_fresh']:.2f}) "
                  f"err {x['err']:.3f} (stale {x['stale_err']:.3f} fresh {x['fresh_err']:.3f}) bad {x['bad']:.3f} grip {x['grip']:.3f}")
