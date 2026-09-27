"""Rules on top of the whitened metric (big library). Usage: p6_rules.py <cell>
cache cells: drift-gated modality switch, success-only candidates, action-cluster classification, per-decision bound.
inf cells: fresh-regime fusion of continuity with the whitened distance.
"""
import json, pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from p1_diag import load_lib, load_q, sigma_d, unit, err_of, aurc, ROOT, SCR, BIG
from p3_white import l2score, fit_transform, kern_synth


def main(cell):
    t0 = time.time()
    ms = cell.rsplit("_", 1)[0]; arm = cell.rsplit("_", 1)[1]; model = ms.split("_")[0]
    L = load_lib(ms, BIG[model]); Q = load_q(cell); sig = sigma_d(ms)
    N = Q["N"]; step = Q["step"]
    heads = (L["a5"] / sig).reshape(L["L"], 35)
    names = ["joint", "vis", "state", "gate_vis_if_drift", "gate_vis_if_drift2", "best_of_joint_vis", "succ_cands", "clu32", "clu64", "clu64_kern",
             "fresh_cont", "fresh_l0p5", "fresh_l1", "fresh_l2", "fresh_l4", "fresh_gate"]
    err = {n: np.full(N, np.nan) for n in names}
    dst = np.full(N, np.nan)
    def feats(P0, P1, rs, vis=True, st=True):
        parts = ([P0[:, :64], P1[:, :64]] if vis else []) + ([rs] if st else [])
        return np.concatenate(parts, 1).astype(np.float64)
    rng = np.random.default_rng(0)
    for t in np.unique(Q["task_id"]):
        qi = np.where(Q["task_id"] == t)[0]; r = np.where(L["task_id"] == t)[0]
        ca5 = L["a5"][r]; ainf = Q["a_inf"][qi]; ep = L["episode"][r]; st = L["step"][r]
        RS = L["rs8"][r]
        DL = np.sqrt(np.maximum((RS ** 2).sum(1)[:, None] - 2 * RS @ RS.T + (RS ** 2).sum(1)[None], 0)); DL[ep[:, None] == ep[None]] = np.inf
        s_d = float(np.median(DL.min(1))) + 1e-6
        drs = np.sqrt(np.maximum((Q["rs8"][qi] ** 2).sum(1)[:, None] - 2 * Q["rs8"][qi] @ RS.T + (RS ** 2).sum(1)[None], 0))
        dst[qi] = drs.min(1) / s_d
        S = {}
        for n, (v, s_) in (("joint", (True, True)), ("vis", (True, False)), ("state", (False, True))):
            Xl = feats(L["P0"][r], L["P1"][r], RS, v, s_); Xq = feats(Q["P0"][qi], Q["P1"][qi], Q["rs8"][qi], v, s_)
            mean, std, W = fit_transform(Xl, heads[r], ep, st, nn=3, lam=0.1)
            S[n] = l2score(((Xq - mean) / std) @ W, ((Xl - mean) / std) @ W)
            a, o, _ = kern_synth(S[n], ca5)
            err[n][qi] = err_of(a, ainf, sig)
        if arm == "cache":
            d = dst[qi]
            err["gate_vis_if_drift"][qi] = np.where(d > 3.0, err["vis"][qi], err["joint"][qi])
            err["gate_vis_if_drift2"][qi] = np.where(d > 2.0, err["vis"][qi], err["joint"][qi])
            err["best_of_joint_vis"][qi] = np.minimum(err["vis"][qi], err["joint"][qi])
            # success-only candidates
            ok = L["success"][r].astype(bool)
            if ok.sum() >= 16:
                a, o, _ = kern_synth(S["joint"][:, ok], ca5[ok]); err["succ_cands"][qi] = err_of(a, ainf, sig)
            # action-cluster classification: k-means on heads (per task), vote among the top-16 (kernel weights), return cluster mean
            for K, nm in ((32, "clu32"), (64, "clu64")):
                H = heads[r]
                C = H[rng.choice(len(H), size=min(K, len(H)), replace=False)]
                for _ in range(15):
                    lab = np.argmin(((H[:, None, :] - C[None]) ** 2).sum(2), axis=1)
                    for k in range(len(C)):
                        if (lab == k).any():
                            C[k] = H[lab == k].mean(0)
                cmean = np.stack([ca5[lab == k].mean(0) if (lab == k).any() else ca5.mean(0) for k in range(len(C))])
                Sj = S["joint"]; n_, Cn = Sj.shape
                part = np.argpartition(-Sj, 15, axis=1)[:, :16]
                ps = np.take_along_axis(Sj, part, 1); o = np.take_along_axis(part, np.argsort(-ps, axis=1), 1)
                dd = -np.take_along_axis(Sj, o, 1); dd = dd - dd[:, :1]; ref = np.maximum(dd[:, 4:5], 1e-6); w = np.exp(-(dd / ref) ** 2)
                votes = np.zeros((n_, len(C)))
                for j in range(16):
                    np.add.at(votes, (np.arange(n_), lab[o[:, j]]), w[:, j])
                win = votes.argmax(1)
                err[nm][qi] = err_of(cmean[win], ainf, sig)
                if K == 64:
                    # kernel mean restricted to the winning cluster's members among the 16
                    msk = (lab[o] == win[:, None]).astype(np.float64) * w
                    a = (ca5[o] * msk[:, :, None, None]).sum(1) / np.maximum(msk.sum(1), 1e-9)[:, None, None]
                    err["clu64_kern"][qi] = err_of(a, ainf, sig)
        else:
            # fresh regime: continuity of the previous chunk's tail (steps >= 1)
            m1 = step[qi] >= 1
            pos = qi[m1]
            tail = (Q["a_exec"][pos - 1][:, 5:10, :] / sig).reshape(len(pos), 35)
            h2 = (heads[r] ** 2).sum(1)
            c = np.sqrt(np.maximum(h2[None] - 2 * tail @ heads[r].T + (tail ** 2).sum(1)[:, None], 0) / 35)
            # scale s_c: median 1-NN of library tails to other-episode heads
            tails = (np.asarray(L["action"][r][:, 5:10, :7], np.float32) / sig).reshape(len(r), 35)
            CC = np.sqrt(np.maximum(h2[None] - 2 * tails @ heads[r].T + (tails ** 2).sum(1)[:, None], 0) / 35); CC[ep[:, None] == ep[None]] = np.inf
            s_c = float(np.median(CC.min(1))) + 1e-6
            Sj = S["joint"][m1]; dmed = np.median(-Sj, axis=1, keepdims=True)
            for lam, nm in ((0.5, "fresh_l0p5"), (1.0, "fresh_l1"), (2.0, "fresh_l2"), (4.0, "fresh_l4")):
                F = Sj / dmed - lam * c / s_c
                a, o, _ = kern_synth(F, ca5); err[nm][pos] = err_of(a, ainf[m1], sig)
            a, o, _ = kern_synth(-c / s_c, ca5); err["fresh_cont"][pos] = err_of(a, ainf[m1], sig)
            # gate: continuity candidates (top-16 by c) re-weighted by the whitened distance
            part = np.argpartition(c, 15, axis=1)[:, :16]
            Fg = np.full_like(Sj, -1e9); np.put_along_axis(Fg, part, np.take_along_axis(Sj / dmed, part, 1), 1)
            a, o, _ = kern_synth(Fg, ca5); err["fresh_gate"][pos] = err_of(a, ainf[m1], sig)
    m = step >= 1
    res = {"cell": cell, "L": int(L["L"]), "n": int(m.sum()), "wall_s": time.time() - t0}
    drift = dst > 3.0
    res["drift_frac"] = float(drift[m].mean())
    for n in names:
        if np.isfinite(err[n][m]).any():
            res[n] = {"err": float(np.nanmean(err[n][m])), "err_drift": float(np.nanmean(err[n][m & drift])) if (m & drift).any() else float("nan"),
                      "err_nodrift": float(np.nanmean(err[n][m & ~drift])), "err_succ": float(np.nanmean(err[n][m & Q["ep_success"]])),
                      "err_fail": float(np.nanmean(err[n][m & ~Q["ep_success"]])) if (m & ~Q["ep_success"]).any() else float("nan")}
    res["step0"] = {n: float(np.nanmean(err[n][step == 0])) for n in ("joint", "vis", "state") }
    json.dump(res, open(SCR / "out" / f"r_{cell}.json", "w"), indent=1)
    print(cell, f"{res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
