"""Closed-loop-oriented checks of the whitened metric on the cache cells (big library). Usage: p5_closedloop.py <cell>
Synthesis sweep (k, kref), magnitude shrinkage / direction agreement vs a_inf, stay rate vs B0, gripper rule, by-step early vs full fit.
"""
import json, pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from p1_diag import load_lib, load_q, sigma_d, unit, err_of, aurc, ROOT, SCR, BIG
from p3_white import l2score, fit_transform


def synth(S, ca5, k, kref, grip_rule="mean"):
    n, C = S.shape
    k = min(k, C)
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    ps = np.take_along_axis(S, part, 1)
    o = np.take_along_axis(part, np.argsort(-ps, axis=1), 1)
    dd = -np.take_along_axis(S, o, 1); dd = dd - dd[:, :1]
    ref = np.maximum(dd[:, min(kref - 1, k - 1):min(kref, k)], 1e-6)
    w = np.exp(-(dd / ref) ** 2)
    a = (ca5[o] * w[:, :, None, None]).sum(1) / w.sum(1)[:, None, None]
    if grip_rule == "wmaj":
        g = np.sign(ca5[o][:, :, :, 6])                      # [n,k,5]
        gm = (g * w[:, :, None]).sum(1)
        a = a.copy(); a[:, :, 6] = np.where(gm >= 0, 1.0, -1.0)
    return a, o


def main(cell):
    t0 = time.time()
    ms = cell.rsplit("_", 1)[0]; model = ms.split("_")[0]
    L = load_lib(ms, BIG[model]); Q = load_q(cell); sig = sigma_d(ms)
    N = Q["N"]; step = Q["step"]
    heads = (L["a5"] / sig).reshape(L["L"], 35)
    cfgs = {f"k{k}_r{r}": (k, r, "mean") for k in (8, 16, 32, 64) for r in (3, 5, 8)}
    cfgs["k16_r5_wmaj"] = (16, 5, "wmaj")
    cfgs["k1"] = (1, 1, "mean")
    err = {n: np.full(N, np.nan) for n in cfgs}
    grip = {n: np.full(N, np.nan) for n in cfgs}
    mag = {n: np.full(N, np.nan) for n in cfgs}
    dircos = {n: np.full(N, np.nan) for n in cfgs}
    top1 = np.full(N, -1); err_early = np.full(N, np.nan); err_full = np.full(N, np.nan)
    def feats(P0, P1, rs): return np.concatenate([P0[:, :64], P1[:, :64], rs], 1).astype(np.float64)
    for t in np.unique(Q["task_id"]):
        qi = np.where(Q["task_id"] == t)[0]; r = np.where(L["task_id"] == t)[0]
        ca5 = L["a5"][r]; ainf = Q["a_inf"][qi]
        Xl = feats(L["P0"][r], L["P1"][r], L["rs8"][r]); Xq = feats(Q["P0"][qi], Q["P1"][qi], Q["rs8"][qi])
        mean, std, W = fit_transform(Xl, heads[r], L["episode"][r], L["step"][r], nn=3, lam=0.1)
        S = l2score(((Xq - mean) / std) @ W, ((Xl - mean) / std) @ W)
        me, se, We = fit_transform(Xl, heads[r], L["episode"][r], L["step"][r], nn=3, lam=0.1, rows_mask=L["step"][r] <= 2)
        Se = l2score(((Xq - me) / se) @ We, ((Xl - me) / se) @ We)
        ae, _ = synth(Se, ca5, 16, 5); err_early[qi] = err_of(ae, ainf, sig)
        af, _ = synth(S, ca5, 16, 5); err_full[qi] = err_of(af, ainf, sig)
        top1[qi] = r[S.argmax(1)]
        for n, (k, kref, gr) in cfgs.items():
            a, o = synth(S, ca5, k, kref, gr)
            err[n][qi] = err_of(a, ainf, sig)
            grip[n][qi] = np.mean((a[:, :, 6] >= 0) != (ainf[:, :, 6] >= 0), axis=1)
            na = np.linalg.norm((a[:, :, :6] / sig[:6]).reshape(len(qi), -1), axis=1); ni = np.linalg.norm((ainf[:, :, :6] / sig[:6]).reshape(len(qi), -1), axis=1)
            mag[n][qi] = na / np.maximum(ni, 1e-6)
            dircos[n][qi] = np.einsum("ij,ij->i", (a[:, :, :6] / sig[:6]).reshape(len(qi), -1), (ainf[:, :, :6] / sig[:6]).reshape(len(qi), -1)) / np.maximum(na * ni, 1e-6)
    m = step >= 1
    res = {"cell": cell, "L": int(L["L"]), "n_stale": int(m.sum())}
    res["synth"] = {n: {"err": float(np.nanmean(err[n][m])), "grip": float(np.nanmean(grip[n][m])), "mag_ratio_p50": float(np.nanmedian(mag[n][m])),
                        "mag_ratio_mean": float(np.nanmean(mag[n][m])), "dircos_mean": float(np.nanmean(dircos[n][m]))} for n in cfgs}
    # a_exec (B0's served chunk) magnitude ratio for reference (cache arm: a_exec == a_hit)
    aex = Q["a_exec"][:, :5, :]
    nex = np.linalg.norm((aex[:, :, :6] / sig[:6]).reshape(N, -1), axis=1); ni = np.linalg.norm((Q["a_inf"][:, :, :6] / sig[:6]).reshape(N, -1), axis=1)
    res["b0_mag_ratio_p50"] = float(np.nanmedian((nex / np.maximum(ni, 1e-6))[m]))
    res["b0_err"] = float(np.nanmean(err_of(aex, Q["a_inf"], sig)[m]))
    # stay rate: same top-1 row as the previous decision of the episode
    prev_same_ep = np.zeros(N, bool); prev_same_ep[1:] = Q["ep"][1:] == Q["ep"][:-1]
    stay = np.zeros(N, bool); stay[1:] = (top1[1:] == top1[:-1]) & prev_same_ep[1:]
    stay_b0 = np.zeros(N, bool); stay_b0[1:] = (Q["rec_top1"][1:] == Q["rec_top1"][:-1]) & prev_same_ep[1:]
    res["stay_rate_base_top1"] = float(stay[m].mean()); res["stay_rate_b0"] = float(stay_b0[m].mean())
    res["err_when_stay"] = float(np.nanmean(err["k16_r5"][m & stay])); res["err_when_switch"] = float(np.nanmean(err["k16_r5"][m & ~stay]))
    # same library episode as previous top-1 (track) rate
    trk = np.zeros(N, bool); trk[1:] = (L["episode"][top1[1:]] == L["episode"][top1[:-1]]) & prev_same_ep[1:]
    res["same_lib_episode_rate"] = float(trk[m].mean())
    # by-step early vs full
    res["by_step"] = {}
    for lo, hi, tag in ((0, 0, "s0"), (1, 1, "s1"), (2, 2, "s2"), (3, 5, "s3-5"), (6, 10000, "s6+")):
        mm = (step >= lo) & (step <= hi)
        res["by_step"][tag] = {"n": int(mm.sum()), "full": float(np.nanmean(err_full[mm])), "early": float(np.nanmean(err_early[mm]))}
    res["wall_s"] = time.time() - t0
    json.dump(res, open(SCR / "out" / f"cl_{cell}.json", "w"), indent=1)
    print(cell, f"{res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
