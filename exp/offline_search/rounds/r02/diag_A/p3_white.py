"""Focused pass on the action-similarity-whitened metric family. Usage: p3_white.py <cell> [lib=big|current]
Writes out/w_<cell>__<lib>.json. err = harness metric. Synthesis: kern16 unless stated (w_i = exp(-(d_i/d_5)^2) over top-16).
"""
import json, pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from p1_diag import load_lib, load_q, sigma_d, unit, err_of, aurc, spearman, ROOT, SCR, BIG

EPS = 1e-8


def l2score(A, B):
    return -np.sqrt(np.maximum((A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None], 0))


def kern_synth(S, ca5, k=16, kref=5):
    n, C = S.shape
    k = min(k, C)
    part = np.argpartition(-S, k - 1, axis=1)[:, :k] if k < C else np.tile(np.arange(C), (n, 1))
    ps = np.take_along_axis(S, part, 1)
    o = np.take_along_axis(part, np.argsort(-ps, axis=1), 1)
    dd = -np.take_along_axis(S, o, 1)
    dd = dd - dd[:, :1]                         # relative to the best (distances >= 0)
    ref = np.maximum(dd[:, min(kref - 1, k - 1):min(kref, k)], 1e-6)
    w = np.exp(-(dd / ref) ** 2)
    a = (ca5[o] * w[:, :, None, None]).sum(1) / w.sum(1)[:, None, None]
    disp = np.sqrt(np.mean((ca5[o[:, :5]] - a[:, None]) ** 2, axis=(1, 2, 3)))
    return a, o, disp


def fit_transform(X, H, ep, step, nn=3, lam=0.1, rank=0, act_mode="head", step_win=0, rows_mask=None, fit_dims=None):
    """X [n,d] raw features; H [n,35 or 70] action heads (sigma-scaled); returns (mean, std, W [d, dout]) with x' = ((x-mean)/std) @ W."""
    if rows_mask is not None:
        X, H, ep, step = X[rows_mask], H[rows_mask], ep[rows_mask], step[rows_mask]
    mean = X.mean(0); std = X.std(0) + 1e-6
    Xn = (X - mean) / std
    h2 = (H * H).sum(1)
    DH = h2[:, None] - 2 * H @ H.T + h2[None, :]
    DH[ep[:, None] == ep[None, :]] = np.inf
    if step_win:
        DH[np.abs(step[:, None] - step[None, :]) > step_win] = np.inf
    k = min(nn, DH.shape[1] - 1)
    nn_idx = np.argpartition(DH, k, axis=1)[:, :k]
    okm = np.isfinite(np.take_along_axis(DH, nn_idx, 1))
    diffs = (Xn[:, None, :] - Xn[nn_idx])[okm]
    Sw = diffs.T @ diffs / max(len(diffs), 1)
    d = Sw.shape[0]
    lamv = lam * np.trace(Sw) / d
    Minv = np.linalg.inv(Sw + lamv * np.eye(d))
    if rank == 0:
        W = np.linalg.cholesky(Minv)                # x' = xn @ W  -> ||x'-y'||^2 = (x-y) Minv (x-y)^T
        return mean, std, W
    St = np.cov(Xn.T) + 1e-6 * np.eye(d)
    A = np.linalg.cholesky(Minv)
    Ms = A.T @ St @ A
    ev, V = np.linalg.eigh((Ms + Ms.T) / 2)
    idx = np.argsort(-ev)[:rank]
    W = A @ V[:, idx]                               # keep the top-rank directions (unit within-class variance each)
    return mean, std, W


def main(cell, libname="big"):
    t0 = time.time()
    ms = cell.rsplit("_", 1)[0]; arm = cell.rsplit("_", 1)[1]; model = ms.split("_")[0]
    L = load_lib(ms, BIG[model] if libname == "big" else "current")
    Lc = L if libname == "current" else load_lib(ms, "current")
    Q = load_q(cell)
    sig = sigma_d(ms)
    N = Q["N"]; step = Q["step"]
    regime = np.where(step == 0, 0, 1 if arm == "inf" else 2)
    heads = (L["a5"] / sig).reshape(L["L"], 35)
    chunk = (np.asarray(L["action"][:, :10, :7], np.float32) / sig).reshape(L["L"], 70)
    # id mapping current -> this library (for the tracking prior); None if not mappable
    ids_c = json.load(open(ROOT / "library" / ms / "current" / "ids.json"))
    ids_L = json.load(open(ROOT / "library" / ms / (BIG[model] if libname == "big" else "current") / "ids.json"))
    pos = {s: i for i, s in enumerate(ids_L)}
    cur2lib = np.array([pos.get(s, -1) for s in ids_c])
    mappable = bool((cur2lib >= 0).all())
    variants = {
        "base": dict(nn=3, lam=0.1, rank=0, pdim=64, act_mode="head"),
        "nn1": dict(nn=1, lam=0.1, rank=0, pdim=64), "nn10": dict(nn=10, lam=0.1, rank=0, pdim=64),
        "lam0p01": dict(nn=3, lam=0.01, rank=0, pdim=64), "lam1": dict(nn=3, lam=1.0, rank=0, pdim=64),
        "p32": dict(nn=3, lam=0.1, rank=0, pdim=32), "p128": dict(nn=3, lam=0.1, rank=0, pdim=128),
        "rank16": dict(nn=3, lam=0.1, rank=16, pdim=64), "rank32": dict(nn=3, lam=0.1, rank=32, pdim=64), "rank64": dict(nn=3, lam=0.1, rank=64, pdim=64),
        "chunk": dict(nn=3, lam=0.1, rank=0, pdim=64, act_mode="chunk"),
        "stepwin3": dict(nn=3, lam=0.1, rank=0, pdim=64, step_win=3),
        "succonly": dict(nn=3, lam=0.1, rank=0, pdim=64, succ=True),
        "shared": dict(nn=3, lam=0.1, rank=0, pdim=64, shared=True),
        "fitcur": dict(nn=3, lam=0.1, rank=0, pdim=64, fitcur=True),
        "novis_state": dict(nn=3, lam=0.1, rank=0, pdim=0),
        "early": dict(nn=3, lam=0.1, rank=0, pdim=64, early=True),
        "st_x3": dict(nn=3, lam=0.1, rank=0, pdim=64, st_scale=3.0),
    }
    names = list(variants) + ["base_mean5", "base_top1", "track_pure", "base_track0p5", "base_track1", "base_track2", "base_antistuck"]
    out = {n: np.full(N, np.nan) for n in names}
    conf = {k: np.full(N, np.nan) for k in ("d1", "disp", "dst", "still", "hv", "d1_rel", "dtrack_rel")}
    oracle = np.full(N, np.nan)
    # shared-transform fit (all tasks) prepared once
    def feats(P0, P1, rs, pdim):
        parts = ([P0[:, :pdim], P1[:, :pdim]] if pdim else []) + [rs]
        return np.concatenate(parts, 1).astype(np.float64)
    shared_fit = None
    for t in np.unique(Q["task_id"]):
        qi = np.where(Q["task_id"] == t)[0]
        r = np.where(L["task_id"] == t)[0]
        ca5 = L["a5"][r]; ainf = Q["a_inf"][qi]
        E = np.empty((len(qi), len(r)), np.float32)
        for lo in range(0, len(qi), 512):
            hi = min(len(qi), lo + 512)
            dif = (ainf[lo:hi, None] - ca5[None]) / sig
            E[lo:hi] = np.sqrt(np.mean(dif * dif, axis=(2, 3)))
        oracle[qi] = E.min(1)
        ep = L["episode"][r]; st = L["step"][r]; succ = L["success"][r].astype(bool)
        # state scale
        RS = L["rs8"][r]
        DL = np.sqrt(np.maximum((RS ** 2).sum(1)[:, None] - 2 * RS @ RS.T + (RS ** 2).sum(1)[None], 0)); DL[ep[:, None] == ep[None]] = np.inf
        s_d = float(np.median(DL.min(1))) + 1e-6
        drs = np.sqrt(np.maximum((Q["rs8"][qi] ** 2).sum(1)[:, None] - 2 * Q["rs8"][qi] @ RS.T + (RS ** 2).sum(1)[None], 0))
        conf["dst"][qi] = drs.min(1) / s_d
        S_base = None
        for n, cfg in variants.items():
            pdim = cfg["pdim"]
            Xl = feats(L["P0"][r], L["P1"][r], L["rs8"][r], pdim); Xq = feats(Q["P0"][qi], Q["P1"][qi], Q["rs8"][qi], pdim)
            if cfg.get("st_scale"):
                Xl = Xl.copy(); Xq = Xq.copy()
            H = chunk[r] if cfg.get("act_mode") == "chunk" else heads[r]
            mask = succ if cfg.get("succ") else None
            if cfg.get("early"):
                mask = st <= 2
            if cfg.get("fitcur"):
                rc = np.where(Lc["task_id"] == t)[0]
                Xf = feats(Lc["P0"][rc], Lc["P1"][rc], Lc["rs8"][rc], pdim)
                mean, std, W = fit_transform(Xf, (Lc["a5"][rc] / sig).reshape(len(rc), 35), Lc["episode"][rc], Lc["step"][rc], nn=cfg["nn"], lam=cfg["lam"], rank=cfg["rank"])
            elif cfg.get("shared"):
                if shared_fit is None:
                    Xa = feats(L["P0"], L["P1"], L["rs8"], pdim)
                    # fit on a subsample of all rows (task-agnostic), action-NN within the same task only
                    rng = np.random.default_rng(0); sel = rng.choice(L["L"], size=min(6000, L["L"]), replace=False)
                    epk = L["episode"][sel] + 100000 * L["task_id"][sel]
                    Ht = heads[sel] + 1000.0 * L["task_id"][sel][:, None]      # keep NN within task
                    shared_fit = fit_transform(Xa[sel], Ht, epk, L["step"][sel], nn=cfg["nn"], lam=cfg["lam"], rank=0)
                mean, std, W = shared_fit
            else:
                mean, std, W = fit_transform(Xl, H, ep, st, nn=cfg["nn"], lam=cfg["lam"], rank=cfg["rank"], step_win=cfg.get("step_win", 0), rows_mask=mask)
            if cfg.get("st_scale"):
                std = std.copy(); std[-8:] /= cfg["st_scale"]
            Zl = ((Xl - mean) / std) @ W; Zq = ((Xq - mean) / std) @ W
            S = l2score(Zq, Zl)
            a, o, disp = kern_synth(S, ca5)
            e = err_of(a, ainf, sig)
            if cfg.get("early"):
                # only meaningful at step 0 (and steps 1-2); elsewhere copy base
                out[n][qi] = e
            else:
                out[n][qi] = e
            if n == "base":
                S_base = S
                conf["d1"][qi] = -S.max(1)
                conf["disp"][qi] = disp
                conf["d1_rel"][qi] = -S.max(1) / (np.median(-S, axis=1) + 1e-6)
                # mean-5 and top-1 synth
                o5 = o[:, :5]
                out["base_mean5"][qi] = err_of(ca5[o5].mean(1), ainf, sig)
                out["base_top1"][qi] = err_of(ca5[o[:, 0]], ainf, sig)
        # stillness
        p1 = qi[Q["step"][qi] >= 1]
        conf["still"][p1] = (unit(Q["P0"][p1]) * unit(Q["P0"][p1 - 1])).sum(1) + (unit(Q["P1"][p1]) * unit(Q["P1"][p1 - 1])).sum(1)
        # tracking prior (cache arm only: previously served row = rec_top1[prev] in current -> mapped)
        if arm == "cache" and mappable:
            prow_c = Q["rec_top1"][p1 - 1]
            prow = cur2lib[prow_c]
            nxt = L["next"][prow]
            local = {rr: j for j, rr in enumerate(r)}
            # candidate bonus: rows in the same library episode as prow with step in [step(prow)+1, step(prow)+2]
            bonus = np.zeros((len(p1), len(r)), np.float32)
            trk_err = np.full(len(p1), np.nan); dtrack_rel = np.full(len(p1), np.nan)
            same_ep = L["episode"][r]
            for j in range(len(p1)):
                pr = prow[j]
                if pr < 0 or L["task_id"][pr] != t:
                    continue
                m = (same_ep == L["episode"][pr]) & (st >= L["step"][pr] + 1) & (st <= L["step"][pr] + 2)
                bonus[j, m] = 1.0
                if nxt[j] >= 0 and nxt[j] in local:
                    jj = local[nxt[j]]
                    trk_err[j] = err_of(ca5[jj][None], ainf[Q["step"][qi] >= 1][j][None], sig)[0]
                    dtrack_rel[j] = (-S_base[np.where(qi == p1[j])[0][0], jj]) / (np.median(-S_base[np.where(qi == p1[j])[0][0]]) + 1e-6)
            out["track_pure"][p1] = trk_err
            conf["dtrack_rel"][p1] = dtrack_rel
            sub = np.searchsorted(qi, p1)
            Sb = S_base[sub]
            scale = np.median(-Sb, axis=1, keepdims=True)
            for beta, nm in ((0.5, "base_track0p5"), (1.0, "base_track1"), (2.0, "base_track2")):
                S2 = Sb + beta * scale * bonus
                a, o, _ = kern_synth(S2, ca5)
                out[nm][p1] = err_of(a, ainf[sub], sig)
            # anti-stuck: if the scene did not change (still > tau) exclude the previously served episode entirely
            still = conf["still"][p1]
            tau = np.nanpercentile(still, 80)
            S3 = Sb.copy()
            for j in range(len(p1)):
                pr = prow[j]
                if pr >= 0 and still[j] > tau:
                    S3[j, same_ep == L["episode"][pr]] -= 100.0
            a, o, _ = kern_synth(S3, ca5)
            out["base_antistuck"][p1] = err_of(a, ainf[sub], sig)
            # hit verify
            ok = nxt >= 0
            dl0 = L["P0"][nxt[ok]] - L["P0"][prow[ok]]; dl1 = L["P1"][nxt[ok]] - L["P1"][prow[ok]]
            qd0 = Q["P0"][p1[ok]] - Q["P0"][p1[ok] - 1]; qd1 = Q["P1"][p1[ok]] - Q["P1"][p1[ok] - 1]
            conf["hv"][p1[ok]] = (unit(dl0) * unit(qd0)).sum(1) + (unit(dl1) * unit(qd1)).sum(1)

    # --- timing of the base method, single thread, on this cell's largest task (projection + whiten + distances + kern16)
    t = max(np.unique(L["task_id"]), key=lambda tt: (L["task_id"] == tt).sum())
    r = np.where(L["task_id"] == t)[0]
    d = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/pca") / ms / BIG[model]
    B0 = np.ascontiguousarray(np.load(d / "v0" / "basis.npy", mmap_mode="r")[:, :64], np.float32); B1 = np.ascontiguousarray(np.load(d / "v1" / "basis.npy", mmap_mode="r")[:, :64], np.float32)
    mu0 = np.load(d / "v0" / "mean.npy"); mu1 = np.load(d / "v1" / "mean.npy")
    Xl = feats(L["P0"][r], L["P1"][r], L["rs8"][r], 64)
    mean, std, W = fit_transform(Xl, heads[r], L["episode"][r], L["step"][r])
    Zl = np.ascontiguousarray((((Xl - mean) / std) @ W).astype(np.float32)); z2 = (Zl ** 2).sum(1)
    Wf = (W / std[:, None]).astype(np.float32); mshift = ((mean / std) @ W).astype(np.float32)
    k0 = np.load(ROOT / "queries" / cell / "key_v0.npy", mmap_mode="r"); k1 = np.load(ROOT / "queries" / cell / "key_v1.npy", mmap_mode="r")
    qs = np.where(Q["task_id"] == t)[0][:300]
    ca5 = L["a5"][r]
    tt = []
    for i in qs:
        a0 = np.asarray(k0[i]); a1 = np.asarray(k1[i]); rs = Q["rs8"][i]
        t1 = time.perf_counter()
        f = np.concatenate([(a0 - mu0) @ B0, (a1 - mu1) @ B1, rs]).astype(np.float32)
        z = f @ Wf - mshift
        dd = z2 - 2.0 * (Zl @ z) + float(z @ z)
        S = -np.sqrt(np.maximum(dd, 0))[None]
        a, o, disp = kern_synth(S, ca5)
        tt.append(time.perf_counter() - t1)
    tt = np.array(tt) * 1e3
    timing = {"cands": int(len(r)), "ms_mean": float(tt.mean()), "ms_p50": float(np.median(tt)), "ms_p95": float(np.percentile(tt, 95))}
    # per-stage
    t1 = time.perf_counter()
    for i in qs[:100]:
        f = np.concatenate([(np.asarray(k0[i]) - mu0) @ B0, (np.asarray(k1[i]) - mu1) @ B1, Q["rs8"][i]])
    timing["ms_projection_only"] = (time.perf_counter() - t1) * 10

    # --- summaries
    def z(x, m):
        v = np.nan_to_num(x[m], nan=np.nanmedian(x[m])); return (v - v.mean()) / (v.std() + 1e-9)
    res = {"cell": cell, "lib": libname, "L": int(L["L"]), "N": int(N), "mappable": mappable, "timing": timing, "wall_s": time.time() - t0}
    for rname, m in (("step0", regime == 0), ("fresh" if arm == "inf" else "stale", regime >= 1)):
        S = {"n": int(m.sum()), "oracle": float(np.nanmean(oracle[m]))}
        drift = conf["dst"] > 3.0
        S["drift_frac"] = float(drift[m].mean()); S["oracle_drift"] = float(np.nanmean(oracle[m & drift])) if (m & drift).any() else float("nan")
        for n in names:
            e = out[n]
            if not np.isfinite(e[m]).any():
                continue
            S[n] = {"err": float(np.nanmean(e[m])), "p50": float(np.nanmedian(e[m])), "err_succ": float(np.nanmean(e[m & Q["ep_success"]])),
                    "err_fail": float(np.nanmean(e[m & ~Q["ep_success"]])) if (m & ~Q["ep_success"]).any() else float("nan"),
                    "err_drift": float(np.nanmean(e[m & drift])) if (m & drift).any() else float("nan"), "err_nodrift": float(np.nanmean(e[m & ~drift]))}
        e = out["base"]
        C = {"-d1": -conf["d1"], "-d1_rel": -conf["d1_rel"], "-disp": -conf["disp"], "-dst": -conf["dst"], "-still": -conf["still"], "hv": conf["hv"],
             "z(-d1)+z(-disp)": z(-conf["d1"], m) + z(-conf["disp"], m), "z(-d1_rel)+z(-disp)": z(-conf["d1_rel"], m) + z(-conf["disp"], m),
             "z(-d1)+z(-disp)+z(-dst)": z(-conf["d1"], m) + z(-conf["disp"], m) + z(-conf["dst"], m),
             "z(-d1)+z(-disp)-z(still)": z(-conf["d1"], m) + z(-conf["disp"], m) - z(conf["still"], m),
             "-d1/med-disp/0.3": -conf["d1_rel"] - conf["disp"] / 0.3}
        S["aurc"] = {}
        for k, v in C.items():
            vv = v if len(v) == N else None
            arr = np.full(N, np.nan); arr[m] = v if len(v) == m.sum() else v[m]
            if np.isfinite(arr[m]).sum() < 10:
                continue
            S["aurc"][k] = aurc(np.nan_to_num(arr[m], nan=-1e9), e[m])
        S["aurc"]["opt"] = aurc(-e[m], e[m])
        S["rho"] = {k: spearman(np.asarray(v if len(v) == N else np.full(N, np.nan)), -e) for k, v in C.items() if len(v) == N}
        res[rname] = S
    (SCR / "out").mkdir(exist_ok=True)
    json.dump(res, open(SCR / "out" / f"w_{cell}__{libname}.json", "w"), indent=1)
    np.savez_compressed(SCR / "out" / f"w_{cell}__{libname}.npz", **{f"err_{n}": out[n] for n in names}, **{f"conf_{k}": v for k, v in conf.items()},
                        oracle=oracle, step=step, ep=Q["ep"], ep_success=Q["ep_success"], regime=regime, task=Q["task_id"])
    print(cell, libname, f"{res['wall_s']:.0f}s", timing, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "big")
