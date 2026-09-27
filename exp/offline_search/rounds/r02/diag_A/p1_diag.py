"""R2 ideation-A diagnostics (read-only, CPU, OMP=1). Usage: p1_diag.py <cell> <lib: current|big>
Writes <SCR>/out/<cell>__<lib>.json with per-regime err / AURC of many vision-centred selectors.
All err numbers: harness metric (RMS over [:5,:7] in sigma_d units vs a_inf), sigma_d from library current.
"""
import json, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path("/dev/shm/offline_search_store")
PCA = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/pca")
SCR = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r02_ideation_A")
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
RSV = 8
EPS = 1e-8


# ------------------------------------------------------------------------------------------------ data
def load_lib(ms, name):
    d = ROOT / "library" / ms / name
    L = {k: np.load(d / f"{k}.npy") for k in ("rs", "task_id", "episode", "step", "progress", "prev", "next", "success", "ep_len")}
    L["action"] = np.load(d / "action.npy", mmap_mode="r")
    L["a5"] = np.ascontiguousarray(np.asarray(L["action"][:, :5, :7], np.float32))
    if name == "current":
        L["P0"] = np.load(SCR / "proj" / f"lib_{ms}_current_v0.npy")
        L["P1"] = np.load(SCR / "proj" / f"lib_{ms}_current_v1.npy")
    else:
        L["P0"] = np.load(PCA / ms / name / "v0" / "proj.npy")
        L["P1"] = np.load(PCA / ms / name / "v1" / "proj.npy")
    L["rs8"] = np.ascontiguousarray(L["rs"][:, :RSV].astype(np.float32))
    L["L"] = len(L["task_id"])
    return L


def load_q(cell):
    d = ROOT / "queries" / cell
    Q = {k: np.load(d / f"{k}.npy") for k in ("rs", "ep", "step", "rec_top1")}
    Q["a_inf"] = np.ascontiguousarray(np.asarray(np.load(d / "a_inf.npy", mmap_mode="r")[:, :5, :7], np.float32))
    Q["a_exec"] = np.ascontiguousarray(np.asarray(np.load(d / "a_exec.npy", mmap_mode="r")[:, :10, :7], np.float32))
    Q["P0"] = np.load(SCR / "proj" / f"{cell}_v0.npy")
    Q["P1"] = np.load(SCR / "proj" / f"{cell}_v1.npy")
    Q["rs8"] = np.ascontiguousarray(Q["rs"][:, :RSV].astype(np.float32))
    eps = json.load(open(d / "episodes.json"))
    ns = np.array([e["num_steps"] for e in eps]); suc = np.array([bool(e["success"]) for e in eps]); tid = np.array([e["task_id"] for e in eps])
    Q["num_steps"] = ns[Q["ep"]]; Q["ep_success"] = suc[Q["ep"]]; Q["task_id"] = tid[Q["ep"]]
    Q["phase"] = Q["step"] / np.maximum(Q["num_steps"] - 1, 1)
    Q["N"] = len(Q["step"])
    return Q


def sigma_d(ms):
    a = np.load(ROOT / "library" / ms / "current" / "action.npy", mmap_mode="r")
    return np.asarray(a[:, :5, :7], np.float32).reshape(-1, 7).std(axis=0)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), EPS)


def err_of(ahat, ainf, sig):
    return np.sqrt(np.mean(((ahat - ainf) / sig) ** 2, axis=(1, 2)))


def aurc(conf, err):
    o = np.argsort(-conf, kind="stable")
    e = err[o]
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return float(risk.mean())


def spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    return float(spearmanr(a[m], b[m]).correlation)


# ------------------------------------------------------------------------------------------------ core eval
def topk_synth(S, cand_a5, k):
    """S [n, C] scores (higher better); returns (mean-k action [n,5,7], top1 idx, topk idx)."""
    n, C = S.shape
    k = min(k, C)
    if k == C:
        o = np.argsort(-S, axis=1)
    else:
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        o = np.take_along_axis(part, np.argsort(-ps, axis=1), 1)
    a = cand_a5[o].mean(axis=1)
    return a, o[:, 0], o


def grip_cluster_synth(S, cand_a5, o):
    """Among the top-k rows o [n,k]: split by gripper majority sign of the executed segment; mean of the heavier cluster."""
    g = np.sign(cand_a5[o][:, :, :, 6].mean(axis=2))          # [n,k] sign per candidate
    w = np.take_along_axis(S, o, 1)
    w = w - w.min(axis=1, keepdims=True) + 1e-3
    pos = (g >= 0)
    wpos = (w * pos).sum(1); wneg = (w * ~pos).sum(1)
    use_pos = wpos >= wneg
    mask = np.where(use_pos[:, None], pos, ~pos).astype(np.float32)
    a = (cand_a5[o] * mask[:, :, None, None]).sum(1) / mask.sum(1)[:, None, None]
    return a


def evaluate(cell, libname):
    t0 = time.time()
    ms = cell.rsplit("_", 1)[0]; arm = cell.rsplit("_", 1)[1]; model = ms.split("_")[0]
    L = load_lib(ms, BIG[model] if libname == "big" else "current")
    Q = load_q(cell)
    sig = sigma_d(ms)
    N = Q["N"]
    step = Q["step"]
    regime = np.where(step == 0, 0, 1 if arm == "inf" else 2)     # 0 step0, 1 fresh, 2 stale
    # library-side derived
    prevL = L["prev"]; nextL = L["next"]
    has_prev = prevL >= 0
    D0 = np.zeros_like(L["P0"]); D1 = np.zeros_like(L["P1"])
    D0[has_prev] = L["P0"][has_prev] - L["P0"][prevL[has_prev]]
    D1[has_prev] = L["P1"][has_prev] - L["P1"][prevL[has_prev]]
    # query-side delta (within episode; step 0 -> zero)
    qD0 = np.zeros_like(Q["P0"]); qD1 = np.zeros_like(Q["P1"])
    m1 = step >= 1
    qD0[m1] = Q["P0"][m1] - Q["P0"][np.where(m1)[0] - 1]
    qD1[m1] = Q["P1"][m1] - Q["P1"][np.where(m1)[0] - 1]
    # state scale (M2-style): median within-task 1-NN rs distance across episodes (library)
    # action head scale
    heads = (L["a5"] / sig).reshape(L["L"], 35)
    res = {"cell": cell, "lib": libname, "L": int(L["L"]), "N": int(N), "episodes_lib": int(len(np.unique(L["episode"])))}
    per = {}   # name -> dict(err [N], conf [N], grip [N], top1 row)

    # ---- fit T1 metrics per task on the library: action-similar-pair whitening + CCA + progress ridge
    metrics = {}
    for t in np.unique(L["task_id"]):
        r = np.where(L["task_id"] == t)[0]
        X = np.concatenate([L["P0"][r, :64], L["P1"][r, :64], L["rs8"][r] * 1.0], axis=1).astype(np.float64)
        Xs = X.std(0) + 1e-6
        Xn = (X - X.mean(0)) / Xs
        ep = L["episode"][r]
        H = heads[r].astype(np.float64)
        # action-similar pairs: 3 nearest heads from other episodes
        h2 = (H * H).sum(1)
        DH = h2[:, None] - 2 * H @ H.T + h2[None, :]
        DH[ep[:, None] == ep[None, :]] = np.inf
        nn = np.argpartition(DH, 3, axis=1)[:, :3]
        diffs = (Xn[:, None, :] - Xn[nn]).reshape(-1, Xn.shape[1])
        Sw = diffs.T @ diffs / len(diffs)
        St = np.cov(Xn.T)
        lam = 0.1 * np.trace(Sw) / Sw.shape[0]
        # whitening transform W = chol((Sw + lam I)^-1)
        Minv = np.linalg.inv(Sw + lam * np.eye(Sw.shape[0]))
        Wm = np.linalg.cholesky(Minv).T                       # x' = x @ Wm.T  -> (x-y) Minv (x-y)^T
        Swv = Sw[:128, :128]; lamv = 0.1 * np.trace(Swv) / 128
        Wmv = np.linalg.cholesky(np.linalg.inv(Swv + lamv * np.eye(128))).T
        # generalized (LDA-like) directions: maximise total/within -> eig of Sw^-1 St
        pass
        # (solve gives non-symmetric; use symmetric form instead)
        A = np.linalg.cholesky(np.linalg.inv(Sw + lam * np.eye(Sw.shape[0])))
        Ms = A.T @ St @ A
        ev, V = np.linalg.eigh((Ms + Ms.T) / 2)
        idx = np.argsort(-ev)[:32]
        Wlda = (A @ V[:, idx]) * np.sqrt(np.maximum(ev[idx], 0))[None, :]   # x' = x @ Wlda
        # CCA between Xn and H (ridge-regularised)
        Hn = (H - H.mean(0)) / (H.std(0) + 1e-6)
        Cxx = np.cov(Xn.T) + 1e-2 * np.eye(Xn.shape[1]); Chh = np.cov(Hn.T) + 1e-2 * np.eye(Hn.shape[1])
        Cxh = (Xn.T @ Hn) / (len(Xn) - 1)
        Lx = np.linalg.cholesky(np.linalg.inv(Cxx)); Lh = np.linalg.cholesky(np.linalg.inv(Chh))
        T = Lx.T @ Cxh @ Lh
        U, s, Vt = np.linalg.svd(T, full_matrices=False)
        kc = 16
        Wcca = (Lx @ U[:, :kc]) * s[:kc][None, :]
        # progress ridge (vision+state) and state-only
        prog = L["progress"][r].astype(np.float64)
        Xa = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
        beta = np.linalg.solve(Xa.T @ Xa + 1.0 * np.eye(Xa.shape[1]), Xa.T @ prog)
        Xs_only = np.concatenate([Xn[:, 128:], np.ones((len(Xn), 1))], 1)
        beta_s = np.linalg.solve(Xs_only.T @ Xs_only + 1.0 * np.eye(Xs_only.shape[1]), Xs_only.T @ prog)
        metrics[int(t)] = dict(mean=X.mean(0), std=Xs, Wm=Wm, Wmv=Wmv, Wlda=Wlda, Wcca=Wcca, beta=beta, beta_s=beta_s, cca_s=s[:kc])
    res["cca_corr_mean"] = float(np.mean([m["cca_s"][:4].mean() for m in metrics.values()]))

    names = ["state_k5", "abs32", "abs128", "abs32_st", "delta128", "delta32", "abs32_delta", "abs32_delta_st",
             "white_l2", "lda32", "cca16", "cca16_st", "band_abs32_st", "abs32_st_gclu",
             "white_vis", "white_k8", "white_kern16", "abs32_st_k8", "abs32_st_step", "white_top1_disp"]
    out = {n: {"err": np.full(N, np.nan), "conf": np.full(N, np.nan), "grip": np.full(N, np.nan), "err1": np.full(N, np.nan),
               "phase_top1": np.full(N, np.nan)} for n in names}
    oracle = np.full(N, np.nan)
    phat = np.full(N, np.nan); phat_s = np.full(N, np.nan)
    vis_motion = np.full(N, np.nan); vis_still = np.full(N, np.nan); hit_verify = np.full(N, np.nan); exp_next_gain = np.full(N, np.nan)
    d1nn_state = np.full(N, np.nan); abs32_top1cos = np.full(N, np.nan)
    Pc = None
    if arm == "cache":
        Lc = L if libname == "current" else load_lib(ms, "current")
        Pc = Lc
    for t in np.unique(Q["task_id"]):
        qi = np.where(Q["task_id"] == t)[0]
        r = np.where(L["task_id"] == t)[0]
        if r.size == 0:
            continue
        ca5 = L["a5"][r]
        ainf = Q["a_inf"][qi]
        # oracle
        # (chunked to bound memory)
        E = np.empty((len(qi), len(r)), np.float32)
        for lo in range(0, len(qi), 512):
            hi = min(len(qi), lo + 512)
            dif = (ainf[lo:hi, None] - ca5[None]) / sig
            E[lo:hi] = np.sqrt(np.mean(dif * dif, axis=(2, 3)))
        oracle[qi] = E.min(1)
        # representations
        u0 = unit(L["P0"][r, :32]); u1 = unit(L["P1"][r, :32])
        U0 = unit(L["P0"][r]); U1 = unit(L["P1"][r])
        q0 = unit(Q["P0"][qi, :32]); q1 = unit(Q["P1"][qi, :32])
        Q0 = unit(Q["P0"][qi]); Q1 = unit(Q["P1"][qi])
        S_abs32 = q0 @ u0.T + q1 @ u1.T
        S_abs128 = Q0 @ U0.T + Q1 @ U1.T
        drs = np.sqrt(np.maximum((Q["rs8"][qi] ** 2).sum(1)[:, None] - 2 * Q["rs8"][qi] @ L["rs8"][r].T + (L["rs8"][r] ** 2).sum(1)[None], 0))
        # state scale: median 1-NN across episodes within library task
        DL = np.sqrt(np.maximum((L["rs8"][r] ** 2).sum(1)[:, None] - 2 * L["rs8"][r] @ L["rs8"][r].T + (L["rs8"][r] ** 2).sum(1)[None], 0))
        DL[L["episode"][r][:, None] == L["episode"][r][None]] = np.inf
        s_d = float(np.median(DL.min(1))) + 1e-6
        d1nn_state[qi] = drs.min(1) / s_d
        # z-scales of abs32 cosine within task (library pseudo-queries)
        CL = u0 @ u0.T + u1 @ u1.T
        CL[L["episode"][r][:, None] == L["episode"][r][None]] = np.nan
        mu_c, sd_c = float(np.nanmean(CL)), float(np.nanstd(CL)) + 1e-6
        S_abs32_st = (S_abs32 - mu_c) / sd_c - drs / s_d
        # delta
        dl0 = unit(D0[r]); dl1 = unit(D1[r])
        dq0 = unit(qD0[qi]); dq1 = unit(qD1[qi])
        S_delta128 = dq0 @ dl0.T + dq1 @ dl1.T
        dl0s = unit(D0[r, :32]); dl1s = unit(D1[r, :32])
        S_delta32 = unit(qD0[qi, :32]) @ dl0s.T + unit(qD1[qi, :32]) @ dl1s.T
        st0 = Q["step"][qi] == 0
        S_delta128[st0] = S_abs128[st0]; S_delta32[st0] = S_abs32[st0]
        S_abs32_delta = (S_abs32 - mu_c) / sd_c + S_delta128 / 0.5
        S_abs32_delta_st = S_abs32_delta - drs / s_d
        # T1 metrics
        m = metrics[int(t)]
        Xl = (np.concatenate([L["P0"][r, :64], L["P1"][r, :64], L["rs8"][r]], 1) - m["mean"]) / m["std"]
        Xq = (np.concatenate([Q["P0"][qi, :64], Q["P1"][qi, :64], Q["rs8"][qi]], 1) - m["mean"]) / m["std"]
        def l2score(A, B):
            return -np.sqrt(np.maximum((A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None], 0))
        S_white = l2score(Xq @ m["Wm"].T, Xl @ m["Wm"].T)
        S_whitev = l2score(Xq[:, :128] @ m["Wmv"].T, Xl[:, :128] @ m["Wmv"].T)
        S_step = S_abs32_st - np.abs(L["step"][r][None, :].astype(np.float32) - Q["step"][qi][:, None].astype(np.float32)) / 3.0
        S_lda = l2score(Xq @ m["Wlda"], Xl @ m["Wlda"])
        S_cca = l2score(Xq @ m["Wcca"], Xl @ m["Wcca"])
        S_cca_st = S_cca / (np.median(-S_cca) + 1e-6) * 1.0 - drs / s_d
        # progress
        Xqa = np.concatenate([Xq, np.ones((len(Xq), 1))], 1)
        phat[qi] = np.clip(Xqa @ m["beta"], 0, 1)
        phat_s[qi] = np.clip(np.concatenate([Xq[:, 128:], np.ones((len(Xq), 1))], 1) @ m["beta_s"], 0, 1)
        band = np.abs(L["progress"][r][None, :] - phat[qi][:, None]) <= 0.15
        S_band = np.where(band, S_abs32_st, S_abs32_st - 100.0)
        # visual motion / stillness
        vis_motion[qi] = np.linalg.norm(qD0[qi], axis=1) / (np.median(np.linalg.norm(D0[r][has_prev[r]], axis=1)) + 1e-6)
        Sv = {"state_k5": -drs, "abs32": S_abs32, "abs128": S_abs128, "abs32_st": S_abs32_st, "delta128": S_delta128,
              "delta32": S_delta32, "abs32_delta": S_abs32_delta, "abs32_delta_st": S_abs32_delta_st, "white_l2": S_white,
              "lda32": S_lda, "cca16": S_cca, "cca16_st": S_cca_st, "band_abs32_st": S_band, "abs32_st_gclu": S_abs32_st,
              "white_vis": S_whitev, "white_k8": S_white, "white_kern16": S_white, "abs32_st_k8": S_abs32_st, "abs32_st_step": S_step,
              "white_top1_disp": S_white}
        for n, S in Sv.items():
            k = 8 if n in ("white_k8", "abs32_st_k8") else 5
            a, i1, o = topk_synth(S, ca5, k)
            if n == "abs32_st_gclu":
                a8, i1, o8 = topk_synth(S, ca5, 8)
                a = grip_cluster_synth(S, ca5, o8)
            if n == "white_kern16":
                a16, i1, o16 = topk_synth(S, ca5, 16)
                dd = -np.take_along_axis(S, o16, 1)
                w = np.exp(-(dd / np.maximum(dd[:, 4:5], 1e-6)) ** 2)
                a = (ca5[o16] * w[:, :, None, None]).sum(1) / w.sum(1)[:, None, None]
                o = o16
            e = err_of(a, ainf, sig)
            out[n]["err"][qi] = e
            out[n]["err1"][qi] = err_of(ca5[i1], ainf, sig)
            out[n]["grip"][qi] = np.mean((a[:, :, 6] >= 0) != (ainf[:, :, 6] >= 0), axis=1)
            top = S[np.arange(len(qi)), i1]
            disp = np.sqrt(np.mean((ca5[o] / sig - (a / sig)[:, None]) ** 2, axis=(1, 2, 3)))
            out[n]["conf"][qi] = top if n != "white_top1_disp" else (top / (np.median(-S) + 1e-6) - disp / 0.3)
            out[n]["phase_top1"][qi] = L["progress"][r][i1]
        abs32_top1cos[qi] = S_abs32.max(1)
        # HIT verification (cache arm): previous served row = rec_top1 of previous decision (library current)
        if arm == "cache":
            pos = qi[Q["step"][qi] >= 1]
            prow = Q["rec_top1"][pos - 1]
            nxt = Pc["next"][prow]
            ok = nxt >= 0
            dl_exp0 = Pc["P0"][nxt[ok]] - Pc["P0"][prow[ok]]
            dl_exp1 = Pc["P1"][nxt[ok]] - Pc["P1"][prow[ok]]
            dq0v = qD0[pos[ok]]; dq1v = qD1[pos[ok]]
            hv = (unit(dl_exp0) * unit(dq0v)).sum(1) + (unit(dl_exp1) * unit(dq1v)).sum(1)
            hit_verify[pos[ok]] = hv
            # expected-next gain: cos(P_t, P[next]) - cos(P_t, P[prow])   (did we arrive where the library went, or stay?)
            pt0 = unit(Q["P0"][pos[ok]]); pt1 = unit(Q["P1"][pos[ok]])
            g = (pt0 * unit(Pc["P0"][nxt[ok]])).sum(1) + (pt1 * unit(Pc["P1"][nxt[ok]])).sum(1) \
                - (pt0 * unit(Pc["P0"][prow[ok]])).sum(1) - (pt1 * unit(Pc["P1"][prow[ok]])).sum(1)
            exp_next_gain[pos[ok]] = g
        # stillness: cos(P_t, P_{t-1}) in 128-d both cams
        pos = qi[Q["step"][qi] >= 1]
        vis_still[pos] = (unit(Q["P0"][pos]) * unit(Q["P0"][pos - 1])).sum(1) + (unit(Q["P1"][pos]) * unit(Q["P1"][pos - 1])).sum(1)

    # ---- summaries per regime
    def summ(mask, tag):
        S = {}
        S["n"] = int(mask.sum())
        S["oracle"] = float(np.nanmean(oracle[mask]))
        for n in names:
            e = out[n]["err"][mask]; c = out[n]["conf"][mask]
            S[n] = {"err": float(np.nanmean(e)), "err1": float(np.nanmean(out[n]["err1"][mask])), "p50": float(np.nanmedian(e)),
                    "grip": float(np.nanmean(out[n]["grip"][mask])), "aurc_top": aurc(c, e),
                    "aurc_opt": aurc(-e, e), "rho_top": spearman(c, -e),
                    "phase_err": float(np.nanmean(np.abs(out[n]["phase_top1"][mask] - Q["phase"][mask])))}
        S["phat_mae_vis"] = float(np.nanmean(np.abs(phat[mask] - Q["phase"][mask])))
        S["phat_mae_state"] = float(np.nanmean(np.abs(phat_s[mask] - Q["phase"][mask])))
        # drift / stuck diagnostics on abs32_st mean-5 err
        e = out["abs32_st"]["err"]
        drift = d1nn_state > 3.0
        S["drift_frac"] = float(np.mean(drift[mask]))
        S["err_drift"] = float(np.nanmean(e[mask & drift])) if (mask & drift).any() else float("nan")
        S["err_nodrift"] = float(np.nanmean(e[mask & ~drift]))
        for sigName, sigv in [("vis_motion", vis_motion), ("vis_still", vis_still), ("hit_verify", hit_verify), ("exp_next_gain", exp_next_gain),
                              ("phat_vs_top1phase", -np.abs(phat - out["abs32_st"]["phase_top1"])), ("neg_d1nn_state", -d1nn_state),
                              ("abs32_top1cos", abs32_top1cos)]:
            v = sigv[mask]
            S[f"rho_{sigName}_negerr"] = spearman(v, -e[mask])
            S[f"aurc_{sigName}"] = aurc(np.nan_to_num(v, nan=-1e9), e[mask]) if np.isfinite(v).any() else float("nan")
            S[f"rho_{sigName}_drift"] = spearman(v, drift[mask].astype(float))
            fm = mask & Q["ep_success"]; ff = mask & ~Q["ep_success"]
            S[f"{sigName}_succ_mean"] = float(np.nanmean(sigv[fm])) if fm.any() else float("nan")
            S[f"{sigName}_fail_mean"] = float(np.nanmean(sigv[ff])) if ff.any() else float("nan")
        # combined confidences for abs32_st
        c1 = out["abs32_st"]["conf"]
        def z(x, mm):
            v = x[mm]; v = np.nan_to_num(v, nan=np.nanmean(v)); return (v - v.mean()) / (v.std() + 1e-9)
        S["aurc_abs32st_plus_still"] = aurc(z(c1, mask) - z(vis_still, mask), e[mask])
        S["aurc_abs32st_plus_motion"] = aurc(z(c1, mask) + z(np.log(vis_motion + 1e-3), mask), e[mask])
        if arm == "cache":
            S["aurc_abs32st_plus_hitverify"] = aurc(z(c1, mask) + z(hit_verify, mask), e[mask])
            S["aurc_abs32st_plus_expnext"] = aurc(z(c1, mask) + z(exp_next_gain, mask), e[mask])
        S["err_succ_eps"] = float(np.nanmean(e[mask & Q["ep_success"]])); S["err_fail_eps"] = float(np.nanmean(e[mask & ~Q["ep_success"]])) if (mask & ~Q["ep_success"]).any() else float("nan")
        return S
    res["step0"] = summ(regime == 0, "step0")
    res["fresh" if arm == "inf" else "stale"] = summ(regime >= 1, "s")
    res["wall_s"] = time.time() - t0
    (SCR / "out").mkdir(exist_ok=True)
    json.dump(res, open(SCR / "out" / f"{cell}__{libname}.json", "w"), indent=1)
    np.savez_compressed(SCR / "out" / f"{cell}__{libname}_sig.npz", d1nn_state=d1nn_state, vis_motion=vis_motion, vis_still=vis_still,
                        hit_verify=hit_verify, exp_next_gain=exp_next_gain, phat=phat, phase=Q["phase"], step=step, ep=Q["ep"],
                        ep_success=Q["ep_success"], err_abs32_st=out["abs32_st"]["err"], err_state=out["state_k5"]["err"],
                        err_delta=out["abs32_delta_st"]["err"], oracle=oracle, abs32_top1cos=abs32_top1cos,
                        conf_abs32_st=out["abs32_st"]["conf"])
    print(cell, libname, f"{res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
