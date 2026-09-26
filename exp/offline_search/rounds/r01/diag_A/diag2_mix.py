"""Diag 2: (A) cache-arm split by prev HIT/MISS (all decisions); (B) subsample with full keys: cascades,
consensus, observation-side validation confidences, PCA-whitened joint space, predicted-action embedding.
numpy only; one process per cell, 1 thread each."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from multiprocessing import Pool

ROOT = "/dev/shm/offline_search_store"
R00 = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
CELLS = ["pi05_spatial_inf", "pi05_spatial_cache", "pi05_l10_inf", "pi05_l10_cache",
         "groot_spatial_inf", "groot_spatial_cache", "groot_l10_inf", "groot_l10_cache"]
NQ = 3000


def err(a, b, sig):
    d = (a[..., :5, :7] - b[..., :5, :7]) / sig
    return np.sqrt(np.mean(d * d, axis=(-2, -1)))


def aurc(conf, e):
    m = np.isfinite(conf) & np.isfinite(e); conf, e = conf[m], e[m]
    o = np.argsort(-conf, kind="stable"); ce = np.cumsum(e[o]) / np.arange(1, len(e) + 1)
    return float(ce.mean())


def zt(x, mu, sg):
    return 0.5 * (np.tanh((x - mu) / (sg if sg > 1e-12 else 1.0)) + 1)


def run(cell):
    sys.path.insert(0, "/home/weiland/projects/openpi")
    from exp.offline_search.harness.store import current_params
    m, s, arm = cell.split("_")
    q = f"{ROOT}/queries/{cell}"; lib = f"{ROOT}/library/{m}_{s}/current"
    a_inf = np.load(f"{q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{q}/a_exec.npy", mmap_mode="r")
    a_hit = np.load(f"{q}/a_hit.npy", mmap_mode="r")
    ep = np.load(f"{q}/ep.npy"); step = np.load(f"{q}/step.npy")
    L_act = np.load(f"{lib}/action.npy"); L_task = np.load(f"{lib}/task_id.npy"); L_rs = np.load(f"{lib}/rs.npy")[:, :8].astype(np.float32)
    L_next = np.load(f"{lib}/next.npy"); L_prev = np.load(f"{lib}/prev.npy")
    d = np.load(f"{R00}/B0_current/{cell}.npz"); b3 = np.load(f"{R00}/B3_oracle/{cell}.npz")
    j = json.load(open(f"{R00}/B0_current/{cell}.json")); sig = np.asarray(j["sigma"], np.float32)
    rec = d["top1"]; orc = d["oracle_row"]; eB0 = d["err"]; task = d["task_id"]; confB0 = d["confidence"]
    N = len(ep); H = a_inf.shape[1]
    A = np.asarray(a_inf[:, :5, :7], np.float32); E = np.asarray(a_exec, np.float32)
    prev = np.arange(N) - 1; hp = step >= 1
    g = A[:, :, 6] >= 0; trans = g.min(1) != g.max(1)
    pg = np.zeros(N, bool); pg[hp] = (E[prev[hp], 4, 6] >= 0) != g[hp, 0]; trans |= pg
    out = {"cell": cell}
    # oracle gripper mismatch on transitions
    out["orc_gm_tr"] = float(b3["grip_mis"][trans].mean()); out["orc_gm_all"] = float(b3["grip_mis"].mean())
    # ---- (A) prev HIT / MISS split, all decisions (offline label via a_hit == a_exec at prev)
    if arm == "cache":
        AH = np.asarray(a_hit[:, :5, :7], np.float32)
        was_hit = np.zeros(N, bool); was_hit[hp] = np.all(np.isclose(E[prev[hp], :5, :7], AH[prev[hp]], atol=1e-6), axis=(1, 2))
        # online-legal label: prev executed chunk equals some library action bit-exactly
        tail = E[prev, 5:10, :7]; e_tail = err(tail, A, sig)
        e_cont = np.full(N, np.nan); e_trk = np.full(N, np.nan)
        for t in np.unique(task):
            cand = np.where(L_task == t)[0]; CA = L_act[cand][:, :5, :7] / sig
            qi = np.where((task == t) & hp)[0]; T = tail[qi] / sig
            D = np.sqrt(np.mean((T[:, None] - CA[None]) ** 2, axis=(2, 3))); b = np.argmin(D, 1)
            e_cont[qi] = err(L_act[cand[b]], A[qi], sig)
        nx = L_next[rec[prev]]; ok = hp & (nx >= 0); e_trk[ok] = err(L_act[nx[ok]], A[ok], sig)
        # decisions since last miss
        since = np.zeros(N, int)
        for i in range(N):
            since[i] = 0 if (step[i] == 0 or not was_hit[i]) else since[i - 1] + 1
        for lab, msk in (("prevMISS", hp & ~was_hit), ("prevHIT", hp & was_hit)):
            out[f"{lab}_n"] = int(msk.sum()); out[f"{lab}_B0"] = float(eB0[msk].mean()); out[f"{lab}_tail"] = float(np.nanmean(e_tail[msk]))
            out[f"{lab}_cont"] = float(np.nanmean(e_cont[msk])); out[f"{lab}_track"] = float(np.nanmean(e_trk[msk]))
            out[f"{lab}_orc"] = float(d["oracle_err"][msk].mean())
        for k in (1, 2, 3, 5):
            msk = hp & (since == k) if k < 5 else hp & (since >= 5)
            out[f"since{k}_n"] = int(msk.sum()); out[f"since{k}_B0"] = float(eB0[msk].mean()); out[f"since{k}_cont"] = float(np.nanmean(e_cont[msk]))
    # ---- (B) subsample with full keys
    rng = np.random.default_rng(0); qi_all = np.sort(rng.choice(N, size=min(NQ, N), replace=False))
    p = current_params(m, s); w = np.asarray(p["weights"], np.float64); mu = p["mu"]; sg = p["sigma"]
    K0 = np.load(f"{lib}/key_v0.npy", mmap_mode="r"); K1 = np.load(f"{lib}/key_v1.npy", mmap_mode="r")
    Q0 = np.load(f"{q}/key_v0.npy", mmap_mode="r"); Q1 = np.load(f"{q}/key_v1.npy", mmap_mode="r"); Qrs = np.load(f"{q}/rs.npy", mmap_mode="r")
    Lk0 = np.asarray(K0, np.float32); Lk1 = np.asarray(K1, np.float32)
    n0 = Lk0 / np.linalg.norm(Lk0, axis=1, keepdims=True); n1 = Lk1 / np.linalg.norm(Lk1, axis=1, keepdims=True)
    # PCA (global, task-centered) via Gram matrix, whitened; 256 comps
    Ltask = L_task
    def tcenter(X, tids, means=None):
        X = X.astype(np.float32).copy()
        if means is None:
            means = {int(t): X[tids == t].mean(0) for t in np.unique(tids)}
        for t in np.unique(tids):
            X[tids == t] -= means[int(t)]
        return X, means
    C0, m0 = tcenter(Lk0, Ltask); C1, m1 = tcenter(Lk1, Ltask)
    r0 = np.sqrt(np.mean(C0 ** 2)); r1 = np.sqrt(np.mean(C1 ** 2))
    Xl = np.concatenate([C0 / r0, C1 / r1], 1)  # L x 65536
    G = Xl @ Xl.T; ev, U = np.linalg.eigh(G.astype(np.float64)); o = np.argsort(-ev); ev, U = ev[o], U[:, o]
    R = 256; U = U[:, :R]; sv = np.sqrt(np.maximum(ev[:R], 1e-9)); comps = (Xl.T @ U / sv).astype(np.float32)  # D x R
    Zl = (Xl @ comps) / sv.astype(np.float32)  # whitened lib coords  L x R (unit variance each)
    rs_mu = L_rs.mean(0); rs_sd = L_rs.std(0) + 1e-6; Zrs_l = (L_rs - rs_mu) / rs_sd
    # ridge: [Z(256) | rs(8)] -> action head (35)
    Y = (L_act[:, :5, :7] / sig).reshape(len(L_act), -1).astype(np.float64)
    F = np.concatenate([Zl, Zrs_l], 1).astype(np.float64); Fm = F.mean(0); Ym = Y.mean(0)
    Fc, Yc = F - Fm, Y - Ym
    ridge = {}
    for lam in (10.0, 100.0, 1000.0):
        ridge[lam] = np.linalg.solve(Fc.T @ Fc + lam * np.eye(F.shape[1]), Fc.T @ Yc)
    # ridge with tail feature (library rows with prev)
    hasp = L_prev >= 0
    Tl = (L_act[L_prev[hasp]][:, 5:10, :7] / sig).reshape(hasp.sum(), -1)
    F2 = np.concatenate([F[hasp], Tl], 1); F2m = F2.mean(0); Y2 = Y[hasp]; Y2m = Y2.mean(0)
    ridge2 = np.linalg.solve((F2 - F2m).T @ (F2 - F2m) + 100.0 * np.eye(F2.shape[1]), (F2 - F2m).T @ (Y2 - Y2m))
    # query features
    Qk0 = np.asarray(Q0[qi_all], np.float32); Qk1 = np.asarray(Q1[qi_all], np.float32); Qr = np.asarray(Qrs[qi_all], np.float32)[:, :8]
    qn0 = Qk0 / np.linalg.norm(Qk0, axis=1, keepdims=True); qn1 = Qk1 / np.linalg.norm(Qk1, axis=1, keepdims=True)
    qt = task[qi_all]
    QC0, _ = tcenter(Qk0, qt, m0); QC1, _ = tcenter(Qk1, qt, m1)
    Xq = np.concatenate([QC0 / r0, QC1 / r1], 1); Zq = (Xq @ comps) / sv.astype(np.float32)
    Zrs_q = (Qr - rs_mu) / rs_sd; Fq = np.concatenate([Zq, Zrs_q], 1).astype(np.float64)
    pred = {lam: (Fq - Fm) @ ridge[lam] + Ym for lam in ridge}
    tail_q = E[prev[qi_all], 5:10, :7] / sig; hpq = hp[qi_all]
    pred2 = (np.concatenate([Fq, tail_q.reshape(len(qi_all), -1)], 1) - F2m) @ ridge2 + Y2m
    Aq = A[qi_all]; eB0q = eB0[qi_all]; confq = confB0[qi_all]; recq = rec[qi_all]
    res = {k: np.full(len(qi_all), np.nan) for k in
           ["b0", "cont", "casc_c5_b0", "casc_c10_b0", "casc_b5_c", "casc_b10_c", "z_c70", "z_c85", "cons_b5", "cons_b5_med", "cons_c3",
            "pca_w", "pca_w_rs", "pred10", "pred100", "pred1000", "pred_cons5", "pred2", "pca_w_cons5", "maha_cons5_med"]}
    conf = {k: np.full(len(qi_all), np.nan) for k in ["cont_d", "cross", "rs_of_cont", "b0score_of_cont", "rankB0_of_cont", "pred_res", "pred2_res", "pca_d", "cont_x_pca"]}
    for t in np.unique(qt):
        cand = np.where(L_task == t)[0]; C = len(cand)
        ii = np.where(qt == t)[0]; n = len(ii)
        c0 = qn0[ii] @ n0[cand].T; c1 = qn1[ii] @ n1[cand].T
        drs = np.sqrt(((np.asarray(Qrs[qi_all[ii]], np.float32)[:, None, :8] - L_rs[cand][None]) ** 2).sum(2))
        S = w[0] * zt(c0, mu[0], sg[0]) + w[1] * zt(c1, mu[1], sg[1]) + w[2] * zt(-drs, mu[2], sg[2])
        oB = np.argsort(-S, 1); bB = oB[:, 0]
        CA = L_act[cand][:, :5, :7]; CAs = CA / sig
        res["b0"][ii] = err(CA[bB], Aq[ii], sig)
        # continuity
        T = tail_q[ii]; D = np.sqrt(np.mean((T[:, None] - CAs[None]) ** 2, axis=(2, 3)))
        oC = np.argsort(D, 1); bC = oC[:, 0]
        res["cont"][ii] = err(CA[bC], Aq[ii], sig); conf["cont_d"][ii] = -D[np.arange(n), bC]
        conf["cross"][ii] = -err(CA[bC], CA[bB], sig)
        conf["rs_of_cont"][ii] = -drs[np.arange(n), bC]; conf["b0score_of_cont"][ii] = S[np.arange(n), bC]
        rankB0 = np.argsort(oB, 1); conf["rankB0_of_cont"][ii] = -rankB0[np.arange(n), bC]
        for mm in (5, 10):
            sl = oC[:, :mm]; pick = sl[np.arange(n), np.argmax(np.take_along_axis(S, sl, 1), 1)]
            res[f"casc_c{mm}_b0"][ii] = err(CA[pick], Aq[ii], sig)
            sl = oB[:, :mm]; pick = sl[np.arange(n), np.argmin(np.take_along_axis(D, sl, 1), 1)]
            res[f"casc_b{mm}_c"][ii] = err(CA[pick], Aq[ii], sig)
        zD = (D - D.mean(1, keepdims=True)) / (D.std(1, keepdims=True) + 1e-6); zS = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)
        for wc, key in ((0.7, "z_c70"), (0.85, "z_c85")):
            pick = np.argmin(wc * zD - (1 - wc) * zS, 1); res[key][ii] = err(CA[pick], Aq[ii], sig)
        # consensus
        k5 = cand[oB[:, :5]]; res["cons_b5"][ii] = err(L_act[k5][:, :, :5, :7].mean(1), Aq[ii], sig)
        res["cons_b5_med"][ii] = err(np.median(L_act[k5][:, :, :5, :7], 1), Aq[ii], sig)
        k3 = cand[oC[:, :3]]; res["cons_c3"][ii] = err(L_act[k3][:, :, :5, :7].mean(1), Aq[ii], sig)
        # PCA-whitened joint L2 (vision only) and + rs
        Dp = np.sqrt(((Zq[ii][:, None] - Zl[cand][None]) ** 2).sum(2)); bp = np.argmin(Dp, 1)
        res["pca_w"][ii] = err(CA[bp], Aq[ii], sig); conf["pca_d"][ii] = -Dp[np.arange(n), bp]
        Dpr = np.sqrt(Dp ** 2 + (32.0) * ((Zrs_q[ii][:, None] - Zrs_l[cand][None]) ** 2).sum(2))  # rs block upweighted (8 dims vs 256)
        bpr = np.argmin(Dpr, 1); res["pca_w_rs"][ii] = err(CA[bpr], Aq[ii], sig)
        kk = cand[np.argsort(Dpr, 1)[:, :5]]; res["pca_w_cons5"][ii] = err(L_act[kk][:, :, :5, :7].mean(1), Aq[ii], sig)
        res["maha_cons5_med"][ii] = err(np.median(L_act[kk][:, :, :5, :7], 1), Aq[ii], sig)
        # continuity x observation product (both as z within task): conf
        conf["cont_x_pca"][ii] = -(zD[np.arange(n), bC] + (Dpr[np.arange(n), bC] - Dpr.mean(1)) / (Dpr.std(1) + 1e-6))
        # predicted action -> nearest library action
        for lam, key in ((10.0, "pred10"), (100.0, "pred100"), (1000.0, "pred1000")):
            P = pred[lam][ii].reshape(n, 5, 7); Dq = np.sqrt(np.mean((P[:, None] - CAs[None]) ** 2, axis=(2, 3))); bq = np.argmin(Dq, 1)
            res[key][ii] = err(CA[bq], Aq[ii], sig)
            if lam == 100.0:
                conf["pred_res"][ii] = -Dq[np.arange(n), bq]
                kk = cand[np.argsort(Dq, 1)[:, :5]]; res["pred_cons5"][ii] = err(L_act[kk][:, :, :5, :7].mean(1), Aq[ii], sig)
        P = pred2[ii].reshape(n, 5, 7); Dq = np.sqrt(np.mean((P[:, None] - CAs[None]) ** 2, axis=(2, 3))); bq = np.argmin(Dq, 1)
        res["pred2"][ii] = err(CA[bq], Aq[ii], sig); conf["pred2_res"][ii] = -Dq[np.arange(n), bq]
    out["sub_n"] = int(hpq.sum()); out["sub_B0_check"] = float(np.nanmean(res["b0"][hpq])); out["sub_B0_rec"] = float(eB0q[hpq].mean())
    for k in res:
        out[f"e_{k}"] = float(np.nanmean(res[k][hpq]))
    out["e_pca_w_all"] = float(np.nanmean(res["pca_w"])); out["e_pred100_all"] = float(np.nanmean(res["pred100"])); out["e_b0_all"] = float(np.nanmean(res["b0"]))
    for k in conf:
        base = res["cont"] if k in ("cont_d", "cross", "rs_of_cont", "b0score_of_cont", "rankB0_of_cont", "cont_x_pca") else (res["pred100"] if k == "pred_res" else res["pred2"] if k == "pred2_res" else res["pca_w"])
        out[f"aurc_{k}"] = aurc(conf[k][hpq], base[hpq])
    out["aurc_B0"] = aurc(confq[hpq], eB0q[hpq])
    # pred_res as confidence for B0's pick? and cross-agreement as confidence for B0's pick
    out["aurc_B0_by_cross"] = aurc(conf["cross"][hpq], res["b0"][hpq])
    out["aurc_pred_for_pred_all"] = aurc(conf["pred_res"], res["pred100"])
    return out


if __name__ == "__main__":
    with Pool(8) as pool:
        outs = pool.map(run, CELLS)
    keys = [k for k in outs[0] if k != "cell"]
    allk = []
    for o in outs:
        for k in o:
            if k != "cell" and k not in allk: allk.append(k)
    for o in outs:
        print(o["cell"], " ".join(f"{k}={o[k]:.3f}" if isinstance(o[k], float) else f"{k}={o[k]}" for k in allk if k in o), flush=True)
    json.dump(outs, open("/home/weiland/.claude/jobs/a607dd74/tmp/ideation_A/diag2.json", "w"), indent=1)
