"""Diag 3: continuity on 10x libraries, two-segment continuity (GR00T), floor gripper on transitions,
agreement confidence for B0 consensus, confidence combos for the continuity pick, staleness detectability."""
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


def gmis(a, b):
    return np.mean((a[..., :5, 6] >= 0) != (b[..., :5, 6] >= 0), axis=-1)


def aurc(conf, e):
    m = np.isfinite(conf) & np.isfinite(e); conf, e = conf[m], e[m]
    o = np.argsort(-conf, kind="stable"); ce = np.cumsum(e[o]) / np.arange(1, len(e) + 1)
    return float(ce.mean())


def zt(x, mu, sg):
    return 0.5 * (np.tanh((x - mu) / (sg if sg > 1e-12 else 1.0)) + 1)


def cont_search(tail, A, task, hp, L_act, L_task, sig, ks=(1, 3), tail2=None, L_ok=None):
    """tail: [N,5,7] prev tail; returns err of top-1 / mean top-k; optional 2nd segment; optional row mask."""
    N = len(A); out = {k: np.full(N, np.nan) for k in ks}; dmin = np.full(N, np.nan); pick = np.full(N, -1)
    for t in np.unique(task):
        cand = np.where((L_task == t) & (L_ok if L_ok is not None else True))[0]
        if len(cand) == 0: continue
        CA = L_act[cand][:, :5, :7] / sig
        qi = np.where((task == t) & hp)[0]; T = tail[qi] / sig
        D = np.mean((T[:, None] - CA[None]) ** 2, axis=(2, 3))
        if tail2 is not None:
            CB = L_act[cand][:, 5:10, :7] / sig; T2 = tail2[qi] / sig
            D = 0.5 * (D + np.mean((T2[:, None] - CB[None]) ** 2, axis=(2, 3)))
        D = np.sqrt(D); o = np.argsort(D, 1)
        for k in ks:
            out[k][qi] = err(L_act[cand[o[:, :k]]][:, :, :5, :7].mean(1), A[qi], sig)
        dmin[qi] = D[np.arange(len(qi)), o[:, 0]]; pick[qi] = cand[o[:, 0]]
    return out, dmin, pick


def run(cell):
    sys.path.insert(0, "/home/weiland/projects/openpi")
    from exp.offline_search.harness.store import current_params
    m, s, arm = cell.split("_")
    q = f"{ROOT}/queries/{cell}"; lib = f"{ROOT}/library/{m}_{s}/current"
    big = f"{ROOT}/library/{m}_{s}/" + ("bpool_cs" if m == "pi05" else "bpool_all")
    a_inf = np.load(f"{q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{q}/a_exec.npy", mmap_mode="r")
    ep = np.load(f"{q}/ep.npy"); step = np.load(f"{q}/step.npy")
    L_act = np.load(f"{lib}/action.npy"); L_task = np.load(f"{lib}/task_id.npy"); L_rs = np.load(f"{lib}/rs.npy")[:, :8].astype(np.float32)
    B_act = np.load(f"{big}/action.npy"); B_task = np.load(f"{big}/task_id.npy"); B_succ = np.load(f"{big}/success.npy").astype(bool)
    d = np.load(f"{R00}/B0_current/{cell}.npz")
    j = json.load(open(f"{R00}/B0_current/{cell}.json")); sig = np.asarray(j["sigma"], np.float32)
    rec = d["top1"]; eB0 = d["err"]; task = d["task_id"]; confB0 = d["confidence"]; binq = d["bin"]
    N = len(ep); H = a_inf.shape[1]
    A = np.asarray(a_inf[:, :5, :7], np.float32); E = np.asarray(a_exec, np.float32)
    prev = np.arange(N) - 1; hp = step >= 1
    g = A[:, :, 6] >= 0; trans = g.min(1) != g.max(1)
    pg = np.zeros(N, bool); pg[hp] = (E[prev[hp], 4, 6] >= 0) != g[hp, 0]; trans |= pg
    out = {"cell": cell, "n": int(hp.sum())}
    tail = E[prev, 5:10, :7]
    # staleness detectability: prev executed chunk bit-equal to the recorded pick's library action (cache arm)
    if arm == "cache":
        out["stale_bitexact"] = float(np.all(E[prev[hp], :, :7] == L_act[rec[prev[hp]]][:, :, :7], axis=(1, 2)).mean())
    else:
        # in inf arm: how often does a fresh chunk coincide with any library chunk (false stale)? check head equality to any lib row of task
        eq = np.zeros(hp.sum(), bool)
        for t in np.unique(task):
            cand = L_act[L_task == t][:, :5, :7]; qi = np.where((task == t) & hp)[0]
            eq[np.searchsorted(np.where(hp)[0], qi)] = np.any(np.all(E[qi][:, None, :5, :7] == cand[None], axis=(2, 3)), 1)
        out["stale_bitexact"] = float(eq.mean())
    # (a) continuity on current vs big library, success-only variant, consensus k=3
    o_cur, dmin_cur, pick_cur = cont_search(tail, A, task, hp, L_act, L_task, sig)
    o_big, dmin_big, _ = cont_search(tail, A, task, hp, B_act, B_task, sig)
    o_bigs, _, _ = cont_search(tail, A, task, hp, B_act, B_task, sig, L_ok=B_succ)
    for lab, o in (("cur", o_cur), ("big", o_big), ("bigS", o_bigs)):
        out[f"cont1_{lab}"] = float(np.nanmean(o[1][hp])); out[f"cont3_{lab}"] = float(np.nanmean(o[3][hp]))
    out["B0"] = float(eB0[hp].mean()); out["orc_cur"] = float(d["oracle_err"][hp].mean())
    # by step bin (inf arms mainly)
    for b in (0, 1, 2):
        mk = hp & (binq == b); out[f"bin{b}_B0"] = float(eB0[mk].mean()); out[f"bin{b}_cont3"] = float(np.nanmean(o_cur[3][mk]))
    out["tr_B0"] = float(eB0[hp & trans].mean()); out["tr_cont3"] = float(np.nanmean(o_cur[3][hp & trans]))
    # (b) two-segment continuity for GR00T (H=16): prev [5:10]->head, prev [10:15]->cand [5:10]
    if H >= 15:
        tail2 = E[prev, 10:15, :7]
        o2, dmin2, _ = cont_search(tail, A, task, hp, L_act, L_task, sig, tail2=tail2)
        out["cont1_2seg"] = float(np.nanmean(o2[1][hp])); out["cont3_2seg"] = float(np.nanmean(o2[3][hp]))
        out["aurc_2seg"] = aurc(-dmin2[hp], o2[1][hp])
    # (c) floor gripper mismatch on transitions
    z = np.load(f"{ROOT}/floor/{m}_{s}/resample.npz")
    ar = z["a_recorded"][:, :5, :7]; af = z["a_fresh"][:, :, :5, :7]; ok = np.isfinite(af).all(axis=(1, 2, 3)) & (z["sec"] > 0)
    gr = ar[:, :, 6] >= 0; tr_f = gr.min(1) != gr.max(1)
    gmf = np.mean([gmis(af[:, k], ar) for k in range(af.shape[1])], 0)
    out["floor_gm_tr"] = float(gmf[ok & tr_f].mean()); out["floor_gm_all"] = float(gmf[ok].mean()); out["floor_tr_share"] = float(tr_f[ok].mean())
    errf = np.mean([err(af[:, k], ar, sig) for k in range(af.shape[1])], 0)
    out["floor_err_tr"] = float(errf[ok & tr_f].mean()); out["floor_err_all"] = float(errf[ok].mean())
    # (d) subsample with keys: B0 top-5 consensus + agreement conf; conf combos for continuity pick
    rng = np.random.default_rng(0); qi_all = np.sort(rng.choice(N, size=min(NQ, N), replace=False))
    p = current_params(m, s); w = np.asarray(p["weights"], np.float64); mu = p["mu"]; sg = p["sigma"]
    K0 = np.load(f"{lib}/key_v0.npy", mmap_mode="r"); K1 = np.load(f"{lib}/key_v1.npy", mmap_mode="r")
    Q0 = np.load(f"{q}/key_v0.npy", mmap_mode="r"); Q1 = np.load(f"{q}/key_v1.npy", mmap_mode="r"); Qrs = np.load(f"{q}/rs.npy", mmap_mode="r")
    Lk0 = np.asarray(K0, np.float32); Lk1 = np.asarray(K1, np.float32)
    n0 = Lk0 / np.linalg.norm(Lk0, axis=1, keepdims=True); n1 = Lk1 / np.linalg.norm(Lk1, axis=1, keepdims=True)
    Qk0 = np.asarray(Q0[qi_all], np.float32); Qk1 = np.asarray(Q1[qi_all], np.float32)
    qn0 = Qk0 / np.linalg.norm(Qk0, axis=1, keepdims=True); qn1 = Qk1 / np.linalg.norm(Qk1, axis=1, keepdims=True)
    qt = task[qi_all]; Aq = A[qi_all]; hpq = hp[qi_all]; tq = tail[qi_all]
    n = len(qi_all)
    R = {k: np.full(n, np.nan) for k in ["b0", "cons5", "cons3", "cont1", "cont3", "gate"]}
    Cf = {k: np.full(n, np.nan) for k in ["b0", "agree5", "agree5_med", "margin", "cont_d", "obs_of_cont", "zmin", "zsum", "obs_rank", "agree_c3", "cont_d_x_agree"]}
    for t in np.unique(qt):
        cand = np.where(L_task == t)[0]; C = len(cand); ii = np.where(qt == t)[0]; nn = len(ii)
        c0 = qn0[ii] @ n0[cand].T; c1 = qn1[ii] @ n1[cand].T
        drs = np.sqrt(((np.asarray(Qrs[qi_all[ii]], np.float32)[:, None, :8] - L_rs[cand][None]) ** 2).sum(2))
        S = w[0] * zt(c0, mu[0], sg[0]) + w[1] * zt(c1, mu[1], sg[1]) + w[2] * zt(-drs, mu[2], sg[2])
        oB = np.argsort(-S, 1); CA = L_act[cand][:, :5, :7]; CAs = CA / sig
        R["b0"][ii] = err(CA[oB[:, 0]], Aq[ii], sig); Cf["b0"][ii] = S[np.arange(nn), oB[:, 0]]
        Cf["margin"][ii] = S[np.arange(nn), oB[:, 0]] - S[np.arange(nn), oB[:, 1]]
        k5 = CA[oB[:, :5]]; mean5 = k5.mean(1); R["cons5"][ii] = err(mean5, Aq[ii], sig); R["cons3"][ii] = err(CA[oB[:, :3]].mean(1), Aq[ii], sig)
        Cf["agree5"][ii] = -np.sqrt(np.mean(((k5 - mean5[:, None]) / sig) ** 2, axis=(1, 2, 3)))
        med5 = np.median(k5, 1); Cf["agree5_med"][ii] = -np.sqrt(np.mean(((k5 - med5[:, None]) / sig) ** 2, axis=(1, 2, 3)))
        # continuity
        T = tq[ii] / sig; D = np.sqrt(np.mean((T[:, None] - CAs[None]) ** 2, axis=(2, 3))); oC = np.argsort(D, 1); bC = oC[:, 0]
        R["cont1"][ii] = err(CA[bC], Aq[ii], sig); k3 = CA[oC[:, :3]]; m3 = k3.mean(1); R["cont3"][ii] = err(m3, Aq[ii], sig)
        Cf["cont_d"][ii] = -D[np.arange(nn), bC]; Cf["obs_of_cont"][ii] = S[np.arange(nn), bC]
        Cf["agree_c3"][ii] = -np.sqrt(np.mean(((k3 - m3[:, None]) / sig) ** 2, axis=(1, 2, 3)))
        zD = (D - D.mean(1, keepdims=True)) / (D.std(1, keepdims=True) + 1e-6); zS = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)
        zc = -zD[np.arange(nn), bC]; zo = zS[np.arange(nn), bC]
        Cf["zmin"][ii] = np.minimum(zc, zo); Cf["zsum"][ii] = zc + zo
        rankB = np.argsort(oB, 1); Cf["obs_rank"][ii] = -rankB[np.arange(nn), bC]
        Cf["cont_d_x_agree"][ii] = Cf["cont_d"][ii] + Cf["agree_c3"][ii]
        # gated: if cont pick within B0 top-10 use cont3 else cons5
        inb = rankB[np.arange(nn), bC] < 10
        R["gate"][ii] = np.where(inb, R["cont3"][ii], R["cons5"][ii])
    for k in R:
        out[f"e_{k}"] = float(np.nanmean(R[k][hpq]))
    out["aurc_B0"] = aurc(Cf["b0"][hpq], R["b0"][hpq]); out["aurc_cons5_b0conf"] = aurc(Cf["b0"][hpq], R["cons5"][hpq])
    out["aurc_cons5_agree"] = aurc(Cf["agree5"][hpq], R["cons5"][hpq]); out["aurc_cons5_agreemed"] = aurc(Cf["agree5_med"][hpq], R["cons5"][hpq])
    out["aurc_cons5_margin"] = aurc(Cf["margin"][hpq], R["cons5"][hpq]); out["aurc_b0_agree"] = aurc(Cf["agree5"][hpq], R["b0"][hpq])
    out["aurc_cons5_b0conf_plus_agree"] = aurc(((Cf["b0"] - np.nanmean(Cf["b0"])) / np.nanstd(Cf["b0"]) + (Cf["agree5"] - np.nanmean(Cf["agree5"])) / np.nanstd(Cf["agree5"]))[hpq], R["cons5"][hpq])
    for k in ["cont_d", "obs_of_cont", "zmin", "zsum", "obs_rank", "agree_c3", "cont_d_x_agree"]:
        out[f"aurc_cont3_{k}"] = aurc(Cf[k][hpq], R["cont3"][hpq])
    np.savez(f"/home/weiland/.claude/jobs/a607dd74/tmp/ideation_A/diag3_{cell}.npz", **{f"R_{k}": v for k, v in R.items()}, **{f"C_{k}": v for k, v in Cf.items()}, hpq=hpq, qi=qi_all)
    return out


if __name__ == "__main__":
    with Pool(8) as pool:
        outs = pool.map(run, CELLS)
    allk = []
    for o in outs:
        for k in o:
            if k != "cell" and k not in allk: allk.append(k)
    for o in outs:
        print(o["cell"], " ".join(f"{k}={o[k]:.3f}" if isinstance(o[k], float) else f"{k}={o[k]}" for k in allk if k in o), flush=True)
    json.dump(outs, open("/home/weiland/.claude/jobs/a607dd74/tmp/ideation_A/diag3.json", "w"), indent=1)
