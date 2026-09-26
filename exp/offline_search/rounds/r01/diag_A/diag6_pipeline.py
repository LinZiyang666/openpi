"""Diag 6: assembled vision-free pipeline per cell (fresh: continuity kernel top-5; stale/step0: state-window top-5),
confidence variants, gripper on transitions, gripper-context hard filter. All decisions."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from multiprocessing import Pool

ROOT = "/dev/shm/offline_search_store"; R00 = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
CELLS = ["pi05_spatial_inf", "pi05_spatial_cache", "pi05_l10_inf", "pi05_l10_cache",
         "groot_spatial_inf", "groot_spatial_cache", "groot_l10_inf", "groot_l10_cache"]


def err(a, b, sig):
    d = (a[..., :5, :7] - b[..., :5, :7]) / sig
    return np.sqrt(np.mean(d * d, axis=(-2, -1)))


def gmis(a, b):
    return np.mean((a[..., :5, 6] >= 0) != (b[..., :5, 6] >= 0), axis=-1)


def aurc(conf, e):
    m = np.isfinite(conf) & np.isfinite(e); conf, e = conf[m], e[m]
    o = np.argsort(-conf, kind="stable"); ce = np.cumsum(e[o]) / np.arange(1, len(e) + 1)
    return float(ce.mean())


def risk_at(conf, e, c):
    o = np.argsort(-conf, kind="stable"); k = max(1, int(round(c * len(e)))); return float(e[o[:k]].mean())


def windows(rs, prev_idx, m):
    n = len(rs); out = np.empty((n, 8 * m), np.float32); cur = np.arange(n)
    for j in range(m):
        out[:, 8 * j:8 * (j + 1)] = rs[cur]; nxt = prev_idx(cur); cur = np.where(nxt >= 0, nxt, cur)
    return out


def run(cell):
    m_, s, arm = cell.split("_")
    q = f"{ROOT}/queries/{cell}"; lib = f"{ROOT}/library/{m_}_{s}/current"; big = f"{ROOT}/library/{m_}_{s}/" + ("bpool_cs" if m_ == "pi05" else "bpool_all")
    a_inf = np.load(f"{q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{q}/a_exec.npy", mmap_mode="r"); step = np.load(f"{q}/step.npy")
    Qrs = np.load(f"{q}/rs.npy")[:, :8].astype(np.float32)
    d = np.load(f"{R00}/B0_current/{cell}.npz"); j = json.load(open(f"{R00}/B0_current/{cell}.json")); sig = np.asarray(j["sigma"], np.float32)
    eB0 = d["err"]; gB0 = d["grip_mis"]; cB0 = d["confidence"]; task = d["task_id"]; N = len(step); prev = np.arange(N) - 1; hp = step >= 1
    A = np.asarray(a_inf[:, :5, :7], np.float32); E = np.asarray(a_exec, np.float32); tail = E[prev, 5:10, :7]
    g = A[:, :, 6] >= 0; trans = g.min(1) != g.max(1); pg = np.zeros(N, bool); pg[hp] = (E[prev[hp], 4, 6] >= 0) != g[hp, 0]; trans |= pg
    L_act = np.load(f"{lib}/action.npy")
    lib_hash = set(bytes(r) for r in np.ascontiguousarray(L_act[:, :, :7]).view(np.uint8).reshape(len(L_act), -1))
    stale = np.zeros(N, bool); stale[hp] = [bytes(r) in lib_hash for r in np.ascontiguousarray(E[prev[hp]][:, :, :7]).view(np.uint8).reshape(hp.sum(), -1)]
    fresh = hp & ~stale
    B_act = np.load(f"{big}/action.npy"); B_task = np.load(f"{big}/task_id.npy"); B_rs = np.load(f"{big}/rs.npy")[:, :8].astype(np.float32); B_prev = np.load(f"{big}/prev.npy")
    sd = B_rs.std(0) + 1e-6
    QW = windows(Qrs / sd, lambda i: np.where(step[i] >= 1, i - 1, -1), 3); LW = windows(B_rs / sd, lambda i: B_prev[i], 3)
    # gripper context of library rows: last executed gripper command of the previous chunk (or own first if no prev)
    B_gctx = np.where(B_prev >= 0, B_act[np.maximum(B_prev, 0)][:, 4, 6] >= 0, B_act[:, 0, 6] >= 0)
    q_gctx = np.zeros(N, bool); q_gctx[hp] = E[prev[hp], 4, 6] >= 0; q_gctx[~hp] = A[~hp, 0, 6] >= 0  # step0: unknown online -> use no filter there
    out = {"cell": cell, "n": N, "fresh_share": float(fresh.mean()), "stale_share": float(stale.mean())}
    act = np.full((N, 5, 7), np.nan, np.float32); conf_d = np.full(N, np.nan); conf_ag = np.full(N, np.nan)
    act_g = np.full((N, 5, 7), np.nan, np.float32)  # with gripper-context filter (stale/step0 branch only)
    act_sw = np.full((N, 5, 7), np.nan, np.float32); conf_sw = np.full(N, np.nan)  # state-window everywhere
    for t in np.unique(task):
        cand = np.where(B_task == t)[0]; CA = B_act[cand][:, :5, :7]; CAs = CA / sig
        # fresh: continuity kernel top-5
        qi = np.where((task == t) & fresh)[0]
        if len(qi):
            T = tail[qi] / sig; D = np.sqrt(np.mean((T[:, None] - CAs[None]) ** 2, axis=(2, 3))); o = np.argsort(D, 1)[:, :5]
            dk = np.take_along_axis(D, o, 1); w = np.exp(-(dk / (dk[:, :1] + 1e-6)) ** 2); w /= w.sum(1, keepdims=True)
            act[qi] = (CA[o] * w[:, :, None, None]).sum(1); conf_d[qi] = -dk[:, 0]
            conf_ag[qi] = -np.sqrt(np.mean(((CA[o] - act[qi][:, None]) / sig) ** 2, axis=(1, 2, 3)))
        # stale/step0: state window top-5 mean (and everywhere for the pure state-window method)
        qi = np.where(task == t)[0]
        D = np.sqrt(((QW[qi][:, None] - LW[cand][None]) ** 2).sum(2)); o = np.argsort(D, 1)[:, :5]
        act_sw[qi] = CA[o].mean(1); conf_sw[qi] = -D[np.arange(len(qi)), o[:, 0]]
        ag = -np.sqrt(np.mean(((CA[o] - act_sw[qi][:, None]) / sig) ** 2, axis=(1, 2, 3)))
        sel = qi[~fresh[qi]]; act[sel] = act_sw[sel]; conf_d[sel] = conf_sw[sel]; conf_ag[sel] = ag[~fresh[qi]]
        # gripper-context filter on the stale/step0 branch (only for hp rows)
        qi2 = qi[stale[qi]]
        if len(qi2):
            Dg = D[stale[qi]].copy(); mask = B_gctx[cand][None, :] != q_gctx[qi2][:, None]; Dg[mask] = np.inf
            og = np.argsort(Dg, 1)[:, :5]; act_g[qi2] = CA[og].mean(1)
    e = err(act, A, sig); gm = gmis(act, A); esw = err(act_sw, A, sig)
    out["e_pipe"] = float(e.mean()); out["e_B0"] = float(eB0.mean()); out["e_sw_all"] = float(esw.mean())
    out["e_pipe_fresh"] = float(e[fresh].mean()) if fresh.any() else float("nan"); out["e_pipe_stale"] = float(e[stale].mean()) if stale.any() else float("nan")
    out["e_pipe_step0"] = float(e[~hp].mean()); out["e_B0_step0"] = float(eB0[~hp].mean())
    out["p50_pipe"] = float(np.median(e)); out["p50_B0"] = float(np.median(eB0))
    out["gm_pipe"] = float(gm.mean()); out["gm_B0"] = float(gB0.mean()); out["gm_pipe_tr"] = float(gm[trans].mean()); out["gm_B0_tr"] = float(gB0[trans].mean())
    out["e_pipe_tr"] = float(e[trans].mean()); out["e_B0_tr"] = float(eB0[trans].mean())
    if stale.any():
        eg = err(act_g, A, sig); out["e_stale_gfilter"] = float(np.nanmean(eg[stale])); out["gm_stale_gfilter"] = float(np.nanmean(gmis(act_g, A)[stale])); out["gm_stale_nofilter"] = float(gm[stale].mean())
    zs = lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    out["aurc_B0"] = aurc(cB0, eB0); out["aurc_pipe_d"] = aurc(conf_d, e); out["aurc_pipe_ag"] = aurc(conf_ag, e); out["aurc_pipe_d+ag"] = aurc(zs(conf_d) + zs(conf_ag), e)
    out["aurc_sw_d"] = aurc(conf_sw, esw)
    out["r30_B0"] = risk_at(cB0, eB0, 0.3); out["r30_pipe"] = risk_at(conf_d, e, 0.3); out["r50_B0"] = risk_at(cB0, eB0, 0.5); out["r50_pipe"] = risk_at(conf_d, e, 0.5)
    out["r70_B0"] = risk_at(cB0, eB0, 0.7); out["r70_pipe"] = risk_at(conf_d, e, 0.7)
    fl = json.load(open(f"{R00}/B0_current/{cell}.json"))["floor"]
    out["indist_pipe"] = float(np.mean(e <= np.array([fl["median"][int(b)] if isinstance(fl.get("median"), list) else np.nan for b in d["bin"]]))) if isinstance(fl.get("median"), list) else float("nan")
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
    json.dump(outs, open("/home/weiland/.claude/jobs/a607dd74/tmp/ideation_A/diag6.json", "w"), indent=1)
