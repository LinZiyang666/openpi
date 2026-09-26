"""Diag 5: state-history window kNN (vision-free observation side), continuity consensus k / kernel weights on
the big library, AURC of continuity on big library. All decisions, all cells, tiny arrays."""
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


def aurc(conf, e):
    m = np.isfinite(conf) & np.isfinite(e); conf, e = conf[m], e[m]
    o = np.argsort(-conf, kind="stable"); ce = np.cumsum(e[o]) / np.arange(1, len(e) + 1)
    return float(ce.mean())


def windows(rs, prev_idx, m):
    """rs [n,8]; prev_idx: function i -> previous row or -1; returns [n, 8*m] (pad by repeating the oldest)."""
    n = len(rs); out = np.empty((n, 8 * m), np.float32); cur = np.arange(n)
    for j in range(m):
        out[:, 8 * j:8 * (j + 1)] = rs[cur]
        nxt = prev_idx(cur); cur = np.where(nxt >= 0, nxt, cur)
    return out


def run(cell):
    m_, s, arm = cell.split("_")
    q = f"{ROOT}/queries/{cell}"; lib = f"{ROOT}/library/{m_}_{s}/current"; big = f"{ROOT}/library/{m_}_{s}/" + ("bpool_cs" if m_ == "pi05" else "bpool_all")
    a_inf = np.load(f"{q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{q}/a_exec.npy", mmap_mode="r"); step = np.load(f"{q}/step.npy")
    Qrs = np.load(f"{q}/rs.npy")[:, :8].astype(np.float32)
    d = np.load(f"{R00}/B0_current/{cell}.npz"); j = json.load(open(f"{R00}/B0_current/{cell}.json")); sig = np.asarray(j["sigma"], np.float32)
    eB0 = d["err"]; task = d["task_id"]; N = len(step); prev = np.arange(N) - 1; hp = step >= 1
    A = np.asarray(a_inf[:, :5, :7], np.float32); E = np.asarray(a_exec, np.float32); tail = E[prev, 5:10, :7]
    out = {"cell": cell, "B0": float(eB0.mean()), "B0_hp": float(eB0[hp].mean())}
    q_prev = lambda i: np.where(step[i] >= 1, i - 1, -1)
    for libname, path in (("cur", lib), ("big", big)):
        L_act = np.load(f"{path}/action.npy"); L_task = np.load(f"{path}/task_id.npy"); L_rs = np.load(f"{path}/rs.npy")[:, :8].astype(np.float32)
        L_prev = np.load(f"{path}/prev.npy")
        l_prev = lambda i: L_prev[i]
        sd = L_rs.std(0) + 1e-6
        for m in (1, 3, 5):
            QW = windows(Qrs / sd, q_prev, m); LW = windows(L_rs / sd, l_prev, m)
            e1 = np.full(N, np.nan); e5 = np.full(N, np.nan); dm = np.full(N, np.nan); ec = np.full(N, np.nan)
            for t in np.unique(task):
                cand = np.where(L_task == t)[0]; qi = np.where(task == t)[0]
                D = np.sqrt(((QW[qi][:, None] - LW[cand][None]) ** 2).sum(2)); o = np.argsort(D, 1)
                e1[qi] = err(L_act[cand[o[:, 0]]], A[qi], sig); e5[qi] = err(L_act[cand[o[:, :5]]][:, :, :5, :7].mean(1), A[qi], sig)
                dm[qi] = D[np.arange(len(qi)), o[:, 0]]
                if libname == "cur" and m == 3:
                    # window + continuity z-sum (fresh regime)
                    qh = qi[hp[qi]]; T = tail[qh] / sig; CAs = L_act[cand][:, :5, :7] / sig
                    Dc = np.sqrt(np.mean((T[:, None] - CAs[None]) ** 2, axis=(2, 3))); Dw = D[hp[qi]]
                    z = (Dc - Dc.mean(1, keepdims=True)) / (Dc.std(1, keepdims=True) + 1e-6) + 0.5 * (Dw - Dw.mean(1, keepdims=True)) / (Dw.std(1, keepdims=True) + 1e-6)
                    oc = np.argsort(z, 1); ec[qh] = err(L_act[cand[oc[:, :3]]][:, :, :5, :7].mean(1), A[qh], sig)
            out[f"rsw{m}_{libname}_e1"] = float(np.nanmean(e1)); out[f"rsw{m}_{libname}_e5"] = float(np.nanmean(e5)); out[f"rsw{m}_{libname}_aurc5"] = aurc(-dm, e5)
            if libname == "cur" and m == 3:
                out["rsw3_plus_cont3"] = float(np.nanmean(ec[hp]))
        # continuity consensus k and kernel weighting on this library
        for k in (1, 3, 5, 8):
            e = np.full(N, np.nan); dm = np.full(N, np.nan); ek = np.full(N, np.nan)
            for t in np.unique(task):
                cand = np.where(L_task == t)[0]; qi = np.where((task == t) & hp)[0]; T = tail[qi] / sig; CAs = L_act[cand][:, :5, :7] / sig
                D = np.sqrt(np.mean((T[:, None] - CAs[None]) ** 2, axis=(2, 3))); o = np.argsort(D, 1)
                e[qi] = err(L_act[cand[o[:, :k]]][:, :, :5, :7].mean(1), A[qi], sig); dm[qi] = D[np.arange(len(qi)), o[:, 0]]
                if k == 5:
                    dk = np.take_along_axis(D, o[:, :5], 1); wgt = np.exp(-(dk / (dk[:, :1] + 1e-6)) ** 2); wgt /= wgt.sum(1, keepdims=True)
                    ek[qi] = err((L_act[cand[o[:, :5]]][:, :, :5, :7] * wgt[:, :, None, None]).sum(1), A[qi], sig)
            out[f"cont{k}_{libname}"] = float(np.nanmean(e[hp]))
            if k == 1: out[f"cont_aurc_{libname}"] = aurc(-dm[hp], e[hp])
            if k == 5: out[f"cont5k_{libname}"] = float(np.nanmean(ek[hp]))
    return out


if __name__ == "__main__":
    with Pool(8) as pool:
        outs = pool.map(run, CELLS)
    allk = [k for k in outs[0] if k != "cell"]
    for o in outs:
        print(o["cell"], " ".join(f"{k}={o[k]:.3f}" for k in allk), flush=True)
    json.dump(outs, open("/home/weiland/.claude/jobs/a607dd74/tmp/ideation_A/diag5.json", "w"), indent=1)
