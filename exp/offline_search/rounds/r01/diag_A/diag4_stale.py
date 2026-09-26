"""Diag 4 (cache arms, diag3 subsample): stale-regime selection variants anchored on the identified previous
library row p (hash of prev executed chunk) and state-consistency confidences."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from multiprocessing import Pool

ROOT = "/dev/shm/offline_search_store"; R00 = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
TMP = "/home/weiland/.claude/jobs/a607dd74/tmp/ideation_A"
CELLS = ["pi05_spatial_cache", "pi05_l10_cache", "groot_spatial_cache", "groot_l10_cache"]


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
    step = np.load(f"{q}/step.npy")
    L_act = np.load(f"{lib}/action.npy"); L_task = np.load(f"{lib}/task_id.npy"); L_rs = np.load(f"{lib}/rs.npy")[:, :8].astype(np.float32)
    L_next = np.load(f"{lib}/next.npy")
    d = np.load(f"{R00}/B0_current/{cell}.npz"); j = json.load(open(f"{R00}/B0_current/{cell}.json")); sig = np.asarray(j["sigma"], np.float32)
    rec = d["top1"]; eB0 = d["err"]; task = d["task_id"]
    z = np.load(f"{TMP}/diag3_{cell}.npz"); qi = z["qi"]; hpq = z["hpq"]
    N = len(step); prev = np.arange(N) - 1
    A = np.asarray(a_inf[:, :5, :7], np.float32); E = np.asarray(a_exec, np.float32)
    # identify p by hashing library chunks (valid dims, full horizon)
    keyf = lambda a: np.ascontiguousarray(a[:, :, :7]).view(np.uint8).reshape(a.shape[0], -1)
    lib_hash = {bytes(r): i for i, r in enumerate(keyf(L_act))}
    P = np.array([lib_hash.get(bytes(r), -1) for r in keyf(E[prev[qi]])]); P[~hpq] = -1
    out = {"cell": cell, "p_found": float((P[hpq] >= 0).mean()), "p_eq_rec": float((P[hpq] == rec[prev[qi]][hpq]).mean())}
    p = current_params(m, s); w = np.asarray(p["weights"], np.float64); mu = p["mu"]; sg = p["sigma"]
    K0 = np.load(f"{lib}/key_v0.npy", mmap_mode="r"); K1 = np.load(f"{lib}/key_v1.npy", mmap_mode="r")
    Q0 = np.load(f"{q}/key_v0.npy", mmap_mode="r"); Q1 = np.load(f"{q}/key_v1.npy", mmap_mode="r"); Qrs = np.load(f"{q}/rs.npy", mmap_mode="r")
    Lk0 = np.asarray(K0, np.float32); Lk1 = np.asarray(K1, np.float32)
    n0 = Lk0 / np.linalg.norm(Lk0, axis=1, keepdims=True); n1 = Lk1 / np.linalg.norm(Lk1, axis=1, keepdims=True)
    Qk0 = np.asarray(Q0[qi], np.float32); Qk1 = np.asarray(Q1[qi], np.float32); Qr = np.asarray(Qrs[qi], np.float32)[:, :8]
    qn0 = Qk0 / np.linalg.norm(Qk0, axis=1, keepdims=True); qn1 = Qk1 / np.linalg.norm(Qk1, axis=1, keepdims=True)
    qt = task[qi]; Aq = A[qi]; tq = E[prev[qi], 5:10, :7]; n = len(qi)
    R = {k: np.full(n, np.nan) for k in ["b0", "cons5", "cont3", "track", "track_cont2", "track_cons4", "cont3_nb", "cons5_nb"]}
    Cf = {k: np.full(n, np.nan) for k in ["b0", "obs_track", "obs_p", "rs_next", "rs_p", "obs_track_plus_rs", "cont_d", "agree_tc2", "obs_track_x_agree"]}
    for t in np.unique(qt):
        cand = np.where(L_task == t)[0]; ii = np.where(qt == t)[0]; nn = len(ii); CA = L_act[cand][:, :5, :7]; CAs = CA / sig
        c0 = qn0[ii] @ n0[cand].T; c1 = qn1[ii] @ n1[cand].T
        drs = np.sqrt(((Qr[ii][:, None] - L_rs[cand][None]) ** 2).sum(2))
        S = w[0] * zt(c0, mu[0], sg[0]) + w[1] * zt(c1, mu[1], sg[1]) + w[2] * zt(-drs, mu[2], sg[2])
        oB = np.argsort(-S, 1); R["b0"][ii] = err(CA[oB[:, 0]], Aq[ii], sig); Cf["b0"][ii] = S[np.arange(nn), oB[:, 0]]
        R["cons5"][ii] = err(CA[oB[:, :5]].mean(1), Aq[ii], sig)
        T = tq[ii] / sig; D = np.sqrt(np.mean((T[:, None] - CAs[None]) ** 2, axis=(2, 3))); oC = np.argsort(D, 1)
        R["cont3"][ii] = err(CA[oC[:, :3]].mean(1), Aq[ii], sig); Cf["cont_d"][ii] = -D[np.arange(nn), oC[:, 0]]
        # tracking anchored on p
        pp = P[ii]; nx = np.where(pp >= 0, L_next[np.maximum(pp, 0)], -1); ok = nx >= 0
        loc = {r: k for k, r in enumerate(cand)}
        for a in range(nn):
            if not ok[a]:
                continue
            r = nx[a]; k = loc.get(int(r))
            if k is None:
                ok[a] = False; continue
            R["track"][ii[a]] = err(CA[k], Aq[ii[a]], sig)
            others = [c for c in oC[a, :3] if c != k][:2]
            R["track_cont2"][ii[a]] = err(np.mean([CA[k]] + [CA[c] for c in others], 0), Aq[ii[a]], sig)
            others = [c for c in oB[a, :5] if c != k][:4]
            R["track_cons4"][ii[a]] = err(np.mean([CA[k]] + [CA[c] for c in others], 0), Aq[ii[a]], sig)
            Cf["obs_track"][ii[a]] = S[a, k]; Cf["rs_next"][ii[a]] = -drs[a, k]
            Cf["agree_tc2"][ii[a]] = -np.sqrt(np.mean(((np.stack([CA[k]] + [CA[c] for c in [c for c in oC[a, :3] if c != k][:2]]) - np.mean([CA[k]] + [CA[c] for c in [c for c in oC[a, :3] if c != k][:2]], 0)) / sig) ** 2))
            kp = loc.get(int(pp[a]))
            if kp is not None:
                Cf["obs_p"][ii[a]] = S[a, kp]; Cf["rs_p"][ii[a]] = -drs[a, kp]
        # no-track available rows: fall back
        R["cont3_nb"][ii] = R["cont3"][ii]; R["cons5_nb"][ii] = R["cons5"][ii]
    okm = hpq & np.isfinite(R["track"])
    out["n_ok"] = int(okm.sum()); out["share_ok"] = float(okm.sum() / hpq.sum())
    for k in ["b0", "cons5", "cont3", "track", "track_cont2", "track_cons4"]:
        out[f"e_{k}"] = float(np.nanmean(R[k][okm]))
    out["e_b0_all"] = float(np.nanmean(R["b0"][hpq])); out["e_cons5_all"] = float(np.nanmean(R["cons5"][hpq]))
    zs = lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    Cf["obs_track_plus_rs"] = zs(Cf["obs_track"]) + zs(Cf["rs_next"]); Cf["obs_track_x_agree"] = zs(Cf["obs_track"]) + zs(Cf["agree_tc2"])
    out["aurc_B0"] = aurc(Cf["b0"][okm], R["b0"][okm])
    for sel in ["track", "track_cont2", "cons5", "cont3"]:
        for cf in ["b0", "obs_track", "obs_p", "rs_next", "rs_p", "obs_track_plus_rs", "cont_d", "agree_tc2", "obs_track_x_agree"]:
            out[f"aurc_{sel}__{cf}"] = aurc(Cf[cf][okm], R[sel][okm])
    return out


if __name__ == "__main__":
    with Pool(4) as pool:
        outs = pool.map(run, CELLS)
    allk = []
    for o in outs:
        for k in o:
            if k != "cell" and k not in allk: allk.append(k)
    for o in outs:
        print(o["cell"], " ".join(f"{k}={o[k]:.3f}" if isinstance(o[k], float) else f"{k}={o[k]}" for k in allk if k in o), flush=True)
    json.dump(outs, open(f"{TMP}/diag4.json", "w"), indent=1)
