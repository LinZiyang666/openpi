"""Diag 1: action-continuity and trajectory-tracking signals (all decisions, all cells). CPU, seconds."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np

ROOT = "/dev/shm/offline_search_store"
R00 = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
CELLS = ["pi05_spatial_inf", "pi05_spatial_cache", "pi05_l10_inf", "pi05_l10_cache",
         "groot_spatial_inf", "groot_spatial_cache", "groot_l10_inf", "groot_l10_cache"]


def err(a, b, sig):
    d = (a[..., :5, :7] - b[..., :5, :7]) / sig
    return np.sqrt(np.mean(d * d, axis=(-2, -1)))


def gmis(a, b):
    return np.mean((a[..., :5, 6] >= 0) != (b[..., :5, 6] >= 0), axis=-1)


def aurc(conf, e):
    o = np.argsort(-conf, kind="stable"); ce = np.cumsum(e[o]) / np.arange(1, len(e) + 1)
    return float(ce.mean())


rows = []
for cell in CELLS:
    m, s, arm = cell.split("_")
    q = f"{ROOT}/queries/{cell}"; lib = f"{ROOT}/library/{m}_{s}/current"
    a_inf = np.load(f"{q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{q}/a_exec.npy", mmap_mode="r")
    ep = np.load(f"{q}/ep.npy"); step = np.load(f"{q}/step.npy"); rs = np.load(f"{q}/rs.npy", mmap_mode="r")[:, :8]
    L_act = np.load(f"{lib}/action.npy"); L_next = np.load(f"{lib}/next.npy"); L_task = np.load(f"{lib}/task_id.npy")
    L_rs = np.load(f"{lib}/rs.npy")[:, :8]; L_step = np.load(f"{lib}/step.npy"); L_prog = np.load(f"{lib}/progress.npy")
    d = np.load(f"{R00}/B0_current/{cell}.npz")
    j = json.load(open(f"{R00}/B0_current/{cell}.json")); sig = np.asarray(j["sigma"], np.float32)
    rec = d["top1"]; orc = d["oracle_row"]; eB0 = d["err"]; task = d["task_id"]; conf = d["confidence"]
    N = len(ep); H = a_inf.shape[1]
    A = np.asarray(a_inf[:, :5, :7], np.float32)
    E = np.asarray(a_exec, np.float32)
    # previous decision of same episode
    prev = np.arange(N) - 1; has_prev = (step >= 1)
    assert np.all(ep[prev[has_prev]] == ep[has_prev])
    # gripper transition label
    g = A[:, :, 6] >= 0
    trans = (g.min(1) != g.max(1))
    pg = np.zeros(N, bool); pg[has_prev] = (E[prev[has_prev], 4, 6] >= 0) != g[has_prev, 0]
    trans = trans | pg
    # (a) tail continuity: prev executed chunk's tail steps 5..9 vs current a_inf[:5]
    tail = E[prev, 5:10, :7]
    e_tail = err(tail, A, sig); gm_tail = gmis(tail, A)
    # (b) track: next of previous B0 pick / oracle pick
    def track_err(pick):
        nx = L_next[pick[prev]]; ok = has_prev & (nx >= 0)
        e = np.full(N, np.nan); e[ok] = err(L_act[nx[ok]], A[ok], sig); return e, ok
    e_trB0, okB0 = track_err(rec); e_trOr, okOr = track_err(orc)
    e_stayOr = np.full(N, np.nan); e_stayOr[has_prev] = err(L_act[orc[prev[has_prev]]], A[has_prev], sig)
    # (c) continuity retrieval: nearest library head to prev tail, within task; also + rs
    e_cont = np.full(N, np.nan); e_cont_rs = np.full(N, np.nan); e_rs = np.full(N, np.nan)
    e_cont_k = {k: np.full(N, np.nan) for k in (3, 5)}; gm_cont = np.full(N, np.nan)
    conf_cont = np.full(N, np.nan); conf_agree = np.full(N, np.nan); e_cont_med = np.full(N, np.nan)
    pick_cont = np.full(N, -1)
    for t in np.unique(task):
        cand = np.where(L_task == t)[0]; CA = L_act[cand][:, :5, :7] / sig  # [C,5,7]
        CR = L_rs[cand]; rs_sd = CR.std(0) + 1e-6
        qi = np.where((task == t) & has_prev)[0]
        T = tail[qi] / sig  # [n,5,7]
        D = np.sqrt(np.mean((T[:, None] - CA[None]) ** 2, axis=(2, 3)))  # [n,C]
        Drs = np.sqrt(np.mean(((np.asarray(rs[qi])[:, None] - CR[None]) / rs_sd) ** 2, axis=2))
        o = np.argsort(D, 1); b = o[:, 0]
        e_cont[qi] = err(L_act[cand[b]], A[qi], sig); gm_cont[qi] = gmis(L_act[cand[b]], A[qi]); pick_cont[qi] = cand[b]
        conf_cont[qi] = -D[np.arange(len(qi)), b]
        for k in (3, 5):
            kk = cand[o[:, :k]]; mean_a = L_act[kk][:, :, :5, :7].mean(1)
            e_cont_k[k][qi] = err(mean_a, A[qi], sig)
        kk = cand[o[:, :5]]; med_a = np.median(L_act[kk][:, :, :5, :7], axis=1); e_cont_med[qi] = err(med_a, A[qi], sig)
        # agreement confidence: spread of top-5 actions
        sp = np.sqrt(np.mean(((L_act[kk][:, :, :5, :7] - med_a[:, None]) / sig) ** 2, axis=(1, 2, 3)))
        conf_agree[qi] = -sp
        # combined: z-scored sum (equal weight) of continuity and rs distance
        z = (D - D.mean(1, keepdims=True)) / (D.std(1, keepdims=True) + 1e-6) + (Drs - Drs.mean(1, keepdims=True)) / (Drs.std(1, keepdims=True) + 1e-6)
        b2 = np.argmin(z, 1); e_cont_rs[qi] = err(L_act[cand[b2]], A[qi], sig)
        b3 = np.argmin(Drs, 1); e_rs[qi] = err(L_act[cand[b3]], A[qi], sig)
    hp = has_prev
    r = dict(cell=cell, n=int(hp.sum()), B0=eB0[hp].mean(), B0_tr=eB0[hp & trans].mean(), trans_share=trans[hp].mean(),
             tail=np.nanmean(e_tail[hp]), tail_p50=np.nanmedian(e_tail[hp]), tail_tr=np.nanmean(e_tail[hp & trans]),
             tail_gm=np.nanmean(gm_tail[hp]), tail_gm_tr=np.nanmean(gm_tail[hp & trans]), B0_gm_tr=d["grip_mis"][hp & trans].mean(),
             trackB0=np.nanmean(e_trB0[okB0]), B0_on_ok=eB0[okB0].mean(), trackOr=np.nanmean(e_trOr[okOr]), stayOr=np.nanmean(e_stayOr[hp]),
             orc=d["oracle_err"][hp].mean(),
             cont=np.nanmean(e_cont[hp]), cont_p50=np.nanmedian(e_cont[hp]), cont_tr=np.nanmean(e_cont[hp & trans]), cont_gm=np.nanmean(gm_cont[hp]),
             cont_gm_tr=np.nanmean(gm_cont[hp & trans]),
             cont_k3=np.nanmean(e_cont_k[3][hp]), cont_k5=np.nanmean(e_cont_k[5][hp]), cont_med5=np.nanmean(e_cont_med[hp]),
             cont_rs=np.nanmean(e_cont_rs[hp]), rs_only=np.nanmean(e_rs[hp]),
             aurc_B0=aurc(conf[hp], eB0[hp]), aurc_cont=aurc(conf_cont[hp], e_cont[hp]), aurc_agree=aurc(conf_agree[hp], e_cont[hp]),
             aurc_tail_as_conf_for_B0=aurc(-e_tail[hp], eB0[hp]),
             flip_vs_B0=(pick_cont[hp] != rec[hp]).mean(), flip_vs_orc=(pick_cont[hp] != orc[hp]).mean(),
             min_B0_cont=np.minimum(eB0[hp], e_cont[hp]).mean())
    rows.append(r)
    print(cell, " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items() if k != "cell"), flush=True)

keys = [k for k in rows[0] if k != "cell"]
print("\n| cell | " + " | ".join(keys) + " |")
for r in rows:
    print(f"| {r['cell']} | " + " | ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")
