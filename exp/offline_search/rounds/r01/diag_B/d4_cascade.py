"""D4: coarse-to-fine over the 10x library: prune task candidates by rs+continuity to top-M, then vision (B0 fused
score, global params) on those M only. Query subsample. Read-only, CPU."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/weiland/projects/openpi")
import numpy as np
from exp.offline_search.harness import metrics, store

ROOT = "/dev/shm/offline_search_store"
RES = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
CELLS = sys.argv[1].split(",") if len(sys.argv) > 1 else [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
EPS = np.float32(1e-8)
MS = (16, 32, 64)


def zt(x, mu, sg):
    return np.float32(0.5) * (np.tanh((x - np.float32(mu)) / np.float32(sg)) + np.float32(1.0))


def errmat(G, Aflat):
    g2 = (G * G).sum(1)[:, None]; a2 = (Aflat * Aflat).sum(1)[None]
    return np.sqrt(np.maximum((g2 - 2.0 * (G @ Aflat.T) + a2) / G.shape[1], 0.0))


def aurc(err, conf):
    return metrics.risk_coverage(np.asarray(err, np.float64), np.asarray(conf, np.float64))["aurc"]


OUT = {}
for cell in CELLS:
    m, s, arm = cell.split("_"); key = f"{m}_{s}"
    p = store.current_params(m, s); w = np.array(p["weights"], np.float32); mu = p["mu"]; sg = p["sigma"]
    z = np.load(f"{RES}/B0_current/{cell}.npz"); N0 = len(z["row"])
    rng = np.random.default_rng(0); sel = np.sort(rng.choice(N0, size=min(NQ, N0), replace=False))
    row = z["row"][sel]; step = z["step"][sel].astype(int); qtask = z["task_id"][sel]; N = len(sel)
    Q = f"{ROOT}/queries/{cell}"
    K0 = np.load(f"{Q}/key_v0.npy", mmap_mode="r"); K1 = np.load(f"{Q}/key_v1.npy", mmap_mode="r")
    qrs = np.asarray(np.load(f"{Q}/rs.npy", mmap_mode="r")[row], np.float32)
    a_inf = np.load(f"{Q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{Q}/a_exec.npy", mmap_mode="r")
    sigma = store.action_sigma(ROOT, key)
    gt = metrics.seg(a_inf[row]); G = (gt / sigma).reshape(N, -1)
    has_prev = step > 0; prow = np.where(has_prev, row - 1, row)
    tail = np.asarray(a_exec[prow], np.float64)[:, 5:10, :7]; T = (tail / sigma).reshape(N, -1)
    res = {"n": N, "B0_current_on_subsample": float(z["err"][sel].mean())}
    for libname in ("current", BIG[m]):
        L = f"{ROOT}/library/{key}/{libname}"
        A = metrics.seg(np.load(f"{L}/action.npy")); AF = (A / sigma).reshape(len(A), -1)
        lrs = np.load(f"{L}/rs.npy"); ltask = np.load(f"{L}/task_id.npy").astype(int)
        LV0 = np.load(f"{L}/key_v0.npy", mmap_mode="r"); LV1 = np.load(f"{L}/key_v1.npy", mmap_mode="r")
        out = {M: {"orc_in_M": np.zeros(N), "b0_on_M": np.zeros(N), "b0cont_on_M": np.zeros(N), "med3_b0cont_on_M": np.zeros(N), "conf_b0cont": np.zeros(N), "conf_cont": np.zeros(N), "agree3": np.zeros(N)} for M in MS}
        e_pre = np.zeros(N); e_orc = np.zeros(N)
        for t in np.unique(qtask):
            qi = np.where(qtask == t)[0]; li = np.where(ltask == t)[0]
            Eall = errmat(G[qi], AF[li]); e_orc[qi] = Eall.min(1)
            d = np.sqrt(np.maximum((qrs[qi][:, :8].astype(np.float64) ** 2).sum(1)[:, None] - 2 * qrs[qi][:, :8].astype(np.float64) @ lrs[li][:, :8].T.astype(np.float64) + (lrs[li][:, :8].astype(np.float64) ** 2).sum(1)[None], 0))
            c = errmat(T[qi], AF[li]); hp = has_prev[qi]
            sd = np.median(d.min(1)) + 1e-9; sc = (np.median(c[hp].min(1)) + 1e-9) if hp.any() else 1.0
            f2 = d / sd + np.where(hp[:, None], c / sc, 0.0)
            e_pre[qi] = Eall[np.arange(len(qi)), np.argmin(f2, 1)]
            order = np.argsort(f2, 1)
            q0 = np.asarray(K0[row[qi]], np.float32); q1 = np.asarray(K1[row[qi]], np.float32)
            qn0 = np.linalg.norm(q0, axis=1); qn1 = np.linalg.norm(q1, axis=1)
            Mmax = min(max(MS), len(li)); cand = order[:, :Mmax]                       # (nq, Mmax) local idx
            # gather keys per query (Mmax rows each) — done per query to bound memory
            c0 = np.zeros((len(qi), Mmax), np.float32); c1 = np.zeros((len(qi), Mmax), np.float32)
            for j in range(len(qi)):
                rows_g = li[cand[j]]; o = np.argsort(rows_g); rg = rows_g[o]
                V0 = np.asarray(LV0[rg], np.float32); V1 = np.asarray(LV1[rg], np.float32)
                cc0 = (V0 @ q0[j]) / np.maximum(np.linalg.norm(V0, axis=1) * qn0[j], EPS)
                cc1 = (V1 @ q1[j]) / np.maximum(np.linalg.norm(V1, axis=1) * qn1[j], EPS)
                c0[j, o] = cc0; c1[j, o] = cc1
            dM = np.take_along_axis(d, cand, 1); cM = np.take_along_axis(c, cand, 1); EM = np.take_along_axis(Eall, cand, 1)
            fB0 = w[0] * zt(c0, mu[0], sg[0]) + w[1] * zt(c1, mu[1], sg[1]) + w[2] * zt(-dM, mu[2], sg[2])
            cz = -cM / sc
            fBC = fB0 + 0.5 * np.where(hp[:, None], cz, 0.0)
            for M in MS:
                Mm = min(M, Mmax); ar = np.arange(len(qi))
                out[M]["orc_in_M"][qi] = EM[:, :Mm].min(1)
                i = np.argmax(fB0[:, :Mm], 1); out[M]["b0_on_M"][qi] = EM[ar, i]
                i = np.argmax(fBC[:, :Mm], 1); out[M]["b0cont_on_M"][qi] = EM[ar, i]
                out[M]["conf_b0cont"][qi] = fBC[ar, i]; out[M]["conf_cont"][qi] = -cM[ar, i]
                o3 = np.argsort(-fBC[:, :Mm], 1)[:, :3]; rows3 = li[np.take_along_axis(cand, o3, 1)]
                top3 = A[rows3]; a = np.median(top3, 1); g = np.sign(top3[..., 6]).sum(1); a[..., 6] = np.where(g >= 0, 1.0, -1.0)
                out[M]["med3_b0cont_on_M"][qi] = metrics.err_seg(a, gt[qi], sigma)
                P = metrics.err_seg(top3[:, :, None], top3[:, None, :], sigma); out[M]["agree3"][qi] = -P.mean((1, 2))
        r = {"L": int(len(ltask)), "oracle": e_orc.mean(), "rs+cont_top1": e_pre.mean()}
        for M in MS:
            o = out[M]; cc = o["conf_cont"].copy(); cc[~has_prev] = np.median(cc[has_prev])
            ra = (np.argsort(np.argsort(cc)) + np.argsort(np.argsort(o["agree3"])) + np.argsort(np.argsort(o["conf_b0cont"]))) / 3.0
            r[f"M{M}"] = {"orc_in_M": o["orc_in_M"].mean(), "b0_on_M": o["b0_on_M"].mean(), "b0cont_on_M": o["b0cont_on_M"].mean(), "med3_b0cont_on_M": o["med3_b0cont_on_M"].mean(),
                          "aurc_b0cont_by_fused": aurc(o["b0cont_on_M"], o["conf_b0cont"]), "aurc_b0cont_by_-cont": aurc(o["b0cont_on_M"], cc),
                          "aurc_b0cont_by_ra3": aurc(o["b0cont_on_M"], ra), "aurc_opt": aurc(o["b0cont_on_M"], -o["b0cont_on_M"]),
                          "aurc_med3_by_ra3": aurc(o["med3_b0cont_on_M"], ra)}
        res[libname] = r
    OUT[cell] = res
    print(cell, json.dumps(res, default=lambda x: round(float(x), 4)), flush=True)
tag = "_".join(CELLS) if len(CELLS) <= 2 else "all"
json.dump(OUT, open(f"/home/weiland/.claude/jobs/a607dd74/tmp/ideation_B/d4_out_{tag}.json", "w"), indent=1, default=float)
