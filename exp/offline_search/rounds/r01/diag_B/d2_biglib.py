"""D2: key-free search (robot_state + action continuity) over current vs the 10x libraries. Read-only, CPU.
Error matrices via GEMM: err = sqrt((|a|^2 - 2 a.g + |g|^2)/35) with a, g in sigma units."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/weiland/projects/openpi")
import numpy as np
from exp.offline_search.harness import metrics, store

ROOT = "/dev/shm/offline_search_store"
RES = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
CELLS = sys.argv[1].split(",") if len(sys.argv) > 1 else [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}


def aurc(err, conf):
    return metrics.risk_coverage(np.asarray(err, np.float64), np.asarray(conf, np.float64))["aurc"]


def errmat(G, Aflat):
    """G (nq,35), Aflat (nl,35) both already divided by sigma -> (nq,nl) RMS err."""
    g2 = (G * G).sum(1)[:, None]; a2 = (Aflat * Aflat).sum(1)[None]
    e2 = (g2 - 2.0 * (G @ Aflat.T) + a2) / G.shape[1]
    return np.sqrt(np.maximum(e2, 0.0))


OUT = {}
for cell in CELLS:
    m, s, arm = cell.split("_"); key = f"{m}_{s}"
    z = np.load(f"{RES}/B0_current/{cell}.npz"); row = z["row"]; step = z["step"].astype(int); N = len(row)
    Q = f"{ROOT}/queries/{cell}"
    a_inf = np.load(f"{Q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{Q}/a_exec.npy", mmap_mode="r")
    qrs = np.asarray(np.load(f"{Q}/rs.npy", mmap_mode="r")[:, :8][row], np.float64)
    qtask = z["task_id"]
    sigma = store.action_sigma(ROOT, key)
    gt = metrics.seg(a_inf[row]); G = (gt / sigma).reshape(N, -1)
    has_prev = step > 0; pidx = np.where(has_prev, np.arange(N) - 1, 0)
    prev_exec = np.asarray(a_exec[row[pidx]], np.float64)
    tail = prev_exec[:, 5:10, :7]; T = (tail / sigma).reshape(N, -1)
    res = {"n": N, "B0": float(z["err"].mean()), "B0_aurc": aurc(z["err"], z["confidence"])}
    for libname in ("current", BIG[m]):
        L = f"{ROOT}/library/{key}/{libname}"
        A = metrics.seg(np.load(f"{L}/action.npy")); AF = (A / sigma).reshape(len(A), -1)
        lrs = np.asarray(np.load(f"{L}/rs.npy")[:, :8], np.float64)
        ltask = np.load(f"{L}/task_id.npy").astype(int); lsucc = np.load(f"{L}/success.npy")
        e_rs = np.zeros(N); e_cont = np.zeros(N); e_fuse_s = np.zeros(N); e_orc = np.zeros(N); e_med3 = np.zeros(N); e_med5 = np.zeros(N)
        e_orc10 = np.zeros(N); conf_c = np.zeros(N); conf_r = np.zeros(N); e_rs_succ = np.zeros(N); e_fuse_z = np.zeros(N)
        e_cont_med3 = np.zeros(N); agree3 = np.zeros(N)
        for t in np.unique(qtask):
            qi_all = np.where(qtask == t)[0]; li = np.where(ltask == t)[0]
            At = AF[li]; RSt = lrs[li]; St = lsucc[li]; Aseg = A[li]
            for c0 in range(0, len(qi_all), 512):
                qi = qi_all[c0:c0 + 512]; nq = len(qi); ar = np.arange(nq)
                Eall = errmat(G[qi], At)
                e_orc[qi] = Eall.min(1)
                d = np.sqrt(np.maximum((qrs[qi] ** 2).sum(1)[:, None] - 2 * qrs[qi] @ RSt.T + (RSt ** 2).sum(1)[None], 0))
                i_rs = np.argmin(d, 1); e_rs[qi] = Eall[ar, i_rs]
                dm = np.where(St[None], d, np.inf); e_rs_succ[qi] = Eall[ar, np.argmin(dm, 1)]
                c = errmat(T[qi], At); hp = has_prev[qi]
                i_c = np.where(hp, np.argmin(c, 1), i_rs); e_cont[qi] = Eall[ar, i_c]
                # fixed-scale fusion: d / typical-NN-distance + c / typical-NN-continuity (scales from this task's queries; T1-fittable from the library)
                sd = np.median(d.min(1)) + 1e-9; sc = (np.median(c[hp].min(1)) + 1e-9) if hp.any() else 1.0
                f2 = d / sd + np.where(hp[:, None], c / sc, 0.0)
                i_f2 = np.argmin(f2, 1); e_fuse_s[qi] = Eall[ar, i_f2]
                zd = (d - d.mean(1, keepdims=True)) / (d.std(1, keepdims=True) + 1e-9); zc = (c - c.mean(1, keepdims=True)) / (c.std(1, keepdims=True) + 1e-9)
                e_fuse_z[qi] = Eall[ar, np.argmin(zd + np.where(hp[:, None], zc, 0.0), 1)]
                o = np.argsort(f2, 1)[:, :10]
                e_orc10[qi] = np.take_along_axis(Eall, o, 1).min(1)
                for k, dst in ((3, e_med3), (5, e_med5)):
                    topk = Aseg[o[:, :k]]; a = np.median(topk, 1); g = np.sign(topk[..., 6]).sum(1); a[..., 6] = np.where(g >= 0, 1.0, -1.0)
                    dst[qi] = metrics.err_seg(a, gt[qi], sigma)
                oc = np.argsort(np.where(hp[:, None], c, f2), 1)[:, :3]
                topk = Aseg[oc]; a = np.median(topk, 1); g = np.sign(topk[..., 6]).sum(1); a[..., 6] = np.where(g >= 0, 1.0, -1.0)
                e_cont_med3[qi] = metrics.err_seg(a, gt[qi], sigma)
                conf_c[qi] = -c[ar, i_f2]; conf_r[qi] = -d[ar, i_f2]
                # agreement among the fused top-3 actions
                P = metrics.err_seg(Aseg[o[:, :3]][:, :, None], Aseg[o[:, :3]][:, None, :], sigma); agree3[qi] = -P.mean((1, 2))
        conf_c[~has_prev] = np.median(conf_c[has_prev])
        ra = (np.argsort(np.argsort(conf_c)) + np.argsort(np.argsort(conf_r)) + np.argsort(np.argsort(agree3))) / 3.0
        out = {"L": int(len(ltask)), "oracle": e_orc.mean(), "rs_top1": e_rs.mean(), "rs_top1_succ_only": e_rs_succ.mean(), "cont_top1": e_cont.mean(),
               "fuse_z_top1": e_fuse_z.mean(), "fuse_fixed_top1": e_fuse_s.mean(), "fuse_fixed_orc_in_top10": e_orc10.mean(),
               "fuse_fixed_med3": e_med3.mean(), "fuse_fixed_med5": e_med5.mean(), "cont_order_med3": e_cont_med3.mean(),
               "aurc_fuse_fixed_by_-cont": aurc(e_fuse_s, conf_c), "aurc_fuse_fixed_by_-rs": aurc(e_fuse_s, conf_r), "aurc_fuse_fixed_by_agree3": aurc(e_fuse_s, agree3),
               "aurc_fuse_fixed_by_rankavg3": aurc(e_fuse_s, ra), "aurc_opt_fuse_fixed": aurc(e_fuse_s, -e_fuse_s),
               "aurc_med3_by_rankavg3": aurc(e_med3, ra), "aurc_opt_med3": aurc(e_med3, -e_med3)}
        res[libname] = out
    OUT[cell] = res
    print(cell, json.dumps(res, default=lambda x: round(float(x), 4)), flush=True)
tag = "_".join(CELLS) if len(CELLS) <= 2 else "all"
json.dump(OUT, open(f"/home/weiland/.claude/jobs/a607dd74/tmp/ideation_B/d2_out_{tag}.json", "w"), indent=1, default=float)
