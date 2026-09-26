"""D5: fusion weight sweep (continuity vs rs), 2-back chunk continuity for GR00T (H=16: steps 10..14 of the chunk two
decisions ago cover this window), and risk@coverage of the candidate method. Read-only, CPU."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/weiland/projects/openpi")
import numpy as np
from exp.offline_search.harness import metrics, store

ROOT = "/dev/shm/offline_search_store"
RES = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
CELLS = sys.argv[1].split(",") if len(sys.argv) > 1 else [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}


def errmat(G, Aflat):
    g2 = (G * G).sum(1)[:, None]; a2 = (Aflat * Aflat).sum(1)[None]
    return np.sqrt(np.maximum((g2 - 2.0 * (G @ Aflat.T) + a2) / G.shape[1], 0.0))


def rc(err, conf):
    return metrics.risk_coverage(np.asarray(err, np.float64), np.asarray(conf, np.float64))


OUT = {}
for cell in CELLS:
    m, s, arm = cell.split("_"); key = f"{m}_{s}"
    z = np.load(f"{RES}/B0_current/{cell}.npz"); row = z["row"]; step = z["step"].astype(int); N = len(row); qtask = z["task_id"]
    Q = f"{ROOT}/queries/{cell}"
    a_inf = np.load(f"{Q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{Q}/a_exec.npy", mmap_mode="r")
    qrs = np.asarray(np.load(f"{Q}/rs.npy", mmap_mode="r")[:, :8][row], np.float64)
    sigma = store.action_sigma(ROOT, key); H = a_inf.shape[1]
    gt = metrics.seg(a_inf[row]); G = (gt / sigma).reshape(N, -1)
    has_prev = step > 0; p1 = np.where(has_prev, np.arange(N) - 1, 0)
    has_prev2 = step > 1; p2 = np.where(has_prev2, np.arange(N) - 2, 0)
    T1 = (np.asarray(a_exec[row[p1]], np.float64)[:, 5:10, :7] / sigma).reshape(N, -1)
    T2 = (np.asarray(a_exec[row[p2]], np.float64)[:, 10:15, :7] / sigma).reshape(N, -1) if H >= 15 else None
    res = {"n": N, "B0": float(z["err"].mean()), "B0_rc": {k: v for k, v in rc(z["err"], z["confidence"]).items() if k in ("aurc", "risk_c30", "risk_c50", "risk_c70")}}
    for libname in ("current", BIG[m]):
        L = f"{ROOT}/library/{key}/{libname}"
        A = metrics.seg(np.load(f"{L}/action.npy")); AF = (A / sigma).reshape(len(A), -1)
        lrs = np.asarray(np.load(f"{L}/rs.npy")[:, :8], np.float64); ltask = np.load(f"{L}/task_id.npy").astype(int)
        alphas = (0.0, 0.25, 0.5, 1.0); betas = (0.0, 0.5, 1.0) if T2 is not None else (0.0,)
        E1 = {(a, b): np.zeros(N) for a in alphas for b in betas}; E3 = {(a, b): np.zeros(N) for a in alphas for b in betas}
        conf = {"cont": np.zeros(N), "rs": np.zeros(N), "agree3": np.zeros(N), "margin": np.zeros(N)}
        for t in np.unique(qtask):
            qi_all = np.where(qtask == t)[0]; li = np.where(ltask == t)[0]; At = AF[li]; RSt = lrs[li]; Aseg = A[li]
            for c0 in range(0, len(qi_all), 512):
                qi = qi_all[c0:c0 + 512]; ar = np.arange(len(qi)); hp = has_prev[qi]; hp2 = has_prev2[qi]
                Eall = errmat(G[qi], At)
                d = np.sqrt(np.maximum((qrs[qi] ** 2).sum(1)[:, None] - 2 * qrs[qi] @ RSt.T + (RSt ** 2).sum(1)[None], 0))
                c1 = errmat(T1[qi], At); c2 = errmat(T2[qi], At) if T2 is not None else None
                sd = np.median(d.min(1)) + 1e-9; sc = (np.median(c1[hp].min(1)) + 1e-9) if hp.any() else 1.0
                sc2 = (np.median(c2[hp2].min(1)) + 1e-9) if (c2 is not None and hp2.any()) else 1.0
                for a in alphas:
                    for b in betas:
                        f = a * d / sd + np.where(hp[:, None], c1 / sc, 0.0)
                        if b > 0: f = f + b * np.where(hp2[:, None], c2 / sc2, 0.0)
                        if a == 0: f = f + np.where(hp[:, None], 0.0, d / sd)   # step 0 fallback: rs
                        o = np.argsort(f, 1)[:, :3]
                        E1[(a, b)][qi] = Eall[ar, o[:, 0]]
                        top3 = Aseg[o]; am = np.median(top3, 1); g = np.sign(top3[..., 6]).sum(1); am[..., 6] = np.where(g >= 0, 1.0, -1.0)
                        E3[(a, b)][qi] = metrics.err_seg(am, gt[qi], sigma)
                        if a == 0.25 and b == 0:
                            conf["cont"][qi] = -c1[ar, o[:, 0]]; conf["rs"][qi] = -d[ar, o[:, 0]]
                            P = metrics.err_seg(top3[:, :, None], top3[:, None, :], sigma); conf["agree3"][qi] = -P.mean((1, 2))
                            fs = np.sort(f, 1); conf["margin"][qi] = fs[:, 1] - fs[:, 0]
        conf["cont"][~has_prev] = np.median(conf["cont"][has_prev])
        ra = (np.argsort(np.argsort(conf["cont"])) + np.argsort(np.argsort(conf["rs"])) + np.argsort(np.argsort(conf["agree3"]))) / 3.0
        e = E3[(0.25, 0.0)]; e1 = E1[(0.25, 0.0)]
        res[libname] = {"top1": {f"a{a}_b{b}": v.mean() for (a, b), v in E1.items()}, "med3": {f"a{a}_b{b}": v.mean() for (a, b), v in E3.items()},
                        "rc_med3_a0.25": {nm: {k: v for k, v in rc(e, cf).items() if k in ("aurc", "risk_c30", "risk_c50", "risk_c70")} for nm, cf in (("cont", conf["cont"]), ("ra3", ra), ("margin", conf["margin"]), ("opt", -e))},
                        "rc_top1_a0.25": {nm: {k: v for k, v in rc(e1, cf).items() if k in ("aurc", "risk_c30", "risk_c50", "risk_c70")} for nm, cf in (("cont", conf["cont"]), ("ra3", ra), ("opt", -e1))},
                        "med3_a0.25_by_bin": {int(t): e[z["bin"] == t].mean() for t in (0, 1, 2)}, "B0_by_bin": {int(t): z["err"][z["bin"] == t].mean() for t in (0, 1, 2)}}
    OUT[cell] = res
    print(cell, json.dumps(res, default=lambda x: round(float(x), 4)), flush=True)
tag = "_".join(CELLS) if len(CELLS) <= 2 else "all"
json.dump(OUT, open(f"/home/weiland/.claude/jobs/a607dd74/tmp/ideation_B/d5_out_{tag}.json", "w"), indent=1, default=float)
