"""D6: gripper-transition veto as a confidence tier; transition/steady split of the continuity picks. Read-only, CPU."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/weiland/projects/openpi")
import numpy as np
from exp.offline_search.harness import metrics, store

ROOT = "/dev/shm/offline_search_store"
RES = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
CELLS = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]


def errmat(G, Aflat):
    g2 = (G * G).sum(1)[:, None]; a2 = (Aflat * Aflat).sum(1)[None]
    return np.sqrt(np.maximum((g2 - 2.0 * (G @ Aflat.T) + a2) / G.shape[1], 0.0))


def rc(err, conf):
    r = metrics.risk_coverage(np.asarray(err, np.float64), np.asarray(conf, np.float64))
    return {k: round(float(v), 3) for k, v in r.items() if k in ("aurc", "risk_c30", "risk_c50", "risk_c70")}


def grip_at_cov(err_unused, grip, conf, c):
    o = np.argsort(-conf, kind="stable"); k = max(1, int(np.ceil(c * len(conf))))
    return float(grip[o[:k]].mean())


def rank(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)


OUT = {}
for cell in CELLS:
    m, s, arm = cell.split("_"); key = f"{m}_{s}"
    z = np.load(f"{RES}/B0_current/{cell}.npz"); row = z["row"]; step = z["step"].astype(int); N = len(row); qtask = z["task_id"]
    Q = f"{ROOT}/queries/{cell}"; L = f"{ROOT}/library/{key}/current"
    a_inf = np.load(f"{Q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{Q}/a_exec.npy", mmap_mode="r")
    qrs = np.asarray(np.load(f"{Q}/rs.npy", mmap_mode="r")[:, :8][row], np.float64)
    A = metrics.seg(np.load(f"{L}/action.npy")); sigma = store.action_sigma(ROOT, key); AF = (A / sigma).reshape(len(A), -1)
    lrs = np.asarray(np.load(f"{L}/rs.npy")[:, :8], np.float64); ltask = np.load(f"{L}/task_id.npy").astype(int)
    gt = metrics.seg(a_inf[row]); G = (gt / sigma).reshape(N, -1)
    has_prev = step > 0; p1 = np.where(has_prev, np.arange(N) - 1, 0)
    prev = np.asarray(a_exec[row[p1]], np.float64); T1 = (prev[:, 5:10, :7] / sigma).reshape(N, -1)
    g_prev = np.sign(prev[:, 4, 6]); g_prev[g_prev == 0] = 1
    trans = ((np.sign(gt[..., 6]) != g_prev[:, None]).any(-1) | (np.sign(gt[:, 1:, 6]) != np.sign(gt[:, :-1, 6])).any(-1)) & has_prev
    # --- B0 pick + its top-10 flip fraction
    topk = z["topk"]; C10 = A[topk]; flip10_b0 = (np.sign(C10[..., 6]) != g_prev[:, None, None]).any(-1).mean(1)
    P5 = metrics.err_seg(C10[:, :5, None], C10[:, None, :5], sigma); agree5 = -P5[:, 0, 1:].mean(1)
    cont_b0 = metrics.err_seg(C10[:, 0], prev[:, 5:10, :7], sigma); cont_b0[~has_prev] = np.median(cont_b0[has_prev])
    e_b0 = z["err"]; g_b0 = z["grip_mis"].astype(np.float64)
    # --- continuity method pick (current lib): score = cont + 0.25 d/sd; top-3 median; flipfrac over its top-10
    e_p1 = np.zeros(N); g_p1 = np.zeros(N); flip10_p1 = np.zeros(N); cont_p1 = np.zeros(N); rs_p1 = np.zeros(N); ag3 = np.zeros(N)
    for t in np.unique(qtask):
        qi = np.where(qtask == t)[0]; li = np.where(ltask == t)[0]; ar = np.arange(len(qi)); hp = has_prev[qi]
        d = np.sqrt(np.maximum((qrs[qi] ** 2).sum(1)[:, None] - 2 * qrs[qi] @ lrs[li].T + (lrs[li] ** 2).sum(1)[None], 0))
        c = errmat(T1[qi], AF[li]); sd = np.median(d.min(1)) + 1e-9
        f = np.where(hp[:, None], c, 0.0) + 0.25 * d / sd + np.where(hp[:, None], 0.0, d / sd)
        o = np.argsort(f, 1)[:, :10]; top3 = A[li[o[:, :3]]]
        a = np.median(top3, 1); gg = np.sign(top3[..., 6]).sum(1); a[..., 6] = np.where(gg >= 0, 1.0, -1.0)
        e_p1[qi] = metrics.err_seg(a, gt[qi], sigma); g_p1[qi] = metrics.grip_mis_seg(a, gt[qi])
        flip10_p1[qi] = (np.sign(A[li[o]][..., 6]) != g_prev[qi][:, None, None]).any(-1).mean(1)
        cont_p1[qi] = -c[ar, o[:, 0]]; rs_p1[qi] = -d[ar, o[:, 0]]
        P = metrics.err_seg(top3[:, :, None], top3[:, None, :], sigma); ag3[qi] = -P.mean((1, 2))
    cont_p1[~has_prev] = np.median(cont_p1[has_prev])
    res = {"n": N, "trans_rate": float(trans.mean()),
           "B0": {"err": e_b0.mean(), "err_trans": e_b0[trans].mean(), "err_steady": e_b0[~trans & has_prev].mean(), "grip": g_b0.mean(), "grip_trans": g_b0[trans].mean()},
           "P1cur_med3": {"err": e_p1.mean(), "err_trans": e_p1[trans].mean(), "err_steady": e_p1[~trans & has_prev].mean(), "grip": g_p1.mean(), "grip_trans": g_p1[trans].mean()}}
    # confidence tiers: base (rank-avg) minus veto
    base_b0 = (rank(z["confidence"]) + rank(agree5) + rank(-cont_b0)) / 3.0
    base_p1 = (rank(cont_p1) + rank(rs_p1) + rank(ag3)) / 3.0
    for nm, e, g, base, ff in (("B0", e_b0, g_b0, base_b0, flip10_b0), ("P1", e_p1, g_p1, base_p1, flip10_p1)):
        r = {"base": rc(e, base), "base_grip@50": grip_at_cov(e, g, base, 0.5), "base_grip@30": grip_at_cov(e, g, base, 0.3)}
        for tau in (0.05, 0.3, 0.5):
            veto = (ff >= tau).astype(np.float64)
            conf = base - 1e9 * veto
            r[f"veto{tau}"] = rc(e, conf); r[f"veto{tau}_grip@50"] = grip_at_cov(e, g, conf, 0.5); r[f"veto{tau}_grip@30"] = grip_at_cov(e, g, conf, 0.3); r[f"veto{tau}_frac"] = float(veto.mean())
        r["opt"] = rc(e, -e)
        res[f"conf_{nm}"] = r
    OUT[cell] = res
    print(cell, json.dumps(res, default=lambda x: round(float(x), 4)), flush=True)
json.dump(OUT, open("/home/weiland/.claude/jobs/a607dd74/tmp/ideation_B/d6_out.json", "w"), indent=1, default=float)
