"""D1: what a cheap re-ranker / confidence over B0's saved top-10 could buy. Read-only; CPU only."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/weiland/projects/openpi")
import numpy as np
from exp.offline_search.harness import metrics, store

ROOT = "/dev/shm/offline_search_store"
RES = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
CELLS = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]
OUT = {}


def aurc(err, conf):
    return metrics.risk_coverage(np.asarray(err, np.float64), np.asarray(conf, np.float64))


def rank_avg(*sigs):
    """average of per-signal ranks (higher = more confident)."""
    r = np.zeros(len(sigs[0]))
    for s in sigs:
        r += np.argsort(np.argsort(s, kind="stable"), kind="stable")
    return r / len(sigs)


for cell in CELLS:
    m, s, arm = cell.split("_")
    key = f"{m}_{s}"
    z = np.load(f"{RES}/B0_current/{cell}.npz")
    Q = f"{ROOT}/queries/{cell}"
    L = f"{ROOT}/library/{key}/current"
    row = z["row"]; step = z["step"].astype(int); topk = z["topk"]; err0 = z["err"]; oerr = z["oracle_err"]
    N = len(row)
    a_inf = np.load(f"{Q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{Q}/a_exec.npy", mmap_mode="r")
    qrs = np.load(f"{Q}/rs.npy", mmap_mode="r")[:, :8]
    A = np.load(f"{L}/action.npy"); nxt = np.load(f"{L}/next.npy"); lrs = np.load(f"{L}/rs.npy")[:, :8]
    sigma = store.action_sigma(ROOT, key)
    K = topk.shape[1]
    assert (topk >= 0).all()
    gt = metrics.seg(a_inf[row])                       # (N,5,7)
    C = metrics.seg(A[topk])                           # (N,K,5,7)
    E = metrics.err_seg(C, gt[:, None], sigma)         # (N,K)
    assert np.allclose(E[:, 0], err0, atol=1e-6), cell
    # previous decision (same episode) = previous npz row when step>0 (npz is in episode/time order)
    has_prev = step > 0
    pidx = np.where(has_prev, np.arange(N) - 1, 0)
    assert (z["ep"][pidx][has_prev] == z["ep"][has_prev]).all()
    prev_exec = np.asarray(a_exec[row[pidx]], np.float64)          # (N,H,32)
    tail = prev_exec[:, 5:10, :7]                                  # forecast of the current executed window
    last = prev_exec[:, 4, :7]                                     # last executed action
    cont5 = metrics.err_seg(C, tail[:, None], sigma)               # (N,K) continuity residual, 5 steps
    cont1 = np.sqrt(np.mean(((C[:, :, 0, :] - last[:, None]) / sigma) ** 2, axis=-1))  # (N,K)
    # gripper
    g_prev = np.sign(last[:, 6]); g_prev[g_prev == 0] = 1
    cand_flip = (np.sign(C[..., 6]) != g_prev[:, None, None]).any(-1)   # (N,K) candidate flips vs last executed gripper
    gt_flip = (np.sign(gt[..., 6]) != g_prev[:, None]).any(-1)           # teacher transition (relative to executed)
    gt_flip_in = (np.sign(gt[:, 1:, 6]) != np.sign(gt[:, :-1, 6])).any(-1)
    trans = (gt_flip | gt_flip_in) & has_prev
    # rs distance among top-K
    drs = np.linalg.norm(lrs[topk] - np.asarray(qrs[row])[:, None], axis=-1)   # (N,K)
    # medoid among top-k: pairwise err among candidates
    def medoid_idx(k):
        Ck = C[:, :k]
        P = metrics.err_seg(Ck[:, :, None], Ck[:, None, :], sigma)   # (N,k,k)
        return np.argmin(P.sum(-1), axis=1), P
    med5, P5 = medoid_idx(5)
    med10, P10 = medoid_idx(10)
    med3, P3 = medoid_idx(3)
    # synthesized per-step median (dims 0..5) + gripper majority over top-k
    def synth_median(k):
        Ck = C[:, :k]
        a = np.median(Ck, axis=1)
        g = np.sign(Ck[..., 6]).sum(1)  # (N,5)
        a[..., 6] = np.where(g >= 0, 1.0, -1.0)
        return metrics.err_seg(a, gt, sigma), metrics.grip_mis_seg(a, gt)
    # --- picks
    def pick_err(idx):
        return E[np.arange(N), idx]
    ar = np.arange(N)
    b0 = E[:, 0]
    res = {"n": N, "B0": b0.mean(), "oracle_task": oerr.mean(), "orc_in_top10": E.min(1).mean(),
           "orc_in_top5": E[:, :5].min(1).mean(), "orc_in_top3": E[:, :3].min(1).mean(),
           "orc_in_top10_is_orc": float((E.min(1) <= oerr + 1e-9).mean())}
    for k in (3, 5, 10):
        i = np.argmin(cont5[:, :k], 1); i = np.where(has_prev, i, 0); res[f"cont5_top{k}"] = pick_err(i).mean()
        i = np.argmin(cont1[:, :k], 1); i = np.where(has_prev, i, 0); res[f"cont1_top{k}"] = pick_err(i).mean()
        i = np.argmin(drs[:, :k], 1); res[f"rs_top{k}"] = pick_err(i).mean()
    res["medoid3"] = pick_err(med3).mean(); res["medoid5"] = pick_err(med5).mean(); res["medoid10"] = pick_err(med10).mean()
    for k in (3, 5, 10):
        e, g = synth_median(k); res[f"synth_med{k}"] = e.mean(); res[f"synth_med{k}_grip"] = g.mean()
    # combined: Borda of B0 rank + cont5 rank within top-k
    for k in (5, 10):
        rb = np.arange(k)[None].repeat(N, 0)
        rc = np.argsort(np.argsort(cont5[:, :k], 1), 1)
        for lam in (0.5, 1.0, 2.0):
            i = np.argmin(rb + lam * rc, 1); i = np.where(has_prev, i, 0)
            res[f"borda_b0+{lam}cont5_top{k}"] = pick_err(i).mean()
        rr = np.argsort(np.argsort(drs[:, :k], 1), 1)
        i = np.argmin(rb + rr, 1); res[f"borda_b0+rs_top{k}"] = pick_err(i).mean()
        i = np.argmin(rb + rr + rc, 1); i = np.where(has_prev, i, 0); res[f"borda_b0+rs+cont5_top{k}"] = pick_err(i).mean()
        i = np.argmin(rr + rc, 1); i = np.where(has_prev, i, 0); res[f"borda_rs+cont5_top{k}"] = pick_err(i).mean()
    # gripper-consistent filter: drop flipping candidates unless all flip (keep B0 order)
    for k in (5, 10):
        allflip = cand_flip[:, :k].all(1)
        ok = ~cand_flip[:, :k] | allflip[:, None]
        i = np.argmax(ok, 1); i = np.where(has_prev, i, 0)
        res[f"gripfilter_top{k}"] = pick_err(i).mean()
        # split by transition
        res[f"gripfilter_top{k}_trans"] = pick_err(i)[trans].mean(); res[f"gripfilter_top{k}_steady"] = pick_err(i)[~trans & has_prev].mean()
    res["B0_trans"] = b0[trans].mean(); res["B0_steady"] = b0[~trans & has_prev].mean(); res["trans_rate"] = float(trans.mean())
    # transition predictability from online signals: fraction of top-10 flipping; AUROC vs trans
    def auroc(score, lab):
        o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
        n1 = lab.sum(); n0 = len(lab) - n1
        return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    ff = cand_flip[:, :10].mean(1)
    res["auroc_flipfrac_vs_trans"] = auroc(ff[has_prev], trans[has_prev])
    # gripper aperture (raw_state dims 6,7) as transition predictor — use |q_rs gripper| vs prev
    # --- track / stay
    prev_top1 = z["top1"][pidx]
    nx = nxt[prev_top1]
    can_track = has_prev & (nx >= 0)
    e_track = np.full(N, np.nan)
    e_track[can_track] = metrics.err_seg(metrics.seg(A[nx[can_track]]), gt[can_track], sigma)
    stay = has_prev & (z["top1"] == prev_top1)
    res["stay_rate"] = float(stay.mean()); res["B0_when_stay"] = b0[stay].mean()
    res["track_when_stay"] = np.nanmean(e_track[stay & can_track]); res["track_all"] = np.nanmean(e_track[can_track])
    res["B0_when_can_track"] = b0[can_track].mean()
    # track if next(prev) in top-k else B0
    for k in (3, 5, 10):
        intop = (topk[:, :k] == nx[:, None]).any(1) & can_track
        e = b0.copy(); e[intop] = e_track[intop]
        res[f"track_if_in_top{k}"] = e.mean(); res[f"track_if_in_top{k}_rate"] = float(intop.mean())
    # stay -> force track
    e = b0.copy(); e[stay & can_track] = e_track[stay & can_track]; res["B0_but_track_on_stay"] = e.mean()
    # --- confidence signals on B0's pick
    conf = {}
    conf["B0"] = z["confidence"]; conf["margin"] = z["x_margin"]; conf["cos_v0"] = z["x_cos_v0"]; conf["cos_v1"] = z["x_cos_v1"]
    conf["-dist_rs"] = -z["x_dist_rs"]
    agree5 = -P5[:, 0, 1:].mean(1)            # top-1 vs the other top-5: action agreement
    conf["agree5"] = agree5
    conf["agree10"] = -P10[:, 0, 1:].mean(1)
    conf["-spread5"] = -P5.mean((1, 2))       # pairwise dispersion of top-5
    c5 = cont5[:, 0].copy(); c5[~has_prev] = np.nanmedian(c5[has_prev]); conf["-cont5"] = -c5
    c1 = cont1[:, 0].copy(); c1[~has_prev] = np.nanmedian(c1[has_prev]); conf["-cont1"] = -c1
    conf["-flipfrac10"] = -ff
    conf["gripunanim5"] = (np.sign(C[:, :5, :, 6]) == np.sign(C[:, :1, :, 6])).all(-1).mean(1)
    conf["ra(margin,agree5)"] = rank_avg(conf["margin"], agree5)
    conf["ra(margin,agree5,-cont5)"] = rank_avg(conf["margin"], agree5, conf["-cont5"])
    conf["ra(agree5,-cont5)"] = rank_avg(agree5, conf["-cont5"])
    conf["ra(agree5,-cont5,-dist_rs)"] = rank_avg(agree5, conf["-cont5"], conf["-dist_rs"])
    conf["ra(agree5,-cont5,-flipfrac)"] = rank_avg(agree5, conf["-cont5"], conf["-flipfrac10"])
    conf["ra(B0,agree5,-cont5)"] = rank_avg(conf["B0"], agree5, conf["-cont5"])
    res["aurc"] = {k: aurc(b0, v)["aurc"] for k, v in conf.items()}
    res["aurc_opt"] = aurc(b0, -b0)["aurc"]
    res["risk30"] = {k: aurc(b0, v)["risk_c30"] for k, v in conf.items()}
    # confidence for the re-ranked pick (borda b0+cont5 top5) — recompute signals for that pick
    rb = np.arange(5)[None].repeat(N, 0); rc = np.argsort(np.argsort(cont5[:, :5], 1), 1)
    i = np.argmin(rb + 1.0 * rc, 1); i = np.where(has_prev, i, 0)
    e_rr = pick_err(i)
    ag = -np.take_along_axis(P5, i[:, None, None].repeat(5, 2), 1)[:, 0]   # (N,5) err of pick vs other top5
    ag = (ag.sum(1) + 0) / 4.0
    c5r = cont5[ar, i].copy(); c5r[~has_prev] = np.nanmedian(c5r[has_prev])
    res["rr_pick_err"] = e_rr.mean()
    res["rr_aurc"] = {"B0conf": aurc(e_rr, conf["B0"])["aurc"], "agree5": aurc(e_rr, ag)["aurc"],
                      "ra(agree5,-cont5)": aurc(e_rr, rank_avg(ag, -c5r))["aurc"],
                      "ra(margin,agree5,-cont5)": aurc(e_rr, rank_avg(conf["margin"], ag, -c5r))["aurc"],
                      "opt": aurc(e_rr, -e_rr)["aurc"]}
    # by third for the re-ranker
    b = z["bin"]
    res["rr_by_bin"] = {int(t): (b0[b == t].mean(), e_rr[b == t].mean()) for t in (0, 1, 2)}
    OUT[cell] = res
    print(cell, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, dict)}), flush=True)
    print("   AURC:", {k: round(v, 3) for k, v in res["aurc"].items()}, "opt", round(res["aurc_opt"], 3))
    print("   risk30:", {k: round(v, 3) for k, v in res["risk30"].items()})
    print("   rr_aurc:", {k: round(v, 3) for k, v in res["rr_aurc"].items()}, "rr_by_bin", res["rr_by_bin"])

json.dump(OUT, open("/home/weiland/.claude/jobs/a607dd74/tmp/ideation_B/d1_out.json", "w"), indent=1, default=float)
