"""D1b: cache-arm split by previous decision HIT/MISS; stacking median-synth on the continuity-reranked list;
conditional strategies. Read-only."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/weiland/projects/openpi")
import numpy as np
from exp.offline_search.harness import metrics, store

ROOT = "/dev/shm/offline_search_store"
RES = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
CELLS = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]


def aurc(err, conf):
    return metrics.risk_coverage(np.asarray(err, np.float64), np.asarray(conf, np.float64))["aurc"]


def rank_avg(*sigs):
    r = np.zeros(len(sigs[0]))
    for s in sigs:
        r += np.argsort(np.argsort(s, kind="stable"), kind="stable")
    return r / len(sigs)


OUT = {}
for cell in CELLS:
    m, s, arm = cell.split("_")
    key = f"{m}_{s}"
    z = np.load(f"{RES}/B0_current/{cell}.npz")
    Q = f"{ROOT}/queries/{cell}"; L = f"{ROOT}/library/{key}/current"
    row = z["row"]; step = z["step"].astype(int); topk = z["topk"]; err0 = z["err"]; N = len(row)
    a_inf = np.load(f"{Q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{Q}/a_exec.npy", mmap_mode="r")
    a_hit = np.load(f"{Q}/a_hit.npy", mmap_mode="r")
    A = np.load(f"{L}/action.npy"); sigma = store.action_sigma(ROOT, key)
    gt = metrics.seg(a_inf[row]); C = metrics.seg(A[topk]); E = metrics.err_seg(C, gt[:, None], sigma)
    has_prev = step > 0; pidx = np.where(has_prev, np.arange(N) - 1, 0)
    prow = row[pidx]
    prev_exec = np.asarray(a_exec[prow], np.float64); prev_inf = np.asarray(a_inf[prow], np.float64)
    prev_was_miss = np.abs(prev_exec[:, :5, :7] - prev_inf[:, :5, :7]).max((1, 2)) < 1e-6
    prev_was_miss &= has_prev
    prev_was_hit = has_prev & ~prev_was_miss
    tail = prev_exec[:, 5:10, :7]
    cont5 = metrics.err_seg(C, tail[:, None], sigma)
    # also: continuity using the *teacher* tail only when available (= after miss) — same as tail there
    ar = np.arange(N); b0 = E[:, 0]
    def pick(i): return E[ar, i]
    res = {"n": N, "hit_rate_prev": float(prev_was_hit[has_prev].mean()), "B0": b0.mean()}
    rb = np.arange(10)[None].repeat(N, 0); rc = np.argsort(np.argsort(cont5, 1), 1)
    picks = {}
    picks["cont5_top10"] = np.where(has_prev, np.argmin(cont5, 1), 0)
    picks["cont5_top5"] = np.where(has_prev, np.argmin(cont5[:, :5], 1), 0)
    picks["borda2_top10"] = np.where(has_prev, np.argmin(rb + 2.0 * rc, 1), 0)
    picks["borda1_top10"] = np.where(has_prev, np.argmin(rb + 1.0 * rc, 1), 0)
    picks["borda1_top5"] = np.where(has_prev, np.argmin(rb[:, :5] + 1.0 * np.argsort(np.argsort(cont5[:, :5], 1), 1), 1), 0)
    # conditional: continuity only after a miss (teacher tail), else B0
    picks["cont5_top10_after_miss_only"] = np.where(prev_was_miss, picks["cont5_top10"], 0)
    picks["borda1_top10_after_miss_only"] = np.where(prev_was_miss, picks["borda1_top10"], 0)
    for name, i in picks.items():
        e = pick(i)
        res[name] = {"all": e.mean(), "after_miss": e[prev_was_miss].mean(), "after_hit": e[prev_was_hit].mean() if prev_was_hit.any() else float("nan"),
                     "step0": e[~has_prev].mean()}
    res["B0_split"] = {"after_miss": b0[prev_was_miss].mean(), "after_hit": b0[prev_was_hit].mean() if prev_was_hit.any() else float("nan"), "step0": b0[~has_prev].mean()}
    res["orc10_split"] = {"after_miss": E.min(1)[prev_was_miss].mean(), "after_hit": E.min(1)[prev_was_hit].mean() if prev_was_hit.any() else float("nan")}
    # stacking: median-synth over the continuity-reranked top-k (order by cont5, take first k)
    def synth_from(order_idx, k):
        Ck = np.take_along_axis(C, order_idx[:, :k, None, None], 1)
        a = np.median(Ck, axis=1); g = np.sign(Ck[..., 6]).sum(1); a[..., 6] = np.where(g >= 0, 1.0, -1.0)
        return metrics.err_seg(a, gt, sigma), metrics.grip_mis_seg(a, gt)
    o_c = np.argsort(cont5, 1, kind="stable"); o_c[~has_prev] = np.arange(10)
    o_b = np.argsort(rb + 1.0 * rc, 1, kind="stable"); o_b[~has_prev] = np.arange(10)
    for k in (3, 5):
        e, g = synth_from(o_c, k); res[f"synth_med{k}_of_cont5order"] = {"all": e.mean(), "after_miss": e[prev_was_miss].mean(), "after_hit": e[prev_was_hit].mean() if prev_was_hit.any() else float("nan"), "grip": g.mean()}
        e, g = synth_from(o_b, k); res[f"synth_med{k}_of_borda1order"] = {"all": e.mean(), "after_miss": e[prev_was_miss].mean(), "after_hit": e[prev_was_hit].mean() if prev_was_hit.any() else float("nan"), "grip": g.mean()}
        e, g = synth_from(np.arange(10)[None].repeat(N, 0), k); res[f"synth_med{k}_of_B0order"] = {"all": e.mean(), "after_miss": e[prev_was_miss].mean(), "after_hit": e[prev_was_hit].mean() if prev_was_hit.any() else float("nan"), "grip": g.mean()}
    # weighted median: cont-ordered top-3 median only when the continuity residual is small, else B0 top-1?
    # confidence split: AURC of -cont5(top1) after miss vs after hit
    c5 = cont5[:, 0].copy(); c5[~has_prev] = np.nanmedian(c5[has_prev])
    P5 = metrics.err_seg(C[:, :5, None], C[:, None, :5], sigma); agree5 = -P5[:, 0, 1:].mean(1)
    res["aurc_split"] = {}
    for nm, msk in (("after_miss", prev_was_miss), ("after_hit", prev_was_hit)):
        if msk.sum() < 100: continue
        res["aurc_split"][nm] = {"B0": aurc(b0[msk], z["confidence"][msk]), "-cont5": aurc(b0[msk], -c5[msk]), "agree5": aurc(b0[msk], agree5[msk]),
                                 "ra(B0,agree5,-cont5)": aurc(b0[msk], rank_avg(z["confidence"][msk], agree5[msk], -c5[msk])),
                                 "ra(agree5,-cont5,-dist_rs)": aurc(b0[msk], rank_avg(agree5[msk], -c5[msk], -z["x_dist_rs"][msk])),
                                 "opt": aurc(b0[msk], -b0[msk])}
    # a hit-aware confidence: after a hit use B0-ish (ra(B0, agree5, -dist_rs)), after a miss use ra(agree5,-cont5); rank within whole cell
    conf_mix = np.where(prev_was_miss, rank_avg(agree5, -c5), rank_avg(z["confidence"], agree5, -z["x_dist_rs"]))
    res["aurc_mix_conf_on_B0pick"] = aurc(b0, conf_mix)
    res["aurc_B0"] = aurc(b0, z["confidence"]); res["aurc_opt"] = aurc(b0, -b0)
    OUT[cell] = res
    print(cell, json.dumps(res, default=lambda x: round(float(x), 4)), flush=True)
json.dump(OUT, open("/home/weiland/.claude/jobs/a607dd74/tmp/ideation_B/d1b_out.json", "w"), indent=1, default=float)
