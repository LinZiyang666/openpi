"""R2 ideation-B diagnostics part 2 (cache arms, current library; uses the caches of diag.py):
  T. trajectory tracking after a HIT with visual verification: the served row is known online (here: rec_top1 of the
     previous decision, whose a_exec == a_hit); candidate next(served) vs re-retrieval (pca32 st1 mean5).
  M. mode-seeking synthesis: when the top-5 heads are multimodal, mean of the largest cluster instead of the mean.
  S. stuck runs: does the selection change while the observation does not? err along the run.
Usage: taskset ... python diag2.py <cell>
"""
import json, pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import diag as D

ROOT = D.ROOT
OUT = D.OUT
cell = sys.argv[1]
model, suite, arm = cell.split("_")
key = f"{model}_{suite}"
cache_dir = OUT / "cache"
t0 = time.time()
pca_cur = D.fit_pca_current(key, 32, cache_dir)
pca = {}
for f in ("v0", "v1"):
    d = D.PCA_DIR / key / D.BIG[model] / f
    pca[f] = (np.load(d / "mean.npy").astype(np.float32), np.ascontiguousarray(np.load(d / "basis.npy", mmap_mode="r")[:, :32], np.float32))
L = D.load_lib(key, "current", pca, 32, cache_dir, pca_cur)
qd = ROOT / "queries" / cell
eps_json = json.load(open(qd / "episodes.json"))
ep = np.asarray(np.load(qd / "ep.npy"), np.int64)
step = np.asarray(np.load(qd / "step.npy"), np.int64)
N = ep.size
task = np.asarray([e["task_id"] for e in eps_json], np.int64)[ep]
ep_success = np.asarray([bool(e["success"]) for e in eps_json])[ep]
rs = np.asarray(np.load(qd / "rs.npy", mmap_mode="r")[:, :8], np.float32)
a_inf = np.asarray(np.load(qd / "a_inf.npy", mmap_mode="r")[:, :5, :7], np.float32)
a_exec = np.asarray(np.load(qd / "a_exec.npy", mmap_mode="r")[:, :10, :7], np.float32)
rec_top1 = np.asarray(np.load(qd / "rec_top1.npy"), np.int64)
QPc = {f: np.load(cache_dir / f"qprojcur_{cell}_{f}.npy")[:, :32] for f in ("v0", "v1")}
sigma = np.asarray(L.act[:, :5, :7], np.float64).std(axis=(0, 1))
sig = sigma.astype(np.float32)
prev_idx = np.full(N, -1, np.int64)
same = np.r_[False, (ep[1:] == ep[:-1]) & (step[1:] == step[:-1] + 1)]
prev_idx[same] = np.arange(N)[same] - 1
regime = np.where(step == 0, 0, 1 if arm == "inf" else 2)
stale = regime == 2


def err_of(ah):
    d = (ah.astype(np.float64) - a_inf) / sigma
    return np.sqrt((d * d).mean(axis=(1, 2)))


# sanity: the previous decision's served row -> its head equals a_exec[prev, :5]
pi = prev_idx >= 0
served = np.full(N, -1, np.int64)
served[pi] = rec_top1[prev_idx[pi]]
chk = np.abs(L.head[served[pi]] - a_exec[prev_idx[pi], :5]).max()
print(f"[{cell}] served-row check max|head - a_exec_prev| = {chk:.2e}", flush=True)

# scores over all candidates (pca32 cur basis, st1) with per-decision top-8 and per-candidate score access
TOP = np.zeros((N, 8), np.int64); SC = np.zeros((N, 8)); Sall = {}
rank_next = np.full(N, -1.0)      # rank of next(served) within the task candidates (0 = best)
score_next = np.full(N, np.nan); score_top = np.full(N, np.nan); score_5th = np.full(N, np.nan)
has_next = np.zeros(N, bool)
for tq in np.unique(task):
    qi = np.nonzero(task == tq)[0]
    ci = np.nonzero(L.task == tq)[0]
    S = np.zeros((qi.size, ci.size))
    for f in ("v0", "v1"):
        Pc = L.Pc[f][ci]; m = Pc.mean(0)
        Zc = D.unit(Pc - m); Zq = D.unit(QPc[f][qi] - m)
        S += D.zrows(Zq @ Zc.T)
    q2 = np.einsum("ij,ij->i", rs[qi], rs[qi])[:, None]; c2 = np.einsum("ij,ij->i", L.rs[ci], L.rs[ci])[None, :]
    Dq = np.sqrt(np.maximum(q2 - 2.0 * (rs[qi] @ L.rs[ci].T) + c2, 0.0))
    S += D.zrows(-Dq)
    o = D.topk_rows(S, 8) if hasattr(D, "topk_rows") else None
    if o is None:
        k = 8
        o = np.argpartition(-S, k - 1, axis=1)[:, :k]
        so = np.take_along_axis(S, o, 1); oo = np.argsort(-so, axis=1, kind="stable"); o = np.take_along_axis(o, oo, 1)
    TOP[qi] = ci[o]; SC[qi] = np.take_along_axis(S, o, 1)
    score_top[qi] = SC[qi, 0]; score_5th[qi] = SC[qi, 4]
    # next(served)
    pos = {int(r): j for j, r in enumerate(ci)}
    for a, i in enumerate(qi):
        s = served[i]
        if s < 0:
            continue
        nx = L.next[s]
        if nx < 0 or int(nx) not in pos:
            continue
        j = pos[int(nx)]
        has_next[i] = True
        score_next[i] = S[a, j]
        rank_next[i] = (S[a] > S[a, j]).sum()
print(f"[{cell}] scored in {time.time() - t0:.0f}s; stale n={stale.sum()} has_next={has_next[stale].mean():.3f}", flush=True)

res = {"cell": cell, "T": {}, "M": {}, "S": {}}
m = stale & has_next
H = L.head
e_mean5 = err_of(H[TOP[:, :5]].mean(1))
e_top1 = err_of(H[TOP[:, 0]])
e_next = np.full(N, np.nan); e_next[has_next] = err_of(H[np.where(has_next, L.next[np.maximum(served, 0)], 0)])[has_next]
e_tail = np.full(N, np.nan); e_tail[pi] = err_of(a_exec[prev_idx[pi], 5:10] if False else np.where(pi[:, None, None], a_exec[np.maximum(prev_idx, 0), 5:10], 0))[pi]
# same-episode continuation: next(served) head vs tail of the served chunk (steps 5..9) -- how different are they?
T = {"n": int(m.sum()), "err_mean5": float(e_mean5[m].mean()), "err_top1": float(e_top1[m].mean()),
     "err_next_served": float(e_next[m].mean()), "err_tail_prev": float(e_tail[m].mean()),
     "err_next_succ_ep": float(e_next[m & ep_success].mean()), "err_mean5_succ_ep": float(e_mean5[m & ep_success].mean()),
     "err_next_fail_ep": float(e_next[m & ~ep_success].mean()), "err_mean5_fail_ep": float(e_mean5[m & ~ep_success].mean()),
     "rank_next_p50": float(np.median(rank_next[m])), "frac_next_in_top5": float((rank_next[m] < 5).mean()),
     "frac_next_in_top1": float((rank_next[m] < 1).mean())}
# gated tracking: track next(served) when its score is within the top-m (visual+state verification), else mean5
for gate in (1, 3, 5, 10, 20):
    tr = m & (rank_next < gate)
    e = e_mean5.copy(); e[tr] = e_next[tr]
    T[f"gate_top{gate}"] = {"frac_track": float(tr[m].mean() if m.any() else 0), "err": float(e[m].mean()),
                            "err_tracked": float(e_next[tr].mean()) if tr.any() else None,
                            "err_mean5_on_tracked": float(e_mean5[tr].mean()) if tr.any() else None,
                            "err_succ_ep": float(e[m & ep_success].mean()), "err_fail_ep": float(e[m & ~ep_success].mean())}
# blend: mean of {next(served), top-4}
nx_head = H[np.maximum(L.next[np.maximum(served, 0)], 0)]
blend = (nx_head + H[TOP[:, :4]].sum(1)) / 5.0
e_bl = err_of(blend)
T["blend_next+top4"] = {"err": float(e_bl[m].mean()), "err_succ_ep": float(e_bl[m & ep_success].mean()), "err_fail_ep": float(e_bl[m & ~ep_success].mean())}
for g in (3, 5):
    tr = m & (rank_next < g)
    e = e_mean5.copy(); e[tr] = e_bl[tr]
    T[f"blend_gate_top{g}"] = {"err": float(e[m].mean()), "err_succ_ep": float(e[m & ep_success].mean()), "err_fail_ep": float(e[m & ~ep_success].mean())}
# by stuck / moving split: motion of the query vs the library's motion at the served row
motion_q = np.zeros(N); motion_q[pi] = np.linalg.norm(rs[pi] - rs[prev_idx[pi]], axis=1)
lib_motion_served = np.zeros(N); lib_motion_served[pi] = L.motion_next[served[pi]]
ratio = motion_q / np.maximum(lib_motion_served, 1e-6)
for nm, msk in (("moving(ratio>=.5)", m & (ratio >= 0.5)), ("slow(ratio<.5)", m & (ratio < 0.5)), ("stuck(ratio<.2)", m & (ratio < 0.2))):
    T[nm] = {"frac": float(msk[m].mean()), "err_mean5": float(e_mean5[msk].mean()) if msk.any() else None,
             "err_next": float(e_next[msk].mean()) if msk.any() else None,
             "err_gate5": float(np.where(rank_next < 5, e_next, e_mean5)[msk].mean()) if msk.any() else None}
res["T"] = T
print(f"[{cell}] T: mean5 {T['err_mean5']:.3f} top1 {T['err_top1']:.3f} next(served) {T['err_next_served']:.3f} tail {T['err_tail_prev']:.3f} "
      f"gate5 {T['gate_top5']['err']:.3f} (track {T['gate_top5']['frac_track']:.2f}) gate3 {T['gate_top3']['err']:.3f} blend {T['blend_next+top4']['err']:.3f}", flush=True)

# ---------------------------------------------------------------- M. mode-seeking synthesis on the top-5
H5 = (H[TOP[:, :5]] / sig).reshape(N, 5, 35)
iu = np.triu_indices(5, 1)
Dp = np.sqrt(((H5[:, iu[0]] - H5[:, iu[1]]) ** 2).mean(2))      # [N,10] pairwise RMS
dmax = Dp.max(1); dmed = np.median(Dp, 1)
# 2-cluster split seeded by the farthest pair; assign each to the nearest seed; largest cluster (ties -> the one with top-1)
far = Dp.argmax(1); a_i, b_i = iu[0][far], iu[1][far]
A = H5[np.arange(N), a_i]; B = H5[np.arange(N), b_i]
da = np.sqrt(((H5 - A[:, None]) ** 2).mean(2)); db = np.sqrt(((H5 - B[:, None]) ** 2).mean(2))
lab = (db < da)                                                    # True -> cluster B
sizeB = lab.sum(1); sizeA = 5 - sizeB
pickB = (sizeB > sizeA) | ((sizeB == sizeA) & lab[:, 0])
inc = np.where(pickB[:, None], lab, ~lab).astype(np.float64)
Hraw = H[TOP[:, :5]]
cl_mean = (Hraw * inc[:, :, None, None]).sum(1) / inc.sum(1)[:, None, None]
e_cl = err_of(cl_mean)
M = {}
for thr_name, thr in (("all", -1.0), ("dmax>1.0", 1.0), ("dmax>1.5", 1.5), ("dmax>2.0", 2.0)):
    mm = stale & (dmax > thr)
    e = e_mean5.copy(); e[mm] = e_cl[mm]
    M[thr_name] = {"frac_multimodal": float(mm[stale].mean()), "err_mean5_on_mm": float(e_mean5[mm].mean()) if mm.any() else None,
                   "err_cluster_on_mm": float(e_cl[mm].mean()) if mm.any() else None, "err_top1_on_mm": float(e_top1[mm].mean()) if mm.any() else None,
                   "err_overall": float(e[stale].mean())}
# how often is the mean action "between modes": distance of the mean to its nearest member relative to member spread
mean5 = H5.mean(1)
dm = np.sqrt(((H5 - mean5[:, None]) ** 2).mean(2)).min(1)
M["mean_to_nearest_member_p50_stale"] = float(np.median(dm[stale])); M["dmax_p50_stale"] = float(np.median(dmax[stale]))
M["dmax_p50_step0"] = float(np.median(dmax[step == 0]))
res["M"] = M
print(f"[{cell}] M: {json.dumps({k: (round(v, 3) if isinstance(v, float) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in M.items()})}", flush=True)

# ---------------------------------------------------------------- S. stuck runs (observation not changing)
thr = np.percentile(L.motion_next[L.has_next], 10)
stuck = np.zeros(N, np.int64)
for i in range(N):
    if prev_idx[i] >= 0 and motion_q[i] < thr:
        stuck[i] = stuck[prev_idx[i]] + 1
# does the selection (top-1 row) change from one stuck decision to the next?
same_top1 = np.zeros(N, bool); same_top1[pi] = TOP[pi, 0] == TOP[prev_idx[pi], 0]
same_served = np.zeros(N, bool); same_served[pi & (prev_idx >= 0)] = False
S = {"motion_thr": float(thr)}
for r in (0, 1, 2, 3, 5):
    mm = stale & (stuck >= r) & (stuck < (r + 1 if r < 5 else 10 ** 6))
    S[f"run={r}{'+' if r == 5 else ''}"] = {"frac": float(mm[stale].mean()), "err_mean5": float(e_mean5[mm].mean()) if mm.any() else None,
                                              "same_top1_as_prev": float(same_top1[mm].mean()) if mm.any() else None,
                                              "frac_fail_ep": float((~ep_success)[mm].mean()) if mm.any() else None}
# B0's served row repeated? (rec_top1 same as previous decision's rec_top1)
rep = np.zeros(N, bool); rep[pi] = rec_top1[pi] == rec_top1[prev_idx[pi]]
S["b0_served_same_row_as_prev_stale"] = float(rep[stale].mean())
S["b0_served_same_row_when_stuck>=2"] = float(rep[stale & (stuck >= 2)].mean()) if (stale & (stuck >= 2)).any() else None
S["b0_served_same_episode_as_prev_stale"] = float((L.ep[np.maximum(rec_top1, 0)][pi] == L.ep[np.maximum(rec_top1[prev_idx[pi]], 0)]).mean())
res["S"] = S
print(f"[{cell}] S: {json.dumps(S)[:600]}", flush=True)
(OUT / f"diag2_{cell}.json").write_text(json.dumps(res, indent=1))
print(f"[{cell}] done {time.time() - t0:.0f}s", flush=True)
