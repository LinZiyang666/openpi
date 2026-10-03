"""Retrieval-regime check for the trainer as built by sol (rounds/r10/train.py), library data only.

sol's trainer poses every library row as a pseudo-query with ``prev_hit=False`` (AWM "fresh" regime 1: distance
d / median(d) + lam_c * continuity / s_c, continuity = RMS between the PREVIOUS row's policy tail [5:10] and each
candidate's head [0:5]).  At deployment that regime only occurs right after a policy call; after a cache HIT (the
common case in A and between guard calls in G) the look uses regime 2 (plain d).  This driver trains on either
regime's syntheses and evaluates held-out episodes served in both regimes:

  loeo_r2      LOEO syntheses in regime 2 (= run_offline "loeo")
  loeo_r1      LOEO syntheses in regime 1 (sol's LOEO)
  loeo_mix     both syntheses, half weight each
  sol_pair     sol's PAIR: regime-1 neighbour order, radius = pooled (all tasks) 95th pct of the regime-1 16th-neighbour
               score, cap 16 nearest, weights summing to the number of PAIRS (fable fit_head normalisation)
  pair_r1_rn   same pairs, weights summing to the number of anchor rows (row-normalised)
  pair_r2_r50  run_offline's best pair variant (regime 2, radius = per-task median 16-NN distance, cap 16, row-normalised)

Held-out conditions: ``r2`` (fresh look after a cache HIT) and ``r1`` (fresh look right after a policy call: the
previous decision's executed chunk is the row's own previous policy chunk).  Step-0 rows use the early metric in both.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r10.analysis_opus.tools.run_offline import kref_for, pair_weights, projections

OUT = Path(__file__).resolve().parents[1] / "out" / "regime"
LAM_C = 0.5
NH = 35
SCHEMES = ("loeo_r2", "loeo_r1", "loeo_mix", "sol_pair", "pair_r1_rn", "pair_r2_r50")


def heads35(act, sig):
    return (act[:, :5, :7] / sig).reshape(len(act), -1)


def tails35(act, sig):
    return (act[:, 5:10, :7] / sig).reshape(len(act), -1)


def rms_dist(A, B):
    d2 = (A * A).sum(1)[:, None] - 2 * A @ B.T + (B * B).sum(1)[None]
    return np.sqrt(np.maximum(d2, 0) / NH)


def scores(met, qX, qstep, qep, qprev_tail, cX, cep, cH, s_c, regime, cand_mask=None, exclude_own=True):
    """Per query the score AWM ranks by: early-metric distance at step 0; main distance (regime 2) or
    d / median_admissible(d) + lam_c * c / s_c (regime 1) at step >= 1.  Returns (S [nq, nc] with inf = excluded,
    main-metric d [nq, nc])."""
    D = core.pdist(met.code(qX), met.code(cX))
    bad = np.zeros_like(D, bool)
    if exclude_own:
        bad |= qep[:, None] == cep[None, :]
    if cand_mask is not None:
        bad |= ~cand_mask[None, :]
    D[bad] = np.inf
    S = D.copy()
    if regime == 1:
        Dm = np.where(bad, np.nan, D)
        med = np.nanmedian(Dm, axis=1) + 1e-12
        C = rms_dist(qprev_tail, cH)
        S = D / med[:, None] + LAM_C * C / s_c
        S[bad] = np.inf
    s0 = qstep == 0
    if s0.any():
        D0 = core.pdist(met.code0(qX[s0]), met.code0(cX))
        D0[bad[s0]] = np.inf
        S[s0] = D0
    return S, D


def synth(S, cact, kref):
    idx, Sk, w = core.topk_kernel(S, kref)
    return np.einsum("qk,qktc->qtc", w, cact[idx]).astype(np.float32)


def prev_tail_of(rows_global_idx, ep, step, act, sig):
    """Tail [5:10] of each row's previous decision in the same episode (rows are episode-contiguous, step-ordered)."""
    prev = rows_global_idx - 1
    ok = (step[rows_global_idx] > 0)
    prev = np.where(ok, prev, rows_global_idx)
    assert np.all(ep[prev] == ep[rows_global_idx]) and np.all(step[prev][ok] == step[rows_global_idx][ok] - 1)
    return tails35(act[prev], sig)


def run_job(job):
    model, suite, size, fold = job
    t0 = time.time()
    out_path = OUT / f"{model}_{suite}_{size}_f{fold}.npz"
    if out_path.exists():
        return str(out_path), 0.0
    C = core.load_cell(model, suite)
    rows, fold_of_row = core.subset_and_folds(C, size)
    tr_m = fold_of_row != fold
    P, _ = projections(model, suite, rows, tr_m, size, "auto")
    act, rs = C["act"][rows], C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    sig_m = act[tr_m][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    sig_r = C["act"][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    kref = kref_for(size)
    X = np.concatenate([P, rs], 1)
    H = heads35(act, sig_m)
    ptail = prev_tail_of(np.arange(len(rows)), ep, step, act, sig_m)
    F = lambda idx, chunk: core.features(P[idx], rs[idx], chunk, step[idx], task[idx], sig_m)
    per_task = {}
    sol_d16 = []
    # pass 1: metric, LOEO syntheses in both regimes, regime-1 scores (for sol's pooled radius)
    for t in range(10):
        itr = np.flatnonzero(tr_m & (task == t))
        ite = np.flatnonzero(~tr_m & (task == t))
        met = core.TaskMetric(X[itr], H[itr], ep[itr], step[itr])
        # AWM's s_c on the candidate library: median over rows of the 1-NN (other episode) tail -> head RMS distance
        Dc = rms_dist(tails35(act[itr], sig_m), H[itr])
        Dc[ep[itr][:, None] == ep[itr][None, :]] = np.inf
        s_c = float(np.median(Dc.min(1))) + 1e-6
        S2, Dm = scores(met, X[itr], step[itr], ep[itr], ptail[itr], X[itr], ep[itr], H[itr], s_c, 2)
        S1, _ = scores(met, X[itr], step[itr], ep[itr], ptail[itr], X[itr], ep[itr], H[itr], s_c, 1)
        s_r2, s_r1 = synth(S2, act[itr], kref), synth(S1, act[itr], kref)
        sol_d16.append(np.sort(S1, 1)[:, 15])
        d16_main = np.sort(Dm, 1)[:, 15]
        per_task[t] = dict(itr=itr, ite=ite, met=met, s_c=s_c, S1=S1, Dm=Dm, s_r2=s_r2, s_r1=s_r1,
                           r50=float(np.median(d16_main)))
    sol_radius = float(np.quantile(np.concatenate(sol_d16), 0.95))
    errs, rec = {}, {k: [] for k in ("task", "ep", "step", "success")}
    diag = dict(sol_radius=sol_radius, tasks=[])
    for t in range(10):
        T = per_task[t]
        itr, ite, met, s_c = T["itr"], T["ite"], T["met"], T["s_c"]
        w_ep = core.episode_weights(ep[itr])
        sets = {"loeo_r2": (F(itr, T["s_r2"]), core.residual(act[itr], T["s_r2"], sig_m), w_ep, len(itr)),
                "loeo_r1": (F(itr, T["s_r1"]), core.residual(act[itr], T["s_r1"], sig_m), w_ep, len(itr))}
        sets["loeo_mix"] = (np.concatenate([sets["loeo_r2"][0], sets["loeo_r1"][0]]),
                            np.concatenate([sets["loeo_r2"][1], sets["loeo_r1"][1]]), np.tile(w_ep, 2) / 2, len(itr))
        # sol's pairs: regime-1 order, global radius, cap 16
        S1 = T["S1"]
        a_i, b_j = [], []
        for i in range(len(itr)):
            order = np.argsort(S1[i], kind="stable")[:16]
            j = order[S1[i, order] <= sol_radius]
            a_i.append(np.full(len(j), i)); b_j.append(j)
        a_i, b_j = np.concatenate(a_i), np.concatenate(b_j)
        w, n_anchor = pair_weights(a_i, ep[itr])
        Xp = core.features(P[itr][a_i], rs[itr][a_i], act[itr][b_j], step[itr][a_i], task[itr][a_i], sig_m)
        Yp = core.residual(act[itr][a_i], act[itr][b_j], sig_m)
        sets["sol_pair"] = (Xp, Yp, w, len(a_i))
        sets["pair_r1_rn"] = (Xp, Yp, w, n_anchor)
        n_sol_pairs = len(a_i)
        # regime-2 r50 c16 pairs (run_offline's best)
        Dm = T["Dm"]
        a2, b2 = [], []
        for i in range(len(itr)):
            cand = np.flatnonzero(Dm[i] <= T["r50"])
            j = cand[np.argsort(Dm[i, cand], kind="stable")[:16]]
            a2.append(np.full(len(j), i)); b2.append(j)
        a2, b2 = np.concatenate(a2), np.concatenate(b2)
        w2, n2 = pair_weights(a2, ep[itr])
        sets["pair_r2_r50"] = (core.features(P[itr][a2], rs[itr][a2], act[itr][b2], step[itr][a2], task[itr][a2], sig_m),
                               core.residual(act[itr][a2], act[itr][b2], sig_m), w2, n2)
        diag["tasks"].append(dict(task=t, n_train=int(len(itr)), sol_pairs=int(n_sol_pairs),
                                  sol_pairs_per_row=n_sol_pairs / len(itr), s_c=s_c,
                                  loeo_r1_err10=float(core.motion_err(act[itr], T["s_r1"], sig_r).mean()),
                                  loeo_r2_err10=float(core.motion_err(act[itr], T["s_r2"], sig_r).mean())))
        # held-out syntheses in both regimes (candidates = training fold, deployment-faithful metric)
        served = {}
        for reg in (2, 1):
            S, _ = scores(met, X[ite], step[ite], ep[ite], ptail[ite], X[itr], ep[itr], H[itr], s_c, reg, exclude_own=False)
            served[f"r{reg}"] = synth(S, act[itr], kref)
        for cond, s_eval in served.items():
            errs.setdefault((cond, "none", "e10"), []).append(core.motion_err(act[ite], s_eval, sig_r))
        for sch in SCHEMES:
            Xs, Ys, ws, tot = sets[sch]
            head = core.Head(Xs, Ys, ws, total=tot)
            for cond, s_eval in served.items():
                corr = head.predict(F(ite, s_eval)).reshape(-1, core.H10, core.MOT) * sig_m[:core.MOT]
                res = act[ite][:, :core.H10, :core.MOT] - s_eval[:, :core.H10, :core.MOT]
                errs.setdefault((cond, sch, "dot"), []).append(((corr * res) / sig_r[:core.MOT] ** 2).sum((1, 2)))
                errs.setdefault((cond, sch, "cc"), []).append(((corr * corr) / sig_r[:core.MOT] ** 2).sum((1, 2)))
                fx = s_eval.copy()
                fx[:, :core.H10, :core.MOT] += 0.5 * corr
                errs.setdefault((cond, sch, "e10_b0.5"), []).append(core.motion_err(act[ite], fx, sig_r))
        rec["task"].append(task[ite]); rec["ep"].append(ep[ite]); rec["step"].append(step[ite])
        rec["success"].append(C["success"][rows][ite])
    arrays = {k: np.concatenate(v) for k, v in rec.items()}
    arrays.update({"|".join(k): np.concatenate(v) for k, v in errs.items()})
    meta = dict(model=model, suite=suite, size=size, fold=fold, kref=kref, diag=diag, wall_s=time.time() - t0,
                schemes=list(SCHEMES))
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, meta_json=np.array(json.dumps(meta)), **arrays)
    tmp.replace(out_path)
    return str(out_path), time.time() - t0


def summarize(pattern="*.npz"):
    import glob
    from collections import defaultdict
    from exp.offline_search.rounds.r10.analysis_opus.tools.analyze import boot_ci
    groups = defaultdict(list)
    for f in sorted(glob.glob(str(OUT / pattern))):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        m = json.loads(str(z["meta_json"]))
        groups[(m["suite"], m["model"], m["size"])].append(({k: z[k] for k in z.files if k != "meta_json"}, m))
    print("| cell | size | held-out regime | none e10 | " + " | ".join(SCHEMES) + " | sol pairs/row |")
    print("|---|---|---|---|" + "---|" * len(SCHEMES) + "---|")
    out = {}
    for key in sorted(groups):
        parts = groups[key]
        if len(parts) != 5:
            continue
        arr = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
        ppr = np.mean([t["sol_pairs_per_row"] for p in parts for t in p[1]["diag"]["tasks"]])
        for cond in ("r2", "r1"):
            base = arr[f"{cond}|none|e10"]
            cells = []
            for s in SCHEMES:
                m, lo, hi = boot_ci(arr[f"{cond}|{s}|e10_b0.5"] - base, arr["ep"])
                b = base.mean()
                star = "*" if (hi < 0 or lo > 0) else ""
                cells.append(f"{100 * m / b:+.1f}{star}")
                out[f"{key[1]}_{key[0]}_{key[2]}|{cond}|{s}"] = dict(rel=m / b, ci=(lo / b, hi / b))
            print(f"| {key[1]} {key[0]} | {key[2]} | {cond} | {base.mean():.3f} | " + " | ".join(cells) + f" | {ppr:.1f} |")
    (OUT.parent / "regime_summary.json").write_text(json.dumps(out, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", default="pi05:l10:50,groot:l10:50,pi05:spatial:50,groot:spatial:50,"
                                      "pi05:l10:500,groot:l10:500,pi05:spatial:500,groot:spatial:500")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(argv)
    if a.summarize:
        summarize()
        return
    jobs = [(s.split(":")[0], s.split(":")[1], int(s.split(":")[2]), f) for s in a.jobs.split(",") for f in range(5)]
    jobs.sort(key=lambda j: -j[2])
    print(f"{len(jobs)} jobs", flush=True)
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for path, dt in pool.imap_unordered(run_job, jobs):
            print(f"done {path} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
