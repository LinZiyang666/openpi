"""k-fold-by-episode comparison of corrector training schemes on the library only (one job = cell x size x fold).

Usage (heavy; pin CPUs):
  taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
    .venv/bin/python -m exp.offline_search.rounds.r10.analysis_opus.tools.run_offline --jobs pi05:l10:50 --workers 16

For every fold f (episodes at permutation position % 5 == f are held out):
  * the cache (PCA for sizes < 500, per-task metric, kernel) is refitted on the training-fold rows;
  * training sets for every scheme are built from the training-fold rows only;
  * every held-out row is served by the training-fold cache ("full"; and "half" = a random half of the training
    episodes as candidates, a sparser-library / drift condition), corrected by each head, and compared with the
    row's own stored policy chunk (motion channels 0-5, 10 steps, sigma units).
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r10.analysis_opus.tools import core

OUT = Path(__file__).resolve().parents[1] / "out" / "kfold"
SCHEMES = ("loeo", "loeo_aug", "xfit", "xfit_aug", "pair_r50_c16", "pair_r95_c64", "pair_r95_c64_raw", "pair_all_c64")
N_INNER = 4
BLENDS = (0.5, 1.0)


def kref_for(size):
    return 8 if size >= 100 else 5          # R8 pure-cache rows: 50-ep libraries kref 5, 500-ep libraries kref 8


def projections(model, suite, rows, train_mask, size, pca):
    if pca == "stored" or (pca == "auto" and size == 500):
        P0, P1 = core.stored_pca(model, suite)
        return np.concatenate([P0[rows], P1[rows]], 1).astype(np.float64), "stored_r01_bpool"
    K0, K1 = core.gather_keys(model, suite, rows)
    out = []
    for Kx in (K0, K1):
        mu, V, _ = core.pca_fit(Kx[train_mask])
        out.append((Kx @ V - mu @ V).astype(np.float64))
        del Kx
    return np.concatenate(out, 1), "refit_trainfold"


def build_pairs(D, ep_tr, radius, cap, rng=None, random_m=None):
    """Anchor/partner index pairs. D: [n, n] main-metric distances with own-episode entries = inf."""
    A, B = [], []
    n = len(D)
    for i in range(n):
        if random_m is not None:
            cand = np.flatnonzero(np.isfinite(D[i]))
            j = rng.choice(cand, size=min(random_m, len(cand)), replace=False)
        else:
            cand = np.flatnonzero(D[i] <= radius)
            if not len(cand):
                continue
            j = cand[np.argsort(D[i, cand], kind="stable")[:cap]]
        A.append(np.full(len(j), i))
        B.append(j)
    return np.concatenate(A), np.concatenate(B)


def pair_weights(a_idx, ep_tr):
    """Each anchor row's pairs share weight 1; each episode's anchor rows share weight 1 (equal weight per episode)."""
    n_pairs = np.bincount(a_idx, minlength=len(ep_tr))[a_idx]
    anchors = np.unique(a_idx)
    _, inv, cnt = np.unique(ep_tr[anchors], return_inverse=True, return_counts=True)
    per_ep = np.zeros(len(ep_tr))
    per_ep[anchors] = 1.0 / cnt[inv]
    return per_ep[a_idx] / n_pairs, len(anchors)


def run_job(job):
    model, suite, size, fold, pca = job[:5]
    mode = job[5] if len(job) > 5 else "main"
    # mode "pool": per-task LOEO / PAIR-r50 heads again, plus ONE pooled head over all tasks (task one-hot input),
    # plus a per-task LOEO head with 10x ridge (alpha 1000) -- head-capacity checks, not the specified recipe
    run_schemes = SCHEMES if mode == "main" else ("loeo", "pair_r50_c16", "loeo_a1000")
    pooled = {"loeo_pool": "loeo", "pair_r50_c16_pool": "pair_r50_c16"} if mode == "pool" else {}
    pool_sets = {k: [] for k in pooled}
    pool_test = []
    t0 = time.time()
    out_path = OUT / f"{model}_{suite}_{size}_f{fold}_{pca}{'' if mode == 'main' else '_' + mode}.npz"
    if out_path.exists():
        return str(out_path), 0.0
    C = core.load_cell(model, suite)
    rows, fold_of_row = core.subset_and_folds(C, size)
    tr_m = fold_of_row != fold
    P, pca_src = projections(model, suite, rows, tr_m, size, pca)
    act = C["act"][rows]
    rs = C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    sig_m = act[tr_m][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)           # method sigma (training fold)
    sig_r = C["act"][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)             # reporting yardstick (500-lib)
    kref = kref_for(size)
    rec = {k: [] for k in ("row", "task", "ep", "step", "ep_len", "success", "d1", "d1norm", "d1norm_in", "d16",
                           "d1_half", "d1norm_half")}
    errs = {}
    diag = []
    X = np.concatenate([P, rs], 1)
    heads = (act[:, :5, :7] / sig_m).reshape(len(act), -1)
    for t in range(10):
        itr = np.flatnonzero(tr_m & (task == t))
        ite = np.flatnonzero(~tr_m & (task == t))
        if not len(ite):
            continue
        met = core.TaskMetric(X[itr], heads[itr], ep[itr], step[itr])
        # LOEO syntheses for the training rows
        s_tr, d1_tr, d16_tr, _, _ = core.serve(X[itr], step[itr], ep[itr], met, X[itr], None, ep[itr], act[itr], kref)
        # cross-fitted LOEO ("xfit"): the metric is refitted without the query's inner group of episodes, so the
        # synthesis sees the query exactly as the deployed cache sees a new episode (no in-sample metric optimism)
        tr_eps = np.unique(ep[itr])
        pos = {int(e): k % N_INNER for k, e in enumerate(np.random.default_rng([fold, t, 5]).permutation(tr_eps))}
        grp = np.array([pos[int(e)] for e in ep[itr]])
        s_xf = np.empty_like(s_tr)
        d1_xf = np.empty(len(itr))
        s_xa = [np.empty_like(s_tr), np.empty_like(s_tr)]
        r3 = np.random.default_rng([fold, t, 17])
        for g in range(N_INNER):
            qg, cg = grp == g, grp != g
            met_g = core.TaskMetric(X[itr][cg], heads[itr][cg], ep[itr][cg], step[itr][cg])
            sg, d1g, *_ = core.serve(X[itr][qg], step[itr][qg], ep[itr][qg], met_g, X[itr][cg], None, ep[itr][cg],
                                     act[itr][cg], kref, exclude_own=False)
            s_xf[qg], d1_xf[qg] = sg, d1g
            ceps = np.unique(ep[itr][cg])
            for slot, frac in enumerate((0.5, 0.25)):
                k = r3.choice(ceps, size=max(1, int(round(frac * len(ceps)))), replace=False)
                s_xa[slot][qg] = core.serve(X[itr][qg], step[itr][qg], ep[itr][qg], met_g, X[itr][cg], None,
                                            ep[itr][cg], act[itr][cg], kref, exclude_own=False,
                                            cand_mask=np.isin(ep[itr][cg], k))[0]
        med_d1 = float(np.median(d1_xf))         # deployment-faithful scale: median d1 of a new episode's rows
        med_d1_in = float(np.median(d1_tr))
        # held-out rows: full training-fold library and a random half of its episodes
        s_te, d1_te, d16_te, _, _ = core.serve(X[ite], step[ite], ep[ite], met, X[itr], None, ep[itr], act[itr], kref,
                                               exclude_own=False)
        rng = np.random.default_rng([fold, t, 7])
        keep = rng.choice(tr_eps, size=max(1, len(tr_eps) // 2), replace=False)
        cm = np.isin(ep[itr], keep)
        s_th, d1_th, _, _, _ = core.serve(X[ite], step[ite], ep[ite], met, X[itr], None, ep[itr], act[itr], kref,
                                          exclude_own=False, cand_mask=cm)
        # ---- training sets
        F = lambda idx_feat, chunk: core.features(P[idx_feat], rs[idx_feat], chunk, step[idx_feat], task[idx_feat], sig_m)
        sets = {}
        w_ep = core.episode_weights(ep[itr])
        sets["loeo"] = (F(itr, s_tr), core.residual(act[itr], s_tr, sig_m), w_ep, len(itr))
        # LOEO + drift augmentation: per training episode, a random 50 % and a random 25 % of the other episodes
        aug_s = [s_tr]
        for frac, salt in ((0.5, 11), (0.25, 12)):
            s_aug = np.empty_like(s_tr)
            r2 = np.random.default_rng([fold, t, salt])
            for e in tr_eps:
                qm = ep[itr] == e
                others = tr_eps[tr_eps != e]
                k = r2.choice(others, size=max(1, int(round(frac * len(others)))), replace=False)
                s_aug[qm] = core.serve(X[itr][qm], step[itr][qm], ep[itr][qm], met, X[itr], None, ep[itr], act[itr],
                                       kref, cand_mask=np.isin(ep[itr], k))[0]
            aug_s.append(s_aug)
        Xa = np.concatenate([F(itr, s) for s in aug_s])
        Ya = np.concatenate([core.residual(act[itr], s, sig_m) for s in aug_s])
        sets["loeo_aug"] = (Xa, Ya, np.tile(w_ep, 3) / 3, len(itr))
        sets["xfit"] = (F(itr, s_xf), core.residual(act[itr], s_xf, sig_m), w_ep, len(itr))
        xs = [s_xf] + s_xa
        sets["xfit_aug"] = (np.concatenate([F(itr, s) for s in xs]),
                            np.concatenate([core.residual(act[itr], s, sig_m) for s in xs]), np.tile(w_ep, 3) / 3, len(itr))
        # pairs (main-metric distance between training rows, own episode excluded)
        Zt = met.code(X[itr])
        D = core.pdist(Zt, Zt)
        D[ep[itr][:, None] == ep[itr][None, :]] = np.inf
        r50, r95 = float(np.median(d16_tr)), float(np.percentile(d16_tr, 95))
        pdiag = {}
        for name, radius, cap, rand in (("pair_r50_c16", r50, 16, None), ("pair_r95_c64", r95, 64, None),
                                        ("pair_all_c64", None, None, 64)):
            a_i, b_j = build_pairs(D, ep[itr], radius, cap, rng=np.random.default_rng([fold, t, 13]), random_m=rand)
            w, n_anchor = pair_weights(a_i, ep[itr])
            Xp = core.features(P[itr][a_i], rs[itr][a_i], act[itr][b_j], step[itr][a_i], task[itr][a_i], sig_m)
            Yp = core.residual(act[itr][a_i], act[itr][b_j], sig_m)
            sets[name] = (Xp, Yp, w, n_anchor)
            if name == "pair_r95_c64":
                sets["pair_r95_c64_raw"] = (Xp, Yp, w, len(a_i))
            pdiag[name] = dict(n_pairs=int(len(a_i)), n_anchor=int(n_anchor), frac_rows_no_pair=1 - n_anchor / len(itr),
                               mean_pair_dist=float(np.mean(D[a_i, b_j])))
        del D
        diag.append(dict(task=t, n_train=int(len(itr)), n_test=int(len(ite)), r50=r50, r95=r95, med_d1_loeo=med_d1_in,
                         med_d1_xfit=med_d1, med_d16_loeo=float(np.median(d16_tr)), pairs=pdiag,
                         loeo_err10=float(core.motion_err(act[itr], s_tr, sig_r).mean()),
                         xfit_err10=float(core.motion_err(act[itr], s_xf, sig_r).mean()),
                         loeo_resid_rms=float(np.sqrt(np.mean(core.residual(act[itr], s_tr, sig_r) ** 2))),
                         xfit_resid_rms=float(np.sqrt(np.mean(core.residual(act[itr], s_xf, sig_r) ** 2)))))
        # ---- fit + evaluate
        Xte_full = F(ite, s_te)
        Xte_half = F(ite, s_th)
        sig6 = sig_m[:core.MOT]
        for cond, s_eval, Xe in (("full", s_te, Xte_full), ("half", s_th, Xte_half)):
            key = (cond, "none")
            errs.setdefault(key + ("e10",), []).append(core.motion_err(act[ite], s_eval, sig_r))
            errs.setdefault(key + ("e5",), []).append(core.motion_err(act[ite], s_eval, sig_r, steps=5))
        sets["loeo_a1000"] = sets["loeo"]
        for k, src in pooled.items():
            Xs, Ys, ws, tot = sets[src]
            pool_sets[k].append((Xs, Ys, ws / ws.sum() * len(np.unique(ep[itr])), tot))   # equal weight per episode
        if pooled:
            pool_test.append((Xte_full, Xte_half, s_te, s_th, act[ite]))
        for sch in run_schemes:
            Xs, Ys, ws, tot = sets[sch]
            head = core.Head(Xs, Ys, ws, total=tot, alpha=1000.0 if sch == "loeo_a1000" else core.ALPHA)
            # input-chunk sensitivity: slope of the correction w.r.t. a small perturbation of the served chunk
            # (-1 = the head replaces the served chunk by its own estimate; 0 = correction ignores the chunk)
            dl = np.random.default_rng([fold, t, 23]).standard_normal((len(ite), core.H10, core.MOT)) * 0.1 * sig6
            s_pert = s_te.copy()
            s_pert[:, :core.H10, :core.MOT] += dl
            c0 = head.predict(Xte_full).reshape(-1, core.H10, core.MOT) * sig6
            c1 = head.predict(F(ite, s_pert)).reshape(-1, core.H10, core.MOT) * sig6
            errs.setdefault(("full", sch, "slope_num"), []).append((((c1 - c0) * dl) / sig6 ** 2).sum((1, 2)))
            errs.setdefault(("full", sch, "slope_den"), []).append(((dl * dl) / sig6 ** 2).sum((1, 2)))
            for cond, s_eval, Xe in (("full", s_te, Xte_full), ("half", s_th, Xte_half)):
                corr = head.predict(Xe).reshape(-1, core.H10, core.MOT) * sig6
                res = (act[ite][:, :core.H10, :core.MOT] - s_eval[:, :core.H10, :core.MOT])
                cn = corr / sig_r[:core.MOT]
                rn = res / sig_r[:core.MOT]
                errs.setdefault((cond, sch, "dot"), []).append((cn * rn).sum((1, 2)))
                errs.setdefault((cond, sch, "cc"), []).append((cn * cn).sum((1, 2)))
                for b in BLENDS:
                    fixed = s_eval.copy()
                    fixed[:, :core.H10, :core.MOT] += b * corr
                    errs.setdefault((cond, sch, f"e10_b{b:g}"), []).append(core.motion_err(act[ite], fixed, sig_r))
                    if b == 0.5:
                        errs.setdefault((cond, sch, "e5_b0.5"), []).append(core.motion_err(act[ite], fixed, sig_r, steps=5))
            del head
        rec["row"].append(rows[ite]); rec["task"].append(task[ite]); rec["ep"].append(ep[ite])
        rec["step"].append(step[ite]); rec["ep_len"].append(C["ep_len"][rows][ite]); rec["success"].append(C["success"][rows][ite])
        rec["d1"].append(d1_te); rec["d1norm"].append(d1_te / med_d1); rec["d1norm_in"].append(d1_te / med_d1_in)
        rec["d16"].append(d16_te)
        rec["d1_half"].append(d1_th); rec["d1norm_half"].append(d1_th / med_d1)
        del sets
    for k, parts in pool_sets.items():
        head = core.Head(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                         np.concatenate([p[2] for p in parts]), total=sum(p[3] for p in parts))
        for cond, xi, si in (("full", 0, 2), ("half", 1, 3)):
            e10, e5, dot, cc, e10b1 = [], [], [], [], []
            for pt in pool_test:
                Xe, s_eval, a_te = pt[xi], pt[si], pt[4]
                corr = head.predict(Xe).reshape(-1, core.H10, core.MOT) * sig_m[:core.MOT]
                res = a_te[:, :core.H10, :core.MOT] - s_eval[:, :core.H10, :core.MOT]
                dot.append(((corr * res) / sig_r[:core.MOT] ** 2).sum((1, 2)))
                cc.append(((corr * corr) / sig_r[:core.MOT] ** 2).sum((1, 2)))
                for b, store in ((0.5, e10), (1.0, e10b1)):
                    fx = s_eval.copy()
                    fx[:, :core.H10, :core.MOT] += b * corr
                    store.append(core.motion_err(a_te, fx, sig_r))
                    if b == 0.5:
                        e5.append(core.motion_err(a_te, fx, sig_r, steps=5))
            errs[(cond, k, "e10_b0.5")] = e10
            errs[(cond, k, "e10_b1")] = e10b1
            errs[(cond, k, "e5_b0.5")] = e5
            errs[(cond, k, "dot")] = dot
            errs[(cond, k, "cc")] = cc
    arrays = {k: np.concatenate(v) for k, v in rec.items()}
    arrays.update({"|".join(k): np.concatenate(v) for k, v in errs.items()})
    meta = dict(model=model, suite=suite, size=size, fold=fold, pca=pca_src, kref=kref, sig_m=sig_m.tolist(),
                sig_r=sig_r.tolist(), diag=diag, wall_s=time.time() - t0, schemes=list(run_schemes) + list(pooled), mode=mode)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, meta_json=np.array(__import__("json").dumps(meta)), **arrays)
    tmp.replace(out_path)
    return str(out_path), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", default="pi05:l10:50,groot:l10:50,pi05:l10:500,groot:l10:500",
                    help="comma list model:suite:size[:pca]  (pca auto|stored|subset)")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    jobs = []
    for spec in a.jobs.split(","):
        parts = spec.split(":")
        m, s, n = parts[0], parts[1], int(parts[2])
        pca = parts[3] if len(parts) > 3 else "auto"
        mode = parts[4] if len(parts) > 4 else "main"
        for f in map(int, a.folds.split(",")):
            jobs.append((m, s, n, f, pca, mode))
    # biggest first
    jobs.sort(key=lambda j: -j[2])
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for path, dt in pool.imap_unordered(run_job, jobs):
            print(f"done {path} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
