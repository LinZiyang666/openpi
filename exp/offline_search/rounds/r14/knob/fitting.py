"""Full-row B-only production fits; serving never imports this module."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r02.g3_recovery import g3_core as core
from exp.offline_search.rounds.r10 import train
from exp.offline_search.rounds.r10.recipe import fitting as r10
from exp.offline_search.rounds.r10.recipe.recipe import _CorrectedCache
from exp.offline_search.rounds.r10.astra.offline import median
from .recipe import _PartitionCache, SWITCHES


def metric_fit(X, H, ep, block=256):
    """Original full-rank AWM equation, bounded action-neighbor memory."""
    X, H = np.asarray(X, np.float64), np.asarray(H, np.float64)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mean) / std
    h2 = (H * H).sum(1)
    gram, count = np.zeros((X.shape[1], X.shape[1])), 0
    for lo in range(0, len(X), block):
        hi = min(lo + block, len(X))
        D = h2[lo:hi, None] - 2 * H[lo:hi] @ H.T + h2[None]
        D[ep[lo:hi, None] == ep[None]] = np.inf
        k = min(3, len(X) - 1)
        ix = np.argpartition(D, k, axis=1)[:, :k]
        ok = np.isfinite(np.take_along_axis(D, ix, 1))
        dif = (Xn[lo:hi, None] - Xn[ix])[ok]
        gram += dif.T @ dif
        count += len(dif)
    sw = gram / max(count, 1)
    W = np.linalg.cholesky(np.linalg.inv(sw + .1 * np.trace(sw) / len(sw) * np.eye(len(sw))))
    return mean, std, W


def equations(C, rows, sigma, pooled):
    X = C['X'][rows].astype(np.float64)
    H = (C['act'][rows, :5].astype(np.float64) / sigma.astype(np.float64)).reshape(len(rows), 35)
    fitter = metric_fit if pooled else awm.fit_metric
    main = fitter(X, H, C['ep'][rows])
    early = C['step'][rows] <= 2
    return main, fitter(X[early], H[early], C['ep'][rows][early])


def table(C, rows, sigma, main, early, source):
    """Same main and affine early representation as AWM.fit, including casts."""
    T = copy.copy(source)
    T.rows = np.asarray(rows, np.int64)
    X = C['X'][rows].astype(np.float64)
    mean, std, W = main
    m0, s0, W0 = early
    T.Z = np.ascontiguousarray(((X - mean) / std) @ W, np.float32)
    Zr = T.Z.astype(np.float64)
    T.z2 = (Zr * Zr).sum(1).astype(np.float32)
    T.Wf = np.ascontiguousarray(W / std[:, None], np.float32)
    T.shift = ((mean / std) @ W).astype(np.float32)
    A = np.linalg.inv(W) @ (np.diag(std / s0) @ W0)
    b = ((mean - m0) / s0) @ W0
    Y = Zr @ A
    T.A0 = np.ascontiguousarray(A, np.float32)
    T.n20 = (Y * Y).sum(1).astype(np.float32)
    T.W0f = np.ascontiguousarray(W0 / s0[:, None], np.float32)
    T.c0 = ((m0 / s0) @ W0 + b).astype(np.float32)
    T.Z0 = T.As0 = None
    H = (C['act'][rows, :5].astype(np.float64) / sigma.astype(np.float64)).reshape(len(rows), 35)
    T.HD = np.ascontiguousarray(H, np.float32)
    T.h2 = (H * H).sum(1).astype(np.float32)
    RS = X[:, 128:]
    T.RS = np.ascontiguousarray(RS, np.float32)
    T.rs2 = (RS * RS).sum(1).astype(np.float32)
    # Diagnostic visual reference follows the query task unless guard is pooled.
    for f in (0, 1):
        V = X[:, f * 64:(f + 1) * 64] - getattr(source, f'Vm{f}')
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        setattr(T, f'V{f}', np.ascontiguousarray(V, np.float32))
    for v in vars(T).values():
        if isinstance(v, np.ndarray):
            v.flags.writeable = False
    return T


def geometry(C, base, scope, metric, *, fit_rows=None, sigma=None):
    fit_rows = np.arange(len(C['ep'])) if fit_rows is None else fit_rows
    sigma = base.sig if sigma is None else sigma
    pooled = equations(C, fit_rows, sigma, True) if metric == 'pool' else None
    out = {}
    for task in range(10):
        own = fit_rows[C['task'][fit_rows] == task]
        donors = fit_rows if scope == 'pool' else own
        fits = pooled if pooled is not None else equations(C, own, sigma, False)
        out[task] = table(C, donors, sigma, *fits, base.tasks[task])
    return out


def distances(T, X, step):
    z = X @ T.Wf - T.shift
    D = T.z2[None] - 2 * (z @ T.Z.T) + (z * z).sum(1)[:, None]
    first = step == 0
    if first.any():
        y = X[first] @ T.W0f - T.c0
        D[first] = T.n20[None] - 2 * ((y @ T.A0.T) @ T.Z.T) + (y * y).sum(1)[:, None]
    return np.sqrt(np.maximum(D, 0.))


def calibrate(C, base, scope, metric):
    parts, eps, folds = [], [], []
    for fold in range(5):
        tr, va = np.flatnonzero(C['fold'] != fold), np.flatnonzero(C['fold'] == fold)
        sigma = np.maximum(C['act'][tr, :5].std((0, 1)), 1e-6).astype(np.float32)
        tables = geometry(C, base, scope, metric, fit_rows=tr, sigma=sigma)
        assert not np.intersect1d(C['ep'][tr], C['ep'][va]).size
        for task, T in tables.items():
            qr = va[C['task'][va] == task]
            # Preserve R10's complete task-query batches, including step zero.
            for lo in range(0, len(qr), 256):
                q = qr[lo:lo + 256]
                D = distances(T, C['X'][q], C['step'][q])
                keep = C['step'][q] > 0
                parts.append(D.min(1)[keep]); eps.append(C['ep'][q][keep])
        folds.append(dict(fold=fold, train_episodes=np.unique(C['ep'][tr]).tolist(),
                          heldout_episodes=np.unique(C['ep'][va]).tolist(),
                          metric_scope=metric, donor_scope=scope, sigma=sigma.tolist(),
                          rows=len(tr), heldout_rows=len(va)))
        print('DISTANCE', scope, metric, fold, flush=True)
    scale = median(np.concatenate(parts), np.concatenate(eps))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid GC_dist calibration')
    return scale, folds


def full_heads(lib, base, C, scope, metric):
    """Every R10 query row, full H x 32 synthesis, own episode excluded."""
    feature = _CorrectedCache.__new__(_CorrectedCache)
    vars(feature).update(vars(base))
    feature.sig_head = base.sig
    by_task = {}
    for task in range(10):
        X, Y, rows, episodes = [], [], [], []
        for i, q in train.queries(lib, C, task):
            T = base.tasks[task]
            donors, ds, xv, rs8 = train.neighbors(base, q, int(lib.episode[i]), cap=16)
            base.reset(q.episode)
            # Bypass the corrected subclass: training proposals are uncorrected.
            cached = awm.AWM.os_synth(base, q, donors, awm._kernel_w(ds.astype(np.float64) - float(ds[0]), base.kref))
            X.append(feature._features(q, xv, rs8, cached)[0])
            Y.append(((np.asarray(lib.action[i, :10, :6], np.float32) - cached[:10, :6]) / base.sig[:6]).ravel())
            rows.append(i); episodes.append(int(lib.episode[i]))
        X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
        rows, episodes = np.asarray(rows), np.asarray(episodes)
        by_task[task] = dict(X=X, Y=Y, rows=rows, episodes=episodes)
        print('QUERIES', scope, metric, task, len(rows), flush=True)
    return by_task


def fit_heads(dataset, sharing):
    groups = dataset.items() if sharing == 'task' else [('pool', {
        k:np.concatenate([d[k] for d in dataset.values()]) for k in ('X', 'Y', 'rows', 'episodes')})]
    heads, counts = {}, {}
    for task, d in groups:
        X = d['X'][:, :-10] if sharing == 'pool' else d['X']
        weights = train.anchor_weights(d['rows'], d['episodes'])
        h = train.fit_head(X, d['Y'], weights)
        for v in h.values():
            v.flags.writeable = False
        heads[str(task)] = h
        counts[str(task)] = dict(rows=len(X), episodes=len(np.unique(d['episodes'])),
                                weight_sum=float(weights.sum()), features=X.shape[1],
                                query_rows_sha256=hashlib.sha256(d['rows'].tobytes()).hexdigest(),
                                exclude_query_episode=True, query_regime='early_step0_stale_later')
    if sharing == 'pool':
        heads = {str(t):heads['pool'] for t in range(10)}
    return heads, counts


def continuity(C, sigma):
    H = (C['act'][:, :5].astype(np.float64) / sigma.astype(np.float64)).reshape(-1, 35)
    tail = (C['act'][:, 5:10].astype(np.float64) / sigma.astype(np.float64)).reshape(-1, 35)
    return float(np.median(awm._masked_min(tail, H, C['ep'], C['ep'], rms_dim=35))) + 1e-6


def pool_guard(judge, lib, C):
    # Raw means use the inherited accumulation/cast convention.
    alltask = np.zeros(lib.L, dtype=np.int32)
    m0 = core.task_means(lib.key_v0, alltask, [0])[0]
    m1 = core.task_means(lib.key_v1, alltask, [0])[0]
    judge.M0, judge.M1 = {t:m0 for t in range(10)}, {t:m1 for t in range(10)}
    ep_rows = core.episode_rows(judge.C)
    vc = core.consecutive_vcos(lib.key_v0, lib.key_v1, judge.M0, judge.M1, judge.C, ep_rows)
    judge.c_thr = float(np.percentile(vc[np.isfinite(vc)], judge.c_pct))
    length = float(np.median([judge.C.ep_len[r[0]] for r in ep_rows.values()]))
    judge.med_len = {t:length for t in range(10)}
    base = judge.base
    scale = np.maximum(base.blind_rs.std(0), .05).astype(np.float32)
    edges = np.flatnonzero(base.blind_next >= 0)
    motion = np.sqrt(np.mean(((base.blind_rs[base.blind_next[edges]] - base.blind_rs[edges]) / scale) ** 2, 1))
    base.state_scale_by_task = {t:scale for t in range(10)}
    base.motion10 = {t:float(np.percentile(motion, 10)) for t in range(10)}
    s_d = float(np.median(awm._masked_min(C['X'][:, 128:].astype(float), C['X'][:, 128:].astype(float), C['ep'], C['ep']))) + 1e-6
    P = C['X'][:, :128].astype(float)
    for T in base.tasks.values():
        T.s_d, T.med_len = s_d, length
        for f in (0, 1):
            mean = P[:, f * 64:(f + 1) * 64].mean(0)
            V = P[T.rows, f * 64:(f + 1) * 64] - mean
            V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
            setattr(T, f'Vm{f}', mean.astype(np.float32))
            setattr(T, f'V{f}', np.ascontiguousarray(V, np.float32))
    scale.flags.writeable = False


def fit_recipe(recipe, ctx):
    """API fitting entry: fresh selected PCA/R10, then independent partitions."""
    r10.fit_recipe(recipe, ctx)
    from exp.offline_search.rounds.r10.recipe.fitting import _Library, library_boundary
    libdir = Path(ctx.root) / 'library' / ctx.lib_key / recipe.library
    with library_boundary(ctx.root, libdir):
        lib = _Library(ctx.root, ctx.lib_key, recipe.library, recipe.episode_subset)
        base = recipe.inner.base
        P = np.concatenate([awm.project(lib.key_v0, base.mu0, base.B0T.T),
                            awm.project(lib.key_v1, base.mu1, base.B1T.T)], 1)
        ef = {int(e):i % 5 for es in lib.chosen.values() for i,e in enumerate(es)}
        C = dict(X=np.c_[P, lib.rs[:, :8]].astype(np.float32), act=np.asarray(lib.action[:, :10, :7], np.float32),
                 ep=lib.episode, step=lib.step, task=lib.task_id,
                 fold=np.asarray([ef[int(e)] for e in lib.episode]))
        scope, metric = recipe.r14_scope, recipe.r14_metric
        changes = scope == 'pool' or metric == 'pool'
        if changes:
            base.tasks = geometry(C, base, scope, metric)
        dataset = full_heads(lib, base, recipe.inner.C, scope, metric) if changes or recipe.r14_head == 'pool' else None
        if dataset is not None:
            base.heads, counts = fit_heads(dataset, recipe.r14_head)
            recipe.fit_info['head_counts'] = counts
            base.head_meta=dict(base.head_meta,task_onehot=recipe.r14_head=='task',per_task=recipe.r14_head=='task')
        if changes:
            base.distance_scale, folds = calibrate(C, base, scope, metric)
            recipe.fit_info.update(distance_scale=base.distance_scale, distance_folds=folds)
        if recipe.r14_continuity == 'pool':
            value = continuity(C, base.sig)
            for T in base.tasks.values():
                T.s_c = value
        if recipe.r14_guard == 'pool':
            pool_guard(recipe.inner, lib, C)
        if any(getattr(recipe, 'r14_' + s) == 'pool' for s in SWITCHES):
            base.__class__ = _PartitionCache
            base.r14_head = recipe.r14_head
        recipe.reset(api.EpisodeView('r14-build', '', 0, 0, 0, 0))
        recipe.fit_info['r14_switches'] = {s:getattr(recipe, 'r14_' + s) for s in SWITCHES}
    if recipe.knob_method != 'off':
        import json
        from exp.offline_search.rounds.r10.data import HERE as R10
        from exp.offline_search.rounds.r11.knob.calibration import calibrate as r11_calibrate
        from exp.offline_search.rounds.r14.impl.common import ASTRA, sha
        expected=json.loads((R10/'subsets'/f'{ctx.model}_{ctx.suite}_{recipe.size}.json').read_text())['episode_ids_by_task']
        if recipe.fit_info['episode_ids_by_task']!=expected:
            raise ValueError('R14 allocation requires the frozen nested subset')
        cell=f'{ctx.model}_{ctx.suite}_{recipe.size}'
        fit=next(r for r in json.loads((ASTRA/'allocation_fits.json').read_text())
                 if r['cell']==cell and r['target']==recipe.target)
        cfg=r11_calibrate(ctx.model,ctx.suite,recipe.size,'periodic',recipe.target)
        assert cfg['setting']==fit['reference_K']
        if recipe.knob_method=='task_periodic':
            cfg=dict(K=fit['K'],reference_K=fit['reference_K'],priority=fit['priority']['z'],
                     gain=fit['gain'],bias=fit['bias'],prior_episodes=fit['prior_episodes'])
        recipe.configure(cfg,ctx.model)
        recipe.r14_fit=dict(frozen_allocation_sha256=sha(ASTRA/'allocation_fits.json'),cell=cell)
