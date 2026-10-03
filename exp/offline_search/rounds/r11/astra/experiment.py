"""Episode-held-out state signals and strictly nested pooled error predictor.

No A data, simulations, models, policy calls, GPU, or production mutations.
The fixed B-library PCA and R10 outer residual labels are reused and hashed.
Outer validation episodes never enter inner predictor features/targets/metric fits.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import time
import numpy as np

from exp.offline_search.rounds.r11.astra.boundary import HERE, READS, dump, install, sha

CELLS = [(m, s, n) for m in ('pi05', 'groot') for s, sizes in
         (('l10', (50, 200, 500)), ('spatial', (50,))) for n in sizes]
FEATURES = ['log_distance', 'log_motion_disagreement', 'log_motion_energy',
            'donor_grip_transition', 'grip_disagreement', 'chunk_grip_transition',
            'relative_distance_spread', 'effective_neighbours']


def episode_weights(ep):
    _, inv, counts = np.unique(ep, return_inverse=True, return_counts=True)
    w = 1. / counts[inv]
    return w * len(w) / w.sum()


def retrieve(C, qr, cr, metric, sig):
    from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
    feats, chunks, nearest, top_rows, progress = [], [], [], [], []
    for lo in range(0, len(qr), 256):
        q = qr[lo:lo + 256]
        D = metric.distances(C['X'][q], C['step'][q])
        D[C['ep'][q, None] == C['ep'][cr][None]] = np.inf
        ids = np.argsort(D, axis=1, kind='stable')[:, :16]
        ds = np.take_along_axis(D, ids, axis=1).astype(np.float64)
        assert np.isfinite(ds).all()
        donors = cr[ids]
        assert not np.any(C['ep'][q, None] == C['ep'][donors])
        w = _kernel_w(ds - ds[:, :1], C['kref'])
        w = (w / w.sum(1, keepdims=True)).astype(np.float32)
        a = C['act'][donors]
        chunk = np.einsum('qk,qktc->qtc', w, a, optimize=False)
        motion_var = np.einsum('qk,qk->q', w,
            np.mean(((a[:, :, :, :6] - chunk[:, None, :, :6]) / sig[:6]) ** 2, axis=(2, 3)))
        energy = np.mean((chunk[:, :, :6] / sig[:6]) ** 2, axis=(1, 2))
        g = a[:, :, :, 6] >= 0
        transition = np.einsum('qk,qk->q', w, (g != g[:, :, :1]).any(2))
        vote = np.einsum('qk,qkt->qt', w, g.astype(np.float32))
        disagreement = np.mean(4 * vote * (1 - vote), axis=1)
        cg = chunk[:, :, 6] >= 0
        chunk_transition = (cg != cg[:, :1]).any(1).astype(float)
        gap = np.log1p((ds[:, -1] - ds[:, 0]) / np.maximum(ds[:, 0], 1e-6))
        effective = 1 / np.sum(w * w, 1) / 16
        feats.append(np.column_stack((np.log1p(ds[:, 0]), np.log1p(motion_var),
            np.log1p(energy), transition, disagreement, chunk_transition, gap, effective)))
        chunks.append(chunk)
        nearest.append(ds[:, 0])
        top_rows.append(donors[:, 0])
        progress.append(C['step'][donors[:, 0]])
    return dict(X=np.concatenate(feats), chunk=np.concatenate(chunks),
                d1=np.concatenate(nearest), top=np.concatenate(top_rows),
                progress=np.concatenate(progress))


def heldout_signals(C, train_rows, query_rows):
    from exp.offline_search.rounds.r10.astra.offline import Metric, sigma
    sig = sigma(C, train_rows)
    X = np.empty((len(query_rows), len(FEATURES)), np.float64)
    y = np.empty(len(query_rows), np.float64)
    d1 = np.empty(len(query_rows), np.float64)
    grip_error = np.empty(len(query_rows), np.float64)
    top = np.empty(len(query_rows), np.int64)
    prog = np.empty(len(query_rows), np.float64)
    for task in range(10):
        cr = train_rows[C['task'][train_rows] == task]
        loc = np.flatnonzero(C['task'][query_rows] == task)
        qr = query_rows[loc]
        if not len(qr):
            continue
        assert not np.intersect1d(C['ep'][cr], C['ep'][qr]).size
        met = Metric(C['X'][cr], C['act'][cr], C['ep'][cr], C['step'][cr], sig)
        r = retrieve(C, qr, cr, met, sig)
        X[loc], d1[loc], top[loc], prog[loc] = r['X'], r['d1'], r['top'], r['progress']
        y[loc] = np.mean(((C['act'][qr, :, :6] - r['chunk'][:, :, :6]) / sig[:6]) ** 2, axis=(1, 2))
        grip_error[loc] = np.mean((C['act'][qr, :, 6] >= 0) != (r['chunk'][:, :, 6] >= 0), axis=1)
    return dict(X=X, m0=y, d1=d1, grip_error=grip_error, top=top, progress=prog)


def predictor_fit(X, y, ep):
    # One cell-wide model, no task identity, no task parameters. Fixed regularizer.
    w = episode_weights(ep)
    mu = np.average(X, weights=w, axis=0)
    sd = np.maximum(np.sqrt(np.average((X - mu) ** 2, weights=w, axis=0)), .05)
    z = np.clip((X - mu) / sd, -8, 8)
    F = np.column_stack((np.ones(len(z)), z))
    reg = np.eye(F.shape[1]) * (.1 * len(X))
    reg[0, 0] = 0
    coef = np.linalg.solve(F.T @ (w[:, None] * F) + reg, F.T @ (w * np.log1p(y)))
    return dict(mean=mu, std=sd, coef=coef)


def predictor_score(head, X):
    z = np.clip((X - head['mean']) / head['std'], -8, 8)
    return np.column_stack((np.ones(len(z)), z)) @ head['coef']


def run_cell(model, suite, size):
    from exp.offline_search.rounds.r10.astra.offline import load, strength
    from exp.offline_search.rounds.r10.data import HERE as R10
    start = time.monotonic()
    tag = f'{model}_{suite}_{size}'
    dest = HERE / 'data' / tag
    dest.mkdir(exist_ok=True, parents=True)
    C = load(model, suite, size)
    oldpath = R10 / 'astra' / 'cv' / tag / 'rows.npz'
    with np.load(oldpath) as z:
        old = {k: np.asarray(z[k]) for k in z.files}
    old = {k: v[np.argsort(old['row'])] for k, v in old.items()}
    np.testing.assert_array_equal(old['row'], C['row'])
    for k in ('ep', 'step', 'task', 'fold'):
        np.testing.assert_array_equal(old[k], C[k])
    residual_strength = strength(old['d1'] / old['scale'], old['step'],
                                 dict(peak=.5, plateau=.75, cutoff=2.))
    risk = np.maximum(0, old['m0'] - 2 * residual_strength * old['yp'] + residual_strength**2 * old['pp'])
    out = {k: C[k] for k in ('row', 'ep', 'step', 'task', 'fold')}
    out.update(risk=risk, m0=old['m0'], distance=old['d1'], correction_strength=residual_strength)
    X = np.empty((len(risk), len(FEATURES)))
    errscore = np.empty(len(risk))
    ge = np.empty(len(risk))
    prog = np.empty(len(risk))
    top = np.empty(len(risk), np.int64)
    audits = []
    for f in range(5):
        tr, va = np.flatnonzero(C['fold'] != f), np.flatnonzero(C['fold'] == f)
        sigs = heldout_signals(C, tr, va)
        np.testing.assert_allclose(sigs['d1'], old['d1'][va], atol=2e-4, rtol=1e-5)
        np.testing.assert_allclose(sigs['m0'], old['m0'][va], atol=3e-5, rtol=1e-4)
        X[va], ge[va], prog[va], top[va] = sigs['X'], sigs['grip_error'], sigs['progress'], sigs['top']
        innerX, innerY, innerEp = [], [], []
        # Leave one of the other outer episode groups out of metric AND donors.
        for j in range(5):
            if j == f:
                continue
            it = tr[C['fold'][tr] != j]
            iq = tr[C['fold'][tr] == j]
            s = heldout_signals(C, it, iq)
            innerX.append(s['X']); innerY.append(s['m0']); innerEp.append(C['ep'][iq])
            assert not np.intersect1d(C['ep'][va], C['ep'][it]).size
            assert not np.intersect1d(C['ep'][va], C['ep'][iq]).size
        head = predictor_fit(np.concatenate(innerX), np.concatenate(innerY), np.concatenate(innerEp))
        errscore[va] = predictor_score(head, sigs['X'])
        audits.append(dict(outer=f, train_episodes=np.unique(C['ep'][tr]).tolist(),
                           heldout_episodes=np.unique(C['ep'][va]).tolist(),
                           head={k: v.tolist() for k, v in head.items()}))
        print(f'{tag} nested fold {f} complete {time.monotonic()-start:.1f}s', flush=True)
    out.update(X=X, predicted_error=errscore, disagreement=np.expm1(X[:, 1]),
               gripper_transition=X[:, 3], gripper_disagreement=X[:, 4],
               chunk_transition=X[:, 5], grip_error=ge, top=top, progress=prog)
    np.savez_compressed(dest / 'signals.npz', **out)
    final_head = predictor_fit(X, out['m0'], C['ep'])
    dump(dest / 'predictor.json', dict(feature_names=FEATURES,
         **{k: v.tolist() for k, v in final_head.items()}, label='log1p(pre-corrector motion MSE)',
         task_indexed=False, fixed_regularizer=.1, clip_z=8))
    dump(dest / 'provenance.json', dict(timestamp=datetime.now(timezone.utc).isoformat(),
        model=model, suite=suite, size=size, rows=len(risk), episodes=len(np.unique(C['ep'])),
        raw_library=f'/home/weiland/trace_runs/offline_search_store/library/{model}_{suite}/' +
                    ('bpool_cs' if model == 'pi05' else 'bpool_all'),
        source_fit_sha256=C['source_fit_sha256'], r10_cv_sha256=sha(oldpath),
        subset_sha256=sha(R10 / 'subsets' / f'{tag}.npy'),
        signals_sha256=sha(dest / 'signals.npz'), feature_names=FEATURES,
        folds=audits, seconds=time.monotonic()-start,
        caveats=['Fixed selected-library PCA as inherited R10 convention.',
                 'Five whole-episode groups, not exact leave-one-single-episode metric refits.',
                 'Risk predictor target is uncorrected MSE; evaluation risk includes R10 corrector.',
                 'R10 correction attenuation was previously selected on this B-library CV.',
                 'Every library episode, including failures, is retained.']))
    print(f'DONE {tag} rows={len(risk)} seconds={time.monotonic()-start:.1f}', flush=True)


def main():
    install()
    p = argparse.ArgumentParser()
    p.add_argument('--cell')
    p.add_argument('--concurrency', type=int, default=4)
    a = p.parse_args()
    cells = [c for c in CELLS if not a.cell or '_'.join(map(str, c)) == a.cell]
    with ThreadPoolExecutor(a.concurrency) as pool:
        futures = [pool.submit(run_cell, *cell) for cell in cells]
        for f in futures:
            f.result()
    dump(HERE / 'experiment_reads.json', sorted(READS))


if __name__ == '__main__':
    main()
