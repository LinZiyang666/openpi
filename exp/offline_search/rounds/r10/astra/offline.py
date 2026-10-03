"""Episode-fold metric/head validation and held-out-episode distance calibration."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time
import numpy as np

from .boundary import HERE, install
from exp.offline_search.rounds.r10 import data, train
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r09.recipe.recipe import _predict

REVISION = 'astra_distance_v1_fixed_pca_nested_metric_cv'
BINS = [0., .75, 1., 1.25, 1.5, 2., 3., float('inf')]
CANDIDATES = [dict(peak=b, plateau=p, cutoff=c)
              for b in (.25, .35, .5) for c in (1.25, 1.5, 2.) for p in (.5, .75)]


def now():
    return datetime.now(timezone.utc).isoformat()


def weights(ep):
    _, inv, counts = np.unique(ep, return_inverse=True, return_counts=True)
    return 1. / counts[inv]


def median(x, ep):
    x = np.asarray(x)
    w = weights(ep)
    order = np.argsort(x, kind='stable')
    return float(x[order[np.searchsorted(np.cumsum(w[order]), w.sum() / 2)]])


def strength(ratio, step, rule):
    r = np.asarray(ratio, np.float64)
    a = rule['peak'] * np.clip((rule['cutoff'] - r) / (rule['cutoff'] - rule['plateau']), 0, 1)
    return np.where((np.asarray(step) == 0) | ~np.isfinite(r), 0., a)


def load(model, suite, size):
    lib = data.SubsetLibrary(data.STORE, f'{model}_{suite}', size)
    path = data.HERE / 'artifacts' / f'r10_{model}_{suite}_{size}_G.pkl'
    with path.open('rb') as f:
        base = pickle.load(f)['method'].inner.base
    assert base.features == 'joint' and base.state_scale == 1 and base.codes == 0
    assert base.norm_cap == base.hyst == 0
    ps = []
    for cam in (0, 1):
        # Verified fresh B-subset PCA only; never the r01 or R8 derived stores.
        with np.load(data.HERE / 'pca' / f'{model}_{suite}' / str(size) / f'v{cam}.npz') as z:
            assert str(z['rows_sha256']) == data.sha(data.HERE / 'subsets' / f'{model}_{suite}_{size}.npy')
            np.testing.assert_array_equal(z['basis'].T, getattr(base, f'B{cam}T'))
            ps.append(np.array(z['proj'], np.float32))
    P = np.concatenate(ps, 1)
    spec = json.loads((data.HERE / 'subsets' / f'{model}_{suite}_{size}.json').read_text())
    ef = {int(e): i % 5 for es in spec['episode_ids_by_task'].values() for i, e in enumerate(es)}
    return dict(P=P, X=np.concatenate([P, lib.rs[:, :8]], 1).astype(np.float32),
                act=np.asarray(lib.action[:, :10, :7], np.float32), ep=np.array(lib.episode),
                step=np.array(lib.step), task=np.array(lib.task_id), row=lib.rows,
                fold=np.asarray([ef[int(e)] for e in lib.episode]), kref=base.kref,
                model=model, suite=suite, size=size, source_fit_sha256=data.sha(path))


class Metric:
    """Same full-rank AWM metric; float32 codes and query transforms as deployed."""
    def __init__(self, X, actions, ep, step, sig, early=True):
        X = X.astype(np.float64)
        heads = (actions[:, :5].astype(np.float64) / sig.astype(np.float64)).reshape(len(X), 35)
        mean, std, W = awm.fit_metric(X, heads, ep, nn=3, lam=.1, rank=0)
        self.Z = (((X - mean) / std) @ W).astype(np.float32)
        self.W = (W / std[:, None]).astype(np.float32)
        self.shift = ((mean / std) @ W).astype(np.float32)
        self.z2 = np.sum(self.Z.astype(np.float64) ** 2, 1).astype(np.float32)
        if early:
            mask = step <= 2
            m, s, w = awm.fit_metric(X[mask], heads[mask], ep[mask], nn=3, lam=.1, rank=0)
            # Use the deployed affine early code representation, not a new metric.
            A = np.linalg.inv(W) @ (np.diag(std / s) @ w)
            b = ((mean - m) / s) @ w
            Y = self.Z.astype(np.float64) @ A
            self.A0 = A.astype(np.float32)
            self.W0 = (w / s[:, None]).astype(np.float32)
            self.c0 = ((m / s) @ w + b).astype(np.float32)
            self.n20 = np.sum(Y * Y, 1).astype(np.float32)

    def distances(self, X, step, early=True):
        z = X @ self.W - self.shift
        D = self.z2[None] - 2 * (z @ self.Z.T) + np.sum(z * z, 1)[:, None]
        if early and (step == 0).any():
            first = step == 0
            y = X[first] @ self.W0 - self.c0
            D[first] = self.n20[None] - 2 * ((y @ self.A0.T) @ self.Z.T) + np.sum(y * y, 1)[:, None]
        return np.sqrt(np.maximum(D, 0.))


def serve(C, qr, cr, metric, *, synth=True):
    nearest, chunks = [], []
    for lo in range(0, len(qr), 256):
        q = qr[lo:lo + 256]
        D = metric.distances(C['X'][q], C['step'][q], early=synth)
        D[C['ep'][q, None] == C['ep'][cr][None]] = np.inf
        assert np.isfinite(D).any(1).all()
        # Stable full sort has exact candidate-ID tie breaking even at the k boundary.
        idx = np.argsort(D, axis=1, kind='stable')[:, :16]
        dk = np.take_along_axis(D, idx, 1).astype(np.float64)
        assert np.isfinite(dk).all()
        nearest.append(dk[:, 0])
        if synth:
            w = awm._kernel_w(dk - dk[:, :1], C['kref'])
            w = (w / w.sum(1, keepdims=True)).astype(np.float32)
            chunks.append(np.einsum('qk,qktc->qtc', w, C['act'][cr[idx]], optimize=False))
    return np.concatenate(nearest), np.concatenate(chunks) if synth else None


def features(C, rows, chunk, sig):
    return np.concatenate([C['P'][rows], C['X'][rows, 128:],
        (chunk / sig).reshape(len(rows), 70),
        (np.minimum(C['step'][rows], 120) / 120.)[:, None],
        np.eye(10, dtype=np.float32)[C['task'][rows]]], 1).astype(np.float32)


def sigma(C, rows):
    return np.maximum(C['act'][rows, :5].std(axis=(0, 1)), 1e-6).astype(np.float32)


def calibration(C, train_rows):
    """Four inner folds: every calibration query episode excluded from metric fit."""
    parts, episodes = [], []
    inner = np.full(len(C['ep']), -1, np.int64)
    for task in range(10):
        es = np.unique(C['ep'][train_rows][C['task'][train_rows] == task])
        es = np.random.default_rng(data.SEED + task).permutation(es)
        for i, e in enumerate(es):
            inner[(C['ep'] == e)] = i % 4
    for f in range(4):
        tr = train_rows[inner[train_rows] != f]
        va = train_rows[inner[train_rows] == f]
        sig = sigma(C, tr)
        for task in range(10):
            cr = tr[C['task'][tr] == task]
            qr = va[(C['task'][va] == task) & (C['step'][va] > 0)]
            assert not np.intersect1d(C['ep'][qr], C['ep'][cr]).size
            metric = Metric(C['X'][cr], C['act'][cr], C['ep'][cr], C['step'][cr], sig, early=False)
            d, _ = serve(C, qr, cr, metric, synth=False)
            parts.append(d); episodes.append(C['ep'][qr])
    return median(np.concatenate(parts), np.concatenate(episodes))


def run_cell(model, suite, size):
    started = time.monotonic()
    dest = HERE / 'cv' / f'{model}_{suite}_{size}'
    dest.mkdir(parents=True, exist_ok=True)
    C = load(model, suite, size)
    out = {k: [] for k in ('ep', 'row', 'step', 'task', 'fold', 'd1', 'scale', 'm0', 'yp', 'pp')}
    folds = []
    for f in range(5):
        tr = np.flatnonzero(C['fold'] != f)
        va = np.flatnonzero(C['fold'] == f)
        assert not np.intersect1d(C['ep'][tr], C['ep'][va]).size
        scale = calibration(C, tr)
        sig = sigma(C, tr)
        for task in range(10):
            cr = tr[C['task'][tr] == task]
            qr = va[C['task'][va] == task]
            met = Metric(C['X'][cr], C['act'][cr], C['ep'][cr], C['step'][cr], sig)
            _, synth_tr = serve(C, cr, cr, met)
            d, synth_va = serve(C, qr, cr, met)
            ytr = ((C['act'][cr, :, :6] - synth_tr[:, :, :6]) / sig[:6]).reshape(len(cr), 60)
            head = train.fit_head(features(C, cr, synth_tr, sig), ytr, train.anchor_weights(cr, C['ep'][cr]))
            y = ((C['act'][qr, :, :6] - synth_va[:, :, :6]) / sig[:6]).reshape(len(qr), 60)
            pred = _predict(head, features(C, qr, synth_va, sig))
            y, pred = y.astype(np.float64), pred.astype(np.float64)
            for key in ('ep', 'row', 'step', 'task', 'fold'):
                out[key].append(C[key][qr])
            for key, val in dict(d1=d, scale=np.full(len(qr), scale),
                    m0=np.mean(y * y, 1), yp=np.mean(y * pred, 1), pp=np.mean(pred * pred, 1)).items():
                out[key].append(val)
        folds.append(dict(fold=f, train_episodes=np.unique(C['ep'][tr]).tolist(),
                          validation_episodes=np.unique(C['ep'][va]).tolist(), distance_scale=scale,
                          sigma=sig.tolist(), train_rows=len(tr), validation_rows=len(va)))
        print(f'CV {model}_{suite}_{size} fold={f} scale={scale:.4f}', flush=True)
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    assert len(np.unique(arrays['row'])) == len(C['row']) == len(arrays['row'])
    assert np.array_equal(np.sort(arrays['row']), C['row'])
    np.savez_compressed(dest / 'rows.npz', **arrays)
    keep = arrays['step'] > 0
    final_scale = median(arrays['d1'][keep], arrays['ep'][keep])
    data.write_json(dest / 'folds.json', dict(revision=REVISION, folds=folds,
        source_fit_sha256=C['source_fit_sha256'], final_distance_scale=final_scale,
        row_count=len(C['row']), episode_count=len(np.unique(C['ep'])), seconds=time.monotonic() - started,
        fit_pool='B', fixed_PCA=True, metric_and_head_exclude_heldout=True))
    print(f'DONE {model}_{suite}_{size} seconds={time.monotonic()-started:.1f}', flush=True)


def mean(values, ep):
    return float(np.average(values, weights=weights(ep)))


def compare(a, rule, mask=None):
    if mask is None:
        mask = np.ones(len(a['ep']), bool)
    if not mask.any():
        return None
    r = a['d1'] / a['scale']
    b = strength(r, a['step'], rule)
    vals = dict(none=a['m0'], GC_loeo=a['m0'] - a['yp'] + .25 * a['pp'],
                GC_dist=a['m0'] - 2 * b * a['yp'] + b * b * a['pp'])
    result = {k: mean(v[mask], a['ep'][mask]) for k, v in vals.items()}
    return dict(rows=int(mask.sum()), episodes=len(np.unique(a['ep'][mask])), **result,
        loeo_change_pct=100 * (result['GC_loeo'] / result['none'] - 1),
        dist_change_pct=100 * (result['GC_dist'] / result['none'] - 1),
        dist_vs_loeo_pct=100 * (result['GC_dist'] / result['GC_loeo'] - 1),
        mean_strength=mean(b[mask], a['ep'][mask]),
        zero_fraction=mean((b[mask] == 0).astype(float), a['ep'][mask]))


def select():
    datasets = []
    for model, suite in data.CELLS:
        for size in data.SIZES:
            key = f'{model}_{suite}_{size}'
            path = HERE / 'cv' / key
            with np.load(path / 'rows.npz') as z:
                a = {k: np.array(z[k]) for k in z.files}
            info = json.loads((path / 'folds.json').read_text())
            datasets.append((key, a, info))
    candidates = []
    for rule in CANDIDATES:
        scores = [compare(a, rule) for _, a, _ in datasets]
        objective = float(np.mean([s['GC_dist'] / s['none'] for s in scores]))
        candidates.append(dict(rule=rule, objective=objective))
    winner = min(candidates, key=lambda c: (c['objective'], c['rule']['peak'], c['rule']['cutoff'], c['rule']['plateau']))
    rule = winner['rule']
    summaries = []
    for key, a, info in datasets:
        ratio = a['d1'] / a['scale']
        bins = [dict(bin='step0', **compare(a, rule, a['step'] == 0))]
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            stats = compare(a, rule, (a['step'] > 0) & (ratio >= lo) & (ratio < hi))
            if stats:
                bins.append(dict(bin=f'[{lo:g},{hi:g})', **stats))
        summary = dict(cell_size=key, distance_scale=info['final_distance_scale'],
                       **compare(a, rule), by_distance=bins)
        summaries.append(summary)
        data.write_json(HERE / 'calibration' / f'{key}.json', dict(revision=REVISION, rule=rule,
            scale=info['final_distance_scale'], source_fit_sha256=info['source_fit_sha256'], fit_pool='B',
            reference='episode-balanced median of non-step0 outer-held-out d1; training-fold metric',
            fixed_PCA=True, cv_sha256=data.sha(HERE / 'cv' / key / 'rows.npz')))
    data.write_json(HERE / 'selection.json', dict(timestamp_utc=now(), revision=REVISION,
        protocol_sha256=data.sha(HERE / 'PROTOCOL.md'), candidates=candidates, selected=winner,
        selection='same tuning CV used for comparison; not independent test performance', summaries=summaries))
    cols = ['cell_size', 'distance_scale', 'none', 'GC_loeo', 'GC_dist', 'loeo_change_pct',
            'dist_change_pct', 'dist_vs_loeo_pct', 'mean_strength', 'zero_fraction']
    import csv
    with (HERE / 'offline_cv.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(summaries)
    with (HERE / 'distance_bins.csv').open('w') as f:
        rows = [dict(cell_size=s['cell_size'], **b) for s in summaries for b in s['by_distance']]
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(winner), flush=True)


def main():
    install()
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['cv', 'select'])
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--model'); p.add_argument('--suite'); p.add_argument('--size', type=int)
    args = p.parse_args()
    if args.action == 'select':
        select(); return
    jobs = [(m, s, n) for m, s in data.CELLS for n in data.SIZES
            if (not args.model or m == args.model) and (not args.suite or s == args.suite)
            and (not args.size or n == args.size)]
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_cell, *job) for job in jobs]
        for f in futures:
            f.result()


if __name__ == '__main__':
    main()
