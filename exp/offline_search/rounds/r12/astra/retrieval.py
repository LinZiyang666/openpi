"""B-library five-fold retrieval screen; no closed-loop SR is inferred here."""
from .common import *
install()
import argparse
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from exp.offline_search.rounds.r02.g1_awm import awm

VARIANTS = ('full', 'wrist', 'agent', 'state', 'wrist_l2', 'marginal',
            'stale2', 'stale4', 'memory2', 'transport2')

def lag(C, n):
    rows = np.arange(len(C['ep']))
    for _ in range(n):
        rows = np.where(C['prev'][rows] >= 0, C['prev'][rows], rows)
    assert np.array_equal(C['ep'][rows], C['ep'])
    return rows

class Metric:
    def __init__(self, X, A, ep, sig, kind='awm'):
        X = X.astype(np.float64)
        if kind == 'awm':
            m, s, w = awm.fit_metric(X, (A[:, :5] / sig).reshape(len(A), -1).astype(np.float64), ep)
        else:
            m, s, w = X.mean(0), X.std(0)+1e-6, np.eye(X.shape[1])
        self.W = (w/s[:, None]).astype(np.float32)
        self.shift = ((m/s)@w).astype(np.float32)
        self.Z = (((X-m)/s)@w).astype(np.float32)
        self.z2 = (self.Z**2).sum(1)

    def dist(self, X):
        z = X@self.W-self.shift
        return np.sqrt(np.maximum((z*z).sum(1)[:, None]+self.z2[None]-2*z@self.Z.T, 0))

def marginal(full, X):
    """Minimize the full squared metric over the unobserved v0 block."""
    obj = Metric.__new__(Metric)
    M = full.W.astype(np.float64) @ full.W.astype(np.float64).T
    S = M[64:, 64:] - M[64:, :64] @ np.linalg.solve(M[:64, :64], M[:64, 64:])
    obj.W = np.linalg.cholesky((S+S.T)/2).astype(np.float32)
    obj.shift = np.zeros(len(S), np.float32)
    obj.Z = X[:, 64:]@obj.W
    obj.z2 = (obj.Z**2).sum(1)
    return obj

def retrieve(metric, Q, qr, cr, C):
    result = {k: [] for k in ('action', 'rows', 'weights', 'distance', 'variance', 'gap', 'effective', 'gripvar')}
    for lo in range(0, len(qr), 192):
        rows = qr[lo:lo+192]
        D = metric.dist(Q[lo:lo+192])
        D[C['ep'][rows, None] == C['ep'][cr][None]] = np.inf
        idx = np.argsort(D, axis=1, kind='stable')[:, :16]
        dk = np.take_along_axis(D, idx, 1).astype(np.float64)
        assert np.isfinite(dk).all()
        w = awm._kernel_w(dk-dk[:, :1], C['kref'])
        w = (w/w.sum(1, keepdims=True)).astype(np.float32)
        a = C['act'][cr[idx]]
        pred = np.einsum('qk,qktc->qtc', w, a, optimize=False)
        g = np.einsum('qk,qkt->qt', w, (a[:, :, :, 6]>=0).astype(np.float32))
        vals = dict(action=pred, rows=cr[idx], weights=w, distance=dk[:, 0],
                    variance=np.einsum('qk,qk->q', w, np.mean(((a[:, :, :, :6]-pred[:, None, :, :6])/C['sigma'][:6])**2, (2, 3))),
                    gap=(dk[:, -1]-dk[:, 0])/np.maximum(dk[:, 0], 1e-6),
                    effective=1/(w*w).sum(1), gripvar=np.mean(4*g*(1-g), 1))
        for k, v in vals.items(): result[k].append(v)
    return {k: np.concatenate(v) for k, v in result.items()}

def run(cell, size):
    start = time.monotonic()
    C, _ = load(cell, size)
    outdir = HERE / 'retrieval' / f'{cell}_{size}'
    outdir.mkdir(parents=True, exist_ok=True)
    P, rs = C['P'], C['rs']
    full = np.c_[P, rs].astype(np.float32)
    past2, past4 = lag(C, 2), lag(C, 4)
    stale2 = np.c_[P[past2, :64], P[:, 64:], rs].astype(np.float32)
    stale4 = np.c_[P[past4, :64], P[:, 64:], rs].astype(np.float32)
    views = dict(full=full, wrist=full[:, 64:], agent=np.c_[P[:, :64], rs], state=rs,
                 wrist_l2=full[:, 64:], memory2=stale2)
    accum = {v: {} for v in VARIANTS}
    meta = {k: [] for k in ('row', 'ep', 'task', 'step', 'fold', 'sigma', 'target', 'past2')}
    for fold in range(5):
        tr = np.flatnonzero(C['fold'] != fold)
        sig = np.maximum(C['act'][tr, :5].std((0, 1)), 1e-6).astype(np.float32)
        C['sigma'] = sig
        for task in range(10):
            cr = tr[C['task'][tr] == task]
            qr = np.flatnonzero((C['fold'] == fold) & (C['task'] == task) & (C['step'] >= 2) & (C['step']%2 == 0))
            assert not np.intersect1d(C['ep'][cr], C['ep'][qr]).size
            results, metrics = {}, {}
            for name, X in views.items():
                metric = Metric(X[cr], C['act'][cr], C['ep'][cr], sig, 'l2' if name == 'wrist_l2' else 'awm')
                metrics[name] = metric
                results[name] = retrieve(metric, X[qr], qr, cr, C)
            results['marginal'] = retrieve(marginal(metrics['full'], full[cr]), full[qr, 64:], qr, cr, C)
            results['stale2'] = retrieve(metrics['full'], stale2[qr], qr, cr, C)
            results['stale4'] = retrieve(metrics['full'], stale4[qr], qr, cr, C)
            r = results['memory2']
            delta = np.einsum('qk,qkd->qd', r['weights'], P[r['rows'], :64]-P[past2[r['rows']], :64])
            transported = stale2[qr].copy()
            transported[:, :64] += delta
            results['transport2'] = retrieve(metrics['full'], transported, qr, cr, C)
            for name, result in results.items():
                a, a0 = result['action'], results['full']['action']
                result['motion_mse'] = np.mean(((a[:, :, :6]-C['act'][qr, :, :6])/sig[:6])**2, (1, 2))
                result['full_delta_mse'] = np.mean(((a[:, :, :6]-a0[:, :, :6])/sig[:6])**2, (1, 2))
                result['grip_error'] = np.mean((a[:, :, 6]>=0) != (C['act'][qr, :, 6]>=0), 1)
                result['grip_flip_full'] = np.mean((a[:, :, 6]>=0) != (a0[:, :, 6]>=0), 1)
                result['top1_full'] = (result['rows'][:, 0] == results['full']['rows'][:, 0]).astype(float)
                for k, val in result.items(): accum[name].setdefault(k, []).append(val)
            vals = {k: C[k][qr] for k in ('row', 'ep', 'task', 'step', 'fold')}
            vals.update(sigma=np.broadcast_to(sig, (len(qr), 7)).copy(), target=C['act'][qr], past2=past2[qr])
            for k, val in vals.items(): meta[k].append(val)
        print(f'{cell}_{size} fold={fold} seconds={time.monotonic()-start:.1f}', flush=True)
    meta = {k: np.concatenate(v) for k, v in meta.items()}
    np.savez_compressed(outdir/'meta.npz', **meta)
    summary = []
    for name, result in accum.items():
        a = {k: np.concatenate(v) for k, v in result.items()}
        np.savez_compressed(outdir/f'{name}.npz', **a)
        summary.append(dict(variant=name, **{k: mean(a[k], meta['ep']) for k in (
            'motion_mse', 'full_delta_mse', 'grip_error', 'grip_flip_full', 'top1_full', 'variance')}))
    result = dict(timestamp=now(), cell=cell, size=size, rows=len(meta['row']),
                  episodes=len(np.unique(meta['ep'])), variants=summary, provenance=C['provenance'],
                  seconds=time.monotonic()-start, validation='five episode folds; fixed B-subset PCA; train-fold metric/sigma/donors; even steps >=2')
    dump(outdir/'summary.json', result)
    dump(outdir/'reads.json', sorted(READS))
    return result

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=int, nargs='+', default=[50, 200, 500])
    ap.add_argument('--cells', nargs='+', default=['pi05_l10', 'pi05_spatial', 'groot_l10', 'groot_spatial'])
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    jobs = [(c, s) for s in args.sizes for c in args.cells]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs = [pool.submit(run, *j) for j in jobs]
        for f in fs:
            r = f.result()
            print(json.dumps(r), flush=True)
