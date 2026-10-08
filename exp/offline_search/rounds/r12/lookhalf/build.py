"""Reproduce the frozen half fits from selected B library, never dev outcomes.

The default reuses the byte-preserved library-only R10 artifact as the full
stack. --fresh-base also rebuilds that stack from the B library. Every added
metric, scale, head and atlas is recomputed; explorer artifacts are validation
references only and are never builder inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.data import STORE, sha
from exp.offline_search.rounds.r10.recipe.build import subset_kwargs
from exp.offline_search.rounds.r10.recipe.fitting import _Library, library_boundary, fit_path
from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe
from exp.offline_search.rounds.r10.train import fit_head, anchor_weights
from exp.offline_search.closed_loop.lookhalf import R12LookHalf, METHODS
from exp.offline_search.closed_loop.lookhalf_math import Metric, features, balanced_median, freeze_fit as freeze
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w

HERE = Path(__file__).resolve().parent
R10 = HERE.parents[1]/'r10'
SPEC = 'exp.offline_search.closed_loop.lookhalf:R12LookHalf'
CELLS = (('pi05', 'l10', 50), ('pi05', 'spatial', 50), ('pi05', 'spatial', 200),
         ('groot', 'spatial', 50), ('groot', 'l10', 50))


def predecessors(lib, n, *, opus=False):
    rows = np.arange(lib.L)
    prev = np.asarray(lib.prev)
    for _ in range(n):
        rows = np.where(prev[rows] >= 0, prev[rows], rows)
    if opus:
        rows = np.where(np.asarray(lib.step) < n, np.arange(lib.L), rows)
    if not np.array_equal(lib.episode[rows], lib.episode):
        raise ValueError('library predecessor crosses episodes')
    return rows


def loeo(metric, X, qr, cr, C, *, early=False, precision='astra'):
    actions, ds, chosen = [], [], []
    batch = 192 if precision == 'astra' else 256
    for lo in range(0, len(qr), batch):
        q = qr[lo:lo+batch]
        D = metric.dist(X[q])
        if early and metric.early is not None:
            initial = C['step'][q] == 0
            if initial.any():
                D[initial] = metric.dist(X[q[initial]], initial=True)
        D[C['ep'][q, None] == C['ep'][cr][None]] = np.inf
        ix = np.argsort(D, axis=1, kind='stable')[:, :16]
        dk = np.take_along_axis(D, ix, 1).astype(np.float64)
        if not np.isfinite(dk).all():
            raise ValueError('at least sixteen other-episode donors required')
        w = _kernel_w(dk-dk[:, :1], C['kref'])
        w = (w/w.sum(1, keepdims=True)).astype(np.float32)
        rows = cr[ix]
        actions.append(np.einsum('qk,qktc->qtc', w, C['act'][rows], optimize=False))
        ds.append(dk[:, 0]); chosen.append(rows[:, 0])
    return np.concatenate(actions), np.concatenate(ds), np.concatenate(chosen)


def scale(X, C, *, precision):
    ds, es = [], []
    for fold in range(5):
        tr = np.flatnonzero(C['fold'] != fold)
        sigma = np.maximum(C['act'][tr, :5].std((0, 1)), 1e-6)
        if precision == 'opus':
            # Exact R10 five-fold distance recipe, replacing only its features.
            from exp.offline_search.rounds.r10.astra.offline import Metric as FullMetric, serve
            Cf = dict(C, X=np.asarray(X, np.float32))
        for task in range(10):
            cr = tr[C['task'][tr] == task]
            qr = np.flatnonzero((C['fold'] == fold) & (C['task'] == task))
            if not len(qr):
                continue
            if precision == 'opus':
                m = FullMetric(Cf['X'][cr], C['act'][cr], C['ep'][cr], C['step'][cr], sigma, early=False)
                distance, _ = serve(Cf, qr, cr, m, synth=False)
            else:
                m = Metric(X[cr], C['act'][cr], C['ep'][cr], sigma)
                _, distance, _ = loeo(m, X, qr, cr, C)
            keep = C['step'][qr] > 0
            ds.append(distance[keep]); es.append(C['ep'][qr][keep])
    value = balanced_median(np.concatenate(ds), np.concatenate(es))
    if not np.isfinite(value) or value <= 0:
        raise ValueError('invalid held-out distance scale')
    return value


def fitted_data(recipe, ctx, lib):
    b = recipe.inner.base
    P = []
    # Only keys from the selected whole B episodes; bounded gathers avoid
    # materializing the unselected parent key corpus.
    for camera in range(2):
        pieces = []
        keys = getattr(lib, f'key_v{camera}')
        basis, center = getattr(b, f'B{camera}T'), getattr(b, f'muB{camera}')
        for lo in range(0, lib.L, 128):
            pieces.append((np.asarray(keys[lo:lo+128], np.float32)@basis.T-center).astype(np.float32))
        P.append(np.concatenate(pieces))
    P = np.concatenate(P, 1)
    ef = {int(e): i%5 for ids in lib.chosen.values() for i, e in enumerate(ids)}
    return dict(P=P, rs=np.asarray(lib.rs[:, :8], np.float32),
                act=np.asarray(lib.action[:, :10, :7], np.float32), ep=np.asarray(lib.episode),
                step=np.asarray(lib.step), task=np.asarray(lib.task_id),
                progress=np.asarray(lib.progress), lens=np.asarray(lib.ep_len),
                fold=np.asarray([ef[int(e)] for e in lib.episode]), kref=b.kref)


def fit_additions(recipe, ctx):
    directory = Path(ctx.root)/'library'/ctx.lib_key/recipe.library
    with library_boundary(ctx.root, directory) as reads:
        lib = _Library(ctx.root, ctx.lib_key, recipe.library, recipe.episode_subset)
        np.testing.assert_array_equal(lib.rows, recipe.row_subset)
        C = fitted_data(recipe, ctx, lib)
        P, rs = C['P'], C['rs']
        past = predecessors(lib, 2)
        sigma = np.maximum(C['act'][:, :5].std((0, 1)), 1e-6)
        osigma = np.asarray(C['act'][:, :5], np.float64).std((0, 1))
        Xs = {'astra_f': np.c_[P, rs].astype(np.float32),
              'astra_h': np.c_[P[past, :64], P[:, 64:], rs, rs-rs[past]].astype(np.float32),
              'astra_w': np.c_[P[:, 64:], rs, rs-rs[past]].astype(np.float32),
              'opus_w': np.c_[P[:, 64:], rs].astype(np.float64)}
        for lag in (2, 4, 6, 8):
            pp = predecessors(lib, lag, opus=True)
            Xs[f'opus_l{lag}'] = np.c_[P[pp, :64], P[:, 64:], rs].astype(np.float64)
        out = dict(metrics={}, heads={}, scales={}, P=P, past2=past, sigma=sigma)
        safe, agreement = np.zeros(lib.L, bool), np.zeros(lib.L, bool)
        for name, X in Xs.items():
            out['metrics'][name] = {}
            if name in ('opus_w', 'astra_h', 'astra_w'):
                out['heads'][name] = {}
            distances, episodes = [], []
            for task in range(10):
                cr = np.flatnonzero(C['task'] == task)
                opus = name.startswith('opus')
                sig = osigma if opus else sigma
                metric = Metric(X[cr], C['act'][cr], C['ep'][cr], sig,
                                precision='opus' if opus else 'astra', early=name == 'opus_w', step=C['step'][cr])
                out['metrics'][name][task] = (metric, cr)
                if name in out['heads']:
                    qr = cr if opus else cr[C['step'][cr] >= 2]
                    raw, d, _ = loeo(metric, X, qr, cr, C, early=opus,
                                    precision='opus' if opus else 'astra')
                    # The serving head normalization is the unchanged R10 sigma.
                    hs = recipe.inner.base.sig_head if opus else sigma
                    y = ((C['act'][qr, :, :6]-raw[:, :, :6])/hs[:6]).reshape(len(qr), 60)
                    out['heads'][name][task] = fit_head(features(X[qr], raw, C['step'][qr], C['task'][qr], hs),
                                                       y, anchor_weights(qr, C['ep'][qr]))
                    distances.append(d); episodes.append(C['ep'][qr])
            if name in out['heads']:
                out['scales'][name] = (balanced_median(np.concatenate(distances), np.concatenate(episodes))
                    if name == 'astra_w' else scale(X, C, precision='opus' if name == 'opus_w' else 'astra'))
            print(f'FIT {ctx.lib_key} {recipe.size} {name}', flush=True)
        # Reproduce the selected-library safety atlas, with failed episodes.
        for task in range(10):
            cr = np.flatnonzero(C['task'] == task)
            qr = cr[C['step'][cr] >= 4]
            if not len(qr):
                continue
            mf = out['metrics']['astra_f'][task][0]
            mh = out['metrics']['astra_h'][task][0]
            af, _, rf = loeo(mf, Xs['astra_f'], qr, cr, C)
            ah, _, rh = loeo(mh, Xs['astra_h'], qr, cr, C)
            _, _, rp = loeo(mf, Xs['astra_f'], past[qr], cr, C)
            agreement[qr] = (np.sqrt(np.mean(((af[:, :, :6]-ah[:, :, :6])/sigma[:6])**2, (1, 2))) <= .15) & ((af[:, :, 6]>=0) == (ah[:, :, 6]>=0)).all(1)
            adv = lambda r: (C['progress'][r]-C['progress'][rp])*np.maximum(C['lens'][r]-1, 1) > .5
            safe[qr] = agreement[qr] & (adv(rf) == adv(rh))
        nxt = np.full(lib.L, -1, int)
        valid = np.flatnonzero(np.asarray(lib.prev) >= 0)
        nxt[np.asarray(lib.prev)[valid]] = valid
        next2 = np.where(nxt >= 0, nxt[np.maximum(nxt, 0)], -1)
        out['atlas'] = dict(next2=next2, safe=safe, action_agreement=agreement, rs=rs,
                            state_scale=np.maximum(rs.std(0), 1e-6))
        out['bytes'] = freeze(out)
    recipe.half_fit = out
    recipe.fit_info = dict(recipe.fit_info, lookhalf_library_only=True, lookhalf_fit_reads=sorted(reads),
        lookhalf_fit_version=1, frozen_dev_constants=dict(opus_tau=.10, opus_margin=1., opus_maxlag=8,
            consensus_tau=.10, strict_tau=.05, support=.99, safe_mass=.75, state_rms=.2,
            witness_progress=.5, advance=.5),
        constants_provenance='post-freeze B-dev calibration in explorer SPEC; no dev data read by builder',
        lookhalf_scales=out['scales'], lookhalf_rows=lib.L,
        numerical_path='GR00T genuine single-camera against joint-library keys; exact-input diagnostic required')


def fit_wrapper(recipe, ctx):
    from exp.offline_search.rounds.r11.knob.fitting import fit_knob
    if recipe.knob_method == 'off':
        from exp.offline_search.rounds.r10.recipe.fitting import fit_recipe
        fit_recipe(recipe, ctx)
    else:
        fit_knob(recipe, ctx)
    if recipe.lookhalf_method != 'off':
        fit_additions(recipe, ctx)


def wrap(base, kwargs, fit=None):
    obj = R12LookHalf(**kwargs)
    # Preserve all deployed fitted fields, while keeping R12/R11 lifecycle
    # fields initialized by its constructor.
    vars(obj).update(vars(base))
    obj.lookhalf_method = kwargs['method']
    obj.half_fit = fit
    obj.reset(api.EpisodeView('r12-build', '', 0, 0, 0, 0))
    obj.name = f'R12_{kwargs["method"]}__{base.name}'
    return obj


def build_cell(model, suite, size, *, fresh_base=False):
    dest = HERE/'artifacts'/f'{model}_{suite}_{size}'
    dest.mkdir(parents=True, exist_ok=True)
    kwargs = subset_kwargs(model, suite, R10/'subsets'/f'{model}_{suite}_{size}.json')
    ctx = api.Context(root=STORE, cell=f'{model}_{suite}_cache', seed=0, scratch=HERE/'scratch'/dest.name)
    if fresh_base:
        base = R10Recipe(**kwargs)
        base.fit(None, ctx)
        provenance = dict(rebuilt_full_stack=True)
    else:
        source = R10/'recipe/artifacts'/f'r10_recipe_{model}_{suite}_{size}.pkl'
        fit_path(source, STORE)
        with source.open('rb') as f:
            blob = pickle.load(f)
        if blob['kwargs'] != kwargs or not blob['method'].fit_info['library_only']:
            raise ValueError('foreign full fit selection')
        base = blob['method']
        directory = Path(STORE)/'library'/ctx.lib_key/kwargs['library']
        if base.fit_info['episodes_sha256'] != sha(directory/'episodes.json'):
            raise ValueError('B source changed')
        provenance = dict(full_fit=str(source), full_fit_sha256=sha(source), rebuilt_full_stack=False)
    methods = ['off', 'opus_m1', 'astra_consensus2']
    if model == 'pi05' and size == 50:
        methods.append('astra_strict')
    fitted = wrap(base, dict(kwargs, method='opus_m1'))
    started = time.monotonic()
    fit_additions(fitted, ctx)
    written = []
    for method in methods:
        kw = dict(kwargs, method=method)
        obj = wrap(base, kw, None if method == 'off' else fitted.half_fit)
        if method != 'off':
            obj.fit_info = fitted.fit_info
        obj.fit_info = dict(obj.fit_info, **provenance)
        api.check_method_attrs(obj)
        path = dest/f'{method}.pkl'
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as f:
            pickle.dump(dict(spec=SPEC, kwargs=kw, cell=ctx.cell, method=obj, registered={},
                             fit_s=time.monotonic()-started), f, protocol=4)
        temporary.replace(path)
        written.append(dict(method=method, artifact=str(path), sha256=sha(path), bytes=path.stat().st_size))
    (dest/'build.json').write_text(json.dumps(dict(model=model, suite=suite, size=size,
        artifacts=written, fit_info=fitted.fit_info, seconds=time.monotonic()-started), indent=2)+'\n')
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cell', choices=[f'{m}:{s}:{n}' for m,s,n in CELLS])
    ap.add_argument('--fresh-base', action='store_true')
    a = ap.parse_args()
    cells = CELLS if a.cell is None else [(a.cell.split(':')[0], a.cell.split(':')[1], int(a.cell.split(':')[2]))]
    for cell in cells:
        print(json.dumps(build_cell(*cell, fresh_base=a.fresh_base)), flush=True)


if __name__ == '__main__':
    main()
