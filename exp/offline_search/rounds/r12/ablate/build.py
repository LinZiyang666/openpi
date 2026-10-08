"""Freeze five layer-one arms per cell; reduced fits read only selected B libraries."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pickle
import time
import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.data import STORE, sha
from exp.offline_search.rounds.r10.recipe.fitting import _Library, library_boundary, fit_path
from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.closed_loop.cache_ablate import ARMS, SPEC, R12CacheAblate, _CameraCache, CameraMetric
from exp.offline_search.rounds.r12.lookhalf.build import CELLS, fitted_data

HERE = Path(__file__).resolve().parent
R10 = HERE.parents[1] / 'r10'


def refit_metric(X, heads, ep, step):
    """Exact deployed AWM full-rank fit and early affine float32 representation."""
    mean, std, W = fit_metric(X, heads, ep, nn=3, lam=.1, rank=0)
    m = CameraMetric()
    m.Z = np.ascontiguousarray(((X-mean)/std) @ W, np.float32)
    Zr = m.Z.astype(np.float64)
    m.z2 = np.sum(Zr*Zr, axis=1).astype(np.float32)
    m.W = np.ascontiguousarray(W/std[:, None], np.float32)
    m.shift = ((mean/std) @ W).astype(np.float32)
    keep = step <= 2
    mean0, std0, W0 = fit_metric(X[keep], heads[keep], ep[keep], nn=3, lam=.1, rank=0)
    A = np.linalg.inv(W) @ (np.diag(std/std0) @ W0)
    b = ((mean-mean0)/std0) @ W0
    Y = Zr @ A
    m.A0 = np.ascontiguousarray(A, np.float32)
    m.n20 = (Y*Y).sum(1).astype(np.float32)
    m.W0 = np.ascontiguousarray(W0/std0[:, None], np.float32)
    m.c0 = ((mean0/std0) @ W0+b).astype(np.float32)
    return m


def build_cell(model, suite, size):
    cell = f'{model}_{suite}_{size}'
    dest = HERE / 'artifacts' / cell
    if dest.exists():
        raise FileExistsError(f'fresh artifacts required: {dest}')
    source = R10 / 'recipe/artifacts' / f'r10_recipe_{cell}.pkl'
    fit_path(source, STORE)
    with source.open('rb') as stream:
        blob = pickle.load(stream)
    recipe = blob['method']
    if not recipe.fit_info['library_only']:
        raise ValueError('source must be B library only')
    library = Path(STORE) / 'library' / f'{model}_{suite}' / recipe.library
    if sha(library / 'episodes.json') != recipe.fit_info['episodes_sha256']:
        raise ValueError('B library source changed')
    ctx = api.Context(root=STORE, cell=f'{model}_{suite}_cache', seed=0, scratch=HERE/'scratch'/cell)
    with library_boundary(STORE, library) as reads:
        lib = _Library(STORE, ctx.lib_key, recipe.library, recipe.episode_subset)
        np.testing.assert_array_equal(lib.rows, recipe.row_subset)
        C = fitted_data(recipe, ctx, lib)
        base = recipe.inner.base
        Xfull = np.c_[C['P'], C['rs']].astype(np.float64)
        # Original AWM fitting casts the context's float32 action sigma to f64.
        heads = (np.asarray(lib.action[:, :5, :7], np.float64)/base.sig.astype(np.float64)).reshape(lib.L, 35)
        metrics = {}
        for camera in (0, 1):
            indices = np.r_[np.arange(camera*64, (camera+1)*64), np.arange(128, 136)]
            for variant in ('sub', 'refit'):
                by_task = {}
                for task, T in base.tasks.items():
                    X = Xfull[T.rows][:, indices]
                    if variant == 'sub':
                        m = CameraMetric()
                        m.restricted_X = X
                        m.transform = T.Wf[indices].astype(np.float64)
                        m.early_transform = T.W0f[indices].astype(np.float64)
                    else:
                        m = refit_metric(X, heads[T.rows], C['ep'][T.rows], C['step'][T.rows])
                    by_task[task] = m
                metrics[camera, variant] = by_task
    dest.mkdir(parents=True)
    artifacts = []
    for selected in ARMS:
        started = time.monotonic()
        kwargs = dict(blob['kwargs'], method=selected)
        method = R12CacheAblate(**kwargs)
        inner = (BlindAWM if selected == 'both' else _CameraCache).__new__(BlindAWM if selected == 'both' else _CameraCache)
        vars(inner).update(vars(base))
        for key in ('heads', 'sig_head', 'head_meta', 'blend', 'chans', 'correct_gripper', 'distance_scale', 'distance_rule'):
            vars(inner).pop(key, None)
        if selected != 'both':
            inner.selected_camera = int(selected.startswith('wrist'))
            inner.camera_metrics = metrics[inner.selected_camera, selected.split('_')[1]]
        method.inner, method.row_subset = inner, recipe.row_subset
        method.size = recipe.size
        method.fit_info = dict(library_only=True, library=recipe.library, episodes=recipe.size,
            episode_ids_by_task=recipe.episode_subset, fit_reads=sorted(reads),
            source_fit=str(source), source_fit_sha256=sha(source),
            episodes_sha256=recipe.fit_info['episodes_sha256'],
            layer1_only=True, guard=False, corrector=False, knob=False, policy_calls=0,
            metric=('unchanged deployed' if selected == 'both' else
                    'principal joint Mahalanobis camera+state block; no refit' if selected.endswith('sub') else
                    'selected B only AWM nn=3 ridge=.1 rank=0; early step<=2'),
            pca='unchanged selected-B per-camera PCA64',
            numerical_path='GR00T one-camera encoding differs about 2% from joint library encoding; retained')
        method.reset(api.EpisodeView('ablate-build', '', 0, 0, 0, 0))
        path = dest / f'{selected}.pkl'
        with path.open('xb') as stream:
            pickle.dump(dict(spec=SPEC, kwargs=kwargs, cell=ctx.cell, method=method,
                             registered={}, fit_s=time.monotonic()-started), stream, protocol=4)
        artifacts.append(dict(method=selected, path=str(path), sha256=sha(path)))
    (dest/'build.json').write_text(json.dumps(dict(cell=cell, fit_pool='B', artifacts=artifacts,
        source_fit=str(source), source_fit_sha256=sha(source), fit_reads=sorted(reads)), indent=2)+'\n')
    print(f'BUILT {cell}: {len(artifacts)} arms', flush=True)


def main():
    for args in CELLS:
        build_cell(*args)


if __name__ == '__main__':
    main()
