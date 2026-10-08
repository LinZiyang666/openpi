"""Build table-independent wrist heads/scales from selected B-library rows."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.data import STORE, sha
from exp.offline_search.rounds.r10.recipe.fitting import _Library, library_boundary, fit_path
from exp.offline_search.rounds.r10.astra.offline import Metric as JointMetric, sigma
from exp.offline_search.rounds.r10.train import fit_head, anchor_weights
from exp.offline_search.rounds.r12.lookhalf.build import CELLS, fitted_data, loeo
from exp.offline_search.closed_loop.cache_ablate import CameraMetric
from exp.offline_search.closed_loop.lookhalf_math import features, balanced_median

HERE = Path(__file__).resolve().parent
R12 = HERE.parent
INDICES = np.r_[np.arange(64, 128), np.arange(128, 136)]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:
        pickle.dump(value, stream, protocol=4)
    temp.replace(path)


def load(path):
    fit_path(path, STORE)
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


class SubMetric:
    """LOEO adapter for the exact restricted metric, with bounded query batches."""
    def __init__(self, metric):
        self.metric = metric
        self.early = True

    def dist(self, X, *, initial=False):
        # Distance evaluation is the ablation's direct-delta arithmetic, including
        # its float32 final rounding; no foreign camera projection is required.
        return np.stack([self.metric.distance(x, initial=initial) for x in X])


def fit_wrist(C, base, metrics):
    X = np.c_[C['P'][:, 64:], C['rs']].astype(np.float32)
    heads, evidence = {}, {}
    for task in range(10):
        rows = base.tasks[task].rows
        raw, d, top = loeo(SubMetric(metrics[task]), X, rows, rows, C, early=True, precision='opus')
        if np.any(C['ep'][top] == C['ep'][rows]):
            raise ValueError('LOEO own episode donor')
        hs = base.sig_head
        y = ((C['act'][rows, :, :6]-raw[:, :, :6])/hs[:6]).reshape(len(rows), 60)
        heads[task] = fit_head(features(X[rows], raw, C['step'][rows], C['task'][rows], hs),
                               y, anchor_weights(rows, C['ep'][rows]))
        evidence[task] = dict(rows=len(rows), episodes=len(np.unique(C['ep'][rows])),
            own_episode_excluded=True, query_rows=rows.tolist(), donor_top1=top.tolist())
        print(f'WRIST HEAD task={task} rows={len(rows)}', flush=True)
    distances, episodes, folds = {t: [] for t in range(10)}, {t: [] for t in range(10)}, []
    Xfull = np.c_[C['P'], C['rs']].astype(np.float32)
    for fold in range(5):
        tr = np.flatnonzero(C['fold'] != fold)
        va = np.flatnonzero(C['fold'] == fold)
        sig = sigma(C, tr)
        folds.append(dict(fold=fold, train_episodes=np.unique(C['ep'][tr]).tolist(),
                          heldout_episodes=np.unique(C['ep'][va]).tolist()))
        if np.intersect1d(C['ep'][tr], C['ep'][va]).size:
            raise ValueError('held-out scale episode in fit')
        for task in range(10):
            cr = tr[C['task'][tr] == task]
            qr = va[(C['task'][va] == task) & (C['step'][va] > 0)]
            if not len(qr):
                continue
            joint = JointMetric(Xfull[cr], C['act'][cr], C['ep'][cr], C['step'][cr], sig, early=False)
            metric = CameraMetric()
            metric.restricted_X = Xfull[cr].astype(np.float64)[:, INDICES]
            metric.transform = joint.W[INDICES].astype(np.float64)
            D = SubMetric(metric).dist(X[qr])
            distances[task].append(D.min(1)); episodes[task].append(C['ep'][qr])
        print(f'WRIST SCALE fold={fold}', flush=True)
    scales = {t: balanced_median(np.concatenate(distances[t]), np.concatenate(episodes[t])) for t in range(10)}
    if any(not np.isfinite(s) or s <= 0 for s in scales.values()):
        raise ValueError('invalid wrist scale')
    return heads, scales, evidence, folds


def build_cell(model, suite, size):
    cell = f'{model}_{suite}_{size}'
    dest = HERE/'fits'/cell
    source = R12/'guard/artifacts'/cell/'h2.pkl'
    ablation = R12/'ablate/artifacts'/cell/'wrist_sub.pkl'
    guard, ablate = load(source)['method'], load(ablation)['method']
    if not guard.fit_info.get('guard_library_only') or not ablate.fit_info['library_only']:
        raise ValueError('unattested source fit')
    np.testing.assert_array_equal(guard.row_subset, ablate.row_subset)
    directory = Path(STORE)/'library'/f'{model}_{suite}'/guard.library
    if sha(directory/'episodes.json') != guard.fit_info['episodes_sha256']:
        raise ValueError('B-library source changed')
    started = time.monotonic()
    ctx = api.Context(root=STORE, cell=f'{model}_{suite}_cache', seed=0, scratch=HERE/'scratch'/cell)
    with library_boundary(STORE, directory) as reads:
        lib = _Library(STORE, ctx.lib_key, guard.library, guard.episode_subset)
        np.testing.assert_array_equal(lib.rows, guard.row_subset)
        C = fitted_data(guard, ctx, lib)
        heads, scales, evidence, folds = fit_wrist(C, guard.inner.base, ablate.inner.camera_metrics)
    fit = dict(heads=heads, scale_sub=scales, camera_metrics=ablate.inner.camera_metrics,
        fit_info=dict(library_only=True, half_library_only=True, fit_reads=sorted(reads),
            episodes_sha256=guard.fit_info['episodes_sha256'], row_subset_sha256=guard.fit_info['row_subset_sha256'],
            head_parameters=dict(features=153, rff=384, ridge=100., seed=0, blend=.5, step0=False),
            loeo=evidence, distance_folds=folds, scale_sub=scales,
            guard_source=str(source), guard_source_sha256=sha(source),
            ablation_source=str(ablation), ablation_source_sha256=sha(ablation),
            metric='restricted deployed joint block; per-fold joint refit for scale only'))
    dump(dest/'wrist.pkl', fit)
    write_json(dest/'BUILD.json', dict(cell=cell, seconds=time.monotonic()-started,
        artifact=str(dest/'wrist.pkl'), sha256=sha(dest/'wrist.pkl'), **fit['fit_info']))
    print(f'BUILT wrist-only additions {cell}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cell', choices=[f'{m}_{s}_{n}' for m,s,n in CELLS])
    args = ap.parse_args()
    for m,s,n in CELLS:
        if args.cell is None or args.cell == f'{m}_{s}_{n}':
            build_cell(m,s,n)


if __name__ == '__main__':
    main()
