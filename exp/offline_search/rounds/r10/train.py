"""LOEO and cross-episode PAIR heads from B libraries only.

Three-fold episode CV conditions on the fixed deployment cache representation.
Held-out episodes are excluded from ALL training query and donor rows. CV is an
offline corrector diagnostic, never used to select a size, hyperparameter or rule.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import pickle
import subprocess
import time
import numpy as np

from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r03.h3_judge.judge import core
from exp.offline_search.rounds.r09.recipe.recipe import RecipeCorrectedBase, _predict
from .data import HERE, STORE, SIZES, CELLS, SubsetLibrary, assert_fit_input, write_json, sha

PAIR_CAP = 16
RADIUS_QUANTILE = .5
REVISION = 'stage2b_regime2_anchor_mass_task_median16'
TRAINING = HERE / 'training2b'
HEADS = HERE / 'heads2b'
N_RFF = 384
ALPHA = 100.
FOLDS = 3
SEED = 20261002


def load_size(model, suite, size):
    path = assert_fit_input(HERE / 'artifacts' / f'r10_{model}_{suite}_{size}_G.pkl')
    with path.open('rb') as f:
        blob = pickle.load(f)
    method = blob['method']
    assert method.variant == 'G' and method.size == size and method.fit_info['library_only']
    assert method.fit_info.get('fit_version') == 2, 'need final subset-only R10 fit'
    lib = SubsetLibrary(STORE, f'{model}_{suite}', size)
    assert np.array_equal(method.row_subset, lib.rows)
    base = method.inner.base
    features = RecipeCorrectedBase.__new__(RecipeCorrectedBase)
    vars(features).update(vars(base))
    features.sig_head = base.sig.copy()
    features.head_meta = dict(task_onehot=True)
    features.blend, features.chans, features.correct_gripper = .5, 6, False
    return lib, base, features, method.inner.C


def queries(lib, table, task):
    for rows in core.episode_rows(table).values():
        if int(table.task[rows[0]]) != task:
            continue
        for pos, i in enumerate(rows):
            # Deployed looks follow cache hits, including G's committed policy
            # tail. Step 0 retains the deployed early metric; later looks use
            # regime 2, without continuity against the recorded policy tail.
            yield int(i), core.PseudoQuery(lib, table, rows, pos, lib.model, prev_hit=True)


def neighbors(base, q, query_episode, allowed_episodes=None, cap=16):
    """Exact deployed metric, restricted BEFORE reference/median.

    The same task, early step-0 branch, continuity, stable tie rule, kNN and
    kernel reference apply. All rows of the query episode are excluded.
    """
    T, _, regime, _, _, xv, rs8, d, _, c, _ = base._dist(q)
    valid = base.lib_ep[T.rows] != query_episode
    if allowed_episodes is not None:
        valid &= np.isin(base.lib_ep[T.rows], list(allowed_episodes))
    pos = np.flatnonzero(valid)
    if not len(pos):
        raise ValueError('no cross-episode donors')
    dt = d[pos]
    if regime == 1:
        med = float(np.median(dt)) + 1e-12
        dt = dt / med + base.lam_c * c[pos] / T.s_c
    order = np.lexsort((pos, dt))[:cap]
    rows = T.rows[pos[order]]
    assert np.all(base.lib_ep[rows] != query_episode)
    return rows, dt[order], xv, rs8


def feature_row(features, q, xv, rs8, action):
    # Single shared implementation: precisely the R9 serving-time recipe.
    return features._features(q, xv, rs8, action)[0]


def anchor_weights(rows, episodes):
    """Equal episode mass; an anchor divides its row weight among its pairs.

    Total mass is the number of represented anchors, never the pair count.
    Radius-empty anchors have no samples and are excluded from that count.
    """
    anchors, first, inverse, counts = np.unique(rows, return_index=True,
                                               return_inverse=True, return_counts=True)
    anchor_eps = episodes[first]
    eps, ep_inverse, ep_counts = np.unique(anchor_eps, return_inverse=True, return_counts=True)
    per_anchor = len(anchors) / (len(eps) * ep_counts[ep_inverse])
    weights = per_anchor[inverse] / counts[inverse]
    assert np.isclose(weights.sum(), len(anchors))
    return weights


def task_dataset(lib, base, features, table, task, radius, allowed=None):
    """Both variants share query projections and retrieval; targets are stored policy chunks."""
    data = {v: dict(X=[], Y=[], ep=[], row=[], donor=[]) for v in ('loeo', 'pair')}
    for i, q in queries(lib, table, task):
        if allowed is not None and int(lib.episode[i]) not in allowed:
            continue
        ep = int(lib.episode[i])
        rows, distances, xv, rs8 = neighbors(base, q, ep, allowed, max(base.k, PAIR_CAP))
        base.reset(q.episode)
        krows, kd = rows[:base.k], distances[:base.k].astype(np.float64)
        w = _kernel_w(kd - kd[0], base.kref)
        cached = base.os_synth(q, krows, w)
        target = np.asarray(lib.action[i, :10, :6], np.float32)
        dl = data['loeo']
        dl['X'].append(feature_row(features, q, xv, rs8, cached))
        dl['Y'].append(((target - cached[:10, :6]) / base.sig[:6]).ravel())
        dl['ep'].append(ep); dl['row'].append(i); dl['donor'].append(-1)
        for j in rows[:PAIR_CAP][distances[:PAIR_CAP] <= radius]:
            action = lib.action[j]
            dp = data['pair']
            dp['X'].append(feature_row(features, q, xv, rs8, action))
            dp['Y'].append(((target - action[:10, :6]) / base.sig[:6]).ravel())
            dp['ep'].append(ep); dp['row'].append(i); dp['donor'].append(int(lib.episode[j]))
    for variant, d in data.items():
        for key in d:
            d[key] = np.asarray(d[key], dtype=np.float32 if key in ('X', 'Y') else np.int64)
        if len(d['X']) == 0:
            raise ValueError(f'empty {variant} task data')
        d['weight'] = anchor_weights(d['row'], d['ep'])
        if variant == 'pair':
            assert np.all(d['ep'] != d['donor'])
    return data


def fit_head(X, Y, weights, seed=0, batch=4096):
    """R9's standardized-input + 384 RFF weighted ridge, alpha 100.

    Accumulate sufficient statistics in float64 batches so pair training is
    bounded in memory. The equation/intercept match R9 fit_head, preserving
    the supplied anchor mass so PAIR multiplicity cannot weaken alpha.
    """
    rng = np.random.default_rng(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    W = (rng.standard_normal((X.shape[1], N_RFF)) / np.sqrt(X.shape[1])).astype(np.float32)
    bias = rng.uniform(0, 2 * np.pi, N_RFF).astype(np.float32)
    dim = X.shape[1] + N_RFF
    A, B = np.zeros((dim, dim)), np.zeros((dim, Y.shape[1]))
    sf, sy, total = np.zeros(dim), np.zeros(Y.shape[1]), 0.
    weights = np.asarray(weights, np.float64)
    assert weights.shape == (len(X),) and np.all(np.isfinite(weights)) and np.all(weights > 0)
    for lo in range(0, len(X), batch):
        hi = min(len(X), lo + batch)
        xn = np.clip((X[lo:hi] - mean) / std, -8, 8)
        F = np.concatenate([xn, np.cos(xn @ W + bias) * np.sqrt(2)], 1).astype(np.float64)
        w = weights[lo:hi]
        yf = Y[lo:hi].astype(np.float64)
        A += F.T @ (F * w[:, None])
        B += F.T @ (yf * w[:, None])
        sf += F.T @ w; sy += yf.T @ w; total += w.sum()
    A -= np.outer(sf, sf) / total
    B -= np.outer(sf, sy) / total
    A.flat[::dim + 1] += ALPHA
    coef = np.linalg.solve(A, B).T
    intercept = sy / total - coef @ (sf / total)
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32), w=W, bias=bias,
                coef=coef.astype(np.float32), intercept=intercept.astype(np.float32))


def radius_for(lib, base, table, tasks=range(10), allowed=None):
    radii, distances = {}, {}
    for task in tasks:
        distances[task] = []
        for i, q in queries(lib, table, task):
            if allowed is not None and int(lib.episode[i]) not in allowed:
                continue
            _, d, _, _ = neighbors(base, q, int(lib.episode[i]), allowed, cap=16)
            distances[task].append(float(d[-1]))
        radii[task] = float(np.median(distances[task]))
    return radii, distances


def heldout_error(base, features, lib, table, task, head, heldout, donors):
    records = []
    for i, q in queries(lib, table, task):
        ep = int(lib.episode[i])
        if ep not in heldout:
            continue
        rows, d, xv, rs8 = neighbors(base, q, ep, donors)
        assert not np.isin(lib.episode[rows], list(heldout)).any()
        base.reset(q.episode)
        kd = d.astype(np.float64)
        cached = base.os_synth(q, rows, _kernel_w(kd - kd[0], base.kref))
        X = feature_row(features, q, xv, rs8, cached)[None]
        Y = ((lib.action[i, :10, :6] - cached[:10, :6]) / base.sig[:6]).ravel()
        pred = _predict(head, X)[0]
        records.append((ep, float(np.mean(Y ** 2)), float(np.mean((Y - .5 * pred) ** 2)),
                        float(np.mean((Y - pred) ** 2))))
    return records


def train_task(model, suite, size, task, radius, cv=True):
    lib, base, features, table = load_size(model, suite, size)
    data = task_dataset(lib, base, features, table, task, radius)
    heads, report = {}, {}
    episodes = np.unique(lib.episode[lib.task_id == task])
    fold_eps = np.array_split(np.random.default_rng(SEED + task).permutation(episodes), FOLDS)
    for v, d in data.items():
        heads[v] = fit_head(d['X'], d['Y'], d['weight'])
        assert heads[v]['coef'].shape == (60, 601) and heads[v]['w'].shape == (217, 384)
        report[v] = dict(task=task, rows=len(d['X']), anchors=len(np.unique(d['row'])),
                        weight_sum=float(d['weight'].sum()), episodes=len(np.unique(d['ep'])), folds=[])
    report['source_fit_sha256'] = sha(HERE / 'artifacts' / f'r10_{model}_{suite}_{size}_G.pkl')
    report['revision'] = REVISION
    if cv:
        for fold, val_eps in enumerate(fold_eps):
            train_eps = set(int(e) for e in episodes if e not in val_eps)
            # Radius is calibrated from training queries/donors only for each fold.
            fold_radius = radius_for(lib, base, table, tasks=(task,), allowed=train_eps)[0][task]
            td = task_dataset(lib, base, features, table, task, fold_radius, train_eps)
            for v, d in td.items():
                assert not np.isin(d['ep'], val_eps).any()
                if v == 'pair': assert not np.isin(d['donor'], val_eps).any()
                head = fit_head(d['X'], d['Y'], d['weight'])
                err = heldout_error(base, features, lib, table, task, head, set(val_eps), train_eps)
                # Per-episode means then mean across episodes (failed episodes retained).
                byep = [np.asarray([r[1:] for r in err if r[0] == e]).mean(0) for e in val_eps]
                mse = np.asarray(byep).mean(0)
                report[v]['folds'].append(dict(fold=fold, heldout_episodes=val_eps.tolist(),
                    train_episodes=sorted(train_eps), radius=fold_radius, n_train=len(d['X']), n_val=len(err),
                    n_anchor=len(np.unique(d['row'])), weight_sum=float(d['weight'].sum()),
                    baseline_mse=float(mse[0]), blend05_mse=float(mse[1]), residual_mse=float(mse[2])))
    out = TRAINING / f'{model}_{suite}_{size}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'task{task}.npz', **{f'{v}_{k}': a for v, h in heads.items() for k, a in h.items()})
    write_json(out / f'task{task}.json', report)
    print(f'TRAIN {model}_{suite}_{size} task={task}', flush=True)


def prepare(model, suite, size):
    lib, base, _, table = load_size(model, suite, size)
    radii, distances = radius_for(lib, base, table)
    write_json(TRAINING / f'{model}_{suite}_{size}' / 'radius.json',
               dict(radius_by_task=radii, quantile=RADIUS_QUANTILE, k=16, pair_cap=PAIR_CAP,
                    rows_by_task={t:len(d) for t,d in distances.items()}, fit_pool='B', revision=REVISION,
                    metric='deployed step0 early metric / prev_hit=True regime 2; own episode excluded',
                    rule='per-task median LOEO 16th-neighbor distance'))
    return radii


def assemble(model, suite, size):
    lib, base, _, _ = load_size(model, suite, size)
    out = TRAINING / f'{model}_{suite}_{size}'
    radius = json.loads((out / 'radius.json').read_text())
    reports = [json.loads((out / f'task{t}.json').read_text()) for t in range(10)]
    for v in ('loeo', 'pair'):
        arrays = {}
        for t in range(10):
            with np.load(out / f'task{t}.npz', allow_pickle=False) as z:
                for k in ('mean', 'std', 'w', 'bias', 'coef', 'intercept'):
                    arrays[f'{t}_{k}'] = np.array(z[f'{v}_{k}'])
        meta = dict(cell=f'{model}_{suite}_cache', size=size, chans=6, sigma=base.sig.tolist(),
                    task_onehot=True, per_task=True, n_rff=N_RFF, alpha=ALPHA, blend=.5, variant=v,
                    revision=REVISION, prev_hit=True, weight_mass='number of represented anchor rows',
                    radius=radius, fit_pool='B', source_fit_sha256=sha(HERE / 'artifacts' / f'r10_{model}_{suite}_{size}_G.pkl'))
        HEADS.mkdir(exist_ok=True)
        np.savez(HEADS / f'{model}_{suite}_{size}_{v}.npz', **arrays, meta_json=json.dumps(meta))
        folds = [r[v]['folds'] for r in reports]
        summary = dict(model=model, suite=suite, size=size, variant=v, revision=REVISION,
            diagnostic='3-fold by episode, equal episode weight; fixed full-subset deployment PCA/metric/sigma',
            candidate_rule='training and validation retrieve only from training-fold episodes; own episode excluded',
            selection='diagnostic only; no selection or tuning', tasks=[r[v] for r in reports])
        if all(folds):
            allfolds = [f for taskfolds in folds for f in taskfolds]
            # Weight fold scores by episode count so each of the subset episodes has equal weight.
            weights = [len(f['heldout_episodes']) for f in allfolds]
            for k in ('baseline_mse', 'blend05_mse', 'residual_mse'):
                summary[k] = float(np.average([f[k] for f in allfolds], weights=weights))
        write_json(out / f'cv_{v}.json', summary)


def main():
    from .build import PREFIX, REPO
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['all', 'task'])
    p.add_argument('--model'); p.add_argument('--suite'); p.add_argument('--size', type=int)
    p.add_argument('--task', type=int); p.add_argument('--radius', type=float)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.action == 'task':
        train_task(a.model, a.suite, a.size, a.task, a.radius)
        return
    for model, suite in CELLS:
        for size in SIZES:
            if a.model and a.model != model or a.suite and a.suite != suite or a.size and a.size != size:
                continue
            path = HERE / 'artifacts' / f'r10_{model}_{suite}_{size}_G.pkl'
            while True:
                try:
                    with path.open('rb') as f:
                        ready = pickle.load(f)['method'].fit_info.get('fit_version') == 2
                except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                    ready = False
                if ready: break
                print(f'WAIT for final B-only fit {path.name}', flush=True)
                time.sleep(10)
            radius = prepare(model, suite, size)
            out = TRAINING / f'{model}_{suite}_{size}'
            def one(task):
                if (out / f'task{task}.json').exists():
                    report = json.loads((out / f'task{task}.json').read_text())
                    if report.get('source_fit_sha256') == sha(path) and report.get('revision') == REVISION: return
                cmd = PREFIX + ['-m', 'exp.offline_search.rounds.r10.train', 'task', '--model', model,
                    '--suite', suite, '--size', str(size), '--task', str(task), '--radius', str(radius[task])]
                with (out / f'task{task}.log').open('w') as f:
                    proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
                if proc.returncode: raise RuntimeError(f'training failure: {out}/task{task}.log')
                print(f'TRAIN {model}_{suite}_{size} task={task}', flush=True)
            with ThreadPoolExecutor(a.workers) as ex:
                list(ex.map(one, range(10)))
            assemble(model, suite, size)
            print(f'HEADS {model}_{suite}_{size}', flush=True)


if __name__ == '__main__':
    main()
