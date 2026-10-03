"""Train-only grouped CV, then an untouched 20..29 shadow-error evaluation."""
import argparse
from datetime import datetime, timezone
import json
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from exp.offline_search.rounds.r09.explore_fable.round2.tools.corrector import fit_head
from exp.offline_search.rounds.r09.explore_fable.round2.tools.methods import load_head, _predict
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from .data import (HERE, STORE, CELLS, VARIANTS, dataset, combine, subset, balanced,
                   episode_errors, control_row, dump, sha, owned)
from .numeric import dense_predict, residual

# Fixed before any evaluation. One configuration is selected jointly over all four cells.
CONFIGS = {
    'shared': dict(dense=True, local=None),
    'row1': dict(dense=False, local=1.0),
    'row10': dict(dense=False, local=10.0),
    'row100': dict(dense=False, local=100.0),
    'shared_row1': dict(dense=True, local=1.0),
    'shared_row10': dict(dense=True, local=10.0),
    'shared_row100': dict(dense=True, local=100.0),
}


def train_assert(d):
    if not np.all((d['init'] >= 0) & (d['init'] < 20)):
        raise ValueError('fit attempted outside 0..19')
    if d['x'].shape[1] != 207:
        raise ValueError('expected observation-only 207-dimensional input')


def design(d, nrows):
    if d['rows'].max() >= nrows:
        raise ValueError('library row outside table')
    return sparse.csr_matrix((d['weights'].ravel(),
        (np.repeat(np.arange(len(d['rows'])), d['rows'].shape[1]), d['rows'].ravel())),
        shape=(len(d['rows']), nrows))


def fit_candidates(d, nrows):
    train_assert(d)
    y = (d['teacher'][:, :, :6] - d['base'][:, :, :6]).reshape(len(d['init']), 60)
    weight = balanced(d['episode'])
    head = fit_head(d['x'], y, weight, seed=260602, n_rff=768, alpha=100.)
    shared = dense_predict(head, d['x'])
    w = design(d, nrows)
    wt = w.T.multiply(weight)
    gram = (wt @ w).tocsc()
    rhs = wt @ np.c_[y, y - shared]
    tables = {}
    for lam in (1., 10., 100.):
        solution = splu(gram + sparse.eye(nrows, format='csc') * lam).solve(rhs)
        tables[False, lam], tables[True, lam] = solution[:, :60], solution[:, 60:]
    return {name: (head if c['dense'] else {},
        tables[c['dense'], c['local']].astype(np.float32) if c['local'] is not None
        else np.zeros((nrows, 60), np.float32)) for name, c in CONFIGS.items()}


def prediction(model, d):
    return d['base'][:, :, :6] + .5 * residual(*model, d['x'], d['rows'], d['weights'])


def library(cell):
    p = STORE / 'library' / cell.rsplit('_', 1)[0] / 'current/action.npy'
    return np.load(p, mmap_mode='r', allow_pickle=False)


def select():
    if (HERE / 'SELECTION.json').exists():
        raise ValueError('selection already frozen')
    all_scores = {}
    for cell in CELLS:
        started = time.monotonic()
        d = combine([dataset(cell, v, 'train') for v in VARIANTS])
        lib = library(cell)
        reconstruction = np.einsum('nk,nkha->nha', d['weights'], lib[d['rows'], :10, :7])
        parity = float(np.max(np.abs(reconstruction[:, :, :6] - d['base'][:, :, :6])))
        assert parity < 1e-4, parity
        sums = {k: [] for k in CONFIGS}
        for fold in range(3):
            tr, va = subset(d, d['init'] % 3 != fold), subset(d, d['init'] % 3 == fold)
            models = fit_candidates(tr, len(lib))
            _, base = episode_errors(va['base'][:, :, :6], va)
            for name, model in models.items():
                _, e = episode_errors(prediction(model, va), va)
                sums[name].append((float(e.sum()), float(base.sum())))
        scores = {k: sum(v[0] for v in vv) / sum(v[1] for v in vv) for k, vv in sums.items()}
        all_scores[cell] = dict(scores=scores, anchors=len(d['init']),
            episodes=len(np.unique(d['episode'])), reconstruction_max_abs=parity,
            library_rows=len(lib), seconds=time.monotonic()-started)
        dump(HERE / 'results' / f'cv_{cell}.json', all_scores[cell])
        print(cell, {k: round(v, 4) for k, v in scores.items()}, flush=True)
    means = {k: float(np.mean([s['scores'][k] for s in all_scores.values()])) for k in CONFIGS}
    winner = min(means, key=means.get)
    dump(HERE / 'SELECTION.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        fit_inits=list(range(20)), selection='three folds by init modulo 3; equal-cell mean relative episode MSE',
        eval_accessed=False, configs=CONFIGS, cells=all_scores, scores=means, winner=winner, blend=.5))
    print('SELECTED', winner, flush=True)


def load_control(cell):
    row = control_row(cell)
    with open(row['kwargs']['corrected_fit'], 'rb') as f:
        b = FitUnpickler(f).load()
    m = b['method']
    assert m.head_meta['train_inits'] == list(range(20))
    assert m.blend == .5 and not m.correct_gripper
    return m


def control_prediction(m, d):
    x = d['x'].copy()
    x[:, 136:206] = (d['base'] / m.sig_head).reshape(len(x), -1)
    if m.head_meta.get('task_onehot', True):
        x = np.c_[x, np.eye(10, dtype=np.float32)[d['task']]]
    out = d['base'][:, :, :6].copy()
    for task in np.unique(d['task']):
        take = d['task'] == task
        corr = _predict(m.heads[str(int(task))], x[take]).reshape(-1, 10, m.chans)
        out[take] += .5 * corr[:, :, :6] * m.sig_head[:6]
    return out


def metric(pred, d, reference=None):
    ep, e = episode_errors(pred, d)
    _, b = episode_errors(d['base'][:, :, :6], d)
    result = dict(anchors=len(pred), episodes=len(ep), mse=float(e.mean()),
        base_mse=float(b.mean()), relative_to_cache=float(e.mean()/b.mean()-1),
        episode_win_fraction=float((e < b).mean()))
    if reference is not None:
        _, c = episode_errors(reference, d)
        # Paired resampling of ten init clusters, all tasks retained within each cluster.
        delta = np.array([(e-c)[ep % 30 == i].mean() for i in range(20, 30)])
        draws = delta[np.random.default_rng(260602).integers(0, 10, (5000, 10))].mean(1)
        result.update(relative_to_control=float(e.mean()/c.mean()-1),
            difference_to_control=float((e-c).mean()),
            difference_ci95=np.quantile(draws, [.025, .975]).tolist(),
            better_than_control=int((e<c).sum()), worse_than_control=int((e>c).sum()))
    return result


def evaluate():
    selection = json.loads((HERE / 'SELECTION.json').read_text())
    name = selection['winner']
    results = {}
    for cell in CELLS:
        d = combine([dataset(cell, v, 'train') for v in VARIANTS])
        models = fit_candidates(d, len(library(cell)))
        head, table = models[name]
        path = owned(HERE / 'artifacts' / f'{cell}.npz')
        meta = dict(cell=cell, fit_inits=list(range(20)), fit_variants=list(VARIANTS),
            anchors=len(d['init']), input_dimension=207, task_id_input=False, task_dispatch=False,
            row_index='existing retrieval only', config=CONFIGS[name], blend=.5,
            selection_sha256=sha(HERE / 'SELECTION.json'))
        np.savez_compressed(path, table=table, **{'head_'+k:v for k,v in head.items()}, meta_json=json.dumps(meta))
        m = load_control(cell)
        paths = {}
        for variant in VARIANTS:
            te = dataset(cell, variant, 'eval')
            control = control_prediction(m, te)
            paths[variant] = dict(control=metric(control, te),
                selected=metric(prediction(models[name], te), te, control),
                shared_ablation=metric(prediction(models['shared'], te), te, control),
                row_ablation=metric(prediction(models['row1'], te), te, control))
        results[cell] = dict(artifact=str(path), sha256=sha(path), meta=meta, paths=paths)
        dump(HERE / 'results' / f'eval_{cell}.json', results[cell])
        print(cell, {v: round(p['selected']['relative_to_control'],4) for v,p in paths.items()}, flush=True)
    dump(HERE / 'results/evaluation.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        selection=name, fit_inits=list(range(20)), eval_inits=list(range(20,30)), cells=results))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['select', 'evaluate'])
    a = p.parse_args()
    select() if a.action == 'select' else evaluate()
