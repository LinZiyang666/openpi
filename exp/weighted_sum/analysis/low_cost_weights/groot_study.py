"""Evaluate corrected GR00T offline objectives and paired low-budget selectors.

Consumes the compact extraction produced by extract_groot.py. Only the original
library is compared to its archived closed-loop grid. The newer W13 library gets
offline diagnostics only. All reruns are CPU computations, with zero new rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
import torch

from exp.weighted_sum.analysis.low_cost_weights.offline_study import (
    FIELDS, calibrate, pair_mse, query_mass, retrieve, simplex,
)
from exp.weighted_sum.analysis.low_cost_weights.backtest import facility_order, race


def backtest(records: list, weights: np.ndarray, winners: np.ndarray, mass: np.ndarray,
             values: dict, index: int, named: dict, repetitions: int = 1000) -> dict:
    """Hide thirty initial states per task while selecting on the other twenty."""
    keys = sorted((r['task'], r['init']) for r in records[0]['episodes'])
    assert len(keys) == len(set(keys)) == 500
    ys = []
    for r in records:
        by_key = {(e['task'], e['init']): e['success'] for e in r['episodes']}
        assert sorted(by_key) == keys
        ys.append([by_key[k] for k in keys])
    y = np.array(ys, dtype=float)
    keys = np.array(keys)
    w = weights[index:]
    ww = winners[index:]
    distances = np.array([(ww != row) @ mass for row in winners[:1225]])
    geometry = np.linalg.norm(weights[:1225, None, :]-w[None, :, :], axis=2)
    uniform = int(np.square(w-1/3).sum(axis=1).argmin())
    jw = np.array(named['J'])
    fixed = {'uniform': uniform, 'J_nearest': int(np.square(w-jw).sum(axis=1).argmin()),
             'listwise_scaled_nearest': int(np.square(w-np.array(named['listwise_scaled7'])).sum(axis=1).argmin()),
             **{f'offline_{k}': int(v[index:].argmin()) for k, v in values.items()}}
    candidates = {
        'J_local_race': np.square(w-jw).sum(axis=1).argsort()[:9],
        'weight_geometry_race': np.array(facility_order(geometry, uniform, 9)),
        'retrieval_behavior_race': np.array(facility_order(distances, uniform, 9)),
        'raw32_shortlist_race': values['raw32_entrymean'][index:].argsort()[:9],
        'valid7_shortlist_race': values['valid7'][index:].argsort()[:9],
        'scaled7_shortlist_race': values['scaled7'][index:].argsort()[:9],
    }
    samples = {k: [] for k in [*fixed, *candidates, 'random9_race', 'full_grid_200', 'full_grid_50']}
    budgets = {}
    rng = np.random.default_rng(20260913)
    for _ in range(repetitions):
        perm = [rng.permutation(np.flatnonzero(keys[:, 0] == t)) for t in range(10)]
        train = np.array([p[:20] for p in perm]).T.flatten()
        test = np.array([p[20:] for p in perm]).T.flatten()
        assert not set(train)&set(test)
        priority = rng.random(len(y))
        choices = {k: (v, 0) for k, v in fixed.items()}
        for k, c in candidates.items():
            choices[k] = race(c, y, train, priority, [(50, 5), (100, 2), (200, 1)])
        choices['random9_race'] = race(rng.choice(len(y), 9, replace=False), y, train, priority,
                                       [(50, 5), (100, 2), (200, 1)])
        for n in [50, 200]:
            choice = int(np.lexsort((priority, -y[:, train[:n]].mean(axis=1)))[0])
            choices[f'full_grid_{n}'] = (choice, len(y)*n)
        for k, (choice, cost) in choices.items():
            samples[k].append((float(y[choice, test].mean()), float(y[choice, train].mean()), choice))
            budgets[k] = cost
    ref = np.array([r[0] for r in samples['full_grid_200']])
    out = {}
    for k, rows in samples.items():
        a = np.array(rows)
        diff = a[:, 0]-ref
        out[k] = {'selection_episodes': budgets[k], 'heldout_episodes': 300,
                  'mean_heldout_sr': float(a[:, 0].mean()), 'mean_train_sr': float(a[:, 1].mean()),
                  'mean_delta_vs_full_grid_pp': float(100*diff.mean()),
                  'delta_split_p10_p90_pp': (100*np.quantile(diff,[.1,.9])).tolist(),
                  'fraction_splits_worse_by_over_1_5pp': float(np.mean(diff < -.015))}
    return {'repetitions': repetitions, 'seed': 20260913, 'methods': out,
            'candidate_weights': {k: w[v].tolist() for k,v in candidates.items()},
            'coverage': {k: float(distances[:,v].min(axis=1).mean()) for k,v in candidates.items()}}


def run(path: Path, output: Path, js: dict) -> None:
    """Keep action-space correction and library-version correction separate."""
    started = time.perf_counter()
    meta = json.loads(path.with_suffix('.json').read_text())
    z = np.load(path)
    acts, task, traj = z['actions'], z['task'], z['traj']
    n = len(acts)
    blocks = []
    for t in np.unique(task):
        ids = z[f'ids_{t}']
        blocks.append((ids, z[f'scores_{t}'], traj[ids, None] != traj[None, ids]))
    mass = query_mass(task, traj)
    valid = acts[..., :7]
    rms = np.maximum(valid.reshape(-1, 7).std(axis=0), .05)
    progress = z['step']/np.maximum(z['length']-1, 1)
    losses = {'raw32': np.sqrt(160*pair_mse(acts)), 'valid7': np.sqrt(35*pair_mse(valid)),
              'scaled7': np.sqrt(pair_mse(valid/rms)), 'valid6': np.sqrt(pair_mse(valid[..., :6])),
              'progress': np.abs(progress[:, None]-progress[None,:]).astype(np.float32)}
    fits = {k: calibrate(blocks, losses[k], mass) for k in ['scaled7', 'valid7', 'progress']}
    jw = np.array([js[f] for f in FIELDS]); jw /= jw.sum()
    named = {'uniform': [1/3]*3, 'J': jw.tolist(),
             **{f'listwise_{k}': f['weights'] for k,f in fits.items()}}
    records = meta.get('closed_loop', [])
    grid = simplex(48)
    weights = np.concatenate([grid, np.array(list(named.values())),
                              np.array([r['weights'] for r in records]).reshape(-1,3)])
    winners = retrieve(blocks, weights, n)
    assert (task[winners] == task[None, :]).all()
    assert (traj[winners] != traj[None, :]).all()
    q = np.arange(n)[None,:]
    values = {k: loss[q,winners] @ mass for k, loss in losses.items()}
    values['raw32_entrymean'] = losses['raw32'][q,winners].mean(axis=1)
    values['valid7_entrymean'] = losses['valid7'][q,winners].mean(axis=1)
    index = len(grid)+len(named)
    sr = np.array([np.mean([e['success'] for e in r['episodes']]) for r in records])
    proxies = {}
    for k, v in values.items():
        best = int(v[:len(grid)].argmin())
        proxies[k] = {'weights': grid[best].tolist(), 'min_loss': float(v[best]),
                      'grid_spread_relative': float(v[:len(grid)].max()/v[:len(grid)].min()-1)}
        if records:
            chosen = int(v[index:].argmin())
            proxies[k].update({'selected_original_setting': records[chosen]['name'],
                              'selected_original_weights': records[chosen]['weights'],
                              'selected_original_sr': float(sr[chosen]),
                              'spearman_with_sr': float(spearmanr(-v[index:], sr).statistic)})
    error = np.square(acts-acts[winners[len(grid)]]).mean(axis=(0,1))
    result = {k: meta[k] for k in ['suite','regime','entries','trajectories','library','library_sha256','normalizers']}
    result.update({'unused_squared_error_share_uniform_retrieval': float(error[7:].sum()/error.sum()),
                   'fits': fits, 'proxies': proxies, 'named_weights': named})
    if records:
        result['grid_max_sr'] = float(sr.max())
        result['grid_cells_in_0_75_to_0_766'] = int(np.sum((sr >= .75)&(sr <= .766)))
        result['backtest'] = backtest(records, weights, winners, mass, values, index, named)
    result['seconds'] = time.perf_counter()-started
    output.mkdir(parents=True, exist_ok=True)
    (output/f'{path.stem}_results.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output/f'{path.stem}_replay.npz', weights=weights, winners=winners,
                        query_mass=mass, hist_start=index, sr=sr, **{f'loss_{k}':v for k,v in values.items()})
    print(json.dumps({'dataset':path.stem,'seconds':result['seconds'],
                      'padding_share':result['unused_squared_error_share_uniform_retrieval'],
                      'proxies':proxies}),flush=True)


def main() -> None:
    """Evaluate the four input bundles with the correct version-specific J values."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    old_j = json.loads((args.input.parent/'groot_J_original.json').read_text())
    w13 = json.loads(Path('exp/libero_groot/config/calibration_normalizers_w13_S3.json').read_text())
    for p in sorted(args.input.glob('*.npz')):
        meta = json.loads(p.with_suffix('.json').read_text())
        suite = meta['suite']
        if meta['regime'] == 'original':
            assert old_j[suite], 'No matching J calibration'
            js = old_j[suite][0]['J']
        else:
            fs = w13[f'{suite}_w13_S3']['fields']
            js = {f: fs[f]['shortlist'][0]['J'] for f in FIELDS}
        run(p, args.output, js)


if __name__ == '__main__':
    main()
