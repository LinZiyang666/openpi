"""Retrospectively replay paired, low-budget selection on a complete episode table.

Uses only the original LIBERO-10 coarse grid. For every repetition, five initial
states per task are used for selection and five different states for evaluation.
The repetitions reuse one archive; their quantiles are split sensitivity, not
independent experimental confidence intervals. Candidate sets use no outcomes.
Dependencies: numpy and the output of offline_study.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def facility_order(distance: np.ndarray, initial: int, count: int) -> list[int]:
    """Greedily minimize mean distance from all sampled policies to representatives."""
    chosen = [initial]
    nearest = distance[:, initial].copy()
    while len(chosen) < count:
        gain = np.minimum(nearest[:, None], distance).mean(axis=0)
        gain[chosen] = np.inf
        nxt = int(gain.argmin())
        chosen.append(nxt)
        nearest = np.minimum(nearest, distance[:, nxt])
    return chosen


def episode_table(records: list, path: Path) -> tuple:
    """Load real success and failure outcomes with complete, aligned episode keys."""
    by_yaml = {r['yaml_id']: {} for r in records}
    duplicates = 0
    for line in path.open():
        row = json.loads(line)
        if row['yaml_id'] not in by_yaml:
            continue
        assert row['phase'] == 'eval'
        # Legacy journal uses 'failed' for normal unsuccessful rollouts, not
        # just transport failures. Filtering to status='done' creates SR=1.
        assert (row['status'], row['success']) in [('done', True), ('failed', False)]
        _, _, task, episode = row['task_uid'].rsplit(':', 3)
        key = (int(task), int(episode))
        previous = by_yaml[row['yaml_id']].get(key)
        if previous is not None:
            duplicates += 1
            assert previous == row['success'], 'Conflicting repeated outcome'
        by_yaml[row['yaml_id']][key] = row['success']
    keys = sorted(next(iter(by_yaml.values())))
    assert len(keys) == 100
    assert all(sorted(x) == keys for x in by_yaml.values())
    y = np.array([[by_yaml[r['yaml_id']][k] for k in keys] for r in records], dtype=float)
    assert np.allclose(y.mean(axis=1), [r['sr'] for r in records])
    return y, np.array(keys), duplicates


def race(candidates: np.ndarray, y: np.ndarray, train: np.ndarray,
         tie_priority: np.ndarray, stages=((20, 5), (40, 2), (50, 1))) -> tuple[int, int]:
    """Spend 300 paired episodes: 9x20, then 5x20 more, then 2x10 more."""
    active = np.array(candidates)
    cost, previous = 0, 0
    for n, keep in stages:
        cost += len(active)*(n-previous)
        scores = y[active][:, train[:n]].mean(axis=1)
        ranking = np.lexsort((tie_priority[active], -scores))
        active = active[ranking[:keep]]
        previous = n
    return int(active[0]), cost


def main() -> None:
    """Compare candidate designs at a matched selection budget, keeping test hidden."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repetitions', type=int, default=1000)
    args = ap.parse_args()
    started = time.perf_counter()
    history = json.loads((args.data/'historical.json').read_text())
    z = np.load(args.data/'replay.npz')
    base_idx = np.array([i for i, r in enumerate(history) if r['stage'] == 'baseline'])
    records = [history[i] for i in base_idx]
    w = np.array([r['weights'] for r in records])
    journal = Path('exp/weighted_sum/data/libero_10/phase2/journal.jsonl')
    y, keys, duplicates = episode_table(records, journal)
    hist_start = int(z['hist_start'])
    selected = hist_start+base_idx
    ww = z['winners'][selected]
    mass = z['query_mass']
    distances = np.array([((ww != row) @ mass) for row in z['winners'][:1225]])
    geometry = np.linalg.norm(z['weights'][:1225, None, :]-w[None, :, :], axis=2)
    summary = json.loads((args.data/'summary.json').read_text())
    unif = int(np.square(w-1/3).sum(axis=1).argmin())
    jw = np.array(summary['candidates']['J']['weights'])
    jnear = np.square(w-jw).sum(axis=1).argsort()
    lw = np.array(summary['candidates']['listwise_valid7_scaled_l2']['weights'])
    fixed = {
        'uniform_nearest': unif,
        'J_nearest': int(jnear[0]),
        'listwise_nearest': int(np.square(w-lw).sum(axis=1).argmin()),
        'raw32_offline': int(z['loss_raw32_l2_entrymean'][selected].argmin()),
        'wire6_offline': int(z['loss_wire6_l2'][selected].argmin()),
    }
    candidates = {
        'J_local_race': jnear[:9],
        'weight_geometry_race': np.array(facility_order(geometry, unif, 9)),
        'retrieval_behavior_race': np.array(facility_order(distances, unif, 9)),
        'raw32_shortlist_race': z['loss_raw32_l2_entrymean'][selected].argsort()[:9],
        'wire6_shortlist_race': z['loss_wire6_l2'][selected].argsort()[:9],
    }
    methods = [*fixed, *candidates, 'random9_race', 'full_grid_20', 'full_grid_50']
    samples = {k: [] for k in methods}
    budget = {}
    rng = np.random.default_rng(20260913)
    for rep in range(args.repetitions):
        perm = [rng.permutation(np.flatnonzero(keys[:, 0] == t)) for t in range(10)]
        assert all(len(x) == 10 for x in perm)
        # Interleave tasks: every prefix of length 10 is exactly task-balanced.
        train = np.array([x[:5] for x in perm]).T.flatten()
        test = np.array([x[5:] for x in perm]).T.flatten()
        assert not set(train) & set(test)
        priority = rng.random(len(records))
        choices = {k: (v, 0) for k, v in fixed.items()}
        for k, c in candidates.items():
            choices[k] = race(c, y, train, priority)
        choices['random9_race'] = race(rng.choice(len(y), 9, replace=False), y, train, priority)
        for n in [20, 50]:
            scores = y[:, train[:n]].mean(axis=1)
            choices[f'full_grid_{n}'] = (int(np.lexsort((priority, -scores))[0]), len(y)*n)
        for method, (choice, cost) in choices.items():
            samples[method].append({'test_sr': float(y[choice, test].mean()),
                                    'train_sr': float(y[choice, train].mean()),
                                    'full_table_sr': float(y[choice].mean()), 'choice': choice})
            budget[method] = cost
    reference = np.array([r['test_sr'] for r in samples['full_grid_50']])
    out = {}
    for method, rows in samples.items():
        values = np.array([r['test_sr'] for r in rows])
        full = np.array([r['full_table_sr'] for r in rows])
        out[method] = {'selection_episodes': budget[method], 'heldout_episodes': 50,
                       'mean_heldout_sr': float(values.mean()),
                       'mean_delta_vs_full_grid_pp': float(100*(values-reference).mean()),
                       'delta_split_p10_p90_pp': (100*np.quantile(values-reference,[.1,.9])).tolist(),
                       'mean_full_table_regret_pp': float(100*(y.mean(axis=1).max()-full).mean()),
                       'mean_train_sr': float(np.mean([r['train_sr'] for r in rows]))}
    coverage = {k: {'mean_disagreement': float(distances[:, v].min(axis=1).mean()),
                    'max_disagreement': float(distances[:, v].min(axis=1).max()),
                    'weights': w[v].tolist()} for k, v in candidates.items()}
    result = {'source': str(journal), 'sha256': hashlib.sha256(journal.read_bytes()).hexdigest(),
              'settings': len(y), 'episodes_per_setting': 100, 'duplicates': duplicates,
              'repetitions': args.repetitions, 'seed': 20260913,
              'interpretation': 'Exploratory retrospective splits of one fixed table, not new rollout validation.',
              'methods': out, 'coverage': coverage, 'seconds': time.perf_counter()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'settings': len(y), 'seconds': result['seconds'], 'methods': out}, indent=2))


if __name__ == '__main__':
    main()
