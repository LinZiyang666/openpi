"""Read-only CPU audit for the 2026-10-15 failed-library ablation.

No fits, views, studies, campaigns, servers, or workers are created. The owner
requires stopping when source facts prevent the binding experiment design.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
R17 = Path('/data/r17_library')
R10 = REPO / 'exp/offline_search/rounds/r10'
LEGACY = Path('/home/weiland/trace_runs/offline_search_store/library')
MW_SIZES = (250, 500, 1000, 1500, 2000, 2500)
LIBERO_SIZES = (50, 100, 200, 300, 400, 500)


def read(path):
    return json.loads(Path(path).read_text())


def summarize(episodes, selected, *, id_key, task_key):
    """Filter each exact collected cohort without resampling or topping up."""
    labels = {e[id_key]: e for e in episodes}
    chosen = [labels[i] for i in selected]
    kept = [e for e in chosen if e['success']]
    counts = Counter(e[task_key] for e in kept)
    tasks = sorted({e[task_key] for e in chosen})
    return dict(collected=len(chosen), failed=len(chosen) - len(kept), kept=len(kept),
                kept_episode_ids=[e[id_key] for e in kept],
                successes_per_task={str(t): counts[t] for t in tasks},
                empty_tasks=[t for t in tasks if counts[t] == 0],
                singleton_tasks=[t for t in tasks if counts[t] == 1],
                two_episode_tasks=[t for t in tasks if counts[t] == 2])


def audit():
    cells = []
    for domain, models, suites, sizes in (
        ('mt50', ('pi05', 'groot', 'dp', 'fastwam'), ('mt50',), MW_SIZES),
        ('libero', ('dp', 'fastwam', 'cosmos'), ('libero_spatial', 'libero_10'), LIBERO_SIZES),
    ):
        for model in models:
            for suite in suites:
                parent = R17 / domain / model
                if domain == 'libero':
                    parent /= suite
                parent /= 'bpool'
                episodes = read(parent / 'READY.json')['metadata']['episodes']
                for size in sizes:
                    view = parent / 'subsets' / f'size_{size // 50 if domain == "mt50" else size}' / 'VIEW.json'
                    selected = read(view)['episode_ids']
                    cells.append(dict(model=model, suite=suite, size=size, source='R17',
                                      parent=str(parent), selection=str(view),
                                      labels=str(parent / 'READY.json') + '#metadata.episodes',
                                      **summarize(episodes, selected, id_key='library_episode_id', task_key='task_name')))
    for model, library in (('pi05', 'bpool_cs'), ('groot', 'bpool_all')):
        for short, suite in (('spatial', 'libero_spatial'), ('l10', 'libero_10')):
            parent = LEGACY / f'{model}_{short}' / library
            episodes = read(parent / 'episodes.json')
            for size in LIBERO_SIZES:
                selection = R10 / 'subsets' / f'{model}_{short}_{size}.json'
                selected = sorted(i for ids in read(selection)['episode_ids_by_task'].values() for i in ids)
                cells.append(dict(model=model, suite=suite, size=size, source='R10',
                                  parent=str(parent), selection=str(selection),
                                  row_indices=str(selection.with_suffix('.npy')),
                                  original_fit=str(R10 / 'artifacts' / f'r10_{model}_{short}_{size}_A.pkl'),
                                  run_fit=f'/home/weiland/trace_runs/os_closed_loop/r10_size_{model}/fits/r10_{model}_{short}_{size}_A.pkl',
                                  labels=str(parent / 'episodes.json'),
                                  **summarize(episodes, selected, id_key='episode_id', task_key='task_id')))
    totals = defaultdict(lambda: dict(collected_parent_episodes=0, kept_parent_episodes=0,
                                    failed_parent_episodes=0, intended_filtered_test_episodes=0,
                                    intended_simulated_test_episodes=0, intended_automatic_failures=0))
    for cell in cells:
        total = totals[cell['model']]
        scored = 2500 if cell['suite'] == 'mt50' else 500
        automatic = 50 * len(cell['empty_tasks'])
        total['intended_filtered_test_episodes'] += scored
        total['intended_simulated_test_episodes'] += scored - automatic
        total['intended_automatic_failures'] += automatic
        if cell['size'] == (2500 if cell['suite'] == 'mt50' else 500):
            for target, source in (('collected_parent_episodes', 'collected'),
                                   ('kept_parent_episodes', 'kept'), ('failed_parent_episodes', 'failed')):
                total[target] += cell[source]
    return dict(status='COUNTS_ONLY',
                empty_task_rule='Do not simulate empty tasks; score all 50 official test inits as failures. Secondary comparison uses only non-empty tasks in both arms.',
                cells=cells, totals=dict(totals))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE / 'COUNTS.json')
    args = p.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], cells=len(result['cells']), totals=result['totals']), indent=2))


if __name__ == '__main__':
    main()
