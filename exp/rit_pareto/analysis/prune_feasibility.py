"""Audit cross-trajectory pruning using each LIBERO suite's production RIT search.

Run with ``python -m exp.rit_pareto.analysis.prune_feasibility``. Outputs are
offline diagnostics, never a replacement library or a closed-loop SR estimate.
The existing SearchStrategy, CacheStorage and InMemoryBackend compute every
pair score; leave-one-trajectory-out diagnostics operate on that exact matrix.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import logging
from pathlib import Path
import pickle
import platform
import time

import h5py
import numpy as np
import torch
import yaml

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import SearchContext, WeightedScoreSumKnnStrategy
from openpi.cache.types import CheckpointID


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _stats(values):
    x = np.asarray(values, dtype=float)
    if not x.size:
        return {'n': 0}
    return dict(n=int(x.size), mean=float(x.mean()), min=float(x.min()),
                p10=float(np.quantile(x, .1)), median=float(np.median(x)),
                p90=float(np.quantile(x, .9)), p95=float(np.quantile(x, .95)), max=float(x.max()))


def _csv(path, rows):
    if rows:
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _strategy(entries, config, top_k=1):
    backend = InMemoryBackend(config['backend']['vector_dims'])
    storage = CacheStorage(backend)
    storage.batch_insert(entries)
    backend.freeze()
    spec = config['checkpoints']['cp1']['search_strategy']
    return WeightedScoreSumKnnStrategy(
        storage, top_k=top_k, step_filter=spec['step_filter'], trajectory_depth=1,
        task_scoped=spec.get('task_scoped', True),
        fusion_weights={k: v['weight'] for k, v in config['keys'].items() if v['enabled']},
        field_similarity=spec['field_similarity'], score_normalization=spec['score_normalization'])


def _context(entry):
    # The production KeyBuilder emits only these enabled fields. Including
    # disabled stored fields would activate backend default weights.
    keys = {k: entry.query_keys[k] for k in ('vision_0', 'vision_1', 'robot_state')}
    return SearchContext(keys, CheckpointID.CP1, entry.step_idx, entry.payload.task_key)


def _prune(pool, scores, remaining, trajectory, threshold):
    # Each deleted point has a DIRECT retained witness. Equal-length points
    # never dominate each other; deleting a point does not delete descendants.
    ordered = sorted(pool, key=lambda i: (remaining[i], i))
    keep, pairs = np.empty(len(pool), dtype=int), []
    size = 0
    for i in ordered:
        active = keep[:size]
        eligible = ((remaining[active] < remaining[i]) & (trajectory[active] != trajectory[i])
                    & (scores[i, active] >= threshold))
        candidates = active[eligible]
        if len(candidates):
            order = np.lexsort((candidates, remaining[candidates], -scores[i, candidates]))
            j = candidates[order[0]]
            pairs.append((i, j))
        else:
            keep[size] = i
            size += 1
    return np.sort(keep[:size]), pairs


def _diagnose(query, winner, baseline, scores, remaining, actions):
    err = np.sqrt(np.mean((actions[query] - actions[winner]) ** 2, axis=(1, 2)))
    old = np.sqrt(np.mean((actions[query] - actions[baseline]) ** 2, axis=(1, 2)))
    score_drop = scores[query, baseline] - scores[query, winner]
    return {
        'n_queries': len(query), 'changed_winner_pct': float(np.mean(winner != baseline) * 100),
        'mean_selected_remaining': float(remaining[winner].mean()),
        'mean_remaining_reduction': float(np.mean(remaining[baseline] - remaining[winner])),
        'mean_action_rmse': float(err.mean()), 'baseline_action_rmse': float(old.mean()),
        'action_rmse_change_pct': float((err.mean() / old.mean() - 1) * 100),
        'action_rmse_worse_pct': float(np.mean(err > old + 1e-7) * 100),
        'gripper_sign_disagreement_pct': float(np.mean(
            np.sign(actions[query, :, 6]) != np.sign(actions[winner, :, 6])) * 100),
        'mean_score_drop': float(score_drop.mean()), 'p95_score_drop': float(np.quantile(score_drop, .95)),
    }


def _bootstrap(trajectories, winners, baseline, remaining, actions):
    # Resample whole query trajectories, preserving repeated-query dependence.
    err = np.sqrt(np.mean((actions - actions[winners]) ** 2, axis=(1, 2)))
    base = np.sqrt(np.mean((actions - actions[baseline]) ** 2, axis=(1, 2)))
    blocks = np.array([[len(v), np.sum(remaining[baseline[v]] - remaining[winners[v]]),
                        err[v].sum(), base[v].sum()] for v in trajectories.values()])
    draws = np.random.default_rng(20260911).integers(len(blocks), size=(4000, len(blocks)))
    sums = blocks[draws].sum(axis=1)
    return dict(remaining_reduction_ci95=np.quantile(sums[:, 1] / sums[:, 0], [.025, .975]).tolist(),
                action_rmse_change_pct_ci95=np.quantile((sums[:, 2] / sums[:, 3] - 1) * 100, [.025, .975]).tolist())


def main():
    """Measure redundancy, prune masks, held-out proxies and CPU search cost."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['libero_spatial', 'libero_10'], default='libero_spatial')
    parser.add_argument('--library', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.library = args.library or Path(f'exp/common/data/cache_artifacts/{args.suite}/cp1_spatial_pool_16.pkl')
    short_name = 'sp' if args.suite == 'libero_spatial' else 'l10'
    args.config = args.config or Path(f'exp/rit_pareto/data/k3_rith/{args.suite}/k3/rith/k3_{short_name}_rith_ir60.yaml')
    args.output = args.output or Path('exp/rit_pareto/analysis/trajectory_prune_20260911') / args.suite
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.ERROR)
    config = yaml.safe_load(args.config.read_text())
    assert config['key_builder']['type'] == 'cp1_spatial_pool_16'
    spec = config['checkpoints']['cp1']['search_strategy']
    assert spec['type'] == 'weighted_score_sum_knn' and spec['step_filter'] == 'all'
    assert spec.get('trajectory_depth', 1) == 1 and spec.get('task_scoped', True)
    assert {k for k, v in config['keys'].items() if v['enabled']} == {'vision_0', 'vision_1', 'robot_state'}
    source_sha = _sha(args.library)
    registry_path = Path(f'exp/data_authority/records/weighted_sum__{args.suite}__cp1_spatial_pool_16.json')
    registry = json.loads(registry_path.read_text())
    export_path = Path(f'exp/rit_pareto/data/k3_rith/{args.suite}/k3/export_record_rith.json')
    export_record = json.loads(export_path.read_text())
    assert source_sha == registry['integrity']['sha256']
    assert export_record['library_pkl'] == config['backend']['in_memory']['preload_path']
    assert export_record['library_sha256'] in {r['sha256'] for r in registry['replicas']}
    with args.library.open('rb') as f:
        artifact = pickle.load(f)
    entries = artifact['entries']
    for entry in entries:
        entry.query_keys = {k: torch.as_tensor(v).float().contiguous() for k, v in entry.query_keys.items()}
        entry.payload.action_chunk = torch.as_tensor(entry.payload.action_chunk).float().contiguous()
    ids = {e.id: i for i, e in enumerate(entries)}
    assert len(ids) == len(entries)
    trajectories, tasks = defaultdict(list), defaultdict(list)
    for i, entry in enumerate(entries):
        trajectories[entry.trajectory_id].append(i)
        tasks[entry.payload.task_key].append(i)
    n = len(entries)
    assert n == registry['content']['entries']
    assert len(trajectories) == registry['content']['trajectories']
    remaining = np.zeros(n, dtype=int)
    trajectory = np.array([e.trajectory_id for e in entries])
    task_name = np.array([e.payload.task_key for e in entries])
    trajectory_rows, h5_rows = [], []
    for traj, members in trajectories.items():
        members.sort(key=lambda i: entries[i].step_idx)
        assert [entries[i].step_idx for i in members] == list(range(len(members)))
        for j, i in enumerate(members):
            assert entries[i].prev_ids == ([] if j == 0 else [entries[members[j - 1]].id])
            assert entries[i].next_ids == ([] if j + 1 == len(members) else [entries[members[j + 1]].id])
            remaining[i] = len(members) - 1 - j
        trajectory_rows.append(dict(trajectory_id=traj, task=task_name[members[0]], points=len(members)))
    for p in sorted(Path(f'exp/common/data/db/libero_cache/{args.suite}').glob('*.h5')):
        with h5py.File(p) as f:
            h5_rows.append(dict(file=p.name, in_library=p.stem in trajectories,
                                success=bool(f.attrs.get('success', False)),
                                num_steps=int(f.attrs.get('num_steps', -1)), task=str(f.attrs.get('task', ''))))
    actions_full = torch.stack([e.payload.action_chunk for e in entries]).numpy()
    actions = actions_full[:, :5, :7]
    strategy = _strategy(entries, config, top_k=n)
    scores = np.full((n, n), -np.inf, dtype=np.float32)
    started = time.perf_counter()
    for i, entry in enumerate(entries):
        results = strategy.search(_context(entry))
        assert len(results) == len(tasks[entry.payload.task_key])
        for hit in results:
            scores[i, ids[hit.id]] = hit.score
        if (i + 1) % 200 == 0:
            print(f'exact production search: {i + 1}/{n}', flush=True)
    same_task = task_name[:, None] == task_name[None, :]
    symmetry_error = float(np.max(np.abs(scores[same_task] - scores.T[same_task])))
    assert symmetry_error < 2e-6
    assert np.all(~np.isfinite(scores[~same_task]))
    cross_traj = trajectory[:, None] != trajectory[None, :]
    nearest_cross = np.max(np.where(same_task & cross_traj, scores, -np.inf), axis=1)
    np.savez_compressed(args.output / 'pair_scores.npz', scores=scores, remaining=remaining,
                        ids=np.array([e.id for e in entries]), trajectories=trajectory, tasks=task_name)
    print('matrix complete', dict(seconds=time.perf_counter() - started,
                                  self_score=_stats(np.diag(scores)), cross_nearest=_stats(nearest_cross)), flush=True)
    thresholds = ([.989, .987, .985, .9825, .98, .975, .97, .95, .90] if args.suite == 'libero_spatial'
                  else [.9995, .999, .9985, .998, .9975, .997, .996, .995, .9925, .99, .985, .98, .975, .97, .95, .90])
    rows, loo_rows, pair_rows, task_rows, masks, prediction_rows = [], [], [], [], {}, []
    query = np.arange(n)
    baseline = np.argmax(np.where(same_task & cross_traj, scores, -np.inf), axis=1)
    for threshold in thresholds:
        label = f'cross_{threshold:g}'
        keep, pairs = _prune(query, scores, remaining, trajectory, threshold)
        removed = np.setdiff1d(query, keep)
        masks[label] = keep
        assert len(keep) + len(pairs) == n and set(removed) == {i for i, j in pairs}
        assert all(j in set(keep) and remaining[j] < remaining[i] and scores[i, j] >= threshold for i, j in pairs)
        assert all(trajectory[i] != trajectory[j] and task_name[i] == task_name[j] for i, j in pairs)
        assert set(np.flatnonzero(remaining == 0)) <= set(keep)
        deleted_set = {entries[i].id for i in removed}
        broken = sum(any(x in deleted_set for x in entries[i].prev_ids + entries[i].next_ids) for i in keep)
        byte_total = sum(sum(v.numel() * v.element_size() for v in e.query_keys.values())
                         + e.payload.action_chunk.numel() * e.payload.action_chunk.element_size()
                         + sum(np.asarray(v).nbytes for v in (e.payload.intermediates or {}).values()) for e in entries)
        ii = np.array([i for i, j in pairs], dtype=int)
        jj = np.array([j for i, j in pairs], dtype=int)
        row = dict(label=label, threshold=threshold, cross_only=True, kept=len(keep), removed=len(removed),
                   removed_pct=float(len(removed) / n * 100),
                   deleted_same_trajectory_pct=float(np.mean(trajectory[ii] == trajectory[jj]) * 100) if len(ii) else 0.,
                   mean_pair_remaining_saved=float(np.mean(remaining[ii] - remaining[jj])) if len(ii) else 0.,
                   pair_action_rmse=float(np.sqrt(np.mean((actions[ii] - actions[jj]) ** 2, axis=(1, 2))).mean()) if len(ii) else 0.,
                   pair_gripper_disagreement_pct=float(np.mean(np.sign(actions[ii, :, 6]) != np.sign(actions[jj, :, 6])) * 100) if len(ii) else 0.,
                   retained_with_dangling_links=broken, array_bytes_total=byte_total)
        rows.append(row)
        for i, j in pairs:
            pair_rows.append(dict(label=label, removed_id=entries[i].id, retained_id=entries[j].id,
                                  task=task_name[i], score=float(scores[i, j]), remaining_removed=int(remaining[i]),
                                  remaining_retained=int(remaining[j]), same_trajectory=bool(trajectory[i] == trajectory[j]),
                                  action_rmse=float(np.sqrt(np.mean((actions[i] - actions[j]) ** 2))),
                                  state_l2=float(torch.linalg.vector_norm(entries[i].query_keys['robot_state'] - entries[j].query_keys['robot_state']))))
        for task, members in tasks.items():
            task_rows.append(dict(label=label, task=task, original=len(members), kept=len(set(members) & set(keep))))
        # The held-out trajectory is removed BEFORE learning a prune mask.
        # Query states/continuations never decide which training points survive.
        winners = np.empty(n, int)
        rng = np.random.default_rng(20260911)
        random_winners = np.empty((20, n), int)
        fold_sizes = []
        for members in trajectories.values():
            task_pool = np.array(tasks[task_name[members[0]]])
            train = np.setdiff1d(task_pool, members)
            selected, _ = _prune(train, scores, remaining, trajectory, threshold)
            winners[members] = selected[np.argmax(scores[np.ix_(members, selected)], axis=1)]
            fold_sizes.append(len(selected) / len(train))
            for seed in range(20):
                random_selected = np.sort(rng.choice(train, size=len(selected), replace=False))
                random_winners[seed, members] = random_selected[np.argmax(scores[np.ix_(members, random_selected)], axis=1)]
        diag = _diagnose(query, winners, baseline, scores, remaining, actions)
        controls = [_diagnose(query, w, baseline, scores, remaining, actions) for w in random_winners]
        diag.update(label=label, threshold=threshold, cross_only=True,
                    mean_fold_retained_fraction=float(np.mean(fold_sizes)),
                    random_rmse_change_pct_mean=float(np.mean([d['action_rmse_change_pct'] for d in controls])),
                    random_remaining_reduction_mean=float(np.mean([d['mean_remaining_reduction'] for d in controls])))
        diag.update(_bootstrap(trajectories, winners, baseline, remaining, actions))
        for i, j in enumerate(winners):
            b = baseline[i]
            prediction_rows.append(dict(label=label, query_id=entries[i].id, trajectory=trajectory[i], task=task_name[i],
                                        baseline_id=entries[b].id, pruned_id=entries[j].id,
                                        baseline_remaining=int(remaining[b]), pruned_remaining=int(remaining[j]),
                                        baseline_score=float(scores[i, b]), pruned_score=float(scores[i, j]),
                                        baseline_action_rmse=float(np.sqrt(np.mean((actions[i] - actions[b]) ** 2))),
                                        pruned_action_rmse=float(np.sqrt(np.mean((actions[i] - actions[j]) ** 2))),
                                        baseline_arm_rmse=float(np.sqrt(np.mean((actions[i, :, :6] - actions[b, :, :6]) ** 2))),
                                        pruned_arm_rmse=float(np.sqrt(np.mean((actions[i, :, :6] - actions[j, :, :6]) ** 2)))))
        loo_rows.append(diag)
        print('sweep', label, row['removed_pct'], diag['mean_remaining_reduction'], diag['action_rmse_change_pct'], flush=True)
    # Recorded winners identify affected original decisions, not counterfactual outcomes.
    exposures = defaultdict(Counter)
    path = Path(f'exp/rit_pareto/data/runs/{args.suite}_hg/per_step.jsonl')
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('phase') != 'eval' or not rec.get('accepted', True):
                continue
            if rec.get('hit_type') in ('FULL_HIT', 'WARM_START'):
                winner = rec['winner_id']
                assert winner in ids
                exposures[rec['yaml_id']][ids[winner]] += 1
    exposure_rows = []
    for label, keep in masks.items():
        removed = set(query) - set(keep)
        for arm, counts in exposures.items():
            exposure_rows.append(dict(label=label, arm=arm, original_hit_decisions=sum(counts.values()),
                                      removed_winner_decisions=sum(counts[i] for i in removed),
                                      affected_pct=100 * sum(counts[i] for i in removed) / sum(counts.values())))
    # Actual warm-cache SearchStrategy timing: same queries, shuffled variant order.
    timed_thresholds = [.987, .985, .98, .975] if args.suite == 'libero_spatial' else [.9985, .998, .997, .995, .99, .98]
    timed_labels = ['baseline'] + [f'cross_{t:g}' for t in timed_thresholds]
    timing_strategies = {label: _strategy(entries if label == 'baseline' else [entries[i] for i in masks[label]], config)
                         for label in timed_labels}
    rng = np.random.default_rng(20260911)
    chosen = rng.choice(n, 100, replace=False)
    contexts = [_context(entries[i]) for i in chosen]
    for s in timing_strategies.values():
        for ctx in contexts:
            s.search(ctx)
    timings, raw_timings = defaultdict(list), []
    for repeat in range(7):
        for label in rng.permutation(timed_labels):
            s = timing_strategies[label]
            for query_id, ctx in zip(chosen, contexts):
                t0 = time.perf_counter_ns()
                s.search(ctx)
                elapsed = (time.perf_counter_ns() - t0) / 1e6
                timings[label].append(elapsed)
                raw_timings.append(dict(label=label, repeat=repeat, query_id=entries[query_id].id, ms=elapsed))
    timing_rows = [dict(label=label, **_stats(values)) for label, values in timings.items()]
    # Match top-k diagnostic scores against the ordinary production top-1 path.
    parity = 0.
    for i in chosen:
        hit = timing_strategies['baseline'].search(_context(entries[i]))[0]
        parity = max(parity, abs(hit.score - float(scores[i].max())))
    assert parity < 2e-6
    direct_checks = []
    validation_thresholds = [.985, .98] if args.suite == 'libero_spatial' else [.9985, .998, .995, .98]
    for task_members in tasks.values():
        members = trajectories[trajectory[task_members[0]]]
        train = np.setdiff1d(task_members, members)
        for threshold in validation_thresholds:
            keep, _ = _prune(train, scores, remaining, trajectory, threshold)
            direct = _strategy([entries[i] for i in keep], config)
            for i in [members[0], members[len(members) // 2], members[-1]]:
                hit = direct.search(_context(entries[i]))[0]
                expected = keep[np.argmax(scores[i, keep])]
                direct_checks.append(dict(query=entries[i].id, cross=True, theta=threshold,
                                          winner_match=hit.id == entries[expected].id,
                                          score_error=abs(hit.score - float(scores[i, expected]))))
    assert all(row['winner_match'] for row in direct_checks)
    direct_error = max(row['score_error'] for row in direct_checks)
    # Changing the candidate matrix size changes float32 reduction rounding.
    # L10 probes retain every winner; the largest observed winning-score
    # difference is 9.42e-6 on a low-score query, far from the prune cut.
    assert direct_error < 1e-5
    (args.output / 'direct_strategy_validation.json').write_text(json.dumps(dict(
        direct_pruned_strategy_queries=len(direct_checks), winner_agreement=1.0,
        max_score_error=direct_error, checks=direct_checks), indent=2) + '\n')
    assert _sha(args.library) == source_sha
    summary = dict(suite=args.suite, library=str(args.library), library_sha256=source_sha, config=str(args.config), config_sha256=_sha(args.config),
                   registry_path=str(registry_path), export_record_path=str(export_path),
                   server_library_sha256=export_record['library_sha256'], registry_and_export_match=True,
                   config_contents=config, n_entries=n, n_trajectories=len(trajectories), n_tasks=len(tasks),
                   trajectory_lengths=_stats([len(v) for v in trajectories.values()]),
                   self_score=_stats(np.diag(scores)), nearest_cross_trajectory_score=_stats(nearest_cross),
                   pair_symmetry_max_error=symmetry_error, top1_score_parity_max_error=parity,
                   all_original_links_verified=True, original_library_unchanged=True,
                   action_dimension_std=actions_full.std(axis=(0, 1)).tolist(),
                   remaining_definition='number of successor entries in the original intact trajectory, excluding current entry',
                   proximity='same task AND different trajectory; exact production WeightedScoreSumKnnStrategy score >= threshold',
                   algorithm='ascending original remaining length; remove only with a strictly shorter retained direct neighbour; equal lengths kept',
                   loo=f'{len(trajectories)} folds: remove query trajectory before pruning; {n} recorded states; no environment rerollout',
                   action_proxy='RMSE over first 5 of 10 predicted actions and first 7 of 32 normalized action dimensions; not a success label',
                   cpu=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                             if line.startswith('model name')), platform.processor()),
                   torch_threads=torch.get_num_threads(), torch_version=torch.__version__,
                   thresholds=thresholds, prune=rows, loo_results=loo_rows, latency=timing_rows,
                   limitations=['no counterfactual closed-loop SR or task duration', 'LOTO queries are successful library trajectories, not independent deployed rollouts',
                                'formal protocol is prune, rerun warmup calibration, then eval; this analysis does not run that protocol',
                                'latency is local CPU search only; excludes key building, GPU inference, network and environment',
                                'physical deletion leaves chain references dangling; an online candidate mask must retain chain semantics'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for name, data in [('prune_sweep', rows), ('leave_one_trajectory_out', loo_rows), ('replacement_pairs', pair_rows),
                       ('per_task_retention', task_rows), ('trajectories', trajectory_rows), ('source_h5_audit', h5_rows),
                       ('recorded_winner_exposure', exposure_rows), ('search_latency', timing_rows),
                       ('leave_one_trajectory_out_predictions', prediction_rows), ('search_latency_samples', raw_timings)]:
        _csv(args.output / f'{name}.csv', data)
    (args.output / 'keep_ids.json').write_text(json.dumps({k: [entries[i].id for i in v] for k, v in masks.items()}, indent=2) + '\n')
    print('DONE', str(args.output), flush=True)


if __name__ == '__main__':
    main()
