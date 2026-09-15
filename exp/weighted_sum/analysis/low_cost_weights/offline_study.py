"""Measure inexpensive modality-weight selectors against archived closed-loop grids.

This analysis never changes serving code or source artifacts. It uses task-scoped,
leave-one-trajectory-out retrieval, production score normalizers, several explicitly
labelled offline losses, and a nonnegative listwise calibration with three fitted
coefficients. Closed-loop outcomes are read only after offline candidates are fixed.
Dependencies: numpy, scipy, torch, and the existing fusion-theory artifact reader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr
import torch
import yaml

from exp.weighted_sum.analysis.fusion_theory import fusion_theory_common as C
from openpi.cache.components.score_normalizers import build_field_normalizers


FIELDS = C.FIELDS


def simplex(n: int) -> np.ndarray:
    """Return the full closed simplex, including all edges and vertices."""
    return np.array([(a / n, b / n, (n-a-b) / n)
                     for a in range(n+1) for b in range(n-a+1)], dtype=np.float32)


def query_mass(task: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """Weight tasks equally, then trajectories equally, then their query steps."""
    mass = np.zeros(len(task), dtype=np.float64)
    tasks = np.unique(task)
    for t in tasks:
        ts = np.unique(traj[task == t])
        for tr in ts:
            ids = np.flatnonzero((task == t) & (traj == tr))
            mass[ids] = 1 / (len(tasks) * len(ts) * len(ids))
    return mass


def retrieve(blocks: list, weights: np.ndarray, n: int) -> np.ndarray:
    """Compute hard top-1 winners in task blocks with trajectory exclusions."""
    out = np.empty((len(weights), n), dtype=np.int32)
    for ids, scores, mask in blocks:
        for start in range(0, len(weights), 32):
            fused = (weights[start:start+32] @ scores.reshape(3, -1)).reshape(-1, len(ids), len(ids))
            fused[:, ~mask] = -np.inf
            out[start:start+len(fused), ids] = ids[fused.argmax(axis=2)]
    return out


def calibrate(blocks: list, loss: np.ndarray, mass: np.ndarray) -> dict:
    """Fit three nonnegative listwise coefficients to fixed soft action targets.

    Each query's best action-error decile is labelled acceptable. Uniform targets
    over that decile avoid inventing a unique correct demonstration. The convex
    cross-entropy has a fixed small ridge penalty; no rollout score is consulted.
    """
    prepared = []
    for ids, score, mask in blocks:
        err = np.where(mask, loss[np.ix_(ids, ids)], np.nan)
        cut = np.nanquantile(err, .1, axis=1)
        target = mask & (err <= cut[:, None])
        target = target / target.sum(axis=1, keepdims=True)
        target_s = np.einsum('mij,ij->mi', score, target)
        prepared.append((ids, score.astype(np.float64), mask, target_s))

    def objective(beta):
        value = 1e-5 * np.dot(beta, beta) / 2
        grad = 1e-5 * beta
        for ids, score, mask, target_s in prepared:
            logits = np.einsum('m,mij->ij', beta, score)
            logits[~mask] = -np.inf
            denom = logsumexp(logits, axis=1)
            prob = np.exp(logits - denom[:, None])
            value += np.dot(mass[ids], denom - beta @ target_s)
            grad += (np.einsum('mij,ij->mi', score, prob) - target_s) @ mass[ids]
        return value, grad

    started = time.perf_counter()
    fit = minimize(objective, np.full(3, 10.), jac=True, bounds=[(0, None)]*3,
                   method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
    if not fit.success or fit.x.sum() <= 0:
        raise RuntimeError(f'Listwise fit failed: {fit.message}')
    return {'weights': (fit.x / fit.x.sum()).tolist(), 'beta': fit.x.tolist(),
            'seconds': time.perf_counter()-started, 'iterations': int(fit.nit),
            'objective': float(fit.fun)}


def pair_mse(x: np.ndarray) -> np.ndarray:
    """Pairwise mean square error, clipping only numerical roundoff."""
    x = x.reshape(len(x), -1).astype(np.float64)
    norm = np.square(x).sum(axis=1)
    return np.maximum((norm[:, None]+norm[None, :]-2*x@x.T)/x.shape[1], 0).astype(np.float32)


def historical(suite: str) -> list[dict]:
    """Load unique first-observed settings; recover exact weights from their YAML."""
    rows = list(csv.DictReader(Path(f'exp/weighted_sum/data/{suite}/phase2/all_results.csv').open()))
    rows = [r for r in rows if r['keybuilder'] == 'cp1_spatial_pool_16' and r['normalizer'] == 'zscore']
    order = {'baseline': 0, 'r1': 1, 'r2': 2, 'r3': 3}
    rows.sort(key=lambda r: (order.get(r['stage'], 99), r['yaml_id']))
    configs = {}
    for p in sorted(Path('exp/weighted_sum/config').rglob('*.yaml')):
        if suite in p.parts:
            configs.setdefault(p.stem, []).append(p)
    unique = {}
    for row in rows:
        key = row['yaml_id']
        if key in unique:
            continue
        # CSV percentages are rounded and can silently change the winning entry.
        paths = configs.get(key, [])
        if not paths:
            raise FileNotFoundError(f'Exact weight YAML missing: {key}')
        preferred = 'phase2' if row['stage'] == 'baseline' else f"round_{row['stage'][1:]}"
        paths.sort(key=lambda p: (preferred not in p.parts, str(p)))
        path = paths[0]
        raw = yaml.safe_load(path.read_text())
        w = np.array([raw['keys'].get(f, {}).get('weight', 0) for f in FIELDS])
        w /= w.sum()
        unique[key] = {'yaml_id': key, 'stage': row['stage'], 'weights': w.tolist(),
                       'sr': float(row['success_rate']), 'n': int(row['n']), 'config': str(path)}
    return list(unique.values())


def run(suite: str, output: Path, cache: Path) -> None:
    """Run one suite and save all candidates, losses, and retrospective comparisons."""
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    library = Path(f'exp/common/data/cache_artifacts/{suite}/cp1_spatial_pool_16.pkl')
    art = C.load_artifact(suite, 'cp1_spatial_pool_16', cache, device='cpu')
    with library.open('rb') as fh:
        original = pickle.load(fh)
    entries = original['entries']
    acts = np.stack([np.asarray(e.payload.action_chunk) for e in entries])[:, :5]
    assert np.allclose(acts, art.actions.reshape(art.n, -1, 32)[:, :5])
    states = np.stack([np.asarray(e.query_keys['robot_state']) for e in entries])
    del original, entries
    config_path = Path(f'exp/ablation_study/cache_prune/config/search_{suite}.yaml')
    config = yaml.safe_load(config_path.read_text())
    ss = config['checkpoints']['cp1']['search_strategy']
    norm = build_field_normalizers([(f, 1., ss['field_similarity'][f]) for f in FIELDS],
                                   ss['score_normalization'], ss['field_similarity'])
    blocks = []
    max_parity = 0.
    for t in np.unique(art.task):
        ids = np.flatnonzero(art.task == t)
        scores = []
        for f in FIELDS:
            raw = art.raw[f][np.ix_(ids, ids)].astype(np.float32)
            normalized = norm[f](torch.from_numpy(raw)).numpy()
            params = ss['score_normalization']['fields'][f]['params']
            ours = C.apply_zscore_tanh(raw, C.SIM_TYPE[f], params)
            max_parity = max(max_parity, float(np.abs(ours-normalized).max()))
            scores.append(normalized)
        mask = art.traj[ids, None] != art.traj[None, ids]
        assert mask.any(axis=1).all()
        blocks.append((ids, np.stack(scores), mask))
    mass = query_mass(art.task, art.traj)
    assert abs(mass.sum()-1) < 1e-10
    q = np.arange(art.n)
    progress = art.step_idx / np.maximum(art.traj_len-1, 1)
    remaining = art.traj_len-1-art.step_idx
    ns = json.loads(Path('assets/pi05_libero/physical-intelligence/libero/norm_stats.json').read_text())['norm_stats']['actions']
    lo, hi = np.array(ns['q01'])[:7], np.array(ns['q99'])[:7]
    wire = (acts[..., :7]+1)/2*(hi-lo+1e-6)+lo
    # Pi0.5's shared LIBERO client forwards the seven outputs unchanged.
    # GR00T has a separate adapter with a different gripper conversion.
    valid = acts[..., :7]
    rms = np.maximum(valid.reshape(-1, 7).std(axis=0), .05)
    losses = {
        'raw32_l2': np.sqrt(32*5*pair_mse(acts)),
        'valid7_l2': np.sqrt(7*5*pair_mse(valid)),
        'valid7_scaled_l2': np.sqrt(pair_mse(valid/rms)),
        'wire7_l2': np.sqrt(pair_mse(wire)),
        'wire6_l2': np.sqrt(pair_mse(wire[..., :6])),
        'displacement_l2': np.sqrt(pair_mse(wire[..., :3].sum(axis=1))),
        'progress_l1': np.abs(progress[:, None]-progress[None, :]).astype(np.float32),
        'remaining_l1': np.abs(remaining[:, None]-remaining[None, :]).astype(np.float32),
        'state_l2': np.sqrt(pair_mse(states[:, :8])),
    }
    calibration = {name: calibrate(blocks, losses[name], mass)
                   for name in ['valid7_scaled_l2', 'wire7_l2', 'progress_l1']}
    grid = simplex(48)
    J = json.loads(Path(f'exp/weighted_sum/data/{suite}/phase1/calibration_normalizers.json').read_text())['cp1_spatial_pool_16']['fields']
    jw = np.array([J[f]['shortlist'][0]['J'] for f in FIELDS]); jw /= jw.sum()
    named = {'uniform': [1/3]*3, 'J': jw.tolist(),
             **{f'listwise_{k}': v['weights'] for k, v in calibration.items()}}
    history = historical(suite)
    weights = np.concatenate([grid, np.array(list(named.values())), np.array([r['weights'] for r in history])])
    winners = retrieve(blocks, weights, art.n)
    for ids, _, _ in blocks:
        assert (art.task[winners[:, ids]] == art.task[ids]).all()
        assert (art.traj[winners[:, ids]] != art.traj[ids]).all()
    values = {}
    for key, pair_loss in losses.items():
        selected = pair_loss[q[None, :], winners]
        values[key] = selected @ mass
        if key == 'raw32_l2':
            values['raw32_l2_entrymean'] = selected.mean(axis=1)
    hist_start = len(grid)+len(named)
    sr = np.array([r['sr'] for r in history])
    proxies = {}
    for key, vals in values.items():
        best = int(vals[:len(grid)].argmin())
        obs_best = int(vals[hist_start:].argmin())
        proxies[key] = {'weights': grid[best].tolist(), 'loss': float(vals[best]),
                        'spearman_with_sr': float(spearmanr(-vals[hist_start:], sr).statistic),
                        'best_observed_setting': history[obs_best],
                        'best_observed_regret_pp': float(100*(sr.max()-sr[obs_best]))}
    candidates = {}
    for i, (name, w) in enumerate(named.items(), len(grid)):
        # Nearest historical policy is diagnostic, not an evaluation of the new w.
        disagreement = (winners[hist_start:] != winners[i]) @ mass
        near = int(disagreement.argmin())
        candidates[name] = {'weights': w, 'losses': {k: float(v[i]) for k, v in values.items()},
                            'nearest_observed_policy': history[near],
                            'nearest_policy_disagreement': float(disagreement[near])}
    # Quantify noise in the coordinates discarded by the actual robot interface.
    ww = winners[len(grid)]
    err = np.square(acts - acts[ww]).mean(axis=(0, 1))
    result = {'suite': suite, 'library': str(library), 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
              'normalizer_config': str(config_path), 'queries': art.n,
              'tasks': len(np.unique(art.task)), 'trajectories': len(np.unique(art.traj)),
              'normalizer_parity_max_abs': max_parity,
              'unused_dims_squared_error_share_uniform': float(err[7:].sum()/err.sum()),
              'per_dim_squared_error_uniform': err.tolist(), 'calibration': calibration,
              'candidates': candidates, 'proxies': proxies,
              'historical_max_sr': float(sr.max()), 'historical_settings': len(history),
              'seconds': time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'historical.json').write_text(json.dumps(history, indent=2)+'\n')
    np.savez_compressed(output/'replay.npz', weights=weights, winners=winners, query_mass=mass,
                        task=art.task, traj=art.traj, hist_start=hist_start,
                        **{f'loss_{k}': v for k, v in values.items()})
    print(json.dumps({'suite': suite, 'seconds': result['seconds'],
                      'historical_settings': len(history), 'output': str(output)}), flush=True)


def main() -> None:
    """Run the selected suites with bounded CPU parallelism."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', choices=['libero_spatial', 'libero_10'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    run(args.suite, args.output, args.cache)


if __name__ == '__main__':
    main()
