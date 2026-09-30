"""Read-only R7 E5 exploration on non-test A recordings; no method fitting.

Run with the CPU/thread/environment prefix documented in PROPOSAL.md.
Only outputs beside this script. No GPU, rollout, or policy import.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
STORE = Path('/home/weiland/trace_runs/offline_search_store/library')
TABLES = Path('/home/weiland/trace_runs/os_closed_loop/r06_c_cal/tables')
COLS = ['uid', 'task_id', 'step', 'Y', 'distance.rms', 'retrieval.rows',
        'retrieval.weights', 'retrieval.dispersion_rms', 'state.normalized']


def stats(a):
    a = np.asarray(a, dtype=float)
    return dict(zip(['q10', 'median', 'q90'], map(float, np.quantile(a, [.1, .5, .9]))))


def rho(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def main():
    summary, all_features = {}, []
    for model in ['pi05', 'groot']:
        for suite in ['l10', 'spatial']:
            for size in [50, 500]:
                cell = f'{model}_{suite}_{size}'
                libname = 'current' if size == 50 else ('bpool_cs' if model == 'pi05' else 'bpool_all')
                lp = STORE / f'{model}_{suite}' / libname
                mf = json.loads((lp / 'manifest.json').read_text())
                load = lambda name: np.load(lp / f'{name}.npy', mmap_mode='r')
                action, episode, nxt, step, success, progress, tid = map(load, ['action', 'episode', 'next', 'step', 'success', 'progress', 'task_id'])
                r, d, g, h = [mf[k] for k in ['exec_steps', 'act_valid_dims', 'gripper_dim', 'H']]
                # Valid dimensions come from metadata, scaling from successful library heads.
                sig = np.std(action[success, :r, :d].reshape(-1, d), axis=0)
                valid = sig > 0
                sig = sig[valid]
                # Exploratory command-mode spans, not semantic grasp/place labels.
                # Midpoint of library gripper extrema; no policy-specific sign assumption.
                grip = action[:, :r, g]
                mid = float((grip.min() + grip.max()) / 2)
                mode = grip >= mid
                spans, episode_switches = [], []
                for ep in np.unique(episode):
                    ix = np.flatnonzero(episode == ep)
                    ix = ix[np.argsort(step[ix])]
                    if not success[ix[0]]:
                        continue
                    z = mode[ix].reshape(-1)
                    cuts = np.flatnonzero(z[1:] != z[:-1]) + 1
                    spans.extend(np.diff(np.r_[0, cuts, len(z)]).tolist())
                    episode_switches.append(len(cuts))
                # Exact sequential successor head rollout; no terminal padding.
                # horizons 3R/4R are exploratory probes, NOT deployed thresholds.
                row_future, row_stable = {}, {}
                for blocks in [2, 3, 4]:
                    rows = np.arange(len(episode))
                    alive = np.ones(len(rows), bool)
                    stable = np.ones(len(rows), bool)
                    initial_mode = mode[:, 0]
                    for b in range(blocks):
                        stable &= alive & np.all(mode[np.maximum(rows, 0)] == initial_mode[:, None], axis=1)
                        if b < blocks - 1:
                            dest = nxt[np.maximum(rows, 0)]
                            ok = dest >= 0
                            safe = np.maximum(dest, 0)
                            ok &= (episode[safe] == episode[np.maximum(rows, 0)])
                            ok &= (tid[safe] == tid[np.maximum(rows, 0)])
                            ok &= (step[safe] == step[np.maximum(rows, 0)] + 1)
                            alive &= ok
                            rows = np.where(alive, dest, -1)
                    row_future[blocks], row_stable[blocks] = alive, stable & alive
                source = TABLES / cell / 'anchors.csv'
                a = pd.read_csv(source, usecols=COLS)
                frows, compute_ms = [], []
                for row in a.to_dict('records'):
                    ix = np.asarray(json.loads(row['retrieval.rows']), int)
                    w = np.asarray(json.loads(row['retrieval.weights']), float)
                    assert len(ix) == 16 and np.isfinite(w).all() and w.sum() > 0
                    assert np.all(tid[ix] == row['task_id']) and np.all(w >= 0)
                    w /= w.sum()
                    tick = perf_counter()
                    acts = np.asarray(action[ix, :h, :d][:, :, valid], float) / sig
                    avg = np.sum(w[:, None, None] * acts, axis=0)
                    perstep = np.sqrt(np.sum(w[:, None, None] * (acts - avg[None])**2, axis=(0, 2)) / len(sig))
                    _, groups = np.unique(episode[ix], return_inverse=True)
                    ew = np.bincount(groups, weights=w)
                    ep_eff = float(1 / np.sum(ew**2))
                    p = float(w @ progress[ix])
                    p_sd = float(np.sqrt(w @ (progress[ix] - p)**2))
                    gm = np.sum(w[:, None] * mode[ix], axis=0)
                    amb = float(np.mean(4 * gm * (1-gm)))
                    rec = {'cell': cell, 'uid': row['uid'], 'task_id': row['task_id'], 'step': row['step'], 'Y': row['Y'],
                           'teacher_rms': row['distance.rms'], 'raw_dispersion': row['retrieval.dispersion_rms'],
                           'dispersion_sigma': float(np.sqrt(np.mean(perstep**2))),
                           'head_dispersion_sigma': float(np.sqrt(np.mean(perstep[:r]**2))),
                           'tail_dispersion_sigma': float(np.sqrt(np.mean(perstep[r:2*r]**2))),
                           'last_block_dispersion_sigma': float(np.sqrt(np.mean(perstep[(h//r-1)*r:(h//r)*r]**2))),
                           'unique_episodes': len(ew), 'ep_eff': ep_eff, 'max_episode_mass': float(ew.max()),
                           'progress_sd': p_sd, 'gripper_split': amb}
                    for b in [2, 3, 4]:
                        rec[f'successor_mass_{b}R'] = float(w @ row_future[b][ix])
                        rec[f'stable_gripper_successor_mass_{b}R'] = float(w @ row_stable[b][ix])
                        rec[f'all_successors_{b}R'] = bool(np.all(row_future[b][ix]))
                        rec[f'all_stable_same_mode_{b}R'] = bool(np.all(row_stable[b][ix]) and np.all(mode[ix, 0] == mode[ix[0], 0]))
                    compute_ms.append((perf_counter()-tick)*1000)
                    frows.append(rec)
                f = pd.DataFrame(frows)
                episodes = f.groupby('uid').agg(Y=('Y', 'first'), n=('step', 'size'), ep_eff=('ep_eff', 'mean'),
                                               rms=('teacher_rms', 'mean'), dispersion=('raw_dispersion', 'mean'))
                correlations = {key: [v for _, grp in f.groupby('uid') if (v := rho(grp[key], grp.teacher_rms)) is not None]
                                for key in ['raw_dispersion', 'dispersion_sigma', 'progress_sd', 'gripper_split', 'ep_eff']}
                res = {'library': str(lp), 'manifest_sha256': hashlib.sha256((lp/'manifest.json').read_bytes()).hexdigest(),
                       'source': str(source), 'source_bytes': source.stat().st_size,
                       'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'H': h, 'R': r, 'D': d, 'library_rows': len(episode), 'library_episodes': len(np.unique(episode)),
                       'library_successful_episodes': len(np.unique(episode[success])),
                       'recording_episodes': len(episodes), 'recording_successes': int(episodes.Y.sum()), 'anchors': len(a),
                       'distinct_episodes': stats(f.unique_episodes), 'effective_episodes': stats(f.ep_eff),
                       'majority_one_episode_anchor_share': float(np.mean(f.max_episode_mass > .5)),
                       'low_eff_lt2_anchor_share': float(np.mean(f.ep_eff < 2)),
                       'gripper_split': stats(f.gripper_split), 'progress_sd': stats(f.progress_sd),
                       'command_mode_span_controls': stats(spans), 'command_switches_per_successful_episode': stats(episode_switches),
                       'same_observation_tail_head_dispersion_ratio': float(f.tail_dispersion_sigma.median() / f.head_dispersion_sigma.median()),
                       'added_feature_compute_ms': stats(compute_ms),
                       'within_episode_spearman_median': {k: float(np.median(v)) if v else None for k,v in correlations.items()},
                       'within_episode_spearman_n': {k:len(v) for k,v in correlations.items()},
                       'future': {str(b*r): {'library_successful_rows_with_successors': float(np.mean(row_future[b][success])),
                                           'library_successful_rows_with_stable_command': float(np.mean(row_stable[b][success])),
                                           'Bval_mean_successor_mass': float(f[f'successor_mass_{b}R'].mean()),
                                           'Bval_mean_stable_command_mass': float(f[f'stable_gripper_successor_mass_{b}R'].mean()),
                                           'Bval_all_successors_anchor_share': float(f[f'all_successors_{b}R'].mean()),
                                           'Bval_all_stable_same_mode_anchor_count': int(f[f'all_stable_same_mode_{b}R'].sum()),
                                           'Bval_all_stable_same_mode_anchor_share': float(f[f'all_stable_same_mode_{b}R'].mean())}
                                  for b in [2,3,4]}}
                summary[cell] = res
                all_features.extend(frows)
                print(cell, 'n=', len(a), 'eps=', len(episodes), 'eff=', round(f.ep_eff.median(), 3),
                      'rhoD=', round(res['within_episode_spearman_median']['raw_dispersion'], 3), flush=True)
    pd.DataFrame(all_features).to_csv(OUT/'anchor_features.csv', index=False)
    result = {'scope': 'Descriptive B-val A recordings, one init per task; no causal value, safety, SR or calibration claim.',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cells': summary}
    (OUT/'evidence.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
