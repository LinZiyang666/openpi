"""R7 E1 exploratory, CPU-only library segmentation. No policy implementation.

Fit inputs: successful library proprioception and executed action heads only.
All thresholds are empirical distributions or the recording/commit resolution.
Use run_analysis.sh for the required environment and affinity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
LIBROOT = Path('/home/weiland/trace_runs/offline_search_store/library')
SCRATCH = Path('/tmp/r7_E1_kinematic_stages')
CELLS = [(m, s, n) for m in ['pi05', 'groot'] for s in ['l10', 'spatial'] for n in [50, 500]]


def cellname(m, s, n):
    return f'{m}_{s}_{n}'


def libpath(m, s, n):
    return LIBROOT / f'{m}_{s}' / ('current' if n == 50 else 'bpool_cs' if m == 'pi05' else 'bpool_all')


def twomeans(x):
    c = np.quantile(x, [0.25, 0.75])
    if c[0] == c[1]:
        c = np.array([x.min(), x.max()])
    for _ in range(100):
        labels = x > c.mean()
        nc = np.array([x[~labels].mean() if (~labels).any() else c[0],
                       x[labels].mean() if labels.any() else c[1]])
        if np.allclose(c, nc):
            break
        c = nc
    return c


def stable_modes(a, threshold):
    g = a > threshold
    # One row is the recording resolution; remove an isolated one-row reversal.
    if len(g) > 2:
        g = np.r_[g[0], (g[:-2].astype(int) + g[1:-1] + g[2:]) >= 2, g[-1]]
    return g.astype(int)


def shortest_partition(z, forced, eps):
    """Exact minimum number of TIME-linear segments between mandatory knots.

    This is an AWE-inspired diagnostic, not a reproduction of its path-distance
    objective. The maximum Euclidean interpolation error is bounded by eps.
    """
    out = [forced[0]]
    maxerr = 0.0
    for start, end in zip(forced[:-1], forced[1:], strict=True):
        n = end - start + 1
        cost = np.full(n, n + 1, dtype=int)
        prev = np.full(n, -1, dtype=int)
        errors = {}
        cost[0] = 0
        for j in range(1, n):
            for i in range(j):
                f = np.arange(j - i + 1)[:, None] / (j - i)
                pred = z[start+i] + f * (z[start+j] - z[start+i])
                err = np.linalg.norm(z[start+i:start+j+1] - pred, axis=1).max()
                errors[i, j] = err
                if err <= eps and cost[i] + 1 < cost[j]:
                    cost[j], prev[j] = cost[i] + 1, i
        j, path = n - 1, [end]
        while j > 0:
            i = prev[j]
            assert i >= 0
            maxerr = max(maxerr, errors[i, j])
            if i > 0:
                path.append(start+i)
            j = i
        out.extend(path[::-1])
    return np.asarray(out), maxerr


def low_runs(v, threshold):
    b = v <= threshold
    edges = np.flatnonzero(np.diff(np.r_[False, b, False]))
    return [a + int(np.argmin(v[a:b])) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def rank(x):
    return rankdata(x, method='average') / len(x)


def fit_cell(m, s, n):
    tick = time.perf_counter()
    cell = cellname(m, s, n)
    p = libpath(m, s, n)
    arr = {k: np.load(p / f'{k}.npy', mmap_mode='r') for k in
           ['rs', 'action', 'episode', 'step', 'task_id', 'success', 'next']}
    manifest = json.loads((p / 'manifest.json').read_text())
    d = manifest['exec_steps']
    rs = np.asarray(arr['rs'][:, :8], dtype=float)
    ag = np.median(arr['action'][:, :d, manifest['gripper_dim']], axis=1)
    good = arr['success'].astype(bool)
    centers = twomeans(ag[good])
    ne = len(rs)
    fields = {k: np.full(ne, np.nan) for k in
              ['stage', 'nstages', 'phase', 'n_events', 'event_near', 'density', 'tightness',
               'difficulty', 'event_end', 'next_boundary', 'phase_fraction', 'support', 'speed']}
    fields['successful'] = good
    fields['task_id'] = np.asarray(arr['task_id'])
    fields['episode'] = np.asarray(arr['episode'])
    fields['step'] = np.asarray(arr['step'])
    taskparams = {}
    summaries, ep_records, stages = [], [], []
    for task in np.unique(arr['task_id']):
        tr = np.flatnonzero((arr['task_id'] == task) & good)
        center = np.median(rs[tr], axis=0)
        scale = np.quantile(rs[tr], .95, axis=0) - np.quantile(rs[tr], .05, axis=0)
        scale[scale <= np.finfo(float).eps] = 1.0
        z8 = (rs - center) / scale
        z = z8[:, :6] / np.sqrt(6)
        episodes = []
        for e in np.unique(arr['episode'][tr]):
            rows = tr[arr['episode'][tr] == e]
            rows = rows[np.argsort(arr['step'][rows])]
            assert np.all(np.diff(arr['step'][rows]) == 1)
            modes = stable_modes(ag[rows], centers.mean())
            events = np.flatnonzero(np.diff(modes)) + 1
            phase = np.cumsum(np.r_[0, np.diff(modes) != 0])
            speed = np.r_[0., np.linalg.norm(np.diff(z[rows], axis=0), axis=1)]
            phasefrac = np.zeros(len(rows))
            phasepaths = {}
            for ph in np.unique(phase):
                ix = np.flatnonzero(phase == ph)
                arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(z[rows[ix]], axis=0), axis=1))]
                phasefrac[ix] = arc / arc[-1] if arc[-1] > 0 else np.linspace(0, 1, len(ix))
                phasepaths[int(ph)] = (ix, arc[-1])
            episodes.append(dict(e=int(e), rows=rows, modes=modes, events=events, phase=phase,
                                 speed=speed, phasefrac=phasefrac, phasepaths=phasepaths))
        speeds = np.concatenate([e['speed'][1:] for e in episodes])
        epsilon = float(np.median(speeds))
        if epsilon <= 0:
            epsilon = float(np.median(speeds[speeds > 0])) if np.any(speeds > 0) else np.finfo(float).eps
        slow = float(np.quantile(speeds, .25))
        taskparams[int(task)] = {'center': center.tolist(), 'scale': scale.tolist(),
                                 'epsilon': epsilon, 'slow': slow, 'n_demos': len(episodes)}
        task_stages = []
        for e in episodes:
            rows, events = e['rows'], e['events']
            stops = [i for i in low_runs(e['speed'][1:], slow)]
            stops = np.array([i+1 for i in stops if i+1 < len(rows)-1], dtype=int)
            forced = np.unique(np.r_[0, events, stops, len(rows)-1]).astype(int)
            knots, err = shortest_partition(z[rows], forced, epsilon)
            half, _ = shortest_partition(z[rows], forced, epsilon/2)
            double, _ = shortest_partition(z[rows], forced, epsilon*2)
            assert err <= epsilon * (1 + 1e-9)
            peers = [o for o in episodes if o['e'] != e['e'] and len(o['events']) == len(events)
                     and o['modes'][0] == e['modes'][0]]
            fields['nstages'][rows] = len(knots)-1
            fields['phase'][rows] = e['phase']
            fields['phase_fraction'][rows] = e['phasefrac']
            fields['n_events'][rows] = len(events)
            fields['speed'][rows] = e['speed']
            fields['event_near'][rows] = (np.min(abs(np.arange(len(rows))[:, None] - events[None, :]), axis=1) <= 1) if len(events) else False
            for k, (a, b) in enumerate(zip(knots[:-1], knots[1:], strict=True)):
                # A boundary is compared at corresponding gripper-event phase/arc.
                ph = e['phase'][b]
                frac = e['phasefrac'][b]
                peer_poses = []
                for o in peers:
                    ix = o['phasepaths'][int(ph)][0]
                    xx = o['phasefrac'][ix]
                    peer_poses.append([np.interp(frac, xx, z[o['rows'][ix], dim]) for dim in range(6)])
                dispersion = float(np.median(np.linalg.norm(np.asarray(peer_poses)-z[rows[b]], axis=1))) if peer_poses else np.nan
                motion = e['phasepaths'][int(ph)][1]
                tightness = motion / (dispersion + epsilon) if peer_poses else np.nan
                event_end = bool(np.any(abs(events-b) <= 1))
                sr = rows[a:b] if k < len(knots)-2 else rows[a:b+1]
                fields['stage'][sr] = k
                fields['density'][sr] = 1/(b-a)
                fields['tightness'][sr] = tightness
                fields['event_end'][sr] = event_end
                fields['support'][sr] = len(peers)
                fields['next_boundary'][sr] = arr['step'][rows[b]] - arr['step'][sr]
                rec = {'cell': cell, 'task': int(task), 'episode': e['e'], 'stage': k,
                       'start_row': int(rows[a]), 'end_row': int(rows[b]), 'duration': int(b-a),
                       'event_end': event_end, 'density': 1/(b-a), 'tightness': tightness,
                       'peer_support': len(peers), 'dispersion': dispersion,
                       'phase': int(ph), 'row_indices': sr.tolist()}
                task_stages.append(rec)
            ep_records.append({'cell': cell, 'task': int(task), 'episode': e['e'], 'rows': len(rows),
                               'events': len(events), 'stops': len(stops), 'stages': len(knots)-1,
                               'stages_half_eps': len(half)-1, 'stages_double_eps': len(double)-1,
                               'compression': len(rows)/len(knots), 'max_reconstruction_error': float(err),
                               'epsilon': epsilon, 'knots': knots.tolist(), 'events_at': events.tolist()})
        # Equal-weight ranks, no evaluation outcomes used. Unsupported precision
        # component gets the neutral median, and is separately marked unknown.
        dense_rank = rank([r['density'] for r in task_stages])
        ti = np.array([r['tightness'] for r in task_stages])
        tight_rank = np.full(len(ti), .5)
        finite = np.isfinite(ti)
        if finite.any(): tight_rank[finite] = rank(ti[finite])
        for i, rec in enumerate(task_stages):
            rec['difficulty'] = float((dense_rank[i]+tight_rank[i]+rec['event_end'])/3)
            fields['difficulty'][rec.pop('row_indices')] = rec['difficulty']
        stages.extend(task_stages)
        eventcounts = [len(e['events']) for e in episodes]
        mode, count = Counter(eventcounts).most_common(1)[0]
        er = [r for r in ep_records if r['cell']==cell and r['task']==task]
        summaries.append({'cell': cell, 'task': int(task), 'episodes': len(episodes),
                          'event_mode': mode, 'event_mode_share': count/len(episodes),
                          'median_events': float(np.median(eventcounts)),
                          'median_stages': float(np.median([r['stages'] for r in er])),
                          'median_compression': float(np.median([r['compression'] for r in er])),
                          'epsilon': epsilon, 'slow': slow})
    np.savez_compressed(HERE / f'labels_{cell}.npz', **fields)
    (HERE / f'calibration_{cell}.json').write_text(json.dumps({'library': str(p), 'task_params': taskparams,
                                                             'gripper_centers': centers.tolist(),
                                                             'fit_seconds': time.perf_counter()-tick}, indent=2))
    print(cell, 'fit_s', round(time.perf_counter()-tick, 2), 'successful_eps',len(ep_records), flush=True)
    return summaries, ep_records, stages


def main():
    all_s, all_e, all_st = [], [], []
    for m, s, n in CELLS:
        ss, es, st = fit_cell(m, s, n)
        all_s.extend(ss); all_e.extend(es); all_st.extend(st)
    pd.DataFrame(all_s).to_csv(HERE/'library_tasks.csv', index=False)
    pd.DataFrame(all_e).to_json(HERE/'library_episodes.json', orient='records', indent=2)
    pd.DataFrame(all_st).to_csv(HERE/'library_stages.csv', index=False)
    report = []
    for cell, es in pd.DataFrame(all_e).groupby('cell', sort=False):
        ts = pd.DataFrame(all_s).query('cell == @cell')
        ls = np.load(HERE/f'labels_{cell}.npz')
        good = ls['successful']
        report.append({'cell':cell,'episodes':len(es),'rows':int(es.rows.sum()),
                       'median_events':float(es.events.median()), 'median_stages':float(es.stages.median()),
                       'median_stages_half_eps':float(es.stages_half_eps.median()),
                       'median_stages_double_eps':float(es.stages_double_eps.median()),
                       'median_compression':float(es.compression.median()),
                       'mean_task_event_mode_share':float(ts.event_mode_share.mean()),
                       'event_row_share':float(ls['event_near'][good].mean()),
                       'event_end_stage_row_share':float(ls['event_end'][good].mean()),
                       'no_peer_row_share':float(np.mean(ls['support'][good]==0)),
                       'median_peer_support':float(np.median(ls['support'][good]))})
    pd.DataFrame(report).to_csv(HERE/'library_summary.csv', index=False)
    print(pd.DataFrame(report).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
