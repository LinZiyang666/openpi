"""Small read-only R8 E1 audit; no serving imports, simulator, or model execution."""
from pathlib import Path
import csv
import json
import time

import h5py
import numpy as np

OUT = Path('/tmp/r8_E1_segmentation')
STORE = Path('/home/weiland/trace_runs/offline_search_store/library')
P3 = Path('/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot')
TRACE = Path('/home/weiland/trace_runs/dual_20260923/runs')


def trace_inventory():
    result = []
    for arm in sorted(TRACE.glob('tr_*')):
        path = sorted((arm / 'trace').glob('*/*.h5'))[0]
        with h5py.File(path, 'r') as f:
            steps = sorted(k for k in f if k.startswith('step_'))
            selected = [steps[i] for i in sorted({0, len(steps)//2, len(steps)-1})]
            datasets = set()
            for step in selected:
                f[step].visititems(lambda k, v: datasets.add(k) if isinstance(v, h5py.Dataset) else None)
            physical = [k for k in sorted(datasets) if any(x in k for x in
                        ('body_xpos', 'contact', 'sim_state', 'object_pos', 'qpos'))]
            result.append(dict(arm=arm.name, file=str(path), steps_checked=selected,
                               physical_datasets=physical, datasets=sorted(datasets)))
    return result


def library_audit():
    result = []
    for model in ('pi05', 'groot'):
        for suite in ('l10', 'spatial'):
            for size in (50, 500):
                path = STORE / f'{model}_{suite}' / ('current' if size == 50 else
                        'bpool_cs' if model == 'pi05' else 'bpool_all')
                a = {k: np.load(path / f'{k}.npy', mmap_mode='r') for k in
                     ('rs', 'action', 'success', 'episode', 'task_id', 'step')}
                manifest = json.loads((path / 'manifest.json').read_text())
                commands = np.median(a['action'][:, :manifest['exec_steps'], manifest['gripper_dim']], axis=1)
                good = np.asarray(a['success'], dtype=bool)
                centers = np.quantile(commands[good], [.25, .75])
                if centers[0] == centers[1]:
                    centers = np.array([commands[good].min(), commands[good].max()])
                for _ in range(100):
                    high = commands[good] > centers.mean()
                    vals = commands[good]
                    new = np.array([vals[~high].mean() if (~high).any() else centers[0],
                                    vals[high].mean() if high.any() else centers[1]])
                    if np.allclose(new, centers):
                        break
                    centers = new
                n_rows = n_event = n_slow = n_slow_interior = n_episodes = 0
                event_counts = []
                for task in np.unique(a['task_id']):
                    ix = np.flatnonzero(good & (a['task_id'] == task))
                    rs = np.asarray(a['rs'][ix, :6], dtype=float)
                    scale = np.quantile(rs, .95, axis=0) - np.quantile(rs, .05, axis=0)
                    scale[scale <= np.finfo(float).eps] = 1.
                    records = []
                    for ep in np.unique(a['episode'][ix]):
                        rows = ix[a['episode'][ix] == ep]
                        rows = rows[np.argsort(a['step'][rows])]
                        assert np.all(np.diff(a['step'][rows]) == 1)
                        mode = (commands[rows] > centers.mean()).astype(int)
                        if len(mode) > 2:
                            mode[1:-1] = (mode[:-2] + mode[1:-1] + mode[2:]) >= 2
                        events = np.flatnonzero(np.diff(mode)) + 1
                        near = (np.min(abs(np.arange(len(rows))[:, None] - events), axis=1) <= 1
                                if len(events) else np.zeros(len(rows), dtype=bool))
                        speed = np.linalg.norm(np.diff(np.asarray(a['rs'][rows, :6]), axis=0) / scale, axis=1)
                        records.append((speed, near[1:]))
                        n_episodes += 1
                        event_counts.append(len(events))
                    threshold = np.quantile(np.concatenate([x[0] for x in records]), .25)
                    for speed, near in records:
                        slow = speed <= threshold
                        n_rows += len(speed)
                        n_event += int(near.sum())
                        n_slow += int(slow.sum())
                        n_slow_interior += int((slow & ~near).sum())
                result.append(dict(cell=f'{model}_{suite}_{size}', successful_episodes=n_episodes,
                    noninitial_rows=n_rows, event_near_rows=n_event, slow_rows=n_slow,
                    slow_interior_rows=n_slow_interior, slow_interior_share=n_slow_interior/n_slow,
                    event_occupancy=n_event/n_rows, median_events=float(np.median(event_counts))))
    return result


def control_audit():
    result = []
    for model in ('pi05', 'groot'):
        for suite in ('l10', 'sp'):
            cell = f'{model}_{suite}_50'
            arm = f'r6p3v2_{cell}_A_r0'
            with (P3 / 'tables' / cell / 'episodes.csv').open() as f:
                episode = next(r for r in csv.DictReader(f) if r['arm'] == arm and
                               r['task_id'] == '0' and r['init'] == '0')
            files = sorted((P3 / 'runs' / arm / 'client_telemetry').glob('*/controls.jsonl'))
            for path in files:
                with path.open() as f:
                    first = json.loads(next(f))
                if first['task_uid'] == episode['uid'] and str(first['attempt']) == episode['attempt']:
                    break
            else:
                raise AssertionError(f'No accepted controls: {cell}')
            rows = []
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    if 'action_issued' in row and not row['wait_phase']:
                        rows.append(row)
            assert len(rows) == int(episode['active_controls'])
            obs_keys = set.intersection(*(set(r['after']['observation_numeric']) for r in rows))
            contact_counts = [len(r['after']['contacts']) for r in rows]
            predicates = [r['after']['predicates'] for r in rows]
            mapping_bytes = sum(len(json.dumps(r['after']['entity_ids'])) for r in rows)
            repeated_before_bytes = sum(len(json.dumps(r['before'])) for r in rows)
            successor_equal = [rows[i]['before'] == rows[i-1]['after'] for i in range(1, len(rows))]
            predicate_changes = sum(predicates[i] != predicates[i-1] for i in range(1,len(rows)))
            event_labels = sorted({json.dumps(r['after']['event_labels'], sort_keys=True) for r in rows})
            result.append(dict(cell=cell, uid=episode['uid'], file=str(path), bytes=path.stat().st_size,
                success=episode['Y'], active_controls=len(rows), present_observation_keys=sorted(obs_keys),
                controls_with_contacts=sum(n > 0 for n in contact_counts), max_contacts=max(contact_counts),
                predicate_keys=sorted({k for p in predicates for k in p}), predicate_changes=predicate_changes,
                actuator_missing=sorted({str(r['actuator_trace_missing']) for r in rows}),
                event_labels=event_labels, entity_mapping_after_json_bytes=mapping_bytes,
                before_json_bytes=repeated_before_bytes, consecutive_before_after_equal=sum(successor_equal),
                consecutive_pairs=len(successor_equal)))
    return result


if __name__ == '__main__':
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(trace=trace_inventory(), libraries=library_audit(), controls=control_audit())
    result['seconds'] = time.perf_counter() - start
    (OUT / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(seconds=result['seconds'], libraries=result['libraries'],
        trace_physical_datasets=[r['physical_datasets'] for r in result['trace']],
        controls=[{k:v for k,v in r.items() if k not in ('present_observation_keys','file','event_labels')}
                  for r in result['controls']]), indent=2))
