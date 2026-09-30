"""Read-only real-data verification; all new products go to the scratch root.

Run from the repository with the affinity/environment in HANDBACK_FAST.md.
Timing is observe-only, on the exact replay_pilot joins; equivalence and detailed
path checks are separate untimed passes. No artifact is refitted or overwritten.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import stall, stall_reference as reference
from .replay_pilot import KEY, TABLES
from .test_stall_fast import assert_identical, check_alignment

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/weiland/trace_runs/os_closed_loop')
SCRATCH = Path('/tmp/codex_stall_fast')
CELLS = tuple(sorted(json.loads((HERE/'source_manifest.json').read_text())))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(4 << 20): h.update(chunk)
    return h.hexdigest()


def load_streams(cell):
    # These are precisely replay_pilot.py's columns, filtering, timestamps,
    # one-to-one join, episode grouping and within-episode ordering.
    root = TABLES/cell
    dec = pd.read_csv(root/'decisions.csv', usecols=KEY+['step', 'actual_controls'])
    dec = dec.sort_values(KEY+['step'])
    dec['control_index'] = dec.groupby(KEY).actual_controls.cumsum()-dec.actual_controls
    cols = KEY+['step', 'task_id', 'init', 'assignment.cohort', 'retrieval.metric_code',
                'retrieval.metric', 'guards.inputs_outputs.os_flags']
    anchors = pd.read_csv(root/'anchors.csv', usecols=cols)
    anchors = anchors[anchors['assignment.cohort'].isin(['A', 'B'])].merge(
        dec[KEY+['step', 'control_index']], on=KEY+['step'], validate='one_to_one')
    streams = []
    for identity, group in anchors.groupby(KEY, sort=True):
        rows = []
        for r in group.sort_values('step').to_dict('records'):
            rows.append((dict(metric_code=json.loads(r['retrieval.metric_code']), metric=r['retrieval.metric']),
                         int(r['control_index']), int(r['step'])))
        streams.append((identity, int(group.task_id.iloc[0]), group['assignment.cohort'].iloc[0], rows))
    return streams


def pilot(cell):
    streams = load_streams(cell)
    model = stall.StallModel.load(Path('/tmp/q3_stall_fits')/cell)
    old = reference.StallModel.load(Path('/tmp/q3_stall_fits')/cell)
    n, active, paths = 0, 0, 0
    states = Counter()
    cohorts = {}
    for identity, task, cohort, rows in streams:
        fast, slow = stall.StallTracker(model, task), reference.StallTracker(old, task)
        for key, control, step in rows:
            fast.observe(key, control); slow.observe(key, control)
            got, expected = fast.status(), slow.status()
            assert got.get('reason') != 'invalid_observation'
            assert_identical(got, expected, f'{cell}/{identity}/{step}')
            if got['state'] != 'inactive':
                span = fast._window[-1][1]-fast._window[0][1]
                paths += check_alignment(model, str(task), [v[0] for v in fast._window], span)
                active += 1
            states[cohort, got['state']] += 1
            cohorts[cohort] = cohorts.get(cohort, 0)+1
            n += 1
    previous = json.loads((HERE/f'pilot_replay_{cell}.json').read_text())
    for s in previous['summaries']:
        if s['scope'] == 'all_anchors':
            assert s['n'] == cohorts[s['cohort']]
            assert s['counts'] == {state: states[s['cohort'], state] for state in s['counts']}
    result = dict(cell=cell, anchors=n, episodes=len(streams), active=active,
        per_template_paths_and_distance_matrices=paths, cohorts=cohorts,
        fields=['state', 'reason', 'delta_hat', 'phase_hat', 'distance_hat', 'spread_hat', 'e90', 'a10',
                'reference_windows', 'window_span', 'W', 'K', 'reference_episodes',
                'all_template_paths', 'selected_template_order', 'every_distance'],
        differences=0, maximum_absolute_float_difference=0.0, fingerprint=model.fingerprint,
        previous_status_counts_identical=True)
    write_json(SCRATCH/f'pilot_{cell}.json', result)
    print(json.dumps(result), flush=True)


def timing(cell):
    streams = load_streams(cell)
    results = []
    # Fresh model per implementation: load remains outside the timer, lazy
    # task preparation is included at its actual first active observation.
    for name, module in [('reference', reference), ('fast', stall)]:
        model = module.StallModel.load(Path('/tmp/q3_stall_fits')/cell)
        samples = []
        for identity, task, cohort, rows in streams:
            tracker = module.StallTracker(model, task)
            for key, control, step in rows:
                cpu_start = time.process_time_ns(); wall_start = time.perf_counter_ns()
                tracker.observe(key, control)
                wall = time.perf_counter_ns()-wall_start
                cpu = time.process_time_ns()-cpu_start
                # Status copy, parsing, joins and I/O are outside the timer.
                state = tracker.status()['state']
                samples.append(dict(arm=identity[0], uid=identity[1], attempt=identity[2],
                    task_id=task, cohort=cohort, step=step, control_index=control,
                    state=state, cpu_us=cpu/1000, wall_us=wall/1000))
        frame = pd.DataFrame(samples)
        frame.to_csv(SCRATCH/f'timing_{name}_{cell}.csv', index=False)
        result = dict(cell=cell, implementation=name, anchors=len(frame),
            cpu_median_ms=float(frame.cpu_us.median()/1000), cpu_p99_ms=float(frame.cpu_us.quantile(.99)/1000),
            wall_median_ms=float(frame.wall_us.median()/1000), wall_p99_ms=float(frame.wall_us.quantile(.99)/1000))
        results.append(result)
        print(json.dumps(result), flush=True)
    write_json(SCRATCH/f'timing_{cell}.json', results)


def artifacts():
    rows = []
    for cell in CELLS:
        deployed_cell = cell.replace('_sp_', '_spatial_')
        for root in (Path('/tmp/q3_stall_fits')/cell, ROOT/'r06_c_cal/stall'/deployed_cell):
            if not root.exists(): root = ROOT/'r06_c_cal/stall'/cell
            file = root/'stall.pkl'; side = root/'stall.json'
            before = sha(file), sha(side)
            model = stall.StallModel.load(root)
            old = reference.StallModel.load(root)
            assert model.fingerprint == old.fingerprint
            assert_identical(model._payload(), old._payload())
            for task, data in model.tasks.items():
                if len(data['references']) < 4: continue
                tracker = stall.StallTracker(model, task)
                t = data['templates'][0]
                for i in range(data['W']+1):
                    tracker.observe(dict(metric_code=t['codes']['main'][i], metric='main'), int(t['controls'][i]))
            assert stall._digest(model._payload()) == model.fingerprint
            # Pickle round-trip need not regenerate the producer's aliases;
            # the original payload SHA and content hash remain checked at load.
            assert before == (sha(file), sha(side))
            rows.append(dict(path=str(root), cell=cell, fingerprint=model.fingerprint,
                payload_sha256=before[0], sidecar_sha256=before[1], loaded_verified=True,
                content_unchanged_after_caches=True))
    write_json(SCRATCH/'artifact_verification.json', rows)
    print(json.dumps(dict(artifact_pairs=len(rows), differences=0)), flush=True)


def audit():
    from . import audit_metrics
    out = SCRATCH/'metric_audit'
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE/'source_manifest.json', out/'source_manifest.json')
    # Run audit_metrics.main verbatim, redirecting its sole output directory.
    with patch.object(audit_metrics, 'HERE', out): audit_metrics.main()
    reports = json.loads((out/'metric_audit.json').read_text())
    assert sum(r['raw_code_checks'] for r in reports) == 1920
    assert all(r['raw_and_code_status_identical'] and r['A_B_metric_bit_identical'] and
               r['saved_metric_matches_A'] for r in reports)


def calibrate(cell):
    from exp.offline_search.rounds.r06.ideation_Q1.method_c import fit_calibration
    prod_cell = cell.replace('_sp_', '_spatial_')
    base = ROOT/'r06_c_cal'
    prod_path = base/'cal'/prod_cell/'calibrated/calibration.json'
    expected = json.loads(prod_path.read_text())
    manifest = base/'manifests'/f'calibration_manifest_{prod_cell}.json'
    assert sha(manifest) == expected['bval_manifest_sha256']
    tables = Path(next(iter(expected['tables']))).parent
    bank = base/'cal'/prod_cell/'r_bank.json'
    client = Path(expected['reset_attestation'][0]['path']).parent.parent
    stall_path = base/'stall'/prod_cell
    if not stall_path.exists(): stall_path = base/'stall'/cell
    out = SCRATCH/'calibration_verified'/cell
    # The consumer's allowlist predates this task and only permits method_c/
    # or /tmp/q1_method_c_fits/. This runtime-only replacement restricts the
    # one authorized scratch destination; no consumer file is changed.
    def scratch_output(path):
        resolved = Path(path).resolve()
        assert resolved == out.resolve(), resolved
        return resolved
    targets = [float(k) for k in expected['library_uniform_budget']]
    with patch.object(fit_calibration, 'output_path', scratch_output):
        got = fit_calibration.fit(tables, bank, out, expected['c1'], targets,
            bval_manifest=manifest, stall_model_path=stall_path, client_root=client,
            cooldown_scope=expected['cooldown_scope'])
    # Compare emitted JSON to emitted JSON (the producer holds integer task
    # keys in memory; JSON necessarily serializes those keys as strings).
    got = json.loads((out/'calibration.json').read_text())
    assert_identical(got, expected, f'calibration/{cell}')
    # The emitted replay product is compared too: cadence topology, compact
    # stall statuses and exact cost inputs, in addition to all budget solutions.
    replay_path = prod_path.parent/'cost_replay.json'
    if replay_path.exists():
        assert_identical(json.loads((out/'cost_replay.json').read_text()), json.loads(replay_path.read_text()),
                         f'cost_replay/{cell}')
    result = dict(cell=cell, production=str(prod_path), output=str(out/'calibration.json'),
        anchors=got['calibration_anchors'], episodes=len(got['calibration_episodes']),
        cadence_nodes=got['cadence_nodes'], solution_count=sum(len(s) for mode in got['solutions'].values() for s in mode.values()),
        differences=0, entire_calibration_json_identical=True,
        entire_cost_replay_identical=replay_path.exists(), stall_fingerprint=got['stall_fingerprint'],
        c1=expected['c1'], targets=targets, tables=str(tables), bank=str(bank),
        bval_manifest=str(manifest), client_root=str(client), stall_model_path=str(stall_path))
    write_json(SCRATCH/f'calibration_{cell}.json', result)
    print(json.dumps(result), flush=True)


def decision_logs():
    root = ROOT/'r06_c_validation'
    arms = json.loads((root/'arms.json').read_text())
    # Only C arms with a configured stall artifact can test the live tracker.
    c = {r['arm']: r for r in arms if r['kwargs'].get('stall_model_path')}
    reports = []
    for arm in sorted(c):
        files = sorted((root/'runs'/arm).rglob('decisions*.jsonl'))
        rows, fresh, code, raw_keys, archives, logged = 0, 0, 0, 0, 0, 0
        for path in files:
            with path.open() as f:
                for line in f:
                    r = json.loads(line)
                    if r.get('ev') != 'dec': continue
                    rows += 1
                    e = r.get('extras', {})
                    if not e.get('os_c_fresh'): continue
                    fresh += 1
                    logged += 'os_c_stall_state' in e
                    # Scan all record content, not just top-level fields.
                    code += 'metric_code' in line
                    raw_keys += 'key_v0' in line and 'key_v1' in line
                    archives += 'absolute_input_archive' in line or 'input_archive' in line
        report = dict(arm=arm, files=len(files), decisions=rows, fresh_anchors=fresh,
            logged_stall_states=logged, metric_code_records=code, raw_visual_key_records=raw_keys,
            input_archive_records=archives, replayable=False)
        # Should any record actually contain a replay source, stop instead of
        # claiming missing data; a decoder would then be required here.
        assert code == raw_keys == archives == 0, report
        reports.append(report)
    result = dict(arms=len(reports), decisions=sum(r['decisions'] for r in reports),
        fresh_anchors=sum(r['fresh_anchors'] for r in reports), replayable=False,
        reason='Decision logs record state and scalar diagnostics, robot_state, retrieval rows/weights/scores; '
               'they omit metric codes, visual keys and any input archive pointer. '
               'Rows/weights and robot state cannot recover the visual query metric code.', reports=reports)
    write_json(SCRATCH/'decision_log_check.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'reports'}), flush=True)


def breakdown(cell):
    """Untimed decomposition pass, after the standalone timing distribution."""
    model = stall.StallModel.load(Path('/tmp/q3_stall_fits')/cell)
    streams = load_streams(cell)
    measurements = {'encode': [], 'distance': [], 'alignment_estimate': [], 'calibrated_status': []}
    originals = {}
    for method, label in [('encode', 'encode'), ('_distances', 'distance'),
                          ('_estimate_distances', 'alignment_estimate'), ('calibrated_status', 'calibrated_status')]:
        func = getattr(model, method)
        originals[method] = func
        def wrapper(*args, _func=func, _label=label, **kwargs):
            start = time.process_time_ns()
            try: return _func(*args, **kwargs)
            finally: measurements[_label].append((time.process_time_ns()-start)/1e6)
        setattr(model, method, wrapper)
    for _, task, _, rows in streams:
        tracker = stall.StallTracker(model, task)
        for key, control, _ in rows: tracker.observe(key, control)
    result = dict(cell=cell, scope='CPU component medians on a separate instrumented pilot pass; '
                  'lazy preparation included; timer overhead means medians are not additive',
        components={k:dict(calls=len(v), median_ms=float(np.median(v)), p99_ms=float(np.quantile(v, .99)))
                    for k,v in measurements.items()})
    write_json(SCRATCH/f'breakdown_{cell}.json', result)
    print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['artifacts', 'pilot', 'timing', 'audit', 'calibration', 'logs', 'breakdown'])
    ap.add_argument('--cell', choices=CELLS)
    args = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if args.stage == 'artifacts': artifacts()
    elif args.stage == 'audit': audit()
    elif args.stage == 'logs': decision_logs()
    else:
        func = {'pilot': pilot, 'timing': timing, 'calibration': calibrate, 'breakdown': breakdown}[args.stage]
        for cell in (args.cell,) if args.cell else CELLS: func(cell)


if __name__ == '__main__': main()
