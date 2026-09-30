"""CPU-only real controller parity, timing and unchanged consumer test runners.

All outputs are confined to /tmp/codex_stall_fast/controller_followup. Existing
method_c scripts are run in memory with output-path redirection and, for their
raw-query recording probe, a dict retaining query attributes. Assertions and
production controller arguments are unchanged.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r06.ideation_Q1.method_c import methods
from . import stall
from .test_stall_fast import assert_identical
from .verify_fast import CELLS, ROOT, SCRATCH, write_json

OUT = SCRATCH/'controller_followup'
METHOD_C = Path(methods.__file__).parent


def before_class():
    name = 'exp.offline_search.rounds.r06.ideation_Q1.method_c._followup_before'
    spec = importlib.util.spec_from_file_location(name, OUT/'methods_before.py')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.CalibratedRescue


def recordings(cell, cohort):
    prod = cell.replace('_sp_', '_spatial_')
    tables = ROOT/('r06_c_cal' if cohort == 'bval' else 'r06_p3_pilot')/'tables'/(prod if cohort == 'bval' else cell)
    keys = ['arm', 'uid', 'attempt', 'step']
    acols = set(keys+['task_id', 'init', 'retrieval.metric_code', 'retrieval.metric', 'history.hit',
                     'history.vision', 'assignment.cohort'])
    anchors = pd.read_csv(tables/'anchors.csv', usecols=lambda k: k in acols)
    dcols = set(keys+['task_id', 'init', 'absolute_input_archive', 'actual_controls', 'vision',
                     'blind_features.retrieval.metric_code', 'blind_features.retrieval.metric'])
    dec = pd.read_csv(tables/'decisions.csv', usecols=lambda k: k in dcols)
    if cohort == 'pilot':
        # First episode per task and A/B cohort, all anchors of that episode.
        candidates = anchors[anchors['assignment.cohort'].isin(['A', 'B'])]
        first = candidates.sort_values(keys).drop_duplicates(['assignment.cohort', 'task_id'])
        ids = set(map(tuple, first[keys[:3]].to_numpy()))
        dec = dec[[tuple(r) in ids for r in dec[keys[:3]].to_numpy()]]
    lookup = {tuple(r[k] for k in keys): r for r in anchors.to_dict('records')}
    for identity, group in dec.groupby(keys[:3], sort=True):
        group = group.sort_values('step')
        rows = []
        previous_keys = None
        previous_action = np.zeros((10, 32), np.float32)
        hist_hit, hist_vision, control = [], [], 0
        for r in group.to_dict('records'):
            record = lookup.get(tuple(r[k] for k in keys))
            use = cohort == 'bval' or record is not None
            # Every archive is read only, to retain the actual preceding action.
            with np.load(r['absolute_input_archive'], allow_pickle=False) as data:
                action = np.array(data['executed_chunk'])
                if 'vision_0' in data:
                    vectors = (np.array(data['vision_0']), np.array(data['vision_1']), np.array(data['robot_state']))
                else: vectors = None
            if record is not None:
                hist_hit = json.loads(record['history.hit'])
                hist_vision = json.loads(record['history.vision'])
            if use:
                assert vectors is not None, (cell, identity, r['step'])
                ep = NS(uid=identity[1], task_id=int(r['task_id']), init=int(r['init']))
                q = NS(episode=ep, task_id=ep.task_id, step=int(r['step']),
                    key_v0=vectors[0], key_v1=vectors[1], rs=vectors[2],
                    prev_hit=bool(hist_hit[-1]) if hist_hit else None, prev_a_exec=previous_action,
                    hist_key_v0=[previous_keys[0]] if previous_keys is not None else [],
                    hist_key_v1=[previous_keys[1]] if previous_keys is not None else [],
                    hist_has_vision=list(hist_vision), hist_hit=list(hist_hit))
                assert control == q.step*5, (cell, q.step, control)
                source = record if record is not None else r
                prefix = 'retrieval.' if record is not None else 'blind_features.retrieval.'
                code = np.asarray(json.loads(source[prefix+'metric_code']), float)
                regime = source[prefix+'metric']
                rows.append((q, dict(metric_code=code, metric=regime), record is not None))
            if vectors is not None and r['vision']: previous_keys = vectors
            previous_action = action
            hist_hit.append(True); hist_vision.append(bool(r['vision']))
            control += int(r['actual_controls'])
        yield identity, rows


def controller(cls, cell):
    prod = cell.replace('_sp_', '_spatial_')
    cal = ROOT/'r06_c_cal/cal'/prod/'calibrated/calibration.json'
    c = cls(.3, placement='R', stall_model_path=ROOT/'r06_c_cal/stall'/cell,
        calibration_path=cal, random_seed=26092903, randomization_key='R6-C-v2/'+prod)
    c._load(prod.rsplit('_', 1)[0]+'_cache')
    return c


def check_result(a, b, where):
    for field in ('topk', 'scores', 'confidence', 'action', 'library', 'extras'):
        assert_identical(getattr(a, field), getattr(b, field), where+'.'+field)


def parity(cell):
    old, new = controller(before_class(), cell), controller(methods.CalibratedRescue, cell)
    totals = {}
    for cohort in ('bval', 'pilot'):
        stats = Counter()
        for identity, rows in recordings(cell, cohort):
            # All observed B-val anchors plus blind diagnostic codes (possible
            # LOOK inputs), and all selected actual pilot anchors.
            for cadence in (('all_fresh', 'C_cadence') if cohort == 'bval' else ('pilot_anchors',)):
                old.reset(rows[0][0].episode); new.reset(rows[0][0].episode)
                i = 0
                while i < len(rows):
                    q, logged_code, actual_anchor = rows[i]
                    raw, regime = new.stall_model.encode(q, str(q.task_id))
                    encoded, saved_regime = new.stall_model.encode(logged_code, str(q.task_id))
                    assert_identical(raw, encoded, f'{cell}/{identity}/{q.step}/recorded_code')
                    assert regime == saved_regime
                    capture = {}
                    real_observe = new.tracker.observe
                    def observe(key, control):
                        got, mode = new.stall_model.encode(key, str(q.task_id))
                        assert isinstance(key, dict) and 'metric_code' in key
                        assert_identical(got, raw, 'controller metric code')
                        assert mode == regime
                        capture['code'] = got
                        return real_observe(key, control)
                    with patch.object(new.tracker, 'observe', observe):
                        a = old.query(q); b = new.query(q)
                    check_result(a, b, f'{cell}/{identity}/{q.step}/{cadence}')
                    assert_identical(old.tracker.status(), new.tracker.status(), 'stall status')
                    assert '_dist' not in new.base.__dict__
                    for name in ('_last_log', '_anchor_index', '_last_cooldown_anchor', '_last_extra_control',
                                 '_look_due_step', '_policy_gate_anchor'):
                        assert_identical(getattr(old, name), getattr(new, name), name)
                    stats[cadence+'_anchors'] += 1
                    stats[cadence+'_actual_recorded_anchors'] += actual_anchor
                    stats['early_codes'] += regime == 'early'; stats['main_codes'] += regime == 'main'
                    stats['calls'] += bool(new._last_log['os_c_call'])
                    stats['cooldowns'] += bool(new._last_log['os_c_cooldown'])
                    stats['extra_LOOKs'] += bool(new._last_log['os_c_extra_look'])
                    step = 1
                    if cadence == 'C_cadence':
                        step = 1 if new._last_log['os_c_extra_look'] else 2
                        if step == 1 and i+1 < len(rows): stats['extra_LOOK_observations'] += 1
                    elif i+1 < len(rows) and new._last_log['os_c_extra_look']:
                        stats['extra_LOOK_observations'] += 1
                    i += step
                stats[cadence+'_episodes'] += 1
        totals[cohort] = dict(stats)
    result = dict(cell=cell, differences=0, maximum_absolute_float_difference=0,
        counts=totals, fields='full stall status; all result arrays/scalars/extras; Ehat, nominal p, p, coin, call, '
        'stall call, cooldown, extra LOOK, anchor/window/control bookkeeping and policy-tail gate')
    write_json(OUT/f'parity_{cell}.json', result); print(json.dumps(result), flush=True)


class TimedTracker(stall.StallTracker):
    def __init__(self, model, task):
        super().__init__(model, task); self.samples = []
    def observe(self, key, control):
        cpu = time.process_time_ns(); wall = time.perf_counter_ns()
        super().observe(key, control)
        elapsed_wall = time.perf_counter_ns()-wall; elapsed_cpu = time.process_time_ns()-cpu
        self.samples.append((elapsed_cpu, elapsed_wall))


def timing(cell):
    # Preload raw vectors/JSON outside timers. Follow the controller call/LOOK
    # cadence on fixed B-val observations and original recorded A histories.
    # This includes actual extra-LOOK input rows from the diagnostic archives.
    streams = list(recordings(cell, 'bval'))
    results = []
    for name, cls in [('raw', before_class()), ('code', methods.CalibratedRescue)]:
        c = controller(cls, cell); c.tracker_class = TimedTracker
        samples = []
        for identity, rows in streams:
            c.reset(rows[0][0].episode)
            i = 0
            while i < len(rows):
                q, _, _ = rows[i]
                # Full query timing includes A retrieval; incremental stall cost
                # is timed by subtracting the original A query's internal work.
                # A._dist timing is nested inside the captured facade hook, so
                # its copy/setup cost remains attributed to the new stall path.
                base_cpu, base_wall, dist_cpu, dist_wall = [], [], [], []
                original_query, original_dist = c.base.query, c.base._dist
                def dist(self, query):
                    cpu = time.process_time_ns(); wall = time.perf_counter_ns()
                    try: return original_dist.__func__(self, query)
                    finally:
                        dist_wall.append(time.perf_counter_ns()-wall); dist_cpu.append(time.process_time_ns()-cpu)
                def query(query):
                    cpu = time.process_time_ns(); wall = time.perf_counter_ns()
                    try: return original_query(query)
                    finally:
                        base_wall.append(time.perf_counter_ns()-wall); base_cpu.append(time.process_time_ns()-cpu)
                hooked_cpu, hooked_wall, capture_cpu, capture_wall = [], [], [], []
                metric_distance = getattr(c, '_metric_distance', None)
                def hooked(*args):
                    cpu = time.process_time_ns(); wall = time.perf_counter_ns()
                    try: return metric_distance(*args)
                    finally:
                        hooked_wall.append(time.perf_counter_ns()-wall); hooked_cpu.append(time.process_time_ns()-cpu)
                def multiply(self, code):
                    cpu = time.process_time_ns(); wall = time.perf_counter_ns()
                    self.code = code
                    mat_cpu = time.process_time_ns(); mat_wall = time.perf_counter_ns()
                    result = self.matrix @ code
                    mat_elapsed_wall = time.perf_counter_ns()-mat_wall
                    mat_elapsed_cpu = time.process_time_ns()-mat_cpu
                    capture_wall.append(time.perf_counter_ns()-wall-mat_elapsed_wall)
                    capture_cpu.append(time.process_time_ns()-cpu-mat_elapsed_cpu)
                    return result
                from contextlib import ExitStack
                with ExitStack() as context:
                    context.enter_context(patch.object(c.base, 'query', query))
                    context.enter_context(patch.object(c.base, '_dist', MethodType(dist, c.base)))
                    if name == 'code':
                        context.enter_context(patch.object(c, '_metric_distance', hooked))
                        context.enter_context(patch.object(methods._MetricCodeMatrix, '__matmul__', multiply))
                    cpu = time.process_time_ns(); wall = time.perf_counter_ns()
                    c.query(q)
                    full_wall = time.perf_counter_ns()-wall; full_cpu = time.process_time_ns()-cpu
                obs_cpu, obs_wall = c.tracker.samples[-1]
                # Incremental C cost over original A includes tracker/status,
                # helper setup/restoration, the entire facade preparation,
                # operand capture, and score/coin/cooldown/LOOK bookkeeping.
                # Subtract only original A work; facade/operand overhead timed
                # inside base.query is added back. Nested timers add overhead,
                # so this is a conservative instrumented host measurement.
                hook_cpu = sum(hooked_cpu)-sum(dist_cpu)+sum(capture_cpu) if name == 'code' else 0
                hook_wall = sum(hooked_wall)-sum(dist_wall)+sum(capture_wall) if name == 'code' else 0
                inc_cpu = full_cpu-sum(base_cpu)+hook_cpu
                inc_wall = full_wall-sum(base_wall)+hook_wall
                samples.append(dict(uid=identity[1], task=q.task_id, step=q.step,
                    observe_cpu_ms=obs_cpu/1e6, observe_wall_ms=obs_wall/1e6,
                    incremental_controller_cpu_ms=inc_cpu/1e6, incremental_controller_wall_ms=inc_wall/1e6,
                    full_controller_cpu_ms=full_cpu/1e6, full_controller_wall_ms=full_wall/1e6,
                    capture_overhead_cpu_ms=hook_cpu/1e6))
                i += 1 if c._last_log['os_c_extra_look'] else 2
        frame = pd.DataFrame(samples); frame.to_csv(OUT/f'timing_{name}_{cell}.csv', index=False)
        summary = dict(cell=cell, input=name, anchors=len(frame),
            **{column+'_median':float(frame[column].median()) for column in frame if column.endswith('_ms')},
            **{column+'_p99':float(frame[column].quantile(.99)) for column in frame if column.endswith('_ms')})
        results.append(summary); print(json.dumps(summary), flush=True)
    write_json(OUT/f'timing_{cell}.json', results)


def replays(cell, case):
    from exp.offline_search.rounds.r06.ideation_Q1.method_c import replay_test
    real_query = methods.CalibratedRescue.query
    current = {}
    class ProbeCode(dict):
        def __getattr__(self, key): return getattr(current['q'], key)
    def payload(*args, **kwargs):
        return ProbeCode(*args, **kwargs) if 'metric_code' in kwargs else dict(*args, **kwargs)
    def query(self, q):
        current['q'] = q
        return real_query(self, q)
    def output(path):
        path = Path(path).resolve(); assert OUT in path.parents
        return path
    reports = []
    for cell in (cell,):
        for case in (case,):
            out = OUT/'replays_v2b_final'/f'{cell}_{case}'
            argv = ['replay_test', '--cell', cell, '--case', 'R' if case == 'R_reverse' else case,
                    '--out', str(out)] + (['--reverse'] if case == 'R_reverse' else [])
            # RecordingTracker only uses the raw attributes for its telemetry
            # probe; dict keys still send the exact code to production observe.
            with patch.object(sys, 'argv', argv), patch.object(replay_test, 'output_path', output), \
                 patch.object(methods.CalibratedRescue, 'query', query), patch.object(methods, 'dict', payload, create=True):
                replay_test.main()
            report = json.loads((out/'report.json').read_text()); reports.append(report)
            # Compare action/source/vision/HIT products with kept v2b replay.
            baseline = Path('/tmp/q1_method_c_fits/replays_v2b')/f'{cell}_{case}'/'served.npz'
            with np.load(out/'served.npz') as a, np.load(baseline) as b:
                assert a.files == b.files
                for key in a.files: assert_identical(a[key], b[key], f'replay/{cell}/{case}/{key}')
    write_json(OUT/f'replay_test_{cell}_{case}.json', dict(status='PASS', replays=len(reports),
        anchors=sum(r['anchors'] for r in reports), decisions=sum(r['decisions'] for r in reports), reports=reports))


def consumer_checks():
    from exp.offline_search.rounds.r06.ideation_Q1.method_c import check_results, check_packaging
    old_out = Path('/tmp/q1_method_c_fits')
    # Read-only fallthrough to original fixtures; redirect every writable
    # scratch path to the authorized root through a small Path-like facade.
    class Overlay:
        def __truediv__(self, part):
            if str(part) == 'replays_v2b': return OUT/'replays_v2b_final'
            if str(part) == 'reset_gate_test' or str(part).endswith('_local_test.yaml'):
                return OUT/part
            return old_out/part
    def writer(path, data): write_json(OUT/Path(path).name, data)
    for module in (check_results, check_packaging):
        with patch.object(module, 'OUT', Overlay()), patch.object(module, 'write_json', writer), \
             patch.object(sys, 'argv', [module.__name__, '--tag', 'v2b']):
            module.main()


def calibrations():
    from . import verify_fast
    with patch.object(verify_fast, 'SCRATCH', OUT):
        for cell in CELLS:
            verify_fast.calibrate(cell)
            got = json.loads((OUT/f'calibration_{cell}.json').read_text())
            prod = Path(got['production']); emitted = Path(got['output'])
            assert emitted.read_bytes() == prod.read_bytes(), cell
            assert (emitted.parent/'cost_replay.json').read_bytes() == (prod.parent/'cost_replay.json').read_bytes(), cell
            got['calibration_byte_equal'] = got['cost_replay_byte_equal'] = True
            write_json(OUT/f'calibration_{cell}.json', got)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['parity', 'timing', 'replays', 'checks', 'calibration'])
    p.add_argument('--cell', choices=CELLS)
    p.add_argument('--case', choices=['A', 'floor', 'uniform', 'R', 'R_reverse', 'R45', 'stall', 'stall30'])
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'replays':
        if not a.cell or not a.case: p.error('each plugin replay needs --cell and --case in a fresh interpreter')
        replays(a.cell, a.case)
    elif a.stage == 'checks': consumer_checks()
    elif a.stage == 'calibration': calibrations()
    else:
        for cell in (a.cell,) if a.cell else CELLS: {'parity':parity, 'timing':timing}[a.stage](cell)


if __name__ == '__main__': main()
