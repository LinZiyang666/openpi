"""Recorded raw-input serving parity and independent frozen-proposal DAG replay.

No simulator, server, worker, model inference or outcome estimate is involved.
Identity replay runs the real A retrieval and both real R6/C3 controllers on
the same raw archives; fake policy actions come from that request's archive.
The separate cost replay holds proposals fixed, as the calibration assumes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from exp.offline_search.closed_loop.blind import LookReason, BlindResult
from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.methods import CalibratedRescue
from exp.offline_search.rounds.r06.ideation_Q1.method_c.stall_bridge import STATE
from exp.offline_search.rounds.r06.ideation_Q1.method_c.budget import simulate_path, expected_cost
from exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.methods import uniform
from .common import ALL_CELLS, SCRATCH, output_path, write_json, stall_path, target_rho
from .methods import CallController
from .recordings import streams, query


def identical(a, b, path='value'):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), path
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), path
    elif isinstance(a, dict):
        assert list(a) == list(b), path + ': keys/order'
        for key in a:
            identical(a[key], b[key], path + '.' + key)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            identical(x, y, path + f'[{i}]')
    elif hasattr(a, '__dataclass_fields__'):
        assert type(a) is type(b), path
        for name in a.__dataclass_fields__:
            identical(getattr(a, name), getattr(b, name), path + '.' + name)
    elif isinstance(a, (float, np.floating)):
        x, y = np.asarray(a), np.asarray(b)
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), path
    else:
        assert a == b, path


def controller(cell, cls, *, placement='uniform', tilt=False):
    kwargs = dict(rho=target_rho(cell), placement=placement, stall_model_path=str(stall_path(cell)),
        calibration_path=str(SCRATCH / 'cal' / cell / ('CT' if tilt else 'CU') / 'calibration.json'),
        random_seed=26092903, randomization_key='R6-C-v2/' + cell, cooldown_scope='stall')
    if cls is CallController:
        kwargs['tilt'] = tilt
    method = cls(**kwargs)
    method._load(cell.rsplit('_', 1)[0] + '_cache')
    return method


def dispatch(method, q, visions, hits):
    if not q.step:
        return LookReason(6, 'start')
    if visions[-1] and not hits[-1]:
        return method.policy_tail_step(q)
    return method.blind_step(q)


def identity(cell):
    totals = {}
    action_hash = hashlib.sha256()
    for placement in ('uniform', 'R'):
        old = controller(cell, CalibratedRescue, placement=placement)
        new = controller(cell, CallController, placement=placement)
        for cohort in ('bval', 'pilot'):
            counts = Counter()
            for episode, data in streams(cell, cohort):
                old.reset(episode); new.reset(episode)
                actions, hits, visions = [], [], []
                counts['episodes'] += 1
                for row in data:
                    q = query(episode, data, actions, hits, visions)
                    a = dispatch(old, q, visions, hits)
                    b = dispatch(new, q, visions, hits)
                    identical(a, b, f'{cell}/{cohort}/{episode.uid}/{q.step}/blind')
                    if isinstance(a, LookReason):
                        a = old.query(q); b = new.query(q)
                        identical(a, b, f'{cell}/{cohort}/{episode.uid}/{q.step}/query')
                        hit = not bool(a.extras['os_force_miss'])
                        action = a.action if hit else row['policy_chunk']
                        visions.append(True)
                        counts['anchors'] += 1
                        counts['calls'] += not hit
                        counts['extra_LOOKs'] += int(a.extras['os_c_extra_look'])
                        counts['stall_cooldowns'] += int(a.extras['os_c_cooldown'])
                        counts['stall_calls'] += int(a.extras['os_c_stall_call'])
                    else:
                        assert isinstance(a, BlindResult)
                        action, hit = a.action, True
                        counts['policy_tails' if visions[-1] and not hits[-1] else 'cache_tails'] += 1
                        visions.append(False)
                    # Serving actions/verdicts/vision flags coincide at every
                    # request. Identical histories are carried to the next one.
                    identical(old._last_log, new._last_log, 'controller decision log')
                    identical(old.tracker.status(), new.tracker.status(), 'stall tracker')
                    actions.append(np.array(action)); hits.append(hit)
                    action_hash.update(np.asarray(action).tobytes())
                    counts['decisions'] += 1
            totals[placement + '/' + cohort] = dict(counts)
    # Real C1 composition with E=0 must retain the complete disabled path.
    from exp.offline_search.rounds.r07.c1_follow.methods import FollowExtension
    old, new = controller(cell, CalibratedRescue), controller(cell, CallController)
    new.install_follow_extension(FollowExtension(None, extend_blocks=0))
    count = 0
    for episode, data in streams(cell, 'bval'):
        old.reset(episode); new.reset(episode)
        actions, hits, visions = [], [], []
        for row in data:
            q = query(episode, data, actions, hits, visions)
            a, b = dispatch(old, q, visions, hits), dispatch(new, q, visions, hits)
            identical(a, b, 'composed E=0 blind')
            if isinstance(a, LookReason):
                a, b = old.query(q), new.query(q)
                identical(a, b, 'composed E=0 query')
                hit = not bool(a.extras['os_force_miss'])
                action = a.action if hit else row['policy_chunk']
                visions.append(True)
            else:
                action, hit = a.action, True
                visions.append(False)
            actions.append(np.array(action)); hits.append(hit); count += 1
    report = dict(status='PASS', cell=cell, differing_actions=0, differing_verdicts=0,
                  differing_vision_flags=0, differing_result_fields=0, counts=totals,
                  composed_E0_decisions=count, combined_action_sha256=action_hash.hexdigest())
    write_json(SCRATCH / 'replays' / f'identity_{cell}.json', report)
    print(json.dumps(report), flush=True)


def cost_paths(cell, seeds=1000):
    root = SCRATCH / 'cal' / cell
    summaries = {}
    for variant, tilt in [('CU', False), ('CT', True)]:
        cal = json.loads((root / variant / 'calibration.json').read_text())
        trees = json.loads((root / variant / 'cost_replay.json').read_text())['modes']['stall']
        method = controller(cell, CallController, tilt=tilt)
        by_uid = {tree['uid']: tree for tree in trees}
        stats = Counter()
        for episode, data in streams(cell, 'bval'):
            tree = by_uid[episode.uid]
            method.reset(episode)
            # Fixed-proposal calibration replay only: substitute the recorded
            # A proposal/code; the real CT feature calculation and inherited
            # R6 controller decisions run unchanged.
            def recorded_proposal(self, q):
                row = data[q.step]
                self.base._remember_anchor(q, row['rows'], row['weights'], row['executed_chunk'])
                return api.Result(row['rows'], np.zeros(len(row['rows'])), 0.,
                                  row['executed_chunk'], self.base.cand_name, {}), row['key']
            visited = []
            step = 0
            with patch.object(CalibratedRescue, '_query_with_metric_code', recorded_proposal):
                while step < len(data):
                    row = data[step]
                    q = NS(episode=episode, task_id=episode.task_id, step=step, rs=row['robot_state'])
                    method.query(q)
                    ex = method._last_log
                    state = next(name for name, code in STATE.items() if code == ex['os_c_stall_state'])
                    visited.append((step, state, bool(ex['os_c_call']), bool(ex['os_c_extra_look'])))
                    if tilt:
                        i = next(i for i, n in enumerate(tree['nodes']) if
                            n['step'] == step and n['state'] == state and n['cooled'] == bool(ex['os_c_cooldown'])
                            and abs(n['call_weight'] - ex['os_c3_weight']) < 1e-14)
                        assert tree['nodes'][i]['call_weight'] == ex['os_c3_weight']
                        stats['deviation_entries'] += ex['os_c3_dev_entry']
                    stats['anchors'] += 1
                    stats['calls'] += ex['os_c_call']; stats['extra_LOOKs'] += ex['os_c_extra_look']
                    stats['cooldowns'] += ex['os_c_cooldown']
                    step += 1 if ex['os_c_extra_look'] else method.commit_controls // method.block_controls
            parameter = method.lambda_
            placement = 'R' if tilt else 'uniform'
            path = simulate_path(tree, parameter, placement, lambda n: uniform(method.randomization_key,
                method.random_seed, episode.task_id, episode.init, n['step'], 'dose-anchor'))
            assert visited == path['path'], (cell, variant, episode.uid)
            stats['episodes'] += 1
        parameter = method.lambda_
        placement = 'R' if tilt else 'uniform'
        samples = []
        for seed in range(seeds):
            costs = []
            for tree in trees:
                path = simulate_path(tree, parameter, placement, lambda n: uniform(method.randomization_key,
                    seed, tree['task'], tree['init'], n['step'], 'dose-anchor'))
                costs.append([(cal['c1'] * path['anchors'] + (1 - cal['c1']) * path['calls']) / tree['nominal_blocks'],
                              path['anchors'], path['calls'], path['extra_LOOKs']])
            samples.append(np.average(costs, axis=0, weights=[t['weight'] for t in trees]))
        samples = np.array(samples)
        expected = expected_cost(trees, parameter, placement, cal['c1'])
        # Dense cells can have exactly constant anchor counts. Accumulate in
        # extended precision so repeated constants do not acquire an artificial
        # standard error; still check zero-variance statistics against roundoff.
        means = np.asarray(samples.mean(0, dtype=np.longdouble), float)
        se = np.asarray(samples.std(0, ddof=1, dtype=np.longdouble) / np.sqrt(seeds), float)
        exact = np.array([expected[k] for k in ['IR', 'anchors', 'calls', 'extra_LOOKs']])
        error = abs(means - exact)
        tolerance = 64 * np.finfo(float).eps * np.maximum(1., abs(exact))
        z = np.divide(error, se, out=np.full_like(se, np.inf), where=se > 0)
        z[error <= tolerance] = 0.
        assert (z < 5).all(), (cell, variant, z)
        assert abs(expected['IR'] - target_rho(cell)) < 1e-14
        summaries[variant] = dict(deployed_fixed_proposal_path_counts=dict(stats),
            seeds=seeds, expected=expected, empirical=dict(zip(['IR', 'anchors', 'calls', 'extra_LOOKs'], means.tolist())),
            max_standard_errors=float(z.max()), path_agreement=True,
            absolute_errors=error.tolist(), roundoff_tolerance=tolerance.tolist())
    report = dict(status='PASS', cell=cell, variants=summaries,
                  assumption='fixed recorded observations and retrieval proposals; no SR estimate')
    write_json(SCRATCH / 'replays' / f'cost_paths_{cell}.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['identity', 'cost'])
    parser.add_argument('--cell', choices=ALL_CELLS, required=True)
    parser.add_argument('--seeds', type=int, default=1000)
    args = parser.parse_args()
    (identity(args.cell) if args.case == 'identity' else cost_paths(args.cell, args.seeds))


if __name__ == '__main__':
    main()
