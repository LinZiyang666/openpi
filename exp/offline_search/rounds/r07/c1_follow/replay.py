"""Recorded P3/B-val identity and fixed-anchor B-val extension audits.

States/actions stay those of the recording; modeled IR is a cycle projection,
never a realized changed-controller IR or a causal success-rate estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from exp.offline_search.closed_loop.blind import BlindQueryView, BlindResult, LookReason
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sources, load_base
from exp.offline_search.rounds.r07.stages.stages import StageTable
from .methods import FollowExtension, StageFollow
from .prefit import OUT, ROOT, adapt, kwargs_for, VARIANTS

RUNS = Path('/home/weiland/trace_runs/os_closed_loop')


def rows_for(cell, campaign, table):
    folder = cell.replace('spatial_', 'sp_') if campaign == 'p3' else cell
    run = 'r06_p3_pilot' if campaign == 'p3' else 'r06_c_cal'
    with (RUNS/run/'tables'/folder/(table + '.csv')).open() as f:
        for row in csv.DictReader(f):
            if campaign != 'p3' or row['arm'].endswith('_A_r0'):
                yield row


def excluded_dist(self, q):
    result = list(self._replay_original_dist(q))
    ids = self._replay_exclusions['tasks'][str(int(q.task_id))]['source_episodes']
    if ids:
        mask = np.isin(self.lib_ep[result[0].rows], ids)
        d, dt = np.array(result[7], copy=True), np.array(result[-1], copy=True)
        d[mask] = np.inf; dt[mask] = np.inf
        result[7], result[8], result[-1] = d, float(np.median(d[~mask])) + 1e-12, dt
    return tuple(result)


def attach_exclusion(obj, cell, campaign):
    if campaign == 'bval':
        path = Path('/tmp/q1_method_c_fits')/cell/'recording_exclusions.json'
        if not path.exists():
            raise FileNotFoundError('exact B-val replay requires existing exclusion map: ' + str(path))
        obj._replay_exclusions = json.loads(path.read_text())
        obj._replay_original_dist = obj._dist
        obj._dist = MethodType(excluded_dist, obj)


def follow(base, source, table, variant='SF1', off=False):
    kw = kwargs_for(source, variant)
    if off:
        kw['extend_blocks'] = 0
    cls = VARIANTS[variant][0]
    obj = adapt(base, kw, cls)
    obj.follow_table = table
    obj.follow_component = FollowExtension(table, extend_blocks=kw['extend_blocks'],
                                           stage_gate=kw['stage_gate'], state_valve=kw['state_valve'])
    return obj


def same_action(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def same_result(a, b, extras=False):
    assert type(a) is type(b)
    if isinstance(a, LookReason):
        assert a == b
    elif isinstance(a, BlindResult):
        same_action(a.action, b.action)
        assert a.rows.tobytes() == b.rows.tobytes() and a.weights.tobytes() == b.weights.tobytes() and a.library == b.library
        if extras:
            assert a.extras == b.extras
    else:
        same_action(a.action, b.action)
        assert a.topk.tobytes() == b.topk.tobytes() and a.scores.tobytes() == b.scores.tobytes()
        assert a.confidence == b.confidence and a.library == b.library
        if extras:
            assert a.extras == b.extras


def identity(cell, source, table, campaign):
    base, _blob = load_base(source)
    disabled = follow(base, source, table, off=True)
    disabled_uniform = follow(base, source, table, variant='UF1', off=True)
    gated = follow(base, source, table)
    for method in (base, disabled, disabled_uniform, gated):
        attach_exclusion(method, cell, campaign)
    counts = dict(episodes=0, decisions=0, anchors=0, blind=0, looks=0,
                  recorded_action_equal=0, recorded_vision_equal=0, recorded_verdict_equal=0,
                  disabled_methods=2, rejected_anchor_decisions=0, rejected_anchors=0,
                  executed_controls=0, parity_failures=0)
    uid, age, rejected = None, 0, False
    histories = None
    for row in rows_for(cell, campaign, 'decisions'):
        step = int(row['step'])
        if row['uid'] != uid:
            uid = row['uid']; counts['episodes'] += 1; age = 0
            episode = SimpleNamespace(uid=uid, init=int(row['init']))
            for method in (base, disabled, disabled_uniform, gated):
                method.reset(episode)
            histories = dict(rs=[], actions=[], vision=[], v0=[], v1=[])
        assert step == len(histories['rs'])
        with np.load(row['absolute_input_archive']) as z:
            current = {k: np.array(z[k], copy=True) for k in ('robot_state','raw_state','vision_0','vision_1','executed_chunk')}
        hrs = np.asarray(histories['rs'], np.float32).reshape(step, -1) if step else np.empty((0,len(current['robot_state'])),np.float32)
        ha = np.asarray(histories['actions'],np.float32) if step else np.empty((0,*current['executed_chunk'].shape),np.float32)
        bq = BlindQueryView(step, int(row['task_id']), episode, current['robot_state'], current['raw_state'],
                            True if step else None, ha[-1] if step else None, ha, np.ones(step,np.int8),
                            hrs, np.asarray(histories['vision'],bool), age)
        a, b = base.blind_step(bq), disabled.blind_step(bq)
        same_result(a, b, extras=True)
        same_result(a, disabled_uniform.blind_step(bq), extras=True)
        assert base.last_blind_extras == disabled.last_blind_extras
        look = isinstance(a, LookReason)
        assert look == (row['vision'] == 'True')
        assert row['source'] == 'cache'  # all accepted verdicts in this A cohort
        counts['recorded_vision_equal'] += 1; counts['recorded_verdict_equal'] += 1
        if rejected:
            g = gated.blind_step(bq)
            same_result(a, g)
            counts['rejected_anchor_decisions'] += 1
        if look:
            counts['looks'] += 1
            q = SimpleNamespace(step=step, task_id=int(row['task_id']), episode=episode,
                                rs=current['robot_state'], raw_state=current['raw_state'],
                                key_v0=current['vision_0'], key_v1=current['vision_1'],
                                hist_key_v0=np.asarray(histories['v0'], np.float32),
                                hist_key_v1=np.asarray(histories['v1'], np.float32),
                                hist_rs=hrs, hist_a_exec=ha, hist_has_vision=np.asarray(histories['vision'],bool),
                                prev_a_exec=ha[-1] if step else None, prev_hit=True if step else None)
            a, b, g = base.query(q), disabled.query(q), gated.query(q)
            same_result(a, b, extras=True); same_result(a, g)
            same_result(a, disabled_uniform.query(q), extras=True)
            rejected = gated._follow_plan.blocks == 0
            if rejected:
                counts['rejected_anchors'] += 1
            counts['anchors'] += 1; age = 0
        else:
            counts['blind'] += 1; age += 1
        same_action(a.action, b.action)
        if a.action.tobytes() == current['executed_chunk'].tobytes():
            counts['recorded_action_equal'] += 1
        histories['rs'].append(current['robot_state']); histories['actions'].append(a.action)
        histories['vision'].append(look); histories['v0'].append(current['vision_0']); histories['v1'].append(current['vision_1'])
        counts['decisions'] += 1; counts['executed_controls'] += int(row['actual_controls'])
    return counts


def extensions(cell, source, table):
    base, _blob = load_base(source)
    methods = {v:follow(base,source,table,v) for v in VARIANTS}
    states, decision_by_ep = {}, defaultdict(dict)
    for row in rows_for(cell, 'bval', 'decisions'):
        with np.load(row['absolute_input_archive']) as z:
            state = np.array(z['robot_state'],copy=True)
        states[row['uid'], int(row['step'])] = state
        decision_by_ep[row['uid']][int(row['step'])] = row
    reports = {v:dict(anchors=0, structural=0, stage=0, granted=0, extension_anchors=0,
                      extra_blocks=0, complete_cycles=0, complete_cycle_blocks=0, censored=0,
                      valve_checks=0, valve_fires=0, first_blind_valve_fires=0,
                      timing_us=[], checks_by_age={}) for v in methods}
    timeline = []
    for row in rows_for(cell, 'bval', 'anchors'):
        uid, step = row['uid'], int(row['step'])
        record = decision_by_ep[uid][step]
        with np.load(record['absolute_input_archive']) as z:
            action = np.array(z['executed_chunk'],copy=True)
        rows = np.asarray(json.loads(row['retrieval.rows']), np.int64)
        weights = np.asarray(json.loads(row['retrieval.weights']), np.float32)
        assert len(rows) == base.k and weights.shape == rows.shape
        anchor = dict(rows=rows, weights=weights, rs=states[uid,step][:len(table.state_scale)].astype(np.float32),
                      action=action, step=step, last_step=step, task=int(row['task_id']), episode=uid, phase=rows.copy())
        for variant, method in methods.items():
            a = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in anchor.items()}
            plan = method.follow_component.plan(a)
            method._anchor, method._follow_plan = a, plan
            stats = reports[variant]
            stats['anchors'] += 1; stats['structural'] += plan.structural; stats['stage'] += plan.stage_ok
            stats['granted'] += bool(plan.blocks)
            extra, ended, censored, cycle_blocks = 0, False, False, 2
            # Include the cap LOOK check for full per-blind timing/cadence audit.
            for age in range(1, table.reference_blocks + plan.blocks + 1):
                current = states.get((uid, step+age))
                if current is None:
                    censored = True; break
                hrs = np.asarray([states[uid,s] for s in range(step+age)],np.float32)
                bq = BlindQueryView(step+age, a['task'], SimpleNamespace(uid=uid), current, current,
                                    True, action, np.empty((0,*action.shape),np.float32),
                                    np.ones(step+age,np.int8), hrs, np.r_[np.ones(step+1,bool),np.zeros(age-1,bool)],age-1)
                start = time.perf_counter_ns(); result = method.blind_step(bq)
                stats['timing_us'].append((time.perf_counter_ns()-start)/1000)
                ex = method.last_blind_extras
                checked = int(ex.get('os_sf_valve_checked',0)); fired = int(ex.get('os_sf_valve_fire',0))
                stats['valve_checks'] += checked; stats['valve_fires'] += fired
                stats['checks_by_age'][str(age)] = stats['checks_by_age'].get(str(age),0)+1
                stats['first_blind_valve_fires'] += bool(fired and age == 1)
                if isinstance(result, LookReason):
                    cycle_blocks = age; ended = True; break
                if age >= table.reference_blocks:
                    extra += 1
                    # Independently validate recorded head/native-tail recipe.
                    expected = (action[age*table.exec_steps:(age+1)*table.exec_steps,:int(table.manifest['act_valid_dims'])]
                                if (age+1)*table.exec_steps <= len(action) else
                                np.tensordot(weights,base.act[table.advance(rows,age),:table.exec_steps,:int(table.manifest['act_valid_dims'])],1))
                    same_action(result.action[:table.exec_steps,:int(table.manifest['act_valid_dims'])],expected)
                    assert result.weights.tobytes() == weights.tobytes()
            stats['extra_blocks'] += extra; stats['extension_anchors'] += bool(extra)
            stats['censored'] += censored
            if ended:
                stats['complete_cycles'] += 1; stats['complete_cycle_blocks'] += cycle_blocks
            timeline.append(dict(cell=cell,variant=variant,uid=uid,step=step,structural=plan.structural,
                                 stage_ok=plan.stage_ok,granted=plan.blocks,extra_blocks=extra,
                                 censored=censored,cycle_blocks=cycle_blocks if ended else None))
    vision_cost = .152 if cell.startswith('pi05_') else .148
    for stats in reports.values():
        timings = np.asarray(stats.pop('timing_us'))
        stats.update(granted_share=stats['granted']/stats['anchors'],
                     extension_share=stats['extension_anchors']/stats['anchors'],
                     valve_fire_rate=stats['valve_fires']/stats['valve_checks'] if stats['valve_checks'] else None,
                     modeled_A_IR=vision_cost/2,
                     modeled_IR=vision_cost*stats['complete_cycles']/stats['complete_cycle_blocks'] if stats['complete_cycle_blocks'] else None,
                     timing=dict(n=len(timings),mean_us=float(timings.mean()),p50_us=float(np.median(timings)),p95_us=float(np.quantile(timings,.95))))
    return reports,timeline


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cells',nargs='+',default=['all'])
    ap.add_argument('--skip-identity',action='store_true'); args=ap.parse_args()
    for cell,source in sources().items():
        if args.cells!=['all'] and cell not in args.cells:
            continue
        start=time.perf_counter()
        table=StageTable.load(OUT/'stages'/(cell+'.pkl'))
        result=dict(cell=cell,identity={} if args.skip_identity else
                    {campaign:identity(cell,source,table,campaign) for campaign in ('p3','bval')})
        result['extensions'],timeline=extensions(cell,source,table)
        result['seconds']=time.perf_counter()-start
        (OUT/(cell+'_replay.json')).write_text(json.dumps(result,indent=2)+'\n')
        (OUT/(cell+'_timeline.json')).write_text(json.dumps(timeline,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
