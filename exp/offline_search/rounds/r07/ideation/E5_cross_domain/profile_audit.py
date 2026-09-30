"""E5 read-only audit of accepted R7 PROFILE requests and saved offline results."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r07.c4_profile.common import accepted_arm

HERE = Path(__file__).resolve().parent
BASE = Path('exp/offline_search/rounds/r07/profile_results')
RUN = Path('/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1/runs')


def variant(name):
    if name.startswith('r7_sw_'):
        return 'SW'
    return name.split('_')[-2]


def contrast(a, b):
    aa = {(e['task'], e['init']): e for e in a['episodes']}
    bb = {(e['task'], e['init']): e for e in b['episodes']}
    assert aa.keys() == bb.keys() and len(aa) == 20
    return dict(n=20, gain=sum(aa[k]['Y'] > bb[k]['Y'] for k in aa),
                loss=sum(aa[k]['Y'] < bb[k]['Y'] for k in aa),
                delta_mean_episode_IR=float(np.mean([aa[k]['IR'] - bb[k]['IR'] for k in aa])),
                delta_request_IR=a['IR'] - b['IR'])


def main():
    report = json.loads((BASE/'closed_loop/profile_report.json').read_text())
    reports = {(r['cell'], variant(r['arm'])): r for r in report['reports']}
    out = dict(arms=[], family={}, contrasts=[], offline={})
    for (cell, v), r in sorted(reports.items()):
        eps, audit = accepted_arm(RUN/r['arm'])
        summary = json.loads((RUN/r['arm']/'summary.json').read_text())
        counter, sources, valve_ages = Counter(), Counter(), Counter()
        cost, times_look, times_blind, events = 0., [], [], []
        for e in eps:
            prev_mode = None
            prev_stage1 = None
            for d in e['decisions']:
                ex, bx = d.get('extras', {}), d.get('blind_extras') or {}
                counter['N'] += 1
                counter['V'] += bool(d['vision'])
                counter['M'] += not bool(d['hit'])
                cv = .152 if cell.startswith('pi05') else .148
                camera = d.get('camera_mode', 'full') if d['vision'] else 'blind'
                completion = d.get('camera_completion_calls', 0)
                dc = (0 if not d['vision'] else .055198 if camera == 'wrist_only' else cv)
                dc += .049890 * completion + (1-cv) * (not bool(d['hit']))
                cost += dc
                if 'owner_cost' in d:
                    assert abs(dc-d['owner_cost']) < 1e-8
                counter['camera_'+camera] += 1
                counter['completions'] += completion
                if 'stage1_calls' in d:
                    if prev_stage1 is None:
                        prev_stage1 = d['stage1_calls'] - int(bool(d['vision']))
                    increment = d['stage1_calls'] - prev_stage1
                    counter['stage1_dispatches'] += increment
                    counter['stage1_count_mismatch'] += increment != int(bool(d['vision']))
                    prev_stage1 = d['stage1_calls']
                if 'camera_stage1_calls' in d:
                    counter['camera_count_mismatch'] += d['camera_stage1_calls'] != int(bool(d['vision']))
                (times_look if d['vision'] else times_blind).append(d['q_us'])
                if d['vision']:
                    counter['grant_anchors'] += ex.get('os_sf_granted', 0) > 0
                    counter['granted_blocks'] += ex.get('os_sf_granted', 0)
                    if bx.get('os_sf_valve_fire'):
                        counter['valve_looks'] += 1
                        age = int(bx.get('os_sf_age', -1))
                        valve_ages[age] += 1
                        events.append(dict(uid=e['uid'], step=d['step'], age=age, Y=e['Y'],
                                           delta=bx.get('os_sf_delta'), radius=bx.get('os_sf_radius')))
                    if 'os_c_fresh' in ex:
                        counter['fresh_calls'] += ex.get('os_c_call', 0)
                        counter['stall_calls'] += ex.get('os_c_stall_call', 0)
                        counter['call_coin_mismatch'] += bool(ex.get('os_c_call')) != (ex['os_c_coin'] < ex['os_c_p'])
                        if v == 'CT30':
                            h, hd = ex['os_c3_h'], ex['os_c3_h_dev']
                            w = (1 + ex['os_c3_event_mass']*(1/h-1)) if h > 0 else 1.
                            if ex['os_c3_dev_entry'] and hd > 0:
                                w /= hd
                            counter['weight_mismatch'] += abs(w-ex['os_c3_weight']) > 1e-7
                            counter['dev_entries'] += ex['os_c3_dev_entry']
                            counter['nominal_p_mismatch'] += abs(min(1,ex['os_c3_lambda']*w)-ex['os_c_nominal_p']) > 1e-7
                extra = (not d['vision']) and (ex.get('os_sf_extension', bx.get('os_sf_extension', 0)) == 1)
                counter['extension_requests'] += extra
                if extra:
                    sources[int(ex.get('os_sf_source', bx.get('os_sf_source', -1)))] += 1
                # Requested normalized gripper sign transitions, not verified physical controls/events.
                head = np.asarray(d.get('served_head', []))
                if len(head):
                    mode = head[:, 6] >= 0
                    if extra:
                        flips = np.count_nonzero(mode[1:] != mode[:-1]) + int(prev_mode is not None and mode[0] != prev_mode)
                        counter['extension_requests_with_command_flip'] += flips > 0
                        counter['extension_command_flips'] += int(flips)
                    prev_mode = bool(mode[-1])
        assert [counter[k] for k in ['N','V','M']] == [r[k] for k in ['N','V','M']]
        assert abs(cost / counter['N'] - r['IR']) < 1e-12
        assert sum(e['Y'] for e in eps) == summary['success']
        reference = reports[(cell, 'A')]
        arm = dict(cell=cell, variant=v, arm=r['arm'], audit=audit, success=summary['success'],
                   request_IR=r['IR'], mean_episode_IR=float(np.mean([e['IR'] for e in r['episodes']])),
                   legacy_summary_IR=summary['cost_ledger']['ir_per_request'],
                   grant_share=counter['grant_anchors']/counter['V'], counters=dict(counter),
                   source_counts=dict(sources), valve_ages=dict(valve_ages), valve_events=events,
                   query_look_us_median=float(np.median(times_look)),
                   query_blind_us_median=float(np.median(times_blind)), versus_A=contrast(r,reference))
        if v == 'CT30':
            arm['versus_CU'] = contrast(r, reports[(cell,'CU30')])
        if v == 'SF2':
            arm['versus_SF1'] = contrast(r,reports[(cell,'SF1')])
        out['arms'].append(arm)
        print(cell, v, 'SR',summary['success'],'IR',round(r['IR'],6),'grant',round(arm['grant_share'],4),
              'ext',counter['extension_requests'],'valve',dict(valve_ages), 'flip',counter['extension_requests_with_command_flip'])
    for v in ['A','SF1','SF2','UF1','SW','CU30','CT30']:
        aa=[a for a in out['arms'] if a['variant']==v]
        out['family'][v] = dict(arms=len(aa), episodes=20*len(aa), successes=sum(a['success'] for a in aa),
                              mean_cell_IR=float(np.mean([a['request_IR'] for a in aa])),
                              gains_A=sum(a['versus_A']['gain'] for a in aa), losses_A=sum(a['versus_A']['loss'] for a in aa),
                              counters=dict(sum((Counter(a['counters']) for a in aa),Counter())))
    val=json.loads((BASE/'offline/value_all/stage_value.json').read_text())
    selected=[x for r in val['reports'] for x in r['results'] if x['estimand']=='first_entry' and x['dose_target']=='natural']
    intervals=[x for x in selected if x['effect'].get('lo') is not None]
    out['offline']['stage_value'] = dict(comparisons=len(selected), intervals=len(intervals),
                                       excludes_zero=sum(x['effect']['lo']>0 or x['effect']['hi']<0 for x in intervals),
                                       supporting=[x for x in intervals if x['effect']['lo']>0 or x['effect']['hi']<0])
    clock=json.loads((BASE/'offline/clock_all/failure_clock.json').read_text())
    failed=[e for r in clock['reports'] for e in r['episodes'] if not e['Y']]
    out['offline']['failure_clock']=dict(failures=len(failed),absolute=sum(e['first_state_deviation'] is not None for e in failed),
                                        valve=sum(e['first_valve_alert'] is not None for e in failed),
                                        stall=sum(e['first_confirmed_stall'] is not None for e in failed))
    (HERE/'profile_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print('FAMILY',json.dumps(out['family']))


if __name__ == '__main__':
    main()
