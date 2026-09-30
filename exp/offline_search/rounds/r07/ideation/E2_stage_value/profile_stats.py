"""Read-only R7 PROFILE audit; writes only E2 analysis artifacts."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r07.c4_profile.common import accepted_arm

HERE = Path(__file__).resolve().parent
R7 = HERE.parents[1]
RUNS = Path('/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1/runs')
report = json.loads((R7 / 'profile_results/closed_loop/profile_report.json').read_text())
refs = {r['cell']: r for r in report['reports'] if '_A_profile' in r['arm']}
pairs = {(r['arm'], r['reference']): r for r in report['paired']}


def variant(arm):
    return 'SW' if arm.startswith('r7_sw_') else arm.removesuffix('_profile').rsplit('_', 1)[1]


metrics, details = [], {}
for r in report['reports']:
    arm, cell = r['arm'], r['cell']
    v = variant(arm)
    eps, audit = accepted_arm(RUNS / arm)
    assert len(eps) == 20 and audit['discarded_prior_prefix'] == 0
    ds = [d for e in eps for d in e['decisions']]
    anchors = [d for d in ds if d['vision']]
    assert (len(ds), len(anchors), sum(not d['hit'] for d in ds)) == (r['N'], r['V'], r['M'])
    summary = json.loads((RUNS / arm / 'summary.json').read_text())
    assert summary['success'] == sum(e['Y'] for e in eps)
    assert summary['client_decisions'] == len(ds)
    for k, n in [('decisions', len(ds)), ('vision_decisions', len(anchors)), ('misses', r['M'])]:
        assert summary['cost_ledger'][k] == n
    components = Counter()
    for d in ds:
        if d['vision']:
            cam = d.get('camera_mode', 'full')
            components['look'] += .055198 if cam == 'wrist_only' else (.152 if cell.startswith('pi05') else .148)
            components['completion'] += .049890 * d.get('camera_completion_calls', 0)
        components['call'] += (not d['hit']) * (.848 if cell.startswith('pi05') else .852)
    assert abs(sum(components.values()) / len(ds) - r['IR']) < 1e-12
    a = refs[cell]
    ys = {(e['task'], e['init']): e['Y'] for e in a['episodes']}
    assert set(ys) == {(e['task'], e['init']) for e in eps}
    wins = sum(e['Y'] > ys[e['task'], e['init']] for e in eps)
    losses = sum(e['Y'] < ys[e['task'], e['init']] for e in eps)
    grants = sum(d.get('extras', {}).get('os_sf_granted', 0) > 0 for d in anchors)
    extension_served = sum(not d['vision'] and d.get('blind_extras', {}).get('os_sf_extension', 0) > 0 for d in ds)
    valves = [d for d in anchors if d['look_reason'] == 11]
    ages = Counter(str(int(d['blind_extras']['os_sf_age'])) for d in valves)
    paired = pairs.get((arm, a['arm']), {}).get('delta_IR', {})
    row = dict(arm=arm, cell=cell, variant=v, SR=r['SR_descriptive'],
               IR=r['IR'], episode_IR=np.mean([e['IR'] for e in r['episodes']]),
               delta_IR_pooled=r['IR']-a['IR'], delta_IR_paired=paired.get('estimate', 0),
               delta_IR_lo=paired.get('lo'), delta_IR_hi=paired.get('hi'),
               N=r['N'], V=r['V'], M=r['M'], wins_A=wins, losses_A=losses,
               grants=grants, eligible=grants/len(anchors), extension_served=extension_served,
               valves=len(valves), early_valves=ages.get('1', 0),
               wrist=r['camera_modes'].get('wrist_only', 0),
               summary_IR=summary['cost_ledger']['ir_per_request'])
    metrics.append(row)
    detail = dict(valve_ages=ages, query_us=r['query_us'])
    if v in ('CU30', 'CT30'):
        fresh = [d for d in ds if d.get('extras', {}).get('os_c_fresh') == 1]
        assert len(fresh) == len(anchors)
        x = [d['extras'] for d in fresh]
        assert all(int(z['os_c_coin'] < z['os_c_p']) == int(not d['hit']) for z, d in zip(x, fresh))
        detail.update(fresh=len(fresh), reasons=Counter(int(z['os_reason']) for z in x),
                      saturated=sum(z['os_c_p'] == 1 for z in x),
                      zero_p=sum(z['os_c_p'] == 0 for z in x),
                      stall=Counter(int(z['os_c_stall_state']) for z in x),
                      extra_look=sum(d['look_reason'] == 8 for d in anchors))
        if v == 'CT30':
            for z in x:
                weight = (1 + z['os_c3_event_mass'] * (1/z['os_c3_h']-1)) * (1/z['os_c3_h_dev'] if z['os_c3_dev_entry'] else 1)
                assert np.isclose(weight, z['os_c3_weight'], rtol=0, atol=1e-10)
                assert np.isclose(min(1., weight*z['os_c3_lambda']), z['os_c_nominal_p'], rtol=0, atol=1e-10)
            detail.update(entries=sum(z['os_c3_dev_entry'] for z in x),
                          entry_calls=sum(z['os_c3_dev_entry'] and not d['hit'] for z, d in zip(x, fresh)),
                          entry_nominal_saturated=sum(z['os_c3_dev_entry'] and z['os_c_nominal_p']==1 for z in x),
                          entry_cooldown=sum(z['os_c3_dev_entry'] and z['os_c_cooldown']>0 for z in x),
                          entry_stall_calls=sum(z['os_c3_dev_entry'] and z['os_c_stall_call']>0 for z in x),
                          high=sum(z['os_c3_deviation'] > z['os_c3_p75'] for z in x),
                          event_positive=sum(z['os_c3_event_mass'] > 0 for z in x),
                          tilted=sum(z['os_c3_weight'] > 1 for z in x),
                          nominal_saturated=sum(z['os_c_nominal_p'] == 1 for z in x),
                          interior_p=x[0]['os_c3_lambda'],
                          entry_by_task=dict(Counter(d['task_id'] for d in fresh if d['extras']['os_c3_dev_entry'])),
                          entry_episodes=len({d['uid'] for d in fresh if d['extras']['os_c3_dev_entry']}))
    details[arm] = detail

value = json.loads((R7 / 'profile_results/offline/value_all/stage_value.json').read_text())
first = [dict(arm=r['audit']['arm'], **s) for r in value['reports'] for s in r['results'] if s['estimand']=='first_entry' and s['dose_target']=='natural']
usable = [r for r in first if r['effect']['lo'] is not None]
value_summary = dict(arms=len(value['reports']), comparisons=len(first), usable=len(usable),
                     supported=sum(r['audit']['supported_anchors'] for r in value['reports']),
                     excludes_zero=[r for r in usable if r['effect']['lo']>0 or r['effect']['hi']<0],
                     stage_names=sorted({r['stage'] for r in first}))
clock = json.loads((R7 / 'profile_results/offline/clock_all/failure_clock.json').read_text())
ceps = [e for r in clock['reports'] for e in r['episodes']]
clock_summary = {}
for y in (0, 1):
    es = [e for e in ceps if e['Y']==y]
    clock_summary[y] = dict(episodes=len(es), **{key:sum(e[key] is not None for e in es) for key in ('first_state_deviation','first_valve_alert','first_confirmed_stall')})
    clock_summary[y]['dev_before_first_command_transition'] = sum(e['first_state_deviation'] is not None and bool(e['wire_command_transition_controls']) and e['first_state_deviation'] < e['wire_command_transition_controls'][0] for e in es)
    clock_summary[y]['valve_before_stall'] = sum(e['first_valve_alert'] is not None and e['first_confirmed_stall'] is not None and e['first_valve_alert'] < e['first_confirmed_stall'] for e in es)
    clock_summary[y]['both_valve_stall'] = sum(e['first_valve_alert'] is not None and e['first_confirmed_stall'] is not None for e in es)
    leads = [e['valve_to_terminal_controls'] for e in es if e['first_valve_alert'] is not None]
    clock_summary[y]['valve_to_terminal_median'] = float(np.median(leads)) if leads else None
    clock_summary[y]['stall_by_campaign'] = {campaign:dict(episodes=sum(e['campaign']==campaign for e in es), alerts=sum(e['campaign']==campaign and e['first_confirmed_stall'] is not None for e in es)) for campaign in ('bval','p3')}
with (R7/'profile_results/offline/clock_all/failure_decisions.csv').open() as f:
    stall_coverage = Counter((r['campaign'],r['stall']) for r in csv.DictReader(f))
clock_summary['stall_decision_coverage'] = {f'{c}:{s}':n for (c,s),n in stall_coverage.items()}

(HERE/'profile_stats.json').write_text(json.dumps(dict(metrics=metrics,details=details,value=value_summary,clock=clock_summary),indent=2)+'\n')
with (HERE/'profile_metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
print('Verified',len(metrics),'arms,',sum(r['N'] for r in metrics),'decisions; 20 matched episodes each; no count, cost, coin or CT-weight mismatch.')
print('CLOCK',clock_summary)
for v in ['SF1','SF2','UF1','SW','CU30','CT30']:
    rows=[r for r in metrics if r['variant']==v]
    print(v,'successes',sum(round(r['SR']*20) for r in rows),'/',len(rows)*20,'IR',np.mean([r['IR'] for r in rows]),'paired deltaA',np.mean([r['delta_IR_paired'] for r in rows]),'grants range',min(r['eligible'] for r in rows),max(r['eligible'] for r in rows),'valves/early/served',sum(r['valves'] for r in rows),sum(r['early_valves'] for r in rows),sum(r['extension_served'] for r in rows))
for r in metrics:
    if r['variant'] in ('CT30','CU30'):print(r['arm'],details[r['arm']])
print('VALUE',len(first),len(usable),'exclude zero',len(value_summary['excludes_zero']))
