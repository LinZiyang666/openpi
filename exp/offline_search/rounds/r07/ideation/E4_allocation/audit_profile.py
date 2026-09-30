"""Read-only R7 profile audit; writes only E4's analysis artifact."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1/runs')
REPORT = HERE.parents[1] / 'profile_results/closed_loop/profile_report.json'


def main():
    report = json.loads(REPORT.read_text())
    refs = {r['cell']: r for r in report['reports'] if '_A_' in r['arm']}
    result = {}
    for r in report['reports']:
        root = ROOT / r['arm']
        summary = json.loads((root/'summary.json').read_text())
        accepted = {}
        for line in (root/'client/journal.jsonl').open():
            j=json.loads(line)
            if j.get('accepted') and j.get('status') in ('done','failed') and not j.get('error'):
                accepted[j['task_uid']]=j
        assert len(accepted)==20
        by_episode=defaultdict(list)
        for path in sorted(root.glob('server_*/decisions_*.jsonl')):
            for line in path.open():
                d=json.loads(line)
                if d.get('ev')=='dec' and d.get('uid') in accepted and d.get('attempt')==accepted[d['uid']]['attempt']:
                    by_episode[d['uid']].append(d)
        counts=Counter(); ct_bins=defaultdict(Counter); owner_cost=0.
        c1=.152 if r['cell'].startswith('pi05') else .148
        for uid, rows in by_episode.items():
            rows.sort(key=lambda d:d['step'])
            assert [d['step'] for d in rows]==list(range(len(rows)))
            latched=False
            for d in rows:
                counts['N']+=1;counts['V']+=bool(d['vision']);counts['M']+=not d['hit']
                assert d.get('ok') is not False
                decision_cost=(.055198 if d.get('camera_mode')=='wrist_only' else c1) if d['vision'] else 0.
                decision_cost+=(1-c1)*(not d['hit'])+.049890*d.get('camera_completion_calls',0)
                owner_cost+=decision_cost
                if d.get('owner_cost') is not None:
                    assert math.isclose(decision_cost,d['owner_cost'],abs_tol=1e-12)
                x=d.get('extras') or {}; b=d.get('blind_extras') or {}
                if d['vision'] and d.get('look_reason')==11:
                    counts['valve_looks']+=1
                    counts['valve_age_'+str(b.get('os_sf_age'))]+=1
                    counts['valve_before_first_tail']+=b.get('os_sf_age')==1
                if 'os_sf_extension' in x and not d['vision'] and x['os_sf_extension']:
                    counts['extended_decisions']+=1
                if d.get('camera_mode'):
                    counts['camera_'+d['camera_mode']]+=1
                    counts['camera_completion_calls']+=int(d.get('camera_completion_calls',0))
                if d['vision'] and 'os_c_fresh' in x:
                    assert x['os_c_fresh']==1
                    assert bool(x['os_c_call']) == (not d['hit'])
                    assert (x['os_c_coin']<x['os_c_p']) == (not d['hit'])
                    counts['fresh_call_decisions']+=1
                    counts['reason_'+str(int(x['os_reason']))]+=1
                    counts['nominal_saturated']+=x['os_c_nominal_p']==1
                    counts['actual_p0']+=x['os_c_p']==0
                    counts['actual_p1']+=x['os_c_p']==1
                    if 'os_c3_weight' in x:
                        event=x['os_c3_event_mass']; h=x['os_c3_h']; hd=x['os_c3_h_dev']
                        entry=bool(x['os_c3_dev_entry'])
                        weight=(1+event*(1/h-1) if h>0 else 1)*(1/hd if entry and hd>0 else 1)
                        assert math.isclose(weight,x['os_c3_weight'],abs_tol=1e-8)
                        assert math.isclose(min(1,weight*x['os_c3_lambda']),x['os_c_nominal_p'],abs_tol=1e-8)
                        if 'os_c3_deviation' in x and 'os_c3_p75' in x:
                            high=x['os_c3_deviation']>x['os_c3_p75']
                            assert entry==(high and not latched)
                            latched=high
                            assert bool(x['os_c3_dev_latched'])==latched
                        counts['ct_entries']+=entry
                        counts['ct_event_nonzero']+=event>0
                        counts['ct_weighted']+=weight>1+1e-8
                        counts['ct_overlap']+=entry and event>0
                        group='entry' if entry else 'event' if event>0 else 'interior_or_unknown'
                        ct_bins[group]['anchors']+=1
                        ct_bins[group]['calls']+=not d['hit']
                        ct_bins[group]['nominal_p_sum']+=x['os_c_nominal_p']
                        ct_bins[group]['p_sum']+=x['os_c_p']
        assert all(counts[k]==r[k] for k in ('N','V','M'))
        assert math.isclose(owner_cost/counts['N'],r['IR'],abs_tol=1e-12)
        ledger=summary['cost_ledger']
        assert (counts['N'],counts['V'],counts['M'])==(ledger['decisions'],ledger['vision_decisions'],ledger['misses'])
        assert sum(int(j['success']) for j in accepted.values())==summary['success']==round(r['SR_descriptive']*20)
        a=refs[r['cell']]
        me=lambda z: sum(e['IR'] for e in z['episodes'])/len(z['episodes'])
        ae={(e['task'],e['init']):e for e in a['episodes']}
        assert set(ae)=={(e['task'],e['init']) for e in r['episodes']}
        grants=sum(s.get('grant_anchors',0) for s in r['by_stage'].values())
        result[r['arm']]=dict(cell=r['cell'],counts=dict(counts),ct_bins=dict(ct_bins),
            IR=r['IR'],mean_episode_IR=me(r),summary_other_basis_IR=ledger['ir_per_five_controls'],
            success=summary['success'],grants=grants,grant_fraction=grants/r['V'],
            saving_from_A_aggregate=a['IR']-r['IR'],saving_from_A_episode_mean=me(a)-me(r),
            win_A=sum(e['Y']>ae[e['task'],e['init']]['Y'] for e in r['episodes']),
            lose_A=sum(e['Y']<ae[e['task'],e['init']]['Y'] for e in r['episodes']),
            cost_per_episode=r['IR']*r['N']/20)
        print(r['arm'],json.dumps(result[r['arm']]),flush=True)
    (HERE/'profile_audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
