"""Read-only R7 profile analysis; writes E1 evidence only, no serving changes."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PROFILE=HERE.parents[1]/'profile_results'
RUN=Path('/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1/runs')


def read_episodes(root):
    accepted={}
    for line in (root/'client/journal.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'):
            accepted[r['task_uid']]=r
    groups={uid:{} for uid in accepted}
    for f in sorted(root.glob('server_*/decisions_*.jsonl')):
        for line in f.open():
            d=json.loads(line);uid=d.get('uid');j=accepted.get(uid)
            if d.get('ev')!='dec' or j is None:continue
            if int(d.get('attempt',1) or 1)!=int(j.get('attempt',1) or 1) or d.get('ts',0)>j['ts']:continue
            assert d['step'] not in groups[uid],(root.name,uid,d['step'])
            groups[uid][d['step']]=d
    assert len(groups)==20
    for uid,g in groups.items():assert sorted(g)==list(range(len(g))),uid
    return [([g[k] for k in sorted(g)],accepted[uid]) for uid,g in groups.items()]


def main():
    report=json.loads((PROFILE/'closed_loop/profile_report.json').read_text())
    metrics=[];audit=[];anomalies=[];calls=[]
    bycell={r['cell']:r for r in report['reports'] if r['arm'].endswith('_A_profile')}
    for r in report['reports']:
        name=r['arm'];cell=r['cell'];model=cell.split('_')[0];cv=.152 if model=='pi05' else .148
        variant='SW' if name.startswith('r7_sw_') else name.rsplit('_',2)[1]
        base=bycell[cell]
        pa=next((v for v in report['paired'] if v['arm']==name and v['reference']==base['arm']),None)
        gr=sum(s.get('grant_anchors',0) for s in r['by_stage'].values())
        metrics.append(dict(cell=cell,variant=variant,episodes=20,successes=round(r['SR_descriptive']*20),
                            N=r['N'],V=r['V'],M=r['M'],IR=r['IR'],mean_episode_IR=np.mean([e['IR'] for e in r['episodes']]),
                            saved_IR=base['IR']-r['IR'],paired_saved_IR=-pa['delta_IR']['estimate'] if pa else 0,
                            grants=gr,grant_share=gr/r['V'],query_mean_us=r['query_us']['mean'],query_p95_us=r['query_us']['p95'],
                            wrist_share=r['camera_modes'].get('wrist_only',0)/r['V']))
        eps=read_episodes(RUN/name);ct=Counter();cost=0.;pvals=[];weights=[];mean_nominal=[]
        threshold=np.mean(json.loads((HERE/f'calibration_{cell}.json').read_text())['gripper_centers'])
        for ds,j in eps:
            anchor=None;oldmode=None;pending_grant=False
            for d in ds:
                assert d.get('ok') is not False and d.get('exec_ok') is not False
                ex=d.get('extras',{});bx=d.get('blind_extras',{})
                ct['N']+=1;ct['V']+=bool(d['vision']);ct['M']+=not bool(d['hit'])
                camera=d.get('camera_mode','full')
                c=(.055198 if camera=='wrist_only' else cv) if d['vision'] else 0.
                c+=.049890*d.get('camera_completion_calls',0)
                c+=(1-cv)*int(not d['hit']);cost+=c
                if 'owner_cost' in d:assert abs(c-d['owner_cost'])<1e-8
                if d['vision']:
                    if d.get('look_reason')==11:
                        age=int(bx.get('os_sf_age',-1));ct['valve_looks']+=1;ct[f'valve_age_{age}']+=1
                    anchor=d
                    pending_grant=ex.get('os_sf_granted',0)>0
                    ct['raw_grants']+=pending_grant
                    if variant=='CT30' and ex.get('os_c_fresh')==1:
                        w=ex['os_c3_weight'];lam=ex['os_c3_lambda'];p=min(1.,w*lam)
                        weights.append(w);pvals.append(p);mean_nominal.append(ex['os_c_nominal_p'])
                        ct['fresh_anchors']+=1;ct['deviation_entries']+=int(ex['os_c3_dev_entry'])
                        ct['probability_clipped']+=int(p==1.)
                        ct['event_mass_positive']+=int(ex['os_c3_event_mass']>0)
                        if abs(p-ex['os_c_nominal_p'])>1e-8:ct['nominal_p_mismatch']+=1
                    if ex.get('os_c_fresh')==1:
                        ct['stall_calls']+=int(ex.get('os_c_stall_call',0))
                        ct['extra_looks']+=int(ex.get('os_c_extra_look',0))
                else:
                    de=ex if 'os_sf_extension' in ex else bx
                    if de.get('os_sf_valve_checked'):ct['valve_checked_blind']+=1
                    if de.get('os_sf_extension'):
                        ct['served_extra_blocks']+=1
                        ct[f"extra_source_{int(de['os_sf_source'])}"]+=1
                        if pending_grant:ct['grants_with_served_extension']+=1;pending_grant=False
                        assert anchor is not None and d['step']-anchor['step']==de['os_sf_age']
                        if variant.startswith('SF'):
                            assert de['os_sf_stage_ok']==1 and de['os_sf_structural']==1
                            assert de['os_sf_valve_fire']==0 and de['os_sf_delta']<=de['os_sf_radius']
                head=np.array(d['served_head']);modes=head[:,6]>threshold
                de=ex if 'os_sf_extension' in ex else bx
                if not d['vision'] and de.get('os_sf_extension'):
                    seq=np.r_[oldmode,modes] if oldmode is not None else modes
                    if np.any(seq[1:]!=seq[:-1]):
                        ct['extra_blocks_with_command_transition']+=1
                        if variant.startswith('SF'):anomalies.append(dict(arm=name,uid=d['uid'],step=d['step'],gripper=head[:,6].tolist(),previous_mode=bool(oldmode),source=de.get('os_sf_source')))
                oldmode=modes[-1]
        assert [ct[k] for k in ['N','V','M']]==[r[k] for k in ['N','V','M']]
        assert ct['raw_grants']==gr
        assert abs(cost/r['N']-r['IR'])<1e-10
        summary=json.loads((RUN/name/'summary.json').read_text())['cost_ledger']
        audit.append(dict(cell=cell,variant=variant,**ct,raw_IR=cost/r['N'],summary_default_IR=summary['ir_per_request'],
                          served_eligibility=ct['grants_with_served_extension']/r['V']))
        if variant=='CT30':calls.append(dict(cell=cell,**ct,mean_weight=np.mean(weights),max_weight=max(weights),
                                              mean_tilted_p=np.mean(pvals),mean_logged_nominal_p=np.mean(mean_nominal)))
        print(name,'reconciled',r['N'],flush=True)
    df=pd.DataFrame(metrics);df.to_csv(HERE/'profile_metrics.csv',index=False)
    pd.DataFrame(audit).fillna(0).to_csv(HERE/'profile_raw_audit.csv',index=False)
    (HERE/'profile_command_transitions.json').write_text(json.dumps(anomalies,indent=2))
    (HERE/'profile_calls_audit.json').write_text(json.dumps(calls,indent=2))
    totals=df.groupby('variant').agg(episodes=('episodes','sum'),successes=('successes','sum'),mean_IR=('IR','mean'),
                                   paired_saving=('paired_saved_IR','mean'),grant_min=('grant_share','min'),grant_max=('grant_share','max'))
    print(totals.round(6).to_string())
    print(pd.DataFrame(audit).fillna(0).to_string(index=False))
    print('SF command-transition blocks',len(anomalies))


if __name__=='__main__':main()
