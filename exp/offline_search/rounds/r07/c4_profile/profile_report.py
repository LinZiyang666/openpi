"""Realized owner cost and descriptive paired PROFILE comparisons from ordinary logs."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import common as C

DEFAULT_KEYS = {
    'camera': ['camera_mode','stage1_mode'],
    'completion': ['camera_completion','camera_completed', 'extras.os_sw_completion_cost'],
    'completion_count': ['camera_completion_calls'],
    'granted': ['extras.os_sf_granted','blind_extras.os_sf_granted'],
    'valve_fire': ['blind_extras.os_sf_valve_fire','extras.os_sf_valve_fire'],
    'valve_stat': ['blind_extras.os_sf_delta','extras.os_sf_delta'],
    'look': ['look_reason','blind_extras.os_sf_look'],
    'wrist_reason': ['extras.os_sw_reason','blind_extras.os_sw_reason'],
    'call_weight': ['extras.os_c3_weight'],
    'call_p': ['extras.os_c_p'],
    'call_coin': ['extras.os_c_coin'],
}


def get_key(row, keys, name):
    choices=keys.get(name,[])
    for key in [choices] if isinstance(choices,str) else choices:
        value=row
        for part in key.split('.'):
            value=value.get(part) if isinstance(value,dict) else None
        if value is not None:return value
    return None


def decision_cost(row,model,keys,allow_full=False):
    vision=bool(row['vision']);call=not bool(row['hit'])
    if call and not vision:raise ValueError('policy call without vision')
    camera=get_key(row,keys,'camera')
    if camera is None and allow_full:camera='full'
    if vision and camera is None:raise ValueError('actual camera mode missing; configure keys or assert --legacy-full-camera')
    component=dict(full_looks=0.,wrist_looks=0.,completion=0.,calls=0.,blind=0.)
    if vision:
        if camera=='full':component['full_looks']=.152 if model=='pi05' else .148
        elif camera=='wrist_only' and model=='pi05':
            component['wrist_looks']=.055198
            completed=get_key(row,keys,'completion')
            count=get_key(row,keys,'completion_count')
            if count is None:count=int(bool(completed))
            if isinstance(count,bool) or not isinstance(count,(int,float)) or count<0 or int(count)!=count:
                raise ValueError('invalid actual missing-camera completion count')
            if call:
                if completed is None:raise ValueError('wrist-origin MISS lacks actual missing-camera completion telemetry')
                if not completed or count<1:raise ValueError('policy ran without missing-camera completion')
            component['completion']=.049890*count
        else:raise ValueError(f'unsupported/unpriced camera mode {model}:{camera}')
    if call:component['calls']=.848 if model=='pi05' else .852
    return component,(camera if vision else 'blind')


def control_counts(root,episodes,manifest=None):
    """Optional authoritative P3 client counts; never use n_steps incl warmup."""
    from exp.offline_search.rounds.r06.p3_profiling.read_v2 import control_join
    counts={};expected={(e['uid'],e['attempt']):e for e in episodes}
    for path in Path(root).rglob('controls.jsonl'):
        seen=[];key=None;active=0;complete=False;reset_hash=None
        for r in C.jsonl(path):
            if 'task_uid' in r:key=r['task_uid'],int(r.get('attempt',1))
            if r.get('ev')=='control':
                seen.append(int(r['control']));active+=r.get('decision_step') is not None
            if r.get('ev')=='rollout_end':complete=True
            if r.get('ev')=='reset':reset_hash=r.get('init_state_sha256')
        if key not in expected:continue
        if key in counts:raise ValueError('duplicate client control trace')
        if not complete or seen!=list(range(len(seen))):raise ValueError('incomplete/gapped authoritative controls')
        ep=expected[key]
        control_join(path,key,{'dec':{d['step']:d for d in ep['decisions']}},{'success':bool(ep['Y'])})
        if manifest and manifest.get('role')=='NONTEST_BVAL':
            selection={(r['task'],r['init']):r for r in manifest['selected']}
            selected=selection.get((ep['task'],ep['init']))
            if selected is None or reset_hash!=selected['state_sha256']:
                raise ValueError('actual reset hash differs from the audited B-val manifest')
        counts[key]=active
    return counts


def report_arm(root,spec,keys,reps,worst,legacy):
    episodes,audit=C.accepted_arm(root);model=spec['model'];cell=f"{model}_{spec.get('suite_short',spec['suite'])}_{spec.get('library_size',50)}"
    # Source library is authoritative for size/cell; no arm-name parsing required.
    suite={'libero_10':'l10','libero_spatial':'spatial','sp':'spatial'}.get(spec['suite'],spec['suite'])
    declared=spec.get('profile_cell')
    if declared:cell=declared
    else:
        libname=episodes[0]['decisions'][0]['lib']
        size=50 if libname=='current' else 500
        cell=f'{model}_{suite}_{size}'
    base,lib,manifest=C.bank(cell);table=C.stage_table(lib,manifest,base)
    manifest_path=spec.get('manifest')
    selection=json.loads(Path(manifest_path).read_text()) if manifest_path and Path(manifest_path).exists() else None
    counts=control_counts(root,episodes,selection)
    stage_counts=defaultdict(Counter);mode_counts=Counter();coverage=Counter();timeline=[];ep_out=[]
    totals=Counter();n=v=m=0
    for e in episodes:
        costs=Counter();stage='unknown';rows=[]
        for d in e['decisions']:
            if d.get('ok') is False or d.get('error'):raise ValueError('accepted decision error')
            n+=1;v+=bool(d['vision']);m+=not bool(d['hit'])
            component,camera=decision_cost(d,model,keys,legacy)
            if d.get('owner_cost') is not None and not np.isclose(d['owner_cost'],sum(component.values()),rtol=0,atol=1e-8):
                raise ValueError('actual owner-cost telemetry disagrees with component accounting')
            totals.update(component);costs.update(component);mode_counts[camera]+=1
            look_stage=stage
            if d['vision'] and len(d.get('rows',[]))==base.k:
                stage=C.stage_label(table,d['rows'],d['weights'])
            stage_counts[stage]['decisions']+=1
            diagnostic={key:get_key(d,keys,key) for key in keys}
            for key,value in diagnostic.items():coverage[key]+=value is not None
            # A grant repeated on blind checks is not another anchor grant.
            if d['vision'] and diagnostic['granted'] is not None:
                stage_counts[stage]['grant_anchors']+=int(diagnostic['granted']>0)
                stage_counts[stage]['granted_blocks']+=diagnostic['granted']
            # The anchor's blind_extras describes its preceding LOOK veto once.
            if d['vision']:
                stage_counts[look_stage]['valve_looks']+=int(bool(diagnostic['valve_fire']))
                stage_counts[look_stage]['look_'+str(diagnostic['look'])]+=1
                stage_counts[stage]['camera_'+camera]+=1
                if diagnostic['wrist_reason'] is not None:stage_counts[stage]['wrist_reason_'+str(diagnostic['wrist_reason'])]+=1
            rows.append(dict(arm=root.name,cell=cell,task=e['task'],init=e['init'],step=d['step'],stage=stage,
                             vision=d['vision'],src=d['src'],camera=camera,cost=sum(component.values()),
                             q_us=d.get('q_us'),**{k:v for k,v in diagnostic.items() if k!='camera'}))
        if lib is not None:
            chosen=np.asarray([d.get('top1',-1) for d in e['decisions']],int)
            move=C.moves(chosen,np.arange(len(chosen))>0,lib['episode'],lib['step'],lib['next'])
            for r,code in zip(rows,move):r['retrieval_move']=C.FLAG[int(code)]
        actual=counts.get((e['uid'],e['attempt']))
        ep_out.append(dict(task=e['task'],init=e['init'],uid=e['uid'],Y=e['Y'],N=len(e['decisions']),
            IR=sum(costs.values())/len(e['decisions']),components=dict(costs),
            active_controls=actual,IR_active_controls=sum(costs.values())/(actual/manifest['exec_steps']) if actual else None))
        timeline.append((ep_out[-1],rows))
    summary=json.loads((root/'summary.json').read_text())
    if summary.get('success') is not None and summary['success']!=sum(e['Y'] for e in episodes):raise ValueError('summary outcome mismatch')
    ledger=summary.get('cost_ledger',{})
    for key,value in [('decisions',n),('vision_decisions',v),('misses',m)]:
        if key in ledger and ledger[key]!=value:raise ValueError(f'summary ledger mismatch: {key}')
    worst_rows=[r for _,rs in sorted(timeline,key=lambda z:(z[0]['Y'],-z[0]['IR']))[:worst] for r in rs]
    report=dict(arm=root.name,cell=cell,variant=spec.get('variant',root.name),audit=audit,N=n,V=v,M=m,
        SR_descriptive=float(np.mean([e['Y'] for e in episodes])),IR=sum(totals.values())/n,
        IR_components={k:totals[k]/n for k in ['full_looks','wrist_looks','completion','calls','blind']},
        camera_modes=dict(mode_counts),by_stage=dict(stage_counts),telemetry_coverage=dict(coverage),
        active_control_episodes=len(counts),IR_active_controls=sum(totals.values())/(sum(counts.values())/manifest['exec_steps']) if len(counts)==len(episodes) and sum(counts.values()) else None,
        query_us=C.quant([d.get('q_us',np.nan) for e in episodes for d in e['decisions']]),episodes=ep_out)
    return report,worst_rows


def paired(a,b,reps):
    x={(r['task'],r['init']):r for r in a['episodes']};y={(r['task'],r['init']):r for r in b['episodes']}
    common=sorted(x.keys() & y.keys());records=[x[k] for k in common]
    if not common:return dict(arm=a['arm'],reference=b['arm'],matched=0)
    result=dict(arm=a['arm'],reference=b['arm'],matched=len(common),unmatched_arm=len(x)-len(common),unmatched_reference=len(y)-len(common),
                delta_IR=C.task_bootstrap(records,[x[k]['IR']-y[k]['IR'] for k in common],reps),
                delta_SR_descriptive=float(np.mean([x[k]['Y']-y[k]['Y'] for k in common])))
    result['delta_component_IR']={name:C.task_bootstrap(records,[x[k]['components'].get(name,0)/x[k]['N']-y[k]['components'].get(name,0)/y[k]['N'] for k in common],reps) for name in ['full_looks','wrist_looks','completion','calls']}
    return result


def main():
    p=C.parser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--arms',nargs='+')
    p.add_argument('--references',nargs='*',default=[]);p.add_argument('--keys',type=Path)
    p.add_argument('--legacy-full-camera',action='store_true',help='Explicitly assert every absent camera mode was full vision')
    p.add_argument('--reps',type=int,default=2000);p.add_argument('--worst',type=int,default=3)
    a=p.parse_args();keys={**DEFAULT_KEYS,**(json.loads(a.keys.read_text()) if a.keys else {})}
    specs=json.loads((a.run_root/'arms.json').read_text());specs={r.get('arm',r.get('name')):r for r in specs}
    names=a.arms or [n for n in specs if (a.run_root/'runs'/n/'summary.json').exists()]
    reports={};ts=[]
    for name in dict.fromkeys(names+a.references):
        r,t=report_arm(a.run_root/'runs'/name,specs[name],keys,a.reps,a.worst,a.legacy_full_camera)
        if r['cell'] not in C.cells(a.cells):continue
        reports[name]=r;ts.extend(t);print(name,'SR',r['SR_descriptive'],'IR',round(r['IR'],6),flush=True)
    if not reports:raise ValueError('no matching accepted arms/cells')
    pairs=[paired(reports[n],reports[ref],a.reps) for n in names for ref in a.references if n in reports and ref in reports and n!=ref and reports[n]['cell']==reports[ref]['cell']]
    C.write(a.out/'profile_report.json',dict(schema='r7.c4.profile.v1',reports=list(reports.values()),paired=pairs,keys=keys,
        caveat='SR descriptive only on non-test PROFILE starts. Request IR uses accepted actual work. Control IR only when every accepted client trace is complete; warmup excluded. Wrist/completion owner prices are the labelled R4 proportional-latency assumption. Paired IR bootstraps task/init clusters on common episodes. Costs omit profiler-only forwards and CPU time; query timing reported separately. Existing summary IR may use a different table: N/V/M reconciliation is mandatory.'))
    C.csv_write(a.out/'worst_timelines.csv',ts)


if __name__=='__main__':main()
