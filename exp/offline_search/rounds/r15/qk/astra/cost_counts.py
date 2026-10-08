"""Reconstruct accepted B-only decision counts; no outcome-based key selection."""
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,RUNS,REPO,dump

def read(path,inputs):
    data=path.read_bytes();inputs[str(path)]=hashlib.sha256(data).hexdigest();return data

def readarm(root,row,inputs):
    assert row['dev'] is True and row['init_pool']=='B'
    run=root/'runs'/row['arm'];accepted={}
    for line in read(run/'client/journal.jsonl',inputs).splitlines():
        r=json.loads(line);assert r.get('init_pool')=='B'
        if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'):accepted[r['task_uid']]=r
    summary=json.loads(read(run/'summary.json',inputs));assert summary['init_pool']=='B'
    assert len(accepted)==summary['complete']
    dec={};reject=0
    for path in sorted(run.glob('server_*/decisions_*.jsonl')):
        for line in read(path,inputs).splitlines():
            d=json.loads(line)
            if d.get('ev')!='dec':continue
            uid=d['uid'];assert d.get('init_pool','B')=='B'
            if uid not in accepted or int(d.get('attempt',1) or 1)!=int(accepted[uid].get('attempt',1) or 1):reject+=1;continue
            assert d['ok'] and type(d['hit']) is bool and type(d['vision']) is bool
            assert d.get('replan_steps',5)==5 and d.get('stage1_mode','full')=='full'
            dec[(uid,int(d['step']))]=d
    bytask=defaultdict(lambda:{'N':0,'V':0,'M':0,'episodes':0})
    for uid in accepted:
        steps=sorted(s for u,s in dec if u==uid);assert steps==list(range(len(steps))) and steps
        task=int(uid.split(':')[-2]);bytask[task]['episodes']+=1
    for (uid,step),d in dec.items():
        task=int(d['task_id']);v=bytask[task];v['N']+=1;v['V']+=int(d['vision']);v['M']+=int(not d['hit'])
    totals={k:sum(v[k] for v in bytask.values()) for k in ('N','V','M','episodes')}
    ledger=summary['cost_ledger'];assert [totals[k] for k in ('N','V','M')]==[ledger[k] for k in ('decisions','vision_decisions','misses')]
    return {'root':str(root),'arm':row['arm'],'model':row['model'],'suite':row.get('suite_short',row.get('suite')),'kwargs':row['kwargs'],'totals':totals,'tasks':bytask,'rejected_attempt_decisions':reject,'ledger_reconciled':True}

def project(v,cv,epsilon):
    N,V,M=(v[k] for k in ('N','V','M'))
    old=(cv*V+(1-cv)*M)/N;new=(epsilon*V+M)/N
    return {**v,'current_ir':old,'new_ir':new,'absolute_saved':old-new,'relative_saved':(old-new)/old,'max_extra_calls_per_100_looks':100*(cv*(V-M)-epsilon*V)/V}

def main():
    inputs={};data=[];unavailable=[]
    for roundname in ('r13_pilot','r14_descent'):
        for model in ('pi05','groot'):
            for suite in ('l10','spatial'):
                root=RUNS/f'{roundname}_{model}_{suite}_50'
                if not (root/'arms.json').exists():unavailable.append(str(root));continue
                arms=json.loads(read(root/'arms.json',inputs))
                for row in arms:
                    method=row['kwargs'].get('method')
                    if method not in ('off','periodic','pure_c10','task_periodic'):continue
                    if not (root/'runs'/row['arm']/'summary.json').exists():unavailable.append(row['arm']);continue
                    d=readarm(root,row,inputs)
                    cv=.152 if model=='pi05' else .148
                    d['owner_cv']=cv;d['at_gate_epsilon']=project(d['totals'],cv,cv/10);d['zero_key_lower_bound']=project(d['totals'],cv,0.)
                    d['task_projections']={str(t):project(v,cv,cv/10) for t,v in d['tasks'].items()}
                    data.append(d);print(root.name,row['arm'],d['totals'],round(d['at_gate_epsilon']['relative_saved']*100,2),flush=True)
    levels=json.loads(read(REPO/'exp/offline_search/rounds/r14/descent_astra/levels.json',inputs))
    stitched={}
    for cell,table in levels.items():
        model,suite,size=cell.split('_');cv=.152 if model=='pi05' else .148;parts={}
        for task,level in table.items():
            method='pure_c10' if level=='c10' else ('off' if level=='off' else 'periodic')
            target=None if method=='off' else (.5 if method=='pure_c10' else int(level[2:])/100)
            matches=[d for d in data if Path(d['root']).name==f'r14_descent_{cell}' and d['kwargs']['method']==method and d['kwargs'].get('target')==target]
            assert len(matches)==1,(cell,task,level,len(matches))
            v=matches[0]['tasks'][int(task)];parts[task]={'level':level,**project(v,cv,cv/10)}
        totals={k:sum(v[k] for v in parts.values()) for k in ('N','V','M','episodes')}
        stitched[cell]={'tasks':parts,'at_gate_epsilon':project(totals,cv,cv/10),'zero_key_lower_bound':project(totals,cv,0.)}
    dump(HERE/'cost_counts.json',{'arms':data,'r14_selected_table_stitched':stitched,'unavailable':unavailable,'input_sha256':inputs,'scope':'counterfactual cost only; fixed accepted trajectories/counts; no new SR or IR outcomes'})
    with (HERE/'cost_counts.csv').open('w',newline='') as f:
        cols=['root','arm','task','N','V','M','episodes','current_ir','new_ir','absolute_saved','relative_saved','max_extra_calls_per_100_looks'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for d in data:
            for task,p in [('all',d['at_gate_epsilon']),*d['task_projections'].items()]:w.writerow({'root':Path(d['root']).name,'arm':d['arm'],'task':task,**p})
    print('stitched',json.dumps({k:v['at_gate_epsilon'] for k,v in stitched.items()}))

if __name__=='__main__':main()
