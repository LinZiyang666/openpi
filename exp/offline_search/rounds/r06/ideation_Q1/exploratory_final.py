"""POST-HOC: closed-loop sources published after Q1 preregistration.

Read summaries and accepted journals; recompute owner cost and paired evidence.
Never execute adapter code, start a rollout, or change any source artifact.
"""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest,rankdata
import pilot_q1 as q

OUT=q.HERE/'exploratory_final'
OUT.mkdir(exist_ok=False)
RUNS=Path('/home/weiland/trace_runs/os_closed_loop')
specs=json.loads((q.HERE.parent/'ideation_Q2/frontier/adapters/replay_baselines.json').read_text())
registry={};source=[];skipped=[];Y={}

def register(path,cell,family,rep=1):
    sp=path/'summary.json';jp=path/'client/journal.jsonl'
    if not sp.exists():skipped.append(dict(path=str(sp),reason='no summary at inventory'));return
    sm=json.loads(sp.read_text())
    if sm.get('complete')!=500:skipped.append(dict(path=str(sp),reason='not 500 completed',complete=sm.get('complete')));return
    name=path.name
    markers=list((path.parents[1]/'state').glob(name+'.DONE'))+list((path.parents[1]/'state').glob(name+'.manifest_*.DONE'))
    if not markers:skipped.append(dict(path=str(sp),reason='no completion marker'));return
    accepted={}
    for line in jp.read_text().splitlines():
        a=json.loads(line)
        if a.get('accepted') and a.get('status') in ['done','failed'] and not a.get('error'):
            key=tuple(map(int,a['task_uid'].split(':')[-2:]))
            assert key not in accepted,(name,key,'duplicate accepted')
            accepted[key]=int(a['success'])
    keys=[(t,i) for t in range(10) for i in range(50)]
    assert set(accepted)==set(keys) and sum(accepted.values())==sm['success'],name
    yy=np.array([accepted[k] for k in keys],float)
    ledger=sm['cost_ledger'];c1=.152 if sm['model']=='pi05' else .148
    n=ledger['decisions'];v=ledger['vision_decisions'];m=ledger['misses'];controls=ledger['controls']
    assert np.isclose(v/n,ledger['v']) and np.isclose(m/n,ledger['m'])
    cost=c1*v+(1-c1)*m
    row=dict(arm=name,cell=cell,model=sm['model'],family=family,rep=rep,success=int(yy.sum()),n=500,SR=yy.mean(),
             owner_IR=cost/(controls/5),owner_IR_per_request=cost/n,vision_fraction=v/n,miss_fraction=m/n,
             stored_IR=ledger['ir_per_five_controls'],summary=str(sp),journal=str(jp),method=sm.get('method'),kwargs=json.dumps(sm.get('kwargs',{}),sort_keys=True))
    registry[name]=row;Y[name]=yy
    for p in [sp,jp,*markers]:source.append(dict(path=str(p),sha256=q.sha(p)))

for s in specs:
    cell=s['cell'];family=s['name'].rsplit('_',1)[-1];run,arm=s['source_arm'].split('/')
    register(RUNS/run/'runs'/arm,cell,family)
    for rep in [2,3]:register(RUNS/'r06_paper/runs'/f'{arm}_rep{rep}',cell,family,rep)

pure={
 'pi05_l10':('r04_cost','r4f_p_l10_inf_k10_L10'),
 'pi05_spatial':('r04_cost','r4f_p_sp_inf_k10_L10'),
 'groot_l10':('r05_q2','r5q2_g_l10_policy_L10'),
 'groot_spatial':('r05_q2','r5q2_g_spatial_policy_L10')}
for cell,(run,arm) in pure.items():register(RUNS/run/'runs'/arm,cell,'P10')
for p in sorted((RUNS/'r06_abl/runs').glob('*/summary.json')):
    match=re.fullmatch(r'r6p2_(direct|identity|no_progress|stuck|terminal|overtime)_([pg])_(l10|sp)_(50|500)',p.parent.name)
    if match:
        typ,model,suite,n=match.groups();cell=f"{'pi05' if model=='p' else 'groot'}_{'spatial' if suite=='sp' else suite}_{n}"
        register(p.parent,cell,'ablate_'+typ)
frontier_inventory=sorted((RUNS/'r06_frontier/runs').glob('*/summary.json'))
for p in frontier_inventory:
    match=re.match(r'r6q2_(pi05|groot)_(l10|spatial)_(50|500)_(.*)',p.parent.name)
    if match:
        model,suite,n,typ=match.groups();register(p.parent,f'{model}_{suite}_{n}','frontier_'+typ)

arms=pd.DataFrame(registry.values());arms.to_csv(OUT/'arms.csv',index=False)
q.dump(OUT/'input_manifest.json',dict(label='POST-HOC',sources=source,skipped=skipped,bootstrap_seed=6062901,
      frontier_inventory=[str(p) for p in frontier_inventory],owner_cost='(c1*vision + (1-c1)*MISS)/(nominal_controls/5)'))
np.savez_compressed(OUT/'episode_outcomes.npz',**Y)
rng=np.random.default_rng(6062901)
boot=np.concatenate([rng.multinomial(50,np.ones(50)/50,size=10000) for _ in range(10)],axis=1)/500
means={};meanIR={}
for (cell,family),g in arms[arms.family.isin(['A','B'])].groupby(['cell','family']):
    assert set(g.rep)=={1,2,3},(cell,family)
    means[cell,family]=np.mean([Y[a] for a in g.sort_values('rep').arm],axis=0)
    meanIR[cell,family]=g.owner_IR.mean()

def pair(name,y,ref,z,cell):
    delta=y-z;draw=boot@delta;ci=np.quantile(draw,[.025,.975])
    result=dict(cell=cell,arm=name,reference=ref,delta=delta.mean(),lo=ci[0],hi=ci[1],paired_init_clusters=500,
                plus=int(np.sum(delta>0)),minus=int(np.sum(delta<0)))
    binary=np.isin(y,[0,1]).all() and np.isin(z,[0,1]).all()
    result['mcnemar_p']=binomtest(result['plus'],result['plus']+result['minus'],.5).pvalue if binary and result['plus']+result['minus'] else (1. if binary else np.nan)
    result['two_pp_NI_95lower']=ci[0]>=-.02
    return result

ab=[];contrasts=[]
for cell in sorted(arms[arms.family=='A'].cell.unique()):
    aa=arms[(arms.cell==cell)&(arms.family=='A')].sort_values('rep')
    bb=arms[(arms.cell==cell)&(arms.family=='B')].sort_values('rep')
    y,z=means[cell,'B'],means[cell,'A'];d=pair('B_mean',y,'A_mean',z,cell)
    plus=minus=0
    for an,bn in zip(aa.arm,bb.arm):
        c=pair(bn,Y[bn],an,Y[an],cell);contrasts.append(c);plus+=c['plus'];minus+=c['minus']
    ab.append(dict(cell=cell,A_SR=z.mean(),A_IR=meanIR[cell,'A'],B_SR=y.mean(),B_IR=meanIR[cell,'B'],
                   delta=d['delta'],lo=d['lo'],hi=d['hi'],A_replicates=aa.SR.tolist(),B_replicates=bb.SR.tolist(),
                   pooled_plus=plus,pooled_minus=minus,nominal_pooled_mcnemar=binomtest(plus,plus+minus,.5).pvalue))
for _,a in arms[arms.family.str.startswith(('ablate_','frontier_'))].iterrows():
    cell=a.cell;y=Y[a.arm]
    for fam in ['A','B']:
        contrasts.append(pair(a.arm,y,fam+'_mean',means[cell,fam],cell))
        for _,r in arms[(arms.cell==cell)&(arms.family==fam)].iterrows():contrasts.append(pair(a.arm,y,r.arm,Y[r.arm],cell))
    key=cell.rsplit('_',1)[0];run,pname=pure[key]
    contrasts.append(pair(a.arm,y,pname,Y[pname],cell))
    if a.family.startswith('frontier_'):
        for _,r in arms[(arms.cell==cell)&arms.family.str.startswith('frontier_')&(arms.arm<a.arm)].iterrows():
            contrasts.append(pair(a.arm,y,r.arm,Y[r.arm],cell))
pd.DataFrame(ab).to_csv(OUT/'AB_three_replicates.csv',index=False)
pd.DataFrame(contrasts).to_csv(OUT/'paired_contrasts.csv',index=False)

# Post-hoc validation of exactly the prior library scores against more precise
# A/B outcomes; this neither changes the frozen survival test nor adds banks.
old=pd.read_csv(q.HERE/'task_scores.csv');rankrows=[];scoreout=[]
cells=sorted(means.keys());cells=sorted({c for c,f in cells})
for src in ['loeo','inf','cache']:
    for candidate,field,sign in [('D','d_pair',1),('R','err10',1),('Qrisk','q',-1),('C95_failure','covered',-1)]:
        xx=[];ys={k:[] for k in ['A_failure','B_gain','P10_gap']};bs={k:[] for k in ys}
        for cell in cells:
            m,s,n=cell.split('_');lib='current' if n=='50' else ('bpool_cs' if m=='pi05' else 'bpool_all')
            tag=f'{m}_{s}_{lib}_refit';sub=old[(old.tag==tag)&(old.source==src)]
            score=sign*sub[field].mean();xx.append(score)
            val={'A_failure':1-means[cell,'A'],'B_gain':means[cell,'B']-means[cell,'A'],
                 'P10_gap':Y[pure[f'{m}_{s}'][1]]-means[cell,'A']}
            for target,y in val.items():
                ys[target].append(y.mean());bs[target].append(boot@y)
                scoreout.append(dict(cell=cell,source=src,candidate=candidate,target=target,score=score,outcome=y.mean()))
        for target,y in ys.items():
            dd=np.stack(bs[target],axis=1);br=q.matrix_rho(np.tile(xx,(len(dd),1)),dd);ci=q.interval(br)
            rankrows.append(dict(source=src,candidate=candidate,target=target,rho=q.rho(xx,y),lo=ci[0],hi=ci[1],banks=8,
                                 uncertainty='conditional on fixed tasks/banks; init clusters shared across 3 replicates'))
pd.DataFrame(scoreout).to_csv(OUT/'quality_outcomes.csv',index=False)
pd.DataFrame(rankrows).to_csv(OUT/'quality_rank_correlations.csv',index=False)
print(json.dumps(dict(arms=len(arms),source_files=len(source),contrasts=len(contrasts),skipped=skipped)),flush=True)
