"""Summarize completed discovery-only E2 jobs; paired bootstrap and call checks."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from callback_calls import OUT, RUN, save


REPS=10000
rng=np.random.default_rng(20261001)
# Same within-task init resampling across models, libraries and variants.
W={suite:np.stack([rng.multinomial(30,np.full(30,1/30),size=REPS) for _ in range(10)],axis=1)
   for suite in ['l10','spatial']}
specs={x['arm']:x for x in json.loads((RUN/'arms.json').read_text())}
arm_rows=[]
support_rows=[]
effects=[]
interactions=[]
draws={}
raw=[]
extra={}
episode_tables={}
stage_costs=[]


def choice_coin(spec,r):
    kw=spec['kwargs']
    if spec['r8']['variant']=='IP':
        key='{}|{}|{}|{}|{}'.format(kw['coin_domain'],int(kw['random_seed']),int(r.task_id),int(r.init),int(r.decision_seq))
        bits=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')
        return (bits >> 11)*2.**-53
    else:
        payload=['Q2-deploy-v1',kw['randomization_key'],int(kw['random_seed']),int(r.task_id),int(r.init),int(r.decision_seq),'dose-anchor']
    bits=int.from_bytes(hashlib.sha256(json.dumps(payload,separators=(',',':'),ensure_ascii=True).encode()).digest()[:8],'big')
    return (bits >> 11)*2.**-53


def estimator(f,ep,endpoint,weights,center=True):
    f=f.sort_values('decision_seq',kind='stable').drop_duplicates('episode_key')
    y=np.full((10,30),np.nan)
    for r in ep.itertuples(): y[int(r.task_id),int(r.init)]=float(r.success)
    assert np.isfinite(y).all()
    rmat=np.zeros((10,30));z=np.zeros((10,30));s=np.zeros((10,30));out=np.zeros((10,30))
    for row in f.itertuples():
        ix=int(row.task_id),int(row.init)
        rmat[ix]=1;z[ix]=row.z;s[ix]=row.z/row.p-(1-row.z)/(1-row.p)
        out[ix]=getattr(row,endpoint)
    assert np.isfinite(out).all()
    baseline=np.zeros((10,30));base_draw=np.zeros((REPS,10,30))
    if center:
        for fold in (0,1):
            test=np.arange(30)%2==fold;train=~test
            baseline[:,test]=y[:,train].mean(1)[:,None]
            denom=weights[:,:,train].sum(2)
            b=(weights[:,:,train]*y[:,train][None,:,:]).sum(2)/denom
            base_draw[:,:,test]=b[:,:,None]
    value=rmat*s*(out-baseline)
    boot=(weights*rmat[None]*s[None]*(out[None]-base_draw)).sum((1,2))
    den=(weights*rmat[None]).sum((1,2))
    boot_cond=np.divide(boot,den,out=np.full(REPS,np.nan),where=den>0)
    point=value.sum()/rmat.sum() if rmat.sum() else np.nan
    result=dict(reached=int(rmat.sum()),calls=int(z.sum()),cache=int(rmat.sum()-z.sum()),
                estimate=float(point),population_estimate=float(value.mean()),
                lo=float(np.nanquantile(boot_cond,.025)),hi=float(np.nanquantile(boot_cond,.975)),
                baseline='other-init-parity task success mean, refit per bootstrap' if center else 'zero',
                endpoint=endpoint)
    return result,boot_cond


for directory in sorted((OUT/'calls').glob('*')):
    if not (directory/'summary.json').exists(): continue
    name=directory.name;spec=specs[name];suite=spec['suite_short'];variant=spec['r8']['variant'];size=spec['r8']['library_size']
    meta=dict(arm=name,model=spec['model'],suite=suite,variant=variant,size=size)
    summary=json.loads((directory/'summary.json').read_text())
    df=pd.read_parquet(directory/'decisions.parquet')
    ep=pd.read_parquet(directory/'episodes.parquet')
    episode_tables[name]=ep
    assert ep.init.max()==29 and df.init.max()==29
    a=df[df.fresh].copy()
    s=a[a.eligible.eq(True)&a.p.between(0,1,inclusive='neither')].copy()
    expected=np.where(a.cooldown.eq(1),0,np.where(a.stall_state.eq(2),1,a.p_nominal)) if variant!='IP' else np.full(len(a),.25)
    formula=int((np.abs(a.p.to_numpy()-expected)>1e-12).sum())
    coinmax=max(abs(choice_coin(spec,r)-r.coin) for r in a.itertuples())
    arm_rows.append(dict(meta,episodes=len(ep),successes=int(ep.success.sum()),decisions=len(df),fresh=len(a),supported=len(s),
        p_zero=int(a.p.eq(0).sum()),p_one=int(a.p.eq(1).sum()),p_min=float(s.p.min()),p_max=float(s.p.max()),
        supported_p_over_95=int(s.p.gt(.95).sum()),supported_p_under_05=int(s.p.lt(.05).sum()),
        nominal_one_nonstall=int((a.p.eq(1)&a.stall_state.ne(2)).sum()),
        forced_stall=int((a.p.eq(1)&a.stall_state.eq(2)).sum()),
        dev_entries=summary['dev_entries'],dev_entries_supported=summary['dev_entries_supported'],
        physical_available=summary['physical_available'],stage_pre_present=summary['stage_pre_present'],
        formula_mismatches=formula,hash_coin_max_error=coinmax,coin_treatment_mismatches=summary['coin_mismatches'],
        dispatch_mismatches=summary['call_count_mismatches'],source_mismatches=summary['source_mismatches'],
        prefix_checks=len(summary['prefix_checks']),prefix_failures=sum(not x['same'] for x in summary['prefix_checks']),
        controls=summary['ledger']['active_controls'],V=summary['ledger']['V'],M=summary['ledger']['M'],
        work=summary['ledger']['owner_work'],IR=summary['ledger']['owner_IR_requests'],IR_control=summary['ledger']['owner_IR_controls']))
    for row in json.loads((directory/'stage_ledger.json').read_text())['tables']['stages']:
        stage_costs.append(dict(meta,**row))
    for family in ['r7_stage','truth_pre']:
        report=json.loads((directory/f'call_value_{family}.json').read_text())
        for row in report['tables']['contrasts']:
            if row.get('endpoint')=='final_success':
                raw.append(dict(meta,family=family,stage=row['stage'],estimand=row['estimand'],status=row['status'],
                                reached=row.get('reached_episodes'),estimate=row.get('conditional_reached',row.get('population',{})).get('estimate'),
                                calls=row.get('treated_n'),cache=row.get('control_n'),call_ESS=row.get('treated_ESS'),cache_ESS=row.get('control_ESS'),
                                interval=row.get('conditional_reached',row.get('population',{})).get('interval')))
        for stage,g in a.groupby(family):
            gs=s[s[family]==stage]
            support_rows.append(dict(meta,family=family,stage=stage,anchors=len(g),supported=len(gs),
                                     p0=int(g.p.eq(0).sum()),p1=int(g.p.eq(1).sum()),calls=int(gs.z.sum()),cache=int(len(gs)-gs.z.sum())))
            if not len(gs): continue
            for endpoint,center in [('final_success',True),('predicate_delta_20',False),('carry_gain_20',False)]:
                est,bs=estimator(gs,ep,endpoint,W[suite],center)
                if not est['calls'] or not est['cache']: continue
                key=(name,family,stage,endpoint)
                draws[key]=bs
                effects.append(dict(meta,family=family,stage=stage,**est))
        for high,low in ([('event','interior'),('mixed','interior')] if family=='r7_stage' else [('grasp_window','approach'),('carry','approach'),('place','approach'),('release','approach')]):
            key1=(name,family,high,'final_success');key0=(name,family,low,'final_success')
            one=next((x for x in effects if (x['arm'],x['family'],x['stage'],x['endpoint'])==key1),None)
            zero=next((x for x in effects if (x['arm'],x['family'],x['stage'],x['endpoint'])==key0),None)
            if one is None or zero is None:continue
            diff=draws[key1]-draws[key0];point=one['estimate']-zero['estimate']
            idx=len(interactions)
            draws['interaction',idx]=diff
            interactions.append(dict(meta,family=family,contrast=high+' minus '+low,estimate=point,
                lo=float(np.nanquantile(diff,.025)),hi=float(np.nanquantile(diff,.975)),se=float(np.nanstd(diff,ddof=1)),
                high_n=one['reached'],low_n=zero['reached'],min_branch=min(one['calls'],one['cache'],zero['calls'],zero['cache'])))
    extra[name]=dict(triggers=summary['triggers'],forensic_labels=summary['forensic_labels'])

# Simultaneous family covers every estimable predeclared cell/stage interaction.
if interactions:
    ts=np.stack([np.abs(draws['interaction',i]-r['estimate'])/r['se'] for i,r in enumerate(interactions)])
    critical=float(np.nanquantile(np.nanmax(ts,axis=0),.95))
    for r in interactions:
        r.update(simultaneous_lo=r['estimate']-critical*r['se'],simultaneous_hi=r['estimate']+critical*r['se'])
else:critical=None

paired=[]
for ct in [r for r in arm_rows if r['variant']=='CT']:
    name=ct['arm'];cu_name=name[:-2]+'CU'
    if cu_name not in episode_tables:continue
    cu=next(r for r in arm_rows if r['arm']==cu_name)
    delta=np.zeros((10,30))
    left=episode_tables[name].set_index(['task_id','init'])['success']
    right=episode_tables[cu_name].set_index(['task_id','init'])['success']
    for (t,i),v in (left.astype(float)-right.astype(float)).items():delta[int(t),int(i)]=v
    bs=(W[ct['suite']]*delta[None]).sum((1,2))/300
    draws['paired',name]=bs
    paired.append(dict(model=ct['model'],suite=ct['suite'],size=ct['size'],episodes=300,
                       CT_successes=ct['successes'],CU_successes=cu['successes'],delta=delta.mean(),
                       lo=np.quantile(bs,.025),hi=np.quantile(bs,.975),CT_IR=ct['IR'],CU_IR=cu['IR'],
                       discordant=int((delta!=0).sum())))
if len(paired)==8:
    bs=np.mean([draws['paired',r['arm']] for r in arm_rows if r['variant']=='CT'],axis=0)
    paired.append(dict(model='equal_cell_pool',suite='both',size='both',episodes=2400,
                       CT_successes=sum(r['CT_successes'] for r in paired),CU_successes=sum(r['CU_successes'] for r in paired),
                       delta=np.mean([r['delta'] for r in paired]),lo=np.quantile(bs,.025),hi=np.quantile(bs,.975)))

pooled=[]
groups=defaultdict(list)
for e in effects:
    groups[e['variant'],e['size'],e['family'],e['stage'],e['endpoint']].append(e)
for (variant,size,family,stage,endpoint),es in groups.items():
    if len(es)!=4:continue
    point=np.mean([e['estimate'] for e in es])
    bs=np.mean([draws[e['arm'],family,stage,endpoint] for e in es],axis=0)
    pooled.append(dict(variant=variant,size=size,family=family,stage=stage,endpoint=endpoint,cells=len(es),
                       reached=sum(e['reached'] for e in es),calls=sum(e['calls'] for e in es),cache=sum(e['cache'] for e in es),
                       estimate=point,lo=np.nanquantile(bs,.025),hi=np.nanquantile(bs,.975)))

# Equal-cell pooled interactions are specified by the same stage pairs as the cell family.
pooled_interactions=defaultdict(list)
for r in interactions:
    pooled_interactions[r['variant'],r['size'],r['family'],r['contrast']].append(r)
for (variant,size,family,contrast),rs in pooled_interactions.items():
    if len(rs)!=4:continue
    indices=[interactions.index(r) for r in rs]
    bs=np.mean([draws['interaction',i] for i in indices],axis=0)
    point=np.mean([r['estimate'] for r in rs])
    draws['interaction',len(interactions)]=bs
    interactions.append(dict(arm='equal_cell_pool',model='both',suite='both',variant=variant,size=size,
        family=family,contrast=contrast,estimate=point,lo=np.nanquantile(bs,.025),hi=np.nanquantile(bs,.975),
        se=float(np.nanstd(bs,ddof=1)),high_n=sum(r['high_n'] for r in rs),low_n=sum(r['low_n'] for r in rs),
        min_branch=min(r['min_branch'] for r in rs)))
if interactions:
    ts=np.stack([np.abs(draws['interaction',i]-r['estimate'])/r['se'] for i,r in enumerate(interactions)])
    complete_family=np.isfinite(ts).all(axis=0)
    critical=float(np.quantile(np.max(ts[:,complete_family],axis=0),.95))
    for r in interactions:
        r.update(simultaneous_lo=r['estimate']-critical*r['se'],simultaneous_hi=r['estimate']+critical*r['se'])

target=OUT/'summary';target.mkdir(exist_ok=True)
for name,rows in [('arms',arm_rows),('support_by_stage',support_rows),('raw_tool_effects',raw),
                  ('centered_first_entry',effects),('interactions',interactions),('pooled_effects',pooled)]:
    pd.DataFrame(rows).to_csv(target/f'{name}.csv',index=False)
pd.DataFrame(paired).to_csv(target/'paired_CT_CU.csv',index=False)
pd.DataFrame(stage_costs).to_csv(target/'stage_costs.csv',index=False)
save(target/'audit.json',dict(arms=len(arm_rows),discovery_episodes=sum(r['episodes'] for r in arm_rows),
     decisions=sum(r['decisions'] for r in arm_rows),bootstraps=REPS,interaction_family=len(interactions),
     simultaneous_complete_bootstraps=int(complete_family.sum()) if interactions else None,
     simultaneous_critical=critical,simultaneous_exclusions=[r for r in interactions if r['simultaneous_lo']>0 or r['simultaneous_hi']<0],
     extras=extra))
print(json.dumps(dict(arms=len(arm_rows),episodes=sum(r['episodes'] for r in arm_rows),interactions=len(interactions),
                      simultaneous_exclusions=int(sum(r['simultaneous_lo']>0 or r['simultaneous_hi']<0 for r in interactions))),indent=2))
