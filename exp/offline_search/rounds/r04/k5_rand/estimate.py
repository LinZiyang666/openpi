"""CALL-minus-CACHE identification, never a learned serving controller.

HT means: mean(I[CALL]*outcome/p), mean(I[CACHE]*outcome/(1-p)).
Bootstrap resamples (task, original init) clusters, retaining both replicates and
recomputing denominators and parent/child shrinkage in each draw. Context tables
condition only on pre-intervention features of exposed episodes. ITT includes
unexposed episodes. Intervals are percentile 95%, pointwise, not simultaneous.
Aggregate collect/KPI reports locate raw logs; aggregates alone cannot recover
assignments, init clustering, or landmark contexts and are explicitly refused.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import numpy as np

from exp.offline_search.rounds.r04.k5_rand.overlay import assignment, RandomizedLandmark

RHOS = {'g50': .3234160220826887, 'g500': .23821622358554875}
CONTEXTS = dict(progress=('<0.5','>=0.5'), gripper=('open','closed'),
                confidence=('>-0.3','<=-0.3'), stall_age=('before','0-9','>=10'))
METRICS = ['Y','N','M','C_g50','C_g500']


def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def load_arm(root, arm, baseline=False):
    """Join accepted terminal attempt only; fail closed on gaps/conflicting duplicates."""
    directory=Path(root)/'runs'/arm
    accepted={}
    for r in jsonl(directory/'client/journal.jsonl'):
        if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'):
            accepted[r['task_uid']]=r
    decisions=defaultdict(dict)
    ends={}
    duplicates=0
    contracts=set()
    for f in sorted(directory.glob('server_*/decisions_*.jsonl')):
        for r in jsonl(f):
            if r.get('ev')=='startup':
                contract={k:r.get(k) for k in ('method_spec','kwargs','cell','judge','stage1_mode','miss_steps')}
                contracts.add(json.dumps(contract,sort_keys=True))
            uid=r.get('uid')
            if uid not in accepted or int(r.get('attempt',1) or 1)!=int(accepted[uid].get('attempt',1) or 1): continue
            if r['ev']=='episode' and r.get('reason')=='episode_end': ends[uid]=r
            if r['ev']!='dec': continue
            step=int(r['step'])
            if step in decisions[uid]:
                old=decisions[uid][step]
                # Ignore transport/timing provenance, never action/verdict/assignment differences.
                fields=['hit','served_head','baseline_verdict','actual_verdict','eligible','opportunity_index',
                        'landmark_class','assigned_treatment','context','trial_id','conf','topk']
                if any(old.get(k)!=r.get(k) for k in fields): raise ValueError(f'{arm} {uid}:{step}: conflicting duplicate')
                duplicates+=1
            decisions[uid][step]=r
    if len(contracts)>1: raise ValueError(f'{arm}: mixed controller configurations')
    controller=json.loads(next(iter(contracts))) if contracts else None
    episodes=[]
    seen=set()
    for uid,j in sorted(accepted.items()):
        ds=[r for _,r in sorted(decisions[uid].items())]
        if not ds or [r['step'] for r in ds]!=list(range(len(ds))):
            raise ValueError(f'{arm} {uid}: missing/non-contiguous accepted-attempt decisions')
        end=ends.get(uid)
        if not end or end['n_decisions']!=len(ds) or end['n_exec']!=len(ds) or bool(end['success'])!=bool(j['success']):
            raise ValueError(f'{arm} {uid}: missing/mismatching episode_end')
        if any(not r.get('ok',True) or r.get('error') for r in ds): raise ValueError(f'{uid}: failed decision')
        task,init=int(ds[0]['task_id']),int(ds[0]['init'])
        if (task,init) in seen: raise ValueError(f'{arm}: duplicate original init {(task,init)}')
        seen.add((task,init))
        row=dict(arm=arm,task_id=task,init=init,Y=int(bool(j['success'])),N=len(ds),
                 M=sum(not r['hit'] for r in ds),uid=uid,attempt=j.get('attempt',1),controller=controller)
        if end.get('n_miss')!=row['M']: raise ValueError(f'{uid}: MISS totals disagree')
        if not baseline:
            first=ds[0]
            a=assignment(first['experiment_seed'],task,init,first['replicate'])
            overlay=RandomizedLandmark(first['experiment_seed'],task,init,first['replicate'])
            exposed=[]
            previous=None
            for d in ds:
                if any(d.get(k)!=v for k,v in a.items()): raise ValueError(f'{uid}: unstable/incorrect assignment')
                # This experiment permits only guard reasons 1-4 (and unforced HITs).
                ex=d.get('extras',{})
                baseline_hit=ex.get('os_force_miss',0)!=1
                if not baseline_hit and int(ex.get('os_reason',0)) not in (1,2,3,4):
                    raise ValueError(f'{uid}: non-guard baseline MISS')
                actual,expected=overlay.apply(d['step'],baseline_hit,d['conf'],ex,previous)
                if any(d.get(k)!=v for k,v in expected.items()): raise ValueError(f'{uid}:{d["step"]}: overlay audit failed')
                if actual!=d['hit'] or d['src']!=('cache' if actual else 'policy'):
                    raise ValueError(f'{uid}: treatment noncompliance')
                head=np.asarray(d['served_head'])
                if head.shape!=(5,7) or not np.isfinite(head).all(): raise ValueError(f'{uid}: invalid served_head')
                if actual and not d.get('exec_ok'): raise ValueError(f'{uid}: cache action mismatch')
                previous=head
                if d['eligible']: exposed.append(d)
            if len(exposed)>1 or end.get('exposed')!=bool(exposed): raise ValueError(f'{uid}: exposure mismatch')
            event=exposed[0] if exposed else None
            row.update(a,exposed=bool(event),context=event['context'] if event else None,
                       actual_treatment=('CACHE' if event['hit'] else 'CALL') if event else None,
                       exposure_step=event['step'] if event else None,
                       baseline_miss_opportunities=overlay.opportunities)
        episodes.append(row)
    return episodes,dict(arm=arm,accepted=len(accepted),loaded=len(episodes),duplicate_decisions=duplicates)


def cluster_draws(rows, boot, seed):
    keys=sorted({(r['task_id'],r['init']) for r in rows})
    index={k:i for i,k in enumerate(keys)}
    cid=np.array([index[(r['task_id'],r['init'])] for r in rows])
    rng=np.random.default_rng(seed)
    # Multinomial weights are equivalent to sampling clusters with replacement.
    weights=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=boot)
    return cid,weights


def values(rows):
    v=np.asarray([[r['Y'],r['N'],r['M']] for r in rows],float)
    return np.column_stack([v,*[.848*v[:,2]+(.152-rho)*v[:,1] for rho in RHOS.values()]])


def effect(rows,mask,cid,weights):
    n=int(mask.sum()); k=weights.shape[1]
    if not n: return dict(n=0,n_clusters=0,supported=False),None
    treatment=np.array([r['assigned_treatment']=='CALL' for r in rows])
    p=np.array([r['propensity'] for r in rows])
    outcome=values(rows)
    call=(treatment/p)*mask; cache=((~treatment)/(1-p))*mask
    num=(call-cache)[:,None]*outcome
    numerator=np.zeros((k,len(METRICS))); denominator=np.zeros(k)
    np.add.at(numerator,cid,num); np.add.at(denominator,cid,mask.astype(float))
    den=weights@denominator
    draws=np.divide(weights@numerator,den[:,None],out=np.full((len(weights),len(METRICS)),np.nan),where=den[:,None]>0)
    delta=num.sum(axis=0)/n
    clusters=len(np.unique(cid[mask])); supported=bool((treatment[mask]).any() and (~treatment[mask]).any() and clusters>=2)
    rec=dict(n=n,n_clusters=clusters,n_call=int((mask&treatment).sum()),n_cache=int((mask&~treatment).sum()),
             supported=supported,call_mean=dict(zip(METRICS,(call[:,None]*outcome).sum(axis=0)/n)),
             cache_mean=dict(zip(METRICS,(cache[:,None]*outcome).sum(axis=0)/n)),delta=dict(zip(METRICS,delta)))
    rec['ci95']={name:np.nanquantile(draws[:,i],[.025,.975]).tolist() if supported else None for i,name in enumerate(METRICS)}
    rec['bootstrap_nonempty']=int(np.isfinite(draws[:,0]).sum())
    return rec,draws


def validate_pairs(rows):
    if not rows: raise ValueError('no randomized episodes')
    contracts={json.dumps(r.get('controller'),sort_keys=True) for r in rows}
    if len(contracts)!=1: raise ValueError('replicate arms have different controller/library/cost configurations')
    groups=defaultdict(list)
    arms=defaultdict(set)
    seeds=set()
    for r in rows:
        groups[(r['task_id'],r['init'])].append(r)
        arms[r['replicate']].add(r['arm']); seeds.add(r['experiment_seed'])
        if r['propensity']!=.5: raise ValueError('expected fixed propensity .5')
        if not (r['Y'] in (0,1) and 0<=r['M']<=r['N'] and r['N']>=1): raise ValueError('invalid Y/N/M')
        a=assignment(r['experiment_seed'],r['task_id'],r['init'],r['replicate'])
        if any(r.get(k)!=v for k,v in a.items()): raise ValueError('incorrect episode assignment')
    if set(arms)!={1,2} or any(len(v)!=1 for v in arms.values()) or len(seeds)!=1:
        raise ValueError('supply exactly one complementary replicate pair with one experiment seed, separately per scale')
    for key,pair in groups.items():
        if len(pair)!=2 or {r['replicate'] for r in pair}!={1,2} or len({r['landmark_class'] for r in pair})!=1 or {r['assigned_treatment'] for r in pair}!={'CALL','CACHE'}:
            raise ValueError(f'incomplete/noncomplementary pair {key}; no complete-case dropping is allowed')
    return groups


def estimate(rows,boot=2000,seed=20260927,baselines=None):
    groups=validate_pairs(rows)
    cid,weights=cluster_draws(rows,boot,seed)
    landmarks=np.array([r['landmark_class'] for r in rows]); exposed=np.array([r['exposed'] for r in rows])
    allmask=np.ones(len(rows),bool)
    itt,_=effect(rows,allmask,cid,weights)
    parents={}; parent_draws={}; itt_landmarks={}
    for landmark in (1,3):
        key=str(landmark)
        itt_landmarks[key]=effect(rows,landmarks==landmark,cid,weights)[0]
        parents[key],parent_draws[key]=effect(rows,exposed&(landmarks==landmark),cid,weights)
    children=[]
    for landmark,bins in itertools.product((1,3),itertools.product(*CONTEXTS.values())):
        context=dict(zip(CONTEXTS,bins))
        mask=exposed&(landmarks==landmark)&np.array([r.get('context')==context for r in rows])
        child,draws=effect(rows,mask,cid,weights)
        child.update(landmark_class=landmark,context=context,shrink_weight=child['n']/(child['n']+100),
                     shrunk_delta=None,shrunk_ci95=None,unsupported_action='baseline CALL')
        parent=parents[str(landmark)]
        if child['supported'] and parent['supported']:
            w=child['shrink_weight']
            child['shrunk_delta']={m:w*child['delta'][m]+(1-w)*parent['delta'][m] for m in METRICS}
            counts=np.bincount(cid[mask],minlength=weights.shape[1])
            boot_n=weights@counts
            boot_w=(boot_n/(boot_n+100))[:,None]
            shrunk=boot_w*draws+(1-boot_w)*parent_draws[str(landmark)]
            child['shrunk_ci95']={m:np.nanquantile(shrunk[:,i],[.025,.975]).tolist() for i,m in enumerate(METRICS)}
        children.append(child)
    exposure=[]
    for rep,lm,tr in itertools.product((1,2),(1,3),('CALL','CACHE')):
        rr=[r for r in rows if r['replicate']==rep and r['landmark_class']==lm and r['assigned_treatment']==tr]
        ee=[r for r in rr if r['exposed']]
        compliant=sum(r['actual_treatment']==tr for r in ee)
        exposure.append(dict(replicate=rep,landmark_class=lm,assigned_treatment=tr,n=len(rr),exposed=len(ee),
            exposure_rate=len(ee)/len(rr) if rr else None, unexposed=len(rr)-len(ee),compliant=compliant,
            noncompliant=len(ee)-compliant,compliance_rate=compliant/len(ee) if ee else None))
    balance=[]
    for feature in ('task_id','landmark_class',*CONTEXTS):
        rr=[r for r in rows if r['exposed']] if feature in CONTEXTS else rows
        levels=sorted({str((r['context'] or {}).get(feature)) if feature in CONTEXTS else str(r[feature]) for r in rr})
        for level in levels:
            counts={tr:sum(r['assigned_treatment']==tr and str((r['context'] or {}).get(feature) if feature in CONTEXTS else r[feature])==level for r in rr) for tr in ('CALL','CACHE')}
            balance.append(dict(feature=feature,level=level,population='exposed' if feature in CONTEXTS else 'ITT',**counts))
    result=dict(schema='causal_rescue_credit.estimate.v1',episodes=len(rows),init_clusters=len(groups),
        estimand='CALL minus CACHE; Horvitz-Thompson means, including unexposed in ITT',
        uncertainty=dict(method='init-cluster percentile bootstrap',draws=boot,seed=seed,coverage=.95,pointwise=True),
        rho=RHOS,cost_formula='.848*dM + (.152-rho)*dN',ITT=itt,ITT_by_landmark=itt_landmarks,
        exposed_by_landmark=parents,contexts=children,exposure_compliance=exposure,balance=balance,
        unknown_context_exposures=sum(r['exposed'] and (r.get('context') is None or any(v is None for v in r['context'].values())) for r in rows),
        discordant_pair_exposure=sum(pair[0]['exposed']!=pair[1]['exposed'] for pair in groups.values()),
        disclosure='Outcome-derived tables borrow information beyond the deployed library; no controller or lambda fitted.')
    if baselines:
        comparisons=[]
        for arm,base in baselines.items():
            bykey={(r['task_id'],r['init']):r for r in base}
            if not set(groups)<=set(bykey): raise ValueError(f'{arm}: baseline missing requested inits')
            target=rows[0].get('controller'); reference=base[0].get('controller')
            if target and reference and any(target.get(k)!=reference.get(k) for k in ('method_spec','kwargs','cell','judge')):
                raise ValueError(f'{arm}: baseline controller/library does not match randomized arms')
            for tr in ('CALL','CACHE'):
                rr=[r for r in rows if r['assigned_treatment']==tr]
                delta=values(rr)-values([bykey[(r['task_id'],r['init'])] for r in rr])
                # One randomized observation of each treatment per init.
                ordered=np.zeros((len(groups),len(METRICS)))
                key_index={key:i for i,key in enumerate(sorted(groups))}
                for r,d in zip(rr,delta): ordered[key_index[(r['task_id'],r['init'])]]=d
                draws=weights@ordered/len(groups)
                comparisons.append(dict(arm=arm,treatment=tr,n=len(rr),descriptive_not_randomized_baseline=True,
                    delta=dict(zip(METRICS,delta.mean(axis=0))),ci95={m:np.quantile(draws[:,i],[.025,.975]).tolist() for i,m in enumerate(METRICS)}))
        result['baseline_comparisons']=comparisons
    return result


def report_sources(path):
    """Resolve aggregate reports to their raw-data roots, never invent per-init data."""
    path=Path(path).resolve(); obj=json.loads(path.read_text())
    if obj.get('schema')=='causal_rescue_credit.episodes.v1': return obj,[]
    roots=[Path(p) for p in obj.get('args',{}).get('run_roots',[])]
    if not roots:
        roots=[p for p in path.parents if (p/'arms.json').exists() and (p/'runs').is_dir()][:1]
    if not roots: raise ValueError(f'{path}: aggregate report lacks init data and no raw run root can be resolved; use --run-root')
    return None,roots


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',action='append',type=Path,default=[])
    ap.add_argument('--input',action='append',default=[],help='collect/KPI report (raw logs must remain) or exported episode JSON')
    ap.add_argument('--arms',nargs=2,required=True,metavar=('REP1','REP2'))
    ap.add_argument('--baseline',action='append',default=[])
    ap.add_argument('--boot',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=20260927)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--episodes-out',type=Path)
    a=ap.parse_args(argv)
    if a.boot<100: ap.error('--boot must be at least 100')
    roots=list(a.run_root); exported=[]
    for file in a.input:
        data,rr=report_sources(file); roots+=rr
        if data: exported+=data['episodes']
    roots=list(dict.fromkeys(r.resolve() for r in roots))
    rows=[]; audits=[]; bases={}
    for arm in [*a.arms,*a.baseline]:
        rr=[r for r in exported if r['arm']==arm]
        if not rr:
            matches=[root for root in roots if (root/'runs'/arm).is_dir()]
            if len(matches)!=1: raise ValueError(f'{arm}: expected exactly one source root, got {matches}')
            rr,audit=load_arm(matches[0],arm,baseline=arm in a.baseline); audits.append(audit)
        if arm in a.baseline: bases[arm]=rr
        else: rows+=rr
    result=estimate(rows,a.boot,a.seed,bases)
    result['input_audit']=audits
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if a.episodes_out:
        a.episodes_out.write_text(json.dumps(dict(schema='causal_rescue_credit.episodes.v1',episodes=rows),indent=2)+'\n')
    print(json.dumps(dict(episodes=result['episodes'],init_clusters=result['init_clusters'],ITT=result['ITT']['delta'],ci95=result['ITT']['ci95'])))

if __name__=='__main__': main()
