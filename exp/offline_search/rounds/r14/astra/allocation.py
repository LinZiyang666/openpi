"""Task-indexed R11 periodic budgets; no adaptive or cross-episode ledger."""
from common import *
import numpy as np
from exp.offline_search.rounds.r11.knob.recipe import uniform
from exp.offline_search.rounds.r11.knob.calibration import tables

DOSES=(.32,.40)
GAIN=.35
PRIOR_EPISODES=10.

def take(p,mask):
    return {k:(v[:,mask] if isinstance(v,np.ndarray) and v.ndim==2 else
               v[mask] if isinstance(v,np.ndarray) and v.ndim==1 else v) for k,v in p.items() if k!='coins'}

def priority(p):
    # Every episode has equal mass; log after episode mean limits tail influence.
    values=[];ns=[]
    for t in range(10):
        vals=[]
        for j in np.flatnonzero(p['task']==t):
            k=p['alive'][:,j]&np.isfinite(p['risk'][:,j])&~p['guard'][:,j]
            if k.any():vals.append(float(np.log1p(p['risk'][k,j].mean())))
        values.append(float(np.mean(vals)) if vals else np.nan);ns.append(len(vals))
    r=np.array(values);n=np.array(ns);mu=float(np.nanmean(r))
    spread=max(float(np.nanstd(r)),.1)
    z=np.where(np.isfinite(r),np.clip((r-mu)/spread,-2,2)*n/(n+PRIOR_EPISODES),0.)
    return z,dict(raw=values,n=ns,mean=mu,spread=spread,shrink=n/(n+PRIOR_EPISODES),z=z)

def settings(K,z,bias=0.,gain=GAIN):
    if gain==0 and bias==0:return np.full(10,K,float)
    return np.maximum(np.expm1(np.log1p(K)+bias-gain*np.asarray(z)),0.)

def prep(p):
    H,E=p['alive'].shape
    coins=np.empty((H+1,E))
    for j in range(E):
        for i in range(-1,H):
            step=-1 if i<0 else int(p['step'][i,j])
            coins[i+1,j]=uniform('R11-gap-v1',0,int(p['task'][j]),int(p['init'][j]),step,'cap')
    p['coins']=coins
    return p

def replay(p,model,K):
    if 'coins' not in p:prep(p)
    H,E=p['alive'].shape
    K=np.broadcast_to(np.asarray(K), (10,))[p['task']]
    floor=np.floor(K);frac=K-floor
    cap=floor+(p['coins'][0]<frac)
    run=np.zeros(E,int);calls=np.zeros((H,E),bool);maxrun=np.zeros(E,int)
    for i in range(H):
        alive=p['alive'][i];call=alive&(p['guard'][i]|(run>=cap));calls[i]=call
        run=np.where(alive,np.where(call,0,run+1),run);maxrun=np.maximum(maxrun,run)
        cap=np.where(call,floor+(p['coins'][i+1]<frac),cap)
    cv,cm=COSTS[model];V=p['alive'].sum(0);M=calls.sum(0);cost=cv*V+cm*M
    eligible=p['alive']&~p['guard'];risk=np.where(np.isfinite(p['risk']),p['risk'],0)
    return dict(ir=float(cost.sum()/p['N'].sum()),cost=cost,M=M,V=V,N=p['N'],calls=calls,
        # Proxy only: avoided imitation loss on recorded states, not success gain.
        capture=float((risk*calls*eligible).sum()/max((risk*eligible).sum(),1e-12)),
        maxrun=maxrun)

def solve(paths,model,K,z,gain=GAIN):
    # Match the exact reference spend separately on each calibration state path.
    refs=[replay(p,model,K)['ir'] for p in paths]
    candidates=[];lo,hi=-8.,8.
    for _ in range(32):
        b=(lo+hi)/2;k=settings(K,z,b,gain)
        irs=[replay(p,model,k)['ir'] for p in paths];errs=np.array(irs)-refs
        candidates.append((float(np.max(np.abs(errs))),abs(b),b,k,irs))
        if (errs.max()+errs.min())/2>0:lo=b
        else:hi=b
    err,_,b,k,irs=min(candidates,key=lambda r:r[:3])
    return dict(bias=b,K=k,reference_irs=refs,irs=irs,max_match_error=err)

def libpack(cell):
    model,suite,n=cell.split('_');a,b,eps,sched=tables(model,suite,int(n))
    parent='bpool_cs' if model=='pi05' else 'bpool_all'
    from exp.offline_search.closed_loop.devset import library_episodes
    _,em,_=library_episodes(model,suite,parent)
    E=len(eps);H=max(len(e.steps) for e in eps)
    p={k:np.zeros((H,E),dtype=bool if k in ('alive','guard') else float) for k in ('alive','guard','step','risk')}
    for j,e in enumerate(eps):
        L=len(e.steps);p['alive'][:L,j]=True;p['guard'][:L,j]=e.guard
        p['step'][:L,j]=e.steps;p['risk'][:L,j]=e.risk**2
    p.update(N=np.array([e.n_slots for e in eps]),task=np.array([e.task for e in eps]),
        init=np.array([int(em[e.ep]['orig_init_state_idx']) for e in eps]),ep=np.array([e.ep for e in eps]))
    return p,sched

def devpack(meta):
    with np.load(HERE/'data'/f'{meta["root"]}__{meta["arm"]}.npz') as z:d={k:z[k] for k in z.files}
    E=len(d['keys']);H=int(d['epV'].max())
    p={k:np.zeros((H,E),dtype=bool if k in ('alive','guard') else float) for k in ('alive','guard','step','risk')}
    p['risk'][:]=np.nan
    for j in range(E):
        ix=np.flatnonzero((d['ep']==j)&d['vision'].astype(bool));L=len(ix);p['alive'][:L,j]=True
        for k in ('guard','step','risk'):p[k][:L,j]=d[k][ix]
    p.update(N=d['epN'],task=d['keys'][:,0].astype(int),init=d['keys'][:,1].astype(int))
    assert np.isfinite(p['guard'][p['alive']]).all()
    return p

def split(p):
    return np.array([int.from_bytes(hashlib.sha256(f'R14-cal:{t}:{i}'.encode()).digest()[:8],'big')%2==0
                     for t,i in zip(p['task'],p['init'])])

def diagnostics(p,model,K,z,cfg):
    base=replay(p,model,K);out=replay(p,model,cfg['K'])
    eligible=p['alive']&~p['guard'];A=base['calls']&eligible;B=out['calls']&eligible
    task_delta=[]
    for t in range(10):
        ix=p['task']==t;den=eligible[:,ix].sum()
        task_delta.append(float((B[:,ix].sum()-A[:,ix].sum())/den) if den else None)
    observed=[x for x in task_delta if x is not None]
    return dict(reference_ir=base['ir'],allocation_ir=out['ir'],delta_ir=out['ir']-base['ir'],
       placement_change=float((A!=B).sum()/max((A|B).sum(),1)),
       capture_delta=out['capture']-base['capture'],max_cache_controls=int(out['maxrun'].max()*10),
       task_knob_delta=task_delta,task_knob_delta_range=max(observed)-min(observed) if observed else None,
       task_ir=[float(out['cost'][p['task']==t].sum()/p['N'][p['task']==t].sum()) if np.any(p['task']==t) else None for t in range(10)],
       task_reference_ir=[float(base['cost'][p['task']==t].sum()/p['N'][p['task']==t].sum()) if np.any(p['task']==t) else None for t in range(10)])

def main():
    fence();fits=[];checks=[];splitchecks=[];priors=[];robust=[];dev={};per_task=[]
    for mp in sorted((HERE/'data').glob('*.json')):
        m=read(mp)
        if m['cell'] in CELLS and m['method'] in ('off','periodic'):
            dev.setdefault(m['cell'],[]).append((m,devpack(m)))
    for cell in CELLS:
        model=cell.split('_')[0];p,sched=libpack(cell)
        cal=split(p); pc=prep(take(p,cal));pv=prep(take(p,~cal))
        z,pinfo=priority(pc);priors.append(dict(cell=cell,**pinfo))
        # Final task priorities use all cross-fitted library episodes, never B SR.
        zfinal,finalinfo=priority(p)
        # Final production calibration uses all allowed B-dev. Keep split-based
        # fits/checks separately: those exposed the small-cohort imbalance risk.
        paths=[prep(v) for _,v in dev.get(cell,[])]
        splitpaths=[prep(take(v,split(v))) for _,v in dev.get(cell,[])]
        if not paths:paths=[prep(p)]
        if not splitpaths:splitpaths=[pc]
        for target in DOSES:
            K=round(sched.solve('periodic',target)[0],4)
            cfg=solve(paths,model,K,zfinal)
            fits.append(dict(cell=cell,target=target,reference_K=K,priority=finalinfo,
                gain=GAIN,prior_episodes=PRIOR_EPISODES,calibration_sources=[m['root']+':'+m['method'] for m,_ in dev.get(cell,[])] or ['library'],**cfg))
            # Honest episode holdout diagnostic for priority estimation/calibration.
            cvfit=solve([pc],model,K,z)
            checks.append(dict(cell=cell,target=target,source='library_identity_holdout',**diagnostics(pv,model,K,z,cvfit)))
            splitfit=solve(splitpaths,model,K,zfinal)
            for m,v in dev.get(cell,[]):
                vv=prep(take(v,~split(v)))
                checks.append(dict(cell=cell,target=target,source=m['root']+':'+m['method']+':development_full',**diagnostics(v,model,K,zfinal,cfg)))
                splitchecks.append(dict(cell=cell,target=target,source=m['root']+':'+m['method']+':identity_holdout',
                    episodes=len(vv['N']),tasks_present=len(np.unique(vv['task'])),**diagnostics(vv,model,K,zfinal,splitfit)))
            for t in range(10):
                per_task.append(dict(cell=cell,target=target,task=t,log_error=finalinfo['raw'][t],
                    episodes=finalinfo['n'][t],priority=zfinal[t],K=cfg['K'][t],reference_K=K))
            # Episode bootstrap of task estimates: fixed bias and recalibrated budget.
            rng=np.random.default_rng(20261004)
            for b in range(40):
                ids=np.concatenate([rng.choice(np.flatnonzero(p['task']==t),max(1,min(5,np.sum(p['task']==t))),replace=True) for t in range(10)])
                zz,_=priority(take(p,ids));kk=settings(K,zz,cfg['bias'])
                r=diagnostics(pv,model,K,zz,dict(K=kk))
                robust.append(dict(cell=cell,target=target,bootstrap=b,**r))
        print(cell,'priority range',float(zfinal.min()),float(zfinal.max()),flush=True)
    dump(HERE/'allocation_fits.json',fits);dump(HERE/'allocation_cv_priorities.json',priors)
    csvout(HERE/'allocation_checks.csv',checks);dump(HERE/'allocation_checks.json',checks)
    dump(HERE/'allocation_split_checks.json',splitchecks)
    csvout(HERE/'allocation_tasks.csv',per_task);dump(HERE/'allocation_robustness.json',robust)
    provenance('allocation')

if __name__=='__main__':main()
