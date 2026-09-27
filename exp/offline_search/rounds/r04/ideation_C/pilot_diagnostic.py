"""Finite A-pool backtest: all-task, reference-outcome-stratified paired pilots.

Target outcomes are excluded from allocation. Historical comparator outcomes set
two pooled Bernoulli variances; every task x reference outcome has positive inclusion.
This does not estimate repeated-rollout stochastic variance.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import json
from pathlib import Path
import numpy as np
from exp.offline_search.rounds.r04.ideation_C.log_diagnostic import journal,OUT,RUN

def ys(run,arm):
    j=journal(run,arm)
    y=np.full((10,50),np.nan)
    for u,r in j.items():
        t,i=map(int,u.split(':')[-2:]); y[t,i]=int(r['success'])
    assert np.isfinite(y).all(),(run,arm)
    return y

def alloc(n,Nh,sd):
    # At least two per nonempty stratum where possible; fixed all-task coverage.
    nh=np.minimum(Nh,2)
    weights=Nh*sd
    while nh.sum()<n:
        target=(n-nh.sum())*weights/max(weights.sum(),1e-12)+nh
        # Greedy marginal variance reduction (using calibration sd only).
        gain=np.where(nh<Nh,(Nh*sd)**2/np.maximum(nh*(nh+1),1),-1)
        nh[np.argmax(gain)]+=1
    return nh

def bench(y0,y1,cal,seed=0):
    d=(y1-y0).ravel(); groups=[np.flatnonzero((np.repeat(np.arange(10),50)==t)&(y0.ravel()==s)) for t in range(10) for s in [0,1]]
    groups=[g for g in groups if len(g)]
    sizes=np.array([len(g) for g in groups]); W=sizes/500
    # Smooth pooled probabilities among past methods, conditional only on reference outcome.
    sds=[]
    probs={}
    for s in [0,1]:
        obs=np.concatenate([yy[y0==s] for yy in cal])
        p=(obs.sum()+1)/(len(obs)+2); probs[s]=p
    for g in groups:
        p=probs[int(y0.ravel()[g[0]])]; sds.append(np.sqrt(p*(1-p)))
    sds=np.array(sds)
    res=dict(delta=float(d.mean()),discordance=float(np.mean(d!=0)),calibration_p=probs)
    for n in [100,200,250]:
        # Exact conditional design variance for fixed historical outcomes, with replacement-free sampling.
        nt=n//10
        vb=sum((.1**2)*(1-nt/50)*np.var((y1-y0)[t],ddof=1)/nt for t in range(10))
        nh=alloc(n,sizes,sds)
        v=sum(W[h]**2*(1-nh[h]/sizes[h])*(np.var(d[g],ddof=1) if len(g)>1 else 0)/nh[h] for h,g in enumerate(groups))
        res[str(n)]=dict(sd_balanced=float(np.sqrt(vb)),sd_stratified=float(np.sqrt(v)),variance_ratio=float(v/vb) if vb else None,
                        allocation=nh.tolist(),sizes=sizes.tolist(),groups=[(int(g[0]//50),int(y0.ravel()[g[0]])) for g in groups])
        if n==100:
            rng=np.random.default_rng(seed); b=[]; st=[]
            for r in range(3000):
                b.append(np.mean([np.mean((y1-y0)[t,rng.choice(50,nt,replace=False)]) for t in range(10)]))
                st.append(sum(W[h]*np.mean(d[rng.choice(g,nh[h],replace=False)]) for h,g in enumerate(groups)))
            for name,a in [('balanced',b),('stratified',st)]:
                a=np.array(a); true=d.mean()
                res[str(n)][f'{name}_rmse']=float(np.sqrt(np.mean((a-true)**2)))
                res[str(n)][f'{name}_sign_error']=float(np.mean(a*np.sign(true)<=0)) if true else None
                res[str(n)][f'{name}_bias']=float(a.mean()-true)
    return res

if __name__=='__main__':
    results=[]
    for scale in [50,500]:
        for model,l in [('pi05','p'),('groot','g')]:
            for suite,s in [('spatial','sp'),('l10','l10')]:
                run=f'r02_g{scale}'; prefix=f'oscl{scale}_{l}_{s}_cl'
                yy={i:ys(run,prefix+str(i)) for i in range(4)}
                targets=[(run,prefix+str(i),yy[i]) for i in [0,1,3]]
                if scale==50:
                    for suffix in (['a05','rm1'] if s=='sp' else ['a05']):
                        name=f'r3f_{l}_{s}_{suffix}'
                        if (RUN/'r03_full'/'runs'/name).exists(): targets.append(('r03_full',name,ys('r03_full',name)))
                if model=='pi05':
                    for suffix in (['g','awm_h70','perk3'] if scale==50 else ['g500','awm500_h70']):
                        name=f'r3mx_p_{s}_{suffix}'
                        if (RUN/'r03_mx'/'runs'/name).exists(): targets.append(('r03_mx',name,ys('r03_mx',name)))
                for trun,name,y in targets:
                    cal=[yy[i] for i in [0,1,3] if not(trun==run and name==prefix+str(i))]
                    r=bench(yy[2],y,cal)
                    tasks=([6,9,0,4,1] if suite=='spatial' else [0,4,6,8,7])
                    r.update(scale=scale,model=model,suite=suite,reference=prefix+'2',target=name,
                        old_pilot_delta=float((y-yy[2])[tasks,:20].mean()))
                    results.append(r)
                    print(scale,model,suite,name,'delta',round(r['delta'],3),'var ratio',round(r['100']['variance_ratio'],3),flush=True)
    (OUT/'pilot_backtest.json').write_text(json.dumps(results,indent=2))
