"""Randomized first-alert call effects under the original IP(.25) future.

Alerts and nuisance heads fit 0..19; causal estimates ONLY 20..29. A/CU monitor
does not use IP treatment/outcomes. No conditioning on post-treatment survival.
Progress outcomes stop at min(horizon, termination), with success absorbing.
This identifies a one-call excursion, NOT a multi-call burst or gate SR.
"""
import json
import numpy as np
from ...round2.tools.learn import fit
from ...round2.inference import predict
from .data import HERE,subset,interval,dump
from .screen import population,selected_entries


def one(cell):
    d=population(cell,'IP')
    if d is None:return None
    if not d['eligible'].all() or not np.allclose(d['p'],.25) or not np.array_equal(d['coin']<d['p'],d['treatment']):
        raise ValueError('randomization audit failed')
    tr,te=subset(d,d['init']<20),subset(d,d['init']>=20)
    with np.load(HERE/'artifacts'/f'{cell}_monitor.npz',allow_pickle=False) as z:head={k:z[k] for k in z.files}
    meta=json.loads((HERE/'artifacts'/f'{cell}_monitor.json').read_text())
    s=predict(head,te['x'])[:,0]
    first,_,_=selected_entries(te,s,meta['threshold'])
    # A privileged first-current-stall diagnostic is pre-treatment, but is NOT
    # a proposed online method. Contrast it to the deployable learned monitor.
    oracle=np.zeros(len(s),bool)
    for ep in np.unique(te['episode']):
        ix=np.flatnonzero((te['episode']==ep)&te['late_stall']&(te['seq']>=12))
        if len(ix):oracle[ix[0]]=True
    a,p=te['treatment'],te['p'];out={}
    for gate,take in [('learned',first),('privileged_stall',oracle)]:
        out[gate]=dict(reached=int(take.sum()),calls=int(a[take].sum()),outcomes={})
        for name in ['progress30','progress50','progress100','success']:
            model=fit(tr['x'],tr[name][:,None].astype(float),tr['episode'],random_features=128,alpha=100.)
            nuisance=np.clip(predict(model,te['x'])[:,0],0,1)
            ht=(a/p-(1-a)/(1-p))*(te[name]-nuisance)
            ep,inv=np.unique(te['episode'],return_inverse=True)
            value=np.bincount(inv,weights=ht*take,minlength=len(ep))
            raw=np.bincount(inv,weights=(a/p-(1-a)/(1-p))*te[name]*take,minlength=len(ep))
            r=dict(centered=interval(value,ep),uncentered=interval(raw,ep),
                observed_call_mean=float(np.mean(te[name][take&a])) if np.any(take&a) else None,
                observed_cache_mean=float(np.mean(te[name][take&~a])) if np.any(take&~a) else None)
            # The conditional interval is resampled jointly with reach; do not
            # mistake the entire-cohort estimate for an effect among alerts.
            if take.any():
                reach=np.bincount(inv,weights=take,minlength=len(ep))
                rng=np.random.default_rng(20261002);clusters=np.unique(ep%30)
                vv=np.array([value[ep%30==i].sum() for i in clusters]);nn=np.array([reach[ep%30==i].sum() for i in clusters])
                draws=rng.integers(0,len(clusters),(3000,len(clusters)))
                boot=vv[draws].sum(1)/np.maximum(nn[draws].sum(1),1)
                r['among_reached']=dict(mean=float(value.sum()/take.sum()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)))
            out[gate]['outcomes'][name]=r
    result=dict(cell=cell,fit_inits=list(range(20)),eval_inits=list(range(20,30)),gates=out,
        warning='Exploratory clustered first-entry effects; no deterministic-gate or burst SR identification')
    dump(HERE/'results'/f'causal_{cell}.json',result)
    print(cell,json.dumps(out),flush=True)
    return result


if __name__=='__main__':dump(HERE/'results/causal.json',[one(c) for c in ['pi05_l10_50','groot_l10_50']])
