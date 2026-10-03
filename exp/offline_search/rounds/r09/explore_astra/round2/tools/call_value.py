"""Learn task-agnostic call moderators from randomized IP(.25) data.

Fit inits 0..19 only, estimate excursions/LOCAL derivatives on 20..29 only.
No shadow disagreement is treated as a causal value label. This tool does NOT
claim to estimate SR under a repeatedly deployed deterministic gate.
"""
import numpy as np
from .data import HERE, CELLS, dataset, subset, masks, features, epmeans, interval, dump
from .learn import fit
from ..inference import predict


def inputs(d):
    # All are known before the current call coin. No outcomes or teacher actions.
    diag=np.nan_to_num(d['diagnostics'],nan=0,posinf=0,neginf=0)
    return np.c_[features(d,True),diag]


def audit(d):
    if not np.all(d['eligible']) or not np.allclose(d['p'],.25):
        raise ValueError('Need randomized interior support at every selected anchor')
    if not np.array_equal(d['coin']<d['p'],d['treatment']):
        raise ValueError('Treatment/coin mismatch')


def first_entry(take,episode):
    out=np.zeros(len(take),bool);seen=set()
    for j in np.flatnonzero(take):
        if episode[j] not in seen: out[j]=True;seen.add(episode[j])
    return out


def sums(value,d):
    ep,inv=np.unique(d['episode'],return_inverse=True)
    return ep,np.bincount(inv,weights=value)


def one(cell):
    d=dataset(cell,'IP')
    if d is None:return None
    audit(d)
    tr,te=masks(d['init']);train,test=subset(d,tr),subset(d,te)
    x,xt=inputs(train),inputs(test)
    # Cross-fitted nuisance on train initial states; no task-specific baseline.
    nuisance=np.zeros(len(x))
    for fold in range(4):
        f=train['init']%4==fold
        m=fit(x[~f],train['success'][~f,None],train['episode'][~f],random_features=128,alpha=100)
        nuisance[f]=np.clip(predict(m,x[f])[:,0],0,1)
    m=fit(x,train['success'][:,None],train['episode'],random_features=128,alpha=100)
    mt=np.clip(predict(m,xt)[:,0],0,1)
    a,p=train['treatment'],train['p']
    target=(a/p-(1-a)/(1-p))*(train['success']-nuisance)
    value=fit(x,target[:,None],train['episode'],random_features=128,alpha=300)
    scores={'value':predict(value,xt)[:,0], 'risk':1-mt,
            'low_motion':-np.sqrt((test['history'][:,:8]**2).mean(1)),
            'distance':np.nan_to_num(test['diagnostics'][:,0])}
    train_scores={'value':predict(value,x)[:,0], 'risk':1-nuisance,
            'low_motion':-np.sqrt((train['history'][:,:8]**2).mean(1)),
            'distance':np.nan_to_num(train['diagnostics'][:,0])}
    a,p=test['treatment'],test['p']
    ht=(a/p-(1-a)/(1-p))*(test['success']-mt)
    out=dict(cell=cell,fit_inits=list(range(20)),eval_inits=list(range(20,30)),
             train_anchors=len(x),eval_anchors=len(xt),gates={})
    for name,score in scores.items():
        threshold=float(np.quantile(train_scores[name],.75))
        high=score>=threshold
        first=first_entry(high,test['episode'])
        ep,e=sums(ht*first,test)
        # Centered tilt uses a threshold fixed on training. Centering rate itself
        # is fixed from training, not from treatment/outcomes in evaluation.
        rate=float(np.mean(train_scores[name]>=threshold))
        _,der=sums(ht*(high.astype(float)-rate),test)
        result=dict(threshold=threshold,eval_anchor_fraction=float(high.mean()),
            reached_episodes=int(first.sum()),calls_at_first=int(a[first].sum()),
            first_entry_call_minus_cache=interval(e,ep),
            local_probability_derivative=interval(der,ep),
            warning='First-entry excursion under IP future; derivative is infinitesimal, neither is gate SR')
        out['gates'][name]=result
    dump(HERE/'results'/f'call_value_{cell}.json',out)
    print(cell,{k:(round(v['first_entry_call_minus_cache']['mean'],3),
        [round(v['first_entry_call_minus_cache'][q],3) for q in ['lo','hi']]) for k,v in out['gates'].items()},flush=True)
    return out


def main():
    result=[one(c) for c in CELLS if c.endswith('_50')]
    dump(HERE/'results/call_value.json',result)


if __name__=='__main__':main()
