"""Audited IP(.25) excursions, using pre-coin observable retrieval diagnostics.

Final-success centered Horvitz-Thompson scores, summed within episodes BEFORE
task-stratified uncertainty. First-entry effects are distinct interventions.
The all-anchor score is only a LOCAL probability-shift derivative, not policy SR.
Shadow disagreement moderation is diagnostic, not an online-free feature.
"""
import json
import numpy as np
from .common import HERE, DERIVED, load_compact, dump, paired_bootstrap


def cluster_sum(values, task, init):
    out=np.zeros((10,30))
    np.add.at(out,(task,init),values)
    return out


def one(meta):
    z=load_compact(meta['arm'])
    keep=z['vision'] & z['eligible']
    p=z['p'][keep];coin=z['coin'][keep];a=z['treatment'][keep]
    if not np.allclose(p,.25) or not np.array_equal(coin<p,a):
        raise ValueError('randomization support or replay failure')
    task,init=z['task'][keep],z['init'][keep]
    y=z['success'][task,init]
    baseline=np.zeros(len(task))
    for fold in range(5):
        train=np.arange(30)%5!=fold
        b=z['success'][:,train].mean(1)
        test=init%5==fold
        baseline[test]=b[task[test]]
    score=(a/p-(1-a)/(1-p))*(y-baseline)
    weight=z['shadow_look_weights'][keep]
    features={k:z[k][keep] for k in ['disp5','d1_rel','dst']}
    features['elapsed']=z['seq'][keep].astype(float)
    features['kernel_entropy']=-(weight*np.log(np.maximum(weight,1e-12))).sum(1)
    features['shadow_motion_disagreement']=((z['shadow_look_cache_chunk'][keep,:,:6]-z['policy_shadow_chunk'][keep,:,:6])**2).mean((1,2))
    results=[]
    for name,x in features.items():
        group=np.full(len(x),False)
        for fold in range(5):
            for t in range(10):
                train=(init%5!=fold)&(task==t)&np.isfinite(x)
                test=(init%5==fold)&(task==t)&np.isfinite(x)
                group[test]=x[test]>np.median(x[train])
        for high in [False,True]:
            eligible=(group==high)&np.isfinite(x)
            first=np.zeros(len(x),bool);seen=set()
            for j in np.flatnonzero(eligible):
                key=(int(task[j]),int(init[j]))
                if key not in seen:
                    first[j]=True;seen.add(key)
            weighted=cluster_sum(np.where(first,score,0.),task,init)
            reached=len(seen)
            estimate=paired_bootstrap(weighted)
            results.append(dict(feature=name,high=high,reached=reached,calls=int(a[first].sum()),
                                caches=int((~a[first]).sum()),population_effect=estimate,
                                reached_effect={k:v*300/reached for k,v in estimate.items()},
                                estimand='first supported entry to this score half, then original IP controller'))
        derivative=cluster_sum(score*np.where(group,1.,-1.),task,init)
        results.append(dict(feature=name,local_high_minus_low_derivative=paired_bootstrap(derivative),
                            estimand='derivative for p -> p + epsilon*(high-low); no finite-policy extrapolation'))
    return dict(arm=meta['arm'],anchors=len(p),coin_mismatches=0,episodes=300,results=results)


def main():
    metas=[json.loads(p.read_text()) for p in (DERIVED/'compact').glob('*.json')]
    results=[one(m) for m in metas if m['variant']=='IP']
    dump(HERE/'results/call_value.json',results)
    for r in results:
        print(r['arm'],r['anchors'],flush=True)
        for e in r['results']:
            if 'local_high_minus_low_derivative' in e:
                print(e['feature'],e['local_high_minus_low_derivative'],flush=True)


if __name__=='__main__':
    main()
