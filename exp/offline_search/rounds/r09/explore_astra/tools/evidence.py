"""Consolidate numeric evidence, simultaneous bounds, and an explicit proxy failure."""
import json
import numpy as np
from scipy.stats import beta,norm
from .common import HERE,DERIVED,load_compact,dump,paired_bootstrap
from .shadows import errors,episode_mean


def cp_bound(candidate,reference,alpha=.05):
    c,r=np.asarray(candidate,bool).ravel(),np.asarray(reference,bool).ravel()
    w=int((c&~r).sum());l=int((~c&r).sum());n=len(c)
    lower=0. if w==0 else beta.ppf(alpha/2,w,n-w+1)
    upper=1. if l==n else beta.ppf(1-alpha/2,l+1,n-l)
    return dict(wins=w,losses=l,n=n,lower=float(lower-upper),alpha=alpha,passes_2pp=bool(lower-upper>-.02))


def main():
    metas=[json.loads(p.read_text()) for p in (DERIVED/'compact').glob('*.json')]
    result=dict(total_arms=len(metas),episodes=len(metas)*300,decisions=sum(m['decisions'] for m in metas),
                baseline=[],refresh_counterexample=[],local_call_multiplicity=[])
    for meta in sorted(metas,key=lambda m:m['arm']):
        if meta['variant']!='A':continue
        z=load_compact(meta['arm'])
        ref=next(m for m in metas if m['variant']=='P10' and m['model']==meta['model'] and m['suite']==meta['suite'])
        p=load_compact(ref['arm'])
        entry={k:meta[k] for k in ['arm','sr','ir','decisions']}
        entry.update(reference_sr=ref['sr'],reference_ir=ref['ir'],paired_sr_difference=paired_bootstrap(z['success']-p['success']),
                     nominal_cp=cp_bound(z['success'],p['success']),eight_cell_cp=cp_bound(z['success'],p['success'],.05/8))
        result['baseline'].append(entry)
        take=~z['vision']
        a=episode_mean(errors(z['served_chunk'],z['policy_shadow_chunk'],5)[0],z['task'],z['init'],take)
        b=episode_mean(errors(z['shadow_look_cache_chunk'],z['policy_shadow_chunk'],5)[0],z['task'],z['init'],take)
        row=dict(arm=meta['arm'],blind_refresh_relative_motion=float(b.mean()/a.mean()-1),
                 same_state_paired_error_change=paired_bootstrap(b-a))
        other=next((m for m in metas if m['arm']==meta['arm']+'5'),None)
        if other:
            o=load_compact(other['arm'])
            row.update(every5_sr=other['sr'],every10_sr=meta['sr'],
                       actual_sr_difference=paired_bootstrap(o['success']-z['success']))
        result['refresh_counterexample'].append(row)
    # Summarize first-entry results, keeping all 48 tests visible.
    cv=json.loads((HERE/'results/call_value.json').read_text())
    for cell in cv:
        for e in cell['results']:
            if 'local_high_minus_low_derivative' not in e:continue
            v=e['local_high_minus_low_derivative']
            # Normal approximation from bootstrap 95% width; only a coarse multiplicity diagnostic.
            se=(v['hi']-v['lo'])/(2*norm.ppf(.975))
            bound=norm.ppf(1-.05/(2*24))*se
            result['local_call_multiplicity'].append(dict(arm=cell['arm'],feature=e['feature'],mean=v['mean'],
                approximate_family_lo=v['mean']-bound,approximate_family_hi=v['mean']+bound,
                method='normal approximation from bootstrap width; descriptive only'))
    result['simple_camera_candidates']=[]
    for meta in metas:
        if meta['model']=='pi05' and meta['suite']=='spatial' and meta['library_size']==500 and meta['variant'] in ['W10','SW','SF1','A','CU','FL']:
            p=load_compact('r8_pi05_spatial_P10');z=load_compact(meta['arm'])
            result['simple_camera_candidates'].append(dict(arm=meta['arm'],sr=meta['sr'],ir=meta['ir'],
                paired_difference=paired_bootstrap(z['success']-p['success']),simultaneous_cp=cp_bound(z['success'],p['success'],.05/8)))
    dump(HERE/'results/evidence.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
