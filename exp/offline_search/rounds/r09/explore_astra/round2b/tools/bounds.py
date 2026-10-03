"""First-intervention coupling bounds and explicit rescue/harm sensitivity.

Before its first alert the method serves unchanged cache actions. Under exact
same-path coupling, only alerted episodes can change success. These finite-data
bounds are NOT confidence intervals and do NOT assert any recovery probability.
"""
import json
import numpy as np
from .data import HERE,COMPACT,validate_identity,dump


def policy_references():
    out=[]
    for name in ['r8_pi05_l10_P10','r8_groot_l10_P10']:
        path=COMPACT/f'{name}.npz'
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['episodes']!=300 or 'inits 0..29' not in meta['split']:raise ValueError('discovery provenance')
        with np.load(path,allow_pickle=False) as z:
            validate_identity(z['task'],z['init'])  # before outcomes
            if z['success'].shape!=(10,30):raise ValueError('forbidden outcome ledger')
            success=z['success'][:,20:30];cost=z['cost'][:,20:30];dec=z['decisions'][:,20:30]
            out.append(dict(arm=name,eval_inits=list(range(20,30)),n=100,
                sr=float(success.mean()),ir=float(cost.sum()/dec.sum())))
    return out


def coupling(n,success,alert_failure,alert_success,q,h):
    if not 0<=q<=1 or not 0<=h<=1:raise ValueError('probabilities')
    if alert_failure>n-success or alert_success>success:raise ValueError('invalid partition')
    lower=(success-alert_success)/n
    upper=(success+alert_failure)/n
    scenario=(success+q*alert_failure-h*alert_success)/n
    return dict(lower=lower,upper=upper,scenario=scenario)


def main():
    refs=policy_references();dump(HERE/'results/policy_reference.json',refs)
    reference={x['arm'].split('_')[1]:x['sr'] for x in refs}
    out=[]
    for r in json.loads((HERE/'results/screen.json').read_text()):
        cell=r['cell'];d=r['populations']['A'];n=d['n_episodes'];success=n-d['failed_episodes']
        af=round(d['failure_coverage']*d['failed_episodes']);ass=d['fired_episodes']-af
        scenarios={str(q):coupling(n,success,af,ass,q,.1)['scenario'] for q in [0,.25,.5,.75,1.]}
        bounds=coupling(n,success,af,ass,0,0)
        ref=reference[cell.split('_')[0]]
        out.append(dict(cell=cell,n=n,baseline_sr=success/n,alert_failed=af,alert_successful=ass,
            coupling_SR_bounds=[bounds['lower'],bounds['upper']],
            scenarios_harm_probability_point1=scenarios,historical_P10_SR=ref,
            rescue_fraction_to_match_P10=(ref*n-success+.1*ass)/af if af else None,
            warning='Same-path coupling assumption; scenarios are assumptions, not fitted SR forecasts. Historical P10 differs from current topology/draw.'))
    dump(HERE/'results/bounds.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
