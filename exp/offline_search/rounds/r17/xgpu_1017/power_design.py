"""CPU planning sensitivity for the PRE-REGISTERED conservative paired interval.

Simulates independent episode pairs at true SR difference zero. This is not a
power claim for clustered task trajectories, nor an adaptive stopping rule.
"""
import argparse
import json
import numpy as np
from scipy.stats import beta
from artifacts import write_json


def design(repetitions=20000):
    rng=np.random.default_rng(101723)
    rows=[]
    for n in (100,500,1000,2500,5000):
        for discordance in (.01,.02,.05,.10,.20):
            cells=rng.multinomial(n,[discordance/2,discordance/2,1-discordance],size=repetitions)
            a,b=cells[:,0],cells[:,1]
            def bounds(k):
                lo=np.where(k==0,0,beta.ppf(.0125,k,n-k+1))
                hi=np.where(k==n,1,beta.ppf(.9875,k+1,n-k))
                return lo,hi
            la,ua=bounds(a);lb,ub=bounds(b)
            accepted=(la-ub>=-.02)&(ua-lb<=.02)
            probability=float(np.mean(accepted))
            rows.append(dict(pairs=n,discordance=discordance,true_difference=0,
                probability_CI_inside_margin=probability,
                monte_carlo_standard_error=float(np.sqrt(probability*(1-probability)/repetitions)),
                median_CI_width=float(np.median((ua-lb)-(la-ub))),
                zero_discordance_halfwidth=float(1-.0125**(1/n))))
    return dict(seed=101723,repetitions=repetitions,margin=.02,confidence=.95,
        method='same Bonferroni Clopper-Pearson interval as paired_sr.py',
        assumption='independent pairs, true difference zero; ignores task/seed clustering',
        status='PLANNING_ONLY_NOT_OBSERVED_SR',rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args();report=design();write_json(a.output,report)
    print(json.dumps(report))
