"""Conservative finite-sample paired SR interval; no zero-discordance collapse."""
import argparse
import json
from pathlib import Path
from scipy.stats import beta, binomtest
from artifacts import write_json


def paired_interval(a, b, margin=.02):
    if not a or len(a) != len(b) or any(type(x) not in (bool,int) or x not in (0,1) for x in a+b):
        raise ValueError('matched binary accepted outcomes required')
    n=len(a); only_a=sum(x and not y for x,y in zip(a,b)); only_b=sum(y and not x for x,y in zip(a,b))
    # Each discordant-cell marginal has an exact binomial CI with alpha=.025;
    # Bonferroni gives >=95% joint coverage, hence a valid difference interval.
    def interval(k):
        return (float(beta.ppf(.0125,k,n-k+1)) if k else 0.,
                float(beta.ppf(.9875,k+1,n-k)) if k<n else 1.)
    la,ua=interval(only_a);lb,ub=interval(only_b)
    ci=[la-ub,ua-lb]
    return dict(pairs=n,only_a=only_a,only_b=only_b,difference=(only_a-only_b)/n,
        CI95=ci,method='conservative exact marginal Clopper-Pearson union bound; paired multinomial cells',
        mcnemar_p=float(binomtest(only_a,only_a+only_b,.5).pvalue) if only_a+only_b else 1.,
        margin=margin,equivalent=ci[0]>=-margin and ci[1]<=margin,
        assumption='independent episode pairs; task/seed clustering requires separate cluster sensitivity analysis')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('left');p.add_argument('right');p.add_argument('--output',required=True)
    a=p.parse_args()
    def load(path):
        rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
        result={}
        for r in rows:
            if r.get('accepted') is not True:raise ValueError('all pairs must have accepted outcomes')
            key=tuple(r[k] for k in ('model','suite','arm','task_id','init_idx','env_seed','policy_seed'))
            if key in result:raise ValueError('duplicate episode identity')
            result[key]=r['success']
        return result
    l,r=load(a.left),load(a.right)
    if set(l)!=set(r):raise ValueError('missing paired episode; never drop failed/missing runs')
    keys=sorted(l)
    write_json(a.output,paired_interval([l[k] for k in keys],[r[k] for k in keys]))


if __name__=='__main__':main()
