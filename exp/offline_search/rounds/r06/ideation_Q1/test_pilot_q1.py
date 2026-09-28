"""Statistical contract checks; no simulator, model or pilot observations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import pilot_q1 as q


def main():
    passed=[]
    # Unequal trajectory lengths must not alter the episode estimand.
    frame=pd.DataFrame([dict(cell='m_s_l',suite='s',task_id=0,init=i,arm='A',uid=str(i),attempt=1,
                            step=j,cohort='A',split='calibration' if i==0 else 'validation',E=1+2*j,
                            **{k:float(j) for k in q.PRIMARY+q.SECONDARY})
                        for i,n in [(0,3),(1,9)] for j in range(n)])
    w=q.hierarchy_weights(frame)
    assert np.isclose(w[:3].sum(),.5) and np.isclose(w[3:].sum(),.5)
    passed.append('equal init/episode weighting despite unequal anchor counts')
    _,params=q.refit(frame)
    changed=frame.copy();changed.loc[changed.split=='validation','E']=1e6
    _,params2=q.refit(changed)
    assert params==params2
    assert np.allclose([params[0]['intercept'],params[0]['slope']],[1,2])
    passed.append('calibration fit is independent of validation labels')
    # Shared cluster multiplicities must preserve exact paired differences.
    ep=pd.DataFrame([dict(cell=cell,suite='s',task_id=t,init=i,uid=f'{cell}/{t}/{i}/{b}',z=float(t+i))
                     for cell in ['a','b'] for t in range(3) for i in range(2) for b in range(3)])
    inf=q.Inference(ep,100)
    pa,ba=inf.aggregate(ep[ep.cell=='a'],['z']);pb,bb=inf.aggregate(ep[ep.cell=='b'],['z'])
    assert np.array_equal(pa,pb) and np.array_equal(ba,bb)
    assert len(inf.index)==6
    passed.append('seed repeats and matched cells share cluster draws')
    fixed=q.Inference(ep,100,fixed_tasks=True)
    for t in range(3):assert np.all(fixed.mult[:,fixed.index.task_id==t].sum(1)==2)
    assert q.budget_screen(pd.DataFrame(),True,{})==[]
    passed.append('fixed-task bootstrap retains all tasks; A-only partial input has no dose selection')
    # Task means are equal even if numbers of init rows vary or values are missing.
    d=ep[ep.cell=='a'].copy();d.loc[(d.task_id==0)&(d.init==0),'z']=np.nan
    expected=d.groupby(['task_id','init']).z.mean().groupby('task_id').mean().mean()
    p,_=inf.aggregate(d,['z']);assert np.isclose(p[0],expected)
    passed.append('hierarchical means retain task weighting with missing ranks')
    a=np.array([[1,1,3,4],[1,4,2,3]],float);b=np.array([[2,2,4,1],[1,2,3,4]],float)
    assert np.allclose(q.matrix_rho(a,b),[spearmanr(x,y).statistic for x,y in zip(a,b)])
    passed.append('vectorized bootstrap Spearman handles repeated task ties')
    xx=np.array([1.,2.,2.,4.]);yy=np.array([2.,4.,3.,1.]);ww=np.array([1,2,3,2])
    assert np.isclose(q.weighted_rho(xx,yy,ww),spearmanr(np.repeat(xx,ww),np.repeat(yy,ww)).statistic)
    passed.append('inverse-selection weighted ranks agree with exact expanded sample')
    u=np.array([0.,1.,2.,3.]);cache=5.
    V=u.var(ddof=1);L=np.mean((cache-u)**2)
    assert np.isclose(L-V,(cache-u.mean())**2-V/4)
    passed.append('K4 signed excess uses unbiased sample variance')
    # Exact enumeration checks marginal mixture and duration/hold package weights.
    p=.125;y0=.2;ys=np.array([.3,.4,.5,.6,.7,.8]);joint=1/6
    mixture=sum(p*joint*(y/p) for y in ys)+(1-p)*(-y0/(1-p))
    assert np.isclose(mixture,ys.mean()-y0)
    package=sum(p*joint*(ys[i]/(p*joint) if i==3 else 0) for i in range(6))+(1-p)*(-y0/(1-p))
    assert np.isclose(package,ys[3]-y0)
    passed.append('factorial HT uses actual propensity and package choice probability')
    state=pd.DataFrame([dict(scope=s,candidate=k,rho=.5,rho_sim_lo=.3,rho_sim_hi=.7,
                             mae_gain=.3,gain_sim_lo=.2,gain_sim_hi=.4,rho_fraction=1)
                        for s in ['pooled','cell'] for k in q.PRIMARY])
    comparisons=pd.DataFrame([dict(scope='pooled',simpler=a,complex=b,lo=-1) for a in q.PRIMARY for b in q.PRIMARY])
    complete={str(i):dict(pilot_complete=True) for i in range(8)}
    assert q.decision_rule(state,comparisons,complete,True)['selected'] is None
    assert q.decision_rule(state,comparisons,{'one':dict(pilot_complete=True)},False)['selected'] is None
    assert q.decision_rule(state,comparisons,complete,False)['selected']=='C'
    passed.append('smoke/partial cells cannot select a method; simplicity order enforced')
    result=dict(passed=len(passed),checks=passed)
    q.dump(q.HERE/'pilot_q1_test_results.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
