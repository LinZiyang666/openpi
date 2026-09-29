"""Continuation planning for pooled endpoints; preserves cross-cell pairing.

Post-estimation design projection only. Selection remains the frozen bootstrap.
"""
import json,math
import numpy as np
import pandas as pd
from scipy.stats import norm
import pilot_q1 as q

OUT=q.HERE/'continuation_measured'
ep=pd.read_csv(q.HERE/'pilot_all_cells/episode_scores.csv')
a=ep[ep.cohort=='A'].copy()
state=pd.read_csv(q.HERE/'pilot_all_cells/state_estimates.csv').set_index(['scope','candidate'])
comparisons=pd.read_csv(q.HERE/'pilot_all_cells/candidate_comparisons.csv')
rows=[];components=[]
targets={}
for candidate in q.PRIMARY:
    targets[candidate+'_rho']=(a['rho_'+candidate],1-.05/16)
    gain=state.loc[('pooled',candidate),'mae_gain'];base=state.loc[('pooled',candidate),'baseline_mae']
    targets[candidate+'_gain_influence']=((a.baseline_mae-a['mae_'+candidate]-gain*a.baseline_mae)/base,1-.05/16)
targets['Qrisk_improvement_over_R']=(a.mae_R-a.mae_Qrisk,1-.05/12)
for label,(target,coverage) in targets.items():
    a['target']=target
    # Equal-cell average inside a suite before ANOVA retains cross-model/bank
    # covariance at each task/init/seed, instead of pretending eight independent cells.
    avg=a.groupby(['suite','task_id','init','block']).target.mean().reset_index()
    stats=[]
    for suite,g in avg.groupby('suite'):
        wide=g.pivot(index=['task_id','init'],columns='block',values='target').sort_index()
        assert wide.shape==(20,3) and wide.notna().all().all()
        yy=wide.to_numpy().reshape(10,2,3);ii=yy.mean(2);tt=ii.mean(1)
        ve=np.sum((yy-ii[:,:,None])**2)/40
        mi=3*np.sum((ii-tt[:,None])**2)/10
        mt=6*np.sum((tt-yy.mean())**2)/9
        vi=(mi-ve)/3;vt=(mt-mi)/6
        comp=dict(label=label,suite=suite,raw_init_variance=vi,raw_task_variance=vt,seed_variance=ve,
                  raw_icc=vi/(vi+ve) if vi+ve>0 else None)
        components.append(comp);stats.append((max(0,vi),max(0,vt),ve))
    assert len(stats)==2
    for stage,repeats in [('pilot_validation',[3]*10),('full_validation',[3]*80+[2]*40+[1]*80)]:
        K=len(repeats)
        vfixed=sum(vi/K+ve*sum(1/r for r in repeats)/K**2 for vi,vt,ve in stats)/4
        vtask=sum(vt/10 for vi,vt,ve in stats)/4
        for scope,var in [('fixed_tasks',vfixed),('task_superpopulation',vfixed+vtask)]:
            rows.append(dict(label=label,stage=stage,scope=scope,variance=var,
                             half_simultaneous=norm.ppf(coverage)*math.sqrt(var)))
pd.DataFrame(rows).to_csv(OUT/'pooled_projections.csv',index=False)
q.dump(OUT/'pooled_components.json',dict(components=components,
     caveat='Normal plug-in projections; all-init variance estimation, coefficients fixed. Negative components clipped only for projection; no new survival test.',
     inputs={str(q.HERE/'pilot_all_cells/episode_scores.csv'):q.sha(q.HERE/'pilot_all_cells/episode_scores.csv')}))
print(pd.DataFrame(rows).query("stage == 'full_validation'").to_string(index=False))
