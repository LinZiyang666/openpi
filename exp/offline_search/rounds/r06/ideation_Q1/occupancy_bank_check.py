"""POST-HOC: aggregate selected R on ten recorded A trajectories.

Evaluate against later A/B outcomes on inits 2..49, excluding recorded inits0/1.
Eight fixed banks remain only eight banks; no universal SR calibration is fit.
"""
import json
import numpy as np
import pandas as pd
import pilot_q1 as q

OUT=q.HERE/'occupancy_bank_check';OUT.mkdir(exist_ok=False)
arms=pd.read_csv(q.HERE/'exploratory_final/arms.csv')
Y=np.load(q.HERE/'exploratory_final/episode_outcomes.npz')
params=json.loads((q.HERE/'few_trajectory_check/calibration_parameters.json').read_text())
pars={r['cell']:r for r in params if r['candidate']=='R'}
rows=[];targets=[]
for m in ['pi05','groot']:
    for s in ['l10','sp']:
        for n in [50,500]:
            cell=f"{m}_{'spatial' if s=='sp' else s}_{n}"
            em=pd.read_csv(q.HERE/f'pilot_{m}_{s}_{n}/episode_scores.csv')
            em=em[(em.cohort=='A')&(em.split=='calibration')&(em.block==0)]
            assert len(em)==10
            p=pars[em.cell.iloc[0]];raw=em.R.mean();pred=p['intercept']+p['slope']*raw
            yy={}
            for family in ['A','B']:
                names=arms[(arms.cell==cell)&(arms.family==family)].arm
                yy[family]=np.mean([Y[a].reshape(10,50)[:,2:].ravel() for a in names],axis=0)
            rows.append(dict(cell=cell,raw_R_mean=raw,predicted_error_mean=pred,Q=1/(1+pred),A_failure=1-yy['A'].mean(),B_gain=(yy['B']-yy['A']).mean()))
            targets.append({'A_failure':1-yy['A'],'B_gain':yy['B']-yy['A']})
frame=pd.DataFrame(rows);frame.to_csv(OUT/'bank_scores.csv',index=False)
rng=np.random.default_rng(6062902);w=np.concatenate([rng.multinomial(48,np.ones(48)/48,size=10000) for _ in range(10)],axis=1)/480
result=[]
for candidate in ['raw_R_mean','predicted_error_mean']:
    for target in ['A_failure','B_gain']:
        draw=np.stack([w@t[target] for t in targets],axis=1)
        ci=q.interval(q.matrix_rho(np.tile(frame[candidate],(len(draw),1)),draw))
        result.append(dict(candidate=candidate,target=target,rho=q.rho(frame[candidate],frame[target]),lo=ci[0],hi=ci[1],
            banks=8,held_out_unique_inits_per_bank=480,scope='POST-HOC fixed-bank/fixed-task init uncertainty, calibration held fixed'))
pd.DataFrame(result).to_csv(OUT/'rank_correlations.csv',index=False)
print(json.dumps(result))
