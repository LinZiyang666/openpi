"""POST-HOC sensitivity: ten recorded calibration trajectories per cell.

Keep only the first seed block (603), init=0, for calibration. Preserve every
preregistered validation A episode. No search over which trajectories to use.
"""
import json
import pandas as pd
import pilot_q1 as q

OUT=q.HERE/'few_trajectory_check';OUT.mkdir(exist_ok=False)
pieces=[];inputs={}
for m in ['pi05','groot']:
    for s in ['l10','sp']:
        for n in [50,500]:
            p=q.HERE/f'pilot_{m}_{s}_{n}/anchor_scores.csv'
            x=pd.read_csv(p,low_memory=False);x=x[x.cohort=='A']
            pieces.append(x[(x.split=='validation')|(x.block==0)])
            inputs[str(p)]=q.sha(p)
x=pd.concat(pieces,ignore_index=True)
x,params=q.refit(x);em=q.episode_metrics(x)
assert all(len(a['calibration_episodes'])==10 for a in params)
inf=q.Inference(em[em.split=='validation'],10000)
state,comparisons=q.state_estimates(em,inf)
state.to_csv(OUT/'state_estimates.csv',index=False)
comparisons.to_csv(OUT/'candidate_comparisons.csv',index=False)
q.dump(OUT/'calibration_parameters.json',params)
q.dump(OUT/'manifest.json',dict(label='POST-HOC sensitivity; does not alter frozen selection',inputs=inputs,
    calibration_rule='A, init0, block0 only: 10 episodes/cell; validation A init1 all three blocks unchanged',
    calibration_episodes=int(em[em.split=='calibration'].uid.nunique()),validation_episodes=int(em[em.split=='validation'].uid.nunique())))
print(state[(state.scope=='pooled')&state.candidate.isin(q.PRIMARY)][['candidate','rho','mae_gain','gain_sim_lo','gain_sim_hi']].to_json(orient='records'))
