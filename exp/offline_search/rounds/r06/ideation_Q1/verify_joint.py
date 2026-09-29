"""Independent assembly check using the coordinator's frozen per-cell scores.

Re-runs the *same frozen* joint bootstrap and rule; this is not a different
estimator or an average of the per-cell confidence intervals.
"""
import json
import numpy as np
import pandas as pd
import pilot_q1 as q

cells=[f'{m}_{s}_{n}' for m in ['pi05','groot'] for s in ['l10','sp'] for n in [50,500]]
em=pd.concat([pd.read_csv(q.HERE/f'pilot_{c}/episode_scores.csv') for c in cells],ignore_index=True)
ep=pd.concat([pd.read_csv(q.HERE/f'pilot_{c}/episode_outcomes.csv') for c in cells],ignore_index=True)
infer=q.Inference(ep[ep.split=='validation'],10000)
s,c=q.state_estimates(em,infer)
decision=q.decision_rule(s,c,q.complete_cells(ep,False),False)
out=q.HERE/'joint_crosscheck';out.mkdir(exist_ok=True)
s.to_csv(out/'state_estimates.csv',index=False);c.to_csv(out/'candidate_comparisons.csv',index=False)
q.dump(out/'decision.json',decision)
result=dict(episodes=len(ep),selection=decision,raw_joint_comparison='not yet available')
joint=q.HERE/'pilot_all_cells'
if (joint/'state_estimates.csv').exists():
    for name in ['state_estimates','candidate_comparisons']:
        a=pd.read_csv(out/(name+'.csv'));b=pd.read_csv(joint/(name+'.csv'))
        assert list(a.columns)==list(b.columns)
        num=a.select_dtypes(include='number').columns
        assert np.allclose(a[num],b[num],rtol=1e-10,atol=1e-11,equal_nan=True),name
    assert decision==json.loads((joint/'decision.json').read_text())
    result['raw_joint_comparison']='PASS: estimates/intervals agree within 1e-10; decision identical'
q.dump(out/'validation.json',result)
print(s[(s.scope=='pooled')&s.candidate.isin(q.PRIMARY)][['candidate','rho','rho_sim_lo','rho_sim_hi','mae_gain','gain_sim_lo','gain_sim_hi']].to_json(orient='records'))
print(json.dumps(decision))
