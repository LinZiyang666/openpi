"""Discovery-only look diagnostics with explicit LIBERO six-motion/one-gripper adapter."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from callback_calls import RUN, OUT, DiscoveryArm, frozen_r7, save
from exp.offline_search.debug.tools.decision import common as C, stage_ledger


p=argparse.ArgumentParser()
p.add_argument('--arms',nargs='+',default=['r8_groot_l10_50_A','r8_groot_l10_50_FL','r8_groot_l10_500_A','r8_groot_l10_500_FL'])
arms=p.parse_args().arms
rng=np.random.default_rng(20261001)
REPS=3000
W=np.stack([rng.multinomial(30,np.full(30,1/30),size=REPS) for _ in range(10)],axis=1)
aggregates=[]
ledgers=[]


def summarize(name,group,label,age):
    row=dict(arm=name,stage=label,age=age,decisions=len(group),episodes=group.episode_key.nunique())
    for metric in ['tail_motion_rms','fresh_motion_rms','motion_improvement','motion_mse_improvement','all7_improvement',
                   'tail_look_motion_rms','tail_look_gripper_flip','gripper_mse_share']:
        means=group.groupby(['task_id','init'])[metric].mean()
        a=np.zeros((10,30));n=np.zeros((10,30))
        for (t,i),v in means.items():a[int(t),int(i)]=v;n[int(t),int(i)]=1
        bs=(W*a[None]).sum((1,2))/(W*n[None]).sum((1,2))
        row[metric]=means.mean()
        row[metric+'_lo'],row[metric+'_hi']=np.quantile(bs,[.025,.975])
        row[metric+'_decision_mean']=group[metric].mean()
    row['fraction_fresh_motion_closer']=(group.motion_improvement>0).mean()
    return row


for name in arms:
    assert (RUN/'state'/f'{name}.AUG_DONE').exists()
    print('start',name,flush=True)
    target=OUT/'shadows'/name
    if (target/'look_summary_rows.json').exists():
        print('already completed',name,flush=True)
        continue
    arm=DiscoveryArm(name)
    frozen_r7(arm)
    arm.stage_column='r7_stage'
    # Existing environment adapter declares dim 6 and threshold 0; no data-fit threshold.
    metadata=C.metadata(arm)
    save(target/'metric_manifest_audit.json',dict(
        raw={k:metadata.get(k) for k in ['valid_action_dims','gripper_dim','gripper_threshold','action_scale','action_std']},
        analysis_adapter=dict(gripper_dim=6,gripper_threshold=0.,motion_dims=list(range(6)),
                              source='debug/tools/physical/adapters.py EnvironmentAdapter.defaults: libero')))
    assert metadata['valid_action_dims']==list(range(7))
    arm.profile_metadata.update(gripper_dim=6,gripper_threshold=0.)
    ds,_=C.inputs(arm)
    ids=ds.decision_id.tolist()
    live=C.arrays(arm,['served_chunk'],ids)['served_chunk'][:,:5,:7].astype(float)
    fresh=C.pick(C.augmentation(arm,'shadow_look',ids),'cache_chunk')[:,:5,:7]
    policy=C.pick(C.augmentation(arm,'policy_shadow',ids),'chunk')[:,:5,:7]
    n=ds.n_applied.to_numpy(int)
    mask=np.arange(5)[None,:]<n[:,None]
    def mse(x,dims):
        v=(x[:,:,dims]**2).mean(axis=2)
        return np.where(mask,v,0.).sum(1)/n
    tail6=mse(live-policy,list(range(6)))
    fresh6=mse(fresh-policy,list(range(6)))
    tail7=mse(live-policy,list(range(7)))
    grip=mse(live-policy,[6])
    frame=ds[['decision_id','episode_key','task_id','init','stage','src','vision','blind_age_controls']].copy()
    frame['tail_motion_rms']=np.sqrt(tail6)
    frame['fresh_motion_rms']=np.sqrt(fresh6)
    frame['motion_improvement']=np.sqrt(tail6)-np.sqrt(fresh6)
    frame['motion_mse_improvement']=tail6-fresh6
    frame['all7_improvement']=np.sqrt(tail7)-np.sqrt(mse(fresh-policy,list(range(7))))
    frame['tail_look_motion_rms']=np.sqrt(mse(live-fresh,list(range(6))))
    frame['tail_look_gripper_flip']=(((live[:,:,6]>0)!=(fresh[:,:,6]>0))*mask).sum(1)/n
    frame['gripper_mse_share']=np.divide(grip,7*tail7,out=np.zeros(len(ds)),where=tail7>0)
    frame=frame[~frame.vision.astype(bool)&frame.src.ne('policy_tail')].copy()
    assert np.isfinite(frame[['tail_motion_rms','fresh_motion_rms']]).all().all()
    frame.to_parquet(target/'look_policy_six_motion.parquet',index=False)
    rows=[]
    for age,group in frame.groupby('blind_age_controls'):
        rows.append(summarize(name,group,'__all__',age))
        for label,g in group.groupby('stage'):
            rows.append(summarize(name,g,label,age))
    save(target/'look_summary_rows.json',rows)
    ledger=stage_ledger.analyze(arm)
    save(target/'stage_ledger.json',ledger)
    ledgers.append(dict(arm=name,**ledger['tables']['stages'][0]))
    print('done',name,len(frame),flush=True)

target=OUT/'summary';target.mkdir(exist_ok=True)
for path in (OUT/'shadows').glob('*/look_summary_rows.json'):
    aggregates.extend(json.loads(path.read_text()))
ledgers=[dict(arm=path.parent.name,**json.loads(path.read_text())['tables']['stages'][0])
         for path in (OUT/'shadows').glob('*/stage_ledger.json')]
pd.DataFrame(aggregates).to_csv(target/'look_summary.csv',index=False)
pd.DataFrame(ledgers).to_csv(target/'look_ledgers.csv',index=False)
print(pd.DataFrame(aggregates).query('stage == "__all__"')[['arm','age','decisions','episodes','motion_improvement','motion_improvement_lo','motion_improvement_hi','tail_look_motion_rms','tail_look_gripper_flip']].to_string(index=False))
