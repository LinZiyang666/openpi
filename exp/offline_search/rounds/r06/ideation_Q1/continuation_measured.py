"""Measured-variance continuation planning; not an additional selection test.

Uses accepted pilot outcomes and the frozen per-cell episode scores. All-init
data are used only to estimate precision, as allowed by the preregistration.
"""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import pilot_q1 as q

OUT=q.HERE/'continuation_measured';OUT.mkdir(exist_ok=False)
CELLS=[f'{m}_{s}_{n}' for m in ['pi05','groot'] for s in ['l10','sp'] for n in [50,500]]
eps=[];scores=[];inputs={}
for cell in CELLS:
    path=q.HERE/('pilot_'+cell)
    meta=json.loads((path/'run_manifest.json').read_text())
    assert meta['script_sha256']=='9a51ec7eb62378fedf8552aecb429253facdfc0c421e3d64c23bb68dc7a1de6d'
    for name,dest in [('episode_outcomes',eps),('episode_scores',scores)]:
        p=path/(name+'.csv');dest.append(pd.read_csv(p));inputs[str(p)]=q.sha(p)
ep=pd.concat(eps,ignore_index=True);em=pd.concat(scores,ignore_index=True)

def components(g,value):
    # Pilot balanced T tasks x two inits x three seed blocks.
    wide=g.pivot(index=['task_id','init'],columns='block',values=value).sort_index()
    if wide.shape!=(20,3) or wide.isna().any().any():return dict(reason='unbalanced/undefined endpoint; no component imputation')
    yy=wide.to_numpy().reshape(10,2,3)
    e=yy.mean(axis=2);t=e.mean(axis=1);grand=yy.mean()
    mse=np.sum((yy-e[:,:,None])**2)/40
    msi=3*np.sum((e-t[:,None])**2)/10
    mst=6*np.sum((t-grand)**2)/9
    vi=(msi-mse)/3;vt=(mst-msi)/6;ve=mse
    rho=vi/(vi+ve) if vi+ve>0 else np.nan
    return dict(mean=grand,observed_variance=float(yy.var(ddof=1)),raw_init_variance=vi,raw_task_variance=vt,
                seed_variance=ve,raw_icc=rho,init_variance=max(0,vi),task_variance=max(0,vt),
                clipping=bool(vi<0 or vt<0),df_task=9,df_init=10,df_seed=40)

def project(stats):
    if 'reason' in stats:return []
    out=[]
    for stage,rs in [('pilot_all',[3]*20),('pilot_validation',[3]*10),
                     ('full_all',[3]*100+[2]*50+[1]*100),('full_validation',[3]*80+[2]*40+[1]*80)]:
        K=len(rs);sigma=stats['init_variance']+stats['seed_variance']
        vv=(stats['init_variance']*K+stats['seed_variance']*sum(1/r for r in rs))/(K*K)
        for scope,var in [('fixed_tasks',vv),('task_superpopulation',vv+stats['task_variance']/10)]:
            row=dict(stage=stage,scope=scope,unique_inits=K,episodes=sum(rs),variance_of_mean=var,
                     half95=1.96*math.sqrt(var),half99=norm.ppf(.995)*math.sqrt(var),
                     half_state_sim=norm.ppf(1-.05/16)*math.sqrt(var),mde80=2.801621*math.sqrt(var),
                     n_eff=sigma/vv if vv>0 else np.nan)
            out.append(row)
    return out

sr=[];state=[]
for cell,cg in ep.groupby('cell'):
    for cohort in ['A','B','dose125','dose25','dose50','factorial','window','dose_mix']:
        for ref in ['P10','A']:
            if cohort==ref:continue
            one=cg[cg.cohort==cohort].merge(cg[cg.cohort==ref],on=['task_id','init','block'],suffixes=('','_ref'),validate='one_to_one')
            assert len(one)==60
            one['delta']=one.Y-one.Y_ref;s=components(one,'delta')
            for p in project(s):sr.append(dict(cell=cell,cohort=cohort,reference=ref,discordance=float((one.delta**2).mean()),**s,**p))
    a=em[(em.cell==cell)&(em.cohort=='A')].copy()
    for candidate in q.PRIMARY:
        for label in ['rho','loss_improvement']:
            a['target']=a['rho_'+candidate] if label=='rho' else a.baseline_mae-a['mae_'+candidate]
            s=components(a,'target')
            if 'reason' in s:state.append(dict(cell=cell,candidate=candidate,label=label,**s));continue
            for p in project(s):state.append(dict(cell=cell,candidate=candidate,label=label,**s,**p))
pd.DataFrame(sr).to_csv(OUT/'SR_projections.csv',index=False)
pd.DataFrame(state).to_csv(OUT/'state_projections.csv',index=False)
q.dump(OUT/'manifest.json',dict(inputs=inputs,estimation='Balanced nested task/init/seed ANOVA; df 9/10/40; paired outcomes before ICC',
    negative_components='Raw values retained. Negative variance components set to zero only for design projections.',
    caveats=['Pilot has only 2 inits/task; component estimates are imprecise.',
             'State loss projection includes calibration-init residuals; not a new held-out test or fit.',
             'Projection assumes stationary variance/effect and frozen coefficients; full recalibration may change both.',
             'More inits cannot remove between-task variation under the preregistered task-superpopulation intervals.',
             'Zero observed variance is not proof of zero future failures; no certification from degenerate plug-in widths.']))
print(json.dumps(dict(SR_projection_rows=len(sr),state_projection_rows=len(state))))
