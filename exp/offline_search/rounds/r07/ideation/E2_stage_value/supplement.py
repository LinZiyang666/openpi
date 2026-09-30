"""Additional descriptive failure localization and gripper-independent error."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
from analyze_stage import OUT,TMP,RUN,SEED,canon,source,finish_sources,boot_weights,histories

pf=histories(pd.read_pickle(TMP/'profile.pkl'))
if 'nongrip_rms' not in pf:
    adds=[]
    for (campaign,cell),f in pf.groupby(['campaign','cell']):
        cn=cell.replace('_spatial_','_sp_') if campaign=='r06_p3_pilot' else cell
        root=RUN/campaign/'tables'/cn
        acts=[]
        cols=['arm','uid','step','chunk_step','cache.6','policy.6','rms']
        for chunk in pd.read_csv(source(root/'action_steps.csv'),usecols=cols,chunksize=100000):
            chunk=chunk[chunk.arm.isin(f.arm.unique())&(chunk.chunk_step<10)]
            chunk['nongrip_mse']=(7*chunk.rms**2-(chunk['cache.6']-chunk['policy.6'])**2).clip(lower=0)/6
            acts.append(chunk.groupby(['arm','uid','step']).nongrip_mse.agg(['sum','count']))
        ag=pd.concat(acts).groupby(level=[0,1,2]).sum()
        ag['nongrip_rms']=np.sqrt(ag['sum']/ag['count'])
        adds.append(ag[['nongrip_rms']].reset_index())
    pf=pf.merge(pd.concat(adds),on=['arm','uid','step'],validate='1:1')
    pf.to_pickle(TMP/'profile.pkl')

sts=[]
for file in Path('exp/offline_search/rounds/r06/ideation_Q3/stall').glob('pilot_status_*.csv'):
    s=pd.read_csv(source(file),usecols=['arm','uid','step','state'])
    sts.append(s.rename(columns={'state':'stall'}))
d=pf[pf.campaign.eq('r06_p3_pilot')].merge(pd.concat(sts),on=['arm','uid','step'],validate='1:1')
rows=[]
for cell,c in d.groupby('cell'):
    last=c.sort_values('step').groupby('uid').tail(1)
    firststall=c[c.stall.eq('slow_confirmed')].sort_values('step').groupby('uid').head(1)
    for pop,g in [('failed_terminal',last[last.Y.eq(0)]),('successful_terminal',last[last.Y.eq(1)]),
                  ('failed_first_stall',firststall[firststall.Y.eq(0)]),('successful_first_stall',firststall[firststall.Y.eq(1)])]:
        for feature in ['phase','event','post_event','grip_state','milestone']:
            for label,n in g[feature].value_counts().items():
                rows.append(dict(cell=cell,population=pop,feature=feature,label=label,n=n,total=len(g),
                                 episodes=int(last.Y.eq(int(pop.startswith('successful'))).sum())))
pd.DataFrame(rows).to_csv(OUT/'failure_localization.csv',index=False)

# Failure-stage shares and uncertainty: fixed tasks, two distinct inits per cell,
# repeats clustered. These describe first detected stalls, not true failure onset.
stats=[]
for cell,c in d.groupby('cell'):
    last=c.sort_values('step').groupby('uid').tail(1)
    fst=c[c.stall.eq('slow_confirmed')].sort_values('step').groupby('uid').head(1)
    for outcome in [0,1]:
        e=last[last.Y.eq(outcome)].copy()
        e['detected']=e.uid.isin(fst.uid).astype(float)
        for metric,v in [('ever_confirmed_stall',e.detected),('terminal_Q4',e.phase.eq('Q4').astype(float))]:
            idx=(e.task*2+e.init).to_numpy(int)
            ar=np.stack([np.bincount(idx,weights=v,minlength=20),np.bincount(idx,minlength=20)],axis=1)
            b=boot_weights(2,seed=SEED+int('_spatial_' in cell))@ar
            with np.errstate(invalid='ignore',divide='ignore'):draw=b[:,0]/b[:,1]
            stats.append(dict(cell=cell,Y=outcome,metric=metric,episodes=len(e),estimate=v.mean(),
                              lo=np.nanquantile(draw,.025),hi=np.nanquantile(draw,.975)))
pd.DataFrame(stats).to_csv(OUT/'failure_episode_rates.csv',index=False)
stats=[]
for feature in ['state_resid','disp','coverage','pred_err','progress_spread']:
    means=[];draws=[]
    for cell,c in d.groupby('cell'):
        rr=[]
        for _,g in c.groupby('uid'):
            r=spearmanr(g[feature],g.rms,nan_policy='omit').statistic
            if np.isfinite(r):rr.append((int(g.task.iloc[0])*2+int(g.init.iloc[0]),r))
        idx=np.array([r[0] for r in rr]);vals=np.array([r[1] for r in rr])
        ar=np.stack([np.bincount(idx,weights=vals,minlength=20),np.bincount(idx,minlength=20)],axis=1)
        z=boot_weights(2,seed=SEED+int('_spatial_' in cell))@ar
        draw=z[:,0]/z[:,1];draws.append(draw);means.append(vals.mean())
        stats.append(dict(cell=cell,feature=feature,rho=means[-1],lo=np.quantile(draw,.025),hi=np.quantile(draw,.975)))
    draw=np.mean(draws,axis=0)
    stats.append(dict(cell='POOLED',feature=feature,rho=np.mean(means),lo=np.quantile(draw,.025),hi=np.quantile(draw,.975)))
pd.DataFrame(stats).to_csv(OUT/'correlation_uncertainty.csv',index=False)
finish_sources('supplement')
print('supplement done',flush=True)
