"""Same-observation diagnostics for the extra-LOOK question; not outcome effects."""
import json
import time
import numpy as np
import pandas as pd
from analyze_stage import OUT,TMP,RUN,SEED,source,finish_sources,canon,boot_weights

t0=time.monotonic()
pf=pd.read_pickle(TMP/'profile.pkl')
allrows=[]
for campaign in ['r06_c_cal','r06_p3_pilot']:
    for root in sorted((RUN/campaign/'tables').iterdir()):
        if not root.is_dir() or '.' in root.name:continue
        cell=canon(root.name)
        f=pf[(pf.campaign==campaign)&(pf.cell==cell)]
        names=['arm','uid','step','parent_anchor','absolute_input_archive','vision','actual_controls']
        dec=pd.read_csv(source(root/'decisions.csv'),usecols=names)
        dec=dec[dec.arm.isin(f.arm.unique()) & ~dec.vision.astype(bool)]
        # Parent state is known before the blind decision, never selected by later Y.
        dec=dec.merge(f[['arm','uid','step','task','init','Y','event','phase','nonadvance','state_resid']],
                      left_on=['arm','uid','parent_anchor'],right_on=['arm','uid','step'],suffixes=('','_anchor'),validate='1:1')
        rows=[]
        for r in dec.to_dict('records'):
            with np.load(source(r['absolute_input_archive'])) as z:
                executed=z['executed_chunk'][:5,:7].astype(float)
                fresh=z['diagnostic_cache_chunk'][:5,:7].astype(float)
                policy=z['policy_chunk'][:5,:7].astype(float)
                olderr=np.sqrt(np.mean((executed-policy)**2))
                newerr=np.sqrt(np.mean((fresh-policy)**2))
                change=np.sqrt(np.mean((executed-fresh)**2))
                grip=np.mean((executed[:,6]>0)!=(fresh[:,6]>0))
            rows.append({k:r[k] for k in ['arm','uid','task','init','step','Y','event','phase','nonadvance','state_resid']}|
                        dict(campaign=campaign,cell=cell,tail_rms=olderr,fresh_rms=newerr,improvement=olderr-newerr,
                             action_change=change,grip_change=grip,actual_controls=r['actual_controls']))
        allrows.extend(rows)
        print('look',campaign,cell,len(rows),flush=True)
df=pd.DataFrame(allrows)
df.to_pickle(TMP/'look.pkl')
summ=[]
for (campaign,cell),c in df.groupby(['campaign','cell']):
    for feature in ['all','event','phase','nonadvance']:
        for label,g in ([('all',c)] if feature=='all' else c.groupby(feature)):
            summ.append(dict(campaign=campaign,cell=cell,feature=feature,label=label,n=len(g),
                             tail_rms=g.tail_rms.mean(),fresh_rms=g.fresh_rms.mean(),improvement=g.improvement.mean(),
                             action_change=g.action_change.mean(),grip_change=g.grip_change.mean(),
                             fraction_closer=(g.improvement>0).mean()))
pd.DataFrame(summ).to_csv(OUT/'look_map.csv',index=False)
summ=[]
for feature in ['all','event','phase','nonadvance']:
    labels=['all'] if feature=='all' else sorted(df[feature].unique())
    for label in labels:
        effects=[];draws=[]
        for cell,c in df[df.campaign.eq('r06_p3_pilot')].groupby('cell'):
            g=c if feature=='all' else c[c[feature].eq(label)]
            idx=(g.task*2+g.init).to_numpy(int)
            arr=np.stack([np.bincount(idx,weights=g.improvement,minlength=20),np.bincount(idx,minlength=20)],axis=1)
            if arr[:,1].sum()==0:continue
            z=boot_weights(2,seed=SEED+int('_spatial_' in cell))@arr
            draw=z[:,0]/z[:,1]
            effects.append(g.improvement.mean());draws.append(draw)
            summ.append(dict(cell=cell,feature=feature,label=label,effect=effects[-1],
                             lo=np.nanquantile(draw,.025),hi=np.nanquantile(draw,.975)))
        if len(effects)==8:
            z=np.nanmean(draws,axis=0)
            summ.append(dict(cell='POOLED',feature=feature,label=label,effect=np.mean(effects),
                             lo=np.nanquantile(z,.025),hi=np.nanquantile(z,.975)))
pd.DataFrame(summ).to_csv(OUT/'look_uncertainty.csv',index=False)
finish_sources('look')
print('done look',round(time.monotonic()-t0,1),flush=True)
