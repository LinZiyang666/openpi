"""Summarize discovery-only R8 E1 tables; no rollout or holdout reads."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path('/tmp/r8cb_E1_segmentation')
SCORES=['gripper_r7','elapsed_fixed']+[f'stop_q{q}_d{d}' for q in (10,25) for d in (1,5,10)]+[
    f'waypoint_x{x}' for x in (0.5,1.0,2.0)]+[f'change_x{x}' for x in (1.0,2.0,4.0)]+['event_mass']


def ap(y,score):
    order=np.argsort(-score,kind='stable');y=y[order];score=score[order]
    ends=np.r_[np.flatnonzero(score[:-1]!=score[1:])+1,len(score)]
    tp=np.cumsum(y)[ends-1]
    return float(np.sum(np.diff(np.r_[0.,tp/y.sum()])*tp/ends)) if y.sum() else np.nan


def top20(df,col):
    flag=np.zeros(len(df),bool)
    # Same tie coins for every candidate: no artificial candidate-specific lottery.
    tie=pd.util.hash_pandas_object(df[['suite','task_id','init','start']],index=False).to_numpy()
    for _,ix in df.groupby('task_id').indices.items():
        order=np.lexsort((tie[ix],df[col].to_numpy()[ix]))
        k=int(np.ceil(.2*len(ix)))
        flag[np.asarray(ix)[order[-k:]]]=True
    return flag


def cluster_ci(df,flag,base,draws=2000):
    z=pd.DataFrame(dict(task=df.task_id,init=df.init,p=df.bad.astype(int),
                       delta=(flag.astype(int)-base.astype(int))*df.bad.astype(int)))
    sums=z.groupby(['task','init'])[['p','delta']].sum()
    rng=np.random.default_rng(80801)
    den=np.zeros(draws);num=np.zeros(draws)
    for task in sorted(df.task_id.unique()):
        arr=np.array([sums.loc[(task,i)].to_numpy() if (task,i) in sums.index else [0,0] for i in range(30)])
        idx=rng.integers(0,30,(draws,30));v=arr[idx].sum(axis=1)
        den+=v[:,0];num+=v[:,1]
    v=num[den>0]/den[den>0]
    return np.quantile(v,[.025,.975]).tolist() if len(v) else [np.nan,np.nan]


def main():
    arms=sorted(p for p in OUT.glob('r8_*') if (p/'COMPLETE.json').exists())
    epi=[];risk=[];bq=[];dups=[];age=[];prefix=[];recon=[];audits=[];clusters=[]
    for path in arms:
        audit=json.loads((path/'COMPLETE.json').read_text());audits.append(audit)
        e=pd.read_parquet(path/'episodes.parquet');d=pd.read_parquet(path/'decisions.parquet')
        if len(e)==0: continue
        assert e.init.max()<30 and d.init.max()<30
        e['onset_available']=e.onset.notna()
        epi.append(e)
        b=pd.read_parquet(path/'boundary.parquet')
        for keys,g in b.groupby(['arm','model','suite','variant','candidate','baseline','tolerance_controls']):
            n=g.predicted_boundaries.sum();t=g.truth_boundaries.sum();m=g.matched_boundaries.sum()
            bq.append(dict(zip(['arm','model','suite','variant','candidate','baseline','tolerance'],keys),
                episodes=len(g),predicted=int(n),truth=int(t),matched=int(m),
                micro_f1=2*m/(n+t) if n+t else 1,macro_f1=g.f1.mean(),precision=m/n if n else np.nan,
                recall=m/t if t else np.nan))
        pre=pd.read_parquet(path/'prefix.parquet');prefix.extend(pre.to_dict('records'))
        rr=pd.read_parquet(path/'reconstruction.parquet')
        for keys,g in rr.groupby(['arm','candidate','resolution']):
            recon.append(dict(zip(['arm','candidate','resolution'],keys),episodes=len(g),
                median_knots=g.knots.median(),p95_translation_max=g.translation_max.quantile(.95),
                p95_rotation_max=g.SO3_rotation_max.quantile(.95)))
        d=d[d.src.isin(['cache','cache_tail','follow']) & d.at_risk & d.observed].copy()
        if not len(d): continue
        for definition,df in [('stock',d),('known_onset_or_success',d[d.known_negative])]:
            df=df[df.event_mass.notna()].reset_index(drop=True)
            if not len(df): continue
            flags={c:top20(df,c) for c in SCORES}
            y=df.bad.to_numpy(bool);base=flags['event_mass']
            for c,flag in flags.items():
                pos=int(y.sum());tp=int((y&flag).sum());n=len(df);nf=int(flag.sum())
                hit_eps=df.loc[y&flag,'episode_key'].nunique();onset_eps=df.loc[y,'episode_key'].nunique()
                ci=cluster_ci(df,flag,base) if pos else [np.nan,np.nan]
                r=dict(arm=path.name,model=df.model.iloc[0],suite=df.suite.iloc[0],variant=df.variant.iloc[0],
                    library_size=int(df.library_size.iloc[0]),definition=definition,candidate=c,
                    windows=n,positive_windows=pos,flagged=nf,true_positive=tp,exposure=nf/n,
                    sensitivity=tp/pos if pos else np.nan,precision=tp/nf,prevalence=pos/n,
                    auprc=ap(y,df[c].to_numpy()),rr=(tp/nf)/((pos-tp)/(n-nf)) if pos>tp else np.nan,
                    onset_episodes=onset_eps,hit_episodes=hit_eps,
                    actionable_hit_episodes=df.loc[y&flag&(df.lead>=5),'episode_key'].nunique(),
                    episode_sensitivity=hit_eps/onset_eps if onset_eps else np.nan,
                    median_lead=df.loc[y&flag,'lead'].median(),
                    delta_vs_catalog=tp/pos-(y&base).sum()/pos if pos else np.nan,
                    delta_ci_lo=ci[0],delta_ci_hi=ci[1])
                risk.append(r)
                z=df[['arm','model','suite','variant','library_size','task_id','init']].copy()
                z['p']=y.astype(int);z['tp']=(y&flag).astype(int);z['base_tp']=(y&base).astype(int)
                z['flag']=flag.astype(int);z['n']=1
                z=z.groupby(['arm','model','suite','variant','library_size','task_id','init']).sum().reset_index()
                z['candidate']=c;z['definition']=definition;clusters.append(z)
            for x,yname in [('stop_q10_d1','stop_q25_d1'),('stop_q10_d5','stop_q25_d5'),
                            ('stop_q10_d10','stop_q25_d10'),('change_x1.0','change_x2.0'),('change_x1.0','change_x4.0')]:
                dups.append(dict(arm=path.name,definition=definition,a=x,b=yname,
                                 unequal_flags=int(np.sum(flags[x]!=flags[yname])),windows=len(df)))
        for (stage,src,blind),g in d.groupby(['truth_stage_pre','src','blind_age']):
            age.append(dict(arm=path.name,stage=stage,src=src,blind_age=blind,windows=len(g),
                            positive_windows=int(g.bad.sum()),unknown_onset_windows=int((~g.known_negative).sum())))
    pd.concat(epi,ignore_index=True).to_parquet(OUT/'all_episodes.parquet',index=False)
    pd.concat(clusters,ignore_index=True).to_parquet(OUT/'risk_clusters.parquet',index=False)
    for name,rows in [('risk_summary',risk),('boundary_summary',bq),('identical_risk_rankings',dups),
                      ('age_summary',age),('prefix_summary',prefix),('reconstruction_summary',recon)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    eps=pd.concat(epi,ignore_index=True)
    brief=dict(arms=len(arms),episodes=len(eps),decisions=sum(a['decisions'] for a in audits),
        controls=sum(a['controls'] for a in audits),active_controls=sum(a['active_controls'] for a in audits),
        errors=sum(a['errors'] for a in audits),capture_error_episodes=sum(a['capture_error_episodes'] for a in audits),
        successes=int(eps.success.sum()),failures=int((~eps.success).sum()),
        failed_with_onset=int((~eps.success & eps.onset.notna()).sum()),
        prefix_checks=len(prefix),prefix_failures=sum(not r['passed'] for r in prefix),
        labels=eps.label.value_counts().to_dict(),holdout_episodes=0)
    (OUT/'summary.json').write_text(json.dumps(brief,indent=2)+'\n')
    print(json.dumps(brief,indent=2))


if __name__=='__main__': main()
