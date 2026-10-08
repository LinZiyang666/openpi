"""Descriptive mechanism analysis; consumes only mechanism_extract.py's B cache.

Uses the same CPU/environment command as mechanism_extract.py. Statistical
predictors are explanatory cross-validation probes, never serving fits.
"""
from __future__ import annotations
from mechanism_extract import HERE, read, dump, sha, frozen, clean
import collections
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy import sparse, optimize, special, stats

COSTS={'groot':(.148,.852),'pi05':(.152,.848)}
SEED=20261004

def dist(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std()),
                **{f'p{int(q*100):02}':float(np.quantile(a,q)) for q in (0,.1,.25,.5,.75,.9,.95,1)}) if len(a) else dict(n=0)

def rate(x): return float(np.mean(x)) if len(x) else None

def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);q=np.isfinite(x)&np.isfinite(y);x=x[q];y=y[q]
    return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and x.std()>0 and y.std()>0 else None

def uniform(task,init,last):
    p=['Q2-deploy-v1','R11-gap-v1',0,int(task),int(init),int(last),'cap']
    b=hashlib.sha256(json.dumps(p,separators=(',',':'),ensure_ascii=True).encode()).digest()[:8]
    return (int.from_bytes(b,'big')>>11)*2.**-53

def agreement(a,b):
    a=np.asarray(a,bool);b=np.asarray(b,bool);inter=int((a&b).sum());union=int((a|b).sum());diff=int((a^b).sum())
    return dict(n=len(a),a=int(a.sum()),b=int(b.sum()),both=inter,a_only=int((a&~b).sum()),b_only=int((b&~a).sum()),
                neither=int((~a&~b).sum()),union=union,symmetric_difference=diff,
                symdiff_over_union=diff/union if union else None,jaccard=inter/union if union else None,
                agreement_all_positions=rate(a==b))

def replays(df,cfgs):
    results=[];errors=collections.Counter()
    for k in ['replay_p','replay_d','p_on_actual_history','d_on_actual_history']:df[k]=False
    df['periodic_min_gap']=np.nan;df['periodic_max_gap']=np.nan;df['replay_theta']=np.nan
    for arm,a in df.groupby('arm',sort=False):
        cell=a.cell.iloc[0];target=a.target.iloc[0]
        targets=[float(target)] if np.isfinite(target) else [.25,.32,.4,.45]
        for target in targets:
            pc=cfgs[(cell,'periodic',target)];dc=cfgs[(cell,'distance_feedback',target)]
            cv,cm=COSTS[a.model.iloc[0]];setting=pc['setting'];floor=math.floor(setting)
            ap=[];ad=[];actual=[];eligible=[];ipp=[];idd=[];thetas=[];ix=[]
            for ep,e in a.groupby('ep',sort=False):
                last=-1;run=V=M=0;last_actual=-1
                for r in e.itertuples():
                    cap=floor+int(uniform(r.task,r.init,last)<setting-floor)
                    p=bool(r.guard or run>=cap)
                    d=r.distance;finite=np.isfinite(d) and d>=0
                    u=float(np.clip((np.log1p(d)-dc['lo'])/(dc['hi']-dc['lo']),0,1)) if finite else np.nan
                    theta=dc['bias']+(cv*V+cm*M-target*r.step)/cm/dc['width']
                    dd=bool(r.guard or not finite or u>theta)
                    cap_act=floor+int(uniform(r.task,r.init,last_actual)<setting-floor)
                    pp_act=bool(r.guard or r.run>=cap_act)
                    theta_act=dc['bias']+(cv*r.V+cm*r.M-target*r.step)/cm/dc['width']
                    dd_act=bool(r.guard or not finite or u>theta_act)
                    if r.method=='periodic':
                        errors['periodic_own_positions']+=int(p!=r.call)
                        errors['periodic_cap']+=int(cap!=r.logged_cap)
                    if r.method=='distance_feedback': errors['feedback_own_positions']+=int(dd!=r.call)
                    ap.append(p);ad.append(dd);actual.append(r.call);eligible.append(not r.guard)
                    ipp.append(pp_act);idd.append(dd_act);thetas.append(theta);ix.append(r.Index)
                    if p:last=r.step;run=0
                    else:run+=1
                    if r.call:last_actual=r.step
                    V+=1;M+=dd
            ap=np.array(ap);ad=np.array(ad);eligible=np.array(eligible);actual=np.array(actual)
            results.append(dict(arm=arm,source=a.source.iloc[0],cell=cell,method=a.method.iloc[0],target=target,rep=int(a.rep.iloc[0]),
                   all_calls=agreement(ad,ap),knob_calls=agreement(ad[eligible],ap[eligible]),
                   actual_vs_periodic=agreement(actual[eligible],ap[eligible]),
                   actual_vs_feedback=agreement(actual[eligible],ad[eligible]),
                   same_history_decisions=agreement(np.array(idd)[eligible],np.array(ipp)[eligible])))
            if np.isfinite(a.target.iloc[0]) or target==.32:
                df.loc[ix,'replay_p']=ap;df.loc[ix,'replay_d']=ad
                df.loc[ix,'p_on_actual_history']=ipp;df.loc[ix,'d_on_actual_history']=idd
                df.loc[ix,'periodic_min_gap']=floor+1;df.loc[ix,'periodic_max_gap']=math.ceil(setting)+1
                df.loc[ix,'replay_theta']=thetas
    assert not any(errors.values()),errors
    return results,dict(errors)

def placement(a):
    e=a.groupby('ep',sort=False);c=a[a.call];k=a[a.knob];g=a[a.guard];eligible=a[~a.guard]
    last=e.tail(1).set_index('ep');last_call=c.groupby('ep',sort=False).tail(1).set_index('ep')
    censored_looks=last['look']-last_call['look']
    final_controls=last['final_controls'] if 'final_controls' in last else 5*last['N']
    censored_controls=final_controls-5*last_call['step']
    return dict(episodes=a.ep.nunique(),looks=len(a),calls=len(c),knobs=len(k),guards=len(g),guard_share_of_calls=len(g)/max(len(c),1),
        gap_any_call=dist(c.gap),gap_knob_since_any=dist(k.gap),gap_knob_since_knob=dist(k.gap_knob),
        controls_any_call=dist(c.gap_controls),controls_knob_since_any=dist(k.gap_controls),controls_knob_since_knob=dist(k.gap_knob_controls),
        first_call_controls=dist(c.groupby('ep',sort=False).controls.first()),
        episodes_without_call=int(a.ep.nunique()-c.ep.nunique()),
        startup_calls=int(c.startup.sum()),knob_after_guard_fraction=rate(k.last_call_kind=='guard'),
        knob_at_first_look_after_guard_fraction=rate((k.last_call_kind=='guard')&(k.gap==1)),
        guards_after_knob_fraction=rate(g.last_call_kind=='knob'),
        knob_shorter_than_periodic_min=int(((k.gap<k.periodic_min_gap)&~k.startup).sum()),
        knob_postfirst_count=int((~k.startup).sum()),
        cache_looks_beyond_periodic_max=int(((eligible.gap>=eligible.periodic_max_gap)&~eligible.call).sum()),
        eligible_cache_looks=int((~eligible.call).sum()),
        right_censored_terminal_gap=dist(censored_looks),
        right_censored_terminal_controls_since_last_call_start=dist(censored_controls),
        terminal_startup_looks_without_any_call=dist(last.loc[~last.index.isin(last_call.index),'look']+1),
        positions={name:{col:dist(z[col]) for col in ('controls','fraction','progress')}
                   for name,z in [('any',c),('knob',k),('guard',g)]})

def hazard(a):
    out=[]
    for gb,z in a.groupby('gap_bin',sort=True):
        eligible=z[~z.guard]
        out.append(dict(gap=gb,looks=len(z),eligible=len(eligible),knobs=int(eligible.knob.sum()),
                        knob_hazard=rate(eligible.knob),guard_hazard=rate(z.guard),all_call_hazard=rate(z.call),
                        mean_distance=float(z.distance.mean()),median_distance=float(z.distance.median()),
                        mean_log_distance=float(np.log1p(z.distance).mean())))
    return out

def position_hazard(a,col):
    out=[]
    bins=np.linspace(0,1,11) if col in ('fraction','progress') else np.arange(0,651,50)
    for left,right in zip(bins[:-1],bins[1:]):
        z=a[(a[col]>=left)&(a[col]<right if right!=1 else a[col]<=right)];e=z[~z.guard]
        out.append(dict(left=left,right=right,looks=len(z),knobs=int(z.knob.sum()),guards=int(z.guard.sum()),
                        eligible=len(e),knob_hazard=rate(e.knob)))
    return out

def matched_positions(df):
    out=[]
    for (source,cell,target,rep),a in df[df.method.isin(['periodic','distance_feedback'])].groupby(['source','cell','target','rep']):
        if set(a.method)!={'periodic','distance_feedback'}:continue
        for kind in ['call','knob']:
            metrics=[];total_union=total_diff=0
            for (task,init),e in a.groupby(['task','init']):
                d=e[(e.method=='distance_feedback')&e[kind]];p=e[(e.method=='periodic')&e[kind]]
                Nd=int(e[e.method=='distance_feedback'].N.iloc[0]);Np=int(e[e.method=='periodic'].N.iloc[0]);end=min(Nd,Np)
                ds=set(d.step[d.step<end]);ps=set(p.step[p.step<end]);union=len(ds|ps);diff=len(ds^ps)
                total_union+=union;total_diff+=diff
                v=dict(task=task,init=init,shared_horizon_controls=5*end,union=union,symdiff=diff,
                       symdiff_over_union=diff/union if union else np.nan)
                for col in ['controls','fraction','progress']:
                    if len(d) and len(p):v[col+'_wasserstein']=float(stats.wasserstein_distance(d[col],p[col]))
                if ds and ps:
                    da=np.array(sorted(ds));pa=np.array(sorted(ps))
                    near=np.r_[np.abs(da[:,None]-pa).min(1),np.abs(pa[:,None]-da).min(1)]
                    v['nearest_controls_mean']=float(5*near.mean());v['within_10_controls']=float((near<=2).mean())
                metrics.append(v)
            out.append(dict(source=source,cell=cell,target=target,rep=int(rep),kind=kind,matched_inits=len(metrics),
                pooled_symdiff_over_union=total_diff/total_union if total_union else None,
                distributions={k:dist([r.get(k,np.nan) for r in metrics]) for k in
                    ['symdiff_over_union','controls_wasserstein','fraction_wasserstein','progress_wasserstein','nearest_controls_mean','within_10_controls']}))
    return out

def within_distance(a):
    # Within-episode demeaning: nonlinear gap categories are compared against a
    # fixed episode mean. This descriptive R2 is in-sample and deliberately
    # distinguished from held-out-identity predictive R2 below.
    z=a[(~a.startup)&np.isfinite(a.distance)].copy()
    y=z.distance.to_numpy();ly=np.log1p(y);ep=z.ep
    design=pd.get_dummies(z.gap_bin,dtype=float)
    X=(design-design.groupby(ep,sort=False).transform('mean')).to_numpy()
    metrics={}
    for name,v in [('raw',y),('log1p',ly)]:
        vy=v-pd.Series(v,index=z.index).groupby(ep,sort=False).transform('mean').to_numpy()
        coef=np.linalg.lstsq(X,vy,rcond=1e-9)[0];res=vy-X@coef
        metrics[name+'_within_episode_gap_R2']=float(1-res@res/(vy@vy)) if vy@vy else None
        metrics[name+'_within_episode_gap_correlation']=correlation(z.gap-pd.Series(z.gap).groupby(ep).transform('mean'),vy)
    corr=[];slopes=[]
    for _,e in z.groupby('ep',sort=False):
        if len(e)>=5 and e.gap.nunique()>1 and e.distance.nunique()>1:
            corr.append(float(stats.spearmanr(e.gap,e.distance).statistic))
            slopes.append(float(np.polyfit(e.gap,np.log1p(e.distance),1)[0]))
    # Within an uninterrupted run, changes cannot be an artefact of averaging
    # different episode distance scales across gap bins.
    cont=z[z.gap>1];after=z[z.gap==1]
    metrics.update(looks=len(z),episode_spearman=dist(corr),episode_log_slope=dist(slopes),
        uninterrupted_run_distance_change=dist(cont.distance_delta),
        uninterrupted_run_fraction_distance_increases=rate(cont.distance_delta>0),
        next_look_after_policy_distance_change=dist(after.distance_delta),
        next_look_after_policy_fraction_distance_decreases=rate(after.distance_delta<0),
        by_gap=hazard(a))
    return metrics

def logistic(X,y):
    # A fixed weak penalty is a numerical regularizer, not controller tuning.
    def fun(w):
        z=X@w;p=special.expit(z)
        loss=np.logaddexp(0,z).sum()-y@z+.5*.1*(w@w)
        grad=np.asarray(X.T@(p-y)).ravel()+.1*w
        return loss,grad
    opt=optimize.minimize(fun,np.zeros(X.shape[1]),method='L-BFGS-B',jac=True,options={'maxiter':500,'ftol':1e-10})
    assert opt.success,opt.message
    return opt.x

def score_metrics(y,p):
    p=np.clip(p,1e-8,1-1e-8);y=np.asarray(y,float);r=stats.rankdata(p)
    n1=y.sum();n0=len(y)-n1
    auc=(r[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan
    return dict(n=len(y),prevalence=float(y.mean()),auc=float(auc),brier=float(np.mean((p-y)**2)),
                logloss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),accuracy=float(((p>=.5)==y).mean()))

def prediction_probes(a):
    a=a[(a.method=='distance_feedback')&~a.guard&np.isfinite(a.distance)].copy().reset_index(drop=True)
    # All repeats/doses of an init go to the same fold. No random look split.
    a['fold']=[int.from_bytes(hashlib.sha256(f'{c}:{t}:{i}'.encode()).digest()[:8],'big')%5
               for c,t,i in zip(a.cell,a.task,a.init)]
    a['cg']=a.cell+':'+a.target.astype(str)
    a['stratum']=a.cg+':'+a.gap_bin
    groups=pd.get_dummies(a.stratum,dtype=float);B=sparse.csr_matrix(groups.to_numpy())
    cells=pd.get_dummies(a.cg,dtype=float);C=sparse.csr_matrix(cells.to_numpy())
    y=a.knob.to_numpy(float);d=np.log1p(a.distance.to_numpy())
    preds={m:np.empty(len(a)) for m in ['gap','gap_distance_residual','gap_ledger','gap_ledger_distance_residual']}
    d_pred=np.empty(len(a));d_base=np.empty(len(a))
    for fold in range(5):
        tr=(a.fold!=fold).to_numpy();te=~tr
        train=a[tr].assign(logd=d[tr]);means=train.groupby('stratum').logd.mean();fallback=train.groupby('cg').logd.mean()
        pred=a.stratum.map(means).fillna(a.cg.map(fallback)).to_numpy();scale=np.std(d[tr]-pred[tr])
        dr=(d-pred)/max(scale,1e-6)
        d_pred[te]=pred[te];d_base[te]=a.cg.map(fallback).to_numpy()[te]
        extra_d=sparse.hstack([C.multiply(np.clip(dr,-5,5)[:,None]),C.multiply((np.clip(dr,-5,5)**2)[:,None])],format='csr')
        # Ledger is the actual predecision threshold. It is available online,
        # unlike episode identity/final length; included to expose the clock
        # inherent in the budget feedback separately from state distance.
        theta=a.theta.to_numpy();theta=(theta-np.mean(theta[tr]))/np.std(theta[tr])
        extra_l=C.multiply(theta[:,None]).tocsr()
        for name,X in [('gap',B),('gap_distance_residual',sparse.hstack([B,extra_d],format='csr')),
                       ('gap_ledger',sparse.hstack([B,extra_l],format='csr')),
                       ('gap_ledger_distance_residual',sparse.hstack([B,extra_l,extra_d],format='csr'))]:
            w=logistic(X[tr],y[tr]);preds[name][te]=special.expit(X[te]@w)
    residual=d-d_pred
    out=dict(looks=len(a),episodes=a.ep.nunique(),folds=5,
        heldout_log_distance_gap_R2_vs_cell_dose_mean=float(1-np.sum((d-d_pred)**2)/np.sum((d-d_base)**2)),
        models={m:score_metrics(y,p) for m,p in preds.items()},by_cell_dose=[])
    for (cell,target),z in a.groupby(['cell','target']):
        ix=z.index.to_numpy()
        out['by_cell_dose'].append(dict(cell=cell,target=target,n=len(z),
            models={m:score_metrics(y[ix],p[ix]) for m,p in preds.items()},
            heldout_log_distance_gap_R2=float(1-np.sum((d[ix]-d_pred[ix])**2)/np.sum((d[ix]-d_base[ix])**2))))
    yc=y-pd.Series(y).groupby(a.stratum).transform('mean').to_numpy()
    out['partial_distance_call_correlation_given_cell_dose_gap']=correlation(residual,yc)
    return out

def shadow_groups(a):
    a=a[~a.guard&np.isfinite(a.risk10)].copy()
    out=[]
    for (source,cell,method),z in a.groupby(['source','cell','method']):
        z=z.copy();z['selection']=np.select([z.replay_d&z.replay_p,z.replay_d&~z.replay_p,~z.replay_d&z.replay_p],
                                            ['both','distance_only','periodic_only'],default='neither')
        for group,e in z.groupby('selection'):
            out.append(dict(source=source,cell=cell,behavior=method,selection=group,n=len(e),
                risk5=dist(e.risk5),risk10=dist(e.risk10),grip10=dist(e.grip10),gap=dist(e.gap),
                by_gap=[dict(gap=gap,n=len(q),risk5_mean=float(q.risk5.mean()),risk10_mean=float(q.risk10.mean()),
                             grip10_mean=float(q.grip10.mean())) for gap,q in e.groupby('gap_bin')]))
    return out

def conditional_shadow(a,metric='risk10',boot=1000):
    # Saturated fixed effects of cell, behavior trajectory, task, and observed
    # exact gap (>=12 pooled, startup separate). This conditions on common
    # support, rather than equating incomparable unconditional errors.
    z=a[(~a.guard)&np.isfinite(a[metric])&(a.replay_d!=a.replay_p)].copy()
    if len(z)==0:return dict(n=0)
    z['stratum']=z.cell+':'+z.method+':'+z.task.astype(str)+':'+z.gap_bin
    z['cluster']=z.cell.str.split('_').str[1]+':'+z.task.astype(str)+':'+z.init.astype(str)
    # The same physical init across BOTH models and both behavior paths remains
    # paired. Task IDs in different suites represent different environments.
    z['choice']=z.replay_d.astype(int)
    groups=sorted(z.stratum.unique());clusters=sorted(z.cluster.unique())
    gi={g:i for i,g in enumerate(groups)};ci={c:i for i,c in enumerate(clusters)}
    n0=np.zeros((len(clusters),len(groups)));n1=n0.copy();s0=n0.copy();s1=n0.copy()
    for r in z.itertuples():
        c=ci[r.cluster];g=gi[r.stratum]
        if r.choice:n1[c,g]+=1;s1[c,g]+=getattr(r,metric)
        else:n0[c,g]+=1;s0[c,g]+=getattr(r,metric)
    def effect(w):
        a0=w@n0;a1=w@n1;b0=w@s0;b1=w@s1
        good=(a0>0)&(a1>0);weight=np.where(good,a0*a1/np.maximum(a0+a1,1),0)
        e0=np.divide(b0,a0,out=np.zeros_like(b0),where=a0>0);e1=np.divide(b1,a1,out=np.zeros_like(b1),where=a1>0)
        den=weight.sum(axis=-1);m0=(weight*e0).sum(axis=-1)/den;m1=(weight*e1).sum(axis=-1)/den
        return m1-m0,m1,m0,good
    delta,md,mp,good=effect(np.ones(len(clusters)))
    support=z.stratum.isin(np.array(groups)[good]);rng=np.random.default_rng(SEED)
    cluster_strata=collections.defaultdict(list)
    for c,i in ci.items():cluster_strata[':'.join(c.split(':')[:-1])].append(i)
    vals=[]
    for start in range(0,boot,100):
        w=np.zeros((min(100,boot-start),len(clusters)))
        for ids in cluster_strata.values():
            draws=rng.choice(ids,(len(w),len(ids)),replace=True)
            for j in range(len(w)):w[j]+=np.bincount(draws[j],minlength=len(clusters))
        vals.extend(effect(w)[0].tolist())
    return dict(n=len(z),common_support_n=int(support.sum()),common_support_strata=int(good.sum()),
        all_strata=len(groups),clusters=len(clusters),metric=metric,
        distance_only_standardized_mean=float(md),periodic_only_standardized_mean=float(mp),
        distance_minus_periodic=float(delta),ratio=float(md/mp),
        bootstrap_replicates=boot,bootstrap_95ci=np.quantile(vals,[.025,.975]).tolist())

def main():
    before=frozen();df=pd.read_pickle(HERE/'mechanism_looks.pkl')
    inputs=read(HERE/'mechanism_inputs.json')
    cfgs={(c['cell'],c['method'],float(c['target'])):c for c in read(HERE/'configs.json')}
    df['gap_bin']=np.where(df.startup,'startup',df.gap.fillna(0).clip(upper=12).astype(int).astype(str))
    replay,checks=replays(df,cfgs);print('replay passed',checks,flush=True)
    arms=[]
    for arm,a in df.groupby('arm',sort=False):
        arms.append(dict(arm=arm,source=a.source.iloc[0],cell=a.cell.iloc[0],method=a.method.iloc[0],target=a.target.iloc[0],rep=int(a.rep.iloc[0]),
            placement=placement(a),gap_hazard=hazard(a),position_hazard={c:position_hazard(a,c) for c in ['controls','fraction','progress']}))
    summaries=[];distance=[]
    for (source,model,method,target),a in df.groupby(['source','model','method','target'],dropna=False):
        summaries.append(dict(source=source,model=model,method=method,target=target,placement=placement(a),gap_hazard=hazard(a)))
    for (source,cell,method),a in df.groupby(['source','cell','method']):
        distance.append(dict(source=source,cell=cell,method=method,**within_distance(a)))
    print('placement and within-episode distance complete',flush=True)
    probes=prediction_probes(df[df.source=='pilot']);print('prediction probes complete',flush=True)
    labels=df[(df.source=='pilot')&(df.target==.32)&(df.rep==0)&df.method.isin(['periodic','distance_feedback'])&np.isfinite(df.risk10)]
    shadows=shadow_groups(labels)
    conditional=dict(pooled={m:conditional_shadow(labels,m) for m in ['risk5','risk10','grip10']},
                     by_cell=[dict(cell=cell,**conditional_shadow(a)) for cell,a in labels.groupby('cell')],
                     by_behavior=[dict(behavior=method,**conditional_shadow(a)) for method,a in labels.groupby('method')])
    primary=df[(df.source=='pilot')&df.method.isin(['periodic','distance_feedback'])]
    pooled_replay=[]
    for (model,target,method),a in primary.groupby(['model','target','method']):
        e=a[~a.guard]
        pooled_replay.append(dict(model=model,target=target,behavior=method,all_calls=agreement(a.replay_d,a.replay_p),
                                  knob_calls=agreement(e.replay_d,e.replay_p),same_history=agreement(e.d_on_actual_history,e.p_on_actual_history)))
    out=dict(schema='r13.B_mechanism.v1',seed=SEED,boundary='B only; all pilot arms; completed devrec arms; descriptive only.',
        inventory=inputs['inventory'],input_manifest_sha=sha(HERE/'mechanism_inputs.json'),look_cache_sha=sha(HERE/'mechanism_looks.pkl'),
        scripts={p.name:sha(p) for p in [HERE/'mechanism_extract.py',HERE/'mechanism_analyze.py']},
        audit=dict(own_rule_replay_mismatches=checks,frozen_files=len(before),frozen_unchanged=before==frozen()),
        counts=dict(arms=df.arm.nunique(),episodes=df.ep.nunique(),looks=len(df),calls=int(df.call.sum()),
                    pilot_arms=int(df[df.source=='pilot'].arm.nunique()),pilot_episodes=int(df[df.source=='pilot'].ep.nunique()),
                    shadow_pilot_looks=len(labels),shadow_pilot_eligible=int((~labels.guard).sum())),
        arms=arms,pooled_placement=summaries,distance_growth=distance,predictors=probes,
        same_state_replay=replay,pooled_replay=pooled_replay,matched_init_positions=matched_positions(df),
        shadow_groups=shadows,conditional_shadow=conditional,
        definitions=dict(gap='Number of real looks after previous actual policy call; next real look = 1. Startup kept separate.',
            gap_bin='Exact gap through 11, 12 includes >=12; startup is its own level.',
            controls='Five robot controls per logged decision slot; elapsed excludes environment warm-up.',
            final_fraction='step/N, retrospective position; not a controller input or survival-adjusted statistic.',
            progress='Logged top1_prog: retrieved demonstration normalized progress; donor can switch.',
            replay='Fixed actual states/look/guard sequence. Independent alternate call history, keyed periodic coins, feedback V/M ledger. Not a counterfactual rollout.',
            shadow='Mean squared policy minus corrected cache motion, divided by frozen per-channel sigma squared; first 5 and first 10 controls. Grip sign disagreement separately.',
            conditional_shadow='Distance-only versus periodic-only replay looks. Cell x behavior x task x observed-gap fixed effects on common support; harmonic count weights. 1000 paired-init cluster bootstraps within suite/task, pairing both models and both behavior paths.',
            predictors='Five folds grouped by cell/task/init across repeats/doses. Cell-dose x gap intercepts, optional cross-fitted log-distance residual and actual predecision ledger threshold; logistic probes, fixed L2=0.1.',
            within_episode_R2='Descriptive in-sample R2 of demeaned raw/log distance regressed on demeaned gap categories; startup removed.'))
    assert out['audit']['frozen_unchanged']
    dump(HERE/'mechanism_numbers.json',out)
    df.to_pickle(HERE/'mechanism_replayed.pkl')
    print('WROTE mechanism_numbers.json',flush=True)

if __name__=='__main__':main()
