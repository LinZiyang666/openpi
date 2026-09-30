"""R7 E2 read-only exploratory analysis. No fitted serving policy is produced.

Run from repo root with the CPU/env prefix in run.sh. Intermediate row tables
live in /tmp/r7_E2_stage_value; summary artifacts live next to this script.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t, spearmanr

OUT = Path(__file__).resolve().parent
TMP = Path('/tmp/r7_E2_stage_value')
RUN = Path('/home/weiland/trace_runs/os_closed_loop')
STORE = Path('/home/weiland/trace_runs/offline_search_store/library')
FRONT = Path('exp/offline_search/rounds/r06/frontier_final')
SEED = 20260930
SOURCES = {}


def source(p):
    p = Path(p)
    s = p.stat()
    SOURCES[str(p)] = dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns)
    return p


def finish_sources(name):
    (OUT / f'sources_{name}.json').write_text(json.dumps(SOURCES, indent=2))


def canon(cell):
    return cell.replace('_sp_', '_spatial_')


def load_lib(cell):
    model, suite, size = canon(cell).split('_')
    if size == '50':
        name = 'current'
    else:
        pts = pd.read_csv(FRONT / 'frontier_points.csv')
        name = pts.loc[(pts.cell == canon(cell)) & (pts.library_nominal == 500), 'library'].mode().iloc[0]
    root = STORE / f'{model}_{suite}' / name
    z = {k: np.load(source(root / f'{k}.npy'), mmap_mode='r') for k in
         ['action', 'rs', 'episode', 'task_id', 'step', 'progress', 'next', 'prev']}
    z['path'] = root
    z['closed_sign'] = 1 if model == 'pi05' else -1  # Existing manifest/adapter semantics.
    z['scales'] = {int(task): np.maximum(np.std(z['rs'][z['task_id'] == task, :8], axis=0), 1e-8)
                   for task in np.unique(z['task_id'])}
    z['quartiles'] = {int(task): np.quantile(z['progress'][z['task_id'] == task], [.25,.5,.75])
                      for task in np.unique(z['task_id'])}
    # Two gripper modes, fitted to library state only. rs[6] is the first finger.
    v = np.asarray(z['rs'][:, 6], dtype=float)
    centers = np.quantile(v, [.1,.9])
    for _ in range(30):
        lab = abs(v[:, None] - centers).argmin(1)
        new = np.array([np.mean(v[lab == k]) for k in [0,1]])
        if np.allclose(centers, new):
            break
        centers = new
    # The initial-state modal cluster is called open-like, not proven object-free.
    starts = z['step'] == 0
    open_mode = np.bincount(lab[starts], minlength=2).argmax()
    z['grip_centers'], z['open_mode'] = centers, open_mode
    return z


def stage_features(z, task, rows, weights, rs, cache, previous_g):
    rows, weights = np.asarray(rows, dtype=int), np.asarray(weights, dtype=float)
    assert len(rows) == len(weights) and len(rows)
    weights = weights / weights.sum()
    prog = float(weights @ z['progress'][rows])
    rg = np.asarray(rs)[:8]
    state_resid = float(np.sqrt(np.mean(((rg - weights @ z['rs'][rows,:8]) / z['scales'][task]) ** 2)))
    mode = abs(rg[6] - z['grip_centers']).argmin()
    g = np.asarray(cache)[:10, 6] * z['closed_sign'] > 0
    if previous_g is not None and np.isfinite(previous_g) and previous_g != 0:
        g = np.r_[previous_g * z['closed_sign'] > 0, g]
    closing, opening = bool(np.any(~g[:-1] & g[1:])), bool(np.any(g[:-1] & ~g[1:]))
    event = ('mixed' if closing and opening else 'closing' if closing else
             'releasing' if opening else 'closed_hold' if g[-1] else 'open_hold')
    return dict(progress=prog, phase='Q'+str(1+np.searchsorted(z['quartiles'][task],prog)),
                state_resid=state_resid, grip_state='open_like' if mode == z['open_mode'] else 'closed_like',
                event=event, progress_spread=float(np.sqrt(weights @ (z['progress'][rows]-prog)**2)))


def histories(df):
    df = df.sort_values(['cell','arm','uid','step']).copy()
    g = df.groupby(['cell','arm','uid'], sort=False)
    df['progress_delta'] = g.progress.diff()
    df['nonadvance'] = np.where(df.progress_delta.isna(), 'start',
                                np.where(df.progress_delta <= 0, 'nonadvance', 'advance'))
    df['previous_event'] = g.event.shift()
    df['post_event'] = np.where(df.previous_event.eq('closing'), 'after_close',
                       np.where(df.previous_event.eq('releasing'), 'after_release','other'))
    if 'milestone' not in df:
        # Valid for cache-only profiles: previous proposed commit was executed.
        wasclose=df.event.isin(['closing','mixed','closed_hold'])
        wasrelease=df.event.isin(['releasing','mixed'])
        pastclose=wasclose.groupby([df.cell,df.arm,df.uid]).transform(lambda x:x.shift(fill_value=False).cummax())
        pastrelease=wasrelease.groupby([df.cell,df.arm,df.uid]).transform(lambda x:x.shift(fill_value=False).cummax())
        df['milestone']=np.where(pastrelease,'after_first_release',np.where(pastclose,'after_close_before_release','before_first_close'))
    return df


def extract_lottery():
    pts = pd.read_csv(source(FRONT / 'frontier_points.csv'))
    selected = pts[(pts.run.eq('r06_c_validation') & pts.arm.str.contains('_U(?:18|30)$')) |
                   (pts.run.eq('r06_frontier') & pts.arm.str.contains('_risk_rho'))]
    outcomes = json.loads(source(FRONT / 'outcomes.json').read_text())
    data, audits = [], []
    for cell, cellarms in selected.groupby('cell', sort=True):
        z = load_lib(cell)
        for armrow in cellarms.itertuples():
            root = RUN / armrow.run / 'runs' / armrow.arm
            accepted = {}
            for line in source(root / 'client/journal.jsonl').open():
                j = json.loads(line)
                if j.get('accepted') and j.get('status') in ['done','failed'] and not j.get('error'):
                    accepted[j['task_uid']] = j
            rows_by_ep = {}
            seen = set()
            dup = discarded = 0
            for file in sorted(root.glob('server_*/decisions_*.jsonl')):
                for line in source(file).open():
                    r = json.loads(line)
                    if r.get('ev') != 'dec' or r.get('uid') not in accepted:
                        continue
                    if r.get('attempt',1) != accepted[r['uid']].get('attempt',1):
                        continue
                    key = (r['uid'],r['step'])
                    dup += key in seen
                    seen.add(key)
                    rows_by_ep.setdefault(r['uid'],[]).append(r)
            local, mismatches, missing = [], 0, 0
            for uid in sorted(accepted):
                candidates = [r for r in rows_by_ep.get(uid,[]) if r['ts'] <= accepted[uid]['ts']]
                # Infrastructure repairs may reuse attempt=1. Select the last
                # full episode prefix beginning at step 0 before acceptance.
                starts = [r['ts'] for r in candidates if r['step'] == 0]
                ep = sorted([r for r in candidates if starts and r['ts'] >= max(starts)],key=lambda r:r['step'])
                discarded += len(rows_by_ep.get(uid,[])) - len(ep)
                if not ep:
                    missing += 1
                    continue
                assert [r['step'] for r in ep] == list(range(len(ep))), (armrow.arm,uid,'gap')
                previous_g = None
                closed_issued = release_issued = False
                for r in ep:
                    ex = r.get('extras',{})
                    if r.get('vision') and 'os_q2_p_call' in ex:
                        p = float(ex['os_q2_p_call'])
                        u = float(ex['os_q2_coin'])
                        a = int(r['src'] == 'policy')
                        mismatches += a != int(u < p)
                        rr, w = r['rows'],r['weights']
                        assert r['lib'] == z['path'].name, (r['lib'],z['path'])
                        cache = np.einsum('i,ijk->jk', np.asarray(w)/sum(w), z['action'][rr,:,:7])
                        task, init = int(r['task_id']),int(r['init'])
                        y = outcomes[armrow.id][f'{task}:{init}']
                        assert y == int(accepted[uid]['success'])
                        f = stage_features(z,task,rr,w,r['robot_state'],cache,previous_g)
                        milestone=('after_first_release' if release_issued else 'after_close_before_release' if closed_issued else 'before_first_close')
                        local.append(dict(cell=cell,arm=armrow.arm,run=armrow.run,uid=uid,task=task,init=init,milestone=milestone,
                                          step=r['step'],Y=y,Z=a,p=p,coin=u,lib=z['path'].name,
                                          d1=ex.get('d1',np.nan),disp=ex.get('disp5',np.nan),**f))
                    previous_g = r['served_head'][-1][6]
                    for a in r['served_head']:
                        if a[6]*z['closed_sign'] > 0:closed_issued=True
                        elif closed_issued:release_issued=True
            audits.append(dict(cell=cell,arm=armrow.arm,episodes=len(accepted),anchors=len(local),
                               duplicates=dup,discarded_prior_prefix=discarded,missing_episodes=missing,coin_mismatches=mismatches,
                               unsupported=sum(r['p'] in [0,1] for r in local)))
            print('audit',audits[-1],flush=True)
            assert mismatches == 0 and missing == 0, audits[-1]
            data.extend(local)
            print('lottery',cell,armrow.arm,len(local),flush=True)
    df = histories(pd.DataFrame(data))
    df.to_pickle(TMP / 'lottery.pkl')
    pd.DataFrame(audits).to_csv(OUT / 'lottery_audit.csv',index=False)
    finish_sources('lottery')


ANCHOR_COLS = ['Y','arm','uid','task_id','init','step','state.normalized','retrieval.rows','retrieval.weights',
              'distance.commit10_rms','distance.rms','retrieval.d1_loeo_quantile','retrieval.dispersion_rms',
              'guards.inputs_outputs.gexec','guards.inputs_outputs.noprog_n','guards.inputs_outputs.top1_prog',
              'guards.inputs_outputs.os_reason','guards.inputs_outputs.pred_err','retrieval.progress',
              'actual_commit_controls','guards.inputs_outputs.term1','calibration.p_values.progress',
              'calibration.p_values.stuck','assignment.cohort','assignment.actual_propensity',
              'assignment.executed_policy','assignment.replicate','resampling.selected',
              'resampling.dispersion_per_step']


def extract_profile():
    alld = []
    for campaign in ['r06_c_cal','r06_p3_pilot']:
        for root in sorted((RUN / campaign / 'tables').iterdir()):
            if not root.is_dir() or '.' in root.name:
                continue
            cell=canon(root.name)
            z=load_lib(cell)
            kept=[]
            for chunk in pd.read_csv(source(root/'anchors.csv'),usecols=lambda c:c in ANCHOR_COLS,chunksize=1000):
                # A alone supplies failure occupancy/shadow diagnostics without treatment mixing.
                chunk=chunk[chunk['assignment.cohort'].eq('A')]
                for r in chunk.to_dict('records'):
                    rr,w=json.loads(r['retrieval.rows']),json.loads(r['retrieval.weights'])
                    cache=np.einsum('i,ijk->jk', np.asarray(w)/sum(w),z['action'][rr,:,:7])
                    f=stage_features(z,int(r['task_id']),rr,w,json.loads(r['state.normalized']),cache,
                                     r['guards.inputs_outputs.gexec'])
                    kept.append(dict(cell=cell,campaign=campaign,arm=r['arm'],uid=r['uid'],task=int(r['task_id']),
                                     init=int(r['init']),step=int(r['step']),Y=int(r['Y']),
                                     rms=r['distance.commit10_rms'],disp=r['retrieval.dispersion_rms'],
                                     coverage=r['retrieval.d1_loeo_quantile'],pred_err=r['guards.inputs_outputs.pred_err'],
                                     noprog_n=r['guards.inputs_outputs.noprog_n'],reason=r['guards.inputs_outputs.os_reason'],
                                     terminal=r['guards.inputs_outputs.term1'],actual_commit=r['actual_commit_controls'],
                                     **f))
            df=pd.DataFrame(kept)
            # Exact normalized cache/policy chunks are separately flattened; compute gripper disagreement.
            acols=['arm','uid','step','chunk_step','cache.6','policy.6','actual_commit','rms']
            acts=[]
            for chunk in pd.read_csv(source(root/'action_steps.csv'),usecols=acols,chunksize=100000):
                chunk=chunk[chunk.arm.isin(df.arm.unique()) & (chunk.chunk_step < 10)]
                chunk['grip_disagree']=(chunk['cache.6'] > 0)!=(chunk['policy.6'] > 0)
                chunk['nongrip_mse']=(7*chunk.rms**2-(chunk['cache.6']-chunk['policy.6'])**2).clip(lower=0)/6
                acts.append(chunk.groupby(['arm','uid','step']).agg(sum=('grip_disagree','sum'),count=('grip_disagree','count'),nongrip_mse=('nongrip_mse','sum')))
            ag=pd.concat(acts).groupby(level=[0,1,2]).sum()
            ag['grip_disagree']=ag['sum']/ag['count']
            ag['nongrip_rms']=np.sqrt(ag.nongrip_mse/ag['count'])
            df=df.merge(ag[['grip_disagree','nongrip_rms']].reset_index(),on=['arm','uid','step'],validate='1:1')
            alld.append(df)
            print('profile',campaign,cell,len(df),flush=True)
    df=histories(pd.concat(alld,ignore_index=True))
    df.to_pickle(TMP/'profile.pkl')
    finish_sources('profile')


def boot_weights(ninit=50,B=2000,seed=SEED):
    rng=np.random.default_rng(seed)
    w=np.zeros((B,10*ninit),dtype=np.float64)
    for k in range(10):
        w[:,k*ninit:(k+1)*ninit]=rng.multinomial(ninit,np.full(ninit,1/ninit),size=B)
    return w


BW=boot_weights()
BWS={'l10':BW,'spatial':boot_weights(seed=SEED+1)}


def causal(group):
    """Hajek anchor-weighted excursion effect and episode-cluster bootstrap.

    We condition on current pre-treatment stratum and the episode's logged dose;
    future actions follow the source controller. This is not a whole-policy OPE.
    """
    g=group[(group.p > 0) & (group.p < 1)].copy()
    if g.empty:
        return None,None
    idx=(g.task*50+g.init).to_numpy(int)
    y,a,p=g.Y.to_numpy(),g.Z.to_numpy(),g.p.to_numpy()
    w1,w0=a/p,(1-a)/(1-p)
    arr=np.stack([np.bincount(idx,weights=w1*y,minlength=500),np.bincount(idx,weights=w1,minlength=500),
                  np.bincount(idx,weights=w0*y,minlength=500),np.bincount(idx,weights=w0,minlength=500)],axis=1)
    s=arr.sum(0)
    if s[1]==0 or s[3]==0:
        return None,None
    effect=s[0]/s[1]-s[2]/s[3]
    b=BWS[g.cell.iloc[0].split('_')[1]]@arr
    draws=b[:,0]/b[:,1]-b[:,2]/b[:,3]
    # Task-cluster sandwich sensitivity to transfer; t9 with only ten tasks.
    psi=(arr[:,0]-s[0]/s[1]*arr[:,1])/s[1]-(arr[:,2]-s[2]/s[3]*arr[:,3])/s[3]
    taskpsi=psi.reshape(10,50).sum(1)
    se=np.sqrt(10/9*np.sum(taskpsi**2))
    lo,hi=np.nanquantile(draws,[.025,.975])
    return dict(n=len(g),episodes=len(np.unique(idx)),calls=int(a.sum()),cache=int((1-a).sum()),
                effect=effect,lo=lo,hi=hi,task_lo=effect-t.ppf(.975,9)*se,task_hi=effect+t.ppf(.975,9)*se,
                mean_p=float(p.mean()),call_SR=s[0]/s[1],cache_SR=s[2]/s[3]),draws


def summarize_lottery():
    df=pd.read_pickle(TMP/'lottery.pkl')
    rows,boots=[],{}
    # State residual threshold comes from ten non-test B-val recordings per cell.
    pf=pd.read_pickle(TMP/'profile.pkl')
    cuts=pf[pf.campaign.eq('r06_c_cal')].groupby('cell').state_resid.quantile(.75).to_dict()
    df['state_deviation']=np.where(df.state_resid > df.cell.map(cuts),'Bval_top_quartile','Bval_lower_three_quartiles')
    (OUT/'state_calibration.json').write_text(json.dumps(cuts,indent=2))
    for dataset,d in [('U',df[df.run.eq('r06_c_validation')]),('risk',df[df.run.eq('r06_frontier')])]:
        for cell,c in d.groupby('cell'):
            for feature in ['all','phase','event','grip_state','nonadvance','post_event','state_deviation','milestone']:
                groups=[('all',c)] if feature=='all' else c.groupby(feature)
                for label,g in groups:
                    for sampling in ['all_anchors','first_entry']:
                        q=g if sampling=='all_anchors' else g.sort_values('step').drop_duplicates(['arm','uid'])
                        rec,draw=causal(q)
                        if rec:
                            key=(dataset,cell,feature,str(label),sampling)
                            boots[key]=draw
                            rows.append(dict(dataset=dataset,cell=cell,feature=feature,label=label,sampling=sampling,
                                             bootstrap_se=np.nanstd(draw,ddof=1),**rec))
        # Equal-cell pooling; same task/init bootstrap draw kept across cells/arms.
        for feature,label,sampling in sorted(set((k[2],k[3],k[4]) for k in boots if k[0]==dataset)):
            keys=[k for k in boots if k[0]==dataset and k[2:]==(feature,label,sampling)]
            if len(keys)!=8:
                continue
            draw=np.mean([boots[k] for k in keys],axis=0)
            rr=[r for r in rows if (r['dataset'],r['feature'],str(r['label']),r['sampling'])==(dataset,feature,label,sampling)
                and r['cell']!='POOLED']
            rows.append(dict(dataset=dataset,cell='POOLED',feature=feature,label=label,sampling=sampling,
                             n=sum(r['n'] for r in rr),episodes=sum(r['episodes'] for r in rr),
                             calls=sum(r['calls'] for r in rr),cache=sum(r['cache'] for r in rr),
                             bootstrap_se=np.nanstd(draw,ddof=1),
                             effect=np.mean([r['effect'] for r in rr]),lo=np.nanquantile(draw,.025),hi=np.nanquantile(draw,.975)))
    pd.DataFrame(rows).to_csv(OUT/'lottery_effects.csv',index=False)
    family=[]
    for sampling in ['all_anchors','first_entry']:
        for name,pooled,m in [('four_pooled_phase_tests',True,4),('32_cell_phase_tests',False,32)]:
            for r in rows:
                if r['dataset']=='U' and r['feature']=='phase' and r['sampling']==sampling and (r['cell']=='POOLED')==pooled:
                    z=norm.ppf(1-.05/(2*m))
                    family.append(dict(family=name,cell=r['cell'],label=r['label'],sampling=sampling,
                                       effect=r['effect'],lo=r['effect']-z*r['bootstrap_se'],hi=r['effect']+z*r['bootstrap_se']))
    pd.DataFrame(family).to_csv(OUT/'phase_family_adjusted.csv',index=False)
    # Prespecified readable contrast candidates, with joint clustered draws.
    contrasts=[]
    for dataset in ['U','risk']:
        for feature,a,b in [('event','closing','open_hold'),('event','releasing','open_hold'),
                             ('phase','Q4','Q1'),('nonadvance','nonadvance','advance'),
                             ('state_deviation','Bval_top_quartile','Bval_lower_three_quartiles')]:
            for sampling in ['all_anchors','first_entry']:
                ds=[];effects=[]
                for cell in sorted(df.cell.unique()):
                    ka,kb=(dataset,cell,feature,a,sampling),(dataset,cell,feature,b,sampling)
                    if ka in boots and kb in boots:
                        ds.append(boots[ka]-boots[kb])
                        ra=next(r for r in rows if (r['dataset'],r['cell'],r['feature'],str(r['label']),r['sampling'])==ka)
                        rb=next(r for r in rows if (r['dataset'],r['cell'],r['feature'],str(r['label']),r['sampling'])==kb)
                        effects.append(ra['effect']-rb['effect'])
                if len(ds)==8:
                    q=np.mean(ds,axis=0)
                    contrasts.append(dict(dataset=dataset,feature=feature,a=a,b=b,sampling=sampling,
                                          contrast=np.mean(effects),lo=np.quantile(q,.025),hi=np.quantile(q,.975)))
    pd.DataFrame(contrasts).to_csv(OUT/'lottery_interactions.csv',index=False)


def summarize_profile():
    df=histories(pd.read_pickle(TMP/'profile.pkl'))
    statuses=[]
    for f in Path('exp/offline_search/rounds/r06/ideation_Q3/stall').glob('pilot_status_*.csv'):
        st=pd.read_csv(source(f),usecols=['cell','arm','uid','step','state'])
        st['cell']=st.cell.map(canon)
        statuses.append(st.rename(columns={'state':'stall'}))
    df=df.merge(pd.concat(statuses),on=['cell','arm','uid','step'],how='left',validate='1:1')
    df['stall']=df.stall.fillna('not_replayed')
    cuts=df[df.campaign.eq('r06_c_cal')].groupby('cell').state_resid.quantile(.75).to_dict()
    df['state_deviation']=np.where(df.state_resid > df.cell.map(cuts),'Bval_top_quartile','Bval_lower_three_quartiles')
    rows=[]
    for (campaign,cell),c in df.groupby(['campaign','cell']):
        for feature in ['all','phase','event','grip_state','nonadvance','post_event','state_deviation','stall','milestone']:
            groups=[('all',c)] if feature=='all' else c.groupby(feature)
            for label,g in groups:
                # Descriptive episode-failure association is occupancy-weighted.
                rows.append(dict(campaign=campaign,cell=cell,feature=feature,label=label,n=len(g),
                                 episodes=g.uid.nunique(),failure_occupancy=(1-g.Y).mean(),
                                 rms=g.rms.mean(),nongrip_rms=g.nongrip_rms.mean(),grip_disagree=g.grip_disagree.mean(),
                                 state_resid=g.state_resid.mean(),share=len(g)/len(c)))
    pd.DataFrame(rows).to_csv(OUT/'profile_map.csv',index=False)
    # Episode-level means make each recorded rollout equally weighted.
    corr=[]
    for (campaign,cell),c in df.groupby(['campaign','cell']):
        for feature in ['state_resid','disp','coverage','pred_err','progress_spread','noprog_n']:
            within=[]
            for _,g in c.groupby('uid'):
                if len(g)>=5 and g[feature].nunique()>1:
                    within.append(spearmanr(g[feature],g.rms,nan_policy='omit').statistic)
            e=c.groupby(['task','init','uid']).agg(Y=('Y','first'),f=(feature,'mean'))
            corr.append(dict(campaign=campaign,cell=cell,feature=feature,
                             rho_disagreement=np.nanmean(within) if within else np.nan,
                             n_episodes=len(e),rho_episode_failure=spearmanr(e.f,1-e.Y).statistic))
    pd.DataFrame(corr).to_csv(OUT/'profile_correlations.csv',index=False)
    # Broad bins: episode-cluster bootstrap within cell, inits held together across seeds.
    tests=[]
    for feature,a,b in [('event','closing','open_hold'),('event','releasing','open_hold'),
                        ('phase','Q4','Q1'),('state_deviation','Bval_top_quartile','Bval_lower_three_quartiles'),
                        ('nonadvance','nonadvance','advance'),('stall','slow_confirmed','ok')]:
        for outcome in ['rms','nongrip_rms','grip_disagree','Y']:
            effs=[];draws=[]
            for cell,c in df[df.campaign.eq('r06_p3_pilot')].groupby('cell'):
                ag=[]
                for label in [a,b]:
                    g=c[c[feature].eq(label)]
                    idx=(g.task*2+g.init).to_numpy(int)
                    ag.append(np.stack([np.bincount(idx,weights=g[outcome],minlength=20),np.bincount(idx,minlength=20)],axis=1))
                arr=np.concatenate(ag,axis=1)
                s=arr.sum(0)
                eff=s[0]/s[1]-s[2]/s[3]
                z=boot_weights(ninit=2,seed=SEED+int('_spatial_' in cell))@arr
                draw=z[:,0]/z[:,1]-z[:,2]/z[:,3]
                effs.append(eff);draws.append(draw)
                tests.append(dict(cell=cell,feature=feature,a=a,b=b,outcome=outcome,effect=eff,
                                  lo=np.nanquantile(draw,.025),hi=np.nanquantile(draw,.975)))
            z=np.nanmean(draws,axis=0)
            tests.append(dict(cell='POOLED',feature=feature,a=a,b=b,outcome=outcome,effect=np.mean(effs),
                              lo=np.nanquantile(z,.025),hi=np.nanquantile(z,.975)))
    pd.DataFrame(tests).to_csv(OUT/'profile_contrasts.csv',index=False)
    finish_sources('profile_summary')


def paired():
    outcomes=json.loads(source(FRONT/'outcomes.json').read_text())
    ab=pd.read_csv(source(FRONT/'paper_AB_recomputed.csv'))
    pure=pd.read_csv(source(FRONT/'pure_references.csv'))
    results=[]
    for cell,g in ab.groupby('cell'):
        m,s,_=cell.split('_')
        p=pure[(pure.model==m)&(pure.suite==s)&(pure.L==10)].iloc[0]
        vals={}
        for label,runs in [('A',json.loads(g[g.label.eq('A')].iloc[0].runs)),
                           ('B',json.loads(g[g.label.eq('B')].iloc[0].runs)),('P10',json.loads(p.runs))]:
            vals[label]=np.array([[outcomes[r][f'{task}:{init}'] for task in range(10) for init in range(50)] for r in runs])
        for label in ['B','P10']:
            a,b=vals['A'],vals[label]
            if len(b)==1: b=np.repeat(b,3,axis=0)
            gap=(b-a).mean(0); benefit=((a==0)&(b==1)).mean(0);harm=((a==1)&(b==0)).mean(0)
            for metric,arr in [('gap',gap),('A_fail_other_success',benefit),('A_success_other_fail',harm)]:
                draw=BW@arr/500
                results.append(dict(cell=cell,reference=label,metric=metric,estimate=arr.mean(),
                                    lo=np.quantile(draw,.025),hi=np.quantile(draw,.975)))
    pd.DataFrame(results).to_csv(OUT/'paired_outcomes.csv',index=False)
    finish_sources('paired')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['lottery','profile','summary','paired']);args=p.parse_args()
    TMP.mkdir(exist_ok=True)
    t0=time.monotonic()
    if args.mode=='lottery':extract_lottery()
    elif args.mode=='profile':extract_profile()
    elif args.mode=='summary':summarize_lottery();summarize_profile()
    elif args.mode=='paired':paired()
    print('done',args.mode,round(time.monotonic()-t0,1),'seconds',flush=True)
