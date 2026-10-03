"""Uncertainty, consistency, control-summary audit, and figures for E1."""
from __future__ import annotations
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_stages import CELLS, HERE, SCRATCH, cellname, libpath
from check_telemetry import auc


def cluster_rr(g, feature='hard', outcome='under_motion', iterations=2000):
    """Bootstrap task/init clusters (all three replicate blocks stay together)."""
    g=g[g.motion_comparable].copy()
    g['num_h']=g[feature].astype(int)*g[outcome].astype(int)
    g['den_h']=g[feature].astype(int)
    g['num_e']=(~g[feature]).astype(int)*g[outcome].astype(int)
    g['den_e']=(~g[feature]).astype(int)
    a=g.groupby(['task','init'])[['num_h','den_h','num_e','den_e']].sum().to_numpy()
    rng=np.random.default_rng(7)
    draws=a[rng.integers(len(a),size=(iterations,len(a)))].sum(axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        rr=draws[:,0]/draws[:,1]/(draws[:,2]/draws[:,3])
    rr=rr[np.isfinite(rr)]
    return [float(x) for x in np.quantile(rr,[.025,.975])] if len(rr) else [None,None]


def main():
    tasks=pd.read_csv(HERE/'library_tasks.csv')
    eps=pd.read_json(HERE/'library_episodes.json')
    consistency=[]
    for n in [50,500]:
        a=tasks[tasks.cell.str.startswith('pi05')&tasks.cell.str.endswith('_'+str(n))].copy(); a['suite']=a.cell.str.split('_').str[1]
        b=tasks[tasks.cell.str.startswith('groot')&tasks.cell.str.endswith('_'+str(n))].copy(); b['suite']=b.cell.str.split('_').str[1]
        j=a.merge(b,on=['suite','task'])
        consistency.append(dict(size=n,event_modes_agree=int((j.event_mode_x==j.event_mode_y).sum()),
                                task_pairs=len(j),median_stage_count_abs_diff=float(np.median(abs(j.median_stages_x-j.median_stages_y)))))
    pd.DataFrame(consistency).to_csv(HERE/'cross_model_consistency.csv',index=False)
    allrows=[]; eligibility=[]; outcome=[]; audits=[]; benchmarks=[]
    for m,s,n in CELLS:
        cell=cellname(m,s,n);lab=np.load(HERE/f'labels_{cell}.npz');good=lab['successful']
        assert np.all(np.isfinite(lab['difficulty'][good]))
        assert np.all(np.isnan(lab['difficulty'][~good]))
        span=lab['next_boundary'][good]; ev=eps[eps.cell==cell]
        eligibility.append(dict(cell=cell,span_at_least_15_controls=float(np.mean(span>=3)),
                                span_at_least_20_controls=float(np.mean(span>=4)),
                                median_slow_knots=float(ev.stops.median()),
                                ends_opposite_gripper_mode=float(np.mean(ev.events%2==1))))
        df=pd.read_csv(HERE/f'telemetry_{cell}.csv');allrows.append(df)
        g=df[(df.cohort=='p3')&(~df.pure_policy)].copy()
        c=g[g.motion_comparable]
        hard_rate=c[c.hard].under_motion.mean(); easy_rate=c[~c.hard].under_motion.mean()
        ee=g.groupby(['uid','task','init']).agg(Y=('Y','first'),hard_share=('hard','mean'),
                                               deviation=('deviation','mean'),events=('event_near','mean')).reset_index()
        onset=g[g.first_stall]
        # Descriptive exposure-normalized rates, plus within-task sign checks.
        task_rr=[]
        for task,cg in c.groupby('task'):
            hr=cg[cg.hard].under_motion.mean(); er=cg[~cg.hard].under_motion.mean()
            task_rr.append(dict(cell=cell,task=int(task),hard_under_motion=hr,easy_under_motion=er,
                                difference=hr-er,hard_anchors=int(cg.hard.sum()),easy_anchors=int((~cg.hard).sum())))
        pd.DataFrame(task_rr).to_csv(HERE/f'task_stalls_{cell}.csv',index=False)
        outcome.append(dict(cell=cell,rr=hard_rate/easy_rate,rr_ci=cluster_rr(g),
                            event_rr=float(c[c.event_near].under_motion.mean()/c[~c.event_near].under_motion.mean()),
                            event_rr_ci=cluster_rr(g,feature='event_near'),
                            event_anchor_share=float(g.event_near.mean()),
                            persistent_anchors=int(g.persistent_under_motion.sum()),
                            persistent_hard_share=float(g[g.persistent_under_motion].hard.mean()),
                            persistent_event_share=float(g[g.persistent_under_motion].event_near.mean()),
                            persistent_episodes=int(g[g.persistent_under_motion].uid.nunique()),
                            persistent_failed_episodes=int(g[g.persistent_under_motion&(g.Y==0)].uid.nunique()),
                            task_positive=int(sum(r['difference']>0 for r in task_rr)),
                            tasks_with_stalls=int(sum(r['hard_under_motion']+r['easy_under_motion']>0 for r in task_rr)),
                            hard_auc_stall=auc(c.under_motion,c.difficulty),
                            event_auc_stall=auc(c.under_motion,c.event_near),
                            tightness_auc_stall=auc(c.under_motion,c.tightness),
                            density_auc_stall=auc(c.under_motion,c.density),
                            deviation_auc_stall=auc(c.under_motion,c.deviation),
                            onset_hard_share=float(onset.hard.mean()),
                            onset_event_share=float(onset.event_near.mean()),
                            failure_episode_hard_share=float(ee[ee.Y==0].hard_share.mean()),
                            success_episode_hard_share=float(ee[ee.Y==1].hard_share.mean())))
        # Audit successor snapshots versus the next observed robot position:
        # controls are actual client transitions, not the unexecuted chunk tail.
        cache=pd.read_pickle(SCRATCH/f'anchors_p3_{cell}.pkl')
        errors=[];actual=[]
        for uid,ep in cache.groupby('uid'):
            ep=ep.sort_values('step').reset_index(drop=True)
            for i in range(len(ep)-1):
                if ep.step.iloc[i+1]-ep.step.iloc[i]!=2 or ep.actual_commit_controls.iloc[i]!=10:continue
                nxt=json.loads(ep['state.raw'].iloc[i+1])[:3]
                v=ep['successor_after_commit.observation_numeric.robot0_eef_pos'].iloc[i]
                if not isinstance(v,str): continue
                succ=json.loads(v)
                errors.append(np.linalg.norm(np.asarray(nxt)-succ))
                actual.append(ep.actual_commit_controls.iloc[i])
        audits.append(dict(cell=cell,actual_commit_to_next_anchor_pairs=len(errors),
                           max_raw_position_discrepancy=float(np.max(errors)),median_actual_controls=float(np.median(actual))))
        # Benchmark ordinary single-query nearest-neighbour lookup separately
        # from the batched query timing reported by check_telemetry.py.
        from scipy.spatial import cKDTree
        rs=np.load(libpath(m,s,n)/'rs.npy',mmap_mode='r')[:,:8]
        params=json.loads((HERE/f'calibration_{cell}.json').read_text())['task_params']
        times=[]
        for task in sorted(params):
            r=np.flatnonzero(good&(lab['task_id']==int(task)))
            sc=np.asarray(params[task]['scale']); mu=np.asarray(params[task]['center'])
            z=(rs[r]-mu)/sc;tree=cKDTree(z)
            for i in range(100):
                x=z[i%len(z)];t=time.perf_counter_ns();tree.query(x);times.append((time.perf_counter_ns()-t)/1e6)
        benchmarks.append(dict(cell=cell,nn_single_p50_ms=float(np.median(times)),nn_single_p95_ms=float(np.quantile(times,.95)),
                               label_file_bytes=(HERE/f'labels_{cell}.npz').stat().st_size))
    pd.DataFrame(eligibility).to_csv(HERE/'stage_eligibility.csv',index=False)
    pd.DataFrame(audits).to_csv(HERE/'control_successor_audit.csv',index=False)
    pd.DataFrame(benchmarks).to_csv(HERE/'cpu_cost.csv',index=False)
    safe_outcome=[{k:(None if isinstance(v,float) and not np.isfinite(v) else v)
                   for k,v in record.items()} for record in outcome]
    (HERE/'stage_outcome_evidence.json').write_text(json.dumps(safe_outcome,indent=2,allow_nan=False))
    df=pd.concat(allrows,ignore_index=True)
    for cohort in ['p3','cal']:
        g=df[(df.cohort==cohort)&(~df.pure_policy)]
        ep=g.groupby('uid').Y.first()
        print(cohort,'episodes',len(ep),'success',int(ep.sum()),'anchors',len(g),'hard',g.hard.mean(),
              'onsets',int(g.first_stall.sum()),'onset_hard',g[g.first_stall].hard.mean())
        first=g[g.first_stall].sort_values('step').groupby('uid').first()
        print('first-onset episodes',len(first),'failed episodes with onset',int((first.Y==0).sum()),
              'failed first onset event-near',first[first.Y==0].event_near.mean(),
              'success first onset event-near',first[first.Y==1].event_near.mean(),
              'event exposure',g.event_near.mean(),'all-onset event share',g[g.first_stall].event_near.mean())
    print(pd.DataFrame(outcome).round(3).to_string(index=False))
    # Scientific figure: results, plus a literal segmentation example.
    os.environ['MPLCONFIGDIR']=str(SCRATCH/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,1,figsize=(10,8),layout='constrained')
    for y,(m,s,n) in enumerate(CELLS):
        cell=cellname(m,s,n);o=next(x for x in outcome if x['cell']==cell)
        lo,hi=o['rr_ci'];rr=o['rr'];axes[0].errorbar(rr,y,xerr=[[max(0,rr-lo)],[max(0,hi-rr)]],fmt='o',color='#345f8d',capsize=3)
    axes[0].set_yticks(range(8),[cellname(*c) for c in CELLS]);axes[0].axvline(1,color='gray',ls='--');axes[0].invert_yaxis()
    axes[0].set_xlabel('Observed under-motion risk ratio: hard / other (95% task-init cluster bootstrap)')
    axes[0].set_title('Kinematic difficulty is not a consistent stall locator across cells')
    for y,m in enumerate(['pi05','groot']):
        cell=cellname(m,'l10',500);ep=eps[(eps.cell==cell)&(eps.task==0)].iloc[0]
        lab=np.load(HERE/f'labels_{cell}.npz');rr=np.flatnonzero(lab['episode']==ep.episode)
        x=np.arange(len(rr));axes[1].plot(x,lab['difficulty'][rr]+y,label=f'{m}, task 0, first successful demo')
        for k in ep.knots:axes[1].plot([k,k],[y,y+.95],color='gray',alpha=.22,lw=.7)
        for k in ep.events_at:axes[1].plot([k,k],[y,y+.95],color='#b63e45',lw=1.4)
    axes[1].set_xlabel('Library decision row (5 executed controls)');axes[1].set_ylabel('Difficulty score + model offset')
    axes[1].set_title('Red: gripper transition; gray: geometric/slow-motion boundary');axes[1].legend(fontsize=8)
    fig.savefig(HERE/'evidence.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
