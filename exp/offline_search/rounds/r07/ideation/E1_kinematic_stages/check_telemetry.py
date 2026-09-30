"""Evaluate frozen E1 library stage labels on accepted P3 A/P10 recordings.

Reads only selected proprioceptive/control-summary columns. Success is used for
evaluation, never segmentation, thresholds, alignment or score fitting.
"""
from __future__ import annotations
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from analyze_stages import CELLS, HERE, SCRATCH, cellname, libpath

P3 = Path('/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables')
CAL = Path('/home/weiland/trace_runs/os_closed_loop/r06_c_cal/tables')
COLS = ['uid','attempt','arm','step','task_id','init','Y','state.normalized','state.raw',
        'retrieval.rows','retrieval.weights','actual_commit_controls','assignment.realized_source',
        'guards.inputs_outputs.noprog_n','guards.inputs_outputs.os_flags',
        'successor_after_head.observation_numeric.robot0_eef_pos',
        'successor_after_commit.observation_numeric.robot0_eef_pos',
        'successor_after_commit.observation_numeric.robot0_gripper_qpos']


def read_selected(path, cell, cohort):
    target = SCRATCH/f'anchors_{cohort}_{cell}.pkl'
    if target.exists(): return pd.read_pickle(target)
    df = pd.read_csv(path/'anchors.csv', usecols=lambda c:c in COLS)
    if cohort == 'p3':
        df = df[df.arm.str.contains(r'_(?:A|P10)_r\d+$',regex=True)].copy()
    df.to_pickle(target)
    return df


def auc(y, x):
    y, x = np.asarray(y).astype(bool), np.asarray(x)
    mask = np.isfinite(x); y,x = y[mask],x[mask]
    if y.sum()==0 or (~y).sum()==0: return np.nan
    return float((rankdata(x)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))


def fit_motion_reference(rs, labels, params):
    """Task thresholds: q05 of same-state LOEO observed/expected displacement.

    Position-only avoids orientation branch cuts. Movement over two stored
    decision intervals matches the A/P10 10-control commit. No test recordings.
    """
    refs = {}
    for task in np.unique(labels['task_id']):
        par = params[str(task)]
        scale = np.asarray(par['scale'])
        center = np.asarray(par['center'])
        rows = np.flatnonzero((labels['task_id']==task)&labels['successful'])
        zz = (rs[rows]-center)/scale
        tree = cKDTree(zz)
        delta = np.full(len(rs), np.nan)
        for ep in np.unique(labels['episode'][rows]):
            rr = rows[labels['episode'][rows]==ep]
            delta[rr[:-2]] = np.linalg.norm((rs[rr[2:],:3]-rs[rr[:-2],:3])/scale[:3],axis=1)
        # LOEO candidates are the nearest state in another successful episode.
        dd, nn = tree.query(zz, k=min(len(rows),max(32, int(np.max(np.bincount(labels['episode'][rows])))+1)))
        peers = rows[nn]
        valid = (labels['episode'][peers]!=labels['episode'][rows,None]) & np.isfinite(delta[peers])
        anyvalid = valid.any(axis=1)
        pick = peers[np.arange(len(rows)),valid.argmax(axis=1)]
        ok = anyvalid & np.isfinite(delta[rows])
        floor = np.finfo(float).eps
        ratios = delta[rows[ok]]/np.maximum(delta[pick[ok]],floor)
        ratio_low=float(np.quantile(ratios,.05))
        # Persistent under-motion: exceed the successful-library q95 maximum
        # run length. Replay both possible anchor parities at the same stride 2.
        lows=np.zeros(len(rows),dtype=bool);lows[ok]=ratios<ratio_low
        maxima=[]
        for ep in np.unique(labels['episode'][rows]):
            ei=np.flatnonzero(labels['episode'][rows]==ep)
            mm=0
            for parity in [0,1]:
                run=0
                for v in lows[ei[parity::2]]:
                    run=run+1 if v else 0;mm=max(mm,run)
            maxima.append(mm)
        run_limit=int(np.quantile(maxima,.95,method='higher'))
        refs[int(task)] = dict(tree=tree, rows=rows, scale=scale, center=center, delta=delta,
                               ratio_low=ratio_low,run_limit=run_limit,
                               hard_threshold=float(np.quantile(labels['difficulty'][rows],.75)),
                               deviation_high=float(np.quantile(dd[np.arange(len(rows)),valid.argmax(axis=1)][anyvalid],.95)),
                               ratios_n=len(ratios))
    return refs


def analyze_cell(m, s, n):
    tick=time.perf_counter()
    cell=cellname(m,s,n)
    lab=np.load(HERE/f'labels_{cell}.npz')
    p=libpath(m,s,n)
    rs=np.asarray(np.load(p/'rs.npy',mmap_mode='r')[:,:8],dtype=float)
    params=json.loads((HERE/f'calibration_{cell}.json').read_text())['task_params']
    refs=fit_motion_reference(rs,lab,params)
    records=[]
    for cohort, root, name in [('p3',P3,cell.replace('spatial','sp')),('cal',CAL,cell)]:
        df=read_selected(root/name,cell,cohort).sort_values(['uid','attempt','step'])
        for (_, _), ep in df.groupby(['uid','attempt'],sort=False):
            ep=ep.sort_values('step').reset_index(drop=True)
            task=int(ep.task_id.iloc[0]); ref=refs[task]
            q=np.array([json.loads(x)[:8] for x in ep['state.normalized']])
            t0=time.perf_counter(); distances, inds=ref['tree'].query((q-ref['center'])/ref['scale']); query_ms=(time.perf_counter()-t0)*1000/len(q)
            nearest=ref['rows'][inds]
            scores=lab['difficulty'][nearest]
            hard=scores>=ref['hard_threshold']
            retrieved=np.array([json.loads(x)[0] for x in ep['retrieval.rows']])
            # Purely proprioceptive labels are primary. Actual A's top-1 labels
            # are a sensitivity analysis that reuses its existing visual lookup.
            rscore=lab['difficulty'][retrieved]
            rhard=rscore>=ref['hard_threshold']
            seen=q[:,:3]/ref['scale'][:3]
            movement=np.r_[np.linalg.norm(np.diff(seen,axis=0),axis=1),np.nan]
            step=ep.step.to_numpy()
            consecutive=np.r_[np.diff(step)==2,False]
            expected=ref['delta'][nearest]
            ratio=movement/np.maximum(expected,np.finfo(float).eps)
            under=(ratio<ref['ratio_low']) & consecutive & np.isfinite(expected)
            comparable=consecutive & np.isfinite(expected)
            persistent=np.zeros(len(ep),dtype=bool);run=0
            for k,v in enumerate(under):
                run=run+1 if v else 0
                persistent[k]=run>ref['run_limit']
            # P3 stores actual client successor after 5/10 issued controls.
            # Motion/reversal ratio is descriptive and has no tuned threshold.
            raw=np.array([json.loads(x) for x in ep['state.raw']])
            physical=[]
            for k,row in ep.iterrows():
                a=row['successor_after_head.observation_numeric.robot0_eef_pos']
                b=row['successor_after_commit.observation_numeric.robot0_eef_pos']
                if isinstance(a,str) and isinstance(b,str):
                    ah=np.array(json.loads(a)); ac=np.array(json.loads(b))
                    physical.append((np.linalg.norm(ah-raw[k,:3]),np.linalg.norm(ac-ah),np.linalg.norm(ac-raw[k,:3])))
                else: physical.append((np.nan,np.nan,np.nan))
            for k,row in ep.iterrows():
                records.append(dict(cell=cell,cohort=cohort,arm=row.arm,uid=row.uid,attempt=int(row.attempt),
                                    task=task,init=int(row.init),step=int(row.step),Y=int(row.Y),
                                    pure_policy=('_P10_' in row.arm),actual_controls=int(row.actual_commit_controls),
                                    nearest_row=int(nearest[k]),retrieved_row=int(retrieved[k]),
                                    stage=int(lab['stage'][nearest[k]]),phase=int(lab['phase'][nearest[k]]),
                                    difficulty=float(scores[k]),hard=bool(hard[k]),
                                    retrieved_hard=bool(rhard[k]),retrieved_known=bool(np.isfinite(rscore[k])),
                                    event_near=bool(lab['event_near'][nearest[k]]),
                                    event_end=bool(lab['event_end'][nearest[k]]),
                                    support=int(lab['support'][nearest[k]]),
                                    tightness=float(lab['tightness'][nearest[k]]),
                                    density=float(lab['density'][nearest[k]]),
                                    deviation=float(distances[k]),out_of_tube=bool(distances[k]>ref['deviation_high']),
                                    under_motion=bool(under[k]),motion_comparable=bool(comparable[k]),
                                    persistent_under_motion=bool(persistent[k]),run_limit=ref['run_limit'],
                                    motion_ratio=float(ratio[k]),motion_cutoff=ref['ratio_low'],
                                    movement=float(movement[k]),expected_movement=float(expected[k]),
                                    actual_head_motion=physical[k][0],actual_tail_motion=physical[k][1],actual_commit_motion=physical[k][2],
                                    first_stall=False, last_anchor=k==len(ep)-1,nn_query_ms=query_ms))
            # Collapse a run of under-motion to its onset; label precedes outcome.
            first=np.flatnonzero(under & ~np.r_[False,under[:-1]])
            base=len(records)-len(ep)
            for k in first: records[base+k]['first_stall']=True
    out=pd.DataFrame(records)
    out.to_csv(HERE/f'telemetry_{cell}.csv',index=False)
    print(cell, 'seconds',round(time.perf_counter()-tick,2),'anchors',len(out),flush=True)
    return out


def main():
    dfs=[analyze_cell(*cell) for cell in CELLS]
    df=pd.concat(dfs,ignore_index=True)
    rows=[]
    for (cell,cohort,pure),g in df.groupby(['cell','cohort','pure_policy']):
        e=g.groupby('uid').agg(Y=('Y','first'), hard_share=('hard','mean'), out_share=('out_of_tube','mean'),
                                hard_last=('hard','last'),hard_first=('hard','first'),score_first=('difficulty','first'),
                                score_mean=('difficulty','mean'),dev_first=('deviation','first'),
                                score_max=('difficulty','max'),event_share=('event_near','mean'))
        c=g[g.motion_comparable]
        hard_rate=c[c.hard].under_motion.mean(); easy_rate=c[~c.hard].under_motion.mean()
        firsts=g[g.first_stall]
        row=dict(cell=cell,cohort=cohort,pure_policy=pure,episodes=len(e),successes=int(e.Y.sum()),
                 anchors=len(g),hard_share=g.hard.mean(),event_share=g.event_near.mean(),
                 under_motion_share=c.under_motion.mean(),hard_under_motion=hard_rate,easy_under_motion=easy_rate,
                 under_motion_risk_ratio=hard_rate/easy_rate if easy_rate>0 else np.nan,
                 stall_onsets=len(firsts),hard_stall_onset_share=firsts.hard.mean(),
                 failed_last_hard=e[e.Y==0].hard_last.mean(),success_last_hard=e[e.Y==1].hard_last.mean(),
                 fail_auc_hardshare=auc(1-e.Y,e.hard_share),fail_auc_initial=auc(1-e.Y,e.score_first),
                 fail_auc_initial_deviation=auc(1-e.Y,e.dev_first),
                 success_hard_share=e[e.Y==1].hard_share.mean(),failed_hard_share=e[e.Y==0].hard_share.mean(),
                 nn_retrieved_hard_agreement=(g.hard==g.retrieved_hard)[g.retrieved_known].mean(),
                 retrieved_known=g.retrieved_known.mean(),out_of_tube_share=g.out_of_tube.mean(),
                 actual_commit_motion_hard=g[g.hard].actual_commit_motion.median(),
                 actual_commit_motion_easy=g[~g.hard].actual_commit_motion.median(),
                 nn_query_ms=float(g.nn_query_ms.median()))
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE/'telemetry_summary.csv',index=False)
    print(pd.DataFrame(rows).query('cohort == "p3" and pure_policy == False')[['cell','episodes','successes','hard_share','under_motion_risk_ratio','stall_onsets','hard_stall_onset_share','fail_auc_initial','nn_retrieved_hard_agreement']].round(3).to_string(index=False))


if __name__=='__main__':
    main()
