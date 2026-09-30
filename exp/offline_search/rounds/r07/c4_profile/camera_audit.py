"""Episode-fold camera metric refits; action and gripper-event errors by stage."""
from __future__ import annotations

import time
import numpy as np

from . import common as C
from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric, _kernel_w, PCA_CUR, PCA_BIG


def action_errors(pred, target, sigma, grip, centers, block):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    pm = np.abs(pred[:, grip, None]-centers).argmin(-1)
    tm = np.abs(target[:, grip, None]-centers).argmin(-1)
    pe = np.flatnonzero(pm[1:] != pm[:-1])+1
    te = np.flatnonzero(tm[1:] != tm[:-1])+1
    # A timing number needs an event in both chunks. Missing/extra events count separately.
    return dict(head_rms=float(np.sqrt(np.mean(((pred[:block]-target[:block])/sigma)**2))),
                commit_rms=float(np.sqrt(np.mean(((pred-target)/sigma)**2))),
                gripper_mode_error=float(np.mean(pm != tm)),
                missing_event=int(len(te)>0 and len(pe)==0), extra_event=int(len(pe)>0 and len(te)==0),
                event_time_error=float(abs(int(pe[0])-int(te[0]))) if len(pe) and len(te) else None)


def summarize(rows, reps):
    out = []
    metrics = ['head_rms', 'commit_rms', 'gripper_mode_error', 'missing_event', 'extra_event', 'event_time_error', 'delete_head_rms_delta']
    for stage in ['all']+sorted({r['stage'] for r in rows}):
        for mode in ['full', 'wrist', 'third_person']:
            g = [r for r in rows if r['camera']==mode and (stage=='all' or r['stage']==stage)]
            if not g:
                continue
            report = dict(stage=stage, camera=mode, rows=len(g), episodes=len({r['episode'] for r in g}))
            for metric in metrics:
                eps = sorted({r['episode'] for r in g if r[metric] is not None})
                means = np.asarray([np.mean([r[metric] for r in g if r['episode']==e and r[metric] is not None]) for e in eps])
                report[metric] = C.PC.EpisodeBootstrap(np.arange(len(means)), reps, C.SEED).mean_ci(means) if len(means) else None
                report[metric+'_episodes'] = len(means)
            if mode != 'full':
                lookup = {(r['row'], r['fold']): r for r in rows if r['camera']=='full'}
                diffs = [r['head_rms']-lookup[r['row'],r['fold']]['head_rms'] for r in g]
                means = np.asarray([np.mean([v for r,v in zip(g,diffs) if r['episode']==e]) for e in sorted({r['episode'] for r in g})])
                report['paired_head_delta_vs_full'] = C.PC.EpisodeBootstrap(np.arange(len(means)), reps, C.SEED).mean_ci(means)
            out.append(report)
    return out


def bval_errors(cell,base,lib,manifest,table,X,actions,source,names,cams):
    """Same-observation shadow comparisons; no B-val outcome enters fitting."""
    b=manifest['exec_steps'];adims=np.arange(manifest['act_valid_dims']);sdims=np.arange(manifest['rs_valid_dims'])
    grip=list(adims).index(manifest['gripper_dim']);horizon=actions.shape[1]
    sigma=np.asarray(lib['action'][:,:b][:,:,adims],float).reshape(-1,len(adims)).std(0)
    sigma=np.where(sigma>1e-12,sigma,1.)
    fits={}
    for task_id in base.tasks:
        train=np.flatnonzero((lib['task_id']==task_id)&lib['success'])
        for name,x in zip(names,X):
            mu,sd,metric=fit_metric(x[train],(actions[train,:b]/sigma).reshape(len(train),-1),source[train],nn=base.nn,lam=base.lam,rank=base.codes)
            fits[task_id,name]=(train,mu,sd,metric,((x[train]-mu)/sd)@metric)
    archives={(r['uid'],int(r['step'])):r['absolute_input_archive'] for r in C.tables(cell,'bval','decisions')}
    rows=[]
    for i,a in enumerate(C.tables(cell,'bval')):
        task=int(a['task_id'])
        with np.load(archives[a['uid'],int(a['step'])],allow_pickle=False) as z:
            state=np.asarray(z['robot_state'][sdims],float)
            pca=[getattr(base,f'B{cam}T')@z[f'vision_{cam}']-getattr(base,f'muB{cam}') for cam in range(len(cams[0]))]
            target=np.asarray(z['policy_chunk'][:horizon][:,adims],float)
        stage=C.stage_label(table,C.decode(a['retrieval.rows']),C.decode(a['retrieval.weights']))
        for name,cam in zip(names,cams):
            train,mu,sd,metric,z=fits[task,name]
            query=np.concatenate([*[pca[j] for j in cam],state]);qz=((query-mu)/sd)@metric
            dist=np.linalg.norm(z-qz,axis=1)
            ix=np.argpartition(dist,base.k-1)[:base.k];ix=ix[np.lexsort((train[ix],dist[ix]))]
            weights=_kernel_w(dist[ix]-dist[ix[0]],base.kref);weights/=weights.sum()
            pred=np.einsum('i,ihd->hd',weights,actions[train[ix]])
            error=action_errors(pred,target,sigma,grip,table.gripper_centers,b)
            unique=np.unique(source[train[ix]])
            mass=[weights[source[train[ix]]==e].sum() for e in unique]
            rows.append(dict(cell=cell,row=i,episode=a['uid'],task=task,fold='bval',stage=stage,camera=name,
                candidate_episodes=len(unique),max_episode_mass=float(max(mass)),delete_head_rms_delta=None,**error))
    return rows


def audit_cell(cell, folds=2, reps=2000, bval=True):
    t0 = time.monotonic()
    base, lib, manifest = C.bank(cell)
    table = C.stage_table(lib, manifest, base)
    model, suite, _ = cell.split('_')
    root = PCA_CUR/f'{model}_{suite}' if base.cand_name=='current' else PCA_BIG/f'{model}_{suite}'/base.cand_name
    # Same policy PCA, fixed independently of current held-out outcome; metrics refit per fold.
    camera_count=len(manifest['img_source_keys']) if manifest.get('img_source_keys') else len(list(lib.dir.glob('key_v[0-9].npy')))
    projections = [np.load(root/f'v{i}'/'proj.npy', mmap_mode='r')[:, :base.B0T.shape[0]] for i in range(camera_count)]
    if len(projections) != 2:
        raise ValueError('full/wrist/third-person adapter requires the declared two-view library')
    sdims = np.arange(manifest['rs_valid_dims']); adims = np.arange(manifest['act_valid_dims'])
    grip = list(adims).index(manifest['gripper_dim'])
    states = np.asarray(lib['rs'][:, sdims], float)
    b = manifest['exec_steps']; horizon = min(manifest['H'], 2*b)
    actions = np.asarray(lib['action'][:, :horizon, adims], float)
    source = __import__(__package__+'.follow_audit', fromlist=['source_episodes']).source_episodes(lib)
    names = ['full','wrist','third_person']; cams = [[0,1],[1],[0]]
    X = [np.concatenate([*[np.asarray(projections[i], float) for i in cam], states], axis=1) for cam in cams]
    # Mode centers are the frozen StageTable gripper-command clusters.
    centers = np.asarray(table.gripper_centers, float)
    rows_out = []
    for task_id in sorted(base.tasks):
        taskrows = np.flatnonzero((lib['task_id']==task_id) & lib['success'])
        eps = np.unique(source[taskrows])
        if len(eps) < 2*folds:
            raise ValueError('folds need at least two training source episodes')
        assignment = {int(e): i%folds for i,e in enumerate(eps)}
        labels = np.asarray([assignment[int(e)] for e in source[taskrows]])
        for fold in range(folds):
            train, test = taskrows[labels!=fold], taskrows[labels==fold]
            sigma = np.asarray(lib['action'][train, :b][:,:,adims], float).reshape(-1,len(adims)).std(0)
            sigma = np.where(sigma>1e-12,sigma,1.)
            supervised = (actions[train,:b]/sigma).reshape(len(train),-1)
            for name, x in zip(names,X):
                mu, sd, wmetric = fit_metric(x[train], supervised, source[train], nn=base.nn, lam=base.lam, rank=base.codes)
                z = ((x[train]-mu)/sd) @ wmetric
                qz = ((x[test]-mu)/sd) @ wmetric
                for row, query in zip(test,qz):
                    d = np.linalg.norm(z-query, axis=1)
                    ix = np.argpartition(d,base.k-1)[:base.k]
                    ix = ix[np.lexsort((train[ix],d[ix]))]
                    weights = _kernel_w(d[ix]-d[ix[0]],base.kref); weights /= weights.sum()
                    pred = np.einsum('i,ihd->hd',weights,actions[train[ix]])
                    error = action_errors(pred,actions[row],sigma,grip,centers,b)
                    # Delete the highest-mass candidate source episode, keep kernel size/rule.
                    unique = np.unique(source[train[ix]])
                    mass = [weights[source[train[ix]]==e].sum() for e in unique]
                    excluded = unique[int(np.argmax(mass))]
                    dd = d.copy(); dd[source[train]==excluded] = np.inf
                    deleted_delta = None
                    if np.isfinite(dd).sum() >= base.k:
                        jj = np.argpartition(dd,base.k-1)[:base.k]; jj = jj[np.lexsort((train[jj],dd[jj]))]
                        ww = _kernel_w(dd[jj]-dd[jj[0]],base.kref); ww /= ww.sum()
                        pp = np.einsum('i,ihd->hd',ww,actions[train[jj]])
                        deleted_delta = action_errors(pp,actions[row],sigma,grip,centers,b)['head_rms']-error['head_rms']
                    rows_out.append(dict(cell=cell, row=int(row), episode=int(source[row]), task=int(task_id), fold=fold,
                        stage=C.stage_label(table,[row],[1.]), camera=name,
                        candidate_episodes=len(unique), max_episode_mass=float(max(mass)),
                        delete_head_rms_delta=deleted_delta, **error))
    bv=bval_errors(cell,base,lib,manifest,table,X,actions,source,names,cams) if bval else []
    return dict(cell=cell, folds=folds, rows=len(rows_out)//len(names), source_episodes=len(np.unique(source)),
                bval_anchors=len(bv)//len(names),bval_metrics=summarize(bv,reps),
                elapsed_s=time.monotonic()-t0, metrics=summarize(rows_out,reps)), [dict(population='library_folds',**r) for r in rows_out]+[dict(population='bval_shadow',**r) for r in bv]


def main():
    p=C.parser(__doc__); p.add_argument('--folds',type=int,default=2); p.add_argument('--reps',type=int,default=2000)
    p.add_argument('--skip-bval',action='store_true');a=p.parse_args()
    if a.folds<2:p.error('at least two episode folds required')
    reports=[];rows=[]
    for cell in C.cells(a.cells):
        report,rr=audit_cell(cell,a.folds,a.reps,not a.skip_bval);reports.append(report);rows.extend(rr)
        print(cell,'rows',report['rows'],'seconds',round(report['elapsed_s'],2),flush=True)
    C.write(a.out/'camera_audit.json',dict(schema='r7.c4.camera.v1',reports=reports,
        caveat='Library imitation screening, not SR or isolated-tower parity. All three action-supervised metrics and action scales refit on training source episodes; policy PCA and frozen library stage vocabulary retained. Main metric at every row; deployed early metric is not simulated. Timing errors condition on an event present in both chunks; missing/extra events reported separately.'))
    C.csv_write(a.out/'camera_rows.csv',rows)


if __name__=='__main__':main()
