"""R8 E1 discovery-only analysis using the delivered physical-tool functions.

No production modifications. All generated products go to callback scratch.
"""
import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from exp.offline_search.debug import reader
from exp.offline_search.debug.tools.physical import segmentation as seg
from exp.offline_search.debug.tools.physical import forensics, segmentation_bench as bench
from exp.offline_search.debug.tools.physical.adapters import calibrate
from exp.offline_search.debug.tools.physical.common import Episode, clean, fingerprint

ROOT = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')
OUT = Path('/tmp/r8cb_E1_segmentation')
LIB = Path('/home/weiland/trace_runs/offline_search_store/library')
FIELDS = ['control_idx','decision_seq','chunk_offset','is_settle','action','reward','done',
          'eef_pos','eef_quat','gripper_qpos','gripper_qvel','obj_pos','obj_quat','predicates']
SPECS = json.loads((ROOT/'arms.json').read_text())
VARIANTS = ('A','CU','CT','FL','P10')


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+'\n')


class ArrayView:
    """Batch-load only served heads; avoid repeated NPZ indexing per episode."""
    def __init__(self, arrays):
        self.arrays = arrays

    def decision_arrays(self, keys, ids):
        assert keys == ['served_chunk']
        return {'served_chunk': np.asarray([self.arrays[i] for i in ids])}


def load(spec, normalized=True, init_limit=30):
    arm = reader.open_arm(ROOT, spec['arm'])
    arm.cache_enabled = False
    # Mask before inspecting outcomes, decisions, controls, or tool fitting.
    frame = arm.episodes()
    frame = frame.loc[frame['init'].between(0, init_limit-1)].copy()
    frame = frame.sort_values(['task_id','init'])
    assert len(frame) == 10*init_limit and frame['init'].max() < 30
    keys = set(frame.episode_key)
    ds = arm.decisions()
    ds = ds.loc[ds.episode_key.isin(keys)].copy()
    assert ds.init.max() < 30
    by_key = {k: clean(v.to_dict('records')) for k,v in ds.groupby('episode_key',sort=False)}
    arrays = {}
    if normalized:
        ids = ds.decision_id.tolist()
        chunks = arm.decision_arrays(['served_chunk'], ids)['served_chunk']
        arrays = dict(zip(ids, chunks))
    view = ArrayView(arrays)
    manifest = dict(arm.server_meta, **arm.manifest)
    episodes=[]
    for row in clean(frame.to_dict('records')):
        row['arm'] = spec['arm']
        controls = arm.controls(row['episode_key'], FIELDS)
        ep = Episode(row, controls, by_key[row['episode_key']], manifest, {}, view)
        ep.require(*FIELDS)
        episodes.append(ep)
    catalog = arm.catalog() if spec['r8']['variant'] != 'P10' else None
    audit = dict(arm=spec['arm'], episodes=len(episodes), decisions=len(ds),
        controls=sum(e.n for e in episodes), active_controls=sum(int((~e.controls['is_settle']).sum()) for e in episodes),
        join_counts=ds.server_join.value_counts().to_dict(),
        capture_error_episodes=sum(bool(e.meta.get('capture_errors')) for e in episodes),
        read_issues=arm.read_issues, aug_done=(ROOT/'state'/f"{spec['arm']}.AUG_DONE").exists())
    return episodes,catalog,audit


def prepare():
    all_eps=[]
    groups={}
    for spec in SPECS:
        if spec['r8']['variant']=='P10':
            eps,_,audit=load(spec, normalized=False)
            groups[(spec['model'],spec['suite_short'])]=eps
            all_eps.extend(eps)
            print('PREP',spec['arm'],len(eps),flush=True)
    config=calibrate(all_eps)
    write_json(OUT/'adapter_calibration.json',config)
    for (model,suite),eps in groups.items():
        for size in (50,500):
            bank='current' if size==50 else 'bpool_cs' if model=='pi05' else 'bpool_all'
            fitted=seg.fit(eps,config,LIB/f'{model}_{suite}'/bank)
            write_json(OUT/f'fit_{model}_{suite}_{size}.json',fitted)
    write_json(OUT/'scope.json',dict(run_root=str(ROOT),init_min=0,init_max=29,
        geometry_fit='successful P10 discovery trajectories; exploratory, not held-out validation',
        physical_calibration='successful discovery P10 in both models; shared per-suite/task radii',
        arm_specs=[s for s in SPECS if s['r8']['variant'] in VARIANTS]))


def catalog_mass(ep,catalog):
    if catalog is None:
        return {}
    lookup=catalog.set_index('row')
    events=lookup.event_near.to_dict()
    success=lookup.success.to_dict()
    out={}; carry=None
    for d in ep.decisions:
        rows=d.get('rows'); weights=d.get('weights')
        if isinstance(rows,list) and isinstance(weights,list) and len(rows)==len(weights) and rows:
            w=np.asarray(weights,float)
            if np.isfinite(w).all() and w.sum()>0 and np.all(w>=0):
                known=np.array([int(r) in events and bool(success.get(int(r),False)) for r in rows])
                carry=dict(value=float(sum(v*bool(events.get(int(r),False)) for r,v in zip(rows,w))/w.sum()),
                           known_mass=float(w[known].sum()/w.sum()), from_decision=d['decision_seq'])
        if carry is not None:
            out[d['decision_id']]=dict(carry,carried=carry['from_decision'] != d['decision_seq'])
    return out


def analyze_group(model,suite,init_limit=30):
    start_time=time.perf_counter()
    config=json.loads((OUT/'adapter_calibration.json').read_text())
    specs=[s for s in SPECS if s['model']==model and s['suite_short']==suite and s['r8']['variant'] in VARIANTS]
    specs.sort(key=lambda s:(s['r8'].get('library_size') or 0,VARIANTS.index(s['r8']['variant'])))
    for spec in specs:
        name=spec['arm'];dest=OUT/name
        if (dest/'COMPLETE.json').exists():
            print('SKIP',name,flush=True);continue
        tick=time.perf_counter()
        eps,catalog,audit=load(spec,init_limit=init_limit)
        size=spec['r8'].get('library_size') or 50
        fitted=json.loads((OUT/f'fit_{model}_{suite}_{size}.json').read_text())
        names=seg.candidates(fitted)
        boundaries=[];summaries=[];decisions=[];reconstruct=[];prefix=[];truth_runs=[];errors=[]
        for ei,ep in enumerate(eps):
            ident={k:ep.meta[k] for k in ('arm','episode_key','task_id','init')}
            ident.update(model=model,suite=suite,variant=spec['r8']['variant'],library_size=spec['r8'].get('library_size'))
            try:
                f=seg.baseline_commands(ep,seg.features(ep,config),fitted)
                forensic,truth=forensics.analyse(ep,config)
                active=np.flatnonzero(f['active']);left,right=int(active[0]),int(active[-1])+1
                gold=(np.flatnonzero(truth['stage'][left+1:right] != truth['stage'][left:right-1])+left+1).tolist()
                counts=Counter(str(x) for x in truth['stage'][active])
                summaries.append(dict(ident,success=ep.outcome['success'],n_controls=ep.n,
                    active_controls=len(active),n_decisions=len(ep.decisions),label=forensic['label'],
                    onset=forensic['onset_control'],truth_boundaries=len(gold),
                    predicate_detail=json.dumps(clean(forensic['predicate_detail'])),
                    stage_counts=json.dumps(dict(counts)),tail_motion=forensic['tail_motion'],
                    near_radius=forensic['adapter_settings']['near_radius'],reset_hash=ep.meta['reset_state_sha256']))
                knots=[left]+gold+[right]
                for a,b in zip(knots[:-1],knots[1:]):
                    truth_runs.append(dict(ident,start=a,end=b,stage=str(truth['stage'][a])))
                scores={candidate:seg.causal_scores(f,fitted,candidate) for candidate in names}
                for candidate in names:
                    pred=seg.retrospective(ep,f,fitted,candidate)
                    uniform=np.linspace(left,right,len(pred)+2)[1:-1].round().astype(int).tolist()
                    for tol in (1,5,10):
                        for label,points in (('candidate',pred),('uniform',uniform)):
                            boundaries.append(dict(ident,candidate=candidate,baseline=label,
                                **seg.boundary_quality(points,gold,tol)))
                    # Full prefix test on init 0 per task; all real trajectories, no simulator.
                    if ep.meta['init']==0:
                        prefix.append(dict(ident,candidate=candidate,passed=seg.prefix_audit(f,fitted,candidate)))
                    if candidate.startswith('waypoint_'):
                        for resolution in (1,5,10):
                            points=sorted(set([left]+seg.retrospective(ep,f,fitted,candidate,resolution)+[right-1]))
                            reconstruct.append(dict(ident,candidate=candidate,resolution=resolution,knots=len(points),
                                **seg.reconstruction(f,points,left,right)))
                cat=catalog_mass(ep,catalog)
                for d in ep.decisions:
                    t=ep.before_index(d)+1
                    if not 0<=t<ep.n: continue
                    onset=forensic['onset_control']
                    bad=onset is not None and t<=onset<t+20
                    observed=ep.n-t>=20 or bad or bool(ep.outcome['success'])
                    rec=dict(ident,decision_id=d['decision_id'],decision_seq=d['decision_seq'],
                        start=t,src=d.get('src'),vision=bool(d.get('vision')),blind_age=d.get('blind_age_controls'),
                        success=ep.outcome['success'],label=forensic['label'],onset=onset,
                        at_risk=onset is None or t<=onset,observed=observed,bad=bad,
                        known_negative=bool(ep.outcome['success']) or onset is not None,
                        lead=onset-t if bad else None,truth_stage_pre=str(truth['stage'][t-1]) if t else 'unknown',
                        applied=d.get('n_applied'),owner_cost=d.get('owner_cost'),
                        event_mass=cat.get(d['decision_id'],{}).get('value'),
                        event_known_mass=cat.get(d['decision_id'],{}).get('known_mass'),
                        event_mass_carried=cat.get(d['decision_id'],{}).get('carried'),
                        **{candidate:float(scores[candidate][t]) for candidate in names})
                    decisions.append(rec)
            except Exception as exc:
                errors.append(dict(ident,error=type(exc).__name__+': '+str(exc)))
            if (ei+1)%50==0: print('PROGRESS',name,ei+1,'seconds',round(time.perf_counter()-tick,1),flush=True)
        dest.mkdir(parents=True,exist_ok=True)
        for filename,rows in [('episodes',summaries),('boundary',boundaries),('decisions',decisions),
                              ('reconstruction',reconstruct),('prefix',prefix),('truth_runs',truth_runs),('errors',errors)]:
            pd.DataFrame(rows).to_parquet(dest/f'{filename}.parquet',index=False)
        audit.update(seconds=time.perf_counter()-tick,analyzed=len(summaries),errors=len(errors),
                     fit_hash=fitted['fit_hash'],physical_config_hash=fingerprint(config),candidates=names,
                     holdout_episodes_read_into_tools=0)
        write_json(dest/'COMPLETE.json',audit)
        print('DONE',name,clean(audit),flush=True)
        del eps,catalog,boundaries,summaries,decisions,reconstruct,prefix,truth_runs
        gc.collect()
    print('GROUP_DONE',model,suite,round(time.perf_counter()-start_time,1),flush=True)


def stock_smoke(model,suite):
    spec=next(s for s in SPECS if s['model']==model and s['suite_short']==suite and
              s['r8']['variant']=='A' and s['r8']['library_size']==50)
    eps,_,audit=load(spec,init_limit=1)
    config=json.loads((OUT/'adapter_calibration.json').read_text())
    result=bench.build(eps,config,LIB/f'{model}_{suite}'/'current')
    path=OUT/f'stock_smoke_{model}_{suite}'
    for name,rows in result.items():
        path.mkdir(parents=True,exist_ok=True)
        if name == 'fits':
            write_json(path/f'{name}.json',rows)
        else:
            pd.DataFrame(clean(rows)).to_parquet(path/f'{name}.parquet',index=False)
    write_json(path/'coverage.json',dict(audit,counts={k:len(v) for k,v in result.items()}))
    print('STOCK_SMOKE',model,suite,{k:len(v) for k,v in result.items()},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['prepare','analyze','smoke'])
    ap.add_argument('--model',choices=['pi05','groot'])
    ap.add_argument('--suite',choices=['l10','spatial'])
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare': prepare()
    elif args.mode=='smoke': stock_smoke(args.model,args.suite)
    else: analyze_group(args.model,args.suite)
