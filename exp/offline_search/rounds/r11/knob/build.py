"""Fit/calibrate R11 and emit 94 A arms plus 20 fixed-setting B pilot arms."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import subprocess
import time
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.data import HERE as R10, RUNS, STORE, sha, write_json
from exp.offline_search.rounds.r10.recipe.build import subset_kwargs
from .calibration import HERE, CELLS, R11, calibrate, tables, ScheduleReplay
from .recipe import R11Knob, METHODS

REPO=R10.parents[3]
SPEC='exp.offline_search.rounds.r11.knob.recipe:R11Knob'
ROOTS=[RUNS/f'r11_knob_{i}' for i in range(1,5)]+[RUNS/'r11_devknob_50']
PREFIX=['taskset','-c','22-37,66-81','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        'MKL_NUM_THREADS=1','PYTHONPATH=.:src','PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=',str(REPO/'.venv/bin/python')]


def grid_cell(model,suite,size):
    levels=(.25,.32,.40) if size==50 else ((.25,.32) if model=='groot' and size==200 else (.25,))
    rows=[(method,t) for t in levels for method in ('random','periodic','distance','error_hybrid')]
    if size==50:
        rows += [('adaptive_error_hybrid',t) for t in (.32,.40)]
        rows += [('disagreement',.32),('periodic_pgt1',.32)]
        if suite=='l10': rows += [('random_tail2',.32)]
    return [('off',None)]+rows


def name_for(model,suite,size,method,target):
    return f'r11_{model}_{suite}_{size}_{method}'+('' if target is None else f'_ir{round(100*target):02d}')


def artifact_path(model,suite,size,method,target):
    return HERE/'artifacts'/f'{name_for(model,suite,size,method,target)}.pkl'


def run_cpu(args,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:
        r=subprocess.run(PREFIX+args,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'CPU command failed {r.returncode}: {log}')


def save(method,kwargs,model,suite,path,seconds=0.):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    blob=dict(spec=SPEC,kwargs=kwargs,cell=f'{model}_{suite}_cache',method=method,registered={},fit_s=seconds)
    with path.open('wb') as f: pickle.dump(blob,f,protocol=4)
    return dict(artifact=str(path),sha256=sha(path),bytes=path.stat().st_size)


def fit_one(model,suite,size,method,target,path):
    path=Path(path).resolve()
    if not path.is_relative_to(HERE): raise ValueError('fit output must remain under knob/')
    if path.exists(): raise FileExistsError(path)
    kwargs=subset_kwargs(model,suite,R10/'subsets'/f'{model}_{suite}_{size}.json')
    kwargs.update(method=method,target=target)
    obj=R11Knob(**kwargs)
    ctx=api.Context(root=STORE,cell=f'{model}_{suite}_cache',seed=0,scratch=HERE/'scratch'/path.stem)
    start=time.monotonic();obj.fit(None,ctx);api.check_method_attrs(obj)
    out=save(obj,kwargs,model,suite,path,time.monotonic()-start)
    print('FIT',json.dumps(out),flush=True)


def prepare_cell(cell):
    model,suite,size=cell
    stem=f'{model}_{suite}_{size}'
    old=R10/'recipe/artifacts'/f'r10_recipe_{stem}.pkl'
    with old.open('rb') as f: baseblob=pickle.load(f)
    kwargs=subset_kwargs(model,suite,R10/'subsets'/f'{stem}.json')
    assert baseblob['kwargs']==kwargs and baseblob['method'].fit_info['library_only']
    # Regenerate every original setting, including arms omitted from final grid.
    schedules=[r for r in json.loads((R11/'opus/out/arm_grid.json').read_text()) if r['cell']==stem]
    states=[r for r in json.loads((R11/'astra/freeze.json').read_text())['arms'] if r['cell']==stem]
    wanted=set(grid_cell(*cell))|{(r['method'],r['target']) for r in schedules}|{(r['method'],r['target_ir']) for r in states}
    configs={}
    for method,target in sorted(wanted,key=lambda x:(x[0],x[1] or 0)):
        configs[method,target]=calibrate(*cell,method,target)
        print('CALIBRATED',stem,method,target,flush=True)
    # Explicit simulator parity at each solved setting, both 8 and 16 reps.
    from exp.offline_search.rounds.r11.opus import ir_model as M
    from exp.offline_search.rounds.r11.opus.arm_grid import MAKERS
    _,_,eps,sched=tables(*cell)
    parity=[]
    for r in schedules:
        x=configs[r['method'],r['target']]['unrounded_setting']
        for reps in (8,16):
            fast=ScheduleReplay(model,eps,reps).ir(r['method'],x)
            ref=M.simulate(model,eps,MAKERS[r['method']][0](x),reps=reps)['IR']
            assert fast==ref,(stem,r['method'],reps,fast,ref)
            parity.append(abs(fast-ref))
    records=[]
    runtime=ScheduleReplay(model,eps,1,runtime=True)
    for method,target in grid_cell(*cell):
        cfg=configs[method,target]
        obj=R11Knob(**kwargs,method=method,target=target)
        vars(obj).update(copy.deepcopy(vars(baseblob['method'])))
        obj.configure(cfg,model)
        obj.reset(api.EpisodeView('r11-build','',0,0,0,0))
        kw=dict(kwargs,method=method,target=target)
        out=save(obj,kw,model,suite,artifact_path(*cell,method,target),baseblob['fit_s'])
        rec=dict(cell=stem,model=model,suite=suite,size=size,method=method,target=target,
                 arm=name_for(*cell,method,target),kwargs=kw,pred_IR_lib=cfg['pred_IR_lib'],**out)
        if method in ('random','periodic','periodic_pgt1','random_tail2'):
            rec['setting']=cfg['setting'];rec['runtime_key_library_IR']=runtime.ir(method,cfg['setting'])
            rec['runtime_minus_reference_IR']=rec['runtime_key_library_IR']-cfg['pred_IR_rounded']
        records.append(rec)
    audit=dict(cell=stem,frozen_schedule_arms=len(schedules),frozen_state_arms=len(states),
               exact_settings=True,max_simulator_deviation=max(parity),records=records,
               base_artifact_sha256=sha(old))
    # Arrays belong in pickles; a JSON summary preserves scalar settings only.
    audit['settings']=[{k:v for k,v in c.items() if k not in ('predictor','score_reference','calibration')}
                       |dict(method=m,target=t) for (m,t),c in configs.items()]
    write_json(HERE/'calibration'/f'{stem}.json',audit)
    return audit


def addendum(records):
    path=HERE/'PREDICTION_ADDENDUM.md'
    lines=['# Library-only prediction addendum',f'Frozen UTC: **{datetime.now(timezone.utc).isoformat()}**.',
           'Written before any R11 evaluation launch or result. No closed-loop outputs were read. '
           'Settings use each explorer’s B-only whole-episode-held-out calibration. '
           'All arms below retain fixed settings for the B pilot. IR is conditional on exogenous library states; '
           'closed-loop state/guard changes remain unmeasured.',
           '| Cell | Method | Target | Predicted library IR | Setting |', '|---|---|---:|---:|---|']
    for r in records:
        if r['method']=='off': continue
        c=next(c for c in json.loads((HERE/'calibration'/f'{r["cell"]}.json').read_text())['settings']
               if c['method']==r['method'] and c['target']==r['target'])
        setting=(f"{c['setting']:.4f}" if 'setting' in c else
                 f"q={c['dose']:.17g}, t={c.get('threshold')}, tie={c.get('tie_probability')}")
        lines.append(f"| {r['cell']} | {r['method']} | {r['target']:.2f} | {r['pred_IR_lib']:.9f} | {setting} |")
    lines += ['', 'Added targets include both Spatial schedules at .25/.32/.40, Spatial post-guard schedules at .32, '
              'and GR00T L10-200 state methods at .32. The complete final grid is listed to make the additions reviewable.',
              'No new SR advantage is assigned to state placement at matched IR. Sparse cells may improve as IR rises; '
              'dense L10 cells are expected near neutral. The pure-policy library cannot measure recovery utility.']
    content='\n\n'.join(lines[:3])+'\n\n'+'\n'.join(lines[3:])+'\n'
    if path.exists():
        # Preserve the prospective timestamp on identical reruns; a changed
        # prediction needs a separate, explicitly reviewed preregistration.
        previous=path.read_text()
        assert previous.split('\n\n',2)[2]==content.split('\n\n',2)[2], 'prediction freeze changed'
    else:
        path.write_text(content)


def emit(records):
    from exp.offline_search.closed_loop.ops.emit_arms import main as emit_arms
    from exp.offline_search.closed_loop import devset
    from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
    content=(RUNS/'r10_size_pi05/eval500.json').read_bytes()
    assert (RUNS/'r10_size_groot/eval500.json').read_bytes()==content
    assert json.loads(content)==[[t,i] for t in range(10) for i in range(50)]
    bins=[[] for _ in range(4)];cost=[0.]*4
    for r in sorted(records,key=lambda r:(-r['pred_IR_lib'],r['arm'])):
        j=min(range(4),key=lambda j:(cost[j],len(bins[j]),j))
        bins[j].append(r);cost[j]+=r['pred_IR_lib']
    summary=[]
    for root,rs in zip(ROOTS[:4],bins):
        root.mkdir(exist_ok=True)
        (root/'eval500.json').write_bytes(content)
        emit_root(root,rs,False)
        summary.append(dict(root=str(root),arms=[r['arm'] for r in rs],arm_count=len(rs),predicted_IR_sum=sum(r['pred_IR_lib'] for r in rs)))
    dev=[r for r in records if r['size']==50 and (r['method']=='off' or (r['method'] in ('random','periodic','distance','error_hybrid') and r['target']==.32))]
    root=ROOTS[4];root.mkdir(exist_ok=True)
    for model,suite,_ in [c for c in CELLS if c[2]==50]:
        manifest=devset.build_manifest(model,suite,r10_size=50,per_task=5,seed=20261003)
        # Allowed former dev manifest is configuration, never an outcome read.
        prior=RUNS/'r11_dev_size50'/f'dev_{model}_{suite}.json'
        assert json.loads(prior.read_text())==manifest
        write_json(root/f'dev_{model}_{suite}.json',manifest)
    emit_root(root,dev,True)
    contracts={}
    for row in json.loads((root/'arms.json').read_text()):
        contracts[row['arm']]=devset.arm_contract(root,row,load_manifest(row['manifest']))
    write_json(root/'dev_contracts.json',contracts)
    summary.append(dict(root=str(root),arms=[r['arm'] for r in json.loads((root/'arms.json').read_text())],arm_count=len(dev),
                        predicted_IR_sum=sum(r['pred_IR_lib'] for r in dev)))
    write_json(HERE/'roots.json',summary)
    print('ROOTS_READY',json.dumps(summary),flush=True)


def emit_root(root,records,dev):
    from exp.offline_search.closed_loop.ops.emit_arms import main as emit_arms
    rows=[]
    for r in records:
        model,suite=r['model'],r['suite']
        name=r['arm'].replace('r11_','r11_devknob_',1) if dev else r['arm']
        flags=['--os-root',str(STORE),'--os-no-shadow-native','--os-tokens','off','--os-blind',
               '--os-policy-tail','--os-policy-tail-blocks','1','--os-judge','guard_only','--os-fit-artifact',r['artifact']]
        rows.append(dict(name=name,model=model,suite=suite,mode='plugin',method=SPEC,kwargs=r['kwargs'],
            full_model=True,cost_ledger=True,manifest=str(root/(f'dev_{model}_{suite}.json' if dev else 'eval500.json')),
            client_overrides=dict(replan_steps=5,**(dict(resize_size=256) if model=='groot' else {})),plugin_args=flags))
    write_json(root/'arms_in.json',rows)
    emit_arms(['--run-root',str(root),'--spec',str(root/'arms_in.json')])
    arms=json.loads((root/'arms.json').read_text())
    for row in arms:
        r=next(r for r in records if row['arm'] == (r['arm'].replace('r11_','r11_devknob_',1) if dev else r['arm']))
        row.update(pred_IR_lib=r['pred_IR_lib'],r11_method=r['method'],target_ir=r['target'],r10_size=r['size'])
        if dev: row.update(dev=True,init_pool='B')
    write_json(root/'arms.json',arms)
    write_json(root/'protocol.json',dict(dev=dev,init_pool='B' if dev else 'A',pairs_per_arm=50 if dev else 500,
               library_only=True,fixed_settings=True,evaluation_launched=False,layer4_task_indexed=False))


def prepare(workers):
    with ProcessPoolExecutor(workers) as pool: audits=list(pool.map(prepare_cell,CELLS))
    records=[r for a in audits for r in a['records']]
    assert len(records)==94
    write_json(HERE/'calibration_audit.json',dict(PASS=True,cells=8,frozen_schedule_arms=sum(a['frozen_schedule_arms'] for a in audits),
        frozen_state_arms=sum(a['frozen_state_arms'] for a in audits),exact_settings=True,
        max_simulator_deviation=max(a['max_simulator_deviation'] for a in audits),records=records))
    addendum(records)
    emit(records)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('fit')
    for field,choices in (('model',('pi05','groot')),('suite',('l10','spatial')),('method',METHODS)):
        f.add_argument('--'+field,choices=choices,required=True)
    f.add_argument('--r10-size',type=int,required=True);f.add_argument('--target',type=float)
    f.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('prepare');q.add_argument('--workers',type=int,default=4)
    r=sub.add_parser('rebuild');r.add_argument('--model',required=True);r.add_argument('--suite',required=True);r.add_argument('--r10-size',type=int,required=True)
    a=p.parse_args()
    if a.action=='fit': fit_one(a.model,a.suite,a.r10_size,a.method,a.target,a.output)
    elif a.action=='prepare': prepare(a.workers)
    else:
        from .calibration import rebuild
        rebuild(a.model,a.suite,a.r10_size)

if __name__=='__main__': main()
