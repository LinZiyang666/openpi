"""SHA-checked R10/R13 reuse and unchanged R11 periodic schedule solves."""
from __future__ import annotations
import copy, json, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace as NS
import numpy as np
from exp.offline_search.closed_loop import devset
from exp.offline_search.rounds.r11.knob.calibration import ScheduleReplay
from exp.offline_search.rounds.r11.opus import ir_model as M
from exp.offline_search.rounds.r13.knob.recipe import R13Knob
from .recipe import R15Knob

HERE=Path(__file__).resolve().parent
ROUNDS=HERE.parent.parent
CELLS=tuple(f'{m}_{s}_{n}' for n in (100,200,500) for m in ('pi05','groot') for s in ('l10','spatial'))
LEVELS=('c10','ir45','ir40','ir32','ir25','off')
SPEC='exp.offline_search.rounds.r13.knob.recipe:R13Knob'


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_bytes(value);tmp.replace(path)


def write_json(path,value):
    dump(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def schedule_episodes(cell):
    model,suite,size=cell.split('_'); size=int(size)
    parent=ROUNDS/'r11/opus/out/anchor'
    paths=[parent/f'{cell}_f{f}.npz' for f in range(5)]
    arrays=[]
    for f,path in enumerate(paths):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['meta_json']))
            assert (meta['model'],meta['suite'],meta['size'],meta['fold'])==(model,suite,size,f)
            arrays.append({k:z[k].copy() for k in ('row','ep','task','step','top1_ep','top1_prog','top1_eplen')})
    d={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    expected=np.load(ROUNDS/'r10/subsets'/f'{cell}.npy',allow_pickle=False)
    assert np.array_equal(np.sort(d['row']),expected)
    spec=json.loads((ROUNDS/'r10/subsets'/f'{cell}.json').read_text())
    assert set(d['ep'])=={e for es in spec['episode_ids_by_task'].values() for e in es}
    assert np.all(d['ep']!=d['top1_ep'])
    order=np.lexsort((d['step'],d['ep']));d={k:v[order] for k,v in d.items()}
    eps=[]
    bounds=np.flatnonzero(np.r_[True,d['ep'][1:]!=d['ep'][:-1],True])
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        assert np.array_equal(d['step'][lo:hi],np.arange(hi-lo))
        a=np.flatnonzero(d['step'][lo:hi]%2==0)+lo
        steps=d['step'][a]
        eps.append(NS(ep=int(d['ep'][lo]),task=int(d['task'][lo]),n_slots=hi-lo,steps=steps,
                      guard=M.guard_flags(steps,d['top1_prog'][a],d['top1_eplen'][a])))
    return eps,{str(p):devset.sha(p) for p in paths}


def fit_cell(cell):
    model,suite,n=cell.split('_');n=int(n)
    catalogs=json.loads((ROUNDS/'r13/impl/fits.json').read_text())
    prior={r['config_key']:r for r in catalogs}
    r10=ROUNDS/'r10/recipe/artifacts'/f'r10_recipe_{cell}.pkl'
    audited=json.loads((ROUNDS/'r10/recipe/artifact_audit.json').read_text())['records']
    record=next(r for r in audited if r['artifact']==str(r10))
    assert devset.sha(r10)==record['sha256']
    base=pickle.loads(r10.read_bytes())
    spec=json.loads((ROUNDS/'r10/subsets'/f'{cell}.json').read_text())
    assert base['method'].size==n and base['method'].fit_info['library_only']
    assert base['method'].fit_info['episode_ids_by_task']==spec['episode_ids_by_task']
    assert base['kwargs']['episode_subset']==spec['episode_ids_by_task']
    eps=None;sched=None;inputs={};results={}
    for level in LEVELS:
        target=None if level in ('off','c10') else int(level[-2:])/100
        key=cell+(':off' if level=='off' else ':pure_c10' if level=='c10' else f':periodic:{target}')
        if key in prior:
            rec=copy.deepcopy(prior[key]);assert devset.sha(rec['path'])==rec['sha256']
            rec.update(reused=True,lower_layer_path=str(r10),lower_layer_sha256=record['sha256'])
            blob=pickle.loads(Path(rec['path']).read_bytes())
            assert blob['method'].fit_info==base['method'].fit_info
            rec['catalog_path']=str(ROUNDS/'r13/impl/fits.json')
            rec['catalog_sha256']=devset.sha(ROUNDS/'r13/impl/fits.json')
        else:
            path=ROUNDS/'r11/knob/artifacts'/f'r15_{cell}_{level}.pkl'
            if level=='c10':
                cfg=dict(setting=0.,dose=0.,pred_IR_lib=.5,library_only=True)
            else:
                if sched is None:
                    eps,inputs=schedule_episodes(cell);sched=ScheduleReplay(model,eps)
                if level=='off':
                    cfg=dict(library_only=True,pred_IR_lib=sched.ir('off',0.))
                else:
                    x,ir,status=sched.solve('periodic',target)
                    cfg=dict(library_only=True,setting=round(x,4),unrounded_setting=x,solve_status=status,
                             pred_IR_lib=ir,pred_IR_rounded=sched.ir('periodic',round(x,4)),
                             v=sched.V/sched.N,g=sched.G/sched.V)
            blob=copy.deepcopy(base);obj=blob['method'];obj.__class__=R13Knob
            obj.knob_method='off' if level=='off' else 'periodic';obj.target=.5 if level=='c10' else target
            obj._controller=None;obj._guard_steps=set();obj._trigger_steps=set()
            obj._proposal_step=-1;obj._proposal=None
            obj._ledger_checkpoint=(0,0,0);obj._accepted_prefix=((),())
            obj.configure(cfg,model)
            kwargs=dict(base['kwargs'],method='pure_c10' if level=='c10' else obj.knob_method,target=obj.target)
            blob.update(spec=SPEC,kwargs=kwargs)
            payload=pickle.dumps(blob,protocol=4)
            if path.exists():
                if path.read_bytes()!=payload: raise ValueError(f'existing fit differs: {path}')
            else: dump(path,payload)
            rec=dict(config_key=key,path=str(path),sha256=devset.sha(path),spec=SPEC,kwargs=kwargs,
                     reused=False,lower_layer_path=str(r10),lower_layer_sha256=record['sha256'],
                     calibration=cfg,calibration_inputs_sha256=inputs,
                     recipe='R11 ScheduleReplay.solve, 8 keyed replicates, 26 bisections, 4 decimal places',
                     schedule_inputs='original held-out anchor top1 progress; corrector risks unused by periodic solve')
        obj=blob['method'];assert obj.fit_info['episode_ids_by_task']==spec['episode_ids_by_task']
        if level=='c10': assert obj.knob_settings['setting']==0
        rec.update(cell=cell,level=level)
        results[level]=rec
    if n==500:
        _,episodes,_=devset.library_episodes(model,suite,base['kwargs']['library'])
        identities={(e['task_id'],e['orig_init_state_idx']):i for i,e in enumerate(episodes)}
        assert len(identities)==500
        for level in LEVELS:
            if level=='c10':continue
            source=results[level];blob=pickle.loads(Path(source['path']).read_bytes())
            obj=blob['method'];obj.__class__=R15Knob;obj.enable_leave_own_episode_out(identities)
            kwargs=dict(blob['kwargs'],leave_own_episode_out=True)
            blob.update(spec='exp.offline_search.rounds.r15.descent.recipe:R15Knob',kwargs=kwargs)
            path=HERE/'artifacts'/f'{cell}_{level}_leave_own_out.pkl'
            dump(path,pickle.dumps(blob,protocol=4))
            results[level]=dict(source,path=str(path),sha256=devset.sha(path),spec=blob['spec'],kwargs=kwargs,
                               serving_wrapper=True,source_path=source['path'],source_sha256=source['sha256'])
    write_json(HERE/'fits'/f'{cell}.json',results)
    print('FIT_READY',cell,flush=True)
    return results


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        out=dict(zip(CELLS,pool.map(fit_cell,CELLS),strict=True))
    write_json(HERE/'fits.json',out)

if __name__=='__main__':main()
