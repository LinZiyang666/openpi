"""CPU-only fit preparation, writing only this task's allowed scratch tree.

Filled specs must point at final artifact locations BEFORE this command. It
refuses dry-run C calibrations and infeasible budgets through the production fit.
"""
import argparse
import json
import pickle
import time
from pathlib import Path
from exp.offline_search.harness import api,store
from exp.offline_search.closed_loop.plugin import load_method_class
from .common import HERE,OUT,STORE,sources,load_base,sha,write_json,output_path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True)
    ap.add_argument('--name',required=True);ap.add_argument('--out',type=Path,default=OUT/'prefits')
    a=ap.parse_args();spec=next(r for r in json.loads(a.spec.read_text()) if r['name']==a.name)
    if any(p in json.dumps(spec) for p in ('<RUN>','<CAL>','<STALL>')):raise ValueError('fill all artifact/run placeholders first')
    out=output_path(a.out);out.mkdir(parents=True,exist_ok=True)
    path=out/(a.name+'.pkl')
    if path.exists():raise FileExistsError(path)
    cell=f'{spec["model"]}_{spec["suite"]}_cache'
    cls,src=load_method_class(spec['method']);method=cls(**spec['kwargs'])
    ctx=api.Context(root=STORE,cell=cell,seed=0,scratch=out/'scratch'/a.name)
    ctx.scratch.mkdir(parents=True,exist_ok=True)
    lib=store.LibraryView(STORE,store.lib_key(cell),'current');start=time.monotonic()
    if cls.__name__=='RiskLottery':
        candidates=[r for c,r in sources().items() if c.startswith(f'{spec["model"]}_{spec["suite"]}_') and r['kwargs']['lib']==spec['kwargs']['lib']]
        assert len(candidates)==1
        base,blob=load_base(candidates[0]);own=dict(method.__dict__)
        method.__dict__.update(base.__dict__)
        method.__dict__.update({k:v for k,v in own.items() if k.startswith(('q2_','_policy_gate_anchor'))})
        method.finish_adapter_fit(lib,ctx)
    elif cls.__name__ in ('BmechCommitJudge','BmechGrootCommitJudge'):
        notes=json.loads((HERE.parents[1]/'ideation_Q3/stall/bmech_artifacts.json').read_text())
        note=next(r for r in notes if r['name']==a.name)
        if sha(note['artifact'])!=note['sha256']:raise ValueError('Q3 Bmech artifact mismatch')
        # Resolve its original file-module classes before loading.
        old=note['source_spec'];old_cls,_=load_method_class(old['method']);old_cls(**old['kwargs'])
        with open(note['artifact'],'rb') as f:blob=pickle.load(f)
        assert blob['spec']==spec['method'] and blob['kwargs']==spec['kwargs']
        method=blob['method']
    else:
        method.fit(lib,ctx)
    api.check_method_attrs(method);method.prof=api.NULL_PROFILER
    blob=dict(method=method,registered=ctx.registered,spec=spec['method'],kwargs=spec['kwargs'],cell=cell,
        fit_s=time.monotonic()-start,provenance=dict(builder=str(Path(__file__)),method_source=src,source_sha256=sha(src)))
    with path.open('xb') as f:pickle.dump(blob,f,protocol=4)
    note=dict(name=a.name,artifact=str(path),sha256=sha(path),bytes=path.stat().st_size,fit_s=blob['fit_s'],
        fit_info=getattr(method,'fit_info',{}),spec=spec['method'],kwargs=spec['kwargs'])
    write_json(path.with_suffix('.json'),note);print(json.dumps(dict(name=a.name,path=str(path),seconds=blob['fit_s'])))

if __name__=='__main__':main()
