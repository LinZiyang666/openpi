"""CPU prefit of exact rendered PROFILE/evaluation specs, preserving frozen A."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from exp.offline_search.harness import api, store
from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import load_base, fingerprint

from . import common as C


def prefit(spec,name,out):
    row=next(r for r in json.loads(Path(spec).read_text()) if r['name']==name)
    if '<RUN>' in json.dumps(row):raise ValueError('render all final paths before prefit')
    out=Path(out).resolve()
    if C.SCRATCH not in out.parents and out!=C.SCRATCH:raise ValueError('prefit writes only /tmp/r7_C4')
    out.mkdir(parents=True,exist_ok=True);path=out/(name+'.pkl')
    if path.exists():raise FileExistsError(path)
    cls,source_path=load_method_class(row['method']);method=cls(**row['kwargs'])
    cell=f'{row["model"]}_{row["suite"]}_cache'
    ctx=api.Context(root=C.STORE,cell=cell,seed=0,scratch=out/'scratch'/name)
    ctx.scratch.mkdir(parents=True,exist_ok=True)
    library=store.LibraryView(C.STORE,store.lib_key(cell),'current')
    start=time.monotonic();registered={}
    if callable(getattr(method,'finish_follow_fit',None)):
        candidates=[s for c,s in C.sources().items() if c.startswith(f'{row["model"]}_{row["suite"]}_') and (s['kwargs']['lib']==row['kwargs']['lib'] or row['kwargs']['lib'] in ('bpool_cs','bpool_all') and c.endswith('_500'))]
        if len(candidates)!=1:raise ValueError('ambiguous frozen A source')
        base,blob=load_base(candidates[0])
        from exp.offline_search.rounds.r07.c1_follow.prefit import adapt
        method=adapt(base,row['kwargs'],cls)
        method.finish_follow_fit(library,ctx)
        if fingerprint(method)!=fingerprint(base):raise ValueError('follow changed A retrieval')
        registered=blob.get('registered',{})
    elif row['method']=='exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM':
        candidates=[s for c,s in C.sources().items() if c.startswith(f'{row["model"]}_{row["suite"]}_') and s['kwargs']==row['kwargs']]
        if len(candidates)!=1:raise ValueError('exact frozen A source missing')
        method,blob=load_base(candidates[0]);registered=blob.get('registered',{})
    else:
        method.fit(library,ctx);registered=ctx.registered
    api.check_method_attrs(method);method.prof=api.NULL_PROFILER
    blob=dict(method=method,registered=registered,spec=row['method'],kwargs=row['kwargs'],cell=cell,
              fit_s=time.monotonic()-start,provenance=dict(builder=__file__,method_source=source_path,source_sha256=C.sha(source_path)))
    with path.open('xb') as f:pickle.dump(blob,f,protocol=4)
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
    with path.open('rb') as f:loaded=FitUnpickler(f).load()
    if loaded['kwargs']!=row['kwargs'] or loaded['spec']!=row['method']:raise ValueError('prefit reload identity failed')
    note=dict(name=name,path=str(path),sha256=C.sha(path),bytes=path.stat().st_size,seconds=blob['fit_s'],reload='PASS')
    C.write(path.with_suffix('.json'),note);print(json.dumps(note),flush=True)
    return note


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--out',type=Path,default=C.SCRATCH/'prefits')
    a=p.parse_args();prefit(a.spec,a.name,a.out)


if __name__=='__main__':main()
