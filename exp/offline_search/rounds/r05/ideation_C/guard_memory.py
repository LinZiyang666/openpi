"""Read existing K7 fits and count arrays; no refit and no GPU allocation."""
from pathlib import Path
import pickle, hashlib, sys, json
import numpy as np
from exp.offline_search.closed_loop.plugin import load_method_class
OUT=Path(__file__).resolve().parent
REPO=OUT.parents[4]
class Loader(pickle.Unpickler):
    def find_class(self,mod,name):
        if mod.startswith('osm_') and mod not in sys.modules:
            for folder in ('r02/g1_awm','r02/g3_recovery','r03/h3_judge','r04/k1_blind','r04/k7_guard'):
                for p in (REPO/'exp/offline_search/rounds'/folder).glob('*.py'):
                    if 'osm_'+hashlib.md5(str(p).encode()).hexdigest()[:12]==mod:
                        load_method_class(str(p)+':'+name)
        return super().find_class(mod,name)
def visit(o,p='',seen=None):
    if seen is None:seen=set()
    if id(o) in seen:return []
    seen.add(id(o))
    if isinstance(o,np.ndarray):return [(p,o)]
    if isinstance(o,dict):items=o.items()
    elif isinstance(o,(tuple,list)):items=enumerate(o)
    elif hasattr(o,'__dict__'):items=vars(o).items()
    else:return []
    return [q for k,v in items for q in visit(v,f'{p}.{k}',seen)]
result=[]
for suite,scale,arm in [('l10',50,'tail1ug'),('l10',500,'tail1ug'),('sp',500,'tail1ug')]:
    name=f'r4k7_p_{suite}_{scale}_{arm}'
    p=Path('/home/weiland/trace_runs/os_closed_loop/r04_k7/fits')/(name+'.pkl')
    m=Loader(p.open('rb')).load()['method'];aa=visit(m);packed=0;targets=[];row_bytes=0
    for n,v in aa:
        size=v.nbytes
        if v.ndim==3 and v.shape[-1]==32 and v.shape[-2]==10:
            size=size*7//32;targets.append(n)
        packed+=size
        isrow=v.ndim>0 and v.shape[0]==len(m.base.act)
        if '.base.tasks.' in n:
            tid=int(n.split('.base.tasks.')[1].split('.')[0])
            isrow=v.ndim>0 and v.shape[0]==len(m.base.tasks[tid].rows)
        if isrow:row_bytes+=size
    r=dict(arm=name,pickle_bytes=p.stat().st_size,array_bytes=sum(v.nbytes for _,v in aa),
           arrays_valid7_action_bytes=packed,packed_paths=targets,packed_variable_bytes=row_bytes,
           packed_row_bytes=row_bytes/len(m.base.act),packed_fixed_bytes=packed-row_bytes,
           arrays=[dict(path=n,shape=list(v.shape),dtype=str(v.dtype),bytes=v.nbytes) for n,v in aa])
    result.append(r);print(json.dumps({k:v for k,v in r.items() if k!='arrays'}))
(OUT/'guard_memory.json').write_text(json.dumps(result,indent=2))
