"""Scientific invariants, deployment-equation parity and pure planner checks."""
from .common import *
install()
import inspect
import time
import numpy as np
from .retrieval import Metric,lag
from .atlas_reference import AtlasConfig,plan,half_features,label_safe
from exp.offline_search.rounds.r10.astra.offline import Metric as R10Metric

checked=0;max_distance_difference=0.;planner_checks=0
for p in sorted((HERE/'retrieval').glob('*/summary.json')):
    j=json.loads(p.read_text());C,_=load(j['cell'],j['size'])
    with np.load(p.parent/'meta.npz') as z:meta={k:np.array(z[k]) for k in z.files}
    assert len(np.unique(meta['row']))==len(meta['row'])
    for name in ('full','wrist','memory2','transport2'):
        with np.load(p.parent/f'{name}.npz') as z:
            donors=np.array(z['rows']);w=np.array(z['weights'])
            assert np.isfinite(z['action']).all()
        assert (C['ep'][donors]!=meta['ep'][:,None]).all()
        assert (C['fold'][donors]!=meta['fold'][:,None]).all()
        assert (C['task'][donors]==meta['task'][:,None]).all()
        np.testing.assert_allclose(w.sum(1),1,rtol=1e-6,atol=1e-6)
        checked+=len(donors)
    tr=np.flatnonzero(C['fold']!=0);sig=np.maximum(C['act'][tr,:5].std((0,1)),1e-6)
    cr=tr[C['task'][tr]==0];qr=np.flatnonzero((C['fold']==0)&(C['task']==0)&(C['step']>=2))[:24]
    X=np.c_[C['P'],C['rs']]
    a=Metric(X[cr],C['act'][cr],C['ep'][cr],sig).dist(X[qr])
    b=R10Metric(X[cr],C['act'][cr],C['ep'][cr],C['step'][cr],sig,early=False).distances(X[qr],C['step'][qr],early=False)
    diff=float(np.max(np.abs(a-b)));max_distance_difference=max(max_distance_difference,diff)
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=2e-4)
    with np.load(HERE/'correction_dynamic'/p.parent.name/'memory2.npz') as z:
        assert np.isfinite(z['action']).all()

cfg=AtlasConfig();rs=np.zeros((20,8),np.float32);nxt=np.arange(20);safe=np.ones(20,bool)
kw=dict(step=4,last_full_step=2,current_rs=np.zeros(8),last_full_rs=np.zeros(8),
        rows=np.arange(16),weights=np.ones(16),next2=nxt,library_rs=rs,safe=safe,state_scale=np.ones(8))
assert plan(cfg,**kw)['half'];planner_checks+=1
for update in (dict(step=3),dict(last_full_step=0),dict(pending_no_progress=True),dict(periodic_due=True),
               dict(lifecycle_valid=False),dict(next2=np.full(20,-1)),dict(current_rs=np.ones(8)),dict(current_rs=np.full(8,np.nan))):
    assert not plan(cfg,**dict(kw,**update))['half'];planner_checks+=1
f=half_features(last_full_v0=np.ones(64),current_wrist=np.ones(64)*2,current_rs=np.ones(8)*3,last_full_rs=np.ones(8))
assert f.shape==(144,);np.testing.assert_array_equal(f[-8:],np.ones(8)*2);planner_checks+=1
assert 'current_v0' not in inspect.signature(half_features).parameters
assert not any('key' in s or 'image' in s for s in inspect.signature(plan).parameters)
a=np.zeros((2,10,7));b=a.copy();b[1,:,6]=-1
np.testing.assert_array_equal(label_safe(a,b,np.ones(7),.15),[True,False]);planner_checks+=1
start=time.perf_counter()
for _ in range(10000):plan(cfg,**kw)
timing=(time.perf_counter()-start)/10000
readsets=[]
for p in HERE.glob('**/*reads.json'):
    xs=json.loads(p.read_text())
    for s in xs:
        pp=Path(s)
        assert not ('r12' in pp.parts and 'opus' in pp.parts)
        assert not (STORE in pp.parents and STORE/'library' not in pp.parents)
        assert not (RUNS in pp.parents and pp.relative_to(RUNS).parts[0] not in DEV)
    readsets+=xs
result=dict(timestamp=now(),retrieval_rows_checked=checked,max_R10_distance_abs_difference=max_distance_difference,
            planner_checks=planner_checks,planner_mean_cpu_seconds=timing,unique_data_reads=len(set(readsets)),
            cpu_affinity=sorted(os.sched_getaffinity(0)),gpu_used=False,pass_all=True)
dump(HERE/'validation.json',result);print(json.dumps(result,indent=2))
