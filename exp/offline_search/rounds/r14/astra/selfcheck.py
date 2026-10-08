from common import *
fence()
import copy
import pickle
from types import SimpleNamespace
import numpy as np
from partition import metric_fit
from allocation import settings, replay, prep
from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric
from exp.offline_search.rounds.r10.data import SubsetLibrary
from exp.offline_search.rounds.r03.h3_judge.judge import core

checks={};rng=np.random.default_rng(20261004)
X=rng.normal(size=(81,12));H=rng.normal(size=(81,35));ep=np.repeat(np.arange(9),9)
a=fit_metric(X,H,ep);b=metric_fit(X,H,ep,block=13)
checks['blocked_metric_max_abs']=max(float(np.max(np.abs(u-v))) for u,v in zip(a,b))
assert checks['blocked_metric_max_abs']<1e-10

count=0;zero_cases=0;all_cases=0
for N in (1,2,3,17,40):
    for gu in (0.,.1,1.):
        E=10;steps=np.arange(0,N,2);shape=(len(steps),E)
        p=dict(alive=np.ones(shape,bool),guard=rng.random(shape)<gu,
             step=np.broadcast_to(steps[:,None],shape),risk=np.ones(shape),N=np.full(E,N),task=np.arange(E),init=np.arange(E))
        for K in (0.,.5,2.,7.):
            z=rng.normal(size=10)
            eq=replay(p,'pi05',settings(K,z,gain=0))
            ref=replay(p,'pi05',K)
            assert np.array_equal(eq['calls'],ref['calls']);count+=1
        always=replay(p,'pi05',np.zeros(10));assert always['calls'].all();all_cases+=1
        noextra=replay(p,'pi05',np.full(10,1e6));assert np.array_equal(noextra['calls'],p['guard']);zero_cases+=1
checks.update(zero_gain_equivalences=count,always_call_cases=all_cases,guard_only_cases=zero_cases)

guard=[]
for cell in CELLS:
    model,suite,n=cell.split('_');m=pickle.load((R10/'recipe/artifacts'/f'r10_recipe_{cell}.pkl').open('rb'))['method']
    v=copy.deepcopy(m);j=v.inner;b=j.base
    # Stronger than ordinary pooling: perturb all inactive task tables and
    # their shared thresholds. Only action, guard flags, donor and lifecycle
    # decisions must match; disabled diagnostics/confidence may change.
    j.M0={t:np.zeros_like(x) for t,x in j.M0.items()};j.M1={t:np.zeros_like(x) for t,x in j.M1.items()}
    j.med_len={t:1. for t in j.med_len};j.c_thr=-1.;j.m_thr=1e3
    b.state_scale_by_task={t:np.ones_like(s) for t,s in b.state_scale_by_task.items()}
    b.motion10={t:1e3 for t in b.motion10}
    lib=SubsetLibrary(STORE,f'{model}_{suite}',int(n));examples=0;active=0
    for t in range(10):
        e=min(np.unique(lib.episode[lib.task_id==t]));rows=np.flatnonzero(lib.episode==e)
        for mode in ('dense','gapped','stationary_gapped'):
            initial=core.PseudoQuery(lib,m.inner.C,rows,0,model,prev_hit=True)
            m.reset(initial.episode);v.reset(initial.episode)
            for pos in np.unique(np.linspace(0,len(rows)-1,min(6,len(rows))).astype(int)):
                q=core.PseudoQuery(lib,m.inner.C,rows,int(pos),model,prev_hit=True)
                attrs=('step','task_id','episode','key_v0','key_v1','rs','prev_hit','prev_a_exec',
                       'hist_key_v0','hist_key_v1','hist_rs','hist_hit','hist_a_exec')
                q=SimpleNamespace(**{k:getattr(q,k) for k in attrs})
                q.hist_has_vision=np.ones(q.step,bool)
                if 'gapped' in mode:q.hist_has_vision[1::2]=False
                if mode=='stationary_gapped':q.key_v0=initial.key_v0;q.key_v1=initial.key_v1;q.rs=initial.rs
                r=m.query(q);s=v.query(q)
                assert np.array_equal(r.topk,s.topk) and np.array_equal(r.action,s.action)
                for key in ('os_force_miss','os_flags','os_reason'):assert r.extras[key]==s.extras[key]
                examples+=1;active+=int(r.extras['os_force_miss'])
    assert active>0
    guard.append(dict(cell=cell,proposals=examples,active_guard_proposals=active));print(cell,'guard-null',examples,active,flush=True)
checks['guard_null']=guard
dump(HERE/'selfcheck.json',dict(timestamp=now(),**checks));provenance('selfcheck')
