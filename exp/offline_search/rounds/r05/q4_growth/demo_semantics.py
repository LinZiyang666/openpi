"""Frozen incremental representation, K1 calibration, and lifecycle checks."""
import json
import pickle
import numpy as np
from types import SimpleNamespace as NS
from exp.offline_search.harness import api,store
from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.closed_loop.blind import BlindResult,LookReason
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r05.q4_growth.common import OUT,SHM,load_base,features,episode_view
from exp.offline_search.rounds.r05.q4_growth.demo_method import DemoAWM,features
from exp.offline_search.rounds.r05.q4_growth.demo_prepare import RUN,SPEC,arm_name


def main():
    result=[]
    for suite in ('l10','spatial'):
        ctx=api.Context(root=SHM,cell=f'pi05_{suite}_cache',seed=0,scratch=RUN/'semantic_scratch')
        C=ctx.open_library('current');B=load_base(suite)
        previous=None;previous_lib=None;code_diffs=[]
        for size in (100,200,300,500):
            name='bpool_cs' if size==500 else f'demo{size}'
            M=DemoAWM(library=name,variant='frozen50');M.prof=api.NULL_PROFILER;M.fit(C,ctx)
            L=ctx.open_library(name)
            if previous is not None:
                # Stable row IDs map even when the existing 500 store uses its
                # original task-major row order rather than the subset prefixes.
                loc={eid:i for i,eid in enumerate(L.ids)}
                for task,T in previous.tasks.items():
                    new=M.tasks[task];mapped=np.array([loc[previous_lib.ids[r]] for r in T.rows])
                    ix=np.searchsorted(new.rows,mapped)
                    delta=float(np.max(np.abs(T.Z-new.Z[ix])))
                    code_diffs.append(dict(smaller=previous_lib.name,larger=name,task=task,max_abs_diff=delta))
                    assert delta==0,(suite,previous_lib.name,name,task,delta)
            previous,previous_lib=M,L
        load_method_class(SPEC)
        p=RUN/'fits'/f'{arm_name(suite,200,"frozen50")}.pkl'
        with p.open('rb') as f:tail=pickle.load(f)['method']
        ref=BlindAWM(lib='current',kref=5,serving='anchor_tail',budget=1,gates='budget_only')
        ref.model='pi05';ref._fit_blind(C)
        bx=features(B,C)
        for task,T in tail.tasks.items():
            assert np.array_equal(tail.state_scale_by_task[task],ref.state_scale_by_task[task])
            assert tail.motion10[task]==ref.motion10[task]
            for fi,sl in ((0,slice(0,64)),(1,slice(64,128))):
                assert np.array_equal(getattr(T,f'Vm{fi}'),bx[B.tasks[task].rows,sl].astype(np.float64).mean(0).astype(np.float32))
        Q=store.QueryCell(SHM,f'pi05_{suite}_cache');A=api.QueryArrays(Q);e=Q.episodes[0];ev=episode_view(e,0)
        def q(step):return api.QueryView(A,e['start']+step,e['start'],step,e['task_id'],ev)
        def bq(step,age=0,**changes):
            v=q(step)
            x={k:getattr(v,k) for k in ('step','task_id','episode','rs','raw_state','prev_a_exec','hist_a_exec','hist_hit','hist_rs')}
            x.update(prev_hit=None if step==0 else True,hist_has_vision=np.ones(step,bool),blind_age=age)
            x.update(changes);return NS(**x)
        tail.reset(ev);assert isinstance(tail.blind_step(bq(0)),LookReason)
        anchor=tail.query(q(0));blind=tail.blind_step(bq(1))
        assert isinstance(blind,BlindResult)
        assert np.array_equal(blind.action[:5,:7],anchor.action[5:10,:7])
        assert isinstance(tail.blind_step(bq(2,1)),LookReason)
        for changes in ({'prev_hit':False},{'task_id':(e['task_id']+1)%10},{'executed_steps':4},{'blind_age':2}):
            tail.reset(ev);tail.query(q(0));r=tail.blind_step(bq(1,**changes))
            assert isinstance(r,LookReason) and r.code==6 and tail._anchor is None
        tail.reset(ev);assert tail._anchor is None
        result.append(dict(suite=suite,frozen_nested_code_checks=code_diffs,all_shared_codes_bit_exact=True,
                           base_task_centers_unchanged=True,base_blind_calibration_unchanged=True,
                           anchor_tail_full_valid_block_exact=True,lifecycle_cases=7,pass_all=True))
    (OUT/'results'/'demo'/'semantics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
