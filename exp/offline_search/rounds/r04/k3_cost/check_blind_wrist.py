"""Fitted batch-four composition: real library queries, blind bypass lifecycle."""
import json,pathlib,pickle,types
import numpy as np
from exp.offline_search.harness import api,store
from exp.offline_search.rounds.r03.h3_judge.judge import core
from exp.offline_search.closed_loop.blind import BlindResult,LookReason
HERE=pathlib.Path(__file__).resolve().parent
FIT=pathlib.Path('/home/weiland/trace_runs/os_closed_loop/r04_cost/fits')
class Blind:
    def __init__(self,q,hit,step,rs,age):
        self.step=step;self.task_id=q.task_id;self.episode=q.episode;self.rs=rs;self.raw_state=rs[:8]
        self.prev_hit=hit;self.prev_a_exec=q.prev_a_exec;self.hist_rs=np.tile(rs,(step,1))
        self.hist_a_exec=np.zeros((step,10,32),np.float32);self.hist_hit=np.ones(step,np.int8)
        self.hist_has_vision=np.ones(step,bool);self.blind_age=age
    def __getattr__(self,name):
        if 'key' in name or 'tok' in name or 'img' in name:raise AssertionError('blind path touched vision')
        raise AttributeError(name)
reports=[]
for suite,short in [('spatial','sp'),('l10','l10')]:
 for lib,scale in [('current','50'),('big','500')]:
    path=FIT/f'r4b4_p_{short}_b2g{scale}_k2_wrist.pkl'
    with path.open('rb') as f:m=pickle.load(f)['method']
    m.prof=api.NULL_PROFILER;L=store.LibraryView('/dev/shm/offline_search_store',f'pi05_{suite}',m.libname)
    rows=next(iter(core.episode_rows(m.C).values()));q=core.PseudoQuery(L,m.C,rows,0,'pi05',None)
    m.reset(q.episode);r=m.blind_step(Blind(q,None,0,q.rs,0));assert isinstance(r,LookReason) and r.code==6
    m.query(q);bq=Blind(q,False,1,L.rs[rows[1]],0);r=m.blind_step(bq);assert isinstance(r,LookReason) and r.code==6
    m.query(q);m.reset(q.episode);r=m.blind_step(Blind(q,True,1,L.rs[rows[1]],0));assert isinstance(r,LookReason) and r.code==6
    # Fixed budget-only mode isolates actual no-vision continuation from optional physical gates.
    m.base.gates='budget_only';m.query(q);r=m.blind_step(Blind(q,True,1,L.rs[rows[1]],0))
    assert isinstance(r,BlindResult),r
    assert len(r.rows)==16 and r.action.shape==(L.H,32)
    reports.append({'suite':suite,'library':m.libname,'step0_look':True,'after_miss_look':True,'reset_look':True,
                    'vision_free_16_member_continuation':True,'artifact':str(path)})
(HERE/'results/blind_wrist_checks.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
