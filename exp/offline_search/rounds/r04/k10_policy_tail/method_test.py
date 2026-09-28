"""Inherited HIT action/gate parity and policy-tail method lifecycle."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import pickle
import numpy as np
from exp.offline_search.closed_loop.blind import BlindQueryView,BlindResult,LookReason,policy_tail_chunk
from exp.offline_search.harness import api,store
from exp.offline_search.rounds.r04.k1_blind.checks import view
from exp.offline_search.rounds.r04.k7_guard.evidence import bit_equal
from exp.offline_search.rounds.r04.k10_policy_tail.judge import PolicyTailJudge
B=Path(__file__).resolve().parent
root='/home/weiland/trace_runs/offline_search_store';rows=[]
for suite,scale in [('l10',500),('l10',50),('spatial',500)]:
    short='sp' if suite=='spatial' else suite
    path=Path(f'/home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_{short}_{scale}_tail1ug.pkl')
    with path.open('rb') as f:k7=pickle.load(f)['method']
    k10=copy.deepcopy(k7);k10.__class__=PolicyTailJudge
    qc=store.QueryCell(root,f'pi05_{suite}_cache');arrays=api.QueryArrays(qc)
    counts=dict(vision=0,blind=0,veto=0)
    for ei in (0,49,250,499):
        e=qc.episodes[ei];q0=view(qc,arrays,e['start'])
        k7.reset(q0.episode);k10.reset(q0.episode)
        for offset in range(min(12,e['end']-e['start'])):
            q=view(qc,arrays,e['start']+offset)
            if offset%2==0:
                bit_equal(k7.query(q),k10.query(q));counts['vision']+=1
            else:
                bq=BlindQueryView(q.step,q.task_id,q.episode,q.rs,q.raw_state,True,q.prev_a_exec,q.hist_a_exec,q.hist_hit,q.hist_rs,np.ones(q.step,bool),0)
                x=k7.blind_step(bq);y=k10.blind_step(bq)
                assert type(x)==type(y)
                if isinstance(x,LookReason):assert x==y;counts['veto']+=1
                else:
                    for name in ('action','rows','weights'):assert np.array_equal(getattr(x,name),getattr(y,name))
                    assert x.library==y.library and x.extras==y.extras;counts['blind']+=1
    q0=view(qc,arrays,qc.episodes[0]['start']);q1=view(qc,arrays,qc.episodes[0]['start']+1)
    def primed():
        k10.reset(q0.episode);k10.query(q0);k10.invalidate_anchor()
        return BlindQueryView(1,q1.task_id,q1.episode,q1.rs,q1.raw_state,False,np.asarray(qc.a_inf[qc.episodes[0]['start']],np.float32),q1.hist_a_exec,np.array([0],np.int8),q1.hist_rs,np.array([True]),0)
    bq=primed();assert k10.base._anchor is None
    r=k10.policy_tail_step(bq);assert isinstance(r,BlindResult) and np.array_equal(r.action,policy_tail_chunk(bq.prev_a_exec))
    assert k10.base._anchor is None and isinstance(k10.policy_tail_step(bq),LookReason)
    for delta in (dict(step=0),dict(task_id=999),dict(blind_age=1),dict(prev_hit=True),dict(hist_has_vision=np.array([False])),dict(episode=SimpleNamespace(uid='other'))):
        bq=primed();assert k10.policy_tail_step(replace(bq,**delta)).code==6
    rows.append(dict(suite=suite,scale=scale,fit=str(path),**counts,policy_tail=True,lifecycle_checks=7))
report=dict(PASS=True,rows=rows)
(B/'results/installed/method_test.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
