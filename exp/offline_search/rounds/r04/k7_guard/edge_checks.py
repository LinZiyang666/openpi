"""Focused tests of confirmations, lifecycle, feature coupling and gap invariants."""
import copy
import json
from types import SimpleNamespace as NS
import numpy as np
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r04.k1_blind.checks import view, blind_view, equal
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindResult, LookReason
from exp.offline_search.rounds.r04.k1_blind.judge import BlindMixedJudge as K1
from exp.offline_search.rounds.r04.k7_guard.judge import VisionConfirmedBlindMixedJudge as K7
from exp.offline_search.rounds.r04.k7_guard.evidence import HERE, ROOT, MaskedView, method


def main():
    checks = []
    m = K7(); m.model='pi05'; m.m_thr=1.; m.c_thr=.95
    m.M0={0:np.zeros(2,np.float32)}; m.M1=copy.deepcopy(m.M0)
    def trial(name, mask, states, keys, expected):
        s=len(mask); rs=np.zeros((s+1,32),np.float32);rs[:,0]=states
        k=np.asarray(keys,np.float32)
        hist=k[:-1].copy(); hist[~np.asarray(mask,bool)]=np.nan
        q=NS(step=s,task_id=0,rs=rs[-1],hist_rs=rs[:-1],key_v0=k[-1],key_v1=k[-1],
            hist_key_v0=hist,hist_key_v1=hist.copy(),hist_has_vision=np.asarray(mask,bool))
        assert m.confirmed_stuck(q)==expected, name
        # Mutating missing-key sentinels must not alter the result.
        hist[~np.asarray(mask,bool)]=12345.
        assert m.confirmed_stuck(q)==expected,name
        checks.append(name)
    trial('step0',[],[0],[[1,0]],0)
    trial('no_left_anchor',[False,False],[0,0,0],[[1,0]]*3,0)
    trial('closed_blind_gap',[True,False,False],[0,0,0,0],[[1,0]]*4,3)
    trial('visual_change_breaks_gap',[True,False,False],[0,0,0,0],[[1,0]]*3+[[0,1]],0)
    trial('one_camera_change',[True],[0,0],[[1,0],[0,1]],0)
    trial('motion_resets_inside_gap',[True,False,False],[0,2,2,2],[[1,0]]*4,2)
    trial('motion_threshold_strict',[True],[0,1],[[1,0]]*2,0)
    trial('consecutive_anchor_intervals',[True,False,True,False],[0]*5,[[1,0]]*5,4)
    trial('unbounded_prefix_excluded',[False,False,True,False],[0]*5,[[1,0]]*5,2)
    trial('prior_visual_change_breaks_run',[True,False,True,False],[0]*5,[[0,1],[0,1],[1,0],[1,0],[1,0]],2)
    # One camera alone vetoes confirmation (min, not mean/max).
    rs=np.zeros((2,32),np.float32)
    q=NS(step=1,task_id=0,rs=rs[1],hist_rs=rs[:1],key_v0=np.array([1,0],np.float32),
         key_v1=np.array([0,1],np.float32),hist_key_v0=np.array([[1,0]],np.float32),
         hist_key_v1=np.array([[1,0]],np.float32),hist_has_vision=np.array([True]))
    assert m.confirmed_stuck(q)==0;checks.append('min_camera_conjunct')
    m.c_thr=1.;q.key_v1=np.array([1,0],np.float32)
    assert m.confirmed_stuck(q)==1;checks.append('cosine_threshold_inclusive')
    try: K7().fit(None, NS(model='groot'))
    except api.SkipCell: checks.append('groot_refused_before_fit')
    else: raise AssertionError('GR00T not refused')
    gap_comparisons=0; lifecycle=[]
    for key in ('pi05_l10','pi05_spatial'):
        for scale in (50,500):
            m=method(key,scale); dense=copy.deepcopy(m);dense.__class__=K1
            qc=store.QueryCell(ROOT,key+'_cache');arrays=api.QueryArrays(qc)
            for ei in (0,499):
                e=qc.episodes[ei];q0=view(qc,arrays,e['start'])
                for obj in (m,dense):obj.reset(q0.episode)
                hv=np.zeros(e['end']-e['start'],bool)
                keys=[np.full((len(hv),len(q0.key_v0)),np.nan,np.float32) for _ in range(2)]
                for i in range(e['start'],e['end']):
                    q=view(qc,arrays,i)
                    if q.step%3:continue
                    hv[q.step]=True; keys[0][q.step]=q.key_v0;keys[1][q.step]=q.key_v1
                    masked=MaskedView(q,hv[:q.step],keys)
                    a=m.query(masked);b=dense.query(masked)
                    equal(a,b,extras=False,confidence=False)
                    assert int(a.extras['os_flags'])&~5 == int(b.extras['os_flags'])&~5
                    assert m._noprog_span==dense._noprog_span
                    assert a.extras['stuck_n']==m.confirmed_stuck(masked)
                    # Recompute V7 confidence with captured features from the actual query.
                    features=[]; orig=m._features
                    def capture(*args,**kw):
                        f=orig(*args,**kw);features.append(f.copy());return f
                    m._features=capture
                    # Repeat query: count must be derived from history, not incremented twice.
                    if q.step:
                        again=m.query(masked)
                        assert again.confidence==a.confidence
                        assert features[-1]['stuck']==min(m.confirmed_stuck(masked),5)
                    del m._features
                    gap_comparisons+=1
            q0=view(qc,arrays,qc.episodes[0]['start']);m.reset(q0.episode)
            assert m.blind_step(blind_view(q0)).code==6
            m.query(q0);m.base.gates='budget_only'
            q1=view(qc,arrays,qc.episodes[0]['start']+1)
            result=m.blind_step(blind_view(q1,prev_hit=False))
            assert result.code==6 and result.name=='after_miss'
            m.guards=False; m.query(q1)
            q2=view(qc,arrays,qc.episodes[0]['start']+2)
            assert isinstance(m.blind_step(blind_view(q2)),BlindResult)
            result=m.blind_step(blind_view(q2,task_id=999))
            assert result.code==6
            other=next(e for e in qc.episodes if e['task_id']!=q0.task_id)
            change=view(qc,arrays,other['start'])
            got=m.query(change); fresh=copy.deepcopy(m);fresh.reset(change.episode)
            equal(got,fresh.query(change)); assert got.extras['stuck_n']==0
            lifecycle.append(dict(key=key,scale=scale,first_decision=6,after_miss=6,task_change=6,reset_equal=True))
    result=dict(PASS=True,synthetic_checks=checks,gap_queries=gap_comparisons,lifecycle=lifecycle)
    (HERE/'results/edges.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
