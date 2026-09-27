"""Parity of the explicit stock guard and memo reset; AWM3 and wrist adapters."""
import copy,json,pickle
import numpy as np
from exp.offline_search.harness import api,store
from exp.offline_search.rounds.r03.h3_judge.judge import MixedJudge
from exp.offline_search.rounds.r03.h1_trap.awm3 import AWM3
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM3
from exp.offline_search.rounds.r04.k1_blind.judge import BlindMixedJudge
from exp.offline_search.rounds.r04.k1_blind.wrist import BlindWristAWM
from exp.offline_search.rounds.r04.k3_cost.method import WristAWM
from exp.offline_search.rounds.r04.k1_blind.checks import ROOT,OUT,view,equal


def main():
    counts=dict(stock_guard=0,memo_reset=0,gap_actions=0,awm3=0,wrist=0)
    diagnostics={}
    for scale in (50,500):
        for suite in ('l10','spatial'):
            key='pi05_'+suite
            with open(f'/tmp/k1_blind_fits/mixed_{key}_{scale}.pkl','rb') as f:m=pickle.load(f)['method']
            old=copy.deepcopy(m);old.__class__=MixedJudge
            m.progress_guard='noprog_n'
            reset=copy.deepcopy(m);reset.memo_reset_after_miss=True
            gap=copy.deepcopy(m);gap.progress_guard='noprog_span'
            for arm in ('inf','cache'):
                qc=store.QueryCell(ROOT,f'{key}_{arm}');arrays=api.QueryArrays(qc)
                for epidx in (0,499):
                    e=qc.episodes[epidx]
                    q0=view(qc,arrays,e['start'])
                    for obj in (old,m,reset,gap):obj.reset(q0.episode)
                    for i in range(e['start'],e['end']):
                        q=view(qc,arrays,i)
                        want=old.query(q);got=m.query(q);equal(want,got);counts['stock_guard']+=1
                        equal(want,gap.query(q),extras=False,confidence=False);counts['gap_actions']+=1
                        r=reset.query(q)
                        equal(want,r,extras=False)
                        if q.prev_hit is False:
                            assert r.extras['noprog_n']==0
                            assert int(want.extras['os_flags'])&~8 == int(r.extras['os_flags'])&~8
                            assert all(np.isnan(x[0]) for x in reset._s['prog'][:-1])
                        else:assert want.extras==r.extras
                        counts['memo_reset']+=1
            # Wrist adapter: camera deletion is intentionally different from full AWM,
            # but wrapping WristAWM cannot alter any ordinary vision result.
            with open(f'/tmp/k1_blind_fits/wrist_{key}_{scale}.pkl','rb') as f:w=pickle.load(f)['method']
            original=copy.copy(w);original.__class__=WristAWM
            qc=store.QueryCell(ROOT,key+'_cache');arrays=api.QueryArrays(qc)
            for i in np.linspace(0,qc.N-1,24,dtype=int):
                q=view(qc,arrays,int(i));equal(original.query(q),w.query(q));counts['wrist']+=1
    for key in ('pi05_l10','groot_l10'):
        qc=store.QueryCell(ROOT,key+'_cache');lib=store.LibraryView(ROOT,key,'current')
        ctx=api.Context(root=ROOT,cell=key+'_cache',seed=0,scratch=OUT/'awm3_scratch')
        for kw in ({},{'ridge_main':1.},{'grip_commit':True}):
            a=AWM3(**kw);a.fit(lib,ctx)
            b=BlindAWM3(**kw);b.__dict__.update(a.__dict__);b._fit_blind(lib)
            arrays=api.QueryArrays(qc);e=qc.episodes[0]
            q0=view(qc,arrays,e['start']);a.reset(q0.episode);b.reset(q0.episode)
            for i in range(e['start'],min(e['end'],e['start']+20)):
                q=view(qc,arrays,i);equal(a.query(q),b.query(q));counts['awm3']+=1
    (OUT/'variant_parity.json').write_text(json.dumps(counts,indent=2));print('VARIANT PARITY PASS',counts)

if __name__=='__main__':main()
