"""Real stored-query tests: both suites/scales, reset, after MISS, camera deletion."""
import json,pathlib,pickle,copy
import numpy as np
from exp.offline_search.harness import api,store
from exp.offline_search.rounds.r04.k3_cost.method import WristView
from exp.offline_search.rounds.r03.h3_judge.judge import core
HERE=pathlib.Path(__file__).resolve().parent
FIT=pathlib.Path('/home/weiland/trace_runs/offline_search_store/derived/r04/k3_cost/fits')

class NoBase:
    def __init__(self,q):self.q=q
    def __getattr__(self,k):
        if k in ('key_v0','hist_key_v0','tok_v0','img0'):raise AssertionError('deleted camera accessed: '+k)
        return getattr(self.q,k)


def snapshot(r):
    return (np.asarray(r.topk).copy(),np.asarray(r.scores).copy(),float(r.confidence),np.asarray(r.action).copy(),dict(r.extras))


def same(a,b):
    return all(np.array_equal(x,y) if isinstance(x,np.ndarray) else x==y for x,y in zip(a,b))


def main():
    reports=[]
    for suite in ('spatial','l10'):
        for lib in ('current','big'):
            path=FIT/f'pi05_{suite}_{lib}.pkl'
            blob=pickle.loads(path.read_bytes());m=blob['method'];m.prof=api.NULL_PROFILER
            C=m.C;L=store.LibraryView('/dev/shm/offline_search_store',f'pi05_{suite}',m.libname)
            er=core.episode_rows(C);eps=list(er)[:2]
            outputs={};n=0;fresh=0
            for order in (eps,list(reversed(eps))):
                for eid in order:
                    rows=er[eid]
                    m.reset(None);got=[]
                    for pos in range(min(8,len(rows))):
                        prev_hit=bool(pos%2) if pos else None
                        q=core.PseudoQuery(L,C,rows,pos,'pi05',prev_hit)
                        r=m.query(NoBase(q));api.validate_result(r,H=L.H,lib_sizes={m.libname:L.L})
                        got.append(snapshot(r));n+=1;fresh+=int(pos>0 and prev_hit is False)
                    if eid in outputs:assert all(same(x,y) for x,y in zip(got,outputs[eid]))
                    outputs[eid]=got
            assert all(t.Z.shape[1]==72 for t in m.base.tasks.values())
            assert m.base.B0T.shape[0]==0
            assert np.array_equal(m.M0[0],m.M1[0])
            reports.append({'suite':suite,'library':m.libname,'episodes':len(er),'entries':L.L,'queries':n,'after_miss':fresh,
                            'reset_bitwise':True,'deleted_camera_never_read':True,'metric_dim':72,
                            'bytes_per_entry':m.bytes_per_entry(),'artifact':str(path),'artifact_bytes':path.stat().st_size,
                            'fit_s':blob['fit_s'],'calibration':m.fit_info['calibration'],'m_thr':m.m_thr,'c_thr':m.c_thr})
    (HERE/'results/method_checks.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
