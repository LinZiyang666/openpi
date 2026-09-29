"""Prove pilot A/B code compatibility with the saved deployed-A metric."""
import json
from pathlib import Path
import pickle
from types import SimpleNamespace as NS
import time

import numpy as np

from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.rounds.r06.p1_groot_commit.make_arms import source_rows
from .fit_models import load_source, sha
from .stall import StallModel, StallTracker, _extract_metric, _digest

HERE=Path(__file__).resolve().parent
RUNS=Path('/home/weiland/trace_runs/os_closed_loop')


def main():
    sources=source_rows();reports=[]
    for cell in sorted(json.loads((HERE/'source_manifest.json').read_text())):
        source,blob,awm,lib=load_source(cell)
        model_name,suite,scale=cell.split('_')
        spec=sources[model_name,'B',suite+'_'+scale]['row']
        cls,_=load_method_class(spec['method']);cls(**spec['kwargs'])
        bpath=RUNS/('r05_q1' if model_name=='pi05' else 'r06_paper')/'fits'/(spec['name']+'.pkl')
        with bpath.open('rb') as f:b=pickle.load(f)['method']
        am=_extract_metric(awm,lib.meta);bm=_extract_metric(b,lib.meta)
        assert _digest(am)==_digest(bm),(cell,'B fit metric differs from deployed A')
        saved=StallModel.load(Path('/tmp/q3_stall_fits')/cell)
        assert _digest(am)==saved.provenance['metric_sha256']
        # Raw-key adapter must reproduce the query codes, in both regimes, with
        # exactly the deployed float32 projection arithmetic.
        n=0;raw_times=[];code_times=[];discrepancies=[];other_episode_errors=[];relative_errors=[];max_error_self=[]
        for task,t in awm.tasks.items():
            rows=t.rows
            for row in rows[np.linspace(0,len(rows)-1,min(12,len(rows)),dtype=int)]:
                for step in (0,1):
                    q=NS(task_id=task,step=step,key_v0=lib.key_v0[row],key_v1=lib.key_v1[row],rs=lib.rs[row],prev_hit=True)
                    table,_,regime,_,_,xv,rs,d,*_=awm._dist(q)
                    early=regime==0 and awm.early
                    feat=awm.feat0 if early else awm.features
                    x=np.concatenate([xv,rs]) if feat=='joint' else xv
                    code=x@table.W0f-table.c0 if early else x@table.Wf-table.shift
                    got,mode=saved.encode(q,str(task))
                    assert np.array_equal(got,code),cell
                    assert mode==('early' if early else 'main')
                    a=StallTracker(saved,task);z=StallTracker(saved,task)
                    # A full window measures observe including the raw projection;
                    # synthetic repetition is a timing/parity check, not an SR proxy.
                    for j in range(saved.tasks[str(task)]['W']+1):
                        tick=j*saved.commit_controls
                        t0=time.perf_counter_ns();a.observe(q,tick);raw_times.append((time.perf_counter_ns()-t0)/1e3)
                        t0=time.perf_counter_ns();z.observe({'metric_code':code,'metric':mode},tick);code_times.append((time.perf_counter_ns()-t0)/1e3)
                        assert a.status()==z.status()
                    # Quantify finite precision versus A's float32 norm/dot form;
                    # comparison excludes its optional action-continuity rerank.
                    C=am['tasks'][str(task)]['codes'][mode].astype(float)
                    dist=np.linalg.norm(C-code.astype(float),axis=1)
                    error=np.abs(dist-d)
                    discrepancies.append(float(np.max(error)))
                    j=int(np.argmax(error));max_error_self.append(bool(table.rows[j]==row))
                    other=np.asarray(lib.episode)[table.rows]!=lib.episode[row]
                    other_episode_errors.append(float(np.max(error[other])))
                    relative_errors.append(float(np.max(error[other]/np.maximum(dist[other],np.finfo(float).tiny))))
                    n+=1
        reports.append(dict(cell=cell,A_B_metric_bit_identical=True,saved_metric_matches_A=True,
            source_A_sha256=sha(source['source_artifact']),source_B_sha256=sha(bpath),raw_code_checks=n,
            raw_and_code_status_identical=True,float64_metric_vs_A_float32_distance_max_abs=max(discrepancies),
            other_episode_distance_max_abs=max(other_episode_errors),other_episode_distance_max_relative=max(relative_errors),
            queries_with_max_error_at_self=sum(max_error_self),
            raw_observe_median_us=float(np.median(raw_times)),raw_observe_p99_us=float(np.quantile(raw_times,.99)),
            code_observe_median_us=float(np.median(code_times)),code_observe_p99_us=float(np.quantile(code_times,.99)),
            timing_scope='240 library queries/cell, both regimes, repeated windows; includes warmup; parity/engineering only'))
        print(json.dumps(reports[-1]),flush=True)
    (HERE/'metric_audit.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
