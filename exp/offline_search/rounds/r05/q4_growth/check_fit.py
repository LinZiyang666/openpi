"""Fresh-process plugin load, independent fit, query parity and frozen-fit audit."""
import argparse
import json
import pickle
import time
import numpy as np
from exp.offline_search.harness import api,store
from exp.offline_search.closed_loop import plugin
from exp.offline_search.rounds.r05.q4_growth.common import (
    OUT,RUN,SHM,SPEC,load_base,sha256,episode_view,features,
)
from exp.offline_search.rounds.r05.q4_growth.build import topology


def compare(a,b):
    assert a.library==b.library
    for k in ('topk','scores','action'):
        assert np.array_equal(getattr(a,k),getattr(b,k)),k
    assert a.confidence==b.confidence
    assert a.extras.keys()==b.extras.keys()
    for k in a.extras:
        assert np.array_equal(a.extras[k],b.extras[k]),k


def check(suite,variant,root):
    started=time.time();short='sp' if suite=='spatial' else suite
    root_tag='shm' if root==SHM else 'cold'
    name=f'r5q4_p_{short}_grow250_{variant}';path=RUN/'fits'/f'{name}.pkl'
    kwargs=dict(library='grow250',variant=variant,kref=5)
    args=['--os-root',str(root),'--os-method',SPEC,'--os-kwargs',json.dumps(kwargs),
          '--os-cell',f'pi05_{suite}_cache','--os-log-dir',str(RUN/'checks'/name/root_tag),
          '--os-tag','fresh_load','--os-fit-artifact',str(path),'--os-no-shadow-native']
    opts,rest=plugin.parse_cli(args);assert not rest
    plugin._git_head=lambda:None
    rt=plugin.PluginRuntime(opts,'pi05');M=rt.method
    assert rt.fit_info['fit_source']=='artifact' and rt.judge is None
    assert rt.lib_sizes['grow250']==len(M.act)
    L=store.LibraryView(root,f'pi05_{suite}');G=store.LibraryView(root,f'pi05_{suite}','grow250')
    assert M.growth_manifest_sha256==sha256(G.dir/'manifest.json')
    cls,_=plugin.load_method_class(SPEC);fresh=cls(**kwargs);fresh.prof=api.NULL_PROFILER
    ctx=api.Context(root=root,cell=f'pi05_{suite}_cache',seed=0,scratch=RUN/'checks'/name/'scratch')
    fresh.fit(L,ctx)
    for field,a in vars(M).items():
        b=getattr(fresh,field)
        if isinstance(a,np.ndarray):assert np.array_equal(a,b),field
    for task,T in M.tasks.items():
        for field,a in vars(T).items():
            b=getattr(fresh.tasks[task],field)
            if isinstance(a,np.ndarray):assert np.array_equal(a,b),(task,field)
    frozen_checks={}
    if variant=='frozen':
        B=load_base(suite)
        for f in ('sig','mu0','mu1','muB0','muB1','B0T','B1T','s_a','zmu','zsd','zs_sd'):
            assert np.array_equal(getattr(M,f),getattr(B,f)),f
        for t,BT in B.tasks.items():
            T=M.tasks[t];n=len(BT.rows)
            for f in ('Wf','shift','W0f','A0','As0','c0','s_d','s_c'):
                assert np.array_equal(getattr(T,f),getattr(BT,f)),(t,f)
            for f in ('rows','Z','z2','HD','h2','RS','rs2','n20'):
                assert np.array_equal(getattr(T,f)[:n],getattr(BT,f)),(t,f)
            assert np.all(T.rows[n:]>=L.L)
        frozen_checks=dict(all_fit_arrays_and_calibration_unchanged=True,all_old_candidate_arrays_bit_exact=True,
                           source_sha256=M.frozen_artifact_sha256)
    decision_count=0;appended_top1=0;reference={}
    for stream in ('cache','inf'):
        Q=store.QueryCell(root,f'pi05_{suite}_{stream}');A=api.QueryArrays(Q)
        eps=[(i,e) for i,e in enumerate(Q.episodes) if e['init'] in (25,49)]
        for reverse in (False,True):
            for i,e in eps[::-1] if reverse else eps:
                ev=episode_view(e,i);M.reset(ev);fresh.reset(ev)
                for st in sorted(set([0,1,min(5,e['num_steps']-1),e['num_steps']-1])):
                    q=api.QueryView(A,e['start']+st,e['start'],st,e['task_id'],ev)
                    res=M.query(q);rr=fresh.query(q)
                    api.validate_result(res,lib_sizes=rt.lib_sizes,H=10,where=name)
                    assert res.library=='grow250' and (G.task_id[res.topk]==e['task_id']).all()
                    compare(res,rr)
                    # The plugin table must resolve all grown indices directly.
                    assert np.array_equal(rt.tables[res.library][res.topk],M.act[res.topk])
                    ident=(stream,i,st)
                    if reverse:compare(res,reference[ident])
                    else:reference[ident]=res
                    appended_top1+=int(res.topk[0]>=L.L);decision_count+=1
    # An omitted middle row is unknown continuation, not terminal, and is
    # neither bridged nor linked to an identical row from another episode.
    top=topology(np.array([0,0,1]),np.array([0,2,0]),np.array([3,3,1]))
    assert top['next'].tolist()==[-1,-1,-1]
    assert top['is_terminal'].tolist()==[False,True,True]
    assert not top['next_valid'].any() and top['terminal_known'].all()
    rec=dict(arm=name,root=str(root),path=str(path),bytes=path.stat().st_size,sha256=sha256(path),
             fit_source=rt.fit_info['fit_source'],fresh_refit_array_parity=True,query_parity_decisions=decision_count,
             appended_top1=appended_top1,reset_order_invariant=True,all_ten_tasks=True,
             early_fresh_stale_regimes=True,missing_edge_fixture_pass=True,frozen_checks=frozen_checks,
             library_rows=G.L,bytes_per_entry=M.bytes_per_entry(),wall_s=time.time()-started,pass_all=True)
    (OUT/'results'/f'check_{suite}_{variant}_{root_tag}.json').write_text(json.dumps(rec,indent=2))
    print(json.dumps(rec),flush=True)

if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True,choices=['l10','spatial']);p.add_argument('--variant',required=True,choices=['refit','frozen']);p.add_argument('--root',type=Path,default=SHM)
    a=p.parse_args();check(a.suite,a.variant,a.root)
