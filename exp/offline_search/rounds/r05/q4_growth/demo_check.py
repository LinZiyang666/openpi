"""Fresh-process artifact + independent-fit replay through existing K1 plugin tests."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from exp.offline_search.harness import api,store
from exp.offline_search.closed_loop import plugin,selftest
from exp.offline_search.rounds.r05.q4_growth.common import OUT,SHM,COLD,load_base,features,sha256,episode_view
from exp.offline_search.rounds.r05.q4_growth.demo_prepare import RUN,SPEC,arm_name,kwargs


def check(suite,size,variant,tag='final'):
    name=arm_name(suite,size,variant);kw=kwargs(size,variant);path=RUN/'fits'/f'{name}.pkl'
    dest=RUN/'checks'/tag/name
    assert not dest.exists(),f'refusing existing test logs: {dest}'
    plugin._git_head=lambda:None
    # Real CPU orchestrators, fake policy, two interleaved connections, logged
    # blind replay independently FITS the method from the store before queries.
    rc=selftest.main(['--cell',f'pi05_{suite}_cache','--yaml',str(RUN/'config'/f'{name}.yaml'),
                     '--method',SPEC,'--kwargs',json.dumps(kw),'--root',str(SHM),
                     '--fit-artifact',str(path),'--blind','--out',str(dest)])
    assert rc==0
    report=json.loads((dest/'selftest_report.json').read_text())
    replay=json.loads((dest/'verify_blind.json').read_text())
    assert report['PASS'] and replay['PASS'] and report['blind']==24 and report['miss']==0
    with path.open('rb') as f:M=pickle.load(f)['method']
    assert M.serving=='anchor_tail' and M.budget==1 and M.gates=='budget_only' and M.kref==5
    B=load_base(suite);frozen={}
    if variant=='frozen50':
        for field in ('sig','mu0','mu1','muB0','muB1','B0T','B1T','s_a','zmu','zsd','zs_sd'):
            assert np.array_equal(getattr(M,field),getattr(B,field)),field
        for task,T in M.tasks.items():
            for field in ('Wf','shift','W0f','A0','As0','c0','s_d','s_c'):
                assert np.array_equal(getattr(T,field),getattr(B.tasks[task],field)),(task,field)
        frozen=dict(PCA_metric_calibration_unchanged=True,source_fit_sha256=M.frozen_artifact_sha256)
    counts={};reference={}
    for root in (SHM,COLD):
        tag='shm' if root==SHM else 'cold'
        opts,rest=plugin.parse_cli(['--os-root',str(root),'--os-method',SPEC,'--os-kwargs',json.dumps(kw),
            '--os-cell',f'pi05_{suite}_cache','--os-log-dir',str(dest/'load_checks'/tag),'--os-tag','load',
            '--os-fit-artifact',str(path),'--os-blind','--os-no-shadow-native'])
        assert not rest
        rt=plugin.PluginRuntime(opts,'pi05');m=rt.method
        G=store.LibraryView(root,f'pi05_{suite}',f'demo{size}')
        assert rt.fit_info['fit_source']=='artifact' and rt.judge is None and rt.blind
        assert m.demo_manifest_sha256==sha256(G.dir/'manifest.json')
        count=0
        for stream in ('cache','inf'):
            Q=store.QueryCell(root,f'pi05_{suite}_{stream}');A=api.QueryArrays(Q)
            for i,e in enumerate(Q.episodes):
                if e['init']!=49:continue
                ep=episode_view(e,i);m.reset(ep)
                for step in (0,1,5):
                    q=api.QueryView(A,e['start']+step,e['start'],step,e['task_id'],ep)
                    r=m.query(q);api.validate_result(r,lib_sizes=rt.lib_sizes,H=10)
                    assert r.library==G.name and (G.task_id[r.topk]==e['task_id']).all()
                    assert np.array_equal(rt.tables[r.library][r.topk],m.act[r.topk])
                    ident=(stream,e['task_id'],step)
                    if root==SHM:reference[ident]=r
                    else:
                        old=reference[ident]
                        for field in ('topk','scores','action','confidence'):assert np.array_equal(getattr(r,field),getattr(old,field))
                        assert r.extras==old.extras
                    count+=1
        counts[tag]=count
    rec=dict(arm=name,bytes=path.stat().st_size,sha256=sha256(path),artifact=str(path),
             plugin_selftest=report,independent_fit_replay=replay,query_checks_by_root=counts,
             all_ten_tasks=True,early_fresh_stale=True,frozen=frozen,pass_all=True)
    (OUT/'results'/'demo'/f'check_{name}.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True,choices=['l10','spatial']);p.add_argument('--size',type=int,required=True,choices=[100,200,300]);p.add_argument('--variant',required=True,choices=['refit','frozen50']);p.add_argument('--tag',default='final');a=p.parse_args();check(a.suite,a.size,a.variant,a.tag)
