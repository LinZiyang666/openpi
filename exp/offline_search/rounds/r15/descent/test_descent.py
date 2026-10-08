"""CPU plugin action/lifecycle equivalence, LOEO and frozen emitter checks."""
from __future__ import annotations
import copy,json,pickle
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from exp.offline_search.closed_loop import devset,plugin
from .prepare import HERE,CELLS,LEVELS,frozen,descent_blob,emit_test_a,install_test_a,check_seal
from .mechanics import Harness,load
from .proof import cpu_fence,lifecycle
from .recipe import R15Knob, hash_chunk
from .verify import verify_records

@pytest.mark.parametrize('ms',('pi05_l10','pi05_spatial','groot_l10','groot_spatial'))
def test_leave_own_out_all_neighbour_paths(ms,tmp_path):
    cell=ms+'_500'
    with cpu_fence():
        h=Harness(ms+'_100',tmp_path/ms)
        try:
            for level in ('off','ir25','ir32','ir40','ir45'):
                enabled=load(cell,level)
                rec=frozen(cell,level,test=True)
                disabled=pickle.loads(Path(rec['path']).read_bytes())['method']
                # A caller with an identity outside B has no own library episode.
                start=h.start
                h.start=lambda conn,task,init=0:start(conn,task,init=999)
                ref=h.run(disabled,steps=8,force_guards=True)
                out=h.run(enabled,steps=8,force_guards=True)
                assert ref[0]==out[0] and lifecycle(ref[2])==lifecycle(out[2])
                h.start=start
                # B init=0 is in the 500 library. Exercise early and stale paths,
                # guard exact chunk lookup, corrector and blind continuation.
                out=h.run(enabled,steps=12,force_guards=True,reverse=True,retries=level!='off')
                assert len(out[2])==120
                conn=h.connection(enabled)
                for task in range(10):
                    h.start(conn,task)
                    for step in range(8):
                        conn.infer(h.obs(task,step))
                        method=conn._osp_sessions[0].method
                        own=method.episode_by_init[(task,0)]
                        assert method._excluded_episode==own
                        assert np.all(method.inner.base.lib_ep[method.inner.base.tasks[task].rows]!=own)
                        rows=np.array(h.logs[-1].get('topk',[]),np.int64)
                        if len(rows):
                            local=np.searchsorted(method.row_subset,rows)
                            assert np.all(method.inner.base.lib_ep[local]!=own)
                        diag=h.logs[-1]['r14']
                        if h.logs[-1]['vision']:
                            assert diag['excluded_episode']==own
                            assert own not in diag['candidate_episode_ids']
                            assert own not in diag['donor_episode_ids']
                        # Exact policy chunk belonging only to own episode cannot match own.
                        ownrows=np.flatnonzero(method.inner.C.ep==own)
                        guard=copy.copy(method.inner.C)
                        guard.L=1
                        guard.horder=ownrows[:1]
                        guard.hsorted=np.array([hash_chunk(guard.act[ownrows[0],:,:7])],np.uint64)
                        matched=guard.find_chunk(guard.act[ownrows[0],:,:7],task)
                        assert matched<0 or method.inner.C.ep[matched]!=own
                    conn.on_episode_end(success=False)
                # A identity with the same integer init must not remove B rows.
                a=copy.deepcopy(enabled)
                from exp.offline_search.harness.api import EpisodeView
                a.reset(EpisodeView('A:test:0:0','',0,0,-1,0))
                assert a._excluded_episode==-1
                assert np.array_equal(a.inner.base.tasks[0].rows,a._full_tasks[0].rows)
            assert R15Knob().leave_own_episode_out is False
        finally:h.rt.flush_all()

@pytest.mark.parametrize('cell',CELLS)
def test_uniform_mixed_emitter_actions(cell,tmp_path):
    with cpu_fence():
        h=Harness(cell,tmp_path/cell)
        try:
            for level in LEVELS:
                blob=descent_blob(cell,{str(t):level for t in range(10)})
                rec=frozen(cell,level,test=True)
                original=pickle.loads(Path(rec['path']).read_bytes())['method']
                ref=h.run(original,tasks=(0,7),steps=8,force_guards=True)
                out=h.run(blob['method'],tasks=(0,7),steps=8,force_guards=True,retries=level!='off')
                assert ref[0]==out[0] and lifecycle(ref[2])==lifecycle(out[2])
            levels={str(t):LEVELS[t%6] for t in range(10)}
            out=h.run(descent_blob(cell,levels)['method'],steps=8,force_guards=True)
            for level in LEVELS:
                tasks=[t for t in range(10) if levels[str(t)]==level]
                rec=frozen(cell,level,test=True)
                ref=h.run(pickle.loads(Path(rec['path']).read_bytes())['method'],tasks=tasks,steps=8,force_guards=True)
                assert {k:v for k,v in out[0].items() if k[0] in tasks}==ref[0]
        finally:h.rt.flush_all()


def test_staging_loading_installation(tmp_path):
    levels={c:{str(t):LEVELS[t%6] for t in range(10)} for c in CELLS}
    uniform={c:'ir32' for c in CELLS}
    with cpu_fence():
        records=emit_test_a(levels,uniform,tmp_path/'stage')
        verify_records(records)
        for rec in records:
            for row in json.loads((Path(rec['root'])/'arms.json').read_text()):
                args=['--os-method',row['method'],'--os-kwargs',json.dumps(row['kwargs']),
                      '--os-cell',row['cell'],'--os-log-dir',str(tmp_path/'loads'/row['arm']),*row['plugin_args']]
                opts,rest=plugin.parse_cli(args);assert not rest
                rt=plugin.PluginRuntime(opts,row['model'])
                try:assert not getattr(rt.method,'leave_own_episode_out',False)
                finally:rt.flush_all()
        targets=install_test_a(tmp_path/'stage',tmp_path/'installed')
        verify_records([check_seal(p) for p in targets])
        with pytest.raises(FileExistsError):install_test_a(tmp_path/'stage',tmp_path/'installed')
        first=Path(records[0]['root'])/'arms.json';first.write_text('[]')
        with pytest.raises(ValueError):install_test_a(tmp_path/'stage',tmp_path/'refused')
        assert not (tmp_path/'refused').exists()

@pytest.mark.parametrize('ms',('pi05_l10','pi05_spatial','groot_l10','groot_spatial'))
def test_leave_own_out_unseen_valid_b_identity(ms,tmp_path):
    model,suite=ms.split('_');cell=ms+'_100'
    _,episodes,_=devset.library_episodes(model,suite,devset.PARENT[model])
    subset=devset.r10_subset(model,suite,100)
    ids={e for es in subset['episode_ids_by_task'].values() for e in es}
    identities={(e['task_id'],e['orig_init_state_idx']):i for i,e in enumerate(episodes) if i in ids}
    unseen={task:next(i for i in range(50) if i not in subset['excluded_inits_by_task'][str(task)]) for task in range(10)}
    with cpu_fence():
        h=Harness(cell,tmp_path/ms)
        try:
            start=h.start;obs=h.obs
            h.start=lambda conn,task,init=0:start(conn,task,init=unseen[task])
            h.obs=lambda task,step,**kwargs:obs(task,step,init=unseen[task],**kwargs)
            for level in ('off','ir25','ir32','ir40','ir45'):
                disabled=load(cell,level);enabled=copy.deepcopy(disabled)
                enabled.__class__=R15Knob;enabled.enable_leave_own_episode_out(identities)
                enabled.validate_leave_own_episode_out(subset)
                ref=h.run(disabled,steps=8,force_guards=True)
                out=h.run(enabled,steps=8,force_guards=True,reverse=True)
                assert ref[0]==out[0] and sorted(lifecycle(ref[2]))==sorted(lifecycle(out[2]))
                assert all(d['r14'].get('excluded_episode',-1)==-1 for d in out[2] if d['vision'])
        finally:h.rt.flush_all()
