import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from . import runtime
from .launch import rewrite, bootstrap
from exp.offline_search.rounds.r17.filter_failed.row_filter import r17_candidates


def test_original_row_filter_keeps_fitted_quantities_and_empty_tasks():
    table=NS(rows=np.array([0,1,2]),Z=np.ones((3,2)),z2=np.ones(3),HD=None,h2=None,early=None,Wf=np.eye(2),shift=np.zeros(2))
    old=NS(tasks={0:table},episode=np.array([2,2,3]),pca={'v':np.eye(2)},heads={0:np.ones(2)})
    new=r17_candidates(old,[3])
    assert new.tasks[0].rows.tolist()==[2]
    assert new.tasks[0].Wf is table.Wf and new.pca is old.pca and new.heads is old.heads
    assert not r17_candidates(old,[]).tasks
    assert r17_candidates(old,[2,3]).tasks[0] is table


def test_launcher_rewrites_local_remote_and_mps_requests():
    target=['python','-m','exp.r17_integration.ops.formal_server','--port','5']
    code='RESIDENT_CODE'
    assert rewrite(target,code)==['python','-c',code,'--port','5']
    remote=['env','python','-m','formal_process','spawn','--request',json.dumps({'argv':target})]
    assert json.loads(rewrite(remote,code)[-1])['argv'][1:3]==['-c',code]
    mps=['python','-m','mps','run','--command',json.dumps(target)]
    remote[-1]=json.dumps({'argv':mps})
    assert json.loads(json.loads(rewrite(remote,code)[-1])['argv'][-1])[1:3]==['-c',code]
    assert rewrite(['python','-m','formal_workers'],code)==['python','-m','formal_workers']
    finalizer='from exp.r17_integration.ops.formal_finalize import background; background("run")'
    assert rewrite(['python','-c',finalizer],code,'CPU_EVIDENCE')[-1]=='CPU_EVIDENCE\n'+finalizer


def test_bootstrap_contains_original_filter_and_no_server_exec_on_compile():
    code=bootstrap()
    compile(code,'ffk-bootstrap','exec')
    assert 'r17_candidates' in code and 'filter_failed.row_filter' in code


def test_bootstrap_freeze_leaves_launched_run_untouched(tmp_path):
    import pytest
    from .build import freeze_bootstrap
    (tmp_path/'LAUNCH.json').write_text('{}')
    campaign={}
    with pytest.raises(ValueError,match='launched definition'):
        freeze_bootstrap(tmp_path,campaign)
    assert campaign=={} and not (tmp_path/'assets').exists()


def test_empty_candidates_skip_retrieval_and_loeo_empty_is_a_miss():
    from exp.r17_integration.search import cache_recipe as cr
    from exp.r17_integration.common import serving
    # Install temporarily; production startup installs once per resident.
    with patch.object(cr,'verified_fit'),patch.object(cr,'load_stall'),patch.object(cr.CacheStudyRecipe,'query'),patch.object(cr.CacheStudyRecipe,'force_look'),patch.object(serving.ConnectionPolicy,'infer',serving.ConnectionPolicy.infer):
        runtime.install({})
        request=NS(task_id=0,original_init_idx=1,decision_seq=0)
        base=NS(tasks={},serve_loeo_episodes={})
        obj=NS(base=base)
        q=NS(request=request)
        miss=cr.CacheStudyRecipe.query(obj,q)
        assert miss.force_miss and miss.extras['empty_candidates'] and not miss.extras['knob_call']
        assert not len(miss.rows) and cr.CacheStudyRecipe.force_look(obj,request,[],[])
        base.tasks={0:NS(rows=np.array([0]))};base.episode=np.array([7]);base.serve_loeo_episodes={(0,1):7}
        assert cr.CacheStudyRecipe.query(obj,q).force_miss


def test_empty_task_runs_real_serving_model_path_on_every_decision(tmp_path):
    import torch
    from exp.r17_integration.common.fake_adapter import fake_provider
    from exp.r17_integration.tests.common.conftest import config_for, with_recipe
    from exp.r17_integration.common.config import build_shared_storage
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.common import serving
    from exp.r17_integration.search import cache_recipe as cr
    provider=fake_provider('ffk.empty-task.cpu')
    cfg=with_recipe(provider,config_for(provider,commit=10),tmp_path)
    base=NS(tasks={},serve_loeo_episodes={},reset=lambda episode:None)
    recipe=cr.CacheStudyRecipe(base,cfg)
    storage=build_shared_storage(cfg)
    policy=make_connection_factory(provider,cfg,storage,recipe_factory=lambda c,s:recipe)()
    # Exercise the exact full-system MISS restriction without a fabricated Q3
    # artifact. Empty tasks do not consult a stall model or any retrieval fit.
    cfg.recipe.miss_mode='stall_only'
    with patch.object(cr,'verified_fit'),patch.object(cr,'load_stall'),patch.object(cr.CacheStudyRecipe,'query',cr.CacheStudyRecipe.query),patch.object(cr.CacheStudyRecipe,'force_look',cr.CacheStudyRecipe.force_look),patch.object(serving.ConnectionPolicy,'infer',serving.ConnectionPolicy.infer):
        runtime.install({})
        policy.on_episode_start(task_id=0,episode_id=0,extra_metadata={'task_name':'reach','original_init_idx':0})
        obs={'state':np.arange(4,dtype=np.float32)}
        first=policy.infer(obs)
        second=policy.infer(dict(obs,__execution__={'decision_seq':0,'executed_count':5}))
        assert first['__hit_meta__']['stage_calls']==second['__hit_meta__']['stage_calls']==(1,1,4)
        assert policy.adapter.counts['stage1']==2 and policy.adapter.counts['nfe']==8
        assert not torch.cuda.is_initialized()


def test_paired_init_difference_keeps_successes_on_empty_tasks():
    from exp.offline_search.rounds.r17.filter_failed_knob.analyze import comparison
    key=('mt50','assembly-v3',7)
    result=comparison({key:{'success':True}},{key:{'success':False}},True)
    assert result['paired_n']==1 and result['gains']==1 and result['losses']==0
    assert result['by_init']['7']['task_mean']==result['by_init']['7']['group_mean']==1.


def test_native_evidence_allows_only_the_empty_candidate_miss(tmp_path):
    from .evidence import empty_uids,install
    from exp.r17_integration.ops import cache_study as study,cache_resume
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    labels=tmp_path/'labels.json'
    write(labels,{'metadata':{'episodes':[dict(library_episode_id=1,task_name='reach-v3',orig_init_idx=0)]}})
    c=dict(domain='metaworld',model='pi05',replicas=1,filter_failed_knob=dict(selection={'fit':[]},
           cells={'mt50/250':dict(labels=str(labels))}))
    write(tmp_path/'campaign.json',c)
    write(tmp_path/'assets/CACHE_STUDY.json',{'groups':[dict(suite='mt50',library_size=250,group='crg_koff',fit_digest='fit',periodic_gap=None)]})
    ex=dict(cache_group='crg_koff',library_size=250,task_name='reach-v3',serve_loeo=False,bundle_identity='b',
            recipe_identity='r',commit_horizon=5,miss_mode='stall_only',corrector=True,num_steps=4,periodic_gap=None)
    write(tmp_path/'EPISODE_PLAN.json',{'tasks':[dict(task_uid='uid',experiment='metaworld_mt50',orig_init_state_idx=0,extra=ex)]})
    write(tmp_path/'driver/DRIVER_RECEIPT.json',dict(status='PASS',elapsed_seconds=1))
    (tmp_path/'driver/journal.jsonl').write_text(json.dumps(dict(task_uid='uid',accepted=True,status='done',success=True,attempt=1))+'\n')
    decision=dict(task_uid='uid',accepted=True,attempt=1,source='policy',stage_calls=[1,1,4],actual_nfe=4,
                  bundle_identity='b',recipe_identity='r',commit_horizon=5,capture_enabled=False,decision_seq=0,cursor=0,
                  resident_pid=1,resident_generation='generation')
    ack=dict(task_uid='uid',accepted=True,attempt=1,_kind='execution_ack',decision_seq=0,executed_count=5,
             cursor_end=5,terminal=True,server_terminal_ack=True)
    (tmp_path/'driver/per_step.jsonl').write_text('\n'.join(map(json.dumps,[decision,ack]))+'\n')
    assert empty_uids(tmp_path)=={'uid'}
    with patch.object(study,'aggregate',study.aggregate),patch.object(cache_resume,'segment_resources',return_value={}):
        install()
        result=study.aggregate(tmp_path)
        assert result['status']=='PASS' and result['groups']['crg_koff/metaworld_mt50']['nfe']==4
        c['filter_failed_knob']['selection']['fit']=[1];write(tmp_path/'campaign.json',c)
        result=study.aggregate(tmp_path)
        assert result['status']=='PARTIAL' and result['violations'][0]['reason']=='ordinary MISS forbidden'
        ex['serve_loeo']=True
        write(tmp_path/'EPISODE_PLAN.json',{'tasks':[dict(task_uid='uid',experiment='metaworld_mt50',orig_init_state_idx=0,extra=ex)]})
        assert empty_uids(tmp_path)=={'uid'}
