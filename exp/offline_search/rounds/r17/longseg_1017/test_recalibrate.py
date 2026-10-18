"""Decision scheduler oracle and real empirical Q3 refit, all on CPU."""
import numpy as np
import pytest

from . import recalibrate as r


@pytest.mark.parametrize('gap',[None,0.,.49,1.5,4.75,20.])
def test_vector_scheduler_equals_literal_served_periodic(gap):
    from exp.r17_integration.common.layers.periodic import periodic
    from exp.r17_integration.common.contracts import RequestContext
    eps=[dict(episode=j,task=0,init=j,rows=np.arange(21+j),forced=np.asarray([
        i in (3,7,15) for i in range(21+j)])) for j in range(3)]
    replay=r.DecisionSchedule(eps);n=m=0
    for e in eps:
        vision=[];hit=[]
        for i,forced in enumerate(e['forced']):
            req=RequestContext('cpu','fake_test','task',e['init'],i,task_id=e['task'])
            knob,_=periodic(req,vision,hit,gap)
            call=bool(knob or forced);vision.append(True);hit.append(int(not call));m+=call;n+=1
    assert replay(gap)==(n,n,m)


def test_refit_uses_only_heads_and_preserves_frozen_metric(tmp_path,monkeypatch):
    from exp.r17_integration.tests.ops.test_cache_study import fixture_model,freeze
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel,_digest
    from exp.r17_integration.search.cache_recipe import validate_deployed_stall_metric,CacheStudyRecipe
    from exp.r17_integration.common.query_view import QueryView
    from exp.r17_integration.common.contracts import RequestContext
    provider=fixture_model('dp_libero');cfg,fit=freeze(provider,tmp_path/'source',10,gated=True)
    before=r.sha(tmp_path/'source/stall/stall.pkl')
    monkeypatch.setattr(r,'WORK',tmp_path/'work')
    record=dict(model='dp',suite='libero_spatial',size=50,old_stall=cfg.recipe.stall_path,
        old_stall_digest=cfg.recipe.stall_digest,fit_digest=cfg.recipe.fitted_digest,commit=10,replan=5)
    data=dict(task=fit.task,episode=fit.episode,step=fit.step,success=np.ones(len(fit.step),bool))
    proof=r.fit_q3(record,data);new=StallModel.load(proof['path']);old=StallModel.load(cfg.recipe.stall_path)
    assert new.exec_steps==new.commit_controls==1
    assert new.provenance['clock_unit']=='segment_decision'
    assert _digest(new.metric)==_digest(old.metric)
    assert new.provenance['constants']==old.provenance['constants']
    assert r.sha(tmp_path/'source/stall/stall.pkl')==before
    assert validate_deployed_stall_metric(fit,new)['mode']=='byte_exact'
    assert all(all(int(fit.step[i])%2==0 for i in t['rows']) for v in new.tasks.values() for t in v['templates'])
    recipe=CacheStudyRecipe(fit.fork(),cfg,model=new)
    obs={name:np.arange(width,dtype=np.float32) for name,width in provider.model_spec.feature_spec.fields.items() if name!='robot_state'}
    for seq in range(4):
        request=RequestContext('cpu','fake_test','reach',0,seq,task_id=0)
        q=QueryView(request,obs,np.asarray(fit.state[seq]),[True]*seq,[1]*seq)
        recipe.query(q)
        assert recipe.tracker._last_control==seq
    assert proof['achieved']['windows']>0
    # The actual shared serving path must observe head indices 0..4; all
    # five frozen-worker tail requests are delivery only.
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.common.config import build_shared_storage
    cfg.recipe.stall_path=proof['path'];cfg.recipe.stall_digest=proof['digest']
    policy=make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    policy.on_episode_start(extra_metadata={'task_name':'reach','task_id':0})
    wire=dict(obs,state=np.asarray(fit.state[2]))
    for delivery in range(10):
        result=policy.infer(dict(wire,**({'__execution__':dict(decision_seq=delivery-1,executed_count=5)} if delivery else {})))
        assert policy.recipe.tracker._last_control==delivery//2
        if delivery%2:
            assert result['__hit_meta__']['source'].endswith('_tail')
            assert result['__hit_meta__']['stage_calls']==(0,0,0)


def test_reported_level_ir_counts_heads_and_keeps_latency_separate(monkeypatch):
    from . import analyze
    monkeypatch.setattr(analyze,'prices',lambda *a:(dict(vision=.02,full_miss=.98,reduced_miss=.1),)*2)
    data={('mt50','assembly-v3',0):dict(success=True,cost=dict(policy=2,library=3,controls=45))}
    st=analyze.stats(data,'dp','reduced','mt50','RTX 4090')
    assert st['level_ir']==st['ir_proxy']==2/5
    assert st['decision_heads']==5
    assert st['latency_ir']==st['ir']==pytest.approx((.02*5+.1*2)*5/45)


def test_native_planner_keeps_UIDs_and_records_new_recipe_identities(tmp_path,monkeypatch):
    import copy,json
    from dataclasses import asdict
    from types import SimpleNamespace
    from exp.r17_integration.tests.ops.test_cache_study import fixture_model,freeze
    from exp.r17_integration.common.recipe import bundle_identity,recipe_identity
    from exp.r17_integration.ops import cache_study as study
    from exp.r17_integration.conductor import manifest
    provider=fixture_model('dp_libero');cfg,_=freeze(provider,tmp_path/'fit',10)
    model=asdict(provider.model_spec);assets=tmp_path/'assets';assets.mkdir()
    (assets/'model_spec.json').write_text(json.dumps(model))
    spec=SimpleNamespace(tasks=[('reach','reach')],ordered_tasks=['reach'],identity='fake_spec',formal_indices=tuple(range(50)),
        dev_indices=tuple(range(30)),library_indices=(),task_table_identity='fake_table',
        max_policy_controls=160,protocol_id='fake_spec',benchmark='metaworld_mt50',replan_steps=5,
        seed_for=lambda partition:7)
    monkeypatch.setattr(manifest,'load_benchmark_spec',lambda path:spec)
    monkeypatch.setattr(manifest,'load_plan_config',lambda path:cfg)
    cfg.recipe.periodic_gap=.5
    row=dict(suite='mt50',group='chrg_k50',library_size=250,file='cache.yaml',indices=[0,1],
        sha256='old_yaml',bundle_id='stable_bundle',bundle_identity=bundle_identity(cfg),
        recipe_identity=recipe_identity(cfg),settings=dict(alias=None))
    seal=dict(model=model,benchmark_protocol=spec.identity,eval_pool='A')
    monkeypatch.setattr(study,'selected_groups',lambda c:(seal,[row]))
    monkeypatch.setattr(study,'file_sha',lambda path:row['sha256'])
    c=dict(model='dp',domain='metaworld',local_assets=str(assets),endpoint_host='localhost',
        endpoint_port=23130,public_port=23130,workers=1,run_name='fixed_scientific_plan')
    _,_,before,_=study.plan(c)
    row['task_namespace']=dict(yaml_identity=row['sha256'],bundle_identity=row['bundle_identity'],
        recipe_identity=row['recipe_identity'])
    cfg.recipe.periodic_gap=1.
    row.update(sha256='new_yaml',bundle_identity=bundle_identity(cfg),recipe_identity=recipe_identity(cfg))
    _,_,after,_=study.plan(c)
    assert [t.task_uid for t in before]==[t.task_uid for t in after]
    assert r.scientific_hash(before)==r.scientific_hash(after)
    assert all(t.extra['yaml_identity']=='new_yaml' and t.extra['recipe_identity']==recipe_identity(cfg) for t in after)
    assert all(t.extra['periodic_gap']==1. for t in after)


def test_best_attained_value_is_explicitly_failed_when_outside_tolerance():
    # Seven heads cannot attain 0.1 with integer model calls.
    replay=r.DecisionSchedule([dict(episode=j,task=0,init=j,rows=np.arange(n),forced=np.zeros(n,bool))
        for j,n in enumerate((3,4))])
    result=r.solve(.1,replay)
    assert not result['passed']
    assert result['absolute_error']>.005
    assert result['predicted_ir']==result['M']/result['N']


@pytest.mark.parametrize('filtered',[False,True])
def test_old_original_evidence_keeps_native_audit_without_empty_exemption(tmp_path,filtered):
    # Old frozen studies have no empty-MISS audit branch. The original arm
    # must retain that audit, while a filtered arm may not bypass its check.
    path=tmp_path/'assets/ffk_evidence.py';path.parent.mkdir()
    path.write_text('def install():\n    raise ValueError("old empty-MISS branch absent")\n')
    campaign=dict(filter_failed_knob=dict(selection={'fit':[1]} if filtered else {}))
    r.pin_plan_evidence(tmp_path,campaign)
    namespace={};exec(path.read_text(),namespace)
    if filtered:
        with pytest.raises(ValueError,match='empty-MISS branch absent'):namespace['install']()
    else:
        namespace['install']()
    assert campaign['ffk_evidence_sha256']==r.sha(path)


def test_metadata_stage_refreshes_a_renamed_calibrated_yaml(tmp_path):
    from .staging import metadata_tree
    assets=tmp_path/'assets';assets.mkdir()
    yaml=assets/'full.yaml';yaml.write_text('old recipe')
    stage=metadata_tree(tmp_path)
    old_inode=yaml.stat().st_ino
    yaml.rename(assets/'old.yaml');yaml.write_text('new recipe')
    assert (stage/'full.yaml').stat().st_ino==old_inode
    assert (metadata_tree(tmp_path)/'full.yaml').read_text()=='new recipe'
    assert (assets/'old.yaml').read_text()=='old recipe'
    q3=assets/'q3_decision1018/calibrated';q3.mkdir(parents=True)
    (q3/'stall.pkl').write_bytes(b'new decision-clock Q3')
    frozen=assets/'cache_inputs';frozen.mkdir()
    (frozen/'fit.npz').write_bytes(b'original retrieval fit')
    stage=metadata_tree(tmp_path)
    assert (stage/'q3_decision1018/calibrated/stall.pkl').read_bytes()==b'new decision-clock Q3'
    assert not (stage/'cache_inputs').exists()
