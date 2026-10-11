"""CPU fixtures exercise real fits/controller/serving; no model weight load."""
import asyncio
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
import yaml

from exp.r17_integration.common.contracts import digest
from exp.r17_integration.common.fake_adapter import fake_provider, fake_spec, fake_sampler
from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.common.recipe import make_recipe_factory, recipe_identity
from exp.r17_integration.common.serving import make_connection_factory, make_bundle_validator
from exp.r17_integration.ops import cache_study as study
from exp.r17_integration.search.cache_recipe import CacheStudyRecipe, metric_mapping, sealed_store_descriptor, verified_fit
from exp.r17_integration.search.fit import fit_recipe, save_fitted_recipe
from exp.r17_integration.search.store import Store, atomic_json
from exp.r17_integration.tests.common.conftest import config_for, with_recipe


def fixture_model(name):
    h, r, views, state, k = {'dp_libero':(16,5,2,8,100),
        'fastwam_libero':(32,5,1,8,10),'cosmos_libero':(16,16,2,9,5)}[name]
    spec = fake_spec(horizon=h,action_dim=7,views=views,model_id=name)
    spec = replace(spec,layout=replace(spec.layout,replan_steps=r,valid_state_indices=tuple(range(state))),
        default_sampler=fake_sampler(k),feature_spec=replace(spec.feature_spec,
        fields={**spec.feature_spec.fields,'robot_state':state}))
    return fake_provider('cache.study.'+uuid.uuid4().hex,spec)


def freeze(provider, tmp_path, c, *, gated=False):
    cfg = with_recipe(provider,config_for(provider,k=provider.model_spec.default_sampler.num_steps,commit=c),
                      tmp_path,corrector=True,steps=8,episodes=6)
    store = Store(cfg.recipe.store_path)
    fitted,receipt = fit_recipe(store,task_names=['reach'],corrector=True,k=cfg.recipe.k,kref=cfg.recipe.kref,
        early=True,guard=False,periodic_gap=None,commit_horizon=c,
        allow_partial_commit=cfg.adapter_binding.get('allow_partial_commit',False))
    ready=save_fitted_recipe(fitted,receipt,tmp_path/'fit')
    cfg.checkpoints={};cfg.recipe.guard=False;cfg.recipe.fitted_path=str(tmp_path/'fit')
    cfg.recipe.fitted_digest=ready['digest'];cfg.recipe.miss_mode='never'
    if gated:
        from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel
        model=StallModel.fit(dict(manifest=dict(H=fitted.layout.horizon,exec_steps=fitted.layout.replan_steps),
            task_id=fitted.task,episode=fitted.episode,step=fitted.step,success=np.ones(len(store.rows),bool)),
            metric=metric_mapping(fitted),commit_controls=min(fitted.layout.horizon,2*fitted.layout.replan_steps))
        model.save(tmp_path/'stall')
        ready=dict(schema='r17.cache_stall.v1',store_digest=cfg.recipe.store_digest,
            fitted_digest=cfg.recipe.fitted_digest,calibration_partition='B_library_successful_LOEO',
            files={p.name:study.file_sha(p) for p in (tmp_path/'stall').iterdir()})
        ready['digest']=digest(ready);atomic_json(tmp_path/'stall/READY.json',ready)
        cfg.recipe.stall_path=str(tmp_path/'stall');cfg.recipe.stall_digest=ready['digest']
        cfg.recipe.miss_mode='stall_only'
    return cfg, fitted


@pytest.mark.parametrize('name',study.MODELS)
def test_sealed_fit_real_layout_cache_corrector_and_tail(name,tmp_path):
    provider=fixture_model(name);c=16 if name=='cosmos_libero' else 10
    cfg,fitted=freeze(provider,tmp_path,c)
    storage=build_shared_storage(cfg);factory=make_connection_factory(provider,cfg,storage)
    obs={'state':np.asarray(fitted.state[2],np.float32),**{n:np.arange(w,dtype=np.float32)
        for n,w in provider.model_spec.feature_spec.fields.items() if n!='robot_state'}}
    policy=factory();policy.on_episode_start(extra_metadata={'task_name':'reach','task_id':0})
    first=policy.infer(obs);meta=first['__hit_meta__']
    assert meta['source']=='library' and meta['actual_nfe']==0 and meta['stage_calls']==(1,0,0)
    assert meta['corrector_enabled'] and not meta['corrector_applied']
    assert policy.adapter.counts['inverse']==1
    count=provider.model_spec.layout.replan_steps
    next_=policy.infer(dict(obs,__execution__={'decision_seq':0,'executed_count':count}))
    assert next_['__hit_meta__']['source']==('library' if c==count else 'library_tail')
    assert next_['__hit_meta__']['actual_nfe']==0
    assert policy.recipe.controls==count
    if c>count:
        assert next_['actions'].tobytes()==first['actions'][count:].tobytes()
        third=policy.infer(dict(obs,__execution__={'decision_seq':1,'executed_count':c-count}))
        assert third['__hit_meta__']['corrector_applied']
        assert third['__hit_meta__']['executed_controls']==c
    else:assert next_['__hit_meta__']['corrector_applied']
    cfg_off=copy.deepcopy(cfg);cfg_off.recipe.corrector=False
    other=make_connection_factory(provider,cfg_off,build_shared_storage(cfg_off))()
    other.on_episode_start(extra_metadata={'task_name':'reach','task_id':0})
    np.testing.assert_array_equal(other.infer(obs)['actions'],first['actions'])
    assert not other.recipe.base.use_corrector
    assert other.recipe.base.action is policy.recipe.base.action
    with pytest.raises(ValueError,match='tasks/settings/corrector/commit'):
        bad=copy.deepcopy(cfg);bad.adapter_binding['commit_horizon']=5 if c!=5 else 10
        verified_fit(bad,Store(cfg.recipe.store_path).manifest['binding'])


@pytest.mark.parametrize('mode,reason',[('never','forbidden'),('stall_only','forbidden')])
def test_pure_cache_rejects_accidental_model_fallback(mode,reason,tmp_path):
    provider=fixture_model('dp_libero');cfg,fitted=freeze(provider,tmp_path,5,gated=mode=='stall_only')
    from exp.r17_integration.common.query_view import RecipeResult
    class Broken:
        def reset(self,*args):pass
        def force_look(self,*args):return True
        def query(self,q):return RecipeResult(np.zeros((16,7),np.float32),np.array([0]),np.array([1.]),np.array([0.]),force_miss=True)
    policy=make_connection_factory(provider,cfg,build_shared_storage(cfg),recipe_factory=lambda *args:Broken())()
    policy.on_episode_start(extra_metadata={'task_name':'reach','task_id':0})
    with pytest.raises(ValueError,match='ordinary MISS|MISS disabled|forbidden'):
        policy.infer({'state':np.arange(8,dtype=np.float32)})
    assert policy.adapter.counts['nfe']==0 and policy.adapter.counts['stage2']==0


def test_original_q3_confirmed_ambiguous_and_one_fresh_anchor_cooldown(tmp_path):
    provider=fixture_model('dp_libero');cfg,_=freeze(provider,tmp_path,5,gated=True)
    base=verified_fit(cfg,Store(cfg.recipe.store_path).manifest['binding']).fork()
    from exp.r17_integration.common.query_view import QueryView
    from exp.r17_integration.common.contracts import RequestContext
    class Tracker:
        def __init__(self,*args):self.index=-1;self.controls=[]
        def observe(self,code,control):self.index+=1;self.controls.append(control)
        def status(self):return dict(state=['slow_confirmed','slow_confirmed','slow_ambiguous','slow_ambiguous','slow_confirmed'][self.index],W=2)
    recipe=CacheStudyRecipe(base,cfg,model=SimpleNamespace(commit_controls=10),tracker_class=Tracker)
    values=[]
    for step in range(5):
        q=QueryView(RequestContext('cpu','fake_mw.v1','reach',0,step,task_id=0),
            {n:np.zeros(w,np.float32) for n,w in provider.model_spec.feature_spec.fields.items()},np.zeros(8,np.float32),
            [True]*step,[1]*step)
        if step:recipe.on_execution(5)
        values.append(recipe.query(q))
        assert recipe.query(q).force_miss==values[-1].force_miss
    assert [r.force_miss for r in values]==[True,False,False,False,True]
    assert [r.extras['stall_extra_look'] for r in values]==[False,False,True,False,False]
    assert recipe.tracker.controls==[0,5,10,15,20]
    recipe.reset(None);assert recipe.controls==0 and recipe.anchor==0 and recipe.tracker is None


def test_library_and_fit_mutation_fences(tmp_path):
    provider=fixture_model('dp_libero');cfg,_=freeze(provider,tmp_path,5)
    factory=make_recipe_factory(provider)
    factory(cfg,None)
    path=Path(cfg.recipe.fitted_path)/'fit.npz';path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='changed'):factory(cfg,None)
    parent=Path(cfg.recipe.store_path);p=parent/'READY.json';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='metadata changed'):sealed_store_descriptor(parent,cfg.recipe.store_digest)


def test_legacy_recipe_identity_keeps_old_default_fields():
    provider=fixture_model('dp_libero');cfg=config_for(provider)
    old=asdict(cfg.recipe)
    for key in ('miss_mode','fitted_path','fitted_digest','stall_path','stall_digest','library_benchmark','serve_loeo'):old.pop(key)
    old.pop('store_path');old['fit_identity']=None
    assert recipe_identity(cfg)==digest(old)


def test_model_code_change_requires_exact_owner_compatibility_receipt(tmp_path):
    from exp.r17_integration.search.cache_recipe import validate_teacher
    from exp.r17_integration.common.recipe import retrieval_model_identity
    provider=fixture_model('dp_libero');cfg,_=freeze(provider,tmp_path,5)
    binding=Store(cfg.recipe.store_path).manifest['binding']
    provider=replace(provider,model_spec=replace(provider.model_spec,code_digest='changed-producer'))
    with pytest.raises(ValueError,match='model-owner library_validator'):validate_teacher(provider,cfg,binding)
    provider=replace(provider,library_validator=lambda *args:dict(status='PASS',evidence=['fixture']))
    with pytest.raises(ValueError,match='missing identities/evidence'):validate_teacher(provider,cfg,binding)
    provider=replace(provider,library_validator=lambda *args:dict(schema='r17.teacher_compatibility.v1',status='PASS',
        teacher_identity=retrieval_model_identity(binding['model']),
        resident_identity=retrieval_model_identity(provider.model_spec),evidence=['CPU fixture; not qualification']))
    assert validate_teacher(provider,cfg,binding)['status']=='PASS'


def test_official_a_bridge_refuses_changed_physics():
    from exp.r17_integration.search.cache_recipe import validate_protocol_bridge
    from exp.r17_integration.common.benchmarks import register_benchmark
    from exp.r17_integration.ops.formal_eval import benchmark
    source,serving=benchmark(),benchmark(True)
    register_benchmark(serving)
    config=SimpleNamespace(adapter_binding={'benchmark_protocol':serving.identity},recipe=SimpleNamespace(
        library_benchmark=dict(source=source.manifest(),serving=serving.manifest())))
    assert validate_protocol_bridge(config,{'benchmark_protocol':source.identity})
    config.recipe.library_benchmark['source']['seed']+=1
    with pytest.raises(ValueError,match='bridge identity mismatch|native benchmark semantics'):
        validate_protocol_bridge(config,{'benchmark_protocol':source.identity})


def test_actual_48_yaml_semantics_and_fail_closed_artifact_prerequisites():
    manifest=study.validate_manifest(study.HERE/'yaml/STUDY.json')
    assert len(manifest['models'])==3 and sum(len(x) for x in manifest['models'].values())==48
    for model,rows in manifest['models'].items():
        for row in rows:
            doc=yaml.safe_load((study.HERE/'yaml'/row['file']).read_text())
            assert len(doc['recipe']['ordered_tasks'])==40
            assert doc['recipe']['guard'] is False and doc['recipe']['periodic_gap'] is None
            assert doc['checkpoints']=={} and doc['recipe']['target_ir'] is None
            assert doc['recipe']['corrector']==('r' in row['group'])
            assert doc['recipe']['miss_mode']==('stall_only' if 'g' in row['group'] else 'never')
            assert doc['adapter_binding']['commit_horizon']==row['settings']['commit']
            if model=='cosmos_libero':assert row['settings']['commit']==16


def test_worker_evidence_is_default_off_and_requires_real_final_ack(monkeypatch):
    from exp.r17_integration.ops.cache_study_worker import evidence,extend_cost_row
    from exp.r17_integration.benchmarks import execution
    result={'__hit_meta__':{'stall_call':True,'resident_pid':42}}
    original=execution.cost_row
    assert extend_cost_row(original)(result,{},0)==original(result,{},0)
    assert extend_cost_row(original)(result,{'cache_study':True},0)['resident_pid']==42
    last=dict(cache_study=True,decision_seq=3,executed_count=2,cursor_end=7,terminal=False)
    client=SimpleNamespace(episode_end=lambda **kwargs:{'execution':dict(decision_seq=3,executed_count=2,cursor_end=7,terminal=True)})
    with evidence():execution.end_episode(client,{},[last])
    assert last['server_terminal_ack']['terminal'] and execution.cost_row is original
    with evidence(),pytest.raises(ValueError,match='terminal execution ACK'):
        execution.end_episode(SimpleNamespace(episode_end=lambda **kwargs:{}),{},[last])


@pytest.mark.parametrize('model,max_batch', [('dp_libero',8),('fastwam_libero',16),('cosmos_libero',1)])
def test_cache_launcher_keeps_control_topology_and_driver(model,max_batch):
    from exp.r17_integration.ops.formal_eval import parse_args
    argv=['--model',model,'--cache-study',str(study.HERE/'yaml/STUDY.json'),'--official-50',
        '--replicas','2','--max-batch',str(max_batch),'--batch-wait-ms','5',
        '--port','23400','--driver-port','23419','--prepare-only']
    if model!='cosmos_libero':argv+=['--mps']
    args=parse_args(argv)
    assert args.driver_mode==('legacy' if model=='cosmos_libero' else 'group-commit')
    assert args.replicas==2 and args.max_batch==max_batch


@pytest.mark.parametrize('model',study.MODELS)
def test_socket_all_eight_yaml_rebinds_without_model_reload(model,tmp_path,monkeypatch):
    from openpi.serving import websocket_policy_server as ws
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from openpi_client import msgpack_numpy
    import websockets.asyncio.client
    provider=fixture_model(model);native=provider.model_spec.layout.replan_steps
    long=16 if model=='cosmos_libero' else 10
    a,_=freeze(provider,tmp_path/'short',native,gated=True)
    b,_=freeze(provider,tmp_path/'long',long,gated=True)
    base=copy.deepcopy(a);base.recipe.enabled=False
    from exp.r17_integration.common.config import RecipeConfig
    base.recipe=RecipeConfig()
    factory=make_connection_factory(provider,base,build_shared_storage(base))
    configs={}
    for group in study.GROUPS:
        cfg=copy.deepcopy(b if 'h' in group else a)
        cfg.recipe.corrector='r' in group
        if 'g' not in group:
            cfg.recipe.miss_mode='never';cfg.recipe.stall_path=None;cfg.recipe.stall_digest=None
        configs[group]=cfg
    for key,value in (('_bundles',{}),('_current_bundle',None),('_bundle_version',0)):monkeypatch.setattr(ws,key,value)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    async def run():
        ready=asyncio.Event();server=WebsocketPolicyServer(object(),host='127.0.0.1',port=port,
            concurrent=True,connection_policy_factory=factory,bundle_validator=make_bundle_validator(provider),
            ready_callback=ready.set,metadata={'fixture':model})
        task=asyncio.create_task(server.run(stop_on_request=True));await asyncio.wait_for(ready.wait(),5)
        packer=msgpack_numpy.Packer();lifetimes=set()
        async def call(client,payload):
            await client.send(packer.pack(payload));result=await asyncio.wait_for(client.recv(),15)
            assert isinstance(result,bytes),result
            return msgpack_numpy.unpackb(result)
        try:
            async with websockets.asyncio.client.connect(f'ws://127.0.0.1:{port}',compression=None) as client:
                await client.recv()
                for group,cfg in configs.items():
                    ack=await call(client,dict(__ctrl__='load_cache_config',bundle_id=group,yaml_content=yaml.safe_dump(asdict(cfg))))
                    assert ack['__ack__']=='load_cache_config',ack
                    assert (await call(client,dict(__ctrl__='select_bundle',bundle_id=group)))['__ack__']=='select_bundle'
                    await call(client,dict(__ctrl__='episode_start',__task__='reach',__extra__={'task_id':0}))
                    obs={'state':np.arange(len(provider.model_spec.layout.valid_state_indices),dtype=np.float32)}
                    out=await call(client,obs);hit=out['__hit_meta__'];assert hit['source']=='library' and hit['actual_nfe']==0
                    lifetimes.add((hit['resident_pid'],hit['resident_generation']))
                    end=await call(client,dict(__ctrl__='episode_end',__execution__={'decision_seq':0,'executed_count':2,'terminal':True}))
                    assert end['execution']['executed_count']==2 and end['execution']['terminal']
                assert len(lifetimes)==1
        finally:server.request_stop();await asyncio.wait_for(task,5)
    asyncio.run(run())


@pytest.mark.parametrize('model,expected',[('dp_libero',8000),('fastwam_libero',8000),('cosmos_libero',4000)])
def test_full_a500_plan_and_smoke_32_keep_cosmos_aliases(model,expected,tmp_path,monkeypatch):
    from exp.r17_integration.common.provider import AdapterProvider,register_provider
    from exp.r17_integration.common.config import load_cache_config
    from exp.r17_integration.common.recipe import bundle_identity
    manifest=study.validate_manifest(study.HERE/'yaml/STUDY.json')
    rows=copy.deepcopy(manifest['models'][model]);groups=[]
    for row in rows:
        parent=study.read(Path(row['library'])/'READY.json')
        source,bench=study.profiles(model,parent)
        spec=replace(study.model_spec(parent['binding']['model']),allowed_benchmark_protocols=(bench.identity,))
        p=AdapterProvider('teacher_template_'+model,spec,lambda:None,lambda k,s=spec:s.default_sampler)
        # Identical templates share one provider; no live model is loaded.
        from exp.r17_integration.common.provider import get_provider
        try:get_provider(p.provider_id)
        except ValueError:register_provider(p)
        config=load_cache_config(study.HERE/'yaml'/row['file'])
        row.update(bundle_id=model+row['suite']+row['group'],bundle_identity=bundle_identity(config),
                   recipe_identity=recipe_identity(config))
        groups.append(row)
    model_doc=asdict(spec)
    atomic_json(tmp_path/'model_spec.json',model_doc);atomic_json(tmp_path/'benchmark.json',bench.manifest())
    seal=dict(model=model_doc,benchmark_protocol=bench.identity,groups=groups)
    seal['digest']=digest(seal);atomic_json(tmp_path/'CACHE_STUDY.json',seal)
    campaign=dict(model=model.rsplit('_',1)[0],local_assets=str(tmp_path),endpoint_host='127.0.0.1',
                  public_port=23400,workers=2,run_name='cpu',smoke=False)
    # Plan verifies YAML bytes/identity without admitting model compatibility;
    # those checks were performed before a real CACHE_STUDY seal is published.
    for row in groups:
        target=tmp_path/row['file'];target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((study.HERE/'yaml'/row['file']).read_bytes())
    _,_,tasks,endpoints=study.plan(campaign)
    assert len(tasks)==expected and len({t.task_uid for t in tasks})==expected
    assert {t.task_id for t in tasks}==set(range(10))|set(range(30,40))
    assert {t.orig_init_state_idx for t in tasks}==set(range(50))
    assert all(t.extra['init_pool']=='A' and t.extra['partition']=='test' for t in tasks)
    assert len(endpoints)==1 and all(endpoints[0].accepts(t) for t in tasks)
    campaign['smoke']=True
    _,_,smoke,_=study.plan(campaign)
    assert len(smoke)==32 and {t.extra['cache_group'] for t in smoke}==set(study.GROUPS)


def test_row_view_rejects_partial_episode_even_with_valid_seals(tmp_path):
    provider=fixture_model('dp_libero');cfg,_=freeze(provider,tmp_path,5)
    parent=study.read(Path(cfg.recipe.store_path)/'READY.json')
    for row in parent['rows']:row['library_episode_id']=int(row['task_uid'].split(':')[-1])
    parent['content_digest']=digest({k:v for k,v in parent.items() if k!='content_digest'})
    atomic_json(Path(cfg.recipe.store_path)/'READY.json',parent)
    view=tmp_path/'library/subsets/size_1';view.mkdir(parents=True)
    np.save(view/'row_indices.npy',np.arange(7,dtype=np.int64))
    manifest=dict(schema='r17.row_index_view.v1',parent_path='../..',parent_digest=parent['content_digest'],
        binding=parent['binding'],size=1,rows=7,episode_ids=[0],row_indices_sha256=study.file_sha(view/'row_indices.npy'))
    manifest['content_digest']=digest(manifest);atomic_json(view/'VIEW.json',manifest)
    with pytest.raises(ValueError,match='episode identity'):sealed_store_descriptor(view,manifest['content_digest'])


def test_generic_batch_audit_accounts_hit_tail_and_stall_call(tmp_path):
    from exp.r17_integration.models.dp.batch_runtime import BatchConfig
    config=BatchConfig(8,5).metadata();rows=[]
    for source,nfe,stages in [('library',0,['stage1']),('library_tail',0,[]),('policy',100,['stage1','stage2','stage3'])]:
        request={'decision_seq':len(rows),'task_name':'reach'}
        entries={stage:dict(stage=stage,size=1,row=0,config_identity=config['identity'],
            members=[dict(request_id='cpu'+stage,request=request)]) for stage in stages}
        rows.append(dict(source=source,nfe=nfe,**request,batching=dict(config=config,stages=entries)))
    target=tmp_path/'replica_0/serving.jsonl';target.parent.mkdir()
    target.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert study.audit_batches({'dp_batching':config},tmp_path)['decisions']==3
    rows[1]['batching']['stages']=rows[0]['batching']['stages']
    target.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='wrong native stages'):study.audit_batches({'dp_batching':config},tmp_path)


def test_cpu_archive_contains_only_original_q3_dependencies(tmp_path):
    import tarfile
    import subprocess
    import os
    from exp.r17_integration.qualification.deploy_cpu import runtime_archive
    archive=tmp_path/'runtime.tar';runtime_archive(study.ROOT,archive);study.append_stall_runtime(archive,study.ROOT)
    with tarfile.open(archive) as tar:
        names=tar.getnames();original=[n for n in names if '/rounds/r06/' in n]
        assert len(original)==3 and all((study.ROOT/n).read_bytes()==tar.extractfile(n).read() for n in original)
        assert not any(n.endswith(('fit.npz','stall.pkl','.yaml')) for n in names)
        tar.extractall(tmp_path/'runtime')
    env=dict(os.environ,PYTHONPATH=str(tmp_path/'runtime'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    code='from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel; from exp.r17_integration.ops.cache_study_worker import evidence'
    result=subprocess.run([__import__('sys').executable,'-c',code],env=env,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('damage',[None,'miss','ack','corrector','restart'])
def test_summary_requires_pure_cache_final_ack_and_resident_lifetime(damage,tmp_path):
    directory=tmp_path;driver=directory/'driver';driver.mkdir()
    campaign=dict(cache_study=True,model='dp',smoke=True,replicas=1)
    task=dict(task_uid='cpu',experiment='libero_spatial',extra=dict(cache_group='c',miss_mode='never',
        corrector=False,num_steps=100,bundle_identity='bundle',recipe_identity='recipe',commit_horizon=5))
    atomic_json(directory/'campaign.json',campaign);atomic_json(directory/'EPISODE_PLAN.json',dict(tasks=[task]))
    journal=dict(task_uid='cpu',attempt=1,status='done',accepted=True,success=True)
    row=dict(task_uid='cpu',attempt=1,accepted=True,source='library',decision_seq=0,cursor=0,
        stage_calls=[1,0,0],actual_nfe=0,bundle_identity='bundle',recipe_identity='recipe',commit_horizon=5,
        corrector_enabled=False,corrector_applied=False,miss_mode='never',capture_enabled=False,
        resident_pid=17,resident_generation='same')
    ack=dict(task_uid='cpu',attempt=1,accepted=True,_kind='execution_ack',decision_seq=0,
        executed_count=2,cursor_end=2,terminal=True,server_terminal_ack={'terminal':True})
    if damage=='miss':row.update(source='policy',stage_calls=[1,1,100],actual_nfe=100)
    if damage=='ack':ack.pop('server_terminal_ack')
    if damage=='corrector':row['corrector_applied']=True
    if damage=='restart':row['resident_generation']=None
    (driver/'journal.jsonl').write_text(json.dumps(journal)+'\n')
    (driver/'per_step.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(ack)+'\n')
    atomic_json(driver/'DRIVER_RECEIPT.json',dict(status='PASS',elapsed_seconds=1))
    result=study.aggregate(directory)
    assert result['status']==('PASS' if damage is None else 'PARTIAL')


@pytest.mark.parametrize('model,maximum',[('dp',8),('fastwam',16)])
def test_model_owned_batch_runtime_cache_stall_and_tails(model,maximum,tmp_path,monkeypatch):
    """Real owned adapters/native scheduler/S07 on small CPU model fixtures."""
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.common.layers import LibraryData,LayoutRecipe
    from exp.r17_integration.common.config import RecipeConfig
    from exp.r17_integration.common.benchmarks import register_benchmark,task_table_identity
    from exp.r17_integration.search.store import write_store
    from exp.r17_integration.debug.validate import model_binding
    monkeypatch.setattr(registry,'_providers',{})
    # Trigger-state injection exercises native gate sampling deterministically;
    # separate tests above verify SHA-bound real Q3 artifacts and thresholds.
    monkeypatch.setattr('exp.r17_integration.search.cache_recipe.load_stall',
        lambda *args:SimpleNamespace(commit_controls=10))
    if model=='dp':
        from exp.r17_integration.tests.models.dp.conftest import make_adapter,observation
        from exp.r17_integration.tests.models.fastwam.conftest import fixture_benchmark_spec
        from exp.r17_integration.models.dp.provider import register_provider as register,make_connection_factory as make
        from exp.r17_integration.models.dp.batch_runtime import create_runtime
        from exp.r17_integration.models.dp.serve import plain_config
        benchmark=replace(fixture_benchmark_spec('libero'),wire_size=96)
        adapter=make_adapter.__wrapped__()('libero',libero_benchmark_specs=[asdict(benchmark)])
        runtime=create_runtime(adapter,maximum,5,tmp_path/'ledger',force=True)
        provider=register(adapter,provider_id='cpu_cache_'+tmp_path.name)
        cfg=plain_config(provider,commit_horizon=10)
        make_factory=lambda c,s,rf:make(adapter,provider=provider,default_config=c,default_storage=s,
                                       stage_core=runtime.core,recipe_factory=rf)
    else:
        from exp.r17_integration.tests.models.fastwam.conftest import make_adapter,observation,fixture_benchmark_spec
        from exp.r17_integration.models.fastwam.provider import register_provider as register,make_connection_factory as make
        from exp.r17_integration.models.fastwam.batch_runtime import create_runtime
        from exp.r17_integration.tests.models.fastwam.test_fastwam_serving import config_for as fast_config
        benchmark=fixture_benchmark_spec('libero')
        adapter=make_adapter('libero',allow_unqualified_batch=False)
        runtime=create_runtime(adapter,maximum,5,tmp_path/'ledger')
        provider=register(adapter,provider_id='cpu_cache_'+tmp_path.name,coordinator=runtime.core)
        cfg=fast_config(provider,k=10,commit=10)
        make_factory=lambda c,s,rf:make(provider,c,s,recipe_factory=rf)
    try:
        spec=provider.model_spec;protocol=spec.allowed_benchmark_protocols[0]
        register_benchmark(benchmark,protocol_identity=protocol)
        tasks=list(benchmark.to_contract().ordered_tasks)
        rng=np.random.default_rng(99);n=24
        lib=LibraryData({name:rng.normal(size=(n,width)).astype(np.float32)
            for name,width in spec.feature_spec.fields.items() if name!='robot_state'},
            rng.normal(size=(n,len(spec.layout.valid_state_indices))).astype(np.float32),
            rng.normal(size=(n,spec.layout.horizon,spec.layout.action_width)).astype(np.float32),
            np.zeros(n,np.int64),np.repeat(np.arange(6),4),np.tile(np.arange(4),6))
        rows=[dict(split='library',campaign='cpu',arm='teacher',bundle_generation='cpu',benchmark_protocol=protocol,
            task_name=tasks[0],task_uid=str(lib.episode[i]),attempt=1,decision_seq=int(lib.step[i]),
            orig_init_idx=10+int(lib.episode[i]),task_data_hash='cpu') for i in range(n)]
        store=write_store(tmp_path/'library',model_binding(spec,protocol),rows,
            {'key_'+k:v for k,v in lib.features.items()}|{'state':lib.state,'normalized_actions':lib.action})
        cfg.enabled=True;cfg.checkpoints={}
        cfg.recipe=RecipeConfig(enabled=True,store_path=str(store.directory),store_digest=store.manifest['content_digest'],
            teacher_sampler_identity=spec.default_sampler.identity,corrector=True,guard=False,
            ordered_tasks=tasks,task_table_identity=task_table_identity(protocol),miss_mode='stall_only',
            stall_path='injected_cpu_tracker',stall_digest='cpu')
        base=LayoutRecipe(spec.layout,len(tasks),commit_horizon=10,corrector=True,guard=False).fit(lib)
        class Tracker:
            def __init__(self,*args):self.index=-1
            def observe(self,*args):self.index+=1
            def status(self):return {'state':'slow_confirmed' if self.index==1 else 'ok'}
        rf=lambda *args:CacheStudyRecipe(base.fork(),cfg,model=SimpleNamespace(commit_controls=10),tracker_class=Tracker)
        policy=runtime.wrap_connection_factory(make_factory(cfg,build_shared_storage(cfg),rf))()
        policy.on_episode_start(extra_metadata={'task_name':tasks[0],'task_id':0,'original_init_idx':0,
            'campaign':'cpu','arm':'qualification','attempt':1,'bundle_generation':'cpu',
            'ack_protocol':'r17.execution.v1'})
        obs=observation('libero');hits=[]
        for seq in range(4):
            value=policy.infer(dict(obs,__request__=dict(campaign='cpu',benchmark_protocol=protocol,
                task_name=tasks[0],original_init_idx=0,decision_seq=seq,task_id=0,arm='qualification',
                attempt=1,bundle_generation='cpu',seed=7,explicit_seed=18),
                **({'__execution__':{'decision_seq':seq-1,'executed_count':5}} if seq else {})))
            hits.append(value['__hit_meta__'])
        assert [h['source'] for h in hits]==['library','library_tail','policy','policy_tail']
        assert [set(h['batching']['stages']) for h in hits]==[{'stage1'},set(),{'stage1','stage2','stage3'},set()]
        assert [h['actual_nfe'] for h in hits]==[0,0,spec.default_sampler.num_steps,0]
        assert hits[2]['stall_call'] and not hits[2]['corrector_applied']
    finally:runtime.close()


@pytest.mark.parametrize('node',['local','h100'])
def test_metadata_fit_transport_and_cosmos_admission_are_cpu_only(node,tmp_path,monkeypatch):
    import ast
    import shutil
    from exp.r17_integration.common.provider import AdapterProvider,register_provider,get_provider
    from exp.r17_integration.ops.sync_back import install_snapshot
    manifest=study.validate_manifest(study.HERE/'yaml/STUDY.json')
    row=manifest['models']['cosmos_libero'][0]
    parent=study.read(Path(row['library'])/'READY.json');source,benchmark=study.profiles('cosmos_libero',parent)
    spec=replace(study.model_spec(parent['binding']['model']),allowed_benchmark_protocols=(benchmark.identity,))
    provider=AdapterProvider('cpu_cosmos_seal_'+uuid.uuid4().hex,spec,lambda:None,lambda k:spec.default_sampler)
    register_provider(provider)
    assets=tmp_path/'assets';assets.mkdir()
    campaign=dict(model='cosmos',cache_study=str(study.HERE/'yaml/STUDY.json'),assets=str(assets),
        server_node=node,suites=['libero_spatial'],run_name='cpu_'+uuid.uuid4().hex,runtime='cpu_only')
    staged={};calls=[]
    def transfer(argv):
        assert argv[:2]==['tether','push'];staged[argv[3].split(':',1)[1]]=argv[2];calls.append(argv)
    def remote(node,argv,**kw):
        assert node=='h100' and len(argv[-1])<4096
        operation=next(n for n in ast.walk(ast.parse(argv[-1])) if isinstance(n,ast.Call) and
            isinstance(n.func,ast.Name) and n.func.id=='install_snapshot')
        source,dest=map(ast.literal_eval,operation.args)
        return json.dumps(install_snapshot(staged[source],dest,expected_sha256=ast.literal_eval(operation.keywords[0].value)))
    launch=SimpleNamespace(transfer=transfer,remote=remote,pythonpath=lambda root:'cpu_only',host_python=lambda c:'python')
    study.stage_inputs(campaign,tmp_path,launch)
    transported=study.read(assets/'cache_inputs/manifest.json')
    assert len(transported['groups'])==8 and not any('image' in p or 'normalized_actions.npy' in p
        or 'state.npy' in p for p in transported['files'])
    assert all(study.file_sha(assets/'cache_inputs'/p)==pin for p,pin in transported['files'].items())
    assert len(calls)==(1 if node=='h100' else 0)
    study.seal(provider,campaign,benchmark)
    sealed=study.read(assets/'CACHE_STUDY.json')
    assert len(sealed['groups'])==8 and sealed['eval_pool']=='A'
    assert provider.config_validator is None and get_provider(provider.provider_id) is provider
