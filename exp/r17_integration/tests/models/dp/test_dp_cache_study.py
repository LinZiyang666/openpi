"""B500 VIEW admission and native DP/S07 YAML cache-study composition."""
import asyncio
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from exp.r17_integration.models.dp.admission import admission_validators, recipe_factory
from exp.r17_integration.models.dp.batch_runtime import BatchConfig, create_runtime
from exp.r17_integration.models.dp.formal_batch_server import cache_study_admission
from exp.r17_integration.models.dp.provider import configure_library_provider, make_connection_factory, register_provider
from exp.r17_integration.models.dp.serve import plain_config
from exp.r17_integration.common.config import RecipeConfig, build_shared_storage
from exp.r17_integration.ops import cache_study as study


@pytest.mark.parametrize('suite', ['libero_spatial','libero_10'])
def test_real_B500_VIEW_teacher_reader_and_no_parent_digest_substitution(make_adapter, suite):
    adapter = make_adapter('libero')
    _,validate = admission_validators(adapter.metadata(),lambda k:adapter.metadata().default_sampler)
    path = Path('/data/r17_library')/('dp_libero_'+suite)/'bpool/subsets/size_500'
    view = json.loads((path/'VIEW.json').read_text())
    cfg = SimpleNamespace(adapter_binding={'num_steps':100,'commit_horizon':5},
        recipe=RecipeConfig(enabled=True,store_path=str(path),store_digest=view['content_digest']))
    validate(cfg)  # Old code raised FileNotFoundError: VIEW has no READY.json.
    provider = SimpleNamespace(config_validator=validate)
    assert recipe_factory(provider,lambda *args:'verified-view')(cfg,None) == 'verified-view'
    cfg.recipe.store_digest = view['parent_digest']
    with pytest.raises(ValueError,match='view mismatch'):
        validate(cfg)


@pytest.fixture
def audited_provider():
    from exp.r17_integration.common.provider import AdapterProvider
    from exp.r17_integration.models.dp.library import verified_producer
    release = verified_producer()
    document = study.read('/data/mw_finetune/eval/hs5dp_ctl_142812/assets/model_spec.json')
    document.update(checkpoint_identity=release['resident_checkpoint'],code_digest=release['resident_code'])
    document['feature_spec']['policy_digest'] = release['resident_checkpoint']
    spec = study.model_spec(document)
    return configure_library_provider(AdapterProvider('P4.audited.synthetic',spec,lambda:None,
        lambda k:spec.default_sampler))


def bound_real_config(provider):
    row = study.validate_manifest(study.HERE/'yaml/STUDY.json')['models']['dp_libero'][0]
    parent = study.read(Path(row['library'])/'READY.json')
    source,serving = study.profiles('dp_libero',parent)
    cfg = SimpleNamespace(adapter_binding={'benchmark_protocol':serving.identity},
        recipe=SimpleNamespace(library_benchmark=dict(source=source.manifest(),serving=serving.manifest())))
    return cfg,parent['binding']


def test_frozen_provider_copy_and_content_bound_compatibility(audited_provider):
    from exp.r17_integration.common.recipe import retrieval_model_identity
    provider = audited_provider
    configured = configure_library_provider(provider)
    assert configured is not provider and configured.provider_id==provider.provider_id
    assert configured.model_spec is provider.model_spec
    cfg,binding = bound_real_config(provider)
    receipt = provider.library_validator(cfg,binding)
    assert receipt['status']=='PASS' and receipt['schema']=='r17.teacher_compatibility.v1'
    assert receipt['teacher_identity']==retrieval_model_identity(binding['model'])
    assert receipt['resident_identity']==retrieval_model_identity(provider.model_spec)
    assert receipt['evidence']['producer_sources'] and receipt['evidence']['weight_language_norm_assets']


@pytest.mark.parametrize('damage',['teacher_code','teacher_weight_id','norm','preprocess','geometry','output','camera','resident_code'])
def test_compatibility_rejects_unknown_producer_and_geometry(audited_provider, damage):
    provider = audited_provider
    cfg,binding = bound_real_config(provider)
    binding = copy.deepcopy(binding)
    if damage=='teacher_code':binding['model']['code_digest']='unknown'
    elif damage=='teacher_weight_id':binding['model']['checkpoint_identity']='unknown'
    elif damage=='norm':binding['model']['norm_identity']='wrong'
    elif damage=='preprocess':binding['model']['feature_spec']['preprocess_digest']='wrong'
    elif damage=='geometry':binding['model']['layout']['horizon']=32
    elif damage=='output':binding['model']['layout']['output_transform_id']='wrong'
    elif damage=='camera':binding['model']['feature_spec']['real_cameras']=['base_0_rgb']
    else:provider=configure_library_provider(replace(provider,model_spec=replace(provider.model_spec,code_digest='unknown')))
    with pytest.raises(ValueError,match='audited|identity|differs'):
        provider.library_validator(cfg,binding)


def test_registered_hook_is_lazy_default_off(make_adapter, monkeypatch):
    from exp.r17_integration.common import provider as registry
    monkeypatch.setattr(registry,'_providers',{})
    monkeypatch.setattr('exp.r17_integration.models.dp.library.verified_producer',
        lambda:pytest.fail('plain/default-off registration read production assets'))
    adapter = make_adapter('libero')
    provider = register_provider(adapter)
    assert registry.get_provider(provider.provider_id) is provider and provider.library_validator is not None
    plain_config(provider)


def test_changed_real_producer_file_is_rejected(tmp_path, monkeypatch):
    from exp.r17_integration.models.dp import library
    from exp.r17_integration.models.dp.manifest import file_digest
    original=Path('exp/mw_finetune/dp/preprocess.py')
    copy_path=tmp_path/'preprocess.py';copy_path.write_bytes(original.read_bytes())
    release=copy.deepcopy(library.RELEASE)
    release['producer_sources']={str(copy_path):file_digest(original)}
    release['assets']={}
    monkeypatch.setattr(library,'RELEASE',release)
    monkeypatch.setattr(library,'_verified',None)
    copy_path.write_bytes(copy_path.read_bytes().replace(b'quantile_normalize',b'quantile_normalizX',1))
    with pytest.raises(ValueError,match='changed feature/action producer'):
        library.verified_producer()


def test_sealed_VIEW_keeps_other_eta_K1_teacher_forbidden(make_adapter, monkeypatch, tmp_path):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.search.store import write_store
    from exp.r17_integration.search.teacher_library import nested_views
    from exp.r17_integration.models.dp.sampler import SamplerProfile
    monkeypatch.setattr(registry,'_providers',{})
    adapter=make_adapter('libero');provider=register_provider(adapter)
    cfg=plain_config(provider,num_steps=2)
    sampler=SamplerProfile('ddim',1,eta=.7).to_spec(adapter.model.cfg)
    binding=model_binding(adapter.metadata(),adapter.metadata().allowed_benchmark_protocols[0])
    binding['sampler']=asdict(sampler)
    parent=write_store(tmp_path/'parent',binding,
        [dict(split='library',init_pool='B',library_episode_id=0)],{'state':[np.zeros(8,np.float32)]})
    view=nested_views(parent,dict(selections={'1':{'all':[0]}},digest='CPU',subset_seed=71),sizes=(1,))[1]
    cfg.recipe=RecipeConfig(enabled=True,store_path=str(view.directory),store_digest=view.manifest['content_digest'],
        teacher_sampler_identity=sampler.identity)
    with pytest.raises(ValueError,match='K1.*teacher'):
        provider.config_validator(cfg)


def test_plain_and_explicit_cache_study_admission(make_adapter, monkeypatch):
    from exp.r17_integration.common import provider as registry
    monkeypatch.setattr(registry,'_providers',{})
    provider = register_provider(make_adapter('libero'))
    config = plain_config(provider);config.enabled=False
    batch = BatchConfig(8,5)
    assert cache_study_admission({},config,batch,provider) is False
    campaign=dict(cache_study='sealed-study',replicas=2,mps=True,server_node='local')
    assert cache_study_admission(campaign,config,batch,provider) is True
    with pytest.raises(ValueError,match='match control'):
        cache_study_admission(dict(campaign,replicas=1),config,batch,provider)
    config.adapter_binding['num_steps']=10
    with pytest.raises(ValueError,match='native DDPM100'):
        cache_study_admission(campaign,config,batch,provider)


def native_configs(adapter,provider,tmp_path):
    from exp.r17_integration.common.benchmarks import get_benchmark,task_table_identity
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.search.store import write_store
    from exp.r17_integration.search.teacher_library import nested_views
    from exp.r17_integration.search.fit import fit_recipe,save_fitted_recipe
    protocol=provider.model_spec.allowed_benchmark_protocols[0]
    tasks=list(get_benchmark(protocol).ordered_tasks)
    rng=np.random.default_rng(71);n=24
    rows=[dict(task_name=tasks[0],task_id=0,task_uid=str(i//4),library_episode_id=i//4,
        decision_seq=i%4,split='library',init_pool='B',campaign='P4_CPU',arm='teacher',attempt=1,
        orig_init_idx=i//4,task_data_hash='CPU',benchmark_protocol=protocol,success=True,
        bundle_generation='CPU') for i in range(n)]
    arrays={f'key_{k}':rng.normal(size=(n,w)).astype(np.float32) for k,w in provider.model_spec.feature_spec.fields.items()}
    arrays.update(state=rng.normal(size=(n,8)).astype(np.float32),
        normalized_actions=rng.normal(size=(n,16,7)).astype(np.float32))
    parent=write_store(tmp_path/'parent',model_binding(provider.model_spec,protocol),rows,arrays)
    view=nested_views(parent,dict(selections={'6':{'all':list(range(6))}},digest='CPU',subset_seed=71),sizes=(6,))[6]
    frozen={}
    for c in (5,10):
        fitted,receipt=fit_recipe(view,task_names=tasks,corrector=True,commit_horizon=c,k=16,kref=8,
            early=True,guard=False,periodic_gap=None)
        path=tmp_path/f'C{c}';ready=save_fitted_recipe(fitted,receipt,path)
        config=plain_config(provider,commit_horizon=c)
        config.recipe=RecipeConfig(enabled=True,store_path=str(view.directory),store_digest=view.manifest['content_digest'],
            teacher_sampler_identity=provider.model_spec.default_sampler.identity,
            fitted_path=str(path),fitted_digest=ready['digest'],miss_mode='never',corrector=True,guard=False,
            ordered_tasks=tasks,task_table_identity=task_table_identity(protocol))
        frozen[c]=config
    configs={}
    for group in study.GROUPS:
        config=copy.deepcopy(frozen[10 if 'h' in group else 5])
        config.recipe.corrector='r' in group
        if 'g' in group:
            config.recipe.miss_mode='stall_only';config.recipe.stall_path='injected-state-only'
            config.recipe.stall_digest='CPU'
        configs[group]=config
    return configs,tasks,protocol


def test_eight_hot_YAML_groups_real_DP_S07_B8_hit_tail_and_stall_only_DDPM100(make_adapter, tmp_path, monkeypatch):
    from exp.r17_integration.common import provider as registry
    from openpi.serving import websocket_policy_server as ws
    from exp.r17_integration.common.serving import make_bundle_validator
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from exp.r17_integration.models.dp.fixtures import libero_observation
    from exp.r17_integration.tests.models.fastwam.conftest import fixture_benchmark_spec
    from openpi_client import msgpack_numpy
    import websockets.asyncio.client
    monkeypatch.setattr(registry,'_providers',{})
    for key,value in (('_bundles',{}),('_current_bundle',None),('_bundle_version',0)):
        monkeypatch.setattr(ws,key,value)
    class Tracker:
        def __init__(self,*args):self.index=-1
        def observe(self,*args):self.index+=1
        def status(self):return {'state':'slow_confirmed' if self.index==1 else 'ok'}
    monkeypatch.setattr('exp.offline_search.rounds.r06.ideation_Q3.stall.stall.StallTracker',Tracker)
    monkeypatch.setattr('exp.r17_integration.search.cache_recipe.load_stall',lambda *args:SimpleNamespace(commit_controls=10))
    benchmark=replace(fixture_benchmark_spec('libero'),wire_size=96)
    adapter=make_adapter('libero',libero_benchmark_specs=[asdict(benchmark)])
    monkeypatch.setattr('exp.mw_finetune.dp.policy.load_policy',lambda *args,**kwargs:pytest.fail('YAML switch reloaded weights'))
    runtime=create_runtime(adapter,8,20,tmp_path/'batching.jsonl')
    try:
        provider=register_provider(adapter)
        configs,tasks,protocol=native_configs(adapter,provider,tmp_path)
        plain=plain_config(provider)
        factory=runtime.wrap_connection_factory(make_connection_factory(adapter,provider=provider,
            default_config=plain,default_storage=build_shared_storage(plain),stage_core=runtime.core))
        obs,_=libero_observation()
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        async def run():
            ready=asyncio.Event();server=WebsocketPolicyServer(object(),host='127.0.0.1',port=port,
                concurrent=True,connection_policy_factory=factory,bundle_validator=make_bundle_validator(provider),
                ready_callback=ready.set,metadata={'fixture':'native DP CPU cache study'})
            running=asyncio.create_task(server.run(stop_on_request=True));await asyncio.wait_for(ready.wait(),5)
            packer=msgpack_numpy.Packer();lifetimes=set();clients=[]
            async def call(client,payload):
                await client.send(packer.pack(payload));result=await asyncio.wait_for(client.recv(),30)
                assert isinstance(result,bytes),result
                return msgpack_numpy.unpackb(result)
            try:
                for _ in range(8):
                    client=await websockets.asyncio.client.connect(f'ws://127.0.0.1:{port}',compression=None)
                    await client.recv();clients.append(client)
                for group,cfg in configs.items():
                    ack=await call(clients[0],dict(__ctrl__='load_cache_config',bundle_id=group,
                        yaml_content=yaml.safe_dump(asdict(cfg))))
                    assert ack['__ack__']=='load_cache_config',ack
                    for i,client in enumerate(clients):
                        assert (await call(client,dict(__ctrl__='select_bundle',bundle_id=group)))['__ack__']=='select_bundle'
                        start=await call(client,dict(__ctrl__='episode_start',__task__=tasks[0],__extra__=dict(
                            campaign='P4_CPU',task_name=tasks[0],task_id=0,original_init_idx=i,seed=7,
                            explicit_seed=18+i,ack_protocol='r17.execution.v1',arm=group)))
                        assert start['__ack__']=='episode_start',start
                    for seq in range(4):
                        replies=await asyncio.gather(*(call(client,dict(obs,
                            **({'__execution__':dict(decision_seq=seq-1,executed_count=5)} if seq else {})))
                            for client in clients))
                        for reply in replies:
                            hit=reply['__hit_meta__'];lifetimes.add((hit['resident_pid'],hit['resident_generation']))
                            assert hit['corrector_enabled']==('r' in group)
                            if 'h' in group and seq%2:
                                assert hit['source'].endswith('_tail') and hit['stage_calls']==[0,0,0]
                            elif 'g' in group and seq==(2 if 'h' in group else 1):
                                assert hit['source']=='policy' and hit['stall_call']
                                assert hit['actual_nfe']==100 and hit['stage_calls']==[1,1,100]
                                assert set(hit['batching']['stages'])=={'stage1','stage2','stage3'}
                            else:
                                assert hit['source']=='library' and hit['actual_nfe']==0
                                assert hit['stage_calls']==[1,0,0] and set(hit['batching']['stages'])=={'stage1'}
                            if hit['source']!='policy':assert hit['actual_nfe']==0
                    for client in clients:
                        end=await call(client,dict(__ctrl__='episode_end',
                            __execution__=dict(decision_seq=3,executed_count=2,terminal=True)))
                        assert end['execution']['terminal'] and end['execution']['executed_count']==2
                assert len(lifetimes)==1
            finally:
                for client in clients:await client.close()
                server.request_stop();await asyncio.wait_for(running,5)
        asyncio.run(run())
        ledger=[json.loads(line) for line in (tmp_path/'batching.jsonl').read_text().splitlines()]
        assert any(r.get('stage')=='stage1' and r['size']==8 for r in ledger)
        assert any(r.get('stage')=='stage3' and r['size']>1 for r in ledger)
    finally:runtime.close()
