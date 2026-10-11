import asyncio
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch
import yaml

from exp.r17_integration.common.config import (BackendConfig,CacheConfig,CheckpointConfig,GateConfig,
    JudgeConfig,KeyBuilderConfig,KeysConfig,SearchStrategyConfig,WritePolicyConfig,
    build_shared_storage,validate_cache_config)
from openpi.cache.storage_types import CacheEntry,CachePayload
from openpi.cache.types import CheckpointID
from exp.r17_integration.common.provider import artifact_binding,binding_for
from exp.r17_integration.common.serving import make_bundle_validator
from exp.r17_integration.models.fastwam.provider import register_provider,make_connection_factory
from exp.r17_integration.tests.models.fastwam.conftest import make_adapter,observation,request


def config_for(provider,k=2,judge="threshold",commit=10):
    keys=KeysConfig()
    for name in vars(keys):
        getattr(keys,name).enabled=name in provider.model_spec.feature_spec.fields
    return CacheConfig(enabled=True,diagnostic_context=True,keys=keys,key_builder=KeyBuilderConfig(type="adapter_native"),
        checkpoints={"cp1":CheckpointConfig(gate=GateConfig(type="always_search"),judge=JudgeConfig(type=judge),
            search_strategy=SearchStrategyConfig(type="weighted_rrf_knn",top_k=1,task_scoped=False))},
        backend=BackendConfig(type="in_memory",vector_dims=dict(provider.model_spec.feature_spec.fields)),
        write_policy=WritePolicyConfig(type="never"),
        adapter_binding=binding_for(provider,provider.model_spec.allowed_benchmark_protocols[0],num_steps=k,commit_horizon=commit))


@pytest.mark.parametrize("benchmark",["libero","metaworld"])
def test_stock_orchestrator_hit_miss_tail_inverse_and_isolation(benchmark,tmp_path):
    a=make_adapter(benchmark);p=register_provider(a,provider_id="p3_"+tmp_path.name+benchmark)
    cfg=config_for(p);validate_cache_config(cfg);storage=build_shared_storage(cfg)
    records=[]
    factory=make_connection_factory(p,cfg,storage,capture_sink=records.append)
    c1,c2=factory(),factory()
    assert c1.adapter is not c2.adapter and c1.adapter.model is c2.adapter.model
    assert c1.orchestrator is not c2.orchestrator
    obs=observation(benchmark)
    plain=c1.infer(obs)["actions"]
    assert records[-1].stage_calls==(1,1,2) and records[-1].actual_nfe==2
    assert records[-1].model_capture["native_grid"]["num_steps"]==2
    assert len(records[-1].native_snapshots)==1
    assert records[-1].model_capture["tokens"].shape[0]==1
    calls={k:list(v) for k,v in a.model.calls.items()}
    tail=c1.infer({**obs,"__execution__":{"decision_seq":0,"executed_count":5}})["actions"]
    assert a.model.calls==calls
    assert tail.tobytes()==plain[5:].tobytes()
    assert records[-1].stage_calls==(0,0,0) and records[-1].cursor==5
    assert not records[-1].model_capture and not records[-1].native_snapshots
    c2.infer(obs)
    assert records[-1].stage_calls==(1,1,2) and c2._cursor==0
    c1.on_episode_end();c1.infer(obs)
    assert records[-1].stage_calls==(1,1,2)
    # No acknowledgment disables reuse instead of inventing executed controls.
    c1.infer(obs)
    assert records[-1].stage_calls==(1,1,2)
    hitcfg=config_for(p,judge="always_hit")
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.cache_storage import CacheStorage
    backend=InMemoryBackend(dict(p.model_spec.feature_spec.fields))
    hitstorage=CacheStorage(backend)
    fields=a.run_stage1(a.prepare(obs,request())).fields
    normalized=torch.full((32,a.metadata().layout.action_width),.25)
    hitstorage.insert(CacheEntry("donor",CheckpointID.CP1,{k:torch.from_numpy(v) for k,v in fields.items()},CachePayload(normalized)))
    backend.artifact_meta={"r17_binding":artifact_binding(hitcfg)}
    backend.freeze()
    hits=[]
    hit=make_connection_factory(p,hitcfg,hitstorage,capture_sink=hits.append)()
    calls={k:len(v) for k,v in a.model.calls.items()}
    actual=hit.infer(obs)["actions"]
    expected=a.to_wire(normalized,None).actions
    assert actual.tobytes()==expected[:10].tobytes()
    assert hits[-1].stage_calls==(1,0,0) and hits[-1].actual_nfe==0
    assert len(a.model.calls["prefill"])==calls["prefill"]
    assert len(a.model.calls["head"])==calls["head"]
    assert len(a.model.calls["context"])==calls["context"]
    assert len(a.model.calls["pre"])==calls["pre"]


def test_binding_rejects_bad_model_norm_geometry_sampler_and_advanced(adapter,tmp_path):
    p=register_provider(adapter,provider_id="p3_"+tmp_path.name)
    cfg=config_for(p)
    for field in ("checkpoint_identity","norm_identity","layout_identity","feature_identity","sampler_identity","benchmark_protocol"):
        bad=config_for(p);bad.adapter_binding[field]="wrong"
        with pytest.raises(ValueError):
            make_bundle_validator(p)(bad)
    for commit in (0,4,33):
        bad=config_for(p);bad.adapter_binding["commit_horizon"]=commit
        with pytest.raises(ValueError):
            make_bundle_validator(p)(bad)
    bad=config_for(p);bad.checkpoints["cp2"]=CheckpointConfig()
    with pytest.raises(ValueError):
        make_bundle_validator(p)(bad)


def test_live_ws_two_clients_hot_yaml_generation_switch_and_bad_rebind(adapter,tmp_path,monkeypatch):
    import websockets.asyncio.server as ws_server
    import websockets.asyncio.client as ws_client
    from openpi_client import msgpack_numpy
    from openpi.serving import websocket_policy_server as wps
    p=register_provider(adapter,provider_id="p3_"+tmp_path.name)
    cfg=config_for(p,k=2);storage=build_shared_storage(cfg)
    captured=[];policies=[]
    factory=make_connection_factory(p,cfg,storage,capture_sink=captured.append)
    def tracked(base=None,bundle_id="default"):
        policy=factory(base,bundle_id);policies.append(policy);return policy
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    server=WebsocketPolicyServer(tracked(),concurrent=True,connection_policy_factory=tracked,
        bundle_validator=make_bundle_validator(p),metadata={"model":"p3_cpu_stub"})
    monkeypatch.setattr(wps,"_bundles",{})
    monkeypatch.setattr(wps,"_current_bundle",None)
    paths=[]
    for k in (1,2):
        path=tmp_path/f"k{k}.yaml"
        path.write_text(yaml.safe_dump(asdict(config_for(p,k=k))))
        paths.append(path)
    bad=config_for(p,k=1);bad.adapter_binding["norm_identity"]="wrong"
    badpath=tmp_path/"bad.yaml";badpath.write_text(yaml.safe_dump(asdict(bad)))
    packer=msgpack_numpy.Packer()
    async def rpc(ws,msg):
        await ws.send(packer.pack(msg));return msgpack_numpy.unpackb(await asyncio.wait_for(ws.recv(),5))
    async def exercise():
        async with ws_server.serve(server._handler,"127.0.0.1",0) as listener:
            port=listener.sockets[0].getsockname()[1]
            async with ws_client.connect(f"ws://127.0.0.1:{port}") as c1, ws_client.connect(f"ws://127.0.0.1:{port}") as c2:
                assert msgpack_numpy.unpackb(await c1.recv())["model"]=="p3_cpu_stub"
                await c2.recv()
                for k,path in zip((1,2),paths):
                    ack=await rpc(c1,{"__ctrl__":"load_cache_config","yaml_path":str(path),"bundle_id":f"k{k}_v1"})
                    assert ack["__ack__"]=="load_cache_config",ack
                assert (await rpc(c1,{"__ctrl__":"select_bundle","bundle_id":"k1_v1"}))["__ack__"]=="select_bundle"
                assert (await rpc(c2,{"__ctrl__":"select_bundle","bundle_id":"k2_v1"}))["__ack__"]=="select_bundle"
                for c in (c1,c2):
                    assert (await rpc(c,{"__ctrl__":"episode_start","task_id":0,"init_state_id":0,"episode_id":0}))["__ack__"]=="episode_start"
                first=await rpc(c1,observation());await rpc(c2,observation())
                assert [r.actual_nfe for r in captured[-2:]]==[1,2]
                await rpc(c1,{**observation(),"__execution__":{"decision_seq":0,"executed_count":5}})
                assert captured[-1].stage_calls==(0,0,0)
                bound=policies[-2]
                assert (await rpc(c1,{"__ctrl__":"select_bundle","bundle_id":"k1_v1"}))["__ack__"]=="select_bundle"
                assert bound._seq==2
                ack=await rpc(c1,{"__ctrl__":"load_cache_config","yaml_path":str(badpath),"bundle_id":"bad"})
                assert ack["__ack__"]=="error" and "bad" not in wps._bundles
                assert (await rpc(c1,{"__ctrl__":"select_bundle","bundle_id":"unknown"}))["__ack__"]=="error"
                for c in (c1,c2):
                    await rpc(c,{"__ctrl__":"episode_end","success":False})
                await rpc(c1,{"__ctrl__":"select_bundle","bundle_id":"k2_v1"})
                await rpc(c1,{"__ctrl__":"episode_start","task_id":0,"init_state_id":0,"episode_id":1})
                await rpc(c1,observation());assert captured[-1].actual_nfe==2
                # A new generation builds a new wrapper; same ID is idempotent.
                await rpc(c1,{"__ctrl__":"load_cache_config","yaml_path":str(paths[0]),"bundle_id":"k1_v2"})
                await rpc(c1,{"__ctrl__":"select_bundle","bundle_id":"k1_v2"})
                assert policies[-1].sampler.num_steps==1
            await asyncio.sleep(.02)
    asyncio.run(exercise())
    assert all(policy._chunk is None for policy in policies)


@pytest.mark.parametrize("benchmark",["libero","metaworld"])
def test_fastwam_layout_consumes_shared_library_recipe_corrector_and_tail(benchmark,tmp_path):
    from exp.r17_integration.common.layers import LayoutRecipe,LibraryData
    from exp.r17_integration.common.query_view import QueryView
    from exp.r17_integration.common.benchmarks import register_benchmark,task_table_identity
    from exp.r17_integration.tests.models.fastwam.conftest import fixture_benchmark_spec
    a=make_adapter(benchmark);spec=a.metadata()
    benchmark_spec=fixture_benchmark_spec(benchmark)
    protocol=spec.allowed_benchmark_protocols[0]
    register_benchmark(benchmark_spec,protocol_identity=protocol)
    tasks=list(benchmark_spec.to_contract().ordered_tasks)
    rng=np.random.default_rng(99);n=24
    lib=LibraryData({"vision_0":rng.normal(size=(n,spec.feature_spec.fields["vision_0"])).astype(np.float32)},
        rng.normal(size=(n,len(spec.layout.valid_state_indices))).astype(np.float32),
        rng.normal(size=(n,32,spec.layout.action_width)).astype(np.float32),
        np.zeros(n,np.int64),np.repeat(np.arange(6),4),np.tile(np.arange(4),6))
    raw=LayoutRecipe(spec.layout,len(tasks),corrector=False,guard=False).fit(lib)
    fitted=LayoutRecipe(spec.layout,len(tasks),corrector=True,guard=False).fit(lib)
    p=register_provider(a,provider_id="p3_recipe_"+tmp_path.name+benchmark)
    cfg=config_for(p);storage=build_shared_storage(cfg);records=[]
    from exp.r17_integration.common.config import RecipeConfig
    from exp.r17_integration.search.store import write_store
    from exp.r17_integration.debug.validate import model_binding
    rows=[dict(split="library",campaign="p3_fixture",arm="teacher",bundle_generation="fixture",benchmark_protocol=protocol,
        task_name=tasks[0],task_uid=f"ep{int(lib.episode[i])}",attempt=1,decision_seq=int(lib.step[i]),
        orig_init_idx=int(lib.episode[i])+10,task_data_hash="synthetic") for i in range(n)]
    store=write_store(tmp_path/"library",model_binding(asdict(spec),protocol),rows,
        {"key_vision_0":lib.features["vision_0"],"state":lib.state,"normalized_actions":lib.action})
    cfg.recipe=RecipeConfig(enabled=True,store_path=str(store.directory),store_digest=store.manifest["content_digest"],
        teacher_sampler_identity=spec.default_sampler.identity,corrector=True,guard=False,
        ordered_tasks=tasks,task_table_identity=task_table_identity(protocol))
    factory=make_connection_factory(p,cfg,storage,recipe_factory=lambda c,s:fitted.fork(),capture_sink=records.append)
    c1,c2=factory(),factory()
    assert c1.recipe.guard is not c2.recipe.guard
    c1.on_episode_start(task_id=0,task=tasks[0],episode_id=0)
    obs=observation(benchmark)
    c1.infer(obs)
    assert records[-1].source=="library" and records[-1].stage_calls==(1,0,0)
    assert not a.model.calls["prefill"] and not a.model.calls["head"]
    query=QueryView(records[-1].request,records[-1].fields,obs["state"],[],[])
    expected_raw=raw.query(query).action
    np.testing.assert_array_equal(records[-1].normalized_chunk[:,spec.layout.gripper_channel],
        expected_raw[:,spec.layout.gripper_channel])
    for fold in fitted.fit_evidence["loeo"]:
        assert fold["query_episode"] not in fold["donor_episodes"]
    c1.infer({**obs,"__execution__":{"decision_seq":0,"executed_count":5}})
    assert records[-1].stage_calls==(0,0,0) and records[-1].source=="library_tail"
    c1.on_episode_end();assert c1._chunk is None
