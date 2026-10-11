import asyncio
import copy
from dataclasses import asdict
import socket

import numpy as np
import pytest
import yaml

from exp.r17_integration.common.provider import artifact_binding
from exp.r17_integration.common.serving import make_bundle_validator, make_connection_factory
from exp.r17_integration.common.config import build_shared_storage, load_cache_config, validate_cache_config
from openpi.serving import websocket_policy_server as ws
from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer

from exp.r17_integration.tests.common.conftest import add_entry, config_for


def test_stock_orchestrator_hit_miss_tail_and_inverse_once(provider):
    config = config_for(provider, commit=10)
    validate_cache_config(config)
    storage = build_shared_storage(config)
    captures = []
    factory = make_connection_factory(provider, config, storage, capture_sink=captures.append)
    p = factory()
    p.on_task_begin(); p.on_episode_start(task_id=2, episode_id=1, extra_metadata={"task_name":"reach"})
    obs = {"state": np.arange(4, dtype=np.float32), "__debug__": {"foo": "bar"}}
    miss = p.infer(obs)
    assert miss["__hit_meta__"]["stage_calls"] == (1,1,4)
    assert captures[0].actual_nfe == 4 and len(captures[0].native_snapshots) == 3
    first = dict(p.adapter.counts)
    tail = p.infer(dict(obs, __execution__={"decision_seq":0,"executed_count":3,"cursor":3}))
    assert p.adapter.counts == first
    assert tail["__hit_meta__"]["stage_calls"] == (0,0,0)
    assert tail["__hit_meta__"]["hit_type"] == "FULL_HIT"
    assert tail["__hit_meta__"]["cursor"] == 3
    np.testing.assert_array_equal(tail["actions"], miss["actions"][3:])
    with pytest.raises(ValueError, match="stale"):
        p.infer(dict(obs, __execution__={"decision_seq":0,"executed_count":1}))
    assert not p.orchestrator.key_builder.cached_data
    p.on_task_end()
    config2 = config_for(provider, commit=10)
    storage2 = build_shared_storage(config2)
    add_entry(storage2, config2)
    hit = make_connection_factory(provider, config2, storage2)()
    out = hit.infer(obs)
    assert out["__hit_meta__"]["stage_calls"] == (1,0,0)
    assert hit.adapter.counts["inverse"] == 1 and hit.adapter.counts["nfe"] == 0
    np.testing.assert_array_equal(out["actions"], np.full((10,4),6.,np.float32))


def test_capture_on_off_byte_parity_and_no_unacknowledged_tail(provider):
    cfg=config_for(provider,commit=10); storage=build_shared_storage(cfg)
    captures=[]
    a=make_connection_factory(provider,cfg,storage)()
    b=make_connection_factory(provider,cfg,storage,capture_sink=captures.append)()
    obs={"state":np.arange(4,dtype=np.float32)}
    for seq in range(3):
        ra,rb=a.infer(obs),b.infer(obs)
        assert ra["actions"].tobytes()==rb["actions"].tobytes()
        assert ra["__hit_meta__"]["stage_calls"]==rb["__hit_meta__"]["stage_calls"]
        assert ra["__hit_meta__"]["sampler_identity"]==rb["__hit_meta__"]["sampler_identity"]
    assert a.adapter.counts==b.adapter.counts
    assert a.adapter.counts["stage1"]==3


def test_shared_sink_execution_identity_and_tail_current_state(provider):
    class Sink:
        def __init__(self):
            self.records, self.acks = [], []
        def __call__(self, record):
            self.records.append(record)
        def on_execution(self, ack):
            self.acks.append(ack)
    cfg=config_for(provider,commit=10);storage=build_shared_storage(cfg);sink=Sink()
    factory=make_connection_factory(provider,cfg,storage,capture_sink=sink)
    a,b=factory(),factory()
    for policy,idx in ((a,3),(b,7)):
        policy.on_episode_start(episode_id=idx,extra_metadata={"task_name":"reach","original_init_idx":idx,"attempt":1})
        policy.infer({"state":np.arange(4,dtype=np.float32)})
    before=dict(a.adapter.counts)
    for policy in (b,a):
        policy.infer({"state":np.arange(4,dtype=np.float32)+10,
                      "__execution__":{"decision_seq":0,"executed_count":5}})
    assert [ack["request"].original_init_idx for ack in sink.acks]==[7,3]
    assert all(ack["request"].decision_seq==0 and ack["request"].attempt==1 for ack in sink.acks)
    assert a.adapter.counts==before
    assert sink.records[-1].stage_calls==(0,0,0)
    np.testing.assert_array_equal(sink.records[-1].fields["robot_state"],np.arange(4,dtype=np.float32)+10)
    assert set(sink.records[-1].fields)=={"robot_state"}


def test_recipe_receives_acknowledged_previous_normalized_action_and_resets(provider,tmp_path):
    from exp.r17_integration.common.query_view import RecipeResult
    class Recipe:
        def __init__(self):self.queries=[]
        def reset(self,episode):pass
        def force_look(self,*args):return True
        def query(self,q):
            self.queries.append(q)
            return RecipeResult(np.zeros((16,4),np.float32),np.array([0]),np.array([1.]),np.array([0.]),force_miss=True)
    cfg=config_for(provider,commit=5)
    from exp.r17_integration.tests.common.conftest import with_recipe
    cfg=with_recipe(provider,cfg,tmp_path)
    storage=build_shared_storage(cfg);recipe=Recipe()
    policy=make_connection_factory(provider,cfg,storage,recipe_factory=lambda c,s:recipe)()
    policy.on_episode_start(extra_metadata={"task_name":"reach", "task_id":0})
    obs={"state":np.arange(4,dtype=np.float32)}
    first=policy.infer(obs)
    policy.infer(dict(obs,__execution__={"decision_seq":0,"executed_count":3}))
    q=recipe.queries[-1]
    np.testing.assert_array_equal(q.previous_action[:5],first["actions"]/2)
    assert q.executed_count==3 and q.prev_hit is False
    policy.on_episode_start(episode_id=2,extra_metadata={"task_name":"reach", "task_id":0})
    policy.infer(obs)
    assert recipe.queries[-1].previous_action is None and recipe.queries[-1].executed_count==0


def test_controller_float64_dtype_and_tail_bytes_preserved(provider):
    from exp.r17_integration.common.contracts import WireChunk
    from exp.r17_integration.common.fake_adapter import FakeAdapter
    class Float64Wire(FakeAdapter):
        def to_wire(self, normalized, prepared):
            self.counts["inverse"]+=1
            return WireChunk(np.asarray(normalized,np.float64)*np.float64(1.123456789))
    cfg=config_for(provider,commit=10);storage=build_shared_storage(cfg);captures=[]
    policy=make_connection_factory(provider,cfg,storage,capture_sink=captures.append,
                                  adapter_factory=lambda:Float64Wire(provider.model_spec))()
    first=policy.infer({"state":np.arange(4,dtype=np.float32)})
    assert first["actions"].dtype==np.float64 and captures[0].wire_actions.dtype==np.float64
    expected=np.tile(np.arange(4,dtype=np.float64),(10,1))*np.float64(1.123456789)
    assert first["actions"].tobytes()==expected.tobytes()
    tail=policy.infer({"state":np.arange(4,dtype=np.float32),"__execution__":{"decision_seq":0,"executed_count":5}})
    assert tail["actions"].tobytes()==first["actions"][5:].tobytes()
    assert policy.adapter.counts["inverse"]==1


def test_lazy_trace_registry_copies_executed_stages_without_forward(provider,monkeypatch):
    from exp.r17_integration.common import provider as registry
    created=[]
    class Trace:
        def __init__(self,adapter):created.append(self)
        def sample_fields(self,s1,s2,result):
            return {"native_target":s2.data.copy(),"normalized":result.normalized_chunk.copy()}
    monkeypatch.setitem(registry._traces,provider.model_spec.model_id,Trace)
    cfg=config_for(provider);storage=build_shared_storage(cfg)
    plain=make_connection_factory(provider,cfg,storage)()
    assert not created
    captures=[];debug=make_connection_factory(provider,cfg,storage,capture_sink=captures.append)()
    obs={"state":np.arange(4,dtype=np.float32)}
    a,b=plain.infer(obs),debug.infer(obs)
    assert len(created)==1 and plain.adapter.counts==debug.adapter.counts
    assert a["actions"].tobytes()==b["actions"].tobytes()
    captures[0].model_capture["normalized"][:]=999
    assert np.max(debug._chunk)<999


@pytest.mark.parametrize("name", ["checkpoint_identity","norm_identity","layout_identity","feature_identity","sampler_identity","benchmark_protocol"])
def test_bad_binding_rejected(provider,name):
    cfg=config_for(provider)
    cfg.adapter_binding[name]="foreign"
    with pytest.raises(ValueError):
        validate_cache_config(cfg)


def test_unknown_provider_builder_and_nested_yaml_typos_fail(provider,tmp_path):
    cfg=config_for(provider)
    raw=asdict(cfg);raw["key_builder"]["typpe"]="typo"
    path=tmp_path/"bad.yaml";path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match="typpe"):
        load_cache_config(path)
    cfg.adapter_binding["provider"]="unregistered"
    with pytest.raises(ValueError,match="unknown adapter"):
        validate_cache_config(cfg)


def test_wrong_artifact_binding_and_payload_geometry_fail(provider):
    cfg=config_for(provider);storage=build_shared_storage(cfg)
    entry=add_entry(storage,cfg)
    storage._backend.artifact_meta["r17_binding"]["norm_identity"]="bad"
    with pytest.raises(ValueError,match="artifact"):
        make_connection_factory(provider,cfg,storage)
    storage._backend.artifact_meta["r17_binding"]=artifact_binding(cfg)
    p=make_connection_factory(provider,cfg,storage)()
    entry.payload.action_chunk=entry.payload.action_chunk[:5]
    with pytest.raises(ValueError,match="geometry"):
        p.infer({"state":np.arange(4,dtype=np.float32)})
    assert not p.orchestrator.key_builder.cached_data


def test_loopback_two_clients_two_bundles_atomic_reject_and_disconnect(provider,monkeypatch):
    from openpi_client import msgpack_numpy
    import websockets.asyncio.client
    cfg=config_for(provider,k=4,commit=10)
    storage=build_shared_storage(cfg)
    factory=make_connection_factory(provider,cfg,storage)
    created=[]
    def track(base,bid):
        p=factory(base,bid);created.append((bid,p));return p
    monkeypatch.setattr(ws,"_bundles",{})
    monkeypatch.setattr(ws,"_current_bundle",None)
    monkeypatch.setattr(ws,"_bundle_version",0)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    async def run():
        ready=asyncio.Event()
        server=WebsocketPolicyServer(object(),host="127.0.0.1",port=port,concurrent=True,
                 connection_policy_factory=track,metadata={"model":"fake"},ready_callback=ready.set,
                 bundle_validator=make_bundle_validator(provider))
        task=asyncio.create_task(server.run(stop_on_request=True))
        await asyncio.wait_for(ready.wait(),5)
        packer=msgpack_numpy.Packer()
        async def call(client,payload):
            await client.send(packer.pack(payload))
            value=await asyncio.wait_for(client.recv(),5)
            assert isinstance(value,bytes),value
            return msgpack_numpy.unpackb(value)
        try:
            async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}",compression=None) as c1, websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}",compression=None) as c2:
                assert msgpack_numpy.unpackb(await c1.recv())=={"model":"fake"}
                await c2.recv()
                for bid,k in (("one.v1",1),("two.v1",2)):
                    ack=await call(c1,{"__ctrl__":"load_cache_config","bundle_id":bid,
                                      "yaml_content":yaml.safe_dump(asdict(config_for(provider,k=k,commit=10)))})
                    assert ack["__ack__"]=="load_cache_config",ack
                assert (await call(c1,{"__ctrl__":"select_bundle","bundle_id":"one.v1"}))["__ack__"]=="select_bundle"
                assert (await call(c2,{"__ctrl__":"select_bundle","bundle_id":"two.v1"}))["__ack__"]=="select_bundle"
                before=len(created)
                await call(c1,{"__ctrl__":"select_bundle","bundle_id":"one.v1"})
                assert len(created)==before
                obs={"state":np.arange(4,dtype=np.float32)}
                out1,out2=await asyncio.gather(call(c1,obs),call(c2,obs))
                assert out1["__hit_meta__"]["actual_nfe"]==1
                assert out2["__hit_meta__"]["actual_nfe"]==2
                # Identity-bearing generation IDs cannot change semantics.
                ack=await call(c1,{"__ctrl__":"load_cache_config","bundle_id":"one.v1",
                                   "yaml_content":yaml.safe_dump(asdict(config_for(provider,k=2,commit=10)))})
                assert ack["__ack__"]=="error"
                await call(c1,{"__ctrl__":"select_bundle","bundle_id":"one.v1"})
                assert len(created)==before and created[0][1].sampler.num_steps==1
                old_version=ws._bundle_version
                for name in ("checkpoint_identity","layout_identity","sampler_identity"):
                    bad=asdict(config_for(provider));bad["adapter_binding"][name]="wrong"
                    assert (await call(c1,{"__ctrl__":"load_cache_config","bundle_id":"one.v1","yaml_content":yaml.safe_dump(bad)}))["__ack__"]=="error"
                assert ws._bundle_version==old_version
                assert (await call(c1,{"__ctrl__":"select_bundle","bundle_id":"missing"}))["__ack__"]=="error"
                tail=await call(c1,dict(obs,__execution__={"decision_seq":0,"executed_count":5}))
                assert tail["__hit_meta__"]["stage_calls"]==[0,0,0]
                assert created[1][1]._cursor==0
                ack=await call(c1,{"__ctrl__":"episode_start","__task__":"reach","__extra__":{"task_id":2,"orig_idx":4}})
                assert ack["__ack__"]=="episode_start"
                assert created[0][1]._seq==0
                assert (await call(c1,{"__ctrl__":"episode_end"}))["__ack__"]=="episode_end"
                await call(c1,{"__ctrl__":"select_bundle","bundle_id":"two.v1"})
                assert created[0][1].adapter.lifecycle[-1]=="end"
                assert created[-1][1].orchestrator.key_builder is not created[1][1].orchestrator.key_builder
            for _ in range(100):
                if all(p.adapter.lifecycle[-1]=="end" for _,p in created):break
                await asyncio.sleep(.01)
            assert all(p.adapter.lifecycle[-1]=="end" for _,p in created)
        finally:
            server.request_stop()
            await asyncio.wait_for(task,5)
    asyncio.run(run())
