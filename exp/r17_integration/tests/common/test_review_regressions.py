"""Behavioral witnesses for REVIEW_A #2–7/#9 and REVIEW_B B1/B2/B9."""
import asyncio
import copy
from dataclasses import asdict, replace
from pathlib import Path
import socket
import subprocess
import time
import uuid

import numpy as np
import pytest
import yaml

from exp.r17_integration.common.config import (
    CacheConfig, RecipeConfig, build_shared_storage, config_from_dict)
from exp.r17_integration.common.contracts import CaptureSpec, digest
from exp.r17_integration.common.fake_adapter import FakeAdapter, fake_provider, fake_spec
from exp.r17_integration.common.provider import (
    artifact_binding, binding_for, validate_artifact)
from exp.r17_integration.common.recipe import (
    bundle_identity, fit_identity, make_recipe_factory, recipe_identity, target_ir_to_gap)
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.common.serving import make_bundle_validator, make_connection_factory
from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
from exp.r17_integration.tests.common.conftest import config_for, with_recipe


OBS = {"state": np.arange(4, dtype=np.float32)}


def context(idx=3):
    return dict(campaign="fix1", task_name="reach", task_id=0, orig_init_state_idx=idx,
                seed=7, explicit_seed=100+idx, arm="full", attempt=1,
                bundle_generation="generation1", ack_protocol="r17.execution.v1")


def strict_policy(provider, commit=10, **kwargs):
    cfg = config_for(provider, commit=commit)
    cfg.diagnostic_context = False
    return make_connection_factory(provider, cfg, build_shared_storage(cfg), **kwargs)()


def test_sealed_shared_source_is_head_and_opt_in_types_are_local():
    files = ["src/openpi/cache/config.py", "src/openpi/cache/types.py",
             "src/openpi/cache/storage_types.py", "src/openpi/serving/websocket_policy_server.py"]
    for name in files:
        assert Path(name).read_bytes() == subprocess.check_output(["git", "show", "HEAD:" + name])
    from openpi.cache.config import CacheConfig as StockConfig
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer as StockServer
    assert "adapter_binding" not in StockConfig.__dataclass_fields__
    assert "recipe" not in StockConfig.__dataclass_fields__
    assert WebsocketPolicyServer is not StockServer
    assert issubclass(WebsocketPolicyServer, StockServer)


def test_q_stationary_teacher_metric_fails_with_explicit_data_error():
    from exp.r17_integration.common.layers.math import fit_metric
    from exp.r17_integration.common.layers import LayoutRecipe
    from exp.r17_integration.tests.common.test_layers import library
    rows = np.ones((12,4),np.float64)
    heads = np.random.default_rng(42).normal(size=(12,20))
    with pytest.raises(ValueError,match="zero-scatter library metric"):
        fit_metric(rows,heads,np.repeat(np.arange(3),4))
    layout,data = library()
    constant = replace(data,features={name:np.zeros_like(v) for name,v in data.features.items()},
                       state=np.zeros_like(data.state))
    with pytest.raises(ValueError,match="zero-scatter library metric"):
        LayoutRecipe(layout,50,corrector=False).fit(constant)


def test_nullable_yaml_and_unknown_nested_recipe_keys(provider):
    raw = asdict(config_for(provider))
    assert raw["checkpoints"]["cp1"]["search_strategy"]["field_similarity"] is None
    loaded = config_from_dict(raw)
    assert isinstance(loaded, CacheConfig) and isinstance(loaded.recipe, RecipeConfig)
    raw["recipe"]["guardd"] = False
    with pytest.raises(ValueError, match="guardd"):
        config_from_dict(raw)
    raw = asdict(config_for(provider))
    raw["checkpoints"]["cp1"]["judge"]["factors"] = [{"type": "fixture", "typpo": 1}]
    with pytest.raises(ValueError, match="typpo"):
        config_from_dict(raw)


def test_full_and_plain_low_k_select_recipe_per_bundle(provider, tmp_path, monkeypatch):
    from openpi.serving import websocket_policy_server as registry
    full = with_recipe(provider, config_for(provider, commit=5), tmp_path, corrector=True)
    cfgs = {"full.recipe": full, "plain.k2": config_for(provider,k=2,commit=5),
            "plain.k1": config_for(provider,k=1,commit=5)}
    gap = copy.deepcopy(full)
    gap.recipe.guard = False
    gap.recipe.corrector = False
    gap.recipe.periodic_gap = 2.
    cfgs["recipe.gap2"] = gap
    bundles = {bid: registry.CurrentCacheBundle("inline", cfg, build_shared_storage(cfg), i, None)
               for i, (bid, cfg) in enumerate(cfgs.items(),1)}
    monkeypatch.setattr(registry,"_bundles",bundles)
    factory = make_connection_factory(provider,full,bundles["full.recipe"].shared_storage)
    policies = {bid: factory(None,bid) for bid in bundles}
    full_policy = policies["full.recipe"]
    full_policy.on_episode_start(extra_metadata=context())
    assert full_policy.sampler.num_steps == 4 and full_policy.recipe.use_corrector
    assert full_policy.infer(OBS)["__hit_meta__"]["source"] == "library"
    for bid,k in (("plain.k2",2),("plain.k1",1)):
        policy = policies[bid]
        assert policy.recipe is None
        assert policy.infer(OBS)["__hit_meta__"]["actual_nfe"] == k
    descent = policies["recipe.gap2"]
    assert not descent.recipe.use_guard and not descent.recipe.use_corrector
    assert descent.recipe.periodic_gap == 2.
    assert descent.bundle_identity != full_policy.bundle_identity
    second = factory(None,"full.recipe")
    assert second.recipe is not full_policy.recipe and second.recipe.guard is not full_policy.recipe.guard
    assert second.recipe._memo_step == -1 and full_policy.recipe._memo_step == 0
    assert second.recipe.action is full_policy.recipe.action


@pytest.mark.parametrize("field,value", [("guard",False),("corrector",True),
    ("periodic_gap",0.),("k",4),("kref",3),("store_digest","foreign")])
def test_recipe_identity_seals_each_arm_setting(provider,tmp_path,field,value):
    cfg = with_recipe(provider,config_for(provider),tmp_path)
    other = copy.deepcopy(cfg)
    setattr(other.recipe,field,value)
    assert recipe_identity(cfg) != recipe_identity(other)
    assert bundle_identity(cfg) != bundle_identity(other)
    if field in ("guard","periodic_gap"):
        assert fit_identity(cfg) == fit_identity(other)
    else:
        assert fit_identity(cfg) != fit_identity(other)
    relocated = copy.deepcopy(cfg)
    relocated.recipe.store_path = "/replica/same_content"
    assert bundle_identity(cfg) == bundle_identity(relocated)


def test_teacher_library_is_independent_of_miss_grid_but_not_model(provider,tmp_path):
    teacher = with_recipe(provider,config_for(provider),tmp_path)
    reduced_spec = replace(provider.model_spec,default_sampler=provider.sampler_for_steps(2))
    reduced = fake_provider("reduced."+uuid.uuid4().hex,reduced_spec)
    cfg = copy.deepcopy(teacher)
    cfg.adapter_binding = binding_for(reduced,"fake_mw.v1",num_steps=2)
    assert make_recipe_factory(reduced)(cfg,None).layout.identity == reduced_spec.layout.identity
    assert artifact_binding(cfg)["sampler_identity"] == teacher.recipe.teacher_sampler_identity
    assert cfg.adapter_binding["sampler_identity"] != artifact_binding(cfg)["sampler_identity"]
    wrong = fake_provider("wrong."+uuid.uuid4().hex,replace(reduced_spec,code_digest="other_code"))
    bad = copy.deepcopy(cfg)
    bad.adapter_binding = binding_for(wrong,"fake_mw.v1",num_steps=2)
    with pytest.raises(ValueError,match="model/norm/layout/features/protocol"):
        make_recipe_factory(wrong)(bad,None)
    cfg.recipe.teacher_sampler_identity = cfg.adapter_binding["sampler_identity"]
    with pytest.raises(ValueError,match="teacher sampler"):
        make_recipe_factory(reduced)(cfg,None)


def test_stock_action_artifact_teacher_provenance_remains_strict(provider):
    from exp.r17_integration.tests.common.conftest import add_entry
    teacher = config_for(provider)
    storage = build_shared_storage(teacher)
    add_entry(storage,teacher)
    reduced = config_for(provider,k=2)
    with pytest.raises(ValueError,match="artifact"):
        validate_artifact(reduced,storage)
    reduced.adapter_binding["library_sampler_identity"] = provider.model_spec.default_sampler.identity
    validate_artifact(reduced,storage)
    wrong = fake_provider("wrong.artifact."+uuid.uuid4().hex,replace(provider.model_spec,code_digest="other"))
    reduced.adapter_binding = binding_for(wrong,"fake_mw.v1",num_steps=2,
        library_sampler_identity=provider.model_spec.default_sampler.identity)
    with pytest.raises(ValueError,match="artifact"):
        validate_artifact(reduced,storage)


def test_actual_stock_loader_retained_model_spec_proves_code_provenance(provider,tmp_path):
    from exp.r17_integration.search.build_library import save_stock_artifact
    from exp.r17_integration.search.store import Store
    teacher = with_recipe(provider,config_for(provider),tmp_path)
    artifact = tmp_path / "teacher.pkl"
    save_stock_artifact(Store(teacher.recipe.store_path),artifact)
    cfg = copy.deepcopy(teacher)
    cfg.backend.in_memory.preload_path = str(artifact)
    storage = build_shared_storage(cfg)
    assert "r17_manifest" not in storage.artifact_meta
    assert "model_spec" in storage.artifact_meta["model"]
    storage.artifact_meta["model"]["r17_binding"].pop("retrieval_model_identity",None)
    validate_artifact(cfg,storage)
    storage.artifact_meta["model"]["model_spec"]["code_digest"] = "foreign"
    with pytest.raises(ValueError,match="artifact"):
        validate_artifact(cfg,storage)


@pytest.mark.parametrize("missing",["campaign","task_name","task_id","orig_init_state_idx","seed","explicit_seed"])
def test_missing_real_context_fails_before_any_model_work(provider,missing):
    policy = strict_policy(provider)
    data = context()
    data.pop(missing)
    policy.on_episode_start(extra_metadata=data)
    with pytest.raises(ValueError,match="missing R17 request context"):
        policy.infer(OBS)
    assert sum(policy.adapter.counts.values()) == 0


def test_original_init_alias_and_suite_identity_change_noise_and_periodic_coin(provider):
    from exp.r17_integration.common.layers.periodic import periodic
    requests = []
    for idx,task,tid in ((3,"spatial/reach",0),(7,"spatial/reach",0),(3,"object/reach",10)):
        policy = strict_policy(provider)
        data = context(idx)
        data.update(task_name=task,task_id=tid)
        policy.on_episode_start(extra_metadata=data)
        requests.append(policy._request({}))
    assert [r.original_init_idx for r in requests] == [3,7,3]
    assert len({RequestRNG(r).identity for r in requests}) == 3
    assert len({RequestRNG(r).seed_for() for r in requests[:2]}) == 2
    assert len({periodic(r,[],[],1.25)[1]["coin"] for r in requests}) == 3
    policy = strict_policy(provider)
    policy.on_episode_start(task_id=41,extra_metadata={"orig_init_state_idx":7,"ack_protocol":"r17.execution.v1"})
    with pytest.raises(ValueError,match="missing R17"):
        policy.infer(OBS)


def test_extended_commit_declares_ack_and_refuses_silent_downgrade(provider):
    policy = strict_policy(provider)
    with pytest.raises(ValueError,match="ack_protocol"):
        policy.on_episode_start(extra_metadata={"task_name":"reach"})
    policy.on_episode_start(extra_metadata=context())
    first = policy.infer(OBS)
    counts = dict(policy.adapter.counts)
    with pytest.raises(ValueError,match="cannot silently downgrade"):
        policy.infer(OBS)
    assert policy.adapter.counts == counts
    tail = policy.infer(dict(OBS,__execution__={"decision_seq":0,"executed_count":5}))
    assert tail["__hit_meta__"]["stage_calls"] == (0,0,0)
    assert tail["actions"].tobytes() == first["actions"][5:].tobytes()
    with pytest.raises(ValueError,match="final execution"):
        policy.on_episode_end()


def test_final_ack_binds_request_and_cursor_before_mutating_state(provider):
    class Sink:
        def __init__(self):self.rows,self.acks = [],[]
        def __call__(self,row):self.rows.append(row)
        def on_execution(self,ack):self.acks.append(ack)
    sink = Sink()
    policy = strict_policy(provider,commit=5,capture_sink=sink)
    policy.on_episode_start(extra_metadata=context())
    policy.infer(OBS)
    request = asdict(sink.rows[-1].request)
    wrong = dict(request,original_init_idx=7)
    for extra in ({"request":wrong},{"cursor_end":4}):
        with pytest.raises(ValueError,match="mismatch"):
            policy.acknowledge_execution(dict(decision_seq=0,executed_count=3,terminal=True,**extra))
        assert policy._cursor == 0 and not sink.acks
    counts = dict(policy.adapter.counts)
    policy.on_episode_end(execution=dict(decision_seq=0,executed_count=3,request=request,cursor_end=3))
    assert sink.acks[-1]["request"] == sink.rows[-1].request
    assert sink.acks[-1]["cursor_end"] == sink.acks[-1]["executed_count"] == 3
    assert sink.acks[-1]["terminal"] and policy.adapter.counts == counts


@pytest.mark.parametrize("commit",[7,12])
def test_nondivisible_metaworld_commit_is_rejected(provider,commit):
    with pytest.raises(ValueError,match="multiple"):
        binding_for(provider,"fake_mw.v1",commit_horizon=commit)


def test_libero_commit32_requires_negotiated_partial_tail_and_ack():
    provider = fake_provider("partial."+uuid.uuid4().hex,fake_spec(horizon=32,action_dim=7))
    cfg = config_for(provider)
    cfg.diagnostic_context = False
    cfg.adapter_binding = binding_for(provider,"fake_libero.v1",commit_horizon=32,allow_partial_commit=True)
    policy = make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    with pytest.raises(ValueError,match="partial_tail_protocol"):
        policy.on_episode_start(extra_metadata=context())
    policy.on_episode_start(extra_metadata=dict(context(),partial_tail_protocol="r17.partial_tail.v1"))
    obs = {"state":np.arange(7,dtype=np.float32)}
    first = policy.infer(obs)
    for seq in range(1,7):
        result = policy.infer(dict(obs,__execution__={"decision_seq":seq-1,"executed_count":5}))
        assert result["__hit_meta__"]["stage_calls"] == (0,0,0)
        assert result["actions"].tobytes() == first["actions"][5*seq:].tobytes()
    assert len(result["actions"]) == 2 and policy.adapter.counts["stage1"] == 1
    receipt = policy.acknowledge_execution({"decision_seq":6,"executed_count":2,"cursor_end":32,"terminal":True})
    assert receipt["cursor_end"] == 32 and receipt["request"]["decision_seq"] == 6
    policy.on_episode_end()


def test_actual_host_cost_stage_ledger_and_capture_pricing_exclusion(provider):
    class Slow(FakeAdapter):
        def run_stage1(self,prepared):
            time.sleep(.002)
            return super().run_stage1(prepared)
        def run_stage2(self,s1,**kwargs):
            time.sleep(.002)
            return super().run_stage2(s1,**kwargs)
        def sample(self,*args):
            time.sleep(.002)
            return super().sample(*args)
    cfg = config_for(provider,commit=10)
    storage = build_shared_storage(cfg)
    plain = make_connection_factory(provider,cfg,storage,adapter_factory=lambda:Slow(provider.model_spec))()
    captures = []
    debug = make_connection_factory(provider,cfg,storage,adapter_factory=lambda:Slow(provider.model_spec),capture_sink=captures.append)()
    a,b = plain.infer(OBS),debug.infer(OBS)
    assert a["actions"].tobytes() == b["actions"].tobytes()
    for out in (a,b):
        meta = out["__hit_meta__"]
        assert meta["cost_ms"] >= 6 and all(meta["stage_ms"][s] >= 2 for s in ("stage1","stage2","stage3"))
        assert meta["cost_ms"] >= sum(meta["stage_ms"].values())
        assert meta["timing_kind"] == "host_wall"
    assert a["__hit_meta__"]["pricing_eligible"] and not a["__hit_meta__"]["capture_enabled"]
    assert not b["__hit_meta__"]["pricing_eligible"] and b["__hit_meta__"]["capture_enabled"]
    assert captures[0].diagnostics["cost_ms"] == b["__hit_meta__"]["cost_ms"]
    assert captures[0].diagnostics["capture_overhead_ms"] > 0
    tail = plain.infer(dict(OBS,__execution__={"decision_seq":0,"executed_count":5}))["__hit_meta__"]
    assert tail["cost_ms"] > 0 and tail["stage_ms"]["state"] > 0
    assert all(tail["stage_ms"][name] == 0 for name in ("prepare","stage1","retrieval","recipe","stage2","stage3","to_wire"))
    native_capture = make_connection_factory(provider,cfg,storage,capture_spec=CaptureSpec(True,True))()
    assert not native_capture.infer(OBS)["__hit_meta__"]["pricing_eligible"]


def test_core_aware_sink_has_actual_bundle_sampler(provider):
    created = []
    def sink_factory(core):
        created.append((core,core.sampler.identity,core.bundle_identity))
        return lambda record: None
    cfg = config_for(provider,k=2)
    factory = make_connection_factory(provider,cfg,build_shared_storage(cfg),capture_sink_factory=sink_factory)
    a,b = factory(),factory()
    assert created[0][0] is a and created[1][0] is b
    assert created[0][1] == provider.sampler_for_steps(2).identity
    assert a.capture.enabled and b.capture.enabled and a.capture_sink is not b.capture_sink


def test_real_ws_recipe_load_is_atomic_and_terminal_ack_is_recorded(provider,tmp_path,monkeypatch):
    from openpi.serving import websocket_policy_server as registry
    from openpi_client import msgpack_numpy
    import websockets.asyncio.client
    class Sink:
        def __init__(self):self.records,self.acks = [],[]
        def __call__(self,row):self.records.append(row)
        def on_execution(self,ack):self.acks.append(ack)
    sink = Sink()
    cfg = config_for(provider,commit=10)
    cfg.diagnostic_context = False
    recipe_cfg = with_recipe(provider,copy.deepcopy(cfg),tmp_path)
    factory = make_connection_factory(provider,cfg,build_shared_storage(cfg),capture_sink=sink)
    monkeypatch.setattr(registry,"_bundles",{})
    monkeypatch.setattr(registry,"_current_bundle",None)
    monkeypatch.setattr(registry,"_bundle_version",0)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port = sock.getsockname()[1]
    async def run():
        ready = asyncio.Event()
        server = WebsocketPolicyServer(object(),host="127.0.0.1",port=port,concurrent=True,
            connection_policy_factory=factory,bundle_validator=make_bundle_validator(provider),ready_callback=ready.set)
        task = asyncio.create_task(server.run(stop_on_request=True))
        await asyncio.wait_for(ready.wait(),5)
        packer = msgpack_numpy.Packer()
        async def call(client,value):
            await client.send(packer.pack(value))
            raw = await asyncio.wait_for(client.recv(),5)
            assert isinstance(raw,bytes),raw
            return msgpack_numpy.unpackb(raw)
        try:
            async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}",compression=None) as client:
                await client.recv()
                load = {"__ctrl__":"load_cache_config","bundle_id":"recipe.full.v1","yaml_content":yaml.safe_dump(asdict(recipe_cfg))}
                ack = await call(client,load)
                assert ack["__ack__"] == "load_cache_config",ack
                assert ack["bundle_identity"] == bundle_identity(recipe_cfg)
                version = registry._bundle_version
                for key,value in (("store_digest","wrong"),("teacher_sampler_identity","wrong"),("fit_identity","wrong")):
                    bad = asdict(recipe_cfg)
                    bad["recipe"][key] = value
                    answer = await call(client,dict(load,bundle_id="bad."+key,yaml_content=yaml.safe_dump(bad)))
                    assert answer["__ack__"] == "error"
                    assert registry._bundle_version == version and "bad."+key not in registry._bundles
                assert (await call(client,{"__ctrl__":"episode_start","bundle_id":"recipe.full.v1",
                    "__task__":"reach","__extra__":dict(context(),bundle_identity=ack["bundle_identity"])}))["__ack__"] == "episode_start"
                result = await call(client,OBS)
                assert result["__hit_meta__"]["source"] == "library"
                assert result["__hit_meta__"]["stage_ms"]["stage2"] == result["__hit_meta__"]["stage_ms"]["stage3"] == 0
                assert (await call(client,{"__ctrl__":"episode_end"}))["__ack__"] == "error"
                answer = await call(client,{"__ctrl__":"episode_end","__execution__":{
                    "decision_seq":0,"executed_count":3,"cursor_end":3}})
                assert answer["__ack__"] == "episode_end"
                execution = answer["execution"]
                assert execution["terminal"] and execution["executed_count"] == 3 and execution["cursor_end"] == 3
                assert execution["request"]["original_init_idx"] == 3
                assert execution["request"]["decision_seq"] == 0
                assert len(sink.records) == len(sink.acks) == 1
                assert sink.acks[0]["terminal"] and sink.acks[0]["request"] == sink.records[0].request
                # Last-decision ack requires no extra model infer.
                assert sink.records[0].stage_calls == (1,0,0)
        finally:
            server.request_stop()
            await asyncio.wait_for(task,5)
    asyncio.run(run())
