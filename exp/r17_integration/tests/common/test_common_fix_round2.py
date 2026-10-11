"""P1 REVIEW_A2 N1/N2/N4 and incoming common N5/B2N6 witnesses.

R16/R11 are read-only CPU oracles. No real outcomes, external stores or GPUs.
"""
import asyncio
from dataclasses import asdict, replace
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from exp.r17_integration.common.benchmarks import register_benchmark, task_table_identity
from exp.r17_integration.common.config import build_shared_storage, config_from_dict
from exp.r17_integration.common.ir import bound_costs, calibrate_periodic, library_schedule, owner_ir
from exp.r17_integration.common.recipe import make_recipe_factory, target_ir_to_gap, verified_store
from exp.r17_integration.common.serving import make_bundle_validator, make_connection_factory
from exp.r17_integration.tests.common.conftest import config_for, with_recipe


@pytest.mark.parametrize("model,k,pool,loeo", [
    ("pi05",1,"A",False),("pi05",2,"B",False),("pi05",10,"B",False),
    ("groot",1,"B",True),("groot",2,"A",False),("groot",8,"A",False)])
def test_n1_reproduces_r16_own_example_configs(tmp_path,model,k,pool,loeo):
    from exp.offline_search.rounds.r16.test_r16 import synthetic
    from exp.offline_search.rounds.r16 import analyze
    from exp.offline_search.rounds.r16.cost import pricing
    seal,name,pairs,_ = synthetic(tmp_path/"r16_example",model=model,pool=pool,k=k,loeo=loeo)
    ep = analyze.load(seal,name,pairs,pool,k,analyze.Reader())
    n,v,m = (int(x.sum()) for x in (ep.n,ep.v,ep.m))
    assert owner_ir(n,v,m,pricing(model,k)) == pytest.approx(ep.stats()["ir"],rel=0,abs=1e-15)
    assert n == 2*v == 2*m  # R16 example has one MISS and one policy tail.


@pytest.mark.parametrize("model", ["pi05","groot"])
@pytest.mark.parametrize("target", [.45,.40,.32,.25])
def test_n1_matches_r11_library_periodic_calibration(model,target):
    from exp.offline_search.rounds.r11.knob.calibration import ScheduleReplay
    from exp.offline_search.rounds.r11.knob.controller import COSTS
    episodes = [SimpleNamespace(ep=e,task=e%2,steps=np.arange(32),
                guard=np.array([i in (7,19) for i in range(32)]),n_slots=32) for e in range(10)]
    oracle = ScheduleReplay(model,episodes)
    costs = dict(zip(("vision","miss"),COSTS[model]))
    result = calibrate_periodic(target,lambda gap:(oracle.N,oracle.V,oracle.counts("periodic",gap)),costs)
    setting,ir,status = oracle.solve("periodic",target)
    assert result["setting"] == round(setting,4)
    assert result["unrounded_setting"] == setting
    assert result["pred_IR_lib"] == ir
    assert result["pred_IR_rounded"] == oracle.ir("periodic",round(setting,4))
    assert result["solve_status"] == status and result["library_only"]


def test_n1_shortcut_and_unbound_costs_fail_closed(provider,tmp_path):
    with pytest.raises(ValueError,match="library schedule replay"):
        target_ir_to_gap(.25)
    cfg = with_recipe(provider,config_for(provider),tmp_path)
    cfg.recipe.target_ir = .25
    with pytest.raises(ValueError,match="coefficient profile"):
        config_from_dict(asdict(cfg))
    cfg.recipe.ir_costs = bound_costs(cfg,vision=.152,miss=.848,source_identity="r16.pi05.diagnostic")
    cfg.recipe.ir_costs["sampler_identity"] = "another_grid"
    with pytest.raises(ValueError,match="identity mismatch"):
        config_from_dict(asdict(cfg))
    with pytest.raises(ValueError,match="feasible interval"):
        calibrate_periodic(.1,lambda gap:(32,32,32 if gap==0 else 0),{"vision":.152,"miss":.848})


def test_n1_bundle_replay_receipt_counts_actual_tails_and_guard(provider,tmp_path):
    cfg = with_recipe(provider,config_for(provider,commit=10),tmp_path)
    factory = make_recipe_factory(provider)
    recipe = factory(cfg,None)
    store = verified_store(cfg.recipe.store_path,cfg.recipe.store_digest)
    cfg.recipe.guard = False
    replay = library_schedule(recipe,store,cfg)
    assert replay(40) == (12,6,0)
    assert replay(0) == (12,12,12)
    costs = bound_costs(cfg,vision=.152,miss=.848,source_identity="r16.pi05.diagnostic")
    cfg.recipe.target_ir,cfg.recipe.ir_costs = .45,costs
    bound = factory(config_from_dict(asdict(cfg)),None)
    receipt = bound.ir_calibration
    assert receipt["pred_IR_rounded"] == owner_ir(receipt["N"],receipt["V"],receipt["M"],costs)
    assert receipt["pred_IR_rounded"] != pytest.approx(receipt["M"]/receipt["N"])
    assert receipt["store_digest"] == cfg.recipe.store_digest
    assert receipt["task_table_identity"] == cfg.recipe.task_table_identity
    assert bound.fit_evidence["ir_calibration"] == receipt
    again = factory(cfg,None)
    assert again.ir_calibration == receipt and again.ir_calibration is not receipt
    cfg.recipe.guard = True
    guarded = library_schedule(recipe,store,cfg)
    # A constant donor progress stalls: guards force fresh looks and MISS.
    original = recipe._retrieve
    def stationary(*args,**kwargs):
        result,xv,rs = original(*args,**kwargs)
        result.rows[:] = np.flatnonzero(recipe.episode != kwargs["exclude_episode"])[0]
        return result,xv,rs
    recipe._retrieve = stationary
    guarded = library_schedule(recipe,store,cfg)
    assert guarded(40)[2] > replay(40)[2]


def test_n1_review_pi_k10_ir25_repro_uses_owner_ir_not_miss_fraction(provider,tmp_path):
    from exp.offline_search.rounds.r16.cost import pricing
    cfg = with_recipe(provider,config_for(provider,commit=5),tmp_path,steps=32,episodes=10)
    cfg.recipe.guard = False
    cfg.recipe.target_ir = .25
    prices = pricing("pi05",10)
    cfg.recipe.ir_costs = bound_costs(cfg,vision=prices["vision"],miss=prices["miss"],
                                    source_identity="diagnostic.r16.pi05.K10:"+prices["source_sha256"])
    fitted = make_recipe_factory(provider)(cfg,None)
    receipt = fitted.ir_calibration
    assert receipt["pred_IR_rounded"] == pytest.approx(.25,abs=.01)
    assert receipt["M"]/receipt["N"] < .13
    assert fitted.periodic_gap > (1-.25)/.25
    # The review's old gap=3 means roughly25% MISS ->36.4% owner IR.
    replay = library_schedule(fitted,verified_store(cfg.recipe.store_path,cfg.recipe.store_digest),cfg)
    assert owner_ir(*replay(3),prices) == pytest.approx(.364,abs=.01)


@pytest.mark.parametrize("mutation", ["reverse","subset","digest","protocol"])
def test_n2_canonical_order_and_spec_digest_reject_before_store_access(provider,tmp_path,monkeypatch,mutation):
    cfg = with_recipe(provider,config_for(provider),tmp_path)
    if mutation == "reverse": cfg.recipe.ordered_tasks = ["assembly","reach"]
    elif mutation == "subset": cfg.recipe.ordered_tasks = ["assembly"]
    elif mutation == "digest": cfg.recipe.task_table_identity = "stale_benchmark_spec"
    else: cfg.adapter_binding["benchmark_protocol"] = "fake_libero.v1"
    import exp.r17_integration.search.store as store_mod
    def forbidden(*args): raise AssertionError("reached store before canonical table validation")
    monkeypatch.setattr(store_mod,"Store",forbidden)
    with pytest.raises(ValueError,match="canonical benchmark|task_table_identity"):
        make_recipe_factory(provider)(cfg,None)


def test_n2_full_mw_canonical_table_and_benchmark_version_are_bound(provider,tmp_path):
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    spec = FORMAL_SPEC.to_contract()
    assert spec.ordered_tasks.index("assembly-v3") == 0
    assert spec.ordered_tasks.index("reach-v3") == 43
    identity = register_benchmark(spec)
    assert identity == task_table_identity(spec.identity)
    changed = replace(spec,version="new_protocol_version")
    assert register_benchmark(changed) != identity
    cfg = with_recipe(provider,config_for(provider),tmp_path)
    cfg.adapter_binding["benchmark_protocol"] = spec.identity
    cfg.recipe.ordered_tasks = ["reach-v3","assembly-v3"]
    cfg.recipe.task_table_identity = identity
    from exp.r17_integration.common.recipe import validate_recipe
    with pytest.raises(ValueError,match="canonical benchmark"):
        validate_recipe(cfg)
    cfg.recipe.ordered_tasks = list(spec.ordered_tasks)
    validate_recipe(cfg)


def test_n2_worker_registry_import_is_stdlib_only():
    subprocess.run([sys.executable,"-c",
        "import sys; import exp.r17_integration.common.benchmarks; "
        "assert not any(n in sys.modules for n in ('torch','jax','mujoco','transformers')); "
        "assert not any(n.startswith('exp.r17_integration.models.') for n in sys.modules)"],check=True,timeout=10)


def test_n2_attested_request_table_is_checked_on_plain_bundle(provider):
    cfg = config_for(provider,commit=5)
    policy = make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    obs = {"state":np.arange(4,dtype=np.float32)}
    for override in (dict(task_name="reach",task_id=0,task_table_identity="foreign"),
                     dict(task_name="assembly",task_id=0,task_table_identity=task_table_identity("fake_mw.v1"))):
        with pytest.raises(ValueError,match="canonical benchmark"):
            policy.infer(dict(obs,__request__=override))
    assert sum(policy.adapter.counts.values()) == 0
    output = policy.infer(dict(obs,__request__={"task_name":"reach","task_id":0,
                                               "task_table_identity":task_table_identity("fake_mw.v1")}))
    assert output["__hit_meta__"]["source"] == "policy"


@pytest.mark.parametrize("enabled", [False,True])
@pytest.mark.parametrize("task_name,task_id", [("assembly",0),("reach",1)])
def test_n2_query_name_index_mismatch_rejects_before_model_work(provider,tmp_path,enabled,task_name,task_id):
    cfg = with_recipe(provider,config_for(provider,commit=5),tmp_path) if enabled else config_for(provider,commit=5)
    if not enabled:
        cfg.recipe.ordered_tasks = ["reach"]
        cfg.recipe.task_table_identity = task_table_identity("fake_mw.v1")
    policy = make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    policy.on_episode_start(extra_metadata={"task_name":task_name,"task_id":task_id})
    with pytest.raises(ValueError,match="canonical recipe task table"):
        policy.infer({"state":np.arange(4,dtype=np.float32)})
    assert sum(policy.adapter.counts.values()) == 0


def test_n4_store_sha_once_per_load_across_connections_and_factories(provider,tmp_path,monkeypatch):
    import exp.r17_integration.search.store as store_mod
    cfg = with_recipe(provider,config_for(provider),tmp_path)
    storage = build_shared_storage(cfg)
    hashes = []
    original = store_mod.file_sha
    def count(path):
        hashes.append(str(path))
        return original(path)
    monkeypatch.setattr(store_mod,"file_sha",count)
    make_bundle_validator(provider)(cfg,storage)
    store = verified_store(cfg.recipe.store_path,cfg.recipe.store_digest)
    assert len(hashes) == len(store.manifest["arrays"])
    policies = [make_connection_factory(provider,cfg,storage)(None,"default") for _ in range(3)]
    assert len(hashes) == len(store.manifest["arrays"])
    assert policies[0].recipe.action is policies[1].recipe.action
    assert policies[0].recipe.guard is not policies[1].recipe.guard
    path = store.directory/next(iter(store.manifest["arrays"].values()))["file"]
    with path.open("ab") as f:f.write(b"changed immutable generation")
    with pytest.raises(ValueError,match="store changed"):
        verified_store(cfg.recipe.store_path,cfg.recipe.store_digest)
    assert len(hashes) == len(store.manifest["arrays"])


def test_n4_generation_changed_during_single_hash_pass_is_never_cached(provider,tmp_path,monkeypatch):
    import exp.r17_integration.search.store as store_mod
    cfg = with_recipe(provider,config_for(provider),tmp_path)
    original = store_mod.Store
    def concurrent_mutation(path):
        store = original(path)
        array = store.directory/next(iter(store.manifest["arrays"].values()))["file"]
        with array.open("ab") as f:f.write(b"concurrent writer")
        return store
    monkeypatch.setattr(store_mod,"Store",concurrent_mutation)
    with pytest.raises(ValueError,match="changed during verification"):
        verified_store(cfg.recipe.store_path,cfg.recipe.store_digest)


def test_n4_linked_ready_target_mutation_invalidates_verified_identity(provider,tmp_path):
    from pathlib import Path
    cfg = with_recipe(provider,config_for(provider),tmp_path)
    ready = Path(cfg.recipe.store_path)/"READY.json"
    target = ready.with_name("linked_manifest.json")
    ready.rename(target)
    ready.symlink_to(target.name)
    verified_store(cfg.recipe.store_path,cfg.recipe.store_digest)
    with target.open("a") as f:f.write("\n")
    with pytest.raises(ValueError,match="store changed"):
        verified_store(cfg.recipe.store_path,cfg.recipe.store_digest)


def test_n4_websocket_load_and_connection_bind_run_off_event_loop(provider,tmp_path,monkeypatch):
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer, _ControlSocket, _socket
    from openpi.serving import websocket_policy_server as registry
    from openpi_client import msgpack_numpy
    import exp.r17_integration.search.store as store_mod
    cfg = with_recipe(provider,config_for(provider),tmp_path)
    storage = build_shared_storage(cfg)
    main = threading.get_ident()
    work_threads = []
    hashes = []
    started,release = threading.Event(),threading.Event()
    original = store_mod.file_sha
    def count(path):
        hashes.append(threading.get_ident())
        return original(path)
    monkeypatch.setattr(store_mod,"file_sha",count)
    def slow_factory(*args):
        work_threads.append(threading.get_ident())
        started.set()
        assert release.wait(5), "test did not release the worker"
        return make_connection_factory(provider,cfg,storage)(*args)
    server = WebsocketPolicyServer(object(),concurrent=True,connection_policy_factory=slow_factory,
                                   bundle_validator=make_bundle_validator(provider))
    monkeypatch.setattr(registry,"_bundles",{})
    monkeypatch.setattr(registry,"_current_bundle",None)
    packer = msgpack_numpy.Packer()
    class Socket:
        async def recv(self):return packer.pack({"__ctrl__":"select_bundle","bundle_id":"worker"})
    async def run():
        ack = await server._load_r17_bundle({"yaml_content":yaml.safe_dump(asdict(cfg)),"bundle_id":"worker"})
        assert ack["__ack__"] == "load_cache_config"
        count_at_load = len(hashes)
        for _ in range(3):
            started.clear();release.clear()
            socket = _ControlSocket(Socket(),server)
            recv = asyncio.create_task(socket.recv())
            assert await asyncio.wait_for(asyncio.to_thread(started.wait,1),2)
            assert not recv.done()  # Event loop runs while the binding worker is blocked.
            release.set()
            await recv
            token = _socket.set(socket)
            try:policy = server._connection_policy_factory(server._policy,"worker")
            finally:_socket.reset(token)
            assert socket.policy is policy
            assert len(hashes) == count_at_load
        assert hashes and all(t != main for t in hashes+work_threads)
    asyncio.run(run())


def test_n5_queue_wait_service_fields_and_missing_instrumentation(provider):
    from exp.r17_integration.common.fake_adapter import FakeAdapter
    class Queued(FakeAdapter):
        wait = 0.
        def drain_queue_wait_ms(self):
            value,self.wait = self.wait,0.
            return value
        def run_stage1(self,*args):
            begin = time.perf_counter();time.sleep(.003)
            self.wait += (time.perf_counter()-begin)*1000
            return super().run_stage1(*args)
    cfg = config_for(provider,commit=5)
    storage = build_shared_storage(cfg)
    policy = make_connection_factory(provider,cfg,storage,adapter_factory=lambda:Queued(provider.model_spec))()
    meta = policy.infer({"state":np.arange(4,dtype=np.float32)})["__hit_meta__"]
    assert meta["queue_wait_ms"] >= 3
    assert meta["service_ms"] == pytest.approx(meta["cost_ms"]-meta["queue_wait_ms"])
    assert meta["stage_service_ms"]["stage1"] == pytest.approx(meta["stage_ms"]["stage1"]-meta["stage_queue_wait_ms"]["stage1"])
    assert meta["pricing_time_field"] == "service_ms" and meta["gpu_service_ms"] is None
    class Unknown(FakeAdapter):drain_queue_wait_ms = None
    unknown = make_connection_factory(provider,cfg,storage,adapter_factory=lambda:Unknown(provider.model_spec))()
    absent = unknown.infer({"state":np.arange(4,dtype=np.float32)})["__hit_meta__"]
    assert absent["queue_wait_ms"] is absent["service_ms"] is absent["pricing_time_field"] is None
    assert not absent["pricing_eligible"]


def test_b2_n6_provider_admission_hooks_apply_to_common_factory_and_capture(provider):
    from exp.r17_integration.common.provider import binding_for, validate_binding
    calls = []
    def bind(binding,sampler):
        calls.append("binding")
        if sampler.num_steps==1 and not binding.get("allow_lossy_sampler"):
            raise ValueError("lossy sampler study opt-in required")
    def config(cfg):
        calls.append("config")
        if cfg.adapter_binding.get("allow_lossy_sampler") and any(cp.enabled for cp in cfg.checkpoints.values()):
            raise ValueError("lossy study must be plain")
    from exp.r17_integration.common.provider import register_provider
    p = replace(provider,provider_id=provider.provider_id+".admission",binding_validator=bind,config_validator=config)
    register_provider(p)
    with pytest.raises(ValueError,match="study opt-in"):
        binding_for(p,"fake_mw.v1",num_steps=1)
    cfg = config_for(p)
    cfg.adapter_binding = binding_for(p,"fake_mw.v1",num_steps=1,allow_lossy_sampler=True,sampler_study_label="plain.k1.diagnostic")
    with pytest.raises(ValueError,match="must be plain"):
        make_connection_factory(p,cfg,build_shared_storage(cfg))()
    for cp in cfg.checkpoints.values():cp.enabled=False
    records = []
    policy = make_connection_factory(p,cfg,build_shared_storage(cfg),capture_sink=records.append)()
    out = policy.infer({"state":np.arange(4,dtype=np.float32)})
    assert out["__hit_meta__"]["sampler_study_label"] == records[0].diagnostics["sampler_study_label"] == "plain.k1.diagnostic"
    assert "config" in calls and "binding" in calls
    cfg.adapter_binding["allow_lossy_sampler"] = "true"
    with pytest.raises(ValueError,match="boolean"):validate_binding(cfg.adapter_binding)
