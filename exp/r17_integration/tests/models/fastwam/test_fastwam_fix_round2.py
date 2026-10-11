"""Round-2 reviewer regressions; all model forwards here are CPU fixtures."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.models.fastwam.loader import LoadSpec, load_adapter
from exp.r17_integration.models.fastwam.protocols import execution_protocol, validate_benchmark_specs
from exp.r17_integration.tests.models.fastwam.conftest import make_adapter, observation, request, fixture_benchmark_spec
from exp.r17_integration.tests.models.fastwam.test_fastwam_serving import config_for


def libero_spec(profile="system", **changes):
    spec = fixture_benchmark_spec("libero")
    if profile == "released_libero":
        spec = replace(spec,seed=42,seed_mode="fixed",replan_steps=32,
                       max_steps_by_suite=(280,280,300,520),protocol_id="cpu_released_reference")
    return replace(spec,**changes)


def load_spec(benchmark="libero",profile="system",spec=None,**changes):
    if spec is None:
        spec = libero_spec(profile) if benchmark == "libero" else mw_spec()
    values = dict(benchmark=benchmark,checkpoint="unused_immutable",wan_root="unused",
        base="unused_base" if benchmark=="metaworld" else "",execution_protocol=profile,
        benchmark_protocols=(spec.identity,),benchmark_specs=(spec.manifest(),))
    return LoadSpec(**{**values,**changes})


def mw_spec(**changes):
    from exp.r17_integration.benchmarks.spec import BenchmarkSpec
    from exp.metaworld.tasks import BENCHMARK, MT50_TASKS
    return BenchmarkSpec(BENCHMARK,"mt50_seed7_corner2_r17_v1",MT50_TASKS,**changes)


@pytest.mark.parametrize("benchmark,profile",[("libero","system"),("libero","released_libero"),("metaworld","system")])
def test_n3_p2_manifest_json_roundtrip_matches_execution_profile(benchmark,profile):
    spec = load_spec(benchmark,profile)
    serialized = LoadSpec(**json.loads(json.dumps(asdict(spec))))
    valid = validate_benchmark_specs(serialized)
    assert len(valid)==1 and valid[0].identity==spec.benchmark_protocols[0]
    from exp.r17_integration.common.benchmarks import get_benchmark
    assert get_benchmark(valid[0].identity).ordered_tasks==tuple(r[0] for r in valid[0].tasks)
    if profile=="system":
        assert execution_protocol(benchmark,profile).seed_rule=="explicit_benchmark_seed"


@pytest.mark.parametrize("changes",[
    {"replan_steps":32},{"settle_steps":11},{"wire_size":96},
    {"seed":42},{"seed_mode":"fixed"},{"max_steps_by_suite":(280,280,300,520)}])
def test_n3_libero_mismatch_rejected_before_checkpoint_probe(changes,monkeypatch):
    import exp.r17_integration.models.fastwam.loader as loader
    def forbidden(*args): pytest.fail("bad protocol reached checkpoint probe")
    monkeypatch.setattr(loader,"inspect_config",forbidden)
    spec = load_spec(spec=libero_spec(**changes))
    with pytest.raises(ValueError,match="execution parameters"):
        load_adapter(spec)


def test_n3_released_system_mixture_identity_tamper_and_missing_docs_fail_closed():
    for profile,other in (("system","released_libero"),("released_libero","system")):
        with pytest.raises(ValueError,match="execution parameters"):
            validate_benchmark_specs(load_spec(profile=profile,spec=libero_spec(other)))
    original = load_spec()
    tampered = {**original.benchmark_specs[0],"protocol_identity":"forged"}
    with pytest.raises(ValueError,match="reconstructed"):
        validate_benchmark_specs(replace(original,benchmark_specs=(tampered,)))
    with pytest.raises(ValueError,match="allowlisted"):
        validate_benchmark_specs(replace(original,benchmark_protocols=("sha256:wrong",)))
    with pytest.raises(ValueError,match="benchmark_specs"):
        validate_benchmark_specs(replace(original,benchmark_specs=(),diagnostic=True))
    assert validate_benchmark_specs(replace(original,benchmark_specs=(),diagnostic=True,
        benchmark_protocols=("fake_p3_diagnostic",)))==()
    with pytest.raises(ValueError,match="benchmark_specs"):
        validate_benchmark_specs(replace(original,benchmark_specs=(),benchmark_protocols=("fake_p3_diagnostic",)))
    for key,value in (("task_table_identity","forged"),("ordered_tasks",["wrong"])):
        with pytest.raises(ValueError,match=key):
            validate_benchmark_specs(replace(original,benchmark_specs=({**original.benchmark_specs[0],key:value},)))


@pytest.mark.parametrize("changes",[
    {"replan_steps":1},{"settle_steps":14},{"render_size":256},
    {"seed":42},{"camera":"corner"},{"max_policy_controls":200},
    {"rotation_degrees":0}])
def test_n3_metaworld_mismatch_rejected(changes):
    with pytest.raises(ValueError,match="canonical"):
        validate_benchmark_specs(load_spec("metaworld",spec=mw_spec(**changes)))


@pytest.mark.parametrize("benchmark,changed",[
    ("metaworld","image"),("metaworld","state"),("metaworld","prompt"),
    ("libero","image"),("libero","image2"),("libero","state"),("libero","prompt")])
def test_n4_native_resume_rejects_other_observation_same_prompt(benchmark,changed):
    a = make_adapter(benchmark)
    obs = observation(benchmark)
    req = request(explicit_seed=42)
    s2 = a.run_stage2(a.run_stage1(a.prepare(obs,req)))
    sampler,rng = a.sampler_for_steps(10),RequestRNG(req)
    result = a.sample(s2,sampler,rng,CaptureSpec(True,True))
    snapshot = result.snapshots[4]
    equal = a.run_stage2(a.run_stage1(a.prepare({k:v.copy() if hasattr(v,"copy") else v for k,v in obs.items()},req)))
    resumed = a.resume(equal,snapshot,sampler,rng)
    torch.testing.assert_close(resumed.normalized_chunk,result.normalized_chunk,rtol=0,atol=0)
    other = {**obs,changed:(obs[changed]+1 if changed!="prompt" else obs[changed]+" new")}
    other_s2 = a.run_stage2(a.run_stage1(a.prepare(other,req)))
    head_calls = len(a.model.calls["head"])
    with pytest.raises(ValueError,match="conditioning"):
        a.resume(other_s2,snapshot,sampler,rng)
    assert len(a.model.calls["head"])==head_calls


@pytest.mark.parametrize("benchmark",["libero","metaworld"])
def test_n5_cost_fraction_includes_prepare_and_wire(benchmark,monkeypatch):
    from exp.r17_integration.models.fastwam import measure
    ticks = iter(range(1000))
    monkeypatch.setattr(measure.time,"perf_counter",lambda:next(ticks))
    a = make_adapter(benchmark)
    report = measure.measure_adapter(a,[observation(benchmark)],repeats=1)
    row = report["rows"][0]
    assert row["prepare_ms"]==1000 and row["wire_ms"]==1000
    assert row["key_ms"]==row["prepare_ms"]+row["stage1_ms"]
    assert row["hit_path_ms"]==row["key_ms"]+row["wire_ms"]
    assert row["key_over_full"]==row["key_ms"]/row["full_ms"]
    assert row["hit_over_full"]==row["hit_path_ms"]/row["full_ms"]
    assert row["hit_over_full"]>row["key_over_full"]>row["stage1_ms"]/row["full_ms"]
    assert row["retrieval_search_ms"] is None
    assert "excluded" in report["cost_definitions"]["retrieval_search_ms"]
    assert "P1_host_wall_ledger" in report["timing_scope"]


@pytest.mark.parametrize("crash",[False,True])
def test_n2_serve_debug_uses_connection_sampler_and_closes(tmp_path,monkeypatch,crash):
    from exp.r17_integration.models.fastwam import serve
    from exp.r17_integration.models.fastwam.provider import register_provider
    from exp.r17_integration.common import config as config_module
    from exp.r17_integration.common import websocket_policy_server as server_module
    from exp.r17_integration.debug import capture_bridge
    from openpi.serving import websocket_policy_server as stock
    a = make_adapter()
    provider = register_provider(a,provider_id="p3_debug_"+tmp_path.name)
    cfg = config_for(provider,k=2)
    monkeypatch.setattr(serve,"load_adapter",lambda *args,**kwargs:a)
    import exp.r17_integration.models.fastwam.provider as providers
    monkeypatch.setattr(providers,"register_provider",lambda *args:provider)
    monkeypatch.setattr(config_module,"load_cache_config",lambda *args:cfg)
    captured = []
    real_capture = capture_bridge.make_capture_sink_factory
    def capture(*args,**kwargs):
        factory = real_capture(*args,**kwargs);captured.append(factory);return factory
    monkeypatch.setattr(capture_bridge,"make_capture_sink_factory",capture)
    cfg1 = config_for(provider,k=1)
    monkeypatch.setattr(stock,"_bundles",{"k1":SimpleNamespace(cache_config=cfg1,
        shared_storage=config_module.build_shared_storage(cfg1),yaml_id="cpu_yaml_k1")})
    class Server:
        def __init__(self,policy,**kwargs): self.policy,self.factory=policy,kwargs["connection_policy_factory"]
        def serve_forever(self):
            second = self.factory(bundle_id="k1")
            for c in (self.policy,second):
                c.on_episode_start(task="reach-v3",episode_id=0,extra_metadata={
                    "campaign":"p3_debug_cpu","arm":"plain","attempt":1,
                    "task_data_hash":"cpu_test_task_data","task_uid":"cpu_"+str(c.sampler.num_steps)})
                c.infer(observation())
                assert c.capture_sink.bridge.binding["sampler"]["num_steps"]==c.sampler.num_steps
                c.on_episode_end(execution={"decision_seq":0,"executed_count":5,"terminal":True})
            assert self.policy.capture_sink is not second.capture_sink
            assert self.policy.capture_sink.bridge is not second.capture_sink.bridge
            if crash: raise RuntimeError("test server stop")
    monkeypatch.setattr(server_module,"WebsocketPolicyServer",Server)
    desc = tmp_path/"descriptor.json"
    desc.write_text(json.dumps({"benchmark":"metaworld","checkpoint":"unused","base":"unused","wan_root":"unused"}))
    monkeypatch.setattr(sys,"argv",["serve","--descriptor",str(desc),"--cache-config","unused",
        "--port","9009","--debug-directory",str(tmp_path/"debug")])
    if crash:
        with pytest.raises(RuntimeError,match="test server stop"): serve.main()
    else: serve.main()
    assert len(captured)==1 and len(captured[0].bridges)==2
    for bridge in captured[0].bridges.values():
        assert bridge._closed
        manifest = json.loads((bridge.directory/"r17_capture.json").read_text())
        assert manifest["status"]=="closed"


def test_n2_debug_and_global_sink_are_mutually_exclusive(tmp_path):
    proc = subprocess.run([sys.executable,"-m","exp.r17_integration.models.fastwam.serve",
        "--descriptor","unused","--debug-directory",str(tmp_path),"--capture-sink","unused:unused"],
        capture_output=True,text=True,timeout=15)
    assert proc.returncode==2 and "not allowed with argument" in proc.stderr


def test_b6_actual_published_raw_released_loader_header_equivalence(tmp_path):
    root = Path("/data/mw_finetune/models/fastwam_libero")
    python = Path("/data/mw_finetune/envs/fastwam/bin/python")
    if not root.is_dir() or not python.is_file():
        pytest.skip("published config/header or Fast-WAM CPU island unavailable")
    receipt = tmp_path/"loader_headers.json"
    proc = subprocess.run([str(python),"-m","exp.r17_integration.models.fastwam.loader_parity",
        "--header-only","--checkpoint",str(root),"--wan-root","/data/xwam/wan22_5b",
        "--receipt",str(receipt)],env={**os.environ,"CUDA_VISIBLE_DEVICES":"","HF_HUB_OFFLINE":"1",
            "TRANSFORMERS_OFFLINE":"1","PYTHONDONTWRITEBYTECODE":"1"},capture_output=True,text=True,timeout=100)
    assert proc.returncode==0,proc.stderr+receipt.read_text()
    result = json.loads(receipt.read_text())
    assert result["matched_unique_parameters"]==1651 and result["tensor_body_reads"]==0
    assert result["published_geometry"]=={"image_size":[224,448],"action_dim":7,"proprio_dim":8,"action_horizon":32}
    assert result["sampler"]["num_steps"]==10
    assert result["device"]=="meta" and len(result["frozen_loader_calls"])==6
    assert result["matched_parameter_dtypes"] and len(result["matched_inference_config_fields"])==24
    assert all("install_raw_loaders.<locals>" in v for v in result["selected_raw_loaders"].values())
    assert result["raw_loader_scope_restored"]
    assert result["released_serialization"]["canonical_state_dict_shapes_match"]
    assert result["released_serialization"]["renamed_checkpoint_keys"]==1620
    assert not result["released_serialization"]["original_key_sets_equal"]
    assert "NOT established" in result["limits"][0]


def test_b6_parity_module_import_is_default_off():
    proc = subprocess.run([sys.executable,"-c",
        "import sys; import exp.r17_integration.models.fastwam.loader_parity; "
        "assert not any(n in sys.modules for n in ('torch','jax','lerobot','fastwam'))"],
        capture_output=True,text=True,timeout=15)
    assert proc.returncode==0,proc.stderr


def test_b6_released_serialization_shim_preserves_values_and_is_strict(tmp_path):
    from safetensors.torch import save_file
    from exp.r17_integration.models.fastwam.loader_parity import _load_released_canonical
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=torch.nn.Module();self.model.mot=torch.nn.Module()
            self.model.mot.layers=torch.nn.ModuleList([torch.nn.Module()])
            self.model.mot.layers[0].blocks=torch.nn.ModuleDict({"action":torch.nn.Linear(3,2,bias=False)})
    policy=Policy()
    values=torch.arange(6,dtype=torch.float32).reshape(2,3)
    old_key="model.mot.mixtures.action.blocks.0.weight"
    file=tmp_path/"model.safetensors"
    save_file({old_key:values},str(file))
    _load_released_canonical(None,policy,str(file),"cpu",True)
    torch.testing.assert_close(policy.model.mot.layers[0].blocks["action"].weight,values,rtol=0,atol=0)
    for bad in ({"extra":values},{old_key:torch.zeros(3,2)},
                {old_key:values,"model.mot.layers.0.blocks.action.weight":values.clone()}):
        save_file(bad,str(file))
        with pytest.raises(ValueError):
            _load_released_canonical(None,policy,str(file),"cpu",True)


def test_b6_gpu_comparison_detects_sidecar_drift():
    from exp.r17_integration.models.fastwam.loader_parity import compare_outputs
    raw = {"latents":np.zeros((1,48,1,7,14)),"actions":np.zeros((32,7))}
    assert compare_outputs(raw,raw,rtol=0,atol=0)["status"]=="passed"
    drift = {**raw,"latents":raw["latents"]+.1}
    report = compare_outputs(raw,drift,rtol=1e-3,atol=1e-3)
    assert report["status"]=="failed" and not report["rows"][1]["passed"]
    with pytest.raises(ValueError,match="geometry"):
        compare_outputs(raw,{**raw,"actions":np.zeros((32,4))},rtol=1e-3,atol=1e-3)
    for bad in (float("inf"),float("nan"),-1.):
        with pytest.raises(ValueError,match="tolerances"):
            compare_outputs(raw,drift,rtol=bad,atol=1e-3)
    assert compare_outputs({"0:input_ids":np.array([32000])},{"0:input_ids":np.array([32001])},
        rtol=1e-3,atol=1e-3)["status"]=="failed"
