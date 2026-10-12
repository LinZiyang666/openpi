"""CPU-only timing and evidence tests. No real weights or CUDA API calls."""
from types import SimpleNamespace
import json
import sys
import types

import numpy as np
import pytest
import torch

from exp.r17_integration.profile import latency as L
from exp.r17_integration.profile import certify as C


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    assert not torch.cuda.is_initialized()
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU profile tests must never initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    assert not torch.cuda.is_initialized()


def test_prices_use_measured_reduced_loop_and_full_denominator():
    costs = L.pricing(2., 3., 15., 4.)
    assert costs == {"vision": .1, "full_miss": .9, "reduced_miss": .35, "denominator_ms": 20.}
    assert costs["vision"] + costs["full_miss"] == 1
    assert L.pricing(2., 3., 15., 15.)["reduced_miss"] == .9
    for bad in ((0, 0, 0, 1), (2, -1, 3, 4), (2, 3, float("nan"), 4)):
        with pytest.raises(ValueError):
            L.pricing(*bad)


def test_quantiles_retain_precision():
    assert L.summary([1., 2., 3., 4., 5.]) == {"median": 3., "p10": 1.4, "p90": 4.6}
    for values in ([], [float("nan")], [-1.]):
        with pytest.raises(ValueError):
            L.summary(values)


@pytest.mark.parametrize("model,k,nfe", [("dp_libero", 100, 100), ("dp_libero", 10, 10),
    ("fastwam_libero", 10, 10), ("fastwam_libero", 1, 1), ("cosmos_libero", 5, 5), ("cosmos_libero", 2, 1)])
def test_actual_network_counts(model, k, nfe):
    assert L.actual_nfe(model, k) == nfe


@pytest.mark.parametrize("observed,captures,skips,new_compiles,compile_count", [
    (None, 0, 0, 0, 3), (599, 0, 0, 0, 3), (600, 1, 0, 0, 3),
    (600, 0, 1, 0, 3), (600, 0, 0, 1, 3), (600, 0, 0, 0, 4)])
def test_incomplete_evidence_never_certifies(observed, captures, skips, new_compiles, compile_count):
    assert not L.certificate(600, observed, captures, skips, new_compiles, compile_count)["certified"]
    assert L.certificate(600, 600, 0, 0, 0, 3)["certified"]


def trace(launches=600, captures=0, marker="my_run"):
    return ('Generating CUDA API Summary...\nTime (%),Total Time (ns),Num Calls,Name\n'
            f'99,2000,"{launches:,}",cudaGraphLaunch_v3020\n'
            f'1,20,{captures},cudaStreamBeginCapture_v10000\n\n'
            'Time (%),Total Time (ns),Instances,Range\n'
            f'100,2020,1,:{marker}\n')


def test_trace_counts_ignore_nvtx_instances_and_require_matching_run():
    assert C.parse_trace(trace(22800), "my_run") == {"cudagraph_launch_count": 22800, "capture_calls_after_warmup": 0}
    assert C.parse_trace(trace(captures=2), "my_run")["capture_calls_after_warmup"] == 2
    with pytest.raises(ValueError):
        C.parse_trace(trace(), "another_run")
    with pytest.raises(ValueError):
        C.parse_trace("my_run\n", "my_run")
    with pytest.raises(ValueError):
        C.parse_trace("", "my_run")


def test_nsys_can_disprove_torch_evidence(tmp_path):
    receipt = dict(trace_marker="my_run", cudagraph_launch_count=600, capture_calls_after_warmup=0,
        expected_cudagraph_launch_count=600, cudagraph_skips=0, compiles_after_warmup=0, compile_count=3, valid=True)
    path = tmp_path / "trace.csv"
    path.write_text(trace())
    assert C.certify_trace(receipt, path.read_text(), path)["certified"]
    path.write_text(trace(599))
    result = C.certify_trace(receipt, path.read_text(), path)
    assert not result["valid"] and not result["certified"]
    assert result["torch_cudagraph_launch_count"] == 600


def test_graph_counters_observe_real_method_calls_and_restore_on_error():
    class Graph:
        def replay(self):
            return "replayed"
        def capture_begin(self):
            return "captured"
    original = Graph.replay
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph))
    with pytest.raises(RuntimeError), L.GraphCounts(fake_torch) as counts:
        g = Graph()
        assert g.capture_begin() == "captured"
        assert g.replay() == "replayed"
        g.replay()
        assert counts.snapshot() == (2, 1)
        raise RuntimeError("exit")
    assert Graph.replay is original


def test_events_follow_adapter_stages_and_count_native_updates():
    clock = SimpleNamespace(value=0., marks=0)
    class Event:
        def __init__(self, **kwargs):
            self.value = None
        def record(self):
            self.value = clock.value
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return other.value - self.value
    def mark():
        clock.marks += 1
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event), compiler=SimpleNamespace(cudagraph_mark_step_begin=mark))
    def network(x):
        clock.value += 11
        return x + 1
    probe = L.StepProbe(fake_torch, network)
    def prepare(obs, request):
        clock.value += 2
        return "prepared"
    def stage1(p):
        assert p == "prepared"
        clock.value += 3
        return "key"
    def stage2(s):
        assert s == "key"
        clock.value += 7
        return "condition"
    def sample(s, sampler):
        assert s == "condition"
        probe.reset()
        x = torch.zeros(1)  # Stub module data lives only on CPU.
        for _ in range(sampler.num_steps):
            x = probe(x)
            clock.value += 13  # Uncompiled native scheduler/update.
        probe.finish()
        return SimpleNamespace(actual_nfe=sampler.num_steps)
    runtime = SimpleNamespace(stage1=stage1, stage2=stage2, sample=sample, probe=probe)
    values, nfe = L.measure_iterations(fake_torch, runtime, SimpleNamespace(prepare=prepare), {}, None,
                                     SimpleNamespace(num_steps=2), 3)
    assert values["stage1_ms"] == [5.] * 3  # Includes prepare.
    assert values["stage2_ms"] == [7.] * 3
    assert values["stage3_ms"] == [48.] * 3
    assert values["total_ms"] == [60.] * 3
    assert values["stage3_step_ms"] == [24.] * 6
    assert values["stage3_network_ms"] == [11.] * 6
    assert nfe == [2, 2, 2] and clock.marks == 3
    assert not probe.enabled


def test_fx_specializes_structure_but_keeps_tensor_values(monkeypatch):
    options = {}
    def compile_stub(graph, **kwargs):
        options.update(kwargs)
        return graph
    monkeypatch.setattr(torch, "compile", compile_stub)
    def native(x, sentinel):
        # The fixed protocol sentinel is a serving-geometry branch.
        if bool(torch.all(sentinel == 1)):
            return x.sin() * 2
        return x.cos()
    with torch.inference_mode():
        compiled = L.compile_tensor_program(native, (torch.tensor([1., 2.]), torch.ones(1)))
        torch.testing.assert_close(compiled(torch.tensor([3., 4.]), torch.ones(1)), torch.tensor([3., 4.]).sin()*2)
    assert options == {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False}


def test_owned_outputs_and_numeric_parity():
    value = {"a": torch.tensor([1., 2.]), "b": [torch.tensor([3.])]}
    owned = L.tensor_copy(value)
    value["a"].zero_()
    assert owned["a"].tolist() == [1., 2.]
    assert L.parity(owned, owned)["passed"]
    assert not L.parity(owned, value)["passed"]
    assert not L.parity(torch.ones(2), torch.ones(3))["passed"]


def test_real_pool_reader_and_cosmos_quaternion_fields(tmp_path, monkeypatch):
    specs = tmp_path / "exp/r17_integration/qualification/specs"
    specs.mkdir(parents=True)
    (specs / "libero_benchmark_spec.json").write_text(json.dumps({"tasks": [["libero_10/task", "real instruction", "libero_10"]]}))
    monkeypatch.setattr(L, "ROOT", tmp_path)
    pool = tmp_path / "bpool"
    pool.mkdir()
    arrays = {"image_0": np.full((1, 256, 256, 3), 19, np.uint8),
              "image_1": np.full((1, 256, 256, 3), 47, np.uint8),
              "state": np.arange(9, dtype=np.float32)[None]}
    for name, array in arrays.items():
        np.save(pool / (name + ".npy"), array)
    (pool / "READY.json").write_text(json.dumps({"rows": [{"task_name": "libero_10/task", "row_id": "row1", "campaign": "teacher"}],
        "arrays": {k: {"file": k+".npy"} for k in arrays}, "binding": {"benchmark_protocol": "native"}}))
    obs, source = L.read_input("cosmos_libero", "libero_10", pool)
    assert obs["prompt"] == "real instruction"
    np.testing.assert_array_equal(obs["observation/cosmos_proprio"], arrays["state"][0])
    assert obs["observation/image"][0, 0, 0] == 19
    assert obs["observation/wrist_image"][0, 0, 0] == 47
    assert source["benchmark_protocol"] == "native" and source["row"] == 0


def test_wrapper_commands_only_include_measured_capture_range(tmp_path):
    out, invocation, stats = C.commands("cosmos_libero", "libero_10", tmp_path, "/island/python", "/bin/nsys")
    assert "--capture-range=cudaProfilerApi" in invocation
    assert "--capture-range-end=stop" in invocation
    assert invocation[-1] == str(out)
    assert "cuda_api_sum,nvtx_pushpop_sum" in stats
    _, plain, stats = C.commands("dp_libero", "libero_spatial", tmp_path, "/island/python")
    assert plain[0] == "/island/python" and stats is None


def test_dp_tensor_cuts_with_cpu_stub_modules(monkeypatch):
    from exp.r17_integration.common.contracts import Stage1, Stage2, SampleResult, SamplerSpec
    module = types.ModuleType("exp.mw_finetune.dp.model")
    module.crop = lambda x, training: x[..., 6:90, 6:90]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    compiled_stages = []
    def compile_stub(fn, inputs):
        compiled_stages.append(fn)
        return fn
    monkeypatch.setattr(L, "compile_tensor_program", compile_stub)
    class Vision(torch.nn.Module):
        def forward(self, pixels, language):
            return pixels.mean(dim=(1, 2, 3))[:, None].expand(-1, 64) + language[:, :1]
    class Denoiser(torch.nn.Module):
        def forward(self, x, t, cond):
            return x + cond.mean() + t.float().sum()
    original = Denoiser()
    core = SimpleNamespace(model=original, normalizer=SimpleNamespace(normalize=lambda k, v: v),
        obs_encoder=SimpleNamespace(obs_nets={"base_0_rgb": Vision(), "left_wrist_0_rgb": Vision()},
        feature_order=("base_0_rgb", "left_wrist_0_rgb", "state", "lang_emb")))
    inputs = {"base_0_rgb": torch.ones(1, 3, 96, 96), "left_wrist_0_rgb": torch.ones(1, 3, 96, 96)*2,
              "state": torch.zeros(1, 8), "lang_emb": torch.ones(1, 3)}
    prepared = SimpleNamespace(data={"inputs": inputs}, state=np.arange(8, dtype=np.float32))
    def eager1(p):
        features = {k: torch.relu(net(module.crop(p.data["inputs"][k], False), inputs["lang_emb"]))
                    for k, net in core.obs_encoder.obs_nets.items()}
        features.update(state=inputs["state"], lang_emb=inputs["lang_emb"])
        return Stage1({"features": features}, {}, p.state)
    def eager2(s):
        return Stage2({"conditioning": torch.cat([s.data["features"][k] for k in core.obs_encoder.feature_order], -1).unsqueeze(1)}, s)
    spec = SamplerSpec("stub", "stub", 1, (1,))
    def sample(s, sampler, rng, capture):
        chunk = core.model(torch.zeros(1, 16, 7), torch.tensor(1), s.data["conditioning"])[0]
        return SampleResult(chunk, 1, sampler)
    adapter = SimpleNamespace(model=SimpleNamespace(_policy=core), device=torch.device("cpu"),
        prepare=lambda obs, request: prepared, run_stage1=eager1, run_stage2=eager2, sample=sample)
    with torch.inference_mode():
        runtime = L.build_stages("dp_libero", adapter, {}, None)
        try:
            s1 = runtime.stage1(prepared)
            assert s1.fields["vision_0"].tolist() == [2.] * 64
            assert s1.fields["vision_1"].tolist() == [3.] * 64
            s2 = runtime.stage2(s1)
            torch.testing.assert_close(s2.data["conditioning"], eager2(eager1(prepared)).data["conditioning"])
            result = runtime.sample(s2, spec, SimpleNamespace(noise_identity=("stub",), explicit_seed=0))
            assert result.actual_nfe == 1 and result.normalized_chunk.shape == (16, 7)
        finally:
            runtime.restore()
    assert len(compiled_stages) == 3 and core.model is original


def test_fastwam_tensor_cuts_with_cpu_stub_modules(monkeypatch):
    from exp.r17_integration.common.contracts import Stage1, Stage2, SampleResult, SamplerSpec
    calls = []
    def compile_stub(fn, inputs):
        calls.append(fn)
        return fn
    monkeypatch.setattr(L, "compile_tensor_program", compile_stub)
    class Video:
        patch_embedding = SimpleNamespace(weight=torch.ones(1))
        fuse_vae_embedding_in_latents = False
        def patchify(self, first):
            return first[:, :, :, ::2, ::2]
        def pre_dit(self, **kw):
            patches = self.patchify(kw["x"])
            return {"tokens": patches.flatten(2).transpose(1, 2), "freqs": torch.ones(1),
                    "t_mod": torch.zeros(1), "context": kw["context"], "context_mask": kw["context_mask"],
                    "meta": {"tokens_per_frame": 8}}
    def tokenizer(prompt, **kw):
        return torch.tensor([[3, 7, 0, 0]]), torch.tensor([[True, True, False, False]])
    def text(ids, mask):
        return ids.float()[:, :, None].expand(-1, -1, 3).clone()
    def encode(prompt):
        ids, mask = tokenizer(prompt)
        context = text(ids, mask)
        context[:, 2:] = 0
        return context, torch.ones_like(mask)
    def append(context, mask, proprio):
        return torch.cat((context, proprio.mean(1).reshape(1, 1, 1).expand(1, 1, 3)), 1), torch.cat((mask, torch.ones(1, 1, dtype=torch.bool)), 1)
    def prefill(**kw):
        return [{"k": kw["video_tokens"].clone(), "v": kw["video_tokens"].clone()*2}]
    def network(**kw):
        return kw["latents_action"] + kw["context"].mean() + kw["video_kv_cache"][0]["v"].mean()
    m = SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32, video_expert=Video(),
        vae=SimpleNamespace(encode=lambda *args, **kw: torch.arange(32.).reshape(1, 1, 1, 4, 8)),
        tokenizer=tokenizer, text_encoder=text, encode_prompt=encode, _append_proprio_to_context=append,
        _build_mot_attention_mask=lambda **kw: torch.ones(40, 40, dtype=torch.bool),
        mot=SimpleNamespace(prefill_video_cache=prefill, mixtures={}), _predict_action_noise_with_cache=network)
    prepared = SimpleNamespace(data={"input_image": torch.ones(1, 3, 224, 448), "prompt": "real prompt",
                                    "proprio": torch.ones(1, 8)}, state=np.zeros(8, np.float32))
    first = m.vae.encode(None)
    patches = m.video_expert.patchify(first)
    data1 = dict(first_latents=first, patches=patches, proprio=prepared.data["proprio"], video_seq_len=8,
        prompt="real prompt", conditioning_source=("real prompt", prepared.data["input_image"], prepared.data["proprio"]),
        model_identity="stub")
    eager1 = Stage1(data1, {}, prepared.state)
    context, mask = append(*encode(["real prompt"]), prepared.data["proprio"])
    tokens = patches.flatten(2).transpose(1, 2)
    eager2 = Stage2(dict(context=context, context_mask=mask, attention_mask=torch.ones(40, 40, dtype=torch.bool),
                        video_kv_cache=[{"k": tokens, "v": tokens*2}]), eager1)
    def sample(s, sampler, rng, capture):
        d = s.data
        x = m._predict_action_noise_with_cache(latents_action=torch.zeros(1, 32, 7), timestep_action=torch.ones(1),
            context=d["context"], context_mask=d["context_mask"], video_kv_cache=d["video_kv_cache"],
            attention_mask=d["attention_mask"], video_seq_len=8)
        return SampleResult(x[0], 1, sampler)
    adapter = SimpleNamespace(model=m, prepare=lambda obs, req: prepared, run_stage1=lambda p: eager1,
        run_stage2=lambda s: eager2, sample=sample, metadata=lambda: SimpleNamespace(layout=SimpleNamespace(horizon=32)))
    request = SimpleNamespace(noise_identity=("stub",), explicit_seed=0)
    with torch.inference_mode():
        runtime = L.build_stages("fastwam_libero", adapter, {}, request)
        try:
            s1 = runtime.stage1(prepared)
            assert s1.fields["vision_0"].shape == (16,)
            assert s1.data["video_seq_len"] == 8
            s2 = runtime.stage2(s1)
            assert L.parity(L.stage_payload("fastwam_libero", s2, 2), L.stage_payload("fastwam_libero", eager2, 2))["passed"]
            result = runtime.sample(s2, SamplerSpec("stub", "stub", 1, (1,)))
            assert result.actual_nfe == 1 and result.normalized_chunk.shape == (32, 7)
        finally:
            runtime.restore()
    assert len(calls) == 3 and m._predict_action_noise_with_cache is network


def test_cosmos_tensor_cuts_with_cpu_stub_condition_and_vendor(monkeypatch):
    from exp.r17_integration.common.contracts import Stage1, Stage2, SampleResult, SamplerSpec
    from exp.r17_integration.models.cosmos import staged
    calls = []
    def compile_stub(fn, inputs):
        calls.append(fn)
        return fn
    monkeypatch.setattr(L, "compile_tensor_program", compile_stub)
    utils = types.ModuleType("cosmos_policy.utils.utils")
    utils.set_seed_everywhere = lambda seed: None
    monkeypatch.setitem(sys.modules, "cosmos_policy", types.ModuleType("cosmos_policy"))
    monkeypatch.setitem(sys.modules, "cosmos_policy.utils", types.ModuleType("cosmos_policy.utils"))
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    latent = torch.arange(9.).reshape(1, 1, 9, 1, 1).expand(1, 16, 9, 28, 28).clone()
    batch = {"video": latent.clone(), "proprio": torch.ones(1, 9)}
    monkeypatch.setattr(staged, "make_data_batch", lambda *args: dict(batch))
    class Model:
        config = SimpleNamespace(use_flowunipc_scheduler=False)
        tensor_kwargs = {"device": "cpu"}
        def _normalize_video_databatch_inplace(self, batch):
            batch["video"] = batch["video"].float()
        def _augment_image_dim_inplace(self, batch):
            pass
        def encode(self, video):
            return video.clone()
        def get_x0_fn_from_batch(self, batch, *args, previous_generated_latent=None, **kw):
            condition = SimpleNamespace(gt_frames=previous_generated_latent.clone(), proprio=batch["proprio"])
            def x0_fn(x, sigma):
                return self.denoise(x, sigma, condition).x0
            return x0_fn
        def denoise(self, x, sigma, condition):
            return SimpleNamespace(x0=x*.2 + condition.gt_frames + condition.proprio.mean())
    m = Model()
    prepared = SimpleNamespace(data={"native": {}, "prompt": "real prompt"}, state=np.zeros(9, np.float32))
    eager1 = Stage1(dict(batch=batch, latent=latent, model_identity="stub"), {}, prepared.state)
    eager2 = Stage2(dict(x0_fn=m.get_x0_fn_from_batch(batch, previous_generated_latent=latent), latent_shape=latent.shape), eager1)
    def sample(s, sampler, rng, capture):
        generated = s.data["x0_fn"](torch.zeros_like(latent), torch.ones(1))
        actions = generated[:, :7, 4].mean((2, 3))[0].expand(16, 7).clone()
        return SampleResult(actions, 1, sampler)
    adapter = SimpleNamespace(model=m, cfg=None, stats=None, embeddings={"real prompt": None}, cu=None,
        prepare=lambda obs, request: prepared, run_stage1=lambda p: eager1, run_stage2=lambda s: eager2,
        sample=sample, metadata=lambda: SimpleNamespace(identity="stub"))
    with torch.inference_mode():
        runtime = L.build_stages("cosmos_libero", adapter, {}, SimpleNamespace(noise_identity=("stub",), explicit_seed=195))
        s1 = runtime.stage1(prepared)
        assert s1.fields["vision_0"].tolist() == [3.] * 256
        assert s1.fields["vision_1"].tolist() == [2.] * 256
        s2 = runtime.stage2(s1)
        assert L.parity(L.stage_payload("cosmos_libero", s2, 2), L.stage_payload("cosmos_libero", eager2, 2))["passed"]
        result = runtime.sample(s2, SamplerSpec("stub", "stub", 2, (1, 0)))
        assert result.actual_nfe == 1 and result.normalized_chunk.shape == (16, 7)
    assert len(calls) == 3
