"""CPU-only timing and evidence tests. No real weights or CUDA API calls."""
from contextlib import nullcontext
from dataclasses import dataclass, replace
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
    ("fastwam_libero", 10, 10), ("fastwam_libero", 1, 1), ("cosmos_libero", 5, 5), ("cosmos_libero", 2, 1),
    ("dp_mw", 100, 100), ("dp_mw", 10, 10), ("fastwam_mw", 10, 10), ("fastwam_mw", 3, 3),
    ("pi05_mw", 10, 10), ("pi05_mw", 2, 2), ("groot_mw", 4, 4), ("groot_mw", 2, 2)])
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


@pytest.mark.parametrize("k", [2, 3, 4, 10, 100])
def test_events_follow_adapter_stages_and_count_native_updates(k):
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
                                     SimpleNamespace(num_steps=k), 3)
    assert values["stage1_ms"] == [5.] * 3  # Includes prepare.
    assert values["stage2_ms"] == [7.] * 3
    assert values["stage3_ms"] == [24. * k] * 3
    assert values["total_ms"] == [12. + 24. * k] * 3
    assert values["stage3_step_ms"] == [24.] * (3 * k)
    assert values["stage3_network_ms"] == [11.] * (3 * k)
    assert nfe == [k] * 3 and clock.marks == 3
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


@pytest.mark.parametrize("model,state_dim,action_dim", [("dp_libero", 8, 7), ("dp_mw", 4, 4)])
def test_dp_tensor_cuts_with_cpu_stub_modules(monkeypatch, model, state_dim, action_dim):
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
        chunk = core.model(torch.zeros(1, 16, action_dim), torch.tensor(1), s.data["conditioning"])[0]
        return SampleResult(chunk, 1, sampler)
    inputs["state"] = torch.zeros(1, state_dim)
    prepared.state = np.arange(state_dim, dtype=np.float32)
    adapter = SimpleNamespace(model=SimpleNamespace(_policy=core, cfg={"action_dim": action_dim}), device=torch.device("cpu"),
        prepare=lambda obs, request: prepared, run_stage1=eager1, run_stage2=eager2, sample=sample)
    with torch.inference_mode():
        runtime = L.build_stages(model, adapter, {}, None)
        try:
            s1 = runtime.stage1(prepared)
            assert s1.fields["vision_0"].tolist() == [2.] * 64
            if model == "dp_libero":
                assert s1.fields["vision_1"].tolist() == [3.] * 64
            else:
                assert set(s1.fields) == {"vision_0", "robot_state"}
            s2 = runtime.stage2(s1)
            torch.testing.assert_close(s2.data["conditioning"], eager2(eager1(prepared)).data["conditioning"])
            result = runtime.sample(s2, spec, SimpleNamespace(noise_identity=("stub",), explicit_seed=0))
            assert result.actual_nfe == 1 and result.normalized_chunk.shape == (16, action_dim)
        finally:
            runtime.restore()
    assert len(compiled_stages) == 3 and core.model is original


@pytest.mark.parametrize("model,state_dim,action_dim,image_width", [
    ("fastwam_libero", 8, 7, 448), ("fastwam_mw", 4, 4, 224)])
def test_fastwam_tensor_cuts_with_cpu_stub_modules(monkeypatch, model, state_dim, action_dim, image_width):
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
    prepared = SimpleNamespace(data={"input_image": torch.ones(1, 3, 224, image_width), "prompt": "real prompt",
                                    "proprio": torch.ones(1, state_dim)}, state=np.zeros(state_dim, np.float32))
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
        x = m._predict_action_noise_with_cache(latents_action=torch.zeros(1, 32, action_dim), timestep_action=torch.ones(1),
            context=d["context"], context_mask=d["context_mask"], video_kv_cache=d["video_kv_cache"],
            attention_mask=d["attention_mask"], video_seq_len=8)
        return SampleResult(x[0], 1, sampler)
    adapter = SimpleNamespace(model=m, prepare=lambda obs, req: prepared, run_stage1=lambda p: eager1,
        run_stage2=lambda s: eager2, sample=sample,
        metadata=lambda: SimpleNamespace(layout=SimpleNamespace(horizon=32, action_width=action_dim)))
    request = SimpleNamespace(noise_identity=("stub",), explicit_seed=0)
    with torch.inference_mode():
        runtime = L.build_stages(model, adapter, {}, request)
        try:
            s1 = runtime.stage1(prepared)
            assert s1.fields["vision_0"].shape == (16,)
            assert s1.data["video_seq_len"] == 8
            s2 = runtime.stage2(s1)
            assert L.parity(L.stage_payload(model, s2, 2), L.stage_payload(model, eager2, 2))["passed"]
            result = runtime.sample(s2, SamplerSpec("stub", "stub", 1, (1,)))
            assert result.actual_nfe == 1 and result.normalized_chunk.shape == (32, action_dim)
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
    assert len(calls) == 2  # stage3 is a captured native denoiser, not a traced program


@pytest.mark.parametrize("model", L.MT50_MODELS)
def test_mt50_pool_preserves_one_rotated_camera_state_and_canonical_prompt(tmp_path, model):
    from exp.metaworld.tasks import PROMPTS
    pool = tmp_path / "bpool"
    pool.mkdir()
    # Distinct corners catch accidental rotation or resize in the reader.
    image = np.zeros((1, 480, 480, 3), np.uint8)
    image[0, 0, 0] = (1, 2, 3)
    image[0, -1, -1] = (4, 5, 6)
    state = np.array([[.1, .2, .3, .4]], np.float32)
    np.save(pool / "image_0.npy", image)
    np.save(pool / "state.npy", state)
    (pool / "READY.json").write_text(json.dumps({
        "rows": [{"task_name": "assembly-v3", "row_id": "row0", "campaign": "teacher"}],
        "arrays": {k: {"file": k + ".npy"} for k in ("image_0", "state")},
        "binding": {"benchmark_protocol": "mt50-native"}}))
    obs, source = L.read_input(model, "mt50", pool)
    assert set(obs) == {"observation/image", "observation/state", "prompt"}
    np.testing.assert_array_equal(obs["observation/image"], image[0])
    np.testing.assert_array_equal(obs["observation/state"], state[0])
    assert obs["prompt"] == PROMPTS["assembly-v3"]
    assert source["geometry"] == {"observation/image": [480, 480, 3], "observation/state": [4]}
    assert source["descriptor"] == str(tmp_path / "campaigns/teacher/assets/load.json")


@pytest.mark.parametrize("model", L.MT50_MODELS)
def test_mt50_wire_npz_input(tmp_path, model):
    path = tmp_path / "wire.npz"
    image = np.full((1, 480, 480, 3), 27, np.uint8)
    state = np.ones((1, 4), np.float32)
    np.savez(path, **{"observation/image": image, "observation/state": state, "prompt": ["Press a button"]})
    obs, source = L.read_input(model, "mt50", None, observations=path)
    assert obs["prompt"] == "Press a button"
    np.testing.assert_array_equal(obs["observation/image"], image[0])
    assert source["row"] == 0 and source["source"] == str(path)


@pytest.mark.parametrize("model", L.MT50_MODELS)
def test_mt50_launcher_uses_one_profile_and_model_island(tmp_path, monkeypatch, model):
    calls = []
    monkeypatch.setattr(C.shutil, "which", lambda name: None)
    def run(command, **kwargs):
        calls.append((command, kwargs))
        out = command[command.index("--out") + 1]
        from pathlib import Path
        Path(out).write_text(json.dumps({"valid": True, "coefficients": {"vision": .1}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(C.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["certify", "--model", model, "--out-dir", str(tmp_path)])
    assert C.main() == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == C.PYTHONS[model]
    assert command[command.index("--suite") + 1] == "mt50"
    assert command[command.index("--out") + 1] == str(tmp_path / (model + "_mt50.json"))
    if model == "fastwam_mw":
        assert "/data/mw_finetune/vendor/FastWAM/src" in kwargs["env"]["PYTHONPATH"]
    if model == "groot_mw":
        assert "/home/weiland/projects/openpi_ext/third_party/gr00t_n15" in kwargs["env"]["PYTHONPATH"]


@pytest.mark.parametrize("model", L.MT50_MODELS)
def test_mt50_loader_dispatches_to_model_owned_loader(tmp_path, monkeypatch, model):
    calls = []
    adapter = object()
    desc = {"checkpoint": "/immutable/checkpoint", "device": "old", "live_k": 4,
            "tokenizer_files": ["tokenizer"], "code_files": ["native.py"]}
    family = model.removesuffix("_mw")
    name = "exp.r17_integration.models." + family + (".adapter" if family == "pi05" else ".loader")
    module = types.ModuleType(name)
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return adapter
    if family == "fastwam":
        module.LoadSpec = lambda **kw: SimpleNamespace(**kw)
    if family == "groot":
        desc.pop("device")
        backend, manifest = object(), object()
        def load_checkpoint(*args, **kwargs):
            calls.append((args, kwargs))
            return backend, manifest
        module.load_checkpoint = load_checkpoint
        provider = types.ModuleType("exp.r17_integration.models.groot.provider")
        provider.GrootMetaWorldAdapter = lambda b, m: adapter if (b, m) == (backend, manifest) else None
        monkeypatch.setitem(sys.modules, provider.__name__, provider)
    else:
        module.load_adapter = load
    monkeypatch.setitem(sys.modules, name, module)
    path = tmp_path / "load.json"
    path.write_text(json.dumps(desc))
    loaded, effective = L.load_adapter(model, path, "cpu")
    assert loaded is adapter and len(calls) == 1
    args, kwargs = calls[0]
    if family in ("dp", "pi05"):
        assert effective["device"] == "cpu" and args[0] == effective
    else:
        assert kwargs["device"] == "cpu"
    if family == "groot":
        assert kwargs["denoising_steps"] == 4
        assert kwargs["tokenizer_files"] == desc["tokenizer_files"]
        assert kwargs["code_files"] == desc["code_files"]


def test_pi05_tensor_cuts_keep_native_keys_cache_and_complete_sample(monkeypatch):
    from exp.r17_integration.common.contracts import PreparedObs, Stage1, Stage2, SampleResult
    from exp.r17_integration.models.pi05.adapter import Pi05MetaWorldAdapter, sampler_for_steps
    @dataclass
    class Observation:
        images: dict
        state: torch.Tensor
    @dataclass
    class Prefix:
        state: torch.Tensor
        prefix_embs: torch.Tensor
        prefix_pad_masks: torch.Tensor
        prefix_att_2d_masks_4d: torch.Tensor
        prefix_position_ids: torch.Tensor
    @dataclass
    class Condition:
        stage1: Prefix
        past_key_values: object
    class Cache(list):
        @classmethod
        def from_legacy_cache(cls, values):
            return cls(values)
    class Model:
        def run_stage1(self, obs):
            x = obs.images["base_0_rgb"].mean()
            return Prefix(obs.state, torch.ones(1, 800, 2048) * x, torch.ones(1, 800, dtype=torch.bool),
                          torch.zeros(1, 1, 800, 800), torch.arange(800)[None])
        def run_stage2(self, s):
            return Condition(s, Cache([(s.prefix_embs[:, :2, :2], s.prefix_embs[:, :2, :2] * 2)]))
        def denoise_step(self, state, mask, kv, x, t):
            return x * .1 + kv[0][0].mean() + t[:, None, None]
    m = Model()
    prepared = PreparedObs(Observation({"base_0_rgb": torch.ones(1, 3, 2, 2)}, torch.zeros(1, 32)), None, np.zeros(32))
    sample_calls = []
    def sample(s, sampler, rng, capture):
        sample_calls.append(sampler.num_steps)
        x = torch.from_numpy(rng.normal((5, 32)))[None]
        for i in range(sampler.num_steps):
            x = x - m.denoise_step(s.data.stage1.state, s.data.stage1.prefix_pad_masks,
                                  s.data.past_key_values, x, torch.tensor([1 - i / sampler.num_steps])) / sampler.num_steps
        return SampleResult(x[0].numpy().copy(), sampler.num_steps, sampler)
    adapter = SimpleNamespace(model=m, prepare=lambda *args: prepared,
        run_stage1=lambda p: Stage1(m.run_stage1(p.data)), run_stage2=lambda s: Stage2(m.run_stage2(s.data), s), sample=sample)
    adapter.retrieval_fields = lambda s: Pi05MetaWorldAdapter.retrieval_fields(adapter, s)
    calls = []
    original_compile = L.compile_tensor_program
    def compile_stub(fn, inputs):
        calls.append(fn)
        # Real FX extraction checks the tensor-only interfaces, on CPU.
        return original_compile(fn, inputs)
    monkeypatch.setattr(L, "compile_tensor_program", compile_stub)
    monkeypatch.setattr(torch, "compile", lambda graph, **kw: graph)
    original = m.denoise_step
    request = SimpleNamespace(noise_identity=("stub",), explicit_seed=0)
    with torch.inference_mode():
        runtime = L.build_stages("pi05_mw", adapter, {}, request)
        try:
            prepared.data.images["base_0_rgb"].fill_(3)
            s1 = runtime.stage1(prepared)
            expected1 = adapter.run_stage1(prepared)
            assert L.parity(s1.fields, adapter.retrieval_fields(expected1))["passed"]
            assert s1.fields["vision_0"].shape == (16 * 2048,) and s1.fields["vision_0"].device.type == "cpu"
            s2 = runtime.stage2(s1)
            assert isinstance(s2.data.past_key_values, Cache)
            assert L.parity(L.stage_payload("pi05_mw", s2, 2), L.stage_payload("pi05_mw", adapter.run_stage2(expected1), 2))["passed"]
            from exp.r17_integration.common.request_rng import RequestRNG
            for k in (10, 2):
                compiled_step = m.denoise_step
                m.denoise_step = original
                expected = adapter.sample(s2, sampler_for_steps(k), RequestRNG(request), None)
                m.denoise_step = compiled_step
                out = runtime.sample(s2, sampler_for_steps(k))
                assert out.actual_nfe == k and out.normalized_chunk.shape == (5, 32)
                assert L.parity(out.normalized_chunk, expected.normalized_chunk)["passed"]
        finally:
            runtime.restore()
    assert len(calls) == 3 and sample_calls == [10, 10, 2, 2] and m.denoise_step == original


def test_groot_tensor_cuts_native_sample_both_endpoints_and_restore(monkeypatch):
    from exp.r17_integration.common.contracts import PreparedObs, RequestContext
    from exp.r17_integration.models.groot.manifest import GrootManifest
    from exp.r17_integration.models.groot.provider import GrootMetaWorldAdapter
    from openpi.cache.groot import staged
    from openpi.cache.types import groot_n15_schedule
    manifest = GrootManifest(*(["sha256:" + "a" * 64] * 5), hidden_width=4)
    head = SimpleNamespace(config=SimpleNamespace(action_horizon=16, action_dim=32, add_pos_embed=False),
        num_inference_timesteps=4, num_timestep_buckets=100, action_horizon=16,
        future_tokens=torch.nn.Embedding(1, 2), state_encoder=lambda state, emb: state[..., :2] + 1,
        action_encoder=lambda x, t, emb: x[..., :2] + t[:, None, None] / 100,
        model=lambda hidden_states, encoder_hidden_states, timestep: hidden_states + encoder_hidden_states.mean(),
        action_decoder=lambda x, emb: x.repeat(1, 1, 16),
        process_backbone_output=lambda data: SimpleNamespace(backbone_features=data["backbone_features"] * 2))
    class Language:
        def get_input_embeddings(self):
            return lambda ids: ids[:, :, None].float().expand(-1, -1, 4).clone()
    eagle = SimpleNamespace(language_model=Language(), image_token_index=99,
                            extract_feature=lambda pixels: pixels.mean().expand(1, 256, 4).clone())
    mask = torch.ones(1, 270, dtype=torch.long)
    mask[:, 4:260] = 99
    state = torch.zeros(1, 1, 64)
    state[..., :4] = torch.arange(4.)
    state_mask = torch.zeros_like(state, dtype=torch.bool)
    state_mask[..., :4] = True
    actions = {"state": state, "state_mask": state_mask, "embodiment_id": torch.tensor([31])}
    backbone = {"eagle_input_ids": mask, "eagle_attention_mask": torch.ones_like(mask),
                "eagle_pixel_values": torch.ones(1, 3, 2, 2)}
    model = SimpleNamespace(action_head=head, prepare_input=lambda normalized: (backbone, actions))
    class Runner:
        _eagle = eagle
        _model = model
        def session(self):
            return nullcontext()
        def live_schedule(self):
            return groot_n15_schedule(head.num_inference_timesteps)
        def run_stage1(self, normalized):
            embeds = eagle.language_model.get_input_embeddings()(backbone["eagle_input_ids"])
            image_mask = backbone["eagle_input_ids"] == 99
            embeds[image_mask] = eagle.extract_feature(backbone["eagle_pixel_values"]).reshape(-1, 4)
            return staged.GrootStage1Output(embeds, backbone["eagle_attention_mask"], image_mask, actions)
        def run_stage2_llm(self, s):
            return staged.GrootStage2Output(s.input_embeds + 2, s.attention_mask, s.action_inputs)
        def _head_inputs(self, s):
            return {"backbone_features": s.backbone_features, "backbone_attention_mask": s.attention_mask}
        def sample_noise(self, s, generator):
            return torch.randn(1, 16, 32, generator=generator)
        def run_stage3(self, s, noise, on_step=None):
            x = staged.denoise_loop(head, self._head_inputs(s), s.action_inputs, noise=noise,
                                   num_steps=head.num_inference_timesteps, on_step=on_step)
            return staged.GrootStage3Output(x, None, head.num_inference_timesteps)
    runner = Runner()
    backend = SimpleNamespace(model=model, normalize_state_only=lambda state: state)
    adapter = GrootMetaWorldAdapter(backend, manifest, runner_factory=lambda m: runner)
    request = RequestContext("profile", manifest.benchmark_protocol, "assembly-v3", 0, 0, explicit_seed=0)
    prepared = PreparedObs({}, request, np.zeros(4, np.float32))
    adapter.prepare = lambda *args: prepared
    original_loop = staged.denoise_loop
    original_compile = L.compile_tensor_program
    calls = []
    def compile_stub(fn, inputs):
        calls.append(fn)
        return original_compile(fn, inputs)
    monkeypatch.setattr(L, "compile_tensor_program", compile_stub)
    monkeypatch.setattr(torch, "compile", lambda graph, **kw: graph)
    with torch.inference_mode():
        runtime = L.build_stages("groot_mw", adapter, {}, request)
        try:
            backbone["eagle_pixel_values"].fill_(3)
            s1 = runtime.stage1(prepared)
            expected1 = adapter.run_stage1(prepared)
            assert L.parity(s1.fields, expected1.fields)["passed"]
            assert s1.fields["vision_0"].shape == (64,) and s1.fields["vision_0"].tolist() == [3.] * 64
            s2 = runtime.stage2(s1)
            assert L.parity(L.stage_payload("groot_mw", s2, 2),
                            L.stage_payload("groot_mw", adapter.run_stage2(expected1), 2))["passed"]
            from exp.r17_integration.common.request_rng import RequestRNG
            for k in (4, 2):
                sampler = L.sampler_for_steps("groot_mw", adapter, k)
                L.select_sampler("groot_mw", adapter, sampler)
                assert adapter.metadata().default_sampler == sampler
                compiled_loop = staged.denoise_loop
                staged.denoise_loop = original_loop
                expected = adapter.sample(s2, sampler, RequestRNG(request))
                staged.denoise_loop = compiled_loop
                got = runtime.sample(s2, sampler)
                assert got.actual_nfe == k and got.normalized_chunk.shape == (16, 32)
                assert L.parity(got.normalized_chunk, expected.normalized_chunk)["passed"]
                # Tensor dt must change the Euler update without another compile.
                assert head.num_inference_timesteps == k
        finally:
            runtime.restore()
    assert len(calls) == 3 and staged.denoise_loop is original_loop
    assert adapter.manifest is manifest and head.num_inference_timesteps == 4


@pytest.mark.parametrize("model", L.MT50_MODELS)
def test_mt50_measurement_receipt_with_cpu_stubs(tmp_path, monkeypatch, model):
    from exp.r17_integration.common.contracts import PreparedObs, Stage1, Stage2, SampleResult, SamplerSpec
    from exp.r17_integration.common.request_rng import RequestRNG
    @dataclass
    class Spec:
        checkpoint_identity: str = "stub"
        allowed_benchmark_protocols: tuple = ("mt50-native",)
    class Graph:
        def replay(self):
            pass
        def capture_begin(self):
            pass
    noop = lambda *args, **kwargs: None
    cuda = SimpleNamespace(CUDAGraph=Graph, is_available=lambda: True, set_device=noop, synchronize=noop,
        max_memory_allocated=lambda: 1024, max_memory_reserved=lambda: 2048, reset_peak_memory_stats=noop,
        current_device=lambda: 0, get_device_properties=lambda device: SimpleNamespace(name="stub", total_memory=4096),
        profiler=SimpleNamespace(start=noop, stop=noop), nvtx=SimpleNamespace(range_push=noop, range_pop=noop))
    fake_torch = SimpleNamespace(cuda=cuda, inference_mode=nullcontext, as_tensor=np.asarray, equal=np.array_equal,
        compiler=SimpleNamespace(cudagraph_mark_step_begin=noop), __version__="stub", version=SimpleNamespace(cuda="stub"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    obs = {"prompt": "Press a button"}
    source = {"benchmark_protocol": "mt50-native", "task_name": "button-press-v3", "descriptor": str(tmp_path / "load.json")}
    (tmp_path / "load.json").write_text("{}")
    class Adapter(SimpleNamespace):
        pass
    loaded = Adapter(prepare=lambda *args: PreparedObs({}, None, np.zeros(4)),
        run_stage1=lambda p: Stage1({}, {"vision_0": np.ones(4)}, p.state),
        run_stage2=lambda s: Stage2({}, s), retrieval_fields=lambda s: s.fields,
        metadata=Spec, to_wire=lambda chunk, p: SimpleNamespace(actions=chunk), language=lambda prompt: None)
    sample_calls = []
    def sample(s, sampler, rng, capture=None):
        sample_calls.append(sampler.num_steps)
        return SampleResult(rng.normal((5, 4)), sampler.num_steps, sampler)
    loaded.sample = sample
    pools = []
    def read_input(model, suite, pool, *args):
        pools.append((suite, str(pool)))
        return obs, source
    monkeypatch.setattr(L, "read_input", read_input)
    monkeypatch.setattr(L, "load_adapter", lambda *args: (loaded, {}))
    monkeypatch.setattr(L, "device_exclusive", lambda: {"exclusive": True})
    monkeypatch.setattr(L, "sampler_for_steps", lambda model, adapter, k: SamplerSpec("stub", "stub", k, tuple(range(k))))
    selected = []
    monkeypatch.setattr(L, "select_sampler", lambda model, adapter, sampler: selected.append(sampler.num_steps))
    monkeypatch.setattr(L, "stage_payload", lambda model, s, which: s.fields if which == 1 else np.ones(4))
    monkeypatch.setattr(L, "tensor_copy", lambda value, **kwargs: value)
    def equal(a, b):
        return all(equal(a[k], b[k]) for k in a) if isinstance(a, dict) else bool(np.array_equal(a, b))
    monkeypatch.setattr(L, "parity", lambda a, b, **kw: {"passed": equal(a, b), "rel_frobenius": 0. if equal(a, b) else 1.})
    compiled = SimpleNamespace(value=0)
    restored = []
    monkeypatch.setattr(L, "counters", lambda: {"stats.unique_graphs": compiled.value})
    def build(*args, **kwargs):
        compiled.value = 3
        return SimpleNamespace(stage1=loaded.run_stage1, stage2=loaded.run_stage2,
            sample=lambda s, sampler, request=None: sample(s, sampler, RequestRNG(request or context)),
            restore=lambda: restored.append(True))
    # The real RequestContext is captured from prepare before build_stages.
    context = None
    def prepare(obs, request):
        nonlocal context
        context = request
        return PreparedObs({}, request, np.zeros(4))
    loaded.prepare = prepare
    monkeypatch.setattr(L, "build_stages", build)
    def measure(torch, runtime, adapter, obs, request, sampler, iters):
        for _ in range(iters * (2 + sampler.num_steps)):
            Graph().replay()
        return ({"stage1_ms": [2.] * iters, "stage2_ms": [3.] * iters,
                 "stage3_ms": [float(sampler.num_steps)] * iters, "total_ms": [5. + sampler.num_steps] * iters,
                 "wall_ms": [6. + sampler.num_steps] * iters,
                 "stage3_step_ms": [1.] * (iters * sampler.num_steps),
                 "stage3_network_ms": [.5] * (iters * sampler.num_steps)}, [sampler.num_steps] * iters)
    monkeypatch.setattr(L, "measure_iterations", measure)
    monkeypatch.setattr(L.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="stub\n"))
    out = tmp_path / "receipt.json"
    monkeypatch.setattr(sys, "argv", ["latency", "--model", model, "--out", str(out)])
    L.main()
    receipt = json.loads(out.read_text())
    full, reduced = L.STEPS[model]
    assert pools == [("mt50", "/data/r17_library/" + model + "_mt50/bpool")]
    assert receipt["schema"] == "r17.batch1_latency.v1" and receipt["profile"] == receipt["suite"] == "mt50"
    assert receipt["model"] == model and receipt["warmup"] == 30 and receipt["iters"] == 200
    assert receipt["batch_size"] == 1 and receipt["valid"] and receipt["compiled_stages"] == 3
    assert receipt["coefficients"] == L.pricing(2., 3., float(full), float(reduced))
    assert receipt["expected_cudagraph_launch_count"] == receipt["cudagraph_launch_count"] == 200 * (4 + full + reduced)
    for k in (full, reduced):
        cell = receipt["cells"][str(k)]
        assert cell["actual_nfe_by_iteration"] == [k] * 200
        assert cell["expected_actual_nfe"] == cell["actual_nfe"] == k
        assert cell["fixed_noise_replay_equal"] and cell["sampled_noise_distinct"] and cell["valid"]
    assert sample_calls.count(full) == sample_calls.count(reduced) == 34
    assert selected == [full, reduced, full, full, reduced, full, reduced, full]
    assert restored == [True]
