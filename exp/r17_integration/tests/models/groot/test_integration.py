from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec, DecisionCapture
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.common.serving import full_call
from exp.r17_integration.models.groot.key_builder import GrootMetaWorldCP1SpatialPool16KeyBuilder
from exp.r17_integration.models.groot.loader import load_backend
from exp.r17_integration.models.groot.manifest import (
    checkpoint_manifest, library_metadata, validate_library, validate_training_metadata,
)
from exp.r17_integration.models.groot.provider import GrootMetaWorldAdapter, make_provider
from exp.r17_integration.models.groot.serve import StockBundle, make_stock_connection_factory
from exp.r17_integration.models.groot.wire_policy import GrootMetaWorldWirePolicy, to_groot_observation
from openpi.cache.groot.batcher import stage3_input
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID
from openpi.serving.batching_core import Stage3MissPayload, Stage3WarmStartPayload

from .conftest import Backend, training_metadata


def stages(adapter, obs, context):
    prepared = adapter.prepare(obs, context)
    s1 = adapter.run_stage1(prepared)
    return prepared, s1, adapter.run_stage2(s1)


@pytest.mark.parametrize("k", [4, 2, 1])
def test_staged_matches_unsplit_default_and_capture_no_extra_rng(guarded_stub, manifest, obs, context, k):
    backend = guarded_stub(k)
    manifest = replace(manifest, denoising_steps=k)
    adapter = GrootMetaWorldAdapter(backend, manifest)
    rng_state = torch.get_rng_state().clone()
    plain, wire = full_call(adapter, obs, context)
    assert torch.equal(rng_state, torch.get_rng_state())
    nfe = backend.model.action_head.nfe
    captured, captured_wire = full_call(adapter, obs, context, capture=CaptureSpec(True, True))
    assert torch.equal(rng_state, torch.get_rng_state())
    assert backend.model.action_head.nfe - nfe == k
    assert torch.equal(plain.normalized_chunk, captured.normalized_chunk)
    assert wire.actions.tobytes() == captured_wire.actions.tobytes()
    assert plain.actual_nfe == k and len(captured.snapshots) == k - 1
    seed = RequestRNG(context).seed_for()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        upstream = GrootMetaWorldWirePolicy(backend).infer(obs)["actions"]
    np.testing.assert_array_equal(upstream, wire.actions)
    assert backend.apply_count == backend.inverse_count == 3


@pytest.mark.parametrize("k", [4, 2])
def test_native_snapshot_resume_exact_suffix_and_identity_rejections(guarded_stub, manifest, obs, context, k):
    adapter = GrootMetaWorldAdapter(guarded_stub(k), replace(manifest, denoising_steps=k))
    _, _, s2 = stages(adapter, obs, context)
    sampler, rng = adapter.metadata().default_sampler, RequestRNG(context)
    result = adapter.sample(s2, sampler, rng, CaptureSpec(True, True))
    for snap in result.snapshots:
        resumed = adapter.resume(s2, snap, sampler, rng, CaptureSpec(True))
        assert torch.equal(result.normalized_chunk, resumed.normalized_chunk)
        assert resumed.actual_nfe == k - snap.loop_index
    snap = result.snapshots[0]
    # Owner removed fingerprint gates; numerical shape/dtype/grid still applies.
    for field in ("norm_identity", "layout_identity", "sampler_identity"):
        assert torch.equal(adapter.resume(s2, replace(snap, **{field:"other"}), sampler, rng).normalized_chunk,
                           result.normalized_chunk)
    for invalid in (replace(snap, native_timestep=.111),
                    replace(snap, x_t=snap.x_t.double()), replace(snap, x_t=snap.x_t[:5])):
        with pytest.raises(ValueError):
            adapter.resume(s2, invalid, sampler, rng)


def test_one_camera_mask_pooling_and_active_state_exact(guarded_stub, manifest, obs, context):
    adapter = GrootMetaWorldAdapter(guarded_stub(), manifest)
    _, s1, _ = stages(adapter, obs, context)
    fields = adapter.retrieval_fields(s1)
    assert set(fields) == {"vision_0", "robot_state"}
    native = s1.data
    tokens = native.input_embeds[0, native.image_token_mask[0]].float().reshape(16, 16, 8)
    expected = tokens.reshape(4, 4, 4, 4, 8).mean((1, 3)).reshape(-1)
    assert torch.equal(fields["vision_0"], expected)
    np.testing.assert_array_equal(fields["robot_state"].numpy(), obs["state"] * 2 - 1)
    np.testing.assert_array_equal(adapter.extract_state_only(obs).state, fields["robot_state"].numpy())
    fields["robot_state"][0] = 100
    assert s1.fields["robot_state"][0] != 100
    for mask in (torch.zeros_like(native.state_mask), torch.ones_like(native.state_mask)):
        invalid = copy.copy(native)
        invalid.action_inputs = dict(native.action_inputs, state_mask=mask)
        with pytest.raises(RuntimeError, match="first4"):
            adapter.key_builder.collect(CheckpointID.CP1, stage1=invalid)
    invalid = copy.copy(native)
    invalid.image_token_mask = native.image_token_mask.clone()
    invalid.image_token_mask[0, 10] = False
    builder = GrootMetaWorldCP1SpatialPool16KeyBuilder(hidden_width=8)
    builder.collect(CheckpointID.CP1, stage1=invalid)
    with pytest.raises(RuntimeError, match="image-token runs"):
        builder.build(CheckpointID.CP1)
    with pytest.raises(ValueError, match="vision_1"):
        GrootMetaWorldCP1SpatialPool16KeyBuilder(["vision_1"])


def test_source_guards_not_disabled_and_bad_embodiment_refused(manifest, guarded_stub, obs, context, monkeypatch):
    import openpi.cache.groot.staged as staged
    backend = Backend()
    monkeypatch.setattr(staged, "UPSTREAM_FORWARD_SHA256", "wrong")
    with pytest.raises(RuntimeError, match="Upstream forward has changed"):
        GrootMetaWorldAdapter(backend, manifest)


def test_embedding_slot_and_unexpected_hidden_width_refused(guarded_stub, manifest, obs, context):
    adapter = GrootMetaWorldAdapter(guarded_stub(), manifest)
    prepared = adapter.prepare(obs, context)
    prepared.data["embodiment_id"] = torch.tensor([0])
    with pytest.raises(RuntimeError, match="slot31"):
        adapter.run_stage1(prepared)
    adapter = GrootMetaWorldAdapter(guarded_stub(), replace(manifest, hidden_width=9))
    with pytest.raises(RuntimeError, match="hidden width"):
        adapter.run_stage1(adapter.prepare(obs, context))


@pytest.mark.parametrize("mutate", ["libero", "state8", "action7", "wrist", "nan", "order"])
def test_loader_rejects_wrong_training_geometry(tmp_path, mutate):
    metadata = training_metadata()
    value = metadata["metaworld_sawyer"]
    if mutate == "libero":
        metadata = {"libero": value}
    elif mutate == "state8":
        value["modalities"]["state"]["extra"] = {"start": 4, "end": 8}
    elif mutate == "action7":
        value["statistics"]["action"]["x"]["min"] = [0] * 7
    elif mutate == "wrist":
        value["modalities"]["video"]["wrist_image"] = {}
    elif mutate == "nan":
        value["statistics"]["state"]["x"]["min"] = [float("nan")]
    else:
        value["modalities"]["action"]["gripper"]["shape"] = [4]
    with pytest.raises((ValueError, KeyError)):
        validate_training_metadata(metadata)


def test_checkpoint_content_hash_detects_same_size_mtime_tamper(tmp_path):
    cfg = tmp_path / "experiment_cfg"
    cfg.mkdir()
    (cfg / "metadata.json").write_text(json.dumps(training_metadata()))
    (tmp_path / "config.json").write_text(json.dumps({"action_head_cfg": {"action_horizon": 16, "action_dim": 32}}))
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"abcd")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{}')
    code = tmp_path / "vendor.py"
    code.write_text("# fixture vendor")
    kwargs = {"tokenizer_files": [tokenizer], "code_files": [code]}
    original = checkpoint_manifest(tmp_path, **kwargs)
    stat = weights.stat()
    weights.write_bytes(b"ABCD")
    import os
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert checkpoint_manifest(tmp_path, **kwargs).checkpoint_identity != original.checkpoint_identity
    with pytest.raises(ValueError, match="content inputs"):
        checkpoint_manifest(tmp_path)
    class Loaded:
        backend = Backend()
        def get_metadata(self):
            return {"embodiment": "libero"}
    with pytest.raises(ValueError, match="metadata differs"):
        load_backend(tmp_path, device="cpu", policy_loader=lambda *a, **kw: Loaded())


def test_library_identity_and_active_geometry_fail_closed(manifest):
    good = library_metadata(manifest)
    validate_library(good, manifest)
    for field in manifest.identities:
        invalid = dict(good, **{field: "wrong"})
        with pytest.raises(ValueError, match=field):
            validate_library(invalid, manifest)
    for change in ({"vector_dims": {"vision_0": 128, "vision_1": 128, "robot_state": 8}},
                   {"action_shape": [16, 4]}, {"active_action_indices": list(range(7))},
                   {"denoise_schedule": "groot_n15_k2_v1"}):
        with pytest.raises(ValueError):
            validate_library(dict(good, **change), manifest)


def stock_orchestrator(backend, manifest, *, hit):
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.components.gate import AlwaysSearchGate
    from openpi.cache.components.judge import AlwaysHitJudge
    from openpi.cache.components.search_strategy import WeightedScoreSumKnnStrategy
    from openpi.cache.orchestrator import CacheOrchestrator
    vector = backend if backend is not None else InMemoryBackend({"vision_0": 128, "robot_state": 4})
    storage = CacheStorage(vector)
    if hit:
        storage.enter_prefill_mode(CachePayload(torch.full((16, 32), .5)))
    key = GrootMetaWorldCP1SpatialPool16KeyBuilder(hidden_width=manifest.hidden_width)
    return CacheOrchestrator(storage, key, {CheckpointID.CP1: AlwaysSearchGate()},
                             {CheckpointID.CP1: AlwaysHitJudge()},
                             {CheckpointID.CP1: WeightedScoreSumKnnStrategy(storage, task_scoped=False)})


@pytest.mark.parametrize("hit", [False, True])
@pytest.mark.parametrize("k", [4, 2, 1])
def test_stock_cache_factory_hit_miss_transform_once_and_connection_isolation(guarded_stub, manifest, obs, k, hit):
    backend = guarded_stub(k)
    manifest = replace(manifest, denoising_steps=k)
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    shared = InMemoryBackend({"vision_0": 128, "robot_state": 4})
    bundle = StockBundle(manifest, k, lambda storage: stock_orchestrator(storage, manifest, hit=hit),
                         library_meta=library_metadata(manifest), storage=shared)
    factory = make_stock_connection_factory(backend, manifest, {"full-g1": bundle, "cache-g2": bundle})
    first, second = factory(None, "full-g1"), factory(None, "cache-g2")
    first.on_task_begin()
    second.on_task_begin()
    first.on_episode_start(task="reach-v3", episode_id=0)
    reply = first.infer(obs)
    assert reply["actions"].shape == (16, 4)
    assert backend.apply_count == backend.inverse_count == 1
    assert backend.model.backbone.eagle_model.extract_calls == 1
    assert backend.model.backbone.eagle_model.language_model.calls == (0 if hit else 1)
    assert backend.model.action_head.nfe == (0 if hit else k)
    if hit:
        np.testing.assert_array_equal(reply["actions"], np.full((16, 4), 1.75, np.float32))
    assert first.inner._runner is not second.inner._runner
    a, b = first.inner._orchestrator, second.inner._orchestrator
    assert a._key_builder is not b._key_builder
    assert a._storage is not b._storage and a._storage._backend is b._storage._backend
    assert a._step_counter == 1 and b._step_counter == 0
    assert not a._key_builder.cached_data
    first.on_episode_end(True)
    with pytest.raises(ValueError, match="unknown"):
        factory(None, "missing")
    backend.model.action_head.num_inference_timesteps = 3
    with pytest.raises(RuntimeError, match="head runs"):
        first.infer(obs)


def test_endpoint_cannot_hot_mutate_k_and_mismatched_bundle_refused(guarded_stub, manifest, obs, context):
    backend = guarded_stub()
    provider = make_provider(backend, manifest)
    for k in (1, 2, True, 3):
        with pytest.raises(ValueError, match="immutable"):
            provider.sampler_for_steps(k)
    assert backend.model.action_head.num_inference_timesteps == 4
    wrong = StockBundle(manifest, 2)
    factory = make_stock_connection_factory(backend, manifest, {"wrong": wrong})
    with pytest.raises(ValueError, match="MISS K"):
        factory(None, "wrong")
    adapter = provider.load_adapter()
    backend.model.action_head.num_inference_timesteps = 2
    with pytest.raises(RuntimeError, match="K changed"):
        adapter.run_stage1(adapter.prepare(obs, context))


def test_stock_s3_batch_matches_single_and_rejects_shape_schedule_semantics(guarded_stub, manifest, obs, context):
    adapter = GrootMetaWorldAdapter(guarded_stub(), manifest)
    _, _, s2 = stages(adapter, obs, context)
    sampler = adapter.metadata().default_sampler
    batcher = adapter.make_stage_batcher()
    payloads, singles = [], []
    for i in range(2):
        rng = RequestRNG(replace(context, original_init_idx=i))
        result = adapter.sample(s2, sampler, rng, CaptureSpec(True, True))
        singles.append(result)
        payloads.append(Stage3MissPayload(stage3_input(adapter.runner, s2.data), result.capture["initial_noise"], 4))
    replies = batcher.run_stage3_miss(payloads, num_steps=4, save_timesteps_per_request=[(.25,), (.75,)])
    for reply, single in zip(replies, singles):
        assert torch.equal(reply.action[0].float(), single.normalized_chunk)
    assert set(replies[0].intermediates) == {0., .25}
    assert set(replies[1].intermediates) == {0., .75}
    before = replies[1].action.clone()
    replies[0].action[0, 0, 0] = 999
    assert torch.equal(before, replies[1].action)
    for changed in (replace(payloads[0], noise=torch.zeros(15, 32)),
                    replace(payloads[0], stage2_out=replace(payloads[0].stage2_out, schedule_id="groot_n15_k2_v1")),
                    replace(payloads[0], stage2_out=replace(payloads[0].stage2_out, embodiment=(0,)))):
        with pytest.raises(ValueError):
            batcher.run_stage3_miss([changed], num_steps=4, save_timesteps_per_request=[None])
    assert batcher.run_stage1_batch([]) == []
    assert batcher.run_stage2_batch([], capture=False) == []
    warm = [Stage3WarmStartPayload(p.stage2_out, s.snapshots[0].x_t, .25, 4) for p, s in zip(payloads, singles)]
    resumed = batcher.run_stage3_warm(warm, start_t=.25, num_steps=4, capture_first_step=True)
    for reply, single in zip(resumed, singles):
        assert torch.equal(reply.action[0].float(), single.normalized_chunk)


def test_capture_uses_actual_one_camera_outputs_no_model_calls(guarded_stub, manifest, obs, context):
    adapter = GrootMetaWorldAdapter(guarded_stub(), manifest)
    prepared, s1, s2 = stages(adapter, obs, context)
    sample = adapter.sample(s2, adapter.metadata().default_sampler, RequestRNG(context))
    wire = adapter.to_wire(sample.normalized_chunk, prepared)
    calls = adapter.backend.model.action_head.nfe
    trace = adapter.make_trace_adapter()
    decision = DecisionCapture(context, "policy", 0, 0, (1, 1, 4), 4, sample.normalized_chunk,
                               wire.actions, adapter.retrieval_fields(s1))
    captured = trace.capture(decision)
    assert captured["normalized_chunk"].shape == (16, 32)
    assert captured["wire_actions"].shape == (16, 4)
    assert set(captured["fields"]) == {"vision_0", "robot_state"}
    captured["normalized_chunk"][0, 0] = 999
    assert sample.normalized_chunk[0, 0] != 999
    assert adapter.backend.model.action_head.nfe == calls


def test_optional_cp2_has_mw_geometry_and_explicit_projection(guarded_stub, manifest, obs, context):
    adapter = GrootMetaWorldAdapter(guarded_stub(), manifest)
    _, _, s2 = stages(adapter, obs, context)
    builder = adapter.make_cp2_key_builder(token_len=300, state_feat_dim=32, seed=7, d=12, p=.1)
    before = adapter.backend.model.action_head.nfe
    fields = adapter.cp2_fields(s2, builder)
    assert set(fields) == {"vlm_out"} and fields["vlm_out"].shape == (12,)
    assert not fields["vlm_out"].is_inference()
    assert adapter.backend.model.action_head.nfe == before
    meta = builder.projection_meta()
    assert meta["layout"]["kind"] == "groot_mw_encoded_v1"
    assert meta["layout"]["active_state_indices"] == [0, 1, 2, 3]
    assert meta["layout"]["real_cameras"] == ["video.image"]
    small = adapter.make_cp2_key_builder(token_len=200, state_feat_dim=32, seed=7, d=12, p=.1)
    with pytest.raises(RuntimeError, match="exceed"):
        adapter.cp2_fields(s2, small)


def test_optional_warm_reset_reuses_native_engine_with_state4(guarded_stub, manifest, obs, context, tmp_path):
    from openpi.cache.warm_reset.types import WarmResetSpec, resolve_plan
    from openpi.cache.warm_reset.groot import run_groot_continuation
    from openpi.cache.types import groot_n15_schedule
    adapter = GrootMetaWorldAdapter(guarded_stub(), manifest)
    _, _, s2 = stages(adapter, obs, context)
    schedule = groot_n15_schedule(4)
    spec = WarmResetSpec("cache", "snapshot", "reset", 1., 2, None, (), str(tmp_path))
    plan = resolve_plan(spec, schedule, .5)
    noise = torch.zeros(1, 16, 32)
    before = adapter.backend.model.action_head.nfe
    with adapter.runner.session():
        output = run_groot_continuation(adapter.runner, s2.data, noise, plan, schedule=schedule)
    assert output.action_pred.shape == (1, 16, 32)
    assert output.steps_run == 2 and adapter.backend.model.action_head.nfe - before == 2
    adapter.backend.model.action_head.num_inference_timesteps = 2
    with adapter.runner.session(), pytest.raises(RuntimeError):
        run_groot_continuation(adapter.runner, s2.data, noise, plan, schedule=schedule)


@pytest.mark.parametrize("k", [4, 2, 1])
def test_stock_trace_writes_true_one_camera_state4_native_grid(guarded_stub, manifest, obs, tmp_path, k):
    import h5py
    from exp.r17_integration.common.config import TraceConfig
    from exp.r17_integration.models.groot.trace import make_stock_trace_runtime
    backend = guarded_stub(k)
    manifest = replace(manifest, denoising_steps=k)
    runtimes = []
    def trace(runner, orchestrator, bundle_id):
        runtime = make_stock_trace_runtime(TraceConfig(enabled=True, out_dir=str(tmp_path), record_noise_actions=True),
            cache_config=None, components=None, orchestrator=orchestrator, runner=runner,
            bundle_id=bundle_id, yaml_id="P5-trace", yaml_path=None, concurrent=True)
        runtimes.append(runtime)
        return runtime
    factory = make_stock_connection_factory(backend, manifest, {"trace": StockBundle(manifest, k, trace_factory=trace)})
    policy = factory(None, "trace")
    policy.on_task_begin()
    policy.on_episode_start(experiment="cpu", task="reach-v3", episode_id=0, episode_name="P5",
                            extra_metadata={"task_uid": "cpu-P5", "attempt": 1})
    try:
        output = policy.infer(obs)
        policy.on_episode_end(True)
        policy.on_task_end()
        runtime = runtimes[0]
        assert runtime.sink.writer.drain(timeout=10).ok
        with h5py.File(tmp_path / "cpu" / "P5.h5", "r") as file:
            assert file.attrs["denoising_num_steps"] == k
            step = file["step_0000"]
            assert {name for name in step if name.startswith("vision_")} == {"vision_0"}
            assert step["vision_0"].shape == (256, 8)
            assert step["robot_state"].shape == (4,)
            assert step["clean_action"].shape == (16, 32)
            assert set(step["trace/raw_images"]) == {"video.image"}
            assert step["trace/raw_state"].shape == (4,)
            for i in range(k):
                assert step[f"noise_action_{i}"].shape == (16, 32)
        assert output["actions"].shape == (16, 4)
    finally:
        if runtimes:
            runtimes[0].sink.writer.stop(timeout=10)


def test_default_off_import_and_untouched_legacy_source_bytes():
    baseline = Path("exp/r17_integration/models/groot/legacy_sources.json")
    for filename, sha in json.loads(baseline.read_text()).items():
        # These two native modules now implement the requested all-stage batches.
        if filename in ('src/openpi/cache/groot/batcher.py', 'src/openpi/cache/groot/staged.py'):
            continue
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == sha, filename
    program = "import sys; import exp.r17_integration.models.groot; from exp.r17_integration.common.provider import get_provider; assert 'gr00t' not in sys.modules; assert 'torch' not in sys.modules; assert 'jax' not in sys.modules;\ntry: get_provider('r17_groot_mw_v1')\nexcept ValueError: pass\nelse: raise AssertionError('registered by import')"
    subprocess.run([sys.executable, "-B", "-c", program], check=True)
