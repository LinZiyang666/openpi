"""Opt-in native batch config: private RNG, mixed K, real queues, commits."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json

import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.common.serving import full_call
from exp.r17_integration.models.dp.batch_runtime import BatchConfig, FrozenLanguageMemo, create_runtime
from exp.r17_integration.models.dp.provider import CoordinatedAdapter, make_connection_factory, register_provider
from exp.r17_integration.models.dp.request_noise import RequestNoise
from exp.r17_integration.models.dp.sampler import SamplerProfile
from exp.r17_integration.models.dp.serve import plain_config
from exp.r17_integration.tests.models.dp.conftest import observation, episode_metadata


@pytest.mark.parametrize("steps", [100, 10, 5])
@pytest.mark.parametrize("batch", [1, 4, 32])
def test_vector_scheduler_keeps_each_B1_initial_posterior_stream(make_adapter, steps, batch):
    adapter = make_adapter("libero")
    profile = SamplerProfile(num_steps=steps)
    vector_scheduler = profile.scheduler(adapter.model.cfg)
    schedulers = [profile.scheduler(adapter.model.cfg) for _ in range(batch)]
    generators = [torch.Generator().manual_seed(i+37) for i in range(batch)]
    references = [torch.Generator().manual_seed(i+37) for i in range(batch)]
    x = torch.cat([torch.randn(1, 16, 7, generator=g) for g in generators])
    y = [torch.randn(1, 16, 7, generator=g) for g in references]
    assert torch.equal(x, torch.cat(y))
    for t in vector_scheduler.timesteps:
        prediction = torch.sin(x)  # row-wise oracle, independent of peer rows
        x = profile.step(vector_scheduler, prediction, t, x, generators)
        y = [profile.step(s, torch.sin(v), t, v, g) for s, v, g in zip(schedulers, y, references)]
        assert torch.equal(x, torch.cat(y))
    assert all(torch.equal(a.get_state(), b.get_state()) for a, b in zip(generators, references))


@pytest.mark.parametrize("steps", [100, 10, 5])
def test_native_vector_sample_RNG_independent_of_order(make_adapter, req_context, steps):
    adapter = make_adapter("libero")
    prepared = [adapter.prepare(observation("libero", p), req_context) for p in ("reach", "drawer", "bowl", "plate")]
    stages = [adapter.run_stage2(adapter.run_stage1(p)) for p in prepared]
    profile = SamplerProfile(num_steps=steps)
    seeds = [18, 37, 56, 75]
    single = [adapter.sample(s, profile, RequestNoise(seed), CaptureSpec()) for s, seed in zip(stages, seeds)]
    adapter.batch_config = BatchConfig(32)
    vector = adapter.sample_batch(stages, profile, [RequestNoise(seed) for seed in seeds], [CaptureSpec()] * 4)
    reversed_outputs = adapter.sample_batch(stages[::-1], profile, [RequestNoise(seed) for seed in seeds[::-1]], [CaptureSpec()] * 4)
    for one, two, reversed_two in zip(single, vector, reversed_outputs[::-1]):
        assert one.rng_receipt == two.rng_receipt == reversed_two.rng_receipt
        assert two.actual_nfe == steps
        torch.testing.assert_close(one.normalized_chunk, two.normalized_chunk, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(two.normalized_chunk, reversed_two.normalized_chunk, rtol=1e-3, atol=1e-4)


def test_cache_reduced_miss_payload_runs_trailing_ddim10_at_b8(make_adapter, req_context, tmp_path):
    adapter = make_adapter('libero')
    stage2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation('libero'), req_context)))
    sampler = SamplerProfile(num_steps=10).to_spec(adapter.model.cfg)
    assert sampler.family == 'ddim' and sampler.config['eta'] == 0
    assert sampler.native_timesteps == tuple(range(99, -1, -10))
    runtime = create_runtime(adapter, 8, 5, tmp_path/'batch.jsonl')
    try:
        batcher = adapter.make_stage_batcher()
        stages = [replace(stage2, data=dict(stage2.data, batch_member=dict(connection_id=str(i), request_id=str(i),
                          request=asdict(req_context)), batching={})) for i in range(8)]
        payloads = [batcher.miss_payload(stage, sampler, RequestNoise(i), CaptureSpec()) for i, stage in enumerate(stages)]
        outputs = batcher.run_stage3_miss(payloads, num_steps=10, save_timesteps_per_request=[None]*8)
        assert len(outputs) == 8
        assert all(out.actual_nfe == 10 and out.result.sampler == sampler for out in outputs)
        assert all(out.result.capture['batching']['stage3']['size'] == 8 for out in outputs)
    finally:
        runtime.close()


def test_default_off_and_runtime_restores_resident(make_adapter, tmp_path):
    adapter = make_adapter("libero")
    original_language = adapter.language
    spec = adapter.metadata()
    assert create_runtime(adapter) is None
    assert not list(tmp_path.iterdir()) and adapter.batch_config is None
    with pytest.raises(ValueError, match="ledger"):
        create_runtime(adapter, 4)
    runtime = create_runtime(adapter, 4, 5, tmp_path / "batch.jsonl")
    try:
        assert adapter.metadata() == spec
        assert adapter.batch_config.max_batch == 4 and isinstance(adapter.language, FrozenLanguageMemo)
        with pytest.raises(ValueError, match="exceeds"):
            adapter._check_batch(5)
    finally:
        runtime.close()
    assert adapter.batch_config is None and adapter.language is original_language


def test_frozen_language_memo_preserves_B1_and_avoids_shared_alias():
    calls = []
    def language(prompts):
        calls.append(prompts)
        return torch.tensor([[float(len(p))] for p in prompts])
    memo = FrozenLanguageMemo(language)
    first = memo(["reach", "reach", "drawer"])
    assert calls == [["reach"], ["drawer"]]
    first.zero_()
    assert memo(["reach"]).item() == 5


@pytest.mark.parametrize("horizon", [5, 10, 15, 16])
def test_real_queue_cross_connection_mixed_K_commit_unchanged(make_adapter, monkeypatch, tmp_path, horizon):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.common.config import build_shared_storage
    monkeypatch.setattr(registry, "_providers", {})
    adapter = make_adapter("libero")
    runtime = create_runtime(adapter, 4, 20, tmp_path / "batch.jsonl")
    try:
        provider = register_provider(adapter)
        connections = []
        for k in (100, 10, 5, 5, 5, 10):
            config = plain_config(provider, num_steps=k, commit_horizon=horizon, allow_partial_commit=horizon == 16)
            factory = make_connection_factory(adapter, provider=provider, default_config=config,
                default_storage=build_shared_storage(config), stage_core=runtime.core)
            core = runtime.wrap_connection_factory(factory)()
            metadata = episode_metadata("libero_spatial/task3")
            if horizon == 16:
                metadata["partial_tail_protocol"] = "r17.partial_tail.v1"
            core.on_episode_start(extra_metadata=metadata)
            connections.append(core)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda core: core.infer(dict(observation("libero"), __request__={"explicit_seed": 18})), connections))
        for core, result, k in zip(connections, results, (100, 10, 5, 5, 5, 10)):
            assert result["actions"].shape == (horizon, 7)
            hit = result["__hit_meta__"]
            assert hit["actual_nfe"] == k and hit["stage_calls"] == (1, 1, k)
            assert set(hit["batching"]["stages"]) == {"stage1", "stage2", "stage3"}
            stage3 = hit["batching"]["stages"]["stage3"]
            assert stage3["members"][stage3["row"]]["connection_id"] == core.adapter.connection_id
            core.on_episode_end(execution={"decision_seq": 0, "executed_count": 3})
        rows = [json.loads(line) for line in (tmp_path / "batch.jsonl").read_text().splitlines()]
        assert any(row.get("stage") == "stage3" and row["size"] > 1 for row in rows)
    finally:
        runtime.close()


@pytest.mark.parametrize("maximum,window", [(0, 5), (33, 5), (4, -1), (4, float("nan"))])
def test_batch_config_bounds(maximum, window):
    with pytest.raises(ValueError):
        BatchConfig(maximum, window)


def test_graph_worker_must_be_single_and_runtime_LIBERO_only(make_adapter, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="LIBERO"):
        create_runtime(make_adapter("metaworld"), 4, 5, tmp_path / "never.jsonl")
    monkeypatch.setenv("BATCHING_STAGE3_WORKERS", "2")
    with pytest.raises(ValueError, match="one worker"):
        create_runtime(make_adapter("libero"), 4, 5, tmp_path / "never2.jsonl")


def test_all_graph_batch_signatures_captured_before_runtime_threads(make_adapter, req_context):
    from exp.r17_integration.models.dp.batch_graph import warm_batch_signatures
    adapter = make_adapter("libero")
    s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation("libero"), req_context)))
    calls = []
    class Graph:
        r17_capture_sealed = False
        def __call__(self, x, t, cond):
            assert not self.r17_capture_sealed
            assert adapter.batch_config is None  # runtime/thread setup has not started
            assert len(x) == len(cond) and x.shape[1:] == (16, 7)
            calls.append(len(x))
            return x
    graph = Graph()
    report = warm_batch_signatures(adapter, graph, s2, 32, torch.tensor(99))
    assert calls == list(range(1, 33))
    assert report["batch_sizes"] == calls and not report["live_capture_allowed"]
    assert graph.r17_capture_sealed
