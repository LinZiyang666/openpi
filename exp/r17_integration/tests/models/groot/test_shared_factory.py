"""P5 exercises P1's exact provider/YAML/factory seams on a CPU GR stub."""
from __future__ import annotations

from dataclasses import replace
import copy
import pytest

from exp.r17_integration.common.provider import binding_for, make_trace_adapter, register_provider, validate_binding
from exp.r17_integration.models.groot.provider import make_provider
from exp.r17_integration.models.groot.serve import make_connection_factory
from .conftest import begin_episode
from exp.r17_integration.common.config import (BackendConfig, CacheConfig, CheckpointConfig, GateConfig, JudgeConfig,
                                KeyBuilderConfig, KeysConfig, SearchStrategyConfig, WritePolicyConfig,
                                build_shared_storage, validate_cache_config)


def config(provider, *, commit=10):
    keys = KeysConfig()
    for name, value in vars(keys).items():
        value.enabled = name in ("vision_0", "robot_state")
    result = CacheConfig(enabled=True, keys=keys,
        key_builder=KeyBuilderConfig(type="adapter_native"),
        checkpoints={"cp1": CheckpointConfig(gate=GateConfig(type="always_search"),
                    judge=JudgeConfig(type="always_hit"),
                    search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", task_scoped=False))},
        backend=BackendConfig(type="in_memory", vector_dims=dict(provider.model_spec.feature_spec.fields)),
        write_policy=WritePolicyConfig(type="never"),
        adapter_binding=binding_for(provider, provider.model_spec.allowed_benchmark_protocols[0], commit_horizon=commit))
    validate_cache_config(result)
    return result


@pytest.fixture
def published(guarded_stub, manifest, monkeypatch):
    import exp.r17_integration.common.provider as registry
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_traces", {})
    provider = make_provider(guarded_stub(), manifest)
    from exp.r17_integration.models.groot.provider import register_provider
    register_provider(provider)
    return provider


@pytest.mark.parametrize("commit", [10, 15])
def test_common_two_bundles_actual_ack_tail_no_model_calls(published, obs, monkeypatch, commit):
    import openpi.serving.websocket_policy_server as ws
    cfg = config(published, commit=commit)
    shared = build_shared_storage(cfg)
    monkeypatch.setattr(ws, "_bundles", {
        "g1": ws.CurrentCacheBundle("g1.yaml", cfg, shared, 1, "g1"),
        "g2": ws.CurrentCacheBundle("g2.yaml", copy.deepcopy(cfg), shared, 2, "g2")})
    captured = []
    factory = make_connection_factory(published, cfg, shared, capture_sink=captured.append)
    first, second = factory(None, "g1"), factory(None, "g2")
    first.on_task_begin()
    second.on_task_begin()
    begin_episode(first, task_id=0, init_state_id=0, episode_id=0,
                           extra_metadata={"task_name": "reach-v3", "original_init_idx": 0})
    reply = first.infer(obs)
    model = first.adapter.backend.model
    assert reply["actions"].shape == (commit, 4)
    assert captured[-1].stage_calls == (1, 1, 4)
    assert captured[-1].actual_nfe == 4
    assert first.adapter is not second.adapter
    assert first.adapter.runner is not second.adapter.runner
    assert first.adapter.key_builder is not second.adapter.key_builder
    assert first.adapter.backend is second.adapter.backend
    assert first.orchestrator._storage._backend is second.orchestrator._storage._backend
    before = (model.backbone.eagle_model.extract_calls, model.backbone.eagle_model.language_model.calls, model.action_head.nfe)
    tail = first.infer(dict(obs, __execution__={"decision_seq": 0, "executed_count": 3, "cursor": 3}))
    assert captured[-1].cursor == 3 and captured[-1].stage_calls == (0, 0, 0)
    assert tail["actions"].tobytes() == reply["actions"][3:].tobytes()
    assert before == (model.backbone.eagle_model.extract_calls, model.backbone.eagle_model.language_model.calls, model.action_head.nfe)
    trace = make_trace_adapter(published.model_spec.model_id, first.adapter)
    assert trace.capture(captured[-1])["wire_actions"].shape == (commit - 3, 4)
    with pytest.raises(ValueError, match="stale"):
        first.infer(dict(obs, __execution__={"decision_seq": 0, "executed_count": 2}))
    assert second._seq == 0 and second._cursor == 0
    first.on_episode_end(success=False, execution={"decision_seq": 1, "executed_count": 2})
    first.on_task_end()
    assert first._chunk is None
    with pytest.raises(ValueError, match="unknown"):
        factory(None, "missing")


def test_common_provider_binding_refuses_wrong_k_norm_layout_and_commit(published):
    good = binding_for(published, published.model_spec.allowed_benchmark_protocols[0], commit_horizon=15)
    validate_binding(good)
    for field, value in (("norm_identity", "other"), ("layout_identity", "other"), ("benchmark_protocol", "libero")):
        validate_binding(dict(good, **{field:value}))  # Existing owner rule: no fingerprint gates.
    for field, value in (("num_steps", 2), ("commit_horizon", 17)):
        with pytest.raises(ValueError):
            validate_binding(dict(good, **{field: value}))


@pytest.mark.parametrize("commit", [10, 15])
def test_actual_execution_exhausts_commit_then_recomputes(published, obs, commit):
    cfg = config(published, commit=commit)
    captures = []
    policy = make_connection_factory(published, cfg, build_shared_storage(cfg), capture_sink=captures.append)()
    begin_episode(policy)
    first = policy.infer(obs)
    nfe = policy.adapter.backend.model.action_head.nfe
    cursor = 0
    seq = 0
    while cursor < commit:
        count = min(5, commit - cursor)
        cursor += count
        reply = policy.infer(dict(obs, __execution__={"decision_seq": seq, "executed_count": count, "cursor": cursor}))
        seq += 1
        if cursor < commit:
            assert captures[-1].stage_calls == (0, 0, 0)
            assert reply["actions"].tobytes() == first["actions"][cursor:].tobytes()
            assert policy.adapter.backend.model.action_head.nfe == nfe
        else:
            assert captures[-1].stage_calls == (1, 1, 4)
            assert captures[-1].cursor == 0
            assert policy.adapter.backend.model.action_head.nfe == nfe + 4


def test_concurrent_requests_semantic_rng_matches_independent_order(published, obs):
    from concurrent.futures import ThreadPoolExecutor
    cfg = config(published)
    factory = make_connection_factory(published, cfg, build_shared_storage(cfg))
    policies = [factory(), factory()]
    from exp.r17_integration.benchmarks.metaworld import decision_seed
    for policy in policies:
        begin_episode(policy)
    requests = [dict(obs, __request__=dict(obs["__request__"], original_init_idx=i,
                                         explicit_seed=decision_seed("reach-v3", i, 0))) for i in (1, 2)]
    with ThreadPoolExecutor(2) as pool:
        parallel = list(pool.map(lambda pair: pair[0].infer(pair[1]), zip(policies, requests)))
    for actual, obs_i in zip(parallel, requests):
        independent = factory()
        begin_episode(independent)
        assert independent.infer(obs_i)["actions"].tobytes() == actual["actions"].tobytes()
    assert policies[0].adapter.model_lock is policies[1].adapter.model_lock
    assert policies[0]._seq == policies[1]._seq == 1


def test_real_stock_payload_hit_validated_and_skips_llm_head(published, manifest, obs):
    import torch
    from exp.r17_integration.models.groot.manifest import library_metadata
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    cfg = config(published)
    shared = build_shared_storage(cfg)
    backend = InMemoryBackend(dict(published.model_spec.feature_spec.fields))
    backend.insert(CacheEntry("donor", CheckpointID.CP1,
                             {name: torch.ones(dim) for name, dim in backend.vector_dims.items()},
                             CachePayload(torch.full((16, 32), .5), task_key="reach-v3")))
    backend.artifact_meta = library_metadata(manifest)
    backend.freeze()
    shared._backend = backend
    factory = make_connection_factory(published, cfg, shared)
    policy = factory()
    begin_episode(policy)
    reply = policy.infer(obs)
    assert reply["__hit_meta__"]["stage_calls"] == (1, 0, 0)
    assert policy.adapter.backend.model.action_head.nfe == 0
    assert policy.adapter.backend.apply_count == policy.adapter.backend.inverse_count == 1
    backend.artifact_meta["norm_identity"] = "wrong"
    with pytest.raises(ValueError, match="norm_identity"):
        factory()


def test_endpoint_catalog_dispatches_episode_to_matching_k_only(manifest):
    from exp.r17_integration.conductor.manifest import episode_tasks
    manifest = replace(manifest, denoising_steps=2)
    endpoint = manifest.endpoint_spec("127.0.0.1", 12502)
    task = episode_tasks(campaign="P5-cpu", arm="plain-k2", bundle_id="k2-g1", endpoint=endpoint.endpoint,
                         model_identity=manifest.checkpoint_identity, norm_identity=manifest.norm_identity,
                         layout_identity=manifest.model_spec().layout.identity, num_steps=2,
                         partition="smoke", names=["reach-v3"], indices=[0])[0]
    assert endpoint.accepts(task)
    assert not replace(manifest, denoising_steps=4).endpoint_spec("127.0.0.1", 12504).accepts(task)
    assert not replace(endpoint, norm_identity="wrong").accepts(task)
