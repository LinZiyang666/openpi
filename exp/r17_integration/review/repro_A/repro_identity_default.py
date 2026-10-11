"""Repro: ConnectionPolicy identity when the client sends the stock MW runner's extra (no R17 envelope)."""
import numpy as np
import exp.r17_integration.common.provider as registry
from exp.r17_integration.common.fake_adapter import fake_provider, fake_spec
from exp.r17_integration.common.provider import binding_for
from exp.r17_integration.common.serving import make_connection_factory
from exp.r17_integration.common.request_rng import RequestRNG
from openpi.cache.config import (BackendConfig, CacheConfig, CheckpointConfig, GateConfig, JudgeConfig,
                                 KeyBuilderConfig, KeysConfig, SearchStrategyConfig, WritePolicyConfig,
                                 build_shared_storage)
p = fake_provider("fake.review.v1", fake_spec(horizon=5))
keys = KeysConfig()
for name, v in vars(keys).items():
    v.enabled = name in p.model_spec.feature_spec.fields
cfg = CacheConfig(enabled=True, keys=keys, key_builder=KeyBuilderConfig(type="adapter_native"),
    checkpoints={"cp1": CheckpointConfig(gate=GateConfig(type="always_skip"), judge=JudgeConfig(type="always_hit"),
                                         search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"))},
    backend=BackendConfig(type="in_memory", vector_dims=dict(p.model_spec.feature_spec.fields)),
    write_policy=WritePolicyConfig(type="never"), adapter_binding=binding_for(p, "fake_mw.v1"))
pol = make_connection_factory(p, cfg, build_shared_storage(cfg))(None, "default")
seeds = {}
for idx in (3, 7):
    # Exactly what exp/metaworld/episode_runner.py sends via WS (server passes these kwargs).
    pol.on_episode_start(experiment="metaworld_mt50", task="reach-v3", episode_id=idx, episode_name="",
                         extra_metadata={"task_id": 41, "orig_init_state_idx": idx, "task_uid": f"u{idx}", "attempt": 1, "seed": 7})
    r = pol._request({})
    seeds[idx] = RequestRNG(r).seed_for()
    print(f"orig idx {idx}: request.original_init_idx={r.original_init_idx} task_name={r.task_name!r} explicit_seed={r.explicit_seed} noise_seed={seeds[idx]}")
print("distinct inits share noise:", seeds[3] == seeds[7])
