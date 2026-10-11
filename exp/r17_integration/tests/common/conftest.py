from dataclasses import replace
import uuid

import pytest

from exp.r17_integration.common.fake_adapter import fake_provider
from exp.r17_integration.common.provider import binding_for

collect_ignore = ["artifacts"]


@pytest.fixture
def provider():
    return fake_provider("test." + uuid.uuid4().hex)


def config_for(provider, *, k=4, commit=None, judge="always_hit"):
    from exp.r17_integration.common.config import (BackendConfig, CacheConfig, CheckpointConfig,
                                    GateConfig, JudgeConfig, KeyBuilderConfig, KeysConfig,
                                    SearchStrategyConfig, WritePolicyConfig)
    dims = dict(provider.model_spec.feature_spec.fields)
    keys = KeysConfig()
    for name, value in vars(keys).items():
        value.enabled = name in dims
    return CacheConfig(enabled=True, diagnostic_context=True, keys=keys, key_builder=KeyBuilderConfig(type="adapter_native"),
                       backend=BackendConfig(type="in_memory", vector_dims=dims),
                       write_policy=WritePolicyConfig(type="never"),
                       checkpoints={"cp1": CheckpointConfig(gate=GateConfig(type="always_search"),
                            judge=JudgeConfig(type=judge), search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", top_k=1))},
                       adapter_binding=binding_for(provider, "fake_mw.v1", num_steps=k, commit_horizon=commit))


def add_entry(storage, config, *, value=3.):
    import torch
    from exp.r17_integration.common.provider import artifact_binding
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID
    from exp.r17_integration.common.provider import validate_binding
    p, _ = validate_binding(config.adapter_binding)
    l = p.model_spec.layout
    entry = CacheEntry("donor", CheckpointID.CP1,
                       {name: torch.arange(dim, dtype=torch.float32) for name, dim in config.backend.vector_dims.items()},
                       CachePayload(torch.full((l.horizon, l.action_width), value)))
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    backend = InMemoryBackend(dict(config.backend.vector_dims))
    backend.insert(entry)
    storage._backend = backend
    storage._backend.artifact_meta = {"r17_binding": artifact_binding(config)}
    storage._backend.freeze()
    return entry


def with_recipe(provider, config, directory, *, corrector=False, steps=4, episodes=3):
    """Small sealed teacher library using P6's actual store/fit format."""
    from dataclasses import asdict
    import numpy as np
    from exp.r17_integration.common.config import RecipeConfig
    from exp.r17_integration.common.contracts import digest
    from exp.r17_integration.common.benchmarks import task_table_identity
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.search.store import write_store
    layout = provider.model_spec.layout
    rng = np.random.default_rng(42)
    n = steps*episodes
    rows = [dict(task_name="reach", task_uid=f"reach:{i // steps}", attempt=1,
                 decision_seq=i % steps, orig_init_idx=i // steps + 10, task_data_hash=f"data:{i // steps}",
                 campaign="p1-library", arm="teacher", bundle_generation="teacher.v1",
                 benchmark_protocol=config.adapter_binding["benchmark_protocol"], split="library", success=True) for i in range(n)]
    arrays = {"state": rng.normal(size=(n, max(layout.valid_state_indices)+1)).astype(np.float32),
              "normalized_actions": rng.normal(size=(n, layout.horizon, layout.action_width)).astype(np.float32)}
    arrays.update({"key_"+name: rng.normal(size=(n, width)).astype(np.float32)
                   for name, width in provider.model_spec.feature_spec.fields.items()})
    store = write_store(directory / "library", model_binding(provider.model_spec,
                        config.adapter_binding["benchmark_protocol"]), rows, arrays,
                        metadata={"key_fields": list(provider.model_spec.feature_spec.fields), "diagnostic": True})
    config.recipe = RecipeConfig(enabled=True, store_path=str(store.directory),
        store_digest=store.manifest["content_digest"], teacher_sampler_identity=digest(asdict(provider.model_spec.default_sampler)),
        ordered_tasks=["reach"], task_table_identity=task_table_identity(config.adapter_binding["benchmark_protocol"]),
        corrector=corrector, k=3, kref=2)
    return config
