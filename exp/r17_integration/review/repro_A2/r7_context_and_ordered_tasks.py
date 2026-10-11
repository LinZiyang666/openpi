"""Round-2 repro: (a) A7 missing context fails closed; (b) recipe task index comes from YAML
ordered_tasks while the query uses the benchmark's request.task_id (no cross-check)."""
import numpy as np
from exp.r17_integration.common.fake_adapter import fake_provider, fake_spec
from exp.r17_integration.common.provider import binding_for
from exp.r17_integration.common.serving import make_connection_factory
from exp.r17_integration.common.config import (BackendConfig, CacheConfig, CheckpointConfig, GateConfig, JudgeConfig,
    KeyBuilderConfig, KeysConfig, SearchStrategyConfig, WritePolicyConfig, build_shared_storage)
from exp.r17_integration.common.contracts import RequestContext
from exp.r17_integration.common.query_view import QueryView
from exp.r17_integration.common.layers import LibraryData, fit_library

spec = fake_spec(horizon=5)
p = fake_provider("fake.review2.v1", spec)
keys = KeysConfig()
for name, v in vars(keys).items():
    v.enabled = name in spec.feature_spec.fields
cfg = CacheConfig(enabled=True, keys=keys, key_builder=KeyBuilderConfig(type="adapter_native"),
    checkpoints={"cp1": CheckpointConfig(gate=GateConfig(type="always_skip"), judge=JudgeConfig(type="always_hit"),
                                         search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"))},
    backend=BackendConfig(type="in_memory", vector_dims=dict(spec.feature_spec.fields)),
    write_policy=WritePolicyConfig(type="never"), adapter_binding=binding_for(p, "fake_mw.v1"))
pol = make_connection_factory(p, cfg, build_shared_storage(cfg))(None, "default")
pol.on_episode_start(experiment="metaworld_mt50", task="reach-v3", episode_id=3, episode_name="",
                     extra_metadata={"task_id": 41, "orig_init_state_idx": 3, "task_uid": "u3", "attempt": 1, "seed": 7})
try:
    pol.infer({"state": np.zeros(4, np.float32)})
    print("(a) UNEXPECTED: stock-runner context accepted")
except ValueError as exc:
    print("(a) stock-runner context ->", exc, "| original_init_idx alias read:", pol._episode.get("original_init_idx"))

# (b) Two tasks; YAML ordered_tasks = ["reach-v3", "assembly-v3"] (not MT50 canonical order, where assembly-v3 is 0).
rng = np.random.default_rng(0)
layout = spec.layout
rows, feats, acts, tids, eps, steps = 0, [], [], [], [], []
for t, offset in ((0, +5.0), (1, -5.0)):            # library task index 0 = reach (+5), 1 = assembly (-5)
    for e in range(3):
        for s in range(4):
            feats.append(rng.normal(offset, .1, 4)); acts.append(np.full((5, layout.action_width), offset, np.float32))
            tids.append(t); eps.append(t * 3 + e); steps.append(s)
lib = LibraryData({"vision_0": np.array(feats, np.float32)}, np.array(feats, np.float32), np.array(acts),
                  np.array(tids), np.array(eps), np.array(steps))
recipe = fit_library(lib, layout=layout, task_count=2, corrector=False, guard=False, periodic_gap=None, k=2, kref=2, early=False)
# Benchmark request for assembly-v3 carries its canonical MT50 task_id 0.
req = RequestContext("c", "fake_mw.v1", "assembly-v3", 0, 1, task_id=0)
out = recipe.query(QueryView(req, {"vision_0": np.full(4, -5., np.float32)}, np.full(4, -5., np.float32), [True], [0], []))
print("(b) assembly-v3 request served from library task index", int(lib.task_id[out.rows[0]]),
      "(=YAML 'reach-v3'); returned action value", float(out.action[0, 0]))
