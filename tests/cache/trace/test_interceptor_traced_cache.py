"""``_infer_traced`` with a library and twins (plan §5, §12-C/E).

Verdict x executed-arm table, warm variants from the twin top-1, the
``warm_exec`` fallback when the real WARM is not among the precomputed
variants, MISS bookkeeping filtered to the model's default snapshot set, and
the CP3 twin running on FULL_HIT steps where the real CP3 handler does not.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.orchestrator import CacheOrchestrator, TwinSet
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
from openpi.cache.trace.types import TraceRuntime
from openpi.cache.types import PI05_V1, CheckpointID
from tests.cache.conftest import TestStorageSearchStrategy, insert_entry
from tests.cache.trace.conftest import D, H, TraceFakeModel, TraceFakePolicy, make_obs, make_plan


class CountingStrategy(TestStorageSearchStrategy):
    def __init__(self, storage, *, top_k=1):
        super().__init__(storage, top_k=top_k)
        self.searches = 0

    def search(self, ctx):
        self.searches += 1
        return super().search(ctx)


def _build(out_dir, *, warm_tiers, plan_tiers, with_cp3=False, **plan_kw):
    backend = InMemoryBackend({"robot_state": D})
    shared = CacheStorage(backend)
    real_storage = shared.per_connection_facade()
    twin_storage = shared.per_connection_facade()
    judge_kw = dict(cp1_threshold=0.9, cp3_threshold=0.95, warm_tiers=warm_tiers)
    cps = [CheckpointID.CP1] + ([CheckpointID.CP3] if with_cp3 else [])
    real_strats = {cp: CountingStrategy(real_storage) for cp in cps}
    twin_strats = {cp: CountingStrategy(twin_storage) for cp in cps}
    twins = TwinSet(
        key_builder=PlaceholderKeyBuilder(),
        gates={cp: AlwaysSearchGate() for cp in cps},
        judges={cp: ThresholdJudge(**judge_kw) for cp in cps},
        strategies=twin_strats, storage=twin_storage, timer=SystemTimer(enabled=False),
    )
    orch = CacheOrchestrator(
        real_storage, PlaceholderKeyBuilder(),
        gates={cp: AlwaysSearchGate() for cp in cps},
        judges={cp: ThresholdJudge(**judge_kw) for cp in cps},
        search_strategies=real_strats, timer=SystemTimer(enabled=False), trace_twins=twins,
    )
    plan = make_plan(checkpoint="CP1", warm_tiers=plan_tiers, **plan_kw)
    writer = TraceWriter(str(out_dir), queue_steps=8)
    rt = TraceRuntime(plan=plan, sink=H5TraceSink(out_dir, plan=plan, writer=writer), twins=twins)
    model = TraceFakeModel(seed=5)
    it = InferenceInterceptor(TraceFakePolicy(model), timer=SystemTimer(enabled=False), eager=True,
                              orchestrator=orch, trace=rt)
    return it, orch, model, backend, shared, writer, real_strats, twin_strats


def _entry(shared, backend, state_dir, *, seed=0, snapshots=(0.7, 0.5, 0.3), entry_id="lib0"):
    g = torch.Generator().manual_seed(seed)
    key = torch.nn.functional.normalize(state_dir, dim=1)
    payload = CachePayload(
        action_chunk=torch.randn(H, D, generator=g),
        intermediates={t: torch.randn(H, D, generator=g) for t in snapshots},
        denoising_num_steps=10, schedule_id=PI05_V1.schedule_id,
    )
    insert_entry(shared, CheckpointID.CP1, key, payload, entry_id=entry_id)
    return payload


def _run_one(it, writer, obs_seed=0):
    it.on_task_begin()
    it.on_episode_start("exp", "task", 1, "ep")
    out = it.infer(make_obs(obs_seed))
    it.on_episode_end(True)
    it.on_task_end()
    assert writer.drain(timeout=10).ok
    return out


def test_miss_with_empty_library(out_dir):
    it, orch, model, backend, shared, writer, *_ = _build(
        out_dir, warm_tiers=[{"threshold": 0.5, "start_t": 0.5}], plan_tiers=(5,)
    )
    out = _run_one(it, writer)
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "MISS" and meta["trace"]["executed_arm"] == "full_inference"
    assert meta["trace"]["tier_status"] == {"warm_05": "no_top1"}
    with h5py.File(out_dir / "exp" / "ep.h5", "r") as f:
        tg = f["step_0000"]["trace"]
        assert "search" in tg and len(tg["search"]["twin_topk_ids"]) == 0
        assert set(tg["actions"].keys()) == {"full_inference", "executed"}
        assert "query_keys" in tg and tg["query_keys"]["robot_state"].shape == (D,)


def test_full_hit_sends_cached_chunk_and_records_all_variants(out_dir):
    it, orch, model, backend, shared, writer, real_strats, twin_strats = _build(
        out_dir, warm_tiers=[{"threshold": 0.5, "start_t": 0.5}], plan_tiers=(5,)
    )
    payload = _entry(shared, backend, model.state)
    backend.freeze()
    out = _run_one(it, writer)
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "FULL_HIT" and meta["winner_id"] == "lib0"
    assert meta["trace"]["executed_arm"] == "full_hit"
    np.testing.assert_allclose(out["actions"], payload.action_chunk.numpy())
    # The full inference and the warm tier ran anyway.
    kinds = [name for name, _ in model.calls]
    assert kinds.count("stage3") == 1 and kinds.count("stage3_from") == 1
    assert real_strats[CheckpointID.CP1].searches == 1 and twin_strats[CheckpointID.CP1].searches == 1
    # FULL_HIT bookkeeping: broadcast + buffer without intermediates (legacy early-return shape).
    assert len(orch._episode_steps) == 0  # write policy None -> buffers cleared at episode end
    with h5py.File(out_dir / "exp" / "ep.h5", "r") as f:
        ag = f["step_0000"]["trace"]["actions"]
        assert set(ag.keys()) == {"full_hit", "warm_05", "full_inference", "executed"}
        np.testing.assert_allclose(ag["full_hit"][...], payload.action_chunk.numpy())
        np.testing.assert_allclose(ag["executed"][...], payload.action_chunk.numpy())
        assert not np.allclose(ag["full_inference"][...], ag["executed"][...])
        assert f["step_0000"]["trace"].attrs["executed_arm"] == "full_hit"


def _warm_query(model, base_key, cos=0.7):
    # Build a model state whose normalized key has cosine ``cos`` with base_key.
    g = torch.Generator().manual_seed(11)
    delta = torch.randn(1, D, generator=g)
    delta = delta - (delta * base_key).sum() * base_key
    delta = delta / delta.norm()
    model.state = cos * base_key + (1 - cos**2) ** 0.5 * delta


def test_warm_start_executes_matching_variant(out_dir):
    it, orch, model, backend, shared, writer, *_ = _build(
        out_dir, warm_tiers=[{"threshold": 0.5, "start_t": 0.5}], plan_tiers=(5,)
    )
    payload = _entry(shared, backend, model.state)
    backend.freeze()
    base_key = torch.nn.functional.normalize(model.state, dim=1)
    _warm_query(model, base_key)
    out = _run_one(it, writer)
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "WARM_START" and meta["start_t"] == 0.5
    assert meta["trace"]["executed_arm"] == "warm_05"
    assert meta["trace"]["tier_status"] == {"warm_05": "ok"}
    expected = model.run_stage3_from(None, payload.intermediates[0.5][None], 0.5, num_steps=10)
    np.testing.assert_allclose(out["actions"], expected.action_chunk[0].numpy())


def test_warm_exec_when_tier_not_enumerated(out_dir):
    # The plan enumerates nothing (as if the tier came from elsewhere) but the
    # real judge still returns WARM_START: the executed arm must be warm_exec.
    it, orch, model, backend, shared, writer, *_ = _build(
        out_dir, warm_tiers=[{"threshold": 0.5, "start_t": 0.5}], plan_tiers=()
    )
    payload = _entry(shared, backend, model.state)
    backend.freeze()
    _warm_query(model, torch.nn.functional.normalize(model.state, dim=1))
    out = _run_one(it, writer)
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "WARM_START"
    assert meta["trace"]["executed_arm"] == "warm_exec"
    assert meta["trace"]["tier_status"]["warm_exec"] == "warm_exec:tier_not_enumerated"
    expected = model.run_stage3_from(None, payload.intermediates[0.5][None], 0.5, num_steps=10)
    np.testing.assert_allclose(out["actions"], expected.action_chunk[0].numpy())
    with h5py.File(out_dir / "exp" / "ep.h5", "r") as f:
        assert "warm_exec" in f["step_0000"]["trace"]["actions"]


def test_missing_snapshot_tier_is_skipped_not_fatal(out_dir):
    it, orch, model, backend, shared, writer, *_ = _build(
        out_dir, warm_tiers=[{"threshold": 0.95, "start_t": 0.3}], plan_tiers=(5, 7)
    )
    _entry(shared, backend, model.state, snapshots=(0.3,))
    backend.freeze()
    out = _run_one(it, writer)
    status = out["__hit_meta__"]["trace"]["tier_status"]
    assert status["warm_05"].startswith("no_snapshot:")
    assert status["warm_07"] == "ok"


def test_miss_bookkeeping_filters_build_snapshots_to_model_default(out_dir):
    from openpi.cache.components.write_policy import AlwaysWritePolicy

    it, orch, model, backend, shared, writer, *_ = _build(
        out_dir, warm_tiers=None, plan_tiers=(), record_noise_actions=True,
        save_timesteps=PI05_V1.timesteps, fail_loud=True,
    )
    orch._write_policy = AlwaysWritePolicy()
    it.on_task_begin()
    it.on_episode_start("exp", "task", 1, "ep")
    it.infer(make_obs())
    step = orch._episode_steps[-1]
    assert sorted(step.intermediates) == sorted((0.7, 0.5, 0.3))
    assert step.denoising_num_steps == 10 and step.schedule_id == PI05_V1.schedule_id
    it.on_episode_end(True)
    it.on_task_end()
    assert writer.drain(timeout=10).ok
    with h5py.File(out_dir / "exp" / "ep.h5", "r") as f:
        g = f["step_0000"]
        assert all(f"noise_action_{i}" in g for i in range(10))


def test_cp3_twin_runs_on_full_hit_where_real_does_not(out_dir):
    it, orch, model, backend, shared, writer, real_strats, twin_strats = _build(
        out_dir, warm_tiers=None, plan_tiers=(), with_cp3=True
    )
    _entry(shared, backend, model.state)
    backend.freeze()
    out = _run_one(it, writer)
    assert out["__hit_meta__"]["hit_type"] == "FULL_HIT"
    assert real_strats[CheckpointID.CP3].searches == 0
    assert twin_strats[CheckpointID.CP3].searches == 1
    with h5py.File(out_dir / "exp" / "ep.h5", "r") as f:
        assert "cp3_twin_json" in f["step_0000"]["trace"].attrs
    # MISS step: both real and twin CP3 run exactly once.
    model.state = torch.randn(1, D)
    it.on_task_begin(); it.on_episode_start("exp", "task", 2, "ep2")
    out = it.infer(make_obs(1))
    it.on_episode_end(True); it.on_task_end()
    assert out["__hit_meta__"]["hit_type"] == "MISS"
    assert real_strats[CheckpointID.CP3].searches == 1
    assert twin_strats[CheckpointID.CP3].searches == 2
    assert writer.drain(timeout=10).ok
