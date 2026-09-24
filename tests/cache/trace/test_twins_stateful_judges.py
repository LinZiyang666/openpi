"""Stateful judges under the twin set (plan §4.1, §4.2, §12-B).

* CRD: the twin runs a complete propose / commit cycle on every step (three
  consecutive steps across verdict transitions never trip the
  "uncommitted proposal" guard), while the real CRD only ever commits on the
  real path; a twin WARM whose snapshot check fails is committed as a legal
  WARM -> MISS downgrade.
* MlpRouter: ``strip_twin_config`` removes the native ``dump_dir`` so the
  twin router never writes a shard; the real router's output is unchanged.
* Composite-style state (rolling window) is exercised through the CRD debt
  ledger: the real ledger with twins equals the real ledger without.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.crd_judge import CumulativeRiskJudge
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import HitType
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.orchestrator import CacheOrchestrator, TwinSet
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.types import PI05_V1, CheckpointID
from tests.cache.conftest import TestStorageSearchStrategy, insert_entry, make_stage1
from tests.cache.test_crd_judge import write_crd

DIM = 32


def _library(storage, *, with_snapshot: bool):
    g = torch.Generator().manual_seed(0)
    inter = {0.3: torch.randn(50, 2, generator=g)} if with_snapshot else {0.7: torch.randn(50, 2, generator=g)}
    for i in range(3):
        state = torch.nn.functional.normalize(torch.randn(1, DIM, generator=g), dim=1)
        payload = CachePayload(
            action_chunk=torch.randn(50, 2, generator=g), intermediates=inter,
            denoising_num_steps=10, schedule_id=PI05_V1.schedule_id,
        )
        insert_entry(storage, CheckpointID.CP1, state, payload, entry_id=f"e{i}")


def _crd_orchestrator(tmp_path, shared, *, twins: bool):
    real_storage = shared.per_connection_facade()
    # delta=0.85 sits between the warm (0.8) and full (0.9) risk of the top
    # score bin, so a near-duplicate query is a WARM_START proposal.
    path = write_crd(tmp_path, name=f"crd_{twins}.npz", delta=0.85)
    judge = CumulativeRiskJudge(path)
    strategy = TestStorageSearchStrategy(real_storage, top_k=1)
    twin_set = None
    if twins:
        twin_storage = shared.per_connection_facade()
        twin_set = TwinSet(
            key_builder=PlaceholderKeyBuilder(), gates={CheckpointID.CP1: AlwaysSearchGate()},
            judges={CheckpointID.CP1: CumulativeRiskJudge(write_crd(tmp_path, name="crd_twin.npz", delta=0.85))},
            strategies={CheckpointID.CP1: TestStorageSearchStrategy(twin_storage, top_k=1)},
            storage=twin_storage, timer=SystemTimer(enabled=False),
        )
    orch = CacheOrchestrator(
        real_storage, PlaceholderKeyBuilder(),
        gates={CheckpointID.CP1: AlwaysSearchGate()}, judges={CheckpointID.CP1: judge},
        search_strategies={CheckpointID.CP1: strategy}, timer=SystemTimer(enabled=False),
        trace_twins=twin_set,
    )
    return orch, judge


def _query(seed, base=None, dist=0.0):
    g = torch.Generator().manual_seed(seed)
    if base is None:
        return torch.nn.functional.normalize(torch.randn(1, DIM, generator=g), dim=1)
    delta = torch.randn(1, DIM, generator=g)
    delta = delta - (delta * base).sum() * base
    return base + dist * delta / delta.norm()


@pytest.mark.parametrize("with_snapshot", [True, False])
def test_crd_twin_commits_every_step_and_real_ledger_is_unchanged(tmp_path, with_snapshot):
    backend = InMemoryBackend({"robot_state": DIM})
    shared = CacheStorage(backend)
    _library(shared, with_snapshot=with_snapshot)
    backend.freeze()
    # The stored keys are normalized; queries at controlled cosine distance.
    e0 = backend._entries["e0"].query_keys["robot_state"][None, :]
    inputs = [_query(1, e0, 0.05), _query(2, e0, 2.0), _query(3, e0, 0.05), _query(4, e0, 2.0)] * 3

    def drive(orch):
        orch.on_task_begin()
        orch.on_episode_start(task_key="", episode_id="1", extra_metadata={"task_id": 0})
        out = []
        for x in inputs:
            try:
                r = orch.check(CheckpointID.CP1, trace=True, fetch_top1=True, stage1=make_stage1(x))
                out.append((r.hit_type, r.start_t, r.entry_id, (r.factor_outputs or {}).get("crd")))
            except ValueError as exc:
                # Real WARM with a missing snapshot raises (HEAD behaviour);
                # the CRD proposal must not be left dangling for the next step.
                out.append(("raise", str(exc)[:30]))
                orch._judges[CheckpointID.CP1].commit_verdict(CheckpointID.CP1, hit_type=HitType.MISS)
            orch.broadcast_action(torch.zeros(50, 2))
            orch.clear()
        orch.on_episode_end()
        return out

    orch_a, judge_a = _crd_orchestrator(tmp_path, shared, twins=False)
    orch_b, judge_b = _crd_orchestrator(tmp_path, shared, twins=True)
    out_a = drive(orch_a)
    out_b = drive(orch_b)
    assert out_a == out_b
    assert judge_a.state == judge_b.state
    assert judge_a._pending is None and judge_b._pending is None
    twin_judge = orch_b._twins.judges[CheckpointID.CP1]
    assert twin_judge._pending is None  # every twin proposal was committed
    if not with_snapshot:
        # Twin proposed WARM at least once and was downgraded (legal WARM->MISS).
        assert any(
            r for r in out_b if r[0] == "raise"
        ), "fixture should have produced a real WARM raise"


def test_strip_twin_config_removes_router_dump_dir(tmp_path):
    from openpi.cache.config import (
        BackendConfig,
        CacheConfig,
        CheckpointConfig,
        JudgeConfig,
        KeyBuilderConfig,
        KeyFieldConfig,
        KeysConfig,
        SearchStrategyConfig,
        WritePolicyConfig,
        build_per_connection_components,
        build_shared_storage,
        validate_cache_config,
    )
    from openpi.cache.trace.runtime import build_trace_twins, strip_twin_config

    dump_dir = tmp_path / "router_dump"
    config = CacheConfig(
        enabled=True,
        keys=KeysConfig(
            vision_0=KeyFieldConfig(enabled=False), vision_1=KeyFieldConfig(enabled=False),
            vision_2=KeyFieldConfig(enabled=False), prompt_emb=KeyFieldConfig(enabled=False),
            robot_state=KeyFieldConfig(enabled=True, weight=1.0),
        ),
        key_builder=KeyBuilderConfig(type="placeholder"),
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                judge=JudgeConfig(
                    type="mlp_router", arms="tc", constant_arm="teacher",
                    feature_fields=["robot_state"], mode="argmax", dump_dir=str(dump_dir),
                ),
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"),
            ),
        },
        write_policy=WritePolicyConfig(type="never"),
    )
    validate_cache_config(config, check_files=False)
    stripped = strip_twin_config(config)
    assert stripped.checkpoints["cp1"].judge.dump_dir is None
    assert config.checkpoints["cp1"].judge.dump_dir == str(dump_dir)
    shared = build_shared_storage(config)
    real = build_per_connection_components(config, shared, quiet=True)
    twins = build_trace_twins(config, shared, real_components=real, yaml_id=None)
    real_judge = real["judges"][CheckpointID.CP1]
    twin_judge = twins.judges[CheckpointID.CP1]
    assert type(real_judge) is type(twin_judge)
    assert real_judge is not twin_judge
    assert getattr(real_judge, "_dump_dir", None) or getattr(real_judge, "dump_dir", None)
    assert not (getattr(twin_judge, "_dump_dir", None) or getattr(twin_judge, "dump_dir", None))
    assert twins.storage is not real["storage"]
    assert twins.timer._enabled is False
