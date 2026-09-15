"""The online RIT feedback chain through a real CacheOrchestrator and the GR00T split."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.online_rit import (
    ContinuationFeedback,
    ContinuationSpec,
    OnlineRiskCurves,
    OnlineRitJudge,
    tier_specs,
)
from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.online_state import CurveRegistry
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import CacheEntry, CachePayload, QuerySpec
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID, groot_n15_schedule

from .conftest import ACTION_DIM, ACTION_HORIZON, STATE_WIDTH
from .test_groot_stage3 import _model_with_flow_head

SCHEDULE = groot_n15_schedule(4)
TIER_TS = [0.75, 0.5, 0.25]  # riskiest first for k=4
TIER_IDX = [3, 2, 1]
LIB_SHA = "ab" * 32


class _StateKeyBuilder:
    """Last-timestep robot state as the CP1 key (the stub's stage-1 state is [B, T, W])."""

    def __init__(self) -> None:
        self.cached_data: dict = {}

    def collect(self, checkpoint_id, **stage_outputs) -> None:
        self.cached_data = {"state": stage_outputs["stage1"].state[0, -1]}

    def build(self, checkpoint_id):
        key = F.normalize(self.cached_data["state"].float(), dim=0)
        return {"robot_state": key.cpu().float().contiguous()}

    def clear(self) -> None:
        self.cached_data = {}


class _Strategy:
    def __init__(self, storage, top_k: int = 1) -> None:
        self._storage, self._top_k = storage, top_k

    def search(self, ctx):
        return self._storage.search(
            QuerySpec(query_keys=ctx.query_keys, top_k=self._top_k, checkpoint_id=ctx.checkpoint_id)
        )


class _Policy:
    def __init__(self, model) -> None:
        self.model = model

    def apply_transforms(self, obs):
        return self.model.build_inputs()

    def unapply_transforms(self, action):
        return {"action.out": action["action"].numpy()}


def _payload(seed: int = 0) -> CachePayload:
    gen = torch.Generator().manual_seed(seed)
    inter = {SCHEDULE.snapshot_t(i): torch.randn(ACTION_HORIZON, ACTION_DIM, generator=gen) for i in range(1, 4)}
    return CachePayload(
        action_chunk=torch.randn(ACTION_HORIZON, ACTION_DIM, generator=gen),
        intermediates=inter,
        denoising_num_steps=4,
        schedule_id=SCHEDULE.schedule_id,
    )


def _spec(feedback_mode: str) -> ContinuationSpec:
    tiers = tier_specs(TIER_TS, SCHEDULE)
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    mask[:5] = True
    return ContinuationSpec(
        tiers=tiers,
        scales={t.index: torch.ones(ACTION_DIM) for t in tiers},
        masks={t.index: mask for t in tiers},
        h_exec=3,
        feedback_mode=feedback_mode,
        schedule=SCHEDULE,
    )


def _curves(update_enabled=True) -> OnlineRiskCurves:
    return OnlineRiskCurves(
        knots=[0.0, 0.5, 1.0], tier_indices=TIER_IDX, alpha=0.05, window=64, n_min=5, update_enabled=update_enabled
    )


def _warm_everywhere(c: OnlineRiskCurves) -> None:
    """Low disagreement at every score for every tier -> every cut sits at the first knot."""
    for n, s in enumerate(np.linspace(0, 1, 40)):
        c.update_batch(float(s), [ContinuationFeedback(t, 0.01, "shadow") for t in TIER_IDX], ("seed", n))


def _build(judge, *, with_entry: bool = True, model_steps: int = 4):
    model = _model_with_flow_head(model_steps)
    backend = InMemoryBackend({"robot_state": STATE_WIDTH})
    storage = CacheStorage(backend)
    if with_entry:
        state = model.build_inputs()["state"][0, -1]
        key = F.normalize(state.float(), dim=0).contiguous()
        storage.insert(
            CacheEntry(
                id="lib:0",
                checkpoint_id=CheckpointID.CP1,
                query_keys={"robot_state": key},
                payload=_payload(),
                trajectory_id="traj0",
            )
        )
    timer = SystemTimer(enabled=False)
    orch = CacheOrchestrator(
        storage,
        _StateKeyBuilder(),
        gates={CheckpointID.CP1: AlwaysSearchGate()},
        judges={CheckpointID.CP1: judge},
        search_strategies={CheckpointID.CP1: _Strategy(storage)},
        timer=timer,
    )
    runner = GrootStagedRunner(model, timer=timer, verify_upstream=False)
    interceptor = GrootCacheInterceptor(_Policy(model), runner, orchestrator=orch, timer=timer)
    return model, orch, interceptor


def _online_judge(feedback_mode: str, curves: OnlineRiskCurves, delta: float = 1.0):
    reg = CurveRegistry()
    key = reg.attach(yaml_id="arm", library_sha256=LIB_SHA, fingerprint="fp", factory=lambda: curves)
    return OnlineRitJudge(registry=reg, registry_key=key, spec=_spec(feedback_mode), delta=delta, yaml_id="arm"), reg, key


def _obs():
    return {"state.x": np.zeros((1, 3), dtype=np.float32)}


def test_inference_exception_marks_online_stream_invalid(monkeypatch):
    c = _curves()
    judge, reg, key = _online_judge("fm1", c)
    _, _, interceptor = _build(judge)
    def failed(*args, **kwargs):
        raise RuntimeError("side step failed")
    monkeypatch.setattr(interceptor._runner, "first_step_updates", failed)
    with pytest.raises(RuntimeError, match="side step"):
        interceptor.get_action(_obs())
    assert reg.is_invalid(key)


def test_host_benchmark_exercises_real_dispatch_registry_and_files(tmp_path):
    from exp.online_rit.bench_fb_cost import measure_host_paths

    model, orch, interceptor = _build(ThresholdJudge(cp1_threshold=.9))
    components = {k: getattr(orch, "_" + k) for k in ("storage", "key_builder", "judges", "search_strategies", "timer", "write_policy", "library_stats")}
    with interceptor._runner.session():
        stage1 = interceptor._runner.run_stage1(model.build_inputs())
    result = measure_host_paths(components, stage1, _spec("fm1"), [0., .5, 1.], LIB_SHA,
                                tmp_path, iters=2, task_key="t", theta=.5)
    assert set(result["dispatch_ms"]) == {"online", "threshold"}
    assert set(result["commit_ms"]) == {f"{m}:{n}" for m in ("learning", "frozen") for n in (0, 1, 3)}
    assert all(v > 0 for v in result["commit_ms"].values())
    assert all(v > 0 for v in result["snapshot_ms"].values())
    assert list(tmp_path.rglob("feedback.jsonl")) and list(tmp_path.rglob("state_latest.json"))


@pytest.mark.parametrize("mode,n_shadow", [("fm1", 2), ("fm0", 0)])
def test_warm_start_yields_executed_feedback_plus_side_evaluations(mode, n_shadow):
    c = _curves()
    _warm_everywhere(c)
    judge, reg, key = _online_judge(mode, c)
    model, orch, interceptor = _build(judge)
    orch.on_episode_start(task_key="t", episode_id="0", extra_metadata={"task_uid": "u", "attempt": 0})
    out = interceptor.get_action(_obs())
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "WARM_START" and meta["start_t"] == 0.75
    diag = meta["online_rit"]
    sources = sorted(fb["source"] for fb in diag["fb"])
    assert sources.count("executed") == 1 and sources.count("shadow") == n_shadow
    assert diag["fb_batch_size"] == n_shadow
    assert diag["learned"] is True and diag["rejected"] == 0
    assert diag["q_pre"]["3"] is not None and diag["verdict"] == "WARM_START"
    assert diag["task_uid"] == "u" and diag["decision_idx"] == 0
    assert reg.describe(key)["n_updates"] == 41
    # every feedback value is a finite disagreement in executed dims only
    assert all(math.isfinite(fb["d"]) for fb in diag["fb"])


@pytest.mark.parametrize("mode,n_shadow", [("fm1", 3), ("fm0", 0)])
def test_miss_with_candidate_side_evaluates_every_tier(mode, n_shadow):
    c = _curves()  # cold: every cut is +inf -> MISS with the candidate kept
    judge, reg, key = _online_judge(mode, c)
    model, orch, interceptor = _build(judge)
    orch.on_episode_start(task_key="t", episode_id="0")
    out = interceptor.get_action(_obs())
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "MISS" and meta["winner_id"] == "lib:0"
    diag = meta["online_rit"]
    assert len(diag["fb"]) == n_shadow and all(fb["source"] == "shadow" for fb in diag["fb"])
    assert diag["cuts"] == {"3": None, "2": None, "1": None}
    assert diag["learned"] is (n_shadow > 0)
    assert out["action.out"].shape == (ACTION_HORIZON, ACTION_DIM)


def test_miss_without_candidate_records_an_empty_row():
    c = _curves()
    judge, reg, key = _online_judge("fm1", c)
    model, orch, interceptor = _build(judge, with_entry=False)
    orch.on_episode_start(task_key="t", episode_id="0")
    out = interceptor.get_action(_obs())
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "MISS" and meta["winner_id"] is None
    assert meta["online_rit"]["fb"] == [] and meta["online_rit"]["candidate"] is False
    assert reg.describe(key)["n_updates"] == 0


def test_legacy_judge_has_no_online_rit_key():
    model, orch, interceptor = _build(ThresholdJudge(cp1_threshold=2.0))
    orch.on_episode_start(task_key="t", episode_id="0")
    out = interceptor.get_action(_obs())
    assert "online_rit" not in out["__hit_meta__"]
    assert out["__hit_meta__"]["hit_type"] == "MISS"


def test_frozen_judge_records_without_learning():
    c = _curves(update_enabled=False)
    judge, reg, key = _online_judge("fm1", c)
    model, orch, interceptor = _build(judge)
    orch.on_episode_start(task_key="t", episode_id="0")
    sha = c.learning_state_sha256()
    out = interceptor.get_action(_obs())
    diag = out["__hit_meta__"]["online_rit"]
    assert len(diag["fb"]) == 3 and diag["learned"] is False
    assert c.learning_state_sha256() == sha and c.n_observed == 3


def test_executed_disagreement_equals_the_hand_computed_value():
    c = _curves()
    _warm_everywhere(c)
    judge, reg, key = _online_judge("fm0", c)
    model, orch, interceptor = _build(judge)
    orch.on_episode_start(task_key="t", episode_id="0")
    payload = _payload()
    runner = interceptor._runner  # noqa: SLF001 - test seam
    with runner.session():
        stage2 = runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))
        out = runner.run_stage3_from(stage2, payload.intermediates[0.75], 0.75, schedule=SCHEDULE, capture_first_step=True)
    from openpi.cache.components.online_rit import continuation_disagreement, reference_update

    u_now = (out.first_step_x[0] - out.first_step_input[0]) * 4
    u_ref = reference_update(payload, 0.75, SCHEDULE)
    spec = _spec("fm0")
    expected = continuation_disagreement(u_now, u_ref, spec.scales[3], spec.masks[3], 3)
    diag = interceptor.get_action(_obs())["__hit_meta__"]["online_rit"]
    assert diag["fb"][0]["d"] == pytest.approx(expected, rel=1e-5)


def test_non_finite_feedback_marks_the_stream_invalid_but_still_serves():
    c = _curves()
    _warm_everywhere(c)
    judge, reg, key = _online_judge("fm1", c)
    model, orch, interceptor = _build(judge)
    orch.on_episode_start(task_key="t", episode_id="0")
    runner = interceptor._runner  # noqa: SLF001 - test seam
    real = runner.first_step_updates

    def poisoned(stage2, snaps, *, schedule):
        pairs = real(stage2, snaps, schedule=schedule)
        return [(x_in, x_out * float("nan")) for x_in, x_out in pairs]

    runner.first_step_updates = poisoned
    out = interceptor.get_action(_obs())
    diag = out["__hit_meta__"]["online_rit"]
    assert out["action.out"].shape == (ACTION_HORIZON, ACTION_DIM)
    assert diag["flow_invalid"] is True and diag["rejected"] == 2 and len(diag["invalid_reasons"]) == 2
    assert reg.describe(key)["flow_invalid"] is True
    # the finite executed observation was still learned, the stream is flagged
    assert len(diag["fb"]) == 1 and diag["fb"][0]["source"] == "executed"


def test_task_end_flushes_a_snapshot(tmp_path):
    reg = CurveRegistry(state_log_root=str(tmp_path))
    c = _curves()
    key = reg.attach(yaml_id="arm", library_sha256=LIB_SHA, fingerprint="fp", factory=lambda: c)
    judge = OnlineRitJudge(registry=reg, registry_key=key, spec=_spec("fm1"), delta=1.0, yaml_id="arm")
    model, orch, interceptor = _build(judge)
    interceptor.on_task_begin()
    orch.on_episode_start(task_key="t", episode_id="0")
    interceptor.get_action(_obs())
    interceptor.on_task_end()
    files = list((tmp_path / f"arm__{LIB_SHA[:12]}").rglob("state_*_task_end.json"))
    assert len(files) == 1
