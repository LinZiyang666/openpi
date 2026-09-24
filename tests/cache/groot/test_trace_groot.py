"""``GrootCacheInterceptor`` under the trace runtime (plan §9, §12-I).

A real ``CacheOrchestrator`` with a twin component set drives the stub flow
head through every verdict: the served action equals the legacy path's under
the same seed (MISS: same global-RNG noise; WARM_START: same resumed loop;
FULL_HIT: the cached chunk and the global RNG untouched), the H5 record holds
every variant, the online RIT feedback is closed for the real *and* the twin
judge with the same numbers as the legacy path, ``warm_exec`` covers a real
verdict outside the enumerated tiers, the CP2-only cycle traces without a
CP1 check, and the coordinator path batches two connections' variants.
"""

from __future__ import annotations

import json
import threading

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import HitType, ThresholdJudge
from openpi.cache.components.search_strategy import WeightedRrfKnnStrategy
from openpi.cache.groot.batcher import GrootStageBatcher
from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.orchestrator import CacheOrchestrator, TwinSet
from openpi.cache.storage_types import CacheEntry, CachePayload, QuerySpec
from openpi.cache.timing import SystemTimer
from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
from openpi.cache.trace.types import TracePlan, TraceRuntime
from openpi.cache.types import VISION_0, VISION_1, VISION_2, CheckpointID, groot_n15_schedule
from openpi.serving.batching_core import BatchingCore

from .conftest import ACTION_DIM, ACTION_HORIZON, STATE_VALID, STATE_WIDTH
from .test_groot_stage3 import _model_with_flow_head
from .test_online_rit_interceptor import _curves, _online_judge, _warm_everywhere

N_STEPS = 4
SCHEDULE = groot_n15_schedule(N_STEPS)
CAMS = (VISION_0, VISION_1, VISION_2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Policy:
    def __init__(self, model) -> None:
        self.model = model

    def apply_transforms(self, obs):
        return self.model.build_inputs()

    def unapply_transforms(self, action):
        return {"action.out": action["action"].numpy()}


class _StateKeyBuilder:
    """Last-timestep state as the key, at CP1 (stage1) or CP2 (stage2)."""

    vision_fields = CAMS

    def __init__(self) -> None:
        self.cached_data: dict = {}
        self.episodes = 0

    def collect(self, checkpoint_id, **stage_outputs) -> None:
        if "stage1" in stage_outputs:
            self.cached_data = {"state": stage_outputs["stage1"].state[0, -1]}
        else:
            self.cached_data = {"state": stage_outputs["stage2"].action_inputs["state"][0, -1]}

    def build(self, checkpoint_id):
        key = F.normalize(self.cached_data["state"].float(), dim=0)
        return {"robot_state": key.cpu().float().contiguous()}

    def clear(self) -> None:
        self.cached_data = {}

    def on_episode_start(self, *a, **k) -> None:
        self.episodes += 1


class _Strategy:
    def __init__(self, storage, top_k: int = 1) -> None:
        self._storage, self._top_k = storage, top_k

    def search(self, ctx):
        return self._storage.search(
            QuerySpec(query_keys=ctx.query_keys, top_k=self._top_k, checkpoint_id=ctx.checkpoint_id)
        )


def _payload(seed: int = 0) -> CachePayload:
    gen = torch.Generator().manual_seed(seed)
    inter = {
        SCHEDULE.snapshot_t(i): torch.randn(ACTION_HORIZON, ACTION_DIM, generator=gen)
        for i in range(1, N_STEPS)
    }
    return CachePayload(
        action_chunk=torch.randn(ACTION_HORIZON, ACTION_DIM, generator=gen),
        intermediates=inter,
        denoising_num_steps=N_STEPS,
        schedule_id=SCHEDULE.schedule_id,
    )


def _library(model, *, entries=("lib:0",), cp=CheckpointID.CP1):
    backend = InMemoryBackend({"robot_state": STATE_WIDTH})
    state = model.build_inputs()["state"][0, -1]
    key = F.normalize(state.float(), dim=0).contiguous()
    for i, eid in enumerate(entries):
        CacheStorage(backend).insert(
            CacheEntry(
                id=eid,
                checkpoint_id=cp,
                query_keys={"robot_state": key},
                payload=_payload(i),
                trajectory_id=f"traj{i}",
            )
        )
    return backend


def _components(backend, judge, *, cp=CheckpointID.CP1, strategy=_Strategy):
    storage = CacheStorage(backend)
    return dict(
        storage=storage,
        key_builder=_StateKeyBuilder(),
        gates={cp: AlwaysSearchGate()},
        judges={cp: judge},
        strategies={cp: strategy(storage)},
    )


def _orchestrator(backend, real_judge, twin_judge, *, cp=CheckpointID.CP1, strategy=_Strategy):
    real = _components(backend, real_judge, cp=cp, strategy=strategy)
    twin = _components(backend, twin_judge, cp=cp, strategy=strategy)
    timer = SystemTimer(enabled=False)
    return CacheOrchestrator(
        real["storage"],
        real["key_builder"],
        gates=real["gates"],
        judges=real["judges"],
        search_strategies=real["strategies"],
        timer=timer,
        trace_twins=TwinSet(
            key_builder=twin["key_builder"],
            gates=twin["gates"],
            judges=twin["judges"],
            strategies=twin["strategies"],
            storage=twin["storage"],
            timer=SystemTimer(enabled=False),
        ),
    ), timer


def _plan(**overrides) -> TracePlan:
    base = dict(
        model="groot_n15",
        schedule=SCHEDULE,
        checkpoint="CP1",
        warm_tiers=(1, 2, 3),
        record_noise_actions=True,
        save_timesteps=SCHEDULE.timesteps,
        record_prefix_tokens=True,
        record_raw_images=True,
        record_model_images=False,
        record_query_keys=True,
        record_search=True,
        record_tokenized_prompt=True,
        raw_image_keys=(),
        rng_isolation="verdict_aware",
        sidecar_jsonl=True,
        fail_loud=True,
    )
    base.update(overrides)
    return TracePlan(**base)


def _runtime(out_dir, **overrides):
    plan = _plan(**overrides)
    writer = TraceWriter(str(out_dir), queue_steps=8)
    return TraceRuntime(plan=plan, sink=H5TraceSink(out_dir, plan=plan, writer=writer)), writer


def _obs():
    return {
        "video.image": np.full((1, 6, 6, 3), 7, dtype=np.uint8),
        "video.wrist": np.full((1, 6, 6, 3), 9, dtype=np.uint8),
        "state.a": np.array([[0.1, 0.2]], dtype=np.float32),
        "state.b": np.array([[0.3]], dtype=np.float32),
        "annotation.task": np.asarray(["open the drawer"]),
    }


def _traced(model, orch, timer, rt, **kw):
    runner = GrootStagedRunner(model, timer=timer, verify_upstream=False)
    return GrootCacheInterceptor(_Policy(model), runner, orchestrator=orch, timer=timer, trace=rt, **kw)


def _legacy(model, orch, timer):
    runner = GrootStagedRunner(model, timer=timer, verify_upstream=False)
    return GrootCacheInterceptor(_Policy(model), runner, orchestrator=orch, timer=timer)


def _episode(it, *, name="ep1", uid="u1"):
    it.on_task_begin()
    it.on_episode_start("exp", "task", 1, name, {"task_uid": uid, "attempt": 1})


def _finish(it, writer):
    it.on_episode_end(True)
    it.on_task_end()
    report = writer.drain(timeout=10)
    assert report.ok, report


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_trace_only_arguments_are_refused_without_trace():
    model = _model_with_flow_head(N_STEPS)
    runner = GrootStagedRunner(model, verify_upstream=False)
    with pytest.raises(ValueError, match="trace-mode"):
        GrootCacheInterceptor(_Policy(model), runner, coordinator=object())
    with pytest.raises(ValueError, match="trace-mode"):
        GrootCacheInterceptor(_Policy(model), runner, model_lock=threading.Lock())


def test_vision_fields_must_match_the_configured_builder(tmp_path):
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    orch, timer = _orchestrator(backend, ThresholdJudge(cp1_threshold=2.0), ThresholdJudge(cp1_threshold=2.0))
    rt, _ = _runtime(tmp_path)
    with pytest.raises(ValueError, match="disagree"):
        _traced(model, orch, timer, rt, trace_vision_fields=(VISION_0, VISION_1))
    it = _traced(model, orch, timer, rt)
    assert it._trace_vision_fields == CAMS  # noqa: SLF001
    with pytest.raises(ValueError, match="required"):
        GrootCacheInterceptor(
            _Policy(model), GrootStagedRunner(model, verify_upstream=False), trace=rt
        )


def test_trace_none_never_reaches_the_traced_path(monkeypatch):
    model = _model_with_flow_head(N_STEPS)
    it = _legacy(model, None, SystemTimer(enabled=False))
    monkeypatch.setattr(it, "_get_action_traced", lambda obs: (_ for _ in ()).throw(AssertionError()))
    assert it.get_action(_obs())["action.out"].shape == (ACTION_HORIZON, ACTION_DIM)


# ---------------------------------------------------------------------------
# No library: the build form
# ---------------------------------------------------------------------------


def test_no_library_build_records_full_inference_and_raw_observation(tmp_path):
    model = _model_with_flow_head(N_STEPS)
    rt, writer = _runtime(tmp_path, checkpoint=None, warm_tiers=())
    timer = SystemTimer(enabled=False)
    it = _traced(model, None, timer, rt, trace_vision_fields=CAMS)
    _episode(it)
    torch.manual_seed(21)
    out = it.get_action(_obs())
    # sampling parity: upstream's own draw under the same seed
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage1 = runner.run_stage1(model.build_inputs())
        stage2 = runner.run_stage2_llm(stage1)
    torch.manual_seed(21)
    with runner.session():
        expected = runner.run_stage3(stage2, noise=None).action_pred[0]
    np.testing.assert_array_equal(out["action.out"], expected.numpy())
    assert out["__hit_meta__"]["hit_type"] == "MISS"
    assert out["__hit_meta__"]["trace"]["executed_arm"] == "full_inference"
    _finish(it, writer)

    with h5py.File(tmp_path / "exp" / "ep1.h5", "r") as f:
        assert f.attrs["denoise_schedule_id"] == SCHEDULE.schedule_id
        assert f.attrs["denoising_num_steps"] == N_STEPS
        assert f.attrs["trace_noise_actions_recorded"]
        g = f["step_0000"]
        assert [k for k in g if k.startswith("vision_")] == ["vision_0", "vision_1", "vision_2"]
        assert g["vision_0"].shape == (256, 8) and g["vision_0"].dtype == np.float16
        assert g["robot_state"].shape == (STATE_VALID,)
        np.testing.assert_array_equal(g["clean_action"][...], expected.numpy())
        # noise_action_0 is the loop's own first input under the same seed
        torch.manual_seed(21)
        with runner.session():
            z = runner.sample_noise(stage2)
        np.testing.assert_array_equal(g["noise_action_0"][...], z[0].numpy())
        assert all(f"noise_action_{i}" in g for i in range(1, N_STEPS))
        assert f"noise_action_{N_STEPS}" not in g
        tg = g["trace"]
        raw = tg["raw_images"]
        assert set(raw.keys()) == {"video.image", "video.wrist"}
        assert raw["video.image"].shape == (6, 6, 3) and int(raw["video.wrist"][0, 0, 0]) == 9
        np.testing.assert_allclose(tg["raw_state"][...], [0.1, 0.2, 0.3])
        assert json.loads(tg["raw_state"].attrs["layout_json"]) == [["state.a", 2], ["state.b", 1]]
        assert tg.attrs["prompt"] == "open the drawer"
        np.testing.assert_array_equal(
            tg["tokenized_prompt"][...], model.build_inputs()["eagle_input_ids"][0].numpy()
        )
        assert "search" not in tg
        assert set(tg["actions"].keys()) == {"full_inference", "executed"}


# ---------------------------------------------------------------------------
# Verdicts through a real orchestrator + twins
# ---------------------------------------------------------------------------


def test_miss_serves_the_legacy_noise_and_records_every_tier(tmp_path):
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    judge = lambda: ThresholdJudge(cp1_threshold=2.0)  # noqa: E731 - never hits
    orch_l, timer_l = _orchestrator(backend, judge(), judge())
    orch_t, timer_t = _orchestrator(backend, judge(), judge())
    legacy = _legacy(model, orch_l, timer_l)
    rt, writer = _runtime(tmp_path)
    traced = _traced(model, orch_t, timer_t, rt)
    for it in (legacy, traced):
        _episode(it)
    torch.manual_seed(5)
    a = legacy.get_action(_obs())
    torch.manual_seed(5)
    b = traced.get_action(_obs())
    np.testing.assert_array_equal(a["action.out"], b["action.out"])
    assert b["__hit_meta__"]["hit_type"] == "MISS" and b["__hit_meta__"]["winner_id"] is None
    tr = b["__hit_meta__"]["trace"]
    assert tr["executed_arm"] == "full_inference" and tr["top1_entry_id"] == "lib:0"
    assert tr["tier_status"] == {"warm_01": "ok", "warm_02": "ok", "warm_03": "ok"}
    _finish(traced, writer)
    with h5py.File(tmp_path / "exp" / "ep1.h5", "r") as f:
        tg = f["step_0000"]["trace"]
        assert set(tg["actions"].keys()) == {
            "full_hit", "warm_01", "warm_02", "warm_03", "full_inference", "executed"
        }
        np.testing.assert_array_equal(tg["actions"]["executed"][...], tg["actions"]["full_inference"][...])
        np.testing.assert_array_equal(tg["actions"]["full_hit"][...], _payload(0).action_chunk.numpy())
        # each warm tier equals the direct resumed loop from the library snapshot
        runner = GrootStagedRunner(model, verify_upstream=False)
        with runner.session():
            stage2 = runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))
        for idx in (1, 2, 3):
            t = SCHEDULE.snapshot_t(idx)
            with runner.session():
                ref = runner.run_stage3_from(stage2, _payload(0).intermediates[t], t, schedule=SCHEDULE)
            np.testing.assert_array_equal(tg["actions"][f"warm_0{idx}"][...], ref.action_pred[0].numpy())
        s = tg["search"]
        assert [x.decode() if isinstance(x, bytes) else x for x in s["twin_topk_ids"][...]] == ["lib:0"]
        assert s.attrs["gate_twin_should_search"]
        proxies = json.loads(tg.attrs["error_proxies_json"])
        assert {"l2_full_hit_vs_full", "l2_warm_01_vs_full", "l2_warm_03_vs_full"} <= set(proxies)
        assert json.loads(tg.attrs["warm_index_map_json"])["2"]["start_t"] == 0.5


def test_warm_start_verdict_serves_the_precomputed_tier(tmp_path):
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    judge = lambda: ThresholdJudge(cp1_threshold=2.0, warm_tiers=[{"threshold": 0.0, "start_t": 0.5}])  # noqa: E731
    orch_l, timer_l = _orchestrator(backend, judge(), judge())
    orch_t, timer_t = _orchestrator(backend, judge(), judge())
    legacy = _legacy(model, orch_l, timer_l)
    rt, writer = _runtime(tmp_path)
    traced = _traced(model, orch_t, timer_t, rt)
    for it in (legacy, traced):
        _episode(it)
    a = legacy.get_action(_obs())
    b = traced.get_action(_obs())
    np.testing.assert_array_equal(a["action.out"], b["action.out"])
    meta = b["__hit_meta__"]
    assert meta["hit_type"] == "WARM_START" and meta["start_t"] == 0.5 and meta["winner_id"] == "lib:0"
    assert meta["trace"]["executed_arm"] == "warm_02"
    assert "warm_exec" not in meta["trace"]["tier_status"]
    _finish(traced, writer)
    with h5py.File(tmp_path / "exp" / "ep1.h5", "r") as f:
        tg = f["step_0000"]["trace"]
        np.testing.assert_array_equal(tg["actions"]["executed"][...], tg["actions"]["warm_02"][...])
        assert tg.attrs["executed_arm"] == "warm_02"
        assert "warm_exec" not in tg["actions"]


def test_warm_exec_covers_a_tier_the_plan_did_not_enumerate(tmp_path):
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    judge = lambda: ThresholdJudge(cp1_threshold=2.0, warm_tiers=[{"threshold": 0.0, "start_t": 0.5}])  # noqa: E731
    orch_l, timer_l = _orchestrator(backend, judge(), judge())
    orch_t, timer_t = _orchestrator(backend, judge(), judge())
    legacy = _legacy(model, orch_l, timer_l)
    rt, writer = _runtime(tmp_path, warm_tiers=(1,))
    traced = _traced(model, orch_t, timer_t, rt)
    for it in (legacy, traced):
        _episode(it)
    a = legacy.get_action(_obs())
    b = traced.get_action(_obs())
    np.testing.assert_array_equal(a["action.out"], b["action.out"])
    tr = b["__hit_meta__"]["trace"]
    assert tr["executed_arm"] == "warm_exec"
    assert tr["tier_status"] == {"warm_01": "ok", "warm_exec": "warm_exec:tier_not_enumerated"}
    _finish(traced, writer)
    with h5py.File(tmp_path / "exp" / "ep1.h5", "r") as f:
        tg = f["step_0000"]["trace"]
        assert set(tg["actions"].keys()) == {"full_hit", "warm_01", "warm_exec", "full_inference", "executed"}
        np.testing.assert_array_equal(tg["actions"]["executed"][...], tg["actions"]["warm_exec"][...])


def test_full_hit_serves_the_cached_chunk_and_keeps_the_global_rng(tmp_path):
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    judge = lambda: ThresholdJudge(cp1_threshold=0.0)  # noqa: E731
    orch, timer = _orchestrator(backend, judge(), judge())
    rt, writer = _runtime(tmp_path)
    traced = _traced(model, orch, timer, rt)
    _episode(traced)
    # Materialise the stub head's lazy layer first: its one-time init draws
    # from the global RNG and would masquerade as a leak of the hit-step noise.
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        runner.run_stage3(runner.run_stage2_llm(runner.run_stage1(model.build_inputs())))
    torch.manual_seed(9)
    state = torch.get_rng_state()
    out = traced.get_action(_obs())
    assert torch.equal(torch.get_rng_state(), state)  # hit-step noise is private
    np.testing.assert_array_equal(out["action.out"], _payload(0).action_chunk.numpy())
    assert out["__hit_meta__"]["hit_type"] == "FULL_HIT"
    assert out["__hit_meta__"]["trace"]["executed_arm"] == "full_hit"
    # the full inference still ran and is recorded
    _finish(traced, writer)
    with h5py.File(tmp_path / "exp" / "ep1.h5", "r") as f:
        g = f["step_0000"]
        assert g["clean_action"].shape == (ACTION_HORIZON, ACTION_DIM)
        assert "noise_action_3" in g
        tg = g["trace"]
        np.testing.assert_array_equal(tg["actions"]["executed"][...], tg["actions"]["full_hit"][...])
        assert not np.array_equal(tg["actions"]["full_inference"][...], tg["actions"]["full_hit"][...])


def test_twin_state_is_isolated_from_the_real_components(tmp_path):
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    judge = lambda: ThresholdJudge(cp1_threshold=2.0)  # noqa: E731
    orch, timer = _orchestrator(backend, judge(), judge())
    rt, writer = _runtime(tmp_path)
    traced = _traced(model, orch, timer, rt)
    _episode(traced)
    traced.get_action(_obs())
    traced.get_action(_obs())
    _finish(traced, writer)
    real_kb = orch.key_builder
    twin_kb = orch.twin_key_builder
    assert real_kb is not twin_kb
    assert real_kb.episodes == twin_kb.episodes == 2  # task_begin (provisional) + episode_start
    assert real_kb.cached_data == {} and twin_kb.cached_data == {}


# ---------------------------------------------------------------------------
# Online RIT: real + twin feedback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode,n_shadow", [("fm1", 2), ("fm0", 0)])
def test_online_rit_feedback_closes_real_and_twin_streams(tmp_path, mode, n_shadow):
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)

    def judge():
        c = _curves()
        _warm_everywhere(c)
        return _online_judge(mode, c)

    real_l, reg_l, key_l = judge()
    real_t, reg_t, key_t = judge()
    twin_t, reg_tw, key_tw = judge()
    orch_l, timer_l = _orchestrator(backend, real_l, real_l)
    orch_t, timer_t = _orchestrator(backend, real_t, twin_t)
    legacy = _legacy(model, orch_l, timer_l)
    rt, writer = _runtime(tmp_path, warm_tiers=(3, 2, 1))
    traced = _traced(model, orch_t, timer_t, rt)
    for it in (legacy, traced):
        _episode(it, uid="u")
    a = legacy.get_action(_obs())
    b = traced.get_action(_obs())
    np.testing.assert_array_equal(a["action.out"], b["action.out"])
    da, db = a["__hit_meta__"]["online_rit"], b["__hit_meta__"]["online_rit"]
    assert da["verdict"] == db["verdict"] == "WARM_START" and db["start_t"] == 0.75
    assert sorted(fb["source"] for fb in db["fb"]) == sorted(fb["source"] for fb in da["fb"])
    assert db["fb_batch_size"] == n_shadow and db["rejected"] == 0 and db["learned"] is True
    # identical disagreement values: the executed pair is the served variant's
    # own first step, the shadows are the same helper call as the legacy path
    for fa, fb in zip(sorted(da["fb"], key=lambda x: x["tier"]), sorted(db["fb"], key=lambda x: x["tier"])):
        assert fa["tier"] == fb["tier"] and abs(fa["d"] - fb["d"]) < 1e-6
    assert reg_t.describe(key_t)["n_updates"] == reg_l.describe(key_l)["n_updates"] == 41
    # the twin stream learned too, from the reused variant captures (no extra batch)
    assert reg_tw.describe(key_tw)["n_updates"] == 41
    _finish(traced, writer)
    with h5py.File(tmp_path / "exp" / "ep1.h5", "r") as f:
        tg = f["step_0000"]["trace"]
        real_j = json.loads(tg.attrs["real_continuation_json"])
        twin_j = json.loads(tg.attrs["twin_continuation_json"])
        assert real_j["fb_batch_size"] == n_shadow
        assert twin_j["fb_batch_size"] == 0 and twin_j["verdict"] == "WARM_START"
        assert sorted(fb["source"] for fb in twin_j["fb"]) == sorted(fb["source"] for fb in real_j["fb"])
        for fr, ft in zip(sorted(real_j["fb"], key=lambda x: x["tier"]), sorted(twin_j["fb"], key=lambda x: x["tier"])):
            assert abs(fr["d"] - ft["d"]) < 1e-6


def test_online_rit_miss_with_candidate_side_evaluates_and_twin_reuses(tmp_path):
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    real, reg_r, key_r = _online_judge("fm1", _curves())  # cold: MISS with candidate
    twin, reg_t, key_t = _online_judge("fm1", _curves())
    orch, timer = _orchestrator(backend, real, twin)
    rt, writer = _runtime(tmp_path, warm_tiers=(3, 2, 1))
    traced = _traced(model, orch, timer, rt)
    _episode(traced, uid="u")
    out = traced.get_action(_obs())
    diag = out["__hit_meta__"]["online_rit"]
    assert diag["verdict"] == "MISS" and len(diag["fb"]) == 3 and diag["fb_batch_size"] == 3
    _finish(traced, writer)
    with h5py.File(tmp_path / "exp" / "ep1.h5", "r") as f:
        twin_j = json.loads(f["step_0000"]["trace"].attrs["twin_continuation_json"])
    assert twin_j["verdict"] == "MISS" and len(twin_j["fb"]) == 3 and twin_j["fb_batch_size"] == 0
    for fr, ft in zip(sorted(diag["fb"], key=lambda x: x["tier"]), sorted(twin_j["fb"], key=lambda x: x["tier"])):
        assert abs(fr["d"] - ft["d"]) < 1e-6
    assert reg_r.describe(key_r)["n_updates"] == reg_t.describe(key_t)["n_updates"] == 1


def test_inference_failure_invalidates_both_online_streams(tmp_path, monkeypatch):
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    real, reg_r, key_r = _online_judge("fm1", _curves())
    twin, reg_t, key_t = _online_judge("fm1", _curves())
    orch, timer = _orchestrator(backend, real, twin)
    rt, writer = _runtime(tmp_path, warm_tiers=(3, 2, 1))
    traced = _traced(model, orch, timer, rt)
    _episode(traced, uid="u")
    monkeypatch.setattr(traced._runner, "run_stage2_llm", lambda s1: (_ for _ in ()).throw(RuntimeError("llm")))  # noqa: SLF001
    with pytest.raises(RuntimeError, match="llm"):
        traced.get_action(_obs())
    assert reg_r.is_invalid(key_r) and reg_t.is_invalid(key_t)


# ---------------------------------------------------------------------------
# CP2-only
# ---------------------------------------------------------------------------


def test_cp2_only_cycle_traces_without_a_cp1_check(tmp_path):
    from .test_groot_cp2_interceptor import _CountingFlowHead

    model = _model_with_flow_head(N_STEPS)
    # The CP2 key source reads the prologue output by key: BatchFeature-shaped head.
    torch.manual_seed(0)
    model.action_head = _CountingFlowHead(N_STEPS)
    backend = _library(model, cp=CheckpointID.CP2)
    judge = lambda: ThresholdJudge(cp1_threshold=2.0, warm_tiers=[{"threshold": 0.0, "start_t": 0.75}])  # noqa: E731
    orch_l, timer_l = _orchestrator(backend, judge(), judge(), cp=CheckpointID.CP2)
    orch_t, timer_t = _orchestrator(backend, judge(), judge(), cp=CheckpointID.CP2)
    assert orch_t.has_checkpoint(CheckpointID.CP2) and not orch_t.has_checkpoint(CheckpointID.CP1)
    legacy = _legacy(model, orch_l, timer_l)
    rt, writer = _runtime(tmp_path, checkpoint="CP2")
    traced = _traced(model, orch_t, timer_t, rt)
    for it in (legacy, traced):
        _episode(it)
    checks = []
    real_check = orch_t.check

    def spy(cp_id, **kw):
        checks.append((cp_id, sorted(kw)))
        return real_check(cp_id, **kw)

    orch_t.check = spy
    a = legacy.get_action(_obs())
    b = traced.get_action(_obs())
    np.testing.assert_array_equal(a["action.out"], b["action.out"])
    assert checks == [(CheckpointID.CP2, ["cp2_source", "fetch_top1", "stage2", "trace"])]
    meta = b["__hit_meta__"]
    assert meta["checkpoint"] == "CP2" and meta["hit_type"] == "WARM_START" and meta["cp1_score"] is None
    assert meta["trace"]["executed_arm"] == "warm_03"
    _finish(traced, writer)
    with h5py.File(tmp_path / "exp" / "ep1.h5", "r") as f:
        tg = f["step_0000"]["trace"]
        assert tg.attrs["checkpoint"] == "CP2"
        assert set(tg["actions"].keys()) == {"full_hit", "warm_01", "warm_02", "warm_03", "full_inference", "executed"}


# ---------------------------------------------------------------------------
# Coordinator: two connections, one same-shape bucket per variant
# ---------------------------------------------------------------------------


def test_two_connections_batch_their_variants_through_the_coordinator(tmp_path):
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    judge = lambda: ThresholdJudge(cp1_threshold=2.0)  # noqa: E731
    shared_runner = GrootStagedRunner(model, verify_upstream=False)
    lock = threading.Lock()
    sizes = []
    real = shared_runner.run_stage3

    def spy(stage2, **kw):
        sizes.append(int(stage2.backbone_features.shape[0]))
        return real(stage2, **kw)

    shared_runner.run_stage3 = spy
    outs = {}
    refs = {}
    with BatchingCore(GrootStageBatcher(shared_runner), device="cpu", max_batch_size=8, max_wait_ms=300.0) as bc:
        its = []
        for i in range(2):
            orch, timer = _orchestrator(backend, judge(), judge())
            rt, writer = _runtime(tmp_path / f"c{i}")
            it = _traced(model, orch, timer, rt, coordinator=bc, bundle_id="default", model_lock=lock)
            _episode(it, name=f"ep{i}")
            its.append((it, writer))
            # reference: the same decision on the direct path with the same private/global noise
        barrier = threading.Barrier(2)

        def worker(i):
            it, _ = its[i]
            barrier.wait()
            torch.manual_seed(100 + i)
            outs[i] = it.get_action(_obs())

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(15)
        for it, writer in its:
            _finish(it, writer)
    assert len(outs) == 2
    # the "full" bucket ran once for both connections (same shape, same schedule)
    assert 2 in sizes, sizes
    for i in range(2):
        tr = outs[i]["__hit_meta__"]["trace"]
        assert tr["executed_arm"] == "full_inference"
        assert tr["tier_status"] == {"warm_01": "ok", "warm_02": "ok", "warm_03": "ok"}
        with h5py.File(tmp_path / f"c{i}" / "exp" / f"ep{i}.h5", "r") as f:
            g = f["step_0000"]
            assert all(f"noise_action_{k}" in g for k in range(N_STEPS))
            tg = g["trace"]
            # warm tiers equal the direct resumed loop (B>1 batched vs single, tight tolerance)
            with shared_runner.session():
                stage2 = shared_runner.run_stage2_llm(shared_runner.run_stage1(model.build_inputs()))
            for idx in (1, 2, 3):
                t = SCHEDULE.snapshot_t(idx)
                with shared_runner.session():
                    ref = shared_runner.run_stage3_from(
                        stage2, _payload(0).intermediates[t], t, schedule=SCHEDULE
                    )
                np.testing.assert_allclose(tg["actions"][f"warm_0{idx}"][...], ref.action_pred[0].numpy(), atol=1e-5)


def test_runtime_twins_are_attached_by_the_interceptor(tmp_path):
    """Production assembly (regression, end-to-end gate): the entry point builds
    the orchestrator first, the runtime carries the twins, the interceptor
    attaches them -- otherwise no twin search and no top-1 would be recorded."""
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    real = _components(backend, ThresholdJudge(cp1_threshold=2.0))
    twin = _components(backend, ThresholdJudge(cp1_threshold=2.0))
    timer = SystemTimer(enabled=False)
    orch = CacheOrchestrator(
        real["storage"], real["key_builder"], gates=real["gates"], judges=real["judges"],
        search_strategies=real["strategies"], timer=timer,
    )
    assert not orch.has_twins
    rt, writer = _runtime(tmp_path)
    rt.twins = TwinSet(
        key_builder=twin["key_builder"], gates=twin["gates"], judges=twin["judges"],
        strategies=twin["strategies"], storage=twin["storage"], timer=SystemTimer(enabled=False),
    )
    traced = _traced(model, orch, timer, rt)
    assert orch.has_twins
    _episode(traced)
    out = traced.get_action(_obs())
    assert out["__hit_meta__"]["trace"]["top1_entry_id"] == "lib:0"
    _finish(traced, writer)
    with h5py.File(tmp_path / "exp" / "ep1.h5", "r") as f:
        assert "search" in f["step_0000"]["trace"]
    # re-attaching the same set is a no-op, a different set is refused
    orch.attach_trace_twins(rt.twins)
    with pytest.raises(RuntimeError, match="already attached"):
        orch.attach_trace_twins(TwinSet(**{k: getattr(rt.twins, k) for k in ("key_builder", "gates", "judges", "strategies", "storage", "timer")}))


def test_close_trace_episode_preserves_real_lifecycle(tmp_path, monkeypatch):
    """The GR00T LIBERO adapter's close hook: the dropped connection's trace
    episode is committed non-terminal and the real components see nothing."""
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    judge = lambda: ThresholdJudge(cp1_threshold=2.0)  # noqa: E731
    orch, timer = _orchestrator(backend, judge(), judge())
    calls = []
    monkeypatch.setattr(orch, "on_task_end", lambda: calls.append("orchestrator.on_task_end"))
    rt, writer = _runtime(tmp_path)
    it = _traced(model, orch, timer, rt)
    assert it.traced and not _legacy(model, orch, timer).traced
    _episode(it, name="dropped")
    it.get_action(_obs())
    it.close_trace_episode()
    assert calls == []
    assert writer.drain(timeout=10).ok
    with h5py.File(tmp_path / "exp" / "dropped.h5", "r") as f:
        assert int(f.attrs["num_steps"]) == 1
        assert bool(f.attrs["trace_terminal"]) is False and bool(f.attrs["trace_closed_ok"]) is True
    it.close_trace_episode()  # idempotent: nothing open
    assert writer.drain(timeout=10).ok


def test_close_trace_episode_releases_only_the_twin_search_sessions(tmp_path):
    """A production knn strategy mints a search session per episode for the
    real and the twin set on one backend; the trace-only close releases the
    twin's and leaves the real one exactly as the untraced LIBERO server does."""
    model = _model_with_flow_head(N_STEPS)
    backend = _library(model)
    judge = lambda: ThresholdJudge(cp1_threshold=2.0)  # noqa: E731
    orch, timer = _orchestrator(
        backend, judge(), judge(), strategy=lambda storage: WeightedRrfKnnStrategy(storage, top_k=1)
    )
    rt, writer = _runtime(tmp_path)
    it = _traced(model, orch, timer, rt)
    _episode(it, name="dropped")
    it.get_action(_obs())
    real_sids = set(orch._real_state.strategy_session_ids)  # noqa: SLF001
    twin_sids = set(orch._twins.state.strategy_session_ids)  # noqa: SLF001
    assert real_sids and twin_sids and not real_sids & twin_sids
    assert backend._active_search_sessions == real_sids | twin_sids  # noqa: SLF001
    it.close_trace_episode()
    assert backend._active_search_sessions == real_sids  # noqa: SLF001
    it.close_trace_episode()  # idempotent
    assert backend._active_search_sessions == real_sids  # noqa: SLF001
    assert writer.drain(timeout=10).ok
