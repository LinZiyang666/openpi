"""Integration: MlpRouterJudge -> Orchestrator -> Interceptor, real two layers.

These are not stub tests: a real ``CacheOrchestrator`` over the in-memory
backend drives a real ``InferenceInterceptor`` with the fake staged model, so
the tri-state FULL_HIT dispatch is exercised end to end.

What is pinned here:

  - each arm reaches the execution path it is supposed to, and *only* that one
    (stage2/3 call counts are the witness that the student arm really skips
    teacher inference and the cache arm really skips both);
  - a payloadless FULL_HIT performs zero payload fetches;
  - the ``hit_override is None`` path — every pre-X14 judge — keeps its exact
    wire, including the absence of a ``router_outputs`` key;
  - ``arm_executed`` is written back from the path that actually ran, including
    the empty-library fallback, so execution cost is billed to the truth;
  - ``decision_idx`` is a dense server-side counter, which is what makes the
    three-source join possible when the client's step index advances by
    ``replan_steps``;
  - the ``query_keys`` injection seam does not perturb CompositeJudge or any
    dump-wrapped legacy config.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi.cache.components.judge import HitType, ThresholdJudge, judge_accepts_query_keys
from openpi.cache.components.mlp_router_judge import MlpRouterJudge
from openpi.cache.components.write_policy import AlwaysWritePolicy
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID, ROBOT_STATE

from tests.cache.conftest import insert_entry, make_counting_orchestrator, make_orchestrator
from tests.cache.test_interceptor import FakeModel, FakePolicy, _make_obs

IDENTITY = {"run_id": "r0", "batch_id": "b0", "task_uid": "y:eval:0:0", "attempt": 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _router(arm: str, arms: str) -> MlpRouterJudge:
    """A deterministic single-arm router — the dispatch matrix wants the arm
    pinned, not sampled."""
    return MlpRouterJudge(
        arms=arms, constant_arm=arm, mode="argmax", feature_fields=[ROBOT_STATE],
    )


class _Sidecar:
    """Stand-in for SidecarExecutor: records calls, returns a marked action."""

    def __init__(self, action_value: float = 7.0, raises: Exception | None = None) -> None:
        self.calls = 0
        self._value = action_value
        self._raises = raises

    def __call__(self, obs: dict) -> dict:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return {"actions": np.full((50, 32), self._value, dtype=np.float32),
                "state": np.zeros(32, dtype=np.float32)}


def _cp1_only_orchestrator(judge, **kwargs):
    """Build a CP1-only orchestrator, mirroring the routing allowlist.

    ``conftest.make_orchestrator`` binds one component instance to BOTH
    checkpoints, which would drive the router's per-episode verdict counter
    twice per inference cycle. Routed configs are locked to ``checkpoints ==
    ['cp1']`` (and ``_validate_mlp_router_static`` rejects a CP3 router), so
    CP1-only is the faithful topology here.
    """
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.components.gate import AlwaysSearchGate
    from openpi.cache.components.key_builder import PlaceholderKeyBuilder
    from openpi.cache.orchestrator import CacheOrchestrator

    from tests.cache.conftest import TestStorageSearchStrategy

    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    orch = CacheOrchestrator(
        storage,
        PlaceholderKeyBuilder(),
        gates={CheckpointID.CP1: AlwaysSearchGate()},
        judges={CheckpointID.CP1: judge},
        search_strategies={CheckpointID.CP1: TestStorageSearchStrategy(storage, top_k=1)},
        timer=SystemTimer(enabled=False),
        **kwargs,
    )
    return orch, storage


def _build(arm: str, arms: str, *, hit_executor=None, seed_cache: bool = False):
    """Wire a router judge into a real orchestrator + interceptor."""
    fixed_state = torch.zeros(1, 32)
    fixed_state[0, 0] = 1.0
    model = FakeModel(fixed_state=fixed_state, fixed_action=torch.randn(1, 50, 32))
    judge = _router(arm, arms)
    orch, storage = _cp1_only_orchestrator(judge)
    if seed_cache:
        insert_entry(storage, CheckpointID.CP1, fixed_state,
                     CachePayload(action_chunk=torch.full((50, 32), -3.0)))
    interceptor = InferenceInterceptor(
        FakePolicy(model=model), timer=SystemTimer(enabled=False),
        orchestrator=orch, hit_executor=hit_executor,
    )
    interceptor.on_task_begin()
    orch.on_episode_start("t", "e", extra_metadata=dict(IDENTITY))
    return interceptor, model, orch


# ---------------------------------------------------------------------------
# Tri-state dispatch matrix
# ---------------------------------------------------------------------------


def test_student_arm_runs_the_sidecar_and_skips_all_inference() -> None:
    sidecar = _Sidecar()
    interceptor, model, _ = _build("student", "ts", hit_executor=sidecar)

    out = interceptor.infer(_make_obs())

    assert sidecar.calls == 1
    assert np.allclose(out["actions"], 7.0)          # sidecar action, not cache
    assert (model.stage2_calls, model.stage3_calls) == (0, 0)
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "FULL_HIT" and meta["winner_id"] is None
    assert meta["executor"] == "override"
    assert meta["router_outputs"]["arm_sampled"] == "student"
    assert meta["router_outputs"]["arm_executed"] == "student"
    assert meta["router_outputs"]["fallback"] is False


def test_student_arm_without_a_sidecar_fails_loud() -> None:
    """A payloadless FULL_HIT with no executor has no action source at all;
    silently falling through would route the student arm to the teacher."""
    interceptor, _, _ = _build("student", "ts", hit_executor=None)
    with pytest.raises(RuntimeError, match="hit_executor"):
        interceptor.infer(_make_obs())


def test_cache_arm_forces_replay_even_with_a_sidecar_wired() -> None:
    """R_tsc wires a sidecar for its student arm; the cache arm must still
    replay the cached action rather than inherit the routing override."""
    sidecar = _Sidecar()
    interceptor, model, _ = _build("cache", "tsc", hit_executor=sidecar, seed_cache=True)

    out = interceptor.infer(_make_obs())

    assert sidecar.calls == 0
    assert np.allclose(out["actions"], -3.0)         # the cached clean action
    assert (model.stage2_calls, model.stage3_calls) == (0, 0)
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "FULL_HIT" and meta["winner_id"] is not None
    assert "executor" not in meta
    assert meta["router_outputs"]["arm_executed"] == "cache"


def test_cache_arm_on_an_empty_library_falls_back_to_the_teacher() -> None:
    interceptor, model, _ = _build("cache", "tc", seed_cache=False)

    out = interceptor.infer(_make_obs())

    assert (model.stage2_calls, model.stage3_calls) == (1, 1)
    ro = out["__hit_meta__"]["router_outputs"]
    assert out["__hit_meta__"]["hit_type"] == "MISS"
    assert (ro["arm_sampled"], ro["arm_executed"], ro["fallback"]) == ("cache", "teacher", True)


def test_teacher_arm_runs_full_inference() -> None:
    interceptor, model, _ = _build("teacher", "tsc", seed_cache=True)

    out = interceptor.infer(_make_obs())

    assert (model.stage2_calls, model.stage3_calls) == (1, 1)
    ro = out["__hit_meta__"]["router_outputs"]
    assert (ro["arm_sampled"], ro["arm_executed"], ro["fallback"]) == ("teacher", "teacher", False)


def test_sidecar_failure_propagates_fail_closed() -> None:
    """A sidecar error must end the episode with an error terminal state, not
    silently degrade the student arm into some other arm."""
    sidecar = _Sidecar(raises=RuntimeError("sidecar down"))
    interceptor, _, _ = _build("student", "ts", hit_executor=sidecar)
    with pytest.raises(RuntimeError, match="sidecar down"):
        interceptor.infer(_make_obs())


# ---------------------------------------------------------------------------
# Orchestrator-level invariants
# ---------------------------------------------------------------------------


def test_payloadless_full_hit_never_fetches_a_payload() -> None:
    # CountingStorage is the witness: the student arm must not touch the library.
    orch, _, storage = make_counting_orchestrator(judge=_router("student", "ts"))
    orch.on_episode_start("t", "e", extra_metadata=dict(IDENTITY))

    result = orch.check(CheckpointID.CP1, stage1=_stage1())

    assert result.hit_type is HitType.FULL_HIT
    assert result.payload is None and result.entry_id is None
    assert result.hit_override is True
    assert result.searched is True
    assert result.query_keys is not None
    assert storage.fetch_payload_call_count == 0


def test_miss_paths_still_carry_router_outputs() -> None:
    """Cost accounting needs every verdict, not only the hit ones."""
    orch, _ = _cp1_only_orchestrator(_router("teacher", "tsc"))
    orch.on_episode_start("t", "e", extra_metadata=dict(IDENTITY))
    result = orch.check(CheckpointID.CP1, stage1=_stage1())
    assert result.hit_type is HitType.MISS
    assert result.router_outputs["arm_sampled"] == "teacher"
    assert result.hit_override is None


def test_legacy_judges_leave_both_new_fields_none() -> None:
    orch, _, _ = make_orchestrator(judge=ThresholdJudge(cp1_threshold=0.98))
    orch.on_task_begin()
    result = orch.check(CheckpointID.CP1, stage1=_stage1())
    assert result.hit_override is None and result.router_outputs is None


def test_decision_idx_is_dense_across_an_episode() -> None:
    """The client's per-step index advances by ``replan_steps`` between infer
    calls, so the join key has to come from the server. This is that counter."""
    interceptor, _, orch = _build("teacher", "tsc")
    replan_steps = 5
    server_idx, client_idx = [], []
    for call in range(4):
        meta = interceptor.infer(_make_obs())["__hit_meta__"]
        server_idx.append(meta["router_outputs"]["decision_idx"])
        client_idx.append(call * replan_steps)  # what the LIBERO client records
    assert server_idx == [0, 1, 2, 3]
    assert client_idx == [0, 5, 10, 15] != server_idx

    # A new episode restarts the counter; the client's would not.
    orch.on_episode_start("t", "e2", extra_metadata=dict(IDENTITY, task_uid="y:eval:0:1"))
    meta = interceptor.infer(_make_obs())["__hit_meta__"]
    assert meta["router_outputs"]["decision_idx"] == 0


def _stage1():
    state = torch.zeros(1, 32)
    state[0, 0] = 1.0
    return type("S", (), {"state": state, "prefix_embs": torch.randn(1, 10, 8)})()


# ---------------------------------------------------------------------------
# hit_override is None: the pre-X14 wire, byte for byte
# ---------------------------------------------------------------------------


_LEGACY_META_KEYS = {"hit_type", "start_t", "winner_id", "cp1_score", "searched"}


def test_legacy_full_hit_replay_wire_is_unchanged() -> None:
    fixed_state = torch.randn(1, 32)
    model = FakeModel(fixed_state=fixed_state, fixed_action=torch.randn(1, 50, 32))
    orch, _, _ = make_orchestrator(write_policy=AlwaysWritePolicy())
    interceptor = InferenceInterceptor(
        FakePolicy(model=model), timer=SystemTimer(enabled=False), orchestrator=orch,
    )
    interceptor.on_task_begin()
    interceptor.infer(_make_obs())           # seeds the cache via AlwaysWritePolicy
    interceptor.on_episode_end(success=True)
    interceptor.on_episode_start("t", "t", 1)

    meta = interceptor.infer(_make_obs())["__hit_meta__"]
    assert meta["hit_type"] == "FULL_HIT"
    assert set(meta) == _LEGACY_META_KEYS      # no router_outputs, no executor


def test_legacy_full_hit_with_a_sidecar_still_overrides() -> None:
    """The pre-X14 ablation routing behaviour (override=None + hit_executor)
    must keep working exactly as before."""
    fixed_state = torch.randn(1, 32)
    model = FakeModel(fixed_state=fixed_state, fixed_action=torch.randn(1, 50, 32))
    orch, _, _ = make_orchestrator(write_policy=AlwaysWritePolicy())
    sidecar = _Sidecar(action_value=1.5)
    interceptor = InferenceInterceptor(
        FakePolicy(model=model), timer=SystemTimer(enabled=False),
        orchestrator=orch, hit_executor=sidecar,
    )
    interceptor.on_task_begin()
    interceptor.infer(_make_obs())
    interceptor.on_episode_end(success=True)
    interceptor.on_episode_start("t", "t", 1)

    out = interceptor.infer(_make_obs())
    assert sidecar.calls == 1
    assert np.allclose(out["actions"], 1.5)
    assert set(out["__hit_meta__"]) == _LEGACY_META_KEYS | {"executor"}


def test_legacy_miss_wire_is_unchanged() -> None:
    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        FakePolicy(), timer=SystemTimer(enabled=False), orchestrator=orch,
    )
    interceptor.on_task_begin()
    meta = interceptor.infer(_make_obs())["__hit_meta__"]
    assert meta["hit_type"] == "MISS"
    assert set(meta) == _LEGACY_META_KEYS


# ---------------------------------------------------------------------------
# query_keys injection seam
# ---------------------------------------------------------------------------


def _miss():
    from openpi.cache.components.judge import JudgeResult

    return JudgeResult(HitType.MISS)


class _DeclaringJudge:
    """Declares ``query_keys`` explicitly, like MlpRouterJudge does.

    The spies below carry *real* signatures rather than ``*args, **kwargs``
    delegates: the probe inspects ``type(judge).__call__``, so a delegate would
    read as "accepts everything" and the test would prove nothing.
    """

    def __init__(self) -> None:
        self.got_query_keys: list[bool] = []

    def __call__(self, results, checkpoint_id, cached_data, *, view=None,
                 history=None, retrieval_signals=None, query_keys=None):
        self.got_query_keys.append(query_keys is not None)
        return _miss()


class _StrictJudge:
    """Keyword-only, no ``query_keys``, no ``**kwargs`` — like CompositeJudge."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, results, checkpoint_id, cached_data, *, view=None,
                 history=None, retrieval_signals=None):
        self.calls += 1
        return _miss()


def test_probe_admits_explicit_and_var_keyword_judges_only() -> None:
    from openpi.cache.components.composite_judge import CompositeJudge

    assert judge_accepts_query_keys(ThresholdJudge()) is True          # **kwargs
    assert judge_accepts_query_keys(ThresholdJudge(), allow_var_keyword=False) is False
    assert judge_accepts_query_keys(_router("teacher", "ts")) is True  # explicit
    # CompositeJudge declares keyword-only params and no **kwargs: injecting
    # would TypeError, which is exactly why the probe exists.
    assert "query_keys" not in CompositeJudge.__call__.__code__.co_varnames


def test_orchestrator_injects_only_into_declaring_judges() -> None:
    declaring, strict = _DeclaringJudge(), _StrictJudge()
    for judge in (declaring, strict):
        orch, _ = _cp1_only_orchestrator(judge)
        orch.on_task_begin()
        orch.check(CheckpointID.CP1, stage1=_stage1())  # a strict judge would TypeError
    assert declaring.got_query_keys == [True]
    assert strict.calls == 1


def test_dump_wrapped_judges_forward_query_keys_only_to_a_router(tmp_path) -> None:
    """Dump-wrapped legacy / composite inners must see a byte-identical call;
    a dump-wrapped router must still receive its features."""
    from openpi.cache.components.dumping_judge import DumpingJudge
    from openpi.cache.components.factors.base import LibraryStats
    from openpi.cache.components.factors.normalization import ZScoreNormalization

    stats = LibraryStats(
        action_sigma=torch.ones(2), action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=torch.ones(2), state_active_mask=torch.ones(2, dtype=torch.bool),
    )

    def wrap(inner, name):
        return DumpingJudge(
            inner=inner, dump_normalization=ZScoreNormalization(stats), dump_factors=[],
            dump_path=str(tmp_path / f"{name}.jsonl"), config_id=name,
        )

    assert wrap(_StrictJudge(), "strict")._forward_query_keys is False
    assert wrap(_DeclaringJudge(), "router")._forward_query_keys is True
    # A **kwargs legacy judge is deliberately excluded too: it would only
    # swallow the kwarg, and withholding it keeps the inner call identical.
    assert wrap(ThresholdJudge(), "threshold")._forward_query_keys is False


def test_dump_wrapped_router_works_end_to_end(tmp_path) -> None:
    from openpi.cache.components.dumping_judge import DumpingJudge
    from openpi.cache.components.factors.base import LibraryStats
    from openpi.cache.components.factors.normalization import ZScoreNormalization

    stats = LibraryStats(
        action_sigma=torch.ones(2), action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=torch.ones(2), state_active_mask=torch.ones(2, dtype=torch.bool),
    )
    wrapped = DumpingJudge(
        inner=_router("teacher", "ts"), dump_normalization=ZScoreNormalization(stats),
        dump_factors=[], dump_path=str(tmp_path / "d.jsonl"), config_id="d",
    )
    orch, _ = _cp1_only_orchestrator(wrapped)
    orch.on_episode_start("t", "e", extra_metadata=dict(IDENTITY))
    result = orch.check(CheckpointID.CP1, stage1=_stage1())
    assert result.router_outputs["arm_sampled"] == "teacher"


# ---------------------------------------------------------------------------
# Config guards: routing must not be able to break the arm semantics
# ---------------------------------------------------------------------------


def _router_cfg(*, arms: str, hit_to=None, miss_to=None):
    from openpi.cache.config import (
        BackendConfig, CacheConfig, CheckpointConfig, GateConfig, JudgeConfig,
        KeyBuilderConfig, KeysConfig, RoutingConfig, SearchStrategyConfig,
        WritePolicyConfig, validate_cache_config,
    )

    keys = KeysConfig()
    for name in ("vision_0", "vision_1", "vision_2", "prompt_emb"):
        getattr(keys, name).enabled = False
    keys.robot_state.enabled = True
    cfg = CacheConfig(
        enabled=True, keys=keys, key_builder=KeyBuilderConfig(type="placeholder"),
        checkpoints={"cp1": CheckpointConfig(
            gate=GateConfig(type="always_search"),
            judge=JudgeConfig(type="mlp_router", arms=arms, mode="argmax",
                              constant_arm="teacher", feature_fields=[ROBOT_STATE], hidden=8),
            search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", top_k=1),
        )},
        backend=BackendConfig(type="in_memory", vector_dims={ROBOT_STATE: 32}),
        write_policy=WritePolicyConfig(type="never"),
        routing=(RoutingConfig(hit_to=hit_to, miss_to=miss_to)
                 if (hit_to or miss_to) else None),
    )
    validate_cache_config(cfg)
    return cfg


def test_mlp_router_rejects_miss_to_routing() -> None:
    """The MISS slot IS the teacher arm. Routing it to a sidecar would execute
    the student model while the verdict, the wire and the cost ledger all still
    say "teacher" — corrupting both the arm semantics and the reward."""
    from openpi.cache.config import ConfigValidationError

    with pytest.raises(ConfigValidationError, match="miss_to"):
        _router_cfg(arms="tc", miss_to="127.0.0.1:7002")
    with pytest.raises(ConfigValidationError, match="miss_to"):
        _router_cfg(arms="tsc", hit_to="127.0.0.1:7002", miss_to="127.0.0.1:7003")


def test_router_arm_routing_pairing_is_bidirectional() -> None:
    from openpi.cache.config import ConfigValidationError

    _router_cfg(arms="tc")                                  # no student, no routing: OK
    _router_cfg(arms="ts", hit_to="127.0.0.1:7002")         # student + sidecar: OK
    with pytest.raises(ConfigValidationError, match="requires routing.hit_to"):
        _router_cfg(arms="ts")
    with pytest.raises(ConfigValidationError, match="no student arm"):
        _router_cfg(arms="tc", hit_to="127.0.0.1:7002")


def test_mlp_router_is_cp1_only() -> None:
    from openpi.cache.config import (
        BackendConfig, CacheConfig, CheckpointConfig, ConfigValidationError,
        GateConfig, JudgeConfig, KeyBuilderConfig, KeysConfig, SearchStrategyConfig,
        WritePolicyConfig, validate_cache_config,
    )

    keys = KeysConfig()
    for name in ("vision_0", "vision_1", "vision_2", "prompt_emb"):
        getattr(keys, name).enabled = False
    keys.robot_state.enabled = True
    judge = JudgeConfig(type="mlp_router", arms="tc", mode="argmax",
                        constant_arm="teacher", feature_fields=[ROBOT_STATE], hidden=8)
    cfg = CacheConfig(
        enabled=True, keys=keys, key_builder=KeyBuilderConfig(type="placeholder"),
        checkpoints={"cp3": CheckpointConfig(
            gate=GateConfig(type="always_search"), judge=judge,
            search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", top_k=1))},
        backend=BackendConfig(type="in_memory", vector_dims={ROBOT_STATE: 32}),
        write_policy=WritePolicyConfig(type="never"),
    )
    with pytest.raises(ConfigValidationError, match="CP1-only"):
        validate_cache_config(cfg)
