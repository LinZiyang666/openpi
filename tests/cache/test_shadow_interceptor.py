"""X15 U3 — the shadow teacher wired into the inference path.

The recorder's own behaviour is covered in ``test_shadow_teacher.py``. What
matters here is the wiring, and every test is about something that must NOT
change when recording is on:

* the dispatched action stays the cache's;
* the global RNG stream is untouched, so later teacher steps draw exactly what
  they would have drawn — the property that makes an on/off byte-parity claim
  possible at all;
* a failing shadow costs a label, not the episode;
* both execution paths (direct model calls and coordinator submissions) are
  covered, because ``_stageN_fn`` is rebound for the latter;
* a truncated connection is marked non-terminal so its labels can be dropped.

Key dependency: ``InferenceInterceptor._record_shadow``.
"""

from __future__ import annotations

import json

import pytest
import torch

from openpi.cache.components.judge import HitType
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.types import CheckpointID
from openpi.cache.shadow_teacher import ShadowTeacherRecorder


class _Result:
    def __init__(self, decision_idx: int = 0) -> None:
        self.router_outputs = {"decision_idx": decision_idx}


def _interceptor(recorder, *, stage3_calls: list) -> InferenceInterceptor:
    """A bare interceptor with only the fields ``_record_shadow`` touches.

    Constructed without ``__init__`` on purpose: the full constructor needs a
    real PyTorch policy, and none of that is under test here.
    """
    obj = object.__new__(InferenceInterceptor)
    object.__setattr__(obj, "_shadow_teacher", recorder)
    object.__setattr__(obj, "_stage_config", None)
    object.__setattr__(obj, "_stage3_device", "cpu")

    def stage2_fn(stage1):
        return stage1

    def stage3_fn(stage2, *, noise=None, num_steps=10):
        stage3_calls.append(noise)
        return torch.ones(4, 3)

    object.__setattr__(obj, "_stage2_fn", stage2_fn)
    object.__setattr__(obj, "_stage3_fn", stage3_fn)
    return obj


# ------------------------------------------------------------------
# Recording
# ------------------------------------------------------------------


def test_a_cache_step_records_one_label(tmp_path) -> None:
    path = tmp_path / "shadow.jsonl"
    rec = ShadowTeacherRecorder(path=str(path))
    rec.begin_episode("u1", 1)
    calls: list = []

    _interceptor(rec, stage3_calls=calls)._record_shadow(
        _Result(3), torch.zeros(4, 3), torch.zeros(1, 8)
    )
    rec.close()

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    assert rows[0]["decision_idx"] == 3
    assert len(calls) == 1, "exactly one teacher forward per decision"


def test_recording_leaves_the_global_rng_untouched(tmp_path) -> None:
    """Without this the extra forward would shift every later teacher step, and
    the on/off byte-parity claim would be false by construction."""
    rec = ShadowTeacherRecorder(path=str(tmp_path / "s.jsonl"))
    rec.begin_episode("u1", 1)
    interceptor = _interceptor(rec, stage3_calls=[])

    torch.manual_seed(0)
    before = torch.random.get_rng_state().clone()
    interceptor._record_shadow(_Result(0), torch.zeros(4, 3), torch.zeros(1, 8))
    after = torch.random.get_rng_state().clone()
    rec.close()

    assert torch.equal(before, after)


def test_the_shadow_noise_is_reproducible_per_decision(tmp_path) -> None:
    seen = []
    for _ in range(2):
        rec = ShadowTeacherRecorder(path=str(tmp_path / "s.jsonl"))
        rec.begin_episode("u1", 1)
        calls: list = []
        _interceptor(rec, stage3_calls=calls)._record_shadow(
            _Result(7), torch.zeros(4, 3), torch.zeros(1, 8)
        )
        rec.close()
        seen.append(calls[0])

    assert torch.equal(seen[0], seen[1])


# ------------------------------------------------------------------
# What must not change
# ------------------------------------------------------------------


def test_disabled_recorder_runs_no_teacher_forward(tmp_path) -> None:
    """Off must be genuinely off — no extra compute, no rows."""
    rec = ShadowTeacherRecorder(path=str(tmp_path / "s.jsonl"), enabled=False)
    rec.begin_episode("u1", 1)
    calls: list = []
    _interceptor(rec, stage3_calls=calls)._record_shadow(
        _Result(0), torch.zeros(4, 3), torch.zeros(1, 8)
    )
    assert calls == []


def test_absent_recorder_is_a_no_op() -> None:
    calls: list = []
    _interceptor(None, stage3_calls=calls)._record_shadow(
        _Result(0), torch.zeros(4, 3), torch.zeros(1, 8)
    )
    assert calls == []


def test_a_failing_teacher_forward_does_not_propagate(tmp_path) -> None:
    """Fail-open: the episode outranks the label."""
    rec = ShadowTeacherRecorder(path=str(tmp_path / "s.jsonl"))
    rec.begin_episode("u1", 1)
    obj = _interceptor(rec, stage3_calls=[])

    def boom(stage2, *, noise=None, num_steps=10):
        raise RuntimeError("stage3 exploded")

    object.__setattr__(obj, "_stage3_fn", boom)
    obj._record_shadow(_Result(0), torch.zeros(4, 3), torch.zeros(1, 8))  # must not raise
    rec.close()

    assert rec.error_count == 1


def test_a_verdict_without_a_decision_index_is_skipped(tmp_path) -> None:
    """A non-router verdict carries no decision coordinate to join on."""
    class _NoRouter:
        router_outputs = None

    rec = ShadowTeacherRecorder(path=str(tmp_path / "s.jsonl"))
    rec.begin_episode("u1", 1)
    calls: list = []
    _interceptor(rec, stage3_calls=calls)._record_shadow(
        _NoRouter(), torch.zeros(4, 3), torch.zeros(1, 8)
    )
    assert calls == []


# ------------------------------------------------------------------
# Both execution paths
# ------------------------------------------------------------------


def test_the_coordinator_path_is_covered_too(tmp_path) -> None:
    """Under a coordinator the stage functions are rebound to submissions;
    the shadow calls them the same way, so no branch is needed."""
    rec = ShadowTeacherRecorder(path=str(tmp_path / "s.jsonl"))
    rec.begin_episode("u1", 1)
    obj = _interceptor(rec, stage3_calls=[])

    submitted: list = []

    def via_coordinator(stage2, *, noise=None, num_steps=10):
        submitted.append(("stage3", noise is not None))
        return torch.ones(4, 3)

    object.__setattr__(obj, "_stage3_fn", via_coordinator)
    obj._record_shadow(_Result(1), torch.zeros(4, 3), torch.zeros(1, 8))
    rec.close()

    assert submitted == [("stage3", True)]


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------


def test_a_dropped_connection_marks_the_episode_non_terminal(tmp_path) -> None:
    """A truncated rollout must be droppable by the joiner rather than being
    mistaken for a complete one."""
    path = tmp_path / "s.jsonl"
    rec = ShadowTeacherRecorder(path=str(path))
    rec.begin_episode("u1", 1)
    rec.finalize_episode(terminal=False)
    rec.close()

    row = [json.loads(line) for line in path.read_text().splitlines() if line.strip()][-1]
    assert row["status"] == "finalize" and row["terminal"] is False


def test_config_refuses_enabled_collection_without_a_path() -> None:
    """Enabling collection with nowhere to write discards every label while
    looking like a successful run."""
    from openpi.cache.config import (
        CacheConfig,
        ConfigValidationError,
        ShadowTeacherConfig,
        validate_cache_config,
    )

    cfg = CacheConfig(shadow_teacher=ShadowTeacherConfig(enabled=True, path=""))
    with pytest.raises(ConfigValidationError, match="shadow_teacher.enabled requires"):
        validate_cache_config(cfg)


def test_disabled_shadow_config_is_the_default() -> None:
    from openpi.cache.config import CacheConfig, validate_cache_config

    cfg = CacheConfig()
    assert cfg.shadow_teacher.enabled is False
    validate_cache_config(cfg)


# ------------------------------------------------------------------
# Contracts the reviewer's probe caught (Round 6)
# ------------------------------------------------------------------


def test_the_label_comes_from_stage3_outputs_action_chunk(tmp_path) -> None:
    """``run_stage3`` returns a ``Stage3Output``, not a bare tensor.

    Treating the wrapper as the chunk silently poisons every label with a
    dataclass repr instead of actions.
    """
    class _Stage3Output:
        def __init__(self, chunk):
            self.action_chunk = chunk
            self.intermediates = None

    rec = ShadowTeacherRecorder(path=str(tmp_path / "s.jsonl"))
    rec.begin_episode("u1", 1)
    obj = _interceptor(rec, stage3_calls=[])
    object.__setattr__(
        obj, "_stage3_fn",
        lambda stage2, *, noise=None, num_steps=10: _Stage3Output(torch.ones(4, 3)),
    )

    obj._record_shadow(_Result(0), torch.zeros(4, 3), torch.zeros(1, 8))
    rec.close()

    rows = [
        json.loads(line)
        for line in (tmp_path / "s.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ok = [r for r in rows if r["status"] == "ok"][0]
    # A real deviation, computed from real numbers.
    assert ok["u"] > 0


def test_exactly_one_terminal_row_even_when_both_hooks_fire(tmp_path) -> None:
    """A normal run calls episode-end AND connection-close. Two rows would
    leave the joiner unable to tell whether the episode completed, because the
    second one says terminal=False."""
    path = tmp_path / "s.jsonl"
    rec = ShadowTeacherRecorder(path=str(path))
    rec.begin_episode("u1", 1)
    rec.finalize_episode(terminal=True)     # on_episode_end
    rec.finalize_episode(terminal=False)    # on_task_end
    rec.close()

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    finals = [r for r in rows if r["status"] == "finalize"]
    assert len(finals) == 1
    assert finals[0]["terminal"] is True


def test_a_new_episode_can_finalize_again(tmp_path) -> None:
    """Idempotence is per episode, not for the recorder's lifetime."""
    path = tmp_path / "s.jsonl"
    rec = ShadowTeacherRecorder(path=str(path))
    for uid in ("u1", "u2"):
        rec.begin_episode(uid, 1)
        rec.finalize_episode(terminal=True)
        rec.finalize_episode(terminal=False)
    rec.close()

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len([r for r in rows if r["status"] == "finalize"]) == 2


def test_the_teacher_arm_is_labelled_too(tmp_path) -> None:
    """Half the label coverage lives on the teacher arm, where the teacher
    chunk is free and the cached candidate is what must be fetched.

    Built on a REAL ``CheckResult``. An earlier version used a stub with a
    ``winner_id`` attribute — a field that exists on ``JudgeResult`` but not on
    ``CheckResult`` — so the production object always took the early return and
    this arm silently recorded nothing while the test passed.
    """
    from openpi.cache.orchestrator import CheckResult

    class _Payload:
        action_chunk = torch.ones(4, 3)

    class _Storage:
        def fetch_payload(self, entry_id):
            assert entry_id == "e1"
            return _Payload()

    class _Orch:
        _storage = _Storage()

    class _Stage3Output:
        action_chunk = torch.zeros(4, 3)

    verdict = CheckResult(
        hit_type=HitType.MISS,
        entry_id="e1",
        router_outputs={"decision_idx": 2},
    )

    path = tmp_path / "s.jsonl"
    rec = ShadowTeacherRecorder(path=str(path))
    rec.begin_episode("u1", 1)
    obj = _interceptor(rec, stage3_calls=[])
    object.__setattr__(obj, "_orchestrator", _Orch())

    obj._record_shadow_teacher_arm(verdict, _Stage3Output())
    rec.close()

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ok = [r for r in rows if r["status"] == "ok"]
    assert ok, "the teacher arm must produce a label"
    assert ok[0]["decision_idx"] == 2 and ok[0]["u"] > 0


def test_a_teacher_verdict_carries_the_passed_over_candidate() -> None:
    """The judge must surface the retrieval top-1 on a MISS.

    Teacher executes either way, so the verdict does not need it — but without
    it the shadow labeller has no idea which cached chunk was declined, and the
    teacher arm degenerates to no labels at all.
    """
    from openpi.cache.components.risk_features import RiskFeatureBuilder
    from openpi.cache.components.risk_router_judge import RiskRouterJudge
    from openpi.cache.storage_types import SearchResultLite

    class _Risk:
        def risk(self, x):
            return 0.99          # forces the teacher arm

    judge = RiskRouterJudge(
        feature_builder=RiskFeatureBuilder(
            task_index=0, replan_steps=5, library_replan_steps=5,
        ),
        risk_model=_Risk(),
        tau=0.5,
    )
    results = [SearchResultLite(id="n1", score=0.9, checkpoint_id=CheckpointID.CP1)]
    verdict = judge(
        results, CheckpointID.CP1, {},
        view=None, history=None, retrieval_signals=None,
        step_features=None, query_keys=None,
    )

    assert verdict.hit_type is HitType.MISS
    assert verdict.winner_id == "n1", "the declined candidate must ride along"


def test_teacher_arm_labelling_fails_open(tmp_path) -> None:
    class _Orch:
        class _storage:
            @staticmethod
            def fetch_payload(entry_id):
                raise KeyError(entry_id)

    from openpi.cache.orchestrator import CheckResult

    verdict = CheckResult(
        hit_type=HitType.MISS, entry_id="gone",
        router_outputs={"decision_idx": 0},
    )

    rec = ShadowTeacherRecorder(path=str(tmp_path / "s.jsonl"))
    rec.begin_episode("u1", 1)
    obj = _interceptor(rec, stage3_calls=[])
    object.__setattr__(obj, "_orchestrator", _Orch())

    obj._record_shadow_teacher_arm(verdict, object())   # must not raise
    rec.close()


def test_server_builds_no_recorder_when_collection_is_off() -> None:
    """Off means no object at all, not a disabled one sitting in the path."""
    import importlib.util
    import pathlib as _pathlib

    spec = importlib.util.spec_from_file_location(
        "_serve_probe", _pathlib.Path("scripts/serve_policy.py")
    )
    assert spec and spec.loader
    # Only the helper is needed; importing the module's heavy deps is avoided
    # by reading the function out of the compiled source.
    from openpi.cache.config import CacheConfig, ShadowTeacherConfig

    src = _pathlib.Path("scripts/serve_policy.py").read_text(encoding="utf-8")
    ns: dict = {"logging": __import__("logging")}
    start = src.index("def _build_shadow_teacher(")
    end = src.index("def _build_routing_executors(", start)
    exec(compile(src[start:end], "serve_policy_helper", "exec"), ns)

    assert ns["_build_shadow_teacher"](CacheConfig()) is None
    on = CacheConfig(shadow_teacher=ShadowTeacherConfig(enabled=True, path="/tmp/x15.jsonl"))
    assert ns["_build_shadow_teacher"](on) is not None
