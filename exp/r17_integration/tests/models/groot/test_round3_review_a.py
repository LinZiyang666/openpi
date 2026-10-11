"""Review A round 3: GR00T-MW canonical task table (A2 N2) and native lock timing (A2 N5)."""
import threading
import time

import numpy as np
import pytest

from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.models.groot.serve import default_config, make_connection_factory
from .conftest import begin_episode
from .test_shared_factory import published  # noqa: F401  (fixture)


def _counts(policy):
    model = policy.adapter.backend.model
    return model.backbone.eagle_model.extract_calls, model.action_head.nfe


def test_round3_default_config_carries_canonical_table_and_rejects_mismatched_task(published, obs):
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    from exp.r17_integration.common.benchmarks import task_table_identity
    cfg = default_config(published, commit_horizon=5)
    assert cfg.recipe.enabled is False
    assert tuple(cfg.recipe.ordered_tasks) == tuple(r[0] for r in FORMAL_SPEC.tasks)
    assert cfg.recipe.task_table_identity == task_table_identity(FORMAL_SPEC.identity)
    policy = make_connection_factory(published, cfg, build_shared_storage(cfg))()
    begin_episode(policy)
    before = _counts(policy)
    # Same canonical index (reach-v3) but another task's name: must fail before any stage.
    wrong = dict(obs, __request__=dict(obs["__request__"], task_name="push-v3"))
    with pytest.raises(ValueError, match="task_name/task_id"):
        policy.infer(wrong)
    assert _counts(policy) == before
    reply = policy.infer(obs)
    assert reply["actions"].shape == (5, 4)


def test_round3_native_model_lock_wait_is_separated_from_service(published, obs):
    from exp.r17_integration.benchmarks.execution import instrument_model_lock
    cfg = default_config(published, commit_horizon=5)
    factory = make_connection_factory(published, cfg, build_shared_storage(cfg))
    first, second = factory(), factory()
    # Exclusion stays process-shared; only the wait ledger is connection-local.
    assert first.adapter.model_lock is second.adapter.model_lock
    assert first.adapter._timed_lock is not second.adapter._timed_lock
    assert instrument_model_lock(first.adapter) is first.adapter  # no double wrapping
    begin_episode(first)
    uncontended = first.infer(obs)["__hit_meta__"]
    assert uncontended["pricing_time_field"] == "service_ms" and uncontended["pricing_eligible"] is True
    assert 0 <= uncontended["queue_wait_ms"] < 20
    assert uncontended["service_ms"] == pytest.approx(uncontended["cost_ms"] - uncontended["queue_wait_ms"])

    begin_episode(first)
    held, release = threading.Event(), threading.Event()
    def holder():
        with first.adapter.model_lock:
            held.set()
            release.wait(5)
    thread = threading.Thread(target=holder)
    thread.start()
    held.wait(5)
    timer = threading.Timer(.05, release.set)
    timer.start()
    try:
        contended = first.infer(obs)
    finally:
        release.set()
        thread.join(5)
        timer.cancel()
    meta = contended["__hit_meta__"]
    assert meta["queue_wait_ms"] >= 30
    assert meta["service_ms"] == pytest.approx(meta["cost_ms"] - meta["queue_wait_ms"])
    stages = meta["stage_queue_wait_ms"]
    assert sum(stages.values()) == pytest.approx(meta["queue_wait_ms"])
    assert all(meta["stage_service_ms"][k] == pytest.approx(meta["stage_ms"][k] - stages[k]) for k in stages)
    # Timing never changes actions: identical request on an uncontended fresh connection.
    fresh = factory()
    begin_episode(fresh)
    assert fresh.infer(obs)["actions"].tobytes() == contended["actions"].tobytes()
    assert first.adapter.drain_queue_wait_ms() == 0.0
