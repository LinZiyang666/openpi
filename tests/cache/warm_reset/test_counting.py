"""Measured step / call counts travel execution -> output -> hit meta -> wrapper -> admission (plan §4.4, §9.4 B2).

Faults are injected where they would really happen -- the loop length inside
the stage-3 entries, an extra or a skipped binding call inside the executor --
never by editing the JSON, and both the direct and the coordinator binding are
exercised.
"""

from __future__ import annotations

import pytest

from openpi.cache.warm_reset import pi05 as P
from openpi.cache.warm_reset.evidence import episode_problems
from tests.cache.warm_reset._support import (
    EPISODE,
    EXTRA,
    expected_for,
    obs_for,
    pi05_stack,
    read_rows,
    run_episode,
)

PATHS = [False, True]  # direct binding, coordinator binding


def _admit(stack, n=3):
    rows = read_rows(stack.served.evidence_path)
    return rows, episode_problems(rows, expected=expected_for(stack, n=n))


@pytest.mark.parametrize("coordinator", PATHS)
@pytest.mark.parametrize("delta", [-1, 1])
def test_continuation_loop_length_is_measured(tmp_path, monkeypatch, coordinator, delta):
    monkeypatch.setattr(P, "_loop_steps", lambda plan: plan.n_steps + delta)
    stack = pi05_stack(tmp_path, "warmreset_t0.2", coordinator=coordinator)
    run_episode(stack, 3)
    rows, result = _admit(stack)
    assert {r["warm_reset"]["continuation_nfe"] for r in rows if r["row_kind"] == "decision"} == {2 + delta}
    assert result["problems"]["steps_mismatch"] == 3 and result["total_nfe"] is None


@pytest.mark.parametrize("coordinator", PATHS)
@pytest.mark.parametrize("delta,arm", [(-1, "selfresetfinal_t0.2"), (1, "selfresetfinal_t0.2"),
                                       (-1, "selfwarmreset_t0.9"), (1, "selfwarmreset_t0.9")])
def test_self_start_loop_length_is_measured(tmp_path, monkeypatch, coordinator, delta, arm):
    monkeypatch.setattr(P, "_self_loop_steps", lambda plan: plan.k + delta)
    t = 0.9 if arm.endswith("0.9") else 0.2
    stack = pi05_stack(tmp_path, arm, t=t, coordinator=coordinator)
    run_episode(stack, 3)
    rows, result = _admit(stack)
    assert {r["warm_reset"]["self_direct_nfe"] for r in rows if r["row_kind"] == "decision"} == {10 + delta}
    assert result["problems"]["self_direct_nfe_mismatch"] == 3


@pytest.mark.parametrize("coordinator", PATHS)
def test_a_snapshot_the_loop_never_reached_is_an_error_not_a_fallback(tmp_path, monkeypatch, coordinator):
    monkeypatch.setattr(P, "_self_loop_steps", lambda plan: plan.k - 1)  # capture index 9 (t = 0.1) never runs
    stack = pi05_stack(tmp_path, "selfwarmreset_t0.1", t=0.1, coordinator=coordinator)
    stack.served.on_task_begin()
    stack.served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    with pytest.raises(RuntimeError, match="never reached"):
        stack.served.infer(obs_for(stack))
    stack.served.on_episode_end(True)
    rows, result = _admit(stack, n=1)
    assert rows[0]["status"] == "error" and result["problems"]["decision_error"] == 1


@pytest.mark.parametrize("coordinator", PATHS)
def test_an_extra_continuation_call_is_counted(tmp_path, coordinator):
    stack = pi05_stack(tmp_path, "resetfinal_t0.2", coordinator=coordinator)
    executor = stack.parts.executor
    once = executor._continue

    def twice(*args):
        once(*args)
        return once(*args)

    executor._continue = twice
    run_episode(stack, 3)
    rows, result = _admit(stack)
    inner = rows[0]["warm_reset"]
    assert (inner["n_stage3_calls"], inner["continuation_nfe"]) == (2, 4)
    assert result["problems"]["extra_stage3_calls"] == 3 and result["problems"]["steps_mismatch"] == 3


@pytest.mark.parametrize("coordinator", PATHS)
@pytest.mark.parametrize("fault", ["extra", "uncounted"])
def test_self_start_calls_are_counted(tmp_path, coordinator, fault):
    stack = pi05_stack(tmp_path, "selfmidfinal_t0.2", coordinator=coordinator)
    executor = stack.parts.executor
    counted = executor._self_start
    if fault == "extra":
        def patched(run_stage3, stage2, noise, plan):
            counted(run_stage3, stage2, noise, plan)
            return counted(run_stage3, stage2, noise, plan)
    else:
        def patched(run_stage3, stage2, noise, plan):
            return run_stage3(stage2, noise, plan)  # the binding ran, nothing was counted
    executor._self_start = patched
    run_episode(stack, 3)
    rows, result = _admit(stack)
    inner = rows[0]["warm_reset"]
    if fault == "extra":
        assert (inner["self_start_calls"], inner["self_direct_nfe"]) == (2, 20)
    else:
        assert (inner["self_start_calls"], inner["self_direct_nfe"]) == (0, 0)
    assert result["problems"]["extra_self_start_calls"] == 3
    assert result["problems"]["self_direct_nfe_mismatch"] == 3


def test_two_connections_never_share_counters(tmp_path):
    a = pi05_stack(tmp_path / "a", "selfwarmreset_t0.2", coordinator=True)
    b = pi05_stack(tmp_path / "b", "warmreset_t0.2", coordinator=True)
    for stack in (a, b):
        stack.served.on_task_begin()
        stack.served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    for _ in range(2):  # interleaved requests
        a.served.infer(obs_for(a))
        b.served.infer(obs_for(b))
    for stack in (a, b):
        stack.served.on_episode_end(True)
    assert not +episode_problems(read_rows(a.served.evidence_path), expected=expected_for(a, n=2))["problems"]
    assert not +episode_problems(read_rows(b.served.evidence_path), expected=expected_for(b, n=2))["problems"]
