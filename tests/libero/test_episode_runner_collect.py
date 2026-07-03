"""Tests for gate-collection wiring in the conductor episode runner.

These drive the REAL ``LiberoEpisodeRunner.run()`` path (stubbed client /
episode_setup / run_episode_fn) so the conductor vision fail-fast and the
episode-summary provenance are exercised the way ``worker_entry`` invokes them,
not a synthetic ``__init__`` condition the real worker never sets.
"""

from types import SimpleNamespace

import pytest


def _make_runner(run_episode_fn, args=None):
    from examples.libero.episode_runner import LiberoEpisodeRunner

    args = args or SimpleNamespace(seed=42, num_trials_per_task=50)
    client = SimpleNamespace(
        select_bundle=lambda b: None,
        episode_start=lambda **k: None,
        episode_end=lambda **k: None,
        close=lambda: None,
    )
    return LiberoEpisodeRunner(
        args,
        episode_setup=lambda t: (None, None, "desc", 5),
        client_factory=lambda server: client,
        run_episode_fn=run_episode_fn,
    )


def _task(**kw):
    # extra carries the REAL per-phase trial count (N=10 here, deliberately != the
    # worker's main.Args default of 50) so tests prove the runner reads extra, not
    # the unrelated worker default.
    d = dict(
        task_uid="y:eval:3:5", yaml_id="y", experiment="libero", task_id=3,
        episode_idx=5, orig_init_state_idx=7, phase="eval", attempt=1, bundle_id="b",
        server=SimpleNamespace(key="s1"), extra={"num_trials_per_task": 10},
    )
    d.update(kw)
    return SimpleNamespace(**d)


def test_conductor_vision_failfast_in_real_loop():
    """A vision field in the collect payload must raise out of run() (un-swallowable)."""
    def run_ep(env, client, init, desc, args, max_steps, *, infer_recorder, step_callback):
        infer_recorder(0, {"hit_type": "MISS"},
                       {"searched": True, "collect": {"robot_state": [1.0], "vision_0": [[0.1]]}})
        return (False, None, None, None, 1)  # episode "fails" but the vision must still raise

    runner = _make_runner(run_ep)
    with pytest.raises(ValueError, match="vision"):
        runner.run(_task(), report=lambda *a, **k: None)


def test_conductor_robot_state_ok_writes_summary():
    def run_ep(env, client, init, desc, args, max_steps, *, infer_recorder, step_callback):
        infer_recorder(0, {"hit_type": "MISS"},
                       {"searched": True, "collect": {"robot_state": [1.0]}, "kb_id": "cp1_mean_pool"})
        return (True, None, None, None, 1)

    result = _make_runner(run_ep).run(_task(), report=lambda *a, **k: None)
    rows = result.per_step_rows
    steps = [r for r in rows if r.get("_kind") != "episode_summary"]
    summaries = [r for r in rows if r.get("_kind") == "episode_summary"]
    # episode_id uses the REAL per-phase N from EpisodeTask.extra (10), NOT the
    # worker's main.Args default (50). This is the B1 regression: conductor and
    # standalone must agree on the canonical id.
    from examples.libero.collect_util import compute_global_episode_id

    expect = compute_global_episode_id(3, 5, 10)  # extra N=10
    assert expect == 3 * 10 + 5
    assert expect != compute_global_episode_id(3, 5, 50)  # worker default NOT used
    assert steps[0]["episode_id"] == expect
    assert steps[0]["episode_id"] != steps[0]["subset_init_state_idx"]
    assert steps[0]["task_uid"] == "y:eval:3:5"
    assert steps[0]["collector_schema_version"] == 1
    assert len(summaries) == 1
    s = summaries[0]
    assert s["episode_id"] == expect
    assert s["kb_id"] == "cp1_mean_pool"
    assert s["seed"] == 42
    assert s["searched_all"] is True
    assert s["collect_fields"] == ["robot_state"]


def test_hit_row_global_episode_id():
    from examples.libero.collect_util import compute_global_episode_id
    from examples.libero.episode_runner import _global_episode_id, _hit_row

    task = _task(task_id=1, episode_idx=3)
    row = _hit_row(task, 2, {"hit_type": "MISS"}, 50)
    assert row["subset_init_state_idx"] == 3
    assert row["step_idx"] == 2
    # Matches the standalone canonical formula (task_id incorporated).
    assert row["episode_id"] == _global_episode_id(task, 50) == compute_global_episode_id(1, 3, 50)
    assert row["episode_id"] != row["subset_init_state_idx"]  # global != subset
    assert "episode_idx" not in row and "step" not in row


def _collect_run_ep(env, client, init, desc, args, max_steps, *, infer_recorder, step_callback):
    infer_recorder(0, {"hit_type": "MISS"}, {"searched": True, "collect": {"robot_state": [1.0]}})
    return (True, None, None, None, 1)


def test_conductor_warmup_and_eval_use_distinct_per_phase_trial_count():
    """B1: warmup (N=2) and eval (N=10) tasks with the SAME (task_id, episode_idx)
    map to DIFFERENT canonical episode_ids because each carries its own per-phase N
    in extra — the single default the worker knows (50) would collide them."""
    from examples.libero.collect_util import compute_global_episode_id

    warm = _task(task_uid="y:warmup:4:1", phase="warmup", task_id=4, episode_idx=1,
                 extra={"num_trials_per_task": 2})
    evl = _task(task_uid="y:eval:4:1", phase="eval", task_id=4, episode_idx=1,
                extra={"num_trials_per_task": 10})
    warm_rows = _make_runner(_collect_run_ep).run(warm, report=lambda *a, **k: None).per_step_rows
    eval_rows = _make_runner(_collect_run_ep).run(evl, report=lambda *a, **k: None).per_step_rows
    warm_id = warm_rows[0]["episode_id"]
    eval_id = eval_rows[0]["episode_id"]
    assert warm_id == compute_global_episode_id(4, 1, 2)   # 9
    assert eval_id == compute_global_episode_id(4, 1, 10)  # 41
    assert warm_id != eval_id


def test_conductor_missing_trial_count_fails_fast():
    """B1: a task with no extra['num_trials_per_task'] must raise, never silently
    fall back to the worker's unrelated default N=50."""
    bad = _task(extra={})
    with pytest.raises(ValueError, match="num_trials_per_task"):
        _make_runner(_collect_run_ep).run(bad, report=lambda *a, **k: None)
