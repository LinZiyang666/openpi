"""Runner-side identity passthrough and per-step router provenance (X14).

The RL trainer joins three sources — the server's feature dump, the conductor
journal, and these client rows — on
``(task_uid, attempt, batch_id, weights_version, decision_idx)``. Two runner
obligations make that join possible:

  1. ``episode_start`` must carry the identity the server needs to name its
     shard, with ``task_uid`` / ``attempt`` taken from the *dispatched task*.
     ``EpisodeTask.extra`` is built once by the strategy and is not rewritten on
     a requeue (the scheduler bumps the generation via ``dataclasses.replace``),
     so trusting a stamped copy would mislabel every dumped step of a retried
     episode.
  2. each per-step row must relay ``router_outputs``, whose ``decision_idx`` is
     the join key — this row's own ``step_idx`` is the physical env step and
     advances by ``replan_steps`` between inference calls.

These drive the REAL ``LiberoEpisodeRunner.run()`` with a stubbed client, the
same way ``worker_entry`` invokes it.
"""

from types import SimpleNamespace

import pytest


class _RecordingClient:
    def __init__(self) -> None:
        self.episode_start_kwargs: list[dict] = []

    def select_bundle(self, bundle_id):
        pass

    def episode_start(self, **kwargs):
        self.episode_start_kwargs.append(kwargs)

    def episode_end(self, **kwargs):
        pass

    def close(self):
        pass


def _make_runner(run_episode_fn, client=None):
    from examples.libero.episode_runner import LiberoEpisodeRunner

    client = client or _RecordingClient()
    runner = LiberoEpisodeRunner(
        SimpleNamespace(seed=42, num_trials_per_task=50),
        episode_setup=lambda t: (None, None, "desc", 5),
        client_factory=lambda server: client,
        run_episode_fn=run_episode_fn,
    )
    return runner, client


def _task(**kw):
    d = dict(
        task_uid="y:eval:3:5", yaml_id="y", experiment="libero", task_id=3,
        episode_idx=5, orig_init_state_idx=7, phase="eval", attempt=1, bundle_id="b",
        server=SimpleNamespace(key="s1"), extra={"num_trials_per_task": 10},
    )
    d.update(kw)
    return SimpleNamespace(**d)


def _run_ep_emitting(metas):
    def run_ep(env, client, init, desc, args, max_steps, *, infer_recorder, step_callback):
        for step_idx, meta in metas:
            infer_recorder(step_idx, meta)
        return (True, None, None, None, len(metas))
    return run_ep


def _steps(result):
    return [r for r in result.per_step_rows if r.get("_kind") is None]


# ---------------------------------------------------------------------------
# episode_start identity
# ---------------------------------------------------------------------------


def test_router_identity_keys_are_forwarded_to_episode_start():
    runner, client = _make_runner(_run_ep_emitting([]))
    task = _task(extra={
        "num_trials_per_task": 10, "run_id": "run0", "batch_id": "b7",
        "weights_version": "v3",
    })
    runner.run(task, report=lambda *a, **k: None)

    extra = client.episode_start_kwargs[0]["extra_metadata"]
    assert extra["run_id"] == "run0"
    assert extra["batch_id"] == "b7"
    assert extra["weights_version"] == "v3"
    # Authoritative from the dispatched task, not from extra.
    assert extra["task_uid"] == "y:eval:3:5"
    assert extra["attempt"] == 1


def test_attempt_comes_from_the_dispatched_task_not_the_stale_extra():
    """A requeued dispatch carries a higher generation on the task itself; the
    strategy's ``extra`` still holds whatever it was built with."""
    runner, client = _make_runner(_run_ep_emitting([]))
    runner.run(_task(attempt=4), report=lambda *a, **k: None)
    assert client.episode_start_kwargs[0]["extra_metadata"]["attempt"] == 4


def test_conflicting_identity_in_extra_fails_loud():
    """Silently preferring one side would mislabel an entire episode's dump."""
    runner, _ = _make_runner(_run_ep_emitting([]))
    task = _task(extra={"num_trials_per_task": 10, "task_uid": "y:eval:9:9"})
    with pytest.raises(ValueError, match="conflicts with"):
        runner.run(task, report=lambda *a, **k: None)

    runner2, _ = _make_runner(_run_ep_emitting([]))
    task2 = _task(attempt=3, extra={"num_trials_per_task": 10, "attempt": 1})
    with pytest.raises(ValueError, match="conflicts with"):
        runner2.run(task2, report=lambda *a, **k: None)


def test_agreeing_duplicate_identity_in_extra_is_accepted():
    runner, client = _make_runner(_run_ep_emitting([]))
    task = _task(extra={"num_trials_per_task": 10, "task_uid": "y:eval:3:5", "attempt": 1})
    runner.run(task, report=lambda *a, **k: None)
    assert client.episode_start_kwargs[0]["extra_metadata"]["task_uid"] == "y:eval:3:5"


def test_non_router_runs_keep_their_episode_start_payload():
    """No router keys in extra => nothing new beyond the identity the dispatched
    task already defines; unrelated extra entries never leak onto the wire."""
    runner, client = _make_runner(_run_ep_emitting([]))
    runner.run(_task(extra={"num_trials_per_task": 10, "unrelated": "x"}),
               report=lambda *a, **k: None)
    extra = client.episode_start_kwargs[0]["extra_metadata"]
    assert extra == {
        "task_id": 3, "orig_init_state_idx": 7,
        "task_uid": "y:eval:3:5", "attempt": 1,
    }


# ---------------------------------------------------------------------------
# per-step rows
# ---------------------------------------------------------------------------


def test_router_outputs_column_is_relayed():
    router_outputs = {
        "decision_idx": 0, "arm_sampled": "student", "arm_executed": "student",
        "probs": [0.3, 0.7], "temperature": 1.0, "weights_version": "v3",
        "seed_ep": 12345, "fallback": False,
    }
    runner, _ = _make_runner(_run_ep_emitting([
        (0, {"hit_type": "FULL_HIT", "router_outputs": router_outputs, "executor": "override"}),
    ]))
    (row,) = _steps(runner.run(_task(), report=lambda *a, **k: None))
    assert row["router_outputs"] == router_outputs
    assert row["executor"] == "override"


def test_decision_idx_is_dense_while_step_idx_jumps_by_replan_steps():
    """The reason the join key is server-side: the client's index skips."""
    replan = 5
    metas = [
        (i * replan, {"hit_type": "MISS", "router_outputs": {"decision_idx": i}})
        for i in range(4)
    ]
    rows = _steps(_make_runner(_run_ep_emitting(metas))[0].run(
        _task(), report=lambda *a, **k: None))
    assert [r["step_idx"] for r in rows] == [0, 5, 10, 15]
    assert [r["router_outputs"]["decision_idx"] for r in rows] == [0, 1, 2, 3]


def test_non_router_rows_carry_a_none_column():
    """Additive column: existing yamls keep every previous field and gain one
    None, so downstream readers that ignore it are unaffected."""
    runner, _ = _make_runner(_run_ep_emitting([(0, {"hit_type": "MISS", "cp1_score": None})]))
    (row,) = _steps(runner.run(_task(), report=lambda *a, **k: None))
    assert row["router_outputs"] is None
    for key in ("yaml_id", "task_id", "subset_init_state_idx", "orig_init_state_idx",
                "episode_id", "task_uid", "phase", "step_idx", "hit_type", "start_t",
                "winner_id", "cp1_score", "searched", "executor"):
        assert key in row
