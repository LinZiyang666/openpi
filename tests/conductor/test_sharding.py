"""Sibling-shard fan-out: the partition, the eval-only refusal, and resume."""

from __future__ import annotations

import pytest

from openpi.conductor import task as T  # noqa: N812
from openpi.conductor.scheduler import EpisodeScheduler
from openpi.conductor.sharding import shard_eval_stage
from openpi.conductor.sharding import stride_partition

SERVERS = [T.ServerEndpoint("h", 23160 + i) for i in range(6)]


def _episodes(yaml_id: str, n: int, *, phase: str = "eval", server=SERVERS[0]):
    return [
        T.EpisodeTask(
            task_uid=T.make_task_uid(yaml_id, phase, i % 10, i),
            yaml_id=yaml_id,
            phase=phase,
            experiment="libero_spatial",
            task_id=i % 10,
            episode_idx=i,
            orig_init_state_idx=i,
            server_host=server.host,
            server_port=server.port,
            bundle_id=yaml_id,
        )
        for i in range(n)
    ]


# -- the partition itself ------------------------------------------------


@pytest.mark.parametrize(("n_items", "n_shards"), [(500, 6), (7, 6), (6, 6), (3, 6), (0, 6), (1, 1)])
def test_stride_partition_is_a_partition(n_items, n_shards):
    items = list(range(n_items))
    shards = stride_partition(items, n_shards)
    assert len(shards) == n_shards
    flat = [x for s in shards for x in s]
    assert sorted(flat) == items  # union is exactly the input
    assert len(flat) == len(set(flat))  # and nothing is duplicated
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_stride_partition_rejects_zero_shards():
    with pytest.raises(ValueError, match="shard count"):
        stride_partition([1, 2, 3], 0)


# -- what the helper refuses ---------------------------------------------


def test_warmup_episodes_are_refused():
    """A sharded warmup would keep 1/N of its calibration dump, silently."""
    with pytest.raises(ValueError, match="eval-only"):
        shard_eval_stage(
            stage_id="w",
            yaml_id="arm",
            episodes=_episodes("arm", 12, phase="warmup"),
            servers=SERVERS,
            episodes_are_idempotent=True,
        )


def test_no_servers_is_refused():
    with pytest.raises(ValueError, match="at least one server"):
        shard_eval_stage(
            stage_id="e",
            yaml_id="arm",
            episodes=_episodes("arm", 4),
            servers=[],
            episodes_are_idempotent=True,
        )


# -- the stages it builds ------------------------------------------------


def test_each_shard_is_retargeted_to_its_own_server():
    """A task still naming the original endpoint would send every worker to one box."""
    stages = shard_eval_stage(
        stage_id="eval__arm",
        yaml_id="arm",
        episodes=_episodes("arm", 500),
        servers=SERVERS,
        episodes_are_idempotent=True,
    )
    assert [s.server for s in stages] == SERVERS
    for stage, server in zip(stages, SERVERS, strict=True):
        assert {(ep.server_host, ep.server_port) for ep in stage.episodes} == {
            (server.host, server.port)
        }
    assert sum(len(s.episodes) for s in stages) == 500


def test_empty_shards_are_kept_and_reach_done():
    """Fewer episodes than servers must not strand the graph."""
    stages = shard_eval_stage(
        stage_id="eval__arm",
        yaml_id="arm",
        episodes=_episodes("arm", 3),
        servers=SERVERS,
        episodes_are_idempotent=True,
    )
    assert len(stages) == 6
    assert sum(1 for s in stages if not s.episodes) == 3

    graph = T.TaskGraph()
    for stage in stages:
        graph.add_stage(stage)
    graph.validate()
    sched = EpisodeScheduler(graph)
    for stage in stages:
        sched.mark_setup_running(stage.stage_id)
        sched.mark_setup_done(stage.stage_id)
    # The three empty siblings complete on setup alone; drain the rest.
    for stage in stages:
        for ep in stage.episodes:
            sched.next_task(stage.server.key)
            sched.mark_result(ep.task_uid, success=True, retriable=False, attempt=1)
    for stage in stages:
        sched.mark_complete_running(stage.stage_id)
        sched.mark_complete_done(stage.stage_id)
    assert sched.all_done()


def test_every_server_can_pull_from_one_sharded_arm():
    """The property the fan-out exists for: one arm, every server busy.

    The single-stage graph is the control -- it is what the scheduler does
    today, and only its owning server gets work.
    """
    sharded = T.TaskGraph()
    stages = shard_eval_stage(
        stage_id="eval__arm",
        yaml_id="arm",
        episodes=_episodes("arm", 60),
        servers=SERVERS,
        episodes_are_idempotent=True,
    )
    for stage in stages:
        sharded.add_stage(stage)
    sched = EpisodeScheduler(sharded)
    for stage in stages:
        sched.mark_setup_running(stage.stage_id)
        sched.mark_setup_done(stage.stage_id)
    served = {s.key for s in SERVERS if sched.next_task(s.key) is not None}
    assert served == {s.key for s in SERVERS}

    single = T.TaskGraph()
    single.add_stage(
        T.Stage("eval__arm", "arm", "eval", SERVERS[0], episodes=_episodes("arm", 60))
    )
    ctl = EpisodeScheduler(single)
    ctl.mark_setup_running("eval__arm")
    ctl.mark_setup_done("eval__arm")
    ctl_served = {s.key for s in SERVERS if ctl.next_task(s.key) is not None}
    assert ctl_served == {SERVERS[0].key}


def test_resume_is_idempotent_across_a_reshard():
    """A run resumed with a different shard count must not re-dispatch done work.

    ``make_task_uid`` does not encode the server, so the journal's uids still
    identify the same episodes after the topology changes underneath them.
    """
    episodes = _episodes("arm", 60)
    done = {ep.task_uid for ep in episodes[:40]}

    graph = T.TaskGraph()
    for stage in shard_eval_stage(
        stage_id="eval__arm",
        yaml_id="arm",
        episodes=episodes,
        servers=SERVERS[:4],
        episodes_are_idempotent=True,
    ):
        graph.add_stage(stage)
    graph.validate()
    sched = EpisodeScheduler(graph)
    sched.mark_preexisting_done(done)
    for sid in list(graph.stages):
        sched.mark_setup_running(sid)
        sched.mark_setup_done(sid)

    dispatched = []
    for _ in range(60):
        for server in SERVERS[:4]:
            task = sched.next_task(server.key)
            if task is not None:
                dispatched.append(task.task_uid)
    assert not (set(dispatched) & done)
    assert set(dispatched) == {ep.task_uid for ep in episodes[40:]}


def test_a_caller_that_does_not_assert_idempotence_is_refused():
    """phase is not a sufficient test -- collection stages are eval too."""
    with pytest.raises(ValueError, match="episodes_are_idempotent"):
        shard_eval_stage(
            stage_id="e",
            yaml_id="arm",
            episodes=_episodes("arm", 12),
            servers=SERVERS,
            episodes_are_idempotent=False,
        )
