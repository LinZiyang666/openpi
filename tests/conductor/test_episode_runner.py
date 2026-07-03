"""LiberoEpisodeRunner adapter tests (M4) — control flow, no LIBERO/CUDA.

Reuses main._run_episode via an injected fake; verifies connection reuse,
bundle binding, episode lifecycle, progress reporting, and per-step collection.
"""

from __future__ import annotations

from examples.libero.episode_runner import LiberoEpisodeRunner
from openpi.conductor import task as T  # noqa: N812

S1 = T.ServerEndpoint("127.0.0.1", 8001)
S2 = T.ServerEndpoint("127.0.0.1", 8002)


class FakeClient:
    def __init__(self, server):
        self.server = server
        self.calls: list[tuple] = []

    def select_bundle(self, bundle_id):
        self.calls.append(("select", bundle_id))

    def episode_start(self, **kw):
        self.calls.append(("start", kw["task"]))

    def episode_end(self, success):
        self.calls.append(("end", success))

    def close(self):
        self.calls.append(("close",))


def _ep(yaml_id="y", phase="eval", task_id=0, idx=0, server=S1, bundle="y", trials=10):
    # extra carries the per-phase trial count the strategy stamps (the runner
    # needs it for the canonical global episode_id; §19.B6).
    return T.EpisodeTask(
        task_uid=T.make_task_uid(yaml_id, phase, task_id, idx),
        yaml_id=yaml_id,
        phase=phase,
        experiment="libero_spatial",
        task_id=task_id,
        episode_idx=idx,
        orig_init_state_idx=idx,
        server_host=server.host,
        server_port=server.port,
        bundle_id=bundle,
        extra={"num_trials_per_task": trials},
    )


def _setup(task):
    # (env, initial_state, task_description, max_steps)
    return (None, None, f"task-{task.task_id}", 10)


def _fake_run_episode(env, client, initial_state, desc, args, max_steps, *, infer_recorder, step_callback):
    infer_recorder(0, {"hit_type": "MISS", "start_t": None, "winner_id": None, "cp1_score": None})
    step_callback(0)
    infer_recorder(1, {"hit_type": "WARM_START", "start_t": 0.5, "winner_id": "w", "cp1_score": 0.9})
    step_callback(1)
    return (True, [], [], None, 2)


def _make_runner(clients: dict):
    def factory(server):
        clients[server.key] = FakeClient(server)
        return clients[server.key]

    return LiberoEpisodeRunner(
        args=object(),
        episode_setup=_setup,
        client_factory=factory,
        run_episode_fn=_fake_run_episode,
    )


def test_run_collects_per_step_and_result():
    clients = {}
    runner = _make_runner(clients)
    progress: list = []
    result = runner.run(_ep(), lambda s, r, h: progress.append((s, h)))
    assert result.success is True
    assert result.n_steps == 2
    assert len(result.per_step_rows) == 2
    assert result.per_step_rows[1]["hit_type"] == "WARM_START"
    assert progress  # step_callback fired


def test_episode_lifecycle_and_bundle_bind():
    clients = {}
    runner = _make_runner(clients)
    runner.run(_ep(bundle="y"), lambda *a: None)
    c = clients[S1.key]
    kinds = [k[0] for k in c.calls]
    assert kinds[0] == "select"
    assert "start" in kinds
    assert "end" in kinds
    assert c.calls[0] == ("select", "y")


def test_client_reused_same_server_bundle():
    clients = {}
    factory_calls = []

    def factory(server):
        factory_calls.append(server.key)
        clients[server.key] = FakeClient(server)
        return clients[server.key]

    runner = LiberoEpisodeRunner(
        args=object(), episode_setup=_setup, client_factory=factory, run_episode_fn=_fake_run_episode
    )
    runner.run(_ep(idx=0, bundle="y"), lambda *a: None)
    runner.run(_ep(idx=1, bundle="y"), lambda *a: None)
    assert factory_calls == [S1.key]  # one connection reused
    selects = [c for c in clients[S1.key].calls if c[0] == "select"]
    assert len(selects) == 1  # bundle bound once


def test_bundle_reselect_on_change():
    clients = {}
    runner = _make_runner(clients)
    runner.run(_ep(idx=0, bundle="y1"), lambda *a: None)
    runner.run(_ep(idx=1, bundle="y2"), lambda *a: None)
    selects = [c for c in clients[S1.key].calls if c[0] == "select"]
    assert selects == [("select", "y1"), ("select", "y2")]
