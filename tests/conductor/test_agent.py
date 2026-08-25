"""WorkerAgent supervision tests (M3) — fork + respawn, with injected spawn."""

from __future__ import annotations

import openpi.conductor.agent as _agent
from openpi.conductor.agent import WorkerAgent
from openpi.conductor.agent import WorkerSpec


class FakeHandle:
    """A process handle that reports alive for ``alive_polls`` polls, then exits."""

    def __init__(self, alive_polls: int = 100):
        self._polls = alive_polls
        self.terminated = False

    def poll(self):
        if self._polls > 0:
            self._polls -= 1
            return None
        return 0  # exited

    def terminate(self):
        self.terminated = True


def _specs(n):
    return [WorkerSpec(worker_id=f"w{i}", server_key="h:8000", gpu_id=str(i)) for i in range(n)]


def test_agent_spawns_all_workers():
    spawned: list[str] = []

    def spawn(spec, host, port):
        spawned.append(spec.worker_id)
        return FakeHandle(alive_polls=100)

    agent = WorkerAgent(_specs(3), "dh", 9000, spawn_fn=spawn)
    agent.start()
    assert sorted(spawned) == ["w0", "w1", "w2"]


def test_agent_respawns_dead_worker():
    counts: dict[str, int] = {}

    def spawn(spec, host, port):
        counts[spec.worker_id] = counts.get(spec.worker_id, 0) + 1
        # first spawn dies immediately; the respawn stays alive
        return FakeHandle(alive_polls=0 if counts[spec.worker_id] == 1 else 100)

    agent = WorkerAgent(_specs(1), "dh", 9000, spawn_fn=spawn)
    agent.start()  # spawn #1 (will report dead)
    agent.supervise_once()  # detect dead -> respawn #2 (alive)
    agent.supervise_once()  # alive -> no respawn
    assert counts["w0"] == 2
    assert agent.restart_counts["w0"] == 1


def test_agent_stop_terminates_handles():
    handles: list[FakeHandle] = []

    def spawn(spec, host, port):
        h = FakeHandle(alive_polls=100)
        handles.append(h)
        return h

    agent = WorkerAgent(_specs(2), "dh", 9000, spawn_fn=spawn)
    agent.start()
    agent.stop()
    assert all(h.terminated for h in handles)


# ----------------------------------------------------------------------
# _default_spawn: conda_env path (LIBERO sim env) vs plain python
# ----------------------------------------------------------------------
def test_default_spawn_conda_env_builds_conda_cmd(monkeypatch):
    import openpi.conductor.agent as agent_mod

    captured: dict = {}

    def fake_popen(cmd, env=None, **kwargs):
        captured["cmd"], captured["env"] = cmd, env
        captured["start_new_session"] = kwargs.get("start_new_session")
        return FakeHandle()

    monkeypatch.setattr(_agent.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("VIRTUAL_ENV", "/some/uv/venv")
    monkeypatch.setenv("PYTHONPATH", "/uv/inject")

    spec = WorkerSpec("w0", "host:8000", "3", conda_env="/scratch/zixuans8/libero_sim")
    _agent._default_spawn(spec, "dh", 9000)
    cmd, env = captured["cmd"], captured["env"]

    # conda run -p <abs prefix> python -m worker_entry ...
    assert cmd[:5] == ["conda", "run", "--no-capture-output", "-p", "/scratch/zixuans8/libero_sim"]
    assert "examples.libero.worker_entry" in cmd
    # env hardening: uv-venv injections stripped, headless GL, GPU pin, repo paths
    assert "VIRTUAL_ENV" not in env
    assert env["MUJOCO_GL"] == "egl"
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert agent_mod._SRC_DIR in env["PYTHONPATH"]
    assert agent_mod._REPO_ROOT in env["PYTHONPATH"]
    assert "/uv/inject" not in env["PYTHONPATH"]  # uv PYTHONPATH not inherited
    # Spawn into a new session/process group so stop() can signal the whole
    # group (the conda wrapper + the real grandchild worker).
    assert captured["start_new_session"] is True


def test_default_spawn_no_conda_uses_plain_python(monkeypatch):

    captured: dict = {}

    def fake_popen(cmd, env=None, **kwargs):
        captured["cmd"], captured["env"] = cmd, env
        captured["start_new_session"] = kwargs.get("start_new_session")
        return FakeHandle()

    monkeypatch.setattr(_agent.subprocess, "Popen", fake_popen)
    spec = WorkerSpec("w0", "host:8000", "0")  # no conda_env (default)
    _agent._default_spawn(spec, "dh", 9000)
    assert captured["cmd"][0] == "python"
    assert "conda" not in captured["cmd"]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"


# ----------------------------------------------------------------------
# _default_spawn: init-state pool forwarding
# ----------------------------------------------------------------------
def _spawn_capture(monkeypatch, spec):

    captured: dict = {}

    def fake_popen(cmd, env=None, **kwargs):
        captured["cmd"] = cmd
        return FakeHandle()

    monkeypatch.setattr(_agent.subprocess, "Popen", fake_popen)
    _agent._default_spawn(spec, "dh", 9000)
    return captured["cmd"]


def test_default_spawn_forwards_the_init_states_dir(monkeypatch):
    """A second, disjoint init pool must reach the worker's CLI."""
    spec = WorkerSpec("w0", "host:8000", "0", init_states_dir="exp/common/data/db_init/libero/libero_10")
    cmd = _spawn_capture(monkeypatch, spec)
    i = cmd.index("--init-states-dir")
    assert cmd[i + 1] == "exp/common/data/db_init/libero/libero_10"


def test_default_spawn_omits_the_flag_when_the_pool_is_unset(monkeypatch):
    """Default stays byte-identical, so existing callers keep the LIBERO pool."""
    cmd = _spawn_capture(monkeypatch, WorkerSpec("w0", "host:8000", "0"))
    assert "--init-states-dir" not in cmd


def test_rollout_knobs_are_forwarded_only_when_set(monkeypatch):
    """A GR00T fleet needs --resize-size 256; omitting it fails every episode.

    The failure is late and uniform: the fleet comes up, connects, and only
    then does the wire contract reject each 224 frame -- so the flag has to be
    expressible where the worker is described, not discovered afterwards.
    """
    seen: dict = {}

    class _P:
        def poll(self):
            return None

        def terminate(self):
            return None

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw.get("env", {})
        return _P()

    monkeypatch.setattr(_agent.subprocess, "Popen", fake_popen)

    bare = WorkerSpec(worker_id="w0", server_key="h:8000", gpu_id="0")
    _agent._default_spawn(bare, "h", 9000)
    assert "--resize-size" not in seen["cmd"]
    assert "--replan-steps" not in seen["cmd"]

    groot = WorkerSpec(
        worker_id="w0",
        server_key="h:8000",
        gpu_id="3",
        resize_size=256,
        replan_steps=5,
        env={"MUJOCO_EGL_DEVICE_ID": "3"},
    )
    _agent._default_spawn(groot, "h", 9000)
    assert seen["cmd"][seen["cmd"].index("--resize-size") + 1] == "256"
    assert seen["cmd"][seen["cmd"].index("--replan-steps") + 1] == "5"
    # CUDA_VISIBLE_DEVICES steers the policy client; EGL picks its render
    # device independently, so the caller's value must survive.
    assert seen["env"]["MUJOCO_EGL_DEVICE_ID"] == "3"
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "3"
