"""WorkerAgent supervision tests (M3) — fork + respawn, with injected spawn."""

from __future__ import annotations

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
