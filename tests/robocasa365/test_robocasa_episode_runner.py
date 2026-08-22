"""Non-manual tests for RobocasaEpisodeRunner: contracts, seed, lifecycle, liveness.

Every collaborator (client, env, gym.make, horizon, handshake probe) is
injected; nothing here touches the simulator, a GPU, or the network.

The pi0.5 observation contract is checked against the ARCHIVED baseline by
re-parsing ``exp/robocasa365/baselines/pi05_step0b_client_ORIGINAL.py`` as
text: transcribing the expected order into this file as a second literal would
let the test and the implementation drift together and certify each other (the
``sys.modules``-fake failure shape).
"""

from __future__ import annotations

import pathlib
import re
import threading
import time
from typing import Any

import numpy as np
import pytest

from openpi.conductor.task import EpisodeTask, ServerEndpoint

from exp.robocasa365.episode_runner import (
    PI05_STATE_CONCAT_ORDER,
    PROMPT_SOURCE_KEY,
    GrootTeacherAdapter,
    Pi05TeacherAdapter,
    RobocasaEpisodeRunner,
    WatchdogRunner,
)

BASELINE = pathlib.Path(__file__).resolve().parents[2] / "exp" / "robocasa365" / "baselines" / "pi05_step0b_client_ORIGINAL.py"

PROMPT = "open the cabinet door"


def _make_task(**extra_overrides: Any) -> EpisodeTask:
    extra = {
        "task_name": "OpenCabinet",
        "layout": 1,
        "style": 1,
        "teacher": "pi05",
        "base_seed": 100,
        "replan_steps": 2,
    }
    extra.update(extra_overrides)
    extra = {k: v for k, v in extra.items() if v is not None}
    return EpisodeTask(
        task_uid="collect_l1s1_pi05__OpenCabinet:eval:0:7",
        yaml_id="collect_l1s1_pi05__OpenCabinet",
        phase="eval",
        experiment="pi05",
        task_id=0,
        episode_idx=7,
        orig_init_state_idx=7,
        server_host="127.0.0.1",
        server_port=8010,
        bundle_id="default",
        attempt=1,
        extra=extra,
    )


def _env_obs() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    obs = {key: rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8) for key in (
        "video.robot0_agentview_left", "video.robot0_eye_in_hand", "video.robot0_agentview_right",
    )}
    obs["state.end_effector_position_relative"] = np.array([1.0, 2.0, 3.0])
    obs["state.end_effector_rotation_relative"] = np.array([4.0, 5.0, 6.0, 7.0])
    obs["state.base_position"] = np.array([8.0, 9.0, 10.0])
    obs["state.base_rotation"] = np.array([11.0, 12.0, 13.0, 14.0])
    obs["state.gripper_qpos"] = np.array([15.0, 16.0])
    obs[PROMPT_SOURCE_KEY] = PROMPT
    return obs


class _FakeEnv:
    def __init__(self, succeed_at: int | None = None) -> None:
        self.reset_seeds: list[int | None] = []
        self.steps = 0
        self.closed = False
        self._succeed_at = succeed_at

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return _env_obs(), {}

    def step(self, action):
        del action
        self.steps += 1
        success = self._succeed_at is not None and self.steps >= self._succeed_at
        return _env_obs(), 0.0, False, False, {"success": success}

    def close(self):
        self.closed = True


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def select_bundle(self, bundle_id):
        self.calls.append(("select_bundle", {"bundle_id": bundle_id}))

    def episode_start(self, **kwargs):
        self.calls.append(("episode_start", kwargs))

    def infer(self, payload):
        self.calls.append(("infer", payload))
        return {"actions": np.zeros((50, 12)), "__hit_meta__": {"hit_type": "MISS"}}

    def episode_end(self, success):
        self.calls.append(("episode_end", {"success": success}))
        return {}

    def close(self):
        self.closed = True

    def named(self, name):
        return [payload for call, payload in self.calls if call == name]


class _FakeAdapter:
    """Minimal adapter: raw action vectors, no conversion, no extra env kwargs."""

    def env_kwargs(self):
        return {}

    def build_observation(self, obs, prompt):
        return {"prompt": prompt, "obs_keys": sorted(obs)[:3]}

    def iter_actions(self, response, replan_steps):
        return iter(list(response["actions"])[:replan_steps])


def _runner(client=None, env=None, horizon=5, probe=None, **kwargs):
    client = client or _FakeClient()
    env = env or _FakeEnv()
    made = []

    def gym_make(task_name, layout, style, **kw):
        made.append((task_name, layout, style, kw))
        return env

    runner = RobocasaEpisodeRunner(
        _FakeAdapter(),
        client_factory=lambda server: client,
        gym_make=gym_make,
        horizon_fn=lambda name: horizon,
        handshake_probe=probe or (lambda server, timeout_s: None),
        connect_deadline_s=0.05,
        connect_retries=2,
        **kwargs,
    )
    return runner, client, env, made


def _report(step, rate, hit):  # noqa: ARG001 - ProgressCallback shape
    return None


# ------------------------------------------------------------------
# §4.3.1a frozen observation/action contracts
# ------------------------------------------------------------------


def test_pi05_state_order_matches_archived_baseline_text():
    source = BASELINE.read_text()
    # Parse the np.concatenate((...)) block of the archive: the obs["state.*"]
    # keys in their literal order.
    concat = re.search(r"np\.concatenate\(\((.*?)\), axis=0\)", source, re.DOTALL)
    assert concat, "baseline archive lost its state-concat block"
    archived_order = re.findall(r'obs\["(state\.[a-z_]+)"\]', concat.group(1))
    assert tuple(archived_order) == PI05_STATE_CONCAT_ORDER


def test_pi05_build_observation_contract():
    adapter = Pi05TeacherAdapter(convert_action=lambda vec: ("converted", vec))
    payload = adapter.build_observation(_env_obs(), PROMPT)
    assert set(payload) == {
        "observation/image", "observation/wrist_image", "observation/right_image",
        "observation/state", "prompt",
    }
    # Language: the natural-language instruction, NOT the canonical task name.
    assert payload["prompt"] == PROMPT
    # 16-dim state in the archived order: 3+4+3+4+2.
    assert payload["observation/state"].shape == (16,)
    assert np.array_equal(payload["observation/state"], np.arange(1.0, 17.0))
    for key in ("observation/image", "observation/wrist_image", "observation/right_image"):
        assert payload[key].shape == (224, 224, 3)
        assert payload[key].dtype == np.uint8


def test_pi05_actions_pass_through_convert_action():
    seen = []
    adapter = Pi05TeacherAdapter(convert_action=lambda vec: seen.append(vec) or ("env_action", len(seen)))
    out = list(adapter.iter_actions({"actions": np.zeros((50, 12))}, 5))
    assert len(out) == len(seen) == 5
    assert out[0] == ("env_action", 1)


def test_render_resolution_difference_is_preserved():
    # Frozen: the pi0.5 baseline passed NO camera sizes; GR00T renders at 512.
    assert Pi05TeacherAdapter(convert_action=lambda v: v).env_kwargs() == {}
    groot_kwargs = GrootTeacherAdapter().env_kwargs()
    assert groot_kwargs == {"camera_heights": 512, "camera_widths": 512}


# ------------------------------------------------------------------
# Runner lifecycle / identity
# ------------------------------------------------------------------


def test_episode_start_uses_canonical_task_name_and_attempt_suffixed_name():
    runner, client, _env, _ = _runner()
    runner.run(_make_task(), _report)
    (start,) = client.named("episode_start")
    # Canonical env name — becomes the h5 ``task`` attr and later the cache
    # task_key. The language prompt must NOT appear here.
    assert start["task"] == "OpenCabinet"
    assert start["task"] != PROMPT
    assert start["experiment"] == "pi05"
    assert start["episode_name"] == "OpenCabinet/episode_0007_a01"
    assert start["extra_metadata"]["task_uid"] == "collect_l1s1_pi05__OpenCabinet:eval:0:7"


def test_reset_seed_is_base_seed_plus_offset():
    runner, _client, env, _ = _runner()
    runner.run(_make_task(), _report)
    assert env.reset_seeds == [107]  # 100 + orig_init_state_idx 7


def test_missing_extra_key_raises():
    runner, _c, _e, _ = _runner()
    with pytest.raises(ValueError, match="replan_steps"):
        runner.run(_make_task(replan_steps=None), _report)


def test_env_cached_per_scene_and_closed_once():
    runner, _client, env, made = _runner()
    runner.run(_make_task(), _report)
    runner.run(_make_task(), _report)
    assert len(made) == 1  # same (task, layout, style) -> one gym.make
    runner.close()
    assert env.closed


def test_episode_end_sent_on_exception_path():
    runner, client, _env, _ = _runner()

    class _Boom(_FakeAdapter):
        def iter_actions(self, response, replan_steps):
            raise RuntimeError("mid-episode failure")

    runner._adapter = _Boom()  # noqa: SLF001
    with pytest.raises(RuntimeError):
        runner.run(_make_task(), _report)
    # Without this the collector never flushes and the h5 silently vanishes.
    assert client.named("episode_end") == [{"success": False}]


def test_per_step_rows_are_small_summaries():
    runner, _client, _env, _ = _runner()
    result = runner.run(_make_task(), _report)
    for row in result.per_step_rows:
        for value in row.values():
            assert not isinstance(value, np.ndarray)


# ------------------------------------------------------------------
# Liveness (§4.3.5): five cases
# ------------------------------------------------------------------


def test_liveness_1_unreachable_server_raises_bounded():
    attempts = []

    def probe(server, timeout_s):
        attempts.append(server.key)
        raise OSError("connection refused")

    runner, _c, _e, _ = _runner(probe=probe)
    with pytest.raises(ConnectionError, match="bounded"):
        runner.run(_make_task(), _report)
    assert len(attempts) == 2  # connect_retries — bounded, not infinite


def _hanging_runner(hang_event: threading.Event):
    """An inner runner that blocks until its client is closed (recv-unblock model)."""

    class _Hanging(RobocasaEpisodeRunner):
        def __init__(self):
            self.closed = threading.Event()

        def run(self, task, report):
            del task, report
            if not hang_event.wait(timeout=10.0):
                raise TimeoutError("test hang never released")
            raise ConnectionError("socket closed by watchdog")

        def close_current_client(self):
            self.closed.set()
            hang_event.set()

        def close(self):
            pass

    return _Hanging()


def test_liveness_2_hang_in_client_construction_exits_within_deadline_plus_grace():
    # Simulates the post-probe race: stuck inside the client constructor, so
    # close_current_client has nothing to close — L3 must fire.
    exits = []

    class _StuckInCtor(RobocasaEpisodeRunner):
        def __init__(self):
            self._released = threading.Event()

        def run(self, task, report):
            del task, report
            self._released.wait(timeout=10.0)
            raise ConnectionError("released")

        def close_current_client(self):  # nothing exists yet: no-op
            return None

        def close(self):
            pass

    inner = _StuckInCtor()
    watchdog = WatchdogRunner(
        inner, episode_deadline_s=0.05, terminate_grace_s=0.05,
        exit_fn=lambda code: (exits.append(code), inner._released.set()),  # noqa: SLF001
    )
    with pytest.raises(ConnectionError):
        watchdog.run(_make_task(), _report)
    assert exits == [3]


def test_liveness_3_and_4_mid_episode_hang_released_by_close():
    # infer-hang and episode_end-hang share the mechanism: the watchdog's L2
    # close unblocks the recv; the episode fails; NO process exit.
    hang = threading.Event()
    inner = _hanging_runner(hang)
    exits = []
    watchdog = WatchdogRunner(inner, episode_deadline_s=0.05, terminate_grace_s=5.0, exit_fn=exits.append)
    with pytest.raises(ConnectionError, match="closed by watchdog"):
        watchdog.run(_make_task(), _report)
    assert inner.closed.is_set()
    assert exits == []  # L2 sufficed; L3 never fired


def test_liveness_5_fast_success_regression_no_stale_timer():
    # A fast episode must disarm+join its watchdog; waiting past
    # deadline+grace afterwards must produce zero timer actions, and a second
    # task must run normally (generation isolation).
    closes = []
    exits = []

    class _Fast(RobocasaEpisodeRunner):
        def __init__(self):
            pass

        def run(self, task, report):
            del report
            return task  # immediate

        def close_current_client(self):
            closes.append(True)

        def close(self):
            pass

    watchdog = WatchdogRunner(_Fast(), episode_deadline_s=0.05, terminate_grace_s=0.05, exit_fn=exits.append)
    assert watchdog.run(_make_task(), _report) is not None
    time.sleep(0.25)  # well past deadline + grace
    assert closes == [] and exits == []
    assert watchdog.run(_make_task(), _report) is not None  # second task unaffected


# ------------------------------------------------------------------
# Liveness: REAL default collaborators (no algorithm stand-ins)
# ------------------------------------------------------------------


def test_real_probe_refused_port_raises_bounded():
    # Real default_handshake_probe + real TCP refusal: bind-then-close to get a
    # port that is guaranteed dead, then assert the runner's bounded retry.
    import socket as _socket

    from exp.robocasa365.episode_runner import default_handshake_probe

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    attempts = []

    def counting_probe(server, timeout_s):
        attempts.append(1)
        return default_handshake_probe(server, timeout_s)

    runner = RobocasaEpisodeRunner(
        _FakeAdapter(),
        client_factory=lambda server: pytest.fail("client must not be built after failed probes"),
        gym_make=lambda *a, **k: pytest.fail("no env on failed connect"),
        horizon_fn=lambda name: 1,
        handshake_probe=counting_probe,
        connect_deadline_s=0.5,
        connect_retries=2,
    )
    task = _make_task()
    task = EpisodeTask(**{**task.__dict__, "server_port": dead_port})
    with pytest.raises(ConnectionError, match="bounded"):
        runner.run(task, _report)
    assert len(attempts) == 2


def test_real_probe_fails_when_ws_handshake_never_completes():
    # The frozen fake listener: accepts TCP, never speaks WS. The REAL probe
    # must fail within its open_timeout instead of hanging.
    import socket as _socket
    import threading as _threading

    from exp.robocasa365.episode_runner import default_handshake_probe

    listener = _socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    accepted = []

    def _accept():
        try:
            conn, _ = listener.accept()
            accepted.append(conn)  # hold it open, say nothing
        except OSError:
            pass

    thread = _threading.Thread(target=_accept, daemon=True)
    thread.start()
    try:
        with pytest.raises(Exception):  # noqa: B017 - any bounded failure is correct; hanging is the bug
            default_handshake_probe(ServerEndpoint("127.0.0.1", port), timeout_s=0.5)
    finally:
        listener.close()
        for conn in accepted:
            conn.close()


def test_real_probe_fails_when_metadata_frame_never_arrives():
    # WS handshake succeeds but the server never sends its first (metadata)
    # frame: the probe must RAISE (a suppressed timeout here would classify a
    # mute server as healthy — the exact G2R3 finding).
    websockets_server = pytest.importorskip("websockets.sync.server")
    import threading as _threading

    from exp.robocasa365.episode_runner import default_handshake_probe

    hold = _threading.Event()

    def handler(connection):
        hold.wait(timeout=5.0)  # never send anything

    with websockets_server.serve(handler, "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
        thread = _threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with pytest.raises(TimeoutError):
                default_handshake_probe(ServerEndpoint("127.0.0.1", port), timeout_s=0.4)
        finally:
            hold.set()
            server.shutdown()


def test_l3_exit_feeds_worker_agent_respawn():
    # The L3 half of the self-heal chain: a worker process that exited (as
    # os._exit(3) would) is respawned by the REAL WorkerAgent.supervise_once.
    from unittest import mock

    from openpi.conductor.agent import WorkerAgent, WorkerSpec

    spawned = []

    def spawn(spec, host, port):
        handle = mock.Mock()
        handle.poll.return_value = None if len(spawned) > 0 else 3  # first dies with code 3
        handle.pid = 4242
        spawned.append(spec.worker_id)
        return handle

    agent = WorkerAgent([WorkerSpec("w0", "127.0.0.1:8010", "0")], driver_host="127.0.0.1", driver_port=1, spawn_fn=spawn)
    agent.start()
    assert spawned == ["w0"]
    agent.supervise_once()  # sees exitcode 3 -> respawn
    assert spawned == ["w0", "w0"]
    assert agent.restart_counts == {"w0": 1}


def test_env_cache_bounded_evicts_and_closes():
    # 8G eval cards: max_cached_envs=1 must close the previous kitchen on a
    # task switch instead of accumulating one env per task (live OOM 2026-08-21).
    class _Env:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    envs = []

    def gym_make(task_name, layout, style, **kw):  # noqa: ARG001
        env = _Env()
        envs.append(env)
        return env

    runner = RobocasaEpisodeRunner(
        _FakeAdapter(),
        client_factory=lambda server: _FakeClient(),
        gym_make=gym_make,
        horizon_fn=lambda name: 5,
        handshake_probe=lambda server, timeout_s: None,
        max_cached_envs=1,
    )
    runner._ensure_env("TaskA", 1, 1)
    runner._ensure_env("TaskB", 1, 1)
    runner._ensure_env("TaskC", 1, 1)
    assert len(runner._envs) == 1
    assert [e.closed for e in envs] == [True, True, False]
    # Re-requesting the cached key must not rebuild.
    runner._ensure_env("TaskC", 1, 1)
    assert len(envs) == 3


def test_env_cache_unbounded_by_default():
    runner, _client, _env, made = _runner()
    runner._ensure_env("TaskA", 1, 1)
    runner._ensure_env("TaskB", 1, 1)
    assert len(runner._envs) == 2  # legacy collection behavior preserved
    assert len(made) == 2
