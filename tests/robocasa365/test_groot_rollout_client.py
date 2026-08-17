"""Non-manual tests for the rollout client's resource handling and seeding.

The simulator is not installed in the main test environment, so ``gymnasium``
and ``robocasa`` are injected into ``sys.modules`` as minimal fakes.  That is
enough because the behaviour under test -- releasing the environment on every
exit path, and replaying one seed list across both scenes -- lives entirely in
``run_one``'s control flow, not in the simulator.
"""

from __future__ import annotations

import json
import math
import sys
import types
from typing import Any

import numpy as np
import pytest

from exp.robocasa365 import groot_keys
from exp.robocasa365.groot_policy_adapter import build_groot_observation
from exp.robocasa365.groot_rollout_client import _select_and_downsample
from exp.robocasa365.groot_rollout_client import (
    episode_seeds,
    parse_scene,
    run_one,
    summarize_exception,
    write_results_atomically,
)

HORIZON = 6


class _FakeEnv:
    """Minimal env stand-in that can succeed, fail, and remember what it ran.

    Recording the executed actions and step counts is what lets the tests observe
    the rollout loop's actual semantics -- success accounting, horizon cut-off,
    plan lifecycle and execution order -- rather than just its exit paths.
    """

    def __init__(
        self, fail_on_step: bool = False, success_at_step: int | None = None
    ) -> None:
        self.closed = 0
        self.seeds: list[Any] = []
        self.fail_on_step = fail_on_step
        self.success_at_step = success_at_step
        self.steps_this_episode = 0
        self.total_steps = 0
        self.executed: list[dict[str, Any]] = []
        self.make_kwargs: dict[str, Any] = {}

    def _obs(self) -> dict[str, Any]:
        # Rendered at the native resolution so run_one actually walks the
        # downsampling path instead of skipping it.
        obs: dict[str, Any] = {
            key: np.zeros(
                (groot_keys.RENDER_RESOLUTION, groot_keys.RENDER_RESOLUTION, 3),
                dtype=np.uint8,
            )
            for key in groot_keys.VIDEO_KEYS
        }
        for key in groot_keys.STATE_KEYS:
            obs[key] = np.zeros(groot_keys.STATE_DIMS[key])
        obs["annotation.human.task_description"] = "do the thing"
        return obs

    def reset(self, seed: Any = None) -> tuple[dict[str, Any], dict]:
        self.seeds.append(seed)
        self.steps_this_episode = 0
        return self._obs(), {}

    def step(self, action: dict[str, Any]):
        if self.fail_on_step:
            raise RuntimeError("mujoco exploded")
        self.executed.append(action)
        self.steps_this_episode += 1
        self.total_steps += 1
        success = (
            self.success_at_step is not None
            and self.steps_this_episode >= self.success_at_step
        )
        return self._obs(), 0.0, False, False, {"success": success}

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def fake_sim(monkeypatch):
    """Install stand-ins for gymnasium and the robocasa task registry."""
    holder: dict[str, _FakeEnv] = {}

    def _make(*args, **kwargs):
        env = holder["env"]
        env.make_kwargs = kwargs
        return env

    monkeypatch.setitem(sys.modules, "gymnasium", types.SimpleNamespace(make=_make))
    # Mirrors the real robocasa layout: TASK_SET_REGISTRY lives in
    # dataset_registry, get_task_horizon in dataset_registry_utils. Injecting a
    # fake that matches whatever the code imports would make this test agree
    # with a wrong import path instead of catching it -- see the manual suite
    # for the check that actually pins the real module.
    robocasa = types.ModuleType("robocasa")
    utils = types.ModuleType("robocasa.utils")
    registry = types.ModuleType("robocasa.utils.dataset_registry")
    registry.TASK_SET_REGISTRY = {"atomic_seen": ["SomeTask"]}
    registry_utils = types.ModuleType("robocasa.utils.dataset_registry_utils")
    registry_utils.get_task_horizon = lambda name: HORIZON
    monkeypatch.setitem(sys.modules, "robocasa", robocasa)
    monkeypatch.setitem(sys.modules, "robocasa.utils", utils)
    monkeypatch.setitem(sys.modules, "robocasa.utils.dataset_registry", registry)
    monkeypatch.setitem(
        sys.modules, "robocasa.utils.dataset_registry_utils", registry_utils
    )
    return holder


class _OkClient:
    """Returns a ramp-valued chunk so individual steps are distinguishable."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_element: dict[str, Any] | None = None
        self.ctrl_frames: list[dict[str, Any]] = []

    def infer(self, element: dict[str, Any]) -> dict[str, Any]:
        if "__ctrl__" in element:
            # Episode framing rides the same call, but it is not an inference:
            # the server dispatches it before the policy is ever consulted.
            self.ctrl_frames.append(element)
            return {}
        self.calls += 1
        self.last_element = element
        base = self.calls * 1000
        return {
            "actions": {
                key: base
                + np.arange(groot_keys.ACTION_HORIZON * dim, dtype=float).reshape(
                    -1, dim
                )
                for key, dim in groot_keys.ACTION_DIMS.items()
            }
        }


class _FailingClient:
    def infer(self, element: dict[str, Any]) -> dict[str, Any]:
        if "__ctrl__" in element:
            return {}
        raise ConnectionError("server went away")


def test_run_one_closes_env_on_success(fake_sim):
    fake_sim["env"] = _FakeEnv()
    result = run_one(_OkClient(), "SomeTask", (1, 1), n_trials=2, replan=3, base_seed=0)
    assert result["n"] == 2
    assert fake_sim["env"].closed == 1


def test_run_one_closes_env_when_inference_fails(fake_sim):
    """A failed inference must not leak the MuJoCo context.

    main() catches per-arm errors and carries on, so an env left open here would
    accumulate across a batch until the GPU runs out.
    """
    fake_sim["env"] = _FakeEnv()
    with pytest.raises(ConnectionError):
        run_one(_FailingClient(), "SomeTask", (1, 1), n_trials=1, replan=3, base_seed=0)
    assert fake_sim["env"].closed == 1


def test_run_one_closes_env_when_step_fails(fake_sim):
    fake_sim["env"] = _FakeEnv(fail_on_step=True)
    with pytest.raises(RuntimeError, match="mujoco exploded"):
        run_one(_OkClient(), "SomeTask", (1, 1), n_trials=1, replan=3, base_seed=0)
    assert fake_sim["env"].closed == 1


def test_run_one_closes_env_when_replan_is_invalid(fake_sim):
    fake_sim["env"] = _FakeEnv()
    with pytest.raises(ValueError):
        run_one(
            _OkClient(),
            "SomeTask",
            (1, 1),
            n_trials=1,
            replan=groot_keys.ACTION_HORIZON + 1,
            base_seed=0,
        )
    # run_one only rejects the bound once it is slicing a chunk, which is after
    # gym.make -- so the env was created and must still have been released.
    # (The eager pre-flight check lives in main(), not here.)
    assert fake_sim["env"].closed == 1


def test_run_one_replays_the_declared_seeds(fake_sim):
    fake_sim["env"] = _FakeEnv()
    result = run_one(
        _OkClient(), "SomeTask", (1, 1), n_trials=3, replan=2, base_seed=41
    )
    assert fake_sim["env"].seeds == [41, 42, 43]
    assert result["seeds"] == [41, 42, 43]


def test_episode_seeds_are_identical_across_scenes():
    # Scene A and scene B are separate gym.make calls; pairing depends entirely
    # on both replaying the same list.
    assert episode_seeds(5, 7) == episode_seeds(5, 7) == [7, 8, 9, 10, 11]


def test_run_one_renders_at_native_resolution(fake_sim):
    fake_sim["env"] = _FakeEnv()
    run_one(_OkClient(), "SomeTask", (3, 5), n_trials=1, replan=2, base_seed=0)
    kwargs = fake_sim["env"].make_kwargs
    assert kwargs["camera_heights"] == groot_keys.RENDER_RESOLUTION
    assert kwargs["camera_widths"] == groot_keys.RENDER_RESOLUTION
    # Only the kitchen may vary between arms.
    assert kwargs["split"] is None
    assert kwargs["obj_instance_split"] == "target"
    assert kwargs["layout_and_style_ids"] == [(3, 5)]


# ------------------------------------------------------------------
# Fail-fast helpers
# ------------------------------------------------------------------


def test_parse_scene_accepts_layout_style_pair():
    assert parse_scene("1,1") == (1, 1)
    assert parse_scene("7,10") == (7, 10)


@pytest.mark.parametrize("bad", ["1", "1,2,3", "", "a,1"])
def test_parse_scene_rejects_malformed(bad):
    """A typo must fail at argument-parse time, not 300 tasks later."""
    with pytest.raises(ValueError):
        parse_scene(bad)


def test_summarize_exception_keeps_the_informative_tail():
    """The inference server returns a traceback; its useful part is last."""
    server_style = (
        "Error in inference server:\n"
        "Traceback (most recent call last):\n"
        '  File "websocket_policy_server.py", line 922, in _handler\n'
        "ValueError: state.base_position contains non-finite values"
    )
    summary = summarize_exception(RuntimeError(server_style))
    assert "non-finite" in summary
    assert "Traceback" not in summary


def test_write_results_atomically_replaces_without_truncation(tmp_path):
    target = tmp_path / "out.json"
    write_results_atomically(target, {"tasks": {"a": 1}})
    write_results_atomically(target, {"tasks": {"a": 1, "b": 2}})
    import json as _json

    assert _json.loads(target.read_text())["tasks"] == {"a": 1, "b": 2}
    # the temp file must not be left behind
    assert list(tmp_path.iterdir()) == [target]


def test_write_results_atomically_leaves_old_file_intact_on_encode_failure(tmp_path):
    target = tmp_path / "out.json"
    write_results_atomically(target, {"ok": 1})
    with pytest.raises(TypeError):
        write_results_atomically(target, {"bad": object()})
    import json as _json

    assert _json.loads(target.read_text()) == {"ok": 1}


# ------------------------------------------------------------------
# Rollout loop semantics -- these produce the experiment's headline numbers
# ------------------------------------------------------------------


def test_run_one_counts_successes_and_rate(fake_sim):
    fake_sim["env"] = _FakeEnv(success_at_step=2)
    result = run_one(_OkClient(), "T", (1, 1), n_trials=3, replan=5, base_seed=0)
    assert result["succ"] == 3
    assert result["n"] == 3
    assert result["sr"] == 1.0


def test_run_one_reports_zero_when_nothing_succeeds(fake_sim):
    fake_sim["env"] = _FakeEnv(success_at_step=None)
    result = run_one(_OkClient(), "T", (1, 1), n_trials=2, replan=5, base_seed=0)
    assert result["succ"] == 0
    assert result["sr"] == 0.0


def test_run_one_stops_the_episode_at_the_horizon(fake_sim):
    """A never-succeeding episode must end at exactly `horizon` steps."""
    fake_sim["env"] = _FakeEnv(success_at_step=None)
    run_one(_OkClient(), "T", (1, 1), n_trials=1, replan=5, base_seed=0)
    assert fake_sim["env"].total_steps == HORIZON


def test_run_one_stops_the_episode_on_success(fake_sim):
    fake_sim["env"] = _FakeEnv(success_at_step=2)
    run_one(_OkClient(), "T", (1, 1), n_trials=1, replan=5, base_seed=0)
    assert fake_sim["env"].total_steps == 2


def test_run_one_executes_actions_in_order(fake_sim):
    """Queued steps must run FIFO and in chunk order.

    Popping from the wrong end, or reversing the split, keeps every shape and
    count intact while driving the robot backwards.
    """
    fake_sim["env"] = _FakeEnv(success_at_step=None)
    client = _OkClient()
    run_one(client, "T", (1, 1), n_trials=1, replan=4, base_seed=0)
    executed = fake_sim["env"].executed
    first = [a["action.end_effector_position"][0] for a in executed[:4]]
    assert first == sorted(first), f"actions ran out of order: {first}"


def test_run_one_discards_the_previous_episode_plan(fake_sim):
    """A new episode must re-plan rather than replay leftover actions."""
    fake_sim["env"] = _FakeEnv(success_at_step=1)
    client = _OkClient()
    run_one(client, "T", (1, 1), n_trials=3, replan=5, base_seed=0)
    # Each episode ends after one step but the chunk held five: if the queue
    # leaked across resets, later episodes would not trigger a new inference.
    assert client.calls == 3


def test_run_one_infers_once_per_replan_window(fake_sim):
    fake_sim["env"] = _FakeEnv(success_at_step=None)
    client = _OkClient()
    run_one(client, "T", (1, 1), n_trials=1, replan=3, base_seed=0)
    assert client.calls == math.ceil(HORIZON / 3)


def test_run_one_sends_downsampled_frames(fake_sim):
    """The env renders at 512; the model must receive 256."""
    fake_sim["env"] = _FakeEnv(success_at_step=None)
    client = _OkClient()
    run_one(client, "T", (1, 1), n_trials=1, replan=5, base_seed=0)
    element = client.last_element
    for key in groot_keys.VIDEO_KEYS:
        assert element[key].shape == (
            groot_keys.MODEL_IMAGE_RESOLUTION,
            groot_keys.MODEL_IMAGE_RESOLUTION,
            3,
        )
        assert element[key].dtype == np.uint8


def test_run_one_sends_every_wire_key(fake_sim):
    fake_sim["env"] = _FakeEnv(success_at_step=None)
    client = _OkClient()
    run_one(client, "T", (1, 1), n_trials=1, replan=5, base_seed=0)
    assert set(client.last_element) == set(groot_keys.wire_observation_keys())


# ------------------------------------------------------------------
# Observation direction: what the client sends must be what the server accepts
# ------------------------------------------------------------------


def _raw_env_obs(resolution: int = groot_keys.RENDER_RESOLUTION) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for index, key in enumerate(groot_keys.VIDEO_KEYS):
        # Distinguishable per camera, so routing mistakes are visible.
        obs[key] = np.full((resolution, resolution, 3), index + 1, dtype=np.uint8)
    for key in groot_keys.STATE_KEYS:
        obs[key] = np.zeros(groot_keys.STATE_DIMS[key], dtype=np.float32)
    obs["annotation.human.task_description"] = "open the drawer"
    return obs


def test_client_output_is_accepted_by_the_server_validator():
    """Closes the loop on the observation side.

    The action direction already has an end-to-end contract test; without this
    one, dropping a key or changing a dtype in _select_and_downsample would only
    surface during a live run.
    """
    element = _select_and_downsample(_raw_env_obs())
    assert set(element) == set(groot_keys.wire_observation_keys())
    build_groot_observation(element)  # must not raise


def test_select_and_downsample_resizes_to_model_resolution():
    element = _select_and_downsample(_raw_env_obs())
    for key in groot_keys.VIDEO_KEYS:
        assert element[key].shape == (
            groot_keys.MODEL_IMAGE_RESOLUTION,
            groot_keys.MODEL_IMAGE_RESOLUTION,
            3,
        )
        assert element[key].dtype == np.uint8


def test_select_and_downsample_keeps_cameras_distinct():
    """Three identical frames would be structurally valid and silently wrong."""
    element = _select_and_downsample(_raw_env_obs())
    means = [element[key].mean() for key in groot_keys.VIDEO_KEYS]
    assert len(set(means)) == 3, f"cameras collapsed: {means}"


def test_select_and_downsample_rejects_non_uint8_frames():
    """Casting instead of rejecting would disarm the server's dtype check."""
    obs = _raw_env_obs()
    obs[groot_keys.VIDEO_KEYS[0]] = np.zeros(
        (groot_keys.RENDER_RESOLUTION, groot_keys.RENDER_RESOLUTION, 3),
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="uint8"):
        _select_and_downsample(obs)


def test_select_and_downsample_rejects_non_square_frames():
    obs = _raw_env_obs()
    obs[groot_keys.VIDEO_KEYS[0]] = np.zeros((256, 512, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="square"):
        _select_and_downsample(obs)


def test_state_dims_cover_every_state_key():
    assert set(groot_keys.STATE_DIMS) == set(groot_keys.STATE_KEYS)
    assert sum(groot_keys.STATE_DIMS.values()) == 16


# ------------------------------------------------------------------
# Episode framing through the real rollout loop
# ------------------------------------------------------------------


class _CtrlRecordingClient(_OkClient):
    """_OkClient that can be told to blow up on the Nth real inference."""

    def __init__(self, fail_on: int | None = None) -> None:
        super().__init__()
        self._fail_on = fail_on

    def infer(self, element: dict[str, Any]) -> dict[str, Any]:
        if "__ctrl__" not in element and self._fail_on is not None:
            if self.calls + 1 == self._fail_on:
                self.calls += 1
                raise ConnectionError("server went away mid-episode")
        return super().infer(element)


def test_run_one_frames_every_episode_with_ctrl(fake_sim):
    fake_sim["env"] = _FakeEnv()
    client = _CtrlRecordingClient()
    run_one(client, "SomeTask", (1, 1), n_trials=2, replan=3, base_seed=0)

    kinds = [frame["__ctrl__"] for frame in client.ctrl_frames]
    assert kinds == ["episode_start", "episode_end", "episode_start", "episode_end"]
    starts = [f for f in client.ctrl_frames if f["__ctrl__"] == "episode_start"]
    # The task key only reaches SearchContext through this field.
    assert all(f["__task__"] == "SomeTask" for f in starts)
    assert [f["__episode_id__"] for f in starts] == [0, 1]


def test_run_one_sends_episode_end_when_the_episode_raises(fake_sim):
    """Without this the server's search session stays open until the socket drops."""
    fake_sim["env"] = _FakeEnv()
    client = _CtrlRecordingClient(fail_on=1)

    with pytest.raises(ConnectionError):
        run_one(client, "SomeTask", (1, 1), n_trials=1, replan=3, base_seed=0)

    kinds = [frame["__ctrl__"] for frame in client.ctrl_frames]
    assert kinds == ["episode_start", "episode_end"]
    assert client.ctrl_frames[-1]["__success__"] is False


def test_run_one_writes_hit_rows_when_the_server_reports_verdicts(fake_sim, tmp_path):
    fake_sim["env"] = _FakeEnv()

    class _CachingClient(_CtrlRecordingClient):
        def infer(self, element):
            out = super().infer(element)
            if "__ctrl__" not in element:
                out = dict(out)
                out["__hit_meta__"] = {
                    "hit_type": "FULL_HIT",
                    "winner_id": f"traj:{self.calls}",
                    "cp1_score": 0.9,
                    "searched": True,
                }
            return out

    log_path = tmp_path / "hits.jsonl"
    with log_path.open("w") as handle:
        run_one(
            _CachingClient(), "SomeTask", (1, 1),
            n_trials=1, replan=3, base_seed=0, hit_log=handle,
        )

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert rows
    assert all(row["task"] == "SomeTask" and row["hit_type"] == "FULL_HIT" for row in rows)
    assert [row["step_idx"] for row in rows] == sorted(row["step_idx"] for row in rows)
