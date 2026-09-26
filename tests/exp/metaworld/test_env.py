"""MetaWorld task table and simulator conventions (fake environments + one real cross-process check)."""

from __future__ import annotations

import collections
import json
import os
import pathlib
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from exp.metaworld import env as E
from exp.metaworld import tasks as T
from exp.warm_reset.plan import validate_tasks

REPO = pathlib.Path(__file__).resolve().parents[3]
SIM_PYTHON = pathlib.Path("~/metaworld_sim/bin/python").expanduser()


# ------------------------------------------------------------------
# Task table
# ------------------------------------------------------------------


def test_task_table_matches_rlinf():
    assert len(T.MT50_TASKS) == 50 == len(set(T.TASK_NAMES))
    assert collections.Counter(T.DIFFICULTY.values()) == {
        "easy": 28,
        "medium": 11,
        "hard": 6,
        "very_hard": 5,
    }
    assert set(T.DIFFICULTY.values()) == set(T.DIFFICULTY_GROUPS)
    # RLinf's training-time label, not LeRobot's later rewrite
    assert T.PROMPTS["push-back-v3"] == "Push the puck to a goal"
    assert T.PROMPTS["push-v3"] == "Push the puck to a goal"
    assert T.PROMPTS["reach-v3"] == "Reach a goal position"
    assert T.PROMPTS["dial-turn-v3"] == "Rotate a dial 180 degrees"
    assert T.TASK_NAMES[0] == "assembly-v3" and T.TASK_NAMES[-1] == "window-close-v3"
    assert all(name.endswith("-v3") for name in T.TASK_NAMES)


def test_export_tasks_is_entry_compatible():
    tasks = T.export_tasks(["reach-v3", "push-back-v3"], 2, init_offset=3)
    validate_tasks(tasks)
    assert tasks == [
        {
            "task_id": T.task_id_of("reach-v3"),
            "name": "reach-v3",
            "init_indices": [3, 4],
        },
        {
            "task_id": T.task_id_of("push-back-v3"),
            "name": "push-back-v3",
            "init_indices": [3, 4],
        },
    ]
    everything = T.export_tasks(None, T.EPISODES_PER_TASK)
    validate_tasks(everything)
    assert [t["task_id"] for t in everything] == list(range(50))
    assert everything[0]["init_indices"] == list(range(20))


@pytest.mark.parametrize(
    ("names", "episodes", "offset"),
    [
        (["reach-v3"], 0, 0),
        (["reach-v3"], 51, 0),
        (["reach-v3"], 2, 49),
        (["reach-v3", "reach-v3"], 1, 0),
        ([], 1, 0),
    ],
)
def test_export_tasks_rejects_bad_requests(names, episodes, offset):
    with pytest.raises(ValueError):
        T.export_tasks(names, episodes, offset)


def test_unknown_task_name():
    with pytest.raises(KeyError):
        T.export_tasks(["reach-v2"], 1)


# ------------------------------------------------------------------
# Fake environment
# ------------------------------------------------------------------


class FakeModel:
    def __init__(self, names=("topview", "corner", "corner2", "corner3")):
        self._names = names
        self.cam_pos = np.zeros((len(names), 3))

    def camera(self, i):
        return SimpleNamespace(name=self._names[i], id=i)


class FakeEnv:
    """Records every step; succeeds at policy step ``success_at`` (1-based) if given."""

    def __init__(self, success_at=None, terminate_at=None):
        self.model = FakeModel()
        self.success_at = success_at
        self.terminate_at = terminate_at
        self.actions = []
        self.resets = 0
        self.t = 0

    def reset(self):
        self.resets += 1
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        return np.full(39, float(self.t))

    def render(self):
        return np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.t += 1
        policy_step = self.t - T.SETTLE_STEPS
        success = self.success_at is not None and policy_step == self.success_at
        terminated = self.terminate_at is not None and policy_step == self.terminate_at
        return self._obs(), 0.0, terminated, False, {"success": float(success)}


def constant_policy(value=0.25):
    requests = []

    def infer(request):
        requests.append(request)
        return {"actions": np.full((5, 4), value, dtype=np.float32)}

    return infer, requests


def test_place_camera_asserts_corner2_id():
    env = FakeEnv()
    E.place_camera(env)
    np.testing.assert_array_equal(env.model.cam_pos[T.CAMERA_ID], T.CAMERA_POS)
    assert T.CAMERA_ID == 2 and T.CAMERA_NAME == "corner2"
    wrong = SimpleNamespace(
        model=FakeModel(("topview", "corner", "corner3", "corner2"))
    )
    with pytest.raises(AssertionError, match="corner2"):
        E.place_camera(wrong)


def test_render_rotated_and_contiguous():
    env = FakeEnv()
    img = E.render_image(env)
    raw = env.render()
    np.testing.assert_array_equal(img, raw[::-1, ::-1])
    assert img.flags["C_CONTIGUOUS"]


def test_observation_request_format():
    env = FakeEnv()
    obs = np.arange(39, dtype=np.float64)
    req = E.observation(env, obs, "Open a drawer")
    assert set(req) == {"observation/image", "observation/state", "prompt"}
    assert req["observation/state"].dtype == np.float32
    np.testing.assert_array_equal(req["observation/state"], [0, 1, 2, 3])
    assert req["prompt"] == "Open a drawer"


def test_settle_steps_are_zero_actions():
    env = FakeEnv()
    obs = E.reset_and_settle(env)
    assert env.resets == 1
    assert len(env.actions) == T.SETTLE_STEPS == 15
    assert all(not a.any() and a.shape == (4,) for a in env.actions)
    assert obs[0] == T.SETTLE_STEPS


def test_step_cap_and_replan_spacing():
    env = FakeEnv()
    infer, requests = constant_policy()
    decisions = []
    out = E.run_episode(
        env, infer, "p", on_decision=lambda step, result: decisions.append(step)
    )
    assert out == E.EpisodeOutcome(False, 160, 32, "step_cap", out.infer_s)
    assert T.MAX_POLICY_STEPS == 160
    assert decisions == list(range(0, 160, 5))
    assert len(env.actions) == T.SETTLE_STEPS + 160
    # the first request sees the settled state, i.e. after the 15 zero steps
    assert requests[0]["observation/state"][0] == T.SETTLE_STEPS
    assert all(np.allclose(a, 0.25) for a in env.actions[T.SETTLE_STEPS :])


def test_success_stops_mid_chunk():
    env = FakeEnv(success_at=7)
    infer, _ = constant_policy()
    out = E.run_episode(env, infer, "p")
    assert (out.success, out.n_steps, out.n_decisions, out.end_reason) == (
        True,
        7,
        2,
        "success",
    )
    assert len(env.actions) == T.SETTLE_STEPS + 7


def test_terminated_is_not_success():
    env = FakeEnv(terminate_at=3)
    infer, _ = constant_policy()
    out = E.run_episode(env, infer, "p")
    assert (out.success, out.n_steps, out.end_reason) == (False, 3, "terminated")


def test_bad_action_chunk_and_replan_rejected():
    def infer(request):
        return {"actions": np.zeros((5, 7))}

    with pytest.raises(ValueError, match="shape"):
        E.run_episode(FakeEnv(), infer, "p")
    with pytest.raises(ValueError, match="replan_steps"):
        E.run_episode(FakeEnv(), constant_policy()[0], "p", replan_steps=6)


# ------------------------------------------------------------------
# Real simulator: identity determinism across processes
# ------------------------------------------------------------------

_PROBE = """
import hashlib, json, sys
from exp.metaworld import env as E
out = []
for task, idx in json.loads(sys.argv[1]):
    env = E.make_env(task, idx)
    obs = E.reset_and_settle(env)
    out.append(hashlib.sha256(obs.tobytes()).hexdigest())
print("PROBE" + json.dumps(out), flush=True)
"""


def _probe(pairs):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")
    }
    env.update(PYTHONPATH=f"{REPO}:{REPO / 'src'}", MUJOCO_GL="egl")
    proc = subprocess.run(
        [str(SIM_PYTHON), "-c", _PROBE, json.dumps(pairs)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=REPO,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE")), None)
    assert line is not None, proc.stderr[-2000:]
    return json.loads(line[len("PROBE") :])


@pytest.mark.skipif(
    not SIM_PYTHON.exists(),
    reason="MetaWorld simulator venv ~/metaworld_sim is not installed",
)
def test_same_identity_same_initial_state_across_processes():
    pairs = [
        ["push-back-v3", 4],
        ["push-back-v3", 5],
        ["reach-v3", 4],
        ["push-back-v3", 4],
    ]
    first, second = _probe(pairs), _probe(pairs)
    assert first == second
    # a repeat inside one process (cached MT1, fresh env) replays the same state
    assert first[3] == first[0]
    # reverse control: a different idx or task is a different initial state
    assert len(set(first[:3])) == 3
