"""CPU proof: observing contacts/controllers/RNG never changes a physical step."""
import copy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.debug.client.adapter import MujocoAdapter, names_by_id
from exp.offline_search.debug.client.capture import Client, EnvTap, FileSink
from exp.offline_search.debug.schema import episode_key
from exp.offline_search.debug.transport.validation import validate_episode


class FakeModel:
    nbody, ngeom, njnt, nsite, nu, nq, nv = 5, 6, 3, 1, 2, 5, 4
    geom_bodyid = np.array([0, 1, 0, 2, 3, 4])
    body_parentid = np.array([0, 0, 0, 0, 3])
    jnt_bodyid = np.array([1, 2, 3])
    jnt_type = np.array([3, 0, 3])
    # The compact geom_names intentionally omits unnamed IDs, as in old P3.
    geom_names = ["robot_collision", "table_collision", "mug_collision", "drawer_collision"]

    def body_id2name(self, i):
        return ["world", "robot0_arm", "mug_main", "drawer_main", "drawer_child"][i]

    def geom_id2name(self, i):
        return [None, "robot_collision", "table_collision", "mug_collision", "drawer_collision", None][i]

    def joint_id2name(self, i):
        return ["robot_joint", "mug_joint", "drawer_joint"][i]

    def site_id2name(self, i):
        return "eef"

    def actuator_id2name(self, i):
        return "act" + str(i)


class FakeSim:
    def __init__(self):
        self.model = FakeModel()
        self.data = SimpleNamespace(qpos=np.zeros(5, np.float64), qvel=np.zeros(4, np.float64),
                                    ctrl=np.zeros(2), act=np.zeros(2), qacc_warmstart=np.zeros(4), time=0.,
                                    mocap_pos=np.empty((0, 3)), mocap_quat=np.empty((0, 4)),
                                    body_xpos=np.array([[0., 0., 0.], [0., 0., 0.], [.02, 0., .1], [.2, 0., 0.], [.2, .02, 0.]]),
                                    body_xquat=np.tile([1., 0., 0., 0.], (5, 1)), cvel=np.zeros((5, 6)),
                                    ncon=1, contact=[SimpleNamespace(geom1=2, geom2=3, dist=-.001,
                                                                   pos=np.zeros(3), frame=np.eye(3).reshape(-1))])
        self.physics_steps = 0

    def step(self):
        self.physics_steps += 1
        self.data.time += .01
        self.data.qpos[:2] += self.data.ctrl * .01
        self.data.qvel[:2] = self.data.ctrl

    def contact_force(self, model, data, i, out):
        assert out.dtype == np.float64
        out[:] = np.arange(6) + 1


class FakeEnv:
    def __init__(self):
        self.sim = FakeSim()
        self.goal_state = [["on", "mug", "target"], ["open", "drawer"]]
        self.obj_body_id = {"mug": 2, "drawer": 3}
        self.robots = [SimpleNamespace(controller=SimpleNamespace(goal=np.array([1., 2.]), counter=3,
                                                                  callback=lambda: None))]
        self.np_random = np.random.RandomState(7)
        self.control_timestep = .02
        self.timestep = 0
        self.cur_time = 0.
        self.satisfied = False
        self.actions = []

    def reset(self):
        self.timestep = 0
        return self.observe()

    def set_init_state(self, state):
        self.sim.data.qpos[:] = state
        return self.observe()

    def get_sim_state(self):
        return np.r_[self.sim.data.time, self.sim.data.qpos, self.sim.data.qvel]

    def observe(self):
        return dict(robot0_eef_pos=np.array([.02, 0., .03]), robot0_eef_quat=np.array([0., 0., 0., 1.]),
                    robot0_gripper_qpos=self.sim.data.qpos[-2:].copy(), robot0_gripper_qvel=self.sim.data.qvel[-2:].copy())

    def _eval_predicate(self, p):
        return self.satisfied if p[0] == "on" else False

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        self.sim.data.ctrl[:] = np.asarray(action)[:2]
        for _ in range(2):
            self.sim.step()
        self.sim.data.qpos[-1] += self.np_random.normal() * .00001
        self.sim.data.body_xpos[2, 2] = max(.02, self.sim.data.body_xpos[2, 2] - .02)
        self.timestep += 1
        self.cur_time += .02
        return self.observe(), 0., False, {}


class FakePolicy:
    def __init__(self, echo=True):
        self.echo = echo
        self.requests = []

    def episode_start(self, **kw):
        self.kw = kw

    def infer(self, obs):
        self.requests.append(copy.deepcopy(obs))
        chunk = np.array([[.1 + i * .01, -.2] for i in range(5)], dtype=np.float32)
        out = {"actions": chunk, "extra": "stock"}
        if self.echo:
            out["__debug__"] = dict(v=1, decision_id=obs["__debug__"]["decision_id"], status="available")
        return out


def make_episode(directory, *, n=132, echo=True, oracle=False, sink_factory=None, fail=False):
    env, policy = FakeEnv(), FakePolicy(echo)
    client = Client(policy, directory, campaign="test", arm="A", oracle=oracle, sink_factory=sink_factory)
    client.episode_start(experiment="fake", task="goal", episode_id=0,
                         extra_metadata=dict(task_uid="A:eval:0:0", attempt=1, dispatch_gen=3, task_id=0, orig_init_state_idx=0))
    client.configure(env, SimpleNamespace(seed=7, num_steps_wait=4, replan_steps=5), n)
    tap = EnvTap(env, client)
    env.reset()
    tap.set_init_state(np.zeros(5))
    for _ in range(4):
        tap.step([0., 0.])
    if fail:
        def broken(*args):
            raise OSError("injected snapshot failure")
        client.save_snapshot = broken
    used, states = 0, []
    while used < n:
        obs = {"observation/image": np.zeros((8, 8, 3), np.uint8), "observation/wrist_image": np.ones((8, 8, 3), np.uint8),
               "observation/state": np.r_[env.observe()["robot0_eef_pos"], np.zeros(3), np.zeros(2)], "prompt": "goal"}
        result = client.infer(obs)
        assert "__debug__" not in result and result["extra"] == "stock"
        assert "__debug__" not in obs and "__oracle__" not in obs
        for action in result["actions"][:min(5, n - used)]:
            tap.step(action.tolist())
            states.append(env.get_sim_state())
            used += 1
    client.finish(False, "step_cap")
    return client, env, policy, np.asarray(states)


def test_capture_parity_rng_contacts_partial_chunks(tmp_path):
    numpy_before, python_before = np.random.get_state(), random.getstate()
    client, env, policy, states = make_episode(tmp_path / "capture")
    reference = FakeEnv()
    reference.reset()
    reference.set_init_state(np.zeros(5))
    reference_states = []
    for action in env.actions:
        reference.step(action.tolist())
        reference_states.append(reference.get_sim_state())
    np.testing.assert_array_equal(states, np.asarray(reference_states[4:]))
    np.testing.assert_array_equal(reference.get_sim_state(), env.get_sim_state())
    assert reference.sim.physics_steps == env.sim.physics_steps
    assert set(reference.sim.__dict__) == set(env.sim.__dict__)
    np.testing.assert_array_equal(reference.np_random.get_state()[1], env.np_random.get_state()[1])
    after = np.random.get_state()
    assert numpy_before[0] == after[0] and numpy_before[2:] == after[2:]
    np.testing.assert_array_equal(numpy_before[1], after[1])
    assert random.getstate() == python_before
    assert not client.errors
    directory = tmp_path / "capture" / episode_key("A:eval:0:0", 1)
    trace = validate_episode(directory, "A:eval:0:0", 1)
    assert trace["status"] == "complete" and trace["n_controls"] == 136 and trace["n_decisions"] == 27
    events = [json.loads(r) for r in (directory / "events.jsonl").read_text().splitlines()]
    decisions = [r for r in events if r["ev"] == "decision"]
    assert decisions[-1]["n_applied"] == 2 and decisions[-1]["control_idx_start"] == 134
    blocks = [np.load(p, allow_pickle=False) for p in sorted(directory.glob("controls_*.npz"))]
    assert [len(b["action"]) for b in blocks] == [64, 64, 8]
    np.testing.assert_array_equal(np.concatenate([b["control_idx"] for b in blocks]), np.arange(136))
    np.testing.assert_array_equal(blocks[0]["contact_geom"][0], [2, 3])
    np.testing.assert_array_equal(blocks[0]["contact_force"][0], np.arange(6) + 1)
    assert blocks[0]["qpos"].dtype == np.float64 and blocks[0]["act_substep"].shape == (64, 2, 2)
    ep = json.loads((directory / "episode.json").read_text())
    assert ep["entities"]["geom_names"][0] is None and ep["entities"]["geom_names"][2] == "table_collision"
    assert len(ep["entities"]["geom_names"]) == 6 and ep["env_seed"] == 7
    assert ep["reset_selfcheck"]["status"] == "available"
    np.testing.assert_allclose(ep["post_settle_obj_pos"][0][2], .02)
    with np.load(directory / "snap_000000.npz", allow_pickle=False) as snap:
        assert not snap["restore_certified"] and "callback:function" in str(snap["controller_skipped_json"])
    for b in blocks:
        b.close()


def test_capture_failure_never_changes_actions_or_states(tmp_path):
    good, _, _, states = make_episode(tmp_path / "good", n=12)
    bad, _, _, other = make_episode(tmp_path / "bad", n=12, fail=True)
    np.testing.assert_array_equal(states, other)
    assert bad.errors and bad.episode["success"] == good.episode["success"]


def test_envelope_without_server_echo_is_unverified(tmp_path):
    client, _, policy, _ = make_episode(tmp_path, n=1, echo=False)
    request = policy.requests[0]["__debug__"]
    assert request["decision_id"] == episode_key("A:eval:0:0", 1) + ":3:0"
    assert request["v"] == 1 and request["attempt"] == 1 and client.episode["server_join"] == "unverified"


def test_geometry_names_and_oracle_post_settle(tmp_path):
    env = FakeEnv()
    adapter = MujocoAdapter(env)
    assert [r["body_id"] for r in adapter.catalog["movable"]] == [2, 3, 4]
    assert adapter.resting_selfcheck()["status"] == "available"
    env.sim.data.contact[0].geom1 = 1
    assert adapter.resting_selfcheck()["missing_table_contacts"] == [2]
    env.sim.data.contact[0].geom1 = 2
    adapter.set_reference()
    oracle = adapter.oracle(env.observe())
    assert oracle["in_window"] and oracle["object_id"] == "mug"
    env.sim.data.body_xpos[2, 2] += .04
    assert not adapter.oracle(env.observe())["in_window"]
    env.sim.data.body_xpos[2, 2] -= .04
    env.satisfied = True
    assert not adapter.oracle(env.observe())["in_window"]
    client, _, policy, _ = make_episode(tmp_path, n=1, oracle=True)
    assert policy.requests[0]["__oracle__"]["in_window"]
    assert client.episode["oracle"]["privileged"]


def test_control_gap_rejected(tmp_path):
    client, _, _, _ = make_episode(tmp_path, n=1)
    directory = tmp_path / client.identity["episode_key"]
    path = directory / "controls_0000.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["control_idx"][0] = 5
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="control gap"):
        validate_episode(directory, "A:eval:0:0", 1)


def test_gripper_joint_descendants_are_robot_bodies():
    env = FakeEnv()
    model = env.sim.model
    body_name, joint_name = model.body_id2name, model.joint_id2name
    model.nbody, model.njnt = 6, 4
    model.body_parentid = np.r_[model.body_parentid, 1]
    model.jnt_bodyid = np.r_[model.jnt_bodyid, 5]
    model.jnt_type = np.r_[model.jnt_type, 3]
    model.body_id2name = lambda i: body_name(i) if i < 5 else "gripper0_left"
    model.joint_id2name = lambda i: joint_name(i) if i < 3 else "gripper_joint"
    adapter = MujocoAdapter(env)
    assert [r["body_id"] for r in adapter.catalog["movable"]] == [2, 3, 4]
