from collections import Counter
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest

from exp.metaworld import env as kernel, tasks as T
from exp.metaworld.episode_runner import MetaworldEpisodeRunner
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, MetaWorldBenchmarkAdapter, R17MetaWorldEpisodeRunner
from exp.r17_integration.benchmarks.physical_mt50 import NativeMetaWorldPhysicalAdapter
from exp.r17_integration.conductor.manifest import episode_tasks
from openpi.conductor.task import ServerEndpoint


class Env:
    def __init__(self, *, success=None, termination=None, truncation=None, fail=None):
        self.success, self.termination, self.truncation, self.fail = success, termination, truncation, fail
        self.actions, self.frames, self.t, self.closed = [], [], 0, False
        self.model = NS(nbody=2, ngeom=2, njnt=1, nsite=1, nu=4,
                        geom_bodyid=np.array([0, 1]), body_parentid=np.array([0, 0]),
                        jnt_bodyid=np.array([1]), jnt_type=np.array([0]), jnt_qposadr=np.array([0]), jnt_dofadr=np.array([0]))
        for kind in ("body", "geom", "joint", "site", "actuator"):
            setattr(self.model, kind, lambda i, k=kind: NS(name=f"{k}{i}"))
        c = NS(geom1=0, geom2=1, dist=-0.002, pos=np.arange(3), frame=np.arange(9))
        self.data = NS(qpos=np.arange(7, dtype=np.float64), qvel=np.arange(6, dtype=np.float64),
                       ctrl=np.zeros(4), act=np.zeros(0), xpos=np.zeros((2, 3)), xquat=np.zeros((2, 4)),
                       cvel=np.zeros((2, 6)), site_xpos=np.zeros((1, 3)), contact=[c], ncon=1, time=0.0)
        self._target_pos = np.array([.1, .2, .3])

    def reset(self):
        self.t = 0
        return np.zeros(39), {}

    def render(self):
        frame = np.full((480, 480, 3), self.t % 256, dtype=np.uint8)
        frame[0, 0] = [1, 2, 3]
        self.frames.append(frame.copy())
        return frame

    def step(self, action):
        idx = self.t - 15 + 1
        if self.fail == idx:
            raise RuntimeError("control fixture failure")
        self.actions.append(np.array(action, copy=True))
        self.t += 1
        self.data.time = self.t * .02
        self.data.qpos[0] = self.t
        return np.full(39, self.t), .0, idx == self.termination, idx == self.truncation, {"success": idx == self.success}

    def close(self):
        self.closed = True


class Client:
    def __init__(self, fail=False, bad=None):
        self.calls, self.requests, self.closed = [], [], False
        self.fail, self.bad = fail, bad

    def select_bundle(self, bundle):
        if bundle == "unknown":
            raise ValueError("unknown bundle")
        self.calls.append(("select", bundle))

    def episode_start(self, **kw):
        self.calls.append(("start", kw))

    def episode_end(self, **kw):
        self.calls.append(("end", kw))

    def infer(self, obs):
        self.requests.append(obs)
        if self.fail:
            raise ConnectionError("lost fixture socket")
        return {"actions": np.array(self.bad, copy=True) if self.bad is not None else np.tile([2, -2, .5, -.7], (5, 1)).astype(np.float32)}

    def close(self):
        self.closed = True


def task(**kwargs):
    base = episode_tasks(campaign="cpu", arm="full", bundle_id="full", endpoint=ServerEndpoint("local", 1),
                         model_identity="model", norm_identity="norm", layout_identity="layout",
                         partition="smoke", names=["reach-v3"])[0]
    return replace(base, **kwargs)


def test_formal_roster_and_split_are_explicit():
    b = MetaWorldBenchmarkAdapter()
    episodes = b.episode_specs()
    assert len(episodes) == 500
    assert len({(r["task_name"], r["original_idx"]) for r in episodes}) == 500
    assert set(r["original_idx"] for r in episodes) == set(range(10))
    assert Counter(r[2] for r in b.tasks()) == {"easy": 28, "medium": 11, "hard": 6, "very_hard": 5}
    assert {r["original_idx"] for r in b.episode_specs("library")} == set(range(10, 30))
    assert {r["original_idx"] for r in b.episode_specs("dev")} == set(range(30, 50))
    assert b.metadata().identity == b.spec.identity
    assert T.EPISODES_PER_TASK == 20  # original September contract unchanged
    assert T.PROMPTS["push-back-v3"] == T.PROMPTS["push-v3"]
    assert T.task_id_of("push-back-v3") != T.task_id_of("push-v3")


def test_task_data_is_selected_by_original_index_not_seed():
    calls = []
    fixtures = [NS(data=f"task-{i}".encode()) for i in range(50)]
    factory = lambda name, seed: (calls.append((name, seed)) or NS(train_tasks=fixtures))
    b = MetaWorldBenchmarkAdapter()
    a = b.task_data_identity("reach-v3", 8, factory)
    assert a == b.task_data_identity("reach-v3", 8, factory)
    assert a["task_data_identity"] != b.task_data_identity("reach-v3", 9, factory)["task_data_identity"]
    assert calls == [("reach-v3", 7)] * 3


@pytest.mark.parametrize("ending", [{"success": 3}, {"termination": 8}, {"truncation": 7}, {}])
def test_kernel_runner_default_off_and_capture_action_observation_parity(ending):
    legacy_env, new_env, captured_env = Env(**ending), Env(**ending), Env(**ending)
    clients = [Client() for _ in range(3)]
    events, acks = [], []
    legacy = MetaworldEpisodeRunner(env_factory=lambda *a: legacy_env, client_factory=lambda _: clients[0])
    new = R17MetaWorldEpisodeRunner(env_factory=lambda *a: new_env, client_factory=lambda _: clients[1],
                                  framework_context=False)
    debug = R17MetaWorldEpisodeRunner(env_factory=lambda *a: captured_env, client_factory=lambda _: clients[2],
                                     capture_sink=events.append, execution_ack=acks.append,
                                     framework_context=False)
    outcomes = [r.run(task(), lambda *a: None) for r in (legacy, new, debug)]
    assert len({(r.success, r.n_steps) for r in outcomes}) == 1
    for other in (new_env, captured_env):
        assert all(a.tobytes() == b.tobytes() for a, b in zip(legacy_env.actions, other.actions, strict=True))
        assert all(a.tobytes() == b.tobytes() for a, b in zip(legacy_env.frames, other.frames, strict=True))
    for other in clients[1:]:
        assert clients[0].calls == other.calls
        for a, b in zip(clients[0].requests, other.requests, strict=True):
            assert a.keys() == b.keys()
            assert a["observation/image"].tobytes() == b["observation/image"].tobytes()
            assert a["observation/state"].tobytes() == b["observation/state"].tobytes()
            assert a["prompt"] == b["prompt"]
    assert sum(r["executed_count"] for r in acks) == outcomes[2].n_steps
    assert events[0]["event"] == "reset" and events[1]["event"] == "settled" and events[-1]["event"] == "terminal"
    controls = [e for e in events if e["event"] == "control"]
    assert len(controls) == outcomes[2].n_steps
    np.testing.assert_array_equal(controls[0]["physical"]["arrays"]["requested_action"], [2, -2, .5, np.float32(-.7)])
    np.testing.assert_array_equal(controls[0]["physical"]["arrays"]["applied_action"], [1, -1, .5, np.float32(-.7)])
    # Every legacy decision row matches with wall clocks excluded.
    for a, b in zip(outcomes[0].per_step_rows[:-1], outcomes[1].per_step_rows[:-1], strict=True):
        assert a == b


@pytest.mark.parametrize("bad", [np.zeros((5, 7)), np.full((5, 4), np.nan), np.zeros((3, 4))])
def test_invalid_actions_close_episode_and_client(bad):
    env, client = Env(), Client(bad=bad)
    runner = R17MetaWorldEpisodeRunner(env_factory=lambda *a: env, client_factory=lambda _: client)
    with pytest.raises(ValueError):
        runner.run(task(attempt=4), lambda *a: None)
    assert env.closed and client.closed
    expected_end = {"success": False}
    if bad.shape == (3, 4):
        # The short finite plan reached the native kernel; no policy control ran.
        expected_end["execution"] = {"decision_seq": 0, "executed_count": 0, "terminal": True}
    assert client.calls[-1] == ("end", expected_end)
    assert len(env.actions) == 15


def test_bundle_switch_client_reuse_and_failure_reconnect():
    clients = [Client(), Client(fail=True), Client()]
    made = []
    def factory(_):
        c = clients[len(made)]
        made.append(c)
        return c
    runner = R17MetaWorldEpisodeRunner(env_factory=lambda *a: Env(success=1), client_factory=factory)
    runner.run(task(), lambda *a: None)
    runner.run(task(bundle_id="k2"), lambda *a: None)
    assert len(made) == 1 and [c for c in made[0].calls if c[0] == "select"] == [("select", "full"), ("select", "k2")]
    with pytest.raises(ValueError, match="unknown"):
        runner.run(task(bundle_id="unknown"), lambda *a: None)
    assert made[0].closed
    with pytest.raises(ConnectionError):
        runner.run(task(), lambda *a: None)
    assert made[1].closed
    runner.run(task(), lambda *a: None)
    assert len(made) == 3


def test_control_exception_acknowledges_only_applied_controls():
    env, client, acks, events = Env(fail=3), Client(), [], []
    r = R17MetaWorldEpisodeRunner(env_factory=lambda *a: env, client_factory=lambda _: client,
                                 capture_sink=events.append, execution_ack=acks.append)
    with pytest.raises(RuntimeError):
        r.run(task(), lambda *a: None)
    assert acks[-1]["executed_count"] == 2 and acks[-1]["last_cursor"] == 1
    assert acks[-1]["terminal"] and acks[-1]["end_reason"] == "error"
    assert events[-1]["end_reason"] == "error"


@pytest.mark.parametrize("factory", [False, True], ids=["runner", "benchmark_factory"])
def test_framework_request_ack_and_original_task_identity(factory):
    client = Client()
    environment_factory = lambda *a: Env(success=7)
    if factory:
        r = MetaWorldBenchmarkAdapter(env_factory=environment_factory).make_episode_runner(
            client_factory=lambda _: client)
    else:
        r = R17MetaWorldEpisodeRunner(env_factory=environment_factory, client_factory=lambda _: client)
    r.run(task(orig_init_state_idx=9, attempt=4), lambda *a: None)
    assert client.requests[0]["__request__"]["original_init_idx"] == 9
    assert client.requests[0]["__request__"]["attempt"] == 4
    assert client.requests[0]["__request__"]["task_name"] == "reach-v3"
    from exp.r17_integration.benchmarks.metaworld import decision_seed
    assert client.requests[0]["__request__"]["explicit_seed"] == decision_seed("reach-v3", 9, 0)
    assert "__execution__" not in client.requests[0]
    assert client.requests[1]["__execution__"] == {"decision_seq": 0, "executed_count": 5, "terminal": False}
    assert client.calls[-1] == ("end", {"success": True,
        "execution": {"decision_seq": 1, "executed_count": 2, "terminal": True}})


def test_native_float64_contacts_and_uncertified_semantics():
    e = Env()
    def force(model, data, i, out):
        out[:] = np.arange(6) + .123456789012
    a = NativeMetaWorldPhysicalAdapter(e, "reach-v3", force)
    c = a.capture(np.arange(39), {"success": False})
    assert c["arrays"]["qpos"].dtype == np.float64
    assert c["arrays"]["contact_force"].dtype == np.float64
    assert c["arrays"]["contact_force"][0, 0] == .123456789012
    assert c["metadata"]["catalog"]["free_joint_body_ids"] == [1]
    assert c["capabilities"]["goal"]["status"] == "available"
    assert c["capabilities"]["predicates"]["status"] == "unsupported"
    assert c["metadata"]["restore_certified"] is False
    e.data.qpos[:] = -8
    assert c["arrays"]["qpos"][0] == 0  # detached numeric capture
    other = NativeMetaWorldPhysicalAdapter(e, "hammer-v3").capture()
    assert other["capabilities"]["goal"]["status"] == "unsupported"
    assert other["capabilities"]["contact_force"]["status"] == "unsupported"


def test_experimental_replan1_has_separate_protocol_identity():
    spec = replace(FORMAL_SPEC, protocol_id="mt50_seed7_corner2_r17_replan1_v1", replan_steps=1)
    assert spec.identity != FORMAL_SPEC.identity


def test_formal_noise_rule_matches_a_and_is_paired_across_arms_retries():
    from exp.mw_finetune.adapters import episode_seed
    from exp.r17_integration.benchmarks.metaworld import decision_seed
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.request_rng import RequestRNG
    for name in T.TASK_NAMES:
        for idx in (0, 9, 30, 49):
            for seq in (0, 1, 31):
                seed = decision_seed(name, idx, seq)
                assert seed == episode_seed(name, idx, decision=seq)
                req = RequestContext("campaign", FORMAL_SPEC.identity, name, idx, seq, explicit_seed=seed)
                a = RequestRNG(req).normal((5, 32))
                b = np.random.default_rng(seed).standard_normal((5, 32)).astype(np.float32)
                assert a.tobytes() == b.tobytes()
                assert a.tobytes() == RequestRNG(replace(req, arm="other", attempt=3, campaign="restarted_campaign")).normal((5, 32)).tobytes()
