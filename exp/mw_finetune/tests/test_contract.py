import collections
import json
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from exp.metaworld import env as E
from exp.metaworld import tasks as T
from exp.mw_finetune import adapters as A
from exp.mw_finetune.aggregate import aggregate
from exp.mw_finetune.pi05 import Pi05Policy
from exp.mw_finetune.serve import ConnectionPolicy, checkpoint_identity
from exp.mw_finetune import runtime
from exp.mw_finetune.journal import read_journal


def wire():
    return {"observation/image": np.zeros((480, 480, 3), np.uint8),
            "observation/state": np.asarray([0.1, 0.7, 0.2, 0.9], np.float32),
            "prompt": T.PROMPTS["reach-v3"]}


def test_task_roster_and_formal_episode_identities():
    assert len(T.TASK_NAMES) == len(set(T.TASK_NAMES)) == 50
    assert collections.Counter(T.DIFFICULTY.values()) == {"easy": 28, "medium": 11, "hard": 6, "very_hard": 5}
    roster = T.export_tasks(None, 10)
    assert len(roster) == 50
    assert all(t["init_indices"] == list(range(10)) for t in roster)
    assert sum(len(t["init_indices"]) for t in roster) == 500
    assert T.BENCH_SEED == 7 and T.SETTLE_STEPS == 15 and T.MAX_POLICY_STEPS == 160
    assert T.PROMPTS["push-back-v3"] == "Push the puck to a goal"


def test_wire_canonical_round_trip_preserves_pixels_state_and_prompt():
    obs = wire()
    obs["observation/image"][0, 1] = [11, 25, 219]
    canonical = A.canonical_observation(obs, task_name="reach-v3", seed=42, decision_index=3)
    restored = A.canonical_to_wire(canonical)
    assert restored["observation/image"] is obs["observation/image"]
    np.testing.assert_array_equal(restored["observation/state"], obs["observation/state"])
    assert restored["prompt"] == obs["prompt"]
    assert canonical["task_name"] == "reach-v3"


@pytest.mark.parametrize("change", [
    {"observation/image": np.zeros((3, 480, 480), np.uint8)},
    {"observation/image": np.zeros((480, 480, 3), np.float32)},
    {"observation/state": np.zeros(8)},
    {"observation/state": np.array([0., 0., 0., np.nan])},
    {"prompt": ""},
])
def test_malformed_observations_are_rejected(change):
    obs = wire()
    obs.update(change)
    with pytest.raises(ValueError):
        A.validate_observation(obs)


def test_physical_actions_are_not_clipped_or_gripper_inverted():
    actions = np.tile([2.3, -1.8, 0.5, -0.7], (16, 1)).astype(np.float32)
    np.testing.assert_array_equal(A.validate_actions(actions), actions)
    for bad in [np.zeros((4, 4)), np.zeros((16, 7)), np.full((16, 4), np.inf), np.zeros(4)]:
        with pytest.raises(ValueError):
            A.validate_actions(bad)


def test_groot_four_channel_mapping():
    obs = wire()
    built = A.groot_observation(obs)
    assert built["video.image"].shape == (1, 480, 480, 3)
    for i, key in enumerate(A.CHANNELS):
        np.testing.assert_array_equal(built[f"state.{key}"][0], obs["observation/state"][i:i+1])
    raw = {f"action.{key}": np.full((1, 16, 1), i / 3) for i, key in enumerate(A.CHANNELS)}
    np.testing.assert_array_equal(A.groot_actions(raw)[0], np.asarray([0, 1/3, 2/3, 1], np.float32))


def test_connection_rng_and_reset_are_independent_of_interleaving():
    class Fake:
        def reset(self):
            self.reset_calls += 1
        def infer(self, obs):
            self.requests.append(obs)
            return {"actions": np.random.normal(size=(5, 4)).astype(np.float32)}
    fake = Fake()
    fake.reset_calls, fake.requests = 0, []
    lock = threading.Lock()
    a, b = ConnectionPolicy(fake, lock), ConnectionPolicy(fake, lock)
    with pytest.raises(RuntimeError):
        a.infer(wire())
    for p, idx in [(a, 0), (b, 1)]:
        p.on_episode_start(T.BENCHMARK, "reach-v3", idx)
    first = a.infer(wire())["actions"]
    b.infer(wire())
    a.infer(wire())
    a.on_episode_start(T.BENCHMARK, "reach-v3", 0)
    np.testing.assert_array_equal(first, a.infer(wire())["actions"])
    assert fake.reset_calls == 4
    assert fake.requests[0]["seed"] != fake.requests[1]["seed"]
    assert fake.requests[0]["seed"] == fake.requests[3]["seed"]
    a.on_episode_end()
    with pytest.raises(RuntimeError):
        a.infer(wire())


def test_connection_rejects_wrong_prompt_and_benchmark_seed():
    p = ConnectionPolicy(Mock(), threading.Lock())
    with pytest.raises(ValueError):
        p.on_episode_start(T.BENCHMARK, "reach-v3", 0, {"seed": 8})
    p.on_episode_start(T.BENCHMARK, "reach-v3", 0)
    obs = wire()
    obs["prompt"] = "LIBERO task"
    with pytest.raises(ValueError):
        p.infer(obs)


def rows_for_all_tasks():
    return [{"task": name, "idx": idx, "seed": 7, "success": T.DIFFICULTY[name] == "easy", "error": None}
            for name in T.TASK_NAMES for idx in range(10)]


def test_aggregation_distinguishes_task_and_group_weighting():
    report = aggregate(rows_for_all_tasks())
    assert report["complete"] and report["completed_episodes"] == 500
    assert report["overall_per_task_mean"] == 28 / 50
    assert report["equal_group_mean"] == 0.25
    assert report["per_group"]["easy"] == {"tasks": 28, "episodes": 280, "sr": 1.0}
    assert report["formal_mt50"] is True
    assert aggregate(rows_for_all_tasks()[:10], ["assembly-v3"], 10)["formal_mt50"] is False


def test_errors_are_missing_and_retry_then_completes_without_double_counting():
    rows = rows_for_all_tasks()
    failed = dict(rows[0], success=None, error="network failed")
    incomplete = [failed, *rows[1:]]
    with pytest.raises(ValueError, match="incomplete"):
        aggregate(incomplete)
    partial = aggregate(incomplete, require_complete=False)
    assert not partial["complete"] and partial["completed_episodes"] == 499
    report = aggregate([*incomplete, rows[0]])
    assert report["complete"] and report["error_attempts"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        aggregate([*rows, rows[0]])
    with pytest.raises(ValueError, match="seed"):
        aggregate([dict(rows[0], seed=8)], require_complete=False)


def test_pi05_uses_existing_wire_keys_and_explicit_repeatable_noise():
    base = Mock()
    base.infer.return_value = {"actions": np.tile([0.1, 0.2, 0.3, -0.6], (5, 1))}
    p = Pi05Policy(base, 10)
    obs = A.canonical_observation(wire(), task_name="reach-v3", seed=101, decision_index=0)
    p.infer(obs)
    first_noise = base.infer.call_args.kwargs["noise"].copy()
    p.reset()
    result = p.infer(obs)
    assert set(base.infer.call_args.args[0]) == {"observation/image", "observation/state", "prompt"}
    assert first_noise.shape == (5, 32)
    np.testing.assert_array_equal(first_noise, base.infer.call_args.kwargs["noise"])
    assert result["actions"][0, 3] == np.float32(-0.6)


def test_checkpoint_identity_detects_changed_config_and_weights(tmp_path):
    (tmp_path / "config.json").write_text('{"action_dim":4}')
    (tmp_path / "model.safetensors").write_bytes(b"one")
    first = checkpoint_identity(tmp_path)
    (tmp_path / "config.json").write_text('{"action_dim":7}')
    assert checkpoint_identity(tmp_path) != first


def test_h100_disk_floor_and_shared_budget(monkeypatch):
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "h100")
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: Mock(free=200e9))
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **kw: "115000000000 /data/mw_finetune\n")
    with pytest.raises(RuntimeError, match="exceeds120"):
        runtime.disk_gate("/data/mw_finetune/test", 10)
    assert runtime.disk_gate("/data/mw_finetune/test", 3) == 200
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: Mock(free=32e9))
    with pytest.raises(RuntimeError, match="floor"):
        runtime.disk_gate("/data/mw_finetune/test", 3)


def test_journal_recovers_only_torn_tail_and_preserves_original(tmp_path):
    path = tmp_path / "episodes.jsonl"
    original = b'{"task":"reach-v3","idx":0}\n{"task":'
    path.write_bytes(original)
    with pytest.raises(ValueError):
        read_journal(path)
    rows = read_journal(path, recover_tail=True)
    assert rows == [{"task": "reach-v3", "idx": 0}]
    assert path.read_bytes() == original.splitlines(keepends=True)[0]
    assert next(tmp_path.glob("episodes.jsonl.torn.*")).read_bytes() == original
    path.write_bytes(b'{bad}\n{"idx":1}\n')
    with pytest.raises(ValueError):
        read_journal(path, recover_tail=True)


def test_valid_final_journal_record_without_newline_can_resume(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text('{"idx":0}')
    assert read_journal(path, recover_tail=True) == [{"idx": 0}]
    with path.open("a") as out:
        out.write('{"idx":1}\n')
    assert read_journal(path) == [{"idx": 0}, {"idx": 1}]


class FakeEnv:
    def __init__(self, success_at=None):
        self.actions = []
        self.success_at = success_at
        self.image = np.arange(480 * 480 * 3, dtype=np.uint32).reshape(480, 480, 3).astype(np.uint8)
    def reset(self):
        return np.arange(39, dtype=np.float32), {}
    def render(self):
        return self.image
    def step(self, action):
        self.actions.append(action.copy())
        return np.arange(39, dtype=np.float32), 0, False, False, {"success": len(self.actions) == self.success_at}


def test_env_conventions_rotation_settle_replan_early_success():
    env = FakeEnv(success_at=18)
    calls = []
    def infer(obs):
        calls.append(obs)
        return {"actions": np.tile([2, -2, .1, -.6], (16, 1))}
    outcome = E.run_episode(env, infer, "Reach a goal position")
    assert outcome.success and outcome.n_steps == 3 and outcome.n_decisions == 1
    assert len(env.actions) == 18
    assert all(np.array_equal(a, np.zeros(4)) for a in env.actions[:15])
    np.testing.assert_array_equal(calls[0]["observation/image"], env.image[::-1, ::-1])
    np.testing.assert_array_equal(calls[0]["observation/state"], np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(env.actions[15], [2, -2, np.float32(.1), np.float32(-.6)])
