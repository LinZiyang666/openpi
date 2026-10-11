import numpy as np
import pytest

from exp.mw_finetune.groot.conventions import (
    CHANNELS, action_chunk, from_groot_actions, modality_metadata, statistics, to_groot_observation,
)
from exp.mw_finetune.groot.policy import MetaWorldPolicy


def observation():
    image = np.zeros((480, 480, 3), dtype=np.uint8)
    image[0, 0] = [1, 2, 3]
    image[-1, -1] = [4, 5, 6]
    return {"image": image, "state": np.array([.1, .7, .2, .6], dtype=np.float32),
            "prompt": "Pick up a nut and place it onto a peg", "task_name": "assembly-v3"}


def test_observation_is_already_render_oriented_and_rgb():
    obs = observation()
    converted = to_groot_observation(obs)
    np.testing.assert_array_equal(converted["video.image"][0], obs["image"])
    for i, name in enumerate(CHANNELS):
        assert converted[f"state.{name}"].shape == (1, 1)
        assert converted[f"state.{name}"][0, 0] == obs["state"][i]
    assert converted["annotation.human.action.task_description"].tolist() == [obs["prompt"]]


@pytest.mark.parametrize("batched", [False, True])
def test_actions_preserve_control_units_and_gripper(batched):
    actions = np.tile([2.3, -1.7, .25, -.6], (16, 1)).astype(np.float32)
    result = {f"action.{name}": actions[:, i:i + 1][None] if batched else actions[:, i:i + 1]
              for i, name in enumerate(CHANNELS)}
    np.testing.assert_array_equal(from_groot_actions(result), actions)


def test_tail_is_zero_padded_without_crossing_episode():
    actions = np.arange(12, dtype=np.float32).reshape(3, 4)
    chunk = action_chunk(actions, 2)
    np.testing.assert_array_equal(chunk[0], actions[-1])
    np.testing.assert_array_equal(chunk[1:], np.zeros((15, 4)))
    assert modality_metadata()["action"]["gripper"]["absolute"] is False


def test_statistics_population_and_quantiles():
    values = np.arange(40).reshape(10, 4)
    stats = statistics(values)
    np.testing.assert_allclose(stats["std"], values.std(axis=0))
    np.testing.assert_allclose(stats["q01"], np.quantile(values, .01, axis=0))
    np.testing.assert_allclose(stats["q99"], np.quantile(values, .99, axis=0))


def test_reset_does_not_queue_or_change_a_full_chunk():
    class Backend:
        calls = 0
        def get_action(self, obs):
            self.calls += 1
            return {f"action.{name}": np.full((16, 1), self.calls + i, dtype=np.float32)
                    for i, name in enumerate(CHANNELS)}
    backend = Backend()
    policy = MetaWorldPolicy(backend, "checkpoint", 4)
    first = policy.infer(observation())["actions"]
    policy.reset()
    second = policy.infer(observation())["actions"]
    assert backend.calls == 2 and first.shape == second.shape == (16, 4)
    np.testing.assert_array_equal(second, first + 1)
    assert policy.get_metadata()["embodiment"] == "metaworld_sawyer"


def test_reject_nan_and_bad_camera():
    obs = observation()
    obs["state"][0] = np.nan
    with pytest.raises(ValueError):
        to_groot_observation(obs)
    obs = observation()
    obs["image"] = obs["image"].astype(np.float32)
    with pytest.raises(ValueError):
        to_groot_observation(obs)


def test_reject_short_or_nonfinite_output():
    values = {f"action.{name}": np.zeros((15, 1)) for name in CHANNELS}
    with pytest.raises(ValueError):
        from_groot_actions(values)
    values = {f"action.{name}": np.full((16, 1), np.nan) for name in CHANNELS}
    with pytest.raises(ValueError):
        from_groot_actions(values)
