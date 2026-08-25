"""Contract tests for the LIBERO<->GR00T translation.

The reason these are worth writing: every plausible mistake in this file
(swapped language key, wrong state split, missing time axis) still yields
finite arrays of a legal shape, so only an explicit contract check catches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from exp.libero_groot import libero_keys as K
from exp.libero_groot.policy_adapter import (
    GrootLiberoPolicyAdapter,
    build_groot_observation,
    chunk_to_libero_actions,
    iter_step_actions,
    normalize_gripper_action,
    validate_action_chunk,
)


def _wire_obs(**overrides):
    obs = {
        K.WIRE_IMAGE: np.zeros((256, 256, 3), np.uint8),
        K.WIRE_WRIST: np.ones((256, 256, 3), np.uint8),
        K.WIRE_STATE: np.arange(8, dtype=np.float64),
        K.WIRE_PROMPT: "pick up the black bowl",
    }
    obs.update(overrides)
    return obs


def _raw_chunk():
    return {k: np.full((K.ACTION_HORIZON, 1), i, dtype=np.float32)
            for i, k in enumerate(K.ACTION_KEYS)}


class TestObservation:
    def test_state_split_matches_the_official_evaluator(self):
        out = build_groot_observation(_wire_obs())
        # six scalars, then the 2-D gripper qpos -- the split the eval script uses
        for i, key in enumerate(K.SCALAR_STATE_KEYS):
            assert out[key].shape == (1, 1)
            assert out[key][0, 0] == float(i)
        assert out[K.GRIPPER_STATE_KEY].shape == (1, 2)
        assert list(out[K.GRIPPER_STATE_KEY][0]) == [6.0, 7.0]

    def test_language_key_is_the_libero_one(self):
        out = build_groot_observation(_wire_obs())
        assert K.LANGUAGE_KEY == "annotation.human.action.task_description"
        # the RoboCasa365 key must NOT appear: it differs by one path segment
        # and would leave the policy unconditioned
        assert "annotation.human.task_description" not in out

    def test_images_gain_a_time_axis_and_keep_dtype(self):
        out = build_groot_observation(_wire_obs())
        for key in K.VIDEO_KEYS:
            assert out[key].shape == (1, 256, 256, 3)
            assert out[key].dtype == np.uint8

    @pytest.mark.parametrize(
        "override, match",
        [
            ({K.WIRE_STATE: np.zeros(7)}, "must have shape"),
            ({K.WIRE_IMAGE: np.zeros((512, 512, 3), np.uint8)}, "must have shape"),
            ({K.WIRE_IMAGE: np.zeros((256, 256, 3), np.float32)}, "must be uint8"),
            ({K.WIRE_STATE: np.full(8, np.nan)}, "non-finite"),
        ],
    )
    def test_malformed_input_is_rejected_not_repaired(self, override, match):
        with pytest.raises(ValueError, match=match):
            build_groot_observation(_wire_obs(**override))

    def test_missing_key_names_the_offender(self):
        obs = _wire_obs()
        del obs[K.WIRE_PROMPT]
        with pytest.raises(ValueError, match=K.WIRE_PROMPT):
            build_groot_observation(obs)


class TestAction:
    def test_chunk_order_is_libero_action_order(self):
        stacked = chunk_to_libero_actions(validate_action_chunk(_raw_chunk()))
        assert stacked.shape == (K.ACTION_HORIZON, 7)
        # key i was filled with value i, so pose column i must be i; the
        # gripper column is converted rather than passed through.
        assert list(stacked[0][:6]) == [float(i) for i in range(6)]
        assert stacked[0][6] == -1.0  # 1 - 2*6 = -11 -> sign

    def test_gripper_is_converted_to_the_libero_convention(self):
        """GR00T emits openness in [0, 1]; robosuite reads +1 as *close*.

        Skipping this leaves the arm reaching for objects it never grips --
        no exception, no NaN, just every episode burning its step budget.
        """
        chunk = np.zeros((3, 7))
        chunk[:, 6] = [0.0, 1.0, 0.9]
        out = normalize_gripper_action(chunk.copy())
        # y = 1 - 2x then sign: fully open -> +1, fully closed -> -1.
        assert list(out[:, 6]) == [1.0, -1.0, -1.0]

    def test_gripper_conversion_leaves_the_pose_columns_alone(self):
        chunk = np.arange(14, dtype=np.float64).reshape(2, 7)
        out = normalize_gripper_action(chunk.copy())
        np.testing.assert_array_equal(out[:, :6], chunk[:, :6])

    def test_iter_step_actions_shares_the_conversion(self):
        rows = list(iter_step_actions(_raw_chunk(), replan_steps=2))
        assert len(rows) == 2
        assert all(row[6] == -1.0 for row in rows)

    def test_short_chunk_is_rejected(self):
        raw = _raw_chunk()
        raw["action.x"] = np.zeros((K.ACTION_HORIZON - 1, 1), np.float32)
        with pytest.raises(ValueError, match="must have shape"):
            validate_action_chunk(raw)

    def test_unexpected_key_is_rejected(self):
        raw = _raw_chunk()
        raw["action.extra"] = np.zeros((K.ACTION_HORIZON, 1), np.float32)
        with pytest.raises(ValueError, match="unexpected action keys"):
            validate_action_chunk(raw)


class TestAdapter:
    def test_infer_round_trip(self):
        class _Stub:
            def get_action(self, observations):
                assert observations[K.LANGUAGE_KEY][0] == "pick up the black bowl"
                return _raw_chunk()

        out = GrootLiberoPolicyAdapter(_Stub()).infer(_wire_obs())
        assert out["actions"].shape == (K.ACTION_HORIZON, 7)

    def test_side_channel_travels_beside_actions_not_inside_them(self):
        """The cache interceptor attaches ``__hit_meta__`` to its infer result.

        Validation rejects unknown keys on purpose, so a side channel that was
        not lifted out first turns every cached step into a hard error -- while
        the collection path, which has no such key, stays green.
        """
        meta = {"hit": True, "kb_id": "lib-1"}

        class _Stub:
            def get_action(self, observations):
                return {**_raw_chunk(), "__hit_meta__": meta}

        out = GrootLiberoPolicyAdapter(_Stub()).infer(_wire_obs())
        assert out["__hit_meta__"] is meta
        assert out["actions"].shape == (K.ACTION_HORIZON, 7)

    def test_unexpected_non_dunder_key_is_still_rejected(self):
        class _Stub:
            def get_action(self, observations):
                return {**_raw_chunk(), "action.extra": np.zeros((K.ACTION_HORIZON, 1))}

        with pytest.raises(ValueError, match="unexpected action keys"):
            GrootLiberoPolicyAdapter(_Stub()).infer(_wire_obs())

    def test_episode_signals_reach_the_wrapped_policy(self):
        """The collector opens/closes its HDF5 on these; swallowing them
        yields a clean-looking rollout that writes nothing."""
        seen = {}

        class _Collector:
            def get_action(self, observations):
                return _raw_chunk()

            def on_episode_start(self, **kw):
                seen["start"] = kw

            def on_episode_end(self, success):
                seen["end"] = success

        adapter = GrootLiberoPolicyAdapter(_Collector())
        adapter.on_episode_start(experiment="libero_spatial", task="t", episode_id=3)
        adapter.on_episode_end(success=True)
        assert seen["start"]["experiment"] == "libero_spatial"
        assert seen["start"]["episode_id"] == 3
        assert seen["end"] is True

    def test_forwarding_is_optional(self):
        class _Plain:
            def get_action(self, observations):
                return _raw_chunk()

        # a policy without the hooks must not raise
        GrootLiberoPolicyAdapter(_Plain()).on_episode_end(success=False)
