"""The pure-noise start and the schedule stamp on the Pi0.5 collection path."""

from __future__ import annotations

from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import torch

from openpi.collect.collection_policy import CollectionPolicy
from openpi.collect.data_collector import EpisodeDataCollector, InferenceEmbeddings


def _policy_with_collector():
    cp = object.__new__(CollectionPolicy)
    cp._collector = MagicMock()
    return cp


def _captures(num_steps: int):
    ins = [torch.full((1, 10, 32), float(i)) for i in range(num_steps)]
    outs = [torch.full((1, 10, 32), -1.0) for _ in range(num_steps)]
    return ins, outs


def test_record_keeps_x0_as_init_noise_and_stamps_the_pi05_schedule():
    cp = _policy_with_collector()
    ins, outs = _captures(10)
    cp._record(
        np.zeros(32, np.float32),
        {},
        [torch.zeros(1, 4, 8)],
        torch.ones(1, 3, 8),
        ins,
        outs,
    )

    embs: InferenceEmbeddings = cp._collector.record_inference.call_args.args[0]
    assert embs.init_noise.shape == (10, 32) and float(embs.init_noise[0, 0]) == 0.0
    assert len(embs.noise_action_steps) == 9
    assert float(embs.noise_action_steps[0][0, 0]) == 1.0  # index 1, not the noise
    stamped = {
        c.args[0]: c.args[1] for c in cp._collector.set_episode_attr.call_args_list
    }
    assert stamped == {"denoise_schedule_id": "pi05_v1", "denoising_num_steps": 10}


def test_record_refuses_a_step_count_the_pi05_schedule_does_not_describe():
    cp = _policy_with_collector()
    ins, outs = _captures(4)
    with pytest.raises(RuntimeError, match="defined as 10"):
        cp._record(
            np.zeros(32, np.float32),
            {},
            [torch.zeros(1, 4, 8)],
            torch.ones(1, 3, 8),
            ins,
            outs,
        )
    cp._collector.record_inference.assert_not_called()


def test_writer_emits_noise_action_0_only_when_init_noise_is_given(tmp_path):
    collector = EpisodeDataCollector(str(tmp_path))
    collector.on_episode_start("exp", "task", 0, episode_name="task_0/ep_0")
    base = dict(
        vision_embs=[np.zeros((4, 8), np.float16)],
        prompt_emb=np.zeros((3, 8), np.float16),
        robot_state=np.zeros(32, np.float32),
        noise_action_steps=[np.full((10, 32), 1.0, np.float32)],
        clean_action=np.zeros((10, 32), np.float32),
    )
    collector.record_inference(
        InferenceEmbeddings(**base, init_noise=np.full((10, 32), 7.0, np.float32))
    )
    collector.record_inference(InferenceEmbeddings(**base))
    collector.on_episode_end(success=True)

    files = list(tmp_path.rglob("*.h5"))
    assert len(files) == 1
    with h5py.File(files[0]) as f:
        assert float(f["step_0000"]["noise_action_0"][0, 0]) == 7.0
        assert "noise_action_1" in f["step_0000"]
        assert "noise_action_0" not in f["step_0001"]
        assert "noise_action_1" in f["step_0001"]
