"""The pure-noise start (``noise_action_0``) on the HDF5 writer.

The Pi0.5 capture itself is the trace serving mode now; its parity with the
legacy hooks is the GPU gate ``tests/cache/trace/test_trace_collect_parity_gpu.py``.
"""

from __future__ import annotations

import h5py
import numpy as np

from openpi.collect.data_collector import EpisodeDataCollector, InferenceEmbeddings


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
