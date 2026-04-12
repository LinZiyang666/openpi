from __future__ import annotations

import h5py
import numpy as np
import torch

from exp.build_in_memory_cache_artifact import _build_fake_stage1, build_artifact


def _write_episode(
    path,
    *,
    success: bool,
    task: str,
    with_vision_2: bool = True,
    step_name: str = "step_000",
):
    with h5py.File(path, "w") as f:
        f.attrs["success"] = success
        f.attrs["task"] = task
        g = f.create_group(step_name)
        g.create_dataset("vision_0", data=np.ones((256, 2048), dtype=np.float32))
        g.create_dataset("vision_1", data=np.full((256, 2048), 2.0, dtype=np.float32))
        if with_vision_2:
            g.create_dataset("vision_2", data=np.full((256, 2048), 3.0, dtype=np.float32))
        g.create_dataset("prompt_emb", data=np.full((10, 2048), 4.0, dtype=np.float32))
        g.create_dataset("robot_state", data=np.arange(32, dtype=np.float32))
        g.create_dataset("clean_action", data=np.ones((10, 32), dtype=np.float32))


def test_build_fake_stage1_zero_fills_missing_vision_2(tmp_path):
    path = tmp_path / "ep.h5"
    _write_episode(path, success=True, task="task_a", with_vision_2=False)

    with h5py.File(path, "r") as f:
        stage1 = _build_fake_stage1(f["step_000"])

    assert stage1.prefix_embs.shape == (1, 256 * 3 + 10, 2048)
    vision_2_slice = stage1.prefix_embs[0, 512:768]
    assert torch.count_nonzero(vision_2_slice) == 0
    assert stage1.state.shape == (1, 32)


def test_build_artifact_skips_failed_episodes_and_parses_step_idx(tmp_path):
    _write_episode(tmp_path / "ok.h5", success=True, task="task_ok", step_name="step_042")
    _write_episode(tmp_path / "bad.h5", success=False, task="task_bad")

    artifact = build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1)

    assert artifact["key_builder_type"] == "cp1_mean_pool"
    assert artifact["checkpoint_id"] == "CP1"
    assert artifact["vector_dims"]["vision_0"] == 2048
    assert len(artifact["entries"]) == 1

    entry = artifact["entries"][0]
    assert entry.id == "ok:42"
    assert entry.step_idx == 42
    assert entry.payload.task_key == "task_ok"
    assert entry.payload.action_chunk.shape == (10, 32)
    assert entry.query_keys["vision_0"].shape == (2048,)
    assert entry.query_keys["robot_state"].shape == (32,)


def test_build_artifact_non_numeric_step_suffix_yields_none(tmp_path):
    _write_episode(tmp_path / "ok.h5", success=True, task="task_ok", step_name="step_warmup")

    artifact = build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1)

    assert len(artifact["entries"]) == 1
    assert artifact["entries"][0].step_idx is None
