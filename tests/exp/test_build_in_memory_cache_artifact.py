from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from exp.common.build_in_memory_cache_artifact import _build_fake_stage1, build_artifact


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


# ---------------------------------------------------------------------------
# Projection key builder (M1 Phase 2) — offline library side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inner", ["cp1_mean_pool", "cp1_spatial_pool_4", "cp1_spatial_pool_64"]
)
def test_projection_identity_matches_inner_via_build_artifact(tmp_path, inner):
    """Identity projection (no weights) built through the top-level build_artifact
    entry point is value-equal to the wrapped pool builder — for both the
    canonical spatial_pool_4 name and its legacy _64 alias."""
    _write_episode(tmp_path / "ok.h5", success=True, task="t", step_name="step_000")

    proj = build_artifact(str(tmp_path), "projection", workers=-1, inner_type=inner)
    ref = build_artifact(str(tmp_path), inner, workers=-1)

    assert proj["key_builder_type"] == "projection"
    assert proj["vector_dims"] == ref["vector_dims"]
    assert proj["projection_params"] == {
        "inner_type": inner,
        "projection_weights_path": None,
    }

    proj_keys = proj["entries"][0].query_keys
    ref_keys = ref["entries"][0].query_keys
    assert set(proj_keys) == set(ref_keys)
    for field in ref_keys:
        # _detach_entries has converted tensors to numpy before returning.
        assert np.array_equal(proj_keys[field], ref_keys[field]), field


def test_get_vector_dims_projection_identity():
    from exp.common.build_in_memory_cache_artifact import _VECTOR_DIMS, _get_vector_dims

    for inner in ("cp1_mean_pool", "cp1_spatial_pool_4", "cp1_spatial_pool_16"):
        assert _get_vector_dims("projection", inner_type=inner) == _VECTOR_DIMS[inner]


def test_projection_weighted_same_head_online_and_offline(tmp_path):
    """The core two-side contract: the SAME saved projection weights produce the
    SAME projected keys through the online config factory and the offline
    artifact builder."""
    from openpi.cache.components.projection_key_builder import (
        ProjectionHead,
        ProjectionParams,
    )
    from openpi.cache.config import (
        KeyBuilderConfig,
        ProjectionKeyBuilderConfig,
        _build_key_builder,
    )
    from openpi.cache.types import (
        PROMPT_EMB,
        ROBOT_STATE,
        VISION_0,
        VISION_1,
        CheckpointID,
    )

    out_dim, emb = 16, 2048
    torch.manual_seed(0)
    params = ProjectionParams(
        {
            VISION_0: ProjectionHead(torch.randn(out_dim, emb)),
            VISION_1: ProjectionHead(torch.randn(out_dim, emb)),
            PROMPT_EMB: ProjectionHead(torch.randn(out_dim, emb)),
        }
    )
    wpath = tmp_path / "proj.pt"
    params.save(wpath)

    _write_episode(
        tmp_path / "ok.h5", success=True, task="t",
        with_vision_2=False, step_name="step_000",
    )

    # Offline: build the projected artifact through the top-level entry point.
    art = build_artifact(
        str(tmp_path), "projection", workers=-1,
        inner_type="cp1_mean_pool", projection_weights_path=str(wpath),
    )
    off = art["entries"][0].query_keys
    assert art["vector_dims"][VISION_0] == out_dim
    assert art["vector_dims"][PROMPT_EMB] == out_dim
    assert art["vector_dims"][ROBOT_STATE] == 32  # passthrough, not projected

    # Online: same weights via the config factory, fed the same reconstructed
    # stage1 (so identical pooled features enter identical heads).
    with h5py.File(tmp_path / "ok.h5", "r") as f:
        stage1 = _build_fake_stage1(f["step_000"])
    kb = _build_key_builder(
        KeyBuilderConfig(
            type="projection",
            projection=ProjectionKeyBuilderConfig(
                inner_type="cp1_mean_pool", weights_path=str(wpath)
            ),
        ),
        enabled_fields=[VISION_0, VISION_1, PROMPT_EMB, ROBOT_STATE],
        vector_dims={VISION_0: out_dim, VISION_1: out_dim, PROMPT_EMB: out_dim, ROBOT_STATE: 32},
    )
    kb.collect(CheckpointID.CP1, stage1=stage1)
    on = kb.build(CheckpointID.CP1)

    for field in (VISION_0, VISION_1, PROMPT_EMB):
        assert on[field].shape == (out_dim,)
        assert np.allclose(on[field].numpy(), off[field], atol=1e-5), field
    assert on[ROBOT_STATE].shape == (32,)
