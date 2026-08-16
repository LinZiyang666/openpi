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

# ---------------------------------------------------------------------------
# Cache-size ablation (X9b): --episode-list + --trajectory-id-mode
# ---------------------------------------------------------------------------


def _write_task_tree(root, n_tasks=3, success=True):
    """Lay episodes out as task_N/episode_0.h5 -- the collision-prone shape."""
    for t in range(n_tasks):
        d = root / f"task_{t}"
        d.mkdir(parents=True, exist_ok=True)
        _write_episode(d / "episode_0.h5", success=success, task=f"task_{t}", step_name="step_000")


def test_relpath_mode_yields_unique_ids_across_tasks(tmp_path):
    _write_task_tree(tmp_path, n_tasks=3)

    artifact = build_artifact(
        str(tmp_path), "cp1_mean_pool", workers=-1, trajectory_id_mode="relpath"
    )

    ids = [e.id for e in artifact["entries"]]
    assert len(ids) == 3
    assert len(set(ids)) == 3, f"ids collided: {ids}"
    assert set(ids) == {"task_0/episode_0:0", "task_1/episode_0:0", "task_2/episode_0:0"}


def test_stem_mode_collides_across_tasks_known_behavior(tmp_path):
    """Pin the collision as known behavior so nobody flips the default silently.

    Under the historical `stem` mode every task's ``episode_0.h5`` maps to the
    same ``episode_0:0``. build() still emits one entry per file -- the loss only
    materializes when a backend keys entries by id (see the load test below).
    """
    _write_task_tree(tmp_path, n_tasks=3)

    artifact = build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1)

    ids = [e.id for e in artifact["entries"]]
    assert len(ids) == 3
    assert set(ids) == {"episode_0:0"}, "stem mode is expected to collide here"


def test_backend_load_drops_entries_under_stem_but_not_relpath(tmp_path):
    """`len(entries) == len(backend._entries)` is the general probe for id collisions."""
    from openpi.cache.backends.in_memory_backend import InMemoryBackend

    _write_task_tree(tmp_path, n_tasks=3)
    dims = {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32}

    def _loaded_count(mode):
        artifact = build_artifact(
            str(tmp_path), "cp1_mean_pool", workers=-1, trajectory_id_mode=mode
        )
        out = tmp_path / f"{mode}.pkl"
        import pickle

        with open(out, "wb") as fh:
            pickle.dump(artifact, fh)
        backend = InMemoryBackend(vector_dims=dims)
        backend.load_artifact(str(out))
        return len(artifact["entries"]), len(backend._entries)

    built_stem, loaded_stem = _loaded_count("stem")
    built_rel, loaded_rel = _loaded_count("relpath")

    assert (built_stem, loaded_stem) == (3, 1), "stem mode silently drops 2 of 3 entries"
    assert built_rel == loaded_rel == 3, "relpath mode must not drop entries"


def test_relpath_rejected_for_model_builder(tmp_path):
    _write_task_tree(tmp_path, n_tasks=1)
    with pytest.raises(ValueError, match="only supported for pool builders"):
        build_artifact(
            str(tmp_path), "cp1_llm_layer_extract", workers=-1, trajectory_id_mode="relpath"
        )


def test_episode_list_selects_subset(tmp_path):
    _write_task_tree(tmp_path, n_tasks=3)
    listing = tmp_path / "subset.txt"
    listing.write_text("task_0/episode_0.h5\ntask_2/episode_0.h5\n")

    artifact = build_artifact(
        str(tmp_path),
        "cp1_mean_pool",
        workers=-1,
        episode_list=str(listing),
        trajectory_id_mode="relpath",
    )

    assert {e.id for e in artifact["entries"]} == {"task_0/episode_0:0", "task_2/episode_0:0"}


def test_episode_list_absent_matches_recursive_scan(tmp_path):
    """Non-regression: leaving both new params unset reproduces the old behavior."""
    from exp.common.build_in_memory_cache_artifact import resolve_h5_paths

    _write_task_tree(tmp_path, n_tasks=3)
    assert resolve_h5_paths(tmp_path, None) == sorted(tmp_path.rglob("*.h5"))


@pytest.mark.parametrize(
    "content, exc, match",
    [
        ("/abs/path.h5\n", ValueError, "absolute path"),
        ("../outside.h5\n", ValueError, "escapes"),
        ("task_0/episode_0.txt\n", ValueError, "not a .h5 file"),
        ("task_0/missing.h5\n", FileNotFoundError, "does not exist"),
        ("task_0/episode_0.h5\n\ntask_1/episode_0.h5\n", ValueError, "blank line"),
        ("task_0/episode_0.h5\ntask_0/../task_0/episode_0.h5\n", ValueError, "duplicate"),
        ("", ValueError, "no episodes listed"),
    ],
)
def test_episode_list_rejects_malformed_entries(tmp_path, content, exc, match):
    from exp.common.build_in_memory_cache_artifact import resolve_h5_paths

    _write_task_tree(tmp_path, n_tasks=2)
    listing = tmp_path / "bad.txt"
    listing.write_text(content)

    with pytest.raises(exc, match=match):
        resolve_h5_paths(tmp_path, listing)


def test_trajectory_id_mode_rejects_unknown_value(tmp_path):
    _write_task_tree(tmp_path, n_tasks=1)
    with pytest.raises(ValueError, match="trajectory_id_mode must be one of"):
        build_artifact(str(tmp_path), "cp1_mean_pool", workers=-1, trajectory_id_mode="nope")
