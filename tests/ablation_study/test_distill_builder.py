"""Distillation builder tests over a canonical trajectory H5 fixture."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from exp.ablation_study.build_distill_dataset import cache_positions_for_task
from exp.ablation_study.build_distill_dataset import check_init_dir
from exp.ablation_study.build_distill_dataset import constrained_split
from exp.ablation_study.build_distill_dataset import iter_frames

TASK_NAME = "pick up the black bowl and place it on the plate"


def _write_episode(path, ep_idx, success=True, cycles=3):
    """Canonical --save_trajectory schema (examples/libero/main.py writer)."""
    with h5py.File(path, "w") as f:
        f.attrs["task_name"] = TASK_NAME
        f.attrs["task_id"] = 0
        f.attrs["init_state_idx"] = ep_idx
        f.attrs["success"] = success
        for i in range(cycles):
            g = f.create_group(f"step_{i:04d}")
            g.create_dataset("agentview_image", data=np.zeros((224, 224, 3), np.uint8))
            g.create_dataset("eye_in_hand_image", data=np.zeros((224, 224, 3), np.uint8))
            g.create_dataset("robot_state", data=np.zeros(8, np.float64))
            g.create_dataset("env_action_chunk", data=np.ones((10, 7), np.float32))
            g.create_dataset("executed_actions", data=np.ones((5, 7), np.float32))


def _raw_tree(tmp_path, n_eps=6):
    task_dir = tmp_path / "task_0"
    task_dir.mkdir()
    for i in range(n_eps):
        _write_episode(task_dir / f"episode_{i}.h5", i, success=(i != 1))
    return tmp_path


def test_iter_frames_reads_task_name_and_filters_failures(tmp_path):
    raw = _raw_tree(tmp_path)
    split = {"task_0": {"train": list(range(6)), "val": [], "task_name": TASK_NAME}}
    frames = list(iter_frames(str(raw), split, "train"))
    # 6 episodes, 1 failed -> 5 successful x 3 cycles.
    assert len(frames) == 15
    for frame, task in frames:
        assert task == TASK_NAME  # canonical attrs["task_name"], not attrs["task"]
        assert frame["actions"].shape == (10, 7)
        assert frame["state"].dtype == np.float32


def test_constrained_split_protects_cache_positions():
    rng = np.random.RandomState(0)
    episodes = list(range(50))
    protected = [3, 11, 27, 40, 49]
    out = constrained_split(episodes, protected, val_n=5, rng=rng)
    assert set(out["val"]).isdisjoint(protected)
    assert set(out["train"]) | set(out["val"]) == set(episodes)
    assert out["protected_in_train"] == sorted(protected)
    assert all(p in out["train"] for p in protected)


def test_constrained_split_insufficient_candidates():
    rng = np.random.RandomState(0)
    with pytest.raises(SystemExit, match="not enough"):
        constrained_split(list(range(6)), [0, 1, 2, 3], val_n=3, rng=rng)


def test_cache_positions_matching():
    pool = np.arange(50 * 4, dtype=np.float64).reshape(50, 4)
    cache = pool[[5, 17, 42]]
    assert cache_positions_for_task(pool, cache) == [5, 17, 42]
    with pytest.raises(SystemExit, match="matched"):
        cache_positions_for_task(pool, np.full((1, 4), -1.0))


def test_emit_split_resolves_underscore_stems(tmp_path, monkeypatch):
    # End-to-end split over the REAL naming contract: H5 carries the natural-
    # language prompt while .init files use the underscore task.name stem.
    import torch

    from exp.ablation_study.build_distill_dataset import emit_split

    raw = tmp_path / "raw"
    raw.mkdir()
    task_dir = raw / "task_0"
    task_dir.mkdir()
    for i in range(8):
        _write_episode(task_dir / f"episode_{i}.h5", i)
    stem = TASK_NAME.replace(" ", "_")
    pool = tmp_path / "pool"
    pool.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    pool_states = np.arange(8 * 4, dtype=float).reshape(8, 4)
    torch.save(pool_states, pool / f"{stem}.init")
    torch.save(pool_states[[0, 2]], cache / f"{stem}.init")
    out = tmp_path / "split.yaml"
    split = emit_split(str(raw), str(pool), str(cache), str(out))
    assert split["task_0"]["init_stem"] == stem
    assert split["task_0"]["task_name"] == TASK_NAME
    assert set(split["task_0"]["protected_in_train"]) == {0, 2}
    assert set(split["task_0"]["val"]).isdisjoint({0, 2})


def test_resolve_init_stem_families(tmp_path):
    from exp.ablation_study.build_distill_dataset import resolve_init_stem

    d = tmp_path / "inits"
    d.mkdir()
    # libero_spatial family: prefix-free stem.
    (d / "pick_up_the_black_bowl_and_place_it_on_the_plate.init").write_bytes(b"x")
    # libero_10/90 family: scene-prefixed stem.
    (d / "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.init").write_bytes(b"x")
    assert resolve_init_stem("pick up the black bowl and place it on the plate", str(d)) \
        == "pick_up_the_black_bowl_and_place_it_on_the_plate"
    assert resolve_init_stem("turn on the stove and put the moka pot on it", str(d)) \
        == "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    # Authoritative map wins over any guessing.
    assert resolve_init_stem("whatever", str(d), {"whatever": "custom_stem"}) == "custom_stem"
    # Ambiguous suffix -> exit.
    (d / "LIVING_ROOM_SCENE1_turn_on_the_stove_and_put_the_moka_pot_on_it.init").write_bytes(b"x")
    with pytest.raises(SystemExit, match="exactly one"):
        resolve_init_stem("turn on the stove and put the moka pot on it", str(d))
    with pytest.raises(SystemExit, match="exactly one"):
        resolve_init_stem("unknown task", str(d))


def test_check_init_dir_rejects_pruned(tmp_path):
    (tmp_path / "a.init").write_bytes(b"x")
    check_init_dir(str(tmp_path))  # clean -> passes
    (tmp_path / "a.pruned_init").write_bytes(b"x")
    with pytest.raises(SystemExit, match="pruned_init"):
        check_init_dir(str(tmp_path))


def test_resolve_pretrained_dir_lerobot_layout(tmp_path):
    from exp.ablation_study.select_student_checkpoint import resolve_pretrained_dir

    out = tmp_path / "run"
    (out / "checkpoints" / "000100" / "pretrained_model").mkdir(parents=True)
    (out / "checkpoints" / "000200" / "pretrained_model").mkdir(parents=True)
    # Without a `last` link the highest step wins.
    assert resolve_pretrained_dir(out).parent.name == "000200"
    last = out / "checkpoints" / "last"
    last.symlink_to(out / "checkpoints" / "000100")
    assert resolve_pretrained_dir(out) == last / "pretrained_model"
    with pytest.raises(SystemExit, match="no LeRobot checkpoints"):
        resolve_pretrained_dir(tmp_path / "empty")
