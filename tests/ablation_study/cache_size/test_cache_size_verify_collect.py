"""P3 collection gate (plan §12 P3).

The fixtures here write **real h5 files in the collector's own layout** rather
than mock objects. That is deliberate and this suite has been bitten twice by
the alternative: a fixture that invents a shape production never produces can
make a broken gate look green.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from exp.ablation_study.cache_size.verify_collect import (
    EPISODES_PER_TASK,
    NUM_TASKS,
    assert_grid_complete,
    assert_results_agree,
    check_one,
    load_results,
    scan_paths,
)


def write_episode(path, *, steps=3, success=True, drop=None, bad_shape=None,
                  declared_steps=None):
    """One episode in the collector's layout and schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["num_steps"] = steps if declared_steps is None else declared_steps
        h.attrs["success"] = success
        h.attrs["task"] = "pick up the black bowl"
        h.attrs["episode_id"] = 0
        h.attrs["experiment_name"] = "libero_spatial"
        for s in range(steps):
            g = h.create_group(f"step_{s:04d}")
            fields = {
                "vision_0": (np.zeros((256, 2048), np.float16)),
                "vision_1": (np.zeros((256, 2048), np.float16)),
                "vision_2": (np.zeros((256, 2048), np.float16)),
                "prompt_emb": (np.zeros((200, 2048), np.float16)),
                "robot_state": (np.zeros((32,), np.float32)),
                "clean_action": (np.zeros((10, 32), np.float32)),
            }
            if bad_shape and s == steps - 1:
                fields[bad_shape] = np.zeros((7, 7), np.float16)
            for name, arr in fields.items():
                if drop and name == drop and s == steps - 1:
                    continue
                g.create_dataset(name, data=arr)
    return path


def build_tree(root, *, tasks=NUM_TASKS, eps=EPISODES_PER_TASK, successes=None):
    """A whole suite tree. ``successes[t]`` = how many of task t's episodes pass."""
    for t in range(tasks):
        n_ok = eps if successes is None else successes[t]
        for e in range(eps):
            write_episode(root / f"task_{t}" / f"episode_{e}.h5", steps=2,
                          success=e < n_ok)
    return root


# ---------------------------------------------------------------------------
# Grid completeness
# ---------------------------------------------------------------------------


def test_complete_grid_passes(tmp_path):
    build_tree(tmp_path, tasks=2, eps=3)
    found = scan_paths(tmp_path)
    assert len(found) == 6
    assert found[(1, 2)].name == "episode_2.h5"


def test_missing_episode_is_fatal(tmp_path):
    build_tree(tmp_path)
    (tmp_path / "task_7" / "episode_31.h5").unlink()
    with pytest.raises(SystemExit, match="1 missing"):
        assert_grid_complete(scan_paths(tmp_path))


def test_extra_episode_is_fatal(tmp_path):
    """An out-of-range episode means the run used a different init pool."""
    build_tree(tmp_path)
    write_episode(tmp_path / "task_3" / "episode_50.h5")
    with pytest.raises(SystemExit, match="unexpected"):
        assert_grid_complete(scan_paths(tmp_path))


def test_unparseable_path_is_fatal(tmp_path):
    write_episode(tmp_path / "task_x" / "episode_0.h5")
    with pytest.raises(SystemExit, match="unparseable"):
        scan_paths(tmp_path)


# ---------------------------------------------------------------------------
# Schema -- the gate that separates this collection from the Phase 0 one
# ---------------------------------------------------------------------------


def test_missing_embedding_is_fatal(tmp_path):
    """The Phase 0 failure mode: well-formed h5, zero embeddings."""
    p = write_episode(tmp_path / "task_0" / "episode_0.h5", steps=3, drop="vision_1")
    with pytest.raises(SystemExit, match="lacks 'vision_1'"):
        check_one(p, deep=True)


def test_missing_embedding_in_a_middle_step_needs_the_deep_check(tmp_path):
    """First+last sampling cannot see a hole in the middle -- state that, don't hide it."""
    p = tmp_path / "task_0" / "episode_0.h5"
    p.parent.mkdir(parents=True)
    with h5py.File(p, "w") as h:
        h.attrs.update({"num_steps": 3, "success": True, "task": "t"})
        for s in range(3):
            g = h.create_group(f"step_{s:04d}")
            for name, arr in (
                ("vision_0", np.zeros((256, 2048), np.float16)),
                ("vision_1", np.zeros((256, 2048), np.float16)),
                ("vision_2", np.zeros((256, 2048), np.float16)),
                ("prompt_emb", np.zeros((200, 2048), np.float16)),
                ("robot_state", np.zeros((32,), np.float32)),
                ("clean_action", np.zeros((10, 32), np.float32)),
            ):
                if s == 1 and name == "vision_2":
                    continue
                g.create_dataset(name, data=arr)
    check_one(p, deep=False)  # shallow pass: the hole is in step 1
    with pytest.raises(SystemExit, match="lacks 'vision_2'"):
        check_one(p, deep=True)


def test_wrong_shape_is_fatal(tmp_path):
    p = write_episode(tmp_path / "task_0" / "episode_0.h5", steps=2,
                      bad_shape="prompt_emb")
    with pytest.raises(SystemExit, match="shape"):
        check_one(p, deep=True)


def test_step_hole_probe(tmp_path):
    """attrs.num_steps > actual groups is what a live cache would leave behind."""
    p = write_episode(tmp_path / "task_0" / "episode_0.h5", steps=4, declared_steps=6)
    with pytest.raises(SystemExit, match="num_steps=6 but 4 step groups"):
        check_one(p, deep=False)


def test_clean_episode_reports_its_outcome(tmp_path):
    p = write_episode(tmp_path / "task_0" / "episode_0.h5", steps=5, success=False)
    info = check_one(p, deep=True)
    assert info == {"num_steps": 5, "success": False, "task": "pick up the black bowl"}


# ---------------------------------------------------------------------------
# Results-JSON join
# ---------------------------------------------------------------------------


def test_results_join_accepts_agreement(tmp_path):
    res = tmp_path / "r.json"
    res.write_text(json.dumps([
        {"task_id": 0, "init_state_idx": 0, "orig_init_state_idx": 0,
         "episode_id": 0, "seed": 7, "success": True},
        {"task_id": 0, "init_state_idx": 1, "orig_init_state_idx": 1,
         "episode_id": 1, "seed": 7, "success": False},
    ]))
    loaded = load_results(res)
    assert_results_agree(loaded, {(0, 0): True, (0, 1): False})


def test_results_join_rejects_a_success_disagreement(tmp_path):
    """The two writers must agree: the size grid selects on *successful* rollouts,
    so one flipped verdict puts a failed trajectory into the library."""
    res = tmp_path / "r.json"
    res.write_text(json.dumps([
        {"task_id": 0, "init_state_idx": 0, "success": True},
    ]))
    with pytest.raises(SystemExit, match="disagree on success"):
        assert_results_agree(load_results(res), {(0, 0): False})


def test_results_join_rejects_a_grid_mismatch(tmp_path):
    res = tmp_path / "r.json"
    res.write_text(json.dumps([{"task_id": 0, "init_state_idx": 0, "success": True}]))
    with pytest.raises(SystemExit, match="different grids"):
        assert_results_agree(load_results(res), {(0, 0): True, (0, 1): True})


def test_results_duplicate_row_is_fatal(tmp_path):
    res = tmp_path / "r.json"
    res.write_text(json.dumps([
        {"task_id": 2, "init_state_idx": 5, "success": True},
        {"task_id": 2, "init_state_idx": 5, "success": False},
    ]))
    with pytest.raises(SystemExit, match="duplicate entry"):
        load_results(res)
