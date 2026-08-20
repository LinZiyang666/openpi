"""Grid construction rules R1-R5 for the cache-size ablation (plan §3.2)."""

from __future__ import annotations

import pytest

from exp.ablation_study.cache_size.emit_size_grid import (
    DEFAULT_TIERS,
    Episode,
    build_grid,
    order_task,
)

TIERS = DEFAULT_TIERS


def _episodes(task_id: int, n: int, *, fail: set[int] | None = None) -> list[Episode]:
    fail = fail or set()
    return [
        Episode(
            task_id=task_id,
            init_idx=i,
            rel_path=f"task_{task_id}/episode_{i}.h5",
            success=i not in fail,
        )
        for i in range(n)
    ]


def _grid(n_tasks=3, n_per_task=50, fail=None, anchors=(0, 1, 2, 3, 4), val=(45, 46, 47, 48, 49)):
    eps: list[Episode] = []
    for t in range(n_tasks):
        eps += _episodes(t, n_per_task, fail=(fail or {}).get(t, set()))
    return build_grid(
        eps,
        anchor_inits_by_task={t: set(anchors) for t in range(n_tasks)},
        val_inits_by_task={t: set(val) for t in range(n_tasks)},
        tiers=TIERS,
    )


def test_r1_per_task_count_is_min_k_n_for_every_tier():
    # task 1 only has 12 successes; every tier must clamp, not just the largest.
    fail = {1: set(range(12, 45))}
    g = _grid(fail=fail)
    counts = g["realized_counts"]["task_1"]
    assert [counts[f"S{i+1}"] for i in range(len(TIERS))] == [1, 2, 5, 10, 12, 12]
    # An unaffected task is unclamped until the B-train budget itself binds.
    counts0 = g["realized_counts"]["task_0"]
    assert [counts0[f"S{i+1}"] for i in range(len(TIERS))] == [1, 2, 5, 10, 20, 45]


def test_r2_zero_success_task_is_fatal():
    fail = {2: set(range(50))}
    with pytest.raises(ValueError, match="zero successful B-train trajectories"):
        _grid(fail=fail)


def test_r3_identical_adjacent_tiers_are_reported_not_dropped():
    # Every task capped at 5 -> S3..S6 all select the same 5 trajectories.
    fail = {t: set(range(5, 45)) for t in range(3)}
    g = _grid(fail=fail)
    assert g["degenerate_pairs"] == ["S3-S4", "S4-S5", "S5-S6"]
    # Tiers themselves survive: the grid still carries all six.
    assert len(g["episodes"]) == len(TIERS)


def test_r4_mean_realized_is_reported_for_x_axis():
    fail = {1: set(range(12, 45))}
    g = _grid(fail=fail)
    # S6: tasks 0 and 2 reach 45, task 1 stops at 12.
    assert g["mean_realized"]["S6"] == pytest.approx((45 + 12 + 45) / 3)


def test_r5_failed_anchor_is_topped_up_and_hit_count_recorded():
    # Two of the five historical anchors failed for task 0.
    g = _grid(fail={0: {1, 3}})
    assert g["anchor_hits"]["task_0"] == {"hit": 3, "of": 5}
    # S3 still holds five trajectories -- topped up from priority 2.
    assert g["realized_counts"]["task_0"]["S3"] == 5
    s3 = g["episodes"]["S3"]["task_0"]
    assert len(s3) == 5
    # The three surviving anchors lead, in orig_init_state_idx order.
    assert s3[:3] == [
        "task_0/episode_0.h5",
        "task_0/episode_2.h5",
        "task_0/episode_4.h5",
    ]


def test_b_val_never_enters_any_tier():
    val = {45, 46, 47, 48, 49}
    g = _grid(val=tuple(val))
    for tier, per_task in g["episodes"].items():
        for task, paths in per_task.items():
            idxs = {int(p.split("episode_")[1].split(".")[0]) for p in paths}
            assert not (idxs & val), f"{tier}/{task} leaked B-val: {idxs & val}"


def test_tiers_are_nested():
    g = _grid()
    names = [f"S{i+1}" for i in range(len(TIERS))]
    for prev, cur in zip(names, names[1:]):
        for task in g["episodes"][prev]:
            assert set(g["episodes"][prev][task]) <= set(g["episodes"][cur][task])


def test_every_tier_every_task_is_non_empty():
    """The premise behind 'hit rate == 1': each task needs >=1 entry in each tier."""
    g = _grid(fail={1: set(range(12, 45))})
    for tier, per_task in g["episodes"].items():
        for task, paths in per_task.items():
            assert paths, f"{tier}/{task} is empty"


def test_ordering_is_deterministic():
    a = _grid()
    b = _grid()
    assert a["episodes"] == b["episodes"]


def test_anchor_order_follows_subset_init_idx():
    """Anchors are ordered by subset init index -- the split file's own space."""
    eps = [
        Episode(task_id=0, init_idx=i, rel_path=f"task_0/episode_{i}.h5", success=True)
        for i in (3, 0, 4, 1, 2)
    ]
    grid = order_task(eps, anchor_inits={0, 1, 2, 3, 4}, val_inits=set())
    assert [e.init_idx for e in grid.ordered] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Collection identity: the roster must be validated, not inferred
# ---------------------------------------------------------------------------


def test_whole_task_missing_is_fatal():
    """A task directory that never appeared is the strongest form of zero coverage."""
    eps = _episodes(0, 50) + _episodes(2, 50)  # task 1 absent entirely
    with pytest.raises(ValueError, match="missing task"):
        build_grid(
            eps,
            anchor_inits_by_task={t: {0, 1, 2, 3, 4} for t in range(3)},
            val_inits_by_task={t: {45, 46, 47, 48, 49} for t in range(3)},
            tiers=TIERS,
        )


def test_unexpected_task_is_fatal():
    eps = _episodes(0, 50) + _episodes(1, 50) + _episodes(7, 50)
    with pytest.raises(ValueError, match="unexpected task"):
        build_grid(
            eps,
            anchor_inits_by_task={t: {0} for t in range(2)},
            val_inits_by_task={t: {49} for t in range(2)},
            tiers=TIERS,
        )


def test_duplicate_init_within_a_task_is_fatal():
    eps = _episodes(0, 50) + _episodes(1, 50)
    eps.append(Episode(task_id=0, init_idx=3, rel_path="task_0/episode_3_dup.h5",
                       success=True))
    with pytest.raises(ValueError, match="collected more than once"):
        build_grid(
            eps,
            anchor_inits_by_task={t: {0} for t in range(2)},
            val_inits_by_task={t: {49} for t in range(2)},
            tiers=TIERS,
        )


def test_short_collection_ledger_is_fatal():
    eps = _episodes(0, 40) + _episodes(1, 50)
    with pytest.raises(ValueError, match="expected 50"):
        build_grid(
            eps,
            anchor_inits_by_task={t: {0} for t in range(2)},
            val_inits_by_task={t: {49} for t in range(2)},
            tiers=TIERS,
            expected_inits_per_task=50,
        )


# ---------------------------------------------------------------------------
# scan_collected against real writer-shaped h5 (not hand-built Episode objects)
# ---------------------------------------------------------------------------


def _write_collect_episode(path, *, success, n_steps=2):
    """Mirror EpisodeDataCollector's output: 6 attrs, step_* groups with keys."""
    import h5py
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["experiment_name"] = "libero_spatial"
        f.attrs["task"] = "pick up the bowl"
        f.attrs["episode_id"] = 0
        f.attrs["num_steps"] = n_steps
        f.attrs["timestamp"] = "2026-08-16T00:00:00"
        f.attrs["success"] = success
        # NOTE: no orig_init_state_idx -- the collector drops extra_metadata.
        for i in range(n_steps):
            g = f.create_group(f"step_{i:04d}")
            g.create_dataset("vision_0", data=np.zeros((4, 8), dtype=np.float16))
            g.create_dataset("vision_1", data=np.zeros((4, 8), dtype=np.float16))
            g.create_dataset("prompt_emb", data=np.zeros((4, 8), dtype=np.float16))
            g.create_dataset("robot_state", data=np.zeros(32, dtype=np.float32))
            g.create_dataset("clean_action", data=np.zeros((10, 32), dtype=np.float32))


def test_scan_collected_reads_real_writer_shape(tmp_path):
    from exp.ablation_study.cache_size.emit_size_grid import scan_collected

    _write_collect_episode(tmp_path / "task_0" / "episode_3.h5", success=True)
    _write_collect_episode(tmp_path / "task_0" / "episode_1.h5", success=False)

    eps = sorted(scan_collected(tmp_path), key=lambda e: e.init_idx)
    assert [(e.task_id, e.init_idx, e.success) for e in eps] == [(0, 1, False), (0, 3, True)]
    assert eps[0].rel_path == "task_0/episode_1.h5"


def test_scan_collected_rejects_the_save_trajectory_dump(tmp_path):
    """Same layout, observations only -- must not be mistaken for the collect corpus."""
    import h5py
    import numpy as np

    from exp.ablation_study.cache_size.emit_size_grid import scan_collected

    d = tmp_path / "task_0"
    d.mkdir(parents=True)
    with h5py.File(d / "episode_0.h5", "w") as f:
        f.attrs["success"] = True
        f.attrs["orig_init_state_idx"] = 17  # present here, absent in --collect
        g = f.create_group("step_0000")
        g.create_dataset("agentview_image", data=np.zeros((4, 4, 3), dtype=np.uint8))
        g.create_dataset("robot_state", data=np.zeros(8, dtype=np.float64))
        g.create_dataset("sim_state", data=np.zeros(92, dtype=np.float64))

    with pytest.raises(ValueError, match="--save_trajectory dump"):
        scan_collected(tmp_path)


# ---------------------------------------------------------------------------
# outcome_filter="all" -- owner ruling 2026-08-17
# ---------------------------------------------------------------------------


def test_all_filter_keeps_failures_and_never_tops_out():
    """Under "all" the axis is collected trajectories, so no tier can top out.

    This is the whole point of the ruling: eligibility stops depending on the
    outcome, so a hard task contributes the same 45 B-train inits as an easy
    one and the R1/R3/R4 topping-out machinery -- which stays wired up -- never
    binds.
    """
    # A brutal task: only 3 of its 45 B-train inits succeeded.
    eps = _episodes(0, 50, fail=set(range(3, 45)))
    grid = order_task(eps, anchor_inits={0, 1, 2, 3, 4}, val_inits={45, 46, 47, 48, 49},
                      outcome_filter="all")
    assert grid.n_success == 45, "all 45 B-train inits are eligible regardless of outcome"
    assert sum(1 for e in grid.ordered if not e.success) == 42

    success_only = order_task(eps, anchor_inits={0, 1, 2, 3, 4},
                              val_inits={45, 46, 47, 48, 49})
    assert success_only.n_success == 3, "the frozen default still counts successes only"


def test_all_filter_still_excludes_b_val():
    """B-val stays out under either filter -- it is the one slice outside both
    the student's training set and every library, and the whole TIER line rests
    on it."""
    eps = _episodes(0, 50, fail={45, 46})
    grid = order_task(eps, anchor_inits={0}, val_inits={45, 46, 47, 48, 49},
                      outcome_filter="all")
    assert grid.n_success == 45
    assert all(e.init_idx < 45 for e in grid.ordered)


def test_all_filter_keeps_anchors_first():
    """R5's ordering is unchanged; under "all" every anchor is present, because
    an anchor is now eligible whether or not its rollout succeeded."""
    eps = _episodes(0, 50, fail={0, 2})  # two of the five anchors failed
    grid = order_task(eps, anchor_inits={0, 1, 2, 3, 4}, val_inits=set(),
                      outcome_filter="all")
    assert [e.init_idx for e in grid.ordered[:5]] == [0, 1, 2, 3, 4]
    assert grid.anchor_hits == 5

    success_only = order_task(eps, anchor_inits={0, 1, 2, 3, 4}, val_inits=set())
    assert success_only.anchor_hits == 3, "the frozen default drops the failed anchors"


def test_grid_records_the_filter_it_used():
    """The emitted grid must say which filter produced it.

    The builder has its own ``--outcome-filter``; if the two disagree the
    builder silently drops listed episodes and the library ends up smaller than
    the tier it claims -- with no error anywhere. Recording it is what makes
    that checkable.
    """
    eps = []
    for t in range(2):
        eps.extend(_episodes(t, 50, fail={10, 11, 12}))
    anchors = {t: {0, 1, 2, 3, 4} for t in range(2)}
    vals = {t: {45, 46, 47, 48, 49} for t in range(2)}

    g_all = build_grid(eps, anchors, vals, expected_inits_per_task=50, outcome_filter="all")
    assert g_all["outcome_filter"] == "all"
    assert set(g_all["n_success"].values()) == {45}
    assert g_all["mean_realized"]["S6"] == 45, "no capping under 'all'"

    g_succ = build_grid(eps, anchors, vals, expected_inits_per_task=50)
    assert g_succ["outcome_filter"] == "success"
    assert set(g_succ["n_success"].values()) == {42}
    assert g_succ["mean_realized"]["S6"] == 42, "capped by the successes that exist"


def test_unknown_outcome_filter_is_rejected():
    with pytest.raises(ValueError, match="outcome_filter"):
        order_task(_episodes(0, 50), anchor_inits=set(), val_inits=set(),
                   outcome_filter="failure")

