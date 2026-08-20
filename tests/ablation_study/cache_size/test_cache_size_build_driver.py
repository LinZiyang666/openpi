"""The build driver's two guards (plan §12 P4).

Both exist because the failure they catch is **silent**: a library that is
smaller than the tier it claims still loads, still evaluates, and still produces
a plausible success rate. Size is the independent variable here, so a quietly
shrunken library does not look like a bug -- it looks like a data point.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import pytest

from exp.ablation_study.cache_size.build_size_artifacts import (
    TIERS,
    build_one,
    verify_list_coverage,
    verify_nesting,
)


@dataclass
class _Entry:
    trajectory_id: str
    id: str


def _write_artifact(path, trajectory_ids, steps=2):
    entries = [
        _Entry(trajectory_id=t, id=f"{t}:{i}")
        for t in trajectory_ids for i in range(steps)
    ]
    path.write_bytes(pickle.dumps({
        "entries": entries,
        "vector_dims": {"vision_0": 32768},
        "key_builder_type": "cp1_spatial_pool_16",
        "checkpoint_id": "CP1",
    }))
    return path


def _write_list(path, rel_paths):
    path.write_text("\n".join(rel_paths) + "\n")
    return path


def test_coverage_passes_when_every_listed_episode_is_present(tmp_path):
    rels = [f"task_{t}/episode_{e}.h5" for t in range(2) for e in range(3)]
    _write_list(tmp_path / "l.txt", rels)
    _write_artifact(tmp_path / "a.pkl", [r.removesuffix(".h5") for r in rels])
    assert verify_list_coverage(str(tmp_path / "a.pkl"), str(tmp_path / "l.txt")) == (6, 6)


def test_coverage_catches_an_outcome_filter_mismatch(tmp_path):
    """The exact accident this guard was added for.

    The grid lists all 45 B-train inits (``--outcome-filter all``); the builder
    is left on its ``success`` default and drops the failed ones. Nothing raises
    -- the artifact is simply a smaller library wearing the S6 label.
    """
    listed = [f"task_0/episode_{e}.h5" for e in range(10)]
    _write_list(tmp_path / "l.txt", listed)
    # the builder kept only the 6 that succeeded
    _write_artifact(tmp_path / "a.pkl", [f"task_0/episode_{e}" for e in range(6)])
    n_listed, n_present = verify_list_coverage(str(tmp_path / "a.pkl"), str(tmp_path / "l.txt"))
    assert (n_listed, n_present) == (10, 6)
    assert n_listed != n_present, "the driver must treat this as fatal"


def test_coverage_ignores_blank_lines_and_the_h5_suffix(tmp_path):
    (tmp_path / "l.txt").write_text("task_0/episode_0.h5\n\n  task_0/episode_1.h5  \n\n")
    _write_artifact(tmp_path / "a.pkl", ["task_0/episode_0", "task_0/episode_1"])
    assert verify_list_coverage(str(tmp_path / "a.pkl"), str(tmp_path / "l.txt")) == (2, 2)


def test_coverage_does_not_credit_extra_artifact_entries(tmp_path):
    """A superset artifact must not mask a missing listed episode.

    Intersecting rather than comparing counts is what makes this hold: an
    artifact carrying 5 trajectories, only 1 of which was listed, is still a
    failure for a 2-episode list.
    """
    _write_list(tmp_path / "l.txt", ["task_0/episode_0.h5", "task_0/episode_9.h5"])
    _write_artifact(tmp_path / "a.pkl",
                    [f"task_0/episode_{e}" for e in (0, 1, 2, 3, 4)])
    assert verify_list_coverage(str(tmp_path / "a.pkl"), str(tmp_path / "l.txt")) == (2, 1)


def test_build_one_forwards_the_outcome_filter_and_relpath_mode(monkeypatch):
    """``--outcome-filter`` and ``--trajectory-id-mode relpath`` must both reach
    the builder. The latter is what keeps ten tasks' ``episode_0`` from
    collapsing onto one id; the former is what this ruling turns on."""
    seen = {}

    def fake_run(cmd, check):
        seen["cmd"] = cmd
        return None

    monkeypatch.setattr(
        "exp.ablation_study.cache_size.build_size_artifacts.subprocess.run", fake_run)
    build_one(data_dir="/d/libero_10", episode_list="/l.txt", output="/o.pkl",
              builder_type="cp1_spatial_pool_16", workers=2, outcome_filter="all")
    cmd = seen["cmd"]
    assert cmd[cmd.index("--outcome-filter") + 1] == "all"
    assert cmd[cmd.index("--trajectory-id-mode") + 1] == "relpath"
    assert cmd[cmd.index("--builder-type") + 1] == "cp1_spatial_pool_16"
    assert cmd[cmd.index("--data-dir") + 1] == "/d/libero_10", (
        "data-dir must include the suite level, or relpath ids gain a prefix"
    )


@pytest.mark.parametrize("bad", ["success", "failure"])
def test_build_one_does_not_silently_default_the_filter(monkeypatch, bad):
    """It is a required argument at the call site, not an optional nicety."""
    monkeypatch.setattr(
        "exp.ablation_study.cache_size.build_size_artifacts.subprocess.run",
        lambda cmd, check: None)
    with pytest.raises(TypeError):
        build_one(data_dir="/d", episode_list="/l", output="/o",
                  builder_type="cp1_spatial_pool_16", workers=1)


# ---------------------------------------------------------------------------
# Nesting -- what makes the size axis a within-library comparison
# ---------------------------------------------------------------------------


def _tier_set(tmp_path, prefix, per_tier):
    for tier, trajs in zip(TIERS, per_tier):
        _write_artifact(tmp_path / f"{prefix}_{tier}.pkl", trajs)


def test_nesting_accepts_a_growing_family(tmp_path):
    growing = [[f"task_0/episode_{i}" for i in range(n)] for n in (1, 2, 5, 10, 20, 45)]
    _tier_set(tmp_path, "cs", growing)
    assert verify_nesting(str(tmp_path), "cs") == []


def test_nesting_allows_a_topped_out_tier_to_equal_its_successor(tmp_path):
    """R1 caps a task at what it has; S5 == S6 is the rule working, not a defect."""
    capped = [[f"task_0/episode_{i}" for i in range(n)] for n in (1, 2, 5, 10, 20, 20)]
    _tier_set(tmp_path, "cs", capped)
    assert verify_nesting(str(tmp_path), "cs") == []


def test_nesting_catches_a_resampled_tier(tmp_path):
    """The failure this gate exists for: S4 is a fresh draw rather than S3 + more.

    Counts still rise, every library still loads, and the evaluation still
    produces a plausible curve -- but each adjacent-tier comparison would then
    mix "more data" with "different data".
    """
    resampled = [
        ["task_0/episode_0"],
        ["task_0/episode_0", "task_0/episode_1"],
        [f"task_0/episode_{i}" for i in range(5)],
        [f"task_0/episode_{i}" for i in range(100, 110)],   # disjoint redraw
        [f"task_0/episode_{i}" for i in range(100, 120)],
        [f"task_0/episode_{i}" for i in range(100, 145)],
    ]
    _tier_set(tmp_path, "cs", resampled)
    v = verify_nesting(str(tmp_path), "cs")
    assert len(v) == 1 and v[0].startswith("S3 is not a subset of S4")
    assert "5 entries" in v[0] or "10 entries" in v[0]


def test_nesting_reports_every_broken_link_not_just_the_first(tmp_path):
    broken = [
        ["task_0/episode_0"],
        ["task_0/episode_9"],                                # S1 orphaned
        ["task_0/episode_9", "task_0/episode_1"],
        ["task_0/episode_9", "task_0/episode_1", "task_0/episode_2"],
        ["task_0/episode_7"],                                # S4 orphaned
        ["task_0/episode_7", "task_0/episode_8"],
    ]
    _tier_set(tmp_path, "cs", broken)
    v = verify_nesting(str(tmp_path), "cs")
    assert len(v) == 2, f"expected both broken links, got {v}"

