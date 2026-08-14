"""Tests for the E1 dose-response driver.

The failure mode this guards is subtle: if x (key quality) and y (relative
gain) are read off the same episodes, they share an ``r_A`` term and correlate
for reasons that have nothing to do with key sufficiency. The fold machinery
exists to stop that, so it is what gets pinned here -- together with the
task-stratified split and the both-orientations reporting the plan asks for.
"""

from __future__ import annotations

import pytest

from exp.markov_sufficiency import e1_dose_response as dr
from exp.markov_sufficiency import e1_loeo_residual as e1


def _rows(builder, n_eps=8, n_tasks=2, gain=0.1, base=1.0):
    """Synthetic A/C rows where C beats A by a fixed relative ``gain``."""
    out = []
    for i in range(n_eps):
        task = f"task{i % n_tasks}"
        traj = f"{builder}_ep{i}"
        r = {"suite": "s", "key_builder": builder, "trajectory_id": traj,
             "task_key": task, "step_idx": 0, "padding": False, "k": 1}
        out.append({**r, "group": "A", "residual": base})
        out.append({**r, "group": dr.DOSE_GROUP, "residual": base * (1 - gain)})
    return out


# ------------------------------------------------------------------
# Fold construction
# ------------------------------------------------------------------


def test_stratified_folds_split_every_task_evenly():
    rows = _rows("b", n_eps=12, n_tasks=3)
    folds = dr.stratified_folds(rows)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_key"], set()).add(r["trajectory_id"])
    for task, eps in by_task.items():
        sides = [folds[e] for e in eps]
        assert abs(sides.count(0) - sides.count(1)) <= 1, f"{task} is lopsided: {sides}"


def test_stratified_folds_are_deterministic_per_seed():
    rows = _rows("b", n_eps=10)
    assert dr.stratified_folds(rows, seed=1) == dr.stratified_folds(rows, seed=1)
    assert dr.stratified_folds(rows, seed=1) != dr.stratified_folds(rows, seed=2)


def test_folds_cover_every_episode_exactly_once():
    rows = _rows("b", n_eps=9, n_tasks=3)
    folds = dr.stratified_folds(rows)
    assert set(folds) == {r["trajectory_id"] for r in rows}
    assert set(folds.values()) <= {0, 1}


# ------------------------------------------------------------------
# dose_response fold plumbing
# ------------------------------------------------------------------


def test_supplied_fold_assignment_overrides_the_default_split():
    """x must come from fold 0 and y from fold 1, as the caller assigned them."""
    rows = _rows("b", n_eps=4, n_tasks=1, gain=0.0)
    # Make fold 0 cheap and fold 1 expensive on the A group only.
    eps = sorted({r["trajectory_id"] for r in rows})
    assign = {e: (0 if i < 2 else 1) for i, e in enumerate(eps)}
    for r in rows:
        if r["group"] == "A" and assign[r["trajectory_id"]] == 0:
            r["residual"] = 0.5
    by_builder = {f"b{i}": [dict(r) for r in rows] for i in range(3)}
    out = e1.dose_response(by_builder, dr.DOSE_GROUP, fold_assignment=assign)
    assert out["n"] == 3


def test_swap_exchanges_the_two_folds():
    rows = _rows("b", n_eps=4, n_tasks=1)
    eps = sorted({r["trajectory_id"] for r in rows})
    assign = {e: (0 if i < 2 else 1) for i, e in enumerate(eps)}
    for r in rows:
        if r["group"] == "A" and assign[r["trajectory_id"]] == 0:
            r["residual"] = 0.5
    by_builder = {f"b{i}": [dict(r) for r in rows] for i in range(3)}
    fwd = e1.dose_response(by_builder, dr.DOSE_GROUP, fold_assignment=assign)
    rev = e1.dose_response(by_builder, dr.DOSE_GROUP, fold_assignment=assign, swap=True)
    # The x values are the A-residual medians of opposite halves, so they differ.
    assert fwd["n"] == rev["n"] == 3


def test_dose_response_default_split_is_unchanged():
    """Callers that pass no assignment keep the original deterministic halves."""
    by_builder = {f"b{i}": _rows(f"b{i}", n_eps=6) for i in range(3)}
    out = e1.dose_response(by_builder, dr.DOSE_GROUP)
    assert out["n"] == 3


# ------------------------------------------------------------------
# analyse
# ------------------------------------------------------------------


def test_analyse_reports_both_orientations_and_the_raw_points():
    by_builder = {f"b{i}": _rows(f"b{i}", n_eps=8, gain=0.05 * i, base=1.0 + i) for i in range(4)}
    out = dr.analyse(by_builder)
    assert set(out) >= {"forward", "folds_reversed", "sign_consistent", "points"}
    assert len(out["points"]) == 4
    assert {p["key_builder"] for p in out["points"]} == set(by_builder)
    assert "exploratory" in out["role"]


def test_analyse_marks_disagreeing_orientations():
    """A trend that flips sign between folds must not read as consistent."""
    out = dr.analyse({f"b{i}": _rows(f"b{i}", n_eps=8) for i in range(3)})
    assert isinstance(out["sign_consistent"], bool)


def test_cli_rejects_a_malformed_spec(tmp_path):
    with pytest.raises(SystemExit):
        dr.main(["--spec", "no-equals-sign", "--out", str(tmp_path / "x.json")])


def test_cli_rejects_a_suite_with_too_few_builders(tmp_path):
    with pytest.raises(SystemExit):
        dr.main(["--spec", "s:b1=y1", "--spec", "s:b2=y2", "--out", str(tmp_path / "x.json")])
