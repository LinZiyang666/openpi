"""Tests for the admission-gate analysis.

The intersection rule here decides which tasks enter the formal cross-scene
experiment, so a silent bug would corrupt every downstream number without ever
raising. These tests pin the statistics against values recorded in
`logs/benchmark_and_teacher_selection.log.md` §12-2 and pin the two failure
modes that would be invisible in the output: an errored arm read as zero
successes, and a task quietly vanishing from the table.

Pure stdlib, no GPU, no islands -- runs in the main environment.
"""

from __future__ import annotations

import json
import math

import pytest

from exp.robocasa365 import analyze_admission_gate as aag


def make_run(tasks: dict[str, dict], *, n_trials: int = 5,
             scene_a=(1, 1), scene_b=(7, 7)) -> dict:
    return {
        "sceneA": list(scene_a),
        "sceneB": list(scene_b),
        "n_trials": n_trials,
        "tasks": tasks,
    }


def arm(succ: int, n: int = 5, wall: float = 1.0) -> dict:
    return {"succ": succ, "n": n, "sr": succ / n, "wall_s": wall}


def write(tmp_path, name: str, payload: dict):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


class TestWilson:
    def test_matches_the_recorded_pi05_interval(self):
        """SR_B = 45/90 is the number the admission verdict turned on."""
        lo, hi = aag.wilson(45, 90)
        assert lo == pytest.approx(0.399, abs=5e-4)
        assert hi == pytest.approx(0.601, abs=5e-4)

    def test_lower_bound_is_below_the_point_estimate(self):
        lo, hi = aag.wilson(37, 90)
        assert lo < 37 / 90 < hi

    def test_zero_successes_still_gives_a_finite_upper_bound(self):
        """A 0/90 arm is not "certainly zero" -- the interval must say so."""
        lo, hi = aag.wilson(0, 90)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert 0 < hi < 0.1

    def test_empty_sample_is_nan_not_zero(self):
        lo, hi = aag.wilson(0, 0)
        assert math.isnan(lo) and math.isnan(hi)


class TestClassify:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (0, 0, "U0"),
            (0, 3, "U1"),
            (3, 0, "U1"),
            (1, 1, "U2"),
            (5, 5, "U2"),
        ],
    )
    def test_truth_table(self, a, b, expected):
        assert aag.classify(a, b) == expected


class TestLoadRun:
    def test_every_task_gets_a_row_including_u0(self, tmp_path):
        """P3: the table reports all tasks, not just the surviving ones."""
        path = write(tmp_path, "r.json", make_run({
            "Good": {"A": arm(3), "B": arm(2)},
            "Dead": {"A": arm(0), "B": arm(0)},
        }))
        rows = aag.load_run(path)["rows"]
        assert {r["task"] for r in rows} == {"Good", "Dead"}
        assert {r["task"]: r["cls"] for r in rows} == {"Good": "U2", "Dead": "U0"}

    def test_errored_arm_is_err_not_zero(self, tmp_path):
        """An infrastructure fault must not be recorded as teacher failure."""
        path = write(tmp_path, "r.json", make_run({
            "Broken": {"A": {"error": "socket closed"}, "B": arm(4)},
        }))
        row = aag.load_run(path)["rows"][0]
        assert row["cls"] == "ERR"
        assert row["a"] is None
        assert row["gap"] is None
        assert "socket closed" in row["error_a"]

    def test_errored_arm_is_excluded_from_pooled_statistics(self, tmp_path):
        """The broken task must not drag SR down as if it had scored 0/5."""
        path = write(tmp_path, "r.json", make_run({
            "Fine": {"A": arm(4), "B": arm(4)},
            "Broken": {"A": {"error": "boom"}, "B": arm(0)},
        }))
        stats = aag.pooled(aag.load_run(path)["rows"])
        assert stats["n"] == 5, "only the analysable task contributes episodes"
        assert stats["sr_a"] == pytest.approx(0.8)

    def test_scene_and_trials_are_carried_through(self, tmp_path):
        path = write(
            tmp_path, "r.json",
            make_run({"T": {"A": arm(1, 10), "B": arm(1, 10)}},
                     n_trials=10, scene_a=(2, 3), scene_b=(9, 9)),
        )
        run = aag.load_run(path)
        assert run["scene_a"] == (2, 3)
        assert run["scene_b"] == (9, 9)
        assert run["n_trials"] == 10


class TestUsableTasks:
    def test_keeps_u1_and_u2_drops_u0_and_err(self, tmp_path):
        """U1 is the most informative class about cross-scene fragility -- keep it."""
        path = write(tmp_path, "r.json", make_run({
            "Both": {"A": arm(3), "B": arm(3)},
            "OneSide": {"A": arm(0), "B": arm(3)},
            "Neither": {"A": arm(0), "B": arm(0)},
            "Broken": {"A": {"error": "x"}, "B": {"error": "x"}},
        }))
        assert aag.usable_tasks(aag.load_run(path)["rows"]) == {"Both", "OneSide"}


class TestIntersection:
    def build(self, tmp_path, left: dict, right: dict, **kw) -> dict:
        reports = {}
        for label, tasks in (("pi05", left), ("groot", right)):
            run = aag.load_run(write(tmp_path, f"{label}.json", make_run(tasks, **kw)))
            reports[label] = {"rows": run["rows"], "run": run}
        return aag.intersect_reports(reports)

    def test_keeps_only_tasks_usable_under_both(self, tmp_path):
        result = self.build(
            tmp_path,
            {"Shared": {"A": arm(3), "B": arm(3)},
             "Pi05Only": {"A": arm(3), "B": arm(3)},
             "NeitherOne": {"A": arm(0), "B": arm(0)}},
            {"Shared": {"A": arm(2), "B": arm(2)},
             "Pi05Only": {"A": arm(0), "B": arm(0)},
             "NeitherOne": {"A": arm(0), "B": arm(0)}},
        )
        assert result["keep"] == {"Shared"}
        assert result["dropped"] == ["NeitherOne", "Pi05Only"]

    def test_u1_under_one_teacher_still_qualifies(self, tmp_path):
        """U1 is explicitly retained by P2; the intersection must not drop it."""
        result = self.build(
            tmp_path,
            {"Fragile": {"A": arm(2), "B": arm(0)}},
            {"Fragile": {"A": arm(0), "B": arm(1)}},
        )
        assert result["keep"] == {"Fragile"}

    def test_errored_arm_drops_the_task_from_the_intersection(self, tmp_path):
        """Unknown is not usable -- but it is reported as ERR, not as U0."""
        result = self.build(
            tmp_path,
            {"T": {"A": arm(3), "B": arm(3)}},
            {"T": {"A": {"error": "socket"}, "B": arm(3)}},
        )
        assert result["keep"] == set()
        assert result["cls_of"]["groot"]["T"] == "ERR"

    def test_teacher_specific_competence_is_surfaced(self, tmp_path):
        result = self.build(
            tmp_path,
            {"Split": {"A": arm(3), "B": arm(3)},
             "Both": {"A": arm(1), "B": arm(1)}},
            {"Split": {"A": arm(0), "B": arm(0)},
             "Both": {"A": arm(1), "B": arm(1)}},
        )
        assert result["split"] == ["Split"]

    def test_task_missing_from_one_run_is_reported_not_silently_kept(self, tmp_path):
        """A partial run is an incomplete run, never a finding."""
        result = self.build(
            tmp_path,
            {"Common": {"A": arm(3), "B": arm(3)},
             "OnlyInPi05": {"A": arm(3), "B": arm(3)}},
            {"Common": {"A": arm(3), "B": arm(3)}},
        )
        assert result["missing"] == ["OnlyInPi05"]
        assert result["keep"] == {"Common"}

    def test_per_teacher_tally_shares_the_intersection_denominator(self, tmp_path):
        """Counting a whole run against the common-task total reads as nonsense."""
        result = self.build(
            tmp_path,
            {"Common": {"A": arm(3), "B": arm(3)},
             "OnlyInPi05": {"A": arm(3), "B": arm(3)}},
            {"Common": {"A": arm(3), "B": arm(3)}},
        )
        assert result["usable"]["pi05"] == {"Common", "OnlyInPi05"}
        assert result["usable_common"]["pi05"] == {"Common"}
        assert len(result["usable_common"]["pi05"]) <= len(result["common"])

    def test_scene_pair_mismatch_is_flagged(self, tmp_path):
        """Different scene pairs make the two runs incomparable."""
        reports = {}
        for label, scene_b in (("pi05", (7, 7)), ("groot", (9, 9))):
            run = aag.load_run(write(
                tmp_path, f"{label}.json",
                make_run({"T": {"A": arm(3), "B": arm(3)}}, scene_b=scene_b),
            ))
            reports[label] = {"rows": run["rows"], "run": run}
        assert aag.intersect_reports(reports)["scene_mismatch"] is not None

    def test_matching_scene_pairs_are_not_flagged(self, tmp_path):
        result = self.build(
            tmp_path,
            {"T": {"A": arm(3), "B": arm(3)}},
            {"T": {"A": arm(3), "B": arm(3)}},
        )
        assert result["scene_mismatch"] is None
        assert result["n_trials"] == 5


class TestPooled:
    def test_reproduces_the_recorded_pi05_pooled_numbers(self):
        """The 18 rows recorded in §12-2, pooled, must give the logged verdict."""
        recorded = [
            (4, 3), (3, 3), (4, 4), (2, 2), (5, 5), (1, 1), (2, 3), (3, 4),
            (3, 4), (3, 4), (3, 5), (2, 5), (2, 0), (0, 2), (0, 0), (0, 0),
            (0, 0), (0, 0),
        ]
        rows = [
            {"task": f"T{i}", "a": a, "b": b, "n": 5, "cls": aag.classify(a, b),
             "gap": (a - b) / 5, "wall": 0.0}
            for i, (a, b) in enumerate(recorded)
        ]
        stats = aag.pooled(rows)
        assert stats["n"] == 90
        assert stats["k_a"] == 37 and stats["k_b"] == 45
        assert stats["ci_b"][0] == pytest.approx(0.399, abs=5e-4)
        assert stats["p1_pass"] is True
        assert stats["gap"] * 100 == pytest.approx(-8.9, abs=0.05)

    def test_p1_fails_when_the_lower_bound_sits_under_the_crash_line(self):
        rows = [
            {"task": "T", "a": 1, "b": 1, "n": 10, "cls": "U2",
             "gap": 0.0, "wall": 0.0}
        ]
        stats = aag.pooled(rows)
        assert stats["sr_b"] == pytest.approx(0.1)
        assert stats["p1_pass"] is False
