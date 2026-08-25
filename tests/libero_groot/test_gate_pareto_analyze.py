"""Aggregation counts what the gate actually did, not just what the judge said.

The teacher ratio has to include gate-skipped steps. A skip is a step where the
gate declined to consult the cache at all, so the teacher ran -- exactly like a
judge-rejected step. Counting only judge MISSes would understate the x-axis by
precisely the gate's own intervention rate, which is the quantity this
experiment exists to measure, so the frontier would be shifted left by an
amount that grows with how hard the gate is working.
"""

from __future__ import annotations

import json

import pytest

from exp.libero_groot.analysis.gate_pareto.analyze_gate_pareto import (
    IntegrityError,
    _f_fh_of,
    aggregate,
    aggregate_arm,
    arms_from_config,
    pareto_front,
    write_manifest,
)


def _results(n: int, successes: int) -> list[dict]:
    return [
        {
            "task_id": 0,
            "init_state_idx": i,
            "orig_init_state_idx": i,
            "episode_id": i,
            "seed": 7,
            "success": i < successes,
        }
        for i in range(n)
    ]


def _row(ep: int, step: int, hit_type: str, searched: bool) -> dict:
    return {
        "yaml_id": "gp_sp_fh40",
        "episode_id": ep,
        "step_idx": step,
        "hit_type": hit_type,
        "searched": searched,
        "cp1_score": 0.9 if searched else None,
    }


def test_gate_skipped_steps_count_as_teacher_calls():
    # Two hits, one judge-MISS, one gate-skip MISS -> half the decisions ran
    # the teacher, and the two MISS kinds are not distinguished in the ratio.
    per_step = [
        _row(0, 0, "FULL_HIT", True),
        _row(0, 1, "MISS", True),      # judge rejected the retrieval
        _row(0, 2, "MISS", False),     # gate declined to search at all
        _row(0, 3, "FULL_HIT", True),
    ]
    point = aggregate_arm(_results(1, 1), per_step, expect_ep=1)
    assert point["decisions"] == 4
    assert point["misses"] == 2
    assert point["teacher_ratio"] == 0.5


def test_a_gate_only_arm_still_reports_a_nonzero_ratio():
    # With the verdict disabled every searched step is a FULL_HIT, so the only
    # teacher calls left are the gate's own -- the point of the ablation.
    per_step = [_row(0, s, "FULL_HIT", True) for s in range(7)]
    per_step.append(_row(0, 7, "MISS", False))
    point = aggregate_arm(_results(1, 1), per_step, expect_ep=1)
    assert point["teacher_ratio"] == pytest.approx(1 / 8)


def test_rows_without_a_verdict_are_not_decisions():
    per_step = [
        _row(0, 0, "FULL_HIT", True),
        dict(_row(0, 1, "MISS", True), hit_type=None),
    ]
    point = aggregate_arm(_results(1, 1), per_step, expect_ep=1)
    assert point["decisions"] == 1


def test_an_arm_with_no_decisions_reports_none_not_zero():
    # Zero would place the arm at the frontier's left edge, which is the most
    # favourable possible position for an arm with no evidence at all.
    point = aggregate_arm(_results(2, 1), [], expect_ep=2)
    assert point["teacher_ratio"] is None
    assert point["success_rate"] == 0.5


@pytest.mark.parametrize(
    "arm,expected",
    [
        ("gp_sp_fh05", 0.05),
        ("gp_l10_fh80", 0.80),
        ("gpgo_sp", None),
        ("gpw_sp", None),
    ],
)
def test_f_fh_is_parsed_from_the_arm_name(arm, expected):
    assert _f_fh_of(arm) == expected


def test_pareto_front_keeps_only_undominated_points():
    points = [(0.3, 0.90), (0.4, 0.88), (0.5, 0.95), (0.6, 0.93)]
    assert pareto_front(points) == [(0.3, 0.90), (0.5, 0.95)]


def _write_arm(results_dir, per_step_dir, arm, results, rows, lanes_found=2):
    (results_dir / f"{arm}.json").write_text(json.dumps(results), encoding="utf-8")
    (per_step_dir / f"{arm}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (per_step_dir / f"{arm}.merge.json").write_text(
        json.dumps({"lanes_expected": 2, "lanes_found": lanes_found, "rows": len(rows)}),
        encoding="utf-8",
    )


@pytest.fixture
def dirs(tmp_path):
    results_dir, per_step_dir = tmp_path / "results", tmp_path / "per_step"
    results_dir.mkdir()
    per_step_dir.mkdir()
    return results_dir, per_step_dir


_ROWS = [lambda: [_row(0, 0, "FULL_HIT", True), _row(1, 0, "MISS", False)]][0]


def test_aggregate_walks_the_expected_arm_set_and_skips_partials(dirs):
    results_dir, per_step_dir = dirs
    _write_arm(results_dir, per_step_dir, "gp_sp_fh40", _results(2, 1), _ROWS())
    # A partial file is a cell the scheduler will re-run; it is not a complete
    # arm, so it must not be aggregated and must not read as an extra arm.
    (results_dir / "gp_sp_fh45.partial.json").write_text("[]", encoding="utf-8")

    summary = aggregate(
        results_dir, per_step_dir, expect_ep=2, expect_arms={"gp_sp_fh40"}
    )
    assert set(summary) == {"gp_sp_fh40"}
    assert summary["gp_sp_fh40"]["teacher_ratio"] == 0.5


def test_aggregate_refuses_an_arm_that_never_produced_a_result(dirs):
    # The failure enumeration cannot see: the summary would simply come back
    # one point smaller, and a frontier through the rest looks perfectly fine.
    results_dir, per_step_dir = dirs
    _write_arm(results_dir, per_step_dir, "gp_sp_fh40", _results(2, 1), _ROWS())
    with pytest.raises(IntegrityError, match="produced no complete results file"):
        aggregate(
            results_dir, per_step_dir, expect_ep=2,
            expect_arms={"gp_sp_fh40", "gp_sp_fh45"},
        )


def test_aggregate_refuses_an_empty_results_directory(dirs):
    results_dir, per_step_dir = dirs
    with pytest.raises(IntegrityError, match="produced no complete results file"):
        aggregate(
            results_dir, per_step_dir, expect_ep=2, expect_arms={"gp_sp_fh40"}
        )


def test_aggregate_refuses_an_empty_expected_set(dirs):
    results_dir, per_step_dir = dirs
    with pytest.raises(IntegrityError, match="refusing to aggregate nothing"):
        aggregate(results_dir, per_step_dir, expect_ep=2, expect_arms=set())


def test_aggregate_refuses_a_result_outside_the_phase(dirs):
    # A stale arm left over from a previous emit would otherwise be plotted as
    # if it belonged to this sweep.
    results_dir, per_step_dir = dirs
    _write_arm(results_dir, per_step_dir, "gp_sp_fh40", _results(2, 1), _ROWS())
    _write_arm(results_dir, per_step_dir, "gp_sp_fh99", _results(2, 1), _ROWS())
    with pytest.raises(IntegrityError, match="unexpected result files"):
        aggregate(
            results_dir, per_step_dir, expect_ep=2, expect_arms={"gp_sp_fh40"}
        )


def test_aggregate_refuses_an_arm_whose_per_step_file_is_absent(dirs):
    results_dir, per_step_dir = dirs
    (results_dir / "gp_sp_fh40.json").write_text(
        json.dumps(_results(2, 1)), encoding="utf-8"
    )
    with pytest.raises(IntegrityError, match="per-step file missing"):
        aggregate(
            results_dir, per_step_dir, expect_ep=2, expect_arms={"gp_sp_fh40"}
        )


def test_aggregate_refuses_an_arm_whose_sidecar_is_absent(dirs):
    # Previously this silently passed merge=None and skipped I6 entirely -- the
    # check was strongest on paper for exactly the runs least able to support it.
    results_dir, per_step_dir = dirs
    _write_arm(results_dir, per_step_dir, "gp_sp_fh40", _results(2, 1), _ROWS())
    (per_step_dir / "gp_sp_fh40.merge.json").unlink()
    with pytest.raises(IntegrityError, match="merge sidecar missing"):
        aggregate(
            results_dir, per_step_dir, expect_ep=2, expect_arms={"gp_sp_fh40"}
        )


def test_aggregate_propagates_a_missing_worker_file(dirs):
    results_dir, per_step_dir = dirs
    _write_arm(
        results_dir, per_step_dir, "gp_sp_fh40", _results(2, 1), _ROWS(),
        lanes_found=1,
    )
    with pytest.raises(IntegrityError, match="1 of 2 worker files"):
        aggregate(
            results_dir, per_step_dir, expect_ep=2, expect_arms={"gp_sp_fh40"}
        )


def test_expected_arms_come_from_the_emitted_recipes(tmp_path):
    yaml_dir = tmp_path / "eval"
    yaml_dir.mkdir()
    for arm in ("gp_sp_fh05", "gp_sp_fh80"):
        (yaml_dir / f"{arm}.yaml").write_text("{}", encoding="utf-8")
    assert arms_from_config(yaml_dir) == {"gp_sp_fh05", "gp_sp_fh80"}


def test_an_empty_recipe_directory_is_refused(tmp_path):
    empty = tmp_path / "eval"
    empty.mkdir()
    with pytest.raises(IntegrityError, match="no arm recipes"):
        arms_from_config(empty)


def test_manifest_digests_every_generated_file(tmp_path):
    (tmp_path / "pareto_libero_spatial.png").write_bytes(b"png-bytes")
    (tmp_path / "plot_data.json").write_text("{}", encoding="utf-8")
    path = write_manifest(tmp_path, {"libero_spatial": "/data/summary.json"})
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["sources"] == {"libero_spatial": "/data/summary.json"}
    names = {f["file"] for f in manifest["files"]}
    assert names == {"pareto_libero_spatial.png", "plot_data.json"}
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])
    # The manifest must not digest itself: rewriting it would change its own
    # hash and the record would never be reproducible.
    assert "MANIFEST.json" not in names
