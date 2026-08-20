"""Sweep-shape invariants: one degree of freedom, no warm tier, gate L intact."""

from __future__ import annotations

import json

import pytest
import yaml

from exp.data_authority.registry import record_path_for
from exp.gate_threshold_pareto import libraries as libs
from exp.gate_threshold_pareto import solve_gtp
from exp.gate_threshold_pareto.emit_gtp_yamls import (
    FORCE_MISS,
    GATE_J,
    GATE_L,
    GATE_PROBE_INTERVAL,
    build_eval,
    build_warmup,
)
from exp.gate_threshold_pareto.run_gtp import validate_arms


# ------------------------------------------------------------------
# Library table
# ------------------------------------------------------------------


def test_four_libraries_two_per_suite():
    assert len(libs.LIBRARIES) == 4
    assert len(libs.for_suite("libero_spatial")) == 2
    assert len(libs.for_suite("libero_10")) == 2
    assert len({libs.arm_key(x) for x in libs.LIBRARIES}) == 4


def test_every_library_is_registered_in_the_data_authority_ledger():
    # The sweep quotes success rates per library; if a library is not in the
    # ledger there is no attested statement of which bytes produced them.
    for lib in libs.LIBRARIES:
        path = record_path_for(lib.dataset_id)
        assert path.is_file(), f"{lib.dataset_id} has no ledger record"
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec["suite"] == lib.suite


def test_by_arm_round_trips_and_rejects_unknown():
    for lib in libs.LIBRARIES:
        assert libs.by_arm(libs.arm_key(lib)) is lib
    with pytest.raises(KeyError):
        libs.by_arm("nope")


# ------------------------------------------------------------------
# Grid
# ------------------------------------------------------------------


def test_grid_is_sixteen_cells_one_axis():
    assert len(solve_gtp.FH_GRID) == 16
    assert solve_gtp.FH_GRID[0] == 0.05
    assert solve_gtp.FH_GRID[-1] == 0.8
    assert list(solve_gtp.FH_GRID) == sorted(solve_gtp.FH_GRID)


def test_theta_reproduces_the_historical_anchor_convention():
    from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds

    scores = [i / 1000 for i in range(1000)]
    got = solve_gtp.solve_arm(scores, arm="x")
    assert (
        got["theta"] == derive_thresholds(scores, solve_gtp.THETA_TOP_FRACTION, 0.0)[0]
    )
    assert solve_gtp.THETA_TOP_FRACTION == 0.85


def test_t_fh_is_monotone_decreasing_in_f_fh():
    scores = [i / 1000 for i in range(1000)]
    cells = solve_gtp.solve_arm(scores, arm="x")["cells"]
    ts = [c["t_fh"] for c in cells]
    assert ts == sorted(ts, reverse=True)


# ------------------------------------------------------------------
# Solver front gates
# ------------------------------------------------------------------


def test_too_few_scores_fails_loud():
    with pytest.raises(SystemExit, match="usable warmup scores"):
        solve_gtp.solve_arm([0.5] * 10, arm="thin")


def test_too_few_distinct_values_fails_loud():
    # 1000 samples but only 4 distinct values: the 16 cells cannot be separated,
    # and duplicated operating points would look like independent evidence.
    with pytest.raises(SystemExit, match="distinct values"):
        solve_gtp.solve_arm([0.1, 0.2, 0.3, 0.4] * 250, arm="flat")


def test_load_scores_drops_nulls_and_groups_by_arm(tmp_path):
    path = tmp_path / "per_step.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"yaml_id": "a", "cp1_score": 0.1},
                {"yaml_id": "a", "cp1_score": None},
                {"yaml_id": "b", "cp1_score": 0.9},
                {"task_uid": "c:eval:0:0", "cp1_score": 0.5},
            ]
        ),
        encoding="utf-8",
    )
    got = solve_gtp.load_scores_by_arm(path)
    assert got == {"a": [0.1], "b": [0.9], "c": [0.5]}


# ------------------------------------------------------------------
# Emitted config shape
# ------------------------------------------------------------------


def test_warmup_forces_miss_and_searches_every_step():
    cfg = build_warmup(libs.LIBRARIES[0])
    cp1 = cfg["checkpoints"]["cp1"]
    assert cp1["gate"] == {"type": "always_search"}
    assert cp1["judge"] == {"type": "threshold", "threshold": FORCE_MISS}
    assert "warm_tiers" not in cp1["judge"]
    assert cfg["write_policy"] == {"type": "never"}


def test_eval_carries_the_hybrid_gate_and_a_binary_verdict():
    lib = libs.LIBRARIES[0]
    cfg = build_eval(lib, theta=0.97, t_fh=0.98)
    cp1 = cfg["checkpoints"]["cp1"]
    assert cp1["gate"] == {
        "type": "score_hysteresis",
        "theta_low": 0.97,
        "theta_high": 0.97,
        "j": GATE_J,
        "probe_interval": GATE_PROBE_INTERVAL,
        "L": GATE_L,
    }
    assert cp1["judge"] == {"type": "threshold", "threshold": 0.98}
    assert "warm_tiers" not in cp1["judge"]
    assert cfg["backend"]["in_memory"]["preload_path"] == lib.preload_path


def test_gate_l_is_the_stage3a_winning_point():
    assert GATE_L == 6
    assert (GATE_J, GATE_PROBE_INTERVAL) == (3, 3)


# ------------------------------------------------------------------
# Runner gates
# ------------------------------------------------------------------


def _arm_file(tmp_path, cfg, name="a"):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return [{"arm": name, "yaml": str(path)}]


def test_runner_accepts_well_formed_arms(tmp_path):
    lib = libs.LIBRARIES[0]
    assert validate_arms(_arm_file(tmp_path, build_warmup(lib), "w"), phase="warmup")
    assert validate_arms(
        _arm_file(tmp_path, build_eval(lib, theta=0.97, t_fh=0.98), "e"), phase="eval"
    )


def test_runner_rejects_a_surviving_warm_tier(tmp_path):
    cfg = build_eval(libs.LIBRARIES[0], theta=0.97, t_fh=0.98)
    cfg["checkpoints"]["cp1"]["judge"]["warm_tiers"] = [
        {"threshold": 0.9, "start_t": 0.5}
    ]
    with pytest.raises(SystemExit, match="warm tier present"):
        validate_arms(_arm_file(tmp_path, cfg, "wt"), phase="eval")


def test_runner_rejects_a_gated_warmup(tmp_path):
    cfg = build_eval(libs.LIBRARIES[0], theta=0.97, t_fh=0.98)
    with pytest.raises(SystemExit, match="expected 'always_search'"):
        validate_arms(_arm_file(tmp_path, cfg, "gw"), phase="warmup")


def test_runner_rejects_an_eval_arm_that_lost_its_L(tmp_path):
    cfg = build_eval(libs.LIBRARIES[0], theta=0.97, t_fh=0.98)
    del cfg["checkpoints"]["cp1"]["gate"]["L"]
    with pytest.raises(SystemExit, match="gate L is"):
        validate_arms(_arm_file(tmp_path, cfg, "nol"), phase="eval")


def test_runner_rejects_an_ungated_eval_arm(tmp_path):
    cfg = build_warmup(libs.LIBRARIES[0])
    with pytest.raises(SystemExit, match="expected 'score_hysteresis'"):
        validate_arms(_arm_file(tmp_path, cfg, "ug"), phase="eval")


# ------------------------------------------------------------------
# Resume filter (2026-08-20 bundle-swap burst)
# ------------------------------------------------------------------


def test_resume_filter_drops_only_completed_arms(tmp_path):
    from exp.gate_threshold_pareto.run_gtp import arms_with_work_left

    journal = tmp_path / "journal.jsonl"
    rows = []
    for i in range(10):  # complete
        rows.append({"yaml_id": "done_arm", "task_uid": f"done_arm:eval:0:{i}"})
    for i in range(4):  # partial
        rows.append({"yaml_id": "partial_arm", "task_uid": f"partial_arm:eval:0:{i}"})
    journal.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    remaining, counts = arms_with_work_left(
        journal, ["done_arm", "partial_arm", "fresh_arm"], expected=10
    )
    assert remaining == ["partial_arm", "fresh_arm"]
    assert counts == {"done_arm": 10, "partial_arm": 4, "fresh_arm": 0}


def test_resume_filter_counts_distinct_uids_not_lines(tmp_path):
    # A retried episode writes a second terminal record for the same task_uid.
    # Counting lines would call a 5-episode arm "complete" at 10 lines and skip
    # the 5 episodes it still owes.
    from exp.gate_threshold_pareto.run_gtp import arms_with_work_left

    journal = tmp_path / "journal.jsonl"
    rows = [{"yaml_id": "a", "task_uid": f"a:eval:0:{i}"} for i in range(5)] * 2
    journal.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    remaining, counts = arms_with_work_left(journal, ["a"], expected=10)
    assert counts == {"a": 5}
    assert remaining == ["a"]


def test_resume_filter_is_a_noop_without_a_journal(tmp_path):
    from exp.gate_threshold_pareto.run_gtp import arms_with_work_left

    remaining, counts = arms_with_work_left(
        tmp_path / "nope.jsonl", ["a", "b"], expected=10
    )
    assert remaining == ["a", "b"]
    assert counts == {}
