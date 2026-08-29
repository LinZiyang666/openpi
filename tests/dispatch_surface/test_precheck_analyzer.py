"""Tests for the precheck adjudicator under the 2026-08-27 cost-axis ruling.

Cost is analytic (verdict counts x frozen CUDA-graph stage costs), the two
gates run on one shared resample, and Gate 2 is an intersection-union test.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from exp.dispatch_surface.analysis.analyze_precheck import (
    ARM_S0,
    ARM_SV,
    CORE_T_ARMS,
    EXPECTED_TRIALS,
    GATE2_COMPUTE_GATE,
    PINNED_START_T_WS,
    STAGE1_MS,
    STAGE2_MS,
    STAGE3_MS,
    arm_cost,
    check_discipline,
    frontier_record,
    gate1,
    gate2,
    load_accepted_episodes,
    load_analytic_cost,
    official_by_task,
    parse_task_uid,
    unit_cost,
)

FULL_MS = STAGE1_MS + STAGE2_MS + STAGE3_MS


# ------------------------------------------------------------------
# Analytic unit cost
# ------------------------------------------------------------------


def test_full_hit_pays_stage1_only():
    assert unit_cost("FULL_HIT", None) == STAGE1_MS


def test_miss_pays_all_three_stages():
    assert unit_cost("MISS", None) == pytest.approx(FULL_MS)


def test_warm_start_runs_start_t_of_stage3():
    """start_t is what REMAINS to run: 0.3 runs 3 of 10 steps, saving 70%."""
    got = unit_cost("WARM_START", PINNED_START_T_WS)
    assert got == pytest.approx(STAGE1_MS + STAGE2_MS + 0.3 * STAGE3_MS)
    assert (FULL_MS - got) / STAGE3_MS == pytest.approx(0.7)


@pytest.mark.parametrize("bad", [None, 0.5, 0.7, 0.0])
def test_off_tier_warm_start_refused(bad):
    """Only the pinned tier has a calibrated cost; anything else is a refusal."""
    with pytest.raises(SystemExit):
        unit_cost("WARM_START", bad)


@pytest.mark.parametrize("bad", [None, "UNPROBED", "unknown", "", "miss"])
def test_unknown_verdict_is_never_silently_billed_as_miss(bad):
    with pytest.raises(SystemExit):
        unit_cost(bad, None)


# ------------------------------------------------------------------
# Ratio-of-sums estimand
# ------------------------------------------------------------------


def test_arm_cost_is_decision_weighted_not_episode_weighted():
    """The ruling's estimand: sum of costs over sum of decisions.

    A short cheap episode and a long expensive one must NOT count equally --
    that would estimate the cost of a random episode instead of a random
    decision, which is a different quantity whenever lengths differ.
    """
    cells = {(0, 0): (10.0, 1), (0, 1): (900.0, 9)}
    assert arm_cost(cells, [(0, 0), (0, 1)]) == pytest.approx(910.0 / 10)
    episode_weighted = (10.0 / 1 + 900.0 / 9) / 2
    assert episode_weighted == pytest.approx(55.0)
    assert arm_cost(cells, [(0, 0), (0, 1)]) != pytest.approx(episode_weighted)


def test_arm_cost_reforms_the_ratio_after_resampling():
    """Duplicating a cell must re-weight both numerator and denominator."""
    cells = {(0, 0): (10.0, 1), (0, 1): (900.0, 9)}
    doubled_short = arm_cost(cells, [(0, 0), (0, 0), (0, 1)])
    assert doubled_short == pytest.approx((10.0 + 10.0 + 900.0) / (1 + 1 + 9))


def test_arm_cost_refuses_an_empty_denominator():
    with pytest.raises(SystemExit):
        arm_cost({(0, 0): (0.0, 0)}, [(0, 0)])


# ------------------------------------------------------------------
# per_step loading -- REAL producer schema
# ------------------------------------------------------------------
#
# The real rows come from two producers and do NOT share a field set:
#   * verdict rows: episode_runner writes hit_type/start_t/orig_init_state_idx,
#     the driver then stamps task_uid/yaml_id/attempt/accepted/run_id.
#   * client_timing rows: episode_runner writes _kind/task_uid/yaml_id/task_id/
#     subset_init_state_idx + the timing fields -- NO orig_init_state_idx
#     (examples/libero/episode_runner.py). An analyzer that reads that field
#     before dispatching on _kind dies on the first real episode.

RUN = "run0123456789"


def _per_step(tmp_path, rows, name="per_step.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def _verdict_rows(arm, task, subset, official, verdicts, *,
                  attempt=1, run_id=RUN, accepted=True, success=True):
    uid = f"{arm}:eval:{task}:{subset}"
    return [{"yaml_id": arm, "task_uid": uid, "task_id": task,
             "orig_init_state_idx": official, "subset_init_state_idx": subset,
             "episode_id": task * EXPECTED_TRIALS + subset, "step_idx": i * 5,
             "phase": "eval",
             "hit_type": h,
             "start_t": PINNED_START_T_WS if h == "WARM_START" else None,
             "attempt": attempt, "accepted": accepted, "run_id": run_id,
             "success": success}
            for i, h in enumerate(verdicts)]


def _timing_row(arm, task, subset, infers, *, attempt=1, run_id=RUN, accepted=True,
                success=True):
    """Exactly the shape episode_runner emits, plus the driver's stamps."""
    return {"_kind": "client_timing", "task_uid": f"{arm}:eval:{task}:{subset}",
            "yaml_id": arm, "task_id": task, "subset_init_state_idx": subset,
            "infer_ms": 100.0 * infers, "infers": infers, "steps": 5 * infers,
            "attempt": attempt, "accepted": accepted, "run_id": run_id,
            "success": success}


def _episode(arm, task, subset, official, verdicts, **kw):
    return (_verdict_rows(arm, task, subset, official, verdicts, **kw)
            + [_timing_row(arm, task, subset, len(verdicts), **kw)])


def _accepted(arm, cells, *, attempt=1, run_id=RUN):
    return {arm: {c: {"success": True, "task_uid": f"{arm}:eval:{c[0]}:{c[1]}",
                      "attempt": attempt, "run_id": run_id} for c in cells}}


OFFICIALS = {0: [3, 11, 27]}


def test_client_timing_without_orig_init_is_accepted(tmp_path):
    """B1: the real client_timing row has no orig_init_state_idx."""
    rows = _episode("a", 0, 0, 3, ["FULL_HIT", "MISS"])
    assert "orig_init_state_idx" not in rows[-1]      # the real shape
    cells, summary = load_analytic_cost(
        _per_step(tmp_path, rows), ["a"], _accepted("a", [(0, 0)]), OFFICIALS)
    assert cells["a"][(0, 0)] == (pytest.approx(STAGE1_MS + FULL_MS), 2)
    assert summary["verdict_counts"]["a"] == {"FULL_HIT": 1, "MISS": 1}


def test_missing_client_timing_row_is_refused(tmp_path):
    """B1: the decision count must be checkable against the episode's own."""
    rows = _verdict_rows("a", 0, 0, 3, ["MISS"])     # no timing row
    with pytest.raises(SystemExit):
        load_analytic_cost(_per_step(tmp_path, rows), ["a"],
                           _accepted("a", [(0, 0)]), OFFICIALS)


def test_infers_mismatch_is_refused(tmp_path):
    rows = _verdict_rows("a", 0, 0, 3, ["MISS", "MISS"]) + [_timing_row("a", 0, 0, 5)]
    with pytest.raises(SystemExit):
        load_analytic_cost(_per_step(tmp_path, rows), ["a"],
                           _accepted("a", [(0, 0)]), OFFICIALS)


def test_stale_attempt_rows_are_excluded_not_billed(tmp_path):
    """B2: a requeued episode's old attempt must not reach the cost."""
    stale = _episode("a", 0, 0, 3, ["MISS", "MISS", "MISS"], attempt=1, accepted=False)
    live = _episode("a", 0, 0, 3, ["FULL_HIT"], attempt=2)
    cells, summary = load_analytic_cost(
        _per_step(tmp_path, stale + live), ["a"],
        _accepted("a", [(0, 0)], attempt=2), OFFICIALS)
    assert cells["a"][(0, 0)] == (pytest.approx(STAGE1_MS), 1)
    assert summary["excluded_stale_rows"]["a"] == len(stale)


def test_fenced_report_from_the_same_attempt_is_excluded(tmp_path):
    """B2: a fenced dispatch reports rows too; only the scheduler's pick counts."""
    fenced = _episode("a", 0, 0, 3, ["MISS", "MISS"], attempt=2, accepted=False)
    live = _episode("a", 0, 0, 3, ["FULL_HIT"], attempt=2)
    cells, _ = load_analytic_cost(
        _per_step(tmp_path, fenced + live), ["a"],
        _accepted("a", [(0, 0)], attempt=2), OFFICIALS)
    assert cells["a"][(0, 0)][1] == 1


def test_rows_from_another_run_are_excluded(tmp_path):
    """B2: resume restarts attempt numbering; run_id is the discriminator."""
    other = _episode("a", 0, 0, 3, ["MISS", "MISS"], run_id="otherrun0001")
    live = _episode("a", 0, 0, 3, ["FULL_HIT"])
    cells, _ = load_analytic_cost(
        _per_step(tmp_path, other + live), ["a"],
        _accepted("a", [(0, 0)]), OFFICIALS)
    assert cells["a"][(0, 0)][1] == 1


def test_row_outside_the_adjudicated_grid_fails_closed(tmp_path):
    """B2: an unexpected cell is a refusal, never a silent skip."""
    rows = _episode("a", 0, 0, 3, ["MISS"]) + _episode("a", 0, 7, 41, ["MISS"])
    with pytest.raises(SystemExit):
        load_analytic_cost(_per_step(tmp_path, rows), ["a"],
                           _accepted("a", [(0, 0)]), OFFICIALS)


def test_official_init_index_is_cross_checked(tmp_path):
    """B3: the row's official index must match what the split manifest froze."""
    rows = _episode("a", 0, 1, 99, ["MISS"])     # subset 1 -> official 11, not 99
    with pytest.raises(SystemExit):
        load_analytic_cost(_per_step(tmp_path, rows), ["a"],
                           _accepted("a", [(0, 1)]), OFFICIALS)


def test_row_without_task_uid_is_refused(tmp_path):
    rows = [{"yaml_id": "a", "task_id": 0, "step_idx": 0, "hit_type": "MISS"}]
    with pytest.raises(SystemExit):
        load_analytic_cost(_per_step(tmp_path, rows), ["a"],
                           _accepted("a", [(0, 0)]), OFFICIALS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accepted", None),
        ("task_uid", "other:eval:0:0"),
        ("task_id", 9),
        ("subset_init_state_idx", 2),
        ("episode_id", 99),
        ("_kind", "episode_summary"),
        ("phase", "warmup"),
        ("success", False),
    ],
)
def test_per_step_identity_mismatches_fail_closed(tmp_path, field, value):
    rows = _episode("a", 0, 0, 3, ["MISS"])
    if value is None:
        rows[0].pop(field)
    else:
        rows[0][field] = value
    with pytest.raises(SystemExit):
        load_analytic_cost(
            _per_step(tmp_path, rows), ["a"], _accepted("a", [(0, 0)]), OFFICIALS
        )


def test_duplicate_verdict_step_is_refused_even_when_infers_count_matches(tmp_path):
    rows = _episode("a", 0, 0, 3, ["FULL_HIT", "MISS"])
    rows[1]["step_idx"] = rows[0]["step_idx"]
    with pytest.raises(SystemExit, match="duplicate verdict step_idx"):
        load_analytic_cost(
            _per_step(tmp_path, rows), ["a"], _accepted("a", [(0, 0)]), OFFICIALS
        )


def test_cost_summary_carries_the_unit_cost_ledger(tmp_path):
    rows = _episode("a", 0, 0, 3, ["MISS"])
    _cells, summary = load_analytic_cost(
        _per_step(tmp_path, rows), ["a"], _accepted("a", [(0, 0)]), OFFICIALS)
    assert summary["unit_cost_ms"]["stage1"] == STAGE1_MS
    assert summary["unit_cost_ms"]["MISS"] == pytest.approx(FULL_MS)
    assert len(summary["per_step_sha256"]) == 64


def test_parse_task_uid():
    assert parse_task_uid("dsp_sv:eval:7:29") == ("dsp_sv", 7, 29)
    with pytest.raises(SystemExit):
        parse_task_uid("nope")


def test_official_by_task_reads_the_frozen_manifest(tmp_path):
    man = tmp_path / "split.json"
    man.write_text(json.dumps({"quota": {"test": 3}, "assignment": {
        str(t): {"test": [40 + t, 2 + t, 17 + t]} for t in range(10)}}))
    out = official_by_task(str(man), 3)
    assert out[0] == [2, 17, 40]          # materialised in sorted order
    with pytest.raises(SystemExit):
        official_by_task(str(man), 30)    # more trials than A' holds


# ------------------------------------------------------------------
# Frontier record
# ------------------------------------------------------------------


def test_bracket_branch_interpolates():
    t = [("a", 0.40, 100.0), ("b", 0.60, 200.0), ("c", 0.80, 300.0)]
    r = frontier_record(t, (0.50, 120.0))
    assert r["branch"] == "bracket"
    assert r["D_sr"] == pytest.approx(0.0)
    assert r["D_c"] == pytest.approx((120.0 - 150.0) / 150.0)


def test_high_branch_clamps_to_argmax_sr_cell():
    t = [("a", 0.40, 100.0), ("b", 0.60, 200.0), ("c", 0.80, 300.0)]
    r = frontier_record(t, (0.90, 240.0))
    assert r["branch"] == "high"
    assert r["D_sr"] == pytest.approx(0.10)
    assert r["D_c"] == pytest.approx((240.0 - 300.0) / 300.0)


def test_low_branch_clamps_and_stays_in_the_cost_distribution():
    t = [("a", 0.40, 100.0), ("b", 0.60, 200.0), ("c", 0.80, 300.0)]
    r = frontier_record(t, (0.20, 90.0))
    assert r["branch"] == "low"
    assert r["D_sr"] == pytest.approx(-0.20)
    # The low replicate still produces D_c: there is no deletion path.
    assert r["D_c"] == pytest.approx((90.0 - 100.0) / 100.0)


def test_nonpositive_comparator_cost_refused():
    """A zero comparator cost would make D_c a division by zero."""
    t = [("a", 0.4, 0.0), ("b", 0.6, 1.0), ("c", 0.8, 2.0)]
    with pytest.raises(SystemExit):
        frontier_record(t, (0.20, 1.0))   # low: clamps onto the zero-cost cell
    with pytest.raises(SystemExit):
        frontier_record(t, (0.40, 1.0))   # exact hit on the zero-cost cell


# ------------------------------------------------------------------
# Gates
# ------------------------------------------------------------------


def _records(d_sr=0.0, d_c=-0.10, n=1000, branch="bracket"):
    return [{"branch": branch, "D_sr": d_sr, "D_c": d_c} for _ in range(n)]


def test_gate1_passes_on_clear_win():
    assert gate1(_records())["pass"]


def test_gate1_fails_on_compute_gate():
    assert not gate1(_records(d_c=-0.01))["pass"]


def test_gate1_low_replicates_fail_sr_floor():
    recs = _records(n=900) + _records(d_sr=-0.05, n=100, branch="low")
    out = gate1(recs)
    assert out["d_sr_p5"] < 0 and not out["pass"]
    assert out["branch_shares"]["low"] == pytest.approx(0.10)


def test_gate1_has_no_latency_condition():
    """The latency gate was deleted with the cost-axis change."""
    out = gate1(_records())
    assert "d_l_p95" not in out


def test_gate2_needs_both_components():
    """Rev 1: the cost side now demands a saving, so "flat cost" no longer passes."""
    rng = np.random.default_rng(0)
    sr_up = rng.normal(0.05, 0.005, 4000)
    sr_flat = rng.normal(0.0, 0.005, 4000)
    cost_saving = rng.normal(-0.12, 0.005, 4000)
    cost_flat = rng.normal(0.0, 0.005, 4000)
    cost_bad = rng.normal(0.10, 0.005, 4000)
    assert gate2(sr_up, cost_saving)["pass"]
    assert not gate2(sr_flat, cost_saving)["pass"]   # SR not shown non-negative
    assert not gate2(sr_up, cost_flat)["pass"]       # no cost saving
    assert not gate2(sr_up, cost_bad)["pass"]        # cost regression
    # An SR gain with an unproven cost claim is reported distinctly.
    out = gate2(sr_up, cost_bad)
    assert out["sr_pass"] and not out["cost_pass"]


def test_gate2_cost_condition_is_a_one_sided_95_upper_bound():
    """Ruling 9.3: the guard runs at alpha=0.05, not 0.025, and not a point."""
    rng = np.random.default_rng(1)
    # Draws centred exactly on the boundary: a 95% upper bound must reject.
    at_boundary = rng.normal(GATE2_COMPUTE_GATE, 0.01, 20000)
    out = gate2(rng.normal(0.05, 0.005, 20000), at_boundary)
    assert not out["cost_pass"]
    assert out["dc_upper_quantile"] == 0.95
    # Rev 1: the cost side now demands a saving, not merely no increase.
    assert GATE2_COMPUTE_GATE == -0.05


def test_gate2_sr_component_is_superiority_not_non_inferiority():
    """Pins a live specification defect -- do not "fix" by loosening the gate.

    Rev 1 states the SR side as non-inferiority, but ``q0.05(dSR) >= 0`` demands
    the 5th percentile itself be non-negative, i.e. dSR significantly positive.
    With a genuinely equal SR the quantile is -1.645*sigma for ANY sigma > 0, so
    the exact scenario Rev 1 was rewritten to accept -- equal SR bought at a
    clearly lower cost, true Pareto dominance -- still fails. A real
    non-inferiority rule needs a pre-registered margin (``>= -m``), which is a
    frozen-criteria decision and is not taken here.
    """
    rng = np.random.default_rng(11)
    cheaper = rng.normal(-0.12, 0.01, 20000)      # 12% cheaper, as offline predicts
    for sigma in (0.02, 0.005, 0.001, 0.0002):
        out = gate2(rng.normal(0.0, sigma, 20000), cheaper)
        assert out["cost_pass"], "the cost side does accept a real saving"
        assert not out["sr_pass"], f"equal SR still fails at sigma={sigma}"
        assert not out["pass"]
    # It only passes once dSR is roughly 1.645 sigma clear of zero.
    out = gate2(rng.normal(0.01, 0.005, 20000), cheaper)
    assert out["sr_pass"] and out["pass"]


def test_gate2_rev1_rejects_an_sr_gain_that_costs_more():
    rng = np.random.default_rng(12)
    out = gate2(rng.normal(0.05, 0.005, 20000), rng.normal(0.02, 0.005, 20000))
    assert out["sr_pass"] and not out["cost_pass"] and not out["pass"]


def test_gate2_rev1_rejects_a_cost_saving_bought_with_lost_sr():
    rng = np.random.default_rng(13)
    out = gate2(rng.normal(-0.03, 0.005, 20000), rng.normal(-0.12, 0.01, 20000))
    assert not out["sr_pass"] and out["cost_pass"] and not out["pass"]


def test_gate2_records_which_sr_quantile_was_used():
    rng = np.random.default_rng(2)
    out = gate2(rng.normal(0.05, 0.005, 1000), rng.normal(0.0, 0.005, 1000))
    assert out["dsr_lower_quantile"] == 0.05


def test_no_gate3_confirmatory_path_exists():
    import inspect

    import exp.dispatch_surface.analysis.analyze_precheck as mod

    src = inspect.getsource(mod)
    assert "gate3" not in src.lower()


# ------------------------------------------------------------------
# Journal loading -- accepted attempt selection
# ------------------------------------------------------------------


def _journal(tmp_path, rows):
    p = tmp_path / "journal.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def _jrow(arm, task, subset, ok=True, *, accepted=True, attempt=1, run_id=RUN):
    return {"yaml_id": arm, "task_uid": f"{arm}:eval:{task}:{subset}",
            "phase": "eval", "accepted": accepted, "attempt": attempt,
            "run_id": run_id, "status": "done" if ok else "failed", "success": ok}


GRID2 = {(0, 0), (0, 1)}


def test_accepted_episodes_load(tmp_path):
    rows = [_jrow(a, 0, i) for a in ("x", "y") for i in (0, 1)]
    out = load_accepted_episodes(_journal(tmp_path, rows), ["x", "y"], GRID2)
    assert set(out["x"]) == GRID2
    assert out["x"][(0, 0)]["attempt"] == 1
    assert out["x"][(0, 0)]["run_id"] == RUN


def test_rejected_attempt_does_not_count_as_coverage(tmp_path):
    """B2: an accepted=false record leaves the cell uncovered."""
    rows = [_jrow("x", 0, 0), _jrow("x", 0, 1, accepted=False)]
    with pytest.raises(SystemExit):
        load_accepted_episodes(_journal(tmp_path, rows), ["x"], GRID2)


def test_two_accepted_records_for_one_cell_refused(tmp_path):
    rows = [_jrow("x", 0, 0), _jrow("x", 0, 0, attempt=2), _jrow("x", 0, 1)]
    with pytest.raises(SystemExit):
        load_accepted_episodes(_journal(tmp_path, rows), ["x"], GRID2)


def test_stale_attempt_beside_the_accepted_one_is_ignored(tmp_path):
    rows = [_jrow("x", 0, 0, accepted=False, attempt=1),
            _jrow("x", 0, 0, attempt=2), _jrow("x", 0, 1)]
    out = load_accepted_episodes(_journal(tmp_path, rows), ["x"], GRID2)
    assert out["x"][(0, 0)]["attempt"] == 2


def test_accepted_record_without_attempt_or_run_refused(tmp_path):
    rows = [_jrow("x", 0, 0), _jrow("x", 0, 1)]
    del rows[0]["run_id"]
    with pytest.raises(SystemExit):
        load_accepted_episodes(_journal(tmp_path, rows), ["x"], GRID2)


def test_episode_outside_the_grid_refused(tmp_path):
    rows = [_jrow("x", 0, 0), _jrow("x", 0, 1), _jrow("x", 0, 9)]
    with pytest.raises(SystemExit):
        load_accepted_episodes(_journal(tmp_path, rows), ["x"], GRID2)


def test_yaml_id_disagreeing_with_task_uid_refused(tmp_path):
    rows = [_jrow("x", 0, 0), _jrow("x", 0, 1)]
    rows[0]["task_uid"] = "y:eval:0:0"
    with pytest.raises(SystemExit):
        load_accepted_episodes(_journal(tmp_path, rows), ["x"], GRID2)


def test_equally_incomplete_grid_refused(tmp_path):
    rows = [_jrow(a, 0, 0) for a in ("x", "y")]
    with pytest.raises(SystemExit):
        load_accepted_episodes(_journal(tmp_path, rows), ["x", "y"], GRID2)


# ------------------------------------------------------------------
# Discipline
# ------------------------------------------------------------------


def _sha(path):
    import hashlib

    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _formal_fit_outputs(tmp_path, *, s0_delta=0.5, s0_inputs=None):
    """Write the same artifact/record integrity chain the formal fitter emits."""
    from openpi.cache.components.surface_judge import CERTIFICATION_EMPIRICAL
    from tests.cache.components.test_surface_judge import _contract, make_artifact

    sv_inputs = {"table": "T", "cohort_manifest": "C", "weights_npz": "W",
                 "rebuild_record": "R", "split_manifest": "S", "cache_yaml": "Y",
                 "d0_record": "D0"}
    common = {
        "d0_binding": {"record_sha256": "d" * 64, "input_rollup_sha256": "i" * 64},
        "dev_membership_sha256": "m" * 64,
        "fold_map_sha256": "f" * 64,
    }
    sv_final = {"q_deploy": "q-sv", "n_dev_rows": 150}
    s0_final = {"q_deploy": "q-s0", "n_dev_rows": 150}
    s0_inputs = sv_inputs if s0_inputs is None else s0_inputs
    (tmp_path / "sv_dir").mkdir(exist_ok=True)
    (tmp_path / "s0_dir").mkdir(exist_ok=True)
    sv_path = make_artifact(
        tmp_path / "sv_dir", delta=0.5, k=5,
        certification_mode=CERTIFICATION_EMPIRICAL, conformal_c=0.0,
        n_calibration_episodes=0,
        retrieval_contract={**_contract(), "top_k": 5},
        meta={"input_digests": sv_inputs, **common, "final_fit_digests": sv_final,
              "delta_name": "primary"},
    )
    s0_path = make_artifact(
        tmp_path / "s0_dir", uses_disagreement=False, delta=s0_delta, k=1,
        certification_mode=CERTIFICATION_EMPIRICAL, conformal_c=0.0,
        n_calibration_episodes=0,
        retrieval_contract={**_contract(), "top_k": 1},
        meta={"input_digests": s0_inputs, **common, "final_fit_digests": s0_final,
              "delta_name": "primary"},
    )
    sv_record = {
        "s_only": False, "certification_mode": CERTIFICATION_EMPIRICAL,
        "delta_star": 0.5, "input_digests": sv_inputs, **common,
        "final_fit_digests": sv_final, "artifacts": {"primary": sv_path},
    }
    s0_record = {
        "s_only": True, "certification_mode": CERTIFICATION_EMPIRICAL,
        "delta_star": s0_delta, "input_digests": s0_inputs, **common,
        "final_fit_digests": s0_final, "artifacts": {"primary": s0_path},
    }
    sv_record_path = tmp_path / "fit_record.json"
    s0_record_path = tmp_path / "fit_record_s_only.json"
    sv_record_path.write_text(json.dumps(sv_record))
    s0_record_path.write_text(json.dumps(s0_record))
    return sv_path, s0_path, sv_record_path, s0_record_path


def _discipline_fixture(tmp_path, *, s0_delta=0.5, s0_inputs=None,
                        launch_trials=30, tamper_arm_yaml=False,
                        launch_fp=None, drop_aprime=False, bad_ledger=False,
                        second_launch=None):
    import yaml as _yaml

    from openpi.cache.components.surface_judge import load_surface_artifact
    sv_path, s0_path, sv_record_path, s0_record_path = _formal_fit_outputs(
        tmp_path, s0_delta=s0_delta, s0_inputs=s0_inputs,
    )
    core = list(CORE_T_ARMS) + [ARM_S0, ARM_SV]
    arms, yaml_shas = {}, {}
    for arm in core:
        y = tmp_path / f"{arm}.yaml"
        if arm == ARM_SV:
            judge = {"surface_artifact_path": str(sv_path)}
        elif arm == ARM_S0:
            judge = {"surface_artifact_path": str(s0_path)}
        else:
            judge = {"type": "threshold"}
        y.write_text(_yaml.safe_dump({"checkpoints": {"cp1": {"judge": judge}}}))
        arms[arm] = str(y)
        yaml_shas[arm] = _sha(y)
    matrix = {"protocol": "dispatch_surface_rev1", "layer": "primary",
              "gate_type": "always_search", "arms": arms, "arm_yaml_sha256": yaml_shas,
              "core_arms": core, "descriptive_arms": [],
              "artifact_paths": {ARM_SV: str(pathlib.Path(sv_path).resolve()),
                                   ARM_S0: str(pathlib.Path(s0_path).resolve())},
              "artifact_sha256": {ARM_SV: _sha(sv_path), ARM_S0: _sha(s0_path)},
              "fit_record_paths": {"sv": str(sv_record_path.resolve()),
                                     "s0": str(s0_record_path.resolve())},
              "fit_record_sha256": {"sv": _sha(sv_record_path),
                                      "s0": _sha(s0_record_path)},
              "library_sha256": load_surface_artifact(sv_path).retrieval_contract[
                  "library_sha256"
              ]}
    matrix_path = tmp_path / "arm_matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True))
    if tamper_arm_yaml:
        # Drift an arm yaml AFTER the matrix froze its digest.
        y = tmp_path / f"{ARM_SV}.yaml"
        y.write_text(y.read_text() + "# drifted\n")
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"suite": "libero_spatial", "quota": {"test": 30},
                                      "assignment": {
        str(t): {"test": list(range(20, 50))} for t in range(10)}}))

    contract = load_surface_artifact(sv_path).retrieval_contract
    entry = {
        "protocol": "dispatch_surface_rev1", "layer": "primary",
        "suite": "libero_spatial",
        "library_sha256": contract["library_sha256"],
        "policy_fingerprint": launch_fp or contract["policy_fingerprint"],
        "contract_binding": {
            "policy_fingerprint": launch_fp or contract["policy_fingerprint"],
            "h_exec": 5,
        },
        "core_arms": core,
        "descriptive_arms": [],
        "trials_per_task": launch_trials,
        "replan_steps": 5,
        "env_seed": 7,
        "aprime_content_sha256": None if drop_aprime else "APRIME",
        "split_manifest_sha256": _sha(split_path),
        "pool": {
            "suite": "libero_spatial", "total_inits": 300,
            "rollup_sha256": None if drop_aprime else "APRIME",
            "split_manifest_sha256": _sha(split_path),
            "state_content_sha256": {},
        },
        "arm_matrix_sha256": _sha(matrix_path),
        "frozen_yaml_sha256": dict(yaml_shas),
        "artifact_sha256": matrix["artifact_sha256"],
        "fit_record_sha256": matrix["fit_record_sha256"],
        "executed_arms": sorted(core),
        "executed_yaml_sha256": dict(yaml_shas),   # frozen BEFORE any drift
        "run_id": RUN,
    }
    ledger = {"schema_version": 1 if bad_ledger else 2, "launches": [entry]}
    if second_launch is not None:
        ledger["launches"].append({**entry, **second_launch})
    (tmp_path / "launch.json").write_text(json.dumps(ledger))

    args = type("Args", (), {})()
    args.fit_record = str(sv_record_path)
    args.launch_manifest = str(tmp_path / "launch.json")
    args.arm_matrix = str(matrix_path)
    args.split_manifest = str(split_path)
    args.trials = 30
    return args, matrix


def test_check_discipline_accepts_consistent_fixture(tmp_path):
    args, matrix = _discipline_fixture(tmp_path)
    out = check_discipline(args, matrix)
    assert out["delta_star"] == 0.5
    assert len(out["arm_matrix_sha256"]) == 64
    assert set(out["arm_yaml_sha256"]) == set(CORE_T_ARMS) | {ARM_SV, ARM_S0}
    assert "analytic" in out["cost_model"]


def test_check_discipline_rejects_artifact_content_drift_after_launch(tmp_path):
    args, matrix = _discipline_fixture(tmp_path)
    pathlib.Path(matrix["artifact_paths"][ARM_SV]).write_bytes(b"replaced artifact")
    with pytest.raises(SystemExit, match="content-drifted"):
        check_discipline(args, matrix)


@pytest.mark.parametrize("break_it", [
    "s0_delta", "s0_inputs", "launch_trials", "arm_yaml_drift",
    "launch_fp", "drop_aprime", "bad_ledger", "ledger_disagrees",
    "duplicate_run_id",
])
def test_check_discipline_refusals(tmp_path, break_it):
    kwargs = {}
    if break_it == "s0_delta":
        kwargs["s0_delta"] = 0.4            # S0 not sharing the frozen delta
    elif break_it == "s0_inputs":
        kwargs["s0_inputs"] = {"table": "OTHER"}   # same delta, different data
    elif break_it == "launch_trials":
        kwargs["launch_trials"] = 25        # frozen 30/task quota broken
    elif break_it == "arm_yaml_drift":
        kwargs["tamper_arm_yaml"] = True    # self-reported sha != actual file
    elif break_it == "launch_fp":
        kwargs["launch_fp"] = "OTHER"       # server ran a different policy
    elif break_it == "bad_ledger":
        kwargs["bad_ledger"] = True         # pre-ledger manifest, unbindable
    elif break_it == "ledger_disagrees":
        # A resume that ran a different arm matrix (G2 B4).
        kwargs["second_launch"] = {"arm_matrix_sha256": "f" * 64, "run_id": "run2"}
    elif break_it == "duplicate_run_id":
        kwargs["second_launch"] = {}        # same run id twice
    else:
        kwargs["drop_aprime"] = True        # no A' content attestation
    args, matrix = _discipline_fixture(tmp_path, **kwargs)
    with pytest.raises(SystemExit):
        check_discipline(args, matrix)


def test_check_discipline_needs_no_cost_bench_inputs(tmp_path):
    """The cost bench is gone: no power record, no cost manifests, no blocks."""
    args, _matrix = _discipline_fixture(tmp_path)
    for gone in ("cost_dir", "power_record", "blocks"):
        assert not hasattr(args, gone)


def test_check_discipline_returns_the_launch_run_ids(tmp_path):
    """B4: the adjudication must know which runs it is allowed to bind to."""
    args, matrix = _discipline_fixture(tmp_path)
    out = check_discipline(args, matrix)
    assert out["launch_run_ids"] == [RUN]
    assert len(out["split_manifest_sha256"]) == 64


def test_check_discipline_accepts_strict_subset_resume(tmp_path):
    """A resume freezes the same experiment but may execute one unfinished arm."""
    args, matrix = _discipline_fixture(tmp_path)
    ledger_path = pathlib.Path(args.launch_manifest)
    ledger = json.loads(ledger_path.read_text())
    resumed = {**ledger["launches"][0], "run_id": "run-resume"}
    resumed["executed_arms"] = [ARM_SV]
    resumed["executed_yaml_sha256"] = {
        ARM_SV: resumed["frozen_yaml_sha256"][ARM_SV]
    }
    ledger["launches"].append(resumed)
    ledger_path.write_text(json.dumps(ledger))
    out = check_discipline(args, matrix)
    assert out["executed_arms_by_run"]["run-resume"] == [ARM_SV]


def test_core_arm_roster_is_five():
    assert len(set(CORE_T_ARMS) | {ARM_SV, ARM_S0}) == 5


# ------------------------------------------------------------------
# End-to-end
# ------------------------------------------------------------------


def _full_experiment(tmp_path, *, sv_advantage=True, layer="primary"):
    """A complete 5-arm x 10-task x 30-init experiment on disk.

    Rows follow the REAL producer contract: verdict rows carry the official
    init index and the driver's (attempt, accepted, run_id) stamps, and each
    episode contributes one client_timing row in episode_runner's shape --
    which has no orig_init_state_idx.
    """
    import yaml as _yaml

    from openpi.cache.components.surface_judge import load_surface_artifact
    core = list(CORE_T_ARMS) + ([ARM_S0, ARM_SV] if layer == "primary" else [ARM_SV])
    sv_path, s0_path, sv_record_path, s0_record_path = _formal_fit_outputs(tmp_path)
    arts = {ARM_SV: sv_path, ARM_S0: s0_path}
    arms, shas = {}, {}
    for arm in core:
        y = tmp_path / f"{arm}.yaml"
        judge = {"surface_artifact_path": arts[arm]} if arm in arts else {"type": "threshold"}
        y.write_text(_yaml.safe_dump({"checkpoints": {"cp1": {"judge": judge}}}))
        arms[arm] = str(y)
        shas[arm] = _sha(y)
    contract = load_surface_artifact(arts[ARM_SV]).retrieval_contract
    matrix = {
        "protocol": "dispatch_surface_rev1", "layer": layer,
        "gate_type": "always_search" if layer == "primary" else "score_hysteresis",
        "arms": arms, "arm_yaml_sha256": shas,
        "core_arms": core, "descriptive_arms": [],
        "artifact_paths": {a: str(pathlib.Path(arts[a]).resolve()) for a in core if a in arts},
        "artifact_sha256": {a: _sha(arts[a]) for a in core if a in arts},
        "fit_record_paths": {"sv": str(sv_record_path.resolve()),
                             "s0": str(s0_record_path.resolve())},
        "fit_record_sha256": {"sv": _sha(sv_record_path), "s0": _sha(s0_record_path)},
        "library_sha256": contract["library_sha256"],
    }
    matrix_path = tmp_path / "arm_matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True))
    # A' holds official inits 20..49 of every task; subset i -> official 20+i.
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"suite": "libero_spatial", "quota": {"test": 30},
                                      "assignment": {
        str(t): {"test": list(range(20, 50))} for t in range(10)}}))

    (tmp_path / "launch.json").write_text(json.dumps({
        "schema_version": 2,
        "launches": [{
            "protocol": "dispatch_surface_rev1", "layer": layer,
            "suite": "libero_spatial",
            "library_sha256": contract["library_sha256"],
            "policy_fingerprint": contract["policy_fingerprint"],
            "contract_binding": {"policy_fingerprint": contract["policy_fingerprint"],
                                 "h_exec": 5},
            "core_arms": core, "descriptive_arms": [],
            "trials_per_task": 30, "replan_steps": 5, "env_seed": 7,
            "aprime_content_sha256": "APRIME",
            "split_manifest_sha256": _sha(split_path),
            "pool": {"suite": "libero_spatial", "total_inits": 300,
                     "rollup_sha256": "APRIME",
                     "split_manifest_sha256": _sha(split_path),
                     "state_content_sha256": {}},
            "arm_matrix_sha256": _sha(matrix_path),
            "frozen_yaml_sha256": dict(shas),
            "artifact_sha256": matrix["artifact_sha256"],
            "fit_record_sha256": matrix["fit_record_sha256"],
            "executed_arms": sorted(core),
            "executed_yaml_sha256": dict(shas),
            "run_id": RUN,
        }],
    }))

    sr_rate = {CORE_T_ARMS[0]: 0.50, CORE_T_ARMS[1]: 0.55, CORE_T_ARMS[2]: 0.60,
               ARM_S0: 0.62, ARM_SV: 0.70 if sv_advantage else 0.40}
    # (FULL, WARM) shares. S0 and SV accept nearly the same fraction of steps;
    # what v buys SV is moving accepted steps from the WARM tier to the much
    # cheaper FULL tier, which is the structural advantage Gate 2 tests. The
    # thresholds sit on the frontier between them.
    tier_share = {CORE_T_ARMS[0]: (0.18, 0.12), CORE_T_ARMS[1]: (0.27, 0.18),
                  CORE_T_ARMS[2]: (0.36, 0.24),
                  ARM_S0: (0.20, 0.52), ARM_SV: (0.40, 0.33)}
    jrows, srows = [], []
    for arm in core:
        full_f, warm_f = tier_share[arm]
        for t in range(10):
            for i in range(30):
                idx = t * 30 + i
                ok = (idx % 100) < int(sr_rate[arm] * 100)
                jrows.append(_jrow(arm, t, i, ok))
                n_dec = 20 + (idx % 5)          # unequal episode lengths
                n_full = int(round(full_f * n_dec))
                n_warm = int(round(warm_f * n_dec))
                verdicts = (["FULL_HIT"] * n_full + ["WARM_START"] * n_warm
                            + ["MISS"] * (n_dec - n_full - n_warm))
                srows.extend(_episode(arm, t, i, 20 + i, verdicts, success=ok))
    journal = tmp_path / "journal.jsonl"
    journal.write_text("".join(json.dumps(r) + "\n" for r in jrows))
    per_step = tmp_path / "per_step.jsonl"
    per_step.write_text("".join(json.dumps(r) + "\n" for r in srows))

    args = type("Args", (), {})()
    args.journal, args.per_step = str(journal), str(per_step)
    args.arm_matrix = str(matrix_path)
    args.fit_record = str(sv_record_path)
    args.launch_manifest = str(tmp_path / "launch.json")
    args.split_manifest = str(split_path)
    args.trials, args.seed = 30, 20260827
    args.out = str(tmp_path / "verdict.json")
    return args


def _run(args, monkeypatch):
    import sys

    import exp.dispatch_surface.analysis.analyze_precheck as mod

    monkeypatch.setattr(mod, "B_REPLICATES", 300)   # keep the test quick
    monkeypatch.setattr(sys, "argv", [
        "analyze_precheck", "--journal", args.journal, "--per-step", args.per_step,
        "--arm-matrix", args.arm_matrix, "--fit-record", args.fit_record,
        "--launch-manifest", args.launch_manifest,
        "--split-manifest", args.split_manifest, "--trials", str(args.trials),
        "--seed", str(args.seed), "--out", args.out,
    ])
    mod.main()
    return json.loads(pathlib.Path(args.out).read_text())


def test_end_to_end_produces_a_verdict_from_one_precheck_run(tmp_path, monkeypatch):
    """The whole adjudication runs off journal + per_step, nothing else."""
    args = _full_experiment(tmp_path)
    out = _run(args, monkeypatch)
    # SV beats S0 on SR (0.70 vs 0.62) AND is materially cheaper: it accepts a
    # similar share of steps but routes far more of them to FULL instead of
    # WARM. Both components of the intersection-union test should clear.
    assert out["verdict"] == "surface_wins_v_confirmed"
    assert out["gate1"]["pass"] and out["gate2"]["pass"]
    # Cost came out of the verdict counts, in the right ballpark.
    for arm, est in out["point_estimates"].items():
        assert STAGE1_MS <= est["cost_ms_per_decision"] <= FULL_MS, arm
    # The cheapest arm must be the one with the highest FULL_HIT share.
    costs = {a: e["cost_ms_per_decision"] for a, e in out["point_estimates"].items()}
    assert min(costs, key=costs.get) == ARM_SV
    # Provenance of the cost model is recorded.
    assert out["discipline"]["cost_inputs"]["unit_cost_ms"]["stage1"] == STAGE1_MS
    assert out["discipline"]["cost_inputs"]["decisions"][ARM_SV] > 0


def test_end_to_end_demotes_the_line_when_the_surface_has_no_sr_edge(
        tmp_path, monkeypatch):
    # SV's SR (0.40) sits below every threshold arm, so the replicate clamps
    # to the argmin-SR cell and the SR floor fails however cheap SV is.
    args = _full_experiment(tmp_path, sv_advantage=False)
    out = _run(args, monkeypatch)
    assert out["verdict"] == "line_demoted"
    assert out["gate1"]["d_sr_p5"] < 0
    assert out["gate1"]["branch_shares"]["low"] == pytest.approx(1.0)
    assert "gate2" not in out          # fixed sequence stops at Gate 1


def test_secondary_layer_is_descriptive_and_never_emits_confirmatory_gates(
        tmp_path, monkeypatch):
    args = _full_experiment(tmp_path, layer="secondary")
    out = _run(args, monkeypatch)
    assert out["analysis_layer"] == "secondary"
    assert out["confirmatory"] is False
    assert out["verdict"] == "secondary_descriptive_complete"
    assert "frontier_descriptive" in out
    assert "gate1" not in out and "gate2" not in out


def test_end_to_end_rejects_run_id_that_did_not_execute_the_arm(tmp_path, monkeypatch):
    """A run_id somewhere in the ledger is insufficient; it must own that arm."""
    args = _full_experiment(tmp_path)
    launch_path = pathlib.Path(args.launch_manifest)
    ledger = json.loads(launch_path.read_text())
    first = ledger["launches"][0]
    first["executed_arms"].remove(ARM_SV)
    first["executed_yaml_sha256"].pop(ARM_SV)
    resumed = {**first, "run_id": "run-resume"}
    resumed["executed_arms"] = [ARM_SV]
    resumed["executed_yaml_sha256"] = {
        ARM_SV: resumed["frozen_yaml_sha256"][ARM_SV]
    }
    ledger["launches"].append(resumed)
    launch_path.write_text(json.dumps(ledger))
    with pytest.raises(SystemExit, match="did not execute that arm"):
        _run(args, monkeypatch)


# ---------------- Rev 1: the retired Gate 2 encoding is refused ----------------

def test_retired_gate2_fields_are_refused():
    from exp.dispatch_surface.analysis.analyze_precheck import reject_retired_gate2

    for field in ("gate2_compute_slack", "dc_upper_slack"):
        with pytest.raises(SystemExit, match="retired Gate 2"):
            reject_retired_gate2({field: 0.05}, "fit record")


def test_a_positive_cost_gate_is_refused_as_the_retired_rule():
    from exp.dispatch_surface.analysis.analyze_precheck import reject_retired_gate2

    with pytest.raises(SystemExit, match="cost-increase ceiling"):
        reject_retired_gate2({"gate2": {"cost_gate": 0.05}}, "fit record")


def test_a_rev1_record_passes_the_retirement_check():
    from exp.dispatch_surface.analysis.analyze_precheck import (
        GATE2_COMPUTE_GATE, reject_retired_gate2,
    )

    reject_retired_gate2({"gate2": {"cost_gate": GATE2_COMPUTE_GATE}}, "fit record")


# ---------------- Rev 1 verdict wording (G2R1-B8) ----------------

@pytest.mark.parametrize("mu_sr,mu_c,verdict", [
    (0.02, -0.12, "surface_wins_v_confirmed"),
    (0.02, 0.02, "surface_v_sr_noninferior_cost_saving_unconfirmed"),
    (-0.03, -0.12, "surface_v_cost_saving_sr_noninferiority_unconfirmed"),
    (-0.03, 0.02, "surface_wins_v_unconfirmed"),
])
def test_gate2_verdict_ids_cover_all_four_corners(mu_sr, mu_c, verdict, tmp_path, monkeypatch):
    """Each corner gets its own id; none of them says "gain"."""
    import re

    src = pathlib.Path("exp/dispatch_surface/analysis/analyze_precheck.py").read_text()
    block = src.split('if g1["pass"]:', 1)[1].split("pathlib.Path(args.out)", 1)[0]
    assert verdict in block
    # Only what the analyzer EMITS is checked; the comments above the branch
    # deliberately quote the retired phrases in order to forbid them.
    emitted = "\n".join(
        ln for ln in block.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "SR gain confirmed" not in emitted
    assert "cost non-inferiority" not in emitted
    assert not re.search(r'verdict"\] = "[^"]*sr_gain', emitted)


def test_verdict_carries_the_suite_for_the_cross_suite_finalizer(tmp_path, monkeypatch):
    """Without this the finalizer refuses every real verdict."""
    from exp.dispatch_surface.analysis.finalize_cross_suite import check_one

    args = _full_experiment(tmp_path)
    out = _run(args, monkeypatch)
    assert out["suite"] == "libero_spatial"
    # And the whole verdict is acceptable to the finalizer's validator.
    assert check_one(out, "verdict.json") == "libero_spatial"


def test_check_discipline_rejects_a_swapped_fit_record_at_the_same_delta(tmp_path):
    """A different fit record that happens to carry the same delta* must still
    be refused: delta alone does not identify the fit that produced the arms."""
    args, matrix = _discipline_fixture(tmp_path)
    record = json.loads(pathlib.Path(args.fit_record).read_text())
    # Same frozen delta, different provenance.
    record["dev_membership_sha256"] = "0" * 64
    record["swapped"] = True
    pathlib.Path(args.fit_record).write_text(json.dumps(record, sort_keys=True))
    with pytest.raises(SystemExit):
        check_discipline(args, matrix)


def test_check_discipline_rejects_a_conformal_artifact_in_place_of_the_empirical_one(tmp_path):
    """Rev 1 claims no certificate; a conformal artifact at the same path would
    be reported as one."""
    from openpi.cache.components.surface_judge import (
        CERTIFICATION_CONFORMAL, load_surface_artifact, save_surface_artifact,
    )

    args, matrix = _discipline_fixture(tmp_path)
    path = matrix["artifact_paths"][ARM_SV]
    art = load_surface_artifact(path)
    swapped = dataclasses.replace(
        art, certification_mode=CERTIFICATION_CONFORMAL,
        conformal_c=0.01, n_calibration_episodes=100,
    )
    save_surface_artifact(swapped, path)
    with pytest.raises(SystemExit):
        check_discipline(args, matrix)
