"""Tests for E2: exposure handling, the task-key gate and the CI verdicts.

Two failure modes are asserted against explicitly, because both would have
produced a confident but wrong conclusion:
  * length confounding -- successful episodes stop on ``done`` while failures
    run to the cap, so "at least one deviation" rises with exposure alone;
  * reading a non-rejection as equivalence.
"""

from __future__ import annotations

import numpy as np
import pytest

from exp.markov_sufficiency import DELTA_E2, K_WINDOW, _stats, e2_winner_forensics as e2


def _episode(task_id, success, n_cycle, dev):
    return {"task_id": task_id, "success": success, "n_cycle": n_cycle, "dev_in_window": dev, "deviations": int(dev)}


def _row(cycle, success, wrong_phase, ep=0, task=0):
    return {
        "yaml_id": "y",
        "task_id": task,
        "subset_init_state_idx": 0,
        "episode_id": ep,
        "attempt": 0,
        "cycle": cycle,
        "success": success,
        "wrong_phase": wrong_phase,
        "winner_task_key": "t",
    }


# ------------------------------------------------------------------
# Exposure window
# ------------------------------------------------------------------


def test_short_episodes_are_excluded_and_reported():
    k = K_WINDOW["libero_spatial"]
    rows = [_row(c, True, False, ep=0) for c in range(k - 5)]  # too short
    rows += [_row(c, False, False, ep=1) for c in range(k + 3)]
    kept, all_eps, exposure = e2.episode_table(rows, "libero_spatial")
    assert exposure["n_dropped_short"] == 1
    assert exposure["dropped_success_fraction"] == 1.0
    assert len(kept) == 1


def test_only_deviations_inside_the_window_count():
    k = K_WINDOW["libero_spatial"]
    rows = [_row(c, False, wrong_phase=(c == k + 2), ep=0) for c in range(k + 5)]
    kept, _all, _ = e2.episode_table(rows, "libero_spatial")
    # The deviation happened after the equal-exposure window, so Y_e^K is False
    # even though the episode did deviate.
    assert kept[0]["dev_in_window"] is False
    assert kept[0]["deviations"] == 1


def test_equal_exposure_removes_pure_length_confounding():
    """With an identical per-cycle deviation rate, the window must not favour failures."""
    k = K_WINDOW["libero_spatial"]
    rows = []
    for ep in range(20):  # successes: short episodes
        rows += [_row(c, True, wrong_phase=(c == 3), ep=ep) for c in range(k + 1)]
    for ep in range(20, 40):  # failures: long episodes, same in-window pattern
        rows += [_row(c, False, wrong_phase=(c == 3), ep=ep) for c in range(3 * k)]
    kept, all_eps, exposure = e2.episode_table(rows, "libero_spatial")
    rates = {
        s: np.mean([e["dev_in_window"] for e in kept if e["success"] is s])
        for s in (True, False)
    }
    assert rates[True] == rates[False] == 1.0
    assert exposure["cycle_median_failure"] > exposure["cycle_median_success"]


# ------------------------------------------------------------------
# Task-key data-integrity gate
# ------------------------------------------------------------------


def test_task_gate_passes_when_winners_match():
    rows = [{"task_id": 0, "winner_task_key": "pick up the bowl"}]
    report = e2.apply_task_gate(rows, {0: "pick up the bowl"})
    assert report["passed"] is True
    assert rows[0]["wrong_task"] is False


def test_task_gate_flags_mismatch_as_integrity_failure():
    rows = [{"task_id": 0, "winner_task_key": "other task"}]
    report = e2.apply_task_gate(rows, {0: "pick up the bowl"})
    assert report["passed"] is False
    assert report["n_wrong_task"] == 1


def test_task_gate_strips_whitespace_but_is_case_sensitive():
    rows = [{"task_id": 0, "winner_task_key": "  pick up  "}, {"task_id": 1, "winner_task_key": "PICK UP"}]
    report = e2.apply_task_gate(rows, {0: "pick up", 1: "pick up"})
    assert rows[0]["wrong_task"] is False
    assert rows[1]["wrong_task"] is True
    assert report["n_wrong_task"] == 1


# ------------------------------------------------------------------
# Verdicts (never read a non-rejection as equivalence)
# ------------------------------------------------------------------


def _ci(est, low, high):
    return _stats.IntervalResult(estimate=est, low=low, high=high, level=0.95, n_resamples=10)


def test_verdict_aliasing_requires_positive_lower_bound_and_effect():
    assert e2.verdict(_ci(0.15, 0.05, 0.25), _ci(0.5, 0.4, 0.6)) == "aliasing"
    # Same point estimate but the interval touches zero -> not enough evidence.
    assert e2.verdict(_ci(0.15, -0.01, 0.30), _ci(0.5, 0.4, 0.6)) == "inconclusive"


def test_verdict_coverage_requires_equivalence_bound_not_just_non_significance():
    # Interval entirely below the practical bound and alignment CI above 70%.
    assert e2.verdict(_ci(0.01, -0.02, 0.05), _ci(0.9, 0.85, 0.95)) == "library_coverage"
    # Wide interval that still permits a >= delta effect must stay inconclusive,
    # even though it "is not significant".
    assert e2.verdict(_ci(0.01, -0.09, 0.12), _ci(0.9, 0.85, 0.95)) == "inconclusive"
    # Alignment point estimate above 70% but its lower bound below it.
    assert e2.verdict(_ci(0.01, -0.02, 0.05), _ci(0.75, 0.60, 0.90)) == "inconclusive"


def test_delta_bound_is_the_preregistered_ten_points():
    assert DELTA_E2 == pytest.approx(0.10)


# ------------------------------------------------------------------
# End-to-end analysis shape
# ------------------------------------------------------------------


def test_analyse_returns_intervals_but_defers_the_verdict():
    """Single-suite output is descriptive; the family owns the verdict."""
    episodes = [_episode(t, True, 40, False) for t in range(5) for _ in range(10)]
    episodes += [_episode(t, False, 40, True) for t in range(5) for _ in range(10)]
    result = e2.analyse(episodes, episodes, seed=0)
    assert result["rate_diff"] == pytest.approx(1.0)
    assert result["descriptive_only"] is True
    assert "verdict" not in result
    assert result["rate_diff_ci"][0] <= result["rate_diff"] <= result["rate_diff_ci"][1]

    family = e2.analyse_family({"libero_spatial": (episodes, episodes), "libero_10": (episodes, episodes)})
    assert all("verdict" in c for c in family["cells"].values())


# ------------------------------------------------------------------
# Secondary estimand and fail-closed gates (G2 round 1)
# ------------------------------------------------------------------


def test_secondary_covers_every_episode_including_short_ones():
    k = K_WINDOW["libero_spatial"]
    rows = [_row(c, True, False, ep=0) for c in range(k - 5)]  # shorter than K
    rows += [_row(c, False, True, ep=1) for c in range(k + 3)]
    kept, all_eps, _ = e2.episode_table(rows, "libero_spatial")
    # The K window is a property of the *primary* estimand only.
    assert len(kept) == 1
    assert len(all_eps) == 2


def test_secondary_glm_uses_length_as_denominator_not_offset():
    eps = [{"task_id": 0, "success": True, "n_cycle": 40, "dev_in_window": False, "deviations": 2} for _ in range(15)]
    eps += [{"task_id": 0, "success": False, "n_cycle": 40, "dev_in_window": True, "deviations": 20} for _ in range(15)]
    out = e2.secondary_quasi_binomial(eps)
    assert out["n"] == 30
    # Failures deviate far more often, so the success coefficient is negative.
    assert out["coef_success"] < 0
    assert out["ci"][0] < out["coef_success"] < out["ci"][1]


def test_task_gate_fails_closed_on_empty_input():
    report = e2.apply_task_gate([], {0: "t"})
    assert report["passed"] is False, "zero rows checked proves nothing about provenance"


def test_task_gate_fails_closed_on_unmapped_task_id():
    rows = [{"task_id": 7, "winner_task_key": "whatever"}]
    report = e2.apply_task_gate(rows, {0: "t"})
    assert report["passed"] is False
    assert report["unmapped_task_ids"] == [7]


def test_family_verdict_requires_holm_agreement():
    # Same data in both suites: a large, consistent difference.
    def make(dev_fail, dev_succ):
        out = [{"task_id": t, "success": False, "n_cycle": 40, "dev_in_window": dev_fail, "deviations": 5}
               for t in range(5) for _ in range(20)]
        out += [{"task_id": t, "success": True, "n_cycle": 40, "dev_in_window": dev_succ, "deviations": 0}
                for t in range(5) for _ in range(20)]
        return out

    strong = make(True, False)
    family = e2.analyse_family({"libero_spatial": (strong, strong), "libero_10": (strong, strong)})
    for cell in family["cells"].values():
        assert cell["holm_level"] >= 0.95  # adjusted intervals are never looser
        assert "cmh_rejected_holm" in cell
        assert cell["verdict"] in ("aliasing", "inconclusive", "library_coverage")


def test_family_downgrades_aliasing_when_holm_does_not_reject():
    # One episode per group: the interval may look positive but CMH cannot reject.
    tiny = [
        {"task_id": 0, "success": False, "n_cycle": 40, "dev_in_window": True, "deviations": 1},
        {"task_id": 0, "success": True, "n_cycle": 40, "dev_in_window": False, "deviations": 0},
    ]
    family = e2.analyse_family({"libero_spatial": (tiny, tiny), "libero_10": (tiny, tiny)})
    for cell in family["cells"].values():
        assert cell["verdict"] != "aliasing"


def test_single_suite_analysis_carries_no_verdict():
    """Only the two-suite Holm family may produce a scientific call."""
    eps = [{"task_id": 0, "success": i % 2 == 0, "n_cycle": 40, "dev_in_window": i % 3 == 0, "deviations": 1}
           for i in range(20)]
    out = e2.analyse(eps, eps)
    assert "verdict" not in out
    assert out["descriptive_only"] is True


def test_family_blocks_a_coverage_call_when_holm_still_rejects():
    strong = [{"task_id": t, "success": False, "n_cycle": 40, "dev_in_window": True, "deviations": 5}
              for t in range(5) for _ in range(20)]
    strong += [{"task_id": t, "success": True, "n_cycle": 40, "dev_in_window": False, "deviations": 0}
               for t in range(5) for _ in range(20)]
    family = e2.analyse_family({"libero_spatial": (strong, strong), "libero_10": (strong, strong)})
    for cell in family["cells"].values():
        if cell["cmh_rejected_holm"]:
            assert cell["verdict"] != "library_coverage"


def test_join_gate_is_fail_closed_for_every_suite():
    """The same gate must guard suite B; NaN and low rates both stop the run."""
    with pytest.raises(SystemExit, match="0.999"):
        e2.assert_join_gate({"join_rate": float("nan")}, "libero_10")
    with pytest.raises(SystemExit, match="0.999"):
        e2.assert_join_gate({"join_rate": 0.5}, "libero_10")
    e2.assert_join_gate({"join_rate": 1.0}, "libero_10")  # must not raise
