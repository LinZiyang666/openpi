"""Tests for the E4/E5 paired rollout analysis.

Covers the parts a reviewer probe can break: the primary contrast must respect
Holm, the interaction must stay estimation-only and be bootstrapped from the
joint four-arm outcome, the discordance downgrade must fire, and E5's verdict
must not depend on the order of its branches.
"""

from __future__ import annotations

import pytest

from exp.markov_sufficiency import DELTA_E4, _stats
from exp.markov_sufficiency import e45_rollout_analysis as e45


def _arm(n_success, n_total, offset=0):
    """Deterministic arm: the first ``n_success`` shared episodes succeed."""
    return {(0, i + offset): i < n_success for i in range(n_total)}


def _four_arms(a0, a1, a2, a3, n=200):
    return {"A0": _arm(a0, n), "A1": _arm(a1, n), "A2": _arm(a2, n), "A3": _arm(a3, n)}


# ------------------------------------------------------------------
# Paired table
# ------------------------------------------------------------------


def test_paired_table_uses_shared_episodes_only():
    a = {(0, 0): True, (0, 1): False}
    b = {(0, 1): True, (0, 2): True}
    table = e45.paired_table(a, b)
    assert table["n_pairs"] == 1
    assert (table["b"], table["c"]) == (0, 1)


def test_pi_d_obs_is_the_discordance_rate():
    a = {(0, i): i < 60 for i in range(100)}
    b = {(0, i): i < 50 for i in range(100)}
    table = e45.paired_table(a, b)
    assert table["pi_d_obs"] == pytest.approx(0.10)


# ------------------------------------------------------------------
# E4
# ------------------------------------------------------------------


def test_risk_difference_uses_paired_episode_denominator():
    table = e45.paired_table(_arm(60, 100), _arm(50, 100))
    result = _stats.mcnemar_exact(table["b"], table["c"], n_pairs=table["n_pairs"])
    # 10 net discordant pairs out of 100 episodes -> 10pp, not 10/10.
    assert result.estimate == pytest.approx(0.10)


def test_e4_downgrades_when_observed_discordance_exceeds_proxy():
    # A3 vs A1 differ on 40% of episodes: far above the spatial proxy Q75 (0.12).
    arms = _four_arms(a0=100, a1=100, a2=100, a3=60, n=200)
    out = e45.analyse_e4({"libero_spatial": arms})
    cell = out["cells"]["libero_spatial"]
    assert cell["downgraded_to_estimation"] is True
    assert cell["verdict"] == "estimation_only_discordance_above_proxy"


def test_e4_equivalence_requires_the_interval_inside_the_bound():
    arms = _four_arms(a0=100, a1=100, a2=100, a3=100, n=400)
    out = e45.analyse_e4({"libero_spatial": arms})
    cell = out["cells"]["libero_spatial"]
    # Identical arms: zero difference, interval collapses inside +-delta.
    assert cell["risk_diff"] == pytest.approx(0.0)
    assert cell["verdict"] == "practically_equivalent"
    assert out["delta_equivalence"] == pytest.approx(DELTA_E4)


def test_e4_interaction_is_estimation_only_and_uses_the_joint_outcome():
    arms = _four_arms(a0=100, a1=110, a2=120, a3=150, n=300)
    out = e45.analyse_e4({"libero_spatial": arms})
    inter = out["cells"]["libero_spatial"]["interaction"]
    assert inter["p_value"] is None, "the interaction must never report a p-value"
    assert inter["ci"][0] <= inter["theta"] <= inter["ci"][1]
    assert inter["n_pairs"] == 300


def test_e4_applies_holm_across_the_two_suites():
    strong = _four_arms(a0=100, a1=100, a2=100, a3=140, n=300)
    out = e45.analyse_e4({"libero_spatial": strong, "libero_10": strong})
    levels = [c["holm_ci"] for c in out["cells"].values()]
    assert len(levels) == 2
    for cell in out["cells"].values():
        assert "holm_rejected" in cell
        # Holm-adjusted intervals are at least as wide as the descriptive ones.
        adj_width = cell["holm_ci"][1] - cell["holm_ci"][0]
        desc_width = cell["descriptive_ci_95"][1] - cell["descriptive_ci_95"][0]
        assert adj_width >= desc_width - 1e-9


def test_e4_verdict_requires_the_five_point_effect_floor():
    """Holm rejection plus a positive CI is not enough below 5pp (plan §5.4)."""
    four_pp = _stats.IntervalResult(0.04, 0.01, 0.07, 0.95, 0)
    assert e45._e4_verdict(four_pp, holm_rejected=True, downgraded=False) != "filter_improves"
    six_pp = _stats.IntervalResult(0.06, 0.02, 0.10, 0.95, 0)
    assert e45._e4_verdict(six_pp, holm_rejected=True, downgraded=False) == "filter_improves"


def test_e4_emits_the_descriptive_contrasts_with_intervals():
    arms = _four_arms(a0=100, a1=110, a2=120, a3=150, n=300)
    cell = e45.analyse_e4({"libero_spatial": arms})["cells"]["libero_spatial"]
    for key in ("reproducibility_A2_minus_A0", "filter_effect_A1_minus_A0"):
        block = cell[key]
        assert block["present"] is True
        assert block["ci"][0] <= block["risk_diff"] <= block["ci"][1]


def test_a4_has_a_paired_interval_not_just_a_point():
    arms = _four_arms(a0=100, a1=100, a2=100, a3=100, n=200)
    arms["A4"] = _arm(20, 200)
    a4 = e45.analyse_e4({"libero_spatial": arms})["cells"]["libero_spatial"]["a4_descriptive"]
    assert a4["vs_A2"]["ci"][0] <= a4["vs_A2"]["risk_diff"] <= a4["vs_A2"]["ci"][1]


def test_task_key_to_id_bridge_skips_unreported_rows():
    ranking = [
        {"task_key": "gated", "adr": None, "reported": False},
        {"task_key": "pick up", "adr": 0.3, "reported": True},
    ]
    assert e45.task_ids_from_adr_ranking(ranking, {"pick up": 2, "gated": 1}) == [2]


def test_e4_reports_a4_as_exploratory():
    arms = _four_arms(a0=100, a1=100, a2=100, a3=100, n=200)
    arms["A4"] = _arm(20, 200)
    out = e45.analyse_e4({"libero_spatial": arms})
    a4 = out["cells"]["libero_spatial"]["a4_descriptive"]
    assert a4["present"] is True
    assert "exploratory" in a4["note"]


# ------------------------------------------------------------------
# E5
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.01, 0.08, "replicated"),
        (0.01, 0.03, "positive_but_below_target"),
        (-0.02, 0.03, "not_supported"),
        (-0.02, 0.09, "inconclusive"),
    ],
)
def test_e5_verdict_partitions_the_interval_space(low, high, expected):
    """Every interval lands in exactly one bucket, independent of branch order."""
    assert e45._e5_verdict(low, high) == expected


def test_e5_verdict_handles_nan():
    assert e45._e5_verdict(float("nan"), float("nan")) == "inconclusive"


def test_e5_is_estimation_only_and_reports_intervals():
    anchor = _arm(100, 300)
    shapes = {"shape0": _arm(140, 300), "shape1": _arm(100, 300)}
    out = e45.analyse_e5(anchor, shapes)
    assert out["target_effect"] == pytest.approx(e45.E5_TARGET)
    for cell in out["shapes"].values():
        assert "p_value" not in cell, "E5 makes no significance claim"
        assert cell["ci"][0] <= cell["risk_diff"] <= cell["ci"][1]
    assert out["shapes"]["shape1"]["risk_diff"] == pytest.approx(0.0)


def test_e5_high_adr_subset_is_flagged_secondary():
    anchor = {(t, i): False for t in range(4) for i in range(50)}
    shape = {(t, i): t == 0 for t in range(4) for i in range(50)}
    out = e45.analyse_e5(anchor, {"s0": shape}, high_adr_tasks=(0,))
    subset = out["shapes"]["s0"]["high_adr_subset"]
    assert subset["n_pairs"] == 50
    assert "not in any comparison family" in subset["note"]


def test_analyse_e4_names_the_missing_arm():
    """A half-collected suite is normal mid-rollout; the error must be diagnosable."""
    full = _four_arms(a0=100, a1=100, a2=100, a3=100, n=50)
    partial = {"A0": _arm(50, 50), "A1": _arm(40, 50)}
    with pytest.raises(SystemExit, match=r"missing arm\(s\) \['A2', 'A3'\]"):
        e45.analyse_e4({"libero_spatial": full, "libero_10": partial})


def test_analyse_e4_does_not_require_the_exploratory_arm():
    """A4 is exploratory: its absence must not block the registered family."""
    arms = _four_arms(a0=100, a1=100, a2=100, a3=120, n=200)
    assert "A4" not in arms
    out = e45.analyse_e4({"libero_spatial": arms, "libero_10": arms})
    assert set(out["cells"]) == {"libero_spatial", "libero_10"}


# ------------------------------------------------------------------
# ADR ranking artifact shapes
# ------------------------------------------------------------------


def test_adr_ranking_reads_the_flat_single_suite_artifact():
    art = {"by_task": [{"task_key": "a", "adr": 0.3, "reported": True}]}
    assert e45.adr_ranking_of(art) == art["by_task"]


def test_adr_ranking_reads_the_two_suite_family_artifact():
    """A family artifact nests by_task per suite; reading it flat yields [] silently."""
    rows = [{"task_key": "a", "adr": 0.3, "reported": True}]
    art = {"mode": "family", "suites": {"libero_10": {"by_task": rows},
                                        "libero_spatial": {"by_task": []}}}
    assert e45.adr_ranking_of(art, "libero_10") == rows


def test_adr_ranking_rejects_a_missing_suite():
    art = {"suites": {"libero_spatial": {"by_task": []}}}
    with pytest.raises(SystemExit, match="pass --adr-suite"):
        e45.adr_ranking_of(art, "libero_10")


def test_adr_ranking_rejects_an_unrecognised_artifact():
    with pytest.raises(SystemExit, match="neither a top-level"):
        e45.adr_ranking_of({"something": "else"})
