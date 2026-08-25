"""X15 U4/U6 — pool isolation, determinism, and the pre-registered estimators.

The statistics here are the ones the plan froze before any data exists, so the
tests check the properties that make them trustworthy rather than specific
numbers: fitting slices never touch the test pool, retraining twice gives the
same artifact, clustered resampling does not overstate precision, and an
un-bracketed curve refuses to produce a headline.

Key dependencies: ``exp.rl_router.train_risk_model``,
``exp.rl_router.analysis.cluster_stats``.
"""

from __future__ import annotations

import math

import pytest
import torch

from exp.rl_router.analysis.cluster_stats import (
    cluster_bootstrap,
    iso_sr_share,
    iso_sr_with_ci,
    isotonic_sr,
    mcnemar_power,
    mean,
)
from exp.rl_router.train_risk_model import (
    assert_pools_are_disjoint,
    fit_isotonic,
    select_delta,
    split_rows,
    train,
)
from openpi.cache.components.risk_features import FEATURE_DIM, feature_schema_digest
from openpi.cache.components.risk_model import RiskModel

# ------------------------------------------------------------------
# Pool isolation
# ------------------------------------------------------------------


def test_clean_ledger_is_accepted() -> None:
    assert_pools_are_disjoint(
        {"gradient": [1, 2], "delta": [3], "cal": [4], "test": [5, 6]}
    )


def test_a_fitting_slice_touching_the_test_pool_is_refused() -> None:
    """The single failure that would turn the held-out number in-sample."""
    with pytest.raises(ValueError, match="must never be fitted on"):
        assert_pools_are_disjoint(
            {"gradient": [1, 5], "delta": [3], "cal": [4], "test": [5]}
        )


def test_overlapping_fitting_slices_are_refused() -> None:
    """Calibrating on gradient data makes the calibration optimistic."""
    with pytest.raises(ValueError, match="share"):
        assert_pools_are_disjoint(
            {"gradient": [1, 2], "delta": [2], "cal": [4], "test": []}
        )


def test_rows_route_by_init_and_unassigned_rows_are_dropped() -> None:
    ledger = {"gradient": ["a"], "delta": ["b"], "cal": ["c"], "test": ["d"]}
    rows = [{"init_id": k} for k in ("a", "b", "c", "d", "unknown")]
    out = split_rows(rows, ledger)

    assert [r["init_id"] for r in out["gradient"]] == ["a"]
    assert [r["init_id"] for r in out["cal"]] == ["c"]
    # Test-pool and unknown rows never enter a fitting slice.
    assert all("d" != r["init_id"] for s in out.values() for r in s)


# ------------------------------------------------------------------
# Threshold and calibration
# ------------------------------------------------------------------


def test_delta_separates_failures_from_successes() -> None:
    u = torch.tensor([0.1, 0.2, 0.9, 1.0])
    success = torch.tensor([True, True, False, False])
    delta = select_delta(u, success)
    assert 0.2 <= delta < 0.9


def test_delta_on_a_degenerate_slice_falls_back_to_the_median() -> None:
    u = torch.tensor([0.1, 0.5, 0.9])
    success = torch.tensor([True, True, True])
    assert select_delta(u, success) == pytest.approx(0.5)


def test_isotonic_fit_is_monotone() -> None:
    u_hat = torch.tensor([0.0, 1.0, 2.0, 3.0])
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    iso = fit_isotonic(u_hat, target)
    values = [iso(v) for v in (0.0, 1.0, 2.0, 3.0)]
    assert values == sorted(values)


# ------------------------------------------------------------------
# Training determinism
# ------------------------------------------------------------------


def _slice(n: int, seed: int) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        x = torch.rand(FEATURE_DIM, generator=g)
        rows.append({
            "init_id": f"i{seed}_{i}",
            "features": x.tolist(),
            "u": float(x.mean()),
            "success": bool(i % 2),
        })
    return rows


def test_training_is_reproducible_and_emits_a_loadable_artifact(tmp_path) -> None:
    slices = {"gradient": _slice(24, 1), "delta": _slice(8, 2), "cal": _slice(8, 3)}

    a = train(slices, epochs=5, hidden=8, seed=0)
    b = train(slices, epochs=5, hidden=8, seed=0)
    x = torch.rand(FEATURE_DIM, generator=torch.Generator().manual_seed(9))
    assert a.risk(x) == pytest.approx(b.risk(x))
    assert a.delta == pytest.approx(b.delta)

    path = str(tmp_path / "risk.pt")
    a.save(path)
    loaded = RiskModel.load(path, expected_schema_sha=feature_schema_digest())
    assert loaded.risk(x) == pytest.approx(a.risk(x))


def test_wrong_feature_width_is_refused() -> None:
    slices = {
        "gradient": [{"init_id": "a", "features": [0.0] * 3, "u": 0.1, "success": True}],
        "delta": _slice(4, 2),
        "cal": _slice(4, 3),
    }
    with pytest.raises(ValueError, match="dims"):
        train(slices, epochs=1, hidden=4, seed=0)


# ------------------------------------------------------------------
# Cluster bootstrap
# ------------------------------------------------------------------


def test_cluster_bootstrap_recovers_the_point_estimate() -> None:
    clusters = [[1.0] * 5, [0.0] * 5, [1.0] * 5, [0.0] * 5]
    out = cluster_bootstrap(clusters, mean, n_boot=500, seed=0)
    assert out["estimate"] == pytest.approx(0.5)
    assert out["ci_low"] <= 0.5 <= out["ci_high"]
    assert out["n_clusters"] == 4 and out["n_obs"] == 20


def test_correlated_seeds_widen_the_interval_versus_pretending_independence() -> None:
    """The whole reason for clustering: 10 identical seeds inside an init carry
    one init's worth of information, not ten."""
    correlated = [[1.0] * 10 if i % 2 else [0.0] * 10 for i in range(10)]
    as_if_independent = [[v] for c in correlated for v in c]

    clustered = cluster_bootstrap(correlated, mean, n_boot=800, seed=0)
    naive = cluster_bootstrap(as_if_independent, mean, n_boot=800, seed=0)
    width = lambda r: r["ci_high"] - r["ci_low"]  # noqa: E731
    assert width(clustered) > width(naive)


def test_empty_input_is_refused() -> None:
    with pytest.raises(ValueError, match="no non-empty clusters"):
        cluster_bootstrap([], mean)


# ------------------------------------------------------------------
# iso-SR
# ------------------------------------------------------------------


def test_iso_sr_interpolates_between_bracketing_points() -> None:
    share = iso_sr_share([(0.2, 0.70), (0.5, 0.90)], target=0.80)
    assert share == pytest.approx(0.35)


def test_iso_sr_refuses_to_extrapolate() -> None:
    """Pre-registered failure rule: an un-bracketed curve yields no headline."""
    assert iso_sr_share([(0.2, 0.60), (0.5, 0.70)], target=0.80) is None
    assert iso_sr_share([(0.2, 0.85), (0.5, 0.95)], target=0.80) is None


def test_non_monotone_points_are_isotonised_before_interpolation() -> None:
    curve = isotonic_sr([(0.2, 0.70), (0.4, 0.65), (0.6, 0.90)])
    values = [y for _, y in curve]
    assert values == sorted(values)


def test_iso_sr_ci_reports_unbracketed_draws_instead_of_hiding_them() -> None:
    points = [
        (0.2, [[0.0] * 4 for _ in range(6)]),
        (0.5, [[0.0] * 4 for _ in range(6)]),
    ]
    out = iso_sr_with_ci(points, target=0.80, n_boot=50, seed=0)
    assert out["bracketed"] is False
    assert out["estimate"] is None
    assert out["unbracketed_draws"] == 50


def test_iso_sr_ci_brackets_a_real_crossing() -> None:
    low = [[0.0, 1.0, 1.0, 1.0] for _ in range(8)]     # SR ~0.75
    high = [[1.0, 1.0, 1.0, 0.0] + [1.0] * 6 for _ in range(8)]  # SR ~0.9
    out = iso_sr_with_ci([(0.2, low), (0.6, high)], target=0.80, n_boot=200, seed=0)
    assert out["bracketed"] is True
    assert 0.2 <= out["estimate"] <= 0.6
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]


# ------------------------------------------------------------------
# Power
# ------------------------------------------------------------------


def test_power_rises_with_pairs_and_with_effect() -> None:
    base = mcnemar_power(500, 0.20, 0.04)
    assert mcnemar_power(2000, 0.20, 0.04) > base
    assert mcnemar_power(500, 0.20, 0.08) > base
    assert 0.0 <= base <= 1.0


def test_high_disagreement_costs_power_at_a_fixed_absolute_effect() -> None:
    """Power scales as ``effect * sqrt(n / discordant_rate)``.

    For a fixed absolute difference in marginals, the SAME difference spread
    over more discordant pairs is a weaker per-pair imbalance and therefore
    harder to detect. This is the direction that makes the pilot estimate
    load-bearing: n alone does not determine whether 500 pairs suffice.
    """
    assert mcnemar_power(500, 0.40, 0.02) < mcnemar_power(500, 0.05, 0.02)


def test_power_collapses_when_the_arms_almost_never_disagree() -> None:
    """The other end of the trade-off, and the reason power is not monotone
    in the discordant rate.

    Lowering ``q`` concentrates a fixed effect into a sharper per-pair
    imbalance (helps) while shrinking the number of discordant pairs (hurts).
    Below roughly ten discordant pairs the second term wins outright: at
    ``q=0.005`` only ~2.5 of 500 pairs disagree, and no imbalance among them
    can clear the critical value. A design cannot be judged from ``n`` alone.
    """
    assert mcnemar_power(500, 0.005, 0.02) == pytest.approx(0.0, abs=1e-6)
    assert mcnemar_power(500, 0.05, 0.02) > 0.4


def test_degenerate_inputs_return_zero_power() -> None:
    assert mcnemar_power(0, 0.2, 0.04) == 0.0
    assert mcnemar_power(500, 0.0, 0.04) == 0.0
    assert not math.isnan(mcnemar_power(500, 1.0, 0.0))


# ------------------------------------------------------------------
# Pre-registration gates
# ------------------------------------------------------------------


def test_primary_hard_fails_on_any_unpaired_slot() -> None:
    """A warning is not enough for the primary: an unpaired slot means the arms
    did not run the same list."""
    from exp.rl_router.analysis.cluster_stats import (
        PreregistrationError,
        assert_no_unpaired_drop,
    )

    assert_no_unpaired_drop(0, 0)                      # the only acceptable case
    with pytest.raises(PreregistrationError, match="identical slot lists"):
        assert_no_unpaired_drop(1, 0)
    with pytest.raises(PreregistrationError, match="identical slot lists"):
        assert_no_unpaired_drop(0, 3)


@pytest.mark.parametrize(
    "phase,reads",
    [("p0", "test"), ("train", "cal"), ("calibrate", "test"), ("tau_grid", "a")],
)
def test_a_fitting_phase_cannot_read_an_evaluation_pool(phase, reads) -> None:
    from exp.rl_router.analysis.cluster_stats import (
        PreregistrationError,
        assert_pool_isolation,
    )

    with pytest.raises(PreregistrationError):
        assert_pool_isolation({}, phase=phase, reads=reads)


@pytest.mark.parametrize(
    "phase,reads",
    [("p0", "gradient"), ("train", "gradient"), ("calibrate", "cal"),
     ("evaluate_btest", "test"), ("evaluate_a", "a")],
)
def test_legitimate_phase_pool_pairs_are_allowed(phase, reads) -> None:
    from exp.rl_router.analysis.cluster_stats import assert_pool_isolation

    assert_pool_isolation({}, phase=phase, reads=reads)


def test_evaluation_refuses_to_run_before_p_hat_is_frozen() -> None:
    """Choosing the comparison point after seeing the evaluation pool is the
    exact failure the ledger exists to make impossible."""
    from exp.rl_router.analysis.cluster_stats import (
        PreregistrationError,
        assert_frozen_before,
    )

    with pytest.raises(PreregistrationError, match="frozen block"):
        assert_frozen_before({}, "p_hat", phase="evaluate_a")

    with pytest.raises(PreregistrationError, match="'value' and 'at'"):
        assert_frozen_before({"frozen": {"p_hat": 0.29}}, "p_hat", phase="evaluate_a")

    # A well-formed ledger needs the first-touch stamp too, so the gate can
    # compare the two times rather than merely confirm a field exists.
    assert_frozen_before(
        {
            "frozen": {"p_hat": {"value": 0.29, "at": "2026-08-22T00:00:00Z"}},
            "touched": {"a": "2026-08-22T06:00:00Z"},
        },
        "p_hat", phase="evaluate_a",
    )


# ------------------------------------------------------------------
# Trainer production contract
# ------------------------------------------------------------------


def _labelled_slice(n: int, seed: int, *, success_rate: float = 0.6) -> list[dict]:
    """Rows with a success flag, so calibration has something to condition on."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        x = torch.rand(FEATURE_DIM, generator=g)
        rows.append({
            "init_id": f"i{seed}_{i}",
            "features": x.tolist(),
            "u": float(x.mean()),
            "success": (i / n) < success_rate,
        })
    return rows


def test_trainer_actually_computes_cp_tau0() -> None:
    """The artifact field must come from the calibration slice, not be left None.

    A None here silently removes the tau grid's centre, which is the one number
    the grid is built around.
    """
    slices = {
        "gradient": _labelled_slice(24, 1),
        "delta": _labelled_slice(8, 2),
        "cal": _labelled_slice(12, 3),
    }
    model = train(slices, epochs=5, hidden=8, seed=0)

    assert model.cp_tau0 is not None
    assert 0.0 <= model.cp_tau0 <= 1.0
    assert model.seed == 0


def test_cp_tau0_is_the_quantile_of_successful_step_risk() -> None:
    """Nonconformity is risk on steps of SUCCESSFUL episodes: the level below
    which the gate would have left a working episode alone."""
    from exp.rl_router.analysis.cluster_stats import mean  # noqa: F401  (parity import)
    from exp.rl_router.train_risk_model import split_conformal_tau0

    risks = torch.tensor([0.1, 0.2, 0.3, 0.9, 0.95])
    success = torch.tensor([True, True, True, False, False])
    tau0 = split_conformal_tau0(risks, success, alpha=0.1)

    # Only the successful steps (0.1, 0.2, 0.3) inform it.
    assert 0.28 <= tau0 <= 0.30


def test_cp_tau0_is_none_when_no_successful_steps_exist() -> None:
    """No calibration basis is reported as absent, not as a fabricated number."""
    from exp.rl_router.train_risk_model import split_conformal_tau0

    risks = torch.tensor([0.5, 0.6])
    assert split_conformal_tau0(risks, torch.tensor([False, False])) is None


def test_trainer_persists_the_task_embedding_table(tmp_path) -> None:
    """The rows the head was fitted against must ride in the artifact, or the
    deployed features cannot be audited against the weights."""
    from openpi.cache.components.risk_features import (
        TASK_EMBED_DIM,
        N_TASKS,
        default_task_embedding_table,
    )

    slices = {
        "gradient": _labelled_slice(24, 1),
        "delta": _labelled_slice(8, 2),
        "cal": _labelled_slice(12, 3),
    }
    model = train(slices, epochs=3, hidden=8, seed=0)
    path = str(tmp_path / "risk.pt")
    model.save(path)
    loaded = RiskModel.load(path, expected_schema_sha=feature_schema_digest())

    assert loaded.task_embedding_table is not None
    assert tuple(loaded.task_embedding_table.shape) == (N_TASKS, TASK_EMBED_DIM)
    assert torch.allclose(loaded.task_embedding_table, default_task_embedding_table())


# ------------------------------------------------------------------
# Freeze / touch ordering (U6)
# ------------------------------------------------------------------


def test_a_parameter_cannot_be_frozen_after_the_evaluation_was_touched(tmp_path) -> None:
    """The hole an existence check leaves open: back-filling ``frozen`` after
    looking at A makes the ledger say the right thing about the wrong history."""
    from exp.rl_router.analysis.cluster_stats import (
        PreregistrationError,
        freeze_parameter,
        record_pool_touch,
    )

    ledger = str(tmp_path / "ledger.json")
    record_pool_touch(ledger, "a", at="2026-08-22T10:00:00Z")

    with pytest.raises(PreregistrationError, match="already touched"):
        freeze_parameter(ledger, "p_hat", 0.29, at="2026-08-22T11:00:00Z")


def test_freezing_before_the_touch_is_allowed_and_ordered(tmp_path) -> None:
    from exp.rl_router.analysis.cluster_stats import (
        assert_frozen_before,
        freeze_parameter,
        record_pool_touch,
    )

    ledger_path = str(tmp_path / "ledger.json")
    freeze_parameter(ledger_path, "p_hat", 0.29, at="2026-08-22T09:00:00Z")
    ledger = record_pool_touch(ledger_path, "a", at="2026-08-22T10:00:00Z")

    assert_frozen_before(ledger, "p_hat", phase="evaluate_a")


def test_a_freeze_stamped_after_the_touch_is_rejected_at_use_time(tmp_path) -> None:
    """Even a hand-edited ledger cannot pass: the check compares timestamps."""
    from exp.rl_router.analysis.cluster_stats import (
        PreregistrationError,
        assert_frozen_before,
    )

    forged = {
        "frozen": {"p_hat": {"value": 0.29, "at": "2026-08-22T12:00:00Z"}},
        "touched": {"a": "2026-08-22T10:00:00Z"},
    }
    with pytest.raises(PreregistrationError, match="at or after"):
        assert_frozen_before(forged, "p_hat", phase="evaluate_a")


def test_a_ledger_with_no_touch_stamp_is_rejected() -> None:
    """Without a first-touch record the ordering is unauditable, so the gate
    refuses rather than assuming the good case."""
    from exp.rl_router.analysis.cluster_stats import (
        PreregistrationError,
        assert_frozen_before,
    )

    only_frozen = {"frozen": {"p_hat": {"value": 0.29, "at": "2026-08-22T09:00:00Z"}}}
    with pytest.raises(PreregistrationError, match="first-touch stamp"):
        assert_frozen_before(only_frozen, "p_hat", phase="evaluate_a")


def test_a_first_touch_stamp_cannot_be_revised(tmp_path) -> None:
    """If the timestamp could be rewritten, the ordering it certifies would be
    worth nothing."""
    from exp.rl_router.analysis.cluster_stats import (
        PreregistrationError,
        record_pool_touch,
    )

    ledger = str(tmp_path / "ledger.json")
    record_pool_touch(ledger, "test", at="2026-08-22T10:00:00Z")
    with pytest.raises(PreregistrationError, match="never revised"):
        record_pool_touch(ledger, "test", at="2026-08-22T09:00:00Z")


def test_refreezing_a_parameter_is_refused(tmp_path) -> None:
    from exp.rl_router.analysis.cluster_stats import (
        PreregistrationError,
        freeze_parameter,
    )

    ledger = str(tmp_path / "ledger.json")
    freeze_parameter(ledger, "p_hat", 0.29, at="2026-08-22T09:00:00Z")
    with pytest.raises(PreregistrationError, match="already frozen"):
        freeze_parameter(ledger, "p_hat", 0.31, at="2026-08-22T09:30:00Z")
