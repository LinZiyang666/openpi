"""Frozen statistical protocol for TRACER Phase 6 (plan §C).

This module holds the pre-registered inference used by the exit-gate / Claim-2
analysis. It is deliberately separate from ``analyze_phase5_rollout.py`` (which
only emits the point BHR/SR/IR): the bootstrap CI and the estimability guard live
HERE so the plan's §C contract is one auditable, seeded implementation.

Frozen constants (plan §C): cluster unit = (task_id, init_state_idx); BHR is a
ratio-of-sums over resampled clusters; B=10_000 bootstrap replicates; seed 7; 95%
CI. Estimability guard N_min: a BHR verdict is refused unless each compared lane
has >= 200 FULL_HIT-labeled steps AND >= 30 distinct hit-contributing clusters.
Claim-2 uses a superiority OR equivalence rule with margin delta=0.03.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# ------------------------------------------------------------------
# Frozen constants (plan §C — do not tune at Code time)
# ------------------------------------------------------------------
B_REPLICATES = 10_000
SEED = 7
CI_LEVEL = 0.95
N_MIN_FH = 200  # min FULL_HIT-labeled steps per lane
N_MIN_CLUSTERS = 30  # min distinct hit-contributing clusters per lane
CLAIM2_EQUIV_DELTA = 0.03  # BHR equivalence margin for Claim-2


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FullHitRow:
    """One FULL_HIT-labeled step for the BHR bootstrap.

    cluster: the (task_id, init_state_idx) resample unit.
    bad:     True if this FULL_HIT is a bad hit (episode ultimately failed).
    """

    cluster: tuple
    bad: bool


def _bhr_ratio_of_sums(rows: list[FullHitRow]) -> float:
    """BHR = sum(bad) / count over a set of FULL_HIT rows (plan §C ratio-of-sums)."""
    n = len(rows)
    if n == 0:
        return float("nan")
    return sum(1 for r in rows if r.bad) / n


def _clusters(rows: list[FullHitRow]) -> dict:
    """Group rows by their resample cluster."""
    by: dict = {}
    for r in rows:
        by.setdefault(r.cluster, []).append(r)
    return by


def estimable(rows: list[FullHitRow]) -> bool:
    """§C estimability guard: enough FULL_HITs AND enough hit-contributing clusters."""
    if len(rows) < N_MIN_FH:
        return False
    return len(_clusters(rows)) >= N_MIN_CLUSTERS


# ------------------------------------------------------------------
# Clustered paired bootstrap on delta-BHR (plan §C)
# ------------------------------------------------------------------
def paired_delta_bhr_ci(
    lane_a: list[FullHitRow],
    lane_b: list[FullHitRow],
    *,
    b_replicates: int = B_REPLICATES,
    seed: int = SEED,
    ci_level: float = CI_LEVEL,
) -> dict:
    """Episode-clustered paired bootstrap of ``BHR_b - BHR_a``.

    Resamples whole clusters WITH REPLACEMENT from the shared cluster set; each
    resampled cluster contributes all its rows in both lanes; BHR is recomputed as a
    ratio-of-sums per replicate. Returns the point estimate and a percentile CI.
    ``excludes_zero`` is True iff the CI does not contain 0. Deterministic in ``seed``.
    """
    a_by = _clusters(lane_a)
    b_by = _clusters(lane_b)
    if set(a_by) != set(b_by):
        # A paired delta must not mix an all-row point estimate with a shared-only
        # resample: both lanes must span the SAME (task, init) cluster set.
        raise ValueError("mismatched cluster sets; delta-BHR must be paired on identical clusters")
    shared = sorted(a_by)
    rng = np.random.default_rng(seed)
    point = _bhr_ratio_of_sums(lane_b) - _bhr_ratio_of_sums(lane_a)
    deltas = np.empty(b_replicates, dtype=float)
    idx = np.arange(len(shared))
    for i in range(b_replicates):
        pick = rng.choice(idx, size=len(shared), replace=True)
        a_rows: list[FullHitRow] = []
        b_rows: list[FullHitRow] = []
        for j in pick:
            c = shared[j]
            a_rows.extend(a_by[c])
            b_rows.extend(b_by[c])
        deltas[i] = _bhr_ratio_of_sums(b_rows) - _bhr_ratio_of_sums(a_rows)
    alpha = 1.0 - ci_level
    deltas = np.asarray(deltas)
    lo, hi = np.quantile(deltas, [alpha / 2, 1.0 - alpha / 2])
    return {
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_clusters": len(shared),
        "pvalue": bootstrap_two_sided_p(deltas),
    }


def bootstrap_two_sided_p(deltas) -> float:
    """Two-sided bootstrap p-value that the true delta is 0 (proportion crossing 0)."""
    d = np.asarray(deltas, dtype=float)
    frac_le = float((d <= 0).mean())
    frac_ge = float((d >= 0).mean())
    return float(min(1.0, 2.0 * min(frac_le, frac_ge)))


def holm_bonferroni(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down over a family {name: p}. Returns {name: reject_bool}."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    reject: dict = {}
    still = True
    for rank, (name, p) in enumerate(ordered):
        thresh = alpha / (m - rank)
        if still and p <= thresh:
            reject[name] = True
        else:
            still = False
            reject[name] = False
    return reject


# ------------------------------------------------------------------
# Verdicts (plan §C / §C.4)
# ------------------------------------------------------------------
def framework_verdict(raw_a: list[FullHitRow], projected_b: list[FullHitRow]) -> dict:
    """A-vs-B primary BHR verdict with the estimability guard.

    Returns status in {INSUFFICIENT, BHR_DOWN, NO_CHANGE}. BHR_DOWN requires the
    paired CI on (BHR_b - BHR_a) to be strictly below 0 (b improves on a).
    """
    if not (estimable(raw_a) and estimable(projected_b)):
        return {"status": "INSUFFICIENT", "reason": "below N_min FULL_HITs/clusters"}
    ci = paired_delta_bhr_ci(raw_a, projected_b)
    down = ci["excludes_zero"] and ci["ci_high"] < 0
    return {"status": "BHR_DOWN" if down else "NO_CHANGE", **ci}


def exit_gate_pass(
    delta_bhr_ci: dict,
    *,
    bhr_significant: bool,
    sr_calib: float,
    sr_base: float,
    ir_calib: float,
    ir_illus: float,
    eps: float = 0.02,
) -> dict:
    """Primary framework exit gate: BHR↓ (significant) AND SR≥SR_base−ε AND IR not worse.

    ``bhr_significant`` is the (Holm-adjusted) BHR-down decision; the CI must also be < 0.
    Returns the joint PASS plus each conjunct so a partial pass is auditable.
    """
    bhr_down = bhr_significant and delta_bhr_ci.get("ci_high", 1.0) < 0
    sr_ok = sr_calib >= sr_base - eps
    ir_ok = ir_calib <= ir_illus
    return {
        "pass": bool(bhr_down and sr_ok and ir_ok),
        "bhr_down": bool(bhr_down),
        "sr_ok": bool(sr_ok),
        "ir_ok": bool(ir_ok),
    }


def suite_family_verdict(
    raw_a: list[FullHitRow],
    proj_b: list[FullHitRow],
    proj_c: list[FullHitRow],
    *,
    sr: dict,
    ir: dict,
    alpha: float = 0.05,
    eps: float = 0.02,
) -> dict:
    """Per-suite confirmatory family {H_AB, H_BC} with Holm-Bonferroni + SR/IR joint.

    ``sr`` = {"base","b"} success rates; ``ir`` = {"illus","b"} inflation ratios (for the
    A-vs-B exit gate). H_AB = A-vs-B Δ-BHR<0 (framework); H_BC = Claim-2 (C vs B). A' is a
    control, not confirmatory. Returns adjusted rejects + the joint framework verdict.
    """
    if not (estimable(raw_a) and estimable(proj_b)):
        return {"status": "INSUFFICIENT", "family": "H_AB not estimable"}
    # The confirmatory family is FIXED at TWO hypotheses; an underpowered H_BC still occupies
    # its family slot (p=1.0, cannot reject) so H_AB is corrected at alpha/2, never alpha.
    ab = paired_delta_bhr_ci(raw_a, proj_b)
    bc_estimable = estimable(proj_b) and estimable(proj_c)
    bc = paired_delta_bhr_ci(proj_b, proj_c) if bc_estimable else None
    pvals = {"H_AB": ab["pvalue"], "H_BC": (bc["pvalue"] if bc else 1.0)}
    reject = holm_bonferroni(pvals, alpha=alpha)

    # Adjusted percentile CIs at each hypothesis's Holm-adjusted alpha (reported beside raw).
    order = sorted(pvals, key=lambda k: pvals[k])
    adj_alpha = {name: alpha / (len(pvals) - rank) for rank, name in enumerate(order)}
    ab_adj = paired_delta_bhr_ci(raw_a, proj_b, ci_level=1.0 - adj_alpha["H_AB"])
    bc_adj = paired_delta_bhr_ci(proj_b, proj_c, ci_level=1.0 - adj_alpha["H_BC"]) if bc_estimable else None

    gate = exit_gate_pass(
        ab,
        bhr_significant=reject["H_AB"],
        sr_calib=sr["b"],
        sr_base=sr["base"],
        ir_calib=ir["b"],
        ir_illus=ir["illus"],
        eps=eps,
    )
    # Claim-2 decided on the Holm-ADJUSTED H_BC interval (not a fresh unadjusted 95%).
    claim2 = _claim2_decision(bc_adj) if bc_adj else {"status": "INCONCLUSIVE"}
    return {
        "status": "PASS" if gate["pass"] else "FAIL",
        "holm_reject": reject,
        "adjusted_alpha": adj_alpha,
        "H_AB": {"raw_ci": [ab["ci_low"], ab["ci_high"]], "adjusted_ci": [ab_adj["ci_low"], ab_adj["ci_high"]], "pvalue": ab["pvalue"]},
        "H_BC": ({"raw_ci": [bc["ci_low"], bc["ci_high"]], "adjusted_ci": [bc_adj["ci_low"], bc_adj["ci_high"]], "pvalue": bc["pvalue"]} if bc else {"status": "UNDERPOWERED"}),
        "claim2": claim2,
        "exit_gate": gate,
    }


def _claim2_decision(ci: dict) -> dict:
    """Claim-2 (C vs B) from a delta-BHR (=BHR_c-BHR_b) CI: SUPERIOR / EQUIVALENT / INDETERMINATE."""
    d = CLAIM2_EQUIV_DELTA
    if ci["ci_high"] < 0 and (ci["ci_low"] > 0 or ci["ci_high"] < 0):  # excludes 0, c below b
        status = "SUPERIOR"
    elif ci["ci_low"] >= -d and ci["ci_high"] <= d:
        status = "EQUIVALENT"
    else:
        status = "INDETERMINATE"
    return {"status": status, **ci}


def claim2_verdict(lane_b: list[FullHitRow], lane_c: list[FullHitRow]) -> dict:
    """Standalone Claim-2 (C vs B) on the raw 95% interval (see §C.4; the per-suite family
    verdict uses the Holm-ADJUSTED interval instead). INCONCLUSIVE if either lane underpowered.
    """
    if not (estimable(lane_b) and estimable(lane_c)):
        return {"status": "INCONCLUSIVE", "reason": "below N_min FULL_HITs/clusters"}
    return _claim2_decision(paired_delta_bhr_ci(lane_b, lane_c))  # BHR_c - BHR_b
