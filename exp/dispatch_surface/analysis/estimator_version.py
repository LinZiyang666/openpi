"""Frozen estimator identities for the dispatch-surface analyses.

Every analysis artifact records the canonical digest of the estimator it was
computed with, so a consumer can refuse to mix outputs of different estimands.
``EXACT_COST_V1`` is the Phase 0 straight-hull estimator (kept only as a
registry entry; its implementation lives untouched in ``frontier_hull``).
``BUDGET_MIXTURE_V1`` is the confirmation-plan estimator (plan section 3.1):
the episode-level randomized-mixture LP under the ratio-of-sums budget.
"""

from __future__ import annotations

import hashlib
import json

EXACT_COST_V1 = {
    "name": "exact_cost_v1",
    "objective": "upper concave hull of (ratio cost, SR) points",
    "pruning": "weak Pareto on (cost, SR)",
    "support": "hull covers both interval ends",
    "support_miss_value": -1.0,
    "max_joint_miss": 0.01,
}

BUDGET_MIXTURE_V1 = {
    "name": "budget_mixture_v1",
    "objective": "max episode-mixture SR",
    "cost_constraint": "C(p) = sum p_i T_i / sum p_i D_i <= B",
    "pruning": "none",
    "basis": "<=2 arms (single or tight pair)",
    "tie_break": "fewer arms, then canonical arm tuple",
    "value_tol": 1e-12,
    "breakpoint_tol_ms": 1e-9,
    "left_tail": "infeasible",
    "right_tail": "implicit (all arms feasible)",
    "integral": ("exact piecewise: constant + linear-fractional closed form; "
                 "breakpoints = single costs + three-point collinearity roots"),
    "numeric_audit": ("internal invariants every call; Simpson abs_tol 1e-10 on frozen "
                      "audit subset; mismatch > 1e-8 fails artifact"),
    "support_miss_value": -1.0,
    "max_joint_miss": 0.01,
    "sensitivity": "measured-policy-only step envelope",
}

REGISTRY = {"exact_cost_v1": EXACT_COST_V1, "budget_mixture_v1": BUDGET_MIXTURE_V1}


def canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def digest(payload: dict = BUDGET_MIXTURE_V1) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def budget_mixture_digest() -> str:
    return digest(BUDGET_MIXTURE_V1)


def assert_estimator(recorded: str | None, *, what: str) -> None:
    """Refuse an artifact computed under a different estimator."""
    if recorded != budget_mixture_digest():
        raise SystemExit(f"{what}: estimator digest {str(recorded)[:12]}... != budget_mixture_v1")
