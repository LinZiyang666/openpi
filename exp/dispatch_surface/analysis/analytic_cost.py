"""Single cost authority for the dispatch-surface line (Rev 2 Phase 0, G1R2-B4).

Every consumer -- the Rev 1 analyzer, the Phase 0 cost-only loader, the
cost-map, the exploratory summary and the discipline validators -- imports the
stage constants, the warm formula and ``unit_cost`` from HERE. Copying the
constants elsewhere is exactly the drift this module exists to prevent, so the
canonical payload and its digest are recorded in every Phase 0 artifact and
cross-checked against the archived Rev 1 verdict (``discipline.cost_inputs``).

The numbers are the frozen CUDA-graph stage costs in milliseconds. They are
written at full precision and must never be rounded in code or documents: the
all-MISS anchor is judged against ``unit_cost("MISS", None)`` with an absolute
tolerance of 1e-9, so a truncated literal (67.5186) would reject a correct arm.
"""

from __future__ import annotations

import hashlib
import json
import math

STAGE1_MS = 10.260266
STAGE2_MS = 27.686469
STAGE3_MS = 29.571860
PINNED_START_T_WS = 0.3
VERDICTS = ("FULL_HIT", "WARM_START", "MISS")

#: Absolute tolerance for comparing a realized ratio-of-sums against a unit
#: cost. rel_tol is deliberately 0: the comparison is between two numbers that
#: are meant to be the same double, not between two measurements.
COST_ABS_TOL = 1e-9


def unit_cost(hit_type: str, start_t) -> float:
    """Analytic per-decision GPU inference cost of one verdict.

    FULL_HIT pays stage1 only. WARM_START resumes the flow at ``start_t`` and
    steps down to 0, running ``round(start_t * num_steps)`` of the stage-3
    steps -- start_t=0.3 runs 3 of 10 and saves 70% (pi0_pytorch.py:691).
    MISS pays all three stages. Anything else is a refusal, never a default.
    """
    if hit_type == "FULL_HIT":
        return STAGE1_MS
    if hit_type == "WARM_START":
        if start_t is None or float(start_t) != PINNED_START_T_WS:
            raise SystemExit(
                f"WARM_START row carries start_t={start_t!r}, expected the pinned "
                f"{PINNED_START_T_WS} — refusing to price an untested warm tier"
            )
        return STAGE1_MS + STAGE2_MS + PINNED_START_T_WS * STAGE3_MS
    if hit_type == "MISS":
        return STAGE1_MS + STAGE2_MS + STAGE3_MS
    raise SystemExit(
        f"unknown hit_type {hit_type!r}; only {VERDICTS} may be priced. Missing "
        "or unprobed verdicts are never silently billed as MISS"
    )


def unit_cost_table() -> dict[str, float]:
    """The per-stage and per-verdict costs, full precision, in the order the
    Rev 1 analyzer records them under ``discipline.cost_inputs.unit_cost_ms``."""
    return {
        "stage1": STAGE1_MS,
        "stage2": STAGE2_MS,
        "stage3": STAGE3_MS,
        "FULL_HIT": unit_cost("FULL_HIT", None),
        "WARM_START": unit_cost("WARM_START", PINNED_START_T_WS),
        "MISS": unit_cost("MISS", None),
    }


def cost_model_payload() -> dict:
    """Canonical, JSON-serialisable description of the cost model."""
    return {
        "model": "analytic GPU inference cost (model-forward compute proxy); "
                 "retrieval CPU excluded; not a measured end-to-end latency",
        "stage_ms": {"stage1": STAGE1_MS, "stage2": STAGE2_MS, "stage3": STAGE3_MS},
        "warm_start_formula": "stage1 + stage2 + start_t * stage3",
        "pinned_start_t_ws": PINNED_START_T_WS,
        "unit_cost_ms": unit_cost_table(),
    }


def cost_model_digest() -> str:
    """sha256 of the canonical payload (sorted keys, compact separators)."""
    blob = json.dumps(cost_model_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cost_matches(realized: float, expected: float) -> bool:
    """Full-precision equality up to ``COST_ABS_TOL``; rel_tol is zero on purpose."""
    return math.isclose(realized, expected, rel_tol=0.0, abs_tol=COST_ABS_TOL)


def assert_unit_costs_match(recorded: dict, *, what: str) -> None:
    """Refuse a consumer whose recorded unit costs differ from this authority.

    ``recorded`` is a ``unit_cost_ms`` mapping as written by a producer (the
    archived Rev 1 verdict, a Phase 0 matrix or ledger). Every key of the
    authority table must be present and equal at full precision.
    """
    table = unit_cost_table()
    if not isinstance(recorded, dict):
        raise SystemExit(f"{what}: unit_cost_ms is not a mapping")
    for key, value in table.items():
        got = recorded.get(key)
        if not isinstance(got, (int, float)) or isinstance(got, bool) or not cost_matches(float(got), value):
            raise SystemExit(
                f"{what}: unit cost {key} = {got!r} differs from the cost authority {value!r}"
            )
