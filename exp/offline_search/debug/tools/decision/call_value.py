"""Actual-propensity randomized CALL excursions; deterministic choices withheld."""
from collections import Counter

from . import common as C
from .contrasts import estimate
from .outcomes import endpoints, availability


def treatment(value):
    if isinstance(value, str):
        label = value.lower()
        if label in ("call", "policy"):
            return True
        if label == "cache":
            return False
    return C.boolean(value)


def analyze(arm, args=None):
    """Estimate first supported stage-entry excursions and probability-shift derivatives."""
    decisions, episodes = C.inputs(arm)
    readiness = C.causal_readiness(decisions, episodes)
    if readiness:
        return {"status": "unavailable", "reason": "; ".join(readiness),
                "coverage": C.coverage(len(decisions), 0), "episode_denominator": len(episodes), "tables": {}}
    values = endpoints(arm, decisions)
    audit, supported = [], []
    reasons = Counter()
    for row in decisions.to_dict("records"):
        p = C.number(C.field(row, "p_effective", "actual_propensity", "os_c_p"))
        z = treatment(C.field(row, "treatment", "executed_policy", "os_c_call"))
        eligible = C.boolean(C.field(row, "eligible", "os_c_eligible", "os_c_fresh"))
        coin = C.number(C.field(row, "coin", "os_c_coin", "os_c_u", "uniform"))
        offset = C.number(row.get("chunk_offset"))
        fresh = C.boolean(row.get("vision")) is True and row.get("src") not in ("policy_tail", "cache_tail", "cache_blind", "follow") and offset in (None, 0)
        reason = None
        if not fresh:
            reason = "not_fresh_anchor"
        elif eligible is not True:
            reason = "ineligible_or_eligibility_unavailable"
        elif p is None or z is None:
            reason = "actual_propensity_or_treatment_unavailable"
        elif p < 0 or p > 1:
            reason = "invalid_propensity"
        elif p in (0., 1.):
            reason = "forced_no_interior_support"
        elif coin is None:
            reason = "coin_unavailable"
        elif not 0 <= coin < 1 or (coin < p) != z:
            reason = "coin_treatment_mismatch"
        audit.append(dict(C.identity(row), p_effective=p, treatment=z, eligible=eligible, fresh=fresh,
                          coin=coin, status="supported" if reason is None else "unsupported", reason=reason,
                          override=C.field(row, "override"), p_nominal=C.number(C.field(row, "p_nominal"))))
        if reason:
            reasons[reason] += 1
            continue
        # Only a logged pre-assignment stage may moderate the random coin.
        stage = row["stage"] if row["stage_source"] == "pre_assignment" else "unknown"
        supported.append(dict(C.identity(row), stage=stage, p_effective=p,
            score_weight=(1 / p if z else -1 / (1 - p)), treated_weight=(1 / p if z else 0.),
            control_weight=(1 / (1 - p) if not z else 0.), outcomes=values[row["decision_id"]]))
    stage_names = sorted(set(row["stage"] for row in supported))
    results = []
    selected_endpoints = ["final_success", "remaining_work", "remaining_active_controls"] + ["predicate_delta_" + str(h) for h in (5, 10, 20)]
    for stage in stage_names:
        entries = [row for row in supported if row["stage"] == stage]
        for endpoint in selected_endpoints:
            for mode in ("first_entry", "probability_shift_derivative"):
                results.append(dict(stage=stage, **estimate(entries, episodes, endpoint, mode, args)))
    if not results:
        results.append(dict(stage="unknown", status="unavailable", reason="no auditable supported randomized anchors"))
    return {"status": "available" if any(row.get("status") == "available" for row in results) else "unavailable",
            "coverage": C.coverage(len(decisions), len(supported)), "episode_denominator": len(episodes),
            "endpoint_coverage": availability(values),
            "support_reasons": dict(reasons), "notes": ["Effective conditional p after overrides is used; nominal p is never substituted.",
                "First-entry population scores include unreached episodes with zero contribution; reached-entry denominator is separate.",
                "Sum of anchor scores estimates a local probability-shift derivative, not controller replacement value.",
                "Unknown includes unavailable assignment-time stage certification; catalogue proxies remain descriptive.",
                "Intervals resample task/init clusters within fixed tasks; fewer than 30 clusters is exploratory."],
            "tables": {"support": audit, "contrasts": results}}


def main():
    C.cli("call_value", analyze)


if __name__ == "__main__":
    main()
