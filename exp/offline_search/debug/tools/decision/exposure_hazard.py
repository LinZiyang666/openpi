"""Follow-lottery dose excursions by pre-assignment stage and planned age."""
from collections import Counter

import numpy as np

from . import common as C
from .contrasts import estimate
from .outcomes import endpoints, availability


def lottery(row):
    support = C.field(row, "support", "E_support", "lottery_support")
    probabilities = C.field(row, "propensities", "E_propensities", "lottery_propensities")
    drawn = C.number(C.field(row, "drawn_e", "drawn_E", "lottery_E"))
    if support is None or probabilities is None or drawn is None:
        raise C.Unavailable("lottery support/actual propensities/drawn E unavailable")
    support = list(map(int, support))
    if isinstance(probabilities, dict):
        probabilities = {int(key): float(value) for key, value in probabilities.items()}
    else:
        # S5's fixed vector is indexed by E, including zero-probability unsupported E.
        probabilities = dict(enumerate(map(float, probabilities)))
    if len(set(support)) != len(support) or int(drawn) != drawn or int(drawn) not in support:
        raise C.Unavailable("invalid lottery support or drawn E")
    if any(not np.isfinite(p) or p < 0 for p in probabilities.values()) or not np.isclose(sum(probabilities.values()), 1.):
        raise C.Unavailable("invalid lottery probability vector")
    if any(probabilities.get(e, 0.) > 0 and e not in support for e in probabilities):
        raise C.Unavailable("positive probability outside structural support")
    if probabilities.get(int(drawn), 0.) <= 0:
        raise C.Unavailable("drawn E has zero actual propensity")
    return support, probabilities, int(drawn)


def analyze(arm, args=None):
    """Identify adjacent randomized extension packages without conditioning on survival to an age."""
    decisions, episodes = C.inputs(arm)
    readiness = C.causal_readiness(decisions, episodes)
    if readiness:
        return {"status": "unavailable", "reason": "; ".join(readiness),
                "coverage": C.coverage(len(decisions), 0), "episode_denominator": len(episodes), "tables": {}}
    values = endpoints(arm, decisions)
    output, entries, reasons = [], {}, Counter()
    for row in decisions.to_dict("records"):
        fresh = C.boolean(row.get("vision")) is True and row.get("src") == "cache"
        if not fresh:
            continue
        try:
            support, ps, drawn = lottery(row)
            coin = C.number(C.field(row, "coin"))
            forced = C.field(row, "override")
            if forced:
                raise C.Unavailable("forced lottery assignment")
            if coin is None or not 0 <= coin < 1:
                raise C.Unavailable("lottery coin unavailable or invalid")
            # S5 uses a uniform support coin; only audit this inversion when
            # the logged probabilities certify the uniform mechanism.
            positive = sorted(e for e in support if ps.get(e, 0.) > 0)
            if len(positive) < 2:
                raise C.Unavailable("no randomized extension support")
            if all(np.isclose(ps[e], 1 / len(positive)) for e in positive):
                if positive[min(int(coin * len(positive)), len(positive) - 1)] != drawn:
                    raise C.Unavailable("lottery coin_treatment_mismatch")
            output.append(dict(C.identity(row), status="supported", support=support, propensities=ps, drawn_e=drawn))
            stage = row["stage"] if row["stage_source"] == "pre_assignment" else "unknown"
            for age in (1, 2):
                control, treated = age - 1, age
                if control not in support or treated not in support or not (0 < ps.get(control, 0) < 1 and 0 < ps.get(treated, 0) < 1):
                    continue
                score = (1 / ps[treated] if drawn == treated else -1 / ps[control] if drawn == control else 0.)
                entry = dict(C.identity(row), stage=stage, score_weight=score,
                             treated_weight=1 / ps[treated] if drawn == treated else 0.,
                             control_weight=1 / ps[control] if drawn == control else 0., outcomes=values[row["decision_id"]])
                entries.setdefault((stage, age), []).append(entry)
        except C.Unavailable as error:
            reasons[str(error)] += 1
            output.append(dict(C.identity(row), status="unsupported", reason=str(error)))
    result = []
    endpoints_list = ["final_success", "next_look_controls", "remaining_work", "remaining_active_controls"] + [
        metric + "_" + str(h) for metric in ("eef_displacement", "robot_demo_drift", "object_relative_drift", "valve_statistic", "stall_entry")
        for h in (5, 10, 20)]
    family = max(1, len(entries) * len(endpoints_list))
    for (stage, age), selected in entries.items():
        for endpoint in endpoints_list:
            result.append(dict(stage=stage, extension_block=age, planned_age_controls=10 + 5 * (age - 1),
                contrast="E%d_minus_E%d" % (age, age - 1), **estimate(selected, episodes, endpoint, "first_entry", args, alpha=.05 / family)))
    if not result:
        result.append(dict(status="unavailable", reason="no supported lottery contrasts"))
    return {"status": "available" if any(row.get("status") == "available" for row in result) else "unavailable",
            "coverage": C.coverage(len(output), sum(row["status"] == "supported" for row in output)),
            "endpoint_coverage": availability(values),
            "episode_denominator": len(episodes), "support_reasons": dict(reasons), "multiplicity_family_size": family,
            "notes": ["Adjacent E packages are scored at assignment; realized age survivors are never a conditioning population.",
               "Stage × planned age uses pre-assignment labels, not stages reached after extension.",
               "Intervals use task/init bootstrap with Bonferroni family correction.",
               "Terminal episodes with no later look are censored for next-look distance, so that endpoint is withheld.",
               "EE displacement is motion, not object-relative drift or a failure label."],
            "tables": {"support": output, "contrasts": result}}


def main():
    C.cli("exposure_hazard", analyze)


if __name__ == "__main__":
    main()
