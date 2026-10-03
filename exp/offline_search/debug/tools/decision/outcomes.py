"""Observed excursion endpoints; terminal states absorb fixed control windows."""
from collections import Counter
import numpy as np

from . import common as C


def endpoints(arm, decisions, physical=True):
    output = {}
    for ek, group in decisions.groupby("episode_key", sort=False):
        rows = group.to_dict("records")
        invalid = any(row.get("termination_reason") == "exception" or
                      (not C.missing(row.get("error")) and bool(row.get("error"))) for row in rows)
        costs = [C.number(row.get("owner_cost")) for row in rows]
        counts = [C.number(row.get("n_applied")) for row in rows]
        controls = {}
        if physical and not invalid:
            try:
                controls = arm.controls(ek)
            except KeyError:
                pass
        for i, row in enumerate(rows):
            success = C.boolean(row.get("success"))
            value = {"final_success": float(success) if success is not None and not invalid else None,
                     "remaining_work": sum(costs[i:]) if not invalid and all(v is not None for v in costs[i:]) else None,
                     "remaining_active_controls": sum(counts[i:]) if not invalid and all(v is not None for v in counts[i:]) else None}
            reasons = {"final_success": "terminal success unavailable",
                       "remaining_work": "live owner costs unavailable",
                       "remaining_active_controls": "applied control counts unavailable",
                       "next_look_controls": "no subsequent look or client control clock; terminal distance is censored"}
            later = next((entry for entry in rows[i + 1:] if C.boolean(entry.get("vision")) is True), None)
            start = C.number(row.get("control_idx_start"))
            nxt = C.number(later.get("control_idx_start")) if later else None
            value["next_look_controls"] = nxt - start if not invalid and start is not None and nxt is not None else None
            value["next_look_censored"] = later is None
            for horizon in (5, 10, 20):
                value["predicate_delta_" + str(horizon)] = None
                value["eef_displacement_" + str(horizon)] = None
                value["valve_statistic_" + str(horizon)] = None
                value["robot_demo_drift_" + str(horizon)] = None
                value["object_relative_drift_" + str(horizon)] = None
                value["stall_entry_" + str(horizon)] = None
                for key in ("predicate_delta", "eef_displacement"):
                    reasons[key + "_" + str(horizon)] = "client physics/reference or complete control window unavailable"
                reasons["valve_statistic_" + str(horizon)] = "exact future decision clock and recorded valve statistic unavailable"
                reasons["stall_entry_" + str(horizon)] = "exact future clock and certified stall-entry diagnostic unavailable"
                reasons["robot_demo_drift_" + str(horizon)] = "aligned library robot-state/physics reference unavailable; EE motion alone is not demo drift"
                reasons["object_relative_drift_" + str(horizon)] = "admitted library physics and matched object frames unavailable"
                if start is not None and not invalid:
                    future = [r for r in rows[i + 1:] if C.number(r.get("control_idx_start")) is not None
                              and 0 < float(r["control_idx_start"]) - start <= horizon]
                    exact = next((r for r in future if float(r["control_idx_start"]) == start + horizon), None)
                    if exact is not None:
                        value["valve_statistic_" + str(horizon)] = C.number(C.field(exact, "shadow_delta", "valve_statistic", "os_sf_delta"))
                    states = [C.boolean(C.field(r, "stall_entry")) for r in future]
                    if states and all(x is not None for x in states) and exact is not None:
                        value["stall_entry_" + str(horizon)] = float(any(states))
            if not invalid and controls and all(key in controls for key in ("control_idx", "is_settle")) and start is not None:
                index = np.asarray(controls["control_idx"])
                active = np.flatnonzero((index >= int(start)) & ~np.asarray(controls["is_settle"], bool))
                before = np.flatnonzero(index < int(start))
                if len(active) and len(before):
                    b = before[-1]
                    term = row.get("termination_reason")
                    absorbs = success is not None and term in ("success", "step_cap", "max_steps", "timeout", "failure", "done")
                    for horizon in (5, 10, 20):
                        if len(active) < horizon and not absorbs:
                            continue
                        end = active[min(horizon, len(active)) - 1]
                        if "predicates" in controls:
                            preds = np.asarray(controls["predicates"], float)
                            if preds.ndim == 2 and preds.shape[1] and np.isfinite(preds[[b, end]]).all():
                                value["predicate_delta_" + str(horizon)] = float(np.mean(preds[end] - preds[b]))
                        if "eef_pos" in controls:
                            pos = np.asarray(controls["eef_pos"], float)
                            if np.isfinite(pos[[b, end]]).all():
                                value["eef_displacement_" + str(horizon)] = float(np.linalg.norm(pos[end] - pos[b]))
            value["valid"] = not invalid
            value["endpoint_reasons"] = {key: "exception/infrastructure error is invalid and excluded" if invalid else reason
                                         for key, reason in reasons.items() if value[key] is None}
            output[row["decision_id"]] = value
    return output


def availability(values):
    """Report source coverage even when an arm has no randomized support."""
    keys = {key for value in values.values() for key in value
            if key not in ("endpoint_reasons", "valid", "next_look_censored")}
    result = {}
    for key in sorted(keys):
        observed = sum(C.number(value.get(key)) is not None for value in values.values())
        reasons = Counter(value.get("endpoint_reasons", {}).get(key, "endpoint source unavailable")
                          for value in values.values() if C.number(value.get(key)) is None)
        result[key] = dict(C.coverage(len(values), observed), unavailable_reasons=dict(reasons))
    return result
