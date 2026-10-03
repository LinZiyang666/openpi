"""Online trigger lead times and stage hit rates against S6 onset labels."""
import json
from pathlib import Path

import pandas as pd

from . import common as C


def applicable(row, trigger, domains):
    domain = domains.get(trigger)
    if domain is None:
        if trigger == "os_sf_valve_fire":
            # A failed blind check returns LookReason; its diagnostics accompany
            # the ensuing vision request, not a served blind block.
            extras = row.get("blind_extras")
            return C.boolean(row.get("vision")) is not True or isinstance(extras, dict) and trigger in extras
        domain = "anchor" if trigger in ("os_c_stall_call", "os_c3_dev_entry") else "all"
    vision = C.boolean(row.get("vision"))
    if domain == "anchor":
        return vision is not False
    if domain == "blind":
        return vision is not True
    if domain != "all":
        raise ValueError("trigger domain must be anchor, blind or all")
    return True


def trigger_value(row, trigger):
    extras = row.get("blind_extras")
    if trigger == "os_sf_valve_fire" and isinstance(extras, dict) and trigger in extras:
        # Current check takes precedence over stale nested method diagnostics.
        return C.number(extras[trigger])
    return C.number(C.field(row, trigger))


def read_labels(path, issues=None):
    if path is None:
        raise C.Unavailable("--onsets S6 labels file required")
    path = Path(path)
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        rows = frame.to_dict("records")
    elif path.suffix == ".jsonl":
        from exp.offline_search.debug.reader import read_jsonl
        rows = read_jsonl(path, issues)
    else:
        value = json.loads(path.read_text())
        if isinstance(value, list):
            rows = value
        elif "tables" in value and "episodes_forensics" in value["tables"]:
            rows = value["tables"]["episodes_forensics"]
        else:
            raise C.Unavailable("onset JSON must be rows or S6 tables.episodes_forensics")
    return rows


def analyze(arm, args=None):
    """Compare independent after-control onset intervals with pre-action online alerts."""
    decisions, episodes = C.inputs(arm)
    label_issues = []
    labels = read_labels(getattr(args, "onsets", None), label_issues)
    name = getattr(arm, "arm_name", None)
    labels = [row for row in labels if "arm" not in row or name is None or row["arm"] == name]
    lookup = {}
    for row in labels:
        key = row.get("episode_key")
        if key is None:
            raise C.Unavailable("S6 onset episode_key absent; task/init cannot substitute an attempt identity")
        if key in lookup:
            raise ValueError("duplicate onset label for episode " + str(key))
        lookup[key] = row
    triggers = getattr(args, "triggers", None) or ["os_sf_valve_fire", "os_c_stall_call", "os_c3_dev_entry", "shadow_valve_fire"]
    domains = C.metadata(arm).get("trigger_domains", {})
    output, leads, summary = [], [], []
    for trigger in triggers:
        # shadow_valve_fire is a diagnostic, explicitly separated from online triggers.
        scope = "shadow_diagnostic" if trigger.startswith("shadow_") else "online"
        available, onset_n, hit_n, early_n, false_n = 0, 0, 0, 0, 0
        for ep in episodes.to_dict("records"):
            group = decisions[decisions.episode_key == ep["episode_key"]]
            label = lookup.get(ep["episode_key"])
            rec = {"episode_key": ep["episode_key"], "task_id": ep["task_id"], "init": ep["init"],
                   "trigger": trigger, "scope": scope, "status": "unavailable"}
            if ep.get("termination_reason") == "exception" or (not C.missing(ep.get("error")) and bool(ep.get("error"))):
                rec["reason"] = "exception/infrastructure error is invalid and excluded"
                output.append(rec)
                continue
            if label is None or label.get("status", "available") != "available":
                rec["reason"] = "onset label unavailable"
                output.append(rec)
                continue
            rows = [row for row in group.to_dict("records") if applicable(row, trigger, domains)]
            observed = [(row, trigger_value(row, trigger)) for row in rows]
            if not observed or any(value is None or C.number(row.get("control_idx_start")) is None for row, value in observed):
                rec["reason"] = "trigger or client control clock unavailable at one or more decisions"
                output.append(rec)
                continue
            available += 1
            onset = C.number(label.get("onset_control"))
            alerts = [row for row, value in observed if value > 0]
            rec.update(status="available", onset_control=onset, alert_denominator=len(alerts),
                       onset_confidence=label.get("onset_confidence"), truth_validated=C.boolean(label.get("truth_validated")),
                       rule_version=label.get("rule_version"), label=label.get("label"))
            if onset is None:
                certified_negative = C.boolean(ep.get("success")) is True or C.boolean(label.get("no_onset_certified")) is True
                rec.update(hit=None, false_alert=bool(alerts) if certified_negative else None,
                           stage="no_onset" if certified_negative else "onset_time_unavailable")
                false_n += bool(alerts) if certified_negative else 0
            else:
                onset_n += 1
                before = [row for row in alerts if float(row["control_idx_start"]) <= onset]
                near = [row for row in before if onset - float(row["control_idx_start"]) <= 20]
                hit = bool(near)
                hit_n += hit
                lead = onset - float(near[0]["control_idx_start"]) if near else None
                rec.update(hit=hit, actionable_hit=bool(lead is not None and lead >= 5),
                           lead_controls=lead, stage=near[0]["stage"] if near else "unhit")
                early_n += rec["actionable_hit"]
                interval = label.get("onset_interval")
                if isinstance(interval, str):
                    try:
                        interval = json.loads(interval)
                    except ValueError:
                        interval = None
                if near and isinstance(interval, list) and len(interval) == 2:
                    rec["lead_interval_controls"] = [bound - float(near[0]["control_idx_start"]) for bound in interval]
                for row in before:
                    leads.append(dict(C.identity(row), trigger=trigger, scope=scope,
                                      lead_controls=onset - float(row["control_idx_start"]), onset_confidence=label.get("onset_confidence")))
            output.append(rec)
        negatives = sum(row["trigger"] == trigger and row["status"] == "available" and row.get("false_alert") is not None for row in output)
        record = dict(trigger=trigger, scope=scope, episode_denominator=len(episodes), available=available,
            onset_episode_denominator=onset_n, hit=hit_n, hit_rate=hit_n / onset_n if onset_n else None,
            actionable_hit=early_n, no_onset_episode_denominator=negatives, onset_time_unavailable=available - onset_n - negatives,
            false_alert=false_n, false_alert_rate=false_n / negatives if negatives else None,
            horizon_controls=20, actionable_lead_controls=5)
        good = [row for row in output if row["trigger"] == trigger and row["status"] == "available" and row.get("hit") is not None]
        if good:
            frame = pd.DataFrame(good)
            frame["numerator"] = frame.hit.astype(float)
            frame["denominator"] = 1.
            record["hit_interval"] = C.cluster_interval(frame, bootstraps=getattr(args, "bootstraps", 1000), seed=getattr(args, "seed", 0))
        summary.append(record)
    # The denominator for a stage-specific hit is onset episodes exposed to that
    # stage in the lookback window, including episodes whose trigger never fired.
    stages = []
    for trigger in triggers:
        for stage in sorted(set(decisions.stage)):
            selected = []
            for row in output:
                if row["trigger"] != trigger or row["status"] != "available" or row.get("onset_control") is None:
                    continue
                onset = row["onset_control"]
                group = decisions[(decisions.episode_key == row["episode_key"]) & (decisions.stage == stage)]
                window = [d for d in group.to_dict("records") if applicable(d, trigger, domains) and 0 <= onset - float(d["control_idx_start"]) <= 20]
                if window:
                    selected.append(dict(task_id=row["task_id"], init=row["init"], denominator=1.,
                        numerator=float(any(trigger_value(d, trigger) > 0 for d in window))))
            if selected:
                stages.append(dict(trigger=trigger, stage=stage, onset_episode_exposure_denominator=len(selected),
                    hit_rate=C.cluster_interval(pd.DataFrame(selected), bootstraps=getattr(args, "bootstraps", 1000), seed=getattr(args, "seed", 0))))
    return {"status": "available" if any(row["available"] for row in summary) else "unavailable",
            "coverage": C.coverage(len(episodes) * len(triggers), sum(row["available"] for row in summary)),
            "onset_input_diagnostics": label_issues,
            "notes": ["S6 onset confidence, rule version and validation status are retained; heuristic labels are not asserted as ground truth.",
               "Hit means a trigger in the 20-control pre-onset window; actionable hit requires at least five controls lead.",
               "Shadow valves are diagnostic scope. Per-stage denominators include unhit onset episodes exposed to that stage."],
            "tables": {"episodes": output, "leads": leads, "triggers": summary, "stages": stages}}


def main():
    C.cli("trigger_vs_onset", analyze)


if __name__ == "__main__":
    main()
