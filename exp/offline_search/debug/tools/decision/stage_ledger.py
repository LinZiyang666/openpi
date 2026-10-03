"""Additive stage compute ledgers, visit costs and actual active controls."""
import numpy as np
import pandas as pd

from . import common as C


MEASURED = {"pi05": (.152, .848), "groot": (.148, .852)}


def prices(arm):
    meta = C.metadata(arm)
    model = meta.get("model", meta.get("model_name"))
    measured = MEASURED.get(model)
    return meta, measured


def decision_cost(row, full, head, wrist, completion):
    stage1 = C.number(C.field(row, "stage1_calls"))
    calls = C.number(C.field(row, "policy_calls"))
    completed = C.number(C.field(row, "camera_completions"))
    if any(value is None for value in (stage1, calls, completed)):
        return None
    if min(stage1, calls, completed) < 0 or any(int(value) != value for value in (stage1, calls, completed)):
        raise ValueError("live dispatch counts must be nonnegative integers")
    mode = row.get("camera_mode")
    if mode == "third_only" and stage1 > 0:
        return None  # No declared measured third-only serving price.
    if mode not in ("full", "wrist_only", "blind"):
        return None
    return stage1 * (wrist if mode == "wrist_only" else full) + head * calls + completion * completed


def analyze(arm, args=None):
    """Account live deployment work by stage and visit, with two price ledgers."""
    decisions, episodes = C.inputs(arm)
    meta, measured = prices(arm)
    if measured is None:
        # New models must supply explicit prices rather than inherit LIBERO prices.
        full, head = C.number(meta.get("full_look")), C.number(meta.get("policy_head"))
        if full is None or head is None:
            raise C.Unavailable("model measured price table unavailable")
        measured = full, head
    wrist = C.number(meta.get("measured_wrist_price"))
    completion = C.number(meta.get("measured_completion_price"))
    wrist = .0646 if wrist is None and meta.get("model") == "pi05" else wrist
    completion = .0502 if completion is None and meta.get("model") == "pi05" else completion
    assumed_full = C.number(meta.get("assumed_full_price"))
    assumed_head = C.number(meta.get("assumed_head_price"))
    assumed_wrist = C.number(meta.get("assumed_wrist_price"))
    assumed_completion = C.number(meta.get("assumed_completion_price"))
    # R4 ledger is a named historical assumption, never the measured owner ledger.
    if meta.get("model") in MEASURED:
        assumed_full = measured[0] if assumed_full is None else assumed_full
        assumed_head = measured[1] if assumed_head is None else assumed_head
        assumed_wrist = .055198 if assumed_wrist is None and meta.get("model") == "pi05" else assumed_wrist
        assumed_completion = .049890 if assumed_completion is None and meta.get("model") == "pi05" else assumed_completion
    output = []
    visits = {}
    for row in decisions.to_dict("records"):
        key = row["episode_key"]
        prev_stage, visit = visits.get(key, (None, -1))
        visit += int(row["stage"] != prev_stage)
        visits[key] = row["stage"], visit
        def cost(table, wp, cp):
            if table is None:
                return None
            # Prices for unused partial-camera paths may be absent safely.
            partial = row.get("camera_mode") == "wrist_only" and C.number(row.get("stage1_calls")) not in (None, 0)
            completing = C.number(row.get("camera_completions")) not in (None, 0)
            if (partial and wp is None) or (completing and cp is None):
                return None
            return decision_cost(row, *table, wp or 0., cp or 0.)
        output.append(dict(C.identity(row), visit=visit, N=1,
            V=C.number(C.field(row, "stage1_calls")), M=C.number(C.field(row, "policy_calls")),
            camera_completions=C.number(C.field(row, "camera_completions")),
            active_controls=C.number(row.get("n_applied")), measured_work=cost(measured, wrist, completion),
            assumed_work=cost((assumed_full, assumed_head) if assumed_full is not None and assumed_head is not None else None,
                              assumed_wrist, assumed_completion), owner_work=C.number(row.get("owner_cost")),
            trigger_stage=C.field(row, "parent_stage", "trigger_stage", default="unknown")))
    frame = pd.DataFrame(output)
    summary, per_visit = [], []
    if len(frame):
        def ledger(group, label):
            result = dict(label, decision_denominator=len(group), N=len(group))
            for key in ("V", "M", "camera_completions", "active_controls", "measured_work", "assumed_work", "owner_work"):
                result[key + "_available"] = int(group[key].notna().sum())
                result[key] = float(group[key].sum()) if group[key].notna().all() else None
            for prefix in ("measured", "assumed", "owner"):
                work = result[prefix + "_work"]
                result[prefix + "_IR_requests"] = work / len(group) if work is not None else None
                controls = result["active_controls"]
                result[prefix + "_IR_controls"] = 5 * work / controls if work is not None and controls else None
                result[prefix + "_share_arm_IR"] = work / len(frame) if work is not None else None
            return result
        total = ledger(frame, {"stage": "__arm__"})
        summary.append(total)
        for stage, group in frame.groupby("stage", dropna=False):
            summary.append(ledger(group, {"stage": stage}))
        for (key, visit), group in frame.groupby(["episode_key", "visit"]):
            per_visit.append(ledger(group, {"episode_key": key, "visit": int(visit), "stage": group.stage.iloc[0]}))
        per_episode = [ledger(group, {"episode_key": key}) for key, group in frame.groupby("episode_key")]
    else:
        total, per_episode = {}, []
    return {"status": "available" if total.get("measured_work") is not None else "unavailable",
            "coverage": C.coverage(len(frame), int(frame.measured_work.notna().sum()) if len(frame) else 0),
            "episode_denominator": len(episodes), "prices": {"measured_full_head": measured, "measured_wrist": wrist,
               "measured_completion": completion, "R4_assumed_full_head": [assumed_full, assumed_head], "assumed_wrist": assumed_wrist},
            "equal_episode_mean_IR": {prefix: float(np.mean([e[prefix + "_IR_requests"] for e in per_episode]))
                if per_episode and all(e[prefix + "_IR_requests"] is not None for e in per_episode) else None
                for prefix in ("measured", "assumed", "owner")},
            "notes": ["Only live dispatch counts enter work. Diagnostic shadows are excluded.",
                "Stage work is additive. Request IR=K/N; control IR=5K/active controls; settling excluded.",
                "Unknown stages are retained. Visits are contiguous runs of recorded stage labels."],
            "tables": {"decisions": output, "stages": summary, "visits": per_visit, "episodes": per_episode}}


def main():
    C.cli("stage_ledger", analyze)


if __name__ == "__main__":
    main()
