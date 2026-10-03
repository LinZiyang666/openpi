"""Served blind/follow proposal versus unused full-look A retrieval."""
import numpy as np

from . import common as C
from .kernels import catalog_map, kernel, overlap
from .metrics import action_metrics
from .library import MemberActions


def analyze(arm, args=None):
    """Measure full-look opportunity on the arm's own path and export a row follow-gap map.

    Follows E4's prototype metrics, replacing its reconstructed A-path second look
    with the same-observation deferred retrieval on the actual closed-loop path.
    """
    decisions, episodes = C.inputs(arm)
    ids = decisions.decision_id.tolist()
    live = C.arrays(arm, ["served_chunk"], ids)
    shadow = C.augmentation(arm, "shadow_look", ids)
    fresh = C.pick(shadow, "cache_chunk", "chunk")
    proposal_reason = "served_chunk or shadow_look cache_chunk unavailable" if "served_chunk" not in live or fresh is None else None
    try:
        catalog = catalog_map(arm)
    except C.Unavailable:
        catalog = None
    output, row_map = [], []
    members = MemberActions(arm)
    eligible = 0
    for i, row in enumerate(decisions.to_dict("records")):
        if C.boolean(row.get("vision")) is not False and row.get("src") not in ("follow", "cache_tail", "cache_blind"):
            continue
        # A policy tail is separately profiled by divergence, not cache following.
        if row.get("src") == "policy_tail":
            continue
        eligible += 1
        n = C.number(row.get("n_applied"))
        head = int(n) if n is not None else None
        if head is None or head <= 0:
            output.append(dict(C.identity(row), src=row.get("src"), status="unavailable", reason="actual applied prefix unavailable",
                               member_spread_status="unavailable", member_spread_reason="actual applied prefix unavailable"))
            continue
        record = dict(C.identity(row), status="unavailable", n_applied=head, src=row.get("src"),
                      blind_age_controls=C.number(row.get("blind_age_controls")), lib=C.field(row, "lib"),
                      library_size=C.metadata(arm).get("library_size"), disagreement_over_member_spread=None)
        try:
            record.update(members.spread(row, head))
        except C.Unavailable as error:
            record.update(member_spread=None, member_spread_status="unavailable", member_spread_reason=str(error))
        if proposal_reason:
            output.append(dict(record, reason=proposal_reason, disagreement_over_member_spread_reason=proposal_reason))
            continue
        metrics = action_metrics(arm, live["served_chunk"][i][:head], fresh[i][:head])
        if not metrics:
            output.append(dict(record, reason="same-observation proposals nonfinite",
                               disagreement_over_member_spread_reason="same-observation proposals nonfinite"))
            continue
        record.update(status="available", offset_denominator=len(metrics),
                      motion_rms=float(np.sqrt(np.mean([m["motion_rms"] ** 2 for m in metrics if m["motion_rms"] is not None]))),
                      rms=float(np.sqrt(np.mean([m["mse"] for m in metrics]))), units=metrics[0]["units"],
                      gripper_flip=float(np.mean([m["gripper_flip"] for m in metrics])) if metrics[0]["gripper_flip"] is not None else None,
                      direction_cosine=C.number(np.nanmean([m["direction_cosine"] if m["direction_cosine"] is not None else np.nan for m in metrics]))
                          if any(m["direction_cosine"] is not None for m in metrics) else None)
        record["disagreement_over_member_spread"] = record["motion_rms"] / record["member_spread"] if record["member_spread"] else None
        record["disagreement_over_member_spread_reason"] = ("member spread unavailable" if record["member_spread"] is None
                                                           else "member spread is zero" if record["member_spread"] == 0 else None)
        rows, weights = C.field(row, "rows"), C.field(row, "weights")
        if rows is not None and weights is not None and "rows" in shadow and "weights" in shadow:
            if np.isfinite(shadow["rows"][i]).all() and np.isfinite(shadow["weights"][i]).all():
                record["kernel_overlap"] = overlap(rows, weights, shadow["rows"][i], shadow["weights"][i])
                if catalog is not None:
                    old = kernel(rows, weights, catalog, record["lib"])
                    new = kernel(shadow["rows"][i], shadow["weights"][i], catalog, record["lib"])
                    record["top_demo_same"] = old["top_demo"] == new["top_demo"] if old["top_demo"] and new["top_demo"] else None
                    record["progress_slip"] = new["progress"] - old["progress"] if old["progress"] is not None and new["progress"] is not None else None
            normalized = np.asarray(weights, float)
            if np.isfinite(normalized).all() and normalized.sum() > 0 and (normalized >= 0).all():
                for rid, weight in zip(rows, normalized / normalized.sum()):
                    row_map.append(dict(C.identity(row), lib=record["lib"], row=int(rid), weight=float(weight),
                                        motion_rms=record["motion_rms"], gripper_flip=record["gripper_flip"]))
        output.append(record)
    gap_map = []
    if row_map:
        import pandas as pd
        frame = pd.DataFrame(row_map)
        for (lib, rid), group in frame.groupby(["lib", "row"], dropna=False):
            mass = group.weight.sum()
            gap_map.append(dict(lib=lib, row=int(rid), decision_denominator=int(group.decision_id.nunique()),
                task_init_clusters=len(group[["task_id", "init"]].drop_duplicates()), weight_mass=float(mass),
                follow_gap=float((group.weight * group.motion_rms).sum() / mass) if mass else None,
                attribution="kernel-weighted decision gap; not a row-specific causal effect"))
    available = sum(row["status"] == "available" for row in output)
    return {"status": "available" if available else "unavailable", "coverage": C.coverage(eligible, available, proposal_reason),
            "member_spread_coverage": C.coverage(eligible, sum(row.get("member_spread_status") == "available" for row in output)),
            "all_decision_denominator": len(decisions), "episode_denominator": len(episodes),
            "notes": ["Unused shadow look is an opportunity proxy, not an observed rescue.",
                "Motion scale is manifest supplied; gripper flips require a declared command threshold.",
                "Member spread is derived from immutable store rows, with native-tail offsets or explicitly recorded successor heads; missing alignment is unavailable."],
            "tables": {"decisions": output, "follow_gap_map": gap_map, "curves": C.summarize(output,
                ["stage", "blind_age_controls", "lib", "library_size", "src"], ["member_spread", "motion_rms", "gripper_flip", "direction_cosine", "kernel_overlap", "progress_slip", "top_demo_same", "disagreement_over_member_spread"])}}


def main():
    C.cli("follow_vs_look", analyze)


if __name__ == "__main__":
    main()
