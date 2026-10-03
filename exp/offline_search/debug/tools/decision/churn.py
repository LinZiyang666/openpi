"""Paired outcome flips: symmetric churn and net bias versus A controls."""
from . import common as C

import pandas as pd


def compare(reference, candidate, args=None, role="lever"):
    from exp.offline_search.debug import reader
    paired = reader.pair(reference, candidate)
    C.require(paired, ["task_id", "init", "success_a", "success_b"])
    output = []
    for row in paired.to_dict("records"):
        a, b = C.boolean(row.get("success_a")), C.boolean(row.get("success_b"))
        value = {"task_id": row["task_id"], "init": row["init"], "role": role,
                 "reference_episode_key": row.get("episode_key_a"), "episode_key": row.get("episode_key_b"),
                 "status": "available" if a is not None and b is not None else "unavailable"}
        if any(row.get("termination_reason_" + side) == "exception" or
               (not C.missing(row.get("error_" + side)) and bool(row.get("error_" + side))) for side in ("a", "b")):
            value.update(status="unavailable", reason="exception/infrastructure error is invalid and excluded")
        verified = True
        conflicts = []
        for key in ("env_seed", "reset_state_sha256", "orig_init_state_idx"):
            x, y = row.get(key + "_a"), row.get(key + "_b")
            if C.missing(x) or C.missing(y):
                verified = False
            elif x != y:
                conflicts.append(key)
        if conflicts:
            value.update(status="unavailable", reason="paired episodes have different " + ", ".join(conflicts))
            verified = False
        value["physical_pair_identity_verified"] = verified
        if value["status"] == "available":
            value.update(reference_success=a, success=b, gain=int(b and not a), loss=int(a and not b),
                         flip=int(a != b), net_success=int(b) - int(a), denominator=1.)
        output.append(value)
    good = [row for row in output if row["status"] == "available"]
    n = len(good)
    summary = dict(role=role, paired_episode_denominator=len(paired), outcome_available=n,
                   unmatched_reference=len(reference.episodes()) - len(paired),
                   unmatched_candidate=len(candidate.episodes()) - len(paired),
                   gain=sum(row["gain"] for row in good), loss=sum(row["loss"] for row in good),
                   status="available" if n and n == len(paired) else "unavailable")
    if summary["status"] == "available":
        summary.update(churn=(summary["gain"] + summary["loss"]) / n,
                       symmetric_churn=2 * min(summary["gain"], summary["loss"]) / n,
                       net_loss_bias=(summary["loss"] - summary["gain"]) / n)
        frame = pd.DataFrame(good)
        summary["net_success"] = C.cluster_interval(frame, numerator="net_success", bootstraps=getattr(args, "bootstraps", 1000), seed=getattr(args, "seed", 0))
        summary["flip_rate"] = C.cluster_interval(frame, numerator="flip", bootstraps=getattr(args, "bootstraps", 1000), seed=getattr(args, "seed", 0))
    else:
        reasons = sorted({row.get("reason", "paired outcome unavailable") for row in output if row["status"] != "available"})
        summary["reason"] = "; ".join(reasons) if reasons else "no paired outcomes; denominator retained"
    return output, summary


def analyze(arm, args):
    """Decompose flips and compare A replication/placebo sensitivity on paired initial states."""
    opened = args.opened
    refname = args.reference or args.arms[0]
    reference = opened[refname]
    output, primary = compare(reference, arm, args)
    primary.update(comparison=getattr(arm, "arm_name", "candidate"), reference=refname)
    summaries = [primary]
    direct = []
    for role, names in (("A_replicate", args.replicates), ("placebo", args.placebos)):
        for name in names:
            rows, result = compare(reference, opened[name], args, role)
            summaries.append(dict(result, comparison=name, reference=refname))
            for entry in rows:
                entry["comparison"] = name
            output.extend(rows)
            # Direct paired net difference respects covariance between lever and control.
            lever = pd.DataFrame([r for r in output if r["role"] == "lever" and r["status"] == "available"])
            control = pd.DataFrame([r for r in rows if r["status"] == "available"])
            if len(lever) and len(control) and primary["status"] == "available" and result["status"] == "available":
                paired = lever.merge(control, on=["task_id", "init"], suffixes=("_lever", "_control"), validate="one_to_one")
                paired["numerator"] = paired.net_success_lever - paired.net_success_control
                paired["denominator"] = 1.
                direct.append(dict(control=name, role=role, estimate=C.cluster_interval(paired, bootstraps=args.bootstraps, seed=args.seed)))
    return {"status": primary["status"], "coverage": C.coverage(primary["paired_episode_denominator"], primary["outcome_available"]),
            "notes": ["churn = symmetric_churn + abs(net_loss_bias); signed net bias is reported separately.",
                "A replicates and shifted-A placebo measure sensitivity under their own perturbations; they do not identify a universal noise correction.",
                "Physical divergence onset is not inferred from outcome flips."],
            "tables": {"pairs": output, "decomposition": summaries, "lever_minus_controls": direct}}


def main():
    C.cli("churn", analyze)


if __name__ == "__main__":
    main()
