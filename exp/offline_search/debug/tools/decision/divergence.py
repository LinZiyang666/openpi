"""Cache/served versus fresh shadow policy, offset curves and sampling floor."""
import numpy as np

from . import common as C
from .metrics import action_metrics, noise_floor


def analyze(arm, args=None):
    """Compare proposal futures and independently sampled policy noise on the same input."""
    decisions, episodes = C.inputs(arm)
    ids = decisions.decision_id.tolist()
    live = C.arrays(arm, ["served_chunk", "cache_chunk"], ids)
    shadow = C.pick(C.augmentation(arm, "policy_shadow", ids), "chunk", "policy_chunk")
    if shadow is None:
        raise C.Unavailable("policy_shadow chunk unavailable")
    draws = C.pick(C.augmentation(arm, "policy_draws", ids), "chunks")
    output, noise = [], []
    library_size = C.metadata(arm).get("library_size")
    for i, row in enumerate(decisions.to_dict("records")):
        floor = {}
        if draws is not None and np.asarray(draws[i]).ndim == 3:
            # Include the independent base shadow sample as the fourth draw.
            sampled = np.concatenate([shadow[i][None], draws[i]], axis=0)
            floor = noise_floor(arm, sampled)
            noise.extend(dict(C.identity(row), future_offset=k, policy_pair_mse=v, variance_floor=v / 2)
                         for k, v in floor.items())
        for key, chunk in live.items():
            limit = C.number(C.field(row, key + "_valid_horizon", "served_len" if key == "served_chunk" else "cache_valid_horizon"))
            candidate = chunk[i][:int(limit)] if limit is not None else chunk[i]
            for metric in action_metrics(arm, candidate, shadow[i]):
                offset = metric["future_offset"]
                n_applied = C.number(row.get("n_applied"))
                output.append(dict(C.identity(row), comparison=key + "_vs_policy_shadow", **metric,
                    blind_age_controls=C.number(row.get("blind_age_controls")), lib=C.field(row, "lib"),
                    library_size=library_size,
                    src=row.get("src"), executed_offset=(offset < n_applied) if n_applied is not None else None,
                    horizon_status="declared" if limit is not None else "array_horizon_validity_unavailable",
                    policy_pair_mse=floor.get(offset), noise_adjusted_mse=metric["mse"] - floor[offset] / 2 if offset in floor else None))
    available = len(set(row["decision_id"] for row in output))
    return {"status": "available" if available else "unavailable", "coverage": C.coverage(len(decisions), available),
            "noise_floor_coverage": C.coverage(len(decisions), len(set(row["decision_id"] for row in noise)), "hashed sample only"),
            "comparison_coverage": {key: C.coverage(len(decisions), len({r["decision_id"] for r in output if r["comparison"].startswith(key + "_vs")}))
                                    for key in ("served_chunk", "cache_chunk")},
            "episode_denominator": len(episodes), "notes": ["Future offsets are proposal comparisons; executed_offset marks the applied prefix.",
             "Noise-adjusted MSE subtracts half policy-pair MSE; negative finite-sample values are preserved.",
             "Disagreement is not a physical failure or a policy rescue effect."],
            "tables": {"offsets": output, "noise_floor": noise, "curves": C.summarize(output,
                ["comparison", "stage", "blind_age_controls", "lib", "library_size", "src", "future_offset"], ["rms", "motion_rms", "gripper_flip", "noise_adjusted_mse"])}}


def main():
    C.cli("divergence", analyze)


if __name__ == "__main__":
    main()
