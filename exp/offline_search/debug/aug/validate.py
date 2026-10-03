"""Noise-aware live MISS/shadow agreement report, using declared valid channels."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import reader


def agreement(arm):
    previous = arm.cache_enabled
    arm.cache_enabled = False
    try:
        return _agreement(arm)
    finally:
        arm.cache_enabled = previous


def _agreement(arm):
    decisions = arm.decisions()
    valid = arm.server_meta.get("valid_action_dims")
    if not valid:
        return dict(status="unavailable", reason="manifest valid_action_dims missing")
    ids = decisions.loc[decisions.policy_calls.gt(0), "decision_id"].tolist() if "policy_calls" in decisions else []
    result = dict(arm=arm.arm_name, live_miss_decisions=len(ids), valid_action_dims=valid)
    live_errors, noise_errors = [], []
    for start in range(0, len(ids), 32):
        selected = ids[start:start + 32]
        try:
            live = arm.decision_arrays(["policy_chunk"], selected)["policy_chunk"]
            shadow = arm.aug("policy_shadow", selected)["chunk"]
        except KeyError as exc:
            return dict(result, status="unavailable", reason=str(exc))
        live, shadow = live[:, :, valid], shadow[:, :, valid]
        finite = np.isfinite(live).all(axis=(1, 2)) & np.isfinite(shadow).all(axis=(1, 2))
        live_errors.extend(np.mean((live[finite].astype(float) - shadow[finite]) ** 2, axis=(1, 2)).tolist())
    parts = arm.aug("policy_draws")
    sampled_ids = parts.get("decision_id", np.array([], str)).astype(str).tolist()
    for start in range(0, len(sampled_ids), 32):
        selected = sampled_ids[start:start + 32]
        draws = arm.aug("policy_draws", selected)["chunks"][:, :, :, valid]
        base = arm.aug("policy_shadow", selected)["chunk"][:, :, valid]
        noise_errors.extend(np.mean((draws.astype(float) - base[:, None]) ** 2, axis=(1, 2, 3)).tolist())
    result.update(finite_live_misses=len(live_errors), sampled_noise_decisions=len(noise_errors),
                  live_shadow_mse=float(np.mean(live_errors)) if live_errors else None,
                  policy_policy_mse=float(np.mean(noise_errors)) if noise_errors else None)
    if not live_errors or not noise_errors:
        return dict(result, status="unavailable", reason="finite live MISS chunks and independent policy draws both required")
    mean_live, mean_noise = float(np.mean(live_errors)), float(np.mean(noise_errors))
    result.update(status="available", mse_ratio=mean_live / mean_noise if mean_noise else None,
                  excess_mse=mean_live - mean_noise,
                  interpretation="Descriptive agreement check: noise sample and MISS population can differ. No universal parity tolerance or statistical PASS is inferred.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    for name in args.arms:
        arm = reader.open_arm(args.run_root, name)
        result = agreement(arm)
        target = arm.debug_dir / "aug/agreement.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
