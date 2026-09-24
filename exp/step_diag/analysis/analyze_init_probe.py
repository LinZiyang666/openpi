"""Validate and summarize fixed-observation probes, clustering uncertainty by episode.

Only accepted terminal conductor attempts enter the analysis. NPZ hashes, loop
shapes, repeated-input equality, cache/RNG preservation, and the shared endpoint
Euler identity are checked before reporting input-to-output distance ratios.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from exp.step_diag.serve_init_probe import action_rmse


def analyze(probe_dir: Path, client_dir: Path, server_dir: Path | None = None, *,
            allow_incomplete: bool = False) -> dict:
    """Reconcile raw probes with accepted episodes and independently recompute metrics."""
    accepted = {}
    for path in client_dir.rglob("journal_*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("accepted") and row.get("status") in ("done", "failed"):
                key = (row["task_uid"], int(row.get("attempt", 1)))
                if key in accepted:
                    raise ValueError(f"duplicate accepted identity: {key}")
                accepted[key] = row
    if not accepted:
        raise ValueError("no accepted terminal episodes")
    if len({uid for uid, _ in accepted}) != len(accepted):
        raise ValueError("multiple accepted attempts for an episode")
    expected = {}
    for path in client_dir.rglob("launch_*.json"):
        launch = json.loads(path.read_text())
        for item in launch["expected"]:
            uid = item["task_uid"]
            if uid in expected and expected[uid] != item:
                raise ValueError("conflicting expected episode identities")
            expected[uid] = item
    accepted_uids = {uid for uid, _ in accepted}
    if expected:
        if not accepted_uids <= set(expected):
            raise ValueError("accepted episode outside requested cohort")
        if not allow_incomplete and set(expected) != accepted_uids:
            raise ValueError("accepted cohort does not equal requested cohort")
    if server_dir is not None:
        manifest = json.loads((server_dir / "manifest_shadow.json").read_text())
        server_rows = defaultdict(list)
        for path in server_dir.glob("rows_*.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                server_rows[(row["task_uid"], int(row["attempt"]))].append(row)
        for identity, terminal in accepted.items():
            evidence = server_rows[identity]
            ends = [r for r in evidence if r.get("status") == "finalize"]
            decisions = [r for r in evidence if r.get("status") != "finalize"]
            if len(ends) != 1 or not ends[0]["terminal"] or ends[0]["outcome"] != terminal["success"]:
                raise ValueError(f"server/worker terminal mismatch: {identity}")
            if ends[0]["stamp_mismatch"] or ends[0]["config_sha"] != manifest["config_sha"]:
                raise ValueError(f"configuration identity mismatch: {identity}")
            if len(decisions) != ends[0]["n_decisions"]:
                raise ValueError("missing executed decisions")
            if any(r["status"] != "ok" or r["executed_steps"] != r["k_full"]
                   or r["n_stage3_calls"] != 1 for r in decisions):
                raise ValueError("executed policy was not the full teacher")
    rows = [json.loads(line) for line in (probe_dir / "probe.jsonl").read_text().splitlines()]
    kept, rejected, seen = [], [], set()
    max_grid_error = 0.0
    for row in rows:
        identity = (row["task_uid"], int(row["attempt"]))
        if identity not in accepted:
            rejected.append(row)
            continue
        if expected:
            item = expected[row["task_uid"]]
            if any(row[k] != item[k] for k in ("task", "env_seed", "init_idx")):
                raise ValueError("probe episode identity differs from launch")
        key = (*identity, row["decision_idx"])
        if key in seen:
            raise ValueError(f"duplicate observation: {key}")
        seen.add(key)
        path = probe_dir / row["arrays"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["arrays_sha256"]:
            raise ValueError(f"array checksum mismatch: {path}")
        if not row["rng_preserved"] or not row["payload_preserved"]:
            raise ValueError("state isolation check failed")
        if (row["winner_task"] != row["task"] or row["winner_task"] != row["random_task"]
                or row["winner_trajectory"] == row["random_trajectory"]):
            raise ValueError("invalid random-cache control")
        with np.load(path, allow_pickle=False) as arrays:
            for init in ("retrieved", "random_same_task", "zero", "gaussian"):
                one, two = arrays[f"{init}_n1"], arrays[f"{init}_n2"]
                if one.shape[0] != 2 or two.shape[0] != 3 or not np.isfinite(one).all() or not np.isfinite(two).all():
                    raise ValueError("invalid loop trajectory")
                if not np.array_equal(one[0], two[0]):
                    raise ValueError("N comparison changed initialization")
                error = float(np.max(np.abs(two[1] - (one[0] + one[1]) / 2)))
                max_grid_error = max(max_grid_error, error)
                if error > 2e-5:
                    raise ValueError(f"shared endpoint Euler identity failed: {error}")
            if len(row["metrics"]) != 6:
                raise ValueError("missing counterfactual metric")
            for m in row["metrics"]:
                a, b = arrays[f"retrieved_n{m['n']}"], arrays[f"{m['control']}_n{m['n']}"]
                for label, idx in (("input", 0), ("first", 1), ("final", -1)):
                    if not np.isclose(action_rmse(a[idx], b[idx]), m[f"{label}_rmse"], atol=1e-9, rtol=1e-7):
                        raise ValueError("stored metric differs from raw arrays")
                initial = action_rmse(a[0], b[0])
                for label, idx in (("first", 1), ("final", -1)):
                    ratio = action_rmse(a[idx], b[idx]) / initial if initial > 1e-12 else None
                    stored = m[f"{label}_ratio"]
                    if ((ratio is None) != (stored is None)
                            or (ratio is not None and not np.isclose(ratio, stored, atol=1e-9, rtol=1e-7))):
                        raise ValueError("stored ratio differs from raw arrays")
                if m["repeat_max_abs"] > 1e-5:
                    raise ValueError("same-input numerical repeat failed")
        kept.append(row)
    missing = set(accepted) - {(r["task_uid"], int(r["attempt"])) for r in kept}
    if missing:
        raise ValueError(f"accepted episodes missing probes: {len(missing)}")
    if server_dir is not None:
        contract = json.loads((probe_dir / "probe_contract.json").read_text())
        for identity in accepted:
            end = next(r for r in server_rows[identity] if r.get("status") == "finalize")
            wanted = {idx for idx in contract["decisions"] if idx < end["n_decisions"]}
            actual = {r["decision_idx"] for r in kept if (r["task_uid"], int(r["attempt"])) == identity}
            if actual != wanted:
                raise ValueError(f"incomplete scheduled observation coverage: {identity}")
    episode_groups = defaultdict(list)
    counts = defaultdict(lambda: {"episodes": set(), "observations": 0})
    for row in kept:
        counts[row["task"]]["episodes"].add(row["task_uid"])
        counts[row["task"]]["observations"] += 1
        for m in row["metrics"]:
            for phase in ("first", "final"):
                if m[f"{phase}_ratio"] is not None:
                    episode_groups[(row["task"], m["n"], m["control"], phase, row["task_uid"])].append(m[f"{phase}_ratio"])
    groups = defaultdict(list)
    for (task, n, control, phase, _), values in episode_groups.items():
        groups[(task, n, control, phase)].append(float(np.mean(values)))
    rng = np.random.default_rng(20260922)
    summaries = []
    for (task, n, control, phase), values in sorted(groups.items()):
        a = np.asarray(values)
        boot = rng.choice(a, (5000, len(a)), replace=True).mean(axis=1)
        summaries.append({"task": task, "n": n, "control": control, "phase": phase,
                          "mean_ratio": float(a.mean()), "episode_values": values,
                          "ci95": np.quantile(boot, [0.025, 0.975]).tolist()})
    return {"accepted_episodes": len(accepted), "observations": len(kept),
            "server_identity_and_full_execution_checked": server_dir is not None,
            "requested_cohort_checked": bool(expected),
            "requested_episodes_in_launched_cells": len(expected) if expected else None,
            "launched_cohort_complete": set(expected) == accepted_uids if expected else None,
            "missing_requested_episodes": sorted(set(expected) - accepted_uids),
            "partial_analysis": allow_incomplete,
            "excluded_unaccepted_observations": len(rejected), "max_euler_identity_error": max_grid_error,
            "tasks": {k: {"episodes": len(v["episodes"]), "observations": v["observations"]} for k, v in counts.items()},
            "summary": summaries, "metric": "output RMSE / input RMSE, first 5 actions x 12 active normalized dimensions",
            "uncertainty": "95% bootstrap CI over episode means within task; 5000 draws",
            "scope": "Fixed teacher-visited observations; no success-rate claim for counterfactual arms."}


def plot(result: dict, output: Path) -> None:
    """Plot episode-level initialization sensitivity with a cancellation reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted(result["tasks"])
    fig, axes = plt.subplots(len(tasks), 2, figsize=(11, 3.6 * len(tasks)), squeeze=False)
    controls = ("random_same_task", "zero", "gaussian")
    for i, task in enumerate(tasks):
        for j, phase in enumerate(("first", "final")):
            ax = axes[i, j]
            for n, offset, color in ((1, -0.17, "#2878a8"), (2, 0.17, "#e18b36")):
                subset = [next(r for r in result["summary"] if (r["task"], r["phase"], r["n"], r["control"]) ==
                               (task, phase, n, control)) for control in controls]
                x = np.arange(3) + offset
                values = [r["mean_ratio"] for r in subset]
                ax.bar(x, values, width=.31, label=f"N={n}", color=color, alpha=.8)
                ax.errorbar(x, values, yerr=np.array([[v-r["ci95"][0], r["ci95"][1]-v]
                                                     for v, r in zip(values, subset)]).T,
                            fmt="none", ecolor="#333333", capsize=3)
            ax.axhline(1, color="#666666", ls="--", lw=1)
            ax.set_xticks(range(3), ["Other same-task cache", "Zero", "Gaussian"])
            ax.set_ylabel("Output distance / input distance")
            ax.set_title(f"{task}\nAfter {phase} update")
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False)
    fig.suptitle("Does restarting erase cache initialization?\n0 = cancellation; 1 = unchanged distance", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    """Write validated JSON and a standalone PNG from downloaded run evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--client-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-figure", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Explicitly analyze only completed episodes of a stopped, incomplete cohort.")
    args = parser.parse_args()
    result = analyze(args.probe_dir, args.client_dir, args.probe_dir.parent / "server",
                     allow_incomplete=args.allow_incomplete)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_figure.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.out_figure)
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2))


if __name__ == "__main__":
    main()
