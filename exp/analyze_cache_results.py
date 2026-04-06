"""Analyze cache experiment results and select top combos.

Usage:
    uv run exp/analyze_cache_results.py \
        --state-file configs/cache_runs/phase1/experiment_state.json \
        --output configs/cache_runs/phase1/analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _extract_combo_and_weights(run_id: str) -> tuple[str, str]:
    """Extract combo key and weight id from run_id.

    E.g. 'phase1_run_001_a_rrf_w1' -> ('a_rrf', 'w1')
    """
    # Pattern: phase*_run_NNN_<combo>_<strategy>_<weight>
    match = re.match(r"phase\d+(?:_\d+)?_run_\d+_(.+)_(w\d+|f\d+|p[\d.]+)$", run_id)
    if match:
        combo_strategy = match.group(1)
        weight_id = match.group(2)
        return combo_strategy, weight_id
    return run_id, ""


def main():
    parser = argparse.ArgumentParser(description="Analyze cache experiment results")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--output", required=True, help="Output analysis JSON")
    args = parser.parse_args()

    data = json.loads(Path(args.state_file).read_text())

    # Filter completed runs with success_rate
    completed = [d for d in data if d["status"] == "done" and d.get("success_rate") is not None]
    if not completed:
        print("No completed runs with success_rate found")
        return

    # Sort by success_rate descending
    completed.sort(key=lambda d: d["success_rate"], reverse=True)

    # Print ranking table
    print(f"\n{'Rank':<6}{'Run ID':<50}{'Success Rate':<15}{'Exit Code':<12}")
    print("-" * 83)
    for i, d in enumerate(completed):
        sr = d["success_rate"]
        print(f"{i+1:<6}{d['run_id']:<50}{sr:<15.4f}{d.get('exit_code', 'N/A'):<12}")

    # Group by combo to find best weight per combo
    combos: dict[str, list[dict]] = {}
    for d in completed:
        combo_key, weight_id = _extract_combo_and_weights(d["run_id"])
        combos.setdefault(combo_key, []).append({**d, "_weight_id": weight_id})

    # Best per combo
    combo_best = []
    print(f"\n{'Combo':<20}{'Best Weight':<15}{'Success Rate':<15}")
    print("-" * 50)
    for combo_key, runs in sorted(combos.items()):
        best = max(runs, key=lambda r: r["success_rate"])
        combo_best.append({
            "combo": combo_key,
            "best_weights": _parse_weights_from_yaml(best["yaml_path"]),
            "best_run_id": best["run_id"],
            "success_rate": best["success_rate"],
        })
        print(f"{combo_key:<20}{best['_weight_id']:<15}{best['success_rate']:<15.4f}")

    combo_best.sort(key=lambda x: x["success_rate"], reverse=True)

    # Top 3 for Phase 1.5
    top3 = combo_best[:3]
    print(f"\nTop 3 combos for Phase 1.5:")
    for entry in top3:
        print(f"  {entry['combo']}: {entry['success_rate']:.4f}")

    # Output analysis JSON
    analysis = {
        "all_runs": completed,
        "combo_best": combo_best,
        "top3": top3,
        "best": combo_best[0] if combo_best else None,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(analysis, indent=2))
    print(f"\nAnalysis saved to {args.output}")


def _parse_weights_from_yaml(yaml_path: str) -> dict[str, float]:
    """Extract fusion weights from a YAML config file."""
    try:
        import yaml

        data = yaml.safe_load(Path(yaml_path).read_text())
        keys = data.get("keys", {})
        return {
            "vision_0": keys.get("vision_0", {}).get("weight", 0.0),
            "vision_1": keys.get("vision_1", {}).get("weight", 0.0),
            "prompt_emb": keys.get("prompt_emb", {}).get("weight", 0.0),
            "robot_state": keys.get("robot_state", {}).get("weight", 0.0),
        }
    except Exception:
        return {}


if __name__ == "__main__":
    main()
