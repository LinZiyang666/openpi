"""Phase-2 entry point: construct WeightSearchStrategy + ConductorDriver and run.

Thin glue. Discovers the emitted eval YAMLs, derives held-out init states from
the init map, and drives the conductor engine. See conductor_tutorial.md §5.

Example:
    uv run exp/weighted_sum/run_phase2.py \
        --yaml-dir exp/weighted_sum/config/phase2_grid \
        --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json \
        --journal exp/weighted_sum/data/phase2/journal.jsonl \
        --servers host:8001,host:8002 --task-ids 0-9 --eval-trials 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openpi.conductor import ConductorDriver, ServerEndpoint

from exp.weighted_sum.init_holdout import held_out_inits
from exp.weighted_sum.weight_search_strategy import WeightSearchStrategy


def _parse_ids(spec: str) -> list[int]:
    """Parse "0-9" or "0,1,2" into a list of ints."""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x != ""]


def main():
    ap = argparse.ArgumentParser(description="Run Phase-2 weighted-sum weight search")
    ap.add_argument("--yaml-dir", required=True)
    ap.add_argument("--init-map", required=True, help="libero_spatial_init_map.json (leak guard)")
    ap.add_argument("--journal", required=True)
    ap.add_argument("--servers", required=True, help="comma-separated host:port endpoints")
    ap.add_argument("--task-ids", default="0-9")
    ap.add_argument("--eval-trials", type=int, default=20)
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--total-inits", type=int, default=50)
    ap.add_argument("--episode-timeout-s", type=int, default=1800)
    args = ap.parse_args()

    yaml_dir = Path(args.yaml_dir)
    yaml_ids = sorted(p.stem for p in yaml_dir.glob("*.yaml"))
    if not yaml_ids:
        raise SystemExit(f"no eval YAMLs in {yaml_dir}")

    task_ids = _parse_ids(args.task_ids)
    holdout = held_out_inits(args.init_map, task_ids, total_inits_per_task=args.total_inits)

    strategy = WeightSearchStrategy(
        task_ids=task_ids,
        eval_trials=args.eval_trials,
        task_suite_name=args.task_suite,
        yaml_dir=str(yaml_dir),
        held_out_inits=holdout,
    )

    servers = []
    for spec in args.servers.split(","):
        host, port = spec.rsplit(":", 1)
        servers.append(ServerEndpoint(host, int(port)))

    from examples.libero.episode_runner import default_client_factory

    driver = ConductorDriver(
        strategy,
        yaml_weights={yid: 100 for yid in yaml_ids},
        servers=servers,
        journal_path=args.journal,
        ctl_factory=default_client_factory,
        episode_timeout_s=args.episode_timeout_s,
    )
    driver.run()


if __name__ == "__main__":
    main()
