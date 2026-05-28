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
import json
import logging
import threading
import time
from pathlib import Path

from openpi.conductor import ConductorDriver, ServerEndpoint, WorkerAgent, WorkerSpec

from exp.weighted_sum.init_holdout import held_out_inits
from exp.weighted_sum.weight_search_strategy import WeightSearchStrategy


def _parse_ids(spec: str) -> list[int]:
    """Parse "0-9" or "0,1,2" into a list of ints."""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x != ""]


def main():
    # Surface driver/scheduler INFO (stage activation, dispatch, results) — the
    # conductor modules log via module loggers; without basicConfig only WARNING+
    # shows, which hides the whole scheduling trace.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
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
    ap.add_argument("--workers", type=int, default=48, help="worker processes on this client machine")
    ap.add_argument("--gpus", type=int, default=8, help="GPUs to round-robin workers across (EGL slots)")
    ap.add_argument(
        "--conda-env", default="",
        help="conda prefix/name for the worker env (e.g. /scratch/zixuans8/libero_sim); "
        "empty = spawn with the agent's own python",
    )
    ap.add_argument("--bind-host", default="127.0.0.1", help="driver pull-server bind host (workers connect here)")
    ap.add_argument(
        "--per-step-out", default="",
        help="if set, write driver.per_step_rows (per-step hit_type / cp1_score) to this JSONL "
        "after the run — needed for the threshold-pareto inference_ratio summary and warmup score "
        "collection. Empty = skip (default; episode SR journal unaffected).",
    )
    ap.add_argument(
        "--eval-concurrency", type=int, default=2,
        help="max eval yamls a server activates simultaneously (scheduler). 1=tightest (least GPU mem, "
        "but end-of-yaml straggler bubble idles workers); 2=fills the bubble (same-keybuilder yamls share "
        "the backend via BackendPool, so ~no extra mem). See docs/experiments/conductor_tutorial.md §12.1.",
    )
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
        bind_host=args.bind_host,
        scheduler_kwargs={"eval_concurrency": args.eval_concurrency},
    )

    # Single-client-machine launch: the driver's pull server + stage loop run on
    # a background thread while the main thread owns the WorkerAgent. Driver,
    # agent, and all workers live on this host; only the worker->server infer
    # calls go to the remote inference server(s).
    driver_thread = threading.Thread(target=driver.run, daemon=True)
    driver_thread.start()
    while driver.port is None:
        time.sleep(0.05)
    print(f"[run_phase2] driver pull port = {driver.port}", flush=True)

    specs = [
        WorkerSpec(
            worker_id=f"w{i}",
            server_key=servers[i % len(servers)].key,
            gpu_id=str(i % args.gpus),
            conda_env=args.conda_env,
        )
        for i in range(args.workers)
    ]
    agent = WorkerAgent(specs, driver_host=args.bind_host, driver_port=driver.port)
    agent_thread = threading.Thread(target=agent.run, daemon=True)
    agent_thread.start()
    print(
        f"[run_phase2] spawned {len(specs)} workers "
        f"(conda_env={args.conda_env or 'none'}, gpus={args.gpus}, server={servers[0].key})",
        flush=True,
    )

    try:
        driver_thread.join()  # returns when all stages are done
    finally:
        agent.stop()
    print("[run_phase2] all stages done", flush=True)

    if args.per_step_out:
        rows = driver.per_step_rows
        out = Path(args.per_step_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[run_phase2] wrote {len(rows)} per-step rows to {out}", flush=True)


if __name__ == "__main__":
    main()
