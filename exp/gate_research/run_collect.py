"""Gate-research collection entry point: eval-only conductor run + gate rows dump.

Thin glue mirroring ``exp/weighted_sum/run_phase2.py`` but for GATE ("search or
not") research: it drives a directory of *self-contained* eval YAMLs (baked
thresholds + normalizer + prebuilt library, ``gate: always_search``) through the
conductor with NO warmup stage, evaluating every task from init states
``0..eval_trials-1`` (no held-out leak guard — gate research wants the full init
distribution). Each CP1 inference step's verdict + model input (``robot_state``)
is collected server-side (``serve_policy --export-collect-meta --collect-fields
robot_state``), rides back on ``EpisodeResult.per_step_rows``, and is dumped to
``--per-step-out`` as JSONL.

Strategy: ``WarmupEvalStrategy(skip_warmup=True)`` — the exact "inits 0..N-1, no
init map" semantics (``orig_init_state_idx = ep``), while still stamping
``extra["num_trials_per_task"]`` so the worker computes the canonical global
``episode_id``.

Example (48 workers on this client, one server with 3 replicas):
    uv run exp/gate_research/run_collect.py \
        --yaml-dir exp/gate_research/config/libero_spatial/eval \
        --journal exp/gate_research/data/libero_spatial/journal.jsonl \
        --per-step-out exp/gate_research/data/libero_spatial/gate_rows.jsonl \
        --servers HOST:8000 --workers 48 --gpus 8 \
        --task-ids 0-9 --eval-trials 50 --task-suite libero_spatial \
        --conda-env /scratch/zixuans8/libero_sim
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path

from openpi.conductor import ConductorDriver, ServerEndpoint, WorkerAgent, WorkerSpec

from exp.verdict_factor_judge.strategies.warmup_eval_strategy import WarmupEvalStrategy


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
    ap = argparse.ArgumentParser(description="Run gate-research eval-only collection")
    ap.add_argument("--yaml-dir", required=True, help="dir of self-contained eval YAMLs (one per config)")
    ap.add_argument("--journal", required=True, help="crash-safe episode resume journal path")
    ap.add_argument("--servers", required=True, help="comma-separated host:port endpoints")
    ap.add_argument("--task-ids", default="0-9")
    ap.add_argument("--eval-trials", type=int, default=50, help="episodes per task (init states 0..N-1)")
    ap.add_argument("--task-suite", default="libero_spatial")
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
        "--per-step-out", required=True,
        help="write driver.per_step_rows (per-step hit_type / cp1_score / robot_state) to this JSONL "
        "after the run — this IS the gate-research collected data.",
    )
    ap.add_argument(
        "--eval-concurrency", type=int, default=2,
        help="max eval yamls a server activates simultaneously (scheduler). 1=tightest GPU mem; "
        "2=fills the end-of-yaml straggler bubble (same-keybuilder yamls share the backend via "
        "BackendPool, so ~no extra mem).",
    )
    ap.add_argument(
        "--server-workers", default="",
        help="comma worker counts per --servers endpoint, e.g. '48' (length must match --servers). "
        "Sum overrides --workers. Empty = even round-robin across servers (legacy).",
    )
    args = ap.parse_args()

    yaml_dir = Path(args.yaml_dir)
    yaml_ids = sorted(p.stem for p in yaml_dir.glob("*.yaml"))
    if not yaml_ids:
        raise SystemExit(f"no eval YAMLs in {yaml_dir}")

    task_ids = _parse_ids(args.task_ids)

    # Eval-only: skip_warmup=True builds ONLY the eval stage per yaml (no warmup,
    # no calibration, no dependency). warmup_trials is unused under skip_warmup.
    # orig_init_state_idx = ep (0..eval_trials-1): full init distribution, no
    # held-out leak guard.
    strategy = WarmupEvalStrategy(
        task_ids=task_ids,
        warmup_trials=0,
        eval_trials=args.eval_trials,
        task_suite_name=args.task_suite,
        yaml_dir=str(yaml_dir),
        skip_warmup=True,
    )

    servers = []
    for spec in args.servers.split(","):
        host, port = spec.rsplit(":", 1)
        servers.append(ServerEndpoint(host, int(port)))

    if args.server_workers:
        counts = [int(x) for x in args.server_workers.split(",")]
        if len(counts) != len(servers):
            raise SystemExit(f"--server-workers has {len(counts)} entries but --servers has {len(servers)}")
        worker_server_keys = [s.key for s, c in zip(servers, counts) for _ in range(c)]
        server_capacities = {s.key: c for s, c in zip(servers, counts)}
    else:
        worker_server_keys = [servers[i % len(servers)].key for i in range(args.workers)]
        server_capacities = None
    n_workers = len(worker_server_keys)

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
        server_capacities=server_capacities,
    )

    driver_thread = threading.Thread(target=driver.run, daemon=True)
    driver_thread.start()
    while driver.port is None:
        time.sleep(0.05)
    print(f"[run_collect] driver pull port = {driver.port}", flush=True)

    specs = [
        WorkerSpec(
            worker_id=f"w{i}",
            server_key=worker_server_keys[i],
            gpu_id=str(i % args.gpus),
            conda_env=args.conda_env,
            task_suite_name=args.task_suite,
        )
        for i in range(n_workers)
    ]
    agent = WorkerAgent(specs, driver_host=args.bind_host, driver_port=driver.port)
    agent_thread = threading.Thread(target=agent.run, daemon=True)
    agent_thread.start()
    print(
        f"[run_collect] spawned {len(specs)} workers "
        f"(conda_env={args.conda_env or 'none'}, gpus={args.gpus}, server={servers[0].key}, "
        f"suite={args.task_suite}, yamls={len(yaml_ids)})",
        flush=True,
    )

    # Incremental append checkpoint (crash safety for the debounce supervisor):
    # per_step_writer is None here, so driver._per_step_rows grows append-only and
    # never drops rows. We periodically append the NEW tail to --per-step-out (open
    # mode "a"), so a mid-run process death loses at most the last CHECKPOINT_S of
    # rows. On journal-resume the driver starts with an empty row buffer and re-runs
    # only the not-yet-done episodes; their rows append after the pre-crash rows.
    # Offline analysis dedups by (task_uid, step_idx) keeping the latest attempt (a
    # requeued in-flight episode is the only source of duplicates). This makes the
    # collected data complete across crashes while keeping efficient episode resume.
    CHECKPOINT_S = 15
    out = Path(args.per_step_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = 0

    def _flush(fh) -> int:
        nonlocal seen
        rows = driver.per_step_rows
        n = len(rows)
        if n > seen:
            for r in rows[seen:n]:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            seen = n
        return n

    # Append mode: a resumed attempt keeps the prior attempt's rows on disk.
    with out.open("a", encoding="utf-8") as fh:
        try:
            while driver_thread.is_alive():
                _flush(fh)
                driver_thread.join(timeout=CHECKPOINT_S)
            final_n = _flush(fh)  # drain the tail written after the last checkpoint
        finally:
            agent.stop()
    print(f"[run_collect] all stages done; appended {final_n} in-run per-step rows to {out}", flush=True)


if __name__ == "__main__":
    main()
