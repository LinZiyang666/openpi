"""Eval driver for the RoboCasa365 weighted-sum search: ONE (server slot, config).

One invocation runs one search cell: 13 eval tasks x N trials of pure-cache
episodes against a single already-running cache server, over the conductor
(journal + resume + watchdogs), with workers on this machine (``--role all``).
The outer slot scheduler (run from the operator's machine over tether) is
responsible for restarting the server with the cell's YAML before invoking
this driver and for tearing it down afterwards — the GR00T serving stack has
no bundle hot-swap (``allow_dynamic_bundles=False``), so config identity is
carried by the server process, and this driver only ever sees one endpoint.

Deviations from ``run_collect.py`` (the collection twin), each deliberate:

- ``--workers`` WorkerSpecs all bind to the single slot server (collection
  binds one worker per server, D-L). Concurrency == worker count: the
  scheduler has no per-server dispatch throttle (``server_capacities`` only
  weights ``assign_servers``), so sizing the worker fleet IS the concurrency
  control — the capacity probe fixed it at 4 per slot.
- ctl stays no-op: run_collect's "eval needs the real ctl" note assumes the
  hot-swap topology where the driver pushes YAMLs; under static per-process
  configs the driver has no bundle operation to perform, and the runner's
  per-connection ``select_bundle("default")`` is the idempotent startup-config
  bind (G2-verified).
- run-plan ``collect_root`` is the absolute sentinel below: nothing is
  collected server-side, but the run-plan JSON stays the executable source of
  the expected-UID set for auditing.
- episode identity: ``run_id = ws1-<cid>__l<layout>s<style>_<teacher>`` so the
  search cell is part of every ``task_uid`` (framework plan §4.3: identity
  must live in yaml_id, ``extra`` is outside ``make_task_uid``).

Example (one cell, invoked by the slot scheduler on timan107)::

    .venv/bin/python -m exp.robocasa365.run_ws_search \
        --teacher groot_tp --server ziyanglin.com:23160 \
        --cid iso_robot_state --env-config exp/robocasa365/config/ws_search_timan107.env \
        --gpu-ids 0,1,2
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import pathlib
import signal
import sys
import threading
import time

from openpi.conductor import ServerEndpoint, WorkerAgent, WorkerSpec
from openpi.conductor.driver import ConductorDriver, assign_servers

from exp.robocasa365.run_collect import (
    RobocasaCollectStrategy,
    _NoOpCtl,
    build_run_plan,
    load_env_config,
    parse_tasks,
    robocasa_spawn_fn,
    validate_teacher_endpoints,
    write_run_plan,
)

# The frozen 13-task eval subset (both-teachers intersection, T5 manifest set).
DEFAULT_EVAL_TASKS = ",".join(
    (
        "CloseBlenderLid",
        "CloseFridge",
        "CoffeeSetupMug",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "SlideDishwasherRack",
        "TurnOnSinkFaucet",
    )
)

# Absolute sentinel: run-plan hashing requires an absolute collect root, but an
# eval server writes nothing. Absolute + obviously-not-a-real-root by content.
EVAL_NO_COLLECT_ROOT = "/tmp/WS_SEARCH_EVAL_NO_COLLECT"


class WsSearchStrategy(RobocasaCollectStrategy):
    """Collection strategy with the search-cell id folded into the identity."""

    def __init__(self, *, cid: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cid = str(cid)
        # Full replacement, not a prefix: the parent's run_id says "collect_",
        # which would mislabel every eval task_uid and journal filename.
        self.run_id = f"ws1-{self._cid}__l{self._layout}s{self._style}_{self._teacher}"


def summarize_journal(journal_path: pathlib.Path, expected_uids: list[str] | None = None) -> dict:
    """Terminal-record SR summary reconciled against the run-plan's UID set.

    Robot failures score 0; infra errors count as ERR (excluded from the SR
    denominator, never as 0); a uid with NO terminal record at all — the
    retry-exhausted case the driver never journals — counts as MISSING. A cell
    is ``complete`` only when nothing is missing and nothing erred; the
    orchestrator keys its done/skip decision on that flag, so a half-dead cell
    gets rerun instead of silently entering the ranking.
    """
    per_task: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    seen: dict[str, dict] = {}
    for line in journal_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # Last accepted terminal record per uid wins (resume may append).
        if rec.get("accepted") is False:
            continue
        seen[rec["task_uid"]] = rec
    for rec in seen.values():
        task = rec["yaml_id"].rsplit("__", 1)[-1]
        if rec.get("error"):
            per_task[task]["err"] += 1
        elif rec.get("success"):
            per_task[task]["succ"] += 1
        else:
            per_task[task]["fail"] += 1
    for uid in expected_uids or []:
        if uid not in seen:
            yaml_id = uid.rsplit(":", 3)[0]
            per_task[yaml_id.rsplit("__", 1)[-1]]["missing"] += 1
    tasks = {
        t: {
            "succ": c["succ"],
            "fail": c["fail"],
            "err": c["err"],
            "missing": c["missing"],
            "n_scored": c["succ"] + c["fail"],
            "sr": (c["succ"] / (c["succ"] + c["fail"])) if (c["succ"] + c["fail"]) else None,
        }
        for t, c in sorted(per_task.items())
    }
    srs = [v["sr"] for v in tasks.values() if v["sr"] is not None]
    n_err = sum(v["err"] for v in tasks.values())
    n_missing = sum(v["missing"] for v in tasks.values())
    return {
        "tasks": tasks,
        # Macro average over tasks (equal task weight, plan D1) — deliberately
        # NOT episode-pooled, and named accordingly.
        "macro_sr": sum(srs) / len(srs) if srs else None,
        "n_err": n_err,
        "n_missing": n_missing,
        "complete": n_err == 0 and n_missing == 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True, choices=("pi05", "groot_tp"))
    ap.add_argument("--server", required=True, help="host:port of THE slot server (exactly one)")
    ap.add_argument("--cid", required=True, help="search cell id (matches the emitted YAML stem)")
    ap.add_argument("--tasks", default=DEFAULT_EVAL_TASKS)
    ap.add_argument("--episodes", type=int, default=8, help="trials per task (owner-ruled 8)")
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=1_000_000, help="eval segment (plan D-E)")
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4, help="worker fleet = the slot's concurrency")
    ap.add_argument("--gpu-ids", default="0", help="comma-separated CUDA slots, round-robin over workers")
    ap.add_argument("--journal-dir", default="", help="default: exp/robocasa365/data/ws_search/<teacher>/")
    ap.add_argument("--env-config", required=True)
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--episode-timeout-s", type=float, default=1800.0)
    ap.add_argument("--connect-deadline-s", type=float, default=600.0)
    ap.add_argument("--episode-deadline-s", type=float, default=1500.0)
    ap.add_argument("--terminate-grace-s", type=float, default=60.0)
    args = ap.parse_args()

    # tmux kill-session delivers SIGHUP, the orchestrator's stuck-kill SIGTERM;
    # both must run the finally-block so agent.stop() reaps the worker process
    # groups instead of orphaning them (start_new_session=True workers survive
    # a plain process-group HUP; the probe-era orphan incident is the lesson).
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(143))

    env_config = load_env_config(args.env_config)
    host, _, port = args.server.rpartition(":")
    slot = ServerEndpoint(host, int(port))
    validate_teacher_endpoints(args.teacher, [slot], env_config)

    tasks = parse_tasks(args.tasks, args.episodes)
    strategy = WsSearchStrategy(
        cid=args.cid,
        teacher=args.teacher,
        layout=args.layout,
        style=args.style,
        base_seed=args.base_seed,
        replan_steps=args.replan_steps,
        tasks=tasks,
    )
    run_id = strategy.run_id

    data_dir = pathlib.Path(__file__).resolve().parent / "data" / "ws_search" / args.teacher
    journal_dir = pathlib.Path(args.journal_dir) if args.journal_dir else data_dir
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal_path = journal_dir / f"journal_{run_id}.jsonl"
    run_plan_path = journal_dir / f"run_plan_{run_id}.json"

    yaml_weights = {yid: n for yid, (_, n) in zip(strategy.yaml_ids, tasks)}
    server_capacities = {slot.key: args.workers}

    assignment = assign_servers(yaml_weights, [slot], None, server_capacities)
    graph = strategy.plan(sorted(yaml_weights), assignment)
    run_plan = build_run_plan(strategy, graph, EVAL_NO_COLLECT_ROOT)
    write_run_plan(run_plan_path, run_plan)
    print(f"[ws_search] run-plan {run_plan_path} plan_hash={run_plan['plan_hash']}", flush=True)

    driver = ConductorDriver(
        strategy,
        yaml_weights=yaml_weights,
        servers=[slot],
        journal_path=str(journal_path),
        ctl_factory=lambda _server: _NoOpCtl(),  # static per-process config: nothing for ctl to do
        episode_timeout_s=args.episode_timeout_s,
        bind_host=args.bind_host,
        server_capacities=server_capacities,
    )
    driver_thread = threading.Thread(target=driver.run, daemon=True)
    driver_thread.start()
    while driver.port is None:
        time.sleep(0.05)
    print(f"[ws_search] driver pull port = {driver.port}", flush=True)

    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    specs = [
        WorkerSpec(worker_id=f"w{i}", server_key=slot.key, gpu_id=gpu_ids[i % len(gpu_ids)])
        for i in range(args.workers)
    ]
    spawn = functools.partial(
        robocasa_spawn_fn,
        worker_python=env_config["WORKER_PYTHON"],
        robocasa_cwd=env_config["ROBOCASA_CWD"],
        repo_root=env_config["REPO_ROOT"],
        egl_lib_dir=env_config["EGL_LIB_DIR"],
        egl_vendor_dir=env_config["EGL_VENDOR_DIR"],
        teacher=args.teacher,
        connect_deadline_s=args.connect_deadline_s,
        episode_deadline_s=args.episode_deadline_s,
        terminate_grace_s=args.terminate_grace_s,
    )
    agent = WorkerAgent(specs, driver_host=args.bind_host, driver_port=driver.port, spawn_fn=spawn)
    agent_thread = threading.Thread(target=agent.run, daemon=True)
    agent_thread.start()
    print(f"[ws_search] agent supervising {len(specs)} worker(s) -> {slot.key}", flush=True)

    try:
        while driver_thread.is_alive():
            driver_thread.join(timeout=5.0)
    finally:
        agent.stop()

    summary = summarize_journal(journal_path, expected_uids=list(run_plan["uids"]))
    summary_path = journal_dir / f"summary_{run_id}.json"
    summary_path.write_text(json.dumps({"cid": args.cid, "teacher": args.teacher, **summary}, indent=1))
    print(
        f"[ws_search] {'DONE' if summary['complete'] else 'INCOMPLETE'} cid={args.cid} "
        f"macro_sr={summary['macro_sr']} n_err={summary['n_err']} "
        f"n_missing={summary['n_missing']} -> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
