"""CLI: tasks, prepare, run, agent and admit for production warm reset experiments.

Run ``python -m exp.warm_reset.run --help``. Model/simulator imports are lazy,
so task export can run inside a simulator island and the driver in the main venv.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import threading
from pathlib import Path

from exp.step_diag.envs import ENVS


def parser() -> argparse.ArgumentParser:
    """Construct the CLI without importing policy or simulator dependencies."""
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser(
        "tasks", help="export trusted task names in the simulator environment"
    )
    tasks.add_argument("--env", required=True, choices=sorted(ENVS))
    tasks.add_argument(
        "--task-ids", default="all", help="LIBERO task indices, comma separated"
    )
    tasks.add_argument(
        "--task-names", default="", help="RoboCasa canonical env names, comma separated"
    )
    tasks.add_argument("--episodes", required=True, type=int)
    tasks.add_argument("--init-offset", default=0, type=int)
    tasks.add_argument("--out", required=True, type=Path)
    prepare = commands.add_parser(
        "prepare", help="freeze YAMLs and the task/rollout plan; launch nothing"
    )
    prepare.add_argument("--env", required=True, choices=sorted(ENVS))
    prepare.add_argument("--base-yaml", required=True, type=Path)
    prepare.add_argument(
        "--arms",
        default="all",
        help="all existing self-family arms or comma-separated arm IDs",
    )
    prepare.add_argument("--tasks", required=True, type=Path)
    prepare.add_argument("--servers", required=True, help="host:port,host:port")
    prepare.add_argument("--server-evidence-root", required=True)
    prepare.add_argument(
        "--namespace",
        required=True,
        help="common self-noise namespace across paired arms",
    )
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--replan-steps", default=5, type=int)
    prepare.add_argument("--seed", default=7, type=int, help="LIBERO worker seed")
    prepare.add_argument(
        "--init-states-dir",
        default="",
        help="LIBERO full pool, indexed by original state ID",
    )
    prepare.add_argument(
        "--base-seed",
        type=int,
        help="RoboCasa seed base; explicitly choose the experiment segment",
    )
    prepare.add_argument("--layout", type=int)
    prepare.add_argument("--style", type=int)
    run = commands.add_parser(
        "run", help="run the conductor driver; workers connect via agent"
    )
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--bind-host", default="0.0.0.0")
    run.add_argument("--port", required=True, type=int)
    run.add_argument(
        "--concurrency",
        default=4,
        type=int,
        help="simultaneously active arms per server",
    )
    run.add_argument("--episode-timeout", default=1800.0, type=float)
    agent = commands.add_parser(
        "agent", help="launch standard simulator workers on this host"
    )
    agent.add_argument("--run-dir", required=True, type=Path)
    agent.add_argument(
        "--server", required=True, help="one inference endpoint from the plan"
    )
    agent.add_argument("--driver-host", required=True)
    agent.add_argument("--driver-port", required=True, type=int)
    agent.add_argument(
        "--gpus", required=True, help="comma-separated local renderer GPU IDs"
    )
    agent.add_argument("--workers-per-gpu", default=4, type=int)
    agent.add_argument(
        "--prefix", default=socket.gethostname(), help="must be unique across agents"
    )
    agent.add_argument(
        "--conda-env", default="", help="LIBERO worker conda environment name/path"
    )
    agent.add_argument(
        "--worker-python", default="", help="RoboCasa island-A interpreter"
    )
    agent.add_argument("--robocasa-cwd", default="")
    agent.add_argument("--egl-lib-dir", default="")
    agent.add_argument("--egl-vendor-dir", default="")
    agent.add_argument("--connect-deadline", default=60.0, type=float)
    agent.add_argument("--episode-deadline", default=1500.0, type=float)
    agent.add_argument("--terminate-grace", default=5.0, type=float)
    agent.add_argument("--max-cached-envs", default=1, type=int)
    admit = commands.add_parser(
        "admit", help="join all three evidence sources; exit 2 on rejection"
    )
    admit.add_argument("--run-dir", required=True, type=Path)
    admit.add_argument(
        "--evidence-dir",
        action="append",
        type=Path,
        help="copied server run directory; repeat for independent servers",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """Execute a requested experiment operation; no server/GPU is started implicitly."""
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.command == "tasks":
        if args.episodes < 1 or args.init_offset < 0:
            raise ValueError(
                "episodes must be positive; init offset must be nonnegative"
            )
        env = ENVS[args.env]
        if env.benchmark == "robocasa365":
            names = [
                name.strip() for name in args.task_names.split(",") if name.strip()
            ]
            if not names:
                raise ValueError("RoboCasa needs --task-names")
            pairs = list(enumerate(names))
        else:
            from libero.libero import benchmark

            suite = benchmark.get_benchmark_dict()[env.benchmark]()
            ids = (
                range(suite.n_tasks)
                if args.task_ids == "all"
                else [int(i) for i in args.task_ids.split(",")]
            )
            pairs = [(i, suite.get_task(i).language) for i in ids]
        tasks = [
            {
                "task_id": i,
                "name": name,
                "init_indices": list(
                    range(args.init_offset, args.init_offset + args.episodes)
                ),
            }
            for i, name in pairs
        ]
        with args.out.open("x") as fh:
            json.dump(tasks, fh, indent=2)
            fh.write("\n")
    elif args.command == "prepare":
        from exp.warm_reset.plan import default_arms, prepare

        plan = prepare(
            out=args.out,
            env_id=args.env,
            base_yaml=args.base_yaml,
            arms=default_arms(args.env) if args.arms == "all" else args.arms.split(","),
            tasks=json.loads(args.tasks.read_text()),
            servers=args.servers.split(","),
            evidence_root=args.server_evidence_root,
            namespace=args.namespace,
            rollout={
                "replan_steps": args.replan_steps,
                "seed": args.seed,
                "init_states_dir": args.init_states_dir,
                "base_seed": args.base_seed,
                "layout": args.layout,
                "style": args.style,
            },
        )
        print(
            json.dumps(
                {
                    "plan": str(args.out / "plan.json"),
                    "arms": len(plan["arms"]),
                    "server_evidence_dir": plan["evidence_dir"],
                },
                indent=2,
            )
        )
    elif args.command == "run":
        from exp.warm_reset.conductor import build_driver
        from exp.warm_reset.plan import read_plan

        if not 0 < args.port < 65536:
            raise ValueError("driver port must be in 1..65535")
        manifest = read_plan(args.run_dir)
        driver, clients = build_driver(
            args.run_dir,
            manifest,
            bind_host=args.bind_host,
            port=args.port,
            concurrency=args.concurrency,
            episode_timeout_s=args.episode_timeout,
        )
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        try:
            driver.run(stop=stop.is_set)
        finally:
            for client in clients:
                client.close()
        print(json.dumps(driver.scheduler.progress(), indent=2))
        if stop.is_set():
            return 130
        from exp.warm_reset.admit import json_rows

        journal = args.run_dir / "journal.jsonl"
        terminals = (
            [r for r in json_rows(journal) if r.get("accepted") is True]
            if journal.exists()
            else []
        )
        expected_count = len(manifest["arms"]) * sum(
            len(t["init_indices"]) for t in manifest["tasks"]
        )
        return (
            0
            if len(terminals) == expected_count
            and all(not r.get("error") for r in terminals)
            else 2
        )
    elif args.command == "agent":
        from exp.warm_reset.conductor import worker_agent

        plan = json.loads((args.run_dir / "plan.json").read_text())
        agent = worker_agent(
            plan,
            server=args.server,
            driver_host=args.driver_host,
            driver_port=args.driver_port,
            gpus=args.gpus.split(","),
            workers_per_gpu=args.workers_per_gpu,
            prefix=args.prefix,
            conda_env=args.conda_env,
            rc_options={
                "worker_python": args.worker_python,
                "robocasa_cwd": args.robocasa_cwd,
                "egl_lib_dir": args.egl_lib_dir,
                "egl_vendor_dir": args.egl_vendor_dir,
                "connect_deadline_s": args.connect_deadline,
                "episode_deadline_s": args.episode_deadline,
                "terminate_grace_s": args.terminate_grace,
                "max_cached_envs": args.max_cached_envs,
            },
        )
        signal.signal(signal.SIGINT, lambda *_: agent.stop())
        signal.signal(signal.SIGTERM, lambda *_: agent.stop())
        try:
            agent.run()
        finally:
            agent.stop()
    else:
        from exp.warm_reset.admit import admit

        try:
            report = admit(args.run_dir, args.evidence_dir)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report = {
                "schema": "warm_reset_admission_v1",
                "ok": False,
                "global_problems": {str(exc): 1},
                "arms": {},
                "episodes": [],
            }
        (args.run_dir / "admission.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2)
        )
        return 0 if report["ok"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
