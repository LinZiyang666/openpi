"""CLI: tasks, prepare, run, agent and admit for production warm reset experiments.

Run ``python -m exp.warm_reset.run --help``. Model/simulator imports are lazy,
so task export can run inside a simulator island and the driver in the main venv.
Environments come from ``exp.warm_reset.envs`` (plugins included); the paired
analysis of finished runs is ``python -m exp.warm_reset.analysis``.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import threading
from pathlib import Path

from exp.warm_reset.envs import env_ids, get_env


def _k_servers(text: str) -> dict[int, list[str]]:
    """``"1=h:p,2=h:p,2=h:q"`` -> ``{1: ["h:p"], 2: ["h:p", "h:q"]}``."""
    out: dict[int, list[str]] = {}
    for item in (x.strip() for x in text.split(",") if x.strip()):
        k, sep, address = item.partition("=")
        if not sep or not k.isdigit() or not address:
            raise ValueError(f"--k-servers entry {item!r} is not <steps>=<host:port>")
        out.setdefault(int(k), []).append(address)
    return out


def parser() -> argparse.ArgumentParser:
    """Construct the CLI without importing policy or simulator dependencies."""
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser(
        "tasks", help="export trusted task names in the simulator environment"
    )
    tasks.add_argument("--env", required=True, choices=env_ids())
    tasks.add_argument(
        "--task-ids", default="all", help="LIBERO task indices, comma separated"
    )
    tasks.add_argument(
        "--task-names",
        default="",
        help="RoboCasa canonical env names (or the environment adapter's names), comma separated",
    )
    tasks.add_argument("--episodes", required=True, type=int)
    tasks.add_argument("--init-offset", default=0, type=int)
    tasks.add_argument("--out", required=True, type=Path)
    prepare = commands.add_parser(
        "prepare", help="freeze YAMLs and the task/rollout plan; launch nothing"
    )
    prepare.add_argument("--env", required=True, choices=env_ids())
    prepare.add_argument(
        "--base-yaml",
        type=Path,
        help="retrieval + frozen library yaml; required by warm-family (library) arms only",
    )
    prepare.add_argument(
        "--arms",
        default="all",
        help="all existing self-family arms or comma-separated arm IDs "
        "(warm family, full, plain_k<k>)",
    )
    prepare.add_argument(
        "--self-trigger",
        default="verdict",
        choices=("verdict", "always"),
        help="always: self arms are library-free (trigger: always), no retrieval",
    )
    prepare.add_argument(
        "--k-servers",
        default="",
        help="GR00T: endpoints started with another --denoising-steps, <steps>=<host:port>,...",
    )
    prepare.add_argument(
        "--experiment-id",
        default=None,
        help="RoboCasa: stable episode experiment id (self noise shared across runs)",
    )
    prepare.add_argument(
        "--pinned-objects",
        default="",
        help="RoboCasa PnP: pin manifest; per-task slot maps are frozen into the plan",
    )
    prepare.add_argument(
        "--init-pool-sha256",
        default=None,
        help="LIBERO: digest of --init-states-dir when that pool is not readable here",
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
    agent.add_argument(
        "--pinned-objects", default="", help="RoboCasa PnP: this host's copy of the plan's pin manifest"
    )
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
        from exp.warm_reset.plan import validate_tasks

        env = get_env(args.env)
        tasks = env.adapter.export_tasks(
            env,
            task_ids=args.task_ids,
            task_names=args.task_names,
            episodes=args.episodes,
            init_offset=args.init_offset,
        )
        validate_tasks(tasks)
        with args.out.open("x") as fh:
            json.dump(tasks, fh, indent=2)
            fh.write("\n")
    elif args.command == "prepare":
        from exp.warm_reset.plan import default_arms, prepare

        plan = prepare(
            out=args.out,
            env_id=args.env,
            base_yaml=args.base_yaml,
            self_trigger=args.self_trigger,
            k_servers=_k_servers(args.k_servers) or None,
            experiment_id=args.experiment_id,
            pinned_objects=args.pinned_objects or None,
            init_pool_sha256=args.init_pool_sha256,
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
        summary = {
            "plan": str(args.out / "plan.json"),
            "arms": len(plan["arms"]),
            "server_evidence_dir": plan["evidence_dir"],
        }
        if "init_pool_sha256" in plan:
            summary["init_pool_sha256"] = plan["init_pool_sha256"]
            if plan["init_pool_sha256"] is None:
                summary["warning"] = (
                    "init pool not readable here and no --init-pool-sha256: "
                    "pairs are keyed by the pool path"
                )
        print(json.dumps(summary, indent=2))
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
            pinned_objects=args.pinned_objects,
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
