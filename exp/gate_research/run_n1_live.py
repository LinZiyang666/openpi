"""Stage 1b live-validation runner: one (config, operating-point) per invocation.

Dispatches by the single eval YAML's CP1 gate type:
  - ``client_controlled`` -> N1 worker (``exp.gate_research.worker_entry_n1``) with
    the N1 thresholds injected via environment variables. ``--theta-low`` /
    ``--theta-high`` / ``--j`` / ``--M`` are required.
  - ``periodic`` -> the default worker (``examples.libero.worker_entry``); no client
    signal is sent (a ``PeriodicGate`` rejects it). ``cache_len`` / ``inference_len``
    are read from the YAML for the manifest.
  - ``follow_winner`` (N2, server-side) -> the default worker, same as periodic (no
    client signal). ``lock_streak`` / ``budget`` are read from the YAML; the server
    stamps per-step ``searched`` on ``__hit_meta__`` (no ``export_collect_meta``).

Both paths reuse the same conductor skeleton as ``run_collect.py`` (eval-only
``WarmupEvalStrategy``, incremental per-step append) and write a run manifest
whose fields are the single source of provenance for ``analyze_n1_live.py``.

Example (client_controlled, point A on libero_spatial fh75_ws10):
    uv run exp/gate_research/run_n1_live.py \
        --yaml-dir exp/gate_research/config/libero_spatial/n1 \
        --run-id spatial_fh75_ws10_A \
        --theta-low 0.968929 --theta-high 0.968929 --j 3 --M 3 --point A \
        --journal exp/gate_research/data/n1_live/spatial_fh75_ws10_A/journal.jsonl \
        --per-step-out exp/gate_research/data/n1_live/spatial_fh75_ws10_A/rows.jsonl \
        --manifest-out exp/gate_research/data/n1_live/spatial_fh75_ws10_A/manifest.json \
        --baseline-journal exp/gate_research/data/libero_spatial/journal.jsonl \
        --baseline-gate-rows exp/gate_research/data/libero_spatial/gate_rows.jsonl \
        --servers HOST:8000 --workers 48 --gpus 8 --task-ids 0-9 --eval-trials 50 \
        --task-suite libero_spatial --conda-env /scratch/zixuans8/libero_sim
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path

import yaml

N1_WORKER_MODULE = "exp.gate_research.worker_entry_n1"
N4_WORKER_MODULE = "exp.gate_research.worker_entry_n4"
DEFAULT_WORKER_MODULE = "examples.libero.worker_entry"
# Sentinel distinguishing "--M not given" from an explicit "--M none" (None is a
# valid value = never probe).
M_UNSET = "__UNSET__"
# Trusted replan-step spacing recorded into the manifest for the analyzer's
# missing-decision check. Matches examples.libero.main.Args.replan_steps default;
# override --replan-steps only if the worker's replan_steps is changed too.
DEFAULT_REPLAN_STEPS = 5


def _parse_ids(spec: str) -> list[int]:
    """Parse "0-9" or "0,1,2" into a list of ints."""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x != ""]


def single_yaml(yaml_dir: str | Path) -> Path:
    """Return the one eval YAML in ``yaml_dir``; SystemExit on zero or many."""
    ys = sorted(Path(yaml_dir).glob("*.yaml"))
    if len(ys) != 1:
        raise SystemExit(f"--yaml-dir must contain exactly one yaml, found {len(ys)} in {yaml_dir}")
    return ys[0]


def gate_info(yaml_path: str | Path) -> dict:
    """Read the CP1 gate config from an eval YAML. Returns
    ``{type, cache_len, inference_len}`` (cache_len/inference_len None unless
    the gate is periodic)."""
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    gate = cfg["checkpoints"]["cp1"]["gate"]
    return {
        "type": gate["type"],
        "cache_len": gate.get("cache_len"),
        "inference_len": gate.get("inference_len"),
        # follow_winner (N2, server-side) params; None unless the gate is N2.
        "lock_streak": gate.get("lock_streak"),
        "budget": gate.get("budget"),
    }


def build_manifest(args, yaml_path: Path, ginfo: dict) -> dict:
    """Assemble the run manifest (deterministic provenance for the analyzer)."""
    return {
        "run_id": args.run_id,
        "suite": args.task_suite,
        "config": yaml_path.stem,
        "point": args.point,
        "gate_type": ginfo["type"],
        # gate_family disambiguates client_controlled runs (n1 vs n4); it is
        # meaningless for periodic (gate_type is the discriminator there). L is
        # the N4 V2 injection threshold, None for n1/periodic. getattr keeps older
        # callers that build the args namespace without these fields working.
        "gate_family": "n2" if ginfo["type"] == "follow_winner" else getattr(args, "gate_family", "n1"),
        "L": getattr(args, "L", None),
        # follow_winner (N2) params; .get keeps hand-built ginfo (older callers /
        # tests without these keys) working -- real gate_info() always supplies them.
        "lock_streak": ginfo.get("lock_streak"),
        "budget": ginfo.get("budget"),
        "theta_low": args.theta_low,
        "theta_high": args.theta_high,
        "j": args.j,
        "M": None if args.M == M_UNSET else args.M,
        "cache_len": ginfo["cache_len"],
        "inference_len": ginfo["inference_len"],
        "replan_steps": args.replan_steps,
        "matched_to": args.matched_to,
        "yaml_id": yaml_path.stem,
        "baseline_yaml_id": args.baseline_yaml_id or yaml_path.stem,
        "journal_path": args.journal,
        "per_step_out_path": args.per_step_out,
        "baseline_journal_path": args.baseline_journal,
        "baseline_gate_rows_path": args.baseline_gate_rows,
        "n_episodes": len(_parse_ids(args.task_ids)) * args.eval_trials,
    }


def _resolve_worker_and_env(ginfo: dict, args) -> str:
    """Validate gate type, set the client env for client_controlled, and return
    the worker module. The client_controlled YAML is shared by N1 and N4 (both
    drive the server-side ClientControlledGate); ``--gate-family`` selects which
    client state machine drives it. Fails fast on missing thresholds / L."""
    gtype = ginfo["type"]
    if gtype == "client_controlled":
        # M may legitimately be None ("never probe"); require it to be PRESENT
        # (distinct from absent) plus the thresholds/j. Shared by n1 and n4.
        if None in (args.theta_low, args.theta_high, args.j) or args.M == M_UNSET:
            raise SystemExit(
                "client_controlled run requires --theta-low --theta-high --j --M "
                "(--M none for never-probe)")
        family = getattr(args, "gate_family", "n1")
        if family == "n4":
            # V2 injection threshold is mandatory for N4; fail fast in the DRIVER
            # before spawning workers that would restart on a per-worker error.
            if args.L is None:
                raise SystemExit("client_controlled --gate-family n4 requires --L")
            from exp.gate_research.n4_gate_client import N4GateState
            try:
                N4GateState(args.theta_low, args.theta_high, args.j, args.M, args.L)
            except ValueError as exc:
                raise SystemExit(f"invalid N4 params: {exc}")
            os.environ["N4_THETA_LOW"] = repr(float(args.theta_low))
            os.environ["N4_THETA_HIGH"] = repr(float(args.theta_high))
            os.environ["N4_J"] = str(int(args.j))
            os.environ["N4_M"] = "none" if args.M is None else str(int(args.M))
            os.environ["N4_L"] = str(int(args.L))
            return N4_WORKER_MODULE
        # Default family "n1": Fail fast on illegal params in the DRIVER.
        from exp.gate_research.n1_gate_client import N1GateState
        try:
            N1GateState(args.theta_low, args.theta_high, args.j, args.M)
        except ValueError as exc:
            raise SystemExit(f"invalid N1 params: {exc}")
        os.environ["N1_THETA_LOW"] = repr(float(args.theta_low))
        os.environ["N1_THETA_HIGH"] = repr(float(args.theta_high))
        os.environ["N1_J"] = str(int(args.j))
        os.environ["N1_M"] = "none" if args.M is None else str(int(args.M))
        return N1_WORKER_MODULE
    if gtype == "periodic":
        if ginfo["cache_len"] is None or ginfo["inference_len"] is None:
            raise SystemExit("periodic yaml must set cache_len and inference_len")
        # Default worker: no client signal (PeriodicGate is server-side).
        return DEFAULT_WORKER_MODULE
    if gtype == "follow_winner":
        # N2 is a server-side event-driven gate (like periodic): the default
        # worker sends no client signal; the server drives lock/blind-replay and
        # stamps per-step ``searched`` on ``__hit_meta__``. lock_streak / budget
        # live in the YAML; fail fast if absent.
        if ginfo.get("lock_streak") is None or ginfo.get("budget") is None:
            raise SystemExit("follow_winner yaml must set lock_streak and budget")
        return DEFAULT_WORKER_MODULE
    raise SystemExit(
        f"unsupported gate.type {gtype!r}; expected client_controlled, periodic, or follow_winner")


def _add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--yaml-dir", required=True, help="dir containing exactly one eval yaml")
    ap.add_argument("--run-id", required=True, help="unique id for this (config, point) run")
    ap.add_argument("--point", default="", help="operating-point label (A|B|periodic)")
    ap.add_argument(
        "--gate-family", choices=("n1", "n4"), default="n1",
        help="client_controlled state machine: n1 (V1 hysteresis) or n4 (V1+V2 injection)")
    ap.add_argument(
        "--L", type=int, default=None,
        help="N4 V2 injection threshold: force skip after L consecutive cache-execution "
        "steps (required for --gate-family n4)")
    ap.add_argument("--theta-low", type=float, default=None)
    ap.add_argument("--theta-high", type=float, default=None)
    ap.add_argument("--j", type=int, default=None)
    ap.add_argument(
        "--M", default=M_UNSET,
        help="probe interval int, or 'none' for never-probe (client_controlled only)")
    ap.add_argument(
        "--matched-to", default=None,
        help="for a periodic run: the N1 run_id it is budget-matched to (analyzer pairing key)")
    ap.add_argument(
        "--replan-steps", type=int, default=DEFAULT_REPLAN_STEPS,
        help="trusted decision spacing recorded to the manifest (== worker main.Args.replan_steps)")
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step-out", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--baseline-journal", required=True, help="Stage-0 same-config journal (SR baseline)")
    ap.add_argument("--baseline-gate-rows", required=True, help="Stage-0 same-config gate_rows (C9 baseline)")
    ap.add_argument(
        "--baseline-yaml-id", default=None,
        help="Stage-0 config yaml_id to filter baseline journal/gate_rows; "
        "defaults to this run's yaml stem (matches when the N1 yaml keeps the Stage-0 stem)")
    ap.add_argument("--servers", required=True)
    ap.add_argument("--task-ids", default="0-9")
    ap.add_argument("--eval-trials", type=int, default=50)
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--episode-timeout-s", type=int, default=1800)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--conda-env", default="")
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--eval-concurrency", type=int, default=1)
    ap.add_argument("--server-workers", default="")


def _normalize_m(args) -> None:
    """Coerce --M ('none' | int) to None | int, preserving the M_UNSET sentinel
    for 'flag not given'."""
    if args.M == M_UNSET:
        return
    if isinstance(args.M, str):
        args.M = None if args.M.strip().lower() == "none" else int(args.M)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    ap = argparse.ArgumentParser(description="Stage 1b N1/periodic live-validation runner")
    _add_args(ap)
    args = ap.parse_args()
    _normalize_m(args)

    yaml_path = single_yaml(args.yaml_dir)
    ginfo = gate_info(yaml_path)
    worker_module = _resolve_worker_and_env(ginfo, args)

    # Write the manifest before launching so provenance survives a crash.
    manifest = build_manifest(args, yaml_path, ginfo)
    mpath = Path(args.manifest_out)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"[run_n1_live] manifest -> {mpath} (gate={ginfo['type']}, worker={worker_module})", flush=True)

    from openpi.conductor import ConductorDriver, ServerEndpoint, WorkerAgent, WorkerSpec

    from exp.verdict_factor_judge.strategies.warmup_eval_strategy import WarmupEvalStrategy

    task_ids = _parse_ids(args.task_ids)
    strategy = WarmupEvalStrategy(
        task_ids=task_ids,
        warmup_trials=0,
        eval_trials=args.eval_trials,
        task_suite_name=args.task_suite,
        yaml_dir=str(Path(args.yaml_dir)),
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
        yaml_weights={yaml_path.stem: 100},
        servers=servers,
        journal_path=args.journal,
        ctl_factory=default_client_factory,  # driver ctl does stage load/preload only, never infer
        episode_timeout_s=args.episode_timeout_s,
        bind_host=args.bind_host,
        scheduler_kwargs={"eval_concurrency": args.eval_concurrency},
        server_capacities=server_capacities,
    )

    driver_thread = threading.Thread(target=driver.run, daemon=True)
    driver_thread.start()
    while driver.port is None:
        time.sleep(0.05)
    print(f"[run_n1_live] driver pull port = {driver.port}", flush=True)

    specs = [
        WorkerSpec(
            worker_id=f"w{i}",
            server_key=worker_server_keys[i],
            gpu_id=str(i % args.gpus),
            conda_env=args.conda_env,
            task_suite_name=args.task_suite,
            worker_module=worker_module,
        )
        for i in range(n_workers)
    ]
    agent = WorkerAgent(specs, driver_host=args.bind_host, driver_port=driver.port)
    agent_thread = threading.Thread(target=agent.run, daemon=True)
    agent_thread.start()
    print(f"[run_n1_live] spawned {len(specs)} workers (module={worker_module})", flush=True)

    # Crash-safe incremental append: driver._per_step_rows grows append-only; we
    # flush the new tail to --per-step-out every CHECKPOINT_S. Dedup happens in
    # the analyzer (global max-attempt per task_uid).
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

    with out.open("a", encoding="utf-8") as fh:
        try:
            while driver_thread.is_alive():
                _flush(fh)
                driver_thread.join(timeout=CHECKPOINT_S)
            final_n = _flush(fh)
        finally:
            agent.stop()
    print(f"[run_n1_live] done; appended {final_n} per-step rows to {out}", flush=True)


if __name__ == "__main__":
    main()
