"""Arm-sequenced conductor driver for the ablation-study evaluation (Phase 3/4).

Reads ``config/arm_matrix_<suite>.yaml`` and evaluates the selected arms on
the official pruned_init test set (the episode runner's default init states),
one conductor eval stage per arm. Arms switch via the existing bundle
hot-swap: ``on_stage_begin`` ships the arm's cache yaml (with its optional
``routing`` section) through ``ctl.load_cache_config`` — the server process is
never restarted.

Before any episode runs, every arm yaml is re-validated locally through
``load_cache_config`` (which enforces the routing positive allowlist), so a
mis-edited arm fails here rather than mid-run.

Usage (client machine, timan107):
    PYTHONPATH=. uv run exp/ablation_study/run_ablation_eval.py \
        --arm-matrix exp/ablation_study/config/arm_matrix_libero_spatial.yaml \
        --task-suite libero_spatial --servers <host>:<port> \
        --workers 8 --journal exp/ablation_study/data/runs/spatial_journal.jsonl
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

import yaml

from openpi.cache.config import load_cache_config
from openpi.conductor import ConductorDriver
from openpi.conductor import ServerEndpoint
from openpi.conductor import WorkerAgent
from openpi.conductor import WorkerSpec
from openpi.conductor import strategy as _strat
from openpi.conductor import task as _task

logger = logging.getLogger(__name__)


def canonical_row(line: str) -> str:
    """Stable identity for a per-step JSONL row (key order independent)."""
    import json

    return json.dumps(json.loads(line), sort_keys=True)


def merge_snapshot(per_step_path: Path, snap_path: Path) -> int:
    """Fold a crash-time snapshot into the authoritative JSONL, deduplicating
    on canonical row identity, then remove the snapshot. Returns merged count."""
    if not snap_path.exists():
        return 0
    seen = set()
    if per_step_path.exists():
        with per_step_path.open(encoding="utf-8") as f:
            seen = {canonical_row(line) for line in f if line.strip()}
    merged = 0
    with per_step_path.open("a", encoding="utf-8") as out:
        with snap_path.open(encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                key = canonical_row(row)
                if key not in seen:
                    seen.add(key)  # dedup within the snapshot itself too
                    out.write(row + "\n")
                    merged += 1
    snap_path.unlink()
    return merged


APPROVAL_DECISIONS = frozenset({"add_episodes", "approved_delta", "underpowered_ok"})


def load_approval(path: str | None) -> dict | None:
    """Validate and normalise the owner approval marker. The recorded delta
    drives BOTH the launch gate and downstream analysis (via the immutable
    preflight artifact)."""
    if not path:
        return None
    import hashlib

    raw = Path(path).read_bytes()
    doc = yaml.safe_load(raw)
    required = {"approved_by", "date", "decision", "delta"}
    if not isinstance(doc, dict) or not required.issubset(doc):
        raise SystemExit(
            f"approval file {path} must be a yaml mapping with fields {sorted(required)} "
            "(decision: add_episodes | approved_delta | underpowered_ok)"
        )
    # Normalise: plain YAML dates parse as datetime.date — JSON-serialisable str.
    doc["date"] = str(doc["date"])
    doc["approved_by"] = str(doc["approved_by"])
    if doc["decision"] not in APPROVAL_DECISIONS:
        raise SystemExit(
            f"approval decision {doc['decision']!r} invalid; "
            f"must be one of {sorted(APPROVAL_DECISIONS)}"
        )
    try:
        doc["delta"] = float(doc["delta"])
    except (TypeError, ValueError):
        raise SystemExit(f"approval delta {doc['delta']!r} is not a number") from None
    if not (0.0 < doc["delta"] <= 0.2):
        raise SystemExit(f"approval delta {doc['delta']} outside sane range (0, 0.2]")
    doc["sha256"] = hashlib.sha256(raw).hexdigest()
    doc["path"] = path
    return doc

NUM_TASKS = 10
TRIALS_PER_TASK = 50  # official pruned_init protocol: 50 eval inits per task


class AblationEvalStrategy(_strat.ExperimentStrategy):
    """Pure-eval strategy: one independent eval stage per arm yaml."""

    def __init__(self, task_suite: str, yaml_paths: dict[str, str], trials: int) -> None:
        self._task_suite = task_suite
        self._yaml_paths = yaml_paths  # yaml_id -> path on the client machine
        self._trials = trials

    def plan(
        self,
        yamls: list[str],
        server_assignment: dict[str, _task.ServerEndpoint],
    ) -> _task.TaskGraph:
        graph = _task.TaskGraph()
        for yaml_id in yamls:
            server = server_assignment[yaml_id]
            stage = _task.Stage(
                stage_id=f"eval__{yaml_id}",
                yaml_id=yaml_id,
                phase="eval",
                server=server,
                setup={"yaml_path": self._yaml_paths[yaml_id]},
            )
            for task_id in range(NUM_TASKS):
                for ep_idx in range(self._trials):
                    stage.episodes.append(
                        _task.EpisodeTask(
                            task_uid=_task.make_task_uid(yaml_id, "eval", task_id, ep_idx),
                            yaml_id=yaml_id,
                            phase="eval",
                            experiment=self._task_suite,
                            task_id=task_id,
                            episode_idx=ep_idx,
                            orig_init_state_idx=ep_idx,
                            server_host=server.host,
                            server_port=server.port,
                            bundle_id=yaml_id,
                            extra={"num_trials_per_task": self._trials},
                        )
                    )
            graph.add_stage(stage)
        return graph

    def on_stage_begin(self, stage, ctl, ctx) -> None:
        yaml_path = stage.setup["yaml_path"]
        ctl.load_cache_config(
            yaml_content=Path(yaml_path).read_text(encoding="utf-8"),
            yaml_id=stage.yaml_id,
            bundle_id=stage.yaml_id,
        )
        logger.info("arm %s: bundle loaded from %s", stage.yaml_id, yaml_path)


def _validate_arms(arm_rows: list[dict]) -> dict[str, str]:
    """Local re-validation of every arm yaml (allowlist enforced at load)."""
    yaml_paths: dict[str, str] = {}
    for row in arm_rows:
        arm, path = row["arm"], row["yaml"]
        cfg = load_cache_config(path)  # raises ConfigValidationError on violation
        routed = cfg.routing is not None
        if arm != "cache_baseline" and not routed:
            raise SystemExit(f"arm {arm}: expected a routing section in {path}")
        if arm == "cache_baseline" and routed:
            raise SystemExit(f"arm {arm}: baseline must not carry routing ({path})")
        yaml_paths[arm] = path
    return yaml_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-matrix", required=True)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--servers", required=True, help="host:port[,host:port...]")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--server-workers", default="")
    parser.add_argument("--arms", default="", help="comma list; empty = all arms")
    parser.add_argument("--trials", type=int, default=TRIALS_PER_TASK)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--per-step-out", required=True, help="crash-safe per-step JSONL")
    parser.add_argument("--expected-discordance", type=float, required=True,
                        help="expected discordant-pair rate for the mandatory TOST power gate")
    parser.add_argument("--preflight-approval", default=None,
                        help="owner-approved marker file allowing an underpowered launch "
                             "(records the approved delta/n decision; plan delta)")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--episode-timeout-s", type=float, default=1800.0)
    parser.add_argument("--eval-concurrency", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--conda-env", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    matrix = yaml.safe_load(Path(args.arm_matrix).read_text(encoding="utf-8"))
    rows = matrix["arms"]
    if args.arms:
        wanted = set(args.arms.split(","))
        rows = [r for r in rows if r["arm"] in wanted]
        missing = wanted - {r["arm"] for r in rows}
        if missing:
            raise SystemExit(f"unknown arms requested: {sorted(missing)}")
    yaml_paths = _validate_arms(rows)

    # Mandatory pre-launch power gate (plan §3): the formal arms may not start
    # underpowered unless the owner's recorded decision says so — and that same
    # recorded delta is what the analyzer must use (single-sourced statistics).
    from exp.ablation_study.analysis.analyze_ablation import DELTA
    from exp.ablation_study.analysis.analyze_ablation import preflight

    approval = load_approval(args.preflight_approval)
    # Decision semantics: approved_delta -> gate re-evaluated under the approved
    # margin; underpowered_ok -> launch allowed with the recorded interpretation
    # (no-equivalence claims); add_episodes -> NO bypass (the fix is --trials).
    delta = float(approval["delta"]) if approval and approval["decision"] == "approved_delta" else DELTA
    gate = preflight(args.trials * NUM_TASKS, args.expected_discordance, delta=delta)
    logger.info("preflight power gate (delta=%s): %s", delta, gate)
    if not gate["decidable"] and not (approval and approval["decision"] == "underpowered_ok"):
        raise SystemExit(
            f"UNDERPOWERED launch blocked: TOST power "
            f"{gate['tost_power_at_zero_diff']:.3f} < {gate['power_target']} at "
            f"n={gate['n_pairs']}, q={args.expected_discordance}, delta={delta}. "
            "Escalate to the owner and encode the decision in a structured "
            "--preflight-approval file (approved_by/date/decision/delta)."
        )
    import json as _json

    Path(args.per_step_out).parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.per_step_out) + ".preflight.json").write_text(_json.dumps(
        {**gate, "approval": approval}, indent=2, default=str))

    servers = []
    for spec in args.servers.split(","):
        host, port = spec.rsplit(":", 1)
        servers.append(ServerEndpoint(host, int(port)))

    if args.server_workers:
        counts = [int(x) for x in args.server_workers.split(",")]
        if len(counts) != len(servers):
            raise SystemExit("--server-workers count mismatch with --servers")
        worker_server_keys = [s.key for s, c in zip(servers, counts) for _ in range(c)]
        server_capacities = {s.key: c for s, c in zip(servers, counts)}
    else:
        worker_server_keys = [servers[i % len(servers)].key for i in range(args.workers)]
        server_capacities = None

    from examples.libero.episode_runner import default_client_factory

    # Crash-safe per-step sink: hit rate / executor provenance / timing rows
    # append per flush instead of living only in driver memory.
    per_step_path = Path(args.per_step_out)
    per_step_path.parent.mkdir(parents=True, exist_ok=True)
    per_step_lock = threading.Lock()

    # Resume-aware merge: fold any crash-time snapshot into the authoritative
    # JSONL (dedup on full row content) BEFORE a resumed run can overwrite it —
    # journal-skipped episodes would otherwise lose their step evidence.
    snap_path = per_step_path.with_suffix(".snapshot.jsonl")
    merged = merge_snapshot(per_step_path, snap_path)
    if merged:
        logger.info("merged %d snapshot rows into %s on resume", merged, per_step_path)

    def _per_step_writer(task_uid: str, rows: list[dict]) -> None:
        import json

        with per_step_lock, per_step_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"task_uid": task_uid, **row}) + "\n")

    def _snapshot_loop(driver, stop_event, interval_s: float = 60.0) -> None:
        """Crash-safe observability: every 60s atomically snapshot the driver's
        in-memory per-step rows so a mid-arm crash loses at most one interval
        (the stage-end writer remains the authoritative final sink)."""
        import json
        import os

        snap = per_step_path.with_suffix(".snapshot.jsonl")
        while not stop_event.wait(interval_s):
            rows = list(driver.per_step_rows)
            tmp = snap.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            os.replace(tmp, snap)

    strategy = AblationEvalStrategy(args.task_suite, yaml_paths, args.trials)
    driver = ConductorDriver(
        strategy,
        yaml_weights={arm: 100 for arm in yaml_paths},
        servers=servers,
        journal_path=args.journal,
        ctl_factory=default_client_factory,
        episode_timeout_s=args.episode_timeout_s,
        bind_host=args.bind_host,
        scheduler_kwargs=(
            {"eval_concurrency": args.eval_concurrency} if args.eval_concurrency else None
        ),
        server_capacities=server_capacities,
        per_step_writer=_per_step_writer,
    )

    driver_thread = threading.Thread(target=driver.run, daemon=True)
    driver_thread.start()
    snapshot_stop = threading.Event()
    snapshot_thread = threading.Thread(
        target=_snapshot_loop, args=(driver, snapshot_stop), daemon=True
    )
    snapshot_thread.start()
    while driver.port is None:
        time.sleep(0.05)
    logger.info("driver pull port = %d", driver.port)

    specs = [
        WorkerSpec(
            worker_id=f"w{i}",
            server_key=worker_server_keys[i],
            gpu_id=str(i % args.gpus),
            conda_env=args.conda_env,
            task_suite_name=args.task_suite,
        )
        for i in range(len(worker_server_keys))
    ]
    # Established conductor lifecycle (run_phase2 precedent): the resident
    # agent runs on a background thread; the driver join is the completion
    # signal and agent.stop() is guaranteed on the way out.
    agent = WorkerAgent(specs, driver_host=args.bind_host, driver_port=driver.port)
    agent_thread = threading.Thread(target=agent.run, daemon=True)
    agent_thread.start()
    try:
        driver_thread.join()
    finally:
        snapshot_stop.set()
        # Join the writer BEFORE folding: an in-flight os.replace after the
        # merge would re-create a stale snapshot (lifecycle race).
        snapshot_thread.join(timeout=90.0)
        agent.stop()
    # Normal exit: the stage writer is authoritative; fold-and-remove any final
    # snapshot so a later resume cannot re-merge stale rows.
    merge_snapshot(per_step_path, snap_path)
    logger.info("all arms done; per-step rows at %s", per_step_path)


if __name__ == "__main__":
    main()
