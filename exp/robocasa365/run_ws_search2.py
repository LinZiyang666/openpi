"""Multi-cell eval driver for the ws2 (text-IVF) RoboCasa365 search round.

One invocation drives a whole arm — every search cell of a phase — against a
pool of already-running ``serve_groot_n15 --concurrent --allow-dynamic-bundles``
servers, over the conductor. Unlike round 1's ``run_ws_search.py`` (one
invocation per cell, server restarted per cell by the slot orchestrator), the
driver here hot-swaps cache configurations over the wire: each cell is one
bundle (``bundle_id = run_id``), loaded once per (server, bundle) at stage
setup, and workers bind to it per task via ``select_bundle`` (the runner
already does this from ``task.bundle_id``).

Frozen contracts (plan `logs/robocasa365_ws_search2_text_ivf_plan.log.md` §3-W3):

- **Identity** is byte-compatible with round 1: per cell,
  ``run_id = <prefix>-<cid>__l<L>s<S>_<teacher>`` and per task
  ``yaml_id = build_yaml_id(run_id, task_name)``; ``task_uid`` comes from
  ``make_task_uid`` unchanged. Only ``Stage.stage_id`` differs — it carries a
  family-interleaved rank prefix so the scheduler's ``sorted(stage_id)``
  traversal serves every weight family from the start instead of walking the
  families in blocks (round-1 mid-run readouts were family-biased samples).
- **Two graphs**: the immutable full graph (built per cell BEFORE the journal
  is read) feeds ``build_run_plan``/``write_run_plan`` — so the plan hash is
  invariant across resumes — and the finalizer's expected-UID sets; the active
  graph excludes UIDs already terminal in the central journal
  (``Journal.record_counts_as_done``), and a cell with nothing left produces
  NO stage at all (zero setup, zero bundle load).
- **Products**: the driver writes one central journal; ``finalize()``
  materializes round-1-shaped per-cell ``journal_<run_id>.jsonl`` +
  ``summary_<run_id>.json`` that the existing ``summarize_ws_search.py`` /
  ``analyze_ws_search_stats.py`` read unchanged. Idempotent; safe mid-run.
- **Evidence**: workers run with ``--episode-header-rows`` (ws2 evidence
  runner), so per-step rows carry one prompt/seed header per episode plus the
  ``__hit_meta__`` rows; the driver appends them per cell to
  ``per_step_<run_id>.jsonl``.
- **Selection manifest**: the ws2c/ws2e phases accept their cell set ONLY from
  ``selection_manifest.json``; a resume verifies the manifest hash recorded on
  first launch and never re-selects.

``ws2_spawn_fn`` is a deliberate copy of ``run_collect.robocasa_spawn_fn``
(plus the evidence flag) so this round leaves the frozen collection spawn
untouched; keep the two in sync if the worker CLI ever changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import logging
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from openpi.conductor import ServerEndpoint, WorkerAgent, WorkerSpec
from openpi.conductor.driver import ConductorDriver, assign_servers
from openpi.conductor import protocol as _proto
from openpi.conductor.journal import Journal
from openpi.conductor.strategy import ExperimentStrategy, StageContext
from openpi.conductor.task import Stage, TaskGraph

from exp.robocasa365.pinned_objects import (
    PNP_CACHE_ARM,
    assert_pnp_eval_identity,
    assert_pnp_run_plan_identity,
    load_pin_manifest,
    resolve_manifest_path,
)
from exp.robocasa365.run_collect import (
    build_run_plan,
    load_env_config,
    parse_tasks,
    validate_teacher_endpoints,
    write_run_plan,
)
from exp.robocasa365.run_ws_search import (
    DEFAULT_EVAL_TASKS,
    EVAL_NO_COLLECT_ROOT,
    WsSearchStrategy,
    summarize_journal,
)

# ------------------------------------------------------------------
# Family interleaving (scheduler-visible order, plan D7)
# ------------------------------------------------------------------

# Fixed family order; a cid's family is its first "_"-segment. All five
# prefixes are distinct as whole segments (iso/grid/grid3/grid3v/grid4), so no
# startswith ambiguity exists.
logger = logging.getLogger(__name__)

FAMILY_ORDER = ("iso", "grid", "grid3", "grid3v", "grid4")


def family_of(cid: str) -> str:
    """Weight-family of a cell id (its first ``_``-separated token)."""
    head = cid.split("_", 1)[0]
    return head if head in FAMILY_ORDER else "other"


def interleave_cells(cids: list[str]) -> list[str]:
    """Round-robin over weight families; deterministic (families/cids sorted).

    Any prefix of the result covers every non-exhausted family, so partial
    scheduler progress is a family-balanced sample (round-1 lesson).
    """
    buckets: dict[str, list[str]] = {f: [] for f in (*FAMILY_ORDER, "other")}
    for cid in sorted(cids):
        buckets[family_of(cid)].append(cid)
    order: list[str] = []
    queues = [buckets[f] for f in (*FAMILY_ORDER, "other") if buckets[f]]
    idx = 0
    while queues:
        queue = queues[idx]
        order.append(queue.pop(0))
        if queue:
            idx += 1
        else:
            # Removing the exhausted queue shifts its successors down by one,
            # so the index already points at the family whose turn is next --
            # advancing here as well would skip that family's turn.
            queues.pop(idx)
        if queues:
            idx %= len(queues)
    return order


# ------------------------------------------------------------------
# Multi-cell strategy (bundle load + ranked stage ids)
# ------------------------------------------------------------------


class Ws2ArmStrategy(ExperimentStrategy):
    """One arm = many cells; each cell is one bundle over the whole pool.

    ``cell_specs`` maps run_id -> {"cid", "yaml_path", "yaml_text", "rank",
    "episodes_by_yaml": {yaml_id: [EpisodeTask, ...]}} where the episode lists
    are ALREADY filtered to the active set (journalled UIDs dropped) and
    stamped with ``bundle_id = run_id``. Cells whose lists are all empty are
    not passed in at all.
    """

    def __init__(self, cell_specs: dict[str, dict[str, Any]]) -> None:
        self._cells = cell_specs
        self._loaded: set[tuple[str, str]] = set()  # (server.key, bundle_id)
        self._lock = threading.Lock()

    def plan(self, yamls: list[str], server_assignment: dict[str, ServerEndpoint]) -> TaskGraph:
        del yamls  # identity lives in the prepared cell specs
        graph = TaskGraph()
        for run_id, spec in self._cells.items():
            for yaml_id, episodes in spec["episodes_by_yaml"].items():
                if not episodes:
                    continue
                graph.add_stage(
                    Stage(
                        # Zero-padded rank first: the scheduler activates and
                        # dispatches in sorted(stage_id) order, so this prefix
                        # IS the execution interleave (plan D7).
                        stage_id=f"{spec['rank']:04d}__{yaml_id}",
                        yaml_id=yaml_id,
                        phase="eval",
                        server=server_assignment[yaml_id],
                        episodes=list(episodes),
                        setup={"bundle_id": run_id, "yaml_text": spec["yaml_text"]},
                    )
                )
        graph.validate()
        return graph

    def _ensure_bundle(self, stage: Stage, ctl: Any) -> None:
        """Load this cell's bundle on this server once — counting only successes.

        Memoising before the call would turn one transient ctl error into a
        dead cell: the driver rolls a failed setup back to SETUP_PENDING and
        retries, the memo would swallow the retry, and every episode of that
        cell on that server would then run against a bundle the server never
        loaded.
        """
        key = (stage.server.key, stage.setup["bundle_id"])
        with self._lock:
            if key in self._loaded:
                return
        ctl.load_cache_config(
            yaml_content=stage.setup["yaml_text"],
            yaml_id=stage.setup["bundle_id"],
            # Never "default": with hot-swap enabled that slot is writable and
            # binding under it would tie provenance to load order.
            bundle_id=stage.setup["bundle_id"],
        )
        with self._lock:
            self._loaded.add(key)

    def on_stage_begin(self, stage: Stage, ctl: Any, ctx: StageContext) -> None:
        del ctx
        if not stage.episodes:
            return
        self._ensure_bundle(stage, ctl)

    def on_resume(self, stage: Stage, ctl: Any, ctx: StageContext) -> None:
        # Same path as begin: the memo makes the load idempotent per process,
        # and a fresh driver process re-sends once per (server, bundle).
        self.on_stage_begin(stage, ctl, ctx)


# ------------------------------------------------------------------
# Worker spawn (evidence runner opt-in)
# ------------------------------------------------------------------


def ws2_spawn_fn(
    spec: WorkerSpec,
    driver_host: str,
    driver_port: int,
    *,
    worker_python: str,
    robocasa_cwd: str,
    repo_root: str,
    egl_lib_dir: str,
    egl_vendor_dir: str,
    teacher: str,
    connect_deadline_s: float,
    episode_deadline_s: float,
    terminate_grace_s: float,
    max_cached_envs: int | None = None,
    pinned_objects_path: str | None = None,
) -> subprocess.Popen:
    """``robocasa_spawn_fn`` copy + ``--episode-header-rows`` (plan §3-W3).

    ``start_new_session=True`` is a SAFETY requirement: ``WorkerAgent.stop()``
    signals ``os.getpgid(pid)``, which without a fresh session is the agent's
    own process group.
    """
    cmd = [
        worker_python,
        "-m",
        "exp.robocasa365.worker_entry",
        "--worker-id", spec.worker_id,
        "--server-key", spec.server_key,
        "--driver-host", driver_host,
        "--driver-port", str(driver_port),
        "--teacher", teacher,
        "--connect-deadline-s", str(connect_deadline_s),
        "--episode-deadline-s", str(episode_deadline_s),
        "--terminate-grace-s", str(terminate_grace_s),
        "--episode-header-rows",
    ]
    if max_cached_envs is not None:
        cmd += ["--max-cached-envs", str(max_cached_envs)]
    if pinned_objects_path:
        cmd += ["--pinned-objects", pinned_objects_path]
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env["PYTHONPATH"] = os.pathsep.join((repo_root, os.path.join(repo_root, "src")))
    env["MUJOCO_GL"] = "egl"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (egl_lib_dir, env.get("LD_LIBRARY_PATH", "")) if p
    )
    env["__EGL_VENDOR_LIBRARY_DIRS"] = egl_vendor_dir
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("MALLOC_TRIM_THRESHOLD_", "134217728")
    return subprocess.Popen(cmd, env=env, cwd=robocasa_cwd, start_new_session=True)


# ------------------------------------------------------------------
# Cell sets (matrix index or selection manifest)
# ------------------------------------------------------------------


def resolve_cells(run_prefix: str, config_dir: pathlib.Path, manifest_path: str) -> tuple[list[str], str]:
    """Return (cids, manifest_sha) for the phase; manifest_sha is "" for ws2.

    - ws2 (screening): every cid in the emitted ``index.json``.
    - ws2c / ws2e: ONLY the cells of the manifest segment named after the
      prefix; the manifest is the single audited source (plan §3-W8) and its
      sha is pinned so a resume can prove it never re-selected.
    """
    if run_prefix == "ws2":
        index = json.loads((config_dir / "index.json").read_text())
        return sorted(index), ""
    if not manifest_path:
        raise SystemExit(f"--run-prefix {run_prefix} requires --manifest (selection_manifest.json)")
    blob = pathlib.Path(manifest_path).read_bytes()
    manifest = json.loads(blob)
    segment = manifest.get("segments", {}).get(run_prefix)
    if segment is None:
        raise SystemExit(f"manifest {manifest_path} has no segment {run_prefix!r}")
    return list(segment["cells"]), hashlib.sha256(blob).hexdigest()


def pinned_config_root(config_dir: pathlib.Path, teacher: str) -> pathlib.Path:
    """Validate and return the root of ``<root>/<teacher>/main``.

    The digest covers both teachers, so checking it alone cannot detect a
    groot run accidentally pointed at the valid pi05 half of the same tree.
    """
    config_dir = pathlib.Path(config_dir).resolve()
    if config_dir.name != "main" or config_dir.parent.name != teacher:
        raise ValueError(
            f"pinned config dir must end in {teacher}/main, got {config_dir}"
        )
    return config_dir.parent.parent


def assert_frozen_cell_set(selected: list[str], frozen: set[str]) -> None:
    """Bind ``index.json`` (the dispatched cells) to the frozen digest table."""
    selected_set = set(selected)
    if len(selected) != len(frozen) or selected_set != frozen:
        raise ValueError(
            "selected cells do not match the frozen digest; "
            f"missing={sorted(frozen - selected_set)[:5]} "
            f"extra={sorted(selected_set - frozen)[:5]}"
        )


def pin_manifest_sha(data_dir: pathlib.Path, run_prefix: str, sha: str) -> None:
    """First launch records the manifest sha; a resume must match it exactly."""
    if not sha:
        return
    pin = data_dir / f"manifest_sha_{run_prefix}.txt"
    if pin.exists():
        stored = pin.read_text().strip()
        if stored != sha:
            raise SystemExit(
                f"selection manifest changed since this run started: recorded {stored}, "
                f"current {sha}. A resume must never re-select; restore the original "
                "manifest or start a fresh run prefix."
            )
        return
    pin.write_text(sha + "\n")


def batched(cells: list[str], size: int) -> list[list[str]]:
    """Split the interleaved cell order into fixed-size, order-preserving runs.

    The scheduler walks every stage on each activation/dispatch call
    (``scheduler.py`` ``_refresh_activation``), so cost grows with the SQUARE of
    the live stage count: measured 3.3 ms at 156 stages (12 cells x 13 tasks)
    against 435 ms at 1716 (all 132). At the whole-matrix size every worker pull
    and every 20 ms driver tick would hold the scheduler lock for ~0.4 s, which
    pegs a core and serialises the fleet for the whole unattended run. Batches
    keep the live graph small; the journal makes the boundaries free.
    """
    if size <= 0:
        return [list(cells)]
    return [cells[i:i + size] for i in range(0, len(cells), size)]


def hold_workers_between_batches(driver: ConductorDriver) -> None:
    """Turn this batch's end-of-run SHUTDOWN into an idle backoff.

    ``handle_pull`` answers MSG_SHUTDOWN once the scheduler is all-done, and a
    worker that receives it exits (``worker.py`` run_forever). Between batches
    that would dismantle the fleet, so non-final batches answer "no task, back
    off" instead: the worker idles, loses the socket when this batch's driver
    stops, and reconnects to the next batch on the same fixed port.
    """
    inner = driver.handle_pull

    def handle(server_key: str):
        msg, payload = inner(server_key)
        if msg == _proto.MSG_SHUTDOWN:
            return _proto.MSG_ASSIGN, {"none": True, "backoff_ms": 500}
        return msg, payload

    driver.handle_pull = handle


def build_cell_specs(
    cell_strategies: dict[str, WsSearchStrategy],
    full_graphs: dict[str, TaskGraph],
    done_uids: set[str],
    ranks: dict[str, int],
    config_dir: pathlib.Path,
) -> dict[str, dict[str, Any]]:
    """Active view: full graphs minus journalled UIDs, bundle stamped.

    A cell with nothing left contributes NO spec (zero stages, zero bundle
    load); episode identity is byte-identical to the full view — only
    ``bundle_id`` is stamped on top.
    """
    cell_specs: dict[str, dict[str, Any]] = {}
    for run_id, strategy in cell_strategies.items():
        episodes_by_yaml: dict[str, list] = {}
        for yaml_id in strategy.yaml_ids:
            stage = full_graphs[run_id].stages[yaml_id]
            active = [
                dataclasses.replace(ep, bundle_id=run_id)
                for ep in stage.episodes
                if ep.task_uid not in done_uids
            ]
            if active:
                episodes_by_yaml[yaml_id] = active
        if episodes_by_yaml:
            cid = strategy._cid  # noqa: SLF001 - own experiment module
            cell_specs[run_id] = {
                "cid": cid,
                "yaml_text": (config_dir / f"{cid}.yaml").read_text(),
                "rank": ranks[cid],
                "episodes_by_yaml": episodes_by_yaml,
            }
    return cell_specs


# ------------------------------------------------------------------
# Finalizer (round-1-shaped per-cell products from the central journal)
# ------------------------------------------------------------------


def finalize(
    central_journal: pathlib.Path,
    data_dir: pathlib.Path,
    *,
    teacher: str,
    expected_by_run: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    """Split the central journal into per-cell journal/summary files.

    ``expected_by_run`` maps run_id -> {"cid", "uids"} from the immutable
    run plans. Raw journal lines are passed through untouched (rejected/stale
    rows included — every consumer already filters on ``accepted`` + status),
    only grouped by the cell their ``task_uid`` belongs to. Idempotent:
    rewrites whole files, safe to run mid-flight for monitoring.
    """
    lines_by_run: dict[str, list[str]] = {run_id: [] for run_id in expected_by_run}
    if central_journal.exists():
        for line in central_journal.read_text().splitlines():
            if not line.strip():
                continue
            try:
                uid = json.loads(line).get("task_uid", "")
            except json.JSONDecodeError:
                # A hard crash can tear the last line mid-write; Journal's own
                # replay tolerates that, so a finished run's products must not
                # be held hostage to hand-repairing the file.
                logger.warning("skipping unparsable journal line in %s", central_journal)
                continue
            # task_uid = <run_id>__<TaskName>:eval:<task_id>:<idx>
            run_id = uid.rsplit("__", 1)[0]
            if run_id in lines_by_run:
                lines_by_run[run_id].append(line)
    complete: dict[str, bool] = {}
    for run_id, spec in expected_by_run.items():
        journal_path = data_dir / f"journal_{run_id}.jsonl"
        journal_path.write_text("".join(f"{line}\n" for line in lines_by_run[run_id]))
        summary = summarize_journal(journal_path, expected_uids=list(spec["uids"]))
        (data_dir / f"summary_{run_id}.json").write_text(
            json.dumps({"cid": spec["cid"], "teacher": teacher, **summary}, indent=1)
        )
        complete[run_id] = bool(summary["complete"])
    return complete


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Both teachers, like every other tool in this family (summarize_ws_search,
    # analyze_ws_search_stats, orchestrate_ws_search). The pi0.5 arm reuses this
    # driver unchanged; ``episode_runner.ADAPTERS`` already carries both, and
    # ``validate_teacher_endpoints`` is what keeps one invocation pointed at a
    # single teacher's endpoint group.
    ap.add_argument("--teacher", required=True, choices=("groot_tp", "pi05"))
    ap.add_argument("--servers", required=True,
                    help='comma-separated "host:port" pool of dynamic-bundle servers')
    ap.add_argument("--run-prefix", default="ws2", choices=("ws2", "ws2c", "ws2e"))
    ap.add_argument("--config-dir", required=True,
                    help="emitted yaml dir for the phase (…/ws_search2/groot_tp/main or …/control)")
    ap.add_argument("--manifest", default="", help="selection_manifest.json (required for ws2c/ws2e)")
    ap.add_argument("--only", default="", help="comma-separated cid subset (rerun/backfill)")
    ap.add_argument("--tasks", default=DEFAULT_EVAL_TASKS)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=1_000_000)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4, help="worker count for THIS agent invocation")
    ap.add_argument("--workers-per-server", type=int, default=8,
                    help="assign_servers capacity weight per server")
    ap.add_argument("--cells-per-batch", type=int, default=12,
                    help="cells driven per graph; the scheduler walks every live "
                    "stage on each call, so the whole 132-cell matrix at once "
                    "costs ~0.4s per pull (0 = one graph, no batching)")
    ap.add_argument("--eval-concurrency", type=int, default=2,
                    help="active (cell, task) stages per server; >1 keeps a worker "
                    "fleet fed across stage boundaries at the cost of per-episode "
                    "bundle re-selection on the connection (facade-level, cheap)")
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--data-dir", default="", help="default: exp/robocasa365/data/ws_search2/<teacher>/")
    ap.add_argument(
        "--pinned-objects",
        default="",
        help="pin table path; pins every object slot to one exact mesh. Must "
        "match the table the library was collected under.",
    )
    ap.add_argument("--env-config", required=True)
    ap.add_argument("--role", default="all", choices=("driver", "agent", "all"))
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--bind-port", type=int, default=0)
    ap.add_argument("--driver-host", default="")
    ap.add_argument("--driver-port", type=int, default=0)
    ap.add_argument("--agent-server", default="",
                    help='agent role: bind this fleet to ONE "host:port" of the pool (required)')
    ap.add_argument("--episode-timeout-s", type=float, default=1800.0)
    ap.add_argument("--connect-deadline-s", type=float, default=600.0)
    ap.add_argument("--episode-deadline-s", type=float, default=1500.0)
    ap.add_argument("--terminate-grace-s", type=float, default=60.0)
    ap.add_argument("--finalize-only", action="store_true",
                    help="just (re)materialize per-cell journals/summaries and exit")
    args = ap.parse_args()

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(143))

    # Resolved before any role branches: the agent role never loads the
    # manifest but still forwards its path to workers whose cwd is the external
    # RoboCasa checkout, where a relative path opens nothing.
    pin_path = resolve_manifest_path(args.pinned_objects) if args.pinned_objects else ""
    pin_id, pinned_objects = (None, None)
    if pin_path:
        pin_id, pinned_objects = load_pin_manifest(pin_path)

    env_config = load_env_config(args.env_config)
    servers = []
    for item in args.servers.split(","):
        host, _, port = item.strip().rpartition(":")
        servers.append(ServerEndpoint(host, int(port)))
    validate_teacher_endpoints(args.teacher, servers, env_config)

    config_dir = pathlib.Path(args.config_dir)
    if pin_path:
        # Before a single cell is dispatched: the 132 yamls this run will ship
        # to the servers must be the exact ones that were frozen, for BOTH
        # teachers. Checking only the teacher in play would let the other half
        # of the experiment drift unnoticed, and the two share cid names.
        from exp.robocasa365.emit_ws_search2_yamls import DIGEST_NAME, verify_index_digest

        try:
            digest_root = pinned_config_root(config_dir, args.teacher)
            digest_doc = verify_index_digest(
                digest_root,
                digest_root / DIGEST_NAME,
                expected_pin_id=pin_id,
            )
        except ValueError as exc:
            raise SystemExit(f"[run_ws_search2] index digest preflight failed: {exc}") from exc
        config_dir = config_dir.resolve()
        print(f"[run_ws_search2] index digest verified under {digest_root}", flush=True)
    cells, manifest_sha = resolve_cells(args.run_prefix, config_dir, args.manifest)
    all_cells = list(cells)
    if pin_path:
        frozen_cells = set(digest_doc["per_teacher"][args.teacher]["cells"])
        try:
            assert_frozen_cell_set(all_cells, frozen_cells)
        except ValueError as exc:
            raise SystemExit(f"[run_ws_search2] {exc}") from exc
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        unknown = wanted - set(cells)
        if unknown:
            raise SystemExit(f"--only cids not in this phase's cell set: {sorted(unknown)}")
        cells = [c for c in cells if c in wanted]

    data_dir = (
        pathlib.Path(args.data_dir)
        if args.data_dir
        else pathlib.Path(__file__).resolve().parent / "data" / "ws_search2" / args.teacher
    )

    tasks = parse_tasks(args.tasks, args.episodes)
    # Ranks come from the phase's IMMUTABLE cell list, so a --only rerun keeps
    # every cell's execution position (plan D7 freezes the order, not the
    # subset); the run order is that list narrowed to the selected cells.
    ranks = {cid: rank for rank, cid in enumerate(interleave_cells(all_cells))}
    ordered_cells = [cid for cid in interleave_cells(all_cells) if cid in set(cells)]

    # ---- immutable full view: per-cell strategies, run plans, expected UIDs
    #
    # Driver-only. A worker agent owns no ledger: building it there would write
    # 132 run plans into the WORKER machine's data dir and, worse, abort the
    # whole agent at launch on any pre-existing plan whose parameters differ
    # (round 1 gated this the same way). Agents need nothing but the endpoint
    # and the worker recipe.
    if pin_path:
        print(f"[run_ws_search2] pin_id={pin_id} manifest={pin_path}", flush=True)
        # Gated on the immutable full cell list, not on `ordered_cells`: a
        # --only resume legitimately dispatches a subset, but the experiment's
        # shape must still be the frozen one.
        assert_pnp_eval_identity(
            tasks, cells=len(all_cells), arm=PNP_CACHE_ARM,
            label="run_ws_search2 (cache arm)",
        )
    cell_strategies: dict[str, WsSearchStrategy] = {}
    yaml_weights: dict[str, int] = {}
    for cid in ordered_cells:
        strategy = WsSearchStrategy(
            cid=cid,
            run_prefix=args.run_prefix,
            teacher=args.teacher,
            layout=args.layout,
            style=args.style,
            base_seed=args.base_seed,
            replan_steps=args.replan_steps,
            tasks=tasks,
            pin_id=pin_id,
            pinned_objects=pinned_objects,
        )
        cell_strategies[strategy.run_id] = strategy
        for yaml_id, (_, n) in zip(strategy.yaml_ids, tasks):
            yaml_weights[yaml_id] = n

    server_capacities = {s.key: args.workers_per_server for s in servers}
    assignment = assign_servers(yaml_weights, servers, None, server_capacities)

    expected_by_run: dict[str, dict[str, Any]] = {}
    full_graphs: dict[str, TaskGraph] = {}
    run_plans: list[dict[str, Any]] = []
    owns_ledger = args.role in ("driver", "all") or args.finalize_only
    if owns_ledger:
        data_dir.mkdir(parents=True, exist_ok=True)
        for run_id, strategy in cell_strategies.items():
            graph = strategy.plan(sorted(strategy.yaml_ids), assignment)
            full_graphs[run_id] = graph
            run_plan = build_run_plan(strategy, graph, EVAL_NO_COLLECT_ROOT)
            run_plans.append(run_plan)
            write_run_plan(data_dir / f"run_plan_{run_id}.json", run_plan)
            expected_by_run[run_id] = {
                "cid": strategy._cid,  # noqa: SLF001
                "uids": list(run_plan["uids"]),
            }
        if pin_path:
            assert_pnp_run_plan_identity(
                run_plans,
                arm=PNP_CACHE_ARM,
                pin_id=pin_id,
                label="run_ws_search2 (cache arm)",
            )

    central_journal = data_dir / f"journal_central_{args.run_prefix}.jsonl"

    if args.finalize_only:
        complete = finalize(central_journal, data_dir, teacher=args.teacher, expected_by_run=expected_by_run)
        done = sum(complete.values())
        print(f"[ws2] finalize-only: {done}/{len(complete)} cells complete", flush=True)
        return

    batches = batched(ordered_cells, args.cells_per_batch)
    driver_thread = None
    drives = args.role in ("driver", "all")
    if drives and len(batches) > 1 and not args.bind_port:
        raise SystemExit(
            "batched driving needs a fixed --bind-port: each batch binds a fresh "
            "driver and the worker fleet reconnects to the same address "
            "(--cells-per-batch 0 disables batching)."
        )
    if drives:
        pin_manifest_sha(data_dir, args.run_prefix, manifest_sha)

        def per_step_writer(yaml_id: str, rows: list[dict[str, Any]]) -> None:
            run_id = yaml_id.rsplit("__", 1)[0]
            path = data_dir / f"per_step_{run_id}.jsonl"
            with path.open("a") as fh:
                for row in rows:
                    fh.write(json.dumps({"yaml_id": yaml_id, **row}) + "\n")

        port_known = threading.Event()
        live_port: dict[str, int] = {}
        # Infrastructure failures must reach the exit code. A daemon thread's
        # traceback is invisible to the parent, so an unattended orchestrator
        # would read a dead driver as an ordinary INCOMPLETE and keep going.
        batch_error: list[BaseException] = []

        def run_batches() -> None:
            try:
                _drive_batches()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                batch_error.append(exc)
                logger.exception("batch driving failed")
            finally:
                # Unblock a main thread still waiting for the first bind.
                port_known.set()

        def _drive_batches() -> None:
            for n, batch in enumerate(batches, start=1):
                # Re-read the journal per batch: earlier batches (and any prior
                # run) have already written their terminal records, so a resume
                # and a fresh start take the same path.
                done_uids = (
                    Journal(str(central_journal)).replay_done_uids()
                    if central_journal.exists() else set()
                )
                batch_strategies = {
                    run_id: s for run_id, s in cell_strategies.items()
                    if s._cid in set(batch)  # noqa: SLF001
                }
                specs = build_cell_specs(
                    batch_strategies, full_graphs, done_uids, ranks, config_dir
                )
                if not specs:
                    print(f"[ws2] batch {n}/{len(batches)}: nothing left, skipping", flush=True)
                    continue
                n_active = sum(
                    len(eps) for spec in specs.values() for eps in spec["episodes_by_yaml"].values()
                )
                print(
                    f"[ws2] batch {n}/{len(batches)} cells={len(specs)} "
                    f"episodes={n_active} servers={len(servers)}",
                    flush=True,
                )
                batch_driver = ConductorDriver(
                    Ws2ArmStrategy(specs),
                    yaml_weights=yaml_weights,
                    servers=servers,
                    journal_path=str(central_journal),
                    ctl_factory=_make_ctl,
                    episode_timeout_s=args.episode_timeout_s,
                    bind_host=args.bind_host,
                    bind_port=args.bind_port,
                    server_capacities=server_capacities,
                    scheduler_kwargs={"eval_concurrency": args.eval_concurrency},
                    per_step_writer=per_step_writer,
                )
                if n < len(batches):
                    hold_workers_between_batches(batch_driver)
                inner_error: list[BaseException] = []

                def drive() -> None:
                    try:
                        batch_driver.run()
                    except BaseException as exc:  # noqa: BLE001 - re-raised below
                        inner_error.append(exc)

                inner = threading.Thread(target=drive, daemon=True)
                inner.start()
                while batch_driver.port is None and inner.is_alive():
                    time.sleep(0.05)
                if batch_driver.port is None:
                    raise RuntimeError(
                        f"batch {n} driver died before binding "
                        f"{args.bind_host}:{args.bind_port}"
                    ) from (inner_error[0] if inner_error else None)
                live_port["port"] = batch_driver.port
                port_known.set()
                print(f"[ws2] batch {n} driver pull port = {batch_driver.port}", flush=True)
                while inner.is_alive():
                    inner.join(timeout=5.0)
                if inner_error:
                    # Stop the whole phase: later batches would run against an
                    # unknown fleet state and their results would be unattributable.
                    raise RuntimeError(f"batch {n} driver raised") from inner_error[0]

        driver_thread = threading.Thread(target=run_batches, daemon=True)
        driver_thread.start()
        # Bounded: a driver that cannot bind must fail the run, not hang it.
        if not port_known.wait(timeout=120.0) and driver_thread.is_alive():
            raise SystemExit("driver did not bind its pull port within 120s")
        if batch_error:
            raise SystemExit(f"ws2 driving failed before serving: {batch_error[0]!r}")

    agent = None
    if args.role in ("agent", "all"):
        driver_host = args.driver_host or args.bind_host
        driver_port = args.driver_port or args.bind_port or live_port.get("port", 0)
        if not driver_port:
            raise SystemExit("--role agent requires --driver-host/--driver-port of a running driver")
        if args.agent_server:
            host, _, port = args.agent_server.rpartition(":")
            fleet_server = ServerEndpoint(host, int(port))
            if fleet_server.key not in {s.key for s in servers}:
                raise SystemExit(f"--agent-server {args.agent_server} is not in --servers")
        elif len(servers) == 1:
            fleet_server = servers[0]
        else:
            raise SystemExit("--role agent with a multi-server pool requires --agent-server")
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
        specs = [
            WorkerSpec(worker_id=f"w{i}", server_key=fleet_server.key, gpu_id=gpu_ids[i % len(gpu_ids)])
            for i in range(args.workers)
        ]
        spawn = functools.partial(
            ws2_spawn_fn,
            worker_python=env_config["WORKER_PYTHON"],
            robocasa_cwd=env_config["ROBOCASA_CWD"],
            repo_root=env_config["REPO_ROOT"],
            egl_lib_dir=env_config["EGL_LIB_DIR"],
            egl_vendor_dir=env_config["EGL_VENDOR_DIR"],
            teacher=args.teacher,
            connect_deadline_s=args.connect_deadline_s,
            episode_deadline_s=args.episode_deadline_s,
            terminate_grace_s=args.terminate_grace_s,
            pinned_objects_path=pin_path or None,
            # 8G eval cards: one cached kitchen per worker (round-1 lesson).
            max_cached_envs=1,
        )
        agent = WorkerAgent(specs, driver_host=driver_host, driver_port=driver_port, spawn_fn=spawn)
        agent_thread = threading.Thread(target=agent.run, daemon=True)
        agent_thread.start()
        print(
            f"[ws2] agent supervising {len(specs)} worker(s) -> {fleet_server.key} "
            f"(driver {driver_host}:{driver_port})",
            flush=True,
        )

    try:
        if driver_thread is not None:
            while driver_thread.is_alive():
                driver_thread.join(timeout=5.0)
        else:
            while True:
                time.sleep(5.0)
    finally:
        if agent is not None:
            agent.stop()

    if drives and batch_error:
        # Non-zero exit, and no finalize: the products would look like an
        # ordinary short run rather than a failed one.
        raise SystemExit(f"ws2 driving failed: {batch_error[0]!r}")

    if drives:
        complete = finalize(central_journal, data_dir, teacher=args.teacher, expected_by_run=expected_by_run)
        done = sum(complete.values())
        rerun = sorted(spec["cid"] for run_id, spec in expected_by_run.items() if not complete[run_id])
        status = "DONE" if done == len(complete) else "INCOMPLETE"
        print(f"[ws2] {status} phase={args.run_prefix} complete={done}/{len(complete)}", flush=True)
        if rerun:
            # MISSING uids come back on a rerun; ERR ones do not -- a fatal
            # error is journalled terminal, so resume drops it and the cell
            # stays incomplete until the journal record itself is dealt with.
            print(f"[ws2] incomplete cells: {','.join(rerun)}", flush=True)
            print("[ws2] rerun heals MISSING uids only "
                  f"(--only {','.join(rerun)}); inspect n_err in the summaries first",
                  flush=True)


def _make_ctl(server: ServerEndpoint):
    """Real ctl: the driver's bundle loads ride the ws protocol."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy(host=server.host, port=server.port)


if __name__ == "__main__":
    main()
