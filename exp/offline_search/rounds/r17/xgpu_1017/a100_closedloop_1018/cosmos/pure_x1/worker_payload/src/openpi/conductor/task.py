"""Core data structures for the two-layer experiment conductor framework.

These are the pure-data types exchanged across the worker / agent / driver
layers and consumed by the scheduler, journal, and strategies. They carry no
experiment semantics (LIBERO / verdict) and no I/O — those live behind the
``EpisodeRunner`` / ``ExperimentStrategy`` interfaces (see ``strategy.py`` /
``worker.py``).

Coupling map:
  DEPENDS ON:  standard library only (no openpi.cache, no exp.*, no LIBERO)
  CONSUMED BY: scheduler.py, driver.py, journal.py, protocol.py, strategy.py,
               worker.py, and every concrete ExperimentStrategy / EpisodeRunner
  IF CHANGED:  wire schema (protocol.py) and journal record shape may need to track
  NOTE:        keep this module dependency-free so the decoupling boundary in
               WORKING_AGREEMENT (src/ mechanism vs exp/ policy) holds.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

# ----------------------------------------------------------------------
# Phase / status literals
# ----------------------------------------------------------------------

# A ``phase`` is an opaque tag forwarded to the worker; the worker never
# interprets it. The conductor core only distinguishes warmup (side-effecting,
# stage-atomic retry) from eval (idempotent, episode-atomic retry).
Phase = Literal["warmup", "eval"]


# ----------------------------------------------------------------------
# Endpoints and identity
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ServerEndpoint:
    """A single inference server *process* endpoint.

    Both deployment shapes the conductor architecture supports map onto this
    type, and which one is in use is a deployment decision the driver does not
    probe: a ``replica_proxy`` public port registers as **one** endpoint whose
    fan-out the driver neither sees nor schedules against, while independent
    single-process servers register as **one entry each**. (An earlier note here
    claimed the router shape was disallowed because its sticky ``fetch_dump``
    would return a partial warmup dump; that stopped being true once the router
    began aggregating -- ``replica_proxy.merge_dump_replies``.)

    The choice is not neutral for scheduling: ``next_task`` selects by
    ``server.key``, so N independent entries are N units the scheduler can place
    work on, whereas one routed entry is a single unit. A strategy that wants
    the pool to work on one yaml at once therefore needs independent entries --
    or a routed endpoint, which achieves the same thing below the driver.
    """

    host: str
    port: int

    @property
    def key(self) -> str:
        return f"{self.host}:{self.port}"


def make_task_uid(yaml_id: str, phase: str, task_id: int, episode_idx: int) -> str:
    """Deterministically derive an episode's unique id.

    Must be deterministic (not random / timestamp based) so that on resume the
    same episode produces the same uid and matches the journal idempotently
    (plan §5.1 / §8.1).
    """
    return f"{yaml_id}:{phase}:{task_id}:{episode_idx}"


# ----------------------------------------------------------------------
# Episode task / result (driver <-> worker payloads)
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EpisodeTask:
    """One episode of work dispatched from the driver to a worker (plan §5.1)."""

    task_uid: str
    yaml_id: str
    phase: Phase
    experiment: str  # task_suite_name, the "experiment kind"
    task_id: int  # index into the task suite
    episode_idx: int  # episode index within the task
    orig_init_state_idx: int  # initial-state mapping (preserves existing filter semantics)
    server_host: str  # owning server of this episode's yaml
    server_port: int
    bundle_id: str  # M2 select_bundle target
    attempt: int = 1  # per-dispatch generation; fences stale results after a requeue (G2R3)
    # Free-form per-task metadata. Producer contract: a strategy whose episodes
    # run through ``examples.libero.episode_runner.LiberoEpisodeRunner`` MUST set
    # ``extra["num_trials_per_task"]`` to this stage's per-phase trial count — the
    # runner derives the canonical global episode_id from it (gate collection
    # §19.B6) and fails fast if it is absent (never falls back to the worker's
    # unrelated main.Args default).
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def server(self) -> ServerEndpoint:
        return ServerEndpoint(self.server_host, self.server_port)


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    """Outcome of running one episode, reported back by the worker (plan §6.2)."""

    task_uid: str
    success: bool
    n_steps: int
    error: str | None = None
    # Per-step observability rows (plan §11.4). Carried back with the result so
    # the driver can merge them centrally without a shared NFS.
    per_step_rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    attempt: int = 1  # echoes the dispatched EpisodeTask.attempt (G2R3 stale-result fence)
    # Wall clock the worker spent on this episode, filled in by ``WorkerLoop``.
    # Without it a phase's worker utilisation -- the thing a capacity change is
    # supposed to move -- cannot be computed from any artifact: the journal
    # records only a terminal timestamp, and the health aggregator is in-memory
    # and never written out. ``None`` for results produced by callers that
    # predate the field.
    duration_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ----------------------------------------------------------------------
# Calibration artifact (warmup/calibration data as a first-class entity)
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CalibrationArtifact:
    """Decouples "calibration data" from "the warmup stage that produced it".

    Supports shared (phase5 G3: one artifact, many eval consumers) and historical
    (phase5 G5: derived from prior raw files, no warmup stage) sources — see
    plan §5.5. A 1:1 ``run_phase`` warmup is just the degenerate case where one
    eval consumes one ``warmup_stage`` artifact.

    Cleanup constraint (G1R2 Item 1): the existing server ``unload_warmup_buffer``
    only accepts one id and derives ``<id>__warmup.jsonl`` server-side; the wire
    never accepts an arbitrary dump name. So every warmup dump MUST be named
    ``<cleanup_id>__warmup``; dump cleanup goes through
    ``unload_warmup_buffer(cleanup_id)``. This keeps everything inside the
    existing protocol (no server change).
    """

    calib_id: str
    source: Literal["warmup_stage", "historical_file"]
    warmup_stage_id: str | None = None  # source == "warmup_stage"
    historical_path: str | None = None  # source == "historical_file"
    cleanup_id: str | None = None  # warmup dump cleanup id; dump named f"{cleanup_id}__warmup"

    def __post_init__(self) -> None:
        if self.source == "warmup_stage":
            if not self.warmup_stage_id or not self.cleanup_id:
                raise ValueError("warmup_stage CalibrationArtifact requires warmup_stage_id and cleanup_id")
        elif self.source == "historical_file":
            if not self.historical_path:
                raise ValueError("historical_file CalibrationArtifact requires historical_path")
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown calibration source: {self.source!r}")

    @property
    def dump_name(self) -> str | None:
        """The server-side dump file stem this artifact is fetched/cleaned by."""
        return f"{self.cleanup_id}__warmup" if self.cleanup_id is not None else None


# ----------------------------------------------------------------------
# Stage graph
# ----------------------------------------------------------------------


@dataclasses.dataclass
class Stage:
    """A group of episodes sharing one (yaml, phase, server) plus calib links.

    ``stage_id`` is the scheduling unit for warmup atomicity (plan §8.2/§9.1):
    warmup stages are retried/resumed whole, never per-episode.
    """

    stage_id: str
    yaml_id: str
    phase: Phase
    server: ServerEndpoint
    episodes: list[EpisodeTask] = dataclasses.field(default_factory=list)
    produces_calib_id: str | None = None  # warmup stage publishes this calib
    consumes_calib_id: str | None = None  # eval stage preloads this calib at on_stage_begin
    # Setup/teardown control info the strategy needs (yaml paths, ids, ...).
    setup: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class StageDependency:
    """``upstream_stage_id`` must fully complete before ``downstream_stage_id``.

    The core runs ``on_stage_complete(upstream)`` then ``on_stage_begin(downstream)``
    in order (plan §5.2/§6.1); there is no single ``barrier`` callable any more —
    the begin/complete lifecycle hooks express ordered setup + handoff.
    """

    upstream_stage_id: str
    downstream_stage_id: str


@dataclasses.dataclass
class TaskGraph:
    """The static plan produced by ``ExperimentStrategy.plan`` (plan §5.2).

    Holds stages + dependencies + calibration artifacts. Ready/blocked state is
    derived by the scheduler, not stored here.
    """

    stages: dict[str, Stage] = dataclasses.field(default_factory=dict)
    dependencies: list[StageDependency] = dataclasses.field(default_factory=list)
    calibrations: dict[str, CalibrationArtifact] = dataclasses.field(default_factory=dict)

    # -- construction helpers --

    def add_stage(self, stage: Stage) -> None:
        if stage.stage_id in self.stages:
            raise ValueError(f"duplicate stage_id: {stage.stage_id!r}")
        self.stages[stage.stage_id] = stage

    def add_dependency(self, upstream_stage_id: str, downstream_stage_id: str) -> None:
        for sid in (upstream_stage_id, downstream_stage_id):
            if sid not in self.stages:
                raise ValueError(f"dependency references unknown stage: {sid!r}")
        self.dependencies.append(StageDependency(upstream_stage_id, downstream_stage_id))

    def add_calibration(self, artifact: CalibrationArtifact) -> None:
        if artifact.calib_id in self.calibrations:
            raise ValueError(f"duplicate calib_id: {artifact.calib_id!r}")
        self.calibrations[artifact.calib_id] = artifact

    # -- queries --

    def upstreams(self, stage_id: str) -> list[str]:
        """Stage ids that must complete before ``stage_id`` may begin."""
        return [d.upstream_stage_id for d in self.dependencies if d.downstream_stage_id == stage_id]

    def downstreams(self, stage_id: str) -> list[str]:
        """Stage ids that depend on ``stage_id`` (its barrier consumers)."""
        return [d.downstream_stage_id for d in self.dependencies if d.upstream_stage_id == stage_id]

    def validate(self) -> None:
        """Reject malformed graphs early: dangling calib refs, cycles, duplicate uids."""
        for stage in self.stages.values():
            for cid in (stage.produces_calib_id, stage.consumes_calib_id):
                if cid is not None and cid not in self.calibrations:
                    raise ValueError(f"stage {stage.stage_id!r} references unknown calib_id {cid!r}")
        self._assert_unique_episode_uids()
        self._assert_acyclic()

    def _assert_unique_episode_uids(self) -> None:
        """No ``task_uid`` may appear in two stages.

        The scheduler indexes episodes by uid into a flat ``{uid: (stage_id,
        task)}`` map, so a duplicate silently rebinds the uid to whichever stage
        was built last. The stage that dispatches it then never sees the result
        -- ``mark_result`` looks up the *other* stage, finds the uid absent from
        its dispatched set, and drops it -- so both stages wait forever and
        ``all_done()`` never holds. Nothing logs.

        Two shapes have to be refused, not one. A uid repeated *inside* one
        stage is dispatched twice from the same pending list, and the second
        dispatch's result is fenced or double-counted; a uid split *across*
        stages rebinds the flat index as described above. Checking only the
        cross-stage case would pass a same-stage duplicate straight through.

        Every strategy shipped before sibling sharding built one stage per yaml,
        which made duplicates unreachable; a strategy that fans one yaml across
        servers is the first shape where a partition bug can produce them, and a
        graph-time check is the only thing standing between that bug and a hung
        run.
        """
        seen: dict[str, str] = {}
        for stage_id in sorted(self.stages):
            for ep in self.stages[stage_id].episodes:
                first = seen.get(ep.task_uid)
                if first is not None:
                    where = (
                        f"twice in stage {stage_id!r}"
                        if first == stage_id
                        else f"in both stage {first!r} and stage {stage_id!r}"
                    )
                    raise ValueError(
                        f"episode {ep.task_uid!r} appears {where}; a uid must "
                        "occur exactly once in the graph or the run cannot complete"
                    )
                seen[ep.task_uid] = stage_id

    def _assert_acyclic(self) -> None:
        # Standard DFS cycle detection over the stage dependency DAG.
        white, grey, black = 0, 1, 2
        color = dict.fromkeys(self.stages, white)

        def visit(sid: str) -> None:
            color[sid] = grey
            for nxt in self.downstreams(sid):
                if color[nxt] == grey:
                    raise ValueError(f"dependency cycle detected at stage {nxt!r}")
                if color[nxt] == white:
                    visit(nxt)
            color[sid] = black

        for sid in self.stages:
            if color[sid] == white:
                visit(sid)
