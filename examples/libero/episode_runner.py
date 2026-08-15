"""LIBERO episode execution kernel as an conductor ``EpisodeRunner``.

Worker-side execution leaf (plan §4.2). Rather than re-implement the (verified)
LIBERO inference loop, this adapter *reuses* ``examples.libero.main._run_episode``
and only adapts it to the conductor contract: connection reuse + bundle
binding, episode_start/end lifecycle, progress reporting, and per-step
``__hit_meta__`` collection (plan §11.4). The server connection is reused across
episodes of the same (server, bundle); re-connect/re-select only on change
(plan §6.4).

The LIBERO-specific bits (``run_episode_fn``, ``episode_setup``, client) are
injected so the adapter logic is unit-testable without LIBERO/CUDA; the real
defaults are wired by ``worker_entry``.

Producer contract: every dispatched ``EpisodeTask`` MUST carry
``extra["num_trials_per_task"]`` (the stage's per-phase trial count). ``run()``
derives the canonical global episode_id from it and fails fast if absent — it
never reads the worker's unrelated ``main.Args`` default (both in-repo
strategies stamp it; a custom strategy must too).

Coupling map:
  DEPENDS ON:  openpi.conductor (task/worker), examples.libero.collect_util,
               examples.libero.main (lazy — the sim loop only)
  CONSUMED BY: examples.libero.worker_entry
"""

from __future__ import annotations

# typing.Callable (subscriptable in py3.8 LIBERO worker env); the EpisodeSetup
# alias below is module-level and evaluated at import, unlike the function
# annotations which are deferred by `from __future__ import annotations`.
from typing import Callable
import contextlib
import time
from typing import Any

from openpi.conductor import task as _task
from openpi.conductor.worker import EpisodeRunner
from openpi.conductor.worker import ProgressCallback

# Pure collection helpers (LIBERO-free): the SAME episode-id formula + schema
# version the standalone harness uses, imported without pulling in the simulator.
from examples.libero.collect_util import COLLECTOR_SCHEMA_VERSION as _COLLECTOR_SCHEMA_VERSION
from examples.libero.collect_util import compute_global_episode_id as _compute_global_episode_id

# episode_setup(task) -> (env, initial_state, task_description, max_steps)
EpisodeSetup = Callable[[_task.EpisodeTask], tuple]


def _global_episode_id(task: _task.EpisodeTask, num_trials_per_task: int) -> int:
    # Canonical global episode id — the SAME collect_util.compute_global_episode_id
    # the standalone harness uses, so conductor and standalone emit the SAME
    # episode_id for the same (task_id, episode_idx). Never the subset position,
    # never a task_uid component.
    return _compute_global_episode_id(task.task_id, task.episode_idx, num_trials_per_task)


def _hit_row(task: _task.EpisodeTask, step: int, hit: dict, num_trials_per_task: int) -> dict:
    # Canonical per-step identity/schema (matches examples/libero/main.py's
    # standalone rows so both harnesses join on the same keys). ``task_uid`` is
    # the single authoritative join key; ``episode_id`` is the canonical global id.
    return {
        "yaml_id": task.yaml_id,
        "task_id": task.task_id,
        "subset_init_state_idx": task.episode_idx,
        "orig_init_state_idx": task.orig_init_state_idx,
        "episode_id": _global_episode_id(task, num_trials_per_task),
        "task_uid": task.task_uid,
        "phase": task.phase,
        "step_idx": step,
        "hit_type": hit.get("hit_type"),
        "start_t": hit.get("start_t"),
        "winner_id": hit.get("winner_id"),
        "cp1_score": hit.get("cp1_score"),
        # searched from the always-on __hit_meta__ (server-side gates like
        # follow_winner have no client stamp). When collect_meta is present
        # (N1/N4 client stamp, or a collection run) infer_recorder overrides
        # this with collect_meta.searched, preserving that provenance.
        "searched": hit.get("searched"),
        # Ablation routing provenance: "override" when a sidecar executor
        # produced this step's action; absent/None on all non-routed steps.
        "executor": hit.get("executor"),
        # X14 RL router provenance ({decision_idx, arm_sampled, arm_executed,
        # probs, ...}); None for every non-router yaml. ``decision_idx`` — not
        # ``step_idx`` — is the join key against the server-side dump: this
        # row's ``step_idx`` is the physical env step, which advances by
        # ``replan_steps`` between consecutive inference calls.
        "router_outputs": hit.get("router_outputs"),
    }


# Identity keys a strategy may stamp onto ``EpisodeTask.extra`` for the RL
# router's server-side dump. ``task_uid`` / ``attempt`` are deliberately NOT in
# this list: they are forced from the dispatched task's top-level fields, since
# ``extra`` is built once by the strategy and does not follow a requeue.
_ROUTER_IDENTITY_KEYS = ("run_id", "batch_id", "weights_version")


def _episode_extra_metadata(task: _task.EpisodeTask) -> dict:
    """Build the ``episode_start`` metadata: task identity + router passthrough.

    The dispatched task is authoritative for ``task_uid`` / ``attempt``. A
    strategy that also wrote them into ``extra`` is stale by construction after
    a requeue (the scheduler bumps the dispatch generation via
    ``dataclasses.replace``, which never touches ``extra``), so a disagreement
    is a producer bug and fails loud rather than silently mislabelling every
    dumped step of the episode.
    """
    extra = dict(task.extra or {})
    meta: dict = {
        "task_id": task.task_id,
        "orig_init_state_idx": task.orig_init_state_idx,
    }
    for key in _ROUTER_IDENTITY_KEYS:
        if extra.get(key) is not None:
            meta[key] = extra[key]
    for key, authoritative in (("task_uid", task.task_uid), ("attempt", task.attempt)):
        stamped = extra.get(key)
        if stamped is not None and stamped != authoritative:
            raise ValueError(
                f"EpisodeTask {task.task_uid!r} extra[{key!r}]={stamped!r} conflicts with "
                f"the dispatched task's {key}={authoritative!r}; the top-level field is "
                "authoritative (extra does not follow a requeue). Remove it from the "
                "strategy's extra payload."
            )
        meta[key] = authoritative
    return meta


def default_client_factory(server: _task.ServerEndpoint):
    """Connect a real WebSocket client to the assigned server."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy(host=server.host, port=server.port)


def _default_run_episode(*args, **kwargs):
    from examples.libero import main as _m

    return _m._run_episode(*args, **kwargs)  # noqa: SLF001 - intentional reuse of main internals


class LiberoEpisodeRunner(EpisodeRunner):
    """Adapts ``main._run_episode`` to the conductor EpisodeRunner contract."""

    def __init__(
        self,
        args: Any,
        episode_setup: EpisodeSetup,
        *,
        client_factory: Callable[[_task.ServerEndpoint], Any] = default_client_factory,
        run_episode_fn: Callable[..., tuple] = _default_run_episode,
    ) -> None:
        self._args = args
        self._episode_setup = episode_setup
        self._client_factory = client_factory
        self._run_episode_fn = run_episode_fn
        self._client: Any | None = None
        self._client_server: str | None = None
        self._bundle: str | None = None

        # Conductor supports robot_state-only gate collection: vision fields risk
        # the 64 MiB per-episode EpisodeResult frame, so vision is standalone-only.
        # Fail fast here at construction (before any episode) instead of deep in
        # the per-step callback, where main._run_episode's broad ``except
        # Exception`` would swallow the error and end the episode as a failure.
        if getattr(args, "collect_gate_dir", "") and getattr(args, "collect_embeddings", "pooled") != "none":
            raise ValueError(
                "conductor gate-collection supports robot_state only; pass "
                "--collect-embeddings none. Vision collection is standalone-only "
                "(per-episode frame vs the 64 MiB EpisodeResult cap)."
            )

    def _ensure_client(self, task: _task.EpisodeTask) -> Any:
        if self._client is None or self._client_server != task.server.key:
            self.close()
            self._client = self._client_factory(task.server)
            self._client_server = task.server.key
            self._bundle = None
        if self._bundle != task.bundle_id:
            self._client.select_bundle(task.bundle_id)
            self._bundle = task.bundle_id
        return self._client

    def run(self, task: _task.EpisodeTask, report: ProgressCallback) -> _task.EpisodeResult:
        client = self._ensure_client(task)
        env, initial_state, task_description, max_steps = self._episode_setup(task)
        client.episode_start(
            experiment=task.experiment,
            task=task_description,
            episode_id=task.episode_idx,
            extra_metadata=_episode_extra_metadata(task),
        )
        per_step: list[dict] = []
        # Per-episode collection state (un-swallowable vision fail-fast +
        # provenance). infer_recorder must NOT raise: a raise inside
        # main._run_episode() is swallowed by its broad ``except Exception`` and
        # merely ends the episode as a failure. We record a violation and re-raise
        # AFTER _run_episode returns (outside that try), where it propagates.
        _ep = {"violation": None, "searched_all": True, "collected": False,
               "fields": None, "kb_id": None}
        # Canonical episode_id needs the REAL per-phase trial count, which only
        # the strategy knows and stamps onto EpisodeTask.extra. Never read the
        # worker's unrelated main.Args default (50): a warmup task (N=2) and an
        # eval task (N=10) must map to distinct global ids matching standalone.
        _ct: dict = {}  # client-side latency accumulators (main._run_episode fills)
        # Signature-filtered pass-through: injected run_episode_fn doubles that
        # predate the client_timing keyword keep working unchanged.
        import inspect

        try:
            _params = inspect.signature(self._run_episode_fn).parameters
            _accepts_ct = "client_timing" in _params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in _params.values()
            )
        except (TypeError, ValueError):
            _accepts_ct = False
        _ct_kwargs = {"client_timing": _ct} if _accepts_ct else {}
        _nt = (task.extra or {}).get("num_trials_per_task")
        if _nt is None:
            raise ValueError(
                f"EpisodeTask {task.task_uid!r} is missing extra['num_trials_per_task']; "
                "the ExperimentStrategy must stamp the per-phase trial count so conductor "
                "and standalone emit the same canonical global episode_id (gate §19.B6)."
            )
        t0 = time.monotonic()

        def infer_recorder(step_idx: int, hit_meta: dict, collect_meta=None) -> None:
            row = _hit_row(task, step_idx, hit_meta or {}, _nt)
            if collect_meta is not None:
                _ep["collected"] = True
                searched = collect_meta.get("searched", True)
                row["searched"] = searched
                _ep["searched_all"] = _ep["searched_all"] and bool(searched)
                row["collector_schema_version"] = _COLLECTOR_SCHEMA_VERSION
                if collect_meta.get("kb_id"):
                    _ep["kb_id"] = collect_meta.get("kb_id")
                collect = collect_meta.get("collect")
                if collect is not None:
                    _ep["fields"] = sorted(collect.keys())
                    # Conductor supports robot_state only: record the violation;
                    # the raise happens after _run_episode (un-swallowable).
                    extra = [k for k in collect if k != "robot_state"]
                    if extra and _ep["violation"] is None:
                        _ep["violation"] = extra
                    row["collect"] = collect
            per_step.append(row)

        def step_callback(step: int) -> None:
            rate = (step + 1) / max(time.monotonic() - t0, 1e-6)
            report(step, rate, None)

        success = False
        n_steps = 0
        try:
            result = self._run_episode_fn(
                env,
                client,
                initial_state,
                task_description,
                self._args,
                max_steps,
                infer_recorder=infer_recorder,
                step_callback=step_callback,
                **_ct_kwargs,
            )
            # Un-swallowable conductor vision fail-fast: raising here (outside
            # main._run_episode's try) propagates to the worker instead of being
            # caught and downgraded to a failed episode.
            if _ep["violation"] is not None:
                raise ValueError(
                    "conductor gate-collection got vision/prompt fields "
                    f"{_ep['violation']}; only robot_state is supported under "
                    "conductor. Vision collection is standalone-only (per-episode "
                    "frame vs the 64 MiB EpisodeResult cap)."
                )
            # Positional dependency on main._run_episode's 5-tuple return
            # (success, images, timestamps, traj, final_env_timestep); if that
            # contract changes order, n_steps below would silently misread.
            success = bool(result[0])
            n_steps = int(result[4]) if len(result) > 4 and result[4] is not None else len(per_step)
        finally:
            with contextlib.suppress(Exception):
                client.episode_end(success=success)
        # Episode-summary provenance row (plan §2.5): distinguished by ``_kind``
        # so per-step analysis filters it out. Carries the join key + provenance
        # not derivable from per-step rows (seed, kb_id).
        if _ep["collected"]:
            per_step.append({
                "_kind": "episode_summary",
                "task_uid": task.task_uid,
                "yaml_id": task.yaml_id,
                "task_id": task.task_id,
                "subset_init_state_idx": task.episode_idx,
                "episode_id": _global_episode_id(task, _nt),
                "phase": task.phase,
                "seed": getattr(self._args, "seed", None),
                "num_steps": len(per_step),
                "success": success,
                "searched_all": _ep["searched_all"],
                "collect_fields": _ep["fields"],
                "kb_id": _ep["kb_id"],
                "collector_schema_version": _COLLECTOR_SCHEMA_VERSION,
                "attempt": task.attempt,
            })
        if _ct:
            # Client latency persistence (ablation plan latency artifact): one
            # summary row per episode, filtered from verdict analysis by _kind.
            per_step.append({
                "_kind": "client_timing", "task_uid": task.task_uid,
                "yaml_id": task.yaml_id, "task_id": task.task_id,
                "subset_init_state_idx": task.episode_idx, **_ct,
            })
        return _task.EpisodeResult(task.task_uid, success=success, n_steps=n_steps, per_step_rows=per_step)

    def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
        self._client = None
        self._client_server = None
        self._bundle = None
