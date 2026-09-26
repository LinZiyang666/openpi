"""Per-connection warm reset runtime: session state, step counting and assembly.

One ``WarmResetSession`` per served connection is the single source of truth
the evidence wrapper and the executor share (plan
``logs/warm_continuation_first_class_plan.log.md`` §4.4, §4.7.2): the wrapper
opens the episode and every decision (``begin_episode`` / ``begin_decision``),
the executor reads the decision index and identity for the self-start seed and
counts every stage-3 binding call and its measured steps. Nothing is shared
across connections and no model object is patched.

``_build`` is the model-agnostic assembler behind ``build_pi05_warm_reset`` /
``build_groot_warm_reset``: it returns ``None`` when the yaml has no
``warm_reset`` block, so the serving entry points change nothing for every
existing config.

Public interface: ``WarmResetSession``, ``WarmResetParts``,
``refuse_warm_reset``, ``decision_meta``, ``META_SCHEMA``.
Internal but shared with the executors: ``_CountedStep``, ``_build``.
Depends on ``openpi.cache.config`` (validation error, schedule) and
``openpi.cache.warm_reset.types``; jax-free.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import uuid
from typing import Any, Callable, Mapping, Optional

from openpi.cache.config import ConfigValidationError, effective_denoise_schedule
from openpi.cache.warm_reset.types import MissSpec, WarmResetSpec

#: Schema tag of ``__hit_meta__["warm_reset"]``.
META_SCHEMA = "warm_reset_meta_v1"

# The episode_start fields every identity carries; ``extra_metadata`` may repeat
# them only with the same value.
_BASE_IDENTITY_KEYS = ("experiment", "task", "episode_id")


# ------------------------------------------------------------------
# Session
# ------------------------------------------------------------------


class WarmResetSession:
    """Episode identity, decision index and per-decision call counters of one connection.

    ``episode_seq`` counts the connection's episodes from 0. ``begin_decision``
    assigns the next decision index and zeroes every per-decision counter at
    once; the executor then adds one call per stage-3 binding entry and the
    measured steps of each successful return. A connection's requests are
    sequential, so the session is not shared between threads.
    """

    def __init__(self, spec: WarmResetSpec, *, conn_id: Optional[str] = None) -> None:
        self.spec = spec
        self.conn_id = conn_id or uuid.uuid4().hex
        self.episode_seq = -1
        self.identity: Optional[Mapping[str, Any]] = None
        self.decision_idx: Optional[int] = None
        self._n_decisions = 0
        self._seed_policy = spec.seed_policy() if spec.self_start else None
        self._zero_counters()

    def _zero_counters(self) -> None:
        self.continuation_calls = 0
        self.self_start_calls = 0
        self.continuation_nfe = 0
        self.self_direct_nfe = 0

    @property
    def in_episode(self) -> bool:
        """Whether an episode is open on this connection."""
        return self.identity is not None

    def begin_episode(
        self,
        *,
        experiment: str,
        task: str,
        episode_id: int,
        extra_metadata: Optional[dict],
    ) -> None:
        """Open the next episode and freeze its identity.

        The identity is ``{experiment, task, episode_id}`` plus every
        ``extra_metadata`` key; a metadata key that repeats one of the three
        with a different value is refused rather than allowed to override it.
        A self-start arm refuses an identity missing any seed key here, before
        the first decision.
        """
        base = {"experiment": str(experiment), "task": str(task), "episode_id": int(episode_id)}
        extra = dict(extra_metadata or {})
        for key in _BASE_IDENTITY_KEYS:
            if key in extra and extra[key] != base[key]:
                raise ValueError(
                    f"episode_start extra_metadata[{key!r}]={extra[key]!r} conflicts with "
                    f"the episode's {key}={base[key]!r}"
                )
        identity = {**extra, **base}
        if self._seed_policy is not None:
            missing = self._seed_policy.missing_keys(identity)
            if missing:
                raise ValueError(
                    f"self-start seed keys {missing} are not in this episode's "
                    f"episode_start identity {sorted(identity)}"
                )
        self.episode_seq += 1
        self.identity = identity
        self.decision_idx = None
        self._n_decisions = 0
        self._zero_counters()

    def begin_decision(self) -> int:
        """Assign the next decision index and zero the per-decision counters."""
        if self.identity is None:
            raise RuntimeError(
                "warm reset: request outside an episode (no episode_start); a warm "
                "reset arm cannot attribute or seed a decision without one"
            )
        self.decision_idx = self._n_decisions
        self._n_decisions += 1
        self._zero_counters()
        return self.decision_idx

    def end_episode(self) -> int:
        """Close the episode; returns the server-side decision count for ``finalize``."""
        n = self._n_decisions
        self.identity = None
        self.decision_idx = None
        return n

    def self_seed(self) -> int:
        """Private-noise seed of the open decision (``RuntimeError`` outside one)."""
        if self._seed_policy is None:
            raise RuntimeError("a cache-start warm reset arm has no self-start seed")
        if self.identity is None or self.decision_idx is None:
            raise RuntimeError("self-start seed requested outside an episode / decision")
        return self._seed_policy.seed(self.identity, self.decision_idx)

    def count_continuation_call(self) -> None:
        """One more entry into the continuation binding in this decision."""
        self.continuation_calls += 1

    def count_self_start_call(self) -> None:
        """One more entry into the self-start binding in this decision."""
        self.self_start_calls += 1

    def add_continuation_steps(self, steps: int) -> None:
        """Add the measured steps of a continuation call that returned."""
        self.continuation_nfe += int(steps)

    def add_self_start_steps(self, steps: int) -> None:
        """Add the measured steps of a self-start call that returned."""
        self.self_direct_nfe += int(steps)


class _CountedStep:
    """Call-local step counter: forwards every call unchanged and counts it.

    Built inside one executor call and wrapped around the model's own step
    function, so nothing on the shared model instance is replaced and two
    connections never share a counter.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._fn(*args, **kwargs)


def decision_meta(
    spec: WarmResetSpec,
    spec_digest: str,
    plan: Any,
    session: WarmResetSession,
    *,
    t: tuple[float, ...],
    dt: float,
    self_seed: Optional[int],
    extra: Optional[dict] = None,
) -> dict:
    """``__hit_meta__["warm_reset"]`` of one decision (plan §4.7.1).

    Plan values (``kind`` ... ``n_steps``, ``t``, ``dt``) come from the frozen
    plan; ``continuation_nfe`` / ``n_stage3_calls`` / ``self_start_calls`` /
    ``self_direct_nfe`` are the session's measured counters, never the budget.
    Only a self-start arm carries ``self_start`` / ``self_seed`` /
    ``self_direct_nfe``; ``decision_nfe`` prices the decision (K + N on a self
    arm, N on a cache arm).
    """
    meta = {
        "schema": META_SCHEMA,
        "spec_digest": spec_digest,
        "kind": plan.kind,
        "source": plan.source,
        "point": plan.point,
        "level": plan.level,
        "start_t": plan.start_t,
        "schedule_id": plan.schedule_id,
        "k": plan.k,
        "n_steps": plan.n_steps,
        "continuation_nfe": session.continuation_nfe,
        "n_stage3_calls": session.continuation_calls,
        "self_start_calls": session.self_start_calls,
        "t": list(t),
        "dt": dt,
        "decision_nfe": session.continuation_nfe
        + (session.self_direct_nfe if spec.self_start else 0),
    }
    if extra:
        meta.update(extra)
    if spec.self_start:
        meta["self_start"] = True
        meta["self_seed"] = int(self_seed)
        meta["self_direct_nfe"] = session.self_direct_nfe
    return meta


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WarmResetParts:
    """What a serving entry point needs for one connection.

    ``executor`` is injected into the interceptor (``warm_reset=``); ``wrap``
    puts the evidence wrapper around that interceptor. Both share ``session``.
    For a ``miss`` block (plain / full arm) ``spec`` is a ``MissSpec``,
    ``executor`` is ``None`` and ``miss_num_steps`` is injected instead
    (``miss_num_steps=``); it is ``None`` for every warm reset arm.
    """

    spec: Any
    session: WarmResetSession
    executor: Any
    wrap: Callable[[Any], Any]
    miss_num_steps: Optional[int] = None

    def interceptor_kwargs(self) -> dict:
        """The interceptor keyword this arm injects: ``warm_reset=`` or ``miss_num_steps=``."""
        if self.miss_num_steps is not None:
            return {"miss_num_steps": self.miss_num_steps}
        return {"warm_reset": self.executor}


def _yaml_sha256(path: Optional[str]) -> Optional[str]:
    """sha256 of the loaded yaml text (the trace runtime's convention), or None."""
    if not path:
        return None
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build(
    config: Any,
    *,
    family: str,
    bundle_id: str,
    yaml_id: Optional[str],
    yaml_path: Optional[str],
    executor_factory: Callable[..., Any],
    wrapper_factory: Callable[..., Any],
) -> Optional[WarmResetParts]:
    """Assemble executor + evidence wrapper for one connection, or ``None`` without a block.

    ``yaml_id`` (the registered bundle's yaml id, ``None`` for a startup yaml)
    and ``bundle_id`` (the connection's selected bundle) are recorded
    independently; ``yaml_path`` is hashed for ``yaml_sha256``. A ``miss``
    block yields the evidence wrapper and ``miss_num_steps`` without an
    executor; the validator refuses both blocks in one config.
    """
    cfg = getattr(config, "warm_reset", None)
    miss = getattr(config, "miss", None)
    if cfg is None and miss is None:
        return None
    if cfg is not None and miss is not None:
        raise ConfigValidationError("miss and warm_reset are mutually exclusive (one evidence stream per arm).")
    schedule = effective_denoise_schedule(config)
    if miss is not None:
        spec: Any = MissSpec.from_config(miss)
        session = WarmResetSession(spec)
        executor = None
    else:
        spec = WarmResetSpec.from_config(cfg)
        session = WarmResetSession(spec)
        executor = executor_factory(spec, session, schedule=schedule)
    yaml_sha256 = _yaml_sha256(yaml_path)

    def wrap(policy: Any) -> Any:
        return wrapper_factory(
            policy,
            session=session,
            family=family,
            bundle_id=str(bundle_id),
            yaml_id=yaml_id,
            yaml_sha256=yaml_sha256,
            schedule_id=schedule.schedule_id,
            k=schedule.num_steps,
        )

    return WarmResetParts(
        spec=spec,
        session=session,
        executor=executor,
        wrap=wrap,
        miss_num_steps=spec.num_steps if miss is not None else None,
    )


def refuse_warm_reset(config: Any, *, where: str) -> None:
    """Raise ``ConfigValidationError`` if ``config`` carries a ``warm_reset`` or ``miss`` block.

    For serving paths that never execute the warm reset continuation or the
    per-bundle MISS step count (trace, shadow / calibration loggers): the
    block would otherwise be silently ignored and the arm would run the exact
    resume / the process-level MISS under its name.
    """
    if getattr(config, "warm_reset", None) is not None:
        raise ConfigValidationError(
            f"{where} does not support a warm_reset block: it would serve the exact "
            "resume under a warm reset arm's name."
        )
    if getattr(config, "miss", None) is not None:
        raise ConfigValidationError(
            f"{where} does not support a miss block: it would serve the process-level "
            "MISS step count under a plain / full arm's name."
        )


__all__ = [
    "META_SCHEMA",
    "WarmResetParts",
    "WarmResetSession",
    "decision_meta",
    "refuse_warm_reset",
]
