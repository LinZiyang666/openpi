"""Frozen warm reset specification, plan resolution and self-start seeds.

A warm reset continuation (plan ``logs/warm_continuation_first_class_plan.log.md``
§4.3) replaces only the stage-3 continuation of a WARM_START verdict: where it
starts (cache / self x snapshot / final), which t grid it walks (reset to an
entry level, or shoot from the start's own flow time) and how many Euler steps
it takes. ``WarmResetSpec`` is the frozen view of the yaml block;
``resolve_plan`` / ``resolve_self_plan`` turn it plus the verdict's
``start_t`` and the library schedule into the frozen, hashable plans the
executors and the batching coordinator consume.

Every public entry point validates plans through ``validate_plan`` /
``validate_self_plan``, so a hand-built plan cannot bypass the resolver's
checks. Levels (``entry_t`` / ``step_budget``) are written in the Pi0.5
flow-time convention (1 = noise, 0 = clean) for every schedule; the GR00T grid
is converted to native ascending time by ``groot_loop_grid``.

``MissSpec`` is the frozen view of the sibling ``miss`` block (plain / full
arms: the MISS step count and its evidence), and ``trigger: always`` on a
self-start spec is the library-free self start (``HIT_SELF_ONLY`` decisions).

Public interface: ``WarmResetSpec``, ``MissSpec``, ``WarmResetPlan``,
``SelfStartPlan``, ``resolve_plan``, ``resolve_self_plan``, ``validate_plan``,
``validate_self_plan``, ``groot_loop_grid``, ``native_grid``, ``SelfSeedPolicy``,
``EpisodeDigestSeed``, ``stable_digest_int``, ``private_noise`` and the
``SOURCE_*`` / ``POINT_*`` / ``KIND_*`` / ``TRIGGER_*`` / ``HIT_SELF_ONLY`` names.
Depends on torch and ``openpi.cache.types`` only (jax-free: the GR00T island
imports it).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Optional, Protocol, Sequence

import torch

from openpi.cache.types import DIRECTION_ASC, DenoiseSchedule, schedule_from_id

SOURCE_CACHE = "cache"
SOURCE_SELF = "self"
POINT_SNAPSHOT = "snapshot"
POINT_FINAL = "final"
KIND_RESET = "reset"
KIND_SHOOT = "shoot"
TRIGGER_VERDICT = "verdict"
TRIGGER_ALWAYS = "always"
#: ``__hit_meta__["hit_type"]`` of a library-free self-start decision
#: (``trigger: always``): no verdict ran, so it is neither MISS nor WARM_START.
HIT_SELF_ONLY = "SELF_ONLY"

_SOURCES = frozenset({SOURCE_CACHE, SOURCE_SELF})
_POINTS = frozenset({POINT_SNAPSHOT, POINT_FINAL})
_KINDS = frozenset({KIND_RESET, KIND_SHOOT})


def _is_int(value: Any) -> bool:
    """An integer that is not a bool (``True`` must never pass for a step count)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_level(value: Any) -> bool:
    return (
        isinstance(value, float)
        and math.isfinite(value)
        and 0.0 < value <= 1.0
    )


# ------------------------------------------------------------------
# Spec
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WarmResetSpec:
    """Frozen view of a validated ``warm_reset`` yaml block.

    ``level`` is the reset ``entry_t`` or the shoot ``step_budget`` (Pi0.5
    convention); ``num_steps`` is ``None`` for ``remaining``. ``trigger`` is
    ``verdict`` (a WARM_START verdict runs the block) or ``always`` (library-free
    self start from the block's own ``start_t``; ``None`` under ``verdict``).
    ``digest()`` is the arm identity every evidence row carries.
    """

    source: str
    point: str
    kind: str
    level: float
    num_steps: Optional[int]
    seed_namespace: Optional[str]
    seed_keys: tuple[str, ...]
    evidence_dir: str
    trigger: str = TRIGGER_VERDICT
    start_t: Optional[float] = None

    @classmethod
    def from_config(cls, cfg: Any) -> WarmResetSpec:
        """Freeze a ``WarmResetConfig`` that ``validate_cache_config`` accepted."""
        grid = cfg.grid
        level = grid.entry_t if grid.kind == KIND_RESET else grid.step_budget
        seed = cfg.self_seed
        trigger = str(getattr(cfg, "trigger", TRIGGER_VERDICT))
        start_t = getattr(cfg, "start_t", None)
        return cls(
            source=str(cfg.start.source),
            point=str(cfg.start.point),
            kind=str(grid.kind),
            level=float(level),
            num_steps=None if cfg.num_steps == "remaining" else int(cfg.num_steps),
            seed_namespace=None if seed is None else str(seed.namespace),
            seed_keys=() if seed is None else tuple(str(k) for k in seed.identity_keys),
            evidence_dir=str(cfg.evidence_dir),
            trigger=trigger,
            start_t=None if start_t is None else float(start_t),
        )

    @property
    def self_start(self) -> bool:
        """Whether the start comes from a direct inference on the decision."""
        return self.source == SOURCE_SELF

    @property
    def always(self) -> bool:
        """Whether every decision runs the block without a verdict (library-free self start)."""
        return self.trigger == TRIGGER_ALWAYS

    def digest(self) -> str:
        """sha256 of the canonical JSON of every field.

        A verdict-triggered spec hashes exactly the fields it had before
        ``trigger`` / ``start_t`` existed, so every digest already written into
        a plan or an evidence row stays valid.
        """
        fields = dataclasses.asdict(self)
        if self.trigger == TRIGGER_VERDICT and self.start_t is None:
            del fields["trigger"], fields["start_t"]
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def seed_policy(self) -> EpisodeDigestSeed:
        """The self-start seed policy; refused on a cache arm."""
        if not self.self_start:
            raise ValueError("a cache-start warm reset arm has no self-start seed policy")
        return EpisodeDigestSeed(namespace=str(self.seed_namespace), keys=tuple(self.seed_keys))


@dataclasses.dataclass(frozen=True)
class MissSpec:
    """Frozen view of a validated ``miss`` yaml block (plain / full arms).

    ``num_steps`` is the Euler step count of every MISS decision of the arm;
    ``digest()`` is the arm identity every evidence row carries. It shares the
    session / evidence-wrapper surface of ``WarmResetSpec`` (``self_start`` is
    always False: a MISS arm draws no private noise).
    """

    num_steps: int
    evidence_dir: str

    @classmethod
    def from_config(cls, cfg: Any) -> MissSpec:
        """Freeze a ``MissConfig`` that ``validate_cache_config`` accepted."""
        return cls(num_steps=int(cfg.num_steps), evidence_dir=str(cfg.evidence_dir))

    @property
    def self_start(self) -> bool:
        """A MISS arm never runs a self start."""
        return False

    def digest(self) -> str:
        """sha256 of the canonical JSON of every field, tagged as a MISS arm."""
        payload = json.dumps({"miss": dataclasses.asdict(self)}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def seed_policy(self) -> EpisodeDigestSeed:
        """Refused: a MISS arm has no self-start seed."""
        raise ValueError("a MISS arm has no self-start seed policy")


# ------------------------------------------------------------------
# Plans
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WarmResetPlan:
    """The continuation of one decision, frozen and hashable.

    ``grid_key`` is what the batching coordinator buckets on: the grid depends
    on the schedule, K, kind, level and N, and a shoot grid also on the
    ``start_t`` it starts from. ``source`` / ``point`` never change the grid,
    so cache / self and snapshot / final arms of one grid share a bucket.
    """

    schedule_id: str
    direction: str
    k: int
    start_t: float
    source: str
    point: str
    kind: str
    level: float
    n_steps: int

    def grid_key(self) -> tuple:
        """Hashable identity of the executed t grid."""
        return (
            self.schedule_id,
            self.k,
            self.kind,
            self.level,
            self.n_steps,
            self.start_t if self.kind == KIND_SHOOT else None,
        )

    def flow_times(self) -> tuple[float, ...]:
        """Per-step flow time fed to the model, in the Pi0.5 convention (evidence).

        Pi0.5 replays the device's float32 accumulation on CPU tensors (a single
        IEEE addition does not depend on the device); GR00T converts the native
        ascending grid, ``t = 1 - tau``.
        """
        if self.direction == DIRECTION_ASC:
            taus, _ = native_grid(self)
            return tuple(1.0 - tau for tau in taus)
        dt = torch.tensor(-self.level / self.n_steps, dtype=torch.float32)
        if self.kind == KIND_RESET:
            timestep = torch.tensor(self.level, dtype=torch.float32)
        else:
            grid = torch.tensor(-1.0 / self.k, dtype=torch.float32)
            timestep = torch.tensor(1.0, dtype=torch.float32)
            replay = self.k - schedule_from_id(self.schedule_id).remaining_steps(self.start_t)
            for _ in range(replay):
                timestep = timestep + grid
        times = []
        for _ in range(self.n_steps):
            times.append(float(timestep))
            timestep = timestep + dt
        return tuple(times)


@dataclasses.dataclass(frozen=True)
class SelfStartPlan:
    """The direct K-step inference of a self-start decision.

    ``capture_index`` is the loop step whose input is the snapshot at
    ``start_t`` (``None`` = keep the final action). ``start_t`` stays on the
    plan so every public entry re-checks it is a recoverable point, final
    included. The grid is the full K-step loop, so ``grid_key`` names only the
    schedule: every self arm of one schedule batches together.
    """

    schedule_id: str
    direction: str
    k: int
    start_t: float
    capture_index: Optional[int]

    def grid_key(self) -> tuple:
        """Hashable identity of the executed (full K-step) grid."""
        return (self.schedule_id, self.k)


def _check_schedule_identity(plan: Any, schedule: DenoiseSchedule) -> None:
    if plan.schedule_id != schedule.schedule_id or plan.direction != schedule.direction:
        raise ValueError(
            f"warm reset plan is for {plan.schedule_id!r} ({plan.direction}) but the "
            f"schedule is {schedule.schedule_id!r} ({schedule.direction})"
        )
    if not _is_int(plan.k) or plan.k != schedule.num_steps:
        raise ValueError(
            f"warm reset plan K={plan.k!r} does not match {schedule.schedule_id} "
            f"({schedule.num_steps} steps)"
        )


def validate_plan(plan: Any, schedule: DenoiseSchedule) -> None:
    """Raise ``ValueError`` unless ``plan`` is a well-formed continuation under ``schedule``.

    Checks the schedule identity and K, that ``start_t`` is a recoverable point
    (``snapshot_index``, also for an explicit N), that N is a non-bool integer
    in ``[1, K]``, the enumerations and the level, and that a final start does
    not shoot.
    """
    if not isinstance(plan, WarmResetPlan):
        raise ValueError(f"expected a WarmResetPlan, got {type(plan).__name__}")
    _check_schedule_identity(plan, schedule)
    schedule.snapshot_index(plan.start_t)
    if not _is_int(plan.n_steps) or not 1 <= plan.n_steps <= plan.k:
        raise ValueError(f"warm reset N={plan.n_steps!r} outside [1, {plan.k}]")
    if plan.source not in _SOURCES or plan.point not in _POINTS or plan.kind not in _KINDS:
        raise ValueError(
            f"warm reset plan has source={plan.source!r} point={plan.point!r} kind={plan.kind!r}"
        )
    if not _is_level(plan.level):
        raise ValueError(f"warm reset level={plan.level!r} must be a float in (0, 1]")
    if plan.kind == KIND_SHOOT and plan.point == POINT_FINAL:
        raise ValueError("a final-action start (T = 0) cannot shoot")


def validate_self_plan(plan: Any, schedule: DenoiseSchedule) -> None:
    """Raise ``ValueError`` unless ``plan`` is a well-formed self-start run under ``schedule``.

    The verdict's ``start_t`` must be recoverable even for a final-action
    capture, and a snapshot capture must name exactly its loop step.
    """
    if not isinstance(plan, SelfStartPlan):
        raise ValueError(f"expected a SelfStartPlan, got {type(plan).__name__}")
    _check_schedule_identity(plan, schedule)
    index = schedule.snapshot_index(plan.start_t)
    if plan.capture_index is not None and (
        not _is_int(plan.capture_index) or plan.capture_index != index
    ):
        raise ValueError(
            f"self-start capture_index={plan.capture_index!r} is not the loop step "
            f"{index} of start_t={plan.start_t}"
        )


def resolve_plan(spec: WarmResetSpec, schedule: DenoiseSchedule, start_t: float) -> WarmResetPlan:
    """Freeze the continuation of a WARM_START verdict at ``start_t`` under ``schedule``.

    ``start_t`` is checked against the schedule first, whether N is explicit or
    ``remaining``; ``ValueError`` on any invalid input.
    """
    schedule.snapshot_index(start_t)
    n_steps = schedule.remaining_steps(start_t) if spec.num_steps is None else spec.num_steps
    plan = WarmResetPlan(
        schedule_id=schedule.schedule_id,
        direction=schedule.direction,
        k=schedule.num_steps,
        start_t=float(start_t),
        source=spec.source,
        point=spec.point,
        kind=spec.kind,
        level=spec.level,
        n_steps=n_steps,
    )
    validate_plan(plan, schedule)
    return plan


def resolve_self_plan(spec: WarmResetSpec, schedule: DenoiseSchedule, start_t: float) -> SelfStartPlan:
    """Freeze the direct inference of a self-start decision; ``ValueError`` on a cache arm."""
    if not spec.self_start:
        raise ValueError("resolve_self_plan needs a self-start spec")
    index = schedule.snapshot_index(start_t)
    plan = SelfStartPlan(
        schedule_id=schedule.schedule_id,
        direction=schedule.direction,
        k=schedule.num_steps,
        start_t=float(start_t),
        capture_index=index if spec.point == POINT_SNAPSHOT else None,
    )
    validate_self_plan(plan, schedule)
    return plan


def groot_loop_grid(plan: WarmResetPlan) -> tuple[Optional[float], float]:
    """``(tau0, dt)`` of a GR00T continuation in native ascending time.

    ``tau0 is None`` selects upstream's own loop (``denoise_loop`` from index 0,
    ``tau_i = i / float(N)``, ``dt = 1 / N``): a reset to pure noise. Otherwise
    the loop is ``tau_i = tau0 + i * dt`` with the expression order of the
    step_diag reference (``dt = (1 - tau0) / N`` for a reset below noise,
    ``tau0 = start_t`` and ``dt = level / N`` for a shoot).
    """
    if plan.direction != DIRECTION_ASC:
        raise ValueError(f"{plan.schedule_id} is not an ascending (GR00T) schedule")
    n = plan.n_steps
    if plan.kind == KIND_SHOOT:
        return float(plan.start_t), plan.level / n
    if plan.level == 1.0:
        return None, 1.0 / n
    tau0 = 1.0 - plan.level
    return tau0, (1.0 - tau0) / n


def native_grid(plan: WarmResetPlan) -> tuple[tuple[float, ...], float]:
    """``(tau_i per step, dt)`` of a GR00T continuation in native time (evidence)."""
    tau0, dt = groot_loop_grid(plan)
    if tau0 is None:
        return tuple(i / float(plan.n_steps) for i in range(plan.n_steps)), dt
    return tuple(tau0 + i * dt for i in range(plan.n_steps)), dt


# ------------------------------------------------------------------
# Self-start seeds and noise
# ------------------------------------------------------------------


def stable_digest_int(*parts: Any) -> int:
    """Deterministic 63-bit integer from the ``|``-joined ``str`` of ``parts``.

    Byte-for-byte the step_diag recorder's algorithm (sha256, first 8 bytes
    little-endian, top bit cleared), never Python's salted ``hash()``.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def private_noise(seed: int, shape: Sequence[int]) -> torch.Tensor:
    """Standard-normal float32 CPU noise from a private generator; the global RNG is untouched."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=gen, dtype=torch.float32)


class SelfSeedPolicy(Protocol):
    """Strategy deciding the private-noise seed of a self-start decision."""

    def seed(self, identity: Mapping[str, Any], decision_idx: int) -> int:
        """The seed of decision ``decision_idx`` of the episode ``identity`` names."""
        ...


@dataclasses.dataclass(frozen=True)
class EpisodeDigestSeed:
    """``stable_digest_int(namespace, *identity[keys], decision_idx, "self")``.

    Neither ``yaml_id`` nor ``bundle_id`` enters the digest, so every self arm
    of one namespace draws the same noise for one (episode, decision); an
    ``attempt`` key makes a retry draw new noise.
    """

    namespace: str
    keys: tuple[str, ...]

    def missing_keys(self, identity: Mapping[str, Any]) -> list[str]:
        """Seed keys ``identity`` does not carry."""
        return [k for k in self.keys if k not in identity]

    def seed(self, identity: Mapping[str, Any], decision_idx: int) -> int:
        """See class docstring; ``KeyError`` if a seed key is missing."""
        missing = self.missing_keys(identity)
        if missing:
            raise KeyError(f"self-start seed keys {missing} missing from the episode identity")
        return stable_digest_int(
            self.namespace, *[identity[k] for k in self.keys], int(decision_idx), "self"
        )


__all__ = [
    "HIT_SELF_ONLY",
    "KIND_RESET",
    "KIND_SHOOT",
    "POINT_FINAL",
    "POINT_SNAPSHOT",
    "SOURCE_CACHE",
    "SOURCE_SELF",
    "TRIGGER_ALWAYS",
    "TRIGGER_VERDICT",
    "EpisodeDigestSeed",
    "MissSpec",
    "SelfSeedPolicy",
    "SelfStartPlan",
    "WarmResetPlan",
    "WarmResetSpec",
    "groot_loop_grid",
    "native_grid",
    "private_noise",
    "resolve_plan",
    "resolve_self_plan",
    "stable_digest_int",
    "validate_plan",
    "validate_self_plan",
]
