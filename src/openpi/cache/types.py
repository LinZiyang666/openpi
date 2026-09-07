"""Base enumerations shared across the cache subsystem.

Kept in a separate module so that both timing.py and storage_types.py can
import from here without creating circular dependencies.

Coupling map:
  DEPENDS ON:  nothing (leaf module)
  CONSUMED BY: KeyBuilder (field name constants), Orchestrator (CheckpointID),
               storage_types (CheckpointID), cache_storage (via storage_types),
               config / judge / surface_judge (DenoiseSchedule + canonical
               timesteps), collectors and artifact builders (schedule ids)
  IF CHANGED:  All consumers must update field references
"""

import re
from dataclasses import dataclass
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Canonical query field names
# ---------------------------------------------------------------------------

# These constants are the only valid keys in CacheEntry.query_keys and
# QuerySpec.query_keys.  Each backend declares the subset it stores via
# VectorStoreBackend.vector_dims.
VISION_0 = "vision_0"
VISION_1 = "vision_1"
VISION_2 = "vision_2"
PROMPT_EMB = "prompt_emb"
ROBOT_STATE = "robot_state"
# CP2 post-backbone single key (ActionCache-style baseline): the backbone's
# prefix output projected to a compact vector by cp2_vlm_key_builder.
VLM_OUT = "vlm_out"

CACHE_QUERY_FIELDS: frozenset[str] = frozenset(
    {
        VISION_0,
        VISION_1,
        VISION_2,
        PROMPT_EMB,
        ROBOT_STATE,
        VLM_OUT,
    }
)


# ---------------------------------------------------------------------------
# Flow-matching denoise schedule identity
# ---------------------------------------------------------------------------
#
# A warm-start payload is a snapshot x_t taken part-way through a flow-matching
# loop, and "t" only means something relative to the loop that produced it.
# Pi0.5 runs 10 Euler steps with time falling 1 -> 0; GR00T N1.5 runs
# `num_inference_timesteps` steps with time rising 0 -> 1, and that count is a
# runtime property of the served policy (4 baked into the RoboCasa checkpoint,
# 8 by CLI override on LIBERO), not a constant of the model family. Two
# libraries with the same geometry but different loops would otherwise
# validate against each other while every cached x_t meant something else.
#
# The schedule is therefore an explicit identity carried by every producer and
# consumer of intermediates: the collector stamps it on the HDF5 file, the
# artifact builder copies it into the pickle, the backend records it on load,
# the config names it, and the assembly-time binding check refuses any
# mismatch. Nothing infers it from a model name, and the GR00T id is derived
# from the live step count so no literal step count exists in code.

DIRECTION_DESC = "noise_to_clean_desc"
"""Flow time runs 1 -> 0 (Pi0.5): step i consumes x at t = 1 - i/N."""

DIRECTION_ASC = "noise_to_clean_asc"
"""Flow time runs 0 -> 1 (GR00T N1.5): step i consumes x at t = i/N."""

_GROOT_SCHEDULE_RE = re.compile(r"^groot_n15_k(?P<k>[1-9][0-9]*)_v1$")


def _round4(value: float) -> float:
    return round(float(value), 4)


@dataclass(frozen=True)
class DenoiseSchedule:
    """Identity of one flow-matching denoise loop and the index <-> t mapping.

    ``schedule_id`` is the sole authority; ``num_steps`` and ``direction`` are
    the two facts every derived quantity is computed from. All producers and
    consumers of ``CachePayload.intermediates`` must go through this object
    for index/timestep arithmetic rather than re-deriving a direction formula
    -- the Pi0.5 formula silently reverses GR00T's remaining-step count.
    """

    schedule_id: str
    num_steps: int
    direction: str

    def __post_init__(self) -> None:
        if self.num_steps < 2:
            raise ValueError(
                f"{self.schedule_id}: num_steps must be >= 2, got {self.num_steps}"
            )
        if self.direction not in (DIRECTION_DESC, DIRECTION_ASC):
            raise ValueError(
                f"{self.schedule_id}: unknown direction {self.direction!r}"
            )

    # -- snapshot geometry ---------------------------------------------------

    @property
    def timesteps(self) -> tuple[float, ...]:
        """Recoverable snapshot timesteps in loop-execution order (N-1 of them).

        Index 0 is the pure-noise start and is never a warm-start point, so it
        is excluded here exactly as the collectors exclude it on disk.
        """
        return tuple(self.snapshot_t(i) for i in range(1, self.num_steps))

    @property
    def timestep_set(self) -> frozenset[float]:
        """``timesteps`` as a set, for membership checks."""
        return frozenset(self.timesteps)

    def snapshot_t(self, index: int) -> float:
        """Flow time of the x consumed by loop step ``index`` (0 <= index < N)."""
        if not 0 <= index < self.num_steps:
            raise ValueError(
                f"{self.schedule_id}: snapshot index {index} outside [0, {self.num_steps})"
            )
        if self.direction == DIRECTION_DESC:
            return _round4(1.0 - index / self.num_steps)
        return _round4(index / self.num_steps)

    def snapshot_index(self, t: float) -> int:
        """Loop step whose input is x_t; raises if ``t`` is not a snapshot point.

        Half-up rounding on purpose: Python's ``round`` is banker's rounding and
        would put an exact .5 boundary on the wrong side.
        """
        if self.direction == DIRECTION_DESC:
            raw = (1.0 - float(t)) * self.num_steps
        else:
            raw = float(t) * self.num_steps
        index = int(raw + 0.5)
        if not 1 <= index < self.num_steps or self.snapshot_t(index) != _round4(t):
            raise ValueError(
                f"{self.schedule_id}: t={t} is not a recoverable timestep; "
                f"valid: {list(self.timesteps)}"
            )
        return index

    def remaining_steps(self, t: float) -> int:
        """Euler steps still to run when resuming from the snapshot at ``t``."""
        return self.num_steps - self.snapshot_index(t)

    def replay_timestep(self, index: int) -> float:
        """Flow time the model itself computes for step ``index``, un-rounded.

        Descending (Pi0.5) accumulates ``dt`` from 1.0 the way the production
        loop does, so a resumed run sees the same bf16-representable value as
        an uninterrupted one; ascending (GR00T) is ``index / N`` exactly as the
        upstream loop writes it before bucketing.
        """
        if not 0 <= index < self.num_steps:
            raise ValueError(
                f"{self.schedule_id}: replay index {index} outside [0, {self.num_steps})"
            )
        if self.direction == DIRECTION_ASC:
            return index / float(self.num_steps)
        t = 1.0
        dt = -1.0 / self.num_steps
        for _ in range(index):
            t = t + dt
        return t


# The Pi0.5 schedule every pre-existing artifact, HDF5 file and yaml implicitly
# used. Its timestep set is, by construction, the historical canonical set.
PI05_V1 = DenoiseSchedule(schedule_id="pi05_v1", num_steps=10, direction=DIRECTION_DESC)


def groot_n15_schedule(num_inference_timesteps: int) -> DenoiseSchedule:
    """The GR00T N1.5 schedule for a *live* step count.

    Callers must pass ``action_head.num_inference_timesteps`` read from the
    policy being served or collected from, never a constant: the count differs
    between deployments of the same checkpoint and is the whole reason the
    step count is part of the identity.
    """
    n = int(num_inference_timesteps)
    if isinstance(num_inference_timesteps, bool) or n < 2:
        raise ValueError(
            f"num_inference_timesteps must be an integer >= 2, got {num_inference_timesteps!r}"
        )
    return DenoiseSchedule(
        schedule_id=f"groot_n15_k{n}_v1", num_steps=n, direction=DIRECTION_ASC
    )


def schedule_from_id(schedule_id: str) -> DenoiseSchedule:
    """Resolve a recorded ``schedule_id`` back to its schedule; unknown ids raise."""
    if schedule_id == PI05_V1.schedule_id:
        return PI05_V1
    match = _GROOT_SCHEDULE_RE.match(str(schedule_id))
    if match is None:
        raise ValueError(
            f"unknown denoise schedule id {schedule_id!r}; expected "
            f"{PI05_V1.schedule_id!r} or 'groot_n15_k<N>_v1'"
        )
    return groot_n15_schedule(int(match.group("k")))


# Backward-compatible alias for the Pi0.5 timestep set. Every existing warm
# start yaml, judge and artifact was written against exactly this set; it is
# now the schedule's derived property rather than an independent formula.
CANONICAL_DENOISE_TIMESTEPS: frozenset[float] = PI05_V1.timestep_set


class CheckpointID(Enum):
    """The three cache checkpoints in the Pi0.5 inference pipeline.

    CP1 — after Stage 1 (vision + tokenisation).  Three outcomes:
          FULL_HIT skips Stage 2 + Stage 3.
          WARM_START runs Stage 2 then partial Stage 3 from a cached x_t.
          MISS runs full Stage 2 + Stage 3.
    CP2 — after Stage 2 (LLM backbone).  Post-backbone single-key cache
          (ActionCache-style baseline; key = projected backbone prefix output).
          FULL_HIT skips Stage 3, WARM_START runs partial Stage 3 from a
          cached x_t, MISS runs Stage 3.  Config validation makes CP2
          mutually exclusive with CP1 and CP3.
    CP3 — after Stage 3 (flow matching).  Schedules a cached action for the
          *next* inference cycle.
    """

    CP1 = auto()
    CP2 = auto()
    CP3 = auto()
