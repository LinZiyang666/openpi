"""Base enumerations shared across the cache subsystem.

Kept in a separate module so that both timing.py and storage_types.py can
import from here without creating circular dependencies.

Coupling map:
  DEPENDS ON:  nothing (leaf module)
  CONSUMED BY: KeyBuilder (field name constants), Orchestrator (CheckpointID),
               storage_types (CheckpointID), cache_storage (via storage_types)
  IF CHANGED:  All consumers must update field references
"""

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

CACHE_QUERY_FIELDS: frozenset[str] = frozenset({
    VISION_0, VISION_1, VISION_2, PROMPT_EMB, ROBOT_STATE,
})


# ---------------------------------------------------------------------------
# Canonical flow-matching denoise timesteps
# ---------------------------------------------------------------------------
#
# Pi0.5 flow matching uses num_steps=10 with timesteps t_i = 1 - i/10 for
# i = 1..9. Warm start payloads are keyed by these exact (rounded-to-4-dp)
# values, and ThresholdJudge.warm_tiers / AlwaysWarmStartJudge both require
# start_t to be one of them. Kept here (leaf module) so judge.py, config.py,
# and any future consumer share one source of truth.
CANONICAL_DENOISE_TIMESTEPS: frozenset[float] = frozenset(
    round(1.0 - i / 10, 4) for i in range(1, 10)
)  # {0.1, 0.2, ..., 0.9}


class CheckpointID(Enum):
    """The three cache checkpoints in the Pi0.5 inference pipeline.

    CP1 — after Stage 1 (vision + tokenisation).  Three outcomes:
          FULL_HIT skips Stage 2 + Stage 3.
          WARM_START runs Stage 2 then partial Stage 3 from a cached x_t.
          MISS runs full Stage 2 + Stage 3.
    CP2 — after Stage 2 (LLM backbone).  Reserved; warm start migrated to CP1.
    CP3 — after Stage 3 (flow matching).  Schedules a cached action for the
          *next* inference cycle.
    """

    CP1 = auto()
    CP2 = auto()
    CP3 = auto()
