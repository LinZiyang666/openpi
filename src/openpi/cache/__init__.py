# Cache system for the Pi0/Pi0.5 inference pipeline.
# See docs/architecture/cache_system.md for the full design and roadmap.
#
# Public API:
#
#   Timing
#   ------
#   SystemTimer    — pluggable, hardware-aware timing for all pipeline components.
#   TaskLifecycle  — protocol for on_task_begin / on_task_end hooks.
#
#   Storage layer (Step 3)
#   ----------------------
#   CheckpointID        — CP1 / CP2 / CP3 enum.
#   CachePayload        — data stored with each cache entry.
#   CacheEntry          — full unit written to the vector store.
#   QueryFilter         — per-query dynamic constraints (task_key, step_range).
#   QuerySpec           — vector + top_k + optional filter.
#   SearchResultLite    — lightweight search result (no payload tensors).
#   SearchResult        — full search result including payload.
#   BatchInsertResult   — outcome of a batch_insert call.
#   UnsupportedFilterError — raised when QueryFilter contains unsupported fields.
#   VectorStoreBackend  — ABC for pluggable vector store backends.
#   CacheStorage        — facade: thread-safe, validates inputs, two-phase search.

try:
    from openpi.cache.timing import SystemTimer, TaskLifecycle
except ImportError:
    pass  # torch not available; timing unused in non-GPU environments (e.g. exp/)
from openpi.cache.types import (
    CheckpointID,
    CACHE_QUERY_FIELDS,
    VISION_0,
    VISION_1,
    VISION_2,
    PROMPT_EMB,
    ROBOT_STATE,
)
from openpi.cache.storage_types import (
    CachePayload,
    CacheEntry,
    QueryFilter,
    QuerySpec,
    SearchResultLite,
    SearchResult,
    BatchInsertResult,
    UnsupportedFilterError,
)
from openpi.cache.backend_base import VectorStoreBackend
from openpi.cache.cache_storage import CacheStorage

# Step 4: Orchestrator + pluggable components
from openpi.cache.components.key_builder import QueryKeyBuilder, PlaceholderKeyBuilder, CP1TemporalPruneKeyBuilder
from openpi.cache.components.token_reducer import (
    PruneResult, TokenReducer, MeanPoolReducer, MaxPoolReducer,
    SpatialPoolReducer, TaskScoringReducer,
)
from openpi.cache.components.gate import GateFunction, AlwaysSearchGate
from openpi.cache.components.judge import (
    AlwaysHitJudge,
    AlwaysWarmStartJudge,
    HitType,
    JudgeResult,
    SimilarityJudge,
    ThresholdJudge,
)
from openpi.cache.orchestrator import CacheOrchestrator, CheckResult

__all__ = [
    # timing (existing)
    "SystemTimer",
    "TaskLifecycle",
    # storage layer
    "CheckpointID",
    "CACHE_QUERY_FIELDS",
    "VISION_0",
    "VISION_1",
    "VISION_2",
    "PROMPT_EMB",
    "ROBOT_STATE",
    "CachePayload",
    "CacheEntry",
    "QueryFilter",
    "QuerySpec",
    "SearchResultLite",
    "SearchResult",
    "BatchInsertResult",
    "UnsupportedFilterError",
    "VectorStoreBackend",
    "CacheStorage",
    # Step 4: orchestrator + components
    "QueryKeyBuilder",
    "PlaceholderKeyBuilder",
    "CP1TemporalPruneKeyBuilder",
    "PruneResult",
    "TokenReducer",
    "MeanPoolReducer",
    "MaxPoolReducer",
    "SpatialPoolReducer",
    "TaskScoringReducer",
    "GateFunction",
    "AlwaysSearchGate",
    "SimilarityJudge",
    "ThresholdJudge",
    "AlwaysHitJudge",
    "AlwaysWarmStartJudge",
    "HitType",
    "JudgeResult",
    "CacheOrchestrator",
    "CheckResult",
]
