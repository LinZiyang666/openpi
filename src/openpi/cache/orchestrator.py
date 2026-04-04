"""CacheOrchestrator: coordinate cache check and write operations.

Combines pluggable components (KeyBuilder, Gate, Judge, SearchStrategy)
with CacheStorage. All storage interaction goes through CacheStorage
facade -- never touches VectorStoreBackend directly.

Data flow overview:
  check():  Interceptor -> collect -> gate -> build -> SearchContext
            -> search_strategy.search(ctx) -> judge -> CheckResult
  write():  Interceptor -> collect -> build -> CacheEntry -> storage.insert

Coupling map:
  DEPENDS ON:  QueryKeyBuilder, GateFunction, SimilarityJudge, SearchStrategy (Step 4 components),
               CacheStorage facade (Step 3) -- insert/fetch_payload (search delegated to SearchStrategy),
               SystemTimer (Step 2) -- optional timing
  CONSUMED BY: InferenceInterceptor (calls check/write at cache checkpoint slots)
  DOES NOT depend on: VectorStoreBackend, Qdrant, or any specific backend
  SHARES: CacheStorage instance with SearchStrategy (Orchestrator: insert/fetch, Strategy: search)
  IF CHANGED:  Interceptor's cache integration logic may need updating
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

import torch

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import GateFunction
from openpi.cache.components.judge import HitType, SimilarityJudge
from openpi.cache.components.key_builder import QueryKeyBuilder
from openpi.cache.components.search_strategy import SearchContext, SearchStrategy
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a cache check at one checkpoint.

    Data flow: Orchestrator.check() -> CheckResult -> Interceptor (decision point)
    Coupling:
      - CONSUMED BY: InferenceInterceptor (reads hit_type to decide stage skip)
      - CONTAINS: CachePayload from CacheStorage.fetch_payload() (Step 3 type)
    """

    hit_type: HitType
    payload: Optional[CachePayload] = None  # non-None only on FULL_HIT
    score: Optional[float] = None
    entry_id: Optional[str] = None


def _stable_hash(checkpoint_id: CheckpointID, query_keys: dict[str, torch.Tensor]) -> str:
    """Compute a deterministic id from checkpoint + query key bytes.

    Same observation at the same checkpoint always maps to the same id,
    so repeated writes are idempotent upserts.
    """
    h = hashlib.sha256()
    h.update(checkpoint_id.name.encode())
    for name in sorted(query_keys.keys()):
        h.update(name.encode())
        h.update(query_keys[name].numpy().tobytes())
    return h.hexdigest()[:32]


class CacheOrchestrator:
    """Orchestrate cache check and write operations.

    Combines pluggable components (KeyBuilder, Gate, Judge, SearchStrategy)
    with CacheStorage. All storage interaction goes through CacheStorage
    facade -- never touches VectorStoreBackend directly.

    Data flow overview:
      check():  Interceptor -> collect -> gate -> build -> SearchContext
                -> search_strategy.search(ctx) -> judge -> CheckResult
      write():  Interceptor -> collect -> build -> CacheEntry -> storage.insert

    Coupling:
      - DEPENDS ON: QueryKeyBuilder, GateFunction, SimilarityJudge, SearchStrategy (Step 4 components)
      - DEPENDS ON: CacheStorage facade (Step 3) -- insert/fetch_payload (search delegated to SearchStrategy)
      - CONSUMED BY: InferenceInterceptor (calls check/write at checkpoint slots)
      - DOES NOT depend on: VectorStoreBackend, Qdrant, or any specific backend
      - SHARES: CacheStorage instance with SearchStrategy (Orchestrator: insert/fetch, Strategy: search)
      - IF CHANGED: Interceptor's cache integration logic may need updating

    Per-checkpoint dispatch:
      gates, judges, search_strategies are dict[CheckpointID, Component].
      CP1 and CP3 can use different Gate/Judge/SearchStrategy instances.
      key_builder is shared across checkpoints (same data extraction logic).

    Step counter:
      _step_counter tracks inference cycles within a task (client connection).
      Reset by on_task_begin(), incremented by check() on CP1.
      Passed to SearchStrategy via SearchContext for step_filter="exact"/"window" modes.
    """

    def __init__(
        self,
        storage: CacheStorage,
        key_builder: QueryKeyBuilder,
        gates: dict[CheckpointID, GateFunction],
        judges: dict[CheckpointID, SimilarityJudge],
        search_strategies: dict[CheckpointID, SearchStrategy],
        timer: Optional[SystemTimer] = None,
    ) -> None:
        self._storage = storage
        self._key_builder = key_builder
        self._gates = gates
        self._judges = judges
        self._search_strategies = search_strategies
        self._timer = timer if timer is not None else SystemTimer(enabled=False)
        self._step_counter: int = 0

        # Register fine-grained probes for each checkpoint's sub-steps.
        for cp in ("cp1", "cp3"):
            for step in ("collect", "gate", "build", "search", "judge", "fetch"):
                self._timer.register_probe(f"{cp}_{step}", backend="cpu")

    def on_task_begin(self) -> None:
        """Reset per-task state. Called when a client connection opens."""
        self._step_counter = 0

    def on_episode_start(self) -> None:
        """Reset per-episode state. Called when simulator sends episode_start."""
        self._step_counter = 0

    def check(self, checkpoint_id: CheckpointID, **stage_outputs) -> CheckResult:
        """Cache check pipeline: collect -> gate -> build -> ctx -> strategy.search() -> judge -> fetch.

        Flow:
          1. key_builder.collect(checkpoint_id, **stage_outputs)
          2. gates[checkpoint_id](checkpoint_id, cached_data) -> if False: MISS
          3. key_builder.build(checkpoint_id) -> query_keys
          4. Construct SearchContext(query_keys, checkpoint_id, current_step=_step_counter)
          5. search_strategies[checkpoint_id].search(ctx) -> results
          6. judges[checkpoint_id](results, checkpoint_id, cached_data) -> (hit_type, winner_id)
          7. if FULL_HIT: storage.fetch_payload(winner_id) -> payload
          8. _step_counter += 1 (only on CP1 check, to count inference cycles)
          9. return CheckResult
        """
        prefix = checkpoint_id.name.lower()

        # If this checkpoint is not configured (e.g. disabled in YAML), skip gracefully.
        if checkpoint_id not in self._gates:
            if checkpoint_id == CheckpointID.CP1:
                self._step_counter += 1
            return CheckResult(hit_type=HitType.MISS)

        gate = self._gates[checkpoint_id]
        judge = self._judges[checkpoint_id]
        strategy = self._search_strategies[checkpoint_id]

        with self._timer.measure(f"{prefix}_collect"):
            self._key_builder.collect(checkpoint_id, **stage_outputs)

        with self._timer.measure(f"{prefix}_gate"):
            should_search = gate(checkpoint_id, self._key_builder.cached_data)
        logger.debug("[step %d] %s gate: %s", self._step_counter, prefix, "SEARCH" if should_search else "SKIP")
        if not should_search:
            if checkpoint_id == CheckpointID.CP1:
                self._step_counter += 1
            return CheckResult(hit_type=HitType.MISS)

        with self._timer.measure(f"{prefix}_build"):
            query_keys = self._key_builder.build(checkpoint_id)

        with self._timer.measure(f"{prefix}_search"):
            ctx = SearchContext(
                query_keys=query_keys,
                checkpoint_id=checkpoint_id,
                current_step=self._step_counter,
            )
            results = strategy.search(ctx)

        with self._timer.measure(f"{prefix}_judge"):
            hit_type, winner_id = judge(
                results, checkpoint_id, self._key_builder.cached_data
            )
        top_score = results[0].score if results else None
        logger.debug("[step %d] %s judge: %s (top_score=%s, winner=%s)", self._step_counter, prefix, hit_type.name, top_score, winner_id)

        if checkpoint_id == CheckpointID.CP1:
            self._step_counter += 1

        if hit_type == HitType.FULL_HIT and winner_id is not None:
            with self._timer.measure(f"{prefix}_fetch"):
                payload = self._storage.fetch_payload(winner_id)
            return CheckResult(
                hit_type=hit_type,
                payload=payload,
                score=results[0].score,
                entry_id=winner_id,
            )

        return CheckResult(hit_type=HitType.MISS)

    def build_keys(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        """Build query keys from already-collected stage outputs.

        Must be called in the main thread (accesses GPU tensors via key_builder).
        Returns CPU tensors safe for cross-thread use.
        """
        return self._key_builder.build(checkpoint_id)

    def write(
        self,
        checkpoint_id: CheckpointID,
        payload: CachePayload,
        **stage_outputs,
    ) -> None:
        """Write a cache entry (synchronous).

        Flow:
          1. key_builder.collect(...) [if not already collected this cycle]
          2. key_builder.build(...) -> query_keys
          3. Construct CacheEntry with stable_hash id
          4. storage.insert(entry)

        Caller must ensure payload tensors are CPU float32.
        """
        self._key_builder.collect(checkpoint_id, **stage_outputs)
        query_keys = self._key_builder.build(checkpoint_id)

        entry_id = _stable_hash(checkpoint_id, query_keys)
        entry = CacheEntry(
            id=entry_id,
            checkpoint_id=checkpoint_id,
            query_keys=query_keys,
            payload=payload,
        )
        self._storage.insert(entry)

    def write_with_keys(
        self,
        checkpoint_id: CheckpointID,
        payload: CachePayload,
        query_keys: dict[str, torch.Tensor],
    ) -> None:
        """Write a cache entry with pre-built query keys. Thread-safe.

        All tensors must already be on CPU. Used by background write threads.
        """
        entry_id = _stable_hash(checkpoint_id, query_keys)
        entry = CacheEntry(
            id=entry_id,
            checkpoint_id=checkpoint_id,
            query_keys=query_keys,
            payload=payload,
        )
        self._storage.insert(entry)

    # ------------------------------------------------------------------
    # CP3 stub interfaces -- real implementation in Step 6
    # ------------------------------------------------------------------

    def schedule_next_action(self, action: torch.Tensor) -> None:
        """CP3: schedule a cached action for the next inference cycle.
        Stub in Step 4 -- does nothing. Step 6 implements with DeferredWriter.
        """
        pass

    def should_skip_inference(self) -> Optional[torch.Tensor]:
        """CP3: check if previous cycle scheduled an action for this cycle.
        Stub in Step 4 -- always returns None (no skip). Step 6 implements.
        """
        return None

    def clear(self) -> None:
        """Release per-cycle state. Called at end of each inference cycle."""
        self._key_builder.clear()
