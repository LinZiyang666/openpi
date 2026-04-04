"""CacheOrchestrator: coordinate cache check and write operations.

Combines pluggable components (KeyBuilder, Gate, Judge) with CacheStorage.
All storage interaction goes through CacheStorage facade — never touches
VectorStoreBackend directly.

Data flow overview:
  check():  Interceptor -> collect -> gate -> build -> storage.search -> judge -> CheckResult
  write():  Interceptor -> collect -> build -> CacheEntry -> storage.insert

Coupling map:
  DEPENDS ON:  QueryKeyBuilder, GateFunction, SimilarityJudge (Step 4 components),
               CacheStorage facade (Step 3) — search/insert/fetch_payload,
               SystemTimer (Step 2) — optional timing
  CONSUMED BY: InferenceInterceptor (calls check/write at cache checkpoint slots)
  DOES NOT depend on: VectorStoreBackend, Qdrant, or any specific backend
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
from openpi.cache.storage_types import CacheEntry, CachePayload, QuerySpec
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

    Combines pluggable components (KeyBuilder, Gate, Judge) with CacheStorage.
    All storage interaction goes through CacheStorage facade — never touches
    VectorStoreBackend directly.

    Data flow overview:
      check():  Interceptor -> collect -> gate -> build -> storage.search -> judge -> CheckResult
      write():  Interceptor -> collect -> build -> CacheEntry -> storage.insert

    Coupling:
      - DEPENDS ON: QueryKeyBuilder, GateFunction, SimilarityJudge (Step 4 components)
      - DEPENDS ON: CacheStorage facade (Step 3) — search/insert/fetch_payload
      - CONSUMED BY: InferenceInterceptor (calls check/write at TODO slots)
      - DOES NOT depend on: VectorStoreBackend, Qdrant, or any specific backend
      - IF CHANGED: Interceptor's cache integration logic may need updating
    """

    def __init__(
        self,
        storage: CacheStorage,
        key_builder: QueryKeyBuilder,
        gate: GateFunction,
        judge: SimilarityJudge,
        timer: Optional[SystemTimer] = None,
    ) -> None:
        self._storage = storage
        self._key_builder = key_builder
        self._gate = gate
        self._judge = judge
        self._timer = timer if timer is not None else SystemTimer(enabled=False)

        # Register fine-grained probes for each checkpoint's sub-steps.
        # NOTE: Step 2 design envisions CUDA Event timing for KeyBuilder once
        # cache_stream is introduced (Step 8). Current CPU backend measures
        # wall-clock time, which equals GPU time when operations run on the
        # default stream with implicit sync (.cpu() calls).
        for cp in ("cp1", "cp3"):
            for step in ("collect", "gate", "build", "search", "judge", "fetch"):
                self._timer.register_probe(f"{cp}_{step}", backend="cpu")

    def check(self, checkpoint_id: CheckpointID, **stage_outputs) -> CheckResult:
        """Cache check pipeline: collect -> gate -> build -> search -> judge -> fetch.

        Flow:
          1. key_builder.collect(checkpoint_id, **stage_outputs)
          2. gate(checkpoint_id, key_builder.cached_data) -> if False: MISS
          3. key_builder.build(checkpoint_id) -> query_keys
          4. storage.search(QuerySpec(query_keys=query_keys, top_k=1, checkpoint_id=checkpoint_id))
          5. judge(results, checkpoint_id, key_builder.cached_data) -> (hit_type, winner_id)
          6. if FULL_HIT: storage.fetch_payload(winner_id) -> payload
          7. return CheckResult

        Note: collect() before gate() so Gate can access cached_data.
        fetch_payload called by Orchestrator, not Judge (Judge is pure judgment).
        """
        prefix = checkpoint_id.name.lower()

        with self._timer.measure(f"{prefix}_collect"):
            self._key_builder.collect(checkpoint_id, **stage_outputs)

        with self._timer.measure(f"{prefix}_gate"):
            should_search = self._gate(checkpoint_id, self._key_builder.cached_data)
        if not should_search:
            return CheckResult(hit_type=HitType.MISS)

        with self._timer.measure(f"{prefix}_build"):
            query_keys = self._key_builder.build(checkpoint_id)

        with self._timer.measure(f"{prefix}_search"):
            spec = QuerySpec(
                query_keys=query_keys,
                top_k=1,
                checkpoint_id=checkpoint_id,
            )
            results = self._storage.search(spec)

        with self._timer.measure(f"{prefix}_judge"):
            hit_type, winner_id = self._judge(
                results, checkpoint_id, self._key_builder.cached_data
            )

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

    def write(
        self,
        checkpoint_id: CheckpointID,
        payload: CachePayload,
        **stage_outputs,
    ) -> None:
        """Write a cache entry (synchronous). Async deferred to Step 8.

        Flow:
          1. key_builder.collect(...) [if not already collected this cycle]
          2. key_builder.build(...) -> query_keys
          3. Construct CacheEntry with stable_hash id
          4. storage.insert(entry)

        Caller must ensure payload tensors are CPU float32.
        """
        # Collect may already have been called in check() this cycle;
        # calling again is idempotent (clears + re-collects same data).
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

    # ------------------------------------------------------------------
    # CP3 stub interfaces — real implementation in Step 6
    # ------------------------------------------------------------------

    def schedule_next_action(self, action: torch.Tensor) -> None:
        """CP3: schedule a cached action for the next inference cycle.
        Stub in Step 4 — does nothing. Step 6 implements with DeferredWriter.
        """
        pass

    def should_skip_inference(self) -> Optional[torch.Tensor]:
        """CP3: check if previous cycle scheduled an action for this cycle.
        Stub in Step 4 — always returns None (no skip). Step 6 implements.
        """
        return None

    def clear(self) -> None:
        """Release per-cycle state. Called at end of each inference cycle."""
        self._key_builder.clear()
