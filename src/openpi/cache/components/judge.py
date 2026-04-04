"""Judge: decide whether search results constitute a cache hit.

Data flow: SearchResultLite.score (from CacheStorage) -> judge() -> (HitType, winner_id)

Coupling map:
  DEPENDS ON:  SearchResultLite.score semantics (Step 3 backend-dependent)
               * Single-field cosine: score in [-1, 1]
               * Multi-field RRF: small positive numbers, scale depends on RRF k param
               * IF backend changes: thresholds MUST be recalibrated
  MAY DEPEND ON: KeyBuilder.cached_data (for future re-scoring judges)
  CONSUMED BY: CacheOrchestrator.check()
  DOES NOT call: CacheStorage or fetch_payload (pure judgment, no side effects)
  IF CHANGED:  Only affects hit/miss decision, no downstream structural impact
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Protocol, runtime_checkable

import torch

from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CheckpointID


class HitType(Enum):
    """Cache hit classification.

    Coupling:
      - CONSUMED BY: CacheOrchestrator (packs into CheckResult), Interceptor (controls stage skip)
    """

    MISS = auto()
    FULL_HIT = auto()
    # WARM_START = auto()  # Step 7: flow matching warm start


@runtime_checkable
class SimilarityJudge(Protocol):
    """Judge whether a search result constitutes a cache hit.

    Data flow: SearchResultLite.score (from CacheStorage) -> judge() -> (HitType, winner_id)
    Coupling:
      - DEPENDS ON: SearchResultLite.score semantics (Step 3 backend-dependent)
        * Single-field cosine: score in [-1, 1]
        * Multi-field RRF: small positive numbers, scale depends on RRF k param
        * IF backend changes: thresholds MUST be recalibrated
      - MAY DEPEND ON: KeyBuilder.cached_data (for future re-scoring judges)
      - CONSUMED BY: CacheOrchestrator.check()
      - DOES NOT call: CacheStorage or fetch_payload (pure judgment, no side effects)
      - IF CHANGED: Only affects hit/miss decision, no downstream structural impact
    """

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> tuple[HitType, Optional[str]]:
        """Judge the top search results.

        Args:
            results: Search results sorted by descending score (from CacheStorage).
            checkpoint_id: CP1 or CP3.
            cached_data: Raw tensors from KeyBuilder.cached_data.

        Returns:
            (hit_type, winner_id): HitType and the id of the winning entry (None if MISS).
        """
        ...


class AlwaysHitJudge:
    """Always returns FULL_HIT for the top-1 result (if any results exist).

    Useful for testing / calibration: confirms the full hit path works
    end-to-end without threshold tuning.
    """

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> tuple[HitType, Optional[str]]:
        if not results:
            return HitType.MISS, None
        return HitType.FULL_HIT, results[0].id


class ThresholdJudge:
    """Simple threshold-based judge: top-1 score > threshold -> FULL_HIT.

    Data flow: results[0].score -> compare threshold -> HitType
    Coupling:
      - DEPENDS ON: score range from CacheStorage backend (see SimilarityJudge docstring)
      - IF backend or key builder changes: threshold value likely needs recalibration
    """

    def __init__(
        self,
        cp1_threshold: float = 0.98,
        cp3_threshold: float = 0.95,
    ) -> None:
        self._thresholds = {
            CheckpointID.CP1: cp1_threshold,
            CheckpointID.CP3: cp3_threshold,
        }

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> tuple[HitType, Optional[str]]:
        if not results:
            return HitType.MISS, None
        top = results[0]
        threshold = self._thresholds.get(checkpoint_id, 0.98)
        if top.score >= threshold:
            return HitType.FULL_HIT, top.id
        return HitType.MISS, None
