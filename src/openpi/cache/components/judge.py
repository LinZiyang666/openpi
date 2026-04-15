"""Judge: decide whether search results constitute a cache hit.

Data flow: SearchResultLite.score (from CacheStorage) -> judge() -> JudgeResult

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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Protocol, runtime_checkable

import torch

from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CANONICAL_DENOISE_TIMESTEPS, CheckpointID


class HitType(Enum):
    """Cache hit classification.

    Coupling:
      - CONSUMED BY: CacheOrchestrator (packs into CheckResult), Interceptor (controls stage skip)
    """

    MISS = auto()
    FULL_HIT = auto()
    WARM_START = auto()


@dataclass
class JudgeResult:
    """Structured return type for SimilarityJudge.

    FULL_HIT and WARM_START must include winner_id; Orchestrator skips
    fetch when winner_id is None.
    """

    hit_type: HitType
    winner_id: str | None = None
    start_t: float | None = None


@runtime_checkable
class SimilarityJudge(Protocol):
    """Judge whether a search result constitutes a cache hit.

    Data flow: SearchResultLite.score (from CacheStorage) -> judge() -> JudgeResult
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
    ) -> JudgeResult:
        """Judge the top search results.

        Args:
            results: Search results sorted by descending score (from CacheStorage).
            checkpoint_id: CP1 or CP3.
            cached_data: Raw tensors from KeyBuilder.cached_data.

        Returns:
            JudgeResult with hit_type, winner_id, and optional start_t.
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
    ) -> JudgeResult:
        if not results:
            return JudgeResult(HitType.MISS)
        return JudgeResult(HitType.FULL_HIT, results[0].id)

    def on_episode_start(self) -> None:
        """Clear internal history buffer. Called by Orchestrator at episode start."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action. Pure local buffer op."""


class AlwaysWarmStartJudge:
    """Always returns WARM_START with a fixed start_t for the top-1 result.

    Used to sweep the success_rate ~ start_t curve under a constant (forced)
    warm-start regime, independent of similarity score. Empty result set
    falls back to MISS (cache truly empty / first step of episode).

    Restricted to CP1 — CP3 has no warm start support. Config-level
    validation rejects CP3 usage; the runtime FULL_HIT fallback below is
    defensive only and should be unreachable for validated configs.
    """

    def __init__(self, start_t: float) -> None:
        st = round(start_t, 4)
        if st not in CANONICAL_DENOISE_TIMESTEPS:
            raise ValueError(
                f"start_t must round to one of {sorted(CANONICAL_DENOISE_TIMESTEPS)}, "
                f"got {start_t}"
            )
        self._start_t = st

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> JudgeResult:
        if not results:
            return JudgeResult(HitType.MISS)
        if checkpoint_id != CheckpointID.CP1:
            # Defensive fallback — config validation rejects always_warm_start on
            # non-CP1 checkpoints, so this branch should be unreachable in
            # validated configs. Keeps behaviour observable if someone bypasses
            # validation (e.g. direct instantiation in tests).
            return JudgeResult(HitType.FULL_HIT, results[0].id)
        return JudgeResult(HitType.WARM_START, results[0].id, start_t=self._start_t)

    def on_episode_start(self) -> None:
        """Clear internal history buffer. Called by Orchestrator at episode start."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action. Pure local buffer op."""


class ThresholdJudge:
    """Multi-tier threshold judge: FULL_HIT / WARM_START / MISS.

    Data flow: results[0].score -> compare threshold -> compare warm_tiers -> JudgeResult
    Coupling:
      - DEPENDS ON: score range from CacheStorage backend (see SimilarityJudge docstring)
      - IF backend or key builder changes: threshold value likely needs recalibration
    """

    def __init__(
        self,
        cp1_threshold: float = 0.98,
        cp3_threshold: float = 0.95,
        warm_tiers: list[dict[str, float]] | None = None,
    ) -> None:
        self._thresholds = {
            CheckpointID.CP1: cp1_threshold,
            CheckpointID.CP3: cp3_threshold,
        }
        self._warm_tiers = warm_tiers or []

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> JudgeResult:
        if not results:
            return JudgeResult(HitType.MISS)
        top = results[0]
        threshold = self._thresholds.get(checkpoint_id, 0.98)
        if top.score >= threshold:
            return JudgeResult(HitType.FULL_HIT, top.id)
        if checkpoint_id == CheckpointID.CP1 and self._warm_tiers:
            for tier in self._warm_tiers:
                if top.score >= tier["threshold"]:
                    return JudgeResult(HitType.WARM_START, top.id, start_t=tier["start_t"])
        return JudgeResult(HitType.MISS)

    def on_episode_start(self) -> None:
        """Clear internal history buffer. Called by Orchestrator at episode start."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action. Pure local buffer op."""
