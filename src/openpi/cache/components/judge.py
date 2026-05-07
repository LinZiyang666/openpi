"""Judge: decide whether search results constitute a cache hit.

Data flow: SearchResultLite.score (from CacheStorage) -> judge() -> JudgeResult

Coupling map:
  DEPENDS ON:  SearchResultLite.score semantics (Step 3 backend-dependent)
               * Single-field cosine: score in [-1, 1]
               * Multi-field RRF: small positive numbers, scale depends on RRF k param
               * IF backend changes: thresholds MUST be recalibrated
  MAY DEPEND ON: KeyBuilder.cached_data (for future re-scoring judges)
  CONSUMED BY: CacheOrchestrator.check()
  Purity contract:
    A judge MUST NOT write to CacheStorage. Read-only access via the
    optional `view` (PayloadView) parameter is permitted at verdict time
    so composite judges can compute factor descriptors over candidate
    payloads + neighbor entries. See `cache_system.md` §5.6 / §5.11 for
    the contract refinement.
  IF CHANGED:  Only affects hit/miss decision, no downstream structural impact
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import torch

from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CANONICAL_DENOISE_TIMESTEPS, CheckpointID

# Verdict-pipeline debug instrumentation. Gated on env var so production
# runs pay zero cost. Toggle from server shell:
#   OPENPI_CACHE_VERDICT_DEBUG=1 uv run scripts/serve_policy.py ... |& tee /tmp/vd.log
# then `grep '\[vd ' /tmp/vd.log` inspects per-verdict structured lines.
_VERDICT_DEBUG = os.environ.get("OPENPI_CACHE_VERDICT_DEBUG") == "1"
_verdict_logger = logging.getLogger("openpi.cache.verdict_debug")

if TYPE_CHECKING:
    from openpi.cache.components.factors.base import HistoryView
    from openpi.cache.components.payload_view import PayloadView


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

    ``factor_outputs`` is an optional diagnostic payload populated only by
    CompositeJudge when its config carries ``export_factor_outputs: true``.
    Schema: ``{"raw": dict[str, float|None], "norm": dict[str, float|None],
    "score": float|None, "sentinel": str|None}``. NaN values are pre-converted
    to ``None`` so the dict round-trips through strict JSON parsers (jq,
    pandas) without relying on Python's lax ``allow_nan=True``. Orchestrator
    forwards this on ``CheckResult``; Interceptor surfaces it through
    ``__hit_meta__`` for client-side per-step logging.
    """

    hit_type: HitType
    winner_id: str | None = None
    start_t: float | None = None
    composer_score: Optional[float] = None
    factor_outputs: Optional[dict] = None


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
      - Purity contract: read-only via `view`; never writes storage.
      - IF CHANGED: Only affects hit/miss decision, no downstream structural impact

    The `view` and `history` keyword-only parameters carry verdict-time
    facade objects injected by Orchestrator. Existing judges that do not
    need them accept and ignore both via `**kwargs`; Orchestrator only
    builds and injects the facades from B1 onward (CompositeJudge land).
    """

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        *,
        view: Optional["PayloadView"] = None,
        history: Optional["HistoryView"] = None,
    ) -> JudgeResult:
        """Judge the top search results.

        Args:
            results: Search results sorted by descending score (from CacheStorage).
            checkpoint_id: CP1 or CP3.
            cached_data: Raw tensors from KeyBuilder.cached_data.
            view: Read-only facade over CacheStorage (B1+ injection;
                None for the B0 path and for legacy judges).
            history: Per-episode action / state snapshot (B1+ injection).

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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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


# ---------------------------------------------------------------------------
# Diagnostic factor-output helpers (used by CompositeJudge when
# export_factor_outputs=True). Pre-converts NaN to None so the dict
# round-trips through strict JSON parsers (jq, pandas) without relying on
# the producer's `allow_nan=True`.
# ---------------------------------------------------------------------------


def _nan_to_none(d: dict[str, float]) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        f = float(v)
        out[k] = None if math.isnan(f) else f
    return out


def _build_factor_outputs(
    raw: dict[str, float],
    norm: dict[str, float],
    *,
    composer_score: Optional[float],
    sentinel: Optional[str],
) -> dict:
    score: Optional[float]
    if composer_score is None:
        score = None
    else:
        s = float(composer_score)
        score = None if math.isnan(s) else s
    return {
        "raw":  _nan_to_none(raw),
        "norm": _nan_to_none(norm),
        "score": score,
        "sentinel": sentinel,
    }


# ---------------------------------------------------------------------------
# B5/B7 refactor — CompositeJudge + DumpingJudge moved out of this file.
# Facade re-exports preserve legacy import paths
#   from openpi.cache.components.judge import CompositeJudge, DumpingJudge
# while the implementations live in dedicated modules.
# ---------------------------------------------------------------------------

from openpi.cache.components.composite_judge import CompositeJudge  # noqa: E402, F401
from openpi.cache.components.dumping_judge import DumpingJudge      # noqa: E402, F401
