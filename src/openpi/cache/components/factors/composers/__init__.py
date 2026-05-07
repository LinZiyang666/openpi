"""Composer Protocol + concrete S1 / S2 / S3 classes.

A Composer aggregates a normalized factor dict into a `JudgeResult`.
Algorithms (orientation flip + tier mapping + NaN handling) live here;
CompositeJudge owns the pipeline (extractor outputs -> normalizer ->
composer.compose).

Coupling map:
  DEPENDS ON:  components/judge.py (JudgeResult, HitType)
  CONSUMED BY: CompositeJudge (composer dependency injection)
  IF CHANGED:  CompositeJudge composition pipeline
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, Protocol, runtime_checkable

from openpi.cache.components.judge import HitType, JudgeResult

# Mirror of judge.py's debug gate so composers can emit a per-compose
# diagnostic line under the same env var. Pays zero cost when not set.
_VERDICT_DEBUG = os.environ.get("OPENPI_CACHE_VERDICT_DEBUG") == "1"
_verdict_logger = logging.getLogger("openpi.cache.verdict_debug")


# Orientation kinds — kept in sync with `_DESCRIPTOR_ORIENTATIONS` in
# `factors/source_window.py` (the single source of truth).
_SAFE = "safe"
_RISKY = "risky"
_NON_MONOTONIC = "non_monotonic"


@runtime_checkable
class Composer(Protocol):
    """Aggregate a normalized factor dict into a JudgeResult.

    `bind_orientations` is called once by `CompositeJudge.__init__` with
    the union of every extractor's `descriptor_orientations`. The composer
    uses these to decide flip direction (safe / risky / non_monotonic)
    when scoring.
    """

    def bind_orientations(self, orientations: dict[str, str]) -> None: ...

    def compose(
        self,
        factors: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        """Return JudgeResult with hit_type in {FULL_HIT, WARM_START, MISS}.

        `winner_id` is the id the composer attaches when emitting
        FULL_HIT or WARM_START — the composer does not pick which
        candidate becomes the winner; it only decides hit-type for the
        already-selected one.
        """
        ...


# ------------------------------------------------------------------
# S1: Weighted percentile sum + tier thresholds
# ------------------------------------------------------------------


class WeightedSumComposer:
    """S1: orientation-aware weighted percentile sum + tier thresholds.

    B0 stub: stores constructor params + records the orientations from
    `bind_orientations`. The actual score aggregation, NaN handling,
    orientation flipping, and tier mapping land in B1+ when CompositeJudge
    is enabled.
    """

    def __init__(
        self,
        weights: dict[str, float],
        full_hit_threshold: float,
        warm_start_threshold: Optional[float] = None,
        warm_start_t: Optional[float] = None,
        directions: Optional[dict[str, str]] = None,
    ) -> None:
        # `directions` is required (validated at bind_orientations time)
        # for any key whose orientation is "non_monotonic" and whose
        # weight is non-zero; format is "high" | "low" | "range:[lo,hi]".
        self._weights = dict(weights)
        self._full_hit_threshold = full_hit_threshold
        self._warm_start_threshold = warm_start_threshold
        self._warm_start_t = warm_start_t
        self._directions = dict(directions) if directions else {}
        self._orientations: dict[str, str] = {}
        # Layer 4 contract (B4 refactor): declare required factor keys
        # so the static yaml validator can fail-fast on missing factors.
        # A zero-weight key has no influence on the score, so it is not
        # a true dependency.
        self.declared_dependencies: set[str] = {
            k for k, w in self._weights.items() if w != 0.0
        }

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        # Cross-check `directions` coverage for non_monotonic keys whose
        # weight is non-zero — fail loud at bind time so invalid YAML
        # never silently aggregates wrong sign.
        self._orientations = dict(orientations)
        missing: list[str] = []
        for k, ori in self._orientations.items():
            if ori != _NON_MONOTONIC:
                continue
            if self._weights.get(k, 0.0) == 0.0:
                continue
            if k not in self._directions:
                missing.append(k)
        if missing:
            raise ValueError(
                f"WeightedSumComposer: non_monotonic keys with non-zero weight "
                f"are missing `directions`: {sorted(missing)}"
            )

    def compose(
        self,
        factors: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        # NaN-tolerant weighted average. Normalizer is expected to map
        # raw factors into [0, 1]; safe orientation contributes the
        # value directly, risky orientation flips (1 - value), and
        # non_monotonic uses the configured direction rule.
        total_w = 0.0
        score = 0.0
        for k, v in factors.items():
            if math.isnan(v):
                continue                                # missing-signal NaN
            w = self._weights.get(k, 0.0)
            if w == 0.0:
                continue
            ori = self._orientations.get(k)
            if ori == _SAFE:
                contrib = v
            elif ori == _RISKY:
                contrib = 1.0 - v
            elif ori == _NON_MONOTONIC:
                contrib = _apply_direction(self._directions[k], v)
            else:
                raise ValueError(
                    f"WeightedSumComposer: unknown orientation {ori!r} for key {k!r}"
                )
            score += w * contrib
            total_w += w
        if total_w == 0.0:
            if _VERDICT_DEBUG:
                _verdict_logger.info(
                    "[vd composer] total_w=0 (every key NaN-skipped) -> MISS"
                )
            return JudgeResult(HitType.MISS, composer_score=None)
        s = score / total_w
        if s >= self._full_hit_threshold:
            if self._warm_start_t is not None:
                # Caller wired this composer for warm-start emission rather
                # than full-hit — treat the threshold pass as WARM_START.
                if _VERDICT_DEBUG:
                    _verdict_logger.info(
                        "[vd composer] score=%.4f >= full_hit=%.2f (warm_start_t set -> WARM_START)",
                        s, self._full_hit_threshold,
                    )
                return JudgeResult(
                    HitType.WARM_START, winner_id=winner_id, start_t=self._warm_start_t,
                    composer_score=s,
                )
            if _VERDICT_DEBUG:
                _verdict_logger.info(
                    "[vd composer] score=%.4f >= full_hit=%.2f -> FULL_HIT",
                    s, self._full_hit_threshold,
                )
            return JudgeResult(
                HitType.FULL_HIT, winner_id=winner_id, composer_score=s,
            )
        if (
            self._warm_start_threshold is not None
            and s >= self._warm_start_threshold
        ):
            if _VERDICT_DEBUG:
                _verdict_logger.info(
                    "[vd composer] score=%.4f >= warm_start=%.2f (< full_hit=%.2f) -> WARM_START",
                    s, self._warm_start_threshold, self._full_hit_threshold,
                )
            return JudgeResult(
                HitType.WARM_START, winner_id=winner_id, start_t=self._warm_start_t,
                composer_score=s,
            )
        if _VERDICT_DEBUG:
            _verdict_logger.info(
                "[vd composer] score=%.4f < %.2f%s -> MISS",
                s, self._full_hit_threshold,
                f" (warm_start={self._warm_start_threshold:.2f})" if self._warm_start_threshold is not None else "",
            )
        return JudgeResult(HitType.MISS, composer_score=s)


# ------------------------------------------------------------------
# S1b: WeightedSum + warm-start fallback when every weighted key is NaN.
# ------------------------------------------------------------------
# Plan §6.5 #2: the framework does not short-circuit on the all-NaN case.
# This subclass implements WARM_START on all-NaN at the *composer* level:
# when every non-zero-weight key is NaN (factor physical edge case), emit
# WARM_START with the configured ``warm_fallback_start_t`` instead of MISS.


class WeightedSumWithWarmFallbackComposer(WeightedSumComposer):
    """WeightedSum + WARM_START fallback on all-NaN weighted dependencies.

    Behaves identically to ``WeightedSumComposer`` except: when every
    declared (non-zero-weight) key in ``calibrated`` is NaN, return
    ``JudgeResult(WARM_START, winner_id, start_t=warm_fallback_start_t)``
    instead of MISS. Useful for factor recipes whose dominant signal can
    NaN out at episode boundaries (e.g. long F1b windows on chains
    shorter than 2P+2F+1).

    Construction params extend the base class with one extra:
        warm_fallback_start_t: float — the ``start_t`` to attach when
        the NaN-fallback path triggers.
    """

    def __init__(
        self,
        weights: dict[str, float],
        full_hit_threshold: float,
        warm_fallback_start_t: float,
        warm_start_threshold: Optional[float] = None,
        warm_start_t: Optional[float] = None,
        directions: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(
            weights=weights,
            full_hit_threshold=full_hit_threshold,
            warm_start_threshold=warm_start_threshold,
            warm_start_t=warm_start_t,
            directions=directions,
        )
        self._warm_fallback_start_t = float(warm_fallback_start_t)

    def compose(
        self,
        factors: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        # All-NaN check covers only the keys that actually weigh in (zero
        # weights have no influence, so their NaN-ness is irrelevant).
        weighted_keys = [k for k, w in self._weights.items() if w != 0.0]
        if weighted_keys and all(
            math.isnan(factors.get(k, float("nan"))) for k in weighted_keys
        ):
            if _VERDICT_DEBUG:
                _verdict_logger.info(
                    "[vd composer] all weighted keys NaN -> WARM_START fallback @ t=%.2f",
                    self._warm_fallback_start_t,
                )
            return JudgeResult(
                HitType.WARM_START,
                winner_id=winner_id,
                start_t=self._warm_fallback_start_t,
                composer_score=None,
            )
        # Otherwise delegate to the base class scoring path.
        return super().compose(factors, winner_id=winner_id)


# ------------------------------------------------------------------
# S2: Conjunctive per-factor thresholds
# ------------------------------------------------------------------


class AndGateComposer:
    """S2: every key in `per_factor_thresholds` must pass its threshold
    in the orientation-aware direction (safe: v >= thr, risky: v <= thr,
    non_monotonic: per-direction). NaN counts as not-passed.
    """

    def __init__(
        self,
        per_factor_thresholds: dict[str, float],
        warm_start_t: Optional[float] = None,
        directions: Optional[dict[str, str]] = None,
    ) -> None:
        self._thresholds = dict(per_factor_thresholds)
        self._warm_start_t = warm_start_t
        self._directions = dict(directions) if directions else {}
        self._orientations: dict[str, str] = {}
        # Layer 4 contract (B4 refactor): every threshold key is required.
        self.declared_dependencies: set[str] = set(self._thresholds.keys())

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        self._orientations = dict(orientations)
        missing = [
            k for k in self._thresholds
            if self._orientations.get(k) == _NON_MONOTONIC and k not in self._directions
        ]
        if missing:
            raise ValueError(
                f"AndGateComposer: non_monotonic keys are missing `directions`: "
                f"{sorted(missing)}"
            )

    def compose(
        self,
        factors: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        for k, thr in self._thresholds.items():
            v = factors.get(k, float("nan"))
            if math.isnan(v):
                return JudgeResult(HitType.MISS)        # NaN = not passed
            if not _passes_threshold(self._orientations.get(k), v, thr, self._directions.get(k)):
                return JudgeResult(HitType.MISS)
        if self._warm_start_t is not None:
            return JudgeResult(
                HitType.WARM_START, winner_id=winner_id, start_t=self._warm_start_t,
            )
        return JudgeResult(HitType.FULL_HIT, winner_id=winner_id)


# ------------------------------------------------------------------
# S3: Disjunctive per-factor thresholds
# ------------------------------------------------------------------


class OrGateComposer:
    """S3: any key passing its threshold (orientation-aware) emits hit.
    NaN keys are skipped (let other keys decide).
    """

    def __init__(
        self,
        per_factor_thresholds: dict[str, float],
        warm_start_t: Optional[float] = None,
        directions: Optional[dict[str, str]] = None,
    ) -> None:
        self._thresholds = dict(per_factor_thresholds)
        self._warm_start_t = warm_start_t
        self._directions = dict(directions) if directions else {}
        self._orientations: dict[str, str] = {}
        # Layer 4 contract (B4 refactor): every threshold key is required.
        self.declared_dependencies: set[str] = set(self._thresholds.keys())

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        self._orientations = dict(orientations)
        missing = [
            k for k in self._thresholds
            if self._orientations.get(k) == _NON_MONOTONIC and k not in self._directions
        ]
        if missing:
            raise ValueError(
                f"OrGateComposer: non_monotonic keys are missing `directions`: "
                f"{sorted(missing)}"
            )

    def compose(
        self,
        factors: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        for k, thr in self._thresholds.items():
            v = factors.get(k, float("nan"))
            if math.isnan(v):
                continue                                # NaN skipped per §2.8.8
            if _passes_threshold(self._orientations.get(k), v, thr, self._directions.get(k)):
                if self._warm_start_t is not None:
                    return JudgeResult(
                        HitType.WARM_START, winner_id=winner_id, start_t=self._warm_start_t,
                    )
                return JudgeResult(HitType.FULL_HIT, winner_id=winner_id)
        return JudgeResult(HitType.MISS)


# ------------------------------------------------------------------
# Direction helpers (non_monotonic descriptors)
# ------------------------------------------------------------------


def _apply_direction(direction: str, v: float) -> float:
    """Return the scoring contribution for a non_monotonic key under the
    configured direction rule. Output is in [0, 1] for `range:[lo,hi]`
    (binary 0/1) and for `low` / `high` it mirrors safe / risky.
    """
    if direction == "high":
        return v
    if direction == "low":
        return 1.0 - v
    if direction.startswith("range:"):
        lo, hi = _parse_range(direction)
        return 1.0 if lo <= v <= hi else 0.0
    raise ValueError(f"Unknown direction {direction!r}; expected 'high' | 'low' | 'range:[lo,hi]'")


def _passes_threshold(
    orientation: Optional[str], v: float, thr: float, direction: Optional[str],
) -> bool:
    if orientation == _SAFE:
        return v >= thr
    if orientation == _RISKY:
        return v <= thr
    if orientation == _NON_MONOTONIC:
        if direction is None:
            raise ValueError("non_monotonic key requires a direction; bind_orientations should have caught this")
        if direction == "high":
            return v >= thr
        if direction == "low":
            return v <= thr
        if direction.startswith("range:"):
            lo, hi = _parse_range(direction)
            return lo <= v <= hi
        raise ValueError(f"Unknown direction {direction!r}")
    raise ValueError(f"Unknown orientation {orientation!r}")


def _parse_range(direction: str) -> tuple[float, float]:
    # Accept "range:[lo,hi]" or "range:lo,hi"; whitespace tolerant.
    body = direction[len("range:"):].strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    lo_s, hi_s = body.split(",")
    return float(lo_s.strip()), float(hi_s.strip())
