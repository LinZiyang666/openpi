"""Composer Protocol + concrete S1 / S2 / S3 classes.

A Composer aggregates a normalized factor dict into a `JudgeResult`.
Three implementations ship as B0 skeletons (constructor + signature +
`bind_orientations` stub); the actual scoring algorithms land in B1+.

Coupling map:
  DEPENDS ON:  components/judge.py (JudgeResult)
  CONSUMED BY: CompositeJudge (composer dependency injection)
  IF CHANGED:  CompositeJudge composition pipeline
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from openpi.cache.components.judge import JudgeResult


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

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        # B0 simply stores; B1+ cross-checks `directions` coverage for
        # non_monotonic keys with non-zero weight.
        self._orientations = dict(orientations)

    def compose(
        self,
        factors: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        raise NotImplementedError(
            "WeightedSumComposer.compose: B1+ algorithm"
        )


# ------------------------------------------------------------------
# S2: Conjunctive per-factor thresholds
# ------------------------------------------------------------------


class AndGateComposer:
    """S2: every key (per `per_factor_thresholds`) must pass its threshold.

    B0 stub. Algorithm body lands in B1+.
    """

    def __init__(
        self,
        per_factor_thresholds: dict[str, float],
        warm_start_t: Optional[float] = None,
    ) -> None:
        self._thresholds = dict(per_factor_thresholds)
        self._warm_start_t = warm_start_t
        self._orientations: dict[str, str] = {}

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        self._orientations = dict(orientations)

    def compose(
        self,
        factors: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        raise NotImplementedError(
            "AndGateComposer.compose: B1+ algorithm"
        )


# ------------------------------------------------------------------
# S3: Disjunctive per-factor thresholds
# ------------------------------------------------------------------


class OrGateComposer:
    """S3: any key passing its threshold emits a hit.

    B0 stub. Algorithm body lands in B1+.
    """

    def __init__(
        self,
        per_factor_thresholds: dict[str, float],
        warm_start_t: Optional[float] = None,
    ) -> None:
        self._thresholds = dict(per_factor_thresholds)
        self._warm_start_t = warm_start_t
        self._orientations: dict[str, str] = {}

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        self._orientations = dict(orientations)

    def compose(
        self,
        factors: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        raise NotImplementedError(
            "OrGateComposer.compose: B1+ algorithm"
        )
