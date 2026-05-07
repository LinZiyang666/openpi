"""Layer 4 Composer Protocol (B4 of the verdict-factor refactor).

A Composer aggregates a calibrated factor dict into a JudgeResult. The
``declared_dependencies`` instance attribute (set by each subclass at
``__init__``) drives the static yaml-time validator: the union of factor
keys produced by the configured Layer 2 factors MUST be a superset of
``composer.declared_dependencies``.

Coupling map:
  DEPENDS ON:  components/judge.py (JudgeResult, HitType)
  CONSUMED BY: components/composite_judge.py (Layer 4 instance)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openpi.cache.components.judge import JudgeResult


@runtime_checkable
class Composer(Protocol):
    """Layer 4 — calibrated factor dict → JudgeResult.

    Instance attributes (each subclass sets these in ``__init__``):
      - ``declared_dependencies: set[str]`` — every factor key the
        composer requires to be present (typically the keys carried by
        non-zero weights / per-factor thresholds). The yaml-time
        validator differs this against the Layer 2 union and rejects the
        config if any declared key is missing.

    The ``bind_orientations`` hook is called once by
    ``CompositeJudge.__init__`` with the union orientation map of every
    Layer 2 factor. Subclasses use it to:
      1. Cross-check that any non_monotonic key with a non-zero weight /
         threshold has an entry in the configured ``directions`` block.
      2. Cache the orientation table for per-key flip / passes during
         ``compose``.

    ``compose`` runs per verdict. CompositeJudge guarantees ``winner_id``
    is non-None (empty results are short-circuited to MISS upstream), so
    subclasses MAY safely emit FULL_HIT / WARM_START with that winner_id.
    """

    declared_dependencies: set[str]

    def bind_orientations(self, orientations: dict[str, str]) -> None: ...

    def compose(
        self,
        calibrated: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        """Aggregate the calibrated factor dict and return the verdict.

        ``calibrated`` may carry NaN values (factor physical edge case).
        The composer subclass owns the NaN handling rule (skip / treat
        as risky / propagate to MISS / fall back to WARM_START etc.).
        """
        ...
