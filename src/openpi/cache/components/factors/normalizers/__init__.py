"""Normalizer Protocol + concrete PercentileRollingNormalizer.

A Normalizer maps raw factor values produced by extractors into a
normalized space (typically [0, 1] percentile rank), so heterogeneous
descriptors can be combined by a Composer. B0 ships the protocol +
`PercentileRollingNormalizer` skeleton; algorithm body lands in B1+.

Coupling map:
  CONSUMED BY: CompositeJudge (normalizer dependency injection)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Normalizer(Protocol):
    """Per-key normalization.

    `bind_keys` is called once by `CompositeJudge.__init__` with the
    full set of keys all extractors will produce, so the implementation
    can pre-allocate per-key state (ring buffers, running stats, etc.).
    """

    def bind_keys(self, keys: list[str]) -> None: ...

    def __call__(self, raw: dict[str, float]) -> dict[str, float]:
        """Map raw factor values -> normalized values (typically [0, 1]).

        NaN inputs propagate as NaN (per the verdict-factor missing-factor
        rule); the Composer is responsible for handling NaN per its own
        type.

        Cold-start sentinel: when the normalizer is in `force_miss` mode
        and the rolling window is not yet ready, it MUST return a dict
        where every key maps to NaN. CompositeJudge detects all-NaN and
        returns HitType.MISS directly without invoking the Composer; this
        keeps the cold-start path representable through the existing
        `dict[str, float]` interface without adding a status channel.
        """
        ...

    def on_episode_start(self) -> None:
        """Lifecycle hook from Orchestrator. Default impls are no-op
        (rolling window survives across episodes for stable percentile)."""
        ...


# ------------------------------------------------------------------
# Default implementation skeleton
# ------------------------------------------------------------------


class PercentileRollingNormalizer:
    """Per-key percentile rank over a rolling window.

    B0 stub: stores params + key list. Algorithm body and
    `cold_start_strategy` semantics land in B1+; the three documented
    strategies are "force_miss" (default; returns all-NaN until window
    fills), "passthrough" (skip normalization), and "lenient" (compute
    percentile from partial samples, returning all-NaN below 10).
    """

    def __init__(
        self,
        window_size: int = 200,
        cold_start_strategy: str = "force_miss",
    ) -> None:
        self._window_size = window_size
        self._cold_start_strategy = cold_start_strategy
        self._keys: list[str] = []

    def bind_keys(self, keys: list[str]) -> None:
        self._keys = list(keys)

    def __call__(self, raw: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError(
            "PercentileRollingNormalizer.__call__: B1+ algorithm"
        )

    def on_episode_start(self) -> None:
        # Default: rolling window survives across episodes so percentile
        # stats are stable. Override in subclasses if per-episode reset
        # semantics are desired.
        return None
