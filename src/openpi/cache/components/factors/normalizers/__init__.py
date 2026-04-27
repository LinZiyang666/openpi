"""Normalizer Protocol + concrete PercentileRollingNormalizer.

A Normalizer maps raw factor values produced by extractors into a
normalized space (typically [0, 1] percentile rank), so heterogeneous
descriptors can be combined by a Composer. B0 ships the protocol +
`PercentileRollingNormalizer` skeleton; algorithm body lands in B1+.

Coupling map:
  CONSUMED BY: CompositeJudge (normalizer dependency injection)
"""

from __future__ import annotations

import collections
import math
from typing import Protocol, runtime_checkable

# Cold-start strategies — see docs §2.8.8.
_FORCE_MISS = "force_miss"
_PASSTHROUGH = "passthrough"
_LENIENT = "lenient"
_VALID_STRATEGIES = frozenset({_FORCE_MISS, _PASSTHROUGH, _LENIENT})

# Below this sample count `lenient` still treats the buffer as cold
# (per docs §2.8.8 "lenient: N < 10 returns all-NaN").
_LENIENT_MIN_SAMPLES = 10


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
        cold_start_strategy: str = _FORCE_MISS,
    ) -> None:
        if cold_start_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"cold_start_strategy must be one of {sorted(_VALID_STRATEGIES)}, "
                f"got {cold_start_strategy!r}"
            )
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._cold_start_strategy = cold_start_strategy
        self._keys: list[str] = []
        self._buffers: dict[str, collections.deque[float]] = {}

    def bind_keys(self, keys: list[str]) -> None:
        self._keys = list(keys)
        self._buffers = {
            k: collections.deque(maxlen=self._window_size) for k in self._keys
        }

    def __call__(self, raw: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in raw.items():
            buf = self._buffers.get(k)
            if buf is None:
                # Defensive: extractor produced an unbound key. CompositeJudge's
                # key contract assertion catches this earlier in the pipeline,
                # so this branch is a safety net for direct normalizer use.
                buf = collections.deque(maxlen=self._window_size)
                self._buffers[k] = buf
            if not math.isnan(v):
                buf.append(v)

            if self._cold_start_strategy == _PASSTHROUGH:
                # Don't normalize at all — useful for ablation experiments
                # comparing rank-based aggregation vs raw-factor aggregation.
                out[k] = v
                continue

            n = len(buf)
            if self._cold_start_strategy == _FORCE_MISS:
                if n < self._window_size:
                    out[k] = float("nan")               # cold-start sentinel
                    continue
            elif self._cold_start_strategy == _LENIENT:
                if n < _LENIENT_MIN_SAMPLES:
                    out[k] = float("nan")
                    continue
                # else: compute percentile from partial buffer (n < window_size)

            out[k] = self._percentile_rank(buf, v)
        return out

    @staticmethod
    def _percentile_rank(buf, v: float) -> float:
        """Fraction of buffered samples <= v. Returns NaN for NaN input.

        The buffer always contains the current `v` (added before this call)
        so a fresh value with no peers maps to 1.0 (top of pool).
        """
        if math.isnan(v):
            return float("nan")
        n = len(buf)
        if n == 0:
            return float("nan")
        rank = sum(1 for x in buf if x <= v)
        return rank / n

    def on_episode_start(self) -> None:
        # Default: rolling window survives across episodes so percentile
        # stats are stable. Override in subclasses if per-episode reset
        # semantics are desired.
        return None
