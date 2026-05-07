"""PercentileRollingCalibration — Layer 3 default implementation.

Per-key rolling-window percentile rank, with the buffer pre-filled to
``window_size`` at ``bind_keys`` time. Cold-start strategies (force_miss
/ passthrough / lenient) from the legacy normalizer have been removed;
the buffer is always saturated when the first verdict arrives, so NaN in
the output can only come from a NaN input (factor physical edge case).

Coupling map:
  DEPENDS ON:  components/factors/base.py (CalibrationSamples),
               components/factors/calibrations/base.py (Calibration Protocol)
  CONSUMED BY: components/composite_judge.py
"""

from __future__ import annotations

import collections
import math

from openpi.cache.components.factors.base import CalibrationSamples


class PercentileRollingCalibration:
    """Per-key rolling-window percentile rank.

    Usage:
        calib = PercentileRollingCalibration(samples, window_size=200)
        calib.bind_keys(union_keys)        # fail-fast if any key < window_size
        normed = calib(raw_factor_dict)    # per verdict

    ``samples.samples`` may carry more keys than ``bind_keys`` declares;
    extras are silently ignored. Conversely, every bound key MUST have at
    least ``window_size`` non-NaN samples — otherwise ``bind_keys`` raises.
    """

    def __init__(
        self,
        samples: CalibrationSamples,
        *,
        window_size: int = 200,
    ) -> None:
        if samples is None:
            raise ValueError("PercentileRollingCalibration requires CalibrationSamples")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._samples = samples
        self._window_size = int(window_size)
        self._buffers: dict[str, collections.deque] = {}
        self._bound: bool = False

    # ------------------------------------------------------------------
    # Setup (called once by CompositeJudge.__init__)
    # ------------------------------------------------------------------

    def bind_keys(self, keys: list[str]) -> None:
        """Validate per-key sample counts and pre-fill rolling buffers.

        Fail-fast on:
          - any bound key absent from ``samples.samples``
          - any bound key with fewer non-NaN samples than ``window_size``

        Both conditions are configuration errors (yaml asks for keys the
        offline / warmup source did not collect, or ran the warmup pass
        too short). Per plan §6.3 there is no graceful cold-start state.
        """
        if self._bound:
            raise RuntimeError(
                "PercentileRollingCalibration.bind_keys called twice; "
                "Calibration instances are one-shot per CompositeJudge"
            )
        for k in keys:
            history = self._samples.samples.get(k)
            if history is None:
                raise KeyError(
                    f"Calibration source missing key {k!r} (declared by factor layer "
                    "but not present in samples)"
                )
            non_nan = [v for v in history if not math.isnan(v)]
            if len(non_nan) < self._window_size:
                raise ValueError(
                    f"Calibration key {k!r}: only {len(non_nan)} non-NaN samples, "
                    f"need >= window_size={self._window_size}"
                )
            buf: collections.deque[float] = collections.deque(maxlen=self._window_size)
            # Take the most recent ``window_size`` samples (drop early ones if oversized).
            for v in non_nan[-self._window_size:]:
                buf.append(float(v))
            self._buffers[k] = buf
        self._bound = True

    # ------------------------------------------------------------------
    # Per-verdict
    # ------------------------------------------------------------------

    def __call__(self, raw: dict[str, float]) -> dict[str, float]:
        if not self._bound:
            raise RuntimeError(
                "PercentileRollingCalibration.__call__ before bind_keys"
            )
        out: dict[str, float] = {}
        for k, v in raw.items():
            buf = self._buffers.get(k)
            if buf is None:
                # Defensive: factor layer produced an unbound key. CompositeJudge's
                # key contract assertion catches this earlier; surface NaN here.
                out[k] = float("nan")
                continue
            if math.isnan(v):
                # NaN propagates; do not enqueue (would corrupt the rolling window).
                out[k] = float("nan")
                continue
            buf.append(float(v))                       # newest into the window
            out[k] = self._percentile_rank(buf, float(v))
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_episode_start(self) -> None:
        # Plan §6.11 #9: rolling window is cross-episode by design (the
        # buffer is the percentile distribution; resetting it would
        # destroy stability). No-op.
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile_rank(buf: "collections.deque[float]", v: float) -> float:
        """Fraction of buffered samples <= v.

        The current ``v`` was just appended (so a fresh, all-time-high
        sample maps to 1.0 = top of pool). For an empty buffer the helper
        returns NaN — but in this class the buffer is never empty after
        ``bind_keys`` has run (window_size >= 1 + non-empty preload).
        """
        n = len(buf)
        if n == 0:
            return float("nan")
        rank = sum(1 for x in buf if x <= v)
        return rank / n
