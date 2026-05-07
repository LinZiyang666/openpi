"""Layer 3 Calibration Protocol.

A Calibration instance is constructed once at server startup from a
``CalibrationSamples`` (per-key historical raw factor values). At
``bind_keys`` time the union of factor keys becomes known and the
implementation MUST fail-fast on any key whose sample count is below
``window_size`` — there is no cold-start state.

Coupling map:
  DEPENDS ON:  components/factors/base.py (CalibrationSamples)
  CONSUMED BY: components/composite_judge.py (Layer 3 instance)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openpi.cache.components.factors.base import CalibrationSamples


@runtime_checkable
class Calibration(Protocol):
    """Layer 3 — per-key calibration of raw factor values.

    Constructor must be ``__init__(samples: CalibrationSamples, **params)``.
    ``samples`` is stashed but not validated yet — the union of keys is not
    known at construction.

    ``bind_keys(keys)`` is called once by ``CompositeJudge.__init__`` with
    the union of factor keys. The implementation MUST fail-fast at that
    point if any key has fewer non-NaN samples than the configured
    ``window_size`` (or its equivalent backing structure).

    ``__call__(raw)`` runs per verdict: NaN inputs propagate to NaN outputs
    unchanged; finite inputs are mapped through the calibration (typically
    ``[0, 1]`` percentile rank).

    Buffers persist across episodes; ``on_episode_start`` is a no-op by
    default (plan §6.11 #9).
    """

    def __init__(self, samples: CalibrationSamples, **params) -> None: ...

    def bind_keys(self, keys: list[str]) -> None: ...

    def __call__(self, raw: dict[str, float]) -> dict[str, float]: ...

    def on_episode_start(self) -> None: ...
