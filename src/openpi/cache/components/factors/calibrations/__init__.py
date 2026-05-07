"""Layer 3 — Calibration (B3 of the verdict-factor refactor).

Public surface:
    - ``Calibration`` Protocol
    - ``PercentileRollingCalibration`` default subclass

Coupling map:
  DEPENDS ON:  components/factors/base.py (CalibrationSamples)
  CONSUMED BY: components/composite_judge.py (B5 Layer 3 instance)
"""

from openpi.cache.components.factors.calibrations.base import Calibration
from openpi.cache.components.factors.calibrations.percentile_rolling import (
    PercentileRollingCalibration,
)

__all__ = ["Calibration", "PercentileRollingCalibration"]
