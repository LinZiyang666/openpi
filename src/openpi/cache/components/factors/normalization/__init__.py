"""Layer 1 — Normalization (B1 of the verdict-factor refactor).

Public surface:
    - ``Normalization`` Protocol (``factors/normalization/base.py``)
    - ``ZScoreNormalization`` default subclass (``factors/normalization/zscore.py``)

Coupling map:
  DEPENDS ON:  components/factors/base.py (LibraryStats)
  CONSUMED BY: components/factors/online.py (B2 per-verdict z-score),
               components/factors/offline.py (B2 episode-level z-score),
               components/composite_judge.py (B5 Layer 1 instance)
"""

from openpi.cache.components.factors.normalization.base import Normalization
from openpi.cache.components.factors.normalization.zscore import ZScoreNormalization

__all__ = ["Normalization", "ZScoreNormalization"]
