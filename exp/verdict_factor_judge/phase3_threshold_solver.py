"""Compat shim — moved to ``exp.verdict_factor_judge.phase3.threshold_solver``.

Replaces this shim with an alias to the real module so that
``sys.modules[__name__]`` and ``getattr(...)`` go directly to the new location
(needed by tests that monkeypatch attributes via the old path).
"""

import sys

from exp.verdict_factor_judge.phase3 import threshold_solver as _real

sys.modules[__name__] = _real
