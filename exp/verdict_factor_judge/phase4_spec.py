"""Compat shim — moved to ``exp.verdict_factor_judge.phase4.spec``."""

import sys

from exp.verdict_factor_judge.phase4 import spec as _real

sys.modules[__name__] = _real
