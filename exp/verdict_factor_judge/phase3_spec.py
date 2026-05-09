"""Compat shim — moved to ``exp.verdict_factor_judge.phase3.spec``."""

import sys

from exp.verdict_factor_judge.phase3 import spec as _real

sys.modules[__name__] = _real
