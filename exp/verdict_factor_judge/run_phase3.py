"""Compat shim — moved to ``exp.verdict_factor_judge.phase3.runner``."""

import sys

from exp.verdict_factor_judge.phase3 import runner as _real

sys.modules[__name__] = _real
