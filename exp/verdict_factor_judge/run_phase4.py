"""Compat shim — moved to ``exp.verdict_factor_judge.phase4.runner``."""

import sys

from exp.verdict_factor_judge.phase4 import runner as _real

sys.modules[__name__] = _real
