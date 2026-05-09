"""Compat shim — moved to ``exp.verdict_factor_judge.common.generate_yamls``."""

import sys

from exp.verdict_factor_judge.common import generate_yamls as _real

sys.modules[__name__] = _real
