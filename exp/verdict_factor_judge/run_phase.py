"""Compat shim — moved to ``exp.verdict_factor_judge.common.run_phase``."""

import sys

from exp.verdict_factor_judge.common import run_phase as _real

sys.modules[__name__] = _real
