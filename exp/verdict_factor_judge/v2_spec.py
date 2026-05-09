"""Compat shim — moved to ``exp.verdict_factor_judge.common.v2_spec``."""

import sys

from exp.verdict_factor_judge.common import v2_spec as _real

sys.modules[__name__] = _real
