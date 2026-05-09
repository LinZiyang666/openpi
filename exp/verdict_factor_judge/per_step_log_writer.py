"""Compat shim — moved to ``exp.verdict_factor_judge.common.per_step_log_writer``."""

import sys

from exp.verdict_factor_judge.common import per_step_log_writer as _real

sys.modules[__name__] = _real
