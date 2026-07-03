"""Compat shim — re-exports the generalized recorder from ``openpi.serving``.

The per-step writer was generalized into
:mod:`openpi.serving.per_step_recorder` (schema-agnostic, two-mode). This module
keeps the verdict_factor_judge experiment's imports and ``_SORT_KEYS`` working
with byte-identical behavior: the default ``stamp_success=False`` mode never
injects a ``success`` key, so ``flush_episode()`` output is unchanged.

**Deprecated.** Scheduled for removal once all runners import the src module
directly and one full Verify passes (see
``logs/gate_data_collection_plan.log.md`` D7).
"""

from __future__ import annotations

from openpi.serving.per_step_recorder import _DEFAULT_SORT_KEYS as _SORT_KEYS
from openpi.serving.per_step_recorder import (  # noqa: F401
    PerStepWriter,
    PerStepWriterPool,
    filter_searched,
    install_atexit,
)


def _row_sort_key(row: dict) -> tuple:
    """Legacy merge sort key (kept for backward-compatible imports)."""
    return tuple(row.get(k, 0) for k in _SORT_KEYS)
