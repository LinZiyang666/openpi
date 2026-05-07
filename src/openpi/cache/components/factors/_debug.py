"""Centralized verdict-factor debug logger.

Replaces the per-module ``_VERDICT_DEBUG`` env-var + ``_verdict_logger``
duplication that existed in ``judge.py`` / ``runtime_continuity.py`` /
``consensus.py`` / ``composers/__init__.py``. All factor / composer / judge
modules read the debug toggle and logger from here.

Toggle from server shell:
    OPENPI_CACHE_VERDICT_DEBUG=1 uv run scripts/serve_policy.py ... |& tee /tmp/vd.log
then ``grep '\[vd ' /tmp/vd.log`` inspects per-verdict structured lines.

Coupling map:
  CONSUMED BY: components/judge.py, components/composite_judge.py,
               components/dumping_judge.py, factors/online.py,
               factors/offline.py, factors/topk.py,
               factors/composers/*.py
"""

from __future__ import annotations

import logging
import os

# Single source of truth for the env-var gate. Zero cost when unset.
VERDICT_DEBUG: bool = os.environ.get("OPENPI_CACHE_VERDICT_DEBUG") == "1"

# Single named logger so all per-verdict trace lines land under one channel.
verdict_logger: logging.Logger = logging.getLogger("openpi.cache.verdict_debug")
