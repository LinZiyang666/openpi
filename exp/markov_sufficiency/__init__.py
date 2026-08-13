"""Markov Sufficiency experiment package (E1-E5).

Tests whether history still carries extractable value for the training-free
cache, under the two-condition framework "(a) does the target depend on time
structure AND (b) can the scoring operator express it".

Public interface
----------------
This module exposes only the constants frozen at G1 time. They are inputs to
the pre-registered analysis, not tunables: every one of them was computed
before any outcome data was inspected, and ``tests/markov_sufficiency`` asserts
that recomputation reproduces them. A drift means the library artifact or the
output chain changed, which invalidates the pre-registration and requires a new
G1 round -- it must never be "fixed" by editing the numbers here.

Key dependencies: the drivers in this package (``e1_*``, ``e2_*``, ``e3_*``,
``emit_e4_yamls``, ``emit_e5_yamls``) and the private helpers ``_library``,
``_scoring``, ``_stats``, ``_timeaxis``.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Frozen constants (plan §3.3.1b / §3.3.1c)
# ------------------------------------------------------------------

SUITES = ("libero_spatial", "libero_10")

#: Physical calibration threshold: P95 of the client-space executed-action
#: distance between adjacent cycles of the same trajectory. Reads as "the upper
#: tail of how much the action changes under normal time evolution".
TAU_A_PHYS = {"libero_spatial": 1.9994, "libero_10": 2.0036}

#: Wrong-phase threshold in inference cycles: max L such that
#: median(D(L)) <= TAU_A_PHYS, derived from library data only (plan §3.3.1b).
W_PHASE = {"libero_spatial": 6, "libero_10": 8}

#: Equal-exposure window in inference cycles: P10 of the success-group cycle
#: count, taken from the independent gate_research batch (plan §3.3.1c). Never
#: recomputed from the analysed batch -- that would let the outcome pick the
#: estimand.
K_WINDOW = {"libero_spatial": 17, "libero_10": 34}

#: Practical-equivalence bounds used by the CI-based verdicts, so that a
#: non-rejection is never read as equivalence (plan §3.3.3 / §5.4).
DELTA_E2 = 0.10
DELTA_E4 = 0.03

#: Tolerance for reproducing the frozen constants in tests. The constants are
#: rounded to four decimals in this file; recomputation must land within this
#: window of the stored value.
FROZEN_TOL = 5e-4

__all__ = [
    "DELTA_E2",
    "DELTA_E4",
    "FROZEN_TOL",
    "K_WINDOW",
    "SUITES",
    "TAU_A_PHYS",
    "W_PHASE",
]
