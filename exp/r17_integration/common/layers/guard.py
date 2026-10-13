"""Retired. The old only-no-progress guard (R8 span arithmetic) is not part of the five-layer design.

Owner ruling (exp/offline_search/rounds/r17/cache/OWNER_RULINGS.md, confirmed 2026-10-13): the stall gate is the
R6/R7 Q3 calibrated stall trigger (R6-C-v2), served with ``recipe.miss_mode="stall_only"`` and a ``stall_path``.
Any use of the old guard is an error.
"""

RETIRED = ("the old no-progress guard is retired; the stall gate is the Q3 calibrated stall trigger "
           "(recipe.miss_mode='stall_only' with a stall_path)")


class NoProgressGuard:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(RETIRED)
