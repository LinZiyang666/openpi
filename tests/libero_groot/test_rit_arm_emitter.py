"""Judge assembly for the LIBERO x GR00T arm set.

The ladder-to-judge step is where a cell can be silently turned into a
different experiment. Two shapes matter:

*   a cell that gives FULL_HIT a **zero** share -- 30 of the 34 GST cells --
    must keep a FULL threshold above the score domain so its warm rungs still
    fire. Dropping the FULL rung instead makes every one of those cells the
    all-MISS arm, which validates, runs, and reports a plausible number.
*   a rung whose cut is not strictly below the rung above it can never fire,
    and the loader rejects that shape; it has to be dropped and recorded.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from exp.libero_groot.emit_rit_arms import (
    CUT_NONE,
    WARM_TS,
    _judge_from_cuts,
    gst_cells,
    gst_cuts,
)


def test_zero_full_share_keeps_the_warm_rungs_alive():
    """f=0 means 'no FULL_HIT', not 'no cache'."""
    judge, dropped = _judge_from_cuts([math.inf, math.inf, 0.99225], WARM_TS)
    assert judge["threshold"] == CUT_NONE
    assert judge["warm_tiers"] == [{"threshold": 0.99225, "start_t": 0.5}]
    assert dropped == ["warm75"]
    # The failure this guards: an all-MISS judge with no warm route at all.
    assert "warm_tiers" in judge


def test_thresholds_are_strictly_decreasing_down_the_ladder():
    judge, dropped = _judge_from_cuts([0.995, 0.990, 0.985], WARM_TS)
    cuts = [judge["threshold"]] + [t["threshold"] for t in judge["warm_tiers"]]
    assert cuts == sorted(cuts, reverse=True)
    assert len(set(cuts)) == len(cuts)
    assert dropped == []


def test_a_rung_that_can_never_fire_is_dropped_and_recorded():
    """warm75 at the same cut as FULL is unreachable: FULL takes those steps."""
    judge, dropped = _judge_from_cuts([0.99, 0.99, 0.98], WARM_TS)
    assert judge["threshold"] == 0.99
    assert [t["start_t"] for t in judge["warm_tiers"]] == [0.5]
    assert dropped == ["warm75"]


def test_full_only_ladder_carries_no_warm_tiers():
    judge, dropped = _judge_from_cuts([0.99, math.inf, math.inf], WARM_TS)
    assert judge == {"type": "threshold", "threshold": 0.99}
    assert dropped == ["warm75", "warm50"]


def test_gst_grid_matches_the_pi05_line():
    cells = gst_cells()
    assert len(cells) == 34
    assert (0, 0, 0) not in cells
    assert all(sum(c) <= 80 and all(x % 20 == 0 for x in c) for c in cells)


def test_gst_cuts_are_descending_quantiles_and_zero_shares_are_infinite():
    scores = np.linspace(0.90, 0.999, 1000)
    cuts = gst_cuts(scores, (0.2, 0.2, 0.0))
    assert cuts[0] > cuts[1]
    assert math.isinf(cuts[2])
    # The first cut admits ~20% of the sample, the second the next ~20%.
    assert 0.18 < float((scores >= cuts[0]).mean()) < 0.22
    assert 0.38 < float((scores >= cuts[1]).mean()) < 0.42


def test_every_gst_cell_yields_a_judge_that_can_serve_something():
    """No cell may collapse into an arm with neither a FULL nor a warm route."""
    scores = np.linspace(0.90, 0.999, 1000)
    for full, wa, wb in gst_cells():
        judge, _ = _judge_from_cuts(
            gst_cuts(scores, (full / 100.0, wa / 100.0, wb / 100.0)), WARM_TS
        )
        serves = judge["threshold"] < CUT_NONE or judge.get("warm_tiers")
        assert serves, f"cell f{full}w{wa}v{wb} serves nothing"


@pytest.mark.parametrize("t,remaining", [(0.75, 2), (0.5, 4)])
def test_warm_rungs_are_the_two_and_four_step_points(t, remaining):
    from openpi.cache.types import groot_n15_schedule

    assert t in WARM_TS
    assert groot_n15_schedule(8).remaining_steps(t) == remaining
