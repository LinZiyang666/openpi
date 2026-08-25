"""Six ways per-step evidence goes missing, and the gate that refuses each.

``teacher_ratio`` is this experiment's x-axis and its denominator is read from
the per-step evidence. Losing part of that evidence does not produce an error
or an implausible number -- it produces a Pareto point in the right region that
is simply wrong. A row count cannot catch it either: one worker's file can
vanish entirely while longer episodes elsewhere push the total past any
threshold a reader would think to set. So the gate checks set equality of
episode identities, and these tests are the six constructions it must reject.
"""

from __future__ import annotations

import pytest

from exp.libero_groot.analysis.gate_pareto.analyze_gate_pareto import (
    IntegrityError,
    aggregate_arm,
    check_arm_integrity,
)

EXPECT = 4
LANES = 3


def _results(n: int = EXPECT) -> list[dict]:
    return [
        {
            "task_id": i // 2,
            "init_state_idx": i % 2,
            "orig_init_state_idx": i % 2,
            "episode_id": i,
            "seed": 7,
            "success": i % 2 == 0,
        }
        for i in range(n)
    ]


def _per_step(episode_ids, steps: int = 3, hit: str = "FULL_HIT") -> list[dict]:
    return [
        {
            "yaml_id": "gp_sp_fh40",
            "episode_id": ep,
            "step_idx": s,
            "hit_type": hit,
            "searched": True,
            "cp1_score": 0.9,
        }
        for ep in episode_ids
        for s in range(steps)
    ]


def _merge(found: int = LANES) -> dict:
    return {"lanes_expected": LANES, "lanes_found": found, "rows": 0}


def test_clean_evidence_passes():
    check_arm_integrity(
        _results(), _per_step(range(EXPECT)), expect_ep=EXPECT, merge=_merge()
    )


def test_i6_a_missing_worker_file_is_named_before_anything_else():
    # The one failure a row count provably cannot see. It is reported first
    # because every downstream mismatch is its symptom, and pointing at the
    # symptom would send the reader to the wrong place.
    with pytest.raises(IntegrityError, match="2 of 3 worker files"):
        check_arm_integrity(
            _results(), _per_step(range(EXPECT)), expect_ep=EXPECT, merge=_merge(2)
        )


def test_i6_a_sidecar_without_lane_counts_is_refused():
    with pytest.raises(IntegrityError, match="lanes_expected"):
        check_arm_integrity(
            _results(), _per_step(range(EXPECT)), expect_ep=EXPECT, merge={"rows": 12}
        )


def test_i1_a_short_arm_is_refused():
    with pytest.raises(IntegrityError, match="expected 4"):
        check_arm_integrity(
            _results(3), _per_step(range(3)), expect_ep=EXPECT, merge=_merge()
        )


def test_i2_a_duplicated_episode_result_is_refused():
    rows = _results()
    rows.append(dict(rows[0]))
    with pytest.raises(IntegrityError, match="duplicate"):
        check_arm_integrity(
            rows, _per_step(range(EXPECT)), expect_ep=EXPECT + 1, merge=_merge()
        )


def test_i3_a_missing_episode_on_the_per_step_side_is_refused():
    """Still refused -- now reported by I4, which names the cause.

    An episode absent from the per-step side has, by definition, no verdict
    row, so I4 catches it first and says so. I3 keeps the "extra" direction,
    which I4 cannot see.
    """
    with pytest.raises(IntegrityError, match="have no verdict row"):
        check_arm_integrity(
            _results(), _per_step(range(EXPECT - 1)), expect_ep=EXPECT, merge=_merge()
        )


def test_i3_an_extra_episode_on_the_per_step_side_is_refused():
    # A stale file from a previous attempt merged in alongside this run's.
    with pytest.raises(IntegrityError, match="extra 1"):
        check_arm_integrity(
            _results(), _per_step(range(EXPECT + 1)), expect_ep=EXPECT, merge=_merge()
        )


def test_i4_an_episode_with_no_verdict_row_is_refused():
    rows = _per_step(range(EXPECT))
    for row in rows:
        if row["episode_id"] == 2:
            row["hit_type"] = None
    with pytest.raises(IntegrityError, match="no verdict row"):
        check_arm_integrity(_results(), rows, expect_ep=EXPECT, merge=_merge())


def test_i5_a_duplicated_decision_is_refused():
    rows = _per_step(range(EXPECT))
    rows.append(dict(rows[0]))
    with pytest.raises(IntegrityError, match=r"duplicate \(episode_id, step_idx\)"):
        check_arm_integrity(_results(), rows, expect_ep=EXPECT, merge=_merge())


def test_the_gate_is_independent_of_the_ratio_it_protects():
    # Aggregation assumes the gate has run; the two must not be entangled, or a
    # future refactor could compute the ratio on evidence that never passed.
    rows = _per_step(range(EXPECT))
    point = aggregate_arm(_results(), rows, expect_ep=EXPECT, arm="gp_sp_fh40")
    assert point["decisions"] == EXPECT * 3
    assert point["teacher_ratio"] == 0.0


def test_the_sidecar_is_not_an_optional_argument():
    # It was optional once, and aggregate() simply passed None when the file was
    # absent -- which skipped I6 entirely for exactly the runs most likely to
    # have lost a worker. A fail-closed check with an opt-out is not fail-closed.
    with pytest.raises(TypeError):
        check_arm_integrity(  # type: ignore[call-arg]
            _results(), _per_step(range(EXPECT)), expect_ep=EXPECT
        )
