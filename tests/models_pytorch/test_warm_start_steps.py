"""Warm-start denoise step-count correctness (regression for the round() bug).

The warm-start path resumes flow-matching from a cached intermediate at
``start_t`` and must run exactly as many Euler steps as the original GPU-tensor
loop ``while timestep >= -dt/2``. That loop runs ``floor(start_t*num_steps+0.5)``
iterations. A previous refactor used ``round()`` (banker's rounding), which
silently drops/adds a step on half-integer boundaries for non-canonical
start_t — these tests pin the correct behaviour.
"""

import pytest

from openpi.models_pytorch.pi0_pytorch import _warm_start_num_steps


@pytest.mark.parametrize("start_t,num_steps,expected", [
    # documented canonical examples
    (0.3, 10, 3),
    (0.5, 10, 5),
    (0.7, 10, 7),
    (0.1, 10, 1),
    (0.9, 10, 9),
    (1.0, 10, 10),
    (0.0, 10, 0),
    # other step counts
    (0.5, 4, 2),
    (0.5, 8, 4),
    (0.25, 4, 1),
    (0.75, 4, 3),
    # half-integer boundaries where round() (banker's) was WRONG:
    (0.05, 10, 1),   # 0.5 -> round()=0, loop ran 1
    (0.25, 10, 3),   # 2.5 -> round()=2, loop ran 3
    (0.45, 10, 5),   # 4.5 -> round()=4, loop ran 5
])
def test_warm_start_num_steps(start_t, num_steps, expected):
    assert _warm_start_num_steps(start_t, num_steps) == expected


def test_warm_start_differs_from_buggy_round_at_boundaries():
    # Regression guard: the half-integer boundaries are exactly where the old
    # round()-based count diverged. Confirm the fix no longer equals round().
    for start_t in (0.05, 0.25, 0.45):
        assert _warm_start_num_steps(start_t, 10) != round(start_t * 10)


def test_warm_start_matches_reference_loop_on_grid():
    # Reproduce the old loop's iteration count via the exact mathematical
    # condition (no float accumulation) across a dense grid incl. boundaries.
    def ref(start_t, num_steps):
        n = 0
        while start_t - n / num_steps >= 1.0 / (2 * num_steps) - 1e-12:
            n += 1
        return n

    for num_steps in (4, 5, 8, 10, 20):
        for k in range(2 * num_steps + 1):
            start_t = k / (2 * num_steps)  # hits every half-step boundary
            assert _warm_start_num_steps(start_t, num_steps) == ref(start_t, num_steps), \
                (start_t, num_steps)
