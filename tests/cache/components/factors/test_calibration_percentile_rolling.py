"""Unit tests for ``PercentileRollingCalibration`` (B3 refactor)."""

from __future__ import annotations

import math

import pytest

from openpi.cache.components.factors.base import CalibrationSamples
from openpi.cache.components.factors.calibrations import (
    Calibration,
    PercentileRollingCalibration,
)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_satisfies_protocol() -> None:
    inst = PercentileRollingCalibration(CalibrationSamples({}))
    assert isinstance(inst, Calibration)


def test_construction_rejects_none_samples() -> None:
    with pytest.raises(ValueError, match="CalibrationSamples"):
        PercentileRollingCalibration(None)  # type: ignore[arg-type]


def test_construction_rejects_non_positive_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        PercentileRollingCalibration(CalibrationSamples({}), window_size=0)


# ----------------------------------------------------------------------
# bind_keys — fail-fast at startup (no cold-start state)
# ----------------------------------------------------------------------


def test_bind_keys_rejects_missing_key() -> None:
    samples = CalibrationSamples({"a": [0.0] * 50})
    calib = PercentileRollingCalibration(samples, window_size=10)
    with pytest.raises(KeyError, match="missing key 'b'"):
        calib.bind_keys(["a", "b"])


def test_bind_keys_rejects_undersized_history() -> None:
    samples = CalibrationSamples({"a": [0.0] * 5})
    calib = PercentileRollingCalibration(samples, window_size=10)
    with pytest.raises(ValueError, match="only 5 non-NaN samples"):
        calib.bind_keys(["a"])


def test_bind_keys_skips_nans_in_count() -> None:
    """A stream with N total samples but lots of NaNs should fail if
    non-NaN count < window_size."""
    samples = CalibrationSamples({"a": [float("nan"), float("nan"), 1.0, 2.0, 3.0]})
    calib = PercentileRollingCalibration(samples, window_size=4)
    with pytest.raises(ValueError, match="only 3 non-NaN samples"):
        calib.bind_keys(["a"])


def test_bind_keys_takes_most_recent_when_oversized() -> None:
    """If the source has more samples than ``window_size``, the buffer
    should be filled with the most-recent slice."""
    samples = CalibrationSamples({"a": list(range(100))})
    calib = PercentileRollingCalibration(samples, window_size=10)
    calib.bind_keys(["a"])
    # Push a value below all stored samples → percentile_rank should be 1/11
    # (only the new value is the floor; the 10 historical [90..99] are above).
    out = calib({"a": -1.0})
    # buffer is now [90..99] then -1 appended (popping 90) → 1 sample <= -1 (the new one) of 10 → 0.1
    assert math.isclose(out["a"], 0.1)


def test_bind_keys_called_twice_raises() -> None:
    samples = CalibrationSamples({"a": [0.0] * 10})
    calib = PercentileRollingCalibration(samples, window_size=5)
    calib.bind_keys(["a"])
    with pytest.raises(RuntimeError, match="bind_keys called twice"):
        calib.bind_keys(["a"])


def test_bind_keys_ignores_extra_unused_keys_in_samples() -> None:
    samples = CalibrationSamples({"a": [0.0] * 50, "z_unused": [9.9]})
    calib = PercentileRollingCalibration(samples, window_size=10)
    calib.bind_keys(["a"])   # should not raise


# ----------------------------------------------------------------------
# __call__ — percentile rank over rolling window
# ----------------------------------------------------------------------


def test_call_before_bind_raises() -> None:
    samples = CalibrationSamples({"a": [0.0] * 10})
    calib = PercentileRollingCalibration(samples, window_size=5)
    with pytest.raises(RuntimeError, match="bind_keys"):
        calib({"a": 0.5})


def test_call_returns_percentile_rank() -> None:
    samples = CalibrationSamples({"a": [float(i) for i in range(10)]})
    calib = PercentileRollingCalibration(samples, window_size=10)
    calib.bind_keys(["a"])
    # Median of [0..9] is 4.5; pushing 4.5 → buffer drops 0, becomes [1..9, 4.5].
    # 5 samples (1, 2, 3, 4, 4.5) <= 4.5 of 10 → 0.5.
    out = calib({"a": 4.5})
    assert math.isclose(out["a"], 0.5)


def test_call_top_value_returns_one() -> None:
    samples = CalibrationSamples({"a": list(range(10))})
    calib = PercentileRollingCalibration(samples, window_size=10)
    calib.bind_keys(["a"])
    out = calib({"a": 100.0})    # all <= 100
    assert math.isclose(out["a"], 1.0)


def test_call_bottom_value_returns_smallest_fraction() -> None:
    samples = CalibrationSamples({"a": list(range(10))})
    calib = PercentileRollingCalibration(samples, window_size=10)
    calib.bind_keys(["a"])
    out = calib({"a": -100.0})
    # Only the new value is <= -100; rest are above. Buffer becomes [1..9, -100],
    # so 1 of 10 samples <= -100 → 0.1.
    assert math.isclose(out["a"], 0.1)


def test_call_propagates_nan_input_without_enqueueing() -> None:
    samples = CalibrationSamples({"a": [0.0] * 5})
    calib = PercentileRollingCalibration(samples, window_size=5)
    calib.bind_keys(["a"])
    out = calib({"a": float("nan")})
    assert math.isnan(out["a"])
    # Buffer untouched: a follow-up with 0.5 sees the original 5 zeros + the new 0.5.
    out = calib({"a": 0.5})
    # Buffer is [0,0,0,0,0.5]: 5 samples <= 0.5 of 5 → 1.0.
    assert math.isclose(out["a"], 1.0)


def test_call_unbound_key_in_raw_returns_nan() -> None:
    """Defensive — CompositeJudge enforces key contract upstream, but we
    still surface NaN rather than KeyError if a stray key sneaks in."""
    samples = CalibrationSamples({"a": [0.0] * 5})
    calib = PercentileRollingCalibration(samples, window_size=5)
    calib.bind_keys(["a"])
    out = calib({"a": 0.5, "stray": 1.0})
    assert math.isclose(out["a"], 1.0)
    assert math.isnan(out["stray"])


# ----------------------------------------------------------------------
# Cross-episode persistence (plan §6.11 #9)
# ----------------------------------------------------------------------


def test_on_episode_start_does_not_reset_buffer() -> None:
    samples = CalibrationSamples({"a": list(range(10))})
    calib = PercentileRollingCalibration(samples, window_size=10)
    calib.bind_keys(["a"])
    calib({"a": 100.0})           # push outlier
    calib.on_episode_start()      # should not clear
    out = calib({"a": 100.0})     # second outlier — buffer carried 100 from before
    # Buffer just before this call: [2..9, 100, 100]. Push 100 → drop 2.
    # buffer becomes [3..9, 100, 100, 100]; samples <= 100 = all 10 → 1.0.
    assert math.isclose(out["a"], 1.0)


# ----------------------------------------------------------------------
# Multi-key independence
# ----------------------------------------------------------------------


def test_multi_key_buffers_are_independent() -> None:
    samples = CalibrationSamples({
        "a": [float(i) for i in range(10)],     # range 0..9
        "b": [float(i) * 100 for i in range(10)],  # range 0..900
    })
    calib = PercentileRollingCalibration(samples, window_size=10)
    calib.bind_keys(["a", "b"])

    # Send the same numeric value through both keys; rank differs because
    # the historical distributions are very different.
    out = calib({"a": 5.0, "b": 5.0})
    # a: buffer was [0..9]; push 5 → drop 0 → [1..9, 5]; samples <= 5 = (1,2,3,4,5,5) = 6 of 10 → 0.6
    assert math.isclose(out["a"], 0.6)
    # b: buffer was [0, 100..900]; push 5 → drop 0 → [100..900, 5]; samples <= 5 = (5) = 1 of 10 → 0.1
    assert math.isclose(out["b"], 0.1)
