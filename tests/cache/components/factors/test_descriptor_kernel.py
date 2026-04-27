"""Tests for `_descriptor_kernel`: the F1a/F1b shared single-window
implementation of jerk / dir / curv_radius / cum_disp.

Synthetic regimes per docs §3.2 — the kernel must produce the expected
relative ordering across regimes so F1a / F1b orientation contracts
hold (jerk: risky / high; dir: safe / high; geometry pair: non_monotonic).
"""

from __future__ import annotations

import math

import pytest
import torch

from openpi.cache.components.factors._descriptor_kernel import (
    all_nan_for,
    compute_descriptors,
    is_all_nan,
)

_ALL = ["jerk", "dir", "curv_radius", "cum_disp"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_window(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (pts, v, j) trio from a [W, D_act] points tensor."""
    v = pts[1:] - pts[:-1]
    j = pts[2:] - 2 * pts[1:-1] + pts[:-2]
    return pts, v, j


# ------------------------------------------------------------------
# Regime: smooth straight sweep (low jerk, high dir, large cum_disp)
# ------------------------------------------------------------------


def test_sweep_regime_low_jerk_high_dir():
    # Linear motion in 2D active subspace: pts[t] = t * direction
    direction = torch.tensor([1.0, 0.5])
    pts = torch.stack([t * direction for t in torch.linspace(0.0, 4.0, 10)], dim=0)
    p, v, j = _make_window(pts)

    out = compute_descriptors(_ALL, p, v, j)

    # jerk should be 0 (perfectly linear → second diff is 0)
    assert out["jerk"] == pytest.approx(0.0, abs=1e-6)
    # dir should be 1.0 (consecutive velocities are identical → cos=1)
    assert out["dir"] == pytest.approx(1.0, abs=1e-6)
    # curv_radius and cum_disp are finite positive
    assert out["curv_radius"] > 0
    assert out["cum_disp"] > 0


# ------------------------------------------------------------------
# Regime: shake (high jerk, lower dir consistency)
# ------------------------------------------------------------------


def test_shake_regime_high_jerk_low_dir():
    # Alternating direction every step → very non-smooth
    pts = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    p, v, j = _make_window(pts)

    out_shake = compute_descriptors(_ALL, p, v, j)

    # Compare to the sweep regime: shake jerk must be strictly greater
    direction = torch.tensor([1.0, 0.0])
    pts_sweep = torch.stack([t * direction for t in torch.linspace(0.0, 5.0, 6)], dim=0)
    p2, v2, j2 = _make_window(pts_sweep)
    out_sweep = compute_descriptors(_ALL, p2, v2, j2)

    assert out_shake["jerk"] > out_sweep["jerk"]
    # Dir for alternating motion should be much lower than 1.0 (cos close to -1)
    assert out_shake["dir"] < out_sweep["dir"]


# ------------------------------------------------------------------
# Regime: stationary (zero displacement, NaN dir, zero geometry)
# ------------------------------------------------------------------


def test_stationary_regime_zero_geometry_nan_dir():
    pts = torch.zeros(6, 2)
    p, v, j = _make_window(pts)

    out = compute_descriptors(_ALL, p, v, j)

    assert out["jerk"] == pytest.approx(0.0, abs=1e-6)
    # dir: v is all zeros → cosine_similarity returns NaN; kernel propagates NaN
    assert math.isnan(out["dir"])
    assert out["curv_radius"] == pytest.approx(0.0, abs=1e-6)
    assert out["cum_disp"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------
# Regime: turn (single direction change → mid jerk, mid dir)
# ------------------------------------------------------------------


def test_turn_regime_intermediate_jerk():
    # Move +x for 4 steps, then +y for 4 steps
    pts = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
        [3.0, 1.0], [3.0, 2.0], [3.0, 3.0],
    ])
    p, v, j = _make_window(pts)

    out = compute_descriptors(_ALL, p, v, j)

    # jerk: spike at the turn, but median over time absorbs the spike
    # (per docs §3.2.1 rationale). So a single turn yields LOW jerk (median 0).
    assert out["jerk"] == pytest.approx(0.0, abs=1e-6)
    # dir: most consecutive cos pairs are 1, one pair near 0 → mean ~0.86
    assert 0.5 < out["dir"] < 1.0
    # geometry: positive
    assert out["curv_radius"] > 0
    assert out["cum_disp"] == pytest.approx(6.0, abs=1e-6)  # 3 + 3


# ------------------------------------------------------------------
# Boundary: degenerate input shapes
# ------------------------------------------------------------------


def test_single_point_window_all_nan():
    pts = torch.tensor([[1.0, 2.0]])
    v = torch.zeros(0, 2)
    j = torch.zeros(0, 2)

    out = compute_descriptors(_ALL, pts, v, j)

    for d in _ALL:
        assert math.isnan(out[d]), f"{d} should be NaN on 1-point window"


def test_empty_window_all_nan():
    pts = torch.zeros(0, 2)
    v = torch.zeros(0, 2)
    j = torch.zeros(0, 2)

    out = compute_descriptors(_ALL, pts, v, j)

    for d in _ALL:
        assert math.isnan(out[d]), f"{d} should be NaN on 0-point window"


def test_two_point_window_dir_nan_others_finite():
    # Two points → exactly one velocity vector → cosine_similarity needs
    # at least two velocity vectors → dir must be NaN; others should be
    # finite positives.
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    v = pts[1:] - pts[:-1]                         # [1, D]
    j = torch.zeros(0, 2)                          # no second diff

    out = compute_descriptors(_ALL, pts, v, j)

    assert math.isnan(out["jerk"])                 # j is empty → NaN
    assert math.isnan(out["dir"])                  # only one velocity → NaN
    assert out["curv_radius"] > 0
    assert out["cum_disp"] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------
# Subset descriptors
# ------------------------------------------------------------------


def test_subset_only_returns_requested_keys():
    direction = torch.tensor([1.0, 0.0])
    pts = torch.stack([t * direction for t in torch.linspace(0.0, 4.0, 6)], dim=0)
    p, v, j = _make_window(pts)

    out = compute_descriptors(["jerk", "dir"], p, v, j)
    assert set(out.keys()) == {"jerk", "dir"}


# ------------------------------------------------------------------
# Unknown descriptor name
# ------------------------------------------------------------------


def test_unknown_descriptor_raises():
    pts = torch.zeros(3, 2)
    p, v, j = _make_window(pts)
    with pytest.raises(NotImplementedError, match="not implemented"):
        compute_descriptors(["dirvar"], p, v, j)


# ------------------------------------------------------------------
# all_nan_for / is_all_nan helpers
# ------------------------------------------------------------------


def test_all_nan_for_returns_nan_per_descriptor():
    out = all_nan_for(_ALL)
    assert set(out.keys()) == set(_ALL)
    for v in out.values():
        assert math.isnan(v)


def test_is_all_nan_true_on_all_nan():
    assert is_all_nan({"a": float("nan"), "b": float("nan")}) is True


def test_is_all_nan_false_on_mixed():
    assert is_all_nan({"a": float("nan"), "b": 0.0}) is False


def test_is_all_nan_false_on_empty():
    # Empty dict means "no descriptors requested", not "all signals missing"
    assert is_all_nan({}) is False
