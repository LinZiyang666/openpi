"""Unit tests for Layer 1 ``ZScoreNormalization``."""

from __future__ import annotations

import math

import pytest
import torch

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.components.factors.normalization import (
    Normalization,
    ZScoreNormalization,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_library_stats(
    action_sigma_vals: list[float],
    state_sigma_vals: list[float],
    *,
    eps: float = 0.01,
) -> LibraryStats:
    a_sigma = torch.tensor(action_sigma_vals, dtype=torch.float32)
    s_sigma = torch.tensor(state_sigma_vals, dtype=torch.float32)
    return LibraryStats(
        action_sigma=a_sigma,
        action_active_mask=a_sigma >= eps,
        state_sigma=s_sigma,
        state_active_mask=s_sigma >= eps,
    )


# ----------------------------------------------------------------------
# Protocol conformance
# ----------------------------------------------------------------------


def test_zscore_satisfies_normalization_protocol() -> None:
    ls = _make_library_stats([1.0], [1.0])
    inst = ZScoreNormalization(ls)
    assert isinstance(inst, Normalization)


# ----------------------------------------------------------------------
# Action channel
# ----------------------------------------------------------------------


def test_normalize_action_divides_by_sigma_and_drops_padded_dofs() -> None:
    # Two active DOFs (sigma 1.0, 2.0) + one padded DOF (sigma 0.0).
    ls = _make_library_stats([1.0, 2.0, 0.0], [1.0])
    inst = ZScoreNormalization(ls)

    raw = torch.tensor([4.0, 8.0, 99.0], dtype=torch.float32)
    out = inst.normalize_action(raw)

    assert out.shape == (2,), f"expected zero padded DOF dropped, got shape {tuple(out.shape)}"
    assert torch.allclose(out, torch.tensor([4.0, 4.0]))   # 4/1, 8/2


def test_normalize_action_handles_2d_input() -> None:
    ls = _make_library_stats([2.0, 0.0, 4.0], [1.0])
    inst = ZScoreNormalization(ls)

    raw = torch.tensor(
        [
            [2.0, 99.0, 8.0],
            [4.0, 99.0, 16.0],
        ],
        dtype=torch.float32,
    )
    out = inst.normalize_action(raw)
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0], [2.0, 4.0]]))


def test_normalize_action_clamps_below_eps_sigma() -> None:
    # sigma exactly at eps stays in active mask; sigma below eps dropped.
    ls = _make_library_stats([0.005, 0.01, 0.5], [1.0])
    inst = ZScoreNormalization(ls)

    raw = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    out = inst.normalize_action(raw)
    assert out.shape == (2,)
    # Active dims are sigma=0.01 and sigma=0.5 → outputs 100 and 2.
    assert math.isclose(float(out[0]), 100.0)
    assert math.isclose(float(out[1]), 2.0)


# ----------------------------------------------------------------------
# State channel
# ----------------------------------------------------------------------


def test_normalize_state_uses_state_sigma() -> None:
    ls = _make_library_stats([99.0], [1.0, 0.5, 0.0])
    inst = ZScoreNormalization(ls)

    raw = torch.tensor([2.0, 1.0, 7.0], dtype=torch.float32)
    out = inst.normalize_state(raw)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([2.0, 2.0]))


def test_empty_state_sigma_returns_zero_width() -> None:
    """Plan §6.3 / base.py LibraryStats fallback: empty state library yields
    zero-length sigma + mask. ``normalize_state`` must surface that as a
    zero-width slice rather than raising — factors detect it and emit NaN."""
    a_sigma = torch.tensor([1.0], dtype=torch.float32)
    ls = LibraryStats(
        action_sigma=a_sigma,
        action_active_mask=torch.tensor([True]),
        state_sigma=torch.zeros(0, dtype=torch.float32),
        state_active_mask=torch.zeros(0, dtype=torch.bool),
    )
    inst = ZScoreNormalization(ls)

    raw = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    out = inst.normalize_state(raw)
    assert out.shape[-1] == 0
    assert out.dtype == torch.float32


def test_zero_active_mask_returns_zero_width() -> None:
    """All-padded channel (every DOF below eps) → empty active subspace."""
    ls = _make_library_stats([0.005, 0.001], [1.0])  # both below eps
    inst = ZScoreNormalization(ls)

    raw = torch.tensor([1.0, 1.0], dtype=torch.float32)
    out = inst.normalize_action(raw)
    assert out.shape == (0,)


# ----------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------


def test_construction_rejects_none_library_stats() -> None:
    with pytest.raises(ValueError, match="non-None library_stats"):
        ZScoreNormalization(None)  # type: ignore[arg-type]


def test_construction_rejects_non_positive_eps() -> None:
    ls = _make_library_stats([1.0], [1.0])
    with pytest.raises(ValueError, match="eps must be > 0"):
        ZScoreNormalization(ls, eps=0.0)
    with pytest.raises(ValueError, match="eps must be > 0"):
        ZScoreNormalization(ls, eps=-0.1)


def test_construction_rejects_library_stats_missing_field() -> None:
    """Plan-mandated fail-fast: a malformed library_stats (missing field) must
    not silently work. AttributeError at __init__ assignment is the contract."""

    class _IncompleteStats:
        # Missing: state_sigma, state_active_mask
        action_sigma = torch.tensor([1.0], dtype=torch.float32)
        action_active_mask = torch.tensor([True])

    with pytest.raises(AttributeError):
        ZScoreNormalization(_IncompleteStats())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Numpy / non-tensor input compatibility
# ----------------------------------------------------------------------


def test_normalize_accepts_numpy_arrays() -> None:
    """Real LibraryStats carries torch tensors but factor inputs may be numpy
    (e.g. raw entries during artifact build). ``torch.as_tensor`` bridges that."""
    import numpy as np

    ls = _make_library_stats([2.0, 4.0], [1.0])
    inst = ZScoreNormalization(ls)

    raw = np.array([4.0, 8.0], dtype=np.float32)
    out = inst.normalize_action(raw)  # type: ignore[arg-type]
    assert torch.allclose(out, torch.tensor([2.0, 2.0]))
