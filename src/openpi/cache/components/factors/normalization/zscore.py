"""ZScoreNormalization — default Layer 1 implementation.

Algorithm: ``raw / sigma.clamp_min(eps)`` followed by active-mask selection
along the trailing DOF axis. Sigma + active mask come from ``LibraryStats``;
``eps`` mirrors the ``LibraryStats.compute_from_entries`` active threshold
(0.01) so a below-eps sigma never enters the divisor.

Coupling map:
  DEPENDS ON:  components/factors/base.py (LibraryStats),
               components/factors/normalization/base.py (Normalization Protocol)
  CONSUMED BY: factors/online.py + factors/offline.py via FactorContext.normalization
"""

from __future__ import annotations

import torch

from openpi.cache.components.factors.base import LibraryStats

# ----------------------------------------------------------------------
# Tunable
# ----------------------------------------------------------------------

# Mirror of the active-mask epsilon used by ``LibraryStats.compute_from_entries``;
# kept private so it can evolve without touching the yaml schema.
_DEFAULT_EPS: float = 0.01


# ----------------------------------------------------------------------
# Implementation
# ----------------------------------------------------------------------


class ZScoreNormalization:
    """Per-DOF z-score with active-mask restriction.

    Constructor fails fast if ``library_stats`` is missing the action or
    state sigma / mask fields (zero-length tensors are tolerated and
    surface as zero-width outputs at call time, signalling to factors
    that there is no valid subspace for that channel).
    """

    def __init__(self, library_stats: LibraryStats, *, eps: float = _DEFAULT_EPS) -> None:
        if library_stats is None:
            raise ValueError("ZScoreNormalization requires a non-None library_stats")
        # The four fields are required attribute names on LibraryStats. A
        # malformed object (e.g. missing one of them) surfaces as AttributeError
        # at this assignment, which is the desired fail-fast.
        self._action_sigma: torch.Tensor = library_stats.action_sigma
        self._action_active_mask: torch.Tensor = library_stats.action_active_mask
        self._state_sigma: torch.Tensor = library_stats.state_sigma
        self._state_active_mask: torch.Tensor = library_stats.state_active_mask
        if eps <= 0:
            raise ValueError(f"ZScoreNormalization eps must be > 0, got {eps}")
        self._eps = float(eps)

    # ------------------------------------------------------------------
    # Public per-channel entrypoints
    # ------------------------------------------------------------------

    def normalize_action(self, raw: torch.Tensor) -> torch.Tensor:
        return self._zscore(raw, self._action_sigma, self._action_active_mask)

    def normalize_state(self, raw: torch.Tensor) -> torch.Tensor:
        return self._zscore(raw, self._state_sigma, self._state_active_mask)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _zscore(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Cast input to float32 on CPU to match the LibraryStats dtype/device.
        # Callers may pass numpy arrays via torch.as_tensor; tolerate that here.
        x = torch.as_tensor(raw, dtype=torch.float32)
        if sigma.numel() == 0:
            # Channel has no library data (e.g. state missing across the
            # entire artifact). Return a zero-width slice along the last
            # dim so the calling factor treats it as a NaN signal.
            return x.new_zeros(*x.shape[:-1], 0)
        denom = sigma.clamp_min(self._eps)
        normed = x / denom                                    # [..., D]
        return normed[..., active_mask]                       # [..., D_active]
