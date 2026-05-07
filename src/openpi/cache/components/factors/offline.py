"""Layer 2 — 8 offline factors (B2 of the verdict-factor refactor).

Each offline factor:
  - Online path: reads pre-computed values from ``winner.payload.factors``
    via the PayloadView. No descriptor math at verdict time.
  - Offline path: implements ``OfflineWriter.compute_for_episode``.
    Builds the per-episode chain sequence (action_chunk[0] for action
    channel, query_keys["robot_state"] for state channel), z-scores it
    via the supplied ``LibraryStats``, then computes a sliding-window
    descriptor at every entry × every window.
  - Multi-window via the ``windows: [{past, future}, ...]`` construction
    param; each window emits one keyed scalar per entry.

Naming convention: ``<descriptor>_offline_<channel>``. Key template:
``<factor_name>__p<P>_f<F>``.

Plan §6.11 #5: each offline factor implements the online path itself
(framework does not auto-read ``payload.factors``); this keeps Factor
subclasses self-contained.

Coupling map:
  DEPENDS ON: components/factors/base.py (Factor + OfflineWriter Protocols,
                                          LibraryStats, FactorContext),
              components/factors/_descriptor_kernel.py.
  CONSUMED BY: components/composite_judge.py (Layer 2 instances),
               exp/common/factor_postprocess.py
                   (enrich_artifact_with_factors collects OfflineWriter list).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from openpi.cache.components.factors._descriptor_kernel import (
    _DESCRIPTOR_ORIENTATIONS,
    all_nan_for,
    compute_descriptors,
)
from openpi.cache.components.factors.base import (
    FactorContext,
    LibraryStats,
    normalize_windows as _normalize_windows,
)
from openpi.cache.components.factors.registry import register

if TYPE_CHECKING:
    from openpi.cache.storage_types import CacheEntry


# ----------------------------------------------------------------------
# Shared offline base
# ----------------------------------------------------------------------


class _OfflineFactorBase:
    """Common machinery for the 8 offline factors.

    Subclasses set:
      - descriptor: "jerk" / "direction" / "dispersion" / "path_length"
      - channel:    "action" or "state"

    Both subclasses share:
      - ``requires_chain_walk = False`` (online path only needs winner; chain
        walking is done at offline-build time, not at verdict time).
      - ``required_top_k = 0``.
      - ``describe(params)`` classmethod.
      - ``extract(ctx)`` — read pre-computed factor values from
        ``winner.payload.factors``.
      - ``required_payload_fields()`` and ``compute_for_episode(entries,
        library_stats)`` for the OfflineWriter half of the surface.
    """

    descriptor: str = ""    # set per subclass
    channel:    str = ""    # set per subclass — "action" or "state"
    requires_chain_walk: bool = False
    required_top_k: int = 0

    def __init__(self, *, windows) -> None:
        self._windows = _normalize_windows(windows)
        if not self._windows:
            raise ValueError(f"{type(self).__name__}: at least one window required")
        self.descriptor_orientations: dict[str, str] = self.__class__.describe(
            {"windows": self._windows}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        if not cls.descriptor or not cls.channel:
            raise TypeError(
                f"{cls.__name__} must set `descriptor` and `channel` class attrs"
            )
        ori = _DESCRIPTOR_ORIENTATIONS[cls.descriptor]
        prefix = f"{cls.descriptor}_offline_{cls.channel}"
        return {
            f"{prefix}__p{p}_f{f}": ori
            for (p, f) in _normalize_windows(params["windows"])
        }

    # ------------------------------------------------------------------
    # Online surface — read from winner.payload.factors
    # ------------------------------------------------------------------

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        keys = list(self.descriptor_orientations.keys())
        if not ctx.results:
            return all_nan_for(keys)
        winner_payload = ctx.view.get(ctx.results[0].id)
        if winner_payload.factors is None:
            return all_nan_for(keys)
        # Read each declared key from payload.factors; missing key → NaN.
        return {
            k: float(winner_payload.factors.get(k, float("nan"))) for k in keys
        }

    # ------------------------------------------------------------------
    # OfflineWriter surface — episode chain → per-entry factor dict
    # ------------------------------------------------------------------

    def required_payload_fields(self) -> set[str]:
        # Current 4 descriptors all read from existing schema (action_chunk[0]
        # or query_keys["robot_state"]); no extra raw payload tensors needed.
        return set()

    def compute_for_episode(
        self,
        entries: list["CacheEntry"],
        library_stats: LibraryStats,
    ) -> list[dict[str, float]]:
        keys = list(self.descriptor_orientations.keys())
        T = len(entries)
        if T == 0:
            return []

        # Channel-specific: pick the right sigma + active mask + extractor.
        sigma, active_mask = self._select_library_arrays(library_stats)
        # Bail-fast guards: empty library or zero active mask → all entries NaN.
        if sigma.numel() == 0 or bool(active_mask.sum() == 0):
            return [all_nan_for(keys) for _ in entries]

        # State-channel additionally guards every-entry presence: any missing
        # robot_state entry corrupts adjacent windows, so emit NaN for the
        # whole episode rather than partial data.
        if self.channel == "state":
            for e in entries:
                if e.query_keys.get("robot_state") is None:
                    return [all_nan_for(keys) for _ in entries]

        seq = self._extract_episode_seq(entries)              # [T, D]
        denom = sigma.clamp_min(self._eps())
        seq_norm = seq / denom                                # [T, D]
        pts_full = seq_norm[..., active_mask]                 # [T, D_active]
        if pts_full.shape[-1] == 0:
            return [all_nan_for(keys) for _ in entries]

        out: list[dict[str, float]] = []
        for k_idx in range(T):
            per_entry: dict[str, float] = {}
            for (P, F) in self._windows:
                lo = k_idx - P
                hi = k_idx + F                                # inclusive end
                if lo < 0 or hi >= T:
                    # Boundary: window pokes past the episode → NaN.
                    per_entry[self._key(P, F)] = float("nan")
                    continue
                pts = pts_full[lo:hi + 1]                     # [W, D_active]
                v = pts[1:] - pts[:-1]                        # [W-1, D_active]
                j = pts[2:] - 2 * pts[1:-1] + pts[:-2]        # [W-2, D_active]
                raw = compute_descriptors([self.descriptor], pts, v, j)
                per_entry[self._key(P, F)] = raw[self.descriptor]
            out.append(per_entry)
        return out

    # ------------------------------------------------------------------
    # Internal hooks (per channel)
    # ------------------------------------------------------------------

    def _select_library_arrays(
        self, library_stats: LibraryStats
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.channel == "action":
            return library_stats.action_sigma, library_stats.action_active_mask
        if self.channel == "state":
            return library_stats.state_sigma, library_stats.state_active_mask
        raise ValueError(f"Unknown channel: {self.channel}")

    def _extract_episode_seq(self, entries: list["CacheEntry"]) -> torch.Tensor:
        if self.channel == "action":
            return torch.stack(
                [
                    torch.as_tensor(e.payload.action_chunk[0], dtype=torch.float32)
                    for e in entries
                ],
                dim=0,
            )
        if self.channel == "state":
            return torch.stack(
                [
                    torch.as_tensor(e.query_keys["robot_state"], dtype=torch.float32)
                    for e in entries
                ],
                dim=0,
            )
        raise ValueError(f"Unknown channel: {self.channel}")

    @staticmethod
    def _eps() -> float:
        # Mirror the LibraryStats default; not exposed to YAML for B1+B6 to
        # keep the calibration contract consistent with the online path.
        return 0.01

    def _key(self, P: int, F: int) -> str:
        return f"{self.descriptor}_offline_{self.channel}__p{P}_f{F}"


# ----------------------------------------------------------------------
# Action / state channel mixins
# ----------------------------------------------------------------------


class _OfflineActionBase(_OfflineFactorBase):
    channel: str = "action"


class _OfflineStateBase(_OfflineFactorBase):
    channel: str = "state"


# ----------------------------------------------------------------------
# 8 thin subclasses: 4 descriptors × 2 channels
# ----------------------------------------------------------------------


@register("jerk_offline_action")
class JerkOfflineAction(_OfflineActionBase):
    descriptor = "jerk"


@register("jerk_offline_state")
class JerkOfflineState(_OfflineStateBase):
    descriptor = "jerk"


@register("direction_offline_action")
class DirectionOfflineAction(_OfflineActionBase):
    descriptor = "direction"


@register("direction_offline_state")
class DirectionOfflineState(_OfflineStateBase):
    descriptor = "direction"


@register("dispersion_offline_action")
class DispersionOfflineAction(_OfflineActionBase):
    descriptor = "dispersion"


@register("dispersion_offline_state")
class DispersionOfflineState(_OfflineStateBase):
    descriptor = "dispersion"


@register("path_length_offline_action")
class PathLengthOfflineAction(_OfflineActionBase):
    descriptor = "path_length"


@register("path_length_offline_state")
class PathLengthOfflineState(_OfflineStateBase):
    descriptor = "path_length"
