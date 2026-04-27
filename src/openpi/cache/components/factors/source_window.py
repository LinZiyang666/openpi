"""F1b — Source-window smoothness factors.

Two factors share a single algorithm base:
  - F1b-A (`f1b_a`): SourceWindowSmoothnessAction — descriptors over
    on-chain action_chunk[0] sequences, normalized by library
    `action_sigma` + `action_active_mask`.
  - F1b-T (`f1b_t`): SourceWindowSmoothnessState — same descriptors over
    on-chain `query_keys['robot_state']` sequences, normalized by
    library `state_sigma` + `state_active_mask`.

Each factor implements both OnlineExtractor (reads pre-computed values
from `payload.factors` at verdict time) and OfflineWriter (computes the
descriptors per episode at artifact build / Orchestrator episode-end).

Window representation contract (per plan §8.1.6 Round 11 Item 1):
  - YAML / FactorConfig.params: `windows: [{past: int, future: int}, ...]`
  - Internal canonical: `list[tuple[int, int]]`
  Both forms feed through `_normalize_windows`; never unpack the raw
  param dict in __init__ / describe / validator.

B0 ships the metadata layer + key-template logic; the `extract` and
`compute_for_episode` bodies raise NotImplementedError until B2 lands
the descriptor algorithm.

Coupling map:
  DEPENDS ON:  components/factors/base.py (OnlineExtractor / OfflineWriter
               protocols, LibraryStats)
  CONSUMED BY: CompositeJudge (via registry); Orchestrator.episode-end
               write path / artifact-build helper (via registry)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import torch

from openpi.cache.components.factors._descriptor_kernel import (
    all_nan_for,
    compute_descriptors,
)
from openpi.cache.components.factors.registry import register

if TYPE_CHECKING:
    from openpi.cache.components.factors.base import HistoryView, LibraryStats
    from openpi.cache.components.payload_view import PayloadView
    from openpi.cache.storage_types import CacheEntry, SearchResultLite


# ------------------------------------------------------------------
# Descriptor orientation table (shared with F1a)
# ------------------------------------------------------------------


_DESCRIPTOR_ORIENTATIONS: dict[str, str] = {
    "jerk":        "risky",
    "dir":         "safe",
    "curv_radius": "non_monotonic",
    "cum_disp":    "non_monotonic",
    # B2+ extension slots (currently unimplemented):
    # "dirvar":   "risky",
    # "path":     "safe",
    # "freq":     "risky",
    # "autocorr": "safe",
}


# ------------------------------------------------------------------
# Window normalization helper
# ------------------------------------------------------------------


def _normalize_windows(windows: Iterable) -> list[tuple[int, int]]:
    """Normalize YAML / params windows to `list[tuple[int, int]]`.

    Accepts:
      - list[dict{"past": int, "future": int}]  (YAML / FactorConfig shape)
      - list[tuple[int, int]] / list[list[int]] (already-normalized form)

    Used in three places — keep them in sync:
      - `_SourceWindowSmoothnessBase.__init__`
      - `_SourceWindowSmoothnessBase.describe` (classmethod, validator path)
      - `validate_cache_config` (composite directions coverage check)
    """
    out: list[tuple[int, int]] = []
    for w in windows:
        if isinstance(w, dict):
            out.append((int(w["past"]), int(w["future"])))
        else:
            p, f = w
            out.append((int(p), int(f)))
    return out


# ------------------------------------------------------------------
# Base + thin subclasses
# ------------------------------------------------------------------


class _SourceWindowSmoothnessBase:
    """Shared algorithm base for F1b-A / F1b-T.

    Subclasses set:
      - source: "action" | "state"   (semantic source of the data this
                                     factor consumes; NOT a YAML param)
      - key_initial: "a" | "t"       (single-letter namespace for the
                                     descriptor keys this factor writes;
                                     matches the registry name suffix
                                     `f1b_a` / `f1b_t` so YAML configs
                                     that reference the registered factor
                                     name find descriptor keys under the
                                     same namespace)
    """

    # ---- Class-level capability flags (read by validator) ----
    source: str          # set per subclass
    key_initial: str     # set per subclass
    requires_library_stats: bool = True
    requires_chain_walk: bool = False
    required_top_k: int = 0

    def __init__(
        self,
        windows,
        descriptors: list[str],
        active_eps: float,
        library_stats: Optional["LibraryStats"] = None,
    ) -> None:
        # `library_stats` is Optional so the offline build path can
        # construct writers via `registry.build(...)` without yet having
        # the LibraryStats instance. Online (CompositeJudge) construction
        # always supplies it through `_build_judge`'s capability-flag
        # injection. OfflineWriter callers pass library_stats per-call
        # via `compute_for_episode(entries, library_stats)`.
        self._windows: list[tuple[int, int]] = _normalize_windows(windows)
        self._descriptors = list(descriptors)
        self._active_eps = active_eps
        self._library_stats = library_stats
        # describe takes the normalized window form so validator-time and
        # runtime-time keys agree exactly.
        self.descriptor_orientations: dict[str, str] = self.__class__.describe(
            {"windows": self._windows, "descriptors": self._descriptors}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        """Return key->orientation for keys an instance with `params`
        will produce. Keys follow the template
        `f1b_<key_initial>_<descriptor>__p<past>_f<future>`, where
        `key_initial` matches the registry name suffix ("a" / "t").
        """
        prefix = f"f1b_{cls.key_initial}"   # "f1b_a" or "f1b_t"
        windows = _normalize_windows(params["windows"])
        out: dict[str, str] = {}
        for d in params["descriptors"]:
            if d not in _DESCRIPTOR_ORIENTATIONS:
                raise ValueError(
                    f"Unknown F1b descriptor {d!r}; known: "
                    f"{sorted(_DESCRIPTOR_ORIENTATIONS)}"
                )
            for (p, f) in windows:
                out[f"{prefix}_{d}__p{p}_f{f}"] = _DESCRIPTOR_ORIENTATIONS[d]
        return out

    # ---- OnlineExtractor surface ----

    def extract(
        self,
        results: list["SearchResultLite"],
        view: "PayloadView",
        history: "HistoryView",
        cached_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        # F1b online path is read-only: descriptor values are pre-computed
        # by the offline path and live in `payload.factors`. Missing
        # keys (legacy artifact / writer skipped) → NaN.
        keys = list(self.descriptor_orientations.keys())
        if not results:
            return all_nan_for(keys)
        winner = view.get(results[0].id)
        factors = winner.factors
        if factors is None:
            return all_nan_for(keys)
        return {k: float(factors.get(k, float("nan"))) for k in keys}

    # ---- OfflineWriter surface ----

    def required_payload_fields(self) -> set[str]:
        # Current descriptor set reads from existing schema (action_chunk
        # + query_keys['robot_state']); no extra raw payload tensors.
        return set()

    def compute_for_episode(
        self,
        entries: list["CacheEntry"],
        library_stats: "LibraryStats",
    ) -> list[dict[str, float]]:
        # Per docs §3.4: per-entry sliding-window descriptor pass over the
        # full episode chain. Returns one factor dict per entry; entries
        # outside any window get all-NaN per-window descriptors.
        keys = list(self.descriptor_orientations.keys())
        T = len(entries)
        if T == 0:
            return []

        sigma, active_mask = self._select_library_arrays(library_stats)

        # State-side fail-safe (a / b): empty library or zero active mask
        # → bail before any state key access / z-score division. Returns
        # one all-NaN dict per entry so the writer's schema (per-entry
        # factor map) stays consistent with online reads.
        if sigma.numel() == 0 or bool(active_mask.sum() == 0):
            return [all_nan_for(keys) for _ in entries]

        # State-side per-entry presence check (c): if any entry is missing
        # the state field, bail for the whole episode. Window descriptors
        # have neighbor dependencies, so dropping one entry corrupts
        # adjacent windows; cleaner to mark the whole chain NaN and let
        # the cold-start sentinel route those verdicts to MISS.
        if self.source == "state":
            for e in entries:
                if e.query_keys.get("robot_state") is None:
                    return [all_nan_for(keys) for _ in entries]

        seq = self._extract_episode_seq(entries)         # [T, D]

        # z-score then active subspace (per docs §3.4)
        eps = self._active_eps
        denom = sigma.clamp_min(eps)
        seq_norm = seq / denom                           # [T, D]
        pts_full = seq_norm[..., active_mask]            # [T, D_act]
        if pts_full.shape[-1] == 0:
            return [all_nan_for(keys) for _ in entries]

        out: list[dict[str, float]] = []
        for k_idx in range(T):
            per_entry: dict[str, float] = {}
            for (p, f) in self._windows:
                lo = k_idx - p
                hi = k_idx + f                          # inclusive end
                if lo < 0 or hi >= T:
                    # Boundary: window pokes past the episode. Per docs
                    # §3.4 boundary rule, all descriptors for this window
                    # are NaN.
                    for d in self._descriptors:
                        per_entry[self._key(d, p, f)] = float("nan")
                    continue
                pts = pts_full[lo:hi + 1]                # [W, D_act]
                v = pts[1:] - pts[:-1]                   # [W-1, D_act]
                j = pts[2:] - 2 * pts[1:-1] + pts[:-2]   # [W-2, D_act]
                raw = compute_descriptors(self._descriptors, pts, v, j)
                for d in self._descriptors:
                    per_entry[self._key(d, p, f)] = raw[d]
            out.append(per_entry)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, descriptor: str, p: int, f: int) -> str:
        return f"f1b_{self.key_initial}_{descriptor}__p{p}_f{f}"

    def _select_library_arrays(
        self, library_stats: "LibraryStats",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.source == "action":
            return library_stats.action_sigma, library_stats.action_active_mask
        if self.source == "state":
            return library_stats.state_sigma, library_stats.state_active_mask
        raise ValueError(f"Unknown source: {self.source}")

    def _extract_episode_seq(self, entries: list["CacheEntry"]) -> torch.Tensor:
        # PRECONDITION: caller already passed the state-side fail-safe and
        # per-entry presence guards above; here we can dereference safely.
        # `torch.as_tensor` bridges numpy entries (post-_detach_entries
        # subprocess output) and torch entries (in-memory tests).
        if self.source == "action":
            return torch.stack(
                [
                    torch.as_tensor(e.payload.action_chunk[0], dtype=torch.float32)
                    for e in entries
                ],
                dim=0,
            )
        if self.source == "state":
            return torch.stack(
                [
                    torch.as_tensor(e.query_keys["robot_state"], dtype=torch.float32)
                    for e in entries
                ],
                dim=0,
            )
        raise ValueError(f"Unknown source: {self.source}")


@register("f1b_a")
class SourceWindowSmoothnessAction(_SourceWindowSmoothnessBase):
    """F1b-A: action-side smoothness; on-chain `payload.action_chunk[0]`
    sequence per episode."""

    source = "action"
    key_initial = "a"


@register("f1b_t")
class SourceWindowSmoothnessState(_SourceWindowSmoothnessBase):
    """F1b-T: state-side smoothness; on-chain
    `query_keys['robot_state']` sequence per episode."""

    source = "state"
    key_initial = "t"
