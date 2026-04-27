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

from typing import TYPE_CHECKING, Iterable

from openpi.cache.components.factors.registry import register

if TYPE_CHECKING:
    import torch

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
        library_stats: "LibraryStats",
    ) -> None:
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
        cached_data: dict[str, "torch.Tensor"],
    ) -> dict[str, float]:
        # B2 reads keys from `view.get(winner_id).factors` and propagates
        # NaN for missing keys per the verdict-factor NaN handling rule.
        raise NotImplementedError(
            "SourceWindowSmoothness.extract: B2 algorithm"
        )

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
        # B2 ships z-score / active_mask / per-window descriptor pass per
        # `verdict_factor_judge.log.md` §2.8.4.
        raise NotImplementedError(
            "SourceWindowSmoothness.compute_for_episode: B2 algorithm"
        )


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
