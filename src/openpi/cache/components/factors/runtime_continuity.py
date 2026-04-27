"""F1a — Runtime continuity factors.

Two factors share a single algorithm base:
  - F1a-A (`f1a_a`): RuntimeContinuityAction — measures continuity
    between the candidate's first action and the recently executed
    action history, capturing discontinuities cache retrieval would
    introduce that an inference run would not.
  - F1a-T (`f1a_t`): RuntimeContinuityState — measures continuity
    between candidate-side state (winner + walk_next) and recent state
    history; explores whether velocity-level state continuity carries
    independent signal from the position-level field similarity already
    used during search.

The two factors are thin subclasses of `_RuntimeContinuityBase`. The
class identity (set by `@register`) determines the source — `source` is
NOT a YAML parameter. Only the class-level attributes change between
subclasses; algorithm logic lives entirely on the base.

B0 ships the metadata layer (capability flags, describe, __init__,
@register). The `extract` body raises NotImplementedError until B1
lands the algorithm.

Coupling map:
  DEPENDS ON:  components/factors/base.py (OnlineExtractor protocol),
               components/factors/source_window.py (_DESCRIPTOR_ORIENTATIONS)
  CONSUMED BY: CompositeJudge via the factor registry
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openpi.cache.components.factors.registry import register

if TYPE_CHECKING:
    import torch

    from openpi.cache.components.factors.base import HistoryView, LibraryStats
    from openpi.cache.components.payload_view import PayloadView
    from openpi.cache.storage_types import SearchResultLite


class _RuntimeContinuityBase:
    """Shared algorithm base for F1a-A / F1a-T.

    Subclasses set:
      - source: "action" | "state"            (semantic source of the
                                              data this factor consumes)
      - key_initial: "a" | "t"                (single-letter namespace for
                                              the descriptor keys this
                                              factor writes; matches the
                                              registry name suffix:
                                              `f1a_a` -> "a",
                                              `f1a_t` -> "t". Decoupled
                                              from `source` so YAML
                                              configs that reference the
                                              registered factor name find
                                              keys under the same
                                              namespace.)
      - requires_chain_walk: bool             (state needs view.walk_next)
    """

    # ---- Class-level capability flags (read by validator) ----
    source: str           # set per subclass
    key_initial: str      # set per subclass
    requires_library_stats: bool = True   # F1a uses z-score per library sigma
    requires_chain_walk: bool             # set per subclass
    required_top_k: int = 0

    def __init__(
        self,
        window_k: int,
        descriptors: list[str],
        library_stats: "LibraryStats",
    ) -> None:
        self._window_k = window_k
        self._descriptors = list(descriptors)
        self._library_stats = library_stats
        # Instance-level orientation map computed via classmethod so
        # validator (which calls describe alone) and runtime stay in sync.
        self.descriptor_orientations: dict[str, str] = self.__class__.describe(
            {"window_k": window_k, "descriptors": self._descriptors}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        """Return key->orientation for the keys this instance produces.

        F1a keys are unsuffixed (single window per call): each descriptor
        maps to one key `f1a_<key_initial>_<descriptor>`, where
        `key_initial` matches the registry name suffix ("a" for action,
        "t" for state/trajectory). Orientation is shared with the F1b
        descriptor table to keep the four core descriptors (jerk / dir /
        curv_radius / cum_disp) consistent across the two factor
        families.
        """
        # Lazy import avoids module-load circularity with source_window.
        from openpi.cache.components.factors.source_window import (
            _DESCRIPTOR_ORIENTATIONS,
        )

        prefix = f"f1a_{cls.key_initial}"   # "f1a_a" or "f1a_t"
        out: dict[str, str] = {}
        for d in params["descriptors"]:
            if d not in _DESCRIPTOR_ORIENTATIONS:
                raise ValueError(
                    f"Unknown F1a descriptor {d!r}; known: "
                    f"{sorted(_DESCRIPTOR_ORIENTATIONS)}"
                )
            out[f"{prefix}_{d}"] = _DESCRIPTOR_ORIENTATIONS[d]
        return out

    def extract(
        self,
        results: list["SearchResultLite"],
        view: "PayloadView",
        history: "HistoryView",
        cached_data: dict[str, "torch.Tensor"],
    ) -> dict[str, float]:
        # B1 ships the algorithm: jerk / dir / curv_radius / cum_disp on
        # the (history-side, candidate-side) splice window, z-scored by
        # library sigma in the active subspace.
        raise NotImplementedError(
            "RuntimeContinuity.extract: B1 algorithm"
        )


@register("f1a_a")
class RuntimeContinuityAction(_RuntimeContinuityBase):
    """F1a-A: continuity between recent executed actions and the
    candidate's first action. Reads `payload.action_chunk[0]` for the
    winner plus `history.actions` from the orchestrator.
    """

    source = "action"
    key_initial = "a"
    requires_chain_walk = False


@register("f1a_t")
class RuntimeContinuityState(_RuntimeContinuityBase):
    """F1a-T: continuity between recent observed state and candidate-side
    state walk. Reads winner `query_keys['robot_state']` plus
    `view.walk_next(winner_id, k)` for downstream entry states; also
    consumes `history.states`.
    """

    source = "state"
    key_initial = "t"
    requires_chain_walk = True
