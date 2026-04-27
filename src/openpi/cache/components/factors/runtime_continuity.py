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

import torch

from openpi.cache.components.factors._descriptor_kernel import (
    all_nan_for,
    compute_descriptors,
)
from openpi.cache.components.factors.registry import register

if TYPE_CHECKING:
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

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def extract(
        self,
        results: list["SearchResultLite"],
        view: "PayloadView",
        history: "HistoryView",
        cached_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        # Empty candidate set — preserve sentinel; CompositeJudge collapses
        # to MISS via its empty-results fast path, but we still owe an
        # all-NaN dict so the key contract assertion sees the declared keys.
        if not results:
            return all_nan_for(list(self.descriptor_orientations.keys()))

        K = self._window_k

        # State-side fail-safe (a/b): empty state library OR zero active
        # mask → bail before any state key access or z-score division.
        # Action-side has the same DOF (pi05 32-dim) but action_sigma is
        # always populated when this code runs (action_chunk is required
        # on every entry, so library_stats.action_sigma can't be empty);
        # we still keep the symmetric check for defensive parity.
        sigma, active_mask = self._select_library_arrays()
        if sigma.numel() == 0 or bool(active_mask.sum() == 0):
            return all_nan_for(list(self.descriptor_orientations.keys()))

        # Build the splice sequence per docs §3.3 (length depends on
        # source). Returns None if any boundary / per-entry presence
        # check fails — caller emits all-NaN.
        try:
            seq = self._build_splice_sequence(results[0].id, view, history, cached_data, K)
        except NotImplementedError:
            # F1a-T fork policy hit (B0 plan §2.3.2). Treat as boundary
            # and emit all-NaN rather than escalating to caller.
            return all_nan_for(list(self.descriptor_orientations.keys()))
        if seq is None:
            return all_nan_for(list(self.descriptor_orientations.keys()))

        # z-score then restrict to active subspace. eps mirrors the
        # `active_eps_*` value used at LibraryStats build time so a
        # below-eps sigma never makes it into the divisor.
        eps = self._library_eps()
        seq_norm = seq / sigma.clamp_min(eps)               # [L, D]
        pts = seq_norm[..., active_mask]                    # [L, D_act]
        if pts.shape[-1] == 0:
            return all_nan_for(list(self.descriptor_orientations.keys()))
        v = pts[1:] - pts[:-1]                              # [L-1, D_act]
        j = pts[2:] - 2 * pts[1:-1] + pts[:-2]              # [L-2, D_act]

        raw = compute_descriptors(self._descriptors, pts, v, j)
        # Re-key under the namespaced descriptor_orientations contract:
        # `f1a_<key_initial>_<descriptor>`.
        prefix = f"f1a_{self.key_initial}"
        return {f"{prefix}_{d}": raw[d] for d in self._descriptors}

    # ------------------------------------------------------------------
    # Internal: library_stats accessors (per source)
    # ------------------------------------------------------------------

    def _select_library_arrays(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.source == "action":
            return self._library_stats.action_sigma, self._library_stats.action_active_mask
        if self.source == "state":
            return self._library_stats.state_sigma, self._library_stats.state_active_mask
        raise ValueError(f"Unknown source: {self.source}")

    def _library_eps(self) -> float:
        # Mirror the LibraryStats default; not exposed via YAML override
        # in B1+B2 (see plan §4.2).
        return 0.01

    # ------------------------------------------------------------------
    # Internal: splice sequence per source (per docs §3.3)
    # ------------------------------------------------------------------

    def _build_splice_sequence(
        self,
        winner_id: str,
        view: "PayloadView",
        history: "HistoryView",
        cached_data: dict[str, torch.Tensor],
        K: int,
    ) -> torch.Tensor | None:
        if self.source == "action":
            return self._build_action_splice(winner_id, view, history, K)
        if self.source == "state":
            return self._build_state_splice(winner_id, view, history, cached_data, K)
        raise ValueError(f"Unknown source: {self.source}")

    def _build_action_splice(
        self,
        winner_id: str,
        view: "PayloadView",
        history: "HistoryView",
        K: int,
    ) -> torch.Tensor | None:
        # docs §3.3.1: splice_a = [a_{-K}, ..., a_{-1}, a_0^cand]  ∈ R^{(K+1) x A}
        # Boundary: episode early when len(history.actions) < K → all-NaN.
        if len(history.actions) < K:
            return None
        winner = view.get(winner_id)
        # winner.action_chunk shape is [chunk_len, A]; first step is the
        # candidate's recommended next action.
        cand = torch.as_tensor(winner.action_chunk[0], dtype=torch.float32)
        hist_tail = history.actions[-K:]
        return torch.stack(
            [torch.as_tensor(a, dtype=torch.float32) for a in hist_tail] + [cand],
            dim=0,
        )                                                   # [K+1, A]

    def _build_state_splice(
        self,
        winner_id: str,
        view: "PayloadView",
        history: "HistoryView",
        cached_data: dict[str, torch.Tensor],
        K: int,
    ) -> torch.Tensor | None:
        # docs §3.3.2: splice_s = [s_{-K}..s_{-1}, s_0^cand, s_1..s_K^cand]
        #              ∈ R^{(2K+1) x S}
        # Boundary handling and state-presence guards are bundled here so
        # extract() doesn't have to duplicate the logic for state.
        if len(history.states) < K:
            return None
        winner_entry = view.get_entry(winner_id)
        winner_state = winner_entry.query_keys.get("robot_state")
        if winner_state is None:
            return None
        # walk_next may raise NotImplementedError on real forks (TRAJECTORY
        # policy not implemented for branching chains in B0); caller
        # catches it and emits all-NaN.
        forward_entries = view.walk_next(winner_id, k=K)
        if len(forward_entries) < K:
            # Trajectory boundary — chain ran out before K forward steps.
            return None
        forward_states_raw: list = []
        for e in forward_entries:
            rs = e.query_keys.get("robot_state")
            if rs is None:
                return None
            forward_states_raw.append(rs)

        hist_tail = history.states[-K:]
        parts: list[torch.Tensor] = (
            [torch.as_tensor(s, dtype=torch.float32) for s in hist_tail]
            + [torch.as_tensor(winner_state, dtype=torch.float32)]
            + [torch.as_tensor(s, dtype=torch.float32) for s in forward_states_raw]
        )
        return torch.stack(parts, dim=0)                    # [2K+1, S]


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
