"""Factor protocols and shared library-level statistics.

A factor may implement OnlineExtractor (verdict-time, on cache hit
candidates), OfflineWriter (artifact-build / episode-end, populates
`payload.factors`), or both. The two protocols are separate so a factor
can declare exactly the surface it provides; CompositeJudge consumes
OnlineExtractors only, while the artifact-build / Orchestrator
write-path collects OfflineWriters.

Coupling map:
  DEPENDS ON:  storage_types.py (CacheEntry, SearchResultLite),
               components/payload_view.py (PayloadView)
  CONSUMED BY: CompositeJudge (OnlineExtractor),
               CacheOrchestrator._build_entry_chain + offline build pkl
               tooling (OfflineWriter)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from openpi.cache.components.payload_view import PayloadView
    from openpi.cache.storage_types import CacheEntry, SearchResultLite


# ------------------------------------------------------------------
# Library-level statistics (F1b normalization basis)
# ------------------------------------------------------------------


@dataclass
class LibraryStats:
    """Per-DOF library-wide statistics for F1b normalization.

    Tensors live on CPU, float32. Computed once per artifact and stored
    on the artifact dict, or computed lazily by InMemoryBackend at
    server startup if the artifact lacks the field.

    The active masks select dimensions whose std exceeds a small epsilon
    (default 0.01). Pi0.5's 32-dim universal action / state space pads
    unused dims with zeros; without the mask, those padded dims would
    distort z-score and cosine descriptors. F1b extractors restrict
    every per-DOF computation to the active subspace.
    """

    action_sigma: "torch.Tensor"        # [action_dim] CPU float32
    action_active_mask: "torch.Tensor"  # [action_dim] bool
    state_sigma: "torch.Tensor"         # [state_dim] CPU float32
    state_active_mask: "torch.Tensor"   # [state_dim] bool

    @classmethod
    def compute_from_entries(
        cls,
        entries: list["CacheEntry"],
        active_eps_action: float = 0.01,
        active_eps_state: float = 0.01,
    ) -> "LibraryStats":
        """Stack `payload.action_chunk[0]` + `query_keys['robot_state']`
        across entries and derive per-DOF std + active masks.

        Tolerates both torch.Tensor and numpy.ndarray entry payloads
        via `torch.as_tensor` — the primary builder path runs
        `_detach_entries` inside the ProcessPool subprocess, so by the
        time enrichment runs in the main process the tensors are numpy.

        Empty entries / state-missing entries: see plan §4.2 — `state_dim`
        falls back to length 0 with a zero-length sigma + mask, and
        F1a-T / F1b-T detect `state_active_mask.sum() == 0` upfront and
        return all-NaN factor dicts.
        """
        if not entries:
            return cls(
                action_sigma=torch.zeros(0, dtype=torch.float32),
                action_active_mask=torch.zeros(0, dtype=torch.bool),
                state_sigma=torch.zeros(0, dtype=torch.float32),
                state_active_mask=torch.zeros(0, dtype=torch.bool),
            )

        # ---- Action side ----
        actions = torch.stack(
            [
                torch.as_tensor(e.payload.action_chunk[0], dtype=torch.float32)
                for e in entries
            ],
            dim=0,
        )                                                # [N, A]
        action_sigma = actions.std(dim=0, unbiased=False)
        action_active_mask = action_sigma >= active_eps_action

        # ---- State side ----
        states_list: list[torch.Tensor] = []
        for e in entries:
            rs = e.query_keys.get("robot_state")
            if rs is not None:
                states_list.append(torch.as_tensor(rs, dtype=torch.float32))

        if states_list:
            states = torch.stack(states_list, dim=0)     # [N', S]
            state_sigma = states.std(dim=0, unbiased=False)
            state_active_mask = state_sigma >= active_eps_state
        else:
            # No entry carries robot_state — placeholder zero-length
            # tensors. Down-stream state-side factors detect the empty
            # mask via the `state_active_mask.sum() == 0` guard at the
            # head of `extract` / `compute_for_episode` and emit all-NaN
            # without touching state.
            state_sigma = torch.zeros(0, dtype=torch.float32)
            state_active_mask = torch.zeros(0, dtype=torch.bool)

        return cls(
            action_sigma=action_sigma,
            action_active_mask=action_active_mask,
            state_sigma=state_sigma,
            state_active_mask=state_active_mask,
        )


# ------------------------------------------------------------------
# Per-episode history snapshot (injected by Orchestrator at B1+)
# ------------------------------------------------------------------


@dataclass
class HistoryView:
    """Per-episode action / state history snapshot.

    Tensors are newest-last (so `actions[-1]` is the most recently
    executed action; `states[-1]` is the most recently observed state).
    Built and injected by Orchestrator.check() in B1+; B0 ships the
    dataclass so judges and factors can type against it.
    """

    actions: list["torch.Tensor"]
    states: list["torch.Tensor"]


# ------------------------------------------------------------------
# Protocols
# ------------------------------------------------------------------


@runtime_checkable
class OnlineExtractor(Protocol):
    """Verdict-time factor extraction. Pure function (no I/O side effect).

    Two-tier metadata model:

    Class-level capability flags (read by validator BEFORE instantiation,
    no params needed):
      - required_top_k:           default minimum top_k this factor type
                                  needs; CompositeJudge feeds the max of
                                  these into the search strategy via
                                  `min_top_k_hint`. May be overridden per
                                  instance via `params`.
      - requires_library_stats:   whether the constructor needs a
                                  `library_stats` kwarg from the config
                                  builder. True for F1a / F1b (z-score
                                  against library sigma); False for F2
                                  (candidate-pool variance is scale
                                  invariant).
      - requires_chain_walk:      whether `extract()` calls
                                  `view.walk_prev` / `view.walk_next`.
                                  True for F1a-T (reads downstream state);
                                  used by validator to fail-fast against
                                  backends that lack `fetch_entry`.

    Instance-level metadata (set in __init__ from params, read by
    CompositeJudge after instantiation):
      - descriptor_orientations:  {key -> "safe"|"risky"|"non_monotonic"}
                                  for every key this *instance* will
                                  produce. Keys depend on params (e.g.
                                  F1b keys encode descriptors x windows
                                  x source), so they cannot be class
                                  level. Instance __init__ should call
                                  `self.descriptor_orientations =
                                  self.__class__.describe(params)`.

    Static key introspection (for validator, before instantiation):
      - describe(params):         classmethod that returns the same
                                  key->orientation map from params alone,
                                  without touching library_stats or any
                                  runtime context. Validator uses this
                                  to cross-check ComposerConfig.directions
                                  coverage for non_monotonic keys without
                                  needing to construct the factor (which
                                  would require library_stats for F1b
                                  before the backend exists).
    """

    required_top_k: int
    requires_library_stats: bool
    requires_chain_walk: bool
    descriptor_orientations: dict[str, str]

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        """Return the {key -> orientation} map an instance built with
        `params` will produce. Pure function — no library_stats, no I/O.
        The classmethod and __init__ MUST agree:
        `instance.descriptor_orientations == cls.describe(params)`.
        """
        ...

    def extract(
        self,
        results: list["SearchResultLite"],
        view: "PayloadView",
        history: "HistoryView",
        cached_data: dict[str, "torch.Tensor"],
    ) -> dict[str, float]:
        """Return {key: value} factor descriptors for this verdict.

        Returned dict's key set MUST equal
        `self.descriptor_orientations.keys()`. CompositeJudge enforces
        this (key contract assertion); drift would silently desynchronize
        Normalizer state, Composer weights, and orientation lookup, so
        we fail loud at the boundary.

        Values may be NaN to signal a missing or invalid measurement
        (e.g. boundary windows, missing payload.factors); Composer is
        responsible for the orientation-aware NaN handling rule.
        """
        ...


@runtime_checkable
class OfflineWriter(Protocol):
    """Artifact-build / episode-end factor computation.

    Implementations consume an entire episode (a list of CacheEntry that
    share a trajectory_id) and return a per-entry factor dict that the
    caller merges into `entries[i].payload.factors`. Artifact-build
    tooling and Orchestrator's episode-end write path both invoke
    `compute_for_episode` through the same interface.
    """

    def required_payload_fields(self) -> set[str]:
        """Extra raw payload fields this writer needs (B2 default: empty
        set — current factors all read from existing schema)."""
        ...

    def compute_for_episode(
        self,
        entries: list["CacheEntry"],
        library_stats: LibraryStats,
    ) -> list[dict[str, float]]:
        """Return per-entry factor dict (parallel to `entries`).
        Caller merges into `entries[i].payload.factors`.
        """
        ...
