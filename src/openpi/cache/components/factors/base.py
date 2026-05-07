"""Factor protocols and shared library-level statistics.

A factor implements ``Factor`` (verdict-time ``extract(ctx)`` over a
``FactorContext``) plus, optionally, ``OfflineWriter`` (artifact-build /
episode-end ``compute_for_episode(entries, library_stats)`` that
populates ``payload.factors``). The two protocols are separate so a
factor can declare exactly the surface it provides; ``CompositeJudge``
consumes ``Factor`` instances at verdict time, while the artifact-build /
Orchestrator write-path collects ``OfflineWriter`` instances.

Coupling map:
  DEPENDS ON:  storage_types.py (CacheEntry, SearchResultLite),
               components/payload_view.py (PayloadView)
  CONSUMED BY: CompositeJudge (Factor),
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


# ------------------------------------------------------------------
# Refactor B1+: Layer 3 calibration samples + Factor verdict context
# ------------------------------------------------------------------


@dataclass
class CalibrationSamples:
    """Layer 3 Calibration startup data: per-key historical raw factor values.

    Loaded from one of two sources at server startup (yaml field
    ``calibration.samples_source.type``):
      - ``offline``: read from a JSONL / pkl file on disk
        (``samples_source.offline.path`` + ``format``).
      - ``warmup``: read from the per-yaml ``WarmupPool`` entry, which
        the warmup yaml's DumpingJudge populated by running the eval
        task distribution while logging raw factor values.

    Calibration subclasses inspect ``samples`` at ``bind_keys`` time and
    fail-fast if any bound key has fewer non-NaN samples than the
    configured window size. This guarantees every Layer 3 buffer is
    saturated before the first verdict — no cold-start state.
    """

    samples: dict[str, list[float]]


@dataclass
class FactorContext:
    """Per-verdict input dataclass for Layer 2 ``Factor.extract``.

    Single dataclass entry point so Factor subclasses do not have to
    juggle four positional arguments. Constructed by ``CompositeJudge.__call__``
    once per verdict and passed unchanged to every active factor.

    Coupling map:
      DEPENDS ON:  storage_types.SearchResultLite, components/payload_view.PayloadView,
                   HistoryView (above), normalization.Normalization (Layer 1).
      CONSUMED BY: components/factors/online.py, offline.py, topk.py.
    """

    results: list["SearchResultLite"]
    view: "PayloadView"
    history: HistoryView
    # Layer 1 instance — Factor subclasses call ``ctx.normalization.normalize_action(seq)``
    # / ``normalize_state(seq)`` when they need z-scored data. Typed as ``object``
    # here to avoid a hard import cycle with ``factors/normalization`` (which
    # depends on this module's ``LibraryStats``).
    normalization: object


@runtime_checkable
class Factor(Protocol):
    """Layer 2 verdict-factor protocol.

    The ``extract(ctx)`` contract collapses verdict-time inputs into a
    single ``FactorContext`` dataclass (``results / view / history /
    normalization``) so subclasses never juggle multiple positional args
    and never persist their own ``library_stats`` (Layer 1 owns
    z-score and is injected as ``ctx.normalization``).

    Class-level capability flags (read by validator BEFORE instantiation):
      - required_top_k:        instance hint upper-bounded by the constructor
                               (e.g. TopkActionVariance sets ``self.required_top_k = K``);
                               CompositeJudge picks the max across factors and
                               feeds search-strategy ``min_top_k_hint``.
      - requires_chain_walk:   ``True`` for any factor whose ``extract`` calls
                               ``ctx.view.walk_prev`` / ``walk_next``. The 8
                               online factors set this True (per plan §6.6 the
                               online splice walks the chain on both action
                               and state channels). The validator uses it to
                               fail-fast against backends without ``fetch_entry``.

    Instance-level metadata:
      - descriptor_orientations: ``{key -> "safe"|"risky"|"non_monotonic"}``
                                 for every key this instance produces. Keys
                                 depend on construction params (windows etc.),
                                 so the map is built in ``__init__`` via
                                 ``self.descriptor_orientations = self.__class__.describe(params)``.

    Static introspection:
      - ``describe(cls, params)`` classmethod: pure mapping from params to
        ``{key -> orientation}``, no runtime context. The validator uses it
        to compute the union of factor keys at yaml-load time and cross-check
        ``Composer.declared_dependencies``.
    """

    required_top_k: int
    requires_chain_walk: bool
    descriptor_orientations: dict[str, str]

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        """Return the ``{key -> orientation}`` map an instance built with
        ``params`` will produce. Pure — no I/O, no library_stats. The
        classmethod and ``__init__`` MUST agree:
        ``instance.descriptor_orientations == cls.describe(params)``.
        """
        ...

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        """Return ``{key: float}`` factor descriptors for this verdict.

        Returned dict's key set MUST equal
        ``self.descriptor_orientations.keys()``. CompositeJudge enforces
        this (key contract assertion); drift silently desynchronizes
        Layer 3 buffer state, Composer weights, and orientation lookup, so
        we fail loud at the boundary.

        Values may be NaN to signal a missing / invalid measurement
        (boundary windows, missing payload.factors, fork detected,
        zero-norm denominator). NaN propagates straight through Layer 3
        and into Composer; the Composer subclass owns NaN handling.
        """
        ...


# ------------------------------------------------------------------
# Shared yaml helper — window list normalization
# ------------------------------------------------------------------


def normalize_windows(raw) -> list[tuple[int, int]]:
    """Normalize a yaml ``windows`` block to ``list[tuple[int, int]]``.

    Accepts either ``[{"past": int, "future": int}, ...]`` (yaml shape) or
    already-normalized ``[(P, F), ...]``. Used by every Layer 2 Factor's
    ``describe`` classmethod and ``__init__`` so the validator and the
    runtime see identical window keys.

    Lives in base.py so both online.py and offline.py can import it
    without creating a sibling-module circular import (registry needs to
    import both at @register time).
    """
    out: list[tuple[int, int]] = []
    for w in raw:
        if isinstance(w, dict):
            out.append((int(w["past"]), int(w["future"])))
        else:
            p, f = w
            out.append((int(p), int(f)))
    return out
