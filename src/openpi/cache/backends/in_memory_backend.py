"""In-memory vector store backend for experiments and testing.

Overview
--------
Stores entries in a Python dict. Supports multi-field search with two fusion
methods: weighted RRF and weighted score sum. Also retains backward-compatible
single-field cosine fallback (fusion_method=None).

Capabilities:
  - Multi-field entry/query (vision_0/1/2, prompt_emb, robot_state)
  - Filters: checkpoint_id, task_key, step_range
  - Field similarity: cosine (vision/prompt) and L2 (robot_state)
  - Fusion: weighted_rrf / weighted_score_sum
  - Brute-force top-k (suitable for < 50k entries)
  - Artifact loading from pickle files

Data flow: CacheStorage -> InMemoryBackend.search/insert/... -> in-process dict

Coupling map:
  DEPENDS ON:  backend_base.py (VectorStoreBackend ABC), storage_types.py
  CONSUMED BY: CacheStorage (via VectorStoreBackend interface)
  IF CHANGED:  tests and experiment cache configs may need updating
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import time
import weakref
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from openpi.cache.backend_base import VectorStoreBackend
from openpi.cache.components.score_normalizers import (
    ScoreNormalizer,
    build_field_normalizers,
)
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    QuerySpec,
    SearchResultLite,
    StepRetrievalFeatures,
)

logger = logging.getLogger(__name__)


def _diag_count_only(results: list[SearchResultLite]) -> StepRetrievalFeatures:
    """Diagnostics for a path that does not decompose per-field scores.

    Only the coverage count is meaningful off the weighted-score-sum single-step
    path; the risk router is config-gated to that path, so the other paths still
    report a truthful ``n_results`` rather than a fabricated decomposition.
    """
    return StepRetrievalFeatures(n_results=len(results))


class SearchSessionActiveError(RuntimeError):
    """Raised when a backend mutation that could pollute cached scores is
    attempted while at least one search session is active.

    See plan §6 mutation contract: during active sessions only `insert(new_id)`
    is allowed. `insert(existing_id)` (upsert), `delete`, and `load_artifact`
    raise this error so violators are exposed instead of silently corrupting
    caches.
    """


class _MultiBranchSentinel(Exception):
    """Internal sentinel raised by `_walk_chain` when a `prev_ids` list has
    more than one element. The caller falls back to the legacy DAG path.
    """


class InMemoryBackend(VectorStoreBackend):
    """In-memory backend with multi-field search and two fusion methods.

    Supports:
      - weighted_rrf: per-field independent ranking + weighted RRF fusion
      - weighted_score_sum: normalized score computation + weighted sum
      - None (fallback): single-field cosine for backward compatibility
    """

    def __init__(self, vector_dims: dict[str, int], text_ivf=None) -> None:
        self._dims = vector_dims
        self._entries: dict[str, CacheEntry] = {}
        # Text-IVF bucket index config (duck-typed: needs .field / .max_buckets,
        # normally a config.TextIvfIndexConfig). None => index disabled; a
        # QuerySpec carrying the text_ivf hint then raises.
        self._text_ivf_cfg = text_ivf
        # Derived bucket index, built lazily / eagerly-at-load and dropped on
        # every mutation. Published atomically via single reference assignment:
        # (buckets: dict[bytes, list[entry_id]],
        #  keys_sorted: list[bytes]  (ascending; doubles as the tie-break order),
        #  reps: FloatTensor [B, dim] | None, rep_norms: FloatTensor [B] | None)
        self._text_ivf_state = None
        # Runtime write-frozen contract (C2): flipped True by freeze().
        self._is_frozen: bool = False
        # Counters for call tracking in tests.
        self.search_call_count: int = 0
        self.fetch_payload_call_count: int = 0

        # Frozen-search caches. The serving path (write_policy=never) issues
        # thousands of searches against an immutable entry set, and rebuilding
        # the filter list plus torch.stack'ing hundreds of key vectors PER
        # QUERY dominated live search cost (measured: 136ms of a 202ms infer,
        # ws_search timing lab 2026-08-22). Both caches are exact — same
        # inputs, same outputs — and any mutation clears them.
        #   _filtered_cache: filter fingerprint -> candidates list (stable
        #     object identity is what keys the matrix cache below).
        #   _field_matrix_cache: (field, id(candidates)) -> (weakref to the
        #     list, valid-index tensor, stacked matrix). The weakref guards
        #     against id() reuse after the original list is garbage collected.
        self._filtered_cache: dict[tuple, list[CacheEntry]] = {}
        self._field_matrix_cache: dict[tuple, tuple] = {}

        # Library-level statistics for verdict-factor F1b normalization.
        # Populated by `load_artifact` (from artifact dict, or computed
        # lazily as fallback when an old artifact lacks the field). None
        # otherwise. Read by CacheStorage.library_stats facade accessor.
        self.library_stats = None

        # Cross-step score memo.
        # Outer key: search_session_id (per-strategy-per-episode uuid4 hex).
        # Inner key: (field_name, query_id, sim_type).
        # Innermost: entry_id -> raw similarity score.
        # Bucket created lazily on first cache miss; dropped on close_search_session.
        self._score_memo: dict[str, dict[tuple, dict[str, float]]] = {}

        # Active session registry, INDEPENDENT from `_score_memo` buckets.
        # Populated by `open_search_session` (called from CacheStorage on
        # behalf of CacheOrchestrator) BEFORE any search runs, so the
        # mutation guard activates without depending on cache write timing.
        self._active_search_sessions: set[str] = set()

        # Test-only flag flipped by `force_legacy_path()` context manager.
        self._force_legacy: bool = False

    @property
    def vector_dims(self) -> dict[str, int]:
        return self._dims

    def supported_filters(self) -> frozenset[str]:
        return frozenset({"checkpoint_id", "task_key", "step_range", "outcome"})

    # ------------------------------------------------------------------
    # Session lifecycle (override ABC default no-op)
    # ------------------------------------------------------------------

    def open_search_session(self, session_id: str) -> None:
        """Register an active search session. Called by CacheOrchestrator
        through CacheStorage *before* any search runs in the session, so the
        mutation guard sees the session as active even before the first
        cache bucket is materialized.
        """
        self._active_search_sessions.add(session_id)

    def close_search_session(self, session_id: str) -> None:
        """Drop the cache bucket and remove the session from the active set.
        Idempotent — closing an unknown sid is a safe no-op.
        """
        self._active_search_sessions.discard(session_id)
        self._score_memo.pop(session_id, None)

    def _has_active_search_sessions(self) -> bool:
        return bool(self._active_search_sessions)

    @contextlib.contextmanager
    def force_legacy_path(self):
        """Test-only: force trajectory search to take the legacy DAG path.

        Use as ``with backend.force_legacy_path(): backend.search(spec)``.
        The legacy path does NOT read or write `_score_memo`, so cache
        state is unaffected by entering / leaving the context.
        """
        prev = self._force_legacy
        self._force_legacy = True
        try:
            yield self
        finally:
            self._force_legacy = prev

    # ------------------------------------------------------------------
    # CRUD with mutation guards
    # ------------------------------------------------------------------

    def insert(self, entry: CacheEntry) -> None:
        # Active-session mutation contract: upsert of existing id is forbidden
        # while any search session is active (would invalidate cached scores).
        # Insert of a brand-new id is always safe (does not touch existing slots).
        if entry.id in self._entries and self._has_active_search_sessions():
            raise SearchSessionActiveError(
                f"Cannot upsert existing entry {entry.id!r} while "
                f"{len(self._active_search_sessions)} search session(s) are active. "
                "Close all sessions before mutation (offline-only operation)."
            )
        self._entries[entry.id] = entry
        self._invalidate_frozen_search_caches()

    def fetch_payload(self, id: str) -> CachePayload:
        self.fetch_payload_call_count += 1
        if id not in self._entries:
            raise KeyError(id)
        return self._entries[id].payload

    def fetch_entry(self, id: str) -> CacheEntry:
        """Return the full CacheEntry by id. O(1) dict lookup.

        Capability used by `CacheStorage.fetch_entry` (duck-typed facade
        method) to support PayloadView chain walks. The Backend ABC does
        not declare this method — it is an InMemoryBackend-specific
        capability that the facade exposes via getattr, so backends that
        cannot keep full entries in memory (e.g. Qdrant) simply do not
        provide it.
        """
        if id not in self._entries:
            raise KeyError(id)
        return self._entries[id]

    def delete(self, ids: list[str]) -> None:
        if self._has_active_search_sessions():
            raise SearchSessionActiveError(
                "Cannot delete entries while search sessions are active. "
                "Offline-only operation."
            )
        for i in ids:
            self._entries.pop(i, None)
        self._invalidate_frozen_search_caches()

    def _invalidate_frozen_search_caches(self) -> None:
        self._filtered_cache.clear()
        self._field_matrix_cache.clear()
        self._text_ivf_state = None

    def count(self) -> int:
        return len(self._entries)

    def load_artifact(self, path: str) -> None:
        """Load pre-built entries from a pickle artifact.

        Artifact format (dict):
          {"key_builder_type": str, "checkpoint_id": str,
           "vector_dims": dict[str, int], "entries": list[CacheEntry],
           "library_stats": LibraryStats (optional, B2 onwards)}

        Validates that artifact vector_dims matches self._dims.

        `library_stats` is optional: when missing (legacy artifacts) the
        backend computes it lazily from the loaded entries so verdict
        factors that depend on `LibraryStats` (F1a / F1b) still have a
        normalization basis. Recompute logs a warning so users notice
        and can rebuild the artifact with the new pipeline.

        Raises SearchSessionActiveError if any search session is active —
        load_artifact replaces backend contents and would invalidate all
        cached scores. Must be called offline (server idle).

        Also raises BackendFrozenError post-freeze (runtime write-frozen
        contract, C2): re-loading an artifact is a database content mutation
        and must happen before freeze() or in offline tooling.
        """
        if self._has_active_search_sessions():
            raise SearchSessionActiveError(
                "Cannot load_artifact while search sessions are active. "
                "Offline-only operation."
            )
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        if data["vector_dims"] != self._dims:
            raise ValueError(
                f"Artifact vector_dims mismatch: "
                f"artifact={data['vector_dims']}, backend={self._dims}"
            )
        # Record which builder produced this artifact so a caller can reject a
        # library whose keys mean something different. `vector_dims` alone
        # cannot do that: mean-pool and max-pool artifacts are dimensionally
        # identical. Recorded, not enforced — the check belongs to whoever
        # knows the expected identity (see CacheStorage.artifact_meta). Legacy
        # artifacts predate these fields and read back as None.
        self.artifact_meta = {
            "key_builder_type": data.get("key_builder_type"),
            "checkpoint_id": data.get("checkpoint_id"),
            # Prompt-pool identity (text-IVF): dict for prompt-pool-aware
            # builds, None for legacy artifacts — the binding check treats
            # None as "must rebuild" whenever the config engages the knobs.
            "prompt_pool": data.get("prompt_pool"),
        }
        for entry in data["entries"]:
            # Backfill trajectory fields for old artifacts that lack them.
            if not hasattr(entry, "prev_ids"):
                entry.prev_ids = []
            if not hasattr(entry, "next_ids"):
                entry.next_ids = []
            if not hasattr(entry, "trajectory_id"):
                entry.trajectory_id = None
            # Backfill the failure-aware outcome tag (TRACER M2 / Phase 3) so
            # old artifacts read as untagged (None) instead of raising on access.
            if not hasattr(entry, "outcome"):
                entry.outcome = None
            # Convert numpy arrays back to torch tensors (memory-efficient artifacts)
            if entry.query_keys:
                entry.query_keys = {
                    k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v
                    for k, v in entry.query_keys.items()
                }
            p = entry.payload
            if p.action_chunk is not None and isinstance(p.action_chunk, np.ndarray):
                p.action_chunk = torch.from_numpy(p.action_chunk).float()
            if p.intermediates:
                p.intermediates = {
                    k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v
                    for k, v in p.intermediates.items()
                }
            self._entries[entry.id] = entry
        logger.info("Loaded %d entries from %s", len(data["entries"]), path)

        # Text-IVF: build the bucket index eagerly so a polluted / mismatched
        # artifact fails at load time (inside BackendPool's per-fingerprint
        # load lock, before freeze) instead of on the first live search.
        if self._text_ivf_cfg is not None:
            self._build_text_ivf_index()

        # ---- library_stats: load from artifact OR fallback recompute ----
        # Distinguish "missing key" (legacy artifact, fallback) from
        # "explicitly None" (build pipeline ran without OfflineWriters
        # configured but didn't compute stats either — same fallback).
        # Either way, recompute lazily from the entries we just loaded.
        ls = data.get("library_stats")
        if ls is None:
            from openpi.cache.components.factors.base import LibraryStats

            t0 = time.time()
            logger.warning(
                "Artifact %s lacks `library_stats`; computing from %d "
                "entries (one-time fallback — rebuild with the B2 pipeline "
                "to skip this).",
                path, len(self._entries),
            )
            ls = LibraryStats.compute_from_entries(list(self._entries.values()))
            logger.warning(
                "library_stats fallback compute finished in %.2fs", time.time() - t0,
            )
        self.library_stats = ls

    # -------------------------------------------------------------------
    # Search dispatch
    # -------------------------------------------------------------------

    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        """Vector search. Thin wrapper that discards X15 diagnostics."""
        results, _ = self.search_with_diagnostics(spec)
        return results

    def search_with_diagnostics(
        self, spec: QuerySpec
    ) -> tuple[list[SearchResultLite], StepRetrievalFeatures]:
        """Search, returning the results and this search's retrieval diagnostics.

        Returning the diagnostics *with* the results is what makes them safe
        under ``BackendPool``, which shares one backend across connections: a
        mutable ``last_*`` slot on the backend could be overwritten by another
        connection between the search and the judge that reads it. The caller
        (a per-connection ``CacheStorage`` facade) owns the returned snapshot.

        Diagnostics are populated only on the weighted-score-sum single-step
        path — the one the X15 risk router is gated to by config validation.
        Every other path returns an empty ``StepRetrievalFeatures``, whose
        ``n_results`` still reflects the real result count.
        """
        self.search_call_count += 1
        diag = StepRetrievalFeatures()
        if not self._entries:
            return [], diag

        if spec.backend_hints and spec.backend_hints.get("text_ivf"):
            candidates = self._text_ivf_candidates(spec)
        else:
            candidates = self._filtered_candidates(spec)
        if not candidates:
            return [], diag

        # ── Trajectory search ──
        if (spec.trajectory_history is not None
                and spec.trajectory_weights is not None
                and len(spec.trajectory_weights) > 1):
            logger.info(
                "Trajectory search: depth=%d, history_len=%d, candidates=%d",
                len(spec.trajectory_weights),
                len(spec.trajectory_history),
                len(candidates),
            )
            if self._force_legacy:
                out = self._search_with_trajectory_legacy(candidates, spec)
                return out, _diag_count_only(out)
            try:
                out = self._search_with_trajectory(candidates, spec)
            except _MultiBranchSentinel:
                logger.warning(
                    "Multi-branch trajectory detected (prev_ids has > 1 "
                    "entry); falling back to legacy DAG path."
                )
                out = self._search_with_trajectory_legacy(candidates, spec)
            return out, _diag_count_only(out)

        # ── Existing single-step search (unchanged) ──
        method = spec.fusion_method
        if method == "weighted_rrf":
            active = self._iter_active_fields(spec)
            if not active:
                return [], diag
            out = self._search_weighted_rrf(candidates, spec, active)
            return out, _diag_count_only(out)
        elif method == "weighted_score_sum":
            active = self._iter_active_fields(spec)
            if not active:
                return [], diag
            if self._wss_collects_diagnostics():
                return self._search_weighted_score_sum(
                    candidates, spec, active, collect_diagnostics=True,
                )
            # A subclass overriding the fusion with its own optimised path
            # (the latency-bench LEAN backends) does not produce the per-field
            # decomposition; it reports a truthful count instead. Probing the
            # signature once mirrors the judge-kwarg seam in components/judge.py.
            out = self._search_weighted_score_sum(candidates, spec, active)
            return out, _diag_count_only(out)
        elif method is None:
            out = self._search_single_field_cosine(candidates, spec)
            return out, _diag_count_only(out)
        else:
            raise ValueError(f"Unknown fusion_method: {method}")

    # -------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------

    class _CandidateList(list):
        """List subclass so the matrix cache can hold weak identity refs
        (plain lists do not support weakref)."""

        __slots__ = ("__weakref__",)

    def _filtered_candidates(
        self,
        spec: QuerySpec,
        bucket: Optional[tuple[bytes, list[str]]] = None,
    ) -> list[CacheEntry]:
        """`_filter_entries` behind the frozen-search cache.

        The returned list is cached by filter fingerprint so repeated searches
        with the same filters (one episode = hundreds of them) reuse the SAME
        list object — which is also what keys the per-field matrix cache.
        Mutations clear the cache, so a hit is always exact.

        ``bucket`` (text-IVF path) restricts the base entry set to one bucket's
        members; its key joins the fingerprint so bucket candidate lists keep a
        stable identity across an episode's searches (None on the legacy path
        leaves legacy fingerprints equivalent).
        """
        key = (
            spec.checkpoint_id,
            None if spec.filters is None else spec.filters.task_key,
            None if spec.filters is None else spec.filters.step_range,
            None if spec.filters is None else spec.filters.outcome,
            None if bucket is None else bucket[0],
        )
        try:
            cached = self._filtered_cache.get(key)
        except TypeError:  # unhashable fingerprint component: skip caching
            return self._filter_entries(spec, bucket=bucket)
        if cached is not None:
            return cached
        result = self._CandidateList(self._filter_entries(spec, bucket=bucket))
        if len(self._filtered_cache) > 256:  # unbounded-fingerprint backstop
            self._filtered_cache.clear()
        self._filtered_cache[key] = result
        return result

    def _filter_entries(
        self,
        spec: QuerySpec,
        bucket: Optional[tuple[bytes, list[str]]] = None,
    ) -> list[CacheEntry]:
        """Filter by checkpoint_id / task_key / step_range."""
        if bucket is None:
            pool = self._entries.values()
        else:
            pool = [self._entries[eid] for eid in bucket[1] if eid in self._entries]
        results = []
        for entry in pool:
            if spec.checkpoint_id is not None and entry.checkpoint_id != spec.checkpoint_id:
                continue
            if spec.filters is not None:
                if spec.filters.task_key is not None:
                    if entry.payload.task_key != spec.filters.task_key:
                        continue
                if spec.filters.step_range is not None:
                    lo, hi = spec.filters.step_range
                    if entry.step_idx is None:
                        continue  # no step_idx → excluded by step_range filter
                    if not (lo <= entry.step_idx <= hi):
                        continue
                if spec.filters.outcome is not None:
                    # getattr guards old pickles that predate the outcome field
                    # (unpickle bypasses __init__); missing => None => unmatched.
                    if getattr(entry, "outcome", None) != spec.filters.outcome:
                        continue
            results.append(entry)
        return results

    # -------------------------------------------------------------------
    # Text-IVF bucket index (screening field = prompt_emb)
    # -------------------------------------------------------------------

    def _build_text_ivf_index(self):
        """Group entries into buckets by byte-identical screening vectors.

        Fail-fast contract: any entry missing the screening field aborts the
        build (a partially screened library silently changes retrieval scope);
        more buckets than ``max_buckets`` aborts too — that shape means either
        a state-in-prompt polluted artifact or one built without
        instruction-span masked pooling.

        The state is assembled in locals and published with one reference
        assignment, so concurrent readers on the lazy path never observe a
        half-built index. Rebuilding from the same entry set is idempotent.
        """
        cfg = self._text_ivf_cfg
        buckets: dict[bytes, list[str]] = {}
        for entry in self._entries.values():
            vec = entry.query_keys.get(cfg.field)
            if vec is None:
                raise ValueError(
                    f"text_ivf: entry {entry.id!r} lacks screening field "
                    f"{cfg.field!r}. The whole library must carry it — rebuild "
                    "the artifact with the screening field enabled."
                )
            buckets.setdefault(vec.numpy().tobytes(), []).append(entry.id)
        if len(buckets) > cfg.max_buckets:
            raise ValueError(
                f"text_ivf: {len(buckets)} buckets exceed max_buckets="
                f"{cfg.max_buckets}. Likely causes: a state-in-prompt model "
                "whose prompt embedding drifts per step, or an artifact built "
                "without instruction-span masked pooling. Rebuild the artifact "
                "with --prompt-masked-pool (and --prompt-instruction-span for "
                "discrete-state prompts), or raise max_buckets if the library "
                "genuinely holds this many instructions."
            )
        keys_sorted = sorted(buckets)
        if keys_sorted:
            reps = torch.stack([
                torch.from_numpy(np.frombuffer(k, dtype=np.float32).copy())
                for k in keys_sorted
            ])
            rep_norms = torch.linalg.vector_norm(reps, dim=1).clamp_min(1e-8)
        else:
            reps, rep_norms = None, None
        state = (buckets, keys_sorted, reps, rep_norms)
        self._text_ivf_state = state
        return state

    def _text_ivf_candidates(self, spec: QuerySpec) -> list[CacheEntry]:
        """Probe the bucket index: exact byte match first, nearest-rep fallback.

        Returns the probed bucket's members passed through the normal filter
        semantics (checkpoint_id / step_range / outcome). Empty library =>
        empty result (a MISS downstream), never an error.
        """
        if self._text_ivf_cfg is None:
            raise RuntimeError(
                "QuerySpec carries the text_ivf hint but this backend has no "
                "text_ivf index configured (backend.in_memory.index_type)."
            )
        state = self._text_ivf_state
        if state is None:
            state = self._build_text_ivf_index()
        buckets, keys_sorted, reps, rep_norms = state
        query = spec.query_keys.get(self._text_ivf_cfg.field)
        if query is None:
            raise ValueError(
                f"text_ivf search requires query field {self._text_ivf_cfg.field!r} "
                "in query_keys."
            )
        if not buckets:
            return self._CandidateList()

        qbytes = query.float().contiguous().numpy().tobytes()
        if qbytes in buckets:
            bucket_key = qbytes
            logger.debug("text_ivf probe: exact bucket hit (%d members)",
                         len(buckets[bucket_key]))
        else:
            q = query.float()
            denom = (rep_norms * torch.linalg.vector_norm(q)).clamp_min(1e-8)
            sims = (reps @ q) / denom
            # argmax returns the FIRST max index; keys_sorted is ascending, so
            # ties deterministically resolve to the smallest bucket key.
            best = int(sims.argmax())
            margin = float("inf")
            if sims.numel() >= 2:
                top2 = sims.topk(2).values
                margin = float(top2[0] - top2[1])
            if margin < 1e-4:
                logger.warning(
                    "text_ivf probe: nearest-bucket margin %.2e is tiny — "
                    "routing may be unstable (collapsed representatives or "
                    "numeric drift).", margin)
            else:
                logger.debug("text_ivf probe: nearest bucket sim=%.6f margin=%.2e",
                             float(sims[best]), margin)
            bucket_key = keys_sorted[best]
        return self._filtered_candidates(spec, bucket=(bucket_key, buckets[bucket_key]))

    # -------------------------------------------------------------------
    # Batched field scoring
    # -------------------------------------------------------------------

    def _compute_field_scores(
        self,
        query_vec: torch.Tensor,
        candidates: list[CacheEntry],
        field_name: str,
        sim_cfg: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure batched similarity computation for one field. NO cache.

        Stacks candidate vectors into a matrix and computes scores in one
        batched operation. Reused by both the cached wrapper
        `_batch_field_scores` and trajectory-search miss-fill paths.

        Returns: (scores, mask) both float32 tensors of shape [N].
                 scores: raw similarity values (undefined where mask=False).
                 mask: 1.0 for candidates that have the field, 0.0 otherwise.
        """
        n = len(candidates)
        idx_t, mat, mask, row_norms = self._candidate_matrix(candidates, field_name)
        if mat is None:
            return torch.zeros(n), torch.zeros(n)

        q = query_vec.float()                      # [D]

        sim_type = sim_cfg.get("type", "cosine")
        if sim_type == "cosine":
            # Manual dot/norms instead of F.cosine_similarity: the broadcast
            # kernel materializes several [V, D] temporaries and measured
            # 43ms/call on [369, 32768] (p1 profile, 2026-08-22) — the matvec
            # with cached row norms is the same math (docs formula:
            # x·y / max(‖x‖‖y‖, ε)) at a fraction of the traffic.
            denom = (row_norms * torch.linalg.vector_norm(q)).clamp_min(1e-8)
            valid_scores = (mat @ q) / denom               # [V]
        elif sim_type == "l2":
            valid_scores = torch.norm(q.unsqueeze(0) - mat, p=2, dim=1)  # [V]
        else:
            raise ValueError(f"Unknown similarity type: {sim_type}")

        scores = torch.zeros(n)
        scores[idx_t] = valid_scores
        return scores, mask.clone()

    def _candidate_matrix(
        self, candidates: list[CacheEntry], field_name: str,
    ) -> tuple[
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
    ]:
        """(valid_idx, matrix, mask, row_norms) for a field, cached per list object.

        The gather + ``torch.stack`` of hundreds of key vectors was the
        dominant per-query cost (frozen-search lab, 2026-08-22); the stacked
        matrix only depends on (candidates list, field), so it is cached keyed
        by ``id(candidates)`` with a weakref identity guard: a hit requires the
        SAME list object to still be alive, which `_filtered_candidates`
        guarantees across an episode's searches. Ad-hoc lists (memo subset
        fills, tests) simply miss and pay the old cost. Returns (None, None,
        None) when no candidate has the field.
        """
        key = (field_name, id(candidates))
        cached = self._field_matrix_cache.get(key)
        if cached is not None:
            ref, idx_t, mat, mask, row_norms = cached
            if ref() is candidates:
                return idx_t, mat, mask, row_norms

        valid_indices = [i for i, e in enumerate(candidates) if field_name in e.query_keys]
        if not valid_indices:
            idx_t, mat, mask, row_norms = None, None, None, None
        else:
            mat = torch.stack(
                [candidates[i].query_keys[field_name] for i in valid_indices]
            ).float()
            idx_t = torch.tensor(valid_indices, dtype=torch.long)
            mask = torch.zeros(len(candidates))
            mask[idx_t] = 1.0
            # Row L2 norms are query-independent: precompute for the manual
            # cosine path so a warm query reads the matrix exactly once.
            row_norms = torch.linalg.vector_norm(mat, dim=1)
        try:
            ref = weakref.ref(candidates)
        except TypeError:
            # Ad-hoc plain list (memo subset fill, direct test call): not
            # weakref-able, so not identity-guardable — compute uncached.
            return idx_t, mat, mask, row_norms
        if len(self._field_matrix_cache) > 64:  # bound resident duplicate matrices
            self._field_matrix_cache.clear()
        self._field_matrix_cache[key] = (ref, idx_t, mat, mask, row_norms)
        return idx_t, mat, mask, row_norms

    def _batch_field_scores(
        self,
        query_vec: torch.Tensor,
        candidates: list[CacheEntry],
        field_name: str,
        sim_cfg: dict[str, Any],
        sid: Optional[str] = None,
        qid: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached field-similarity wrapper.

        - sid/qid both None  → trunk uncached path (single-step search,
          back-compat with existing callers like _search_weighted_rrf).
        - sid/qid both set   → look up in `_score_memo[sid][(field, qid, sim_type)]`
          and only compute the missing entries; cache the new scores.
        - sid not in `_active_search_sessions` → defensive fallback to uncached path
          + warning (lifecycle bug somewhere upstream).
        """
        if sid is None or qid is None:
            return self._compute_field_scores(query_vec, candidates, field_name, sim_cfg)

        # Defensive: refuse to create a bucket for an unregistered sid.
        if sid not in self._active_search_sessions:
            logger.warning(
                "Search with unregistered search_session_id %r; falling back "
                "to uncached path. Indicates a lifecycle bug (strategy minted "
                "sid but orchestrator did not register).", sid,
            )
            return self._compute_field_scores(query_vec, candidates, field_name, sim_cfg)

        sim_type = sim_cfg.get("type", "cosine")
        inner_key = (field_name, qid, sim_type)

        bucket = self._score_memo.setdefault(sid, {})
        slot = bucket.setdefault(inner_key, {})

        # Partition candidates into hits / misses with one Python pass.
        # Bulk index_put for the hit fill avoids creating one 0-d tensor per
        # candidate, which dominated the loop at N≥5k in benchmarks.
        n = len(candidates)
        hit_indices: list[int] = []
        hit_values: list[float] = []
        miss_indices: list[int] = []
        for i, e in enumerate(candidates):
            cached = slot.get(e.id)
            if cached is not None:
                hit_indices.append(i)
                hit_values.append(cached)
            elif field_name in e.query_keys:
                miss_indices.append(i)

        scores = torch.zeros(n)
        mask = torch.zeros(n)
        if hit_indices:
            idx_t = torch.tensor(hit_indices, dtype=torch.long)
            val_t = torch.tensor(hit_values, dtype=torch.float32)
            scores[idx_t] = val_t
            mask[idx_t] = 1.0

        if miss_indices:
            sub = [candidates[i] for i in miss_indices]
            sub_scores, sub_mask = self._compute_field_scores(
                query_vec, sub, field_name, sim_cfg,
            )
            valid_mask = sub_mask.bool()
            if valid_mask.any():
                valid_local = valid_mask.nonzero(as_tuple=True)[0]
                miss_idx_t = torch.tensor(miss_indices, dtype=torch.long)
                target_idx = miss_idx_t[valid_local]
                target_vals = sub_scores[valid_local].to(torch.float32)
                scores[target_idx] = target_vals
                mask[target_idx] = 1.0
                # Persist the freshly computed scores in the cache slot.
                # Slot is a plain dict; per-id write is unavoidable but it
                # only runs on misses, not on the steady-state hit path.
                vals_list = target_vals.tolist()
                for k, j_local in enumerate(valid_local.tolist()):
                    slot[candidates[miss_indices[j_local]].id] = vals_list[k]
        return scores, mask

    def _iter_active_fields(
        self,
        spec: QuerySpec,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return active fields: [(field_name, weight, sim_config), ...]

        Active = present in query_keys AND in vector_dims AND weight > 0.
        """
        result = []
        weights = spec.fusion_weights or {}
        sim_configs = spec.field_similarity or {}
        for field_name in spec.query_keys:
            if field_name not in self._dims:
                continue
            w = weights.get(field_name, 1.0)
            if w <= 0:
                continue
            sim_cfg = sim_configs.get(field_name, {"type": "cosine"})
            result.append((field_name, w, sim_cfg))
        return result

    # -------------------------------------------------------------------
    # Weighted RRF fusion
    # -------------------------------------------------------------------

    def _search_weighted_rrf(
        self,
        candidates: list[CacheEntry],
        spec: QuerySpec,
        active_fields: list[tuple[str, float, dict[str, Any]]],
    ) -> list[SearchResultLite]:
        """Weighted Reciprocal Rank Fusion (batched).

        1. Per-field: batched score computation, then argsort for ranks
        2. RRF: score(x) = sum_f w_f / (rrf_k + rank_f(x))
        3. Sort by RRF score descending, return top_k
        """
        rrf_k = 60
        if spec.backend_hints:
            rrf_k = spec.backend_hints.get("rrf_k", 60)

        n = len(candidates)
        rrf_scores = torch.zeros(n)

        for field_name, weight, sim_cfg in active_fields:
            scores, mask = self._batch_field_scores(
                spec.query_keys[field_name], candidates, field_name, sim_cfg,
            )
            valid_idx = mask.nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            valid_scores = scores[valid_idx]

            sim_type = sim_cfg.get("type", "cosine")
            if sim_type == "cosine":
                order = valid_scores.argsort(descending=True)
            else:
                order = valid_scores.argsort(descending=False)
            # Only rank valid entries; invalid entries get no RRF contribution.
            ranks = torch.empty(valid_idx.numel(), dtype=torch.float32)
            ranks[order] = torch.arange(1, valid_idx.numel() + 1, dtype=torch.float32)
            rrf_scores[valid_idx] += weight / (rrf_k + ranks)

        top_k = min(spec.top_k, n)
        top_indices = rrf_scores.topk(top_k).indices.tolist()

        results = []
        for idx in top_indices:
            entry = candidates[idx]
            results.append(
                SearchResultLite(id=entry.id, score=float(rrf_scores[idx]),
                                 checkpoint_id=entry.checkpoint_id)
            )
        return results

    # -------------------------------------------------------------------
    # Weighted Score Sum fusion
    # -------------------------------------------------------------------

    def _ensure_normalizers(
        self,
        normalizers: Optional[dict[str, ScoreNormalizer]],
        active_fields: list[tuple[str, float, dict[str, Any]]],
        spec: QuerySpec,
    ) -> dict[str, ScoreNormalizer]:
        """Return a normalizer dict covering every active field.

        Builds the full set when none was threaded in. When a prebuilt set was
        threaded from a trajectory entry point, fills any field it does not
        cover — a history layer may carry a field the current step's set lacks
        (build cost is incurred only for the missing fields).
        """
        if normalizers is None:
            return build_field_normalizers(
                active_fields, spec.score_normalization, spec.field_similarity,
            )
        missing = [af for af in active_fields if af[0] not in normalizers]
        if missing:
            return {
                **normalizers,
                **build_field_normalizers(
                    missing, spec.score_normalization, spec.field_similarity,
                ),
            }
        return normalizers

    def _search_weighted_score_sum(
        self,
        candidates: list[CacheEntry],
        spec: QuerySpec,
        active_fields: list[tuple[str, float, dict[str, Any]]],
        normalizers: Optional[dict[str, ScoreNormalizer]] = None,
        collect_diagnostics: bool = False,
    ):
        """Two-layer weighted score sum (batched).

        Layer 1 (normalization): each field's raw score (cosine value or L2
        distance from `_batch_field_scores`) is mapped to a bounded, comparable
        scalar by a per-field `ScoreNormalizer` (built from spec config).
        Layer 2 (fusion): Score(x) = sum_f w_f * s_hat_f(x), then top_k.

        `normalizers` may be supplied by the trajectory entry points so they are
        constructed once per search rather than once per chain layer.

        With ``collect_diagnostics`` the return becomes
        ``(results, StepRetrievalFeatures)``: the per-field normalized scores
        this method already computes are captured for the X15 risk router
        instead of being dropped after fusion. The default stays list-returning
        so the trajectory entry points are byte-identical.
        """
        normalizers = self._ensure_normalizers(normalizers, active_fields, spec)

        n = len(candidates)
        final_scores = torch.zeros(n)
        # Retained only under collect_diagnostics: {field: (masked scores)}.
        per_field_masked: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        for field_name, weight, sim_cfg in active_fields:
            raw, mask = self._batch_field_scores(
                spec.query_keys[field_name], candidates, field_name, sim_cfg,
            )
            # Layer 1: raw similarity -> bounded normalized score.
            s = normalizers[field_name](raw)
            # Layer 2: only accumulate for entries that have this field.
            contribution = s * mask
            final_scores += weight * contribution
            if collect_diagnostics:
                # Keep the masked normalized score (pre-weight): the router
                # compares fields to each other, so the fusion weight — a
                # config constant — would only rescale every step identically.
                # The mask rides along: a candidate lacking this field scores 0
                # after masking, and counting that zero as a runner-up would
                # fabricate a margin out of an absent candidate.
                per_field_masked[field_name] = (contribution, mask)

        top_k = min(spec.top_k, n)
        top_indices = final_scores.topk(top_k).indices.tolist()

        results = []
        for idx in top_indices:
            entry = candidates[idx]
            results.append(
                SearchResultLite(id=entry.id, score=float(final_scores[idx]),
                                 checkpoint_id=entry.checkpoint_id)
            )
        if not collect_diagnostics:
            return results
        return results, self._build_step_features(
            results, top_indices, final_scores, per_field_masked,
        )

    @classmethod
    def _wss_collects_diagnostics(cls) -> bool:
        """Whether this class's fusion accepts the diagnostics flag.

        Cached per class: the answer is fixed at import time, and the probe is
        on the search hot path.
        """
        cached = cls.__dict__.get("_WSS_DIAG_SUPPORTED")
        if cached is None:
            params = inspect.signature(cls._search_weighted_score_sum).parameters
            cached = "collect_diagnostics" in params
            cls._WSS_DIAG_SUPPORTED = cached
        return cached

    @staticmethod
    def _build_step_features(
        results: list[SearchResultLite],
        top_indices: list[int],
        final_scores: torch.Tensor,
        per_field_masked: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> StepRetrievalFeatures:
        """Assemble the X15 diagnostics from one fusion pass.

        ``winner_per_field`` decomposes the FUSED winner; ``field_own_margin``
        instead ranks each field independently, which is what exposes a field
        that cannot separate its own best two candidates even when the fused
        score looks confident.
        """
        if not results:
            return StepRetrievalFeatures()

        winner_idx = top_indices[0]
        winner_per_field = {
            name: float(scores[winner_idx])
            for name, (scores, _) in per_field_masked.items()
        }

        field_own_margin: dict[str, float] = {}
        for name, (scores, mask) in per_field_masked.items():
            # Only candidates that actually carry the field are ranked: with
            # fewer than two of them the field has no second place, so the
            # margin is undefined rather than zero.
            scored = scores[mask > 0]
            if scored.numel() < 2:
                continue
            top2 = scored.topk(2).values
            field_own_margin[name] = float(top2[0] - top2[1])

        fused_margin = 0.0
        if len(top_indices) >= 2:
            fused_margin = float(
                final_scores[top_indices[0]] - final_scores[top_indices[1]]
            )

        return StepRetrievalFeatures(
            fused_topk=tuple((r.id, r.score) for r in results),
            winner_per_field=winner_per_field,
            field_own_margin=field_own_margin,
            fused_margin=fused_margin,
            n_results=len(results),
        )

    # -------------------------------------------------------------------
    # Trajectory search — new single-chain main path
    # -------------------------------------------------------------------

    def _search_with_trajectory(
        self,
        candidates: list[CacheEntry],
        spec: QuerySpec,
    ) -> list[SearchResultLite]:
        """Single-chain trajectory search with cross-step score memo.

        Specialised for ``len(prev_ids) <= 1`` per entry (the only shape
        present in the current cache). Multi-branch entries cause
        `_walk_chain` to raise `_MultiBranchSentinel`; the dispatcher in
        `search()` catches it and falls back to `_search_with_trajectory_legacy`.

        Pipeline:
          (1) Walk `prev_ids` once to flatten ancestors into a [N, L] grid.
          (2) Per layer, fuse field-level similarities (RRF rank scope is
              still confined to that layer's reachable set — same contract
              as the legacy path, just without redundant top-k sort).
          (3) Accumulate per-candidate trajectory score: weighted sum over
              layers using `level_scores[l].get(ancestor_ids[i, l], 0.0)`.
          (4) Partial top-k.

        Backend score memo (when `spec.search_session_id` and
        `spec.trajectory_query_ids` are both non-None) keys raw per-field
        similarity by (session_id, field, query_id, sim_type). Same query
        tensor at different layers shares a query_id, so reuse holds across
        steps even though the layer index moves.
        """
        history = spec.trajectory_history    # newest-first
        weights = spec.trajectory_weights     # newest-first
        qids = spec.trajectory_query_ids      # newest-first or None
        sid = spec.search_session_id            # None → uncached path
        # Effective depth — history may be shorter than weights when episode
        # has not yet accumulated enough steps. Excess weight slots are unused.
        L = min(len(history), len(weights))
        if L <= 0:
            return []

        # Build per-field normalizers once (params are query-independent), shared
        # across all chain layers. Only weighted_score_sum uses them — building
        # for the RRF path would be wasted work and could raise on a stray type.
        traj_normalizers = (
            build_field_normalizers(
                self._iter_active_fields(spec), spec.score_normalization, spec.field_similarity,
            )
            if spec.fusion_method == "weighted_score_sum"
            else None
        )

        # (1) Flatten ancestors. ancestor_ids[i][l] is candidate i's ancestor
        # l steps back (None when chain ends earlier than L).
        ancestor_ids = self._walk_chain(candidates, depth=L,
                                        expected_checkpoint_id=spec.checkpoint_id)

        # (2) Per-layer batched fusion -> {entry_id: layer_score}.
        level_scores: list[dict[str, float]] = []
        for layer_idx in range(L):
            layer_entry_ids: set[str] = set()
            for row in ancestor_ids:
                eid = row[layer_idx]
                if eid is not None:
                    layer_entry_ids.add(eid)
            if not layer_entry_ids:
                level_scores.append({})
                continue
            layer_entries = [
                self._entries[eid] for eid in layer_entry_ids
                if eid in self._entries
            ]
            if not layer_entries:
                level_scores.append({})
                continue
            qid = qids[layer_idx] if qids is not None else None
            level_scores.append(self._compute_level_scores(
                layer_entries, history[layer_idx], spec, sid=sid, qid=qid,
                normalizers=traj_normalizers,
            ))

        # (3) Accumulate per-candidate trajectory score.
        traj_scores = self._accumulate(ancestor_ids, level_scores, weights)

        # (4) Partial top-k.
        top_k = min(spec.top_k, len(candidates))
        if top_k <= 0:
            return []
        top_indices = traj_scores.topk(top_k).indices.tolist()

        if logger.isEnabledFor(logging.INFO) and top_indices:
            best = top_indices[0]
            logger.info(
                "  Trajectory result: winner=%s, traj_score=%.6f",
                candidates[best].id, float(traj_scores[best]),
            )
        return [
            SearchResultLite(
                id=candidates[i].id,
                score=float(traj_scores[i]),
                checkpoint_id=candidates[i].checkpoint_id,
            )
            for i in top_indices
        ]

    def _walk_chain(
        self,
        candidates: list[CacheEntry],
        depth: int,
        expected_checkpoint_id: Any = None,
    ) -> list[list[Optional[str]]]:
        """Iteratively flatten each candidate's ancestor chain to depth L.

        Returns a list of length len(candidates), each element a list of
        length `depth` where slot l is the entry id l steps back (or None
        when the chain ended earlier or hit a checkpoint mismatch).

        Raises `_MultiBranchSentinel` immediately on encountering any entry
        with len(prev_ids) > 1 — such DAGs are handled by the legacy path.
        """
        out: list[list[Optional[str]]] = []
        for entry in candidates:
            row: list[Optional[str]] = [None] * depth
            cur: Optional[CacheEntry] = entry
            for layer in range(depth):
                if cur is None:
                    break
                if (expected_checkpoint_id is not None
                        and cur.checkpoint_id != expected_checkpoint_id):
                    break
                row[layer] = cur.id
                prev_ids = getattr(cur, "prev_ids", None) or []
                if len(prev_ids) > 1:
                    raise _MultiBranchSentinel()
                if not prev_ids:
                    break
                cur = self._entries.get(prev_ids[0])
            out.append(row)
        return out

    def _compute_level_scores(
        self,
        entries: list[CacheEntry],
        query_keys: dict[str, torch.Tensor],
        spec: QuerySpec,
        sid: Optional[str] = None,
        qid: Optional[int] = None,
        normalizers: Optional[dict[str, ScoreNormalizer]] = None,
    ) -> dict[str, float]:
        """Per-layer fusion -> {entry_id: layer_score}.

        Recomputes active fields against the layer's query_keys (older
        history may lack some fields). Bypasses single-step `_search_*`
        functions' final top-k sort: returns a flat dict directly.

        RRF rank scope is confined to this layer's `entries` — identical
        to the legacy path's "per-level reachable-set RRF" semantics.
        """
        if not entries:
            return {}

        temp_spec = QuerySpec(
            query_keys=query_keys,
            top_k=len(entries),
            checkpoint_id=spec.checkpoint_id,
            fusion_weights=spec.fusion_weights,
            fusion_method=spec.fusion_method,
            field_similarity=spec.field_similarity,
            score_normalization=spec.score_normalization,
            backend_hints=spec.backend_hints,
        )
        active_fields = self._iter_active_fields(temp_spec)
        if not active_fields:
            return {}

        method = spec.fusion_method
        n = len(entries)

        # Per-field cached similarity scores (sid/qid plumbed through).
        per_field_scores: list[tuple[str, float, dict[str, Any], torch.Tensor, torch.Tensor]] = []
        for field_name, weight, sim_cfg in active_fields:
            scores, mask = self._batch_field_scores(
                query_keys[field_name], entries, field_name, sim_cfg,
                sid=sid, qid=qid,
            )
            per_field_scores.append((field_name, weight, sim_cfg, scores, mask))

        if method == "weighted_rrf":
            rrf_k = 60
            if spec.backend_hints:
                rrf_k = spec.backend_hints.get("rrf_k", 60)
            rrf_scores = torch.zeros(n)
            for field_name, weight, sim_cfg, scores, mask in per_field_scores:
                valid_idx = mask.nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                valid_scores = scores[valid_idx]
                sim_type = sim_cfg.get("type", "cosine")
                if sim_type == "cosine":
                    order = valid_scores.argsort(descending=True)
                else:
                    order = valid_scores.argsort(descending=False)
                ranks = torch.empty(valid_idx.numel(), dtype=torch.float32)
                ranks[order] = torch.arange(1, valid_idx.numel() + 1, dtype=torch.float32)
                rrf_scores[valid_idx] += weight / (rrf_k + ranks)
            return {entries[i].id: float(rrf_scores[i]) for i in range(n)}

        if method == "weighted_score_sum":
            normalizers = self._ensure_normalizers(normalizers, active_fields, spec)
            final_scores = torch.zeros(n)
            for field_name, weight, sim_cfg, raw, mask in per_field_scores:
                # Layer 1: per-field bounded normalization (shared with single-step).
                s = normalizers[field_name](raw)
                final_scores += weight * s * mask
            return {entries[i].id: float(final_scores[i]) for i in range(n)}

        # Fallback (fusion_method=None) — use first field's cosine.
        if per_field_scores:
            field_name, weight, sim_cfg, scores, mask = per_field_scores[0]
            return {
                entries[i].id: float(scores[i])
                for i in range(n)
                if mask[i] > 0
            }
        return {}

    def _accumulate(
        self,
        ancestor_ids: list[list[Optional[str]]],
        level_scores: list[dict[str, float]],
        weights: list[float],
    ) -> torch.Tensor:
        """Combine per-layer scores into per-candidate trajectory scores.

        traj[i] = sum_l weights[l] * level_scores[l].get(ancestor_ids[i][l], 0.0)

        Missing ancestors (chain ended early) and missing layer entries
        contribute 0 to the sum.
        """
        n = len(ancestor_ids)
        L = len(weights)
        traj = torch.zeros(n)
        for i, row in enumerate(ancestor_ids):
            total = 0.0
            for l in range(L):
                eid = row[l] if l < len(row) else None
                if eid is None:
                    continue
                score = level_scores[l].get(eid)
                if score is None:
                    continue
                total += weights[l] * score
            traj[i] = total
        return traj

    # -------------------------------------------------------------------
    # Trajectory search — legacy DAG path (multi-branch fallback + parity gold reference)
    # -------------------------------------------------------------------

    def _search_with_trajectory_legacy(
        self,
        candidates: list[CacheEntry],
        spec: QuerySpec,
    ) -> list[SearchResultLite]:
        """Trajectory search: two-pass recursive + batched per-level scoring.

        Original implementation kept verbatim as the multi-branch fallback
        (entries with `len(prev_ids) > 1`) and as the parity gold reference
        for tests via `force_legacy_path()`.

        Reuses existing per-step fusion (RRF / score_sum) and adds cross-step
        weighted fusion on top.

        Phase A — collect: walk prev_ids to gather entry sets per depth level.
        Phase B — score: batch-score each level using configured fusion_method.
        Phase C — aggregate: walk same paths again, look up pre-computed scores,
                  weighted sum across levels, take max over branching paths.

        RRF note: when fusion_method="weighted_rrf", each level's RRF ranking
        is scoped to that level's reachable entry set (per-level reachable-set
        RRF score), which differs from single-step RRF semantics.
        """
        history = spec.trajectory_history   # newest-first
        weights = spec.trajectory_weights   # newest-first
        max_depth = len(weights) - 1

        # Phase A: collect entry ids per depth level
        level_entries: list[set[str]] = [set() for _ in range(len(weights))]
        for entry in candidates:
            self._collect_trajectory_entries(
                entry_id=entry.id,
                depth=max_depth,
                max_depth=max_depth,
                level_entries=level_entries,
                query_history_len=len(history),
                expected_checkpoint_id=spec.checkpoint_id,
            )
        level_sizes = [len(s) for s in level_entries]
        logger.info("  Phase A collect: level_sizes=%s (level 0=current)", level_sizes)

        # Build per-field normalizers once, shared across all depth levels; only
        # the weighted_score_sum path consumes them.
        traj_normalizers = (
            build_field_normalizers(
                self._iter_active_fields(spec), spec.score_normalization, spec.field_similarity,
            )
            if spec.fusion_method == "weighted_score_sum"
            else None
        )

        # Phase B: batch-score each level
        level_scores: list[dict[str, float]] = []
        for idx in range(len(weights)):
            entry_ids = level_entries[idx]
            if not entry_ids:
                level_scores.append({})
                continue
            entries_at_level = [self._entries[eid] for eid in entry_ids if eid in self._entries]
            if not entries_at_level:
                level_scores.append({})
                continue
            scores = self._batch_step_scores(
                entries_at_level, history[idx], spec, normalizers=traj_normalizers,
            )
            level_scores.append(scores)

        # Phase C: aggregate trajectory scores
        scored: list[tuple[CacheEntry, float]] = []
        for entry in candidates:
            path_scores = self._score_trajectory(
                entry_id=entry.id,
                depth=max_depth,
                max_depth=max_depth,
                weights=weights,
                accumulated_sim=0.0,
                level_scores=level_scores,
                query_history_len=len(history),
                expected_checkpoint_id=spec.checkpoint_id,
            )
            traj_score = max(path_scores) if path_scores else 0.0
            scored.append((entry, traj_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[0] if scored else None
        if top:
            logger.info(
                "  Phase C result: winner=%s, traj_score=%.6f",
                top[0].id, top[1],
            )
        return [
            SearchResultLite(id=e.id, score=s, checkpoint_id=e.checkpoint_id)
            for e, s in scored[:spec.top_k]
        ]

    def _collect_trajectory_entries(
        self,
        entry_id: str,
        depth: int,
        max_depth: int,
        level_entries: list[set[str]],
        query_history_len: int,
        expected_checkpoint_id=None,
    ) -> None:
        """First pass: collect entry ids that need scoring at each depth level.

        Skips entries whose checkpoint_id doesn't match expected_checkpoint_id
        (defensive check against cross-checkpoint pointer pollution).
        """
        entry = self._entries.get(entry_id)
        if entry is None:
            return
        if expected_checkpoint_id is not None and entry.checkpoint_id != expected_checkpoint_id:
            return

        idx = max_depth - depth
        if idx >= query_history_len:
            return

        level_entries[idx].add(entry_id)

        if depth == 0 or not entry.prev_ids:
            return
        for prev_id in entry.prev_ids:
            self._collect_trajectory_entries(
                prev_id, depth - 1, max_depth,
                level_entries, query_history_len,
                expected_checkpoint_id,
            )

    def _batch_step_scores(
        self,
        entries: list[CacheEntry],
        query_keys: dict[str, 'torch.Tensor'],
        spec: QuerySpec,
        normalizers: Optional[dict[str, ScoreNormalizer]] = None,
    ) -> dict[str, float]:
        """Batch-score one level's entries using configured fusion method.

        Builds a temporary QuerySpec with the level's query_keys, recomputes
        active_fields for that level (historical queries may lack some fields),
        then delegates to existing fusion methods.

        Returns {entry_id: step_score}.
        """
        if not entries:
            return {}

        temp_spec = QuerySpec(
            query_keys=query_keys,
            top_k=len(entries),
            checkpoint_id=spec.checkpoint_id,
            fusion_weights=spec.fusion_weights,
            fusion_method=spec.fusion_method,
            field_similarity=spec.field_similarity,
            score_normalization=spec.score_normalization,
            backend_hints=spec.backend_hints,
        )

        level_active_fields = self._iter_active_fields(temp_spec)
        if not level_active_fields:
            return {}

        if spec.fusion_method == "weighted_rrf":
            results = self._search_weighted_rrf(entries, temp_spec, level_active_fields)
        elif spec.fusion_method == "weighted_score_sum":
            results = self._search_weighted_score_sum(
                entries, temp_spec, level_active_fields, normalizers=normalizers,
            )
        else:
            results = self._search_single_field_cosine(entries, temp_spec)

        return {r.id: r.score for r in results}

    def _score_trajectory(
        self,
        entry_id: str,
        depth: int,
        max_depth: int,
        weights: list[float],
        accumulated_sim: float,
        level_scores: list[dict[str, float]],
        query_history_len: int,
        expected_checkpoint_id=None,
    ) -> list[float]:
        """Second pass: look up pre-computed scores, weighted sum, handle branching.

        Index mapping: idx = max_depth - depth.
        Early termination when idx >= query_history_len (consistent with _collect).
        Checkpoint guard: skips entries whose checkpoint_id doesn't match, preventing
        dirty paths (e.g. CP1->CP3->CP1) from borrowing scores computed via other
        valid paths in level_scores.
        Returns list of accumulated scores for all complete paths.
        """
        entry = self._entries.get(entry_id)
        if entry is None:
            return [accumulated_sim]

        if expected_checkpoint_id is not None and entry.checkpoint_id != expected_checkpoint_id:
            return [accumulated_sim]

        idx = max_depth - depth
        if idx >= query_history_len:
            return [accumulated_sim]
        if idx >= len(level_scores):
            return [accumulated_sim]

        step_score = level_scores[idx].get(entry_id, 0.0)
        accumulated_sim += weights[idx] * step_score

        if depth == 0 or not entry.prev_ids:
            return [accumulated_sim]

        all_paths: list[float] = []
        for prev_id in entry.prev_ids:
            all_paths.extend(self._score_trajectory(
                prev_id, depth - 1, max_depth,
                weights, accumulated_sim, level_scores,
                query_history_len,
                expected_checkpoint_id,
            ))
        return all_paths

    # -------------------------------------------------------------------
    # Backward-compatible single-field cosine (fusion_method=None)
    # -------------------------------------------------------------------

    def _search_single_field_cosine(
        self, candidates: list[CacheEntry], spec: QuerySpec
    ) -> list[SearchResultLite]:
        """Original single-field cosine search (batched). Existing tests depend on this."""
        sim_cfg = {"type": "cosine"}
        for field in spec.query_keys:
            scores, mask = self._batch_field_scores(
                spec.query_keys[field], candidates, field, sim_cfg,
            )
            if mask.sum() == 0:
                continue
            # Set missing entries to -inf so they sort last.
            scores[mask == 0] = float("-inf")
            top_k = min(spec.top_k, int(mask.sum().item()))
            top_indices = scores.topk(top_k).indices.tolist()
            return [
                SearchResultLite(
                    id=candidates[i].id,
                    score=float(scores[i]),
                    checkpoint_id=candidates[i].checkpoint_id,
                )
                for i in top_indices
            ]
        return []
