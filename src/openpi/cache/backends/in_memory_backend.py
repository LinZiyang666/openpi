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
import logging
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from openpi.cache.backend_base import VectorStoreBackend
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    QuerySpec,
    SearchResultLite,
)

logger = logging.getLogger(__name__)


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

    def __init__(self, vector_dims: dict[str, int]) -> None:
        self._dims = vector_dims
        self._entries: dict[str, CacheEntry] = {}
        # Counters for call tracking in tests.
        self.search_call_count: int = 0
        self.fetch_payload_call_count: int = 0

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
        return frozenset({"checkpoint_id", "task_key", "step_range"})

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
        for entry in data["entries"]:
            # Backfill trajectory fields for old artifacts that lack them.
            if not hasattr(entry, "prev_ids"):
                entry.prev_ids = []
            if not hasattr(entry, "next_ids"):
                entry.next_ids = []
            if not hasattr(entry, "trajectory_id"):
                entry.trajectory_id = None
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
        self.search_call_count += 1
        if not self._entries:
            return []

        candidates = self._filter_entries(spec)
        if not candidates:
            return []

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
                return self._search_with_trajectory_legacy(candidates, spec)
            try:
                return self._search_with_trajectory(candidates, spec)
            except _MultiBranchSentinel:
                logger.warning(
                    "Multi-branch trajectory detected (prev_ids has > 1 "
                    "entry); falling back to legacy DAG path."
                )
                return self._search_with_trajectory_legacy(candidates, spec)

        # ── Existing single-step search (unchanged) ──
        method = spec.fusion_method
        if method == "weighted_rrf":
            active = self._iter_active_fields(spec)
            if not active:
                return []
            return self._search_weighted_rrf(candidates, spec, active)
        elif method == "weighted_score_sum":
            active = self._iter_active_fields(spec)
            if not active:
                return []
            return self._search_weighted_score_sum(candidates, spec, active)
        elif method is None:
            return self._search_single_field_cosine(candidates, spec)
        else:
            raise ValueError(f"Unknown fusion_method: {method}")

    # -------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------

    def _filter_entries(self, spec: QuerySpec) -> list[CacheEntry]:
        """Filter by checkpoint_id / task_key / step_range."""
        results = []
        for entry in self._entries.values():
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
            results.append(entry)
        return results

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
        valid_indices = [i for i, e in enumerate(candidates) if field_name in e.query_keys]

        if not valid_indices:
            return torch.zeros(n), torch.zeros(n)

        vecs = [candidates[i].query_keys[field_name] for i in valid_indices]
        mat = torch.stack(vecs).float()           # [V, D]
        q = query_vec.float().unsqueeze(0)         # [1, D]

        sim_type = sim_cfg.get("type", "cosine")
        if sim_type == "cosine":
            valid_scores = F.cosine_similarity(q, mat)     # [V]
        elif sim_type == "l2":
            valid_scores = torch.norm(q - mat, p=2, dim=1)  # [V]
        else:
            raise ValueError(f"Unknown similarity type: {sim_type}")

        scores = torch.zeros(n)
        mask = torch.zeros(n)
        idx_t = torch.tensor(valid_indices, dtype=torch.long)
        scores[idx_t] = valid_scores
        mask[idx_t] = 1.0
        return scores, mask

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

    def _search_weighted_score_sum(
        self,
        candidates: list[CacheEntry],
        spec: QuerySpec,
        active_fields: list[tuple[str, float, dict[str, Any]]],
    ) -> list[SearchResultLite]:
        """Weighted Score Sum with normalization (batched).

        1. Per-field batched raw score (cosine or L2)
        2. Convert to [0,1] similarity:
           - cosine: (cos + 1) / 2
           - L2: exp(-d / tau)
        3. Percentile normalization: clip((s - p5) / (p95 - p5), 0, 1)
        4. Weighted sum: Score(x) = sum_f w_f * s_hat_f(x)
        5. Sort descending, return top_k
        """
        norm_config = spec.score_normalization or {}
        norm_type = norm_config.get("type", "none")
        norm_fields = norm_config.get("fields", {})

        if norm_type == "percentile":
            missing = [fn for fn, _, _ in active_fields if fn not in norm_fields]
            if missing:
                raise ValueError(
                    f"Percentile normalization missing stats for active fields: {missing}. "
                    f"Available: {list(norm_fields)}. Re-run calibration."
                )

        n = len(candidates)
        final_scores = torch.zeros(n)

        for field_name, weight, sim_cfg in active_fields:
            sim_type = sim_cfg.get("type", "cosine")
            raw, mask = self._batch_field_scores(
                spec.query_keys[field_name], candidates, field_name, sim_cfg,
            )

            # Convert to [0, 1] similarity
            if sim_type == "cosine":
                s = (raw + 1.0) / 2.0
            elif sim_type == "l2":
                to_sim = sim_cfg.get("to_similarity", {})
                tau = to_sim.get("tau", 1.0)
                s = torch.exp(-raw / tau)
            else:
                s = raw

            # Percentile normalization
            if norm_type == "percentile":
                p5 = norm_fields[field_name]["p5"]
                p95 = norm_fields[field_name]["p95"]
                denom = p95 - p5
                if denom > 0:
                    s = ((s - p5) / denom).clamp(0.0, 1.0)
                else:
                    s = torch.full_like(s, 0.5)

            # Only accumulate for entries that have this field.
            final_scores += weight * s * mask

        top_k = min(spec.top_k, n)
        top_indices = final_scores.topk(top_k).indices.tolist()

        results = []
        for idx in top_indices:
            entry = candidates[idx]
            results.append(
                SearchResultLite(id=entry.id, score=float(final_scores[idx]),
                                 checkpoint_id=entry.checkpoint_id)
            )
        return results

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
            norm_config = spec.score_normalization or {}
            norm_type = norm_config.get("type", "none")
            norm_fields = norm_config.get("fields", {})
            if norm_type == "percentile":
                missing = [fn for fn, _, _ in active_fields if fn not in norm_fields]
                if missing:
                    raise ValueError(
                        f"Percentile normalization missing stats for active fields: "
                        f"{missing}. Available: {list(norm_fields)}. Re-run calibration."
                    )
            final_scores = torch.zeros(n)
            for field_name, weight, sim_cfg, raw, mask in per_field_scores:
                sim_type = sim_cfg.get("type", "cosine")
                if sim_type == "cosine":
                    s = (raw + 1.0) / 2.0
                elif sim_type == "l2":
                    to_sim = sim_cfg.get("to_similarity", {})
                    tau = to_sim.get("tau", 1.0)
                    s = torch.exp(-raw / tau)
                else:
                    s = raw
                if norm_type == "percentile":
                    p5 = norm_fields[field_name]["p5"]
                    p95 = norm_fields[field_name]["p95"]
                    denom = p95 - p5
                    if denom > 0:
                        s = ((s - p5) / denom).clamp(0.0, 1.0)
                    else:
                        s = torch.full_like(s, 0.5)
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
            scores = self._batch_step_scores(entries_at_level, history[idx], spec)
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
            results = self._search_weighted_score_sum(entries, temp_spec, level_active_fields)
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
