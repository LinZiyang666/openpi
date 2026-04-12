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

import logging

import numpy as np
from typing import Any

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

    @property
    def vector_dims(self) -> dict[str, int]:
        return self._dims

    def supported_filters(self) -> frozenset[str]:
        return frozenset({"checkpoint_id", "task_key", "step_range"})

    def insert(self, entry: CacheEntry) -> None:
        self._entries[entry.id] = entry

    def fetch_payload(self, id: str) -> CachePayload:
        self.fetch_payload_call_count += 1
        if id not in self._entries:
            raise KeyError(id)
        return self._entries[id].payload

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._entries.pop(i, None)

    def count(self) -> int:
        return len(self._entries)

    def load_artifact(self, path: str) -> None:
        """Load pre-built entries from a pickle artifact.

        Artifact format (dict):
          {"key_builder_type": str, "checkpoint_id": str,
           "vector_dims": dict[str, int], "entries": list[CacheEntry]}

        Validates that artifact vector_dims matches self._dims.
        """
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
            return self._search_with_trajectory(candidates, spec)

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

    def _batch_field_scores(
        self,
        query_vec: torch.Tensor,
        candidates: list[CacheEntry],
        field_name: str,
        sim_cfg: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute raw similarity scores for one field across all candidates.

        Stacks candidate vectors into a matrix and computes scores in one
        batched operation instead of per-entry loops.

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
    # Trajectory search: two-pass recursive + batched per-level scoring
    # -------------------------------------------------------------------

    def _search_with_trajectory(
        self,
        candidates: list[CacheEntry],
        spec: QuerySpec,
    ) -> list[SearchResultLite]:
        """Trajectory search: two-pass recursive + batched per-level scoring.

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
