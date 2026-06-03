"""PrebuiltMatrixBackend — ROUND 1 stack-elimination subclass of InMemoryBackend.

ROUND 1 first cut of the cp1_search progressive tuning (Owner-overridden workflow, see
``logs/cache_latency_bench_round1_stack_elim.log.md``). Lives entirely under ``exp/`` and
changes ZERO src code.

Responsibility
--------------
Eliminate the per-query candidate-matrix rematerialization in
``InMemoryBackend._compute_field_scores`` (in_memory_backend.py:386-387:
``torch.stack([c.query_keys[field] for c in valid]).float()``). On a write-frozen library
(``write_policy: never``) the candidate vectors never change, so the per-(checkpoint,
task_key, field) matrix is built ONCE at prebuild time and rows are gathered by id at
search time. The downstream operators (``F.cosine_similarity`` / ``torch.norm``) and the
scatter-back logic (in_memory_backend.py:398-402) are reused VERBATIM, so the numerical
path is bit-identical and the 3-way verdict is preserved — the equivalence ROUND 1 must
prove by the comparison harness.

Safety window
-------------
The fast path fires only when every candidate that has the field maps into a SINGLE
prebuilt (checkpoint, task_key, field) bucket and ``sim_type in {cosine, l2}``. Any other
shape — cross-bucket candidates, an id absent from the bucket, an unknown sim_type, or the
memo path (``_batch_field_scores`` called with sid/qid) — falls back to
``super()._compute_field_scores(...)``. Because ONLY ``_compute_field_scores`` is
overridden, the trajectory / rrf / score-memo paths run the parent unchanged.

Public interface
----------------
- ``_prebuild_task_matrices()``: build the per-bucket matrices; call AFTER the library is
  loaded (``_entries`` populated) and before timing.
- ``_compute_field_scores(...)``: override; same signature as the parent.

Dependencies: ``openpi.cache.backends.in_memory_backend.InMemoryBackend`` (parent).
Entries are duck-typed (``.id`` / ``.checkpoint_id`` / ``.payload.task_key`` /
``.query_keys``), matching ``CacheEntry``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from openpi.cache.backends.in_memory_backend import InMemoryBackend


class PrebuiltMatrixBackend(InMemoryBackend):
    """InMemoryBackend with per-(checkpoint, task_key, field) prebuilt fp32 matrices.

    ROUND 1 scope: only ``_compute_field_scores`` is overridden; everything else
    (search dispatch, fusion, normalization, trajectory, score-memo) is inherited and
    runs exactly as in the parent.
    """

    def __init__(self, vector_dims: dict[str, int]) -> None:
        super().__init__(vector_dims)
        # (checkpoint_id, task_key, field) -> contiguous fp32 [B, D] bucket matrix.
        self._mat: dict[tuple, torch.Tensor] = {}
        # (checkpoint_id, task_key, field) -> {entry_id: row_index} into the matrix.
        self._id2row: dict[tuple, dict[str, int]] = {}
        self._prebuilt: bool = False
        # Diagnostics: prove the fast path actually fires (vs both sides silently
        # taking the parent path, which would make a 0.0 diff a false positive).
        self._fast_hits: int = 0
        self._fallbacks: int = 0

    # ------------------------------------------------------------------
    # Prebuild
    # ------------------------------------------------------------------

    def _prebuild_task_matrices(self) -> None:
        """Stack each (checkpoint, task_key, field) bucket into a contiguous fp32 matrix.

        Iterates ``self._entries.values()`` in dict-insertion order — the SAME order
        ``_filter_entries`` (in_memory_backend.py:343) yields candidates. Row lookup is
        by id, so the gather reproduces the parent's per-query stack content regardless of
        later candidate ordering. Only entries that actually carry the field contribute a
        row (mirrors the parent's ``field_name in e.query_keys`` valid filter).

        Frozen-library contract: assumes ``write_policy: never`` — the matrices snapshot
        ``_entries`` ONCE and the fast path never revalidates a candidate's live
        ``query_keys[field]`` against its prebuilt row. Any post-prebuild in-place vector
        mutation (or insert/upsert) would serve stale rows; valid only on a write-frozen
        library. Re-call this method after any (offline) library change.
        """
        mat_rows: dict[tuple, list[torch.Tensor]] = {}
        id2row: dict[tuple, dict[str, int]] = {}
        for entry in self._entries.values():
            key_prefix = (entry.checkpoint_id, entry.payload.task_key)
            for field, vec in entry.query_keys.items():
                key = (*key_prefix, field)
                rows = mat_rows.setdefault(key, [])
                rowmap = id2row.setdefault(key, {})
                rowmap[entry.id] = len(rows)
                rows.append(vec.float())
        self._mat = {k: torch.stack(v).contiguous() for k, v in mat_rows.items()}
        self._id2row = id2row
        self._prebuilt = True

    # ------------------------------------------------------------------
    # Override: stack-free field scoring (safe window only)
    # ------------------------------------------------------------------

    def _compute_field_scores(
        self,
        query_vec: torch.Tensor,
        candidates: list,
        field_name: str,
        sim_cfg: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack-free fast path with bit-identical operators; else parent fallback.

        See module docstring for the safety window. The fast path gathers candidate rows
        from the prebuilt bucket (contiguous) and runs the SAME ``F.cosine_similarity`` /
        ``torch.norm`` as the parent, then scatters into the ``[n]`` scores/mask exactly
        like in_memory_backend.py:398-402.
        """
        n = len(candidates)
        # Same valid filter as the parent (in_memory_backend.py:381).
        valid_indices = [i for i, e in enumerate(candidates) if field_name in e.query_keys]
        if not valid_indices:
            return torch.zeros(n), torch.zeros(n)

        sim_type = sim_cfg.get("type", "cosine")
        key = self._fast_path_key(candidates, valid_indices, field_name, sim_type)
        if key is None:
            self._fallbacks += 1
            return super()._compute_field_scores(query_vec, candidates, field_name, sim_cfg)
        self._fast_hits += 1

        rowmap = self._id2row[key]
        rows = [rowmap[candidates[i].id] for i in valid_indices]
        full = self._mat[key]
        # Zero-copy steady state: when the candidates are exactly the whole bucket in
        # matrix-row order (step_filter=all over a frozen task bucket — _filter_entries
        # and prebuild iterate _entries in the SAME order), feed the resident matrix
        # directly and eliminate the per-query ~104MB rematerialization. Otherwise gather
        # the needed rows. Both feed bit-identical content to the operator.
        if rows == list(range(full.shape[0])):
            mat = full                                              # [N, D] resident view
        else:
            mat = full.index_select(0, torch.tensor(rows, dtype=torch.long)).contiguous()
        q = query_vec.float().unsqueeze(0)                          # [1, D]

        if sim_type == "cosine":
            valid_scores = F.cosine_similarity(q, mat)              # [V]
        else:  # "l2" — guaranteed by _fast_path_key
            valid_scores = torch.norm(q - mat, p=2, dim=1)          # [V]

        scores = torch.zeros(n)
        mask = torch.zeros(n)
        idx_t = torch.tensor(valid_indices, dtype=torch.long)
        scores[idx_t] = valid_scores
        mask[idx_t] = 1.0
        return scores, mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fast_path_key(self, candidates, valid_indices, field_name, sim_type):
        """Return the single bucket key if the fast path is safe, else ``None``.

        Safe iff: prebuild done, ``sim_type in {cosine, l2}``, all valid candidates share
        one (checkpoint, task_key) bucket that has a matrix for this field, and every valid
        candidate id is present in that bucket's id->row map.
        """
        if not self._prebuilt or sim_type not in ("cosine", "l2"):
            return None
        first = candidates[valid_indices[0]]
        key = (first.checkpoint_id, first.payload.task_key, field_name)
        rowmap = self._id2row.get(key)
        if rowmap is None:
            return None
        for i in valid_indices:
            e = candidates[i]
            if e.checkpoint_id != key[0] or e.payload.task_key != key[1] or e.id not in rowmap:
                return None
        return key
