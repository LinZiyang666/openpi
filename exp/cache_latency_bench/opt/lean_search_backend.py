"""LeanSearchBackend — ROUND 3 framework-overhead elimination (cp1_search).

Round 2 (prenorm-dot) left cp1_search at ~4.7ms median, but a roofline probe showed the
pure operator (2x GEMV + normalize + weighted-sum + argmax) is only ~2.5ms — the ~2ms gap
is backend FRAMEWORK overhead, not the math:
  - ``_batch_field_scores`` hit/miss wrapper (depth_1 has no memo → all "miss"),
  - ``_compute_field_scores`` building ``torch.zeros[n]`` + scattering valid_scores back
    by index — an IDENTITY no-op when candidates == the whole bucket in row order,
  - ``topk(1)`` instead of ``argmax``,
  - the per-field dispatch / facade wrapping.

This subclass overrides ``_search_weighted_score_sum`` with a LEAN path for the steady
state (depth_1 + whole-bucket-in-matrix-row-order + every active field fast-pathable +
top_k==1): score each field directly on the resident matrix (cosine → ``M_unit.mv(qn)``;
l2 → ``torch.norm``), weighted-sum, ``argmax``. Bit-identical to the Round-2 path (same mv
/ normalize / sum operators — only the zeros/scatter/wrapper/topk plumbing is dropped).
Anything outside the steady state (LOO/subset candidates, top_k>1, a field not fast-
pathable) falls back to ``super()._search_weighted_score_sum`` (the full Round-2 path,
which itself prenorm-mv's via the overridden ``_compute_field_scores``).

Frozen-library contract inherited from PrenormDotBackend (write_policy:never). Zero src
changes.

Dependencies: ``exp.cache_latency_bench.opt.prenorm_dot_backend.PrenormDotBackend``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from openpi.cache.storage_types import SearchResultLite

from exp.cache_latency_bench.opt.prenorm_dot_backend import PrenormDotBackend


class LeanSearchBackend(PrenormDotBackend):
    """PrenormDotBackend with a scatter/wrapper-free steady-state search path.

    ROUND 3 scope: only ``_search_weighted_score_sum`` is overridden; prebuild,
    normalization, and ``_compute_field_scores`` (used by the fallback) are inherited.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lean_hits: int = 0       # diagnostics: lean path fired
        self._lean_fallbacks: int = 0  # diagnostics: fell back to the full Round-2 path
        # (ckpt, task, n) buckets whose whole-bucket-in-order property was verified once.
        # On a frozen library with a deterministic _filter_entries the candidate order is
        # invariant, so verify O(n) the first time, then trust O(1) thereafter.
        self._verified_buckets: set = set()

    # ------------------------------------------------------------------
    # Lean steady-state search
    # ------------------------------------------------------------------

    def _lean_bucket(self, candidates, active_fields):
        """Return (ckpt, task_key) iff the lean path is valid, else None.

        Valid iff every active field has a resident matrix, the candidate set IS the whole
        bucket in matrix-row order (so ``fused[i]`` aligns with ``candidates[i]`` and the
        resident matrix can be used zero-copy), cosine fields are normalized (in
        ``_unit_keys``) and l2 fields are raw. Mirrors ``_fast_path_key`` but at the search
        level and with the whole-bucket-in-order requirement.
        """
        if not candidates:
            return None
        first = candidates[0]
        ck, tk = first.checkpoint_id, first.payload.task_key
        n = len(candidates)
        for field, _weight, sim_cfg in active_fields:
            sim_type = sim_cfg.get("type", "cosine")
            key = (ck, tk, field)
            mat = self._mat.get(key)
            if mat is None or mat.shape[0] != n:
                return None
            if field in self._cosine_fields:
                if sim_type != "cosine" or key not in self._unit_keys:
                    return None
            elif field in self._l2_fields:
                if sim_type != "l2":
                    return None
            else:
                return None
        # Whole-bucket-in-order: verify O(n) ONCE per (ckpt,task,n), then O(1) endpoint
        # spot-check. A frozen library + deterministic _filter_entries makes the candidate
        # order invariant, so the per-query O(n) id scan (which ate the lean speedup) is
        # amortized; the endpoint check guards the order precondition cheaply.
        cache_key = (ck, tk, n)
        rowmap = self._id2row[(ck, tk, active_fields[0][0])]
        if cache_key not in self._verified_buckets:
            for i in range(n):
                if rowmap.get(candidates[i].id) != i:
                    return None
            self._verified_buckets.add(cache_key)
        elif rowmap.get(candidates[0].id) != 0 or rowmap.get(candidates[-1].id) != n - 1:
            # G2 MAJOR: a reordered same-(ckpt,task,n) bucket would misalign fused[i] vs
            # candidates[i]; the endpoints catch it without the full O(n) rescan.
            return None
        return (ck, tk)

    def _search_weighted_score_sum(
        self,
        candidates: list,
        spec,
        active_fields,
        normalizers: Optional[dict] = None,
    ) -> list:
        """Lean steady-state path (top_k==1, whole bucket) or full Round-2 fallback."""
        bucket = None
        if spec.top_k == 1:
            bucket = self._lean_bucket(candidates, active_fields)
        if bucket is None:
            self._lean_fallbacks += 1
            return super()._search_weighted_score_sum(candidates, spec, active_fields, normalizers)

        ck, tk = bucket
        normalizers = self._ensure_normalizers(normalizers, active_fields, spec)
        fused = None
        for field, weight, sim_cfg in active_fields:
            mat = self._mat[(ck, tk, field)]                       # resident: normed cosine / raw l2
            q = spec.query_keys[field].float()
            if field in self._cosine_fields:
                raw = mat.mv(q / q.norm().clamp_min(1e-12))        # cosine = unit·unit
            else:  # l2
                raw = torch.norm(q.unsqueeze(0) - mat, p=2, dim=1)
            s = weight * normalizers[field](raw)
            fused = s if fused is None else fused + s

        self._lean_hits += 1
        # topk(1) (not argmax) matches the full path's tie-break (last-max) exactly; on
        # continuous fp32 fused scores exact ties don't occur, so this is belt-and-braces.
        top = int(fused.topk(1).indices)
        e = candidates[top]
        return [SearchResultLite(id=e.id, score=float(fused[top]), checkpoint_id=e.checkpoint_id)]
