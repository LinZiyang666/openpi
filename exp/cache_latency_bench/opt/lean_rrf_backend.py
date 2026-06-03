"""LeanRRFBackend — ROUND-RRF R3 framework-overhead elimination for weighted_rrf.

The RRF roofline probe showed full ``_search_weighted_rrf`` ~5.2ms vs a lean
(GEMV+argsort+RRF) ~3.1ms — a ~1.7ms backend framework gap (the ``_batch_field_scores``
wrapper, ``_compute_field_scores`` zeros+scatter, ``mask.nonzero``), same shape as the
weighted_sum LEAN cut. This subclass overrides ``_search_weighted_rrf`` with a lean
steady-state path; the RRF rank fusion (per-field argsort + ``weight/(rrf_k+rank)``) is the
irreducible RRF cost that lean cannot remove.

Inherits ``LeanSearchBackend`` to REUSE ``_lean_bucket`` / ``_verified_buckets`` (the
whole-bucket-in-order check applies identically to RRF — both need the resident matrix in
candidate order). The inherited ``_search_weighted_score_sum`` override (weighted_sum LEAN)
is never reached under a weighted_rrf config, so it is harmless; ``_lean_hits`` (its counter)
stays 0 while this class tracks ``_lean_rrf_hits`` separately.

Equivalence: vs the parent PrenormDot ``_search_weighted_rrf`` it is bit-identical (same
``mat.mv`` cosine / ``torch.norm`` l2 → same argsort order → same ranks → same RRF score),
only the wrapper/scatter plumbing is dropped. Non-whole-bucket / top_k>1 falls back to
``super()._search_weighted_rrf`` (the Round-2 prenorm RRF path). fp32-ONLY (RRF argsort is
perturbation-sensitive: fp16=7%/bf16=31% winner flips on record).

Dependencies: ``exp.cache_latency_bench.opt.lean_search_backend.LeanSearchBackend``.
"""

from __future__ import annotations

import torch

from openpi.cache.storage_types import SearchResultLite

from exp.cache_latency_bench.opt.lean_search_backend import LeanSearchBackend


class LeanRRFBackend(LeanSearchBackend):
    """LeanSearchBackend with a scatter/wrapper-free steady-state weighted_rrf path."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lean_rrf_hits: int = 0
        self._lean_rrf_fallbacks: int = 0

    def _search_weighted_rrf(self, candidates: list, spec, active_fields) -> list:
        """Lean steady-state RRF (top_k==1, whole bucket) or full Round-2 prenorm fallback."""
        bucket = None
        if spec.top_k == 1:
            bucket = self._lean_bucket(candidates, active_fields)
        if bucket is None:
            self._lean_rrf_fallbacks += 1
            return super()._search_weighted_rrf(candidates, spec, active_fields)

        ck, tk = bucket
        rrf_k = (spec.backend_hints or {}).get("rrf_k", 60)
        n = len(candidates)
        rrf = torch.zeros(n)
        for field, weight, sim_cfg in active_fields:
            mat = self._mat[(ck, tk, field)]                 # resident: normed cosine / raw l2
            q = spec.query_keys[field].float()
            if field in self._cosine_fields:
                raw = mat.mv(q / q.norm().clamp_min(1e-12))  # cosine via prenorm GEMV
                descending = True
            else:  # l2 — raw distance, ascending (smaller = better), matches src :542
                raw = torch.norm(q.unsqueeze(0) - mat, p=2, dim=1)
                descending = False
            order = raw.argsort(descending=descending)
            ranks = torch.empty(n, dtype=torch.float32)
            ranks[order] = torch.arange(1, n + 1, dtype=torch.float32)
            rrf += weight / (rrf_k + ranks)

        self._lean_rrf_hits += 1
        top = int(rrf.topk(1).indices)                       # topk(1) matches src tie-break
        e = candidates[top]
        return [SearchResultLite(id=e.id, score=float(rrf[top]), checkpoint_id=e.checkpoint_id)]
