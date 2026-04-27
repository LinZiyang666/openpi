"""F2 — Top-K action consensus factor.

Measures the candidate-pool agreement on the next action. High
inter-candidate variance signals retrieval ambiguity ("the cache is
guessing") and is treated as risky. Drives `min_top_k_hint` in B1+ so
the search strategy returns enough candidates to compute the consensus.

B0 ships the metadata layer (class-level flags + instance-level
`required_top_k = K` + describe + register). The `extract` body raises
NotImplementedError until B1 lands the variance algorithm.

Coupling map:
  DEPENDS ON:  components/factors/base.py (OnlineExtractor protocol)
  CONSUMED BY: CompositeJudge (via registry); SearchStrategy
               `min_top_k_hint` wiring in B1+
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from openpi.cache.components.factors.registry import register

if TYPE_CHECKING:
    from openpi.cache.components.factors.base import HistoryView
    from openpi.cache.components.payload_view import PayloadView
    from openpi.cache.storage_types import SearchResultLite

# Threshold below which a per-DOF candidate variance is considered
# "constant across the candidate pool" (most likely a Pi0.5 padded DOF
# carrying zeros across the entire library). Dropped from the average so
# `f2_var` reflects only DOFs the cache actually disagrees about.
_F2_CANDIDATE_LOCAL_EPS: float = 1e-8


@register("f2")
class TopKActionConsensus:
    """Top-K candidate consensus on action_chunk[0].

    `K` is fed back to the search strategy via CompositeJudge's
    `min_required_top_k` so the strategy returns at least K candidates.
    The strategy's own `top_k` YAML field semantics are preserved —
    `max(top_k, min_top_k_hint)` controls the actual fetch.
    """

    # ---- Class-level capability flags ----
    source: Optional[str] = None             # F2 has no source dimension
    requires_library_stats: bool = False     # candidate-pool variance is scale invariant
    requires_chain_walk: bool = False

    def __init__(self, K: int) -> None:
        if K < 1:
            raise ValueError(f"TopKActionConsensus K must be >= 1, got {K}")
        self.K = K
        # Instance-level required_top_k (CompositeJudge picks the max
        # across extractors).
        self.required_top_k = K
        self.descriptor_orientations: dict[str, str] = self.__class__.describe(
            {"K": K}
        )

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        """F2 emits a single risky descriptor `f2_var` regardless of K
        (the variance is a single scalar).
        """
        return {"f2_var": "risky"}

    def extract(
        self,
        results: list["SearchResultLite"],
        view: "PayloadView",
        history: "HistoryView",
        cached_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        # Variance over candidate pool's first-step actions, restricted
        # to the candidate-LOCAL active subspace (per docs §3.5). This
        # path keeps `requires_library_stats=False`: we deliberately do
        # NOT consult `library_stats.action_active_mask`. Pi0.5 padded
        # DOFs carry exactly 0 across the full library, so per-DOF
        # candidate variance is also exactly 0 on those dims and they
        # drop out of the average via the eps gate.
        K_eff = min(self.K, len(results))
        if K_eff < 2:
            # Single candidate (or empty): no consensus signal — propagate
            # NaN so Composer treats it as missing-signal per §2.8.8.
            return {"f2_var": float("nan")}
        ids = [r.id for r in results[:K_eff]]
        payloads = view.get_many(ids)               # PayloadView memoizes
        chunk0 = torch.stack(
            [
                torch.as_tensor(p.action_chunk[0], dtype=torch.float32)
                for p in payloads
            ],
            dim=0,
        )                                            # [K_eff, A]
        var_d = chunk0.var(dim=0, unbiased=False)    # [A] population variance
        active = var_d > _F2_CANDIDATE_LOCAL_EPS     # candidate-local mask
        if not bool(active.any()):
            # Entire candidate pool agrees on every dim — distinguish
            # "perfect consensus" (which would naively give 0 and look
            # like a hard FULL_HIT signal) from "no useful active dims".
            # Return NaN to defer the decision to other factors.
            return {"f2_var": float("nan")}
        return {"f2_var": float(var_d[active].mean())}
