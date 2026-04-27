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

from openpi.cache.components.factors.registry import register

if TYPE_CHECKING:
    import torch

    from openpi.cache.components.factors.base import HistoryView
    from openpi.cache.components.payload_view import PayloadView
    from openpi.cache.storage_types import SearchResultLite


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
        cached_data: dict[str, "torch.Tensor"],
    ) -> dict[str, float]:
        # B1 ships: view.get_many([r.id for r in results[:K]]) → per-DOF
        # variance in the active subspace → return {"f2_var": float}.
        raise NotImplementedError(
            "TopKActionConsensus.extract: B1 algorithm"
        )
