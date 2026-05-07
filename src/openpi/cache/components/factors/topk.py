"""Layer 2 — TopkActionVariance (B2 of the verdict-factor refactor).

The 17th factor: per-DOF variance of the top-K candidate pool's
``action_chunk[0]``. High variance signals retrieval ambiguity ("the
cache is guessing") — treated as risky.

Plan §6.11 #7: registered name is ``topk_action_variance``. Single
descriptor key (no windows): ``topk_action_variance``. Construction
param: ``K: int >= 2``.

Differences from the 8 online / 8 offline factors:
  - ``required_top_k = K`` (so search-strategy ``min_top_k_hint`` lifts
    fetch breadth to at least K candidates).
  - Does NOT consult Layer 1 normalization. Variance over the candidate
    pool is scale-invariant and Pi0.5 padded DOFs carry exactly 0 across
    the entire library, so per-DOF candidate variance is also exactly 0
    on those dims and they drop out of the active mask via the eps gate.
  - ``requires_chain_walk = False``.

Coupling map:
  DEPENDS ON: components/factors/base.py (Factor, FactorContext),
              components/factors/_debug.py (verdict logger),
              components/factors/registry.py.
  CONSUMED BY: components/composite_judge.py (Layer 2 instance);
               search_strategy via ``min_top_k_hint``.
"""

from __future__ import annotations

import torch

from openpi.cache.components.factors._debug import VERDICT_DEBUG, verdict_logger
from openpi.cache.components.factors.base import FactorContext
from openpi.cache.components.factors.registry import register

# Per-DOF candidate-pool variance below this threshold is treated as
# "constant across pool" (Pi0.5 padding). Mirrors the legacy F2 constant.
_F2_CANDIDATE_LOCAL_EPS: float = 1e-8


@register("topk_action_variance")
class TopkActionVariance:
    """Variance over top-K candidate ``action_chunk[0]`` vectors."""

    requires_chain_walk: bool = False
    descriptor_orientations: dict[str, str]
    required_top_k: int

    def __init__(self, *, K: int) -> None:
        if K < 2:
            raise ValueError(f"TopkActionVariance K must be >= 2, got {K}")
        self.K = int(K)
        self.required_top_k = int(K)
        self.descriptor_orientations = self.__class__.describe({"K": K})

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        # Single risky-orientation key regardless of K (the variance is a
        # single scalar per verdict).
        return {"topk_action_variance": "risky"}

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        K_eff = min(self.K, len(ctx.results))
        if K_eff < 2:
            if VERDICT_DEBUG:
                verdict_logger.info(
                    "[vd topk_action_variance bail] K_eff=%d < 2 (results=%d, requested K=%d)",
                    K_eff, len(ctx.results), self.K,
                )
            return {"topk_action_variance": float("nan")}
        ids = [r.id for r in ctx.results[:K_eff]]
        payloads = ctx.view.get_many(ids)
        chunk0 = torch.stack(
            [
                torch.as_tensor(p.action_chunk[0], dtype=torch.float32)
                for p in payloads
            ],
            dim=0,
        )                                                # [K_eff, A]
        var_d = chunk0.var(dim=0, unbiased=False)        # [A]
        active = var_d > _F2_CANDIDATE_LOCAL_EPS
        if not bool(active.any()):
            if VERDICT_DEBUG:
                verdict_logger.info(
                    "[vd topk_action_variance bail] candidate pool agreement "
                    "(max var=%.6e <= eps=%.0e); top ids=%s",
                    float(var_d.max()), _F2_CANDIDATE_LOCAL_EPS, ids[:3],
                )
            return {"topk_action_variance": float("nan")}
        return {"topk_action_variance": float(var_d[active].mean())}
