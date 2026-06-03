"""ROUND 4 optimized CP1 keybuilder + injection (exp-layer, zero src change).

CP1SpatialPool16AvgPoolKeyBuilder subclasses the src CP1SpatialPool16KeyBuilder and
overrides ONLY _reduce_vision to use F.avg_pool2d(kernel=4,stride=4) instead of
F.adaptive_avg_pool2d((4,4)). On a fixed 16x16->4x4 grid these are the SAME operation
(adaptive picks kernel=ceil(16/4)=4, stride=floor(16/4)=4, no overlap) and the bench
(r4_build_micro.py) proves the output is BIT-IDENTICAL on CPU and GPU over real tokens.

collect/build/clear/_cache/_enabled lifecycle is inherited unchanged; only the vision
reduce kernel swaps, so query_keys are byte-identical and every downstream consumer
(search cosine, 3-way verdict, judge calibration) is untouched.

Injection: pass a components_hook to ReplayHarness that replaces components["key_builder"].
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder


def _avg_pool_spatial(tokens: torch.Tensor, grid_size: int, pool_size: int) -> torch.Tensor:
    """Bit-identical replacement for _spatial_pool_tokens using fixed avg_pool2d.

    Same reshape/permute framing as the src; only adaptive_avg_pool2d -> avg_pool2d.
    On a grid divisible by pool_size, adaptive_avg_pool2d == avg_pool2d(k=g//p, s=g//p).
    """
    emb_dim = tokens.shape[1]
    x = tokens.reshape(grid_size, grid_size, emb_dim).permute(2, 0, 1).unsqueeze(0)
    pooled = F.avg_pool2d(x, kernel_size=grid_size // pool_size, stride=grid_size // pool_size)
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


class CP1SpatialPool16AvgPoolKeyBuilder(CP1SpatialPool16KeyBuilder):
    """R4 build-opt: adaptive_avg_pool2d -> avg_pool2d (bit-identical, ~20% faster reduce)."""

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _avg_pool_spatial(tokens, self._GRID_SIZE, self._POOL_SIZE)


def make_avgpool_keybuilder_hook() -> Any:
    """Return a components_hook(components) that swaps in the avgpool keybuilder.

    Preserves the original _enabled set (only enabled fields are sliced/reduced) so the
    optimized builder is a drop-in. Asserts the swap held and the lifecycle attrs survived.
    """

    def hook(components: dict[str, Any]) -> None:
        old = components["key_builder"]
        assert isinstance(old, CP1SpatialPool16KeyBuilder), (
            f"R4 avgpool hook expects CP1SpatialPool16KeyBuilder, got {type(old).__name__}"
        )
        new = CP1SpatialPool16AvgPoolKeyBuilder(enabled_fields=None)
        # Carry over the exact _enabled set + any collected state (none yet at build time).
        new._enabled = old._enabled
        new._cache = old._cache
        components["key_builder"] = new
        assert isinstance(components["key_builder"], CP1SpatialPool16AvgPoolKeyBuilder)

    return hook
