"""ROUND 4 build-segment optimization — bit-equal batched-avgpool key builder.

Drop-in subclass of the src ``CP1SpatialPool16KeyBuilder`` that replaces the
vision-pool path. ZERO src change — only ``build()`` is overridden in this exp
subclass; ``collect`` / ``clear`` / ``cached_data`` / ``_enabled`` lifecycle and
the slice/robot_state/prompt logic are inherited unchanged.

What changes vs src ``_spatial_pool_tokens`` (key_builder.py:192):
  src  : per vision field, adaptive_avg_pool2d on a NON-contiguous [1,2048,16,16]
         NCHW view (the permute(2,0,1)); then per-field .cpu().float().contiguous().
  here : stack the two active vision fields -> [2,256,2048], ONE
         avg_pool2d(kernel=4,stride=4) on the batched [2,2048,16,16] view, then
         ONE .cpu().float().contiguous() D2H for both, then split.

Why it is correct (bit-equal): for fixed 16->4 with no remainder,
adaptive_avg_pool2d == avg_pool2d(kernel=4,stride=4) (verified torch.equal=True
on 60 real libero_10 tokens, CPU and CUDA, max|diff|=0.0). Stacking does not
change per-field arithmetic. Output flat layout (oh,ow,c) is preserved by the
same squeeze/permute(...,2,0)/reshape, so query keys are byte-identical ->
retrieval cosine, fusion, and the 3-way verdict are unchanged by construction.

Why it is faster (measured t=4, 60-token median, per build call, 2 vision):
  src adaptive (2x pool + 2x D2H)  ~1.90 ms
  batched avgpool (1x pool + 1 D2H) ~0.47 ms   (~75% on vision pool work)
The win has three independent sources: (1) avg_pool2d drops the adaptive
output-grid computation; (2) batching halves Python/kernel dispatch AND the D2H
count; (3) the batched [2,2048,16,16] non-contiguous view dispatches to a far
faster CPU pooling path than the single [1,2048,16,16] view (a torch-internal
heuristic; see r4_pool_micro.py). On CUDA the win is the single D2H (production
keeps the irreducible ~0.12 ms transfer for 2x[32768]).

NOTE on portability: lever (1) and (2) are portable bench+production. Lever (3)
(the single-vs-batched non-contiguous pool-kernel gap) is a CPU-only,
version-sensitive artifact — it helps the CPU bench a lot but is not the source
of the production (GPU) win, which comes from (1)+(2)+single-D2H. Bit-equality
is independent of all three and holds on both devices.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from openpi.cache.components.key_builder import (
    CP1SpatialPool16KeyBuilder,
    _mean_pool_tokens,
    _to_cpu_float32,
)
from openpi.cache.types import (
    PROMPT_EMB,
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
    CheckpointID,
)

_VISION_FIELDS = (VISION_0, VISION_1, VISION_2)


def _batched_spatial_pool(tokens_stack: torch.Tensor, grid: int, pool: int) -> torch.Tensor:
    """[K, grid*grid, emb] -> [K, pool*pool*emb], bit-equal to per-field src pool.

    avg_pool2d(kernel=grid//pool, stride=grid//pool) == adaptive_avg_pool2d to
    (pool,pool) for the integer-divisible case (grid % pool == 0). Output flat
    order (oh, ow, c) matches src _spatial_pool_tokens exactly.
    """
    k = grid // pool
    emb = tokens_stack.shape[-1]
    x = tokens_stack.reshape(-1, grid, grid, emb).permute(0, 3, 1, 2)  # [K, emb, grid, grid]
    pooled = F.avg_pool2d(x, kernel_size=k, stride=k)                  # [K, emb, pool, pool]
    return pooled.permute(0, 2, 3, 1).reshape(pooled.shape[0], -1)     # [K, pool*pool*emb]


class CP1SpatialPool16BatchedKeyBuilder(CP1SpatialPool16KeyBuilder):
    """ROUND 4 build-optimized spatial-pool-16 key builder.

    Bit-equivalent to ``CP1SpatialPool16KeyBuilder`` but pools all enabled vision
    fields in one batched avg_pool2d + one D2H. Falls back to the inherited
    per-field path if 0/1 vision fields are enabled (nothing to batch).
    """

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        if checkpoint_id not in (CheckpointID.CP1, CheckpointID.CP3):
            raise ValueError(f"Unsupported checkpoint_id: {checkpoint_id}")

        # Routed through the base class's overridable hook rather than calling
        # the module-level slicer: a subclass that lays its prefix out
        # differently (the GR00T builders locate image tokens by mask, not by
        # fixed offset) would otherwise be sliced with Pi0.5's offsets while
        # believing it had inherited the batched build. Identical result here.
        raw = self._slice()
        keys: dict[str, torch.Tensor] = {}

        active_vision = [f for f in _VISION_FIELDS if f in raw]
        if len(active_vision) >= 2:
            # Fail loud if a camera isn't the expected grid*grid token count — reshape
            # would otherwise silently mis-shape a non-256-token vision field.
            for f in active_vision:
                assert raw[f].shape[0] == self._GRID_SIZE ** 2, (
                    f"batched pool expects {self._GRID_SIZE ** 2}-token vision, "
                    f"got {tuple(raw[f].shape)} for {f}"
                )
            # Batched path: stack [K,256,emb] -> one pool -> one D2H -> split.
            stack = torch.stack([raw[f] for f in active_vision], dim=0)
            pooled = _batched_spatial_pool(stack, self._GRID_SIZE, self._POOL_SIZE)
            pooled_cpu = _to_cpu_float32(pooled)  # single D2H for all vision fields
            for i, f in enumerate(active_vision):
                keys[f] = pooled_cpu[i].contiguous()
        else:
            # 0 or 1 vision field: nothing to batch, use inherited per-field reduce.
            for f in active_vision:
                keys[f] = _to_cpu_float32(self._reduce_vision(raw[f]))

        if PROMPT_EMB in raw:
            keys[PROMPT_EMB] = _to_cpu_float32(_mean_pool_tokens(raw[PROMPT_EMB]))

        if ROBOT_STATE in raw:
            keys[ROBOT_STATE] = _to_cpu_float32(raw[ROBOT_STATE])

        return keys
