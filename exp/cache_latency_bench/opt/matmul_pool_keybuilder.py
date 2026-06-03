"""ROUND 4 build-segment optimization: matmul spatial pool key builder.

The stock ``CP1SpatialPool16KeyBuilder._reduce_vision`` does, per vision field:

    tokens[256,2048].reshape(16,16,2048).permute(2,0,1).unsqueeze(0)
      -> F.adaptive_avg_pool2d((4,4))
      -> .squeeze(0).permute(1,2,0).reshape(-1)   # [32768]

Average-pool over FIXED non-overlapping 4x4 blocks of a 16x16 grid is a LINEAR
operation, so it equals a single sparse pooling matrix multiply::

    P[16, 256] @ tokens[256, 2048] -> [16, 2048] -> reshape(-1) -> [32768]

where ``P[o, p] = 1/16`` iff source patch ``p = r*16 + c`` falls in output block
``o = (r//4)*4 + (c//4)``, else 0. ``reshape(-1)`` over the contiguous ``[16,2048]``
gives output order ``[block, channel]`` row-major, IDENTICAL to the stock
``permute(1,2,0).reshape(-1)`` (``[pool_i, pool_j, channel]`` with block index
``pool_i*4 + pool_j``).

Why it is faster (micro_bench_build2/3.py, threads=4):
  * single GEMM (OpenBLAS/MKL) is ~10x the pooling kernel on this tiny tensor;
  * GEMM writes a contiguous ``[16,2048]`` directly -> no permute, no non-contiguous
    intermediate, no reshape copy. The stock path's two permutes feed a
    non-contiguous tensor into the pool, and the trailing permute->reshape forces
    a copy. matmul sidesteps both.
  * full-build (2 vision pools + 3 _to_cpu_float32) drops ~1.28ms -> ~0.59ms
    (5.7x) at threads=4.

Equivalence: BIT-EXACT (max|diff|=0.0) vs the stock adaptive_avg_pool2d across
200 random seeds x value scales AND on real LIBERO vision_0 tokens
(micro_bench_build3.py). So query keys are byte-identical -> cosine retrieval,
3-way verdict, and warm-tier classification are unchanged. This is NOT a
lossy approximation; it is the same average pool computed by a faster kernel.

Portability (bench CPU vs production GPU):
  * The MATMUL POOL win is PORTABLE to production GPU: cuBLAS GEMM vs the cuDNN/
    ATen adaptive pool kernel on a [256,2048] tensor; the permute-copy avoidance
    matters on GPU too. The pooling matrix P moves to the encoder's device once.
  * The ``_to_cpu_float32`` D2H is NOT optimizable in production (real GPU->CPU
    transfer of the [32768] reduced vector is mandatory and dominates GPU build);
    on the bench it is a no-op (fake stage1 is already CPU float32), so the bench
    build time is pool-bound, which is exactly the part matmul accelerates.

Injection: ``components["key_builder"]`` is replaced via ReplayHarness's
``components_hook``. We copy the stock builder's ``_enabled`` set so field
selection (depth_1.yaml: vision_0 + vision_1 + robot_state) is preserved. Zero
src change; the override touches only ``_reduce_vision``.
"""

from __future__ import annotations

import torch

from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder


def _build_pool_matrix(grid_size: int, pool_size: int) -> torch.Tensor:
    """Sparse average-pool matrix P[pool^2, grid^2] for fixed non-overlapping blocks.

    P[o, p] = 1/(block_area) iff patch p=(r*grid+c) is in output block
    o=(r//k)*pool + (c//k), where k = grid//pool. Output index order matches the
    stock permute(1,2,0).reshape(-1): block index = pool_i*pool + pool_j.
    """
    assert grid_size % pool_size == 0, "grid must divide evenly into pool blocks"
    k = grid_size // pool_size
    area = float(k * k)
    P = torch.zeros(pool_size * pool_size, grid_size * grid_size, dtype=torch.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            o = (r // k) * pool_size + (c // k)
            P[o, r * grid_size + c] = 1.0 / area
    return P.contiguous()


class CP1SpatialPool16MatmulKeyBuilder(CP1SpatialPool16KeyBuilder):
    """Bit-exact, faster drop-in for CP1SpatialPool16KeyBuilder.

    Replaces adaptive_avg_pool2d + double-permute with one P @ tokens GEMM.
    The pooling matrix is precomputed once at construction (16x256, ~16 KB).
    """

    def __init__(self, enabled_fields=None) -> None:
        super().__init__(enabled_fields=enabled_fields)
        # Precompute the sparse pooling matrix once. _GRID_SIZE=16, _POOL_SIZE=4.
        self._pool_matrix = _build_pool_matrix(self._GRID_SIZE, self._POOL_SIZE)

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [256, emb_dim] on the encoder device. P @ tokens -> [16, emb_dim]
        # contiguous; reshape(-1) -> [16*emb_dim]. Identical output to the stock
        # path's permute(1,2,0).reshape(-1). In production the matrix is moved to
        # the token device once (lazy, cached) so the GEMM stays on-device.
        P = self._pool_matrix
        if P.device != tokens.device or P.dtype != tokens.dtype:
            # Production GPU path: lazily migrate/cast the pooling matrix to match
            # the encoder output. Bench stays CPU float32 (no-op).
            P = P.to(device=tokens.device, dtype=tokens.dtype)
            self._pool_matrix = P
        return (P @ tokens).reshape(-1)
