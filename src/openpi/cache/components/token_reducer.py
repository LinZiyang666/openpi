"""Pluggable token reduction strategies for pruned vision tokens.

Receives PruneResult from Step 1 (temporal pruning) and compresses
the remaining token set into a fixed-dimension key vector for cache lookup.

This module is the experimentation hotspot -- implementations are swapped
frequently. The interface (TokenReducer Protocol + PruneResult dataclass)
must remain stable.

Data flow:
  CP1TemporalPruneKeyBuilder._temporal_prune()
    -> PruneResult
    -> TokenReducer.reduce(prune_result, prompt_emb=...)
    -> [output_dim] GPU tensor
    -> KeyBuilder handles CPU transfer

Coupling map:
  DEPENDS ON:  nothing (leaf module)
  CONSUMED BY: CP1TemporalPruneKeyBuilder (key_builder.py)
  IF CHANGED:  output_dim must match backend vector_dims
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

# SigLIP embedding dimension (ViT-So400m/14: hidden_size=1152 projected to 2048
# by multi_modal_projector). All vision tokens share this dimension.
_EMB_DIM = 2048


# ------------------------------------------------------------------
# PruneResult: Step 1 output, Step 2 input
# ------------------------------------------------------------------


@dataclass
class PruneResult:
    """Output of Step 1 (token pruning), input to Step 2 (token reduction).

    Decouples pruning logic from reduction logic. All reducer implementations
    receive this uniform structure regardless of which pruner produced it.
    """

    tokens: torch.Tensor
    """[W, K, emb_dim] -- pruned token set across W frames."""

    token_indices: torch.Tensor
    """[K] int64 -- positions in the original 256-token grid."""

    pruned: bool
    """False when window < prune_window_size (no pruning was done)."""

    temporal_scores: torch.Tensor | None = None
    """[K] optional temporal change scores, for Step 2 reference."""


# ------------------------------------------------------------------
# TokenReducer Protocol
# ------------------------------------------------------------------


@runtime_checkable
class TokenReducer(Protocol):
    """Reduce pruned vision tokens to a fixed-dim key vector.

    Coupling:
      DEPENDS ON:  PruneResult (same file)
      CONSUMED BY: CP1TemporalPruneKeyBuilder.build()
      IF CHANGED:  output_dim must match backend vector_dims
    """

    def reduce(
        self,
        prune_result: PruneResult,
        *,
        prompt_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """PruneResult -> [output_dim] tensor (GPU, not yet transferred to CPU).

        prompt_emb: optional [num_tokens, emb_dim] for task-aware reduction.
        Reducers that don't use it simply ignore the parameter.
        """
        ...

    @property
    def output_dim(self) -> int:
        """Dimension of the output key vector. Must be deterministic."""
        ...


# ------------------------------------------------------------------
# MeanPoolReducer
# ------------------------------------------------------------------


class MeanPoolReducer:
    """Mean over time and token dims. output_dim = emb_dim (2048)."""

    @property
    def output_dim(self) -> int:
        return _EMB_DIM

    def reduce(
        self,
        prune_result: PruneResult,
        *,
        prompt_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [W, K, D] -> mean over W -> [K, D] -> mean over K -> [D]
        return prune_result.tokens.mean(dim=0).mean(dim=0)


# ------------------------------------------------------------------
# MaxPoolReducer
# ------------------------------------------------------------------


class MaxPoolReducer:
    """Mean over time, then per-dim max over tokens. output_dim = emb_dim (2048)."""

    @property
    def output_dim(self) -> int:
        return _EMB_DIM

    def reduce(
        self,
        prune_result: PruneResult,
        *,
        prompt_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [W, K, D] -> mean over W -> [K, D] -> max over K -> [D]
        return prune_result.tokens.mean(dim=0).max(dim=0).values


# ------------------------------------------------------------------
# SpatialPoolReducer
# ------------------------------------------------------------------


class SpatialPoolReducer:
    """Fill pruned tokens back into 16x16 grid, then adaptive avg pool.

    output_dim = output_tokens * emb_dim.
    output_tokens must be a perfect square (4, 16, 64, etc.).
    """

    _GRID_SIZE = 16  # sqrt(256), SigLIP patch grid

    def __init__(self, output_tokens: int = 16):
        if output_tokens < 1:
            raise ValueError(
                f"output_tokens must be >= 1, got {output_tokens}"
            )
        self._output_tokens = output_tokens
        self._pool_size = int(output_tokens**0.5)
        if self._pool_size**2 != output_tokens:
            raise ValueError(
                f"output_tokens must be a perfect square, got {output_tokens}"
            )

    @property
    def output_dim(self) -> int:
        return self._output_tokens * _EMB_DIM

    def reduce(
        self,
        prune_result: PruneResult,
        *,
        prompt_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. time average: [W, K, D] -> [K, D]
        token_means = prune_result.tokens.mean(dim=0)

        # 2. fill back into 16x16 grid using token_indices
        D = token_means.shape[-1]
        grid = torch.zeros(
            self._GRID_SIZE * self._GRID_SIZE,
            D,
            device=token_means.device,
            dtype=token_means.dtype,
        )
        grid[prune_result.token_indices] = token_means
        # -> [16, 16, D] -> [1, D, 16, 16]
        grid = grid.reshape(self._GRID_SIZE, self._GRID_SIZE, D)
        grid = grid.permute(2, 0, 1).unsqueeze(0)

        # 3. adaptive avg pool -> [1, D, pool, pool]
        pooled = F.adaptive_avg_pool2d(grid, (self._pool_size, self._pool_size))

        # 4. flatten -> [output_tokens * D]
        return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


# ------------------------------------------------------------------
# TaskScoringReducer
# ------------------------------------------------------------------


class TaskScoringReducer:
    """Select top-K by task relevance, then weighted pool.

    Uses cos(token, prompt_emb) to score each token's relevance to
    the language instruction, selects top-K, and produces a weighted
    softmax pool. Suitable for mixed-task databases where task signal
    helps distinguish relevant vs irrelevant dynamic tokens.

    Falls back to mean pool when prompt_emb is None or empty.
    """

    def __init__(self, select_k: int = 32, temperature: float = 1.0):
        if select_k < 1:
            raise ValueError(f"select_k must be >= 1, got {select_k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self._select_k = select_k
        self._temperature = temperature

    @property
    def output_dim(self) -> int:
        return _EMB_DIM

    def reduce(
        self,
        prune_result: PruneResult,
        *,
        prompt_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. time average: [W, K, D] -> [K, D]
        token_means = prune_result.tokens.mean(dim=0)

        if prompt_emb is None or prompt_emb.shape[0] == 0:
            # fallback: no task signal, plain mean pool
            return token_means.mean(dim=0)

        # 2. task scoring: cos(token, mean(prompt_emb))
        token_norm = F.normalize(token_means, dim=-1)
        prompt_vec = F.normalize(prompt_emb.mean(dim=0, keepdim=True), dim=-1)
        task_scores = (token_norm @ prompt_vec.T).squeeze(-1)  # [K]

        # 3. top-K selection
        k = min(self._select_k, token_means.shape[0])
        topk_indices = task_scores.topk(k).indices
        selected = token_means[topk_indices]  # [k, D]
        selected_scores = task_scores[topk_indices]  # [k]

        # 4. weighted softmax pool
        weights = F.softmax(selected_scores / self._temperature, dim=0)
        return (weights.unsqueeze(-1) * selected).sum(dim=0)  # [D]
