"""Stage I/O helpers — pure functions for stack/split of stage{1,2,3} outputs.

Used by the BatchingCoordinator (M1) to combine per-request unbatched
observations into a batched forward call, and to split the batched outputs
back into per-request slices for the post-stage CPU work.

Contracts (plan §4.1.1 / §4.1.2 / §4.1.3, G1 R2 Item 2)
-------------------------------------------------------
* ``stack_observation`` consumes **unbatched** per-request obs (no batch dim)
  and returns a batched dict with leaf shape ``[N, ...]``. The Interceptor
  MUST NOT pre-add a batch dimension before submitting — double-batching
  would emit ``[N, 1, ...]`` and break ``run_stage1``.
* ``stack_stage{1,2}_output`` concatenate per-request outputs along dim 0,
  expecting each input to already carry a unit batch dimension ``[1, ...]``
  (as produced by ``split_stage{1,2}_output``).
* ``split_stage{1,2}_output`` slice the batched output along dim 0 into a
  list of length N. For Stage2, ``DynamicCache`` is split with
  ``DynamicCache.batch_split`` and each shard's key/value tensors are
  cloned to defeat view-aliasing so downstream stage3 forwards on one
  shard cannot leak into another.
* ``split_stage3_intermediates`` slices the optional MISS-only intermediates
  dict ``{t: [B, ...]}`` into N per-request dicts ``[1, ...]``.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import numpy as np
import torch

from openpi.models_pytorch.pi0_pytorch import (
    Stage1Output,
    Stage2Output,
    Stage3Output,
)


# ------------------------------------------------------------------
# Observation (Stage1 input)
# ------------------------------------------------------------------


def stack_observation(obs_list: Sequence[Any], device: str | torch.device) -> Any:
    """Stack N unbatched obs dicts/trees into one batched obs.

    Args:
        obs_list: Sequence of N obs (each a nested dict / list / tuple of
            numpy arrays or tensors with NO batch dim).
        device: Target device for the stacked tensors.

    Returns:
        A nested structure (same shape as one element of ``obs_list``) whose
        leaves are ``torch.Tensor`` of shape ``[N, ...]`` on ``device``.

    Raises:
        ValueError: when ``obs_list`` is empty.
    """
    if not obs_list:
        raise ValueError("stack_observation: obs_list must be non-empty")

    device = torch.device(device)

    def _stack_leaf(*xs):
        # Each x may be np.ndarray, torch.Tensor, or scalar — normalise then stack.
        tensors = [torch.as_tensor(np.asarray(x) if isinstance(x, np.ndarray) else x)
                   for x in xs]
        stacked = torch.stack(tensors, dim=0)
        return stacked.to(device)

    return jax.tree.map(_stack_leaf, *obs_list)


# ------------------------------------------------------------------
# Stage1Output
# ------------------------------------------------------------------


def stack_stage1_output(out_list: Sequence[Stage1Output]) -> Stage1Output:
    """Concatenate N ``[1, ...]`` Stage1Outputs along batch dim 0.

    Each field of ``Stage1Output`` is a tensor with leading batch dim. We use
    ``torch.cat`` (rather than ``stack``) so the inputs and outputs both have
    a leading batch dim — ``stack_observation`` adds the batch dim once at the
    start of the pipeline.
    """
    if not out_list:
        raise ValueError("stack_stage1_output: out_list must be non-empty")
    return Stage1Output(
        state=torch.cat([o.state for o in out_list], dim=0),
        prefix_embs=torch.cat([o.prefix_embs for o in out_list], dim=0),
        prefix_pad_masks=torch.cat([o.prefix_pad_masks for o in out_list], dim=0),
        prefix_att_2d_masks_4d=torch.cat([o.prefix_att_2d_masks_4d for o in out_list], dim=0),
        prefix_position_ids=torch.cat([o.prefix_position_ids for o in out_list], dim=0),
    )


def split_stage1_output(out: Stage1Output, n: int) -> list[Stage1Output]:
    """Slice a batched Stage1Output into N ``[1, ...]`` shards."""
    return [
        Stage1Output(
            state=out.state[i:i + 1],
            prefix_embs=out.prefix_embs[i:i + 1],
            prefix_pad_masks=out.prefix_pad_masks[i:i + 1],
            prefix_att_2d_masks_4d=out.prefix_att_2d_masks_4d[i:i + 1],
            prefix_position_ids=out.prefix_position_ids[i:i + 1],
        )
        for i in range(n)
    ]


# ------------------------------------------------------------------
# Stage2Output (incl. DynamicCache past_key_values)
# ------------------------------------------------------------------


def _stack_dynamic_cache(caches: Sequence[Any]) -> Any:
    """Concatenate a list of N HuggingFace DynamicCache instances (or
    tuple-of-tuples legacy format) along the batch dim.

    Both representations: per-layer K, V tensors of shape
    ``[B, num_heads, L, head_dim]``. Stacking concatenates along dim 0.

    Returns the same wrapper type as the input (DynamicCache or tuple).
    """
    if not caches:
        raise ValueError("_stack_dynamic_cache: caches must be non-empty")

    # Prefer the official DynamicCache helper when available.
    first = caches[0]
    if hasattr(first, "from_batch_splits"):
        # All inputs must be DynamicCache. ``from_batch_splits`` expects
        # a list of independent caches and rebuilds a merged one. Keeps
        # the original wrapper semantics (refcount, seen_tokens, ...).
        return type(first).from_batch_splits(list(caches))

    # Legacy tuple-of-tuples fallback (older transformers): each cache is
    # ``tuple[ tuple[K_layer, V_layer], ... ]`` per layer.
    num_layers = len(first)
    merged_layers = []
    for layer_idx in range(num_layers):
        keys = torch.cat([c[layer_idx][0] for c in caches], dim=0)
        values = torch.cat([c[layer_idx][1] for c in caches], dim=0)
        merged_layers.append((keys, values))
    return tuple(merged_layers)


def _split_dynamic_cache(cache: Any, n: int) -> list[Any]:
    """Split a batched DynamicCache (or tuple-of-tuples) into N independent
    shards of batch size 1.

    Per-layer key/value tensors are cloned so each shard owns independent
    storage — downstream stage3 forwards on one shard cannot pollute another
    via view aliasing (G1 R2 Item 2 risk register).
    """
    if hasattr(cache, "batch_split"):
        # DynamicCache.batch_split(full_batch_size, split_size) returns a list
        # of caches with split_size each. We want N shards of size 1.
        shards = cache.batch_split(full_batch_size=n, split_size=1)
        # Clone per-layer K/V to break view aliasing on the underlying tensors.
        for shard in shards:
            shard.key_cache = [t.clone() for t in shard.key_cache]
            shard.value_cache = [t.clone() for t in shard.value_cache]
        return list(shards)

    # Tuple-of-tuples fallback: each layer K/V is [B, H, L, D]; slice along
    # dim 0 + clone.
    shards: list[Any] = []
    for i in range(n):
        layers = []
        for layer in cache:
            k, v = layer
            layers.append((k[i:i + 1].clone(), v[i:i + 1].clone()))
        shards.append(tuple(layers))
    return shards


def _stack_prefix_out(out_list: Sequence[Stage2Output]) -> torch.Tensor | None:
    """Batch the optional CP2 ``prefix_out``: all-None -> None, all-set -> cat.

    A mixed batch is a programming error (a capture decision is made once per
    batch, see ``BatchingCoordinator._run_batch``), so it fails loud instead of
    silently dropping some requests' keys.
    """
    outs = [o.prefix_out for o in out_list]
    if all(p is None for p in outs):
        return None
    if any(p is None for p in outs):
        raise ValueError(
            "stack_stage2_output: prefix_out must be all-None or all-set across the batch"
        )
    return torch.cat(outs, dim=0)


def stack_stage2_output(out_list: Sequence[Stage2Output]) -> Stage2Output:
    """Concatenate N Stage2Outputs along batch dim 0 (incl. DynamicCache and prefix_out)."""
    if not out_list:
        raise ValueError("stack_stage2_output: out_list must be non-empty")
    return Stage2Output(
        stage1=stack_stage1_output([o.stage1 for o in out_list]),
        past_key_values=_stack_dynamic_cache([o.past_key_values for o in out_list]),
        prefix_out=_stack_prefix_out(out_list),
    )


def split_stage2_output(out: Stage2Output, n: int) -> list[Stage2Output]:
    """Slice a batched Stage2Output into N shards (incl. DynamicCache and prefix_out split)."""
    stage1_shards = split_stage1_output(out.stage1, n)
    cache_shards = _split_dynamic_cache(out.past_key_values, n)
    if out.prefix_out is None:
        prefix_shards: list[torch.Tensor | None] = [None] * len(stage1_shards)
    else:
        # Mirror the stage1 shard batch sizes so prefix_out stays aligned with
        # the request order whatever split policy stage1 used.
        sizes = [s1.state.shape[0] for s1 in stage1_shards]
        prefix_shards = list(torch.split(out.prefix_out, sizes, dim=0))
    return [
        Stage2Output(stage1=s1, past_key_values=kv, prefix_out=po)
        for s1, kv, po in zip(stage1_shards, cache_shards, prefix_shards)
    ]


# ------------------------------------------------------------------
# Stage3Output intermediates split (MISS path only)
# ------------------------------------------------------------------


def split_stage3_intermediates(
    intermediates: dict[float, torch.Tensor] | None, n: int,
) -> list[dict[float, torch.Tensor] | None]:
    """Slice the optional MISS-path intermediates ``{t: [B, ...]}`` into N
    per-request dicts of shape ``[1, ...]``.

    Returns ``[None] * n`` when ``intermediates is None`` (WARM_START path
    or callers that opted out of ``return_intermediates``).
    """
    if intermediates is None:
        return [None] * n
    return [
        {t: tensor[i:i + 1] for t, tensor in intermediates.items()}
        for i in range(n)
    ]
