"""Build InMemoryBackend artifact from HDF5 episode data.

Usage:
    uv run exp/common/build_in_memory_cache_artifact.py \
        --data-dir exp/common/data/libero_spatial \
        --builder-type cp1_mean_pool \
        --output exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool.pkl

    # Build all 4 artifacts at once:
    for bt in cp1_mean_pool cp1_spatial_pool_16 cp1_spatial_pool_64 cp1_max_pool; do
        uv run exp/common/build_in_memory_cache_artifact.py \
            --data-dir exp/common/data/libero_spatial \
            --builder-type $bt \
            --output exp/common/data/cache_artifacts/libero_spatial/${bt}.pkl
    done

    # B2: enrich with verdict-factor `payload.factors` + top-level
    # `library_stats` by passing a minimal YAML listing F1b factors to
    # --factors-yaml. See `enrich_artifact_with_factors` in
    # exp/common/factor_postprocess.py for the YAML shape.

Input: HDF5 episode files with vision_0/1/2, prompt_emb, robot_state fields
Output: pickle file loadable by InMemoryBackend.load_artifact()
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vector dims per builder type
# ---------------------------------------------------------------------------

_VECTOR_DIMS: dict[str, dict[str, int]] = {
    "cp1_mean_pool":       {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32},
    "cp1_spatial_pool_16": {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32},
    # cp1_spatial_pool_4: canonical name (4 output tokens, 2x2 pool).
    "cp1_spatial_pool_4":  {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32},
    # Backward-compat alias of cp1_spatial_pool_4 (old `_64` = 64x compression).
    "cp1_spatial_pool_64": {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32},
    "cp1_max_pool":        {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32},
}

# cp1_llm_layer_extract dims depend on prefix_reducer. Vision dim varies
# per reducer; prompt_emb is always 2048 (mean-pooled fallback because the
# lang segment is variable-length and not square-shaped).
_LLM_LAYER_EXTRACT_DIMS: dict[str, dict[str, int]] = {
    "prefix_mean_pool":  {"vision_0": 2048, "robot_state": 32},
    "per_modality_mean_pool": {"vision_0": 2048, "vision_1": 2048,
                           "vision_2": 2048, "prompt_emb": 2048,
                           "robot_state": 32},
    "per_modality_max_pool": {"vision_0": 2048, "vision_1": 2048,
                               "vision_2": 2048, "prompt_emb": 2048,
                               "robot_state": 32},
    # vision: 16 tokens (4x4) * 2048 = 32768; prompt_emb: 2048 (masked mean fallback).
    "per_modality_spatial_pool_16": {"vision_0": 32768, "vision_1": 32768,
                                      "vision_2": 32768, "prompt_emb": 2048,
                                      "robot_state": 32},
    # vision: 4 tokens (2x2) * 2048 = 8192; prompt_emb: 2048.
    "per_modality_spatial_pool_4":  {"vision_0": 8192, "vision_1": 8192,
                                      "vision_2": 8192, "prompt_emb": 2048,
                                      "robot_state": 32},
}


def _build_artifact_reducer(
    reducer_type: str,
    output_tokens: int = 16,
    select_k: int = 32,
    temperature: float = 1.0,
):
    """Build a TokenReducer for artifact generation (no config.py dependency)."""
    from openpi.cache.components.token_reducer import (
        MaxPoolReducer,
        MeanPoolReducer,
        SpatialPoolReducer,
        TaskScoringReducer,
    )

    if reducer_type == "mean_pool":
        return MeanPoolReducer()
    elif reducer_type == "max_pool":
        return MaxPoolReducer()
    elif reducer_type == "spatial_pool":
        return SpatialPoolReducer(output_tokens=output_tokens)
    elif reducer_type == "task_scoring":
        return TaskScoringReducer(select_k=select_k, temperature=temperature)
    else:
        raise ValueError(f"Unknown reducer_type: {reducer_type}")


# Builder types that use dynamic vector dims (not in _VECTOR_DIMS)
_ALL_BUILDER_TYPES = list(_VECTOR_DIMS.keys()) + ["cp1_temporal_prune", "cp1_llm_layer_extract"]


def _get_vector_dims(
    builder_type: str,
    reducer_type: str = "mean_pool",
    output_tokens: int = 16,
    prefix_reducer_type: str = "prefix_mean_pool",
) -> dict[str, int]:
    if builder_type in _VECTOR_DIMS:
        return _VECTOR_DIMS[builder_type]
    if builder_type == "cp1_temporal_prune":
        reducer = _build_artifact_reducer(reducer_type, output_tokens)
        vision_dim = reducer.output_dim
        return {
            "vision_0": vision_dim, "vision_1": vision_dim,
            "prompt_emb": 2048, "robot_state": 32,
        }
    if builder_type == "cp1_llm_layer_extract":
        if prefix_reducer_type not in _LLM_LAYER_EXTRACT_DIMS:
            raise ValueError(
                f"Unknown prefix_reducer_type: {prefix_reducer_type}. "
                f"Valid: {sorted(_LLM_LAYER_EXTRACT_DIMS.keys())}"
            )
        return _LLM_LAYER_EXTRACT_DIMS[prefix_reducer_type]
    raise ValueError(f"Unknown builder_type: {builder_type}")


# ---------------------------------------------------------------------------
# Builder factory
# ---------------------------------------------------------------------------


def _create_builder(
    builder_type: str,
    reducer_type: str = "mean_pool",
    output_tokens: int = 16,
    prune_window_size: int = 4,
    temporal_keep_ratio: float = 0.5,
    select_k: int = 32,
    temperature: float = 1.0,
    extract_layer: int = 0,
    prefix_reducer_type: str = "prefix_mean_pool",
):
    from openpi.cache.components.key_builder import (
        CP1MaxPoolKeyBuilder,
        CP1MeanPoolKeyBuilder,
        CP1SpatialPool16KeyBuilder,
        CP1SpatialPool64KeyBuilder,
    )

    builders = {
        "cp1_mean_pool": CP1MeanPoolKeyBuilder,
        "cp1_spatial_pool_16": CP1SpatialPool16KeyBuilder,
        "cp1_spatial_pool_64": CP1SpatialPool64KeyBuilder,
        "cp1_max_pool": CP1MaxPoolKeyBuilder,
    }
    if builder_type in builders:
        return builders[builder_type]()
    if builder_type == "cp1_temporal_prune":
        from openpi.cache.components.key_builder import CP1TemporalPruneKeyBuilder

        reducer = _build_artifact_reducer(reducer_type, output_tokens, select_k, temperature)
        return CP1TemporalPruneKeyBuilder(
            reducer=reducer,
            prune_window_size=prune_window_size,
            temporal_keep_ratio=temporal_keep_ratio,
        )
    if builder_type == "cp1_llm_layer_extract":
        from openpi.cache.components.llm_layer_key_builder import (
            CP1LLMLayerExtractKeyBuilder,
        )
        from openpi.cache.components.prefix_reducer import (
            PerModalityMaxPoolReducer,
            PerModalityMeanPoolReducer,
            PerModalitySpatialPoolReducer,
            PrefixMeanPoolReducer,
        )

        if prefix_reducer_type == "prefix_mean_pool":
            prefix_reducer = PrefixMeanPoolReducer()
        elif prefix_reducer_type == "per_modality_mean_pool":
            prefix_reducer = PerModalityMeanPoolReducer()
        elif prefix_reducer_type == "per_modality_max_pool":
            prefix_reducer = PerModalityMaxPoolReducer()
        elif prefix_reducer_type == "per_modality_spatial_pool_16":
            prefix_reducer = PerModalitySpatialPoolReducer(output_tokens=16)
        elif prefix_reducer_type == "per_modality_spatial_pool_4":
            prefix_reducer = PerModalitySpatialPoolReducer(output_tokens=4)
        else:
            raise ValueError(
                f"Unknown prefix_reducer_type: {prefix_reducer_type}. "
                f"Valid: {sorted(_LLM_LAYER_EXTRACT_DIMS.keys())}"
            )
        # Caller must invoke `attach_model(model)` before the first build().
        return CP1LLMLayerExtractKeyBuilder(
            reducer=prefix_reducer,
            extract_layer=extract_layer,
        )
    raise ValueError(f"Unknown builder_type: {builder_type}. Valid: {_ALL_BUILDER_TYPES}")


# ---------------------------------------------------------------------------
# Fake stage1 output
# ---------------------------------------------------------------------------


class _FakeStage1:
    """Mimics Stage1Output structure for KeyBuilder.collect().

    The four mask-related fields are required by `cp1_llm_layer_extract`
    (which needs them to replay layer-N attention) and ignored by all
    SigLIP-only builders. All fields default to None for backward compat
    with the legacy `_build_fake_stage1` path.
    """

    def __init__(
        self,
        prefix_embs: torch.Tensor,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor | None = None,
        prefix_position_ids: torch.Tensor | None = None,
        prefix_att_2d_masks_4d: torch.Tensor | None = None,
    ):
        self.prefix_embs = prefix_embs                     # [1, prefix_len, emb_dim]
        self.state = state                                 # [1, state_dim]
        self.prefix_pad_masks = prefix_pad_masks           # [1, prefix_len] bool
        self.prefix_position_ids = prefix_position_ids     # [1, prefix_len] int64
        self.prefix_att_2d_masks_4d = prefix_att_2d_masks_4d  # [1, 1, prefix_len, prefix_len]


def _build_fake_stage1(group: h5py.Group) -> _FakeStage1:
    """Reconstruct prefix_embs from HDF5 step group.

    HDF5 fields:
      vision_0: [256, 2048], vision_1: [256, 2048],
      vision_2: [256, 2048] (may not exist),
      prompt_emb: [num_tokens, 2048], robot_state: [32]

    Rebuilds prefix_embs = concat([vision_0, vision_1, vision_2, prompt_emb])
    with batch dim. vision_2 zero-filled if absent (maintains token offsets).
    """
    parts = []
    for vfield in ("vision_0", "vision_1", "vision_2"):
        if vfield in group:
            parts.append(torch.from_numpy(np.array(group[vfield])).float())
        else:
            emb_dim = parts[0].shape[1] if parts else 2048
            parts.append(torch.zeros(256, emb_dim))

    prompt = torch.from_numpy(np.array(group["prompt_emb"])).float()
    parts.append(prompt)

    prefix_embs = torch.cat(parts, dim=0).unsqueeze(0)  # [1, prefix_len, emb_dim]
    state = torch.from_numpy(np.array(group["robot_state"])).float()
    if state.dim() == 1:
        state = state.unsqueeze(0)  # [1, state_dim]

    return _FakeStage1(prefix_embs, state)


# ---------------------------------------------------------------------------
# cp1_llm_layer_extract — model-aware Stage1 reconstruction
# ---------------------------------------------------------------------------


# Pi0.5 prefix layout (mirrors src/openpi/cache/components/key_builder.py:142-147).
_VISION_SLOTS = ("vision_0", "vision_1", "vision_2")
_TOKENS_PER_IMAGE = 256
_PI05_TOKENIZER_SOURCE = "gs://big_vision/paligemma_tokenizer.model"
_PI05_TOKENIZER_MAX_LEN = 200


def _load_pi05_for_llm_extract(checkpoint_dir: str, config_name: str, device: str):
    """Load PI0Pytorch model + PaligemmaTokenizer for offline layer-N extraction.

    Returns (model, tokenizer). Both are loaded once per build run; the model
    is the same instance KeyBuilder.attach_model() will borrow layer refs from.
    """
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(config_name)
    logger.info("Loading PI0Pytorch checkpoint from %s on %s", checkpoint_dir, device)
    policy = _policy_config.create_trained_policy(
        train_config, checkpoint_dir, pytorch_device=device,
    )
    model = policy._model  # noqa: SLF001 -- canonical access pattern (interceptor.py:156)
    tokenizer = PaligemmaTokenizer(max_len=_PI05_TOKENIZER_MAX_LEN)
    return model, tokenizer


def _build_fake_stage1_with_masks(
    group: h5py.Group,
    task_str: str,
    tokenizer,
    model,
    device: torch.device,
) -> _FakeStage1:
    """Reconstruct a `Stage1Output`-shaped object from HDF5 for layer-N extract.

    Recovers `lang_masks` deterministically by re-running the tokenizer on
    `task_str + robot_state` (Pi0.5 prompt format from `tokenizer.py:24-29`),
    then synthesizes `prefix_pad_masks`, `prefix_position_ids`, and
    `prefix_att_2d_masks_4d` to match what `embed_prefix()` produces online.
    """
    import openpi.models_pytorch.pi0_pytorch as pi0_module

    # Vision parts: zero-fill any absent camera (sets pad_mask=False below).
    parts = []
    present_cameras: list[bool] = []
    for vfield in _VISION_SLOTS:
        if vfield in group:
            parts.append(torch.from_numpy(np.array(group[vfield])).float())
            present_cameras.append(True)
        else:
            emb_dim = parts[0].shape[1] if parts else 2048
            parts.append(torch.zeros(_TOKENS_PER_IMAGE, emb_dim))
            present_cameras.append(False)

    # prompt_emb is padded to max_token_len at collect time and captured via
    # a hook that already applies the `× sqrt(emb_dim)` scale in-place before
    # store (`collection_policy.py:82-84`), so HDF5 prompt_emb is already what
    # `embed_prefix` would feed to layer 0. No extra scaling needed here.
    prompt = torch.from_numpy(np.array(group["prompt_emb"])).float()
    parts.append(prompt)

    prefix_embs = torch.cat(parts, dim=0).unsqueeze(0).to(device)  # [1, 968, 2048]

    state_np = np.array(group["robot_state"], dtype=np.float32)
    state = torch.from_numpy(state_np).float()
    if state.dim() == 1:
        state = state.unsqueeze(0)
    state = state.to(device)

    # Re-tokenize deterministically. Whether `state` is folded into the discrete
    # prompt depends on `Pi0Config.discrete_state_input`: for Pi0.5 variants
    # where it is False (e.g. `pi05_libero`), prompt is `task + "\n"` only;
    # state goes through the continuous action-expert path, not the prompt.
    # Must match the collector: `TokenizePrompt` in `transforms.py:252-266`
    # reads the same flag to decide whether to pass `state`.
    tok_state = state_np if model.config.discrete_state_input else None
    _, lang_mask_np = tokenizer.tokenize(task_str, state=tok_state)
    lang_mask = torch.from_numpy(lang_mask_np).bool()

    # Vision pad mask: True for present cameras, False for absent ones (the
    # zero-filled segments must not influence attention or masked pooling).
    vision_pad = torch.zeros(len(_VISION_SLOTS) * _TOKENS_PER_IMAGE, dtype=torch.bool)
    for i, present in enumerate(present_cameras):
        if present:
            vision_pad[i * _TOKENS_PER_IMAGE:(i + 1) * _TOKENS_PER_IMAGE] = True

    prefix_pad_masks = torch.cat([vision_pad, lang_mask]).unsqueeze(0).to(device)

    # Position ids: cumulative count over real (unpadded) tokens, zero-indexed
    # — matches `_stage1_token_prep` (`pi0_pytorch.py:478`).
    prefix_position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
    prefix_position_ids = prefix_position_ids.to(device)

    # Prefix-LM attention: att_masks all zero -> full attention among valid
    # tokens, padding columns excluded by the pad mask. Mirrors `embed_prefix`
    # (`pi0_pytorch.py:308, 323`).
    prefix_att_masks = torch.zeros_like(prefix_pad_masks, dtype=torch.bool, device=device)
    prefix_att_2d_masks = pi0_module.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001

    return _FakeStage1(
        prefix_embs=prefix_embs,
        state=state,
        prefix_pad_masks=prefix_pad_masks,
        prefix_position_ids=prefix_position_ids,
        prefix_att_2d_masks_4d=prefix_att_2d_masks_4d,
    )


def _self_check_tokenizer_consistency(
    h5_path: Path, model, tokenizer, device: torch.device
) -> None:
    """Verify tokenizer/embed against HDF5 prompt_emb on the first step.

    Re-tokenizes the first step's task+state, embeds via the loaded model,
    and asserts the result matches the stored `prompt_emb` at non-padding
    positions (bf16 tolerance). Raises with a clear pointer if data and
    model came from different builds.

    HDF5 `prompt_emb` is captured via a hook on `language_model.embed_tokens`
    that already multiplies by `sqrt(emb_dim)` in-place before storing
    (`collection_policy.py:82-84`). The re-computed embedding here therefore
    also applies the same scale to match the stored representation.

    Whether `state` is folded into the discrete prompt follows the same rule
    as the collector: it is keyed off `Pi0Config.discrete_state_input`.
    """
    import math

    with h5py.File(h5_path, "r") as f:
        task_str = str(f.attrs.get("task", ""))
        step_names = sorted(k for k in f.keys() if k.startswith("step_"))
        if not step_names:
            return
        group = f[step_names[0]]
        state_np = np.array(group["robot_state"], dtype=np.float32)
        hdf5_prompt = torch.from_numpy(np.array(group["prompt_emb"])).float()  # [200, 2048]

    tok_state = state_np if model.config.discrete_state_input else None
    lang_tokens_np, lang_mask_np = tokenizer.tokenize(task_str, state=tok_state)
    lang_tokens = torch.from_numpy(lang_tokens_np).long().unsqueeze(0).to(device)
    with torch.no_grad():
        lang_emb = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
    lang_emb = lang_emb[0].float().cpu()                    # [200, 2048]
    mask = torch.from_numpy(lang_mask_np).bool()

    if not mask.any():
        logger.warning("Tokenizer self-check: empty prompt at first step (skipped)")
        return

    diff = (lang_emb[mask] - hdf5_prompt[mask]).abs().max().item()
    if not torch.allclose(lang_emb[mask], hdf5_prompt[mask], rtol=1e-2, atol=1e-2):
        raise RuntimeError(
            f"Tokenizer/embed self-check failed (max abs diff: {diff:.4g}). "
            f"HDF5 prompt_emb at {h5_path} was produced by a different "
            f"tokenizer or model checkpoint than the one loaded for this build. "
            f"Verify --checkpoint-dir matches the data collection model and that "
            f"PaligemmaTokenizer source ({_PI05_TOKENIZER_SOURCE}) is unchanged."
        )
    logger.info("Tokenizer self-check passed (max abs diff: %.4g)", diff)


def _process_episode_with_model(
    h5_path: Path,
    builder,
    model,
    tokenizer,
    device: torch.device,
    checkpoint_id_str: str,
) -> list | None:
    """Process one HDF5 file using a model-attached cp1_llm_layer_extract builder."""
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID

    cp_id = CheckpointID[checkpoint_id_str]

    with h5py.File(h5_path, "r") as f:
        task = str(f.attrs.get("task", ""))
        success = bool(f.attrs.get("success", False))
        if not success:
            return None

        trajectory_id = h5_path.stem

        if hasattr(builder, "on_episode_start"):
            builder.on_episode_start()

        def _step_sort_key(name: str) -> tuple[bool, int, str]:
            suffix = name.split("_", 1)[1] if "_" in name else ""
            if suffix.isdigit():
                return (False, int(suffix), name)
            return (True, 0, name)

        step_names = sorted(
            (k for k in f.keys() if k.startswith("step_")),
            key=_step_sort_key,
        )
        episode_entries: list[CacheEntry] = []
        for step_name in step_names:
            group = f[step_name]

            fake_stage1 = _build_fake_stage1_with_masks(
                group, task_str=task, tokenizer=tokenizer, model=model, device=device,
            )
            builder.collect(cp_id, stage1=fake_stage1)
            query_keys = builder.build(cp_id)
            builder.clear()

            step_idx = None
            suffix = step_name.split("_", 1)[1] if "_" in step_name else ""
            if suffix.isdigit():
                step_idx = int(suffix)

            entry_id = f"{trajectory_id}:{step_idx if step_idx is not None else step_name}"

            action = torch.from_numpy(np.array(group["clean_action"])).float()
            if action.dim() == 1:
                action = action.unsqueeze(0)

            _NUM_STEPS = 10
            intermediates = None
            denoising_num_steps = None
            noise_indices = []
            for k in group.keys():
                if k.startswith("noise_action_"):
                    suffix = k.split("_")[-1]
                    if suffix.isdigit():
                        idx = int(suffix)
                        if 1 <= idx < _NUM_STEPS:
                            noise_indices.append(idx)
            if noise_indices:
                denoising_num_steps = _NUM_STEPS
                intermediates = {}
                for i in sorted(noise_indices):
                    t = round(1.0 - i / _NUM_STEPS, 4)
                    intermediates[t] = torch.from_numpy(
                        np.array(group[f"noise_action_{i}"])
                    ).float()

            payload = CachePayload(
                action_chunk=action,
                task_key=task,
                intermediates=intermediates,
                denoising_num_steps=denoising_num_steps,
            )
            entry = CacheEntry(
                id=entry_id,
                checkpoint_id=cp_id,
                query_keys=query_keys,
                payload=payload,
                step_idx=step_idx,
                trajectory_id=trajectory_id,
            )
            episode_entries.append(entry)

        for i in range(len(episode_entries)):
            if i > 0:
                episode_entries[i].prev_ids = [episode_entries[i - 1].id]
            if i < len(episode_entries) - 1:
                episode_entries[i].next_ids = [episode_entries[i + 1].id]

        _detach_entries(episode_entries)
        return episode_entries


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def _process_episode(
    h5_path_str: str,
    builder_type: str,
    checkpoint_id_str: str,
    reducer_type: str = "mean_pool",
    output_tokens: int = 16,
    prune_window_size: int = 4,
    temporal_keep_ratio: float = 0.5,
    select_k: int = 32,
    temperature: float = 1.0,
) -> list | None:
    """Process a single H5 file in a worker process. Returns list of CacheEntry or None."""
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID

    h5_path = Path(h5_path_str)
    cp_id = CheckpointID[checkpoint_id_str]
    builder = _create_builder(
        builder_type, reducer_type, output_tokens,
        prune_window_size, temporal_keep_ratio,
        select_k, temperature,
    )

    with h5py.File(h5_path, "r") as f:
        task = str(f.attrs.get("task", ""))
        success = bool(f.attrs.get("success", False))
        if not success:
            return None

        trajectory_id = h5_path.stem

        # Notify stateful builders to reset history for this episode
        if hasattr(builder, 'on_episode_start'):
            builder.on_episode_start()

        def _step_sort_key(name: str) -> tuple[bool, int, str]:
            suffix = name.split("_", 1)[1] if "_" in name else ""
            if suffix.isdigit():
                return (False, int(suffix), name)
            return (True, 0, name)

        step_names = sorted(
            (k for k in f.keys() if k.startswith("step_")),
            key=_step_sort_key,
        )
        episode_entries: list[CacheEntry] = []
        for step_name in step_names:
            group = f[step_name]

            fake_stage1 = _build_fake_stage1(group)
            builder.collect(cp_id, stage1=fake_stage1)
            query_keys = builder.build(cp_id)
            builder.clear()

            step_idx = None
            suffix = step_name.split("_", 1)[1] if "_" in step_name else ""
            if suffix.isdigit():
                step_idx = int(suffix)

            entry_id = f"{trajectory_id}:{step_idx if step_idx is not None else step_name}"

            action = torch.from_numpy(np.array(group["clean_action"])).float()
            if action.dim() == 1:
                action = action.unsqueeze(0)

            _NUM_STEPS = 10
            intermediates = None
            denoising_num_steps = None
            noise_indices = []
            for k in group.keys():
                if k.startswith("noise_action_"):
                    suffix = k.split("_")[-1]
                    if suffix.isdigit():
                        idx = int(suffix)
                        if 1 <= idx < _NUM_STEPS:
                            noise_indices.append(idx)
            if noise_indices:
                denoising_num_steps = _NUM_STEPS
                intermediates = {}
                for i in sorted(noise_indices):
                    t = round(1.0 - i / _NUM_STEPS, 4)
                    intermediates[t] = torch.from_numpy(np.array(group[f"noise_action_{i}"])).float()

            payload = CachePayload(
                action_chunk=action,
                task_key=task,
                intermediates=intermediates,
                denoising_num_steps=denoising_num_steps,
            )
            entry = CacheEntry(
                id=entry_id,
                checkpoint_id=cp_id,
                query_keys=query_keys,
                payload=payload,
                step_idx=step_idx,
                trajectory_id=trajectory_id,
            )
            episode_entries.append(entry)

        # Link prev_ids / next_ids within episode
        for i in range(len(episode_entries)):
            if i > 0:
                episode_entries[i].prev_ids = [episode_entries[i - 1].id]
            if i < len(episode_entries) - 1:
                episode_entries[i].next_ids = [episode_entries[i + 1].id]

        # Convert tensors before crossing process boundaries. Returning raw
        # torch.Tensor objects from ProcessPool workers can fail when Python
        # reconstructs shared-memory file descriptors in the parent process.
        _detach_entries(episode_entries)
        return episode_entries


def _detach_entries(entries: list) -> None:
    """Convert all torch tensors in entries to numpy arrays in-place.

    Frees PyTorch allocator memory while keeping data intact for pickling.
    InMemoryBackend.load_artifact() converts them back to torch on load.
    """
    for entry in entries:
        if entry.query_keys:
            entry.query_keys = {
                k: v.numpy() if isinstance(v, torch.Tensor) else v
                for k, v in entry.query_keys.items()
            }
        p = entry.payload
        if p.action_chunk is not None and isinstance(p.action_chunk, torch.Tensor):
            p.action_chunk = p.action_chunk.numpy()
        if p.intermediates:
            p.intermediates = {
                k: v.numpy() if isinstance(v, torch.Tensor) else v
                for k, v in p.intermediates.items()
            }


def build_artifact(
    data_dir: str,
    builder_type: str,
    checkpoint_id_str: str = "CP1",
    workers: int = 0,
    reducer_type: str = "mean_pool",
    output_tokens: int = 16,
    prune_window_size: int = 4,
    temporal_keep_ratio: float = 0.5,
    select_k: int = 32,
    temperature: float = 1.0,
    extract_layer: int = 0,
    prefix_reducer_type: str = "prefix_mean_pool",
    checkpoint_dir: str | None = None,
    config_name: str | None = None,
    device: str = "cuda",
) -> dict:
    """Build artifact dict from HDF5 data.

    Scans data_dir for .h5 files, uses only successful episodes.
    For each step: collect() -> build() -> CacheEntry.
    Entry id = "{episode_file_stem}_{step_name}" (deterministic, traceable).
    action_chunk keeps real horizon from data (no padding to [50,32]).

    Args:
        workers: Number of parallel workers. 0 = all CPUs. Forced to -1
            (serial in-process) for `cp1_llm_layer_extract` since the
            partial paligemma forward must run in the main process where
            the model is loaded once on GPU.
    """
    vector_dims = _get_vector_dims(
        builder_type, reducer_type, output_tokens, prefix_reducer_type,
    )

    h5_paths = sorted(Path(data_dir).rglob("*.h5"))
    if not h5_paths:
        logger.warning("No .h5 files found in %s", data_dir)
        return {"key_builder_type": builder_type, "checkpoint_id": checkpoint_id_str,
                "vector_dims": vector_dims, "entries": []}

    # cp1_llm_layer_extract takes a model-aware code path that cannot be
    # shipped through ProcessPool workers (each would reload the model into
    # VRAM). Force serial mode and route through `_process_episode_with_model`.
    if builder_type == "cp1_llm_layer_extract":
        if checkpoint_dir is None or config_name is None:
            raise ValueError(
                "cp1_llm_layer_extract requires --checkpoint-dir and --config-name"
            )
        if workers != -1:
            logger.info(
                "Forcing workers=-1 for cp1_llm_layer_extract "
                "(model load + GPU forward must stay in main process)"
            )
            workers = -1

        torch_device = torch.device(device)
        model, tokenizer = _load_pi05_for_llm_extract(checkpoint_dir, config_name, device)
        builder = _create_builder(
            builder_type,
            extract_layer=extract_layer,
            prefix_reducer_type=prefix_reducer_type,
        )
        builder.attach_model(model)

        # First-step parity check: ensures HDF5 prompt_emb was produced by
        # the same tokenizer + model checkpoint we just loaded.
        _self_check_tokenizer_consistency(h5_paths[0], model, tokenizer, torch_device)

        entries: list = []
        logger.info("Processing %d H5 files in serial mode (cp1_llm_layer_extract)",
                    len(h5_paths))
        for i, p in enumerate(h5_paths, 1):
            result = _process_episode_with_model(
                p, builder, model, tokenizer, torch_device, checkpoint_id_str,
            )
            if result is not None:
                entries.extend(result)
                del result
            gc.collect()
            if i % 10 == 0 or i == len(h5_paths):
                logger.info("Progress: %d/%d files, %d entries",
                            i, len(h5_paths), len(entries))
        logger.info("Built %d entries for %s from %s",
                    len(entries), builder_type, data_dir)
        return {
            "key_builder_type": builder_type,
            "checkpoint_id": checkpoint_id_str,
            "vector_dims": vector_dims,
            "entries": entries,
            "reducer_params": {
                "extract_layer": extract_layer,
                "prefix_reducer_type": prefix_reducer_type,
                "apply_final_norm": False,
                "checkpoint_dir": str(checkpoint_dir),
                "config_name": config_name,
                "tokenizer_class": "PaligemmaTokenizer",
                "tokenizer_source": _PI05_TOKENIZER_SOURCE,
                "tokenizer_max_len": _PI05_TOKENIZER_MAX_LEN,
            },
        }

    _ep_args = (
        builder_type, checkpoint_id_str,
        reducer_type, output_tokens, prune_window_size, temporal_keep_ratio,
        select_k, temperature,
    )

    entries: list = []

    if workers == -1:
        # Serial mode: run in main process (avoids fork overhead in tests)
        logger.info("Processing %d H5 files in serial mode", len(h5_paths))
        for i, p in enumerate(h5_paths, 1):
            result = _process_episode(str(p), *_ep_args)
            if result is not None:
                entries.extend(result)
                del result
            gc.collect()
            if i % 10 == 0 or i == len(h5_paths):
                logger.info("Progress: %d/%d files, %d entries", i, len(h5_paths), len(entries))
    else:
        num_workers = workers if workers > 0 else (os.cpu_count() or 1)
        logger.info("Processing %d H5 files with %d workers", len(h5_paths), num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(_process_episode, str(p), *_ep_args): p
                for p in h5_paths
            }
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                if result is not None:
                    entries.extend(result)
                if done_count % 10 == 0 or done_count == len(h5_paths):
                    logger.info("Progress: %d/%d files, %d entries", done_count, len(h5_paths), len(entries))

    logger.info("Built %d entries for %s from %s", len(entries), builder_type, data_dir)
    artifact = {
        "key_builder_type": builder_type,
        "checkpoint_id": checkpoint_id_str,
        "vector_dims": vector_dims,
        "entries": entries,
    }
    # Record reducer params for traceability (offline/online consistency audit)
    if builder_type == "cp1_temporal_prune":
        artifact["reducer_params"] = {
            "reducer_type": reducer_type,
            "output_tokens": output_tokens,
            "prune_window_size": prune_window_size,
            "temporal_keep_ratio": temporal_keep_ratio,
            "select_k": select_k,
            "temperature": temperature,
        }
    return artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build InMemoryBackend artifact from HDF5 data")
    parser.add_argument("--data-dir", required=True, help="Directory with .h5 episode files")
    parser.add_argument("--builder-type", required=True, choices=_ALL_BUILDER_TYPES)
    parser.add_argument("--output", required=True, help="Output .pkl path")
    parser.add_argument("--checkpoint-id", default="CP1", choices=["CP1"])
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (0 = all CPUs, -1 = serial in main process)")
    parser.add_argument("--reducer-type", default="mean_pool",
                        choices=["mean_pool", "max_pool", "spatial_pool", "task_scoring"])
    parser.add_argument("--output-tokens", type=int, default=16,
                        help="SpatialPoolReducer output_tokens (must be perfect square)")
    parser.add_argument("--prune-window-size", type=int, default=4)
    parser.add_argument("--temporal-keep-ratio", type=float, default=0.5)
    parser.add_argument("--select-k", type=int, default=32,
                        help="TaskScoringReducer: number of top-k tokens to select")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="TaskScoringReducer: softmax temperature")
    # cp1_llm_layer_extract-only args.
    parser.add_argument("--extract-layer", type=int, default=0,
                        help="cp1_llm_layer_extract: PaliGemma layer index to extract (0..17)")
    parser.add_argument("--prefix-reducer-type", default="prefix_mean_pool",
                        choices=sorted(_LLM_LAYER_EXTRACT_DIMS.keys()),
                        help="cp1_llm_layer_extract: how to pool layer-N hidden states")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="cp1_llm_layer_extract: PI0Pytorch checkpoint dir (model.safetensors)")
    parser.add_argument("--config-name", default=None,
                        help="cp1_llm_layer_extract: TrainConfig name (e.g. pi05_libero)")
    parser.add_argument("--device", default="cuda",
                        help="cp1_llm_layer_extract: torch device for the model "
                             "(default: cuda; use cpu only for tiny smoke tests)")
    # B2 verdict-factor enrichment.
    parser.add_argument(
        "--factors-yaml", default=None,
        help="Path to a minimal YAML listing OfflineWriter-capable factors "
             "(F1b-A / F1b-T). When set, the artifact is enriched with per-entry "
             "`payload.factors` and a top-level `library_stats` field. When unset, "
             "only an empty `library_stats` placeholder is computed (still safe "
             "to load — InMemoryBackend will fall back to recompute at startup)."
    )
    args = parser.parse_args()

    artifact = build_artifact(
        args.data_dir, args.builder_type, args.checkpoint_id,
        workers=args.workers,
        reducer_type=args.reducer_type,
        output_tokens=args.output_tokens,
        prune_window_size=args.prune_window_size,
        temporal_keep_ratio=args.temporal_keep_ratio,
        select_k=args.select_k,
        temperature=args.temperature,
        extract_layer=args.extract_layer,
        prefix_reducer_type=args.prefix_reducer_type,
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        device=args.device,
    )

    # B2 — enrich with verdict-factor fields. Runs AFTER build_artifact's
    # _detach_entries (subprocess path keeps tensors numpy in memory for
    # IPC efficiency); enrich_artifact_with_factors / LibraryStats /
    # OfflineWriter all bridge numpy via torch.as_tensor.
    # sys.path injection mirrors build_llm_layer_matrix.py:50 — exp/ is
    # not on the package path when invoked as `uv run python exp/...`.
    import sys as _sys
    _here = str(Path(__file__).parent.resolve())
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from factor_postprocess import (
        _load_offline_writers_from_yaml,
        enrich_artifact_with_factors,
    )
    offline_writers = (
        _load_offline_writers_from_yaml(args.factors_yaml)
        if args.factors_yaml else []
    )
    if artifact["entries"]:
        library_stats = enrich_artifact_with_factors(artifact["entries"], offline_writers)
        artifact["library_stats"] = library_stats

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved {len(artifact['entries'])} entries to {args.output}")


if __name__ == "__main__":
    main()
