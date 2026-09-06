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
import hashlib
import json
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

# GR00T pools reuse the Pi0.5 pooling code: the HDF5 stores each field
# separately, so `_build_fake_stage1` reassembles a prefix whose offsets are
# correct by construction and the offset-based slicing applies unchanged. Only
# the geometry differs -- three cameras instead of two, and a 20-wide state --
# and the artifact is stamped with the GR00T builder name so a library can
# never be loaded under a Pi0.5 recipe that happens to share its dimensions.
_GROOT_TO_PI05_BUILDER: dict[str, str] = {
    "cp1_groot_mean_pool":              "cp1_mean_pool",
    "cp1_groot_spatial_pool_16":        "cp1_spatial_pool_16",
    "cp1_groot_spatial_pool_4":         "cp1_spatial_pool_4",
    "cp1_groot_max_pool":               "cp1_max_pool",
    # LIBERO post-trains of the same architecture: two cameras, 8-wide state.
    "cp1_groot_libero_mean_pool":       "cp1_mean_pool",
    "cp1_groot_libero_spatial_pool_16": "cp1_spatial_pool_16",
}
_GROOT_VISION_SLOTS = 3
_GROOT_ROBOT_STATE_DIM = 20
# Geometry is carried by the builder NAME, not by CLI flags: `_reshape_dims`
# fails silently in the direction that matters -- an under-declared camera count
# just drops that field from `vector_dims`, and the backend then omits it from
# every query without comment. Binding the geometry to the name makes the
# LIBERO artifact impossible to build under the RoboCasa recipe by omission.
_GROOT_GEOMETRY: dict[str, tuple[int, int]] = {  # builder -> (vision_slots, state_dim)
    "cp1_groot_libero_mean_pool":       (2, 8),
    "cp1_groot_libero_spatial_pool_16": (2, 8),
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
_ALL_BUILDER_TYPES = list(_VECTOR_DIMS.keys()) + list(_GROOT_TO_PI05_BUILDER) + [
    "cp1_temporal_prune", "cp1_llm_layer_extract", "projection",
]


def _projection_vector_dims(inner_type: str, params) -> dict[str, int]:
    """Stored vector dims for a projection artifact.

    Projected fields take their head out_dim; every other field keeps the inner
    pool dim. Identity (params is None) returns the inner dims unchanged.
    """
    dims = dict(_VECTOR_DIMS[inner_type])
    if params is not None:
        for field, head in params.heads.items():
            dims[field] = int(head.weight.shape[0])
    return dims


def _reshape_dims(
    dims: dict[str, int], vision_slots: int, robot_state_dim: int
) -> dict[str, int]:
    """Adapt a Pi0.5 dims table to another camera count / state width.

    The base tables describe LIBERO: two cameras, 32-wide state. Leaving them
    unadapted for a three-camera embodiment is not a loud failure -- the third
    field is simply absent from `vector_dims`, and the backend then drops that
    key from every query without comment.
    """
    out = {k: v for k, v in dims.items() if not k.startswith("vision_")}
    vision_dim = dims["vision_0"]
    for i in range(vision_slots):
        out[f"vision_{i}"] = vision_dim
    if "robot_state" in out:
        out["robot_state"] = robot_state_dim
    return {k: out[k] for k in sorted(out, key=lambda n: (not n.startswith("vision_"), n))}


def _get_vector_dims(
    builder_type: str,
    reducer_type: str = "mean_pool",
    output_tokens: int = 16,
    prefix_reducer_type: str = "prefix_mean_pool",
    inner_type: str = "cp1_mean_pool",
    projection_weights_path: str | None = None,
    vision_slots: int | None = None,
    robot_state_dim: int | None = None,
) -> dict[str, int]:
    if builder_type in _GROOT_TO_PI05_BUILDER:
        base = _VECTOR_DIMS[_GROOT_TO_PI05_BUILDER[builder_type]]
        default_slots, default_state = _GROOT_GEOMETRY.get(
            builder_type, (_GROOT_VISION_SLOTS, _GROOT_ROBOT_STATE_DIM)
        )
        return _reshape_dims(
            base,
            default_slots if vision_slots is None else vision_slots,
            default_state if robot_state_dim is None else robot_state_dim,
        )
    if builder_type in _VECTOR_DIMS:
        dims = _VECTOR_DIMS[builder_type]
        if vision_slots is None and robot_state_dim is None:
            return dims
        return _reshape_dims(
            dims,
            len([k for k in dims if k.startswith("vision_")]) if vision_slots is None else vision_slots,
            dims["robot_state"] if robot_state_dim is None else robot_state_dim,
        )
    if builder_type == "projection":
        from openpi.cache.components.projection_key_builder import ProjectionParams

        params = (
            ProjectionParams.load(projection_weights_path)
            if projection_weights_path else None
        )
        return _projection_vector_dims(inner_type, params)
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
    inner_type: str = "cp1_mean_pool",
    projection_weights_path: str | None = None,
    prompt_masked_pool: bool = False,
    prompt_instruction_span: bool = False,
    tokenizer_factory=None,
):
    # GR00T artifacts are built by the equivalent Pi0.5 pooling class; see
    # _GROOT_TO_PI05_BUILDER for why that is the same computation.
    builder_type = _GROOT_TO_PI05_BUILDER.get(builder_type, builder_type)
    from openpi.cache.components.key_builder import (
        CP1MaxPoolKeyBuilder,
        CP1MeanPoolKeyBuilder,
        CP1SpatialPool4KeyBuilder,
        CP1SpatialPool16KeyBuilder,
        CP1SpatialPool64KeyBuilder,
    )

    builders = {
        "cp1_mean_pool": CP1MeanPoolKeyBuilder,
        "cp1_spatial_pool_16": CP1SpatialPool16KeyBuilder,
        # Canonical 4-token name (2x2 pool); cp1_spatial_pool_64 is its legacy alias.
        "cp1_spatial_pool_4": CP1SpatialPool4KeyBuilder,
        "cp1_spatial_pool_64": CP1SpatialPool64KeyBuilder,
        "cp1_max_pool": CP1MaxPoolKeyBuilder,
    }
    _pool_knobs = {
        "prompt_masked_pool": prompt_masked_pool,
        "prompt_instruction_span": prompt_instruction_span,
        "tokenizer_factory": tokenizer_factory,
    }
    if builder_type in builders:
        return builders[builder_type](**_pool_knobs)
    if builder_type == "projection":
        from openpi.cache.components.projection_key_builder import (
            ProjectionKeyBuilder,
            ProjectionParams,
            validate_projection_params,
        )

        inner_cls = builders.get(inner_type)
        if inner_cls is None:
            raise ValueError(
                f"projection inner_type '{inner_type}' must be a stateless pool "
                f"builder. Valid: {sorted(builders)}"
            )
        params = (
            ProjectionParams.load(projection_weights_path)
            if projection_weights_path else None
        )
        # Validate against the projected stored dims (out_dim per head); in_dim
        # is checked against the inner pool output for the chosen inner_type.
        validate_projection_params(
            params, inner_type, _projection_vector_dims(inner_type, params)
        )
        return ProjectionKeyBuilder(inner_cls(**_pool_knobs), params)
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


def _build_fake_stage1(group: h5py.Group, lang_mask: "torch.Tensor | None" = None) -> _FakeStage1:
    """Reconstruct prefix_embs from HDF5 step group.

    ``lang_mask`` (masked prompt pooling): [num_prompt_tokens] bool from the
    deterministic re-tokenization of the stored prompt attr; when given, a
    prefix_pad_masks is synthesized (vision positions all True) so the online
    masked-pool code path runs identically offline.

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

    prefix_pad_masks = None
    if lang_mask is not None:
        n_vision = prefix_embs.shape[1] - lang_mask.shape[0]
        prefix_pad_masks = torch.cat(
            [torch.ones(n_vision, dtype=torch.bool), lang_mask.bool()]
        ).unsqueeze(0)

    return _FakeStage1(prefix_embs, state, prefix_pad_masks=prefix_pad_masks)


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
# Episode selection + trajectory id resolution
# ---------------------------------------------------------------------------

_TRAJECTORY_ID_MODES = ("stem", "relpath")


def resolve_from_manifest(data_dir: str | Path, manifest_path: str | Path) -> tuple[list[Path], dict]:
    """Return the h5 files named by an audit manifest, verifying every digest.

    ``verify_collection_artifacts`` already decided which episodes are
    admissible and recorded a sha256 for each. Feeding the builder that manifest
    -- rather than re-scanning the directory -- is what makes "the library was
    built from exactly the audited set" a checkable statement instead of a
    convention: a file edited or replaced after the audit changes its digest and
    is rejected here.

    Returns:
        ``(paths, manifest)`` so the caller can stamp the manifest's identity
        (plan hashes, pin id) onto the artifact.
    """
    root = Path(data_dir).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    if "tasks" not in manifest:
        raise ValueError(f"{manifest_path}: not an audit manifest (no 'tasks' key)")

    manifest_pin_id = manifest.get("pin_id")
    paths: list[Path] = []
    seen: set[Path] = set()
    for task_name in sorted(manifest["tasks"]):
        for row in manifest["tasks"][task_name]:
            resolved = (root / row["path"]).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"{manifest_path}: {row['path']!r} escapes --data-dir {root}")
            if not resolved.is_file():
                raise FileNotFoundError(f"{manifest_path}: {resolved} does not exist")
            if resolved in seen:
                raise ValueError(f"{manifest_path}: duplicate entry {row['path']!r}")
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(
                    f"{manifest_path}: {row['path']} hashes to {digest} but the "
                    f"manifest recorded {row['sha256']}; the file changed after "
                    "the audit admitted it"
                )
            if manifest_pin_id is not None:
                # The digests only prove the bytes are unchanged; they say
                # nothing about WHICH experiment those bytes came from. Without
                # this the manifest's pin_id is an unbacked claim, and editing
                # that one field (touching no episode at all) would stamp a
                # false identity onto the library that the serve-time binding
                # check would then happily accept.
                with h5py.File(resolved, "r") as handle:
                    episode_pin = handle.attrs.get("pin_id")
                if episode_pin is None:
                    raise ValueError(
                        f"{manifest_path}: {row['path']} records no pin_id but the "
                        f"manifest claims {manifest_pin_id}"
                    )
                if str(episode_pin) != manifest_pin_id:
                    raise ValueError(
                        f"{manifest_path}: {row['path']} was collected under pin_id "
                        f"{str(episode_pin)} but the manifest claims {manifest_pin_id}"
                    )
            seen.add(resolved)
            paths.append(resolved)
    if not paths:
        raise ValueError(f"{manifest_path}: manifest admits no episodes")
    return paths, manifest


def resolve_h5_paths(data_dir: str | Path, episode_list: str | Path | None) -> list[Path]:
    """Return the h5 files to build from, either by scan or by explicit list.

    Without ``episode_list`` this is the historical ``rglob`` scan. With one, the
    listed paths replace it entirely. Every rejection below is fail-fast rather
    than a silent skip: a silently shrinking library would be indistinguishable
    from a genuinely smaller one, and library size is the independent variable
    in the cache-size ablation.
    """
    root = Path(data_dir).resolve()
    if episode_list is None:
        return sorted(Path(data_dir).rglob("*.h5"))

    raw = Path(episode_list).read_text().splitlines()
    paths: list[Path] = []
    seen: set[Path] = set()
    for lineno, line in enumerate(raw, 1):
        rel = line.strip()
        if not rel:
            raise ValueError(f"{episode_list}:{lineno}: blank line (expected a relative .h5 path)")
        if Path(rel).is_absolute():
            raise ValueError(
                f"{episode_list}:{lineno}: absolute path {rel!r}; entries must be "
                "relative to --data-dir so the library stays reproducible"
            )
        resolved = (root / rel).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"{episode_list}:{lineno}: {rel!r} escapes --data-dir {root}")
        if resolved.suffix != ".h5":
            raise ValueError(f"{episode_list}:{lineno}: {rel!r} is not a .h5 file")
        if not resolved.is_file():
            raise FileNotFoundError(f"{episode_list}:{lineno}: {resolved} does not exist")
        if resolved in seen:
            raise ValueError(f"{episode_list}:{lineno}: duplicate entry {rel!r} (after normalization)")
        seen.add(resolved)
        paths.append(resolved)

    if not paths:
        raise ValueError(f"{episode_list}: no episodes listed")
    return paths


def trajectory_id_for(h5_path: Path, data_dir: str | Path, mode: str) -> str | None:
    """Resolve one file's trajectory id, or None to keep the builder's default.

    ``stem`` (default) returns None so ``_process_episode`` falls back to
    ``h5_path.stem`` exactly as before. ``relpath`` returns the suffix-stripped
    path relative to ``data_dir``, which stays unique across sub-directories --
    required whenever episodes are laid out as ``task_N/episode_M.h5``, since
    stem-based ids would collide across tasks and silently overwrite each other
    in ``InMemoryBackend.load_artifact``.
    """
    if mode == "stem":
        return None
    if mode != "relpath":
        raise ValueError(f"trajectory_id_mode must be one of {_TRAJECTORY_ID_MODES}, got {mode!r}")
    root = Path(data_dir).resolve()
    return h5_path.resolve().relative_to(root).with_suffix("").as_posix()


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


_TOKENIZER_CACHE: dict[int, object] = {}


def _get_pi05_tokenizer(max_len: int):
    """Per-process PaligemmaTokenizer cache keyed by max_len.

    max_len is derived from the stored prompt_emb sequence length so the
    offline re-tokenization pads exactly like the deployment did.
    """
    tok = _TOKENIZER_CACHE.get(max_len)
    if tok is None:
        from openpi.models.tokenizer import PaligemmaTokenizer

        tok = PaligemmaTokenizer(max_len=max_len)
        _TOKENIZER_CACHE[max_len] = tok
    return tok


def _trailing_identical_rows(seq: np.ndarray) -> int:
    """Length of the trailing run of rows identical to the last row."""
    last = seq[-1]
    n = 0
    for i in range(len(seq) - 1, -1, -1):
        if np.array_equal(seq[i], last):
            n += 1
        else:
            break
    return n


def _masked_prompt_step_context(
    tokenizer,
    prompt_str: str,
    state_np: np.ndarray | None,
    prompt_emb_np: np.ndarray,
    instruction_span: bool,
    h5_name: str,
):
    """Re-tokenize one step's prompt and self-check against the stored rows.

    Returns (tokens_int64_1d, lang_mask_bool_tensor). Aborts (ValueError) on:
      - mask/stored-length mismatch;
      - dual-source mismatch (mask.sum() != seq_len - trailing pad rows,
        checked only when padding exists);
      - instruction_span with the " State:" marker not occurring exactly once.
    """
    from openpi.cache.components.key_builder import find_instruction_span

    tokens, mask = tokenizer.tokenize(prompt_str, state=state_np)
    if len(mask) != prompt_emb_np.shape[0]:
        raise ValueError(
            f"{h5_name}: re-tokenized mask length {len(mask)} != stored "
            f"prompt_emb rows {prompt_emb_np.shape[0]} (tokenizer max_len drift)."
        )
    valid = int(np.asarray(mask).sum())
    if valid < prompt_emb_np.shape[0]:
        trailing = _trailing_identical_rows(prompt_emb_np)
        if prompt_emb_np.shape[0] - trailing != valid:
            raise ValueError(
                f"{h5_name}: dual-source mask check failed — re-tokenized "
                f"valid={valid} but stored rows imply "
                f"{prompt_emb_np.shape[0] - trailing}. The H5 prompt attr and "
                "the captured embeddings disagree; refusing to build a "
                "mismatched artifact."
            )
    tokens_np = np.asarray(tokens, dtype=np.int64)
    if instruction_span:
        marker = np.asarray(tokenizer.encode_fragment(" State:"), dtype=np.int64)
        valid_ids = tokens_np[:valid]
        first = find_instruction_span(valid_ids, marker)
        if first is None or first <= 0:
            raise ValueError(
                f"{h5_name}: --prompt-instruction-span set but the ' State:' "
                "marker was not found in the tokenized prompt."
            )
        second = find_instruction_span(valid_ids[first + 1:], marker)
        if second is not None:
            raise ValueError(
                f"{h5_name}: ' State:' marker occurs more than once in the "
                "tokenized prompt; the span rule is ambiguous."
            )
    lang_mask = torch.from_numpy(np.asarray(mask).astype(bool))
    return tokens_np, lang_mask


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
    inner_type: str = "cp1_mean_pool",
    projection_weights_path: str | None = None,
    outcome_filter: str = "success",
    trajectory_id: str | None = None,
    *,
    prompt_masked_pool: bool = False,
    prompt_instruction_span: bool = False,
    discrete_state_input: bool = False,
) -> list | None:
    """Process a single H5 file in a worker process. Returns list of CacheEntry or None.

    ``trajectory_id`` overrides the default ``h5_path.stem``. The caller resolves
    it (it needs ``data_dir`` to build a relative path, which workers do not have),
    so ``None`` preserves the historical stem-based ids byte for byte.

    ``outcome_filter`` selects episodes by the HDF5 ``success`` attr: "success"
    (default; the unchanged legacy code path, behavior-preserving) keeps successful episodes,
    "failure" keeps only failed ones (TRACER Phase 4 D- collection), "all" keeps
    every episode.
    """
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID

    h5_path = Path(h5_path_str)
    cp_id = CheckpointID[checkpoint_id_str]
    tokenizer = None
    builder = _create_builder(
        builder_type, reducer_type, output_tokens,
        prune_window_size, temporal_keep_ratio,
        select_k, temperature,
        inner_type=inner_type,
        projection_weights_path=projection_weights_path,
        prompt_masked_pool=prompt_masked_pool,
        prompt_instruction_span=prompt_instruction_span,
        tokenizer_factory=(lambda: tokenizer) if prompt_masked_pool else None,
    )

    with h5py.File(h5_path, "r") as f:
        task = str(f.attrs.get("task", ""))
        success = bool(f.attrs.get("success", False))
        # Outcome filter (TRACER Phase 4). Default "success" runs the unchanged
        # legacy code path (behavior-preserving); "failure" collects the D- pool;
        # "all" keeps both.
        if outcome_filter == "success" and not success:
            return None
        if outcome_filter == "failure" and success:
            return None

        if trajectory_id is None:
            trajectory_id = h5_path.stem

        # Notify stateful builders to reset history for this episode
        if hasattr(builder, 'on_episode_start'):
            builder.on_episode_start()

        prompt_str = None
        episode_prompt_ctx = None
        if prompt_masked_pool:
            prompt_str = str(f.attrs.get("prompt", "") or task)
            if not prompt_str:
                raise ValueError(
                    f"{h5_path.name}: masked prompt pooling needs the episode "
                    "prompt but neither the 'prompt' nor the 'task' attr is set."
                )

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

            if prompt_masked_pool:
                prompt_emb_np = np.array(group["prompt_emb"])
                if tokenizer is None:
                    tokenizer = _get_pi05_tokenizer(prompt_emb_np.shape[0])
                if discrete_state_input:
                    # State is baked into the prompt: tokens vary per step.
                    step_ctx = _masked_prompt_step_context(
                        tokenizer, prompt_str,
                        np.array(group["robot_state"], dtype=np.float32),
                        prompt_emb_np, prompt_instruction_span, h5_path.name,
                    )
                else:
                    if episode_prompt_ctx is None:
                        episode_prompt_ctx = _masked_prompt_step_context(
                            tokenizer, prompt_str, None,
                            prompt_emb_np, prompt_instruction_span, h5_path.name,
                        )
                    step_ctx = episode_prompt_ctx
                tokens_np, lang_mask = step_ctx
                fake_stage1 = _build_fake_stage1(group, lang_mask=lang_mask)
                builder.collect(cp_id, stage1=fake_stage1, tokenized_prompt=tokens_np)
            else:
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
    inner_type: str = "cp1_mean_pool",
    projection_weights_path: str | None = None,
    outcome_filter: str = "success",
    episode_list: str | None = None,
    manifest: str | None = None,
    trajectory_id_mode: str = "stem",
    vision_slots: int | None = None,
    robot_state_dim: int | None = None,
    prompt_masked_pool: bool = False,
    prompt_instruction_span: bool = False,
    discrete_state_input: bool = False,
) -> dict:
    """Build artifact dict from HDF5 data.

    Scans data_dir for .h5 files, keeping episodes per ``outcome_filter``
    ("success" = legacy default, "failure" = TRACER Phase 4 D- pool, "all").
    For each step: collect() -> build() -> CacheEntry.
    Entry id = "{episode_file_stem}_{step_name}" (deterministic, traceable).
    action_chunk keeps real horizon from data (no padding to [50,32]).

    Args:
        workers: Number of parallel workers. 0 = all CPUs. Forced to -1
            (serial in-process) for `cp1_llm_layer_extract` since the
            partial paligemma forward must run in the main process where
            the model is loaded once on GPU.
    """
    if outcome_filter not in ("success", "failure", "all"):
        raise ValueError(
            f"outcome_filter must be one of success/failure/all, got {outcome_filter!r}"
        )
    # Prompt-pool knob allowlist — same single-source constant as online config
    # validation. Rejecting BEFORE any H5 is touched also closes the
    # lying-metadata path: an unsupported builder would emit full-pool keys
    # while the artifact claimed masked semantics.
    if prompt_masked_pool or prompt_instruction_span:
        from openpi.cache.config import PROMPT_POOL_KNOB_BUILDERS

        supported = builder_type in PROMPT_POOL_KNOB_BUILDERS or (
            builder_type == "projection" and inner_type in PROMPT_POOL_KNOB_BUILDERS
        )
        if not supported:
            raise ValueError(
                f"--prompt-masked-pool / --prompt-instruction-span are only "
                f"honoured by {sorted(PROMPT_POOL_KNOB_BUILDERS)} (or projection "
                f"with such an inner); got builder_type={builder_type!r}."
            )
    if prompt_instruction_span and not prompt_masked_pool:
        raise ValueError("--prompt-instruction-span requires --prompt-masked-pool.")
    # cp1_llm_layer_extract routes through _process_episode_with_model, which
    # this phase does not parameterize. Reject non-default filters fail-loud at
    # the very top -- BEFORE any checkpoint/model load -- so the failure is
    # CPU-only (TRACER Phase 4 plan §4.1 / G1 R2).
    if outcome_filter != "success" and builder_type == "cp1_llm_layer_extract":
        raise ValueError(
            "outcome_filter is only supported for pool builders (_process_episode); "
            "cp1_llm_layer_extract (model path) accepts only the default 'success'."
        )
    if trajectory_id_mode not in _TRAJECTORY_ID_MODES:
        raise ValueError(
            f"trajectory_id_mode must be one of {_TRAJECTORY_ID_MODES}, got {trajectory_id_mode!r}"
        )
    # The model path (_process_episode_with_model) derives ids independently and
    # is not parameterized here; reject rather than silently ignore the request.
    if trajectory_id_mode != "stem" and builder_type == "cp1_llm_layer_extract":
        raise ValueError(
            "trajectory_id_mode is only supported for pool builders (_process_episode); "
            "cp1_llm_layer_extract (model path) accepts only the default 'stem'."
        )
    vector_dims = _get_vector_dims(
        builder_type, reducer_type, output_tokens, prefix_reducer_type,
        inner_type=inner_type, projection_weights_path=projection_weights_path,
        vision_slots=vision_slots, robot_state_dim=robot_state_dim,
    )

    # Validated once here; both the serial and the ProcessPool branch below
    # consume this same list, so the two paths cannot diverge.
    if manifest is not None:
        if episode_list is not None:
            raise ValueError("--manifest and --episode-list are mutually exclusive inputs")
        h5_paths, manifest_doc = resolve_from_manifest(data_dir, manifest)
    else:
        h5_paths, manifest_doc = resolve_h5_paths(data_dir, episode_list), None
    if not h5_paths:
        logger.warning("No .h5 files found in %s", data_dir)
        return {"key_builder_type": builder_type, "checkpoint_id": checkpoint_id_str,
                "vector_dims": vector_dims, "entries": [],
                "prompt_pool": {"masked": prompt_masked_pool,
                                "instruction_span": prompt_instruction_span}}

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
        inner_type, projection_weights_path,
        outcome_filter,
    )
    _ep_kwargs = {
        "prompt_masked_pool": prompt_masked_pool,
        "prompt_instruction_span": prompt_instruction_span,
        "discrete_state_input": discrete_state_input,
    }

    traj_ids = {p: trajectory_id_for(p, data_dir, trajectory_id_mode) for p in h5_paths}

    entries: list = []

    if workers == -1:
        # Serial mode: run in main process (avoids fork overhead in tests)
        logger.info("Processing %d H5 files in serial mode", len(h5_paths))
        for i, p in enumerate(h5_paths, 1):
            result = _process_episode(str(p), *_ep_args, traj_ids[p], **_ep_kwargs)
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
                pool.submit(_process_episode, str(p), *_ep_args, traj_ids[p], **_ep_kwargs): p
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
        # Prompt-pool identity, force-checked at serve time against the
        # configured knobs (_check_text_ivf_artifact_binding).
        "prompt_pool": {"masked": prompt_masked_pool,
                        "instruction_span": prompt_instruction_span},
    }
    if manifest_doc is not None:
        # Provenance of the SET this library was built from: which collection
        # run-plans produced it, and which object pin table those episodes ran
        # under. The pin id is what _check_pin_identity_binding compares a
        # serving config against.
        artifact["plan_hashes"] = manifest_doc.get("plan_hashes")
        if manifest_doc.get("pin_id") is not None:
            artifact["pin_id"] = manifest_doc["pin_id"]
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
    # Record projection provenance (inner pool + weights source) for the same
    # offline/online consistency audit.
    if builder_type == "projection":
        artifact["projection_params"] = {
            "inner_type": inner_type,
            "projection_weights_path": (
                str(projection_weights_path) if projection_weights_path else None
            ),
        }
    return artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _enrich_existing_pkl_main(argv: list[str]) -> None:
    """``enrich-existing-pkl`` sub-command: read an existing pkl, run new
    OfflineWriter list, write a new pkl with merged factors.

    Plan §16 B6 / B6.5: this is the smoke acceptance gate for the
    refactor. Reuses the input pkl's ``library_stats`` so the smoke run
    completes in seconds rather than re-computing per-DOF sigma over
    every entry (30+ min on libero_spatial).
    """
    parser = argparse.ArgumentParser(
        prog="enrich-existing-pkl",
        description=(
            "Read an existing InMemoryBackend pkl artifact, run the new "
            "17-factor OfflineWriter list against its entries, and write "
            "a new pkl with merged payload.factors. Reuses the input pkl's "
            "library_stats — does NOT recompute per-DOF sigma."
        ),
    )
    parser.add_argument("--input", required=True, help="Source pkl path (read-only)")
    parser.add_argument("--factors-yaml", required=True, help="Factor spec yaml")
    parser.add_argument("--output", required=True, help="Destination pkl path")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.is_file():
        raise SystemExit(f"--input pkl not found: {in_path}")

    logger.info("Loading source artifact from %s", in_path)
    with open(in_path, "rb") as fh:
        artifact = pickle.load(fh)
    if "entries" not in artifact:
        raise SystemExit(f"--input pkl {in_path} missing 'entries' field")
    library_stats = artifact.get("library_stats")
    if library_stats is None:
        logger.warning(
            "input pkl has no library_stats; will compute_from_entries (slow)"
        )

    # sys.path injection mirrors the legacy build path so
    # ``from factor_postprocess import ...`` works when invoked as
    # ``uv run python -m exp.common.build_in_memory_cache_artifact ...``.
    import sys as _sys
    _here = str(Path(__file__).parent.resolve())
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from factor_postprocess import (
        _load_offline_writers_from_yaml,
        enrich_artifact_with_factors,
    )

    offline_writers = _load_offline_writers_from_yaml(args.factors_yaml)
    new_library_stats = enrich_artifact_with_factors(
        artifact["entries"],
        offline_writers,
        library_stats=library_stats,   # G1 R3 Item 4: skip recompute when present
    )
    artifact["library_stats"] = new_library_stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(artifact, fh)
    logger.info(
        "wrote enriched pkl: %d entries, %d offline writers applied → %s",
        len(artifact["entries"]), len(offline_writers), out_path,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # B6 refactor: dispatch on first positional ``argv`` token. The legacy
    # build-from-HDF5 path runs unchanged when no sub-command is given.
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "enrich-existing-pkl":
        _enrich_existing_pkl_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="Build InMemoryBackend artifact from HDF5 data")
    parser.add_argument("--data-dir", required=True, help="Directory with .h5 episode files")
    parser.add_argument("--builder-type", required=True, choices=_ALL_BUILDER_TYPES)
    parser.add_argument("--output", required=True, help="Output .pkl path")
    parser.add_argument("--checkpoint-id", default="CP1", choices=["CP1"])
    parser.add_argument(
        "--vision-slots", type=int, default=None,
        help="Number of vision_* fields in the artifact. Defaults to the "
             "builder's own convention (2 for the Pi0.5 pools, 3 for cp1_groot_*).",
    )
    parser.add_argument(
        "--robot-state-dim", type=int, default=None,
        help="Width of the robot_state vector. Defaults per builder (32 for the "
             "Pi0.5 pools, 20 for cp1_groot_*).",
    )
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
    # projection-only args.
    parser.add_argument("--inner-type", default="cp1_mean_pool",
                        help="projection: stateless pool builder to wrap")
    parser.add_argument("--projection-weights", default=None,
                        help="projection: torch-saved ProjectionParams path "
                             "(None -> identity, output equals the inner pool)")
    # TRACER Phase 4: failure-pool D- collection filter.
    parser.add_argument(
        "--outcome-filter", default="success", choices=["success", "failure", "all"],
        help="Which episodes to keep by HDF5 success attr: 'success' (default, "
             "unchanged legacy behavior), 'failure' (TRACER Phase 4 D- "
             "collection), or 'all'. Pool builders only; cp1_llm_layer_extract "
             "rejects a non-default value fail-loud."
    )
    # Cache-size ablation (X9b): explicit episode subsets + collision-free ids.
    parser.add_argument(
        "--episode-list", default=None,
        help="Path to a newline-separated list of .h5 paths relative to "
             "--data-dir. When given it replaces the recursive scan, so a "
             "library can be built from an explicit subset without copying "
             "files. Rejects absolute paths, '..' escapes, non-.h5 entries, "
             "missing files, blank lines and post-normalization duplicates "
             "fail-fast (a silently shrinking library would be "
             "indistinguishable from a genuinely smaller one)."
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Path to a verify_collection_artifacts manifest. Replaces both the "
             "scan and --episode-list: the library is built from exactly the "
             "admitted episodes, each one's sha256 re-verified, and the "
             "manifest's plan hashes plus pin_id are stamped onto the artifact."
    )
    parser.add_argument(
        "--trajectory-id-mode", default="stem", choices=list(_TRAJECTORY_ID_MODES),
        help="How to derive each entry's trajectory_id. 'stem' (default) uses "
             "the file stem, preserving historical artifacts byte for byte. "
             "'relpath' uses the suffix-stripped path relative to --data-dir; "
             "required for layouts like task_N/episode_M.h5 where stems repeat "
             "across sub-directories and would otherwise collide (and silently "
             "overwrite) in InMemoryBackend.load_artifact. Pool builders only."
    )
    # Text-IVF instruction-span masked prompt pooling (plan U5).
    parser.add_argument(
        "--prompt-masked-pool", action="store_true",
        help="Pool prompt_emb over real prompt tokens only (padding excluded), "
             "reconstructing the mask by re-tokenizing the stored prompt attr. "
             "Only honoured by the cp1_* pool builders (and projection with "
             "such an inner); other builder types abort before any H5 is read."
    )
    parser.add_argument(
        "--prompt-instruction-span", action="store_true",
        help="Additionally cut the prompt at the ' State:' marker "
             "(discrete-state prompts). Requires --prompt-masked-pool; the "
             "build aborts unless the marker occurs exactly once per prompt."
    )
    parser.add_argument(
        "--discrete-state-input", action="store_true",
        help="The deployment bakes discretized state into the prompt "
             "(Pi0Config.discrete_state_input=True, e.g. pi05_robocasa): "
             "re-tokenize per step with that step's state."
    )
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
        prompt_masked_pool=args.prompt_masked_pool,
        prompt_instruction_span=args.prompt_instruction_span,
        discrete_state_input=args.discrete_state_input,
        inner_type=args.inner_type,
        projection_weights_path=args.projection_weights,
        outcome_filter=args.outcome_filter,
        episode_list=args.episode_list,
        manifest=args.manifest,
        trajectory_id_mode=args.trajectory_id_mode,
        vision_slots=args.vision_slots,
        robot_state_dim=args.robot_state_dim,
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
