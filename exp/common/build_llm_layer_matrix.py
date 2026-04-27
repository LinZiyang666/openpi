"""Batched artifact builder for cp1_llm_layer_extract across a (layers, reducers) matrix.

Motivation
----------
The per-config `build_in_memory_cache_artifact.py` path loads the Pi0.5 model
and runs a full prefix forward once per (extract_layer, prefix_reducer) combo.
For a sweep of L layers × R reducers this pays 20x the fixed cost for little
gain: the layer-N hidden state for layer N<=max(L) is already a byproduct of
the forward to max(L). This script amortizes model load, tokenizer self-check,
stage-1 reconstruction, and layer forwards across the whole matrix — one
episode pass produces all 20 artifact pickles simultaneously.

Reuses (without duplication):
  - `_load_pi05_for_llm_extract`       (model + tokenizer loading)
  - `_self_check_tokenizer_consistency` (startup parity guard)
  - `_build_fake_stage1_with_masks`    (per-step prefix reconstruction)
  - `_cast_to_layer_dtype`              (layer-0 dtype promotion rule)

Usage:
    uv run python exp/common/build_llm_layer_matrix.py \
        --data-dir exp/common/data/db/libero_cache/libero_spatial \
        --output-dir exp/common/data/cache_artifacts/libero_spatial/llm_layer_extract \
        --checkpoint-dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch" \
        --config-name pi05_libero \
        --layers 0,1,2,3 \
        --reducers prefix_mean_pool,per_modality_mean_pool,per_modality_max_pool,\
per_modality_spatial_pool_16,per_modality_spatial_pool_4

Output layout (same naming convention as per-config path):
    {output_dir}/cp1_llm_l{L}_{reducer}.pkl
"""

from __future__ import annotations

import argparse
import gc
import logging
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# Allow running this file directly (`python exp/common/...`) without PYTHONPATH;
# sibling module is in the same dir.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from build_in_memory_cache_artifact import (  # noqa: E402  (sys.path mutated above)
    _LLM_LAYER_EXTRACT_DIMS,
    _PI05_TOKENIZER_MAX_LEN,
    _PI05_TOKENIZER_SOURCE,
    _build_fake_stage1_with_masks,
    _detach_entries,
    _load_pi05_for_llm_extract,
    _self_check_tokenizer_consistency,
)
from openpi.cache.components.key_builder import _PROMPT_START, _VISION_OFFSETS
from openpi.cache.components.llm_layer_key_builder import _cast_to_layer_dtype
from openpi.cache.components.prefix_reducer import (
    LLMLayerExtractResult,
    PerModalityMaxPoolReducer,
    PerModalityMeanPoolReducer,
    PerModalitySpatialPoolReducer,
    PrefixMeanPoolReducer,
    PrefixReducer,
)
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import PROMPT_EMB, ROBOT_STATE, CheckpointID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reducer factory (keeps this script independent of the full cache.config path)
# ---------------------------------------------------------------------------


def _build_reducer(reducer_type: str) -> PrefixReducer:
    if reducer_type == "prefix_mean_pool":
        return PrefixMeanPoolReducer()
    if reducer_type == "per_modality_mean_pool":
        return PerModalityMeanPoolReducer()
    if reducer_type == "per_modality_max_pool":
        return PerModalityMaxPoolReducer()
    if reducer_type == "per_modality_spatial_pool_16":
        return PerModalitySpatialPoolReducer(output_tokens=16)
    if reducer_type == "per_modality_spatial_pool_4":
        return PerModalitySpatialPoolReducer(output_tokens=4)
    raise ValueError(f"Unknown reducer_type: {reducer_type}")


# ---------------------------------------------------------------------------
# Multi-layer forward
# ---------------------------------------------------------------------------


def _forward_collect_layers(
    fake_stage1,
    layers: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    target_layers: set[int],
) -> dict[int, torch.Tensor]:
    """Run layers[0..max(target_layers)] once; return hidden states at each target.

    Mirrors `CP1LLMLayerExtractKeyBuilder._extract` but collects multiple layer
    outputs in a single forward. `target_layers` are 0-indexed; the returned
    dict maps layer index -> hidden state [L, D] (batch dim dropped).
    """
    max_layer = max(target_layers)
    prefix_embs = fake_stage1.prefix_embs                       # [1, L, D]
    attention_mask = fake_stage1.prefix_att_2d_masks_4d         # [1, 1, L, L]
    position_ids = fake_stage1.prefix_position_ids              # [1, L]

    hidden = _cast_to_layer_dtype(prefix_embs, layers[0])
    cos, sin = rotary_emb(hidden, position_ids)

    collected: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for layer_idx in range(max_layer + 1):
            layer_out = layers[layer_idx](
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=(cos, sin),
                adarms_cond=None,     # paligemma side: use_adarms[0]=False (Pi0.5)
            )
            hidden = layer_out[0]
            if layer_idx in target_layers:
                # Clone to decouple from the rolling `hidden` reassigned next step.
                collected[layer_idx] = hidden[0].detach().clone()
    return collected


# ---------------------------------------------------------------------------
# Per-step processing: one forward, all (layer, reducer) reductions
# ---------------------------------------------------------------------------


def _build_entry(
    query_keys: dict[str, torch.Tensor],
    state_cpu: torch.Tensor,
    trajectory_id: str,
    step_idx: int,
    step_name: str,
    cp_id: CheckpointID,
    group: h5py.Group,
    task: str,
) -> CacheEntry:
    """Assemble CacheEntry + CachePayload from the per-step HDF5 group.

    Logic is a targeted copy of
    `build_in_memory_cache_artifact._process_episode_with_model`; kept inlined
    here to avoid restructuring that function into a public helper.
    """
    # Move GPU keys to CPU fp32 contiguous (matching the single-config path).
    keys_cpu: dict[str, torch.Tensor] = {
        f: v.detach().cpu().to(torch.float32).contiguous()
        for f, v in query_keys.items()
    }
    keys_cpu[ROBOT_STATE] = state_cpu

    entry_id = f"{trajectory_id}:{step_idx if step_idx is not None else step_name}"

    action = torch.from_numpy(np.array(group["clean_action"])).float()
    if action.dim() == 1:
        action = action.unsqueeze(0)

    _NUM_STEPS = 10
    intermediates = None
    denoising_num_steps = None
    noise_indices: list[int] = []
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
    return CacheEntry(
        id=entry_id,
        checkpoint_id=cp_id,
        query_keys=keys_cpu,
        payload=payload,
        step_idx=step_idx,
        trajectory_id=trajectory_id,
    )


def _process_episode_matrix(
    h5_path: Path,
    model,
    tokenizer,
    device: torch.device,
    layers: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    target_layers: set[int],
    reducers: dict[str, PrefixReducer],
    cp_id: CheckpointID,
    entries_by_key: dict[tuple[int, str], list[CacheEntry]],
) -> int:
    """Process one HDF5 episode, appending entries to all matrix slots. Returns #steps."""
    with h5py.File(h5_path, "r") as f:
        task = str(f.attrs.get("task", ""))
        success = bool(f.attrs.get("success", False))
        if not success:
            return 0
        trajectory_id = h5_path.stem

        def _step_sort_key(name: str) -> tuple[bool, int, str]:
            suffix = name.split("_", 1)[1] if "_" in name else ""
            if suffix.isdigit():
                return (False, int(suffix), name)
            return (True, 0, name)

        step_names = sorted(
            (k for k in f.keys() if k.startswith("step_")),
            key=_step_sort_key,
        )

        # We need to also backfill prev/next ids per matrix slot at the end.
        per_slot_this_episode: dict[tuple[int, str], list[CacheEntry]] = {
            k: [] for k in entries_by_key
        }

        for step_name in step_names:
            group = f[step_name]
            fake_stage1 = _build_fake_stage1_with_masks(
                group, task_str=task, tokenizer=tokenizer,
                model=model, device=device,
            )

            # Single forward -> hidden states at every target layer.
            layer_hiddens = _forward_collect_layers(
                fake_stage1, layers, rotary_emb, target_layers,
            )

            pad_mask = fake_stage1.prefix_pad_masks[0]          # [L]
            prefix_len = pad_mask.shape[0]
            segment_offsets = {
                field: (start, end) for field, start, end in _VISION_OFFSETS
            }
            segment_offsets[PROMPT_EMB] = (_PROMPT_START, prefix_len)

            state_cpu = (
                fake_stage1.state[0].detach().cpu().to(torch.float32).contiguous()
            )

            suffix = step_name.split("_", 1)[1] if "_" in step_name else ""
            step_idx = int(suffix) if suffix.isdigit() else None

            for layer_idx, hidden in layer_hiddens.items():
                result = LLMLayerExtractResult(
                    hidden_states=hidden,
                    pad_mask=pad_mask,
                    segment_offsets=segment_offsets,
                    extract_layer=layer_idx,
                )
                for reducer_type, reducer in reducers.items():
                    slot = (layer_idx, reducer_type)
                    # Skip reducer work for slots whose artifact already exists
                    # on disk (pre-filtered into `entries_by_key`).
                    if slot not in per_slot_this_episode:
                        continue
                    gpu_keys = reducer.reduce(result)
                    entry = _build_entry(
                        query_keys=gpu_keys,
                        state_cpu=state_cpu,
                        trajectory_id=trajectory_id,
                        step_idx=step_idx,
                        step_name=step_name,
                        cp_id=cp_id,
                        group=group,
                        task=task,
                    )
                    per_slot_this_episode[slot].append(entry)

            # Free forward-time GPU tensors eagerly.
            del layer_hiddens

        # Backfill prev/next ids within this episode, per matrix slot.
        for slot_key, slot_entries in per_slot_this_episode.items():
            for i, e in enumerate(slot_entries):
                if i > 0:
                    e.prev_ids = [slot_entries[i - 1].id]
                if i < len(slot_entries) - 1:
                    e.next_ids = [slot_entries[i + 1].id]
            entries_by_key[slot_key].extend(slot_entries)

        return len(step_names)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _vector_dims_for_reducer(reducer_type: str) -> dict[str, int]:
    return _LLM_LAYER_EXTRACT_DIMS[reducer_type]


def _save_artifact(
    entries: list[CacheEntry],
    *,
    out_path: Path,
    extract_layer: int,
    reducer_type: str,
    checkpoint_dir: str,
    config_name: str,
    factors_yaml: str | None = None,
) -> None:
    # Convert torch.Tensor fields to numpy in-place (matches single-config path).
    _detach_entries(entries)

    # B2 — verdict-factor enrichment runs AFTER detach. Helper internals
    # bridge numpy through torch.as_tensor.
    library_stats = None
    if entries:
        from exp.common.factor_postprocess import (
            _load_offline_writers_from_yaml,
            enrich_artifact_with_factors,
        )
        offline_writers = (
            _load_offline_writers_from_yaml(factors_yaml) if factors_yaml else []
        )
        library_stats = enrich_artifact_with_factors(entries, offline_writers)

    artifact = {
        "key_builder_type": "cp1_llm_layer_extract",
        "checkpoint_id": "CP1",
        "vector_dims": _vector_dims_for_reducer(reducer_type),
        "entries": entries,
        "library_stats": library_stats,
        "reducer_params": {
            "extract_layer": extract_layer,
            "prefix_reducer_type": reducer_type,
            "apply_final_norm": False,
            "checkpoint_dir": str(checkpoint_dir),
            "config_name": config_name,
            "tokenizer_class": "PaligemmaTokenizer",
            "tokenizer_source": _PI05_TOKENIZER_SOURCE,
            "tokenizer_max_len": _PI05_TOKENIZER_MAX_LEN,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_matrix(
    *,
    data_dir: str,
    output_dir: str,
    checkpoint_dir: str,
    config_name: str,
    device: str,
    layers: list[int],
    reducer_types: list[str],
    skip_existing: bool = True,
    factors_yaml: str | None = None,
) -> None:
    target_layers = set(layers)
    for r in reducer_types:
        if r not in _LLM_LAYER_EXTRACT_DIMS:
            raise ValueError(
                f"Unknown reducer_type {r!r}. "
                f"Valid: {sorted(_LLM_LAYER_EXTRACT_DIMS)}"
            )

    # Compute which (layer, reducer) slots still need building.
    slots: list[tuple[int, str]] = []
    out_paths: dict[tuple[int, str], Path] = {}
    for L in layers:
        for r in reducer_types:
            p = Path(output_dir) / f"cp1_llm_l{L}_{r}.pkl"
            out_paths[(L, r)] = p
            if skip_existing and p.exists():
                logger.info("[skip existing] %s", p.name)
                continue
            slots.append((L, r))

    if not slots:
        logger.info("All requested artifacts already exist; nothing to do.")
        return

    # Layers actually required by the remaining slots.
    target_layers = {L for (L, _) in slots}
    required_reducer_types = sorted({r for (_, r) in slots})
    reducers = {r: _build_reducer(r) for r in required_reducer_types}

    h5_paths = sorted(Path(data_dir).rglob("*.h5"))
    if not h5_paths:
        logger.warning("No .h5 files found in %s", data_dir)
        return

    torch_device = torch.device(device)
    model, tokenizer = _load_pi05_for_llm_extract(checkpoint_dir, config_name, device)

    language_model = model.paligemma_with_expert.paligemma.language_model
    model_layers = language_model.layers
    rotary_emb = language_model.rotary_emb
    language_model.config._attn_implementation = "eager"  # noqa: SLF001
    depth = len(model_layers)
    for L in target_layers:
        if not (0 <= L < depth):
            raise ValueError(f"extract_layer={L} out of range (model depth={depth})")

    # Tokenizer self-check (once, on first episode).
    _self_check_tokenizer_consistency(h5_paths[0], model, tokenizer, torch_device)

    # Entry buckets per slot.
    entries_by_key: dict[tuple[int, str], list[CacheEntry]] = {s: [] for s in slots}

    # Only track reducer types that are in `slots`; still use per-step bucket only
    # for the slots we need.
    active_slots = set(slots)

    def _reducers_for_layer(L: int) -> dict[str, PrefixReducer]:
        # Filter to reducer types with an active slot at this layer.
        return {
            r: reducers[r] for r in required_reducer_types if (L, r) in active_slots
        }

    cp_id = CheckpointID.CP1
    t_start = time.time()
    total_steps = 0

    logger.info(
        "Matrix build: %d layers × %d reducers = %d slots, %d episodes",
        len(target_layers), len(required_reducer_types), len(slots), len(h5_paths),
    )
    for i, p in enumerate(h5_paths, 1):
        # Per-episode: pick reducers-per-layer dict fresh so each step only
        # reduces what we need. Reducers are cheap tensor ops, no state.
        reducers_per_layer_loop: dict[str, PrefixReducer] = {
            r: reducers[r] for r in required_reducer_types
        }
        _process_episode_matrix(
            h5_path=p,
            model=model,
            tokenizer=tokenizer,
            device=torch_device,
            layers=model_layers,
            rotary_emb=rotary_emb,
            target_layers=target_layers,
            reducers=reducers_per_layer_loop,
            cp_id=cp_id,
            entries_by_key=entries_by_key,
        )
        gc.collect()
        if i % 5 == 0 or i == len(h5_paths):
            # Report progress using the first active slot as a sentinel — all
            # slots grow in lock-step (same episodes, same success filter).
            sentinel_slot = slots[0]
            logger.info(
                "Progress: %d/%d episodes, slot[%s] #entries=%d, elapsed=%.1fs",
                i, len(h5_paths), sentinel_slot,
                len(entries_by_key[sentinel_slot]), time.time() - t_start,
            )
        total_steps += 1

    logger.info(
        "Forward complete in %.1fs. Saving %d artifacts...",
        time.time() - t_start, len(slots),
    )

    for (L, r), bucket in entries_by_key.items():
        out_path = out_paths[(L, r)]
        _save_artifact(
            bucket,
            out_path=out_path,
            extract_layer=L,
            reducer_type=r,
            checkpoint_dir=checkpoint_dir,
            config_name=config_name,
            factors_yaml=factors_yaml,
        )
        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info("  saved %s  (%d entries, %.1f MB)", out_path.name, len(bucket), size_mb)

    logger.info("Matrix build done in %.1fs total.", time.time() - t_start)


def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_layers(s: str) -> list[int]:
    return [int(x) for x in _parse_list(s)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batched cp1_llm_layer_extract artifact matrix builder "
                    "(1 model load, 1 forward per step -> all (layer,reducer) pkls).",
    )
    parser.add_argument("--data-dir", required=True, help="Dir of HDF5 episode files.")
    parser.add_argument("--output-dir", required=True,
                        help="Dir to write cp1_llm_l{L}_{reducer}.pkl into.")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="PI0Pytorch checkpoint dir (model.safetensors).")
    parser.add_argument("--config-name", required=True,
                        help="TrainConfig name (e.g. pi05_libero).")
    parser.add_argument("--device", default="cuda",
                        help="torch device; default cuda.")
    parser.add_argument("--layers", default="0,1,2,3",
                        help="Comma-separated layer indices to extract. Default 0,1,2,3.")
    parser.add_argument(
        "--reducers",
        default="prefix_mean_pool,per_modality_mean_pool,per_modality_max_pool,"
                "per_modality_spatial_pool_16,per_modality_spatial_pool_4",
        help="Comma-separated reducer types to run.",
    )
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Rebuild even if the output pkl already exists.")
    parser.add_argument(
        "--factors-yaml", default=None,
        help="Path to a YAML listing OfflineWriter-capable factors "
             "(F1b-A / F1b-T) — see exp/common/factor_postprocess.py. "
             "Each (layer, reducer) artifact gets per-entry "
             "`payload.factors` + a top-level `library_stats` field."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    build_matrix(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        device=args.device,
        layers=_parse_layers(args.layers),
        reducer_types=_parse_list(args.reducers),
        skip_existing=not args.no_skip_existing,
        factors_yaml=args.factors_yaml,
    )


if __name__ == "__main__":
    main()
