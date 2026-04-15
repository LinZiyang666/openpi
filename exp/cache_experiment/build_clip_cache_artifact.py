"""Build InMemoryBackend artifact from HDF5 data using CLIP image encoder.

Single-process GPU batch encoding.  Does NOT use ProcessPoolExecutor
(CLIP model must be loaded once; multi-process GPU duplication would OOM).

Input:  HDF5 episode files with input_images, prompt_emb, robot_state fields.
Output: pickle file loadable by InMemoryBackend.load_artifact().

Usage:
    uv run exp/cache_experiment/build_clip_cache_artifact.py \
        --data-dir data/db/libero_cache/libero_spatial \
        --clip-model ViT-B-32 \
        --clip-pretrained openai \
        --output data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl

    # Use ViT-L-14 instead:
    uv run exp/cache_experiment/build_clip_cache_artifact.py \
        --data-dir data/db/libero_cache/libero_spatial \
        --clip-model ViT-L-14 \
        --clip-pretrained openai \
        --output data/cache_artifacts/libero_spatial/clip_vit_l_14.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from openpi.cache.components.clip_key_builder import (
    _SLOT_TO_FIELD,
    clip_prompt_key_from_tokens,
    clip_state_key,
)
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import PROMPT_EMB, ROBOT_STATE, CheckpointID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIP model wrapper (loaded once in main process)
# ---------------------------------------------------------------------------


class _CLIPEncoder:
    """Thin wrapper around open_clip for batch image encoding."""

    def __init__(self, model_name: str, pretrained: str, device: str) -> None:
        import open_clip

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device(device)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self.embed_dim = model.visual.output_dim
        logger.info(
            "Loaded %s (pretrained=%s) on %s, embed_dim=%d",
            model_name, pretrained, device, self.embed_dim,
        )

    def encode_batch(self, images: list[np.ndarray]) -> torch.Tensor:
        """Encode a list of uint8 numpy images.

        Args:
            images: list of (H, W, 3) uint8 ndarray.

        Returns:
            [N, embed_dim] CPU float32 L2-normalized tensor.
        """
        tensors = [self._preprocess(Image.fromarray(img)) for img in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embs = self._model.encode_image(batch, normalize=True)
        return embs.cpu().float().contiguous()


# ---------------------------------------------------------------------------
# HDF5 processing
# ---------------------------------------------------------------------------


def _step_sort_key(name: str) -> tuple[bool, int, str]:
    suffix = name.split("_", 1)[1] if "_" in name else ""
    if suffix.isdigit():
        return (False, int(suffix), name)
    return (True, 0, name)


def _process_episode(
    h5_path: Path,
    encoder: _CLIPEncoder,
    enabled_fields: set[str],
    checkpoint_id_str: str,
    batch_size: int,
) -> list | None:
    """Process one HDF5 episode file. Returns list of CacheEntry or None."""
    cp_id = CheckpointID[checkpoint_id_str]

    with h5py.File(h5_path, "r") as f:
        task = str(f.attrs.get("task", ""))
        success = bool(f.attrs.get("success", False))
        if not success:
            return None

        step_names = sorted(
            (k for k in f.keys() if k.startswith("step_")),
            key=_step_sort_key,
        )
        if not step_names:
            return None

        trajectory_id = h5_path.stem

        # -- Pass 1: collect all images for batch encoding --
        # Per-step: {field_name: image_np} for enabled vision slots.
        step_image_map: list[dict[str, np.ndarray]] = []
        all_images: list[np.ndarray] = []
        all_image_indices: list[tuple[int, str]] = []  # (step_idx_in_list, field_name)

        for step_name in step_names:
            group = f[step_name]
            img_grp = group.get("input_images")
            step_imgs: dict[str, np.ndarray] = {}
            if img_grp is not None:
                for slot_name, field_name in _SLOT_TO_FIELD:
                    if field_name not in enabled_fields:
                        continue
                    if slot_name not in img_grp:
                        continue
                    img = np.array(img_grp[slot_name])
                    step_imgs[field_name] = img
                    all_images.append(img)
                    all_image_indices.append((len(step_image_map), field_name))
            step_image_map.append(step_imgs)

        # -- Batch encode all images --
        all_embeddings: list[torch.Tensor] = []
        for batch_start in range(0, len(all_images), batch_size):
            batch = all_images[batch_start:batch_start + batch_size]
            embs = encoder.encode_batch(batch)  # [N, embed_dim]
            for i in range(embs.shape[0]):
                all_embeddings.append(embs[i])

        # Scatter embeddings back to per-step dicts.
        step_vision_keys: list[dict[str, torch.Tensor]] = [{} for _ in step_names]
        for emb_idx, (step_list_idx, field_name) in enumerate(all_image_indices):
            step_vision_keys[step_list_idx][field_name] = all_embeddings[emb_idx]

        # -- Pass 2: build CacheEntry per step --
        episode_entries: list[CacheEntry] = []
        for i, step_name in enumerate(step_names):
            group = f[step_name]
            query_keys: dict[str, torch.Tensor] = {}

            # Vision keys from CLIP.
            query_keys.update(step_vision_keys[i])

            # prompt_emb: mean pool from HDF5 prompt_emb field.
            if PROMPT_EMB in enabled_fields and PROMPT_EMB in group:
                prompt = torch.from_numpy(np.array(group[PROMPT_EMB])).float()
                query_keys[PROMPT_EMB] = clip_prompt_key_from_tokens(prompt)

            # robot_state: raw vector.
            if ROBOT_STATE in enabled_fields and ROBOT_STATE in group:
                state = torch.from_numpy(np.array(group[ROBOT_STATE])).float()
                if state.dim() > 1:
                    state = state.squeeze(0)
                query_keys[ROBOT_STATE] = clip_state_key(state)

            step_idx = None
            suffix = step_name.split("_", 1)[1] if "_" in step_name else ""
            if suffix.isdigit():
                step_idx = int(suffix)

            entry_id = f"{trajectory_id}:{step_idx if step_idx is not None else step_name}"

            action = torch.from_numpy(np.array(group["clean_action"])).float()
            if action.dim() == 1:
                action = action.unsqueeze(0)

            # Mirror build_in_memory_cache_artifact.py:250-266 — read noise_action_*
            # to populate payload.intermediates, so WARM_START judges can lift
            # cached x_t from this artifact without hitting the Orchestrator
            # "WARM_START payload incomplete" downgrade path.
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
                for nidx in sorted(noise_indices):
                    t = round(1.0 - nidx / _NUM_STEPS, 4)
                    intermediates[t] = torch.from_numpy(
                        np.array(group[f"noise_action_{nidx}"])
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

        # Link prev_ids / next_ids within episode.
        for i in range(len(episode_entries)):
            if i > 0:
                episode_entries[i].prev_ids = [episode_entries[i - 1].id]
            if i < len(episode_entries) - 1:
                episode_entries[i].next_ids = [episode_entries[i + 1].id]

        return episode_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_artifact(
    data_dir: str,
    clip_model: str,
    clip_pretrained: str,
    enabled_fields: list[str],
    device: str = "cpu",
    checkpoint_id_str: str = "CP1",
    batch_size: int = 64,
) -> dict:
    """Build artifact dict from HDF5 data using CLIP encoder."""
    encoder = _CLIPEncoder(clip_model, clip_pretrained, device)

    # Determine vector_dims from encoder + enabled fields.
    vector_dims: dict[str, int] = {}
    for _, field_name in _SLOT_TO_FIELD:
        if field_name in enabled_fields:
            vector_dims[field_name] = encoder.embed_dim
    if PROMPT_EMB in enabled_fields:
        vector_dims[PROMPT_EMB] = 2048  # PaliGemma emb_dim
    if ROBOT_STATE in enabled_fields:
        vector_dims[ROBOT_STATE] = 32

    enabled_set = set(enabled_fields)

    h5_paths = sorted(Path(data_dir).rglob("*.h5"))
    if not h5_paths:
        logger.warning("No .h5 files found in %s", data_dir)

    entries: list = []
    skipped = 0
    for idx, h5_path in enumerate(h5_paths):
        result = _process_episode(
            h5_path, encoder, enabled_set, checkpoint_id_str, batch_size,
        )
        if result is None:
            skipped += 1
        else:
            entries.extend(result)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(h5_paths):
            logger.info(
                "Progress: %d/%d files, %d entries, %d skipped",
                idx + 1, len(h5_paths), len(entries), skipped,
            )

    # Prune vector_dims to only fields that actually appear in entries.
    observed_fields: set[str] = set()
    for entry in entries:
        observed_fields.update(entry.query_keys.keys())
    vector_dims = {k: v for k, v in vector_dims.items() if k in observed_fields}

    logger.info(
        "Built %d entries from %d files (%d skipped) for clip/%s/%s",
        len(entries), len(h5_paths), skipped, clip_model, clip_pretrained,
    )

    return {
        "key_builder_type": "clip",
        "checkpoint_id": checkpoint_id_str,
        "vector_dims": vector_dims,
        "clip_model_name": clip_model,
        "clip_pretrained": clip_pretrained,
        "entries": entries,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Build InMemoryBackend artifact from HDF5 data using CLIP encoder",
    )
    parser.add_argument("--data-dir", required=True, help="Directory with .h5 episode files")
    parser.add_argument("--clip-model", default="ViT-B-32", help="open_clip model name")
    parser.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag")
    parser.add_argument("--output", required=True, help="Output .pkl path")
    parser.add_argument("--checkpoint-id", default="CP1", choices=["CP1"])
    parser.add_argument("--device", default="cpu", help="Device for CLIP encoding (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for CLIP encoding")
    parser.add_argument(
        "--fields", default="vision_0,robot_state",
        help="Comma-separated enabled fields (default: vision_0,robot_state)",
    )
    args = parser.parse_args()

    enabled_fields = [f.strip() for f in args.fields.split(",")]

    artifact = build_artifact(
        data_dir=args.data_dir,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        enabled_fields=enabled_fields,
        device=args.device,
        checkpoint_id_str=args.checkpoint_id,
        batch_size=args.batch_size,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(artifact, f)
    print(
        f"Saved {len(artifact['entries'])} entries to {args.output} "
        f"(clip_model={args.clip_model}, clip_pretrained={args.clip_pretrained}, "
        f"vector_dims={artifact['vector_dims']})"
    )


if __name__ == "__main__":
    main()
