"""Build InMemoryBackend artifact from HDF5 episode data.

Usage:
    uv run exp/build_in_memory_cache_artifact.py \
        --data-dir data/libero_spatial \
        --builder-type cp1_mean_pool \
        --output data/cache_artifacts/libero_spatial/cp1_mean_pool.pkl

    # Build all 4 artifacts at once:
    for bt in cp1_mean_pool cp1_spatial_pool_16 cp1_spatial_pool_64 cp1_max_pool; do
        uv run exp/build_in_memory_cache_artifact.py \
            --data-dir data/libero_spatial \
            --builder-type $bt \
            --output data/cache_artifacts/libero_spatial/${bt}.pkl
    done

Input: HDF5 episode files with vision_0/1/2, prompt_emb, robot_state fields
Output: pickle file loadable by InMemoryBackend.load_artifact()
"""

from __future__ import annotations

import argparse
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
    "cp1_spatial_pool_64": {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32},
    "cp1_max_pool":        {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32},
}


def _get_vector_dims(builder_type: str) -> dict[str, int]:
    return _VECTOR_DIMS[builder_type]


# ---------------------------------------------------------------------------
# Builder factory
# ---------------------------------------------------------------------------


def _create_builder(builder_type: str):
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
    if builder_type not in builders:
        raise ValueError(f"Unknown builder_type: {builder_type}. Valid: {list(builders)}")
    return builders[builder_type]()


# ---------------------------------------------------------------------------
# Fake stage1 output
# ---------------------------------------------------------------------------


class _FakeStage1:
    """Mimics Stage1Output structure for KeyBuilder.collect()."""

    def __init__(self, prefix_embs: torch.Tensor, state: torch.Tensor):
        self.prefix_embs = prefix_embs  # [1, prefix_len, emb_dim]
        self.state = state               # [1, state_dim]


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
# Main build function
# ---------------------------------------------------------------------------


def _process_episode(h5_path_str: str, builder_type: str, checkpoint_id_str: str) -> list | None:
    """Process a single H5 file in a worker process. Returns list of CacheEntry or None."""
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID

    h5_path = Path(h5_path_str)
    cp_id = CheckpointID[checkpoint_id_str]
    builder = _create_builder(builder_type)

    with h5py.File(h5_path, "r") as f:
        task = str(f.attrs.get("task", ""))
        success = bool(f.attrs.get("success", False))
        if not success:
            return None

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

        return episode_entries


def build_artifact(
    data_dir: str,
    builder_type: str,
    checkpoint_id_str: str = "CP1",
    workers: int = 0,
) -> dict:
    """Build artifact dict from HDF5 data.

    Scans data_dir for .h5 files, uses only successful episodes.
    For each step: collect() -> build() -> CacheEntry.
    Entry id = "{episode_file_stem}_{step_name}" (deterministic, traceable).
    action_chunk keeps real horizon from data (no padding to [50,32]).

    Args:
        workers: Number of parallel workers. 0 = all CPUs.
    """
    vector_dims = _get_vector_dims(builder_type)

    h5_paths = sorted(Path(data_dir).rglob("*.h5"))
    if not h5_paths:
        logger.warning("No .h5 files found in %s", data_dir)
        return {"key_builder_type": builder_type, "checkpoint_id": checkpoint_id_str,
                "vector_dims": vector_dims, "entries": []}

    num_workers = workers if workers > 0 else (os.cpu_count() or 1)
    logger.info("Processing %d H5 files with %d workers", len(h5_paths), num_workers)

    entries: list = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_process_episode, str(p), builder_type, checkpoint_id_str): p
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
    return {
        "key_builder_type": builder_type,
        "checkpoint_id": checkpoint_id_str,
        "vector_dims": vector_dims,
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build InMemoryBackend artifact from HDF5 data")
    parser.add_argument("--data-dir", required=True, help="Directory with .h5 episode files")
    parser.add_argument("--builder-type", required=True, choices=list(_VECTOR_DIMS.keys()))
    parser.add_argument("--output", required=True, help="Output .pkl path")
    parser.add_argument("--checkpoint-id", default="CP1", choices=["CP1"])
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (0 = all CPUs)")
    args = parser.parse_args()

    artifact = build_artifact(args.data_dir, args.builder_type, args.checkpoint_id, workers=args.workers)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved {len(artifact['entries'])} entries to {args.output}")


if __name__ == "__main__":
    main()
