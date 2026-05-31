"""HDF5 trajectory source for the cache latency replay benchmark.

Reads the real per-episode HDF5 files produced by the data collector
(``exp/common/data/db/libero_cache/<suite>/*.h5``) and yields, for each
step, the inputs needed to drive ``CacheOrchestrator.check(CP1, ...)``
without loading any model: a reconstructed ``Stage1Output`` (vision and
prompt token embeddings already projected to LLM space, so no vision
tower) plus the ground-truth ``clean_action`` of that step.

One HDF5 file == one episode. Public interface: ``H5EpisodeSource``
iterating ``EpisodeInput`` -> ``StepInput``.

The H5 -> fake-Stage1 reconstruction reuses
``exp.common.build_in_memory_cache_artifact._build_fake_stage1`` (a
controlled cross-module private coupling, plan R6) so the benchmark and
the artifact builder stay bit-identical on that path.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import glob
import os

import h5py
import numpy as np
import torch

# Controlled cross-module private reuse (plan R6): the exact same H5 ->
# fake Stage1 reconstruction the artifact builder uses, so retrieval keys
# match the library that was built from these same H5 files. No src/ change.
from exp.common.build_in_memory_cache_artifact import _build_fake_stage1
from exp.common.build_in_memory_cache_artifact import _FakeStage1

# ----------------------------------------------------------------------
# Step / Episode records
# ----------------------------------------------------------------------


@dataclass
class StepInput:
    """One inference step replayed from an HDF5 ``step_<N>`` group."""

    fake_stage1: _FakeStage1  # reconstructed Stage1Output (CPU float32)
    clean_action: torch.Tensor  # [H, action_dim] ground-truth action chunk
    step_idx: int


@dataclass
class EpisodeInput:
    """One episode == one HDF5 file (lazily reads steps on iteration)."""

    path: str
    task_key: str
    episode_id: int
    success: bool
    num_steps: int

    def steps(self) -> Iterator[StepInput]:
        """Yield each ``step_<N>`` group in numeric order as a ``StepInput``."""
        with h5py.File(self.path, "r") as f:
            for name in _sorted_step_names(f):
                group = f[name]
                yield StepInput(
                    fake_stage1=_build_fake_stage1(group),
                    clean_action=_read_clean_action(group),
                    step_idx=_step_index(name),
                )


# ----------------------------------------------------------------------
# Source
# ----------------------------------------------------------------------


class H5EpisodeSource:
    """Scan an H5 directory and iterate episodes in filename order."""

    def __init__(self, h5_dir: str) -> None:
        self._h5_dir = h5_dir
        self._paths = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
        if not self._paths:
            raise FileNotFoundError(f"No .h5 files under {h5_dir!r}")

    def __len__(self) -> int:
        return len(self._paths)

    def iter_episodes(self) -> Iterator[EpisodeInput]:
        """Yield one ``EpisodeInput`` per HDF5 file (filename-sorted)."""
        for path in self._paths:
            with h5py.File(path, "r") as f:
                attrs = f.attrs
                num_steps = sum(1 for k in f.keys() if k.startswith("step_"))
                yield EpisodeInput(
                    path=path,
                    task_key=str(attrs.get("task", "")),
                    episode_id=int(attrs.get("episode_id", -1)),
                    success=bool(attrs.get("success", False)),
                    num_steps=num_steps,
                )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _step_index(name: str) -> int:
    return int(name.split("_")[-1])


def _sorted_step_names(f: h5py.File) -> list[str]:
    return sorted((k for k in f.keys() if k.startswith("step_")), key=_step_index)


def _read_clean_action(group: h5py.Group) -> torch.Tensor:
    """Read the step's ground-truth action chunk as ``[H, action_dim]``.

    This matches the unbatched action shape ``broadcast_action`` /
    ``buffer_for_write`` expect (the orchestrator indexes ``[0]`` for the
    per-step anchor); the library entries were built from the same field.
    """
    return torch.from_numpy(np.array(group["clean_action"])).float()
