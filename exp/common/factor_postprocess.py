"""Helper: enrich an artifact's entries with per-entry `payload.factors`
and compute the artifact-level `library_stats`.

Used by `exp/common/build_*.py` scripts after their per-episode entry
list is finalized but before pickling. Tolerates both numpy.ndarray and
torch.Tensor payload tensors via `torch.as_tensor`, so the helper is
safe to call **after** the primary builder's `_detach_entries` has run
in its ProcessPool subprocess (post-detach the payload tensors are
numpy, but `LibraryStats.compute_from_entries` and OfflineWriter
implementations bridge through `torch.as_tensor`).

Public surface:

  enrich_artifact_with_factors(entries, offline_writers, *,
                               active_eps_action=0.01,
                               active_eps_state=0.01) -> LibraryStats

  _load_offline_writers_from_yaml(yaml_path) -> list[OfflineWriter]

Coupling map:
  DEPENDS ON:  openpi.cache.components.factors.base (LibraryStats),
               openpi.cache.components.factors.registry (build / get_class),
               openpi.cache.config (ConfigValidationError)
  CONSUMED BY: exp/common/build_in_memory_cache_artifact.py,
               exp/common/build_clip_cache_artifact.py,
               exp/common/build_llm_layer_matrix.py
"""

from __future__ import annotations

import logging
from typing import Any

import yaml

from openpi.cache.components.factors import registry
from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.config import ConfigValidationError

logger = logging.getLogger(__name__)


def enrich_artifact_with_factors(
    entries: list[Any],
    offline_writers: list[Any],
    *,
    library_stats: LibraryStats | None = None,
    active_eps_action: float = 0.01,
    active_eps_state: float = 0.01,
) -> LibraryStats:
    """In-place: write per-entry `payload.factors` (per writer) + compute
    `LibraryStats` from the entry pool. Returns LibraryStats so the caller
    can attach it to the artifact dict's `library_stats` field.

    G1 R3 Item 4 / plan §16 B6.5: ``library_stats`` may be passed in to
    skip the recompute (used by the ``enrich-existing-pkl`` smoke
    sub-command which reads stats from the input pkl and saves a 30+min
    recompute on every run). Default ``None`` preserves the legacy
    HDF5 build path: stats are computed from the entry pool.

    Per-episode segmentation: writers receive per-trajectory entry slices
    keyed by `entry.trajectory_id`, sorted by `entry.step_idx` (the
    actual field name on `CacheEntry`). Entries with `step_idx is None`
    go to a separate "unnamed" bucket and are processed in collection
    order; if any such bucket exists the helper logs a warning since the
    writer's window semantics depend on monotonic step ordering.

    Tolerates both torch.Tensor and numpy.ndarray payload tensors:
    `LibraryStats.compute_from_entries` and OfflineWriter implementations
    use `torch.as_tensor(...)` internally. Safe to call after
    `_detach_entries` has run.
    """
    if library_stats is None:
        library_stats = LibraryStats.compute_from_entries(
            entries,
            active_eps_action=active_eps_action,
            active_eps_state=active_eps_state,
        )
    if not offline_writers:
        return library_stats

    # Group entries by trajectory_id, then sort by step_idx with a
    # separate bucket for None step_idx (logged but not raised — we still
    # produce factors so the artifact stays loadable).
    by_traj: dict[Any, list[Any]] = {}
    for e in entries:
        by_traj.setdefault(e.trajectory_id, []).append(e)

    for traj_id, traj_entries in by_traj.items():
        named = [e for e in traj_entries if e.step_idx is not None]
        unnamed = [e for e in traj_entries if e.step_idx is None]
        named.sort(key=lambda e: e.step_idx)
        if unnamed:
            logger.warning(
                "trajectory %s has %d entries with step_idx=None; "
                "appending in collection order (window descriptors may be invalid)",
                traj_id, len(unnamed),
            )
        ordered = named + unnamed

        for writer in offline_writers:
            per_entry_factors = writer.compute_for_episode(ordered, library_stats)
            for entry, factors in zip(ordered, per_entry_factors, strict=True):
                if entry.payload.factors is None:
                    entry.payload.factors = {}
                entry.payload.factors.update(factors)

    return library_stats


def _load_offline_writers_from_yaml(yaml_path: str) -> list[Any]:
    """Read a minimal YAML and return list of OfflineWriter instances.

    Validates that each ``type`` is registered AND has
    ``compute_for_episode`` (i.e. is one of the 8 offline factors:
    ``jerk_offline_action`` / ``jerk_offline_state`` / ... /
    ``path_length_offline_state``). Online-only factors (8 online + topk)
    raise ConfigValidationError to make misconfiguration loud.

    YAML shape (refactor):

      factors:
        - type: jerk_offline_action
          params:
            windows: [{past: 0, future: 5}, {past: 5, future: 5}]
        - type: dispersion_offline_state
          params: { ... }
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    out: list[Any] = []
    for entry in data.get("factors", []):
        type_name = entry["type"]
        cls = registry.get_class(type_name)
        if not hasattr(cls, "compute_for_episode"):
            raise ConfigValidationError(
                f"factor type {type_name!r} has no compute_for_episode "
                "(only the 8 offline factors implement the OfflineWriter surface; "
                "online + topk_action_variance run only at verdict time)"
            )
        out.append(registry.build(type_name, **entry.get("params", {})))
    return out
