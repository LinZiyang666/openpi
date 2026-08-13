"""Read-only access to offline cache-library artifacts for E1/E2/E3.

Responsibilities:
  * load a ``cp1_*.pkl`` artifact into an indexed, read-only container;
  * walk the ``prev_ids`` ancestor chain with production semantics (a missing
    ancestor is reported as ``None`` so callers can score it as 0.0, never
    silently truncated);
  * reproduce the production output chain so that actions compared in E1/E3
    are the client-space actions LIBERO actually executes, not the raw
    model-space chunk stored in the payload.

Public interface: :class:`Library`, :func:`load_library`,
:func:`walk_ancestors`, :func:`build_output_chain`, :func:`executed_action`,
:func:`raw_action`.

Key dependencies: ``openpi.cache.storage_types.CacheEntry`` (artifact schema),
``openpi.transforms.Unnormalize`` and ``openpi.policies.libero_policy``
(output chain), both used directly so the offline path cannot drift from
production.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
import pickle
from typing import Any, Callable, Iterator, Optional

import numpy as np

# ------------------------------------------------------------------
# Artifact container
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Library:
    """An offline cache artifact, indexed for lookup by id and by trajectory.

    ``meta`` carries provenance that downstream manifests must record: the
    artifact path, the key-builder type, and whether the entries carry outcome
    tags. Legacy artifacts have ``outcome=None`` on every entry -- that is not
    an error (the build-time success filter is a separate mechanism), but any
    claim that depends on per-entry outcome labels is invalid on them.
    """

    entries: list[Any]
    by_id: dict[str, Any]
    by_traj: dict[str, list[Any]]
    vector_dims: dict[str, int]
    key_builder_type: str
    meta: dict[str, Any]

    def __len__(self) -> int:
        return len(self.entries)

    def trajectories(self) -> Iterator[tuple[str, list[Any]]]:
        """Yield ``(trajectory_id, entries sorted by step_idx)`` pairs."""
        for traj_id, items in self.by_traj.items():
            yield traj_id, items


def load_library(path: str | pathlib.Path) -> Library:
    """Load a pickled cache artifact and build the id / trajectory indices.

    The artifact is treated as read-only; entries keep their original objects
    so that ``id`` values stay join-compatible with per-step rollout logs.
    """
    path = pathlib.Path(path)
    with path.open("rb") as fh:
        raw = pickle.load(fh)

    entries = list(raw["entries"])
    by_id = {e.id: e for e in entries}
    by_traj: dict[str, list[Any]] = {}
    for entry in entries:
        by_traj.setdefault(entry.trajectory_id, []).append(entry)
    for items in by_traj.values():
        items.sort(key=lambda e: (e.step_idx if e.step_idx is not None else -1))

    outcomes = {getattr(e, "outcome", None) for e in entries}
    meta = {
        "artifact_path": str(path),
        "n_entries": len(entries),
        "n_trajectories": len(by_traj),
        "outcome_values": sorted(o for o in outcomes if o is not None),
        # True for legacy artifacts: the build-time filter ran, but the tag was
        # never written onto the entries.
        "outcome_all_none": outcomes == {None},
    }
    return Library(
        entries=entries,
        by_id=by_id,
        by_traj=by_traj,
        vector_dims=dict(raw.get("vector_dims", {})),
        key_builder_type=str(raw.get("key_builder_type", "")),
        meta=meta,
    )


# ------------------------------------------------------------------
# Ancestor chain
# ------------------------------------------------------------------


def walk_ancestors(lib: Library, entry_id: str, depth: int) -> list[Optional[str]]:
    """Return the ``depth`` ancestor ids of ``entry_id``, newest first.

    Element ``l`` is the id of the ancestor ``l+1`` steps back, or ``None``
    when the chain ends (episode prefix) or the id is dangling. Production
    scores a missing ancestor as 0.0 rather than skipping the level, so the
    list is always exactly ``depth`` long -- callers must not shorten it.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    out: list[Optional[str]] = []
    current = lib.by_id.get(entry_id)
    for _ in range(depth):
        if current is None:
            out.append(None)
            continue
        prev_ids = getattr(current, "prev_ids", None) or []
        if not prev_ids:
            current = None
            out.append(None)
            continue
        # Artifacts built by this project are linear chains; a fan-in would make
        # "the" ancestor ambiguous, so refuse instead of picking arbitrarily.
        if len(prev_ids) > 1:
            raise ValueError(
                f"entry {getattr(current, 'id', '?')!r} has {len(prev_ids)} parents; "
                "multi-parent chains are out of scope for this experiment"
            )
        parent_id = prev_ids[0]
        out.append(parent_id if parent_id in lib.by_id else None)
        current = lib.by_id.get(parent_id)
    return out


# ------------------------------------------------------------------
# Output chain (model-space chunk -> client-space executed action)
# ------------------------------------------------------------------

#: Default norm-stats asset for the pi05 LIBERO policy.
DEFAULT_NORM_STATS = (
    "assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
)


@functools.lru_cache(maxsize=4)
def build_output_chain(
    norm_stats_path: str = DEFAULT_NORM_STATS,
    *,
    use_quantiles: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the production output chain for actions.

    Mirrors ``policy_config.create_trained_policy``'s output transforms
    (``policy_config.py``): ``model_transforms.outputs`` (empty for PI05)
    -> ``Unnormalize(norm_stats, use_quantiles)`` -> ``data_transforms.outputs``
    (``LiberoOutputs``, which slices the first 7 dims). Slicing alone is *not*
    the executed action: unnormalisation is a per-dimension affine map, so it
    changes L2 residuals and distance thresholds.

    ``use_quantiles`` follows ``model_type != PI0``; pi05 is quantile-normalised.
    Returns a callable mapping ``[T, 32]`` model-space chunks to ``[T, 7]``
    client-space actions.
    """
    from openpi import transforms as _transforms
    from openpi.policies import libero_policy

    with pathlib.Path(norm_stats_path).open() as fh:
        raw = json.load(fh)
    stats = raw["norm_stats"]["actions"]
    norm_stats = {
        "actions": _transforms.NormStats(
            mean=np.asarray(stats["mean"], dtype=np.float32),
            std=np.asarray(stats["std"], dtype=np.float32),
            q01=np.asarray(stats["q01"], dtype=np.float32) if "q01" in stats else None,
            q99=np.asarray(stats["q99"], dtype=np.float32) if "q99" in stats else None,
        )
    }
    unnormalize = _transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles)
    to_client = libero_policy.LiberoOutputs()

    def chain(chunk: np.ndarray) -> np.ndarray:
        data = {"actions": np.asarray(chunk, dtype=np.float32)}
        return np.asarray(to_client(unnormalize(data))["actions"])

    return chain


def executed_action(
    entry: Any,
    *,
    out_chain: Callable[[np.ndarray], np.ndarray],
    index: int = 0,
) -> np.ndarray:
    """Return the client-space executed action at chunk position ``index``.

    Always goes through ``out_chain``; a bare ``action_chunk[index][:7]`` is
    model-space and must never stand in for this value.
    """
    chunk = np.asarray(entry.payload.action_chunk, dtype=np.float32)
    if chunk.ndim != 2:
        raise ValueError(f"expected a [T, D] action chunk, got shape {chunk.shape}")
    return out_chain(chunk)[index]


def raw_action(entry: Any, index: int = 0) -> np.ndarray:
    """Return the raw model-space action (all 32 dims), for appendix sensitivity only."""
    return np.asarray(entry.payload.action_chunk, dtype=np.float32)[index]
