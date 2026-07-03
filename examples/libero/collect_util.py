"""Pure gate-research collection helpers (no LIBERO/CUDA/torch-sim deps).

``examples.libero.main`` imports the LIBERO simulator at module top, so any code
that needs these helpers — the conductor ``episode_runner`` and unit tests —
would be forced to install the sim just to reach a pure ndarray→list codec or the
canonical episode-id formula. This module holds those pure helpers (numpy only)
so both the standalone harness (``main``) and the conductor harness
(``episode_runner``) import the SAME implementations, and both are unit-testable
without LIBERO (plan §4 decoupling mandate / §19.B6 single-source identity).

Coupling map:
  DEPENDS ON:  numpy
  CONSUMED BY: examples.libero.main, examples.libero.episode_runner, tests
"""

from __future__ import annotations

import numpy as np

# Bump when the collected row schema changes (collect field encoding, key names).
# The single source of truth for both harnesses.
COLLECTOR_SCHEMA_VERSION = 1


def compute_global_episode_id(task_id: int, episode_idx: int, num_trials_per_task: int) -> int:
    """Derive the global episode id from (task_id, subset episode idx).

    Both serial and concurrent evaluation paths MUST agree on this mapping
    (plan §4 / §19.B6): if they diverge, client-side GT HDF5, server-side
    collected HDF5, results JSON, and downstream Layer F unit keys fall out of
    alignment as soon as the caller uses ``--task_ids`` or ``--episode_filter``
    to skip episodes. Keep as the only place this formula lives.
    """
    return int(task_id) * int(num_trials_per_task) + int(episode_idx)


def encode_collect_meta(collect_meta):
    """Convert a ``__collect_meta__`` payload to JSONL/msgpack-safe lists.

    The server sends ``{"collect": {field: np.ndarray | None}, "searched": bool}``
    (plus an optional ``kb_id`` provenance string). Conductor's ``msgpack.packb``
    and the JSONL ``json.dumps`` cannot encode ndarrays, so upcast each array to
    float32 (lossless from the wire's float32) and ``.tolist()`` here at the
    client boundary. Returns ``{"collect": {...lists...} | None, "searched": bool}``
    (with ``kb_id`` passed through when present) ready to merge into a per-step
    row, or ``None`` when collection is off.
    """
    if collect_meta is None:
        return None
    collect = collect_meta.get("collect")
    if collect is not None:
        collect = {
            name: (None if arr is None else np.asarray(arr, dtype=np.float32).tolist())
            for name, arr in collect.items()
        }
    out = {"collect": collect, "searched": bool(collect_meta.get("searched", True))}
    # Pass through server-side provenance (kb_id) — dropping it here left the
    # episode-summary's kb_id permanently None.
    if collect_meta.get("kb_id") is not None:
        out["kb_id"] = collect_meta.get("kb_id")
    return out


def merge_collect(row: dict, collect_meta) -> None:
    """Merge an encoded collect payload into a per-step row in place.

    Provenance: the row carries ``searched`` + ``collector_schema_version`` and
    (when present) the ``collect`` dict whose keys ARE the collected fields.
    A per-episode ``episode_summary`` row is written separately (see
    :func:`episode_summary_row`) for provenance not fully derivable from the
    per-step rows (``seed``, ``kb_id``).
    """
    if collect_meta is None:
        return
    row["searched"] = collect_meta.get("searched", True)
    row["collector_schema_version"] = COLLECTOR_SCHEMA_VERSION
    collect = collect_meta.get("collect")
    if collect is not None:
        row["collect"] = collect


def update_summary_acc(acc: dict, collect_meta) -> None:
    """Accumulate per-episode collection provenance for the episode-summary row."""
    if collect_meta is None:
        return
    acc["collected"] = True
    acc["searched_all"] = acc["searched_all"] and bool(collect_meta.get("searched", True))
    if collect_meta.get("kb_id"):
        acc["kb_id"] = collect_meta.get("kb_id")
    collect = collect_meta.get("collect")
    if collect is not None:
        acc["fields"] = sorted(collect.keys())


def episode_summary_row(acc, *, task_uid, yaml_id, task_id, subset_init_state_idx,
                        episode_id, phase, seed, success) -> dict:
    """Build the per-episode provenance summary row (plan §2.5).

    Distinguished by ``_kind`` so per-step analysis filters it out. Carries the
    join key + provenance not fully derivable from per-step rows (seed, kb_id).
    """
    return {
        "_kind": "episode_summary",
        "task_uid": task_uid,
        "yaml_id": yaml_id,
        "task_id": int(task_id),
        "subset_init_state_idx": int(subset_init_state_idx),
        "episode_id": int(episode_id),
        "phase": phase,
        "seed": seed,
        "num_steps": acc["n"],
        "success": bool(success),
        "searched_all": acc["searched_all"],
        "collect_fields": acc["fields"],
        "kb_id": acc["kb_id"],
        "collector_schema_version": COLLECTOR_SCHEMA_VERSION,
    }
