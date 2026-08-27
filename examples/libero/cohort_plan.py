"""Dispatch-surface cohort plan adapter for the LIBERO client.

Lightweight (no libero / torch imports) so tests and the collection launcher
can consume it without the simulator environment. Bridges the cohort plan
emitted by ``exp/dispatch_surface/collect_query_cohort.py plan`` onto the
LIBERO client's existing episode-filter machinery, and assembles the full
four-field identity metadata the episode collector persists into H5 attrs
(task_id / orig_init_state_idx / subset_init_state_idx / split).

The loop index of the LIBERO client is the SUBSET position inside the
materialised init pool; the plan carries both that and the official index, so
the assembled metadata can never conflate the two spaces (G2R2-B1).
"""

from __future__ import annotations

import json
import pathlib


def load_cohort_map(path: str) -> dict[tuple[int, int], dict]:
    """(task_id, subset_init_state_idx) -> {orig, split} from a cohort plan."""
    plan = json.loads(pathlib.Path(path).read_text())
    episodes = plan.get("episodes")
    if not episodes:
        raise ValueError(f"cohort plan {path} has no episodes")
    out: dict[tuple[int, int], dict] = {}
    for e in episodes:
        key = (int(e["task_id"]), int(e["subset_init_state_idx"]))
        if key in out:
            raise ValueError(f"cohort plan duplicates (task, subset)={key}")
        out[key] = {
            "orig": int(e["orig_init_state_idx"]),
            "split": str(e["split"]),
        }
    return out


def episode_filter_pairs(cohort_map: dict[tuple[int, int], dict]):
    """(filter_pairs, orig_map) in the shapes _load_episode_filter returns."""
    filter_pairs = set(cohort_map)
    orig_map = {key: info["orig"] for key, info in cohort_map.items()}
    return filter_pairs, orig_map


def cohort_extra_metadata(
    cohort_map: dict[tuple[int, int], dict],
    task_id: int,
    subset_init_state_idx: int,
    orig_init_state_idx: int,
) -> dict:
    """Full identity metadata for one planned episode.

    Raises when the episode is not in the plan or the official index the
    client derived disagrees with the plan — silent divergence here is
    exactly the fit/cal mislabelling failure this module exists to prevent.
    """
    key = (int(task_id), int(subset_init_state_idx))
    info = cohort_map.get(key)
    if info is None:
        raise ValueError(f"(task, subset)={key} is not part of the cohort plan")
    if info["orig"] != int(orig_init_state_idx):
        raise ValueError(
            f"client derived official init {orig_init_state_idx} for {key} but the "
            f"plan says {info['orig']} — init pool and plan are out of sync"
        )
    return {
        "task_id": int(task_id),
        "orig_init_state_idx": int(orig_init_state_idx),
        "subset_init_state_idx": int(subset_init_state_idx),
        "split": info["split"],
    }
