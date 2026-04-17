from __future__ import annotations

import torch

from exp.common.calibrate_score_sum_stats import compute_field_stats, sanity_check
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


def _entry(eid: str, *, task: str, query_keys: dict[str, torch.Tensor]) -> CacheEntry:
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys=query_keys,
        payload=CachePayload(action_chunk=torch.zeros(10, 32), task_key=task),
    )


def test_compute_field_stats_for_cosine_returns_bounded_percentiles():
    entries = [
        _entry("a", task="t1", query_keys={"vision_0": torch.tensor([1.0, 0.0])}),
        _entry("b", task="t1", query_keys={"vision_0": torch.tensor([0.0, 1.0])}),
        _entry("c", task="t2", query_keys={"vision_0": torch.tensor([1.0, 1.0])}),
    ]

    stats = compute_field_stats(entries, "vision_0", "cosine", num_pairs=32)

    assert 0.0 <= stats["p5"] <= stats["p95"] <= 1.0
    assert 0.0 <= stats["mean"] <= 1.0
    assert stats["std"] >= 0.0


def test_compute_field_stats_for_l2_uses_tau_mapping():
    entries = [
        _entry("a", task="t1", query_keys={"robot_state": torch.tensor([0.0, 0.0])}),
        _entry("b", task="t1", query_keys={"robot_state": torch.tensor([0.1, 0.0])}),
        _entry("c", task="t2", query_keys={"robot_state": torch.tensor([2.0, 2.0])}),
    ]

    stats = compute_field_stats(entries, "robot_state", "l2", tau=0.334717, num_pairs=32)

    assert 0.0 <= stats["p5"] <= stats["p95"] <= 1.0
    assert 0.0 <= stats["mean"] <= 1.0


def test_compute_field_stats_with_too_few_entries_returns_defaults():
    entries = [_entry("a", task="t1", query_keys={"vision_0": torch.tensor([1.0, 0.0])})]

    stats = compute_field_stats(entries, "vision_0", "cosine")

    assert stats == {"p5": 0.0, "p95": 1.0, "mean": 0.5, "std": 0.0}


def test_sanity_check_returns_positive_separation_for_easy_case():
    entries = [
        _entry("a1", task="task_a", query_keys={"vision_0": torch.tensor([1.0, 0.0])}),
        _entry("a2", task="task_a", query_keys={"vision_0": torch.tensor([0.9, 0.1])}),
        _entry("b1", task="task_b", query_keys={"vision_0": torch.tensor([0.0, 1.0])}),
        _entry("b2", task="task_b", query_keys={"vision_0": torch.tensor([0.1, 0.9])}),
        _entry("a3", task="task_a", query_keys={"vision_0": torch.tensor([1.0, 0.0])}),
        _entry("b3", task="task_b", query_keys={"vision_0": torch.tensor([0.0, 1.0])}),
        _entry("a4", task="task_a", query_keys={"vision_0": torch.tensor([0.95, 0.05])}),
        _entry("b4", task="task_b", query_keys={"vision_0": torch.tensor([0.05, 0.95])}),
        _entry("a5", task="task_a", query_keys={"vision_0": torch.tensor([0.98, 0.02])}),
        _entry("b5", task="task_b", query_keys={"vision_0": torch.tensor([0.02, 0.98])}),
    ]

    stats = sanity_check(entries, "vision_0", "cosine", num_pairs=200)

    assert stats["same_task_mean"] > stats["cross_task_mean"]
    assert stats["separation"] > 0.0
