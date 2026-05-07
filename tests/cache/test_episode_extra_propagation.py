"""DumpingJudge.on_episode_start propagates extra_metadata to inner judge."""

from __future__ import annotations

from pathlib import Path

import torch

from openpi.cache.components.dumping_judge import DumpingJudge
from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.components.factors.normalization import ZScoreNormalization
from openpi.cache.components.judge import AlwaysHitJudge


def _make_judge(tmp_path: Path) -> DumpingJudge:
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    library_stats = LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )
    norm = ZScoreNormalization(library_stats)
    return DumpingJudge(
        inner=AlwaysHitJudge(),
        dump_normalization=norm,
        dump_factors=[],
        dump_path=str(tmp_path / "dump.jsonl"),
        config_id="cfg-id",
    )


def test_dumping_judge_stashes_extra_metadata_for_jsonl_rows(tmp_path: Path) -> None:
    judge = _make_judge(tmp_path)
    judge.on_episode_start(extra_metadata={"task_id": "t99", "orig_init_state_idx": 4})
    assert judge._current_extra == {"task_id": "t99", "orig_init_state_idx": 4}
    assert judge._step_idx == 0


def test_dumping_judge_resets_step_idx_each_episode(tmp_path: Path) -> None:
    judge = _make_judge(tmp_path)
    judge.on_episode_start(extra_metadata={"task_id": "t1"})
    # Simulate verdicts incrementing step_idx (object.__setattr__ guard means
    # step_idx is mutable through the wrapper).
    object.__setattr__(judge, "_step_idx", 17)
    judge.on_episode_start(extra_metadata={"task_id": "t2"})
    assert judge._step_idx == 0


def test_dumping_judge_inner_lifecycle_does_not_break_when_inner_lacks_kwarg(
    tmp_path: Path,
) -> None:
    """AlwaysHitJudge.on_episode_start has no extra_metadata parameter; the
    filtered-dispatch contract must drop the kwarg silently."""
    judge = _make_judge(tmp_path)
    # Should not raise.
    judge.on_episode_start(extra_metadata={"task_id": "t-skip"})
