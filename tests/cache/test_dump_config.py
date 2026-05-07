"""DumpConfig + DumpingJudge wrapping tests (refactor)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from openpi.cache.components.dumping_judge import DumpingJudge
from openpi.cache.components.factors.base import (
    CalibrationSamples,
    FactorContext,
    HistoryView,
    LibraryStats,
)
from openpi.cache.components.factors.calibrations import PercentileRollingCalibration
from openpi.cache.components.factors.composers import WeightedSumComposer
from openpi.cache.components.factors.normalization import ZScoreNormalization
from openpi.cache.components.factors.topk import TopkActionVariance
from openpi.cache.components.judge import AlwaysHitJudge, HitType
from openpi.cache.config import DumpConfig, FactorConfig
from openpi.cache.storage_types import CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_lib_stats() -> LibraryStats:
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    return LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )


class _StubView:
    def __init__(self, payloads: dict[str, CachePayload]) -> None:
        self._payloads = payloads

    def get(self, eid: str) -> CachePayload:
        return self._payloads[eid]

    def get_many(self, ids: list[str]) -> list[CachePayload]:
        return [self._payloads[i] for i in ids]


# ----------------------------------------------------------------------
# DumpConfig dataclass
# ----------------------------------------------------------------------


def test_dump_config_defaults() -> None:
    cfg = DumpConfig()
    assert cfg.path == ""
    assert cfg.config_id == ""
    assert cfg.factors == []
    assert cfg.deferred is False


def test_dump_config_factors_list_round_trip() -> None:
    cfg = DumpConfig(
        path="/tmp/x.jsonl",
        config_id="some_yaml",
        factors=[FactorConfig(type="topk_action_variance", params={"K": 5})],
    )
    assert cfg.factors[0].type == "topk_action_variance"
    assert cfg.factors[0].params == {"K": 5}


# ----------------------------------------------------------------------
# DumpingJudge: wraps inner, dumps independent factors
# ----------------------------------------------------------------------


def test_dumping_judge_writes_jsonl_and_returns_inner_result(tmp_path: Path) -> None:
    inner = AlwaysHitJudge()
    norm = ZScoreNormalization(_make_lib_stats())
    dump_path = tmp_path / "dump.jsonl"
    judge = DumpingJudge(
        inner=inner,
        dump_normalization=norm,
        dump_factors=[TopkActionVariance(K=2)],
        dump_path=str(dump_path),
        config_id="test_yaml",
    )

    payloads = {
        "a": CachePayload(action_chunk=torch.tensor([[1.0, 0.0]], dtype=torch.float32)),
        "b": CachePayload(action_chunk=torch.tensor([[2.0, 0.0]], dtype=torch.float32)),
    }
    results = [
        SearchResultLite(id="a", score=1.0, checkpoint_id=CheckpointID.CP1),
        SearchResultLite(id="b", score=0.9, checkpoint_id=CheckpointID.CP1),
    ]
    view = _StubView(payloads)
    history = HistoryView(actions=[], states=[])

    judge.on_episode_start(extra_metadata={"task_id": "t0", "orig_init_state_idx": 7})
    out = judge(results, CheckpointID.CP1, {}, view=view, history=history)

    # Inner result preserved byte-for-byte.
    assert out.hit_type == HitType.FULL_HIT
    assert out.winner_id == "a"

    # JSONL row appended with config_id, identity metadata, dump factor raw.
    rows = [json.loads(line) for line in dump_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["config_id"] == "test_yaml"
    assert row["task_id"] == "t0"
    assert row["orig_init_state_idx"] == 7
    assert row["winner_id"] == "a"
    assert "topk_action_variance" in row["factor_raw"]


def test_dumping_judge_min_required_top_k_max_of_inner_and_dump() -> None:
    """min_required_top_k = max(inner_hint, max(dump_factor.required_top_k))."""

    class _InnerWithHint:
        min_required_top_k = 3
        def __call__(self, *a, **kw):
            from openpi.cache.components.judge import HitType, JudgeResult
            return JudgeResult(HitType.MISS)
        def on_episode_start(self, **kw): pass
        def record_action(self, *a, **kw): pass

    inner = _InnerWithHint()
    norm = ZScoreNormalization(_make_lib_stats())
    judge = DumpingJudge(
        inner=inner,
        dump_normalization=norm,
        dump_factors=[TopkActionVariance(K=7)],
        dump_path="/tmp/_unused.jsonl",
        config_id="x",
    )
    assert judge.min_required_top_k == 7   # max(3, 7)


def test_dumping_judge_extra_metadata_forwarded_to_inner() -> None:
    """on_episode_start(extra_metadata=...) reaches the inner judge via
    the orchestrator's filtered dispatch contract."""
    received: list[dict | None] = []

    class _InnerCapturingExtra:
        min_required_top_k = 0
        def __call__(self, *a, **kw):
            from openpi.cache.components.judge import HitType, JudgeResult
            return JudgeResult(HitType.MISS)
        def on_episode_start(self, *, extra_metadata=None):
            received.append(extra_metadata)
        def record_action(self, *a, **kw): pass

    inner = _InnerCapturingExtra()
    norm = ZScoreNormalization(_make_lib_stats())
    judge = DumpingJudge(
        inner=inner,
        dump_normalization=norm,
        dump_factors=[],
        dump_path="/tmp/_unused2.jsonl",
        config_id="x",
    )
    payload = {"task_id": "x", "orig_init_state_idx": 1}
    judge.on_episode_start(extra_metadata=payload)
    assert received == [payload]


def test_dumping_judge_per_factor_failure_isolated(tmp_path: Path) -> None:
    """A misbehaving dump factor produces NaN row entries instead of crashing."""

    class _BoomFactor:
        requires_chain_walk = False
        required_top_k = 0
        descriptor_orientations = {"boom_factor": "risky"}

        @classmethod
        def describe(cls, params): return {"boom_factor": "risky"}

        def extract(self, ctx):
            raise RuntimeError("boom")

    inner = AlwaysHitJudge()
    norm = ZScoreNormalization(_make_lib_stats())
    dump_path = tmp_path / "boom.jsonl"
    judge = DumpingJudge(
        inner=inner,
        dump_normalization=norm,
        dump_factors=[_BoomFactor()],
        dump_path=str(dump_path),
        config_id="test_boom",
    )
    results = [SearchResultLite(id="a", score=1.0, checkpoint_id=CheckpointID.CP1)]
    payloads = {"a": CachePayload(action_chunk=torch.tensor([[0.0]], dtype=torch.float32))}
    judge.on_episode_start(extra_metadata={})
    out = judge(results, CheckpointID.CP1, {},
                view=_StubView(payloads), history=HistoryView(actions=[], states=[]))
    assert out.hit_type == HitType.FULL_HIT
    rows = [json.loads(line) for line in dump_path.read_text().splitlines() if line.strip()]
    assert rows[0]["factor_nan"]["boom_factor"] is True
