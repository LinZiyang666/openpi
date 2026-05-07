"""Unit tests for TopkActionVariance (B2 refactor, the 17th factor)."""

from __future__ import annotations

import math

import pytest
import torch

from openpi.cache.components.factors.base import (
    Factor,
    FactorContext,
    HistoryView,
)
from openpi.cache.components.factors.registry import get_class
from openpi.cache.components.factors.topk import TopkActionVariance
from openpi.cache.storage_types import CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID


# ----------------------------------------------------------------------
# Stub view returning canned payloads keyed by id
# ----------------------------------------------------------------------


class _StubView:
    def __init__(self, payloads: dict[str, CachePayload]) -> None:
        self._payloads = payloads

    def get_many(self, ids: list[str]) -> list[CachePayload]:
        return [self._payloads[i] for i in ids]


def _payload(action_first: list[float]) -> CachePayload:
    return CachePayload(action_chunk=torch.tensor([action_first], dtype=torch.float32))


def _ctx(results: list[SearchResultLite], payloads: dict[str, CachePayload]) -> FactorContext:
    return FactorContext(
        results=results,
        view=_StubView(payloads),
        history=HistoryView(actions=[], states=[]),
        normalization=None,
    )


def _result(eid: str, score: float = 1.0) -> SearchResultLite:
    return SearchResultLite(id=eid, score=score, checkpoint_id=CheckpointID.CP1)


# ----------------------------------------------------------------------
# Construction + protocol conformance
# ----------------------------------------------------------------------


def test_topk_satisfies_factor_protocol() -> None:
    inst = TopkActionVariance(K=5)
    assert isinstance(inst, Factor)


def test_topk_registered() -> None:
    cls = get_class("topk_action_variance")
    assert cls is TopkActionVariance


def test_topk_required_top_k_equals_K() -> None:
    inst = TopkActionVariance(K=5)
    assert inst.required_top_k == 5
    inst = TopkActionVariance(K=2)
    assert inst.required_top_k == 2


def test_topk_does_not_require_chain_walk() -> None:
    inst = TopkActionVariance(K=5)
    assert inst.requires_chain_walk is False


def test_topk_K_below_2_rejected() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        TopkActionVariance(K=1)
    with pytest.raises(ValueError, match=">= 2"):
        TopkActionVariance(K=0)


# ----------------------------------------------------------------------
# describe(params)
# ----------------------------------------------------------------------


def test_describe_emits_single_risky_key_regardless_of_K() -> None:
    assert TopkActionVariance.describe({"K": 5}) == {"topk_action_variance": "risky"}
    assert TopkActionVariance.describe({"K": 2}) == {"topk_action_variance": "risky"}


# ----------------------------------------------------------------------
# extract — happy path
# ----------------------------------------------------------------------


def test_extract_emits_finite_variance_for_disagreeing_pool() -> None:
    payloads = {
        "a": _payload([1.0, 0.0]),
        "b": _payload([2.0, 0.0]),
        "c": _payload([3.0, 0.0]),
    }
    results = [_result("a"), _result("b"), _result("c")]
    f = TopkActionVariance(K=3)
    out = f.extract(_ctx(results, payloads))
    assert "topk_action_variance" in out
    val = out["topk_action_variance"]
    # Per-DOF variance: dim 0 has var > 0; dim 1 is constant 0, masked out.
    assert math.isfinite(val) and val > 0


def test_extract_emits_nan_when_pool_unanimous() -> None:
    payloads = {
        "a": _payload([1.0, 0.0]),
        "b": _payload([1.0, 0.0]),
        "c": _payload([1.0, 0.0]),
    }
    results = [_result(i) for i in ("a", "b", "c")]
    f = TopkActionVariance(K=3)
    out = f.extract(_ctx(results, payloads))
    # Both dims have ~zero variance → no active dim → NaN.
    assert math.isnan(out["topk_action_variance"])


def test_extract_emits_nan_when_K_eff_below_2() -> None:
    payloads = {"a": _payload([1.0, 2.0])}
    f = TopkActionVariance(K=5)
    out = f.extract(_ctx([_result("a")], payloads))
    assert math.isnan(out["topk_action_variance"])


def test_extract_with_empty_results_emits_nan() -> None:
    f = TopkActionVariance(K=5)
    out = f.extract(_ctx([], {}))
    assert math.isnan(out["topk_action_variance"])


def test_extract_uses_only_top_K_results() -> None:
    """Pool has 4 entries but K=2 → variance computed over first 2 only."""
    payloads = {
        "a": _payload([1.0, 0.0]),
        "b": _payload([1.0, 0.0]),     # agree with a → variance=0
        "c": _payload([99.0, 0.0]),    # would disagree if used
        "d": _payload([99.0, 0.0]),
    }
    results = [_result(i) for i in ("a", "b", "c", "d")]
    f = TopkActionVariance(K=2)
    out = f.extract(_ctx(results, payloads))
    # Top-2 (a, b) are unanimous → NaN per active-mask rule.
    assert math.isnan(out["topk_action_variance"])
