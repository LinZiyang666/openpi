"""Tests for the ``__hit_meta__`` field that ``InferenceInterceptor.infer``
attaches to its return value (verdict_factor_judge B1 observability).

The field is consumed by the LIBERO client's per-step JSONL writer to
persist per-verdict ``(hit_type, start_t, winner_id, cp1_score)`` so any
yaml's behaviour can be re-analysed offline. Each test below covers one of
the four documented return paths:

  1. CP1 FULL_HIT early return
  2. MISS late return (orchestrator engaged, no cache hit)
  3. WARM_START late return (orchestrator engaged, judge picks a tier)
  4. Orchestrator disabled — late return emits a MISS placeholder so the
     wire schema is uniform with cold-start MISSes.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.write_policy import AlwaysWritePolicy
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID

from tests.cache.conftest import insert_entry, make_orchestrator
from tests.cache.test_interceptor import FakeModel, FakePolicy, _make_obs


# ---------------------------------------------------------------------------
# Path 1: CP1 FULL_HIT early return
# ---------------------------------------------------------------------------


def test_full_hit_attaches_hit_meta() -> None:
    """A CP1 FULL_HIT must carry hit_type=FULL_HIT plus a populated
    winner_id / cp1_score (from the matched entry) and start_t=None
    (FULL_HIT skips warm-start tiers)."""
    fixed_state = torch.randn(1, 32)
    fixed_action = torch.randn(1, 50, 32)
    model = FakeModel(fixed_state=fixed_state, fixed_action=fixed_action)
    policy = FakePolicy(model=model)

    orch, _, _ = make_orchestrator(write_policy=AlwaysWritePolicy())
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )

    # First call: MISS (still attaches __hit_meta__, see other test); the
    # important effect is that AlwaysWritePolicy queues this entry so the
    # next call with the same state is a guaranteed FULL_HIT.
    obs = _make_obs()
    interceptor.on_task_begin()
    interceptor.infer(obs)
    interceptor.on_episode_end(success=True)
    interceptor.on_episode_start("test", "test", 1)

    result = interceptor.infer(obs)
    meta = result["__hit_meta__"]
    assert meta["hit_type"] == "FULL_HIT"
    assert meta["start_t"] is None
    assert isinstance(meta["winner_id"], str) and meta["winner_id"]
    # ThresholdJudge fires FULL_HIT only on score >= 0.98; the writer-stored
    # entry uses the same query_keys so similarity ≈ 1.0.
    assert meta["cp1_score"] is not None and meta["cp1_score"] >= 0.98


# ---------------------------------------------------------------------------
# Path 2: MISS late return
# ---------------------------------------------------------------------------


def test_miss_attaches_hit_meta() -> None:
    """Empty cache + orchestrator engaged → MISS path. ``__hit_meta__`` must
    report hit_type=MISS with start_t=None and a None winner_id (no entry
    matched). cp1_score may be None because the search returns no hits."""
    policy = FakePolicy()
    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )
    interceptor.on_task_begin()

    result = interceptor.infer(_make_obs())
    meta = result["__hit_meta__"]
    assert meta["hit_type"] == "MISS"
    assert meta["start_t"] is None
    assert meta["winner_id"] is None
    # ThresholdJudge returns score=None on MISS (no candidate met threshold).
    assert meta["cp1_score"] is None


# ---------------------------------------------------------------------------
# Path 3: WARM_START late return
# ---------------------------------------------------------------------------


def test_warm_start_attaches_hit_meta_with_start_t() -> None:
    """Stored entry similar enough to trigger warm_tiers but below FULL_HIT
    threshold → WARM_START path. ``__hit_meta__`` must surface the judge's
    chosen ``start_t`` so offline analysis can compare warm-start cadence."""
    stored_state = torch.zeros(1, 32)
    stored_state[0, 0] = 1.0

    target_cos = 0.96
    sin_val = math.sqrt(1.0 - target_cos ** 2)
    query_state = torch.zeros(1, 32)
    query_state[0, 0] = target_cos
    query_state[0, 1] = sin_val
    query_state = F.normalize(query_state, dim=1)

    fixed_action = torch.randn(1, 50, 32)
    model = FakeModel(fixed_state=query_state, fixed_action=fixed_action)

    # FakeModel doesn't ship ``run_stage3_from`` by default; provide a stub
    # so the WARM_START branch can complete. The interceptor calls it with
    # positional + ``num_steps=...`` so the parameter names are fixed; the
    # ``del`` statement marks them as intentionally unused.
    def _fake_run_stage3_from(stage2, start_x, start_t, *, num_steps=10):
        del stage2, start_x, start_t, num_steps
        return SimpleNamespace(action_chunk=fixed_action.clone())

    model.run_stage3_from = _fake_run_stage3_from

    policy = FakePolicy(model=model)

    warm_tiers = [{"threshold": 0.95, "start_t": 0.3}]
    judge = ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95, warm_tiers=warm_tiers)
    orch, _, storage = make_orchestrator(judge=judge)
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )

    payload = CachePayload(
        action_chunk=torch.randn(50, 32),
        intermediates={0.3: torch.randn(50, 32), 0.5: torch.randn(50, 32)},
        denoising_num_steps=10,
    )
    insert_entry(storage, CheckpointID.CP1, stored_state, payload)

    interceptor.on_task_begin()
    result = interceptor.infer(_make_obs())
    meta = result["__hit_meta__"]
    assert meta["hit_type"] == "WARM_START"
    assert meta["start_t"] == 0.3
    assert isinstance(meta["winner_id"], str) and meta["winner_id"]
    assert meta["cp1_score"] is not None
    # ThresholdJudge picks the tier when score >= warm_tiers[0].threshold and
    # below cp1_threshold; verify we landed in that band.
    assert 0.95 <= meta["cp1_score"] < 0.98


# ---------------------------------------------------------------------------
# Path 4: Orchestrator disabled (CP1 disabled fallback)
# ---------------------------------------------------------------------------


def test_no_orchestrator_attaches_miss_placeholder() -> None:
    """Without an orchestrator the interceptor never produces a CheckResult;
    the helper must emit a MISS placeholder so client analysis tools see one
    uniform schema regardless of whether caching is on."""
    interceptor = InferenceInterceptor(
        FakePolicy(), timer=SystemTimer(enabled=False)
    )
    result = interceptor.infer(_make_obs())
    meta = result["__hit_meta__"]
    assert meta == {
        "hit_type": "MISS",
        "start_t": None,
        "winner_id": None,
        "cp1_score": None,
    }
