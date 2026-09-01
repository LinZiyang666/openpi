"""H-CRD through the real orchestrator (G2 R1 B2 / B4 / B6 integration).

* a WARM_START the judge proposes is downgraded by the orchestrator when the
  payload lacks intermediates; the judge must book the executed MISS (D = 0,
  RECOVERY) and the step's ``factor_outputs["crd"]`` must record both verdicts;
* assembling a CRD judge behind any gate other than always_search is refused.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from openpi.cache.components import crd_judge as crd
from openpi.cache.components.gate import AlwaysSearchGate, AlwaysSkipGate
from openpi.cache.components.judge import HitType
from openpi.cache.storage_types import CachePayload
from openpi.cache.types import CheckpointID
from tests.cache.conftest import insert_entry, make_orchestrator, make_stage1
from tests.cache.test_crd_judge import write_crd


def _unit(dim=32, index=0):
    """Standard basis vector as [1, dim] (the batched shape the key builder expects)."""
    v = torch.zeros(1, dim)
    v[0, index] = 1.0
    return v


def _with_cosine(base, target):
    other = torch.zeros_like(base)
    other[0, 1] = 1.0
    q = target * base + (1.0 - target ** 2) ** 0.5 * other
    return F.normalize(q, dim=1)


def _judge(tmp_path, **kw):
    # beta between d_warm (0.4) and d_full (0.5) at the top bins -> FULL infeasible, WARM feasible
    j = crd.CumulativeRiskJudge(write_crd(tmp_path, gamma=1.0, beta=0.45, j_bad=None, l_max=None, **kw))
    return j


def test_warm_downgrade_is_committed_as_miss_and_logged(tmp_path):
    judge = _judge(tmp_path)
    orch, _, storage = make_orchestrator(judge=judge)
    orch.on_episode_start(extra_metadata={"task_id": 0})
    state = _unit(32, 0)
    insert_entry(storage, CheckpointID.CP1, state, CachePayload(action_chunk=torch.randn(50, 32)))  # no intermediates
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(_with_cosine(state, 0.96)))
    assert result.hit_type == HitType.MISS
    diag = (result.factor_outputs or {}).get("crd")
    assert diag is not None
    assert diag["proposed"] == "WARM_START" and diag["executed"] == "MISS" and diag["reason"] == "downgrade"
    assert judge.state["D"] == 0.0 and judge.state["mode"] == crd.RECOVERY and judge.state["pending"] is False
    orch.clear()


def test_full_hit_commits_and_logs_debt(tmp_path):
    judge = crd.CumulativeRiskJudge(write_crd(tmp_path, gamma=1.0, beta=5.0, j_bad=None, l_max=None))
    orch, _, storage = make_orchestrator(judge=judge)
    orch.on_episode_start(extra_metadata={"task_id": 0})
    state = _unit(32, 0)
    insert_entry(storage, CheckpointID.CP1, state, CachePayload(action_chunk=torch.randn(50, 32)))
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(_with_cosine(state, 0.96)))
    assert result.hit_type == HitType.FULL_HIT
    diag = result.factor_outputs["crd"]
    assert diag["executed"] == "FULL_HIT" and diag["D_after"] == pytest.approx(0.5) and diag["fh_run"] == 1
    assert judge.state["D"] == pytest.approx(0.5) and judge.state["pending"] is False
    orch.clear()


def test_episode_start_without_task_identity_fails_loud(tmp_path):
    judge = _judge(tmp_path)
    orch, _, _ = make_orchestrator(judge=judge)
    with pytest.raises(ValueError, match="task scale"):
        orch.on_episode_start(extra_metadata={})


def test_real_connection_lifecycle_waits_for_identified_episode_start(tmp_path):
    """WebsocketPolicyServer calls task_begin before the client's episode_start."""
    judge = _judge(tmp_path)
    orch, _, _ = make_orchestrator(judge=judge)
    orch.on_task_begin()  # connection open: reset only, no fabricated task id
    assert judge.state["task_scale"] is None
    with pytest.raises(RuntimeError, match="before on_episode_start"):
        judge([type("R", (), {"score": 0.9, "id": "e0"})()], CheckpointID.CP1, {})
    orch.on_episode_start(extra_metadata={"task_id": 0})
    assert judge.state["task_scale"] == 1.0


@pytest.mark.parametrize("gate", [AlwaysSkipGate()])
def test_assembly_refuses_non_always_search_gate(tmp_path, gate):
    with pytest.raises(ValueError, match="always_search"):
        make_orchestrator(gate=gate, judge=_judge(tmp_path))


def test_assembly_accepts_always_search_gate(tmp_path):
    make_orchestrator(gate=AlwaysSearchGate(), judge=_judge(tmp_path))
