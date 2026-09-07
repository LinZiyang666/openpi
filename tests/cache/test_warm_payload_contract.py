"""Runtime warm-start guards: identity, exact lookup, and snapshot geometry."""

import pytest
import torch

from openpi.cache.components.judge import AlwaysWarmStartJudge
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID, groot_n15_schedule
from tests.cache.conftest import insert_entry, make_orchestrator
from tests.cache.test_interceptor import FakeModel, FakePolicy, _make_obs


def _payload():
    return CachePayload(
        action_chunk=torch.zeros(50, 32),
        intermediates={0.5: torch.ones(50, 32)},
        denoising_num_steps=4,
        schedule_id="groot_n15_k4_v1",
    )


@pytest.mark.parametrize(
    "bad", [torch.zeros(1, 32), torch.zeros(50, 32, dtype=torch.float64)]
)
def test_runtime_rejects_snapshot_broadcast_or_dtype_drift(bad):
    payload = _payload()
    payload.intermediates[0.5] = bad
    with pytest.raises(ValueError, match="shape|dtype"):
        payload.validate_for_warm_start(groot_n15_schedule(4), 0.5)


def test_runtime_requires_the_key_that_consumers_actually_index():
    payload = _payload()
    with pytest.raises(ValueError, match="no snapshot"):
        payload.validate_for_warm_start(groot_n15_schedule(4), 0.49999)


def test_pi05_consumer_rejects_internally_consistent_groot_payload():
    state = torch.randn(1, 32)
    judge = AlwaysWarmStartJudge(0.5, schedule=groot_n15_schedule(4))
    orchestrator, _, storage = make_orchestrator(judge=judge)
    insert_entry(storage, CheckpointID.CP1, state, _payload())
    interceptor = InferenceInterceptor(
        FakePolicy(FakeModel(fixed_state=state)),
        timer=SystemTimer(enabled=False),
        orchestrator=orchestrator,
    )
    interceptor.on_task_begin()
    with pytest.raises(ValueError, match="does not match"):
        interceptor.infer(_make_obs())
