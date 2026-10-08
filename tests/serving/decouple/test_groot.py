from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from openpi.cache.groot.staged import GrootStagedRunner
from openpi.serving.split_service import RemoteGrootRunner, StageService, make_groot_coordinator
from tests.cache.groot.test_groot_stage3 import _model_with_flow_head
from tests.serving.decouple.test_stages import Harness


@pytest.mark.parametrize("k", [1, 2, 8])
def test_groot_backbone_head_exact_with_k_and_interleaved_connections(k):
    model = _model_with_flow_head(k)
    direct = GrootStagedRunner(model, verify_upstream=False)
    coordinator = make_groot_coordinator(direct, max_batch_size=1, max_wait_ms=0)
    harness = Harness(StageService(model, runner=direct, coordinator=coordinator))
    remote = RemoteGrootRunner(harness.client, model)
    normalized = model.build_inputs()
    # Materialize the test head's LazyLinear before any concurrent forwards.
    with direct.session():
        direct.run_stage2(direct.run_stage1(normalized))
    def infer(i):
        with direct.session():
            s1 = direct.run_stage1(normalized)
            s2 = direct.run_stage2_llm(s1)
            # Use the stub's real action shape rather than production constants.
            noise = torch.full((1, model.action_head.config.action_horizon,
                                model.action_head.config.action_dim), i * 0.01)
            expected = direct.run_stage3(s2, noise=noise)
        with harness.client.decision():
            actual_s1 = remote.run_stage1(normalized)
            actual_s2 = remote.run_stage2_llm(actual_s1)
            actual = remote.run_stage3(actual_s2, noise=noise)
            assert torch.equal(actual_s1.input_embeds, s1.input_embeds)
            assert torch.equal(actual_s2.backbone_features, s2.backbone_features)
            assert actual.steps_run == k
            assert torch.equal(actual.action_pred, expected.action_pred)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(infer, range(6)))
        assert all(not handles for handles in harness.service.sessions.values())
    finally:
        harness.close()
        coordinator.stop()


def test_groot_seeded_miss_matches_upstream_stub():
    model = _model_with_flow_head(4)
    direct = GrootStagedRunner(model, verify_upstream=False)
    coordinator = make_groot_coordinator(direct)
    harness = Harness(StageService(model, runner=direct, coordinator=coordinator))
    remote = RemoteGrootRunner(harness.client, model)
    normalized = model.build_inputs()
    try:
        torch.manual_seed(35)
        with direct.session():
            expected = direct.run_stage2(direct.run_stage1(normalized))
        torch.manual_seed(35)
        with harness.client.decision():
            actual = remote.run_stage2(remote.run_stage1(normalized))
        assert torch.equal(expected.action_pred, actual.action_pred)
    finally:
        harness.close()
        coordinator.stop()


@pytest.mark.parametrize("start_t", [0.25, 0.5, 0.75])
def test_groot_warm_continuation_and_first_step_exact(start_t):
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    normalized = model.build_inputs()
    with runner.session():
        s2 = runner.run_stage2_llm(runner.run_stage1(normalized))
        runner.run_stage3(s2)
        start_x = torch.full((1, model.action_head.config.action_horizon,
                              model.action_head.config.action_dim), start_t)
        expected = runner.run_stage3_from(s2, start_x, start_t, schedule=runner.live_schedule(),
                                         capture_first_step=True)
    coordinator = make_groot_coordinator(runner)
    harness = Harness(StageService(model, runner=runner, coordinator=coordinator))
    remote = RemoteGrootRunner(harness.client, model)
    try:
        with harness.client.decision():
            s2 = remote.run_stage2_llm(remote.run_stage1(normalized))
            actual = remote.run_stage3_from(s2, start_x, start_t, schedule=remote.live_schedule(),
                                           capture_first_step=True)
        assert actual.steps_run == expected.steps_run
        for name in ("action_pred", "first_step_input", "first_step_x"):
            assert torch.equal(getattr(actual, name), getattr(expected, name))
    finally:
        harness.close()
        coordinator.stop()
