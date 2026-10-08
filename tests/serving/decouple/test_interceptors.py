"""Original cache interceptors with remote model adapters, CPU stage stubs."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.decouple.recorded import load
from openpi.cache.components.judge import HitType
from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.interceptor import InferenceInterceptor
from openpi.serving.batching_coordinator import BatchingCoordinator
from openpi.serving.split_runtime import DecisionPolicy, export_blueprint, validate_arguments
from openpi.serving.split_service import RemoteCoordinator, RemoteGrootRunner, RemotePiModel, StageService, make_groot_coordinator
from tests.cache.conftest import make_orchestrator
from tests.cache.groot.test_groot_interceptor import _StubOrchestrator, _StubPolicy, _obs
from tests.cache.groot.test_groot_stage3 import _model_with_flow_head
from tests.cache.test_interceptor import FakePolicy, _make_obs
from tests.serving.decouple.stubs import PiModel
from tests.serving.decouple.test_stages import Harness


@pytest.mark.parametrize("k", [1, 2, 10])
def test_pi_real_interceptor_recorded_state_and_noise_exact(k):
    model = PiModel()
    core = BatchingCoordinator(model, max_batch_size=1, max_wait_ms=0)
    core.start()
    harness = Harness(StageService(model, coordinator=core))
    direct_orch, _, _ = make_orchestrator()
    front_orch, _, _ = make_orchestrator()
    direct = InferenceInterceptor(FakePolicy(model), orchestrator=direct_orch,
                                  coordinator=core, eager=True, miss_num_steps=k)
    front = DecisionPolicy(InferenceInterceptor(FakePolicy(RemotePiModel(harness.client, model.config)),
                           orchestrator=front_orch, coordinator=RemoteCoordinator(harness.client),
                           eager=True, miss_num_steps=k), harness.client)
    data = load("pi05", limit=1)
    obs = _make_obs()
    obs["state"][:] = 0
    state = data.arrays["rs"][int(data.rows[0])]
    obs["state"][:len(state)] = state
    noise = np.linspace(-1, 1, 320, dtype=np.float32).reshape(10, 32)
    try:
        expected = direct.infer(obs, noise=noise)
        actual = front.infer(obs, noise=noise)
        for name in ("actions", "state"):
            assert np.array_equal(actual[name], expected[name])
        assert actual["__hit_meta__"] == expected["__hit_meta__"]
    finally:
        harness.close()
        core.stop()


@pytest.mark.parametrize("k", [1, 2, 8])
def test_groot_real_interceptor_seeded_miss_exact(k):
    model = _model_with_flow_head(k)
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        runner.run_stage2(runner.run_stage1(model.build_inputs()))
    core = make_groot_coordinator(runner)
    harness = Harness(StageService(model, runner=runner, coordinator=core))
    policy = _StubPolicy(model)
    direct = GrootCacheInterceptor(policy, runner, orchestrator=_StubOrchestrator(HitType.MISS), miss_num_steps=k)
    front = GrootCacheInterceptor(policy, RemoteGrootRunner(harness.client, model),
                                  orchestrator=_StubOrchestrator(HitType.MISS), miss_num_steps=k)
    try:
        torch.manual_seed(917)
        expected = direct.get_action(_obs())
        torch.manual_seed(917)
        with harness.client.decision():
            actual = front.get_action(_obs())
        assert np.array_equal(actual["action.out"], expected["action.out"])
        assert actual["__hit_meta__"] == expected["__hit_meta__"]
    finally:
        harness.close()
        core.stop()


def test_cpu_blueprint_refuses_model_parameters():
    policy = FakePolicy(PiModel())
    policy._sample_kwargs, policy._metadata = {}, {}
    assert export_blueprint(policy, "pi05", {})
    policy._metadata = {"accidental_weight": policy._model.anchor}
    with pytest.raises(ValueError, match="parameters"):
        export_blueprint(policy, "pi05", {})


def test_split_argument_gates_cover_native_cli_spellings():
    validate_arguments("pi05", ["--cache_config", "dummy.yaml", "--os-stage1-mode", "full"])
    for flag in ("--stage1_device", "--os-stage1-mode=dummy_cached", "--os-pack-prefix"):
        with pytest.raises(ValueError, match="split:"):
            validate_arguments("pi05", ["--cache-config", "dummy.yaml", flag, "cuda"])
