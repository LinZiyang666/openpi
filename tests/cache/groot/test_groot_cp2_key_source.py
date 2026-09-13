"""``GrootStagedRunner.run_cp2_key_source``: the encoded conditioning as a key source.

Off-island, with the stub head's counting encoders; the real-head comparison
is the island parity gate (exp/libero_groot/groot_cp2_parity.py).
"""

from __future__ import annotations

import pytest
import torch

from openpi.cache.groot.cp2_key_builder import GrootCP2TernaryKeyBuilder
from openpi.cache.groot.staged import GrootCP2KeySource, GrootStagedRunner
from openpi.cache.types import VLM_OUT, CheckpointID

from .conftest import EMB_DIM, STATE_FEAT_DIM, StubGrootModel


def _stage2(runner, model):
    stage1 = runner.run_stage1(model.build_inputs())
    return runner.run_stage2_llm(stage1)


def test_requires_the_session():
    model = StubGrootModel()
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = _stage2(runner, model)
    with pytest.raises(RuntimeError, match="must run inside GrootStagedRunner.session"):
        runner.run_cp2_key_source(stage2)


def test_returns_the_heads_own_encodings_once_each():
    model = StubGrootModel()
    head = model.action_head
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = _stage2(runner, model)
        raw = stage2.backbone_features.clone()
        src = runner.run_cp2_key_source(stage2)
        # Reference: the head's encoders applied by hand to the same stage-2 output.
        expect_vl = (raw * 3.0 + 1.0)[0]
        expect_state = head.state_encoder(stage2.action_inputs["state"], stage2.action_inputs["embodiment_id"])[0, -1]
    assert isinstance(src, GrootCP2KeySource)
    assert src.vl_encoded.shape == (raw.shape[1], EMB_DIM)
    assert src.state_encoded.shape == (STATE_FEAT_DIM,)
    assert torch.equal(src.vl_encoded.float(), expect_vl.float())
    assert torch.equal(src.state_encoded.float(), expect_state.float())
    assert head.process_calls == 1 and head.state_calls == 2  # 1 by the helper + 1 by the reference above
    assert head.calls == 0  # the denoise loop never ran


def test_stage2_output_is_not_written_back_and_rng_is_untouched():
    """``process_backbone_output`` normalises in place; the helper must hand it a
    fresh BatchFeature so ``stage2`` still feeds a later WARM / MISS unchanged."""
    model = StubGrootModel()
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = _stage2(runner, model)
        feats_before = stage2.backbone_features.clone()
        state_before = stage2.action_inputs["state"].clone()
        rng_before = torch.get_rng_state()
        runner.run_cp2_key_source(stage2)
        runner.run_cp2_key_source(stage2)
    assert torch.equal(feats_before, stage2.backbone_features)
    assert torch.equal(state_before, stage2.action_inputs["state"])
    assert torch.equal(rng_before, torch.get_rng_state())
    assert model.action_head.process_calls == 2  # once per call, no caching either


def test_source_may_be_inference_tensors_but_the_stored_key_is_not():
    model = StubGrootModel()
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        src = runner.run_cp2_key_source(_stage2(runner, model))
    assert src.vl_encoded.is_inference() and src.state_encoded.is_inference()
    builder = GrootCP2TernaryKeyBuilder(seed=1, d=8, p=0.5, token_len=src.vl_encoded.shape[0] + 1,
                                        feature_dim=EMB_DIM, state_feat_dim=STATE_FEAT_DIM)
    builder.collect(CheckpointID.CP2, cp2_source=src)
    key = builder.build(CheckpointID.CP2)[VLM_OUT]
    assert not key.is_inference() and key.dtype is torch.float32 and key.device.type == "cpu"


def test_training_mode_head_is_refused():
    model = StubGrootModel()
    model.action_head.training = True
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = _stage2(runner, model)
        with pytest.raises(RuntimeError, match="training mode"):
            runner.run_cp2_key_source(stage2)
    assert model.action_head.process_calls == 0


def test_missing_action_inputs_and_non_finite_encodings_are_refused():
    model = StubGrootModel()
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = _stage2(runner, model)
        stage2.action_inputs = None
        with pytest.raises(RuntimeError, match="action_inputs is None"):
            runner.run_cp2_key_source(stage2)
        stage2 = _stage2(runner, model)
        model.action_head.poison_state = True
        with pytest.raises(RuntimeError, match="non-finite"):
            runner.run_cp2_key_source(stage2)


def test_the_probe_is_registered_and_records_one_measurement_per_call():
    from openpi.cache.timing import SystemTimer
    from openpi.serving import monitor

    previous = monitor.get_monitor_level()
    monitor.set_monitor_level(monitor.MonitorLevel.BASIC)
    try:
        model = StubGrootModel()
        timer = SystemTimer(enabled=True)
        runner = GrootStagedRunner(model, timer=timer, verify_upstream=False)
        with runner.session():
            stage2 = _stage2(runner, model)
            runner.run_cp2_key_source(stage2)
            runner.run_cp2_key_source(stage2)
        counts = {name: stats.count for name, stats in timer.summary(task_only=False).items()}
        assert counts.get("cp2_encode") == 2
        assert counts.get("stage2_action", 0) == 0
    finally:
        monitor.set_monitor_level(previous)
