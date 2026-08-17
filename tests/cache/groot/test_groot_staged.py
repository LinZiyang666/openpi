"""The two-stage split: preconditions, ordering, and the guards that keep it honest."""

from __future__ import annotations

import pytest
import torch

from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.timing import SystemTimer

from .conftest import (
    EMB_DIM,
    IMAGE_TOKEN_ID,
    N_CAMERAS,
    TOKENS_PER_IMAGE,
    StubGrootModel,
)


def test_stage1_scatters_vision_into_the_language_sequence(runner, stub_model):
    inputs = stub_model.build_inputs()
    with runner.session():
        stage1 = runner.run_stage1(inputs)

    ids = inputs["eagle_input_ids"]
    assert stage1.input_embeds.shape == (1, ids.shape[1], EMB_DIM)
    assert torch.equal(stage1.image_token_mask, ids == IMAGE_TOKEN_ID)
    assert int(stage1.image_token_mask.sum()) == N_CAMERAS * TOKENS_PER_IMAGE

    # The scattered positions hold the vision output, not the token embedding.
    vit = stub_model.backbone.eagle_model.extract_feature(inputs["eagle_pixel_values"])
    scattered = stage1.input_embeds[0][stage1.image_token_mask[0]]
    assert torch.allclose(scattered, vit.reshape(-1, EMB_DIM).to(scattered.dtype))


def test_stage2_reads_the_final_layer_and_rebuilds_its_own_batchfeature(runner, stub_model):
    inputs = stub_model.build_inputs()
    with runner.session():
        stage1 = runner.run_stage1(inputs)
        first = runner.run_stage2(stage1)
        second = runner.run_stage2(stage1)

    # Re-running against the same stage1 must be reproducible. It would not be
    # if the backbone-output mapping were cached and the action head's in-place
    # normalisation applied twice.
    assert torch.equal(first.action_pred, second.action_pred)
    assert stub_model.action_head.calls == 2
    assert stub_model.validate_calls == 2


def test_stage1_is_not_rerun_by_stage2(runner, stub_model):
    inputs = stub_model.build_inputs()
    with runner.session():
        stage1 = runner.run_stage1(inputs)
        assert stub_model.backbone.eagle_model.extract_calls == 1
        runner.run_stage2(stage1)
    assert stub_model.backbone.eagle_model.extract_calls == 1


def test_both_stages_refuse_to_run_outside_a_session(runner, stub_model):
    inputs = stub_model.build_inputs()
    with pytest.raises(RuntimeError, match="must run inside"):
        runner.run_stage1(inputs)

    with runner.session():
        stage1 = runner.run_stage1(inputs)
    with pytest.raises(RuntimeError, match="must run inside"):
        runner.run_stage2(stage1)


def test_unexpected_eagle_input_is_refused(runner, stub_model):
    inputs = stub_model.build_inputs()
    inputs["eagle_pixel_values_videos"] = torch.zeros(1)
    with runner.session(), pytest.raises(RuntimeError, match="Unexpected eagle inputs"):
        runner.run_stage1(inputs)


def test_batch_larger_than_one_is_refused(runner, stub_model):
    inputs = stub_model.build_inputs()
    inputs["eagle_input_ids"] = inputs["eagle_input_ids"].repeat(2, 1)
    with runner.session(), pytest.raises(RuntimeError, match="only B=1"):
        runner.run_stage1(inputs)


def test_image_token_count_mismatch_is_refused_not_truncated(runner, stub_model):
    inputs = stub_model.build_inputs()
    ids = inputs["eagle_input_ids"]
    ids[0, 0] = IMAGE_TOKEN_ID  # one image token too many
    with runner.session(), pytest.raises(RuntimeError, match="image tokens"):
        runner.run_stage1(inputs)


def test_training_mode_model_is_refused():
    model = StubGrootModel()
    model.training = True
    with pytest.raises(RuntimeError, match="eval-mode"):
        GrootStagedRunner(model, verify_upstream=False)


def test_select_layer_must_match_the_truncated_stack():
    model = StubGrootModel(n_layers=3)
    model.backbone.select_layer = 2
    with pytest.raises(RuntimeError, match="select_layer"):
        GrootStagedRunner(model, verify_upstream=False)


def test_upstream_hash_guard_fires_on_a_foreign_forward():
    model = StubGrootModel()
    with pytest.raises(RuntimeError, match="Upstream forward has changed"):
        GrootStagedRunner(model, verify_upstream=True)


def test_probe_counts_distinguish_a_skipped_stage2(stub_model):
    from openpi.serving import monitor

    previous = monitor.get_monitor_level()
    monitor.set_monitor_level(monitor.MonitorLevel.BASIC)
    try:
        timer = SystemTimer(enabled=True)
        runner = GrootStagedRunner(stub_model, timer=timer, verify_upstream=False)
        inputs = stub_model.build_inputs()

        with runner.session():
            stage1 = runner.run_stage1(inputs)
        assert _samples(timer, "stage1_vision") == 1
        assert _samples(timer, "stage2_llm") == 0
        assert _samples(timer, "stage2_action") == 0

        with runner.session():
            runner.run_stage2(stage1)
        assert _samples(timer, "stage2_llm") == 1
        assert _samples(timer, "stage2_action") == 1
    finally:
        monitor.set_monitor_level(previous)


def test_timer_defaults_to_disabled_so_the_handshake_records_nothing(stub_model):
    runner = GrootStagedRunner(stub_model, verify_upstream=False)
    inputs = stub_model.build_inputs()
    with runner.session():
        runner.run_stage1(inputs)
    assert _samples(runner._timer, "stage1_vision") == 0  # noqa: SLF001


def _samples(timer: SystemTimer, probe: str) -> int:
    """How many times a probe recorded. Absent probe reads as zero, not an error."""
    stats = timer.summary(task_only=False).get(probe)
    return 0 if stats is None else stats.count
