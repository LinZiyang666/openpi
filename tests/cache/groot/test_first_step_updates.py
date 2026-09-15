"""Opt-in first-step capture and batched side evaluation on the flow-head stub."""

from __future__ import annotations

import torch

from openpi.cache.groot.staged import GrootStagedRunner, denoise_loop, denoise_step
from openpi.cache.types import groot_n15_schedule

from .conftest import ACTION_DIM, ACTION_HORIZON
from .test_groot_stage3 import _fixed_noise, _model_with_flow_head


def _stage2(model, runner):
    return runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))


def test_capture_is_an_observer_and_matches_the_loops_first_step():
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    schedule = groot_n15_schedule(4)
    snap = _fixed_noise()
    with runner.session():
        stage2 = _stage2(model, runner)
        plain = runner.run_stage3_from(stage2, snap, 0.5, schedule=schedule)
        captured = runner.run_stage3_from(stage2, snap, 0.5, schedule=schedule, capture_first_step=True)
        # the first step, computed on its own
        head_inputs = runner._head_inputs(stage2)  # noqa: SLF001
        seen = []
        denoise_loop(
            model.action_head,
            head_inputs,
            stage2.action_inputs,
            noise=snap,
            num_steps=4,
            start_index=2,
            on_step=lambda t, x_in, x_out: seen.append((t, x_in.clone(), x_out.clone())),
        )
    assert plain.first_step_x is None and plain.first_step_input is None
    assert torch.equal(plain.action_pred, captured.action_pred)
    assert torch.equal(captured.first_step_input, snap)
    assert seen[0][0] == 2
    assert torch.equal(captured.first_step_x, seen[0][2])
    assert len(seen) == 2


def test_batched_side_evaluation_matches_serial_steps():
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    schedule = groot_n15_schedule(4)
    gen = torch.Generator().manual_seed(11)
    snaps = [(t, torch.randn(ACTION_HORIZON, ACTION_DIM, generator=gen)) for t in (0.75, 0.5, 0.25)]
    with runner.session():
        stage2 = _stage2(model, runner)
        batched = runner.first_step_updates(stage2, snaps, schedule=schedule)
        serial = []
        for t, x in snaps:
            head_inputs = runner._head_inputs(stage2)  # noqa: SLF001
            head = model.action_head
            processed = head.process_backbone_output(head_inputs)
            vl = processed.backbone_features
            emb = stage2.action_inputs["embodiment_id"]
            sf = head.state_encoder(stage2.action_inputs["state"], emb)
            index = schedule.snapshot_index(t)
            bucket = int(index / 4 * head.num_timestep_buckets)
            ts = torch.full((1,), bucket)
            serial.append(denoise_step(head, vl, sf, emb, x[None], ts, 0.25))
        single = runner.first_step_updates(stage2, snaps[:1], schedule=schedule)
        empty = runner.first_step_updates(stage2, [], schedule=schedule)
    assert empty == []
    assert len(batched) == 3
    for (x_used, b), s, (t, x) in zip(batched, serial, snaps):
        assert b.shape == (1, ACTION_HORIZON, ACTION_DIM)
        assert torch.allclose(b, s, atol=1e-6)
        # the returned input is the one the head consumed (head dtype), not the caller's tensor
        assert x_used.dtype == b.dtype
        assert torch.equal(x_used, x[None].to(dtype=b.dtype))
    assert torch.allclose(single[0][1], serial[0], atol=1e-6)


def test_consumed_input_is_the_cast_snapshot_not_the_stored_one():
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    schedule = groot_n15_schedule(4)
    stored = torch.full((ACTION_HORIZON, ACTION_DIM), 1.001)  # not representable in BF16
    with runner.session():
        stage2 = _stage2(model, runner)
        ((x_used, x_out),) = runner.first_step_updates(stage2, [(0.5, stored)], schedule=schedule)
        cap = runner.run_stage3_from(stage2, stored, 0.5, schedule=schedule, capture_first_step=True)
    if x_used.dtype != torch.float32:
        assert not torch.equal(x_used.float(), stored[None])  # rounding happened in the cast
    assert torch.equal(cap.first_step_input, x_used)  # executed path reports the same consumed input
    assert torch.equal(cap.first_step_x, x_out)


def test_side_evaluation_refuses_a_foreign_schedule():
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = _stage2(model, runner)
        try:
            runner.first_step_updates(stage2, [(0.5, _fixed_noise())], schedule=groot_n15_schedule(8))
        except RuntimeError as exc:
            assert "groot_n15_k8_v1" in str(exc)
        else:
            raise AssertionError("expected a schedule mismatch")


def test_split_miss_path_equals_run_stage2():
    """run_stage2_llm + run_stage3 is exactly what run_stage2 does (the online MISS path)."""
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    noise = _fixed_noise()
    with runner.session():
        stage1 = runner.run_stage1(model.build_inputs())
        stage2 = runner.run_stage2_llm(stage1)
        split = runner.run_stage3(stage2, noise=noise).action_pred
        stage2b = runner.run_stage2_llm(stage1)
        again = runner.run_stage3(stage2b, noise=noise).action_pred
    assert torch.equal(split, again)
