"""CUDA readiness contract of the GR00T stage-3 batcher (plan §6.2, §12-I).

The producer writes ``stage2`` / ``noise`` / ``start_x`` on its own CUDA
stream and records ``ready_events`` there; the worker waits on them before it
stacks anything. With the stub head on the GPU the batched reply must equal
the same-stream direct computation. Skipped without CUDA.
"""

from __future__ import annotations

import threading

import pytest
import torch

from openpi.cache.groot import batcher as gb
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.types import groot_n15_schedule
from openpi.serving.batching_core import (
    BatchingCore,
    Stage3MissPayload,
    Stage3WarmStartPayload,
    record_ready_events,
)

from .conftest import ACTION_DIM, ACTION_HORIZON, TOKENS_PER_IMAGE, EMB_DIM
from .test_groot_stage3 import _FlowHeadStub, _model_with_flow_head

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

N_STEPS = 4
SCHEDULE = groot_n15_schedule(N_STEPS)


def _cuda_model():
    model = _model_with_flow_head(N_STEPS)
    dev = torch.device("cuda")
    eagle = model.backbone.eagle_model
    eagle.language_model._embed.to(dev)  # noqa: SLF001 - stub internals

    def extract_feature(pixel_values):
        eagle.extract_calls += 1
        n_images = pixel_values.shape[0]
        base = torch.arange(n_images, dtype=torch.float32, device=pixel_values.device).view(n_images, 1, 1)
        return base + torch.ones(n_images, TOKENS_PER_IMAGE, EMB_DIM, device=pixel_values.device)

    eagle.extract_feature = extract_feature
    model.action_head = _FlowHeadStub(N_STEPS).to(dev).eval()
    model.device = "cuda"
    return model


def _inputs(model):
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in model.build_inputs().items()}


def test_record_ready_events_marks_the_producer_stream():
    assert len(record_ready_events("cuda")) == 1
    assert record_ready_events("cpu") == ()


def test_batched_reply_equals_direct_under_a_side_stream_producer():
    model = _cuda_model()
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = runner.run_stage2_llm(runner.run_stage1(_inputs(model)))
        z = runner.sample_noise(stage2, generator=torch.Generator(device="cuda").manual_seed(3))
    torch.cuda.synchronize()
    # materialise the lazy head layer on the default stream first
    direct, caps = gb.run_miss(runner, stage2, z, schedule=SCHEDULE, capture=True)
    warm_t = SCHEDULE.snapshot_t(2)
    warm_direct = gb.run_warm(runner, stage2, caps[2], warm_t, schedule=SCHEDULE, capture_first_step=True)
    torch.cuda.synchronize()

    results = {}
    with BatchingCore(gb.GrootStageBatcher(runner), device="cuda", max_batch_size=4, max_wait_ms=100.0) as bc:

        def producer():
            # Host identity is prepared before the stream work (no sync later).
            cond0 = gb.stage3_input(runner, stage2)
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                # Hold the producer stream busy first so the writes below are
                # still pending when the worker pulls the request: without the
                # ready events the worker would stack half-written inputs.
                torch.cuda._sleep(300_000_000)
                # fresh copies written on the side stream, then the events
                feats = stage2.backbone_features.clone() * 1.0
                s2 = type(stage2)(
                    backbone_features=feats,
                    attention_mask=stage2.attention_mask.clone(),
                    action_inputs=stage2.action_inputs,
                )
                noise = z.clone()
                start_x = caps[2].clone()
                cond = gb.GrootStage3Input(
                    stage2=s2, schedule_id=cond0.schedule_id,
                    embodiment=cond0.embodiment, exec_domain=cond0.exec_domain,
                )
                events = record_ready_events("cuda")
            payloads = [
                Stage3MissPayload(cond, noise[0], N_STEPS, save_timesteps=SCHEDULE.timesteps, ready_events=events),
                Stage3WarmStartPayload(cond, start_x[0], warm_t, N_STEPS, capture_first_step=True, ready_events=events),
            ]
            results["outs"] = bc.submit_many_to_stage(3, "default", payloads, timeout=30.0)

        t = threading.Thread(target=producer)
        t.start()
        t.join(60)
    outs = results["outs"]
    torch.cuda.synchronize()
    assert torch.equal(outs[0].action, direct.action_pred)
    for i in range(N_STEPS):
        assert torch.equal(outs[0].intermediates[SCHEDULE.snapshot_t(i)], caps[i])
    assert torch.equal(outs[1].action, warm_direct.action_pred)
    assert torch.equal(outs[1].first_step_x, warm_direct.first_step_x)
