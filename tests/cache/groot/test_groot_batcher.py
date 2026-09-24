"""GR00T stage-3 additions for the trace mode (plan §6.3, §9-3, §9-4, §9-7).

Runner: ``run_stage3(on_step=)`` observes every step and refuses to observe
upstream's own loop; ``sample_noise`` reproduces upstream's draw (sampling
parity under a seed), a private generator leaves the global stream alone and
the dtype probe is cached and RNG-neutral. Batcher: B=1 through the batcher
is bit-identical to the direct call, a same-shape B>1 bucket agrees with the
per-row runs, different prompt lengths never share a bucket, a MISS bucket
mixing legacy / trace requests hands each its own snapshot set, and the
zero-padding counterexample is kept: padding changes the actions.
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
)

from .conftest import ACTION_DIM, ACTION_HORIZON
from .test_groot_stage3 import _fixed_noise, _model_with_flow_head

N_STEPS = 4
SCHEDULE = groot_n15_schedule(N_STEPS)


def _runner(model) -> GrootStagedRunner:
    return GrootStagedRunner(model, verify_upstream=False)


def _stage2(runner, model, *, prompt_tokens=None, state_scale=1.0):
    inputs = model.build_inputs(prompt_tokens=prompt_tokens)
    inputs["state"] = inputs["state"] * state_scale
    with runner.session():
        stage1 = runner.run_stage1(inputs)
        return runner.run_stage2_llm(stage1)


# ---------------------------------------------------------------------------
# staged.py: on_step / sample_noise / dtype probe
# ---------------------------------------------------------------------------


def test_on_step_without_noise_is_refused():
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    stage2 = _stage2(runner, model)
    with runner.session(), pytest.raises(ValueError, match="on_step"):
        runner.run_stage3(stage2, noise=None, on_step=lambda *a: None)


def test_on_step_fires_once_per_step_and_does_not_change_the_loop():
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    stage2 = _stage2(runner, model)
    noise = _fixed_noise()
    seen: list[tuple[int, torch.Tensor]] = []
    with runner.session():
        observed = runner.run_stage3(
            stage2, noise=noise, on_step=lambda s, x_in, x_out: seen.append((s, x_in.clone()))
        )
        plain = runner.run_stage3(stage2, noise=noise)
    assert [s for s, _ in seen] == list(range(N_STEPS))
    assert torch.equal(observed.action_pred, plain.action_pred)
    # step 0 consumed the noise itself (cast to the head dtype)
    assert torch.equal(seen[0][1], noise.to(seen[0][1].dtype))


def test_sampling_parity_with_upstream_draw():
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    stage2 = _stage2(runner, model)
    torch.manual_seed(11)
    with runner.session():
        upstream = runner.run_stage3(stage2, noise=None).action_pred
    torch.manual_seed(11)
    with runner.session():
        z = runner.sample_noise(stage2)
        ours = runner.run_stage3(stage2, noise=z).action_pred
    assert z.shape == (1, ACTION_HORIZON, ACTION_DIM)
    assert torch.equal(upstream, ours)


def test_private_generator_leaves_the_global_stream_untouched():
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    stage2 = _stage2(runner, model)
    torch.manual_seed(3)
    before = torch.get_rng_state()
    gen = torch.Generator().manual_seed(99)
    with runner.session():
        z1 = runner.sample_noise(stage2, generator=gen)
    assert torch.equal(torch.get_rng_state(), before)
    gen2 = torch.Generator().manual_seed(99)
    with runner.session():
        z2 = runner.sample_noise(stage2, generator=gen2)
    assert torch.equal(z1, z2)


def test_dtype_probe_runs_once_and_is_rng_neutral():
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    stage2 = _stage2(runner, model)
    calls = {"n": 0}
    real = model.action_head.process_backbone_output

    def counted(backbone_output):
        calls["n"] += 1
        return real(backbone_output)

    model.action_head.process_backbone_output = counted
    torch.manual_seed(5)
    state = torch.get_rng_state()
    with runner.session():
        dtype = runner._noise_dtype(stage2)  # noqa: SLF001 - the probe itself
    assert torch.equal(torch.get_rng_state(), state)
    assert calls["n"] == 1 and dtype == torch.float32
    with runner.session():
        runner._noise_dtype(stage2)  # noqa: SLF001
        runner.sample_noise(stage2, generator=torch.Generator().manual_seed(0))
    assert calls["n"] == 1  # cached: no second prologue run


# ---------------------------------------------------------------------------
# batcher.py
# ---------------------------------------------------------------------------


def _core(runner, **kw):
    return BatchingCore(gb.GrootStageBatcher(runner), device="cpu", max_wait_ms=100.0, **kw)


def test_stage1_and_stage2_are_refused():
    model = _model_with_flow_head(N_STEPS)
    b = gb.GrootStageBatcher(_runner(model))
    with pytest.raises(NotImplementedError):
        b.run_stage1_batch([])
    with pytest.raises(NotImplementedError):
        b.run_stage2_batch([], capture=False)


def test_b1_through_batcher_matches_direct_bitwise():
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    stage2 = _stage2(runner, model)
    noise = _fixed_noise()
    cond = gb.stage3_input(runner, stage2)
    direct, caps = gb.run_miss(runner, stage2, noise, schedule=SCHEDULE, capture=True)
    with _core(runner) as bc:
        out = bc.submit_to_stage(
            3, "default",
            Stage3MissPayload(cond, noise[0], N_STEPS, save_timesteps=SCHEDULE.timesteps),
            timeout=5.0,
        )
    assert torch.equal(out.action, direct.action_pred)
    assert sorted(out.intermediates) == sorted((0.0,) + SCHEDULE.timesteps)
    for i in range(N_STEPS):
        assert torch.equal(out.intermediates[SCHEDULE.snapshot_t(i)], caps[i])
    # warm: resume from the captured step-2 input reproduces the tail
    start_t = SCHEDULE.snapshot_t(2)
    warm_direct = gb.run_warm(runner, stage2, caps[2], start_t, schedule=SCHEDULE, capture_first_step=True)
    with _core(runner) as bc:
        w = bc.submit_to_stage(
            3, "default",
            Stage3WarmStartPayload(cond, caps[2][0], start_t, N_STEPS, capture_first_step=True),
            timeout=5.0,
        )
    assert torch.equal(w.action, warm_direct.action_pred)
    assert torch.equal(w.action, direct.action_pred)
    assert torch.equal(w.first_step_input, warm_direct.first_step_input)
    assert torch.equal(w.first_step_x, caps[3])


def test_same_shape_bucket_agrees_with_per_row_runs():
    """Three connections, same prompt length, different state -> one bucket, per-row parity."""
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    stage2s = [_stage2(runner, model, state_scale=s) for s in (1.0, 2.0, 0.5)]
    noises = [torch.randn(1, ACTION_HORIZON, ACTION_DIM, generator=torch.Generator().manual_seed(i)) for i in range(3)]
    singles = [gb.run_miss(runner, s2, z, schedule=SCHEDULE, capture=False)[0].action_pred for s2, z in zip(stage2s, noises)]
    conds = [gb.stage3_input(runner, s2) for s2 in stage2s]
    keys = {gb.GrootStageBatcher.bucket_key(Stage3MissPayload(c, z[0], N_STEPS)) for c, z in zip(conds, noises)}
    assert len(keys) == 1
    results = {}
    barrier = threading.Barrier(3)
    calls = []
    real = runner.run_stage3

    def spy(stage2, **kw):
        calls.append(int(stage2.backbone_features.shape[0]))
        return real(stage2, **kw)

    runner.run_stage3 = spy
    with _core(runner, max_batch_size=8) as bc:
        def worker(i):
            barrier.wait()
            results[i] = bc.submit_to_stage(3, "default", Stage3MissPayload(conds[i], noises[i][0], N_STEPS), timeout=5.0)
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in ts: t.start()
        for t in ts: t.join(10)
    assert calls == [3]
    for i in range(3):
        assert torch.allclose(results[i].action, singles[i], atol=1e-5, rtol=1e-5)
        assert results[i].intermediates is None


def test_different_prompt_lengths_never_share_a_bucket():
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    a = gb.stage3_input(runner, _stage2(runner, model, prompt_tokens=5))
    b = gb.stage3_input(runner, _stage2(runner, model, prompt_tokens=7))
    z = torch.zeros(ACTION_HORIZON, ACTION_DIM)
    ka = gb.GrootStageBatcher.bucket_key(Stage3MissPayload(a, z, N_STEPS))
    kb = gb.GrootStageBatcher.bucket_key(Stage3MissPayload(b, z, N_STEPS))
    assert ka != kb
    with pytest.raises(ValueError, match="shapes differ"):
        gb.cat_stage2([a, b])


def test_zero_padding_would_change_the_actions():
    """The counterexample the same-shape rule exists for (no conditioning mask upstream)."""
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    stage2 = _stage2(runner, model, prompt_tokens=5)
    noise = _fixed_noise()
    ref = gb.run_miss(runner, stage2, noise, schedule=SCHEDULE, capture=False)[0].action_pred
    pad = 8
    from openpi.cache.groot.staged import GrootStage2Output

    padded = GrootStage2Output(
        backbone_features=torch.nn.functional.pad(stage2.backbone_features, (0, 0, 0, pad)),
        attention_mask=torch.nn.functional.pad(stage2.attention_mask, (0, pad)),
        action_inputs=stage2.action_inputs,
    )
    out = gb.run_miss(runner, padded, noise, schedule=SCHEDULE, capture=False)[0].action_pred
    assert not torch.allclose(out, ref)


def test_miss_bucket_union_and_per_request_filter():
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    cond = gb.stage3_input(runner, _stage2(runner, model))
    z = torch.zeros(ACTION_HORIZON, ACTION_DIM)
    b = gb.GrootStageBatcher(runner)
    outs = b.run_stage3_miss(
        [Stage3MissPayload(cond, z, N_STEPS), Stage3MissPayload(cond, z, N_STEPS, save_timesteps=(0.5,)),
         Stage3MissPayload(cond, z, N_STEPS, save_timesteps=SCHEDULE.timesteps)],
        num_steps=N_STEPS,
        save_timesteps_per_request=[None, (0.5,), SCHEDULE.timesteps],
    )
    assert outs[0].intermediates is None
    assert sorted(outs[1].intermediates) == [0.0, 0.5]
    assert sorted(outs[2].intermediates) == sorted((0.0,) + SCHEDULE.timesteps)


def test_all_none_bucket_keeps_the_unobserved_call_shape():
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    cond = gb.stage3_input(runner, _stage2(runner, model))
    seen = {}
    real = runner.run_stage3

    def spy(stage2, **kw):
        seen.update(kw)
        return real(stage2, **kw)

    runner.run_stage3 = spy
    gb.GrootStageBatcher(runner).run_stage3_miss(
        [Stage3MissPayload(cond, torch.zeros(ACTION_HORIZON, ACTION_DIM), N_STEPS)],
        num_steps=N_STEPS, save_timesteps_per_request=[None],
    )
    assert seen["on_step"] is None


def test_wrong_num_steps_is_refused():
    model = _model_with_flow_head(N_STEPS)
    runner = _runner(model)
    cond = gb.stage3_input(runner, _stage2(runner, model))
    with pytest.raises(RuntimeError, match="steps"):
        gb.GrootStageBatcher(runner).run_stage3_miss(
            [Stage3MissPayload(cond, torch.zeros(ACTION_HORIZON, ACTION_DIM), N_STEPS + 1)],
            num_steps=N_STEPS + 1, save_timesteps_per_request=[None],
        )
