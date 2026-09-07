"""Stage-3 equivalence gate on a full-shape stub of the upstream action head.

The upstream ``FlowmatchingActionHead`` is not importable here, so the stub
below implements ``get_action`` with the *same loop shape* upstream has -- an
internal ``torch.randn`` draw, ascending ``t/N`` time, integer bucketing, the
encoder / DiT / decoder call sequence -- with tiny deterministic modules. That
is enough to prove the two things the real gate on island B re-proves on the
real head: the pinned transcription reproduces ``get_action`` exactly under the
same noise, and resuming from a captured snapshot reproduces the tail of the
full loop exactly.
"""

from __future__ import annotations

import types

import pytest
import torch

from openpi.cache.groot.staged import GrootStagedRunner, denoise_loop
from openpi.cache.types import groot_n15_schedule

from .conftest import ACTION_DIM, ACTION_HORIZON, StubGrootModel

HIDDEN = 8
BUCKETS = 1000


class _FlowHeadStub(torch.nn.Module):
    """Upstream ``get_action``'s loop with stand-in modules of the same names."""

    def __init__(self, num_inference_timesteps: int) -> None:
        super().__init__()
        self.num_inference_timesteps = num_inference_timesteps
        self.num_timestep_buckets = BUCKETS
        self.action_horizon = ACTION_HORIZON
        self.config = types.SimpleNamespace(add_pos_embed=True)
        self.action_encoder_proj = torch.nn.Linear(ACTION_DIM, HIDDEN)
        self.position_embedding = torch.nn.Embedding(ACTION_HORIZON, HIDDEN)
        self.future_tokens = torch.nn.Embedding(2, HIDDEN)
        self.action_decoder_proj = torch.nn.Linear(HIDDEN, ACTION_DIM)
        self.vl_proj = torch.nn.LazyLinear(HIDDEN)

    # -- the modules the transcription calls, by name ---------------------

    def process_backbone_output(self, backbone_output):
        return types.SimpleNamespace(
            backbone_features=backbone_output["backbone_features"].float()
        )

    def state_encoder(self, state, embodiment_id):
        del embodiment_id
        value = state.float().mean()
        return torch.zeros(state.shape[0], 1, HIDDEN) + value

    def action_encoder(self, actions, timesteps, embodiment_id):
        del embodiment_id
        # Timestep enters here so a resumed loop that mis-bucketed would differ.
        return (
            self.action_encoder_proj(actions.float())
            + timesteps.float()[:, None, None] * 1e-3
        )

    def model(self, *, hidden_states, encoder_hidden_states, timestep):
        cond = self.vl_proj(encoder_hidden_states).mean(1, keepdim=True)
        return hidden_states + cond + timestep.float()[:, None, None] * 1e-3

    def action_decoder(self, model_output, embodiment_id):
        del embodiment_id
        return self.action_decoder_proj(model_output)

    # -- upstream get_action, loop shape verbatim, noise drawn inside --------

    def get_action(self, backbone_output, action_input):
        processed = self.process_backbone_output(backbone_output)
        vl = processed.backbone_features
        embodiment_id = action_input["embodiment_id"]
        state_features = self.state_encoder(action_input["state"], embodiment_id)
        batch_size = vl.shape[0]
        actions = torch.randn(
            size=(batch_size, self.action_horizon, ACTION_DIM),
            dtype=vl.dtype,
            device=vl.device,
        )
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=vl.device
            )
            action_features = self.action_encoder(
                actions, timesteps_tensor, embodiment_id
            )
            if self.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=vl.device
                )
                action_features = action_features + self.position_embedding(
                    pos_ids
                ).unsqueeze(0)
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
                vl.shape[0], -1, -1
            )
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            actions = actions + dt * pred[:, -self.action_horizon :]
        return {"action_pred": actions}


def _model_with_flow_head(num_steps: int) -> StubGrootModel:
    torch.manual_seed(0)
    model = StubGrootModel()
    model.action_head = _FlowHeadStub(num_steps)
    return model


def _fixed_noise() -> torch.Tensor:
    return torch.randn(
        1, ACTION_HORIZON, ACTION_DIM, generator=torch.Generator().manual_seed(7)
    )


def _upstream_with_noise(head, backbone_outputs, action_inputs, noise):
    """Call the stub's own get_action with its internal randn pinned to ``noise``."""
    real = torch.randn

    def fake(*args, **kwargs):
        return noise.to(dtype=kwargs.get("dtype", noise.dtype))

    torch.randn = fake
    try:
        return head.get_action(backbone_outputs, action_inputs)["action_pred"]
    finally:
        torch.randn = real


@pytest.mark.parametrize("num_steps", [4, 8])
def test_transcription_matches_upstream_get_action_under_the_same_noise(num_steps):
    model = _model_with_flow_head(num_steps)
    runner = GrootStagedRunner(model, verify_upstream=False)
    noise = _fixed_noise()
    with runner.session():
        stage2 = runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))
        transcribed = runner.run_stage3(stage2, noise=noise).action_pred
        head_inputs = runner._head_inputs(stage2)  # noqa: SLF001
        upstream = _upstream_with_noise(
            model.action_head, head_inputs, stage2.action_inputs, noise
        )
    assert torch.equal(transcribed, upstream)


def test_run_stage2_is_the_upstream_full_path_and_carries_llm_output():
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage1 = runner.run_stage1(model.build_inputs())
        full = runner.run_stage2(stage1)
    assert full.action_pred.shape == (1, ACTION_HORIZON, ACTION_DIM)
    assert full.backbone_features.shape[0] == 1
    assert torch.equal(full.attention_mask, stage1.attention_mask)


@pytest.mark.parametrize("resume", [False, True])
def test_transcribed_outputs_satisfy_the_real_batchfeature_contract(resume):
    """Real GR00T rejects plain dicts even when action shape and values are right."""
    from transformers.feature_extraction_utils import BatchFeature

    model = _model_with_flow_head(4)
    checked = []

    def validate(action_outputs, backbone_outputs, *, is_training):
        assert isinstance(action_outputs, BatchFeature)
        assert isinstance(backbone_outputs, BatchFeature)
        assert not is_training
        checked.append(True)

    model.validate_data = validate
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))
        if resume:
            runner.run_stage3_from(
                stage2, _fixed_noise(), 0.5, schedule=groot_n15_schedule(4)
            )
        else:
            runner.run_stage3(stage2, noise=_fixed_noise())
    assert checked == [True]


@pytest.mark.parametrize("num_steps", [4, 8])
def test_resuming_from_a_hooked_snapshot_reproduces_the_full_loop_tail(num_steps):
    """Snapshots captured the way the collector captures them (action_encoder
    input, one per step) resume to exactly the full run's answer, at every t."""
    model = _model_with_flow_head(num_steps)
    runner = GrootStagedRunner(model, verify_upstream=False)
    schedule = groot_n15_schedule(num_steps)
    noise = _fixed_noise()
    captures: list[torch.Tensor] = []
    head = model.action_head
    original = head.action_encoder

    def hooked(actions, timesteps, embodiment_id):
        captures.append(actions.detach().clone())
        return original(actions, timesteps, embodiment_id)

    head.action_encoder = hooked
    with runner.session():
        stage2 = runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))
        full = runner.run_stage3(stage2, noise=noise).action_pred
    head.action_encoder = original
    assert len(captures) == num_steps
    assert torch.equal(captures[0].float(), noise.to(captures[0].dtype).float())

    for index in range(1, num_steps):
        start_t = schedule.snapshot_t(index)
        with runner.session():
            resumed = runner.run_stage3_from(
                stage2, captures[index][0], start_t, schedule=schedule
            )
        assert torch.equal(resumed.action_pred, full), f"t={start_t}"
        assert resumed.start_t == start_t
        assert resumed.steps_run == num_steps - index


def test_resume_refuses_a_schedule_the_head_is_not_running():
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))
        with pytest.raises(RuntimeError, match="groot_n15_k8_v1"):
            runner.run_stage3_from(
                stage2,
                torch.zeros(ACTION_HORIZON, ACTION_DIM),
                0.5,
                schedule=groot_n15_schedule(8),
            )


def test_resume_refuses_a_timestep_that_is_not_a_snapshot_point():
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))
        with pytest.raises(ValueError, match="not a recoverable timestep"):
            runner.run_stage3_from(
                stage2,
                torch.zeros(ACTION_HORIZON, ACTION_DIM),
                0.3,
                schedule=groot_n15_schedule(4),
            )


def test_remaining_steps_are_the_ascending_count_not_the_pi05_formula():
    schedule = groot_n15_schedule(4)
    assert [schedule.remaining_steps(t) for t in schedule.timesteps] == [3, 2, 1]
    # Pi0.5's floor(start_t * N + 0.5) would give 1, 2, 3 -- the reverse.
    assert [int(t * 4 + 0.5) for t in schedule.timesteps] == [1, 2, 3]


def test_live_schedule_tracks_the_head_step_count():
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    assert runner.live_schedule() == groot_n15_schedule(4)
    model.action_head.num_inference_timesteps = 8
    assert runner.live_schedule() == groot_n15_schedule(8)


def test_stage3_refuses_to_run_outside_a_session():
    model = _model_with_flow_head(4)
    runner = GrootStagedRunner(model, verify_upstream=False)
    with runner.session():
        stage2 = runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))
    with pytest.raises(RuntimeError, match="run_stage3"):
        runner.run_stage3(stage2)
    with pytest.raises(RuntimeError, match="run_stage3_from"):
        runner.run_stage3_from(
            stage2,
            torch.zeros(ACTION_HORIZON, ACTION_DIM),
            0.5,
            schedule=groot_n15_schedule(4),
        )


def test_denoise_loop_start_index_skips_exactly_the_steps_before_it():
    """Direct check of the loop primitive: the timesteps it feeds the encoder."""
    head = _FlowHeadStub(4)
    fed: list[int] = []
    original = head.action_encoder

    def hooked(actions, timesteps, embodiment_id):
        fed.append(int(timesteps[0]))
        return original(actions, timesteps, embodiment_id)

    head.action_encoder = hooked
    backbone = {"backbone_features": torch.zeros(1, 3, HIDDEN)}
    action_input = {"state": torch.zeros(1, 1, 4), "embodiment_id": torch.tensor([0])}
    denoise_loop(
        head,
        backbone,
        action_input,
        noise=torch.zeros(1, ACTION_HORIZON, ACTION_DIM),
        num_steps=4,
        start_index=2,
    )
    assert fed == [int(2 / 4 * BUCKETS), int(3 / 4 * BUCKETS)]
