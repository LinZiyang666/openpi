"""GR00T warm reset: bit parity with step_diag (K=4 / K=8), entry guards, measured counts (plan §4.3.3, §4.6, §9).

The stub head is shaped for the real ``staged.denoise_step`` (encoder -> DiT ->
decoder, all elementwise so autocast cannot reorder anything) and records the
timestep bucket and input of every step; both sides run on the same real
``GrootStagedRunner`` inside its session. Real-model parity is the manual GPU
test's claim.
"""

from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest
import torch

from exp.step_diag import envs as E
from exp.step_diag import groot as G
from exp.step_diag.serve_diag_groot import GROOT_VARIANT_MODES
from openpi.cache.components.judge import HitType
from openpi.cache.config import CacheConfig, WarmResetConfig, _dict_to_dataclass
from openpi.cache.groot import staged as S
from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot.staged import GrootStage2Output, GrootStagedRunner
from openpi.cache.storage_types import CachePayload
from openpi.cache.types import groot_n15_schedule
from openpi.cache.warm_reset import groot as WG
from openpi.cache.warm_reset.evidence import ExpectedEpisode, episode_problems
from openpi.cache.warm_reset.groot import (
    GrootWarmResetExecutor,
    build_groot_warm_reset,
    groot_self_start,
    run_groot_continuation,
)
from openpi.cache.warm_reset.runtime import WarmResetSession
from openpi.cache.warm_reset.types import private_noise, resolve_plan, resolve_self_plan
from tests.cache.warm_reset._arms import block_of, spec_for

from .conftest import ACTION_DIM, ACTION_HORIZON, StubGrootModel
from .test_groot_interceptor import _obs, _StubOrchestrator, _StubPolicy

H, D = ACTION_HORIZON, ACTION_DIM
BUCKETS = 1000


class _DiTHead:
    """Upstream head surface for ``staged.denoise_step``; records every step's bucket and input."""

    num_timestep_buckets = BUCKETS

    def __init__(self, k: int) -> None:
        self.num_inference_timesteps = k
        self.config = types.SimpleNamespace(action_horizon=H, action_dim=D, add_pos_embed=False)
        self.action_horizon = H
        self.future_tokens = types.SimpleNamespace(weight=torch.linspace(-0.3, 0.3, 2 * D).view(2, D))
        self.log: list = []
        self.process_calls = 0

    def process_backbone_output(self, backbone_outputs):
        self.process_calls += 1
        backbone_outputs["backbone_features"] = backbone_outputs["backbone_features"] * 1.5 + 0.25
        return backbone_outputs

    def state_encoder(self, state, embodiment_id):
        return torch.tanh(state.float()[..., :D])

    def action_encoder(self, actions, timesteps, embodiment_id):
        self.log.append((timesteps.tolist(), actions.detach().clone()))
        return torch.sin(2.0 * actions + timesteps.float()[:, None, None] / BUCKETS)

    def model(self, hidden_states, encoder_hidden_states, timestep):
        cond = encoder_hidden_states.float().mean(dim=(1, 2))[:, None, None]
        return torch.tanh(1.3 * hidden_states + cond + timestep.float()[:, None, None] / BUCKETS)

    def action_decoder(self, model_output, embodiment_id):
        return model_output * 0.7


def _model(k: int) -> StubGrootModel:
    """A stub with a DiT head; seeded so every instance has the same embedding weights."""
    torch.manual_seed(1234)
    model = StubGrootModel()
    model.action_head = _DiTHead(k)
    return model


def _runner(k: int):
    model = _model(k)
    return GrootStagedRunner(model, verify_upstream=False), model.action_head


def _stage2(value: float = 0.2) -> GrootStage2Output:
    state = torch.full((1, 1, 10), value)
    return GrootStage2Output(
        backbone_features=torch.linspace(-1.0, 1.0, 5 * 8).view(1, 5, 8) + value,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        action_inputs={"state": state, "embodiment_id": torch.zeros(1, dtype=torch.long)},
    )


def _like(seed: int = 3) -> torch.Tensor:
    return torch.randn(H, D, generator=torch.Generator().manual_seed(seed))


def _recording_step(log):
    original = S.denoise_step

    def step(head, vl, state_features, embodiment_id, actions, timesteps, dt):
        log.append((timesteps.tolist(), dt, actions.detach().clone()))
        return original(head, vl, state_features, embodiment_id, actions, timesteps, dt)

    return step


def _same_log(a, b) -> bool:
    return len(a) == len(b) and all(x[0] == y[0] and x[1] == y[1] and torch.equal(x[2], y[2]) for x, y in zip(a, b))


RC_CASES = [(4, mode, s) for mode in GROOT_VARIANT_MODES for s in (0.75, 0.5)]
LIBERO_CASES = [(8, E.warm_mode_of(arm), arm) for arm in E.LIBERO_SELF_ARMS_BY_POLICY["groot"]
                if E.warm_steps_of(arm) is not None]


def _arm(k: int, mode: str, s_or_arm):
    return (s_or_arm, E.warm_t_of(s_or_arm)) if k == 8 else (f"{mode}_t{s_or_arm}", s_or_arm)


# ------------------------------------------------------------------
# Continuation parity
# ------------------------------------------------------------------


@pytest.mark.parametrize("k,mode,s", RC_CASES + LIBERO_CASES)
def test_continuation_equals_groot_warm_variant_stage3(monkeypatch, k, mode, s):
    arm, start_t = _arm(k, mode, s)
    schedule = groot_n15_schedule(k)
    plan = resolve_plan(spec_for("groot", arm), schedule, start_t)
    stage2, start = _stage2(), _like()
    ref_runner, _ = _runner(k)
    ref_log: list = []
    with ref_runner.session():
        ref = G.groot_warm_variant_stage3(ref_runner, stage2, start, start_t, schedule=schedule,
                                          variant=GROOT_VARIANT_MODES[mode], step_fn=_recording_step(ref_log),
                                          num_steps=E.warm_steps_of(arm))
    new_runner, _ = _runner(k)
    new_log: list = []
    monkeypatch.setattr(S, "denoise_step", _recording_step(new_log))
    with new_runner.session():
        new = run_groot_continuation(new_runner, stage2, start, plan, schedule=schedule)
    assert _same_log(new_log, ref_log) and len(new_log) == plan.n_steps
    assert torch.equal(new.action_pred, ref.action_pred)
    assert new.steps_run == ref.steps_run == plan.n_steps and new.start_t == start_t
    # the evidence grid is the grid the loop executed
    assert [entry[0][0] for entry in new_log] == [int(tau * BUCKETS) for tau in WG.native_grid(plan)[0]]


# ------------------------------------------------------------------
# Self-start parity
# ------------------------------------------------------------------

SELF_CASES = [(4, m, s) for m in GROOT_VARIANT_MODES if m.startswith("self") for s in (0.75, 0.5)] + [
    c for c in LIBERO_CASES if c[1].startswith("self")]


@pytest.mark.parametrize("k,mode,s", SELF_CASES)
def test_self_start_equals_step_diag(k, mode, s):
    arm, start_t = _arm(k, mode, s)
    schedule = groot_n15_schedule(k)
    stage2, like = _stage2(-0.1), _like()
    ref_runner, ref_head = _runner(k)
    with ref_runner.session():
        ref = G.groot_self_start(ref_runner, stage2, like, start_t, schedule=schedule,
                                 variant=GROOT_VARIANT_MODES[mode], seed=4242)
    new_runner, new_head = _runner(k)
    plan = resolve_self_plan(spec_for("groot", arm), schedule, start_t)
    noise = private_noise(4242, tuple(like.shape[-2:]))[None, ...].to(device=like.device)
    with new_runner.session():
        x, steps = groot_self_start(new_runner, stage2, noise, plan, schedule=schedule)
    new = x.to(device=like.device, dtype=like.dtype).reshape(like.shape)
    assert torch.equal(new, ref) and steps == k
    assert [e[0] for e in new_head.log] == [e[0] for e in ref_head.log]
    assert all(torch.equal(a[1], b[1]) for a, b in zip(new_head.log, ref_head.log, strict=True))


# ------------------------------------------------------------------
# Served decision: GrootCacheInterceptor + executor vs step_diag install_warm_variant
# ------------------------------------------------------------------

EPISODE = {"experiment": "robocasa365", "task": "CloseFridge", "episode_id": 7}
EXTRA = {"task_uid": "y:eval:1:7", "attempt": 1, "task_id": 1, "orig_init_state_idx": 7}


def _payload(k: int, start_t: float) -> CachePayload:
    gen = torch.Generator().manual_seed(11)
    return CachePayload(action_chunk=torch.randn(H, D, generator=gen),
                        intermediates={start_t: torch.randn(H, D, generator=gen)},
                        denoising_num_steps=k, schedule_id=f"groot_n15_k{k}_v1")


def _groot_parts(tmp_path, arm: str, k: int, start_t: float, *, yaml_id="y", bundle_id="b"):
    spec = spec_for("groot", arm, evidence_dir=str(tmp_path / "evidence"))
    cfg = CacheConfig(denoise_schedule=f"groot_n15_k{k}_v1",
                      warm_reset=_dict_to_dataclass(WarmResetConfig, block_of(spec)))
    return spec, build_groot_warm_reset(cfg, bundle_id=bundle_id, yaml_id=yaml_id, yaml_path=None)


def _served_production(tmp_path, arm, k, start_t, seed=None):
    """The served GR00T stack; ``seed`` pins the self-start seed integer (parity only)."""
    model = _model(k)
    runner = GrootStagedRunner(model, verify_upstream=False)
    orch = _StubOrchestrator(HitType.WARM_START, _payload(k, start_t), start_t=start_t)
    spec, parts = _groot_parts(tmp_path, arm, k, start_t)
    if seed is not None:
        parts.session.self_seed = lambda: seed
    served = parts.wrap(GrootCacheInterceptor(_StubPolicy(model), runner, orchestrator=orch,
                                              warm_reset=parts.executor))
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    return served, model, spec, parts


@pytest.mark.parametrize("k,arm,start_t", [
    (4, "warmreset_t0.5", 0.5), (4, "resetfinal_t0.75", 0.75), (4, "midreset_t0.5", 0.5),
    (4, "midshoot_t0.5", 0.5), (4, "selfmidfinal_t0.5", 0.5), (4, "selfwarmshoot_t0.75", 0.75),
    (8, "midreset50_t0.5_n2", 0.5), (8, "selfmidreset_t0.75_n1", 0.75), (8, "selfresetfinal_t0.5_n2", 0.5),
])
def test_served_decision_equals_the_step_diag_path(tmp_path, monkeypatch, k, arm, start_t):
    mode = E.warm_mode_of(arm)
    seed = 99
    ref_model = _model(k)
    ref_runner = GrootStagedRunner(ref_model, verify_upstream=False)
    ref_orch = _StubOrchestrator(HitType.WARM_START, _payload(k, start_t), start_t=start_t)
    ref_icpt = GrootCacheInterceptor(_StubPolicy(ref_model), ref_runner, orchestrator=ref_orch)
    G.install_warm_variant(ref_runner, ref_orch, GROOT_VARIANT_MODES[mode], groot_n15_schedule(k),
                           self_start=E.is_self_mode(mode), num_steps=E.warm_steps_of(arm))
    monkeypatch.setattr(G._SELF, "seed", seed, raising=False)  # what GrootEvidencePolicy publishes
    ref = ref_icpt.get_action(_obs())
    served, model, _, _ = _served_production(tmp_path, arm, k, start_t, seed)
    new = served.get_action(_obs())
    np.testing.assert_array_equal(new["action.out"], ref["action.out"])
    assert [e[0] for e in model.action_head.log] == [e[0] for e in ref_model.action_head.log]
    meta = new["__hit_meta__"]["warm_reset"]
    n = E.warm_steps_of(arm) or groot_n15_schedule(k).remaining_steps(start_t)
    assert meta["continuation_nfe"] == n and meta["n_stage3_calls"] == 1 and meta["k"] == k
    assert len(meta["tau"]) == len(meta["bucket"]) == len(meta["t"]) == n
    assert meta["t"] == pytest.approx([1.0 - tau for tau in meta["tau"]])
    assert meta["decision_nfe"] == n + (k if E.is_self_mode(mode) else 0)


# ------------------------------------------------------------------
# Entry guards: all before any head call (plan §4.3.3, F13)
# ------------------------------------------------------------------


def _untouched(head) -> bool:
    return head.process_calls == 0 and head.log == []


def _enter(entry, runner, plan, schedule):
    if entry == "continuation":
        return run_groot_continuation(runner, _stage2(), _like(), plan, schedule=schedule)
    return groot_self_start(runner, _stage2(), _like()[None], plan, schedule=schedule)


@pytest.mark.parametrize("entry", ["continuation", "self_start"])
@pytest.mark.parametrize("case", ["library_k4_live_k8", "outside_session", "plan_other_schedule"])
def test_guards_refuse_before_the_first_head_call(entry, case):
    runner, head = _runner(8 if case == "library_k4_live_k8" else 4)
    schedule = groot_n15_schedule(4)
    if entry == "continuation":
        plan = resolve_plan(spec_for("groot", "midreset_t0.5"), schedule, 0.5)
    else:
        plan = resolve_self_plan(spec_for("groot", "selfmidreset_t0.5"), schedule, 0.5)
    if case == "plan_other_schedule":
        plan = dataclasses.replace(plan, schedule_id="groot_n15_k8_v1", k=8)
    if case == "outside_session":
        with pytest.raises(RuntimeError, match="session"):
            _enter(entry, runner, plan, schedule)
    else:
        with runner.session(), pytest.raises((RuntimeError, ValueError)):
            _enter(entry, runner, plan, schedule)
    assert _untouched(head)


def test_explicit_n_with_an_unrecoverable_start_is_refused_everywhere():
    runner, head = _runner(8)
    schedule = groot_n15_schedule(8)
    with pytest.raises(ValueError):
        resolve_plan(spec_for("groot", "midreset_t0.75_n1"), schedule, 0.7)
    plan = dataclasses.replace(resolve_plan(spec_for("groot", "midreset_t0.75_n1"), schedule, 0.75), start_t=0.7)
    with runner.session(), pytest.raises(ValueError, match="not a recoverable"):
        run_groot_continuation(runner, _stage2(), _like(), plan, schedule=schedule)
    self_plan = dataclasses.replace(resolve_self_plan(spec_for("groot", "selfresetfinal_t0.75_n1"), schedule, 0.75),
                                    start_t=0.7)
    with runner.session(), pytest.raises(ValueError, match="not a recoverable"):
        groot_self_start(runner, _stage2(), _like()[None], self_plan, schedule=schedule)
    assert _untouched(head)


def test_live_step_count_changed_after_assembly_is_refused_on_the_next_decision(tmp_path):
    served, model, _, _ = _served_production(tmp_path, "selfmidreset_t0.5", 4, 0.5)
    served.get_action(_obs())
    head = model.action_head
    calls = head.process_calls
    head.num_inference_timesteps = 8
    with pytest.raises(RuntimeError, match="action head is"):
        served.get_action(_obs())
    assert head.process_calls == calls


@pytest.mark.parametrize("entry", ["continuation", "self_start"])
@pytest.mark.parametrize("bad_batch", [0, 2])
def test_entry_rejects_start_batch_before_preprocessing(entry, bad_batch):
    runner, head = _runner(4)
    schedule = groot_n15_schedule(4)
    spec = spec_for("groot", "selfmidreset_t0.5")
    x = _like()[None].expand(bad_batch, -1, -1).clone()
    if entry == "continuation":
        plan, run = resolve_plan(spec, schedule, 0.5), run_groot_continuation
    else:
        plan, run = resolve_self_plan(spec, schedule, 0.5), groot_self_start
    with runner.session(), pytest.raises(ValueError, match="batch"):
        run(runner, _stage2(), x, plan, schedule=schedule)
    assert _untouched(head)


# ------------------------------------------------------------------
# Interceptor wiring
# ------------------------------------------------------------------


def _executor(arm="warmreset_t0.5"):
    spec = spec_for("groot", arm)
    return GrootWarmResetExecutor(spec, WarmResetSession(spec))


def test_interceptor_refuses_trace_cp2_and_online_rit():
    runner, _ = _runner(4)
    policy = _StubPolicy(runner._model)
    with pytest.raises(ValueError, match="warm_reset"):
        GrootCacheInterceptor(policy, runner, trace=types.SimpleNamespace(twins=None, plan=None),
                              trace_vision_fields=("vision_0",), warm_reset=_executor())
    cp2 = types.SimpleNamespace(has_checkpoint=lambda cp: cp.name == "CP2", artifact_meta={})
    with pytest.raises(ValueError, match="CP2"):
        GrootCacheInterceptor(policy, runner, orchestrator=cp2, warm_reset=_executor())
    rit = types.SimpleNamespace(has_checkpoint=lambda cp: False, continuation_spec=lambda cp: object())
    with pytest.raises(ValueError, match="online_rit"):
        GrootCacheInterceptor(policy, runner, orchestrator=rit, warm_reset=_executor())


def test_absent_block_keeps_the_exact_resume_call_and_wire():
    runner, _ = _runner(4)
    orch = _StubOrchestrator(HitType.WARM_START, _payload(4, 0.5), start_t=0.5)
    icpt = GrootCacheInterceptor(_StubPolicy(runner._model), runner, orchestrator=orch)
    seen = []
    resume = runner.run_stage3_from

    def spy(stage2, start_x, start_t, *, schedule, **kw):
        seen.append((start_x, start_t, schedule, kw))
        return resume(stage2, start_x, start_t, schedule=schedule, **kw)

    runner.run_stage3_from = spy
    out = icpt.get_action(_obs())
    ((start_x, start_t, schedule, kw),) = seen
    assert torch.equal(start_x, orch._payload.intermediates[0.5]) and start_t == 0.5
    assert schedule == groot_n15_schedule(4) and kw == {}
    assert "warm_reset" not in out["__hit_meta__"]


# ------------------------------------------------------------------
# Measured counts through the evidence, and K + N pricing
# ------------------------------------------------------------------


def _episode(served, n):
    for _ in range(n):
        served.get_action(_obs())
    served.on_episode_end(True)
    rows = [json.loads(line) for line in served.evidence_path.read_text().splitlines()]
    return rows


def _expected(spec, k, start_t, n):
    return ExpectedEpisode(
        task_uid=EXTRA["task_uid"], attempt=1, outcome=True, n_decisions=n, yaml_id="y", bundle_id="b",
        spec=spec, spec_digest=spec.digest(), yaml_sha256=None, schedule_id=f"groot_n15_k{k}_v1", k=k,
        start_t=start_t, identity={**EPISODE, **EXTRA},
    )


def test_groot_libero_self_arm_is_priced_k_plus_n(tmp_path):
    served, _, spec, _ = _served_production(tmp_path, "selfmidreset_t0.75_n1", 8, 0.75)
    rows = _episode(served, 3)
    result = episode_problems(rows, expected=_expected(spec, 8, 0.75, 3))
    assert not +result["problems"], result["problems"]
    assert (result["continuation_nfe"], result["self_start_nfe"], result["total_nfe"]) == (3, 24, 27)


@pytest.mark.parametrize("arm", ["warmreset_t0.5_n2", "midreset_t0.5_n2"])
@pytest.mark.parametrize("delta", [-1, 1])
def test_continuation_step_count_is_measured(tmp_path, monkeypatch, arm, delta):
    monkeypatch.setattr(WG, "_loop_steps", lambda plan: plan.n_steps + delta)
    served, _, spec, _ = _served_production(tmp_path, arm, 8, 0.5)
    rows = _episode(served, 2)
    assert rows[0]["warm_reset"]["continuation_nfe"] == 2 + delta
    assert episode_problems(rows, expected=_expected(spec, 8, 0.5, 2))["problems"]["steps_mismatch"] == 2


@pytest.mark.parametrize("delta", [-1, 1])
def test_self_start_step_count_is_measured(tmp_path, delta):
    served, _, spec, _ = _served_production(tmp_path, "selfresetfinal_t0.5_n2", 8, 0.5)
    runner = served._inner._runner
    run_stage3 = runner.run_stage3

    def faulty(stage2, *, noise=None, on_step=None):
        if delta < 0:  # the observer misses the first step
            return run_stage3(stage2, noise=noise, on_step=lambda i, a, b: None if i == 0 else on_step(i, a, b))
        out = run_stage3(stage2, noise=noise, on_step=on_step)
        on_step(8, out.action_pred, out.action_pred)  # one step too many
        return out

    runner.run_stage3 = faulty
    rows = _episode(served, 2)
    assert rows[0]["warm_reset"]["self_direct_nfe"] == 8 + delta
    problems = episode_problems(rows, expected=_expected(spec, 8, 0.5, 2))["problems"]
    assert problems["self_direct_nfe_mismatch"] == 2
