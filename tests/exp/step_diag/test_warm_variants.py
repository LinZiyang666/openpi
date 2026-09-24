"""CPU tests for the warm-start continuation variants (``warmreset`` / ``warmshoot`` arms).

The variants keep the call count of the resume (``floor(start_t * K + 0.5)`` Euler steps) but use
``dt = -1/remaining_steps``; ``reset_t`` restarts the flow time at 1, ``overshoot`` keeps the cache's
own start time. The tests pin the exact ``(t, dt)`` sequences the network sees, the routing of the
served WARM_START call through the variant (shadow bracket untouched), the arm-id / yaml validation
and the analysis-side parsing of the new arm ids.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from exp.step_diag import envs as _envs
from exp.step_diag import pi05 as P

H, D = 50, 32


class _TimeRecordingModel:
    """``denoise_step`` with the production signature; records the t it is asked at, returns a constant."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        self.times.append(float(timestep[0]))
        return torch.ones_like(x_t)

    # staged API stubs the interceptor binds at construction
    def run_stage1(self, observation):
        return SimpleNamespace(state=torch.zeros(1, D), prefix_embs=torch.zeros(1, 4, 8))

    def run_stage2(self, stage1):
        return _stage2()

    def sample_noise(self, shape, device, generator=None):
        return torch.randn(*shape, generator=generator)

    def run_stage3(self, stage2, noise=None, num_steps=10, return_intermediates=False, save_timesteps=()):
        x = noise if noise is not None else torch.randn(1, H, D)
        return SimpleNamespace(action_chunk=x, intermediates=None)

    def run_stage3_from(self, stage2, start_x, start_t, *, num_steps=10):
        # the exact resume, as pi0_pytorch.run_stage3_from
        dt = torch.tensor(-1.0 / num_steps)
        n = int(start_t * num_steps + 0.5)
        t = torch.tensor(1.0)
        for _ in range(num_steps - n):
            t = t + dt
        x = start_x
        for _ in range(n):
            x = x + dt * self.denoise_step(None, None, None, x, t.expand(1))
            t = t + dt
        return SimpleNamespace(action_chunk=x)


def _stage2():
    return SimpleNamespace(stage1=SimpleNamespace(state=torch.zeros(1, D), prefix_pad_masks=torch.ones(1, 4, dtype=torch.bool)),
                           past_key_values=None)


@pytest.mark.parametrize(
    "variant,start_t,expected_t,expected_dt",
    [
        ("reset_t", 0.2, [1.0, 0.5], -0.5),
        ("overshoot", 0.2, [0.2, -0.3], -0.5),
        ("reset_t", 0.3, [1.0, 2 / 3, 1 / 3], -1 / 3),
        ("overshoot", 0.1, [0.1], -1.0),
        ("reset_final", 0.2, [1.0, 0.5], -0.5),
        ("reset_final", 0.3, [1.0, 2 / 3, 1 / 3], -1 / 3),
        ("mid_final", 0.1, [0.9], -0.9),          # one K=10 grid step below noise, 1 step to 0
        ("mid_final", 0.2, [0.9, 0.45], -0.45),   # same entry, 2 steps to 0
        ("mid_final50", 0.1, [0.5], -0.5),        # entry 0.5, 1 step
        ("mid_final50", 0.2, [0.5, 0.25], -0.25),  # entry 0.5, 2 steps
        ("mid_snap", 0.2, [0.9, 0.45], -0.45),    # snapshot start, midfinal's entry
        ("mid_snap50", 0.2, [0.5, 0.25], -0.25),  # snapshot start fed at 0.5
    ],
)
def test_variant_time_and_step_sequences(variant, start_t, expected_t, expected_dt):
    model = _TimeRecordingModel()
    x0 = torch.zeros(1, H, D)
    out = P.warm_variant_stage3(model, _stage2(), x0, start_t, num_steps=10, variant=variant)
    assert len(model.times) == len(expected_t)
    assert model.times == pytest.approx(expected_t, abs=1e-6)
    # every step adds dt * 1, so the chunk moves by n * dt = -1 in total
    assert torch.allclose(out.action_chunk, torch.full_like(x0, len(expected_t) * expected_dt))


def test_variant_matches_resume_call_count_and_differs_in_grid():
    model_r, model_v = _TimeRecordingModel(), _TimeRecordingModel()
    x0 = torch.zeros(1, H, D)
    model_r.run_stage3_from(_stage2(), x0, 0.2, num_steps=10)
    P.warm_variant_stage3(model_v, _stage2(), x0, 0.2, num_steps=10, variant="reset_t")
    assert len(model_r.times) == len(model_v.times) == 2
    assert model_r.times == pytest.approx([0.2, 0.1], abs=1e-6)
    assert model_v.times == pytest.approx([1.0, 0.5], abs=1e-6)


def test_unknown_variant_and_empty_resume_are_refused():
    with pytest.raises(ValueError):
        P.warm_variant_stage3(_TimeRecordingModel(), _stage2(), torch.zeros(1, H, D), 0.2, num_steps=10, variant="nope")
    with pytest.raises(ValueError):
        P.warm_variant_stage3(_TimeRecordingModel(), _stage2(), torch.zeros(1, H, D), 0.01, num_steps=10, variant="reset_t")


def test_interceptor_routes_executed_warm_start_to_the_variant_but_not_the_shadow_bracket(tmp_path):
    """The served WARM_START call (outside ``in_shadow``) uses the variant grid; the shadow resume does not."""
    from exp.step_diag import recorder as R
    from openpi.cache.timing import SystemTimer
    from tests.cache.test_interceptor import FakePolicy

    model = _TimeRecordingModel()
    model.config = SimpleNamespace(pytorch_compile_mode=None, action_horizon=H, action_dim=D)
    spec = R.DiagSpec(experiment_id="e", env_id="pi05_rc", arm_id="warmreset_t0.2", mode="warmreset", k_full=10,
                      k_set=(), warm_ts=(), n_primary=4, n_dense_extra=2, action_shape=(H, D), config_sha="c")
    rec = R.DiagRecorder(spec, tmp_path)
    icpt = P.Pi05DiagInterceptor(FakePolicy(model), timer=SystemTimer(enabled=False), diag=rec, mode="warmreset",
                                 exec_steps=None, warm_variant="reset_t")
    icpt.infer  # the wrapped bindings exist
    icpt._tl.n_steps = 0
    icpt._tl.n_calls = 0
    x0 = torch.zeros(1, H, D)
    model.run_stage3_from(_stage2(), x0, 0.2, num_steps=10)        # the executed path (not in shadow)
    assert model.times == pytest.approx([1.0, 0.5], abs=1e-6)       # variant grid
    assert icpt._tl.n_calls == 1 and icpt._tl.n_steps == 2
    model.times.clear()
    icpt._tl.in_shadow = True
    model.run_stage3_from(_stage2(), x0, 0.2, num_steps=10)        # shadow bracket: exact resume
    icpt._tl.in_shadow = False
    assert model.times == pytest.approx([0.2, 0.1], abs=1e-6)
    assert icpt._tl.n_calls == 1                                     # shadow calls are not counted
    other = _TimeRecordingModel()
    other.config = model.config
    with pytest.raises(ValueError):
        P.Pi05DiagInterceptor(FakePolicy(other), timer=SystemTimer(enabled=False), diag=rec,
                              mode="warmreset", warm_variant="bogus")


def test_warm_family_arm_id_parsing():
    assert _envs.warm_t_of("warm_t0.2") == 0.2
    assert _envs.warm_t_of("warmreset_t0.2") == 0.2
    assert _envs.warm_t_of("warmshoot_t0.3") == 0.3
    assert _envs.warm_mode_of("warm_t0.2") == "warm"
    assert _envs.warm_mode_of("warmreset_t0.2") == "warmreset"
    assert _envs.warm_mode_of("warmshoot_t0.1") == "warmshoot"
    with pytest.raises(ValueError):
        _envs.warm_t_of("plain_k2")
    assert _envs.WARM_VARIANT_MODES == {"warmreset": "reset_t", "warmshoot": "overshoot", "resetfinal": "reset_final",
                                        "midfinal": "mid_final", "midfinal50": "mid_final50",
                                        "midreset": "mid_snap", "midreset50": "mid_snap50"}
    assert _envs.warm_t_of("resetfinal_t0.2") == 0.2 and _envs.warm_mode_of("resetfinal_t0.2") == "resetfinal"


def test_validate_arm_accepts_variant_modes_with_the_warm_yaml(tmp_path):
    import yaml

    cfg = {"checkpoints": {"cp1": {"judge": {"type": "always_warm_start", "start_t": 0.2}}}}
    path = tmp_path / "warm_t0.2.yaml"
    path.write_text(yaml.safe_dump(cfg))
    _envs.validate_arm("pi05_rc", "warm", "warm_t0.2", None, str(path))
    _envs.validate_arm("pi05_rc", "warmreset", "warmreset_t0.2", None, str(path))
    _envs.validate_arm("pi05_rc", "warmshoot", "warmshoot_t0.2", None, str(path))
    with pytest.raises(ValueError):  # arm id must carry the mode
        _envs.validate_arm("pi05_rc", "warmreset", "warm_t0.2", None, str(path))
    with pytest.raises(ValueError):  # start_t must match the yaml's judge
        _envs.validate_arm("pi05_rc", "warmreset", "warmreset_t0.3", None, str(path))
    with pytest.raises(ValueError):  # pi0.5 only
        _envs.validate_arm("groot_rc", "warmreset", "warmreset_t0.75", None, str(path))


def test_var500_round_is_a_superset_of_the_formal_identities():
    """The 500-episode round keeps the formal seed, so its first 50/100 identities are the Q-B ones."""
    from exp.step_diag import run_diag as RD

    assert set(_envs.VAR500_ARMS) == {"plain_k2", "warm_t0.2", "warmreset_t0.2", "warmshoot_t0.2"}
    assert set(_envs.VAR500_TASKS) <= set(_envs.qb_tasks("pi05"))
    assert _envs.VAR500_EPISODES == 500 > _envs.QB_FLAT_EPISODES

    def ids(exp, n, arm="warm_t0.2"):
        s = RD.StepDiagStrategy(arm_id=arm, experiment_id=exp, lane="main", teacher="pi05", layout=1, style=1,
                                base_seed=_envs.RC_FORMAL_BASE_SEED, replan_steps=5, tasks=[("CloseFridge", n)])
        return {tuple(sorted((k, v) for k, v in dict(ident).items() if k not in ("experiment_id", "task_uid")))
                for ident in s.expected_identities()}

    formal, big = ids("sdiag_v1", 50), ids(_envs.VAR500_EXPERIMENT_ID, 500)
    assert len(big) == 500 and len(formal) == 50 and formal <= big


def test_xseed_round_uses_the_historical_segment_and_ladder_m_matches_the_resume():
    from exp.step_diag import run_diag as RD
    from openpi.models_pytorch.pi0_pytorch import _warm_start_num_steps

    assert _envs.RC_XCHECK_BASE_SEED == 1_000_000 and _envs.XSEED_EPISODES == 50
    s = RD.StepDiagStrategy(arm_id="warmreset_t0.2", experiment_id=_envs.XSEED_EXPERIMENT_ID, lane="main", teacher="pi05",
                            layout=1, style=1, base_seed=_envs.RC_XCHECK_BASE_SEED, replan_steps=5, tasks=[("CloseFridge", 50)])
    seeds = sorted(i["env_seed"] for i in s.expected_identities())
    assert seeds == list(range(1_000_000, 1_000_050))
    for t in _envs.QB_WARM_TS["pi05"]:
        assert int(t * 10 + 0.5) == _warm_start_num_steps(t, 10)


def test_reset_final_starts_from_the_payload_final_chunk_with_the_reset_grid(tmp_path):
    from exp.step_diag import recorder as R
    from openpi.cache.timing import SystemTimer
    from tests.cache.test_interceptor import FakePolicy

    model = _TimeRecordingModel()
    model.config = SimpleNamespace(pytorch_compile_mode=None, action_horizon=H, action_dim=D)
    spec = R.DiagSpec(experiment_id="e", env_id="pi05_rc", arm_id="resetfinal_t0.2", mode="resetfinal", k_full=10,
                      k_set=(), warm_ts=(), n_primary=4, n_dense_extra=2, action_shape=(H, D), config_sha="c")
    rec = R.DiagRecorder(spec, tmp_path)
    icpt = P.Pi05DiagInterceptor(FakePolicy(model), timer=SystemTimer(enabled=False), diag=rec,
                                 mode="resetfinal", exec_steps=None, warm_variant="reset_final")
    final = torch.full((H, D), 7.0)
    icpt._orchestrator = SimpleNamespace(_storage=SimpleNamespace(fetch_payload=lambda eid: SimpleNamespace(action_chunk=final if eid == "e1" else None)))
    icpt._tl.n_steps = 0
    icpt._tl.n_calls = 0
    icpt._tl.cp1_result = SimpleNamespace(entry_id="e1")
    snapshot = torch.zeros(1, H, D)  # what the production path would hand in (x at start_t)
    out = model.run_stage3_from(_stage2(), snapshot, 0.2, num_steps=10)
    assert model.times == pytest.approx([1.0, 0.5], abs=1e-6)
    # started from the final chunk (7) not the snapshot (0): two steps of dt=-0.5 with v=1 -> 7 - 1
    assert torch.allclose(out.action_chunk, torch.full((1, H, D), 6.0))
    assert icpt._tl.n_steps == 2 and icpt._tl.n_calls == 1
    icpt._tl.cp1_result = None
    with pytest.raises(RuntimeError):
        model.run_stage3_from(_stage2(), snapshot, 0.2, num_steps=10)
    assert rec is not None


def _cell(rows):
    """rows: (task_uid, init_idx, success) on one task/lane; identity = init_idx."""
    return {"_outcomes": {u: {"task": "T", "init_idx": i, "env_seed": 2_000_000 + i, "lane": "main", "pin_id": None,
                              "layout": 1, "style": 1, "success": s} for u, i, s in rows}}


def test_replicated_identities_are_averaged_and_order_independent():
    from exp.step_diag.analysis import warm_variants as W
    first = [("r1_0", 0, 1), ("r1_1", 1, 1)]
    second = [("r2_0", 0, 0), ("r2_1", 1, 1)]
    ref = _cell([("f0", 0, 0), ("f1", 1, 1)])
    ab = W._paired(_cell(first + second), ref)
    ba = W._paired(_cell(second + first), ref)
    assert ab == ba == [(0.5, 0.0), (1.0, 1.0)]
    assert W._identity_sr(_cell(first + second)) == 0.75  # pooled 3/4, same as the identity mean when balanced


def test_degenerate_pairs_get_a_nonzero_interval():
    import numpy as np
    from exp.step_diag.analysis import warm_variants as W
    d = W._boot_delta([(1, 1)] * 25 + [(0, 0)] * 25, np.random.default_rng(0))
    assert d["degenerate"] and d["point"] == 0.0
    assert d["lower"] < 0 < d["upper"] and abs(d["upper"] - 1.959963984540054 ** 2 / (50 + 1.959963984540054 ** 2)) < 1e-12
    d2 = W._boot_delta([(1, 0)] * 5 + [(0, 0)] * 45, np.random.default_rng(0))
    assert not d2["degenerate"] and d2["lower"] < d2["point"] < d2["upper"]
