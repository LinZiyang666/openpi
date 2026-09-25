"""CPU tests for the GR00T shoot ablation (``warmshoot`` / ``midshoot`` / ``midshoot50`` arms, owner 2026-09-24).

No reset: the start (the cached or self-produced snapshot at the arm's ``start_t``) stays at its own GR00T time and
takes ``n = remaining_steps(start_t)`` steps of ``dt = SHOOT_ENTRY_T / n`` -- the warm-reset / midreset / midreset50
step size -- so the time runs past the clean end. The tests pin the (bucket, dt) sequences, the start tensor, the
arm tables and the validation rules.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from openpi.cache.types import groot_n15_schedule

from exp.step_diag import envs as _envs
from exp.step_diag import groot as G
from exp.step_diag import recorder as R

B, H, D, BUCKETS = 1, 16, 32, 1000


class _Head:
    num_timestep_buckets = BUCKETS

    def process_backbone_output(self, backbone_output):
        return SimpleNamespace(backbone_features=torch.zeros(B, 4, 8))

    def state_encoder(self, state, embodiment_id):
        return torch.zeros(B, 8)


class _Runner:
    def __init__(self):
        self._model = SimpleNamespace(action_head=_Head(), validate_data=lambda *a, **k: None)
        self._timer = SimpleNamespace(measure=lambda name: _noop())

    def _head_inputs(self, stage2):
        return {"backbone_features": torch.zeros(B, 4, 8)}

    def run_stage3_from(self, *a, **kw):
        raise AssertionError("replaced by the variant")


@contextmanager
def _noop():
    yield


def _stage2():
    return SimpleNamespace(action_inputs={"embodiment_id": torch.zeros(B, dtype=torch.long), "state": torch.zeros(B, 8)})


def _recording_step(log):
    def step_fn(head, vl, state_features, embodiment_id, actions, timesteps_tensor, dt):
        log.append((int(timesteps_tensor[0]), float(dt)))
        return actions + 1.0
    return step_fn


# (arm variant, arm start_t, expected (bucket, dt) sequence): GR00T time t_i = start_t + i * entry / n
SHOOT_GRID = [
    ("overshoot", 0.75, [(750, 1.0)]),                        # T=0.25, N=1, warmreset step (entry 1)
    ("mid_shoot", 0.75, [(750, 0.75)]),                       # T=0.25, N=1, midreset step (entry 0.75)
    ("mid_shoot50", 0.75, [(750, 0.5)]),                      # T=0.25, N=1, midreset50 step (entry 0.5)
    ("overshoot", 0.5, [(500, 0.5), (1000, 0.5)]),            # T=0.5, N=2: second query at the clean end
    ("mid_shoot", 0.5, [(500, 0.375), (875, 0.375)]),         # T=0.5, N=2
]


@pytest.mark.parametrize("variant,start_t,expected", SHOOT_GRID)
def test_groot_shoot_grid_starts_at_the_snapshot_time(variant, start_t, expected):
    sched = groot_n15_schedule(4)
    log = []
    start = torch.full((H, D), 3.0)
    out = G.groot_warm_variant_stage3(_Runner(), _stage2(), start, start_t, schedule=sched, variant=variant,
                                      step_fn=_recording_step(log))
    assert log == expected
    assert out.steps_run == sched.remaining_steps(start_t) == len(expected)
    assert torch.allclose(out.action_pred, torch.full((B, H, D), 3.0 + len(expected)))  # the start itself is kept


def test_mid_shoot50_at_half_is_the_exact_resume_grid():
    """T=0.5, N=2 with the midreset50 step (0.25) is the exact resume -- which is why that arm is not run."""
    from openpi.cache.groot import staged as S

    sched = groot_n15_schedule(4)
    shoot, resume = [], []
    G.groot_warm_variant_stage3(_Runner(), _stage2(), torch.zeros(H, D), 0.5, schedule=sched, variant="mid_shoot50",
                                step_fn=_recording_step(shoot))
    S.denoise_loop(_Head(), {}, _stage2().action_inputs, noise=torch.zeros(B, H, D), num_steps=4,
                   start_index=sched.snapshot_index(0.5), step_fn=_recording_step(resume))
    assert shoot == resume == [(500, 0.25), (750, 0.25)]


def test_shoot_loop_refuses_bad_parameters():
    for t0, dt, n in ((1.0, 0.5, 1), (0.5, 0.0, 1), (0.5, 0.5, 0), (-0.1, 0.5, 1)):
        with pytest.raises(ValueError):
            G.shoot_denoise_loop(_Head(), {}, _stage2().action_inputs, start=torch.zeros(B, H, D), t0=t0, dt=dt,
                                 num_steps=n, step_fn=_recording_step([]))


def test_cache_shoot_keeps_the_snapshot_and_self_shoot_takes_the_direct_run_snapshot():
    sched = groot_n15_schedule(4)
    # cache start: the snapshot the interceptor hands in is used as is (not the payload's final chunk)
    runner, log = _Runner(), []
    orch = SimpleNamespace(check=lambda *a, **k: SimpleNamespace(payload=SimpleNamespace(action_chunk=torch.full((H, D), 99.0))))
    G.install_warm_variant(runner, orch, "mid_shoot", sched, step_fn=_recording_step(log))
    out = runner.run_stage3_from(_stage2(), torch.full((H, D), 5.0), 0.75, schedule=sched)
    assert log == [(750, 0.75)] and torch.allclose(out.action_pred, torch.full((B, H, D), 6.0))
    # self start: the direct 4-step run's snapshot at native 0.75 (input of step 3), then the shoot step
    runner, log = _Runner(), []
    G.install_warm_variant(runner, orch, "overshoot", sched, step_fn=_recording_step(log), self_start=True)
    G._SELF.seed, G._SELF.info = 4321, None
    out = runner.run_stage3_from(_stage2(), torch.full((H, D), 99.0), 0.75, schedule=sched)
    noise = R.make_noise(4321, (H, D))[None, ...]
    assert log == [(0, 0.25), (250, 0.25), (500, 0.25), (750, 0.25), (750, 1.0)]
    assert torch.allclose(out.action_pred, noise + 3 + 1) and out.steps_run == 1
    assert G._SELF.info["self_direct_nfe"] == 4


def test_shoot_arm_tables_and_validation(tmp_path):
    import yaml

    shoot = [a for a in _envs.SELF13_ARMS_BY_POLICY["groot"] if "shoot" in a]
    assert sorted(shoot) == sorted(["warmshoot_t0.75", "midshoot_t0.75", "midshoot50_t0.75", "warmshoot_t0.5",
                                    "midshoot_t0.5", "selfwarmshoot_t0.75", "selfmidshoot_t0.75",
                                    "selfmidshoot50_t0.75", "selfwarmshoot_t0.5", "selfmidshoot_t0.5"])
    assert not any("shoot" in a for a in _envs.SELF13_ARMS_BY_POLICY["pi05"])
    for arm in shoot:
        mode, t = _envs.warm_mode_of(arm), _envs.warm_t_of(arm)
        assert mode in _envs.GROOT_SHOOT_MODES or mode in _envs.SELF_SHOOT_MODES
        assert _envs.is_self_mode(mode) == arm.startswith("self")
        path = tmp_path / f"{arm}.yaml"
        path.write_text(yaml.safe_dump({"checkpoints": {"cp1": {"judge": {"type": "always_warm_start", "start_t": t}}}}))
        _envs.validate_arm("groot_rc", mode, arm, None, str(path))
    path = tmp_path / "w.yaml"
    path.write_text(yaml.safe_dump({"checkpoints": {"cp1": {"judge": {"type": "always_warm_start", "start_t": 0.2}}}}))
    for mode in ("midshoot", "selfwarmshoot"):  # GR00T only
        with pytest.raises(ValueError):
            _envs.validate_arm("pi05_rc", mode, f"{mode}_t0.2", None, str(path))
    _envs.validate_arm("pi05_rc", "warmshoot", "warmshoot_t0.2", None, str(path))  # the pi0.5 arm that already ran
    from exp.step_diag import serve_diag_groot as SG

    for mode, variant in {**_envs.GROOT_SHOOT_MODES, **_envs.SELF_SHOOT_MODES}.items():
        assert SG.GROOT_VARIANT_MODES[mode] == variant
    assert _envs.SHOOT_ENTRY_T == {"overshoot": 1.0, "mid_shoot": 0.75, "mid_shoot50": 0.5}
