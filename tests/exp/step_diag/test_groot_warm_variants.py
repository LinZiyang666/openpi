"""CPU tests for the GR00T warm-start continuation variants (``warmreset`` / ``resetfinal`` arms, no overshoot).

GR00T's loop is ascending with K = 4 (``t = i/K``, ``dt = 1/K``). The exact resume runs steps
``snapshot_index(t)..K-1`` on the library grid; the variants run a fresh ``n = K - i`` step loop from
``t = 0`` with ``dt = 1/n``. The tests pin the (timestep bucket, dt) sequences through a recording
``step_fn``, the ``steps_run`` stamp the evidence policy reads, the payload substitution of
``reset_final`` and the arm / driver allow-lists.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from openpi.cache.types import groot_n15_schedule

from exp.step_diag import envs as _envs
from exp.step_diag import groot as G

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


@pytest.mark.parametrize("start_t,expected", [
    (0.75, [(0, 1.0)]),                       # 1 remaining step: a fresh 1-step loop, t=0, dt=1
    (0.5, [(0, 0.5), (int(0.5 * BUCKETS), 0.5)]),  # 2 remaining steps: t = 0, 1/2 with dt = 1/2
])
def test_groot_reset_variant_runs_a_fresh_short_loop(start_t, expected):
    sched = groot_n15_schedule(4)
    log = []
    out = G.groot_warm_variant_stage3(_Runner(), _stage2(), torch.zeros(H, D), start_t, schedule=sched,
                                      variant="reset_t", step_fn=_recording_step(log))
    assert log == expected
    assert out.steps_run == len(expected) == sched.remaining_steps(start_t) and out.start_t == start_t
    assert out.action_pred.shape == (B, H, D) and torch.allclose(out.action_pred, torch.full((B, H, D), float(len(expected))))


def test_groot_exact_resume_grid_differs_from_the_variant():
    """The production resume at t=0.5 visits buckets for t=0.5, 0.75 with dt=1/4; the variant visits 0, 0.5 with dt=1/2."""
    from openpi.cache.groot import staged as S

    sched = groot_n15_schedule(4)
    log = []
    S.denoise_loop(_Head(), {}, _stage2().action_inputs, noise=torch.zeros(B, H, D), num_steps=4,
                   start_index=sched.snapshot_index(0.5), step_fn=_recording_step(log))
    assert log == [(int(0.5 * BUCKETS), 0.25), (int(0.75 * BUCKETS), 0.25)]


def test_groot_reset_final_takes_the_payload_chunk_and_refuses_without_it():
    sched = groot_n15_schedule(4)
    runner = _Runner()
    chunk = torch.full((H, D), 7.0)
    orch = SimpleNamespace(check=lambda *a, **k: SimpleNamespace(payload=SimpleNamespace(action_chunk=chunk)))
    G.install_warm_variant(runner, orch, "reset_final", sched, step_fn=_recording_step([]))
    orch.check("cp1")  # the stashing wrapper records the hit
    out = runner.run_stage3_from(_stage2(), torch.zeros(H, D), 0.75, schedule=sched)
    # the loop ran from the chunk (7), not the snapshot (0); the stub step adds 1 per step
    assert torch.allclose(out.action_pred, torch.full((B, H, D), 8.0)) and out.steps_run == 1
    runner2 = _Runner()
    G.install_warm_variant(runner2, SimpleNamespace(check=lambda *a, **k: SimpleNamespace(payload=None)), "reset_final", sched,
                           step_fn=_recording_step([]))
    with pytest.raises(RuntimeError):
        runner2.run_stage3_from(_stage2(), torch.zeros(H, D), 0.75, schedule=sched)
    with pytest.raises(ValueError):
        G.install_warm_variant(_Runner(), None, "bogus", sched)


def test_groot_variant_arm_validation_and_allow_lists(tmp_path):
    import yaml

    cfg = {"checkpoints": {"cp1": {"judge": {"type": "always_warm_start", "start_t": 0.75}}}}
    path = tmp_path / "warm_t0.75.yaml"
    path.write_text(yaml.safe_dump(cfg))
    _envs.validate_arm("groot_rc", "warmreset", "warmreset_t0.75", None, str(path))
    _envs.validate_arm("groot_rc", "resetfinal", "resetfinal_t0.75", None, str(path))
    _envs.validate_arm("groot_rc", "warmshoot", "warmshoot_t0.75", None, str(path))  # shoot ablation (2026-09-24)
    with pytest.raises(ValueError):  # midshoot is GR00T-only
        _envs.validate_arm("pi05_rc", "midshoot", "midshoot_t0.75", None, str(path))
    assert "resetfinal_t0.75" in _envs.MACRO13_ARMS_BY_POLICY["groot"] and "warmshoot_t0.75" not in _envs.MACRO13_ARMS_BY_POLICY["groot"]
    assert _envs.XSEED_ARMS_BY_POLICY["pi05"] == _envs.VAR500_ARMS
    env = _envs.ENVS["groot_rc"]
    assert env.schedule.remaining_steps(0.75) == 1 and env.schedule.remaining_steps(0.5) == 2
    assert _envs.ENVS["pi05_rc"].schedule.remaining_steps(0.2) == 2


@pytest.mark.parametrize("start_t,expected", [
    (0.75, [(int(0.25 * BUCKETS), 0.75)]),                                # 1 step: GR00T t 0.25 -> 1 (pi0.5 0.75 -> 0)
    (0.5, [(int(0.25 * BUCKETS), 0.375), (int(0.625 * BUCKETS), 0.375)]),  # 2 steps of 0.375
])
def test_groot_mid_final_enters_one_grid_step_below_noise(start_t, expected):
    sched = groot_n15_schedule(4)
    log = []
    out = G.groot_warm_variant_stage3(_Runner(), _stage2(), torch.zeros(H, D), start_t, schedule=sched,
                                      variant="mid_final", step_fn=_recording_step(log))
    assert log == [(b, pytest.approx(dt)) for b, dt in expected]
    assert out.steps_run == len(expected) == sched.remaining_steps(start_t)
    assert _envs.MID_ENTRY_T == {"pi05": 0.9, "groot": 0.75}


@pytest.mark.parametrize("n", [1, 2, 4])
def test_mid_denoise_loop_at_entry_zero_is_upstreams_loop(n):
    from openpi.cache.groot import staged as S

    a, b = [], []
    x = torch.randn(B, H, D)
    ya = G.mid_denoise_loop(_Head(), {}, _stage2().action_inputs, start=x, entry_t=0.0, num_steps=n, step_fn=_recording_step(a))
    yb = S.denoise_loop(_Head(), {}, _stage2().action_inputs, noise=x, num_steps=n, step_fn=_recording_step(b))
    assert a == b and torch.equal(ya, yb)


def test_groot_mid_final_takes_the_payload_chunk_and_arm_is_allowed(tmp_path):
    import yaml

    sched = groot_n15_schedule(4)
    runner = _Runner()
    chunk = torch.full((H, D), 7.0)
    orch = SimpleNamespace(check=lambda *a, **k: SimpleNamespace(payload=SimpleNamespace(action_chunk=chunk)))
    G.install_warm_variant(runner, orch, "mid_final", sched, step_fn=_recording_step([]))
    orch.check("cp1")
    out = runner.run_stage3_from(_stage2(), torch.zeros(H, D), 0.75, schedule=sched)
    assert torch.allclose(out.action_pred, torch.full((B, H, D), 8.0)) and out.steps_run == 1
    cfg = {"checkpoints": {"cp1": {"judge": {"type": "always_warm_start", "start_t": 0.5}}}}
    path = tmp_path / "warm_t0.5.yaml"
    path.write_text(yaml.safe_dump(cfg))
    _envs.validate_arm("groot_rc", "midfinal", "midfinal_t0.5", None, str(path))
    assert _envs.warm_mode_of("midfinal_t0.5") == "midfinal" and _envs.warm_t_of("midfinal_t0.5") == 0.5


@pytest.mark.parametrize("start_t,expected", [
    (0.75, [(int(0.5 * BUCKETS), 0.5)]),                                 # entry 0.5, 1 step
    (0.5, [(int(0.5 * BUCKETS), 0.25), (int(0.75 * BUCKETS), 0.25)]),    # entry 0.5, 2 steps
])
def test_groot_mid_final50_enters_at_half(start_t, expected):
    sched = groot_n15_schedule(4)
    log = []
    out = G.groot_warm_variant_stage3(_Runner(), _stage2(), torch.zeros(H, D), start_t, schedule=sched,
                                      variant="mid_final50", step_fn=_recording_step(log))
    assert log == [(b, pytest.approx(dt)) for b, dt in expected] and out.steps_run == len(expected)
    assert "midfinal50_t0.75" in _envs.MACRO13_ARMS_BY_POLICY["groot"] and _envs.warm_mode_of("midfinal50_t0.5") == "midfinal50"


def test_groot_mid_snap_keeps_the_snapshot_and_enters_at_075():
    """midreset: the snapshot handed in is NOT replaced by the payload chunk; entry = midfinal's (GR00T t 0.25)."""
    sched = groot_n15_schedule(4)
    runner = _Runner()
    chunk = torch.full((H, D), 7.0)
    orch = SimpleNamespace(check=lambda *a, **k: SimpleNamespace(payload=SimpleNamespace(action_chunk=chunk)))
    log = []
    G.install_warm_variant(runner, orch, "mid_snap", sched, step_fn=_recording_step(log))
    orch.check("cp1")
    out = runner.run_stage3_from(_stage2(), torch.full((H, D), 3.0), 0.5, schedule=sched)
    assert torch.allclose(out.action_pred, torch.full((B, H, D), 5.0)) and out.steps_run == 2
    assert log == [(int(0.25 * BUCKETS), pytest.approx(0.375)), (int(0.625 * BUCKETS), pytest.approx(0.375))]
    assert "midreset_t0.5" in _envs.MACRO13_ARMS_BY_POLICY["groot"]


@pytest.mark.parametrize("start_t,expected", [
    (0.75, [(int(0.5 * BUCKETS), 0.5)]),
    (0.5, [(int(0.5 * BUCKETS), 0.25), (int(0.75 * BUCKETS), 0.25)]),
])
def test_groot_mid_snap50_keeps_snapshot_and_enters_at_half(start_t, expected):
    sched = groot_n15_schedule(4)
    runner = _Runner()
    orch = SimpleNamespace(check=lambda *a, **k: SimpleNamespace(payload=SimpleNamespace(action_chunk=torch.full((H, D), 7.0))))
    log = []
    G.install_warm_variant(runner, orch, "mid_snap50", sched, step_fn=_recording_step(log))
    orch.check("cp1")
    out = runner.run_stage3_from(_stage2(), torch.full((H, D), 3.0), start_t, schedule=sched)
    assert log == [(b, pytest.approx(dt)) for b, dt in expected]
    assert torch.allclose(out.action_pred, torch.full((B, H, D), 3.0 + len(expected)))  # snapshot kept, not the payload
    assert "midreset50_t0.5" in _envs.MACRO13_ARMS_BY_POLICY["groot"]
