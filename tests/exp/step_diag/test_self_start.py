"""CPU tests for the self-start ablation arms (``self<variant>_t<t>``, owner 2026-09-24).

A self arm runs the same warm-reset continuation as its cache arm, but the start is the policy's own direct
full inference on the current observation (its snapshot at ``start_t`` or its final action) instead of the
retrieved cache entry. The tests pin: which tensor becomes the start, the (t, dt) grid of the continuation,
the executed-step / stage-3-call accounting (the direct run is bracketed out and reported as
``self_direct_nfe``), the private per-decision noise, the untouched global RNG, and the arm / driver tables.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from openpi.cache.types import groot_n15_schedule

from exp.step_diag import envs as _envs
from exp.step_diag import groot as G
from exp.step_diag import pi05 as P
from exp.step_diag import recorder as R

H, D = 50, 32


# -- pi0.5 ----------------------------------------------------------------------------------------------


class _Pi05Model:
    """Production-signature stubs: ``denoise_step`` returns v = 1 and records t; ``run_stage3`` walks K Euler
    steps through the INSTANCE ``denoise_step`` (so the interceptor's step counter sees them) and returns the
    snapshots at ``save_timesteps`` with the production convention (x before the step at that t)."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.config = SimpleNamespace(pytorch_compile_mode=None, action_horizon=H, action_dim=D)

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        self.times.append(float(timestep[0]))
        return torch.ones_like(x_t)

    def run_stage1(self, observation):
        return SimpleNamespace(state=torch.zeros(1, D), prefix_embs=torch.zeros(1, 4, 8))

    def run_stage2(self, stage1):
        return _stage2()

    def sample_noise(self, shape, device, generator=None):
        return torch.randn(*shape, generator=generator)

    def run_stage3(self, stage2, noise=None, num_steps=10, return_intermediates=False, save_timesteps=()):
        x = noise.clone()
        save_at = {round((1.0 - st) * num_steps): st for st in save_timesteps}
        inter = {}
        t = torch.tensor(1.0)
        dt = torch.tensor(-1.0 / num_steps)
        for step in range(num_steps):
            if step in save_at:
                inter[save_at[step]] = x.clone()
            x = x + dt * self.denoise_step(None, None, None, x, t.expand(1))
            t = t + dt
        return SimpleNamespace(action_chunk=x, intermediates=inter if return_intermediates else None)

    def run_stage3_from(self, stage2, start_x, start_t, *, num_steps=10):
        raise AssertionError("the self arm must route the executed call through the variant")


def _stage2():
    return SimpleNamespace(stage1=SimpleNamespace(state=torch.zeros(1, D), prefix_pad_masks=torch.ones(1, 4, dtype=torch.bool)),
                           past_key_values=None)


def _identity():
    return R.EpisodeIdentity(benchmark="robocasa365", task="CloseFridge", episode_id=0, task_uid="u:eval:1:3",
                             attempt=1, task_id=1, init_idx=3, env_seed=2_000_003)


def _pi05_interceptor(tmp_path, mode: str, variant: str):
    from openpi.cache.timing import SystemTimer
    from tests.cache.test_interceptor import FakePolicy

    model = _Pi05Model()
    spec = R.DiagSpec(experiment_id="sdiag_self13", env_id="pi05_rc", arm_id=f"{mode}_t0.2", mode=mode, k_full=10,
                      k_set=(), warm_ts=(), n_primary=4, n_dense_extra=28, action_shape=(H, D), config_sha="c")
    icpt = P.Pi05DiagInterceptor(FakePolicy(model), timer=SystemTimer(enabled=False), diag=R.DiagRecorder(spec, tmp_path),
                                 mode=mode, exec_steps=None, warm_variant=variant, self_start=True)
    icpt._tl.n_steps = 0
    icpt._tl.n_calls = 0
    icpt._tl.self_info = None
    return model, icpt, spec


def _expected_noise(spec, ident, decision_idx=0):
    seed = R.noise_seed(spec.experiment_id, spec.env_id, ident.task, (ident.env_seed, ident.init_idx, None),
                        ident.attempt, decision_idx, "self")
    return seed, R.make_noise(seed, (H, D))[None, ...]


@pytest.mark.parametrize("variant,start_shift,cont_t,cont_shift", [
    ("reset_t", -0.8, [1.0, 0.5], -1.0),        # snapshot at t=0.2 = noise - 8 * 0.1; two steps of -0.5
    ("reset_final", -1.0, [1.0, 0.5], -1.0),    # final action = noise - 10 * 0.1
    ("mid_final", -1.0, [0.9, 0.45], -0.9),     # final action fed at 0.9, dt = -0.45
    ("mid_snap50", -0.8, [0.5, 0.25], -0.5),    # snapshot fed at 0.5, dt = -0.25
])
def test_pi05_self_start_feeds_the_direct_run_not_the_cache(tmp_path, variant, start_shift, cont_t, cont_shift):
    mode = {"reset_t": "selfwarmreset", "reset_final": "selfresetfinal", "mid_final": "selfmidfinal",
            "mid_snap50": "selfmidreset50"}[variant]
    model, icpt, spec = _pi05_interceptor(tmp_path, mode, variant)
    ident = _identity()
    icpt._diag.begin_episode(ident)
    seed, noise = _expected_noise(spec, ident)
    rng_before = torch.get_rng_state()
    cached = torch.full((1, H, D), 99.0)  # the retrieved snapshot the production path hands in
    out = model.run_stage3_from(_stage2(), cached, 0.2, num_steps=10)
    # direct run: the full K=10 grid; then the variant's continuation grid
    assert model.times[:10] == pytest.approx([1.0 - 0.1 * i for i in range(10)], abs=1e-5)
    assert model.times[10:] == pytest.approx(cont_t, abs=1e-6)
    assert torch.allclose(out.action_chunk, noise + start_shift + cont_shift, atol=1e-5)
    # accounting: the executed continuation only; the direct run is reported separately
    assert icpt._tl.n_steps == len(cont_t) and icpt._tl.n_calls == 1
    assert icpt._tl.self_info == {"self_start": True, "self_seed": seed, "self_direct_nfe": 10}
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_pi05_self_start_seed_follows_the_decision_and_needs_an_episode(tmp_path):
    model, icpt, spec = _pi05_interceptor(tmp_path, "selfwarmreset", "reset_t")
    with pytest.raises(RuntimeError):
        model.run_stage3_from(_stage2(), torch.zeros(1, H, D), 0.2, num_steps=10)
    ident = _identity()
    icpt._diag.begin_episode(ident)
    s0 = icpt._diag.self_start_seed()
    icpt._diag.record(a_exec=torch.zeros(H, D), executed_steps=2, n_stage3_calls=1, hit_type="WARM_START",
                      start_t=0.2, schedule_id="pi05_v1")
    s1 = icpt._diag.self_start_seed()
    assert s0 == _expected_noise(spec, ident, 0)[0] and s1 == _expected_noise(spec, ident, 1)[0] and s0 != s1


def test_pi05_self_start_refuses_without_a_reset_variant(tmp_path):
    from openpi.cache.timing import SystemTimer
    from tests.cache.test_interceptor import FakePolicy

    spec = R.DiagSpec(experiment_id="e", env_id="pi05_rc", arm_id="x", mode="warm", k_full=10, k_set=(), warm_ts=(),
                      n_primary=4, n_dense_extra=28, action_shape=(H, D), config_sha="c")
    for variant in (None, "overshoot"):
        with pytest.raises(ValueError):
            P.Pi05DiagInterceptor(FakePolicy(_Pi05Model()), timer=SystemTimer(enabled=False),
                                  diag=R.DiagRecorder(spec, tmp_path), mode="warm", warm_variant=variant, self_start=True)


# -- GR00T ------------------------------------------------------------------------------------------------

B, HG, DG, BUCKETS = 1, 16, 32, 1000


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

    def run_stage3(self, *a, **kw):
        raise AssertionError("not used by a warm arm")

    def run_stage3_from(self, *a, **kw):
        raise AssertionError("replaced by the variant")


@contextmanager
def _noop():
    yield


def _gstage2():
    return SimpleNamespace(action_inputs={"embodiment_id": torch.zeros(B, dtype=torch.long), "state": torch.zeros(B, 8)})


def _recording_step(log):
    def step_fn(head, vl, state_features, embodiment_id, actions, timesteps_tensor, dt):
        log.append((int(timesteps_tensor[0]), float(dt)))
        return actions + 1.0
    return step_fn


@pytest.mark.parametrize("variant,start_t,direct_to_start,cont", [
    ("reset_t", 0.75, 3, [(0, 1.0)]),                                    # T=0.25 snapshot (input of step 3), t=1
    ("mid_snap", 0.75, 3, [(int(0.25 * BUCKETS), 0.75)]),                 # T=0.25 fed at pi0.5 t=0.75
    ("mid_snap50", 0.75, 3, [(int(0.5 * BUCKETS), 0.5)]),                 # T=0.25 fed at pi0.5 t=0.5
    ("reset_final", 0.75, 4, [(0, 1.0)]),                                 # T=0 final action, t=1
    ("mid_final50", 0.75, 4, [(int(0.5 * BUCKETS), 0.5)]),                # T=0 fed at 0.5
    ("reset_t", 0.5, 2, [(0, 0.5), (int(0.5 * BUCKETS), 0.5)]),           # T=0.5 snapshot, t = 1, 0.5
    ("mid_snap", 0.5, 2, [(int(0.25 * BUCKETS), 0.375), (int(0.625 * BUCKETS), 0.375)]),
    ("mid_final", 0.5, 4, [(int(0.25 * BUCKETS), 0.375), (int(0.625 * BUCKETS), 0.375)]),
])
def test_groot_self_start_feeds_the_direct_run_not_the_cache(variant, start_t, direct_to_start, cont):
    sched = groot_n15_schedule(4)
    runner, log = _Runner(), []
    orch = SimpleNamespace(check=lambda *a, **k: SimpleNamespace(payload=SimpleNamespace(action_chunk=torch.full((HG, DG), 99.0))))
    G.install_warm_variant(runner, orch, variant, sched, step_fn=_recording_step(log), self_start=True)
    G._SELF.seed, G._SELF.info = 12345, None
    rng_before = torch.get_rng_state()
    out = runner.run_stage3_from(_gstage2(), torch.full((HG, DG), 99.0), start_t, schedule=sched)
    noise = R.make_noise(12345, (HG, DG))[None, ...]
    # direct run: the native 4-step grid; then the variant's grid; the stub step adds 1
    assert log[:4] == [(0, 0.25), (250, 0.25), (500, 0.25), (750, 0.25)] and log[4:] == cont
    assert torch.allclose(out.action_pred, noise + direct_to_start + len(cont))
    assert out.steps_run == sched.remaining_steps(start_t) == len(cont)
    assert G._SELF.info == {"self_start": True, "self_seed": 12345, "self_direct_nfe": 4}
    assert torch.equal(torch.get_rng_state(), rng_before)
    G._SELF.seed = None
    with pytest.raises(RuntimeError):
        runner.run_stage3_from(_gstage2(), torch.zeros(HG, DG), start_t, schedule=sched)


def test_groot_evidence_wrapper_publishes_the_seed_and_records_the_direct_nfe(tmp_path):
    sched = groot_n15_schedule(4)
    runner = _Runner()
    G.install_warm_variant(runner, None, "reset_t", sched, step_fn=_recording_step([]), self_start=True)
    spec = R.DiagSpec(experiment_id="sdiag_self13", env_id="groot_rc", arm_id="selfwarmreset_t0.75", mode="selfwarmreset",
                      k_full=4, k_set=(), warm_ts=(), n_primary=4, n_dense_extra=28, action_shape=(HG, DG), config_sha="c")
    rec = R.DiagRecorder(spec, tmp_path)

    class _Inner:
        def get_action(self, obs):
            runner.run_stage3_from(_gstage2(), torch.zeros(HG, DG), 0.75, schedule=sched)
            return {"action": 0, "__hit_meta__": {"hit_type": "WARM_START", "start_t": 0.75}}

    pol = G.GrootEvidencePolicy(_Inner(), runner, rec, schedule_id=sched.schedule_id)
    ident = _identity()
    pol._diag.begin_episode(ident)
    want = pol._diag.self_start_seed()
    pol.get_action({})
    row = pol._diag._pending_rows[-1]
    assert row["hit_type"] == "WARM_START" and row["executed_steps"] == 1 and row["n_stage3_calls"] == 1
    assert row["self_start"] is True and row["self_seed"] == want and row["self_direct_nfe"] == 4


# -- arm tables -------------------------------------------------------------------------------------------


def test_self_arm_tables_and_validation(tmp_path):
    import yaml

    assert _envs.SELF_VARIANT_MODES == {"selfwarmreset": "reset_t", "selfresetfinal": "reset_final",
                                        "selfmidfinal": "mid_final", "selfmidfinal50": "mid_final50",
                                        "selfmidreset": "mid_snap", "selfmidreset50": "mid_snap50"}
    assert len(_envs.SELF13_ARMS_BY_POLICY["pi05"]) == 3 and len(_envs.SELF13_ARMS_BY_POLICY["groot"]) == 21
    for policy, arms in _envs.SELF13_ARMS_BY_POLICY.items():
        env_id = f"{policy}_rc"
        for arm in arms:
            if not arm.startswith("self"):
                continue  # the cache shoot arms run in this round (checked in test_groot_shoot)
            mode, t = _envs.warm_mode_of(arm), _envs.warm_t_of(arm)
            assert _envs.is_self_mode(mode) and t in _envs.QB_WARM_TS[policy]
            cache_arm = arm[len("self"):]  # the paired cache arm ran in the macro-13 round or runs in this one
            assert cache_arm in _envs.MACRO13_ARMS_BY_POLICY[policy] or cache_arm in arms
            path = tmp_path / f"{arm}.yaml"
            path.write_text(yaml.safe_dump({"checkpoints": {"cp1": {"judge": {"type": "always_warm_start", "start_t": t}}}}))
            _envs.validate_arm(env_id, mode, arm, None, str(path))
    # the exact resume has no self counterpart (a run's own snapshot on its own grid reproduces that run)
    assert "selfwarm" not in _envs.SELF_VARIANT_MODES and "selfwarmshoot" not in _envs.SELF_VARIANT_MODES
    # GR00T N=2 exact-resume equivalent (snapshot T=0.5 fed at t=0.5) is not run either
    assert "selfmidreset50_t0.5" not in _envs.SELF13_ARMS_BY_POLICY["groot"]


# -- analysis admission (G2 Round 1, B3) ------------------------------------------------------------------------


def test_robocasa_self_arm_admission_checks_the_self_start_evidence(tmp_path):
    """RoboCasa self rows (no initial-state pool) are admitted only with ``self_start``, the session's seed for that
    decision and ``self_direct_nfe = K`` (pi0.5 10); the total NFE per episode counts the direct inference."""
    import json

    from exp.step_diag.analysis import aggregate_arms as AG
    from tests.exp.step_diag.test_aggregate import _write_arm

    def write(root, edit=None):
        arms, srv = root / "arms", root / "srv"
        _write_arm(arms, srv, "selfwarmreset_t0.2", "warm", 2, {"CloseFridge": [1, 0]})
        path = srv / "pi05" / "selfwarmreset_t0.2" / "rows_x.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for r in rows:
            if r["status"] == "ok":
                seed = R.noise_seed("e", "pi05_rc", r["task"], (r["env_seed"], r["init_idx"], None), 1, r["decision_idx"], "self")
                r.update(self_start=True, self_seed=seed, self_direct_nfe=10)
                if edit:
                    edit(r)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return AG.cell_admission(AG.load_arm(arms / "pi05" / "selfwarmreset_t0.2"),
                                 AG.load_server_rows(srv / "pi05" / "selfwarmreset_t0.2"), "CloseFridge", kind="warm", m=2)

    ok = write(tmp_path / "ok")
    assert ok["equal_nfe"] and ok["self_start"] and ok["k_self"] == 10
    assert set(ok["episode_total_nfe"].values()) == {3 * 2 + 3 * 10}  # 3 decisions: continuation 6 + self 30
    bad = write(tmp_path / "bad", edit=lambda r: r.update(self_seed=r["self_seed"] ^ 1, self_direct_nfe=4))
    assert not bad["equal_nfe"] and bad["problems"] == {"self_seed_mismatch": 6, "self_direct_nfe_mismatch": 6}
