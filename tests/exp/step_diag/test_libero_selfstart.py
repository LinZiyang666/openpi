"""CPU tests for the LIBERO warm-reset / self-start round (``sdiag_libero_self``, logs/step_diag_libero_selfstart_plan.log.md).

Pins: every GR00T K=8 arm of plan §3.2 as (start, t sequence, dt, N) through a recording ``step_fn`` -- and against
RoboCasa's K=4 arm of the same (T, N, t) --, the exact resume of the two ``warm_t`` arms on the native grid, the
default step count of the variants being the remaining-step count (RoboCasa unchanged), the arm tables and
``validate_arm``, the LIBERO warm yamls (shadow with only the judge replaced), the GR00T LIBERO server argv and
per-connection composition, the driver's validation / expected identities / layout, and the LIBERO analysis
(admission with the K=8 schedule, self vs cache / full / plain pairing, success lengths net of settle steps).
G2 Round 1 revisions: pairing on the initial-state pool and env seed, the formal 10 x 50 status against runs in
flight and smoke data, per-decision self-start evidence, continuation / self-start / total NFE, and success lengths
taken only from accepted, consistent attempts (RoboCasa and LIBERO).
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from openpi.cache.types import groot_n15_schedule

from exp.step_diag import emit_arms as EM
from exp.step_diag import envs as E
from exp.step_diag import groot as G
from exp.step_diag import recorder as R
from exp.step_diag import run_libero_diag as RL
from exp.step_diag import serve_diag_groot as SG
from exp.step_diag.analysis import aggregate_arms as AG
from exp.step_diag.analysis import success_length as SL
from exp.step_diag.analysis import warm_variants as WV

B, H, D, BUCKETS = 1, 16, 32, 1000
K8 = groot_n15_schedule(8)
K4 = groot_n15_schedule(4)
GROOT_VARIANT_ARMS = [a for a in E.LIBERO_SELF_ARMS_BY_POLICY["groot"] if E.warm_steps_of(a) is not None]

# plan §3.2: (T, pi0.5-notation t sequence) of every GR00T warm-reset row, by base mode and N
PLAN_T_SEQ = {
    ("warmreset", 1): [1.0], ("midreset", 1): [0.75], ("midreset50", 1): [0.5],
    ("warmreset", 2): [1.0, 0.5], ("midreset", 2): [0.75, 0.375], ("midreset50", 2): [0.5, 0.25],
}
PLAN_T_SEQ.update({("resetfinal", n): PLAN_T_SEQ[("warmreset", n)] for n in (1, 2)})
PLAN_T_SEQ.update({("midfinal", n): PLAN_T_SEQ[("midreset", n)] for n in (1, 2)})
PLAN_T_SEQ.update({("midfinal50", n): PLAN_T_SEQ[("midreset50", n)] for n in (1, 2)})
PLAN_SNAPSHOT_T = {1: 0.25, 2: 0.5}  # pi0.5-notation T of the snapshot rows; final-action rows have T = 0


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


def _stage2():
    return SimpleNamespace(action_inputs={"embodiment_id": torch.zeros(B, dtype=torch.long), "state": torch.zeros(B, 8)})


def _recording_step(log):
    def step_fn(head, vl, state_features, embodiment_id, actions, timesteps_tensor, dt):
        log.append((int(timesteps_tensor[0]), float(dt)))
        return actions + 1.0
    return step_fn


def _base_mode(mode: str) -> str:
    return mode[len("self"):] if mode in E.SELF_VARIANT_MODES else mode


# -- GR00T K=8: every row of plan §3.2 ----------------------------------------------------------------------


@pytest.mark.parametrize("arm", GROOT_VARIANT_ARMS)
def test_groot_k8_arm_rows_match_the_plan(arm):
    mode, start_t, n = E.warm_mode_of(arm), E.warm_t_of(arm), E.warm_steps_of(arm)
    self_start = mode in E.SELF_VARIANT_MODES
    base = _base_mode(mode)
    variant = SG.GROOT_VARIANT_MODES[mode]
    final = variant in G.GROOT_FINAL_START_VARIANTS
    assert start_t == E.GROOT_LIBERO_START_T_BY_N[n] and 1.0 - start_t == PLAN_SNAPSHOT_T[n]
    runner, log = _Runner(), []
    orch = SimpleNamespace(check=lambda *a, **k: SimpleNamespace(payload=SimpleNamespace(action_chunk=torch.full((H, D), 99.0))))
    G.install_warm_variant(runner, orch, variant, K8, step_fn=_recording_step(log), self_start=self_start, num_steps=n)
    orch.check("cp1")
    G._SELF.seed, G._SELF.info = 12345, None
    out = runner.run_stage3_from(_stage2(), torch.full((H, D), 3.0), start_t, schedule=K8)
    t_seq = PLAN_T_SEQ[(base, n)]
    want = [(round((1.0 - t) * BUCKETS), pytest.approx(t_seq[0] / n)) for t in t_seq]  # native t = 1 - t, dt = t0/N
    if self_start:
        # the direct run: the native K=8 grid from private noise; its snapshot at start_t = the input of step 6 / 4
        assert log[:8] == [(125 * i, 0.125) for i in range(8)] and log[8:] == want
        noise = R.make_noise(12345, (H, D))[None, ...]
        start = noise + (8 if final else K8.snapshot_index(start_t))
        assert K8.snapshot_index(start_t) / 8 == pytest.approx(start_t)
        assert G._SELF.info == {"self_start": True, "self_seed": 12345, "self_direct_nfe": 8}
    else:
        assert log == want
        start = torch.full((B, H, D), 99.0 if final else 3.0)  # payload final chunk / retrieved snapshot
    assert torch.allclose(out.action_pred, start + n)
    assert out.steps_run == n == E.executed_steps_of("groot_libero_10", arm) and out.start_t == start_t
    G._SELF.seed = None


@pytest.mark.parametrize("arm", [a for a in GROOT_VARIANT_ARMS if not a.startswith("self")])
def test_groot_k8_cache_arm_equals_the_robocasa_k4_arm(arm):
    """Same (T, N, t): the K=8 arm with explicit N runs exactly the continuation RoboCasa's K=4 arm runs."""
    rc_arm, n = E.split_warm_steps(arm)
    assert rc_arm in E.MACRO13_ARMS_BY_POLICY["groot"] and E.ENVS["groot_rc"].remaining_steps(E.warm_t_of(rc_arm)) == n
    variant = SG.GROOT_VARIANT_MODES[E.warm_mode_of(arm)]
    x = torch.randn(B, H, D)
    log8, log4 = [], []
    y8 = G.groot_warm_variant_stage3(_Runner(), _stage2(), x, E.warm_t_of(arm), schedule=K8, variant=variant,
                                     step_fn=_recording_step(log8), num_steps=n)
    y4 = G.groot_warm_variant_stage3(_Runner(), _stage2(), x, E.warm_t_of(rc_arm), schedule=K4, variant=variant,
                                     step_fn=_recording_step(log4))
    assert log8 == log4 and torch.equal(y8.action_pred, y4.action_pred) and y8.steps_run == y4.steps_run == n
    # and the snapshot it starts from sits at the same native time on both grids
    assert K8.snapshot_t(K8.snapshot_index(E.warm_t_of(arm))) == K4.snapshot_t(K4.snapshot_index(E.warm_t_of(rc_arm)))


@pytest.mark.parametrize("arm,expected", [
    ("warm_t0.875", [(875, 0.125)]),                 # ours, T = 0.125, N = 1, t = 0.125
    ("warm_t0.75", [(750, 0.125), (875, 0.125)]),    # ours, T = 0.25, N = 2, t = 0.25, 0.125
])
def test_groot_k8_exact_resume_stays_on_the_native_grid(arm, expected):
    from openpi.cache.groot import staged as S

    t = E.warm_t_of(arm)
    log = []
    S.denoise_loop(_Head(), {}, _stage2().action_inputs, noise=torch.zeros(B, H, D), num_steps=8,
                   start_index=K8.snapshot_index(t), step_fn=_recording_step(log))
    assert log == expected and E.warm_steps_of(arm) is None
    assert E.executed_steps_of("groot_libero_spatial", arm) == K8.remaining_steps(t) == len(expected)


@pytest.mark.parametrize("sched", [K4, K8])
@pytest.mark.parametrize("variant", G.GROOT_WARM_VARIANTS)
def test_default_step_count_is_the_remaining_steps_bit_for_bit(sched, variant):
    """No ``num_steps`` (every RoboCasa arm): the continuation is the one before the decoupling, n = remaining."""
    def step(head, vl, sf, emb, actions, tt, dt):
        return actions * 0.9 + float(tt[0]) / 997.0 + dt

    for start_t in sorted(sched.timestep_set):
        x = torch.randn(B, H, D)
        base = G.groot_warm_variant_stage3(_Runner(), _stage2(), x, start_t, schedule=sched, variant=variant, step_fn=step)
        none = G.groot_warm_variant_stage3(_Runner(), _stage2(), x, start_t, schedule=sched, variant=variant, step_fn=step,
                                           num_steps=None)
        expl = G.groot_warm_variant_stage3(_Runner(), _stage2(), x, start_t, schedule=sched, variant=variant, step_fn=step,
                                           num_steps=sched.remaining_steps(start_t))
        assert torch.equal(base.action_pred, none.action_pred) and torch.equal(base.action_pred, expl.action_pred)
        assert base.steps_run == none.steps_run == expl.steps_run == sched.remaining_steps(start_t)
    runners = []
    for kw in ({}, {"num_steps": None}):
        r, log = _Runner(), []
        orch = SimpleNamespace(check=lambda *a, **k: SimpleNamespace(payload=SimpleNamespace(action_chunk=torch.ones(H, D))))
        G.install_warm_variant(r, orch, variant, sched, step_fn=_recording_step(log), **kw)
        orch.check("cp1")
        out = r.run_stage3_from(_stage2(), torch.zeros(H, D), sorted(sched.timestep_set)[0], schedule=sched)
        runners.append((log, out.action_pred, out.steps_run))
    assert runners[0][0] == runners[1][0] and torch.equal(runners[0][1], runners[1][1])


# -- arm tables and validation ------------------------------------------------------------------------------


def _judge_yaml(tmp_path, t):
    path = tmp_path / f"warm_t{t:g}.yaml"
    path.write_text(yaml.safe_dump({"checkpoints": {"cp1": {"judge": {"type": "always_warm_start", "start_t": t}}}}))
    return str(path)


def test_libero_arm_tables():
    pi05 = E.LIBERO_SELF_ARMS_BY_POLICY["pi05"]
    assert len(pi05) == 9 and set(pi05) == {"full", "plain_k2", "warm_t0.2", "warmreset_t0.2", "resetfinal_t0.2",
                                            "midfinal_t0.2", *E.SELF13_ARMS_BY_POLICY["pi05"]}
    groot = E.LIBERO_SELF_ARMS_BY_POLICY["groot"]
    cache = [a for a in GROOT_VARIANT_ARMS if not a.startswith("self")]
    selfs = [a for a in GROOT_VARIANT_ARMS if a.startswith("self")]
    assert len(groot) == len(set(groot)) == 29 and len(cache) == len(selfs) == 12
    assert groot[:5] == ("full", "plain_k1", "plain_k2", "warm_t0.875", "warm_t0.75")
    assert set(selfs) == {f"self{a}" for a in cache}
    assert {(E.warm_mode_of(a), E.warm_t_of(a), E.warm_steps_of(a)) for a in cache} == {
        (mode, t, n) for n, t in ((1, 0.75), (2, 0.5)) for mode in E.GROOT_LIBERO_RESET_MODES}
    assert E.LIBERO_SELF_SUITES == ("libero_spatial", "libero_10") and E.LIBERO_SELF_EPISODES * E.LIBERO_N_TASKS == 500
    # parsing: RoboCasa ids carry no step count and keep their meaning
    assert E.split_warm_steps("warm_t0.2") == ("warm_t0.2", None) and E.warm_steps_of("midreset50_t0.5") is None
    assert E.split_warm_steps("selfmidfinal50_t0.5_n2") == ("selfmidfinal50_t0.5", 2)
    assert E.warm_t_of("selfmidfinal50_t0.5_n2") == 0.5 and E.warm_mode_of("selfmidfinal50_t0.5_n2") == "selfmidfinal50"
    assert E.executed_steps_of("pi05_libero_10", "selfmidfinal_t0.2") == 2 and E.executed_steps_of("groot_libero_10", "full") == 8
    assert E.executed_steps_of("groot_rc", "warmreset_t0.5") == 2 and E.executed_steps_of("pi05_rc", "plain_k3") == 3


@pytest.mark.parametrize("env_id", ["pi05_libero_spatial", "pi05_libero_10", "groot_libero_spatial", "groot_libero_10"])
def test_every_libero_arm_validates(tmp_path, env_id):
    env = E.ENVS[env_id]
    for arm in E.LIBERO_SELF_ARMS_BY_POLICY[env.policy]:
        if arm == "full":
            E.validate_arm(env_id, "full", arm, env.k_full, None)
        elif arm.startswith("plain_k"):
            E.validate_arm(env_id, "plain", arm, int(arm[len("plain_k"):]), None)
        else:
            E.validate_arm(env_id, E.warm_mode_of(arm), arm, None, _judge_yaml(tmp_path, E.warm_t_of(arm)))


@pytest.mark.parametrize("env_id,mode,arm,yaml_t", [
    ("groot_libero_10", "warmreset", "warmreset_t0.75", 0.75),         # a GR00T LIBERO variant must name N
    ("groot_libero_10", "warm", "warm_t0.75_n2", 0.75),                # the exact resume never names N
    ("groot_libero_10", "midreset", "midreset_t0.75_n0", 0.75),        # N >= 1
    ("groot_libero_10", "midreset", "midreset_t0.75_n9", 0.75),        # N <= K
    ("groot_libero_10", "warmreset", "warmreset_t0.25_n1", 0.25),      # start_t outside warm_ts
    ("groot_libero_10", "midreset", "midreset_t0.75_n1", 0.5),         # judge start_t differs from the arm
    ("groot_libero_10", "warmshoot", "warmshoot_t0.75_n1", 0.75),      # no overshoot for GR00T
    ("pi05_libero_10", "warmreset", "warmreset_t0.2_n2", 0.2),         # pi0.5 runs remaining_steps(start_t)
    ("groot_rc", "warmreset", "warmreset_t0.75_n1", 0.75),             # RoboCasa arm ids are unchanged
    ("pi05_rc", "selfwarmreset", "selfwarmreset_t0.2_n2", 0.2),
])
def test_validate_arm_rejections(tmp_path, env_id, mode, arm, yaml_t):
    with pytest.raises(ValueError):
        E.validate_arm(env_id, mode, arm, None, _judge_yaml(tmp_path, yaml_t))


# -- yamls ----------------------------------------------------------------------------------------------------


def _without_judge(cfg):
    out = copy.deepcopy(cfg)
    out["checkpoints"]["cp1"]["judge"] = None
    return out


@pytest.mark.parametrize("env_id", ["pi05_libero_spatial", "pi05_libero_10", "groot_libero_spatial", "groot_libero_10"])
def test_libero_warm_yaml_is_the_shadow_with_only_the_judge_replaced(tmp_path, env_id):
    env = E.ENVS[env_id]
    index = EM.emit(tmp_path, (env_id,))[env_id]
    shadow = yaml.safe_load((tmp_path / index["cells"]["shadow"]["file"]).read_text())
    for t in env.warm_ts:
        cell = index["cells"][f"warm_t{t:g}"]
        text = (tmp_path / cell["file"]).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == cell["sha256"]
        warm = yaml.safe_load(text)
        assert warm["checkpoints"]["cp1"]["judge"] == {"type": "always_warm_start", "start_t": t}
        assert _without_judge(warm) == _without_judge(shadow)
    # every arm of the round names an emitted cell (warm family) or none (plain / full)
    for arm, spec in index["libero_self"]["arms"].items():
        assert spec["exec_steps"] == E.executed_steps_of(env_id, arm)
        assert (spec["cell"] is None) == (arm == "full" or arm.startswith("plain_k"))
        assert spec["cell"] is None or spec["cell"] in index["cells"]
    base, _ = EM.load_base(env_id)
    off_set = 0.25 if env.policy == "groot" else 0.5  # a grid timestep the library check does not require
    with pytest.raises(ValueError, match="warm_ts"):
        EM.build_warm_cell(env_id, base, off_set)


# -- GR00T LIBERO server --------------------------------------------------------------------------------------

LCOMMON = ["--experiment-id", E.LIBERO_SELF_EXPERIMENT_ID, "--diag-out", "/tmp/o"]


def test_groot_libero_parse_every_mode():
    args, rest = SG.parse(["--benchmark", "libero", "--mode", "plain", "--exec-steps", "1", "--env-id", "groot_libero_10",
                           "--arm-id", "plain_k1", *LCOMMON, "--", "--port", "1"])
    assert rest == ["--port", "1", "--denoising-steps", "1", "--concurrent"]
    args, rest = SG.parse(["--benchmark", "libero", "--mode", "full", "--env-id", "groot_libero_10", "--arm-id", "full",
                           *LCOMMON, "--", "--concurrent", "--denoising-steps=8"])
    assert args.exec_steps == 8 and rest == ["--concurrent", "--denoising-steps=8"]
    args, rest = SG.parse(["--benchmark", "libero", "--mode", "selfmidreset", "--env-id", "groot_libero_spatial",
                           "--arm-id", "selfmidreset_t0.75_n1", "--cache-config", "w.yaml", *LCOMMON])
    assert rest[:5] == ["--cache-config", "w.yaml", "--denoising-steps", "8", "--concurrent"]
    assert rest[5] == "--rit-shadow-out" and rest[7:] == ["--rit-warm-ts", "0.875,0.75,0.5"]
    for argv in (["--mode", "full", "--arm-id", "full", "--", "--denoising-steps", "4"],       # K contradicts the arm
                 ["--mode", "warm", "--arm-id", "warm_t0.75", "--cache-config", "w", "--", "--denoising-steps", "4"],
                 ["--mode", "midreset", "--arm-id", "midreset_t0.75_n1"]):                   # no yaml
        own = argv[:argv.index("--")] if "--" in argv else argv
        tail = argv[argv.index("--"):] if "--" in argv else []
        with pytest.raises(SystemExit):
            SG.parse(["--benchmark", "libero", "--env-id", "groot_libero_10", *own, *LCOMMON, *tail])


class _FakeCapturedRunner:
    def __init__(self, k):
        self._model = SimpleNamespace(action_head=SimpleNamespace(num_inference_timesteps=k))
        self._timer = None

    def live_schedule(self):
        return groot_n15_schedule(self._model.action_head.num_inference_timesteps)

    def run_stage3(self, *a, **kw):
        raise AssertionError("not called here")

    def run_stage3_from(self, *a, **kw):
        raise AssertionError("not called here")


def _fake_policy(k):
    return SimpleNamespace(model=SimpleNamespace(action_head=SimpleNamespace(
        num_inference_timesteps=k, config=SimpleNamespace(action_horizon=16, action_dim=32))))


@pytest.mark.parametrize("mode,arm,k", [
    ("plain", "plain_k1", 1), ("full", "full", 8), ("shadow", "shadow", 8), ("warm", "warm_t0.875", 8),
    ("midreset50", "midreset50_t0.5_n2", 8), ("selfresetfinal", "selfresetfinal_t0.75_n1", 8),
])
def test_groot_libero_connection_factory_composes_one_served_object_per_connection(tmp_path, monkeypatch, mode, arm, k):
    from exp.libero_groot import serve_groot_libero as srv
    import openpi.cache.groot.interceptor as gi

    monkeypatch.setattr(srv, "_build_shadow_factory", srv._build_shadow_factory)
    monkeypatch.setattr(srv, "_build_concurrent_factory", srv._build_concurrent_factory)
    runners, installs = [], []

    def parts(srv_args, config, shared_storage, model):
        runners.append(_FakeCapturedRunner(k))
        return (None if config is None else SimpleNamespace(check=lambda *a, **kw: None)), runners[-1], None

    monkeypatch.setattr(SG, "_libero_connection_parts", parts)
    monkeypatch.setattr(G, "install_warm_variant", lambda *a, **kw: installs.append((a[2], kw)))
    monkeypatch.setattr(gi, "GrootCacheInterceptor", lambda policy, runner, **kw: SimpleNamespace(runner=runner, **kw))
    cache = [] if mode in ("plain", "full") else ["--cache-config", "w.yaml"]
    steps = ["--exec-steps", "1"] if mode == "plain" else []
    args, _ = SG.parse(["--benchmark", "libero", "--mode", mode, "--env-id", "groot_libero_10", "--arm-id", arm,
                        *steps, *cache, *LCOMMON])
    rec = R.DiagRecorder(R.DiagSpec(experiment_id="e", env_id="groot_libero_10", arm_id=arm, mode=mode, k_full=8,
                                    action_shape=(16, 32)), tmp_path)
    SG.install_libero(args, rec)
    if mode in ("plain", "full"):
        factory, label = srv._build_concurrent_factory(_fake_policy(k), SimpleNamespace())
    else:
        factory, label = srv._build_shadow_factory(SimpleNamespace(), object(), None, threading.Lock())
    served = [factory(_fake_policy(k)) for _ in range(2)]
    inner = [s._inner._policy for s in served]  # _InferLockedPolicy -> GrootLiberoPolicyAdapter -> served object
    assert inner[0] is not inner[1] and inner[0]._diag is not inner[1]._diag  # one DiagSession per connection
    if mode in ("plain", "full", "shadow"):
        assert all(isinstance(p, G.GrootDiagPolicy) for p in inner) and inner[0]._shadow_on == (mode == "shadow")
        assert inner[0]._live == k
    else:
        assert all(isinstance(p, G.GrootEvidencePolicy) for p in inner)
    variant = SG.GROOT_VARIANT_MODES.get(mode)
    if variant is None:
        assert installs == []
    else:
        assert installs == [(variant, {"self_start": mode in E.SELF_VARIANT_MODES, "num_steps": E.warm_steps_of(arm)})] * 2
    with pytest.raises(RuntimeError):  # the live head must run the arm's K
        factory(_fake_policy(2))


# -- driver -------------------------------------------------------------------------------------------------------


def _args(**kw):
    base = dict(env_id="groot_libero_10", experiment_id=E.LIBERO_SELF_EXPERIMENT_ID, config_sha="cfg")
    base.update(kw)
    return SimpleNamespace(**base)


def test_strategy_expected_identities_follow_the_task_subset():
    from openpi.conductor import ServerEndpoint

    names = [f"task{i}" for i in range(10)]
    s = RL.LiberoDiagStrategy(_args(), names, "pool", "L", arm_id="selfwarmreset_t0.75_n1", tasks=(7, 3),
                              episodes=50, run_prefix="sdq23160t7")
    assert s.run_id.startswith("sdq23160t7-selfwarmreset_t0.75_n1-") and s.yaml_ids == [f"{s.run_id}_t7", f"{s.run_id}_t3"]
    graph = s.plan([], {y: ServerEndpoint("h", 1) for y in s.yaml_ids})
    eps = [e for st in graph.stages.values() for e in st.episodes]
    expected = s.expected()
    assert len(eps) == len(expected) == 100 and {x["task_uid"] for x in expected} == {e.task_uid for e in eps}
    assert {(x["task_id"], x["init_idx"]) for x in expected} == {(t, i) for t in (3, 7) for i in range(50)}
    assert all(x["task"] == names[x["task_id"]] and x["env_seed"] == 7 and x["init_pool_sha256"] == "pool" for x in expected)
    for e in eps:
        assert e.orig_init_state_idx == e.episode_idx and e.experiment == "libero_10"
        assert e.extra["arm_id"] == "selfwarmreset_t0.75_n1" and e.extra["num_trials_per_task"] == 50 and e.extra["seed"] == 7
    other = RL.LiberoDiagStrategy(_args(), names, "pool", "L", arm_id="selfwarmreset_t0.75_n1", tasks=(7,), episodes=50,
                                  run_prefix="sdq23161t7")
    assert other.run_id != s.run_id
    # the shadow keeps its historical run id and its 10 x 10 identity set
    shadow = RL.LiberoDiagStrategy(_args(experiment_id="sdiag_v1"), names, "pool", "L")
    assert shadow.run_id == "sdiag-lib-" + E.sha256_json(["groot_libero_10", "sdiag_v1", "cfg", "pool"])[:16]
    assert len(shadow.expected()) == 100 and {x["init_idx"] for x in shadow.expected()} == set(range(10))
    with pytest.raises(ValueError):
        RL.LiberoDiagStrategy(_args(), names, "pool", "L", arm_id="full", tasks=(10,), episodes=50)


def _cell(*extra, env_id="groot_libero_10", exp=E.LIBERO_SELF_EXPERIMENT_ID, root="/tmp/libero_self"):
    a = RL.build_parser().parse_args(["--env-id", env_id, "--experiment-id", exp, "--config-sha", "c", "--pool", "p",
                                      "--out-root", root, *extra])
    return RL.check_cell(a, E.ENVS[env_id])


def test_driver_argument_validation():
    assert _cell() == (tuple(range(10)), 10)  # the shadow (default arm)
    assert _cell("--arm-id", "selfmidreset_t0.75_n1", "--tasks", "3") == ((3,), 50)
    assert _cell("--arm-id", "selfmidfinal_t0.2", env_id="pi05_libero_spatial") == (tuple(range(10)), 50)
    assert _cell("--arm-id", "warmreset_t0.2", "--episodes", "2", "--tasks", "0,4", exp="sdiag_libero_smoke") == ((0, 4), 2)
    bad = [
        ("--tasks", "3"),                                                    # shadow: all tasks
        ("--episodes", "50"),                                                # shadow: 10 episodes
        ("--arm-id", "selfmidreset_t0.75_n1", "--episodes", "10"),           # the round runs 50 per task
        ("--arm-id", "warmshoot_t0.2"),                                      # not an arm of the round
        ("--arm-id", "selfmidfinal_t0.2"),                                   # a pi0.5 arm on a GR00T environment
        ("--arm-id", "full", "--tasks", "10"),                               # task ids 0..9
        ("--arm-id", "full", "--tasks", "1,1"),
    ]
    for extra in bad:
        with pytest.raises(ValueError):
            _cell(*extra)
    with pytest.raises(ValueError, match="out-root"):
        _cell("--arm-id", "full", root=RL.DEFAULT_OUT_ROOT)
    with pytest.raises(ValueError):
        _cell("--arm-id", "full", exp="sdiag_v1")                            # other experiments: shadow only
    with pytest.raises(ValueError):
        _cell("--arm-id", "full", exp="sdiag_libero_smoke")                  # smoke names its episode count
    with pytest.raises(ValueError):
        _cell("--arm-id", "bogus_arm", "--episodes", "1", exp="sdiag_libero_smoke")
    with pytest.raises(SystemExit):  # main refuses before hashing the pool or touching LIBERO
        RL.main(["--env-id", "groot_libero_10", "--experiment-id", "sdiag_v1", "--config-sha", "c", "--pool", "/nonexistent",
                 "--arm-id", "full"])
    assert RL.cell_dir("r", "groot_tp", "groot_libero_10", "shadow").as_posix() == "r/groot_tp/shadow_groot_libero_10"
    assert RL.cell_dir("r", "pi05", "pi05_libero_10", "warm_t0.2").as_posix() == "r/pi05/pi05_libero_10/warm_t0.2"


def test_runner_summary_records_the_settle_steps():
    from openpi.conductor.task import EpisodeTask
    from tests.exp.step_diag.test_worker_entry import _FakeClient, _FakeEnv

    client, env = _FakeClient(), _FakeEnv()
    task = EpisodeTask(task_uid="u", yaml_id="y", phase="eval", experiment="libero_10", task_id=2, episode_idx=3,
                       orig_init_state_idx=3, server_host="h", server_port=1, bundle_id="default",
                       extra=dict(num_trials_per_task=50, launch_id="L", arm_id="full", experiment_id="e",
                                  config_sha="c", init_pool_sha256="pool", seed=7))

    def episode(env, client, *args, **kwargs):
        env.reset()
        for _ in range(10):
            env.step(None)  # settle steps
        client.infer({})
        env.step(None)
        return True, [], [], [], 11

    runner = RL.LiberoDiagRunner(SimpleNamespace(seed=7, num_steps_wait=10), lambda _: (env, None, "task", 10),
                                 client_factory=lambda _: client, pool_sha="pool", run_episode_fn=episode)
    summary = next(r for r in runner.run(task, lambda *a: None).per_step_rows if r.get("row") == "episode_summary")
    assert summary["n_env_steps"] == 11 and summary["num_steps_wait"] == 10 and summary["n_decisions"] == 1


# -- analysis -----------------------------------------------------------------------------------------------------

ENV_ID, TEACHER = "groot_libero_10", "groot_tp"
TASKS = {0: "put both the alphabet soup and the tomato sauce in the basket", 1: "turn on the stove and put the moka pot on it"}
TASK_NAMES = {**TASKS, **{i: f"libero_10 task {i}" for i in range(2, E.LIBERO_N_TASKS)}}


def _self_seed(experiment, env_id, ident, attempt, decision_idx):
    """What ``DiagSession.self_start_seed`` gives a LIBERO decision (pinned against the session below)."""
    return R.noise_seed(experiment, env_id, ident["task"], (ident["env_seed"], ident["init_idx"], ident["init_pool_sha256"]),
                        attempt, decision_idx, "self")


def _write_libero_arm(root, server_root, arm_id, outcomes, *, steps=None, schedule=None, n_decisions=3, decisions=None,
                      env_id=ENV_ID, experiment=E.LIBERO_SELF_EXPERIMENT_ID, pool="pool", env_seed=7, self_edit=None,
                      summary_edit=None, journal_edit=None, attempts=None):
    """outcomes: {task_id: [success per init_idx]}; one LIBERO cell of the round (driver + server side), shaped like
    ``run_libero_diag`` (launch with env_id / env_seed, worker summary with the client stamp) and the recorder (a
    self arm's rows carry ``self_start`` / ``self_seed`` / ``self_direct_nfe = K``). Hooks: ``self_edit(row)``,
    ``summary_edit(uid, row) -> [rows]`` (the rows written for that identity), ``journal_edit(uid, rec)``;
    ``attempts``: {(task_id, i): accepted attempt}, an earlier attempt leaving a retriable failure's summary."""
    env = E.ENVS[env_id]
    teacher = "pi05" if env.policy == "pi05" else "groot_tp"
    warm = arm_id != "full" and not arm_id.startswith("plain_k")
    is_self = warm and E.is_self_mode(E.warm_mode_of(arm_id))
    m = E.executed_steps_of(env_id, arm_id)
    d = root / teacher / env_id / arm_id
    s = server_root / teacher / env_id / arm_id
    d.mkdir(parents=True)
    s.mkdir(parents=True)
    manifest = E.RunManifest(experiment_id=experiment, env=env.to_json(), arm_id=arm_id,
                             mode=E.warm_mode_of(arm_id) if warm else "plain", exec_steps=None if warm else m,
                             checkpoint="ckpt", checkpoint_sha256="weights", cache_config="c" if warm else None,
                             cache_config_sha256="cache" if warm else None, library="lib" if warm else None,
                             library_sha256="library" if warm else None, code_commit="x",
                             extras={"h_exec": 5, "runtime": {"source_sha256": "s"}})
    cfg = manifest.write(s / f"manifest_{arm_id}.json")
    stamp = {"launch_id": "L0", "arm_id": arm_id, "experiment_id": experiment, "config_sha": cfg}
    worker_stamp = {**stamp, "init_pool_sha256": pool, "seed": env_seed}  # LiberoDiagRunner's client stamp
    schedule = schedule or (("pi05_v1" if env.policy == "pi05" else f"groot_n15_k{8 if warm else m}_v1"))
    max_dec = max([n_decisions, *(decisions or {}).values()])
    (s / "arrays").mkdir()
    rel = "arrays/shared.npz"  # every episode's arrays are zeros: one file serves them all
    np.savez(s / rel, **{f"a_exec_{j:04d}": np.zeros((env.action_horizon, env.action_dim), dtype=np.float32)
                         for j in range(max_dec)})
    meta = {"arrays": rel, "arrays_sha256": hashlib.sha256((s / rel).read_bytes()).hexdigest(), "env_id": env_id}
    expected, journal, rows, per_step = [], [], [], []
    for task_id, succ in outcomes.items():
        for i, ok in enumerate(succ):
            uid = f"run_t{task_id}:eval:{task_id}:{i}"
            ident = {"task_uid": uid, "task": TASK_NAMES[task_id], "task_id": task_id, "init_idx": i, "env_seed": env_seed,
                     "init_pool_sha256": pool}
            expected.append(ident)
            att = (attempts or {}).get((task_id, i), 1)
            rec = {"accepted": True, "run_id": "driver", "task_uid": uid, "status": "done", "success": bool(ok),
                   "attempt": att, "error": None}
            journal.append(journal_edit(uid, rec) if journal_edit else rec)
            n_dec = n_decisions if decisions is None else decisions[(task_id, i)]
            summ = {"row": "episode_summary", "step_idx": -2, "task_uid": uid, "attempt": att, "accepted": True,
                    "run_id": "driver", "task_id": task_id, "init_idx": i, "success": bool(ok), "error": None,
                    "n_decisions": n_dec, "n_env_steps": 10 + 5 * n_dec - 2, "num_steps_wait": 10,
                    "worker_runtime": {"host": "timan107", "source_sha256": "s"}, **worker_stamp}
            summs = [summ]
            if att > 1:  # the retriable failure of the earlier attempt left its own (accepted) summary
                summs.insert(0, dict(summ, attempt=att - 1, success=False, error="ws timeout", n_decisions=1,
                                     n_env_steps=13))
            per_step += summary_edit(uid, summs) if summary_edit else summs
            per_step.append({"_kind": "client_timing", "task_uid": uid, "infer_ms": 1.0})  # LIBERO rows without step_idx
            for j in range(n_dec):
                row = {**ident, **meta, "config_sha": cfg, "start_t": E.warm_t_of(arm_id) if warm else None,
                       "schedule_id": schedule, "decision_idx": j, "status": "ok",
                       "hit_type": "WARM_START" if warm else "MISS", "executed_steps": m if steps is None else steps,
                       "n_stage3_calls": 1, "attempt": att, "experiment_id": experiment}
                if is_self:
                    row.update(self_start=True, self_seed=_self_seed(experiment, env_id, ident, att, j),
                               self_direct_nfe=env.k_full)
                if self_edit:
                    self_edit(row)
                rows.append(row)
            rows.append({**ident, **meta, "config_sha": cfg, "outcome": bool(ok), "status": "finalize", "terminal": True,
                         "n_decisions": n_dec, "client_stamp": stamp, "stamp_mismatch": [], "attempt": att})
    launch = {"launch_id": "L0", "driver_run_id": "driver", "config_sha": cfg, "experiment_id": experiment,
              "arm_id": arm_id, "teacher": teacher, "env_id": env_id, "env_seed": env_seed, "expected": expected}
    (d / "launch_L0.json").write_text(json.dumps(launch))
    (d / "journal_x.jsonl").write_text("\n".join(json.dumps(r) for r in journal) + "\n")
    (d / "per_step_x.jsonl").write_text("\n".join(json.dumps(r) for r in per_step) + "\n")
    (s / "rows_x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_libero_analysis_pairs_self_against_cache_full_and_plain(tmp_path):
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    data = {
        "full": {0: [1, 1, 1, 1], 1: [1, 1, 0, 1]},
        "plain_k1": {0: [0, 1, 0, 0], 1: [0, 0, 0, 1]},
        "warm_t0.875": {0: [1, 1, 0, 0], 1: [1, 0, 0, 1]},
        "midreset_t0.75_n1": {0: [1, 0, 1, 0], 1: [0, 1, 0, 1]},
        "selfmidreset_t0.75_n1": {0: [1, 1, 1, 0], 1: [1, 1, 0, 1]},
    }
    for aid, outcomes in data.items():
        _write_libero_arm(arms, srv, aid, outcomes)
    res = WV.analyze_libero(ENV_ID, arms, srv, m=1)
    assert res["m"] == 1 and "selfwarmreset_t0.75_n1" in res["missing_arms"] and "full" not in res["missing_arms"]
    assert [r["task_id"] for r in res["tasks"].values()] == [0, 1]
    for r in res["tasks"].values():
        assert not r["comparison_problems"]
        for aid, c in r["cells"].items():
            assert c["complete"] and c["equal_nfe"], (aid, c["problems"])
        assert r["cells"]["midreset_t0.75_n1"]["mean_executed_steps"] == 1.0
    t0 = res["tasks"][TASKS[0]]["paired_deltas"]
    assert set(t0) == {"warm_t0.875 - plain_k1", "warm_t0.875 - full", "midreset_t0.75_n1 - plain_k1",
                       "midreset_t0.75_n1 - warm_t0.875", "midreset_t0.75_n1 - full",
                       "selfmidreset_t0.75_n1 - midreset_t0.75_n1", "selfmidreset_t0.75_n1 - full",
                       "selfmidreset_t0.75_n1 - plain_k1"}
    d = t0["selfmidreset_t0.75_n1 - midreset_t0.75_n1"]
    assert d["n"] == 4 and d["point"] == pytest.approx(0.25) and (d["n10"], d["n01"]) == (1, 0)
    # 2 tasks x 4 episodes: a run in flight, never the formal 10 x 50 round (B2)
    assert res["status"] == "partial" and not res["formal"]["complete"] and "macro" not in res
    for aid in data:
        rec = res["formal"]["arms"][aid]
        assert not rec["formal"] and "identity_set_not_frozen" in rec["problems"]
        assert all(r["cells"][aid]["formal"] is False for r in res["tasks"].values())
    assert all(r["status"] == "partial" for r in res["tasks"].values())
    macro = res["macro_partial"]
    assert macro["status"].startswith("partial")
    assert macro["sr"]["selfmidreset_t0.75_n1"] == {"n_tasks": 2, "macro_sr": pytest.approx(0.75)}
    assert macro["paired_deltas"]["selfmidreset_t0.75_n1 - plain_k1"]["point"] == pytest.approx(0.75 - 0.25)
    md = WV.markdown_libero(res)
    assert "t0: put both" in md and "status: **PARTIAL**" in md and "Progress macro — PARTIAL" in md
    assert "Macro — formal" not in md and "| partial |" in md and "K + m" in md
    with pytest.raises(ValueError):
        WV.analyze_libero(ENV_ID, arms, srv, m=4)  # not a panel of the round


def test_libero_admission_gates_n_and_the_k8_schedule(tmp_path):
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    _write_libero_arm(arms, srv, "midfinal_t0.75_n1", {0: [1, 0]}, steps=2)         # ran 2 steps, the arm names 1
    _write_libero_arm(arms, srv, "midfinal50_t0.75_n1", {0: [1, 0]}, schedule="groot_n15_k4_v1")
    _write_libero_arm(arms, srv, "resetfinal_t0.75_n1", {0: [1, 0]})
    for aid, problem in (("midfinal_t0.75_n1", "steps_mismatch"), ("midfinal50_t0.75_n1", "schedule_mismatch"),
                         ("resetfinal_t0.75_n1", None)):
        arm = AG.load_arm(arms / TEACHER / ENV_ID / aid)
        server = AG.load_server_rows(srv / TEACHER / ENV_ID / aid)
        c = AG.cell_admission(arm, server, TASKS[0], kind="warm", m=1, env_id=ENV_ID)
        assert c["complete"] and c["equal_nfe"] == (problem is None) and (problem is None or c["problems"][problem] == 6)


def test_libero_success_length_nets_settle_steps_and_pairs_self_with_cache(tmp_path):
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    decisions = {(0, i): 10 + i for i in range(4)}
    _write_libero_arm(arms, srv, "full", {0: [1, 1, 1, 1]}, decisions=decisions)
    _write_libero_arm(arms, srv, "resetfinal_t0.5_n2", {0: [1, 1, 0, 1]}, decisions={k: v + 2 for k, v in decisions.items()})
    _write_libero_arm(arms, srv, "selfresetfinal_t0.5_n2", {0: [1, 1, 1, 1]}, decisions={k: v - 1 for k, v in decisions.items()})
    loaded, problems = {}, []
    for aid in ("full", "resetfinal_t0.5_n2", "selfresetfinal_t0.5_n2"):
        loaded[aid], p = SL.arm_episodes(TEACHER, aid, (arms,), env_dir=ENV_ID)
        problems += p
    key = SL.identity_key({"task": TASKS[0], "init_idx": 2, "env_seed": 7, "init_pool_sha256": "pool"}, ENV_ID)
    assert key == (TASKS[0], 2, 7, "pool", ENV_ID)
    assert problems == [] and loaded["full"][key] == [(True, 5 * 12 - 2, 12)]  # 10 settle steps netted out
    res = SL.analyse(loaded, [], ["full", "resetfinal_t0.5_n2", "selfresetfinal_t0.5_n2"], 2, np.random.default_rng(0),
                     vs_cache=True)
    row = res["arms"]["selfresetfinal_t0.5_n2"]
    assert row["paired"]["macro_diff"] == pytest.approx(-1.0) and row["paired_vs_cache"]["macro_diff"] == pytest.approx(-3.0)
    assert row["paired_vs_cache"]["n_pairs"] == 3 and "paired_vs_cache" not in res["arms"]["resetfinal_t0.5_n2"]
    panels = SL.libero_panels(("groot_libero_10", "pi05_libero_spatial"))
    assert set(panels) == {"groot_libero_10_m1", "groot_libero_10_m2", "pi05_libero_spatial_m2"}
    assert len(panels["groot_libero_10_m1"][2]) == 1 + 1 + 1 + 12 and len(panels["pi05_libero_spatial_m2"][2]) == 9


# -- G2 Round 1 revisions: pairing identity (B1), formal completeness (B2), self-start evidence (B3), NFE (B4) and
# -- accepted success lengths (B5) ---------------------------------------------------------------------------------

PI_ENV = "pi05_libero_spatial"


def _cell_of(arms, srv, aid, *, task=TASKS[0], env_id=ENV_ID, kind="warm", m=1):
    teacher = "pi05" if E.ENVS[env_id].policy == "pi05" else "groot_tp"
    return AG.cell_admission(AG.load_arm(arms / teacher / env_id / aid), AG.load_server_rows(srv / teacher / env_id / aid),
                             task, kind=kind, m=m, env_id=env_id)


def test_libero_pairing_identity_includes_the_initial_state_pool_and_env_seed(tmp_path):
    same = {0: [1, 0, 1, 0], 1: [0, 1, 1, 0]}
    arms, srv = tmp_path / "same", tmp_path / "same_srv"
    _write_libero_arm(arms, srv, "plain_k1", {0: [0, 0, 1, 1], 1: [0, 0, 0, 1]})
    _write_libero_arm(arms, srv, "midreset_t0.75_n1", same)
    res = WV.analyze_libero(ENV_ID, arms, srv, m=1)
    for r in res["tasks"].values():  # one pool, one seed: every identity pairs
        assert r["comparison_problems"] == [] and r["paired_deltas"]["midreset_t0.75_n1 - plain_k1"]["n"] == 4
    assert "midreset_t0.75_n1 - plain_k1" in res["macro_partial"]["paired_deltas"]
    for field, value, problem in (("pool", "another pool", "init_pool_mismatch"), ("env_seed", 8, "env_seed_mismatch")):
        arms, srv = tmp_path / field, tmp_path / f"{field}_srv"
        _write_libero_arm(arms, srv, "plain_k1", {0: [0, 0, 1, 1], 1: [0, 0, 0, 1]})
        _write_libero_arm(arms, srv, "midreset_t0.75_n1", same, **{field: value})  # consistent evidence, other initial states
        assert _cell_of(arms, srv, "midreset_t0.75_n1")["equal_nfe"]               # admissible on its own
        res = WV.analyze_libero(ENV_ID, arms, srv, m=1)
        for r in res["tasks"].values():
            assert problem in r["comparison_problems"]
            assert r["paired_deltas"]["midreset_t0.75_n1 - plain_k1"]["n"] == 0  # the same init_idx never pairs
        assert "midreset_t0.75_n1 - plain_k1" not in res["macro_partial"]["paired_deltas"]
        md = WV.markdown_libero(res)
        assert "MISMATCH (task excluded from macro)" in md and problem in md


def test_success_length_pairs_only_the_same_pool_and_reports_a_mismatch(tmp_path):
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    _write_libero_arm(arms, srv, "full", {0: [1, 1, 1, 1]})
    _write_libero_arm(arms, srv, "plain_k1", {0: [1, 1, 1, 1]}, n_decisions=4)
    _write_libero_arm(arms, srv, "midreset_t0.75_n1", {0: [1, 1, 1, 1]}, n_decisions=2, pool="another pool")
    data = {aid: SL.arm_episodes(TEACHER, aid, (arms,), env_dir=ENV_ID)[0] for aid in ("full", "plain_k1", "midreset_t0.75_n1")}
    res = SL.analyse(data, [], list(data), 2, np.random.default_rng(0))
    assert res["arms"]["plain_k1"]["paired"]["n_pairs"] == 4 and res["arms"]["plain_k1"]["paired"]["macro_diff"] == 1.0
    assert "paired" not in res["arms"]["midreset_t0.75_n1"]  # another pool: no pair with full
    problems = SL.identity_mismatches(data, list(data))
    assert len(problems) == 1 and problems[0].startswith("4 (task, init_idx)") and "midreset_t0.75_n1" in problems[0]
    assert SL.identity_mismatches(data, ["full", "plain_k1"]) == []


def _manifest_arm(*, experiment=E.LIBERO_SELF_EXPERIMENT_ID, tasks=range(E.LIBERO_N_TASKS), episodes=E.LIBERO_SELF_EPISODES,
                  env_id=PI_ENV, seed=7, pools=("pool",)):
    expected = {f"u{t}:{i}": {"task_uid": f"u{t}:{i}", "task": TASK_NAMES[t], "task_id": t, "init_idx": i, "env_seed": seed,
                              "init_pool_sha256": pools[i % len(pools)]} for t in tasks for i in range(episodes)}
    return {"launches": {"L0": {"experiment_id": experiment, "env_id": env_id, "env_seed": seed}}, "expected": expected,
            "manifest_problems": []}


@pytest.mark.parametrize("kw,problem", [
    ({"experiment": "sdiag_libero_smoke"}, "experiment_not_formal"),     # smoke data
    ({"experiment": "sdiag_v1"}, "experiment_not_formal"),
    ({"episodes": 49}, "identity_set_not_frozen"),                        # truncated manifest
    ({"episodes": 4}, "identity_set_not_frozen"),
    ({"tasks": range(9)}, "identity_set_not_frozen"),                     # a task subset
    ({"seed": 8}, "env_seed_not_frozen"),
    ({"pools": ("pool", "another pool")}, "init_pool_not_unique"),
    ({"env_id": "groot_libero_10"}, "launch_environment_mismatch"),
])
def test_libero_formal_status_needs_the_frozen_design(kw, problem):
    assert WV.libero_formal_problems(PI_ENV, _manifest_arm()) == []
    assert problem in WV.libero_formal_problems(PI_ENV, _manifest_arm(**kw))


@pytest.fixture(scope="module")
def formal_pi05_round(tmp_path_factory):
    """Every arm of the pi0.5 LIBERO panel (m = 2) on the frozen 10 tasks x 50 init_idx of the formal experiment."""
    root = tmp_path_factory.mktemp("formal")
    arms, srv = root / "arms", root / "srv"
    rng = np.random.default_rng(7)
    for aid, _, _ in WV.libero_arm_specs(PI_ENV, 2):
        outcomes = {t: [int(x) for x in rng.random(E.LIBERO_SELF_EPISODES) < 0.6] for t in range(E.LIBERO_N_TASKS)}
        _write_libero_arm(arms, srv, aid, outcomes, env_id=PI_ENV, n_decisions=1)
    return arms, srv


def _run_success_length(monkeypatch, tmp_path, arms, env_id):
    out = tmp_path / "sl.json"
    monkeypatch.setattr("sys.argv", ["success_length", "--benchmark", "libero", "--roots", str(arms), "--env-ids", env_id,
                                     "--out", str(out)])
    SL.main()
    return json.loads(out.read_text())


def test_libero_formal_round_is_formal_and_its_macro_uses_formal_arms_only(formal_pi05_round, tmp_path, monkeypatch):
    arms, srv = formal_pi05_round
    res = WV.analyze_libero(PI_ENV, arms, srv, m=2)
    assert res["status"] == "formal" and res["formal"]["complete"] and res["missing_arms"] == []
    assert all(rec["formal"] and rec["problems"] == [] for rec in res["formal"]["arms"].values())
    assert len(res["tasks"]) == 10 and all(r["status"] == "formal" for r in res["tasks"].values())
    assert "macro_partial" not in res and res["macro"]["status"] == "formal"
    assert all(v["n_tasks"] == 10 for v in res["macro"]["sr"].values())
    assert res["macro"]["paired_deltas"]["selfwarmreset_t0.2 - warmreset_t0.2"]["n_tasks"] == 10
    md = WV.markdown_libero(res)
    assert "status: **FORMAL**" in md and "Macro — formal" in md and "PARTIAL" not in md
    sl = _run_success_length(monkeypatch, tmp_path, arms, PI_ENV)["decisions"]["pi05_libero_spatial_m2"]
    assert sl["status"] == "formal" and all(row["status"] == "formal" for row in sl["arms"].values())


def test_libero_smoke_arm_is_never_formal(formal_pi05_round, tmp_path, monkeypatch):
    import shutil

    arms0, srv0 = formal_pi05_round
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    shutil.copytree(arms0, arms)
    shutil.copytree(srv0, srv)
    for root in (arms, srv):
        shutil.rmtree(root / "pi05" / PI_ENV / "selfwarmreset_t0.2")
    smoke = {t: [1] * E.LIBERO_SELF_EPISODES for t in range(E.LIBERO_N_TASKS)}  # the full 10 x 50, but smoke data
    _write_libero_arm(arms, srv, "selfwarmreset_t0.2", smoke, env_id=PI_ENV, n_decisions=1, experiment="sdiag_libero_smoke")
    res = WV.analyze_libero(PI_ENV, arms, srv, m=2)
    rec = res["formal"]["arms"]["selfwarmreset_t0.2"]
    assert res["status"] == "partial" and not rec["formal"] and "experiment_not_formal" in rec["problems"]
    assert res["macro"]["sr"]["selfwarmreset_t0.2"] == {"n_tasks": 0, "macro_sr": None}
    assert res["macro"]["sr"]["warmreset_t0.2"]["n_tasks"] == 10
    assert not any("selfwarmreset_t0.2" in k for k in res["macro"]["paired_deltas"])
    assert "selfwarmreset_t0.2 - warmreset_t0.2" in res["macro_partial"]["paired_deltas"]  # progress view, labelled
    assert res["macro_partial"]["status"].startswith("partial")
    sl = _run_success_length(monkeypatch, tmp_path, arms, PI_ENV)["decisions"]["pi05_libero_spatial_m2"]
    assert sl["status"] == "partial" and sl["arms"]["selfwarmreset_t0.2"]["status"] == "partial"
    assert sl["arms"]["warmreset_t0.2"]["status"] == "formal"


def test_libero_success_length_in_flight_is_partial(tmp_path, monkeypatch):
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    for aid in ("full", "plain_k1", "midreset_t0.75_n1"):
        _write_libero_arm(arms, srv, aid, {0: [1, 1, 0, 1]})
    sl = _run_success_length(monkeypatch, tmp_path, arms, ENV_ID)["env_steps"]["groot_libero_10_m1"]
    assert sl["status"] == "partial" and sl["problems"] == []
    assert all(row["status"] == "partial" for row in sl["arms"].values())
    assert any(p.startswith("accepted_lengths:") for p in sl["formal"]["full"]["problems"])


@pytest.mark.parametrize("arm,edit,problems", [
    ("selfmidreset_t0.75_n1", lambda r: [r.pop(k) for k in ("self_start", "self_seed", "self_direct_nfe")],
     {"self_start_missing", "self_seed_mismatch", "self_direct_nfe_mismatch"}),              # no evidence at all
    ("selfmidreset_t0.75_n1", lambda r: r.update(self_start=False), {"self_start_missing"}),
    ("selfmidreset_t0.75_n1", lambda r: r.update(self_direct_nfe=0), {"self_direct_nfe_mismatch"}),
    ("selfmidreset_t0.75_n1", lambda r: r.update(self_direct_nfe=4), {"self_direct_nfe_mismatch"}),  # RoboCasa's K
    ("selfmidreset_t0.75_n1", lambda r: r.update(self_seed=r["self_seed"] + 1), {"self_seed_mismatch"}),
    ("selfmidreset_t0.75_n1",                                                              # the next decision's seed
     lambda r: r.update(self_seed=_self_seed(E.LIBERO_SELF_EXPERIMENT_ID, ENV_ID, r, 1, r["decision_idx"] + 1)),
     {"self_seed_mismatch"}),
    ("selfmidreset_t0.75_n1",                                                              # another pool's seed
     lambda r: r.update(self_seed=_self_seed(E.LIBERO_SELF_EXPERIMENT_ID, ENV_ID, dict(r, init_pool_sha256="x"), 1,
                                             r["decision_idx"])), {"self_seed_mismatch"}),
    ("midreset_t0.75_n1", lambda r: r.update(self_start=True, self_seed=1, self_direct_nfe=8), {"self_start_on_cache_arm"}),
    ("midreset_t0.75_n1", lambda r: r.update(self_start=True), {"self_start_on_cache_arm"}),
])
def test_self_start_evidence_is_required_on_self_arms_and_refused_on_cache_arms(tmp_path, arm, edit, problems):
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    _write_libero_arm(arms, srv, arm, {0: [1, 0]})
    assert _cell_of(arms, srv, arm)["equal_nfe"]  # the recorder's evidence as written is admitted
    arms, srv = tmp_path / "bad", tmp_path / "bad_srv"
    _write_libero_arm(arms, srv, arm, {0: [1, 0]}, self_edit=edit)
    c = _cell_of(arms, srv, arm)
    assert c["complete"] and not c["equal_nfe"] and set(c["problems"]) == problems
    assert all(c["problems"][p] == 6 for p in problems)  # every decision of both episodes
    res = WV.analyze_libero(ENV_ID, arms, srv, m=1)
    assert not res["tasks"][TASKS[0]]["cells"][arm]["equal_nfe"]


def test_plain_arm_with_self_fields_is_refused(tmp_path):
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    _write_libero_arm(arms, srv, "plain_k1", {0: [1, 0]}, self_edit=lambda r: r.update(self_start=True))
    c = _cell_of(arms, srv, "plain_k1", kind="plain")
    assert not c["equal_nfe"] and c["problems"] == {"self_start_on_cache_arm": 6}


@pytest.mark.parametrize("env_id,extra", [("groot_libero_10", {"init_pool_sha256": "pool"}), ("groot_rc", {})])
def test_expected_self_seed_is_the_sessions_self_start_seed(tmp_path, env_id, extra):
    """The analysis recomputes exactly what ``DiagSession.self_start_seed`` publishes, decision by decision, from the
    expected identity (LIBERO with its pool; RoboCasa without one)."""
    spec = R.DiagSpec(experiment_id="exp", env_id=env_id, arm_id="a", mode="selfmidreset", k_full=8, action_shape=(16, 32))
    sess = R.DiagRecorder(spec, tmp_path).session()
    sess.begin_episode(R.EpisodeIdentity.from_episode_start(
        experiment="bench", task="T", episode_id=3,
        extra_metadata={"task_uid": "u", "attempt": 2, "orig_init_state_idx": 3, "seed": 17, **extra}))
    ident = {"task": "T", "init_idx": 3, "env_seed": 17, **extra}
    for j in range(3):
        assert sess.self_start_seed() == AG.expected_self_seed("exp", env_id, ident, 2, j)
        sess.record(a_exec=torch.zeros(16, 32), executed_steps=1, n_stage3_calls=1, hit_type="WARM_START", start_t=0.75,
                    schedule_id="s")
    assert AG.expected_self_seed("exp", env_id, ident, 1, 0) != AG.expected_self_seed("exp", env_id, ident, 2, 0)


def test_nfe_accounting_counts_the_self_start_direct_inference(tmp_path):
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    # GR00T LIBERO: K = 8, N = 1, 3 decisions per episode
    _write_libero_arm(arms, srv, "selfmidreset_t0.75_n1", {0: [1, 0]}, n_decisions=3)
    _write_libero_arm(arms, srv, "midreset_t0.75_n1", {0: [1, 0]}, n_decisions=3)
    s = _cell_of(arms, srv, "selfmidreset_t0.75_n1")
    assert s["equal_nfe"] and s["self_start"] and s["k_self"] == 8 and s["mean_executed_steps"] == 1.0
    assert s["mean_self_direct_nfe"] == 8.0 and s["n_decisions"] == 6
    for uid in s["episode_total_nfe"]:
        assert (s["episode_continuation_nfe"][uid], s["episode_self_start_nfe"][uid], s["episode_total_nfe"][uid]) == (3, 24, 27)
    c = _cell_of(arms, srv, "midreset_t0.75_n1")
    assert not c["self_start"] and c["k_self"] == 0
    for uid in c["episode_total_nfe"]:
        assert (c["episode_continuation_nfe"][uid], c["episode_self_start_nfe"][uid], c["episode_total_nfe"][uid]) == (3, 0, 3)
    # pi0.5 LIBERO: K = 10, N = 2, 3 decisions per episode
    _write_libero_arm(arms, srv, "selfwarmreset_t0.2", {0: [1, 0]}, n_decisions=3, env_id=PI_ENV)
    p = _cell_of(arms, srv, "selfwarmreset_t0.2", env_id=PI_ENV, m=2)
    assert p["equal_nfe"] and p["k_self"] == 10 and p["mean_executed_steps"] == 2.0
    for uid in p["episode_total_nfe"]:
        assert (p["episode_continuation_nfe"][uid], p["episode_self_start_nfe"][uid], p["episode_total_nfe"][uid]) == (6, 30, 36)
    # the continuation gate and stage-3 accounting are unchanged: the direct run is not an executed stage-3 call
    _write_libero_arm(arms, srv, "selfmidreset_t0.5_n2", {0: [1, 0]}, self_edit=lambda r: r.update(n_stage3_calls=2))
    assert _cell_of(arms, srv, "selfmidreset_t0.5_n2", m=2)["problems"] == {"extra_stage3_calls": 6}
    # a self arm whose direct-inference count is missing has no total
    _write_libero_arm(arms, srv, "selfwarmreset_t0.75_n1", {0: [1]}, self_edit=lambda r: r.pop("self_direct_nfe"))
    m = _cell_of(arms, srv, "selfwarmreset_t0.75_n1")
    assert list(m["episode_self_start_nfe"].values()) == [None] and list(m["episode_total_nfe"].values()) == [None]
    # the report shows the cost apart: per decision K + m on the self arm, m on its cache arm
    res = WV.analyze_libero(ENV_ID, arms, srv, m=1)
    cells = res["tasks"][TASKS[0]]["cells"]
    assert cells["selfmidreset_t0.75_n1"]["nfe_per_decision"] == {"continuation": 1.0, "self_start": 8.0, "total": 9.0}
    assert cells["selfmidreset_t0.75_n1"]["episode_nfe_mean"] == {"continuation": 3.0, "self_start": 24.0, "total": 27.0}
    assert cells["midreset_t0.75_n1"]["nfe_per_decision"] == {"continuation": 1.0, "self_start": 0.0, "total": 1.0}
    md = WV.markdown_libero(res)
    assert "| selfmidreset_t0.75_n1 | 2 | 0.50 | partial | 1.0 + 8 self |" in md
    assert "| midreset_t0.75_n1 | 2 | 0.50 | partial | 1.0 |" in md


BAD = "run_t0:eval:0:1"


def _lengths(root, **kw):
    arms = root / "arms"
    _write_libero_arm(arms, root / "srv", "full", {0: [1, 1, 1, 0]}, **kw)
    return SL.arm_episodes(TEACHER, "full", (arms,), env_dir=ENV_ID)


def _edit_bad(**fields):
    return lambda uid, rows: [dict(r, **fields) for r in rows] if uid == BAD else rows


@pytest.mark.parametrize("name,kw", [
    ("summary_not_accepted", {"summary_edit": _edit_bad(accepted=False)}),
    ("summary_success_mismatch", {"summary_edit": _edit_bad(success=False)}),
    ("summary_error", {"summary_edit": _edit_bad(error="ws timeout")}),
    ("summary_other_run", {"summary_edit": _edit_bad(run_id="another driver")}),
    ("summary_other_arm", {"summary_edit": _edit_bad(arm_id="plain_k1")}),
    ("summary_other_config", {"summary_edit": _edit_bad(config_sha="another config")}),
    ("summary_other_pool", {"summary_edit": _edit_bad(init_pool_sha256="another pool")}),
    ("summary_missing_counts", {"summary_edit": _edit_bad(n_env_steps=None)}),
    ("terminal_error", {"journal_edit": lambda uid, rec: dict(rec, error="boom") if uid == BAD else rec}),
    ("attempt2_summary_missing", {"attempts": {(0, 1): 2},                     # only the stale attempt 1 reported
                                  "summary_edit": lambda uid, rows: [r for r in rows if r["attempt"] == 1]}),
    ("two_accepted_summaries", {"summary_edit": lambda uid, rows: rows + [dict(rows[-1], n_decisions=5)] if uid == BAD else rows}),
])
def test_success_length_takes_only_accepted_consistent_attempts(tmp_path, name, kw):
    data, problems = _lengths(tmp_path, **kw)
    assert len(problems) == 1 and BAD in problems[0], problems
    assert sorted(k[1] for k in data) == [0, 2, 3]  # the three other identities keep their lengths


def test_success_length_retry_mixtures_use_the_accepted_attempt(tmp_path):
    key = SL.identity_key({"task": TASKS[0], "init_idx": 1, "env_seed": 7, "init_pool_sha256": "pool"}, ENV_ID)
    # accepted on attempt 2: the attempt-1 retriable failure's summary (1 decision, error) never gives the length
    data, problems = _lengths(tmp_path / "retry", attempts={(0, 1): 2})
    assert problems == [] and data[key] == [(True, 13, 3)]
    # a fenced (not accepted) duplicate report of the accepted attempt is ignored
    data, problems = _lengths(tmp_path / "fenced", attempts={(0, 1): 2}, summary_edit=lambda uid, rows: (
        rows + [dict(rows[-1], accepted=False, n_decisions=7, n_env_steps=38)] if uid == BAD else rows))
    assert problems == [] and data[key] == [(True, 13, 3)]


def test_success_length_propagates_manifest_problems(tmp_path):
    arms = tmp_path / "arms"
    _write_libero_arm(arms, tmp_path / "srv", "full", {0: [1, 1]})
    d = arms / TEACHER / ENV_ID / "full"
    launch = json.loads((d / "launch_L0.json").read_text())
    (d / "launch_L0b.json").write_text(json.dumps(dict(launch, started="later")))  # one launch id, two contents
    _, problems = SL.arm_episodes(TEACHER, "full", (arms,), env_dir=ENV_ID)
    assert any("manifest problem duplicate_or_missing_launch_id" in p for p in problems)
    (d / "launch_L0b.json").unlink()
    (d / "launch_L0.json").write_text(json.dumps(dict(launch, env_id="groot_libero_spatial")))
    _, problems = SL.arm_episodes(TEACHER, "full", (arms,), env_dir=ENV_ID)
    assert any("another environment" in p for p in problems)


def test_success_length_robocasa_path_applies_the_same_acceptance(tmp_path):
    from tests.exp.step_diag.test_aggregate import _write_arm

    arms, srv = tmp_path / "arms", tmp_path / "srv"
    _write_arm(arms, srv, "plain_k2", "plain", 2, {"CloseFridge": [1, 0, 1]})
    d = arms / "pi05" / "plain_k2"
    launch = json.loads((d / "launch_0.json").read_text())
    stamp = {k: launch[k] for k in ("launch_id", "arm_id", "experiment_id", "config_sha")}  # StepDiagEpisodeRunner's stamp
    rows = [dict(json.loads(line), **stamp) for line in (d / "per_step_x.jsonl").read_text().splitlines() if line.strip()]
    (d / "per_step_x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    data, problems = SL.arm_episodes("pi05", "plain_k2", (arms,))
    assert problems == [] and sorted(data) == [("CloseFridge", i, 2_000_000 + i, None, None) for i in range(3)]
    rows[1]["accepted"] = False
    (d / "per_step_x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    data, problems = SL.arm_episodes("pi05", "plain_k2", (arms,))
    assert len(data) == 2 and len(problems) == 1 and "CloseFridge:1" in problems[0]
