"""Adversarial design, staged dispatch, client alignment and ICC checks on CPU."""
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

from .assignment import assign
from .design import Design
from .v2_engine import Engine
from .telemetry import Client, EnvTap
from .read_v2 import control_join
from .pilot_stats import icc


def staged(model, coordinator, pvalue, root):
    H = 10 if model == "pi05" else 16
    saved = torch.random.get_rng_state().clone()
    calls = dict(stage1=0, stage2=0, stage3=0)
    def f1(obs):
        calls["stage1"] += 1
        return obs
    def f2(s):
        calls["stage2"] += 1
        return s
    def f3(s, noise=None, **kw):
        calls["stage3"] += 1
        return NS(action_chunk=noise + .1, action_pred=noise + .1)
    def sample(shape, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)
    cache = np.full((H, 32), .4, np.float32)
    record = dict(schema="r6p3.anchor.v2", assignment=assign(603, 0, 0, 0, 0, pvalue),
                  retrieval={"rows": np.array([0])}, guards={}, state={}, timing={}, extensions={})
    method = NS(p=pvalue, seed=603, replicate=0, strata=None, record=record,
        design=Design({}), design_config=Design({}).config, catalog_sha256="test", catalog={},
        resample_p=1., resample_draws=4, blind_shadow=True, commit_controls=10)
    method.base = NS(_dist=lambda: None, tasks={}, act=cache[None])
    emitted = []
    rt = NS(blind=True, policy_tail=True, policy_tail_blocks=1, gpu=None, randomized=False,
        judge=NS(mode="guard_only", cap=0, burst=1, step0="judge"), opts=NS(os_log_dir=str(root)),
        model=model, suite="l10", H=H, tag="test", emit=emitted.append)
    s = NS(rt=rt, method=method, conn=1, miss_steps=10 if model == "pi05" else 8,
        ep=NS(task_id=0, init=0), ep_meta=dict(uid=f"{model}_{coordinator}_{pvalue}:eval:0:0", attempt=1), _s1_ms=1.,
        stage1_calls=1, cur_obs={}, b_aex=NS(a=np.zeros((1, H, 32), np.float32)),
        b_v0=NS(a=np.ones((1, 8))), b_v1=NS(a=np.ones((1, 8))), b_rs=NS(a=np.zeros((1, 8))), b_raw=NS(a=np.zeros((1, 8))))
    def search(ctx):
        s._dec = dict(step=0, hit=True, served=cache, lib="current", q_us=1., wire_actions=cache)
    s.on_search, s.after_infer = search, lambda *args: None
    s.on_executed = lambda x: s.b_aex.a.__setitem__(0, x)
    s.wire_diag = lambda: {}
    p = NS(_trace=None, _warm_reset=None)
    if model == "pi05":
        p._stage1_fn, p._stage2_fn, p._stage3_fn = f1, f2, f3
        p._model, p._stage3_device, p._stage_config = NS(run_stage3=f3, sample_noise=sample), "cpu", None
    else:
        from openpi.cache.groot.staged import GrootStage2Output
        def g2(s):
            calls["stage2"] += 1
            return GrootStage2Output(backbone_features=torch.zeros(1, 2, 2), attention_mask=None)
        class Head:
            training = False
            num_inference_timesteps = 8
            def process_backbone_output(self, inputs):
                return {"backbone_features": inputs["backbone_features"]}
        p._runner = NS(run_stage1=f1, run_stage2_llm=g2, run_stage3=f3,
            run_stage2=lambda _: (_ for _ in ()).throw(AssertionError("second full forward")),
            session=contextlib.nullcontext, _model=NS(action_head=Head()),
            _head_inputs=lambda s: {"backbone_features": s.backbone_features})
    adapter = NS(policy=p, fake=False, prepare=lambda obs: (np.zeros(8), None),
                 output=lambda chunk, state: {"actions": chunk[:, :7].copy()})
    conn = NS(_osp_sessions=[s], _osp_adapter=adapter)
    engine = Engine(conn)
    value = p._stage1_fn(torch.zeros(1)) if model == "pi05" else p._runner.run_stage1(torch.zeros(1))
    s.on_search(None)
    shadow = engine.profile["policy_chunk"].copy()
    if pvalue:
        if model == "pi05":
            st2 = p._stage2_fn(value)
            if coordinator:
                st3 = p._stage3_fn(st2, noise=p._model.sample_noise((1, H, 32), "cpu"))
            else:
                st3 = p._model.run_stage3(st2)
            chosen = st3.action_chunk[0].numpy()
        else:
            chosen = p._runner.run_stage2(value).action_pred[0].numpy()
        assert chosen.tobytes() == shadow.tobytes(), "resampling replaced the execution draw"
    else:
        chosen = cache
    s.on_executed(chosen)
    s._dec["wire_actions"] = chosen[:, :7]
    s.after_infer(5., True)
    assert calls == dict(stage1=1, stage2=1, stage3=4), calls
    assert torch.equal(saved, torch.random.get_rng_state())
    dec = next(r for r in emitted if r["ev"] == "p3_decision")
    assert dec["stage_invocations"] == calls
    assert engine.stage1 is engine.stage2 is engine.stage3 is engine.noise is None
    anchor_counts = dict(calls)
    class Builder:
        def __init__(self):
            self._cache = {"live": torch.ones(1)}
        def collect(self, checkpoint_id, **kw):
            self._cache["shadow"] = kw["stage1"]
        def build(self, checkpoint_id):
            return {k: torch.ones(8) for k in ("vision_0", "vision_1", "robot_state")}
        def clear(self):
            self._cache.clear()
    s.kb = Builder()
    cache_id = id(s.kb._cache)
    if model == "pi05":
        p._coordinator, p._pytorch_device = NS() if coordinator else None, "cpu"
        p._input_transform = lambda obs: {"image": {"test": np.zeros((2, 2, 3), np.uint8)},
            "image_mask": {"test": np.array(True)}, "state": np.zeros(32, np.float32)}
    else:
        p._policy = NS(apply_transforms=lambda x: x)
    size = 256 if model == "groot" else 224
    obs = {"observation/image": np.zeros((size, size, 3), np.uint8),
           "observation/wrist_image": np.zeros((size, size, 3), np.uint8),
           "observation/state": np.zeros(8), "prompt": "test"}
    shadow2, keys = engine.blind_probe(obs, 725)
    assert shadow2.shape == (H, 32) and set(keys) == {"vision_0", "vision_1", "robot_state"}
    assert id(s.kb._cache) == cache_id and set(s.kb._cache) == {"live"}
    assert calls == dict(stage1=2, stage2=2, stage3=5), calls
    assert torch.equal(saved, torch.random.get_rng_state())
    engine.last_log_ok = True
    def fail_log(*args):
        raise RuntimeError("injected logging failure")
    engine.decision = fail_log
    try:
        s.after_infer(1., True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("logging failure suppressed")
    assert not engine.last_log_ok and engine.stage1 is engine.stage2 is engine.stage3 is engine.noise is None
    return dict(model=model, coordinator=coordinator, p=pvalue, measured_counts=anchor_counts,
                blind_counts=dict(stage1=1, stage2=1, stage3=1), live_key_builder_unchanged=True,
                global_torch_rng_unchanged=True, primary_shadow_preserved=True, logging_failure_closed=True)


def controls_test(root):
    class Env:
        def __init__(self):
            self.env = self
            self.timestep, self.cur_time = 0, 0.
            self.state = np.zeros(12)
            self.robots = []
            self.sim = NS(data=NS(ctrl=np.zeros(7), qpos=self.state, qvel=np.zeros(12)))
            self.sim.step = lambda: None
        def get_sim_state(self):
            return self.state.copy()
        def set_init_state(self, state):
            self.state[:] = state
            return {"robot0_gripper_qpos": self.state[6:8].copy()}
        def step(self, action):
            self.sim.data.ctrl[:] = np.clip(action, -1, 1)*2
            for _ in range(3):
                self.sim.step()
            self.state[:7] += action
            self.timestep += 1
            self.cur_time += .05
            return {"robot0_gripper_qpos": self.state[6:8].copy()}, 0., self.timestep == 7, {}
    class Inner:
        n = 0
        def episode_start(self, **kw):
            pass
        def infer(self, obs):
            n = self.n
            self.n += 1
            return dict(actions=np.full((10, 7), .125, np.float32),
                __hit_meta__={"factor_outputs": {"osplug": {"p3_anchor": n == 0, "p3_step": n}}}, __p3__=dict(
                p3_version=2, p3_anchor=n == 0, p3_step=n, parent_anchor=0,
                source="cache" if n == 0 else "cache_blind", commit_controls=10))
    c = Client(Inner(), root)
    c.episode_start(task="test", extra_metadata=dict(task_uid="test:eval:0:0", attempt=1))
    env = Env()
    c.env, c.max_env_timestep = env, 10
    tapped = EnvTap(env, c)
    tapped.set_init_state(np.zeros(12))
    obs = {"observation/image": np.zeros((2, 2, 3), np.uint8), "observation/wrist_image": np.zeros((2, 2, 3), np.uint8),
           "observation/state": np.zeros(8), "prompt": "test"}
    actions = []
    for n in (5, 2):
        response = c.infer(obs)
        for action in response["actions"][:n]:
            actions.append(action.copy())
            tapped.step(action.tolist())
    c.emit(dict(ev="rollout_end", success=True, controls=7, final_chunk_completed=2))
    c.file.close()
    controls, decisions, end, by_step = control_join(c.path / "controls.jsonl", ("test:eval:0:0", 1),
        {"dec": {0: {}, 1: {}}}, {"success": True})
    assert len(controls) == 7 and [len(by_step[j]) for j in (0, 1)] == [5, 2]
    assert all(len(r["actuator_ctrl_each_physics_step"]) == 3 for r in controls)
    assert np.array_equal(env.state[:7], np.sum(actions, axis=0))
    assert len(list(c.path.glob("step_*.npz"))) == 1
    # Loss audit refuses truncated or altered actuation.
    rows = [json.loads(x) for x in (c.path / "controls.jsonl").read_text().splitlines()]
    bad = root / "bad.jsonl"
    for row in rows:
        if row["ev"] == "control":
            row["action_issued"][0] += 1
            break
    bad.write_text("".join(json.dumps(x) + "\n" for x in rows))
    try:
        control_join(bad, ("test:eval:0:0", 1), {"dec": {0: {}, 1: {}}}, {"success": True})
    except ValueError:
        pass
    else:
        raise AssertionError("altered actual control accepted")
    return dict(controls=7, physics_substeps=21, decisions=2, snapshots=1, partial_terminal_controls=2,
                altered_control_rejected=True, factual_actions_unchanged=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    root = ap.parse_args().out
    root.mkdir(exist_ok=False)
    staged_rows = [staged(m, c, p, root / "staged") for m, c in (("pi05", False), ("pi05", True), ("groot", False)) for p in (0, 1)]
    # Test marginal immediate propensities by actual randomization, not a second
    # implementation of the design's branch logic.
    calls, delays, durations = [], [], []
    for init in range(12000):
        d = Design(dict(cap=1, delays=[0, 1, 2]))
        a = d.resolve(assign(603, 0, init, 0, 0, .3), True)
        calls.append(a["executed_policy"])
        assert abs(a["actual_propensity"] - .1) < 1e-12
        for step in (2, 4, 6, 8):
            b = d.resolve(assign(603, 0, init, 0, step, .3), True)
        assert d.spent <= 1
        z = Design(dict(pre_guard=True, durations=[5, 10], holds=[1, 2, 3]))
        r = z.resolve(assign(603, 0, init, 0, 0, .5), False)
        durations.append(r["duration_choice"])
        assert r["eligible"] and r["actual_propensity"] == .5
    assert .09 < np.mean(calls) < .11 and .48 < np.mean(np.asarray(durations) == 5) < .52
    assert icc([[0, 0], [1, 1], [0, 0], [1, 1]])["icc"] == 1
    fixed = icc([[0, 1], [2, 3], [100, 101], [102, 103]], [0, 0, 1, 1])
    assert np.isclose(fixed["icc"], 7/9) and fixed["between_df"] == 2
    unequal = icc([[0, 2], [1, 3, 5], [100, 102], [101, 103, 105]], [0, 0, 1, 1])
    assert np.isclose(unequal["icc"], 11/71)
    assert icc([[0, 1], [100, 101]], [0, 1])["icc"] is None
    control = controls_test(root / "client")
    # The process may load both versions. The dispatch must still choose the
    # original Engine for an old fit/connection without profile_version=2.
    from . import v2
    from .test_units import stage_check
    coexist = [stage_check(m, c, p) for m, c in (("pi05", False), ("pi05", True), ("groot", False)) for p in (0, 1)]
    result = dict(PASS=True, staged=staged_rows, randomized_units=12000,
        delayed_immediate_rate=float(np.mean(calls)), duration5_rate=float(np.mean(np.asarray(durations) == 5)),
        controls=control, icc_perfect_repeat=1., icc_task_fixed_effects=fixed,
        icc_unequal_repeats=unequal, legacy_after_v2_import=coexist)
    (Path(__file__).parent / "results/units_v2.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
