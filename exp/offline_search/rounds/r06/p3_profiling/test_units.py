"""CPU staged-path, assignment, estimator and snapshot checks; no simulator."""
import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

from .assignment import assign
from .hooks import Engine, Proxy
from .snapshots import Client
from .estimate import effect


def stage_check(model, coordinator, treatment):
    H = 10 if model == "pi05" else 16
    calls = dict(s1=0, s2=0, s3=0)
    saved = torch.random.get_rng_state().clone()

    def s1(obs):
        calls["s1"] += 1
        return obs

    def s2(s):
        calls["s2"] += 1
        return s

    def s3(s, noise=None, **kw):
        calls["s3"] += 1
        return NS(action_chunk=noise + .1, action_pred=noise + .1)

    def sample(shape, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)

    cache = np.full((H, 32), .4, np.float32)
    assignment = assign(603, 0, 0, 0, 0, treatment)
    record = dict(schema="r6p3.anchor.v1", assignment=assignment, retrieval={}, guards={}, state={},
                  timing={}, extensions={})
    method = NS(p=treatment, seed=603, replicate=0, strata=None, record=record)
    emitted = []
    rt = NS(blind=True, policy_tail=True, policy_tail_blocks=1, gpu=None, randomized=False,
            judge=NS(mode="guard_only", cap=0, burst=1, step0="judge"), opts=NS(),
            model=model, suite="l10", H=H, tag="test", emit=emitted.append)
    session = NS(rt=rt, method=method, conn=1, miss_steps=10 if model == "pi05" else 8,
                 ep=NS(task_id=0, init=0), ep_meta=dict(uid="test:0:0", attempt=1), _s1_ms=1.,
                 b_aex=NS(a=np.zeros((1, H, 32), np.float32)))

    def search(ctx):
        session._dec = dict(step=0, hit=True, served=cache, lib="current", q_us=1., wire_actions=cache)
        return [1]

    session.on_search = search
    session.after_infer = lambda *args: None
    session.on_executed = lambda x: session.b_aex.a.__setitem__(0, x)
    session.wire_diag = lambda: {}
    p = NS(_trace=None, _warm_reset=None)
    if model == "pi05":
        p._stage1_fn, p._stage2_fn, p._stage3_fn = s1, s2, s3
        p._model = NS(run_stage3=s3, sample_noise=sample)
        p._stage3_device, p._stage_config = "cpu", None
    else:
        # Real dataclass shape; controlled minimal head/stages on CPU.
        from openpi.cache.groot.staged import GrootStage2Output
        def g2(s):
            calls["s2"] += 1
            return GrootStage2Output(backbone_features=torch.zeros(1, 2, 2), attention_mask=None)
        class Head:
            training = False
            num_inference_timesteps = 8
            def process_backbone_output(self, inputs):
                return {"backbone_features": inputs["backbone_features"]}
        def upstream(s):
            raise AssertionError("MISS should reuse the shadow result")
        p._runner = NS(run_stage1=s1, run_stage2_llm=g2, run_stage3=s3, run_stage2=upstream,
                       session=contextlib.nullcontext, _model=NS(action_head=Head()),
                       _head_inputs=lambda s: {"backbone_features": s.backbone_features})
    conn = NS(_osp_sessions=[session], _osp_adapter=NS(policy=p, fake=False))
    engine = Engine(conn)
    if model == "pi05":
        v = p._stage1_fn(torch.zeros(1))
    else:
        v = p._runner.run_stage1(torch.zeros(1))
    session.on_search(None)
    try:
        session.on_executed(cache + .25)
    except RuntimeError:
        pass
    else:
        raise AssertionError("wrong chunk reached history broadcast")
    if treatment:
        if model == "pi05":
            st2 = p._stage2_fn(v)
            if coordinator:
                # Real coordinator MISS prologue requests noise before submit.
                noise = p._model.sample_noise((1, H, 32), "cpu")
                st3 = p._stage3_fn(st2, noise=noise)
            else:
                st3 = p._model.run_stage3(st2)
            executed = st3.action_chunk[0].numpy()
        else:
            executed = p._runner.run_stage2(v).action_pred[0].numpy()
    else:
        executed = cache
    session.on_executed(executed)
    session._dec["wire_actions"] = executed
    session.after_infer(5., True)
    assert calls == dict(s1=1, s2=1, s3=1), calls
    assert torch.equal(saved, torch.random.get_rng_state()), "global torch RNG advanced"
    assert engine.stage1 is engine.stage2 is engine.stage3 is engine.noise is None
    assert emitted[-1]["cost"]["additional_forward_for_injection"] == 0
    assert np.array_equal(emitted[-1]["executed_chunk"], emitted[-1]["policy_chunk"] if treatment else cache)
    return dict(model=model, coordinator=coordinator, p=treatment, calls=calls,
                rng_unchanged=True, wrong_chunk_rejected_before_broadcast=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    root = ap.parse_args().out
    root.mkdir(exist_ok=False)
    from .method import loeo_cdf
    toy = NS(early=True, lib_ep=np.array([0, 0, 1, 1]), tasks={0: NS(
        rows=np.arange(4), Z=np.array([[0.], [1.], [10.], [11.]], np.float32),
        Z0=None, A0=np.array([[2.]], np.float32), As0=None)})
    cdf = loeo_cdf(toy, block=2)
    assert np.array_equal(cdf[0, 'main'], [9., 9., 10., 10.])
    assert np.array_equal(cdf[0, 'early'], [18., 18., 20., 20.])
    stage = [stage_check(m, c, p) for m, c in (("pi05", False), ("pi05", True), ("groot", False)) for p in (0, 1)]
    assignments = [assign(603, 3, i, rep, step, .2) for rep in range(4) for i in range(50) for step in range(0, 100, 2)]
    for a in reversed(assignments):
        assert a == assign(a["seed"], a["task_id"], a["init"], a["replicate"], a["step"], a["propensity"])
    assert .18 < np.mean([a["assigned_call"] for a in assignments]) < .22
    for bad in (-.1, 1.1, float("nan")):
        try:
            assign(1, 0, 0, 0, 0, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid p accepted")
    # Planted terminal effect, all eligible observations independently drawn;
    # five repeated anchors per init test cluster retention in bootstrap.
    rng = np.random.default_rng(17)
    effects = {}
    for delta in (0., .2):
        rows = []
        for i in range(1000):
            for step in range(5):
                a = assign(17, 0, i, 0, step, .2)
                y = rng.random() < .4 + delta * a["assigned_call"]
                rows.append(dict(arm="toy", task_id=0, init=i, Y=y,
                    **{"assignment.eligible": True, "assignment.assigned_call": a["assigned_call"],
                       "assignment.propensity": .2}))
        r = effect(rows, boot=400)
        assert r["n_clusters"] == 1000 and abs(r["delta"] - delta) < .05
        assert r["ci95"][0] <= delta <= r["ci95"][1]
        effects[str(delta)] = r
    class Inner:
        def episode_start(self, **kw):
            pass
        def infer(self, obs):
            return {"actions": np.ones((16, 7), np.float32), "diag": {"p3_anchor": True, "p3_step": 2}}
    client = Client(Inner(), root, every=1)
    client.env = NS(get_sim_state=lambda: np.arange(12), env=NS(timestep=20, cur_time=1., robots=[]))
    client.episode_start(extra_metadata=dict(task_uid="toy:eval:0:0", attempt=1), task="toy")
    obs = {"observation/image": np.zeros((2, 2, 3), np.uint8), "observation/wrist_image": np.zeros((2, 2, 3), np.uint8),
           "observation/state": np.zeros(8), "prompt": "toy"}
    result = client.infer(obs)
    path = next(root.glob("*/step_*.npz"))
    with np.load(path) as data:
        assert np.array_equal(data["sim_state"], np.arange(12))
        assert json.loads(str(data["metadata_json"]))["decision_step"] == 2
        assert np.array_equal(data["served_wire_chunk"], result["actions"])
    summary = dict(PASS=True, loeo_main_and_early_exclusion=True, staged=stage, assignments=len(assignments),
                   assigned_calls=sum(a["assigned_call"] for a in assignments), estimator=effects, snapshots=1)
    out = Path(__file__).parent / "results/units.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
