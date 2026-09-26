"""Per-bundle MISS steps (``miss:``) and the library-free self start (``trigger: always``), Pi0.5.

* config: both blocks validate with their exclusions; absent blocks keep every
  existing spec digest (frozen values from the pre-change code);
* serving: a ``miss`` bundle runs its own step count on the direct and the
  coordinator MISS path and reports it additively (``miss_nfe``); without the
  block the MISS reads the module ``_NUM_STEPS`` at call time (the step_diag
  process pin still applies) and the wire has no new key;
* a library-free self arm is bit-equal to the same self arm served over a
  library, carries ``SELF_ONLY`` and is admitted with K + N pricing;
* evidence admission of both arm kinds, positive and negative.
"""

from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest
import yaml

from exp.warm_reset.envs import get_env
from exp.warm_reset.plan import library_free_base
from openpi.cache import interceptor as icpt_module
from openpi.cache.config import (
    ConfigValidationError,
    build_shared_storage,
    is_library_free,
    load_cache_config,
)
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.types import PI05_V1
from openpi.cache.warm_reset.evidence import ExpectedEpisode, episode_problems
from openpi.cache.warm_reset.pi05 import Pi05WarmResetExecutor
from openpi.cache.warm_reset.runtime import WarmResetSession, refuse_warm_reset
from openpi.cache.warm_reset.types import HIT_SELF_ONLY, MissSpec, WarmResetSpec
from scripts import serve_policy
from tests.cache.test_interceptor import FakePolicy
from tests.cache.warm_reset._support import (
    EPISODE,
    EXTRA,
    Pi05Model,
    SyncCoordinator,
    block_of,
    obs_for,
    pi05_stack,
    read_rows,
    run_episode,
    spec_for,
)

ENV = get_env("pi05_libero_10")


# ------------------------------------------------------------------
# yaml helpers
# ------------------------------------------------------------------


def _miss_raw(tmp_path, steps: int) -> dict:
    raw = library_free_base(ENV)
    raw["miss"] = {"num_steps": steps, "evidence_dir": str(tmp_path / "evidence")}
    return raw


def _self_only_block(tmp_path, arm: str, t: float) -> dict:
    block = block_of(spec_for("pi05", arm, evidence_dir=str(tmp_path / "evidence")))
    return {**block, "trigger": "always", "start_t": t}


def _self_only_raw(tmp_path, arm: str, t: float) -> dict:
    raw = library_free_base(ENV)
    raw["warm_reset"] = _self_only_block(tmp_path, arm, t)
    return raw


def _load(tmp_path, raw: dict, name: str = "arm.yaml"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_cache_config(path), path


def _served(tmp_path, monkeypatch, raw, *, model=None, coordinator=False, yaml_id="y", bundle_id="b"):
    """The stack ``_wrap_policy`` serves for a hot-loaded bundle of ``raw`` (eager, CPU)."""
    from openpi.serving import websocket_policy_server as wps

    cfg, path = _load(tmp_path, raw, f"{yaml_id}.yaml")
    bundle = types.SimpleNamespace(cache_config=cfg, shared_storage=build_shared_storage(cfg), yaml_id=yaml_id,
                                   config_path=str(path))
    monkeypatch.setattr(wps, "get_current_cache_bundle", lambda bundle_id=None: bundle)
    model = model or Pi05Model()
    coord = SyncCoordinator(model) if coordinator else None
    served = serve_policy._wrap_policy(
        FakePolicy(model), serve_policy.Args(), quiet=True, eager=True,
        shared_cache={"coordinator": coord} if coord is not None else None, bundle_id=bundle_id,
    )
    return types.SimpleNamespace(served=served, model=model, coordinator=coord, cfg=cfg, path=path,
                                 yaml_id=yaml_id, bundle_id=bundle_id)


def _spy_stage3(model: Pi05Model) -> list:
    seen = []
    inner = model.run_stage3

    def spy(stage2, **kw):
        seen.append(kw.get("num_steps"))
        return inner(stage2, **kw)

    model.run_stage3 = spy
    return seen


def _episode(served, n: int, *, success: bool = True) -> list:
    served.on_task_begin()
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    outs = [served.infer(obs_for(None)) for _ in range(n)]
    served.on_episode_end(success)
    return outs


def _expected(stack, spec, *, n: int, start_t, outcome: bool = True, **overrides) -> ExpectedEpisode:
    import hashlib

    fields = dict(
        task_uid=EXTRA["task_uid"], attempt=EXTRA["attempt"], outcome=outcome, n_decisions=n,
        yaml_id=stack.yaml_id, bundle_id=stack.bundle_id, spec=spec, spec_digest=spec.digest(),
        yaml_sha256=hashlib.sha256(stack.path.read_text().encode("utf-8")).hexdigest(),
        schedule_id=PI05_V1.schedule_id, k=PI05_V1.num_steps, start_t=start_t, identity={**EPISODE, **EXTRA},
    )
    fields.update(overrides)
    return ExpectedEpisode(**fields)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------


def test_verdict_spec_digests_are_the_pre_change_values():
    """A spec without the new fields hashes exactly as before (frozen HEAD values)."""
    self_spec = WarmResetSpec(source="self", point="snapshot", kind="reset", level=1.0, num_steps=None,
                              seed_namespace="ns", seed_keys=("experiment", "task", "orig_init_state_idx", "attempt"),
                              evidence_dir="/ev")
    cache_spec = WarmResetSpec(source="cache", point="final", kind="reset", level=0.9, num_steps=2,
                               seed_namespace=None, seed_keys=(), evidence_dir="/ev")
    assert self_spec.digest() == "ee61a99e306b8f17593f5386ab86fc9804368e2755a5d8fe94e9434ff4321e0e"
    assert cache_spec.digest() == "2b0933fb0d2071b9b988b1e42778a94d8a9ebf26d446a6ac90fe12bea60a6cf3"
    always = dataclasses.replace(self_spec, trigger="always", start_t=0.2)
    assert always.digest() != self_spec.digest() and always.always and not self_spec.always


def test_absent_blocks_load_as_none(tmp_path):
    raw = library_free_base(ENV)
    cfg, _ = _load(tmp_path, raw)
    assert cfg.miss is None and cfg.warm_reset is None and not is_library_free(cfg)


def test_miss_block_round_trips(tmp_path):
    cfg, _ = _load(tmp_path, _miss_raw(tmp_path, 2))
    assert cfg.miss.num_steps == 2 and is_library_free(cfg)
    spec = MissSpec.from_config(cfg.miss)
    assert spec == MissSpec(num_steps=2, evidence_dir=str(tmp_path / "evidence")) and not spec.self_start
    with pytest.raises(ValueError):
        spec.seed_policy()


def test_self_only_block_round_trips(tmp_path):
    cfg, _ = _load(tmp_path, _self_only_raw(tmp_path, "selfwarmreset_t0.2", 0.2))
    spec = WarmResetSpec.from_config(cfg.warm_reset)
    assert spec.always and spec.start_t == 0.2 and spec.self_start and is_library_free(cfg)


@pytest.mark.parametrize("mutate,match", [
    (lambda raw: raw["miss"].update(num_steps=0), "miss.num_steps"),
    (lambda raw: raw["miss"].update(num_steps=True), "miss.num_steps"),
    (lambda raw: raw["miss"].update(evidence_dir=""), "miss.evidence_dir"),
    (lambda raw: raw["write_policy"].update(type="always"), "write_policy.type must be"),
    (lambda raw: raw.update(trace={"enabled": True, "out_dir": "/tmp/x"}), "trace"),
    (lambda raw: raw.update(shadow_teacher={"enabled": True, "path": "/tmp/s.jsonl"}), "shadow_teacher"),
    (lambda raw: raw.update(warm_reset={"start": {"source": "self", "point": "final"},
                                        "grid": {"kind": "reset", "entry_t": 1.0}, "evidence_dir": "/tmp/e",
                                        "self_seed": {"namespace": "n"}, "trigger": "always",
                                        "start_t": 0.2}), "mutually exclusive"),
])
def test_miss_block_rejections(tmp_path, mutate, match):
    raw = _miss_raw(tmp_path, 2)
    mutate(raw)
    with pytest.raises(ConfigValidationError, match=match):
        _load(tmp_path, raw)


def test_miss_at_k_may_keep_a_writing_policy(tmp_path):
    raw = _miss_raw(tmp_path, PI05_V1.num_steps)
    raw["write_policy"] = {"type": "always"}
    _load(tmp_path, raw)  # full K: a written MISS entry is stamped with the right loop


@pytest.mark.parametrize("mutate,match", [
    (lambda raw: raw["warm_reset"].update(trigger="sometimes"), "trigger"),
    (lambda raw: raw["warm_reset"].pop("start_t"), "start_t is required"),
    (lambda raw: raw["warm_reset"].update(start_t=0.25), "start_t"),
    (lambda raw: raw["checkpoints"]["cp1"].update(enabled=True, gate={"type": "always_search"},
                                                  judge={"type": "always_warm_start", "start_t": 0.2}),
     "disable every checkpoint"),
])
def test_self_only_rejections(tmp_path, mutate, match):
    raw = _self_only_raw(tmp_path, "selfwarmreset_t0.2", 0.2)
    mutate(raw)
    with pytest.raises(ConfigValidationError, match=match):
        _load(tmp_path, raw)


def test_self_only_needs_a_self_source(tmp_path):
    raw = library_free_base(ENV)
    raw["warm_reset"] = {**block_of(spec_for("pi05", "warmreset_t0.2", evidence_dir=str(tmp_path / "e"))),
                         "trigger": "always", "start_t": 0.2}
    with pytest.raises(ConfigValidationError, match="start.source: self"):
        _load(tmp_path, raw)


def test_verdict_block_refuses_a_start_t(tmp_path):
    raw = library_free_base(ENV)
    raw["checkpoints"]["cp1"].update(enabled=True, gate={"type": "always_search"},
                                     judge={"type": "always_warm_start", "start_t": 0.2})
    raw["warm_reset"] = {**block_of(spec_for("pi05", "warmreset_t0.2", evidence_dir=str(tmp_path / "e"))),
                         "start_t": 0.2}
    with pytest.raises(ConfigValidationError, match="only valid with trigger: always"):
        _load(tmp_path, raw)


def test_refuse_warm_reset_also_refuses_a_miss_block(tmp_path):
    cfg, _ = _load(tmp_path, _miss_raw(tmp_path, 2))
    with pytest.raises(ConfigValidationError, match="miss block"):
        refuse_warm_reset(cfg, where="--trace-out")


# ------------------------------------------------------------------
# Serving: per-bundle MISS steps
# ------------------------------------------------------------------


@pytest.mark.parametrize("coordinator", [False, True])
@pytest.mark.parametrize("steps", [1, 2, 10])
def test_miss_bundle_runs_its_own_step_count(tmp_path, monkeypatch, coordinator, steps):
    stack = _served(tmp_path, monkeypatch, _miss_raw(tmp_path, steps), coordinator=coordinator)
    seen = _spy_stage3(stack.model)
    outs = _episode(stack.served, 3)
    assert seen == [steps] * 3
    assert all(o["__hit_meta__"]["miss_nfe"] == steps and o["__hit_meta__"]["hit_type"] == "MISS" for o in outs)
    if coordinator:
        from openpi.serving.batching_coordinator import (
            Pi05StageBatcher,
            Stage3MissPayload,
        )

        misses = [p for _, _, p in stack.coordinator.calls if isinstance(p, Stage3MissPayload)]
        assert [p.num_steps for p in misses] == [steps] * 3
        assert Pi05StageBatcher.bucket_key(misses[0]) == ("miss", None, steps)


def test_two_bundles_on_one_model_keep_their_own_counts(tmp_path, monkeypatch):
    model = Pi05Model()
    a = _served(tmp_path / "a", monkeypatch, _miss_raw(tmp_path / "a", 2), model=model, yaml_id="a")
    b = _served(tmp_path / "b", monkeypatch, _miss_raw(tmp_path / "b", 3), model=model, yaml_id="b")
    seen = _spy_stage3(model)
    a.served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    b.served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    a.served.infer(obs_for(None))
    b.served.infer(obs_for(None))
    a.served.infer(obs_for(None))
    assert seen == [2, 3, 2]


def test_absent_block_reads_the_module_step_count_at_call_time(tmp_path, monkeypatch):
    """No ``miss`` block: MISS takes ``_NUM_STEPS`` when it runs (a process pin still applies), no new key."""
    raw = library_free_base(ENV)
    raw["checkpoints"]["cp1"].update(enabled=True, gate={"type": "always_search"}, judge={"type": "threshold"})
    raw["backend"]["vector_dims"] = {"robot_state": 8}  # the stub model's state; the library is empty
    stack = _served(tmp_path, monkeypatch, raw)
    assert type(stack.served) is InferenceInterceptor and stack.served._miss_num_steps is None
    seen = _spy_stage3(stack.model)
    stack.served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    out = stack.served.infer(obs_for(None))
    monkeypatch.setattr(icpt_module, "_NUM_STEPS", 3)
    stack.served.infer(obs_for(None))
    assert seen == [10, 3]
    assert "miss_nfe" not in out["__hit_meta__"]
    assert not (tmp_path / "evidence").exists()


def test_miss_evidence_is_admitted_and_priced(tmp_path, monkeypatch):
    stack = _served(tmp_path, monkeypatch, _miss_raw(tmp_path, 2))
    _episode(stack.served, 4, success=False)
    rows = read_rows(stack.served.evidence_path)
    spec = MissSpec.from_config(stack.cfg.miss)
    decisions = [r for r in rows if r["row_kind"] == "decision"]
    assert [r["miss_nfe"] for r in decisions] == [2] * 4 and all(r["warm_reset"] is None for r in decisions)
    result = episode_problems(rows, expected=_expected(stack, spec, n=4, start_t=None, outcome=False))
    assert not +result["problems"], result["problems"]
    assert (result["miss_nfe"], result["total_nfe"], result["continuation_nfe"]) == (8, 8, None)


@pytest.mark.parametrize("case,code", [
    ("steps", "steps_mismatch"),
    ("missing", "miss_nfe_missing"),
    ("hit", "hit_type_mismatch"),
    ("start_t", "schedule_mismatch"),
])
def test_miss_evidence_negatives(tmp_path, monkeypatch, case, code):
    stack = _served(tmp_path, monkeypatch, _miss_raw(tmp_path, 2))
    _episode(stack.served, 2)
    rows = read_rows(stack.served.evidence_path)
    spec = MissSpec.from_config(stack.cfg.miss)
    expected = _expected(stack, spec, n=2, start_t=None)
    decision = next(r for r in rows if r["row_kind"] == "decision")
    if case == "steps":
        other = MissSpec(num_steps=3, evidence_dir=spec.evidence_dir)
        rows = [dict(r, spec_digest=other.digest()) for r in rows]
        expected = _expected(stack, other, n=2, start_t=None)
    elif case == "missing":
        del decision["miss_nfe"]  # a server that ignores the block
    elif case == "hit":
        decision["hit_type"] = "WARM_START"
    else:
        decision["start_t"] = 0.2
    result = episode_problems(rows, expected=expected)
    assert result["problems"][code] >= 1 and result["total_nfe"] is None


def test_miss_expectation_itself_is_checked(tmp_path, monkeypatch):
    stack = _served(tmp_path, monkeypatch, _miss_raw(tmp_path, 2))
    _episode(stack.served, 1)
    rows = read_rows(stack.served.evidence_path)
    spec = MissSpec.from_config(stack.cfg.miss)
    bad = _expected(stack, spec, n=1, start_t=0.2)
    assert episode_problems(rows, expected=bad)["problems"]["invalid_expected"] == 1


@pytest.mark.parametrize("kwargs,match", [
    ({"miss_num_steps": 0}, "miss_num_steps"),
    ({"miss_num_steps": True}, "miss_num_steps"),
    ({"miss_num_steps": 2, "orchestrator": None}, "orchestrator"),
    ({"miss_num_steps": 2, "hit_executor": lambda obs: obs}, "hit_executor"),
    ({"miss_num_steps": 2, "shadow_teacher": object()}, "shadow"),
])
def test_interceptor_refuses_invalid_miss_steps(kwargs, match):
    from tests.cache.conftest import make_orchestrator

    orch, _, _ = make_orchestrator(vector_dims={"robot_state": 8})
    kw = {"orchestrator": orch, **kwargs}
    with pytest.raises(ValueError, match=match):
        InferenceInterceptor(FakePolicy(Pi05Model()), eager=True, **kw)


def test_interceptor_refuses_miss_steps_with_warm_reset():
    from tests.cache.conftest import make_orchestrator

    orch, _, _ = make_orchestrator(vector_dims={"robot_state": 8})
    spec = spec_for("pi05", "warmreset_t0.2")
    with pytest.raises(ValueError, match="mutually exclusive"):
        InferenceInterceptor(FakePolicy(Pi05Model()), orchestrator=orch, eager=True, miss_num_steps=2,
                             warm_reset=Pi05WarmResetExecutor(spec, WarmResetSession(spec)))


# ------------------------------------------------------------------
# Serving: library-free self start
# ------------------------------------------------------------------


@pytest.mark.parametrize("coordinator", [False, True])
@pytest.mark.parametrize("arm,t", [("selfwarmreset_t0.2", 0.2), ("selfresetfinal_t0.2", 0.2),
                                   ("selfmidfinal_t0.3", 0.3), ("selfmidreset50_t0.1", 0.1)])
def test_library_free_self_arm_equals_the_library_self_arm(tmp_path, monkeypatch, coordinator, arm, t):
    ref = pi05_stack(tmp_path / "ref", arm, t, coordinator=coordinator)
    raw = _self_only_raw(tmp_path, arm, t)
    raw["warm_reset"]["self_seed"]["namespace"] = "ns"  # the reference stack's namespace
    free = _served(tmp_path, monkeypatch, raw, coordinator=coordinator)
    ref_outs = run_episode(ref, 3)
    free_outs = _episode(free.served, 3)
    for a, b in zip(ref_outs, free_outs):
        assert np.array_equal(a["actions"], b["actions"])
        ma, mb = a["__hit_meta__"], b["__hit_meta__"]
        assert ma["hit_type"] == "WARM_START" and mb["hit_type"] == HIT_SELF_ONLY
        assert mb["start_t"] == t and mb["winner_id"] is None
        wa = {k: v for k, v in ma["warm_reset"].items() if k != "spec_digest"}
        wb = {k: v for k, v in mb["warm_reset"].items() if k != "spec_digest"}
        assert wa == wb
    if coordinator:
        assert not any(type(p).__name__ == "Stage3MissPayload" for _, _, p in free.coordinator.calls)


def test_library_free_self_arm_is_admitted_with_k_plus_n(tmp_path, monkeypatch):
    stack = _served(tmp_path, monkeypatch, _self_only_raw(tmp_path, "selfresetfinal_t0.2", 0.2))
    _episode(stack.served, 3)
    rows = read_rows(stack.served.evidence_path)
    spec = WarmResetSpec.from_config(stack.cfg.warm_reset)
    assert all(r["hit_type"] == HIT_SELF_ONLY for r in rows if r["row_kind"] == "decision")
    result = episode_problems(rows, expected=_expected(stack, spec, n=3, start_t=0.2))
    assert not +result["problems"], result["problems"]
    assert (result["continuation_nfe"], result["self_start_nfe"], result["total_nfe"]) == (6, 30, 36)
    # the same rows read as a verdict arm are rejected (hit type / spec identity)
    verdict = dataclasses.replace(spec, trigger="verdict", start_t=None)
    bad = episode_problems(rows, expected=_expected(stack, verdict, n=3, start_t=0.2))
    assert bad["problems"]["hit_type_mismatch"] == 3


def test_self_only_executor_refuses_the_wrong_entry():
    spec = spec_for("pi05", "selfwarmreset_t0.2")
    always = dataclasses.replace(spec, trigger="always", start_t=0.2)
    verdict_exec = Pi05WarmResetExecutor(spec, WarmResetSession(spec))
    always_exec = Pi05WarmResetExecutor(always, WarmResetSession(always))
    with pytest.raises(RuntimeError, match="run_self_only needs"):
        verdict_exec.run_self_only(stage2=None, action_shape=(10, 8), device="cpu", run_stage3=None, timer=None)
    with pytest.raises(RuntimeError, match="never a verdict"):
        always_exec.run(stage2=None, cp_result=None, snapshot_x=None, run_stage3=None, timer=None)
    assert always_exec.trigger_always and not verdict_exec.trigger_always


def test_library_free_yaml_needs_no_library_file(tmp_path, monkeypatch):
    """The bundle builds from the yaml alone: no preload_path, no artifact."""
    stack = _served(tmp_path, monkeypatch, _self_only_raw(tmp_path, "selfwarmreset_t0.2", 0.2))
    assert stack.cfg.backend.in_memory.preload_path in ("", None)
    assert json.loads(json.dumps(_episode(stack.served, 1)[0]["__hit_meta__"]["warm_reset"]))["self_start"] is True


def test_pi05_executor_refuses_a_foreign_schedule():
    from openpi.cache.types import groot_n15_schedule

    spec = spec_for("pi05", "warmreset_t0.2")
    with pytest.raises(ValueError, match="pi05_v1"):
        Pi05WarmResetExecutor(spec, WarmResetSession(spec), schedule=groot_n15_schedule(8))


def test_self_only_executor_needs_an_orchestrator():
    spec = dataclasses.replace(spec_for("pi05", "selfwarmreset_t0.2"), trigger="always", start_t=0.2)
    with pytest.raises(ValueError, match="needs an orchestrator"):
        InferenceInterceptor(FakePolicy(Pi05Model()), eager=True,
                             warm_reset=Pi05WarmResetExecutor(spec, WarmResetSession(spec)))
