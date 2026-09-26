"""GR00T: the ``miss`` step assertion and the library-free self start (``trigger: always``).

GR00T's MISS runs the live head's ``num_inference_timesteps`` (process level),
so a ``miss`` block is an assertion checked before every MISS and reported as
``miss_nfe``; a library-free self arm is bit-equal to the same self arm served
over a library (stub head, real ``GrootStagedRunner``). The load guard admits
both library-free recipes without a checkpoint or an artifact and checks their
loop identity against the live head.
"""

from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest

from openpi.cache.components.judge import HitType
from openpi.cache.config import (
    CacheConfig,
    CheckpointConfig,
    ConfigValidationError,
    MissConfig,
    SearchStrategyConfig,
    WarmResetConfig,
    WritePolicyConfig,
    _dict_to_dataclass,
)
from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot.load_guard import (
    validate_artifact_identity,
    validate_groot_cache_config,
)
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.warm_reset.evidence import ExpectedEpisode, episode_problems
from openpi.cache.warm_reset.groot import GrootWarmResetExecutor, build_groot_warm_reset
from openpi.cache.warm_reset.runtime import WarmResetSession
from openpi.cache.warm_reset.types import HIT_SELF_ONLY, MissSpec
from tests.cache.warm_reset._arms import block_of, spec_for

from .conftest import StubGrootModel
from .test_groot_interceptor import _obs, _StubOrchestrator, _StubPolicy
from .test_warm_reset_groot import EPISODE, EXTRA, _model, _served_production


def _disabled_cp1() -> dict:
    return {"cp1": CheckpointConfig(enabled=False, search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"))}


def _miss_config(tmp_path, steps: int, k: int = 8) -> CacheConfig:
    return CacheConfig(enabled=True, denoise_schedule=f"groot_n15_k{k}_v1", checkpoints=_disabled_cp1(),
                       write_policy=WritePolicyConfig(type="never"),
                       miss=MissConfig(num_steps=steps, evidence_dir=str(tmp_path / "evidence")))


def _self_only_config(tmp_path, arm: str, t: float, k: int) -> tuple:
    spec = dataclasses.replace(spec_for("groot", arm, evidence_dir=str(tmp_path / "evidence")),
                               trigger="always", start_t=t)
    block = {**block_of(spec), "trigger": "always", "start_t": t}
    cfg = CacheConfig(enabled=True, denoise_schedule=f"groot_n15_k{k}_v1", checkpoints=_disabled_cp1(),
                      write_policy=WritePolicyConfig(type="never"),
                      warm_reset=_dict_to_dataclass(WarmResetConfig, block))
    return spec, cfg


def _miss_model(k: int) -> StubGrootModel:
    """The upstream-shaped head (``get_action``: the production MISS loop) at live step count ``k``."""
    model = StubGrootModel()
    model.action_head.num_inference_timesteps = k
    return model


def _served_miss(tmp_path, steps: int, live_k: int):
    model = _miss_model(live_k)
    runner = GrootStagedRunner(model, verify_upstream=False)
    parts = build_groot_warm_reset(_miss_config(tmp_path, steps), bundle_id="b", yaml_id="y", yaml_path=None)
    icpt = GrootCacheInterceptor(_StubPolicy(model), runner, orchestrator=_StubOrchestrator(HitType.MISS),
                                 **parts.interceptor_kwargs())
    return parts.wrap(icpt), model, parts


def _expected(spec, *, n: int, start_t, k: int = 8) -> ExpectedEpisode:
    return ExpectedEpisode(
        task_uid=EXTRA["task_uid"], attempt=1, outcome=True, n_decisions=n, yaml_id="y", bundle_id="b",
        spec=spec, spec_digest=spec.digest(), yaml_sha256=None, schedule_id=f"groot_n15_k{k}_v1", k=k,
        start_t=start_t, identity={**EPISODE, **EXTRA},
    )


def _episode(served, n: int) -> list:
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    outs = [served.get_action(_obs()) for _ in range(n)]
    served.on_episode_end(True)
    return outs


def _rows(served) -> list:
    return [json.loads(line) for line in served.evidence_path.read_text().splitlines()]


# ------------------------------------------------------------------
# MISS step assertion
# ------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 8])
def test_miss_arm_reports_the_live_step_count_and_is_admitted(tmp_path, k):
    served, _, parts = _served_miss(tmp_path, k, k)
    outs = _episode(served, 3)
    assert parts.executor is None and parts.miss_num_steps == k
    assert all(o["__hit_meta__"]["miss_nfe"] == k and o["__hit_meta__"]["hit_type"] == "MISS" for o in outs)
    result = episode_problems(_rows(served), expected=_expected(parts.spec, n=3, start_t=None))
    assert not +result["problems"], result["problems"]
    assert result["miss_nfe"] == 3 * k


def test_miss_arm_on_a_server_with_another_step_count_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="process-level"):
        _served_miss(tmp_path, 2, 8)


def test_live_step_change_is_refused_before_any_head_call(tmp_path):
    served, model, _ = _served_miss(tmp_path, 8, 8)
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    served.get_action(_obs())
    calls = model.action_head.process_calls
    model.action_head.num_inference_timesteps = 4
    with pytest.raises(RuntimeError, match="action head runs 4"):
        served.get_action(_obs())
    assert model.action_head.process_calls == calls and model.action_head.calls == 1


def test_absent_block_keeps_the_miss_wire():
    model = _miss_model(8)
    runner = GrootStagedRunner(model, verify_upstream=False)
    icpt = GrootCacheInterceptor(_StubPolicy(model), runner, orchestrator=_StubOrchestrator(HitType.MISS))
    out = icpt.get_action(_obs())
    assert "miss_nfe" not in out["__hit_meta__"] and icpt._miss_num_steps is None


@pytest.mark.parametrize("kwargs,match", [
    ({"trace": types.SimpleNamespace(twins=None, plan=None), "trace_vision_fields": ("vision_0",)}, "trace"),
    ({"orchestrator": types.SimpleNamespace(has_checkpoint=lambda cp: cp.name == "CP2", artifact_meta={})}, "CP2"),
    ({"orchestrator": types.SimpleNamespace(has_checkpoint=lambda cp: False, continuation_spec=lambda cp: object())},
     "online_rit"),
])
def test_interceptor_refuses_miss_steps_with_trace_cp2_and_online_rit(kwargs, match):
    model = _model(8)
    runner = GrootStagedRunner(model, verify_upstream=False)
    with pytest.raises(ValueError, match=match):
        GrootCacheInterceptor(_StubPolicy(model), runner, miss_num_steps=8, **kwargs)


# ------------------------------------------------------------------
# Library-free self start
# ------------------------------------------------------------------


@pytest.mark.parametrize("k,arm,t", [
    (4, "selfwarmreset_t0.75", 0.75), (4, "selfmidfinal_t0.5", 0.5), (4, "selfwarmshoot_t0.75", 0.75),
    (8, "selfmidreset_t0.75_n1", 0.75), (8, "selfresetfinal_t0.5_n2", 0.5),
])
def test_library_free_self_arm_equals_the_library_self_arm(tmp_path, k, arm, t):
    seed = 4242
    ref, _, _, _ = _served_production(tmp_path / "ref", arm, k, t, seed)
    _, cfg = _self_only_config(tmp_path, arm, t, k)
    parts = build_groot_warm_reset(cfg, bundle_id="b", yaml_id="y", yaml_path=None)
    parts.session.self_seed = lambda: seed
    model = _model(k)
    served = parts.wrap(GrootCacheInterceptor(_StubPolicy(model), GrootStagedRunner(model, verify_upstream=False),
                                              orchestrator=_StubOrchestrator(HitType.MISS),
                                              **parts.interceptor_kwargs()))
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    for _ in range(2):
        a, b = ref.get_action(_obs()), served.get_action(_obs())
        assert np.array_equal(a["action.out"], b["action.out"])
        assert b["__hit_meta__"]["hit_type"] == HIT_SELF_ONLY and b["__hit_meta__"]["start_t"] == t
        wa = {key: v for key, v in a["__hit_meta__"]["warm_reset"].items() if key != "spec_digest"}
        wb = {key: v for key, v in b["__hit_meta__"]["warm_reset"].items() if key != "spec_digest"}
        assert wa == wb


def test_library_free_self_arm_is_admitted_with_k_plus_n(tmp_path):
    spec, cfg = _self_only_config(tmp_path, "selfmidreset_t0.75_n1", 0.75, 8)
    parts = build_groot_warm_reset(cfg, bundle_id="b", yaml_id="y", yaml_path=None)
    model = _model(8)
    served = parts.wrap(GrootCacheInterceptor(_StubPolicy(model), GrootStagedRunner(model, verify_upstream=False),
                                              orchestrator=_StubOrchestrator(HitType.MISS),
                                              **parts.interceptor_kwargs()))
    _episode(served, 3)
    result = episode_problems(_rows(served), expected=_expected(spec, n=3, start_t=0.75))
    assert not +result["problems"], result["problems"]
    assert (result["continuation_nfe"], result["self_start_nfe"], result["total_nfe"]) == (3, 24, 27)


def test_self_only_needs_its_schedule_and_refuses_a_verdict(tmp_path):
    spec, cfg = _self_only_config(tmp_path, "selfwarmreset_t0.75", 0.75, 4)
    with pytest.raises(ValueError, match="schedule"):
        GrootWarmResetExecutor(spec, WarmResetSession(spec))
    parts = build_groot_warm_reset(cfg, bundle_id="b", yaml_id="y", yaml_path=None)
    model = _model(4)
    icpt = GrootCacheInterceptor(_StubPolicy(model), GrootStagedRunner(model, verify_upstream=False),
                                 orchestrator=_StubOrchestrator(HitType.WARM_START), **parts.interceptor_kwargs())
    served = parts.wrap(icpt)
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    with pytest.raises(RuntimeError, match="must enable no checkpoint"):
        served.get_action(_obs())


def test_self_only_on_a_head_with_another_loop_is_refused_before_the_head(tmp_path):
    _, cfg = _self_only_config(tmp_path, "selfwarmreset_t0.75", 0.75, 4)
    parts = build_groot_warm_reset(cfg, bundle_id="b", yaml_id="y", yaml_path=None)
    model = _model(8)
    served = parts.wrap(GrootCacheInterceptor(_StubPolicy(model), GrootStagedRunner(model, verify_upstream=False),
                                              orchestrator=_StubOrchestrator(HitType.MISS),
                                              **parts.interceptor_kwargs()))
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    with pytest.raises(RuntimeError, match="action head is"):
        served.get_action(_obs())
    assert model.action_head.process_calls == 0


# ------------------------------------------------------------------
# Load guard
# ------------------------------------------------------------------


def test_guard_admits_library_free_recipes_without_checkpoint_or_artifact(tmp_path):
    miss = _miss_config(tmp_path, 2)
    validate_groot_cache_config(miss, num_inference_timesteps=2)
    validate_artifact_identity(types.SimpleNamespace(artifact_meta=None), miss)
    _, self_only = _self_only_config(tmp_path, "selfwarmreset_t0.75", 0.75, 8)
    validate_groot_cache_config(self_only, num_inference_timesteps=8)
    validate_artifact_identity(types.SimpleNamespace(artifact_meta=None), self_only)


def test_guard_checks_the_library_free_loop_against_the_live_head(tmp_path):
    with pytest.raises(ConfigValidationError, match="process-level"):
        validate_groot_cache_config(_miss_config(tmp_path, 2), num_inference_timesteps=8)
    _, self_only = _self_only_config(tmp_path, "selfwarmreset_t0.75", 0.75, 8)
    with pytest.raises(ConfigValidationError, match="runs 4 steps"):
        validate_groot_cache_config(self_only, num_inference_timesteps=4)
    self_only.denoise_schedule = None
    with pytest.raises(ConfigValidationError, match="must name denoise_schedule"):
        validate_groot_cache_config(self_only, num_inference_timesteps=8)


def test_guard_still_refuses_an_empty_checkpoint_set_without_a_library_free_block():
    cfg = CacheConfig(enabled=True, checkpoints=_disabled_cp1())
    with pytest.raises(ConfigValidationError, match="enabled checkpoints must be exactly"):
        validate_groot_cache_config(cfg)
    with pytest.raises(ConfigValidationError, match="artifact identity"):
        validate_artifact_identity(types.SimpleNamespace(artifact_meta=None), cfg)


def test_miss_spec_digest_names_the_arm(tmp_path):
    parts = build_groot_warm_reset(_miss_config(tmp_path, 1), bundle_id="b", yaml_id="y", yaml_path=None)
    assert parts.spec == MissSpec(num_steps=1, evidence_dir=str(tmp_path / "evidence"))
    assert parts.interceptor_kwargs() == {"miss_num_steps": 1}
