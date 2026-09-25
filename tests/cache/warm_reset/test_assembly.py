"""Serving assembly: ``_wrap_policy`` and the GR00T entry points (plan §4.8, §9.3, §9.4).

With a ``warm_reset`` block every cache-stack construction point yields
"evidence wrapper around an interceptor with the injected executor", with the
registered yaml id and the selected bundle id recorded independently; without
one, the built stack and the interceptor call are exactly today's.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import sys
import threading
import types

import pytest
import yaml

from openpi.cache.config import ConfigValidationError, load_cache_config
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.warm_reset.evidence import (
    GrootWarmResetEvidencePolicy,
    WarmResetEvidencePolicy,
)
from openpi.cache.warm_reset.groot import GrootWarmResetExecutor
from openpi.cache.warm_reset.pi05 import Pi05WarmResetExecutor
from openpi.cache.warm_reset.runtime import refuse_warm_reset
from scripts import serve_policy
from tests.cache.test_interceptor import FakePolicy
from tests.cache.warm_reset._support import block_of, pi05_config, spec_for


def _yaml(tmp_path, block) -> pathlib.Path:
    raw = {
        "enabled": True,
        "keys": {"robot_state": {"enabled": True, "weight": 1.0}},
        "key_builder": {"type": "placeholder"},
        "checkpoints": {"cp1": {"gate": {"type": "always_search"},
                                "judge": {"type": "always_warm_start", "start_t": 0.2},
                                "search_strategy": {"type": "weighted_rrf_knn", "top_k": 1}}},
        "backend": {"type": "in_memory", "vector_dims": {"robot_state": 32}},
        "write_policy": {"type": "never"},
    }
    if block is not None:
        raw["warm_reset"] = block
    path = tmp_path / ("arm_with_block.yaml" if block is not None else "arm.yaml")
    path.write_text(yaml.safe_dump(raw))
    return path


def _block(tmp_path, arm="selfwarmreset_t0.2", policy="pi05"):
    return block_of(spec_for(policy, arm, evidence_dir=str(tmp_path / "evidence")))


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_text().encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Pi0.5: scripts/serve_policy._wrap_policy
# ------------------------------------------------------------------


def test_startup_yaml_branch_wraps_and_injects(tmp_path):
    path = _yaml(tmp_path, _block(tmp_path))
    args = dataclasses.replace(serve_policy.Args(), cache_config=str(path))
    policy = serve_policy._wrap_policy(FakePolicy(), args, quiet=True, eager=True)
    assert isinstance(policy, WarmResetEvidencePolicy)
    inner = policy._inner
    assert isinstance(inner, InferenceInterceptor) and isinstance(inner._warm_reset, Pi05WarmResetExecutor)
    assert inner._warm_reset._session is policy._session  # one session, two users
    assert (policy._yaml_id, policy._bundle_id, policy._yaml_sha256) == (None, "default", _sha(path))
    assert "stage3_self_start" in inner._timer._probes


def test_bundle_branch_records_yaml_id_and_bundle_id_independently(tmp_path, monkeypatch):
    from openpi.cache.config import build_shared_storage
    from openpi.serving import websocket_policy_server as wps

    path = _yaml(tmp_path, _block(tmp_path, "warmreset_t0.2"))
    cfg = load_cache_config(path)
    bundle = types.SimpleNamespace(cache_config=cfg, shared_storage=build_shared_storage(cfg), yaml_id="arm-x",
                                   config_path=str(path))
    monkeypatch.setattr(wps, "get_current_cache_bundle", lambda bundle_id=None: bundle)
    policy = serve_policy._wrap_policy(FakePolicy(), serve_policy.Args(), quiet=True, eager=True,
                                       bundle_id="bundle-7")
    assert isinstance(policy, WarmResetEvidencePolicy)
    assert (policy._yaml_id, policy._bundle_id, policy._yaml_sha256) == ("arm-x", "bundle-7", _sha(path))
    assert "stage3_self_start" not in policy._inner._timer._probes


def test_record_keeps_the_policy_recorder_outermost(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _yaml(tmp_path, _block(tmp_path))
    args = dataclasses.replace(serve_policy.Args(), cache_config=str(path), record=True)
    policy = serve_policy._wrap_policy(FakePolicy(), args, quiet=True, eager=True)
    assert type(policy).__name__ == "PolicyRecorder"
    assert isinstance(policy._policy, WarmResetEvidencePolicy)


def test_absent_block_builds_exactly_the_interceptor(tmp_path):
    args = dataclasses.replace(serve_policy.Args(), cache_config=str(_yaml(tmp_path, None)))
    policy = serve_policy._wrap_policy(FakePolicy(), args, quiet=True, eager=True)
    assert type(policy) is InferenceInterceptor and policy._warm_reset is None
    assert not hasattr(policy, "_stage3_warm_reset_fn")
    assert not (tmp_path / "evidence").exists()


def test_refuse_warm_reset():
    refuse_warm_reset(pi05_config(None), where="x")
    refuse_warm_reset(None, where="x")
    with pytest.raises(ConfigValidationError, match="--trace-out"):
        refuse_warm_reset(pi05_config(block_of(spec_for("pi05", "warmreset_t0.2"))), where="--trace-out")


# ------------------------------------------------------------------
# GR00T entry points: seam fakes (the pattern of tests/robocasa365/test_groot_concurrent_serving.py)
# ------------------------------------------------------------------


class _FakeInterceptor:
    """Stands in for GrootCacheInterceptor and records every constructor keyword."""

    instances: list = []

    def __init__(self, policy, runner, **kwargs):
        self.policy, self.runner, self.kwargs = policy, runner, kwargs
        _FakeInterceptor.instances.append(self)

    def get_action(self, obs):
        return {}


def _components():
    return {"timer": types.SimpleNamespace(enable_csv=lambda d: None), "storage": object(), "key_builder": object(),
            "gates": {}, "judges": {}, "search_strategies": {}, "write_policy": object(), "offline_writers": (),
            "library_stats": None}


def _groot_config(tmp_path, block, k=4):
    cfg = pi05_config(block, start_t=0.5, denoise_schedule=f"groot_n15_k{k}_v1")
    cfg.key_builder.type = "cp1_groot_libero_mean"
    return cfg


def _policy(k=4):
    return types.SimpleNamespace(model=types.SimpleNamespace(action_head=types.SimpleNamespace(num_inference_timesteps=k)))


@pytest.fixture
def groot_seams(monkeypatch):
    import openpi.cache.config as cache_config
    import openpi.cache.groot.interceptor as gi
    import openpi.cache.groot.load_guard as lg
    import openpi.cache.groot.staged as gs
    import openpi.cache.orchestrator as orch

    state = types.SimpleNamespace(config=None, bundle=None)
    monkeypatch.setattr(cache_config, "load_cache_config", lambda path: state.config)
    monkeypatch.setattr(cache_config, "validate_cache_config", lambda c: None)
    monkeypatch.setattr(cache_config, "build_shared_storage", lambda c: object())
    monkeypatch.setattr(cache_config, "build_cache_components", lambda c: _components())
    monkeypatch.setattr(cache_config, "build_per_connection_components", lambda c, s, **kw: _components())
    monkeypatch.setattr(lg, "validate_groot_cache_config", lambda c, **kw: None)
    monkeypatch.setattr(lg, "validate_artifact_identity", lambda s, c: None)
    monkeypatch.setattr(orch, "CacheOrchestrator", lambda **kw: types.SimpleNamespace(**kw))
    monkeypatch.setattr(gs, "GrootStagedRunner", lambda model, **kw: types.SimpleNamespace(model=model))
    monkeypatch.setattr(gi, "GrootCacheInterceptor", _FakeInterceptor)
    from openpi.serving import websocket_policy_server as wps

    monkeypatch.setattr(wps, "get_current_cache_bundle", lambda bundle_id=None: state.bundle)
    _FakeInterceptor.instances = []
    return state


def _rc_args(path, **kw):
    ns = types.SimpleNamespace(cache_config=str(path), concurrent=True, compile_stage1=False, trace_out=None,
                               trace_build_cache=None, rit_shadow_out=None, allow_dynamic_bundles=False)
    ns.__dict__.update(kw)
    return ns


def _libero_args(path, **kw):
    ns = _rc_args(path, **kw)
    ns.__dict__.setdefault("online_state_dir", None)
    ns.__dict__.setdefault("loto_log_out", None)
    ns.__dict__.update(kw)
    return ns


def _assert_wired(served, *, yaml_id, bundle_id):
    assert isinstance(served, GrootWarmResetEvidencePolicy)
    interceptor = served._inner
    executor = interceptor.kwargs["warm_reset"]
    assert isinstance(executor, GrootWarmResetExecutor) and executor._session is served._session
    assert (served._yaml_id, served._bundle_id, served._k) == (yaml_id, bundle_id, 4)


def test_rc_single_connection_stack(tmp_path, groot_seams):
    from exp.robocasa365 import serve_groot_n15 as sgn

    groot_seams.config = _groot_config(tmp_path, _block(tmp_path, "selfmidreset_t0.5", "groot"))
    served, _ = sgn._build_served_policy(_policy(), _rc_args(tmp_path / "a.yaml"))
    _assert_wired(served, yaml_id=None, bundle_id="default")
    groot_seams.config = _groot_config(tmp_path, None)
    plain, _ = sgn._build_served_policy(_policy(), _rc_args(tmp_path / "a.yaml"))
    assert type(plain) is _FakeInterceptor and "warm_reset" not in plain.kwargs


def test_rc_rit_shadow_refuses_a_block(tmp_path, groot_seams):
    from exp.robocasa365 import serve_groot_n15 as sgn

    groot_seams.config = _groot_config(tmp_path, _block(tmp_path, "warmreset_t0.5", "groot"))
    with pytest.raises(ConfigValidationError, match="--rit-shadow-out"):
        sgn._build_served_policy(_policy(), _rc_args(tmp_path / "a.yaml", rit_shadow_out="x.jsonl"))


@pytest.mark.parametrize("entry", ["rc", "libero"])
def test_concurrent_factory_with_a_dynamic_bundle(tmp_path, groot_seams, entry):
    if entry == "rc":
        from exp.robocasa365 import serve_groot_n15 as srv
        from exp.robocasa365.groot_policy_adapter import GrootPolicyAdapter as Adapter
    else:
        from exp.libero_groot import serve_groot_libero as srv
        from exp.libero_groot.policy_adapter import GrootLiberoPolicyAdapter as Adapter
    groot_seams.config = _groot_config(tmp_path, None)  # the startup yaml has no block
    loaded = _groot_config(tmp_path, _block(tmp_path, "warmreset_t0.5", "groot"))
    groot_seams.bundle = types.SimpleNamespace(cache_config=loaded, shared_storage=object(), yaml_id="arm-x",
                                               config_path=None)
    args = (_rc_args if entry == "rc" else _libero_args)(tmp_path / "a.yaml", allow_dynamic_bundles=True)
    factory, _ = srv._build_concurrent_factory(_policy(), args)
    locked = factory(_policy(), "bundle-7")
    assert isinstance(locked, srv._InferLockedPolicy) and isinstance(locked._inner, Adapter)
    _assert_wired(locked._inner._policy, yaml_id="arm-x", bundle_id="bundle-7")
    groot_seams.bundle = None  # "default" -> the startup yaml, no block: today's stack
    plain = factory(_policy(), "default")
    assert type(plain._inner._policy) is _FakeInterceptor and "warm_reset" not in plain._inner._policy.kwargs


@pytest.mark.parametrize("entry", ["rc", "libero"])
def test_concurrent_trace_refuses_a_block(tmp_path, groot_seams, monkeypatch, entry):
    if entry == "rc":
        from exp.robocasa365 import serve_groot_n15 as srv
    else:
        from exp.libero_groot import serve_groot_libero as srv
    monkeypatch.setattr(srv, "_start_trace_coordinator", lambda policy: object())
    groot_seams.config = _groot_config(tmp_path, _block(tmp_path, "warmreset_t0.5", "groot"))
    args = (_rc_args if entry == "rc" else _libero_args)(tmp_path / "a.yaml", trace_out=str(tmp_path / "trace"))
    factory, _ = srv._build_concurrent_factory(_policy(), args)
    with pytest.raises(ConfigValidationError, match="--trace-out"):
        factory(_policy(), "default")
    assert _FakeInterceptor.instances == []


def test_libero_shadow_and_loto_factories_refuse_a_block(tmp_path, groot_seams):
    from exp.libero_groot import serve_groot_libero as sgl

    cfg = _groot_config(tmp_path, _block(tmp_path, "warmreset_t0.5", "groot"))
    with pytest.raises(ConfigValidationError, match="--rit-shadow-out"):
        sgl._build_shadow_factory(_libero_args(tmp_path / "a.yaml", rit_warm_ts="0.5"), cfg, object(),
                                  threading.Lock())
    with pytest.raises(ConfigValidationError, match="--loto-log-out"):
        sgl._build_loto_factory(_libero_args(tmp_path / "a.yaml"), cfg, object(), threading.Lock())


def test_libero_single_connection_stack(tmp_path, groot_seams, monkeypatch):
    """``serve_groot_libero.main`` without --concurrent, with gr00t and the server stubbed."""
    from exp.libero_groot import serve_groot_libero as sgl
    from exp.libero_groot.policy_adapter import GrootLiberoPolicyAdapter
    from openpi.serving import websocket_policy_server as wps

    gr00t = types.ModuleType("gr00t")
    gr00t.__spec__ = types.SimpleNamespace(name="gr00t")
    policy_mod = types.ModuleType("gr00t.model.policy")
    policy_mod.Gr00tPolicy = lambda **kw: types.SimpleNamespace(model=_policy(8).model, denoising_steps=8)
    data_mod = types.ModuleType("custom_data_config")
    data_mod.LiberoDataConfig = lambda: types.SimpleNamespace(modality_config=lambda: None, transform=lambda: None)
    for name, module in (("gr00t", gr00t), ("gr00t.model", types.ModuleType("gr00t.model")),
                         ("gr00t.model.policy", policy_mod), ("custom_data_config", data_mod)):
        monkeypatch.setitem(sys.modules, name, module)
    served = {}

    class _Server:
        def __init__(self, policy=None, **kw):
            served["policy"] = policy

        def serve_forever(self):
            return None

    monkeypatch.setattr(wps, "WebsocketPolicyServer", _Server)
    for block in (_block(tmp_path, "selfmidreset_t0.75_n1", "groot"), None):
        groot_seams.config = _groot_config(tmp_path, block, k=8)
        monkeypatch.setattr(sys, "argv", ["serve_groot_libero.py", "--cache-config", str(tmp_path / "a.yaml"),
                                          "--port", "1", "--checkpoint", str(tmp_path)])
        sgl.main()
        adapter = served["policy"]
        assert isinstance(adapter, GrootLiberoPolicyAdapter)
        if block is None:
            assert type(adapter._policy) is _FakeInterceptor and "warm_reset" not in adapter._policy.kwargs
        else:
            assert isinstance(adapter._policy, GrootWarmResetEvidencePolicy)
            assert isinstance(adapter._policy._inner.kwargs["warm_reset"], GrootWarmResetExecutor)
            assert adapter._policy._k == 8
