"""serve_policy routing wiring tests: executor construction from the yaml
routing section, stage-placement rejection, and probe fail-fast (plan §9)."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import serve_policy  # noqa: E402

from openpi.cache.config import load_cache_config  # noqa: E402
from openpi.cache.sidecar_executor import SidecarError  # noqa: E402
from openpi.models_pytorch.stage_device_placement import StageDeviceConfig  # noqa: E402

ROUTED_YAML = pathlib.Path("exp/ablation_study/config/small_at_hit/libero_spatial_hit_smolvla.yaml")
BASELINE_YAML = pathlib.Path("exp/ablation_study/config/common/libero_spatial_baseline.yaml")


def test_no_routing_returns_none_pair():
    cfg = load_cache_config(BASELINE_YAML)
    assert serve_policy._build_routing_executors(cfg, None) == (None, None)


def test_meta_stage_placement_rejected():
    cfg = load_cache_config(ROUTED_YAML)
    stage_config = StageDeviceConfig(stage1="cpu", stage2="meta", stage3="meta")
    with pytest.raises(ValueError, match="stage placement"):
        serve_policy._build_routing_executors(cfg, stage_config)


def test_absent_sidecar_probe_fails_fast(monkeypatch):
    cfg = load_cache_config(ROUTED_YAML)
    # The emitted yaml points at 127.0.0.1:7001; nothing listens there in CI.
    monkeypatch.setattr(cfg.routing, "connect_timeout_s", 1.0)
    with pytest.raises(SidecarError):
        serve_policy._build_routing_executors(cfg, None)


def test_hit_vs_miss_slot_selection(monkeypatch):
    import openpi.cache.sidecar_executor as se

    monkeypatch.setattr(se, "probe_endpoint", lambda ep, timeout_s=5.0: None)
    monkeypatch.setattr(serve_policy, "probe_endpoint", lambda ep, timeout_s=5.0: None, raising=False)
    cfg_hit = load_cache_config(ROUTED_YAML)
    hit_ex, miss_ex = serve_policy._build_routing_executors(cfg_hit, None)
    assert hit_ex is not None and miss_ex is None
    hit_ex.close()
    cfg_miss = load_cache_config(
        "exp/ablation_study/config/small_at_miss/libero_spatial_miss_smolvla.yaml"
    )
    hit_ex2, miss_ex2 = serve_policy._build_routing_executors(cfg_miss, None)
    assert hit_ex2 is None and miss_ex2 is not None
    miss_ex2.close()


def _minimal_routed_yaml(tmp_path):
    """Minimal allowlist-conforming routed yaml with the placeholder key
    builder (cp1_* builders require an artifact preload; the wrapper-path
    tests only exercise executor construction, not retrieval quality)."""
    import yaml as _yaml

    raw = {
        "enabled": True,
        "keys": {
            "vision_0": {"enabled": False}, "vision_1": {"enabled": False},
            "vision_2": {"enabled": False}, "prompt_emb": {"enabled": False},
            "robot_state": {"enabled": True, "weight": 1.0},
        },
        "key_builder": {"type": "placeholder"},
        "checkpoints": {
            "cp1": {
                "gate": {"type": "always_search"},
                "judge": {"type": "threshold", "threshold": 0.9},
                "search_strategy": {"type": "weighted_rrf_knn", "top_k": 1},
            }
        },
        "backend": {"type": "in_memory", "vector_dims": {"robot_state": 32}},
        "write_policy": {"type": "never"},
        "routing": {"hit_to": "127.0.0.1:7001"},
    }
    p = tmp_path / "routed_minimal.yaml"
    p.write_text(_yaml.safe_dump(raw))
    return p


def test_static_cache_config_path_builds_executor(tmp_path, monkeypatch):
    import dataclasses

    import openpi.cache.sidecar_executor as se
    from tests.cache.test_interceptor import FakePolicy

    monkeypatch.setattr(se, "probe_endpoint", lambda ep, timeout_s=5.0: None)
    args = dataclasses.replace(serve_policy.Args(), cache_config=str(_minimal_routed_yaml(tmp_path)))
    policy = serve_policy._wrap_policy(FakePolicy(), args, quiet=True, eager=True)
    from openpi.cache.interceptor import InferenceInterceptor

    assert isinstance(policy, InferenceInterceptor)
    assert policy._hit_executor is not None and policy._miss_executor is None
    policy.on_task_end()  # deterministic close


def test_runtime_bundle_path_builds_executor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import openpi.cache.sidecar_executor as se
    from openpi.cache.config import build_shared_storage
    from openpi.cache.config import load_cache_config as _load
    from openpi.serving import websocket_policy_server as wps
    from tests.cache.test_interceptor import FakePolicy

    monkeypatch.setattr(se, "probe_endpoint", lambda ep, timeout_s=5.0: None)
    cfg = _load(_minimal_routed_yaml(tmp_path))
    storage = build_shared_storage(cfg)
    bundle = SimpleNamespace(cache_config=cfg, shared_storage=storage, yaml_id="arm-x")
    monkeypatch.setattr(wps, "get_current_cache_bundle", lambda bundle_id=None: bundle)
    policy = serve_policy._wrap_policy(FakePolicy(), serve_policy.Args(), quiet=True, eager=True)
    from openpi.cache.interceptor import InferenceInterceptor

    assert isinstance(policy, InferenceInterceptor)
    assert policy._hit_executor is not None
    policy.on_task_end()


def test_timing_csv_enabled_and_artifact_written(tmp_path, monkeypatch):
    import dataclasses

    import openpi.cache.sidecar_executor as se
    from openpi.serving import monitor as _monitor
    from tests.cache.test_interceptor import FakePolicy

    # SystemTimer is gated by OPENPI_MONITOR_LEVEL (below BASIC => no-op); the
    # latency pass runs the server with BASIC+, mirrored here via the test-only
    # setter.
    _prev = _monitor.get_monitor_level()
    _monitor.set_monitor_level(_monitor.MonitorLevel.BASIC)
    try:
        monkeypatch.setattr(se, "probe_endpoint", lambda ep, timeout_s=5.0: None)
        csv_dir = tmp_path / "timing"
        csv_dir.mkdir()
        args = dataclasses.replace(
            serve_policy.Args(),
            cache_config=str(_minimal_routed_yaml(tmp_path)),
            timing_csv_dir=str(csv_dir),
        )
        policy = serve_policy._wrap_policy(FakePolicy(), args, quiet=True, eager=True)
        # enable_csv wired through _wrap_policy; a task cycle flushes a real CSV.
        policy.on_task_begin()
        policy.infer(
            __import__("tests.cache.test_interceptor", fromlist=["_make_obs"])._make_obs()
        )
        policy.on_task_end()
        assert list(csv_dir.glob("*.csv")), "SystemTimer CSV artifact not written"
    finally:
        _monitor.set_monitor_level(_prev)
