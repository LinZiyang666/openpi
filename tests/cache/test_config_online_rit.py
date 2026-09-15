"""online_rit config: validation, warm-timestep enumeration, include_ws, factory identity."""

from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.cache.components.gate import ScoreHysteresisGate
from openpi.cache.components.online_rit import OnlineRiskCurves, OnlineRitJudge
from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    CheckpointConfig,
    ConfigValidationError,
    GateConfig,
    InMemoryConfig,
    JudgeConfig,
    KeyFieldConfig,
    KeysConfig,
    SearchStrategyConfig,
    _build_gate,
    _build_judge,
    required_warm_timesteps,
    validate_cache_config,
)
from openpi.cache.online_state import CurveRegistry
from openpi.cache.types import groot_n15_schedule

D = 32
TIERS = [0.875, 0.75, 0.5]
TIER_IDX = [7, 6, 4]


LIB_A = "1" * 64
LIB_B = "2" * 64


def write_scales(path, tier_indices=TIER_IDX, dim=D, library_sha256=LIB_A):
    mask = np.zeros(dim, dtype=bool)
    mask[:7] = True
    arrays = {"exec_mask": mask, "mask_D": mask, "scale_D": np.ones(dim, np.float32)}
    for i in tier_indices:
        arrays[f"mask_d_{i}"] = mask
        arrays[f"scale_d_{i}"] = np.ones(dim, np.float32)
    arrays["meta_json"] = np.array(json.dumps({"exec_dims": 7, "library_sha256": library_sha256, "schedule_id": "groot_n15_k8_v1"}))
    np.savez(path, **arrays)
    return str(path)


def scales_sha(path) -> str:
    import hashlib

    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def fixed_for(path, h_exec=5) -> dict:
    return {"scales_sha256": scales_sha(path), "schedule_id": "groot_n15_k8_v1", "h_exec": h_exec}


def judge_cfg(scales_path, **over) -> JudgeConfig:
    base = dict(
        type="online_rit",
        tiers=list(TIERS),
        alpha=0.05,
        delta=0.5,
        knots=[0.0, 0.25, 0.5, 0.75, 1.0],
        update_scales_path=scales_path,
        feedback_mode="fm1",
        update_enabled=True,
        window=128,
        n_min=20,
        h_exec=5,
        snapshot_every=200,
    )
    base.update(over)
    return JudgeConfig(**base)


def _config(judge: JudgeConfig, *, gate: GateConfig | None = None, schedule="groot_n15_k8_v1"):
    return CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(
            type="in_memory",
            vector_dims={"robot_state": D},
            in_memory=InMemoryConfig(preload_path="lib.pkl"),
        ),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=gate or GateConfig(type="always_search"),
                judge=judge,
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"),
            ),
        },
        denoise_schedule=schedule,
    )


@pytest.fixture
def scales(tmp_path):
    return write_scales(tmp_path / "scales.npz")


def test_valid_config_passes(scales):
    validate_cache_config(_config(judge_cfg(scales)))


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"tiers": [0.75, 0.875]}, "strictly decreasing"),
        ({"tiers": [0.3]}, "not a snapshot"),
        ({"tiers": []}, "non-empty"),
        ({"alpha": 0.7}, "alpha"),
        ({"delta": -1.0}, "delta"),
        ({"knots": [0.0, 0.5]}, "at least three"),
        ({"knots": [0.0, 0.6, 0.5]}, "strictly increasing"),
        ({"knots": [0.0, 0.5, 1.5]}, "domain"),
        ({"feedback_mode": "fm2"}, "feedback_mode"),
        ({"update_enabled": None}, "update_enabled"),
        ({"window": 10, "n_min": 20}, "window >= n_min"),
        ({"h_exec": 0}, "h_exec"),
        ({"update_scales_path": "/nonexistent.npz"}, "not found"),
        ({"warm_tiers": [{"threshold": 0.5, "start_t": 0.5}]}, "warm_tiers"),
    ],
)
def test_static_rejections(scales, over, needle):
    with pytest.raises(ConfigValidationError, match=needle):
        validate_cache_config(_config(judge_cfg(scales, **over)))


def test_driver_side_validation_can_skip_server_file_checks(tmp_path):
    scales = write_scales(tmp_path / "scales.npz")
    cfg = _config(judge_cfg(scales, update_scales_path="/nonexistent/scales.npz", init_state_path="/nonexistent/init.json"))
    with pytest.raises(ConfigValidationError, match="not found"):
        validate_cache_config(cfg)
    validate_cache_config(cfg, check_files=False)


def test_requires_named_schedule_and_cp1(scales):
    with pytest.raises(ConfigValidationError, match="denoise_schedule"):
        validate_cache_config(_config(judge_cfg(scales), schedule=None))


def test_required_warm_timesteps_include_successor_snapshots(scales):
    cfg = _config(judge_cfg(scales))
    assert required_warm_timesteps(cfg) == frozenset({0.875, 0.75, 0.625, 0.5})


def test_hysteresis_gate_needs_include_ws(scales):
    gate = GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.9, j=3, probe_interval=3, L=6)
    with pytest.raises(ConfigValidationError, match="include_ws"):
        validate_cache_config(_config(judge_cfg(scales), gate=gate))
    gate.include_ws = True
    validate_cache_config(_config(judge_cfg(scales), gate=gate))
    built = _build_gate(gate)
    assert isinstance(built, ScoreHysteresisGate) and built._include_ws is True  # noqa: SLF001


def test_include_ws_is_a_stray_field_on_other_gates():
    cfg = _config(JudgeConfig(type="always_hit"), gate=GateConfig(type="always_search", include_ws=True))
    with pytest.raises(ConfigValidationError, match="cannot set"):
        validate_cache_config(cfg)
    gate = GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.9, j=3, include_ws="yes")
    with pytest.raises(ConfigValidationError, match="include_ws must be a bool"):
        validate_cache_config(_config(JudgeConfig(type="always_hit"), gate=gate))


def test_legacy_hysteresis_gate_builds_unchanged():
    gate = GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.9, j=3, probe_interval=3, L=6)
    validate_cache_config(_config(JudgeConfig(type="always_hit"), gate=gate))
    built = _build_gate(gate)
    assert built._include_ws is False  # noqa: SLF001


def test_factory_requires_registry_and_identity(scales):
    cfg = judge_cfg(scales)
    sched = groot_n15_schedule(8)
    with pytest.raises(ConfigValidationError, match="CurveRegistry"):
        _build_judge(cfg, yaml_id="a", schedule=sched, library_sha256="1" * 64)
    reg = CurveRegistry()
    with pytest.raises(ConfigValidationError, match="yaml_id"):
        _build_judge(cfg, schedule=sched, online_registry=reg, library_sha256="1" * 64)
    with pytest.raises(ConfigValidationError, match="sha256"):
        _build_judge(cfg, yaml_id="a", schedule=sched, online_registry=reg)


def test_factory_shares_state_per_identity_and_refuses_fingerprint_conflicts(scales):
    reg = CurveRegistry()
    sched = groot_n15_schedule(8)
    j1 = _build_judge(judge_cfg(scales), yaml_id="a", schedule=sched, online_registry=reg, library_sha256="1" * 64)
    j2 = _build_judge(judge_cfg(scales), yaml_id="a", schedule=sched, online_registry=reg, library_sha256="1" * 64)
    assert isinstance(j1, OnlineRitJudge)
    assert j1._key == j2._key  # noqa: SLF001
    assert reg.describe(j1._key)["n_attached"] == 2  # noqa: SLF001
    with pytest.raises(ValueError, match="fingerprint"):
        _build_judge(judge_cfg(scales, delta=0.9), yaml_id="a", schedule=sched, online_registry=reg, library_sha256="1" * 64)
    j3 = _build_judge(judge_cfg(scales), yaml_id="b", schedule=sched, online_registry=reg, library_sha256="1" * 64)
    assert j3._key != j1._key  # noqa: SLF001


def test_factory_refuses_scales_from_another_library_unless_mapped(scales, tmp_path):
    reg = CurveRegistry()
    sched = groot_n15_schedule(8)
    with pytest.raises(ValueError, match="not the scales' library"):
        _build_judge(judge_cfg(scales), yaml_id="x", schedule=sched, online_registry=reg, library_sha256=LIB_B)
    # declared S3 -> S3b mapping: scales from A, served B
    j = _build_judge(judge_cfg(scales, source_library_sha256=LIB_A), yaml_id="s3b", schedule=sched, online_registry=reg, library_sha256=LIB_B)
    assert isinstance(j, OnlineRitJudge)
    with pytest.raises(ValueError, match="source_library_sha256"):
        _build_judge(judge_cfg(scales, source_library_sha256=LIB_B), yaml_id="y", schedule=sched, online_registry=reg, library_sha256=LIB_B)
    no_meta = write_scales(tmp_path / "nometa.npz", library_sha256="")
    with pytest.raises(ValueError, match="no library_sha256"):
        _build_judge(judge_cfg(no_meta), yaml_id="z", schedule=sched, online_registry=reg, library_sha256=LIB_A)


def test_factory_refuses_init_state_with_foreign_fixed_params_or_tampered_content(scales, tmp_path):
    sched = groot_n15_schedule(8)
    reg = CurveRegistry()
    foreign = OnlineRiskCurves(knots=[0.0, 0.25, 0.5, 0.75, 1.0], tier_indices=TIER_IDX, alpha=0.05, window=128, n_min=20, fixed_params={**fixed_for(scales), "scales_sha256": "0" * 64})
    p = tmp_path / "foreign.json"
    p.write_text(json.dumps(foreign.snapshot()))
    with pytest.raises(ValueError, match="fixed_params"):
        _build_judge(judge_cfg(scales, init_state_path=str(p)), yaml_id="f1", schedule=sched, online_registry=reg, library_sha256=LIB_A)
    good = OnlineRiskCurves(knots=[0.0, 0.25, 0.5, 0.75, 1.0], tier_indices=TIER_IDX, alpha=0.05, window=128, n_min=20, fixed_params=fixed_for(scales))
    for n in range(25):
        good.update_batch(0.5, [__import__("openpi.cache.components.online_rit", fromlist=["ContinuationFeedback"]).ContinuationFeedback(7, 0.1, "shadow")], ("t", n))
    snap = good.snapshot()
    snap["windows"]["7"][2][0][0] = 999.0  # tamper, keep hashes
    p2 = tmp_path / "tampered.json"
    p2.write_text(json.dumps(snap))
    with pytest.raises(ValueError, match="state_sha256"):
        _build_judge(judge_cfg(scales, init_state_path=str(p2), update_enabled=False), yaml_id="f2", schedule=sched, online_registry=reg, library_sha256=LIB_A)


def test_factory_loads_init_state_and_checks_its_layout(scales, tmp_path):
    curves = OnlineRiskCurves(knots=[0.0, 0.25, 0.5, 0.75, 1.0], tier_indices=TIER_IDX, alpha=0.05, window=128, n_min=20, fixed_params=fixed_for(scales))
    state_path = tmp_path / "init.json"
    state_path.write_text(json.dumps(curves.snapshot()))
    reg = CurveRegistry()
    sched = groot_n15_schedule(8)
    j = _build_judge(
        judge_cfg(scales, init_state_path=str(state_path), update_enabled=False),
        yaml_id="f", schedule=sched, online_registry=reg, library_sha256="1" * 64,
    )
    c = reg.curves(j._key)  # noqa: SLF001
    assert c.update_enabled is False and c.learning_state_sha256() == curves.learning_state_sha256()
    with pytest.raises(ValueError, match="knots"):
        _build_judge(
            judge_cfg(scales, init_state_path=str(state_path), knots=[0.0, 0.5, 1.0]),
            yaml_id="g", schedule=sched, online_registry=reg, library_sha256="1" * 64,
        )


def test_groot_load_guard_accepts_online_rit(scales):
    from openpi.cache.groot.load_guard import validate_groot_cache_config

    cfg = _config(judge_cfg(scales))
    cfg.write_policy.type = "never"
    validate_groot_cache_config(cfg, num_inference_timesteps=8)
    with pytest.raises(ConfigValidationError):
        validate_groot_cache_config(cfg, num_inference_timesteps=4)
