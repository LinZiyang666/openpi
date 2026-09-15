"""Entry points the online RIT arms pass through: identity, retries, per-step slot, validator."""

from __future__ import annotations

import pathlib
import types

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "exp/libero_groot/config/rit/libero_10/template.yaml"


def test_yaml_identity_prefers_the_bundle_then_the_startup_stem():
    from exp.libero_groot.serve_groot_libero import yaml_identity

    assert yaml_identity("l10_online_ir70", "/x/startup.yaml") == "l10_online_ir70"
    assert yaml_identity("default", "/x/startup.yaml") == "startup"
    assert yaml_identity("default", None) == "default"


def test_scheduler_kwargs_carry_the_retry_cap():
    from exp.gate_threshold_pareto.run_gtp import scheduler_kwargs_from_args

    assert scheduler_kwargs_from_args(types.SimpleNamespace(eval_concurrency=0, max_episode_retries=None)) is None
    assert scheduler_kwargs_from_args(types.SimpleNamespace(eval_concurrency=0, max_episode_retries=0)) == {
        "max_episode_retries": 0
    }
    assert scheduler_kwargs_from_args(types.SimpleNamespace(eval_concurrency=2, max_episode_retries=1)) == {
        "eval_concurrency": 2,
        "max_episode_retries": 1,
    }
    with pytest.raises(SystemExit):
        scheduler_kwargs_from_args(types.SimpleNamespace(eval_concurrency=0, max_episode_retries=-1))


def test_hit_row_passes_the_online_rit_slot_through():
    from examples.libero.episode_runner import _hit_row

    task = types.SimpleNamespace(
        yaml_id="y", task_id=1, episode_idx=2, orig_init_state_idx=3, task_uid="u", phase="eval", attempt=0
    )
    diag = {"decision_idx": 4, "q_pre": {"7": 0.1}, "fb": []}
    row = _hit_row(task, 5, {"hit_type": "MISS", "online_rit": diag}, 50)
    assert row["online_rit"] == diag
    assert _hit_row(task, 5, {"hit_type": "MISS"}, 50)["online_rit"] is None


def _online_arm(tmp_path: pathlib.Path) -> pathlib.Path:
    from tests.cache.test_config_online_rit import write_scales

    scales = write_scales(tmp_path / "scales.npz")
    doc = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    cp1 = doc["checkpoints"]["cp1"]
    cp1["judge"] = {
        "type": "online_rit",
        "tiers": [0.875, 0.75, 0.5],
        "alpha": 0.05,
        "delta": 0.4,
        "knots": [0.0, 0.25, 0.5, 0.75, 1.0],
        "update_scales_path": scales,
        "feedback_mode": "fm1",
        "update_enabled": True,
        "window": 128,
        "n_min": 20,
        "h_exec": 5,
        "snapshot_every": 200,
    }
    cp1["gate"] = {
        "type": "score_hysteresis",
        "theta_low": 0.9973,
        "theta_high": 0.9973,
        "j": 3,
        "probe_interval": 3,
        "L": 6,
        "include_ws": True,
    }
    doc["write_policy"] = {"type": "never"}
    path = tmp_path / "l10_online_ir70.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def test_run_gtp_init_map_stamps_original_indices(tmp_path):
    from exp.gate_threshold_pareto.run_gtp import SweepStrategy
    from openpi.conductor.task import ServerEndpoint

    idx = {t: list(range(t, t + 50, 2)) for t in range(10)}
    strat = SweepStrategy("libero_10", {"a": "/a.yaml"}, 25, init_index_map=idx)
    eps = strat._episodes("a", ServerEndpoint("h", 1))  # noqa: SLF001
    assert [e.orig_init_state_idx for e in eps[:3]] == idx[0][:3]
    assert all(e.episode_idx == i % 25 for i, e in enumerate(eps))


def test_run_gtp_validator_admits_an_online_rit_arm(tmp_path):
    from exp.gate_threshold_pareto.run_gtp import JUDGE_TYPES, validate_arms

    assert "online_rit" in JUDGE_TYPES
    path = _online_arm(tmp_path)
    rows = [{"arm": "l10_online_ir70", "yaml": str(path), "suite": "libero_10"}]
    out = validate_arms(rows, phase="eval", judge_type="online_rit", eval_gate="score_hysteresis")
    assert out == {"l10_online_ir70": str(path)}
    with pytest.raises(SystemExit, match="expected 'threshold'"):
        validate_arms(rows, phase="eval", judge_type="threshold", eval_gate="score_hysteresis")


def test_groot_guard_and_loader_accept_the_arm(tmp_path):
    from openpi.cache.config import load_cache_config, required_warm_timesteps
    from openpi.cache.groot.load_guard import validate_groot_cache_config

    cfg = load_cache_config(str(_online_arm(tmp_path)))
    validate_groot_cache_config(cfg, allow_hysteresis_gate=True, num_inference_timesteps=8)
    assert required_warm_timesteps(cfg) == frozenset({0.875, 0.75, 0.625, 0.5})
