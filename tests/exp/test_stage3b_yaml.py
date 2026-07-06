"""Stage 3b: the n4_server operating-point YAMLs load, validate, and build the
score_hysteresis gate with the correct 3a winning params (L=6). Non-manual."""

from __future__ import annotations

import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]

# (yaml_path, theta, j, probe_interval, L)
_N4_SERVER_YAMLS = [
    (
        _REPO / "exp/gate_research/config/libero_spatial/n4_server"
        "/cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@43__d1__fh75_ws10_quantile.yaml",
        0.968929, 3, 3, 6,
    ),
    (
        _REPO / "exp/gate_research/config/libero_10/n4_server"
        "/cp1_spatial_pool_16__grid3_vision_0@56_vision_1@25_robot_state@18__d1__fh5_ws40_quantile.yaml",
        0.996873, 3, 3, 6,
    ),
]


@pytest.mark.parametrize("yaml_path,theta,j,pi,L", _N4_SERVER_YAMLS,
                         ids=lambda p: p.parent.parent.name if isinstance(p, pathlib.Path) else "")
def test_n4_server_yaml_loads_validates_and_builds(yaml_path, theta, j, pi, L):
    from openpi.cache.config import _build_gate, load_cache_config, validate_cache_config
    from openpi.cache.components.gate import ScoreHysteresisGate

    cfg = load_cache_config(str(yaml_path))
    validate_cache_config(cfg)  # must not raise

    gate_cfg = cfg.checkpoints["cp1"].gate
    # score_hysteresis, NOT the client_controlled it was copied from.
    assert gate_cfg.type == "score_hysteresis"
    assert gate_cfg.theta_low == theta and gate_cfg.theta_high == theta
    assert gate_cfg.j == j and gate_cfg.probe_interval == pi and gate_cfg.L == L

    gate = _build_gate(gate_cfg)
    assert isinstance(gate, ScoreHysteresisGate) and gate._L == L
