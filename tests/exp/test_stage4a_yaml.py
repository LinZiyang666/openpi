"""Stage 4a: the n2_server operating-point YAMLs load, validate, and build the
follow_winner gate with the expected params. Non-manual."""

from __future__ import annotations

import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]

# (yaml_path, lock_streak, budget)
_N2_SERVER_YAMLS = [
    (
        _REPO / "exp/gate_research/config/libero_spatial/n2_server"
        "/cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@43__d1__fh75_ws10_quantile.yaml",
        3, 5,
    ),
    (
        _REPO / "exp/gate_research/config/libero_10/n2_server"
        "/cp1_spatial_pool_16__grid3_vision_0@56_vision_1@25_robot_state@18__d1__fh5_ws40_quantile.yaml",
        3, 5,
    ),
]


@pytest.mark.parametrize("yaml_path,lock_streak,budget", _N2_SERVER_YAMLS,
                         ids=lambda p: p.parent.parent.name if isinstance(p, pathlib.Path) else "")
def test_n2_server_yaml_loads_validates_and_builds(yaml_path, lock_streak, budget):
    from openpi.cache.config import _build_gate, load_cache_config, validate_cache_config
    from openpi.cache.components.gate import FollowWinnerGate

    cfg = load_cache_config(str(yaml_path))
    validate_cache_config(cfg)  # must not raise

    # follow_winner requires an in_memory backend (blind replay walks the chain).
    assert cfg.backend.type == "in_memory"

    gate_cfg = cfg.checkpoints["cp1"].gate
    assert gate_cfg.type == "follow_winner"
    assert gate_cfg.lock_streak == lock_streak and gate_cfg.budget == budget

    gate = _build_gate(gate_cfg)
    assert isinstance(gate, FollowWinnerGate)
    assert gate._lock_streak == lock_streak and gate._budget == budget
