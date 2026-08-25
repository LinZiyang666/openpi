"""The emitted arms differ from the search template in exactly two fields.

The template carries this experiment's entire retrieval identity -- the weights
the search spent 46,000 episodes finding, the per-field normalizer parameters
fitted against the S3 library, the backend dims. If an emitter touched any of
that, every arm would still load, still run, and still produce a plausible
frontier, just not of the recipe anyone thinks it is. These tests hold the
template fixed and check that only the gate and the judge move.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from exp.libero_groot import emit_gate_yamls as emit
from exp.libero_groot import gate_pareto_bindings as gpb

_TEMPLATE = {
    "enabled": True,
    "timer": {"enabled": False},
    "keys": {
        "vision_0": {"enabled": True, "weight": 0.4166666666666667},
        "vision_1": {"enabled": True, "weight": 0.3333333333333333},
        "vision_2": {"enabled": False, "weight": 0.0},
        "prompt_emb": {"enabled": False, "weight": 0.0},
        "robot_state": {"enabled": True, "weight": 0.25},
    },
    "key_builder": {"type": "cp1_groot_libero_spatial_pool_16"},
    "checkpoints": {
        "cp1": {
            "enabled": True,
            "gate": {"type": "always_search"},
            "judge": {"type": "always_hit"},
            "search_strategy": {
                "type": "weighted_score_sum_knn",
                "top_k": 1,
                "step_filter": "all",
                "field_similarity": {
                    "vision_0": {"type": "cosine"},
                    "vision_1": {"type": "cosine"},
                    "robot_state": {
                        "type": "l2",
                        "to_similarity": {"type": "exp", "tau": 1.0},
                    },
                },
                "score_normalization": {
                    "type": "per_field",
                    "fields": {
                        "vision_0": {
                            "method": "zscore",
                            "params": {"mu": 0.93, "sigma": 0.024, "squash": "tanh"},
                        },
                        "vision_1": {
                            "method": "zscore",
                            "params": {"mu": 0.84, "sigma": 0.040, "squash": "tanh"},
                        },
                        "robot_state": {
                            "method": "zscore",
                            "params": {"mu": -1.9, "sigma": 0.888, "squash": "tanh"},
                        },
                    },
                },
            },
        },
        "cp3": {"enabled": False, "search_strategy": {"type": "weighted_rrf_knn"}},
    },
    "backend": {
        "type": "in_memory",
        "vector_dims": {
            "vision_0": 32768,
            "vision_1": 32768,
            "prompt_emb": 2048,
            "robot_state": 8,
        },
        "in_memory": {
            "preload_path": "/data/does-not-need-to-exist.pkl",
            "index_type": "brute_force",
        },
    },
    "write_policy": {"type": "never"},
}


@pytest.fixture
def binding(tmp_path, monkeypatch):
    """A real Binding whose four slots point into tmp_path."""
    monkeypatch.setattr(gpb, "REPO_ROOT", tmp_path)
    b = gpb.for_suite("libero_spatial")
    b.config_root.mkdir(parents=True, exist_ok=True)
    b.template_path.write_text(
        yaml.safe_dump(copy.deepcopy(_TEMPLATE), sort_keys=False), encoding="utf-8"
    )
    return b


def _untouched(cfg: dict) -> dict:
    """Everything an emitter must not have changed."""
    cp1 = cfg["checkpoints"]["cp1"]
    return {
        "keys": cfg["keys"],
        "key_builder": cfg["key_builder"],
        "search_strategy": cp1["search_strategy"],
        "vector_dims": cfg["backend"]["vector_dims"],
    }


@pytest.mark.parametrize("mode", ["warmup", "eval", "gate_only"])
def test_only_gate_and_judge_move(binding, mode):
    if mode == "warmup":
        cfg = emit.build_warmup(binding)
    elif mode == "eval":
        cfg = emit.build_eval(binding, theta=0.97, t_fh=0.93)
    else:
        cfg = emit.build_gate_only(binding, theta=0.97)
    assert _untouched(cfg) == _untouched(_TEMPLATE)


def test_warmup_forces_every_step_to_miss(binding):
    cp1 = emit.build_warmup(binding)["checkpoints"]["cp1"]
    assert cp1["gate"] == {"type": "always_search"}
    assert cp1["judge"] == {"type": "threshold", "threshold": emit.FORCE_MISS}
    # Above the closed [0, 1] score range, or the warmup would record scores
    # from a policy that was partly replaying the cache.
    assert emit.FORCE_MISS > 1.0


def test_gate_only_forces_every_search_to_hit(binding):
    cp1 = emit.build_gate_only(binding, theta=0.97)["checkpoints"]["cp1"]
    assert cp1["judge"] == {"type": "threshold", "threshold": emit.FORCE_HIT}
    assert emit.FORCE_HIT < 0.0
    # Same gate as the sweep: this point is the frontier's left-hand limit, not
    # a different gate's.
    assert cp1["gate"]["L"] == emit.GATE_L


@pytest.mark.parametrize("theta,t_fh", [(0.97, 0.99), (0.5, 0.1)])
def test_eval_carries_the_full_hybrid_gate(binding, theta, t_fh):
    cp1 = emit.build_eval(binding, theta=theta, t_fh=t_fh)["checkpoints"]["cp1"]
    assert cp1["gate"] == {
        "type": "score_hysteresis",
        "theta_low": theta,
        "theta_high": theta,
        "j": emit.GATE_J,
        "probe_interval": emit.GATE_PROBE_INTERVAL,
        "L": emit.GATE_L,
    }
    assert cp1["judge"] == {"type": "threshold", "threshold": t_fh}


def test_library_binding_overrides_whatever_the_template_carried(binding):
    cfg = emit.build_warmup(binding)
    assert cfg["backend"]["in_memory"]["preload_path"] == binding.library
    assert cfg["write_policy"] == {"type": "never"}


def test_emitted_files_land_in_the_config_slot(binding):
    emitted = emit.emit_warmup(binding)
    assert set(emitted) == {"gpw_sp"}
    path = binding.config_root / "warmup" / "gpw_sp.yaml"
    assert path.is_file()
    assert str(path).endswith("exp/libero_groot/config/gate_pareto/"
                              "libero_spatial/warmup/gpw_sp.yaml")


def test_sweep_covers_the_shared_grid_and_names_cells_by_percent(binding):
    solved = {
        "arms": {
            "gpw_sp": {
                "theta": 0.97,
                "cells": [{"f_fh": f, "t_fh": 0.9} for f in emit.FH_GRID],
            }
        }
    }
    emitted = emit.emit_eval(binding, solved)
    assert len(emitted) == len(emit.FH_GRID) == 16
    assert "gp_sp_fh05" in emitted and "gp_sp_fh80" in emitted


def test_a_drifted_grid_is_refused(binding):
    # A grid that lost or gained a cell would make this line's frontier no
    # longer comparable cell-for-cell with the pi0.5 one.
    solved = {
        "arms": {
            "gpw_sp": {
                "theta": 0.97,
                "cells": [{"f_fh": f, "t_fh": 0.9} for f in emit.FH_GRID[:-1]],
            }
        }
    }
    with pytest.raises(SystemExit, match="FH_GRID"):
        emit.emit_eval(binding, solved)


def test_a_missing_warmup_arm_is_named(binding):
    with pytest.raises(SystemExit, match="gpw_sp"):
        emit.emit_eval(binding, {"arms": {"gpw_l10": {"theta": 0.5, "cells": []}}})


def test_gate_only_reuses_the_sweeps_solved_theta(binding):
    solved = {
        "arms": {
            "gpw_sp": {
                "theta": 0.9689,
                "cells": [{"f_fh": f, "t_fh": 0.9} for f in emit.FH_GRID],
            }
        }
    }
    emit.emit_eval(binding, solved)
    emitted = emit.emit_gate_only(binding)
    cfg = yaml.safe_load(
        (binding.config_root / "gate_only" / "gpgo_sp.yaml").read_text(encoding="utf-8")
    )
    assert cfg["checkpoints"]["cp1"]["gate"]["theta_low"] == 0.9689
    assert set(emitted) == {"gpgo_sp"}


def test_gate_only_refuses_to_guess_a_theta(binding):
    with pytest.raises(SystemExit, match="emit the sweep first"):
        emit.emit_gate_only(binding)


@pytest.mark.parametrize("mode", ["warmup", "eval", "gate_only"])
def test_a_warm_tier_in_the_template_cannot_survive(binding, mode):
    # Every builder replaces the judge wholesale rather than merging into it,
    # which is what makes the warm-start route unreachable by construction --
    # a WARM_START verdict against a GR00T library carries no intermediates and
    # is silently downgraded to MISS, so a leaked tier would present as an
    # inexplicably low hit rate rather than an error.
    leaked = copy.deepcopy(_TEMPLATE)
    leaked["checkpoints"]["cp1"]["judge"] = {
        "type": "threshold",
        "threshold": 0.9,
        "warm_tiers": [{"threshold": 0.5, "start_t": 3}],
    }
    binding.template_path.write_text(yaml.safe_dump(leaked, sort_keys=False), "utf-8")
    if mode == "warmup":
        cfg = emit.build_warmup(binding)
    elif mode == "eval":
        cfg = emit.build_eval(binding, theta=0.97, t_fh=0.93)
    else:
        cfg = emit.build_gate_only(binding, theta=0.97)
    assert "warm_tiers" not in cfg["checkpoints"]["cp1"]["judge"]


def test_a_missing_template_names_the_fix(binding):
    binding.template_path.unlink()
    with pytest.raises(SystemExit, match="--mode template"):
        emit.build_warmup(binding)
