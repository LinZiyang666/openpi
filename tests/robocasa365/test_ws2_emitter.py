"""Tests for the ws2 YAML emitter: arm shapes, cell counts, real validation.

Every emitted config is run through the production loader + validator inside
the emitter itself, so these tests mainly pin the arm-shape invariants, the
manifest coupling of the control arm and the byte-reproducibility of a
re-emit.
"""

from __future__ import annotations

import json

import pytest
import yaml

from exp.robocasa365.emit_ws_search2_yamls import (
    CALIB_STEM,
    PRELOAD,
    build_cell,
    emit_arm,
    verify_cell,
)
from exp.robocasa365.emit_ws_search_yamls import weight_matrix

CALIB = {
    "builder_type": "cp1_groot_spatial_pool_16",
    "vector_dims": {"vision_0": 32768, "vision_1": 32768, "vision_2": 32768,
                    "prompt_emb": 2048, "robot_state": 20},
    "fields": {
        # zscore params mirror the real calibration json (mu/sigma/squash).
        f: {"sim_type": "cosine" if f.startswith("vision") else "l2",
            "selected": {"method": "zscore",
                         "params": {"mu": 0.85, "sigma": 0.03, "squash": "tanh"}}}
        for f in ("vision_0", "vision_1", "vision_2", "robot_state")
    },
}


def test_matrix_is_the_frozen_132_cell_set():
    configs = weight_matrix()
    assert len(configs) == 132
    assert all(abs(sum(w.values()) - 1.0) < 1e-9 for w in configs.values())


def test_main_arm_cell_carries_the_three_text_ivf_keys():
    weights = {"vision_2": 0.875, "robot_state": 0.125}
    cfg = build_cell(weights, CALIB, text_ivf=True)
    assert cfg["checkpoints"]["cp1"]["search_strategy"]["type"] == "text_ivf_knn"
    assert cfg["backend"]["in_memory"]["index_type"] == "text_ivf"
    assert cfg["keys"]["prompt_emb"] == {"enabled": True, "weight": 0.0}
    assert cfg["backend"]["in_memory"]["preload_path"] == PRELOAD
    # prompt_emb screens; it must never score.
    sn = cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]
    assert "prompt_emb" not in sn


def test_control_arm_cell_is_the_round1_shape_over_the_same_library():
    weights = {"vision_2": 0.875, "robot_state": 0.125}
    cfg = build_cell(weights, CALIB, text_ivf=False)
    assert cfg["checkpoints"]["cp1"]["search_strategy"]["type"] == "weighted_score_sum_knn"
    assert cfg["backend"]["in_memory"]["index_type"] == "brute_force"
    assert cfg["keys"]["prompt_emb"]["enabled"] is False
    # Same library as the main arm: that is what makes the pair matched.
    assert cfg["backend"]["in_memory"]["preload_path"] == PRELOAD


def test_both_arms_keep_the_pure_cache_recipe():
    for text_ivf in (True, False):
        cfg = build_cell({"vision_0": 1.0}, CALIB, text_ivf=text_ivf)
        cp1 = cfg["checkpoints"]["cp1"]
        assert cp1["gate"]["type"] == "always_search"
        assert cp1["judge"]["type"] == "always_hit"
        assert cp1["search_strategy"]["top_k"] == 1
        assert cfg["write_policy"] == {"type": "never"}
        assert cfg["timer"]["enabled"] is False
        assert cfg["checkpoints"]["cp3"] == {"enabled": False,
                                             "search_strategy": {"type": "weighted_rrf_knn"}}


def test_verify_cell_rejects_a_half_converted_config():
    cfg = build_cell({"vision_0": 1.0}, CALIB, text_ivf=True)
    cfg["backend"]["in_memory"]["index_type"] = "brute_force"
    with pytest.raises(AssertionError):
        verify_cell(cfg, "iso_vision_0", text_ivf=True)


def test_emit_arm_writes_index_and_reproduces_byte_for_byte(tmp_path, monkeypatch):
    # The real validator needs the artifact only by path, never opens it, but
    # keep this unit hermetic: stub the on-disk validation.
    monkeypatch.setattr("exp.robocasa365.emit_ws_search2_yamls.validate_on_disk", lambda path: None)
    configs = weight_matrix()
    cids = sorted(configs)[:5]

    index = emit_arm(tmp_path / "main", cids, configs, CALIB, text_ivf=True)
    assert set(index) == set(cids)
    first = {cid: (tmp_path / "main" / f"{cid}.yaml").read_text() for cid in cids}

    emit_arm(tmp_path / "main", cids, configs, CALIB, text_ivf=True)
    assert {cid: (tmp_path / "main" / f"{cid}.yaml").read_text() for cid in cids} == first

    written = json.loads((tmp_path / "main" / "index.json").read_text())
    assert {c: written[c]["weights"] for c in cids} == {c: configs[c] for c in cids}
    loaded = yaml.safe_load(first[cids[0]])
    assert loaded["checkpoints"]["cp1"]["search_strategy"]["type"] == "text_ivf_knn"


def test_calibration_stem_names_the_full704_library():
    assert CALIB_STEM.endswith("_full704")
    assert "full704" in PRELOAD and PRELOAD.startswith("/data/")


@pytest.mark.parametrize("text_ivf", [True, False])
def test_both_arms_pass_the_production_validator(tmp_path, text_ivf):
    """The rule-6 change is what makes the main arm loadable at all.

    Not a stub: this is the loader the server runs, so a regression in the
    GR00T x text_ivf allowance fails here rather than at the first cell.
    """
    from openpi.cache.config import load_cache_config, validate_cache_config

    cfg = build_cell({"vision_2": 0.875, "robot_state": 0.125}, CALIB, text_ivf=text_ivf)
    path = tmp_path / "cell.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    validate_cache_config(load_cache_config(path))
