"""The warm-start emitter: schedule-bound cells, its own digest, no step-count default."""

from __future__ import annotations

import json

import pytest
import yaml

from exp.robocasa365 import emit_ws_warmstart_yamls as emitter
from exp.robocasa365.emit_ws_search_yamls import weight_matrix

PIN_ID = "4d13ac5e" + "0" * 56


def _calib_entry(builder: str, state_dim: int) -> dict:
    return {
        "builder_type": builder,
        "vector_dims": {
            "vision_0": 32768,
            "vision_1": 32768,
            "vision_2": 32768,
            "prompt_emb": 2048,
            "robot_state": state_dim,
        },
        "fields": {
            f: {
                "sim_type": "cosine" if f.startswith("vision") else "l2",
                "selected": {
                    "method": "zscore",
                    "params": {"mu": 0.85, "sigma": 0.03, "squash": "tanh"},
                },
            }
            for f in ("vision_0", "vision_1", "vision_2", "robot_state")
        },
    }


def _calib() -> dict:
    return {
        "groot_tp_spatial_pool_16_pnp_warm": _calib_entry(
            "cp1_groot_spatial_pool_16", 20
        ),
        "pi05_spatial_pool_16_pnp_warm": _calib_entry("cp1_spatial_pool_16", 32),
    }


@pytest.fixture(scope="module")
def warm_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws_warmstart_pnp")
    cid = sorted(weight_matrix())[0]
    digest = emitter.emit_warm_arms(
        root, _calib(), PIN_ID, weight_cids=[cid], groot_steps=4
    )
    return root, cid, digest


def test_groot_cells_sweep_the_live_schedule_and_name_it(warm_tree):
    root, cid, digest = warm_tree
    cells = digest["per_teacher"]["groot_tp"]["cells"]
    assert sorted(cells) == [f"{cid}__ws0.2500", f"{cid}__ws0.5000", f"{cid}__ws0.7500"]
    for name in cells:
        cfg = yaml.safe_load(
            (root / "groot_tp" / emitter.ARM / f"{name}.yaml").read_text()
        )
        assert cfg["denoise_schedule"] == "groot_n15_k4_v1"
        assert cfg["checkpoints"]["cp1"]["judge"]["type"] == "always_warm_start"
        assert cfg["backend"]["in_memory"]["expected_pin_id"] == PIN_ID
        assert cfg["backend"]["in_memory"]["preload_path"].endswith(
            "groot_tp_spatial_pool_16_pnp_warm.pkl"
        )
        assert cfg["write_policy"] == {"type": "never"}


def test_pi05_cells_sweep_nine_points_under_the_legacy_reading(warm_tree):
    root, cid, digest = warm_tree
    cells = digest["per_teacher"]["pi05"]["cells"]
    assert len(cells) == 9
    cfg = yaml.safe_load(
        (root / "pi05" / emitter.ARM / f"{cid}__ws0.3000.yaml").read_text()
    )
    assert "denoise_schedule" not in cfg
    assert cfg["checkpoints"]["cp1"]["judge"] == {
        "type": "always_warm_start",
        "start_t": 0.3,
    }


def test_index_records_the_schedule_per_cell(warm_tree):
    root, cid, _ = warm_tree
    index = json.loads((root / "groot_tp" / emitter.ARM / "index.json").read_text())
    assert index[f"{cid}__ws0.5000"]["schedule_id"] == "groot_n15_k4_v1"
    assert index[f"{cid}__ws0.5000"]["start_t"] == 0.5


def test_digest_verifies_and_catches_a_tampered_cell(warm_tree, tmp_path):
    root, cid, _ = warm_tree
    emitter.verify_index_digest(
        root, root / emitter.DIGEST_NAME, expected_pin_id=PIN_ID
    )
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    path = copy / "groot_tp" / emitter.ARM / f"{cid}__ws0.5000.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["checkpoints"]["cp1"]["judge"]["start_t"] = 0.25
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match="hashes to"):
        emitter.verify_index_digest(
            copy, copy / emitter.DIGEST_NAME, expected_pin_id=PIN_ID
        )


def test_digest_is_not_the_ws2_digest(warm_tree):
    _, _, digest = warm_tree
    assert set(digest["source_sha256"]) == {
        "emit_ws_search_yamls.py",
        "emit_ws_search2_yamls.py",
        "emit_ws_warmstart_yamls.py",
    }


def test_groot_step_count_has_no_default(tmp_path):
    cid = sorted(weight_matrix())[0]
    with pytest.raises(SystemExit, match="--groot-steps is required"):
        emitter.emit_warm_arms(
            tmp_path, _calib(), PIN_ID, weight_cids=[cid], groot_steps=None
        )


def test_a_timestep_outside_the_schedule_is_refused(tmp_path):
    cid = sorted(weight_matrix())[0]
    with pytest.raises(AssertionError):
        emitter.emit_warm_arms(
            tmp_path,
            _calib(),
            PIN_ID,
            weight_cids=[cid],
            groot_steps=4,
            timesteps=[0.3],
        )


def test_unknown_weight_cell_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="not in the round-1 matrix"):
        emitter.emit_warm_arms(
            tmp_path, _calib(), PIN_ID, weight_cids=["nope"], groot_steps=4
        )
