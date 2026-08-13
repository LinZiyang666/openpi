"""Tests for the E4/E5 yaml emitters.

The emitters exist to keep the arm matrix a single-variable comparison: only
depth and the nominal index filter may move for E4, only the trajectory shape
for E5, and E5 must stay on the base the original screening used.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from exp.markov_sufficiency import emit_e4_yamls as e4
from exp.markov_sufficiency import emit_e5_yamls as e5

REPO = pathlib.Path(__file__).resolve().parents[2]
E4_SOURCE = REPO / (
    "exp/weighted_sum/config/trajectory/libero_spatial/"
    "cp1_spatial_pool_16__grid3_vision_0@6_vision_1@44_robot_state@50__d3.yaml"
)
E5_SOURCE = REPO / (
    "exp/weighted_sum/config/trajectory_weight_alloc/libero_10/"
    "cp1_spatial_pool_16__grid_vision_0@62_vision_1@37__d3.yaml"
)

needs_e4_source = pytest.mark.skipif(not E4_SOURCE.exists(), reason="requires the spatial d3 base yaml")
needs_e5_source = pytest.mark.skipif(not E5_SOURCE.exists(), reason="requires the 0.62/0.37/0 d3 base yaml")


def _load(path):
    with path.open() as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------------
# E4 derivation
# ------------------------------------------------------------------


@needs_e4_source
def test_derivation_touches_only_the_allowed_keys():
    source = _load(E4_SOURCE)
    derived = e4.derive_arm(source, depth=3, step_filter="window", step_window=5)
    assert e4.diff_keys(source, derived) <= e4.ALLOWED_KEYS


@needs_e4_source
def test_depth_one_drops_trajectory_weights():
    source = _load(E4_SOURCE)
    derived = e4.derive_arm(source, depth=1, step_filter="all", step_window=None)
    ss = derived["checkpoints"]["cp1"]["search_strategy"]
    assert ss["trajectory_depth"] == 1
    assert "trajectory_weights" not in ss


@needs_e4_source
def test_derivation_preserves_scoring_and_backend_bytes():
    source = _load(E4_SOURCE)
    derived = e4.derive_arm(source, depth=3, step_filter="exact", step_window=None)
    ss_a = source["checkpoints"]["cp1"]["search_strategy"]
    ss_b = derived["checkpoints"]["cp1"]["search_strategy"]
    for key in ("field_similarity", "score_normalization", "type", "top_k"):
        assert ss_a.get(key) == ss_b.get(key)
    assert source["keys"] == derived["keys"]
    assert source["backend"] == derived["backend"]
    assert source["checkpoints"]["cp1"]["judge"] == derived["checkpoints"]["cp1"]["judge"]


@needs_e4_source
def test_emit_suite_writes_five_valid_arms(tmp_path):
    written = e4.emit_suite(E4_SOURCE, tmp_path, d_best=3, prefix="spatial")
    assert len(written) == len(e4.ARMS)
    arms = {p.stem.split("__")[-1] for p in written}
    assert arms == {"A0", "A1", "A2", "A3", "A4"}
    a1 = _load(tmp_path / "spatial__A1.yaml")["checkpoints"]["cp1"]["search_strategy"]
    assert (a1["trajectory_depth"], a1["step_filter"], a1["step_window"]) == (1, "window", 5)
    a4 = _load(tmp_path / "spatial__A4.yaml")["checkpoints"]["cp1"]["search_strategy"]
    assert a4["step_filter"] == "exact"


@needs_e4_source
def test_emitter_rejects_out_of_scope_changes():
    source = _load(E4_SOURCE)
    derived = e4.derive_arm(source, depth=3, step_filter="all", step_window=None)
    derived["keys"]["vision_0"]["weight"] = 0.99  # a change the arm matrix forbids
    assert not e4.diff_keys(source, derived) <= e4.ALLOWED_KEYS


# ------------------------------------------------------------------
# E5 derivation
# ------------------------------------------------------------------


@needs_e5_source
def test_e5_accepts_only_the_screened_base():
    e5.check_base(_load(E5_SOURCE))  # must not raise


@needs_e4_source
def test_e5_rejects_a_different_base():
    # The spatial base has robot_state enabled, so it is not the 0.62/0.37/0
    # config the d3-trough was screened on.
    with pytest.raises(SystemExit, match="screened base"):
        e5.check_base(_load(E4_SOURCE))


@needs_e5_source
def test_e5_anchor_is_same_base_depth_one():
    source = _load(E5_SOURCE)
    anchor = e5.derive_anchor(source)
    ss = anchor["checkpoints"]["cp1"]["search_strategy"]
    assert ss["trajectory_depth"] == 1
    assert "trajectory_weights" not in ss
    # The anchor differs from the shapes only in the trajectory settings.
    assert anchor["keys"] == source["keys"]


@needs_e5_source
def test_e5_shape_substitution_sets_depth_from_the_shape():
    derived = e5.derive_shape(_load(E5_SOURCE), [0.2, 0.3, 0.5])
    ss = derived["checkpoints"]["cp1"]["search_strategy"]
    assert ss["trajectory_depth"] == 3
    assert ss["trajectory_weights"] == [0.2, 0.3, 0.5]


@needs_e5_source
def test_emit_arms_writes_anchor_plus_shapes(tmp_path):
    written = e5.emit_arms(E5_SOURCE, tmp_path, [[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]], prefix="l10")
    assert len(written) == 3
    assert (tmp_path / "l10__d1_anchor.yaml").exists()
    assert (tmp_path / "l10__shape0.yaml").exists()
